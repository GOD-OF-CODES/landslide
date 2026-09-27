// PARTICLES (SLIDE workstream): dust clouds, debris chips + mud splatter, the slide-front cloud, rain, rain veils,
// splashes, drips, wheel droplets, ground mist and the windshield (refracting drops + wipers in the cockpit view).
//
// API (DESIGN.md "particles"):
//   dust(pos, size = 1, energy = 0.5)   billowing, sky-lit dust burst (soft particles against the scene depth)
//   debris(pos, count = 8, opts)        rock chips + mud splatter thrown ballistically (land and settle, then fade)
//   splash(pos)                         a single rain/water splash
//   drop(pos, vel, diameterMm)          one ballistic water drop (motion-blurred streak + splash where it lands)
//   rain / veils / drips / windshield: automatic (intensity from ctx.env.rain; windshield when cameraRig.mode === 'car-cockpit')
//   It listens to 'impact' events and spawns dust + debris automatically.
//
// PHYSICAL MODEL (what is matched, see also the report in scratch/rain/):
//   * Drop sizes: Marshall-Palmer N(D) = N0 exp(-Lambda D), Lambda = 4.1 R^-0.21 /mm, R = rain rate (mm/h) from env.rain.
//   * Fall speed: Atlas/Gunn-Kinzer terminal velocity v(D) = 9.65 - 10.3 exp(-0.6 D) m/s (D in mm): 2 m/s .. 8 m/s.
//   * Streak = drop motion during a 1/55 s video exposure (dashcam/phone): 4..15 cm long, relative to the moving camera.
//   * Photometry (Garg & Nayar 2006/07): a drop refracts ~165 deg of the environment, so its radiance is an average of
//     sky and ground radiance, never brighter than the sky. It covers a pixel only for (drop size / streak length) of
//     the exposure, so a streak composites as mix(background, E_drop, coverage) with coverage = pi/4 D_px^2 / streak
//     area. Result: streaks are faint, visible against dark trees and rock, and vanish against the bright overcast.
//   * Depth of field: camera focused at ~8 m with a ~4 mm entrance pupil: near drops (< 1 m) are wide, soft and faint.
//   * Beyond ~7 m individual drops are unresolvable: camera-centred rain "shells" (veil) add the statistical streak
//     texture and wind-driven rain curtains, soft-faded against the scene depth.
//   * Splashes: crown + secondary droplets of a 2-3 mm drop are 1-4 cm and last 60-150 ms (Worthington jets).
//   * Drips from spruce crowns and rock ledges: 4-5.5 mm drops falling from rest, every few seconds per drip point.
// Everything is pooled in fixed GPU ring buffers: the CPU only writes a few attributes when something spawns; motion,
// growth, fading and lighting happen in the shaders. All materials get the global height fog.
import * as THREE from 'three';
import { G, groups } from '../physics/world.js';

const clamp = THREE.MathUtils.clamp;
const lerp = THREE.MathUtils.lerp;
const _v = new THREE.Vector3(), _w = new THREE.Vector3(), _u = new THREE.Vector3(), _m = new THREE.Matrix4();
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _c3 = new THREE.Vector3();
const _down = new THREE.Vector3(0, -1, 0);
const CULL = 'gl_Position = vec4(2.0, 2.0, 2.0, 1.0);';

function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// ------------------------------------------------------------------------------------------------ fog + depth plumbing
function fogUniforms(ctx) {
  return THREE.UniformsUtils.merge([THREE.UniformsLib.fog, ctx.env?.fogUniforms ?? {}]);
}
const hasFog = (ctx) => !!ctx.scene.fog;
/** Will post.depthUniforms exist? (post inits after us; decide at construction so no recompile is needed later) */
const expectSoft = (ctx) => !ctx.flags?.nopost && (!ctx.flags?.only || ctx.flags.only.includes('post'));
const _depthDummy = (() => { const t = new THREE.DataTexture(new Float32Array([1, 1, 1, 1]), 1, 1, THREE.RGBAFormat, THREE.FloatType); t.needsUpdate = true; return t; })();
function softUniforms() {
  return { tSceneDepth: { value: _depthDummy }, sceneDepthRes: { value: new THREE.Vector2(1, 1) }, sceneCamNF: { value: new THREE.Vector2(0.08, 6000) } };
}
const SOFT_GLSL = /* glsl */`
#ifdef SOFT
uniform sampler2D tSceneDepth; uniform vec2 sceneDepthRes; uniform vec2 sceneCamNF;
float sceneViewZAt(vec2 fc) { return perspectiveDepthToViewZ(texture2D(tSceneDepth, fc / sceneDepthRes).r, sceneCamNF.x, sceneCamNF.y); }
/** 0 where the particle touches opaque geometry, 1 once it is 'softness' metres in front of it */
float softFade(float softness) {
  float fragZ = perspectiveDepthToViewZ(gl_FragCoord.z, sceneCamNF.x, sceneCamNF.y);
  return clamp((fragZ - sceneViewZAt(gl_FragCoord.xy)) / softness, 0.0, 1.0);
}
#else
float softFade(float softness) { return 1.0; }
#endif
`;

// Radiance of a water drop (Garg & Nayar): the drop images ~165 deg of the scene behind it (inverted), so it shows a
// blend of sky and ground/forest radiance, plus a few % of front-surface sky reflection. uSkyL = horizon sky radiance.
const WATER_GLSL = /* glsl */`
uniform vec3 uSkyL;
uniform vec3 uGndL;
uniform float uFlash;
vec3 dropRadiance(vec3 d) {
  float up = clamp(0.5 + 0.7 * d.y, 0.0, 1.0);
  return (mix(uGndL, uSkyL, 0.52 + 0.33 * up) + uSkyL * 0.04) * (1.0 + uFlash * 2.5);
}
`;
function waterUniforms(shared) {
  return { uSkyL: shared.skyL, uGndL: shared.gndL, uFlash: shared.flash };
}

// Motion-blurred drop streak, built in screen space (see the header for the photometric model).
const STREAK_VERT_GLSL = /* glsl */`
uniform float uExp;       // exposure time (s)
uniform float uPix;       // radians per pixel (vertical)
uniform float uAperture;  // entrance pupil (m) for defocus
uniform float uFocus;     // focus distance (m)
varying vec3 vQ;          // x across (-1..1 at the drop edge), y along (0 head .. 1 tail), z cap fraction
varying float vA;
varying vec3 vDir;
// h, t: view-space head / tail of the streak; D: drop diameter (m); gain: statistical multiplier.
// Returns the view-space vertex; alpha = peak coverage (0 = cull).
vec4 streakVertex(vec3 h, vec3 t, float D, float gain, out float alpha) {
  alpha = 0.0;
  float zh = -h.z, zt = -t.z;
  if (zh < 0.1 && zt < 0.1) return vec4(0.0, 0.0, -1.0, 1.0);
  if (zt < 0.1) t = mix(h, t, (zh - 0.1) / max(zh - zt, 1e-4));
  if (zh < 0.1) h = mix(t, h, (zt - 0.1) / max(zt - zh, 1e-4));
  zh = -h.z; zt = -t.z;
  vec2 sh = h.xy / zh, st = t.xy / zt;
  vec2 dv = st - sh;
  float Lpx = length(dv) / uPix;
  vec2 dir = Lpx > 1e-3 ? dv / (Lpx * uPix) : vec2(0.0, -1.0);
  vec2 perp = vec2(-dir.y, dir.x);
  float z = 0.5 * (zh + zt);
  float fp = z * uPix;                                         // metres per pixel at the drop
  float Dpx = D / fp;
  float bPx = uAperture * abs(z - uFocus) / (uFocus * fp);    // defocus blur circle (px)
  float Wpx = max(sqrt(Dpx * Dpx + bPx * bPx), 1.0);
  float hw = 0.5 * Wpx + 0.75;
  float y = position.y;
  vec2 s = mix(sh - dir * hw * uPix, st + dir * hw * uPix, y) + perp * position.x * hw * uPix;
  float zz = mix(zh, zt, y);
  // conserved coverage integral: pi/4 * D_px^2 (px^2), spread over the blurred streak footprint
  float area = (Lpx + Wpx) * Wpx * 0.6;
  alpha = gain * 0.7854 * Dpx * Dpx / max(area, 0.6);
  vQ = vec3(position.x * hw / (0.5 * Wpx + 0.5), y, hw / (Lpx + 2.0 * hw));
  return vec4(s * zz, -zz, 1.0);
}
`;
const STREAK_FRAG = /* glsl */`
#include <common>
#include <fog_pars_fragment>
${WATER_GLSL}
varying vec3 vQ;
varying float vA;
varying vec3 vDir;
void main() {
  float across = clamp(1.0 - vQ.x * vQ.x, 0.0, 1.0);
  float cf = min(max(vQ.z, 1e-3), 0.5);
  float along = smoothstep(0.0, 2.0 * cf, vQ.y) * (1.0 - smoothstep(1.0 - 2.0 * cf, 1.0, vQ.y));
  float a = vA * across * along;
  if (a < 0.0012) discard;
  gl_FragColor = vec4(dropRadiance(normalize(vDir)), a);
  #include <fog_fragment>
}`;
function streakUniforms(shared) {
  return { uExp: shared.exp, uPix: shared.pix, uAperture: shared.aperture, uFocus: shared.focus, ...waterUniforms(shared) };
}
function quadStrip() {
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, 0, 0, 1, 0, 0, 1, 1, 0, -1, 1, 0]), 3));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  return g;
}

// ------------------------------------------------------------------------------------------------ puff texture
/** 2x2 atlas of billowy, cauliflower puffs (R = density), built procedurally. */
function makePuffTexture(size = 192) {
  const N = size * 2;
  const data = new Uint8Array(N * N * 4);
  const rnd = mulberry(99);
  const L = 64, lat = new Float32Array(L * L);
  for (let i = 0; i < lat.length; i++) lat[i] = rnd();
  const vn = (x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = lat[((yi & 63) * L) + (xi & 63)], b = lat[((yi & 63) * L) + ((xi + 1) & 63)];
    const c = lat[(((yi + 1) & 63) * L) + (xi & 63)], d = lat[(((yi + 1) & 63) * L) + ((xi + 1) & 63)];
    return lerp(lerp(a, b, u), lerp(c, d, u), v);
  };
  for (let t = 0; t < 4; t++) {
    const ox = (t % 2) * size, oy = Math.floor(t / 2) * size;
    const sx = rnd() * 40, sy = rnd() * 40;
    // cauliflower: a cluster of overlapping round lobes, bigger in the middle
    const blobs = [];
    for (let k = 0; k < 16; k++) {
      const a = rnd() * Math.PI * 2, r = Math.sqrt(rnd()) * 0.27;
      blobs.push([Math.cos(a) * r, Math.sin(a) * r, 0.09 + rnd() * 0.12 * (1 - r * 1.8), 0.6 + rnd() * 0.5]);
    }
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const u = x / size - 0.5, v = y / size - 0.5;
      let b = 0;
      for (const [bx, by, br, bw] of blobs) { const dd = ((u - bx) ** 2 + (v - by) ** 2) / (br * br); b += bw * Math.exp(-dd * 1.6); }
      b = 1 - Math.exp(-b * 1.3);
      const r = Math.hypot(u, v) * 2;
      let n = 0, amp = 0.5, f = 4.5;
      for (let o = 0; o < 5; o++) { n += vn(sx + u * f, sy + v * f) * amp; amp *= 0.5; f *= 2.13; }
      // billow structure: ridged low octaves sharpen lobe edges, high octaves add fine wisps
      const ridge = 1 - Math.abs(2 * vn(sx * 1.7 + u * 7, sy * 1.3 + v * 7) - 1);
      // (QA) ridge weight 0.25 -> 0.08: the ridged octave's contour loops survived the age erosion as bright rings
      // ("soap bubbles") in old dust clouds
      let dens = b * (0.42 + n * 1.05 + ridge * 0.08) - 0.28;
      dens = clamp(dens * 1.7, 0, 1) * clamp((0.94 - r) * 2.8, 0, 1);
      const i = ((oy + y) * N + ox + x) * 4;
      data[i] = Math.round(Math.pow(dens, 0.85) * 255); data[i + 1] = data[i]; data[i + 2] = data[i]; data[i + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true; tex.needsUpdate = true;
  return tex;
}

// ================================================================================================= dust + mist (soft puffs)
const DUST_VERT = /* glsl */`
#include <common>
#include <fog_pars_vertex>
attribute vec3 iPos;
attribute vec3 iVel;
attribute vec4 iTime;   // x birth, y life, z ground y, w seed
attribute vec4 iSize;   // x size0, y size1, z opacity, w buoyancy (m/s^2)
attribute vec4 iCol;    // rgb albedo, a: 0 = sky-lit mist (flat, wide), 1 = lit dust
attribute vec2 iExt;    // x aspect (width/height), y drag k (1/s)
uniform float uTime;
uniform vec3 uWind;
uniform vec3 uSunV;     // sun direction in view space
uniform vec3 uUpV;      // world up in view space
varying vec2 vUv;
varying float vAlpha;
varying vec4 vCol;
varying vec3 vWorld;
varying float vGround;
varying float vSize;
varying vec2 vLight;
varying vec2 vUpS;
varying float vTile;
varying float vAge;
void main() {
  float age = uTime - iTime.x;
  float a = age / iTime.y;
  vAlpha = 0.0;
  if (a < 0.0 || a > 1.0) { ${CULL} return; }
  float k = max(iExt.y, 0.05);
  vec3 p = iPos + iVel * (1.0 - exp(-k * age)) / k + uWind * age * 0.7 + vec3(0.0, 0.5 * iSize.w * age * age, 0.0);
  float sz = mix(iSize.x, iSize.y, 1.0 - pow(1.0 - a, 2.4));
  float ang = iTime.w * 6.2831 + age * (fract(iTime.w * 7.13) - 0.5) * 0.35;
  if (iCol.a < 0.5) ang = (fract(iTime.w * 3.7) - 0.5) * 0.3;     // mist stays level
  float cs = cos(ang), sn = sin(ang);
  vec2 c = position.xy * vec2(iExt.x, 1.0);
  vec2 rc = vec2(c.x * cs - c.y * sn, c.x * sn + c.y * cs);
  vec4 mvPosition = viewMatrix * vec4(p, 1.0);
  // pull the sprite toward the camera a little: less clipping into the ground/rocks it was spawned on
  mvPosition.xyz += normalize(-mvPosition.xyz) * min(sz * 0.25, max(-mvPosition.z - 0.5, 0.0));
  mvPosition.xy += rc * sz;
  vec3 camRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 camUp = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
  vWorld = p + camRight * rc.x * sz + camUp * rc.y * sz;
  gl_Position = projectionMatrix * mvPosition;
  vUv = position.xy + 0.5;
  vTile = floor(fract(iTime.w * 13.7) * 4.0);
  vAlpha = iSize.z * smoothstep(0.0, 0.06, a) * (1.0 - smoothstep(0.55, 1.0, a)) * smoothstep(0.6, 4.0, -mvPosition.z);
  vCol = iCol;
  vGround = iTime.z;
  vSize = sz;
  vAge = a;
  // light directions in the sprite's own (rotated) frame
  vLight = vec2(uSunV.x * cs + uSunV.y * sn, -uSunV.x * sn + uSunV.y * cs);
  vUpS = vec2(uUpV.x * cs + uUpV.y * sn, -uUpV.x * sn + uUpV.y * cs);
  #include <fog_vertex>
}`;
const DUST_FRAG = /* glsl */`
#include <common>
#include <packing>
#include <fog_pars_fragment>
${SOFT_GLSL}
uniform sampler2D uPuff;
uniform vec3 uAmb;
uniform vec3 uSun;
uniform float uTime;
varying vec2 vUv;
varying float vAlpha;
varying vec4 vCol;
varying vec3 vWorld;
varying float vGround;
varying float vSize;
varying vec2 vLight;
varying vec2 vUpS;
varying float vTile;
varying float vAge;
vec2 tileUv(vec2 uv, float t) { return clamp(uv, 0.01, 0.99) * 0.5 + vec2(mod(t, 2.0), floor(t / 2.0)) * 0.5; }
void main() {
  if (vAlpha <= 0.001) discard;
  // slow internal boiling: the puff drifts against a second, counter-rotating lobe field
  vec2 cuv = vUv - 0.5;
  float bt = vAge * 1.6 + vTile;
  vec2 w1 = vec2(cos(bt), sin(bt)) * 0.035 * vAge;
  float d = texture2D(uPuff, tileUv(vUv + w1, vTile)).r;
  if (d < 0.01) discard;
  if (vCol.a > 0.5) {   // dust only: the mist is a thin veil, one sample is enough
    float d2 = texture2D(uPuff, tileUv(0.5 + mat2(0.8, -0.6, 0.6, 0.8) * cuv * 1.35 - w1, mod(vTile + 1.0, 4.0))).r;
    d = mix(d, d * (0.55 + 0.9 * d2), 0.55);
  }
  // erosion: as the cloud ages and thins, its edges fray first (threshold rises), instead of a uniform fade
  float thr = 0.04 + 0.32 * vAge * vAge;
  float dens = clamp((d - thr) / max(1.0 - thr, 0.05), 0.0, 1.0);
  // quad-edge fade: at low mips the neighbouring atlas tile bleeds in, never let that draw the sprite's rectangle
  vec2 eq = min(vUv, 1.0 - vUv);
  float alpha = smoothstep(0.0, 0.65, dens) * vAlpha * smoothstep(0.0, 0.12, min(eq.x, eq.y));
  vec3 col;
  if (vCol.a > 0.5) {
    // single-scattering proxy: density gradient toward the sun and toward the bright overcast above
    float dl = texture2D(uPuff, tileUv(vUv + normalize(vLight + 1e-4) * 0.07, vTile)).r;
    float du = texture2D(uPuff, tileUv(vUv + normalize(vUpS + 1e-4) * 0.08, vTile)).r;
    float sunSh = clamp(0.6 + (d - dl) * 3.0, 0.12, 1.3);
    float skySh = clamp(0.78 + (d - du) * 2.4, 0.3, 1.2);
    float core = mix(1.0, 0.72, smoothstep(0.35, 0.95, d));       // optically thick cores are darker
    col = vCol.rgb * (uAmb * skySh * core + uSun * sunSh);
  } else {
    col = vCol.rgb * uAmb;                                          // rain mist: sky-lit, forward-scattering
  }
  // soft intersection with the ground it was spawned on, and with any opaque geometry (scene depth)
  alpha *= smoothstep(vGround - 0.05, vGround + max(vSize * (vCol.a > 0.5 ? 0.3 : 0.12), 0.1), vWorld.y);
  alpha *= softFade(max(vSize * (vCol.a > 0.5 ? 0.35 : 0.5), 0.2));
  if (alpha < 0.002) discard;
  gl_FragColor = vec4(col, alpha);
  #include <fog_fragment>
}`;

class DustSystem {
  constructor(ctx, cap = 1400, name = 'fx_dust') {
    this.ctx = ctx; this.cap = cap; this.head = 0;
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]), 3));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    const mk = (n, nm) => { const a = new THREE.InstancedBufferAttribute(new Float32Array(cap * n), n); a.setUsage(THREE.DynamicDrawUsage); g.setAttribute(nm, a); return a; };
    this.aPos = mk(3, 'iPos'); this.aVel = mk(3, 'iVel'); this.aTime = mk(4, 'iTime'); this.aSize = mk(4, 'iSize'); this.aCol = mk(4, 'iCol'); this.aExt = mk(2, 'iExt');
    for (let i = 0; i < cap; i++) this.aTime.array[i * 4] = -1e6;
    g.instanceCount = cap;
    this.soft = expectSoft(ctx);
    this.uniforms = {
      ...fogUniforms(ctx), ...softUniforms(),
      uTime: { value: 0 }, uWind: { value: new THREE.Vector3(0.6, 0, 0.25) }, uSunV: { value: new THREE.Vector3(0, 1, 0) }, uUpV: { value: new THREE.Vector3(0, 1, 0) },
      uPuff: { value: DustSystem.puff || (DustSystem.puff = makePuffTexture()) }, uAmb: { value: new THREE.Color(0.62, 0.66, 0.72) }, uSun: { value: new THREE.Color(0.55, 0.52, 0.47) },
    };
    this.mat = new THREE.ShaderMaterial({
      uniforms: this.uniforms, vertexShader: DUST_VERT, fragmentShader: DUST_FRAG,
      transparent: true, depthWrite: false, fog: hasFog(ctx), side: THREE.DoubleSide,
      defines: this.soft ? { SOFT: '' } : {},
    });
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.frustumCulled = false; this.mesh.renderOrder = 5; this.mesh.name = name;
    ctx.scene.add(this.mesh);
    this.geo = g;
    this._dirtyLo = Infinity; this._dirtyHi = -1;
    this.time = 0;
  }
  /** lit: 1 = dust (sun + sky self-shadowing), 0 = rain mist (flat sky-lit). aspect > 1 = wide sprite. drag k (1/s). */
  emit(p, vel, life, size0, size1, opacity, col, ground, buoy = 0.15, lit = 1, aspect = 1, drag = 1.3) {
    const i = this.head; this.head = (this.head + 1) % this.cap;
    this.aPos.array.set([p.x, p.y, p.z], i * 3);
    this.aVel.array.set([vel.x, vel.y, vel.z], i * 3);
    this.aTime.array.set([this.time, life, ground, Math.random()], i * 4);
    this.aSize.array.set([size0, size1, opacity, buoy], i * 4);
    this.aCol.array.set([col.r, col.g, col.b, lit], i * 4);
    this.aExt.array.set([aspect, drag], i * 2);
    this._dirtyLo = Math.min(this._dirtyLo, i); this._dirtyHi = Math.max(this._dirtyHi, i);
  }
  update(dt) {
    this.time += dt;
    this.uniforms.uTime.value = this.time;
    if (this._dirtyHi >= 0) {
      const lo = this._dirtyLo, n = this._dirtyHi - lo + 1;
      for (const [a, k] of [[this.aPos, 3], [this.aVel, 3], [this.aTime, 4], [this.aSize, 4], [this.aCol, 4], [this.aExt, 2]]) {
        a.clearUpdateRanges(); a.addUpdateRange(lo * k, n * k); a.needsUpdate = true;
      }
      this._dirtyLo = Infinity; this._dirtyHi = -1;
    }
  }
}

// ================================================================================================= chips + mud (lit meshes)
const CHIP_HEAD = /* glsl */`
attribute vec3 iP0;
attribute vec3 iV;
attribute vec4 iT;     // birth, life, ground y, spin rate
attribute vec4 iS;     // scale, kind (0 rock chip, 1 mud, 2 wood splinter), seed, unused
uniform float uTime;
varying float vKind;
varying float vShade;
varying vec3 vChL;
mat3 chipRot(vec3 ax, float a) {
  ax = normalize(ax); float s = sin(a), c = cos(a), oc = 1.0 - c;
  return mat3(oc * ax.x * ax.x + c, oc * ax.x * ax.y + ax.z * s, oc * ax.z * ax.x - ax.y * s,
              oc * ax.x * ax.y - ax.z * s, oc * ax.y * ax.y + c, oc * ax.y * ax.z + ax.x * s,
              oc * ax.z * ax.x + ax.y * s, oc * ax.y * ax.z - ax.x * s, oc * ax.z * ax.z + c);
}
`;
const CHIP_NORMAL = /* glsl */`
float chAge = uTime - iT.x;
float chAlive = step(0.0, chAge) * step(chAge, iT.y);
float chDisc = iV.y * iV.y + 2.0 * 9.81 * max(iP0.y - iT.z, 0.0);
float chTl = (iV.y + sqrt(chDisc)) / 9.81;
float chTe = min(chAge, chTl);
vec3 chAx = vec3(fract(iS.z * 13.1) - 0.5, fract(iS.z * 7.7) - 0.5, fract(iS.z * 3.3) - 0.5) + 1e-3;
mat3 chR = chipRot(chAx, iT.w * chTe + iS.z * 6.0);
vec3 objectNormal = chR * normal;
bool chSplat = iS.y > 0.5 && iS.y < 1.5 && chAge > chTl;
if (chSplat) objectNormal = normalize(objectNormal * vec3(1.0, 4.0, 1.0));
#ifdef USE_TANGENT
  vec3 objectTangent = vec3( tangent.xyz );
#endif
`;
const CHIP_VERT = /* glsl */`
vec3 chP = iP0 + iV * chTe + vec3(0.0, -4.905 * chTe * chTe, 0.0);
if (chAge > chTl) { chP.xz += iV.xz * min(chAge - chTl, 0.2) * 0.25; chP.y = iT.z; }
float chSc = iS.x * chAlive * (1.0 - smoothstep(0.8, 1.0, chAge / max(iT.y, 1e-3)));
vec3 chShape = iS.y > 1.5 ? vec3(0.9, 0.22, 2.0) : vec3(1.0);   // wood: thin flat splinter
// (QA) a flattened icosahedron is a pointed lens (read as seeds / petals in the chop shot); hatchet chips from spruce
// are split slabs with blunt, torn ends: push the wood chip's vertices toward the box corners
vec3 chPos = iS.y > 1.5 ? sign(position) * pow(abs(position), vec3(0.4)) : position;
vChL = chPos * chShape;
vec3 transformed = chR * (chPos * chShape * chSc);
if (chSplat) { transformed.y *= 0.22; transformed.xz *= 1.7; transformed.y += chSc * 0.05; }
transformed += chP;
vKind = iS.y;
vShade = 0.7 + 0.6 * fract(iS.z * 31.7);
`;

class ChipSystem {
  constructor(ctx, cap = 900) {
    this.ctx = ctx; this.cap = cap; this.head = 0; this.time = 0;
    // angular chip: perturbed icosahedron, flat shaded
    let base = new THREE.IcosahedronGeometry(1, 0);
    if (base.index) base = base.toNonIndexed();
    const pos = base.attributes.position, rnd = mulberry(5);
    const seen = new Map();
    for (let i = 0; i < pos.count; i++) {
      const key = `${pos.getX(i).toFixed(3)},${pos.getY(i).toFixed(3)},${pos.getZ(i).toFixed(3)}`;
      if (!seen.has(key)) seen.set(key, [0.6 + rnd() * 0.6, 0.35 + rnd() * 0.4, 0.6 + rnd() * 0.6]);
      const s = seen.get(key);
      pos.setXYZ(i, pos.getX(i) * s[0], pos.getY(i) * s[1], pos.getZ(i) * s[2]);
    }
    base.computeVertexNormals();
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', base.attributes.position);
    g.setAttribute('normal', base.attributes.normal);
    const mk = (n, name) => { const a = new THREE.InstancedBufferAttribute(new Float32Array(cap * n), n); a.setUsage(THREE.DynamicDrawUsage); g.setAttribute(name, a); return a; };
    this.aP0 = mk(3, 'iP0'); this.aV = mk(3, 'iV'); this.aT = mk(4, 'iT'); this.aS = mk(4, 'iS');
    for (let i = 0; i < cap; i++) this.aT.array[i * 4] = -1e6;
    g.instanceCount = cap;
    this.uTime = { value: 0 };
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8, metalness: 0, flatShading: true });
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uTime = this.uTime;
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\n' + CHIP_HEAD)
        .replace('#include <beginnormal_vertex>', CHIP_NORMAL)
        .replace('#include <begin_vertex>', CHIP_VERT);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying float vKind;\nvarying float vShade;\nvarying vec3 vChL;')
        .replace('#include <color_fragment>', `#include <color_fragment>
          // rock chip / wet mud / fresh spruce wood (pale sapwood, ~1 in 5 a dark bark flake)
          vec3 chWood = vShade > 1.18 ? vec3(0.075, 0.045, 0.03) : vec3(0.47, 0.335, 0.17) * (0.75 + 0.35 * vShade);
          // growth rings / fibres run along the splinter (local z): fine lengthwise stripes, latewood darker
          float chGr = fract(sin(floor(vChL.x * 3.5 + vChL.y * 2.0 + vShade * 17.0) * 43.7) * 7613.1);
          chWood *= 0.8 + 0.2 * chGr - 0.1 * smoothstep(0.85, 1.0, abs(vChL.z) / 2.0);
          diffuseColor.rgb = vKind < 0.5 ? vec3(0.19, 0.18, 0.165) * vShade : (vKind < 1.5 ? vec3(0.06, 0.045, 0.032) * vShade : chWood);`)
        .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = vKind < 0.5 ? 0.72 : (vKind < 1.5 ? 0.22 : 0.6);');
    };
    mat.customProgramCacheKey = () => 'fxchip2';
    this.mesh = new THREE.Mesh(g, mat);
    this.mesh.frustumCulled = false; this.mesh.name = 'fx_chips';
    this.mesh.castShadow = false; this.mesh.receiveShadow = true;
    ctx.scene.add(this.mesh);
    this._lo = Infinity; this._hi = -1;
  }
  emit(p, v, life, scale, kind, ground, spin) {
    const i = this.head; this.head = (this.head + 1) % this.cap;
    this.aP0.array.set([p.x, p.y, p.z], i * 3);
    this.aV.array.set([v.x, v.y, v.z], i * 3);
    this.aT.array.set([this.time, life, ground, spin], i * 4);
    this.aS.array.set([scale, kind, Math.random(), 0], i * 4);
    this._lo = Math.min(this._lo, i); this._hi = Math.max(this._hi, i);
  }
  update(dt) {
    this.time += dt; this.uTime.value = this.time;
    if (this._hi >= 0) {
      const lo = this._lo, n = this._hi - lo + 1;
      for (const [a, k] of [[this.aP0, 3], [this.aV, 3], [this.aT, 4], [this.aS, 4]]) { a.clearUpdateRanges(); a.addUpdateRange(lo * k, n * k); a.needsUpdate = true; }
      this._lo = Infinity; this._hi = -1;
    }
  }
}

// ================================================================================================= rain streaks (near field)
const RAIN_VERT = /* glsl */`
#include <common>
#include <fog_pars_vertex>
${STREAK_VERT_GLSL}
attribute vec3 iOff;
attribute vec4 iRnd;      // x size quantile, y brightness jitter, z turbulence phase, w unused
uniform float uTime;
uniform vec3 uCam;
uniform vec3 uBox;
uniform vec3 uDisp;       // integrated wind drift (m)
uniform vec3 uWind;       // mean wind (m/s)
uniform vec4 uGust;       // xy gust-front direction, z gust amplitude (fraction), w gust phase
uniform vec3 uCamVel;
uniform float uCount;     // instances in use (box A first, then box B)
uniform float uCountA;    // near box instances
uniform vec3 uBoxA;       // near box: resolved drops at close to real density, low statistical gain
uniform float uGainA;
uniform float uLambda;    // Marshall-Palmer slope (1/mm)
uniform float uDmin;      // smallest rendered drop (mm)
uniform float uGain;      // far box: each streak stands in for many unresolved drops
uniform float uFadeR;
uniform mat4 uCarInv;
uniform float uCarOn;
void main() {
  vA = 0.0; vQ = vec3(0.0); vDir = vec3(0.0, 1.0, 0.0);
  if (float(gl_InstanceID) >= uCount) { ${CULL} return; }
  float D = min(uDmin - log(1.0 - 0.985 * iRnd.x) / uLambda, 5.5);   // mm
  float vt = 9.65 - 10.3 * exp(-0.6 * D);                              // terminal velocity (m/s)
  bool nearBox = float(gl_InstanceID) < uCountA;
  vec3 box = nearBox ? uBoxA : uBox;
  vec3 p = iOff * box + vec3(0.0, -vt * uTime, 0.0) + uDisp;
  p = uCam + mod(p - uCam, box) - 0.5 * box;
  // gust fronts sweep through as bands of stronger wind; small drops also follow the turbulence
  vec2 gd = uGust.xy;
  float gb = sin(dot(p.xz, gd) * 0.11 - uGust.w) * 0.65 + sin(dot(p.xz, vec2(-gd.y, gd.x)) * 0.07 + uGust.w * 0.6) * 0.35;
  vec3 wind = uWind * (1.0 + gb * uGust.z);
  float turb = (0.25 + 0.5 / D) * (0.4 + uGust.z);
  wind.xz += vec2(sin(uTime * 2.1 + iRnd.z * 37.0), cos(uTime * 1.7 + iRnd.z * 23.0)) * turb;
  p.xz += (wind.xz - uWind.xz) * 0.3;
  if (uCarOn > 0.5) {
    vec3 lp = (uCarInv * vec4(p, 1.0)).xyz;
    if (abs(lp.x) < 1.0 && lp.y > -0.3 && lp.y < 2.05 && abs(lp.z) < 2.25) { ${CULL} return; }
  }
  vec3 rel = vec3(wind.x, -vt, wind.z) - uCamVel;
  rel *= min(1.0, 24.0 / max(length(rel), 1e-3));
  vec4 hv = viewMatrix * vec4(p, 1.0);
  vec4 tv = viewMatrix * vec4(p - rel * uExp, 1.0);
  float a;
  vec4 mvPosition = streakVertex(hv.xyz, tv.xyz, D * 1e-3, nearBox ? uGainA : uGain, a);
  float dist = length(p - uCam);
  float rA = 0.5 * uBoxA.x;
  // the far box hands the inner sphere over to the near box; the near box fades before its own wrap boundary
  a *= nearBox ? (1.0 - smoothstep(rA * 0.75, rA * 0.97, dist)) * smoothstep(0.12, 0.45, dist)
               : (1.0 - smoothstep(uFadeR * 0.6, uFadeR, dist)) * smoothstep(rA * 0.75, rA * 0.97, dist);
  a *= 0.7 + 0.6 * iRnd.y;
  vA = min(a, 0.3);
  if (vA < 0.0012) { ${CULL} return; }
  vDir = normalize(p - uCam);
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}`;

class RainSystem {
  constructor(ctx, maxN, shared) {
    this.ctx = ctx; this.maxN = maxN;
    const g = quadStrip();
    const rnd = mulberry(7);
    const off = new Float32Array(maxN * 3), rr = new Float32Array(maxN * 4);
    for (let i = 0; i < maxN; i++) { off[i * 3] = rnd(); off[i * 3 + 1] = rnd(); off[i * 3 + 2] = rnd(); for (let k = 0; k < 4; k++) rr[i * 4 + k] = rnd(); }
    g.setAttribute('iOff', new THREE.InstancedBufferAttribute(off, 3));
    g.setAttribute('iRnd', new THREE.InstancedBufferAttribute(rr, 4));
    g.instanceCount = maxN;
    this.u = {
      ...fogUniforms(ctx), ...streakUniforms(shared),
      uTime: { value: 0 }, uCam: { value: new THREE.Vector3() }, uBox: { value: new THREE.Vector3(14, 12, 14) },
      uDisp: shared.disp, uWind: shared.wind, uGust: shared.gust, uCamVel: shared.camVel,
      uCount: { value: 0 }, uCountA: { value: 0 }, uBoxA: { value: new THREE.Vector3(5, 5, 5) }, uGainA: { value: 3 },
      uLambda: { value: 2.7 }, uDmin: { value: 0.8 }, uGain: { value: 8 }, uFadeR: { value: 6.8 },
      uCarInv: { value: new THREE.Matrix4() }, uCarOn: { value: 0 },
    };
    this.mat = new THREE.ShaderMaterial({
      uniforms: this.u, vertexShader: RAIN_VERT, fragmentShader: STREAK_FRAG,
      transparent: true, depthWrite: false, fog: hasFog(ctx), side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.frustumCulled = false; this.mesh.renderOrder = 8; this.mesh.name = 'fx_rain';
    ctx.scene.add(this.mesh);
  }
}

// ================================================================================================= ballistic water drops
// Wheel droplets, drips from trees / rock ledges: linear drag toward the wind + gravity, drawn as motion-blurred streaks.
const DROP_VERT = /* glsl */`
#include <common>
#include <fog_pars_vertex>
${STREAK_VERT_GLSL}
attribute vec3 iP0;
attribute vec3 iV0;
attribute vec4 iT;      // birth, life, ground y, diameter (mm)
attribute vec2 iK;      // drag k (1/s), gain
uniform float uTime;
uniform vec3 uWind;
void main() {
  vA = 0.0; vQ = vec3(0.0); vDir = vec3(0.0, 1.0, 0.0);
  float age = uTime - iT.x;
  if (age < 0.0 || age > iT.y) { ${CULL} return; }
  float k = max(iK.x, 0.02);
  float e = exp(-k * age), f = (1.0 - e) / k;
  vec3 w = vec3(uWind.x, 0.0, uWind.z);
  vec3 p = iP0 + (iV0 - w) * f + w * age;
  p.y -= 9.81 / k * (age - f);
  vec3 v = (iV0 - w) * e + w;
  v.y -= 9.81 / k * (1.0 - e);
  if (p.y < iT.z) { ${CULL} return; }
  vec4 hv = viewMatrix * vec4(p, 1.0);
  vec4 tv = viewMatrix * vec4(p - v * min(uExp, age + 0.002), 1.0);
  float a;
  vec4 mvPosition = streakVertex(hv.xyz, tv.xyz, iT.w * 1e-3, iK.y, a);
  a *= smoothstep(0.1, 0.35, length(p - cameraPosition));
  vA = min(a, 0.6);
  if (vA < 0.0012) { ${CULL} return; }
  vDir = normalize(p - cameraPosition);
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}`;

class DropSystem {
  constructor(ctx, cap, shared) {
    this.ctx = ctx; this.cap = cap; this.head = 0; this.time = 0;
    const g = quadStrip();
    const mk = (n, name) => { const a = new THREE.InstancedBufferAttribute(new Float32Array(cap * n), n); a.setUsage(THREE.DynamicDrawUsage); g.setAttribute(name, a); return a; };
    this.aP0 = mk(3, 'iP0'); this.aV0 = mk(3, 'iV0'); this.aT = mk(4, 'iT'); this.aK = mk(2, 'iK');
    for (let i = 0; i < cap; i++) this.aT.array[i * 4] = -1e6;
    g.instanceCount = cap;
    this.u = { ...fogUniforms(ctx), ...streakUniforms(shared), uTime: { value: 0 }, uWind: shared.wind };
    this.mat = new THREE.ShaderMaterial({
      uniforms: this.u, vertexShader: DROP_VERT, fragmentShader: STREAK_FRAG,
      transparent: true, depthWrite: false, fog: hasFog(ctx), side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.frustumCulled = false; this.mesh.renderOrder = 8; this.mesh.name = 'fx_drops';
    ctx.scene.add(this.mesh);
    this._lo = Infinity; this._hi = -1;
  }
  /** delay (s) lets a drip start in the future; life should cover the fall to 'ground'. */
  emit(p, v, life, ground, dMm, drag, gain = 1, delay = 0) {
    const i = this.head; this.head = (this.head + 1) % this.cap;
    this.aP0.array.set([p.x, p.y, p.z], i * 3);
    this.aV0.array.set([v.x, v.y, v.z], i * 3);
    this.aT.array.set([this.time + delay, life, ground, dMm], i * 4);
    this.aK.array.set([drag, gain], i * 2);
    this._lo = Math.min(this._lo, i); this._hi = Math.max(this._hi, i);
  }
  update(dt) {
    this.time += dt; this.u.uTime.value = this.time;
    if (this._hi >= 0) {
      const lo = this._lo, n = this._hi - lo + 1;
      for (const [a, k] of [[this.aP0, 3], [this.aV0, 3], [this.aT, 4], [this.aK, 2]]) { a.clearUpdateRanges(); a.addUpdateRange(lo * k, n * k); a.needsUpdate = true; }
      this._lo = Infinity; this._hi = -1;
    }
  }
}

// ================================================================================================= rain veil (far field)
// One full-screen pass: the view ray crosses three camera-centred rain shells (8, 18, 42 m). Each shell carries a
// tiling texture of unresolved streaks (the statistical image of the drops around that distance), falling and slanting
// with the wind relative to the camera, modulated by drifting rain curtains, soft-faded against the scene depth.
// Distant rain shafts add a curtain-modulated veil of sky light over the far slopes.
const VEIL_VERT = /* glsl */`
varying vec2 vNdc;
void main() { vNdc = position.xy; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const VEIL_FRAG = /* glsl */`
#include <common>
#include <packing>
${WATER_GLSL}
uniform sampler2D tSceneDepth; uniform vec2 sceneDepthRes; uniform vec2 sceneCamNF;
uniform sampler2D uStreak;
uniform sampler2D uNoise;
uniform mat4 uInvProj;
uniform mat4 uCamWorld;
uniform vec3 uCam;
uniform float uTime;
uniform vec3 uWindRel;   // wind - camera velocity (m/s)
uniform vec3 uDisp;
uniform float uRain;
uniform vec4 uGainV;     // shell gains, w = shaft strength
uniform float uLayers;
uniform float uFall;     // representative fall speed (m/s)
varying vec2 vNdc;
float curtain(vec2 xz) {
  vec2 q = (xz - uDisp.xz) * 0.011;
  float n = texture2D(uNoise, q).r * 0.65 + texture2D(uNoise, q * 2.9 + 0.37).r * 0.35;
  return smoothstep(0.28, 0.78, n);
}
float curtain1(vec2 xz) { return smoothstep(0.25, 0.8, texture2D(uNoise, (xz - uDisp.xz) * 0.011).r); }
void main() {
  vec4 vp = uInvProj * vec4(vNdc, 1.0, 1.0);
  vec3 vdir = normalize(vp.xyz / vp.w);
  vec3 rd = normalize((uCamWorld * vec4(vdir, 0.0)).xyz);
  float dep = texture2D(tSceneDepth, gl_FragCoord.xy / sceneDepthRes).r;
  float sceneT = dep >= 0.99999 ? 1e5 : -perspectiveDepthToViewZ(dep, sceneCamNF.x, sceneCamNF.y) / max(-vdir.z, 1e-4);
  if (sceneT < 7.0) discard;                  // near geometry (road at the feet, the cabin) covers every shell
  float hl = max(length(rd.xz), 0.05);
  float phi = atan(rd.z, rd.x);
  vec2 tang = vec2(-sin(phi), cos(phi));
  float wt = dot(uWindRel.xz, tang);
  float slant = clamp(wt / uFall, -2.5, 2.5);
  float a = 0.0;
  float cur = mix(0.35, 1.0, curtain1(uCam.xz + rd.xz * (14.0 / hl)));   // curtains are ~100 m features: one lookup
  for (int i = 0; i < 3; i++) {
    if (float(i) >= uLayers) break;
    float R = i == 0 ? 8.0 : (i == 1 ? 18.0 : 42.0);
    float t = R / hl;
    if (t > sceneT) continue;
    float soft = clamp((sceneT - t) / (0.6 * R), 0.0, 1.0);
    float v = rd.y * t;                       // height relative to the eye
    float u = phi * R;                        // arc length on the shell
    // streak lines follow the relative velocity (wt, -fall): x is constant along a streak, y scrolls with the fall
    float tw = R * 0.12;                      // shell tile width (m): 128 texels
    vec2 st = vec2((u + slant * v) / tw + float(i) * 0.31, (v + uFall * uTime * (1.0 + 0.1 * float(i))) / (tw * 4.0));
    vec3 tx = texture2D(uStreak, st).rgb;
    float s = i == 0 ? tx.r : (i == 1 ? tx.g : tx.b);
    float g = i == 0 ? uGainV.x : (i == 1 ? uGainV.y : uGainV.z);
    // high up the streaks thin out (seen end-on) and the band near the eye is left to the particle rain
    float la = s * g * cur * soft * smoothstep(1.4, 0.2, abs(v) / R);
    a = a + la * (1.0 - a);
  }
  // rain shafts: a curtain-modulated veil of rain over the distant slopes
  float tH = min(sceneT, 1500.0);
  float shaft = (1.0 - exp(-tH * 0.0028)) * curtain(uCam.xz + rd.xz * min(tH, 260.0) * 0.6) * uGainV.w;
  vec3 E = dropRadiance(rd);
  float aTot = a + shaft * (1.0 - a);
  if (aTot < 0.002) discard;
  vec3 col = (E * a + uSkyL * (1.0 + uFlash * 2.5) * 0.9 * shaft * (1.0 - a)) / aTot;
  gl_FragColor = vec4(col, aTot * uRain);
}`;

function makeStreakTexture() {
  const W = 128, H = 512;
  const acc = new Float32Array(W * H * 3);
  const rnd = mulberry(4242);
  // per channel: streak length (texels) and count; coverage ~6 %
  const spec = [[16, 250], [8, 480], [4, 900]];
  for (let c = 0; c < 3; c++) {
    const [L0, n] = spec[c];
    for (let k = 0; k < n; k++) {
      const x = rnd() * W, y0 = rnd() * H;
      const L = L0 * (0.6 + rnd() * 0.8);
      const I = Math.pow(0.2 + 0.8 * rnd(), 2) * 1.4;      // drop-size spread (coverage ~ D^2)
      const xi = Math.floor(x), fx = x - xi;
      for (let j = 0; j < L; j++) {
        const y = Math.floor(y0 + j) % H;
        const e = Math.min(1, j + 1, L - j);                 // soft ends
        for (const [dx, w] of [[0, 1 - fx], [1, fx]]) {
          const xx = (xi + dx) % W;
          acc[(y * W + xx) * 3 + c] += I * w * e;
        }
      }
    }
  }
  const data = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) data[i * 4 + c] = Math.round(clamp(acc[i * 3 + c], 0, 1) * 255);
    data[i * 4 + 3] = 255;
  }
  const t = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearMipmapLinearFilter; t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}
function makeNoiseTexture(N = 128) {
  const rnd = mulberry(77);
  const oct = [[8, 0.5], [16, 0.3], [32, 0.2]];
  const lat = oct.map(([f]) => { const a = new Float32Array(f * f); for (let i = 0; i < a.length; i++) a[i] = rnd(); return a; });
  const data = new Uint8Array(N * N * 4);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = 0;
    oct.forEach(([f, w], k) => {
      const fx = x / N * f, fy = y / N * f, xi = Math.floor(fx), yi = Math.floor(fy), ax = fx - xi, ay = fy - yi;
      const sx = ax * ax * (3 - 2 * ax), sy = ay * ay * (3 - 2 * ay), L = lat[k];
      const g = (i, j) => L[((j % f + f) % f) * f + ((i % f + f) % f)];
      v += w * lerp(lerp(g(xi, yi), g(xi + 1, yi), sx), lerp(g(xi, yi + 1), g(xi + 1, yi + 1), sx), sy);
    });
    const i = (y * N + x) * 4;
    data[i] = data[i + 1] = data[i + 2] = Math.round(v * 255); data[i + 3] = 255;
  }
  const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearMipmapLinearFilter; t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

class RainVeil {
  constructor(ctx, shared, layers) {
    this.ctx = ctx;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    this.u = {
      ...softUniforms(), ...waterUniforms(shared),
      uStreak: { value: makeStreakTexture() }, uNoise: { value: makeNoiseTexture() },
      uInvProj: { value: new THREE.Matrix4() }, uCamWorld: { value: new THREE.Matrix4() }, uCam: { value: new THREE.Vector3() },
      uTime: { value: 0 }, uWindRel: { value: new THREE.Vector3() }, uDisp: shared.disp, uRain: { value: 0 },
      uGainV: { value: new THREE.Vector4(0.17, 0.14, 0.11, 0.12) }, uLayers: { value: layers }, uFall: { value: 5.5 },
    };
    // (QA) shell gains: Marshall-Palmer at ~4 mm/h gives ~220 drops > 0.8 mm per m^3; each 2 mm drop covers a pixel
    // for only D / streak length ~ 1.5 % of a 1/55 s exposure, so the drops 5-40 m out integrate to a mean veil opacity
    // of ~0.5-1 %: a faint, fine texture, not distinct white dashes. The former 0.42/0.34/0.26 read as snowfall
    // against the dark headwall and rock cut (~2.5x the physical coverage).
    this.mat = new THREE.ShaderMaterial({
      uniforms: this.u, vertexShader: VEIL_VERT, fragmentShader: VEIL_FRAG,
      transparent: true, depthWrite: false, depthTest: false, fog: false,
    });
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.frustumCulled = false; this.mesh.renderOrder = 6; this.mesh.name = 'fx_rain_veil';
    this.mesh.visible = false;
    ctx.scene.add(this.mesh);
  }
}

// ================================================================================================= splashes
const SPLASH_VERT = /* glsl */`
#include <common>
#include <fog_pars_vertex>
attribute vec4 iP;     // xyz position, w birth
attribute vec3 iS;     // size (m), seed, life (s)
uniform float uTime;
uniform float uPix;
varying vec2 vUv;
varying float vA;
varying float vSeed;
varying float vCov;
varying vec3 vDir;
void main() {
  float age = (uTime - iP.w) / iS.z;
  vA = -1.0; vCov = 0.0; vSeed = 0.0; vUv = vec2(0.0); vDir = vec3(0.0, 1.0, 0.0);
  if (age < 0.0 || age > 1.0) { ${CULL} return; }
  #ifdef LOCAL
    vec3 P0 = (modelMatrix * vec4(iP.xyz, 1.0)).xyz;       // splashes riding on the car (bonnet)
  #else
    vec3 P0 = iP.xyz;
  #endif
  // cylindrical billboard (stays upright)
  vec3 toCam = cameraPosition - P0;
  float dist = length(toCam);
  toCam.y = 0.0;
  vec3 fw = normalize(toCam + vec3(1e-4, 0.0, 0.0));
  vec3 rt = vec3(fw.z, 0.0, -fw.x);
  // never smaller than ~3 px on screen; the coverage (alpha) shrinks instead (energy conserving)
  float S = iS.x, Smin = 3.0 * dist * uPix;
  float Sd = max(S, Smin);
  vCov = (S / Sd) * (S / Sd);
  vec3 wp = P0 + rt * position.x * Sd + vec3(0.0, 1.0, 0.0) * (position.y + 0.5) * Sd * 0.75;
  vec4 mvPosition = viewMatrix * vec4(wp, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  vUv = position.xy + vec2(0.5, 0.5);
  vA = age;
  vSeed = iS.y;
  vDir = normalize(wp - cameraPosition);
  #include <fog_vertex>
}`;
const SPLASH_FRAG = /* glsl */`
#include <common>
#include <fog_pars_fragment>
${WATER_GLSL}
varying vec2 vUv;
varying float vA;
varying float vSeed;
varying float vCov;
varying vec3 vDir;
float h1(float n) { return fract(sin(n * 12.9898 + vSeed * 78.233) * 43758.5453); }
void main() {
  if (vA < 0.0) discard;
  float a = vA;
  vec2 p = (vUv - vec2(0.5, 0.0)) * vec2(1.0, 0.75);
  // features thinner than a pixel are widened to ~1 px with their intensity scaled down (coverage is conserved)
  float px = max(fwidth(p.x), 1e-4);
  // crown: a thin rising water wall (seen side-on as a rim with a faint sheet), breaking up after ~40% of the life
  float cw = 0.07 + 0.2 * sqrt(a);
  float ch = 0.2 * sin(3.1416 * clamp(a * 1.7, 0.0, 1.0));
  float ww = 0.022, wwd = max(ww, px * 0.85);
  float top = smoothstep(ch + px * 0.5, ch - px * 0.5, p.y) * smoothstep(0.0, 0.01, ch);
  float wall = exp(-pow((abs(p.x) - cw) / wwd, 2.0)) * (ww / wwd) * top;
  float sheet = smoothstep(cw + px, cw - px, abs(p.x)) * top * 0.22;
  float crown = (wall + sheet) * (1.0 - smoothstep(0.3, 0.55, a));
  // secondary droplets from the rim, on short parabolas
  float drops = 0.0;
  for (int i = 0; i < 6; i++) {
    float fi = float(i);
    float h = h1(fi + 1.0);
    float dir = (fi / 5.0 - 0.5) * 1.8 + (h - 0.5) * 0.4;
    float sp = 0.5 + 0.7 * h;
    vec2 q = vec2(dir * a * 0.34 * sp, (sp * 1.3 * a - 1.9 * a * a) * 0.42 + 0.02);
    float r = 0.022 * (1.0 - a * 0.5), rd = max(r, px * 0.75);
    drops += smoothstep(rd, rd * 0.3, length(p - q)) * step(0.0, q.y) * (r / rd) * (r / rd);
  }
  float c = clamp(crown * 0.7 + drops * 0.85, 0.0, 1.0);
  float alpha = c * vCov * (1.0 - smoothstep(0.6, 1.0, a)) * 0.8;
  if (alpha < 0.003) discard;
  // water catches the sky: slightly brighter than a falling drop (reflective crown at grazing angles)
  gl_FragColor = vec4(dropRadiance(normalize(vDir)) * 1.12, alpha);
  #include <fog_fragment>
}`;

class SplashSystem {
  constructor(ctx, cap, shared, local = false) {
    this.ctx = ctx; this.cap = cap; this.head = 0; this.time = 0;
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]), 3));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    this.aP = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.aS = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3), 3).setUsage(THREE.DynamicDrawUsage);
    for (let i = 0; i < cap; i++) { this.aP.array[i * 4 + 3] = -1e6; this.aS.array[i * 3 + 2] = 0.1; }
    g.setAttribute('iP', this.aP); g.setAttribute('iS', this.aS);
    g.instanceCount = cap;
    this.u = { ...fogUniforms(ctx), ...waterUniforms(shared), uTime: { value: 0 }, uPix: shared.pix };
    this.mat = new THREE.ShaderMaterial({
      uniforms: this.u, vertexShader: SPLASH_VERT, fragmentShader: SPLASH_FRAG,
      transparent: true, depthWrite: false, fog: hasFog(ctx), side: THREE.DoubleSide,
      defines: local ? { LOCAL: '' } : {},
    });
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.frustumCulled = false; this.mesh.renderOrder = 7; this.mesh.name = local ? 'fx_splash_car' : 'fx_splash';
    if (!local) ctx.scene.add(this.mesh);
    this._lo = Infinity; this._hi = -1;
  }
  emit(p, size, life = 0.12, delay = 0) {
    const i = this.head; this.head = (this.head + 1) % this.cap;
    this.aP.array.set([p.x, p.y, p.z, this.time + delay], i * 4);
    this.aS.array.set([size, Math.random(), life], i * 3);
    this._lo = Math.min(this._lo, i); this._hi = Math.max(this._hi, i);
  }
  update(dt) {
    this.time += dt; this.u.uTime.value = this.time;
    if (this._hi >= 0) {
      const lo = this._lo, n = this._hi - lo + 1;
      for (const [a, k] of [[this.aP, 4], [this.aS, 3]]) { a.clearUpdateRanges(); a.addUpdateRange(lo * k, n * k); a.needsUpdate = true; }
      this._lo = Infinity; this._hi = -1;
    }
  }
}

// ================================================================================================= scene colour copy
// Half-resolution copy (with a mip chain) of the OPAQUE scene of the current frame, for the windshield drops to refract.
// Same trick as post's depth copy: an invisible transparent hook sorted before every other transparent object.
class SceneColorCopy {
  constructor(ctx) {
    this.ctx = ctx; this.want = false; this.ok = false;
    this.rt = new THREE.WebGLRenderTarget(2, 2, {
      type: THREE.HalfFloatType, depthBuffer: false, stencilBuffer: false,
      minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: true,
    });
    this.rt.texture.name = 'fx.sceneColorHalf';
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0, 0, 0, 0], 3));
    g.setDrawRange(0, 0);
    const m = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, depthTest: false, colorWrite: false, fog: false });
    this.hook = new THREE.Mesh(g, m);
    this.hook.name = 'fx_scene_color_hook';
    this.hook.frustumCulled = false; this.hook.renderOrder = -1e9 + 1;
    this.hook.onBeforeRender = (renderer, _s, cam) => { try { this._copy(renderer, cam); } catch (e) { this.want = false; this._failed = true; console.warn('[particles] scene colour copy failed', e); } };
    ctx.scene.add(this.hook);
  }
  _copy(renderer, cam) {
    if (!this.want || this._failed) return;
    const src = this.ctx.post?.composer?.inputBuffer;
    if (!src || cam !== this.ctx.camera || renderer.getRenderTarget() !== src) return;
    const w = Math.max(2, src.width >> 1), h = Math.max(2, src.height >> 1);
    if (this.rt.width !== w || this.rt.height !== h) this.rt.setSize(w, h);
    const props = renderer.properties;
    let dst = props.get(this.rt).__webglFramebuffer;
    if (!dst) { renderer.initRenderTarget(this.rt); dst = props.get(this.rt).__webglFramebuffer; }
    const srcFbo = props.get(src).__webglFramebuffer;
    if (!srcFbo || !dst || Array.isArray(dst)) return;
    const gl = renderer.getContext();
    const prevR = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING), prevD = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, srcFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, dst);
    gl.blitFramebuffer(0, 0, src.width, src.height, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.LINEAR);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, prevR);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, prevD);
    const tex = props.get(this.rt.texture).__webglTexture;
    if (tex) { renderer.state.bindTexture(gl.TEXTURE_2D, tex); gl.generateMipmap(gl.TEXTURE_2D); }
    if (!this._checked) {
      this._checked = true;
      const e = gl.getError();
      if (e) { console.warn('[particles] scene colour copy: GL error', e); this._failed = true; return; }
    }
    this.ok = true;
  }
  dispose() { this.ctx.scene.remove(this.hook); this.hook.geometry.dispose(); this.hook.material.dispose(); this.rt.dispose(); }
}

// ================================================================================================= windshield
const WS_VERT = /* glsl */`
varying vec2 vP;       // windshield plane coords (m): x across (+ = car left), y up along the glass
void main() {
  vP = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const WS_FRAG = /* glsl */`
uniform float uTime;
uniform float uRain;
uniform float uSpeed;      // car speed m/s
uniform vec4 uWiper;       // x period, y max angle, z blade tip radius, w on
uniform vec4 uPivots;      // xy pivot A, zw pivot B (plane coords)
uniform float uCycle0;     // time the wiper cycling started
uniform float uSweepF;     // fraction of the cycle the blades move (intermittent mode parks them the rest)
uniform vec3 uSky;
uniform vec3 uGround;
uniform vec4 uCrackA;      // boulder-hit crack: xy centre (plane coords), z radius (m), w strength
uniform vec4 uCrackB;
uniform sampler2D tScene;  // half-res opaque scene (mip chain)
uniform vec2 uRes;         // drawing-buffer size (px)
uniform float uHasScene;
uniform float uFilm;       // accumulated water film (wipers parked) 0..1
uniform vec2 uWH;          // windscreen plane width, height (m)
uniform float uStopT;      // no new drops arrive after this time (under the tunnel roof)
uniform float uPxRad;      // screen pixels per radian
varying vec2 vP;
float sst(float e0, float e1, float x) { float t = clamp((x - e0) / (e1 - e0), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 h22(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
// wiper angle at time t (radians from the parked position, sweeping up)
float wAng(float t) {
  float ph = fract((t - uCycle0) / uWiper.x);
  float s = clamp(ph / uSweepF, 0.0, 1.0);        // sweep, then parked (intermittent)
  return uWiper.y * (0.5 - 0.5 * cos(6.2831853 * s));
}
// most recent time (<= now) the blade passed angle a
float lastPass(float a) {
  if (uWiper.w < 0.5) return -1e6;
  float P = uWiper.x;
  float k = clamp(a / uWiper.y, 0.0, 1.0);
  float s1 = acos(1.0 - 2.0 * k) / 6.2831853;
  float s2 = 1.0 - s1;
  float t0 = uTime - uCycle0;
  float c = floor(t0 / P);
  float up = (c + s1 * uSweepF) * P, dn = (c + s2 * uSweepF) * P;
  float best = -1e6;
  if (up <= t0) best = max(best, up);
  if (dn <= t0) best = max(best, dn);
  if (best < -1e5) best = (c - 1.0 + s2 * uSweepF) * P;
  return best + uCycle0;
}
// blades park pointing toward the car's right (-x) and sweep up: angle measured from -x toward +y
float bladeAngle(vec2 d) { return atan(d.y, -d.x); }
float sweptTime(vec2 p, vec2 piv) {
  vec2 d = p - piv;
  float r = length(d);
  if (r > uWiper.z || r < 0.07) return -1e6;
  float a = bladeAngle(d);
  if (a < -0.05 || a > uWiper.y + 0.02) return -1e6;
  return lastPass(clamp(a, 0.0, uWiper.y));
}
// laminated-glass impact: a crushed core, jittered radial cracks and broken concentric rings
float crack(vec2 p, vec4 C, float seed) {
  if (C.w <= 0.0) return 0.0;
  vec2 d = p - C.xy;
  float r = length(d);
  if (r > C.z) return 0.0;
  float a = atan(d.y, d.x);
  float m = 0.0;
  for (int i = 0; i < 13; i++) {
    float fi = float(i);
    float ai = (fi + 0.8 * h21(vec2(fi, seed))) * 6.2831853 / 13.0;
    float len = C.z * (0.35 + 0.65 * h21(vec2(fi, seed + 3.0)));
    float wob = 0.05 * sin(r * 31.0 + fi * 3.1) + 0.02 * sin(r * 97.0 + fi * 1.7);
    float da = abs(mod(a - ai - wob + 3.14159265, 6.2831853) - 3.14159265);
    float w = (0.0005 + 0.0015 * (1.0 - r / len)) * (0.6 + 0.8 * h21(vec2(fi * 3.7, seed)));
    float fork = step(0.55, h21(vec2(floor(r * 22.0), fi + seed))) * 0.5;
    float dd = min(da * r, abs(da - fork * 0.35) * r + 0.002);
    m = max(m, sst(w, 0.0, dd) * step(r, len) * (0.55 + 0.45 * (1.0 - r / len)));
  }
  for (int k = 1; k <= 4; k++) {
    float fk = float(k);
    float rk = C.z * (0.08 + 0.13 * fk) * (0.85 + 0.3 * h21(vec2(fk, seed)));
    float wr = rk + 0.008 * sin(a * (4.0 + fk) + fk * 2.0);
    float seg = step(0.55, h21(vec2(floor(a * (2.0 + fk) + fk * 7.0), seed + fk)));
    m = max(m, sst(0.0010, 0.0, abs(r - wr)) * seg * 0.6);
  }
  m = max(m, sst(0.022, 0.005, r) * 0.85);
  m += sst(0.06, 0.0, r) * 0.25 * h21(floor(p * 900.0));
  return clamp(m, 0.0, 1.0) * C.w;
}
vec3 sceneAt(vec2 px, float lod) {
  return textureLod(tScene, clamp(px / uRes, vec2(0.002), vec2(0.998)), lod).rgb;
}
// What a drop on the OUTSIDE of the glass sees at screen offset offPx: the world ahead. Where that direction leaves the
// windscreen (the screen copy shows the cabin there) substitute the overcast sky above and the bonnet / road below.
vec3 outsideAt(vec2 frag, vec2 offPx, mat2 toPlane, float lod) {
  vec3 c = uHasScene > 0.5 ? sceneAt(frag + offPx, lod) : mix(uGround, uSky, smoothstep(-80.0, 80.0, offPx.y));
  vec2 q = vP + toPlane * offPx;                       // where the sample falls on the windscreen plane
  float above = smoothstep(uWH.y - 0.06, uWH.y + 0.03, q.y);
  float below = smoothstep(0.02, -0.06, q.y);
  float side = smoothstep(uWH.x * 0.5 - 0.04, uWH.x * 0.5 + 0.04, abs(q.x));
  c = mix(c, uSky * 1.05, max(above, side * 0.7));
  c = mix(c, uGround * 1.6, below);
  return c;
}
void main() {
  float wipe = max(sweptTime(vP, uPivots.xy), sweptTime(vP, uPivots.zw));
  float air = clamp((uSpeed - 4.0) / 20.0, 0.0, 1.0);        // airflow pushes drops up the glass at speed
  // new drops arrive faster when driving into the rain
  float period = mix(10.0, 2.2, uRain) / (1.0 + uSpeed * 0.12);
  // plane metres -> screen pixels (the glass is tilted and seen in perspective)
  vec2 dx = dFdx(vP), dy = dFdy(vP);
  float det = dx.x * dy.y - dx.y * dy.x;
  mat2 toPx = abs(det) > 1e-12 ? mat2(dy.y, -dx.y, -dy.x, dx.x) / det : mat2(0.0);
  float mPerPx = sqrt(abs(det));                             // plane metres per pixel
  // ---- drops: find the (largest) drop covering this point -------------------------------------------------
  float best = 0.0; vec2 bn = vec2(0.0); float brad = 0.0; float bage = 0.0;
  for (int layer = 0; layer < 3; layer++) {
    float cs = layer == 0 ? 0.042 : (layer == 1 ? 0.021 : 0.0095);
    vec2 q = vP / cs;
    vec2 cell = floor(q);
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
      vec2 c = cell + vec2(float(i), float(j));
      vec2 hh = h22(c + float(layer) * 17.0);
      float per = period * (layer == 2 ? 0.6 : 1.0);
      float birth = floor((uTime + hh.x * per) / per) * per - hh.x * per;
      float age = uTime - birth;
      vec2 ctr = c + 0.15 + 0.7 * h22(c * 1.37 + 5.0 + float(layer));
      // only the big beads creep up the glass in the airflow (the rest are pinned by surface tension); the
      // search covers +-1 cell, so the creep is capped (drops that really run are the rivulets below)
      float drift = layer == 0 ? min(air * age * (0.4 + hh.y * 0.8), 0.75) : 0.0;
      ctr.y += drift; ctr.x += drift * 0.18 * (hh.x - 0.5);
      if (birth < wipe || birth > uStopT) continue;
      // radii (m): large merged beads 3.4-7 mm, medium 1.7-3.5 mm, fine mist 0.6-1.3 mm (big for a clean screen, as on
      // an old, hydrophobic-dirty windscreen just before the beads run); they grow as drops merge
      float r0 = layer == 0 ? 0.0034 : (layer == 1 ? 0.0017 : 0.0006);
      float rad = r0 * (1.0 + 1.1 * hh.y) * mix(0.55, 1.0, smoothstep(0.0, 2.5, age)) / cs;
      rad *= 1.0 - air * 0.25;
      vec2 d = q - ctr;
      d.y *= 1.0 + air * 0.8;
      float r = length(d) / rad;
      if (r >= 1.0) continue;
      float wgt = rad * cs * (1.0 - r * r * 0.2);
      if (wgt > best) { best = wgt; bn = d / rad; brad = rad * cs; bage = age; }
    }
  }
  vec2 frag = gl_FragCoord.xy;
  vec3 col = vec3(0.0);
  float alpha = 0.0;
  if (best > 0.0) {
    float r = length(bn);
    // plano-convex water lens (n = 1.33, contact angle ~70 deg): a wide-angle lens that images ~+-25 deg of the world
    // ahead, inverted and strongly minified (sky in its lower half, road / bonnet in its upper half); the steep rim
    // sees the dark cabin (total internal reflection band). Deviation grows with the local surface tilt.
    vec2 dirPx = toPx * bn;
    dirPx = length(dirPx) > 1e-6 ? normalize(dirPx) : vec2(0.0, 1.0);
    float dev = 0.44 * r * sqrt(r);                            // radians
    vec2 offPx = -dirPx * dev * uPxRad;
    float mag = length(toPx * vec2(brad, 0.0)) * 2.0;        // drop diameter in px
    float lod = clamp(log2(max(dev * uPxRad / max(mag * 0.5, 0.5), 1.0)) - 1.0, 0.0, 5.5);
    vec3 refr = outsideAt(frag, offPx, mat2(dx, dy), lod);
    float rim = smoothstep(0.62, 1.0, r);
    vec3 dcol = mix(refr * 0.93, uGround * 0.45, rim * 0.8);
    // faint sheen: the overcast reflected at the water surface (no sun, so no hard glint)
    dcol += uSky * (smoothstep(0.35, 0.0, length(bn - vec2(0.25, 0.45))) * 0.18);
    // defocus: the eye / lens is focused on the road, a 4 mm pupil blurs the glass (0.7 m) by ~3 px
    float blurR = 2.2 / max(mag * 0.5, 1.0);
    float edge = 1.0 - smoothstep(max(1.0 - blurR, 0.15), 1.0 + blurR * 0.3, r);
    // (QA) a bead smaller than that ~5.5 mrad blur circle cannot show as a crisp dot: its light is spread over the
    // blur disc at a fraction of the contrast. Drawn only over its own 1-2 px it was a full-contrast white pixel (the
    // sky image in its lower half) -> hundreds of sparkles against the dark headwall. Fade sub-blur beads by
    // diameter / blur (between peak-preserving and energy-conserving); beads larger than the blur are unchanged.
    float sub = clamp(mag / max(0.0055 * uPxRad, 1.0), 0.2, 1.0);
    col = dcol; alpha = edge * smoothstep(0.0, 0.12, bage) * sub;
  }
  // ---- rivulets: big drops run down (slow) or are blown up the glass (fast), leaving a wet trail ----------------
  {
    float lw = 0.045;
    float lane = floor(vP.x / lw);
    float hs = h21(vec2(lane, 3.1));
    float on = step(0.62, hs) * uRain;
    if (on > 0.0) {
      float dirY = air > 0.2 ? 1.0 : -1.0;
      float spd = mix(0.035, 0.25, air) * (0.6 + hs);
      float cyc = 3.5 + hs * 4.0;
      float tb = floor((uTime + hs * cyc) / cyc) * cyc - hs * cyc;
      float t = uTime - tb;
      float y0 = dirY < 0.0 ? 0.55 - hs * 0.1 : 0.02;
      float yh = y0 + dirY * spd * t;
      float xw = (lane + 0.5) * lw + 0.008 * sin(vP.y * 23.0 + hs * 40.0) + 0.004 * sin(vP.y * 61.0 + hs * 13.0);
      float along = (vP.y - yh) * -dirY;                     // distance behind the head along the trail
      float dxw = abs(vP.x - xw);
      if (tb >= wipe && tb <= uStopT && along > -0.004 && along < min(spd * t, 0.22)) {
        float headR = 0.0038;
        float w = mix(0.0016, 0.0007, clamp(along / 0.2, 0.0, 1.0));
        float trail = sst(w + mPerPx, w * 0.3, dxw) * step(0.0, along) * 0.55;
        vec2 hd = vec2(vP.x - xw, (vP.y - yh) * 0.8);
        float head = sst(headR + mPerPx, headR * 0.6, length(hd));
        float m = max(trail, head) * on;
        if (m > 0.01) {
          vec2 n2 = head > trail ? hd / headR : vec2((vP.x - xw) / w, 0.0);
          vec2 offPx = toPx * (-clamp(n2, -1.0, 1.0) * (head > trail ? headR * 12.0 : w * 6.0));
          vec3 refr = outsideAt(frag, offPx, mat2(dx, dy), 1.5);
          vec3 c2 = refr * 0.96 + uSky * 0.06 * clamp(1.0 - abs(n2.x), 0.0, 1.0);
          col = mix(col, c2, m); alpha = max(alpha, m);
        }
      }
    }
  }
  // ---- thin film: a smeared water layer right behind the blade that beads up within ~0.4 s, and a slow haze
  // of mist droplets building up while the wipers are parked -------------------------------------------------------
  float since = uTime - wipe;
  float smear = uWiper.w > 0.5 && wipe > -1e5 ? exp(-since / 0.22) * (0.2 + 0.15 * uRain) : 0.0;
  float haze = clamp(uFilm + uRain * 0.05 * clamp(since / 8.0, 0.0, 1.0), 0.0, 0.5);
  // the smear is streaky along the blade arc (rubber edge imperfections leave lines of water)
  vec2 rel = vP - uPivots.xy;
  float arcLines = 0.55 + 0.45 * sin(length(rel) * 900.0 + h21(vec2(floor(length(rel) * 140.0), 1.0)) * 6.0);
  float film = max(smear * arcLines, haze);
  if (film > 0.004 && alpha < 0.99) {
    vec2 wv = vec2(h21(floor(vP * 900.0)), h21(floor(vP * 900.0) + 7.0)) - 0.5;
    vec2 streakDir = normalize(vec2(-(vP.y - uPivots.y), vP.x - uPivots.x) + 1e-4);   // smear follows the blade arc
    vec2 offPx = toPx * (wv * 0.004 + streakDir * 0.003 * sin(dot(vP, streakDir.yx) * 700.0)) * film;
    vec3 refr = uHasScene > 0.5 ? sceneAt(frag + offPx, 1.0 + film * 3.0) : uSky * 0.7;
    vec3 c3 = mix(refr, uSky * 0.85, film * 0.35);
    float fa = film * (1.0 - alpha);
    col = (col * alpha + c3 * fa) / max(alpha + fa, 1e-4); alpha += fa;
  }
  // ---- the blades (the parked blades are part of the car model; these are the moving ones) ------------------
  if (uWiper.w > 0.5) {
    float ang = wAng(uTime);
    if (ang > 0.035) {
      for (int b = 0; b < 2; b++) {
        vec2 piv = b == 0 ? uPivots.xy : uPivots.zw;
        vec2 dir = vec2(-cos(ang), sin(ang));
        vec2 d = vP - piv;
        float along = dot(d, dir);
        float across = abs(d.x * dir.y - d.y * dir.x);
        if (along > 0.0 && along < uWiper.z) {
          float blade = sst(0.009 + mPerPx, 0.006, across) * step(0.07, along);
          float arm = sst(0.0065 + mPerPx, 0.004, abs(across - 0.016)) * step(along, uWiper.z * 0.97);
          float m = max(blade, arm * 0.95);
          col = mix(col, vec3(0.01), m); alpha = max(alpha, m);
        }
      }
    }
  }
  // boulder damage: the cracks catch the sky light
  float ck = max(crack(vP, uCrackA, 1.0), crack(vP, uCrackB, 7.0));
  if (ck > 0.001) { col = mix(col, uSky * 1.05 + 0.05, ck * 0.8); alpha = max(alpha, ck * 0.7); }
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(col, alpha);
}`;

class Windshield {
  constructor(ctx, colorCopy, carSplash = null) {
    this.ctx = ctx; this.mesh = null; this._tried = false; this.cycle0 = 0; this.wiperOn = false;
    this.copy = colorCopy; this.film = 0; this.carSplash = carSplash; this._hoodAcc = 0;
  }
  /** Rain hitting the bonnet (seen from the driver's seat): splashes in car-local coordinates. */
  _hoodSplashes(dt, rain) {
    const S = this.carSplash, H = this.hood;
    if (!S || !H?.length || rain < 0.02 || dt <= 0) return;
    this._hoodAcc += dt * 110 * rain * (1 + Math.abs(this.ctx.car?.speed ?? 0) * 0.04);
    let n = 0;
    while (this._hoodAcc >= 1 && n++ < 8) {
      this._hoodAcc -= 1;
      const c = H[Math.floor(Math.random() * H.length)];
      const big = Math.random();
      S.emit(_w.set(c[0] + (Math.random() - 0.5) * c[3], c[1] + 0.003, c[2] + (Math.random() - 0.5) * c[4]), 0.03 + big * big * 0.05, 0.07 + big * 0.08);
    }
  }
  _build() {
    this._tried = true;
    const car = this.ctx.car;
    const root = car?.object;
    if (!root) return;
    root.updateMatrixWorld(true);
    const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
    const pts = [];
    const nm = new THREE.Matrix3();
    root.traverse((m) => {
      if (!m.isMesh) return;
      const mats = Array.isArray(m.material) ? m.material : [m.material];
      if (!mats.some((x) => /glass/i.test(x?.name || ''))) return;
      const g = m.geometry, p = g.attributes.position, n = g.attributes.normal;
      if (!p || !n) return;
      _m.multiplyMatrices(inv, m.matrixWorld);
      nm.getNormalMatrix(_m);
      for (let i = 0; i < p.count; i++) {
        _v.fromBufferAttribute(n, i).applyMatrix3(nm).normalize();
        if (_v.z < 0.45 || _v.y < 0.1) continue;              // forward + upward facing: the windscreen
        _w.fromBufferAttribute(p, i).applyMatrix4(_m);
        if (_w.y < 0.9 || _w.z < -0.3) continue;
        pts.push(_w.clone());
      }
    });
    // bonnet: upward-facing body vertices ahead of the screen base -> a coarse height grid for rain splashes
    const hood = [];
    const HX = 6, HZ = 8, hx0 = -0.78, hx1 = 0.78, hz0 = 0.85, hz1 = 1.95;
    const hgrid = new Float32Array(HX * HZ).fill(-1e9);
    root.traverse((m) => {
      if (!m.isMesh) return;
      const mats = Array.isArray(m.material) ? m.material : [m.material];
      if (mats.some((x) => /glass|headlight|interior/i.test(x?.name || ''))) return;
      const g = m.geometry, p = g.attributes.position, n = g.attributes.normal;
      if (!p || !n || p.count > 400000) return;
      _m.multiplyMatrices(inv, m.matrixWorld);
      nm.getNormalMatrix(_m);
      for (let i = 0; i < p.count; i++) {
        _w.fromBufferAttribute(p, i).applyMatrix4(_m);
        if (_w.z < hz0 || _w.z > hz1 || _w.x < hx0 || _w.x > hx1 || _w.y < 0.7 || _w.y > 1.3) continue;
        _v.fromBufferAttribute(n, i).applyMatrix3(nm).normalize();
        if (_v.y < 0.85) continue;
        const ix = Math.min(HX - 1, Math.floor((_w.x - hx0) / (hx1 - hx0) * HX)), iz = Math.min(HZ - 1, Math.floor((_w.z - hz0) / (hz1 - hz0) * HZ));
        hgrid[iz * HX + ix] = Math.max(hgrid[iz * HX + ix], _w.y);
      }
    });
    for (let iz = 0; iz < HZ; iz++) for (let ix = 0; ix < HX; ix++) {
      const y = hgrid[iz * HX + ix];
      if (y > -1e8) hood.push([hx0 + (ix + 0.5) / HX * (hx1 - hx0), y, hz0 + (iz + 0.5) / HZ * (hz1 - hz0), (hx1 - hx0) / HX, (hz1 - hz0) / HZ]);
    }
    this.hood = hood;
    let x0, x1, yb, yt, zb, zt;
    if (pts.length >= 4) {
      x0 = Math.min(...pts.map((p) => p.x)); x1 = Math.max(...pts.map((p) => p.x));
      yb = Math.min(...pts.map((p) => p.y)); yt = Math.max(...pts.map((p) => p.y));
      const lo = pts.filter((p) => p.y < yb + 0.06), hi = pts.filter((p) => p.y > yt - 0.06);
      zb = lo.reduce((a, p) => a + p.z, 0) / lo.length; zt = hi.reduce((a, p) => a + p.z, 0) / hi.length;
    } else {
      // fallback: typical boxy 4x4 screen relative to the seat camera
      x0 = -0.68; x1 = 0.68; yb = 1.12; yt = 1.62; zb = 0.62; zt = 0.28;
    }
    const inset = 0.015;
    const W = x1 - x0 - 0.06, bl = new THREE.Vector3(x0 + 0.03, yb + 0.02, zb), tl = new THREE.Vector3(x0 + 0.03, yt - 0.02, zt);
    const up = new THREE.Vector3().subVectors(tl, bl);
    const H = up.length();
    const upN = up.clone().normalize();
    const nrm = new THREE.Vector3(1, 0, 0).cross(up).normalize(); // points forward/up (outside)
    const geo = new THREE.BufferGeometry();
    const P = [], UV = [];
    const corner = (u, v) => { const q = bl.clone().addScaledVector(new THREE.Vector3(1, 0, 0), u * W).addScaledVector(up, v).addScaledVector(nrm, -inset); P.push(q.x, q.y, q.z); UV.push(u * W - W / 2, v * H); };
    corner(0, 0); corner(1, 0); corner(1, 1); corner(0, 1);
    geo.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(UV, 2));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    // wiper pivots of car.glb (tools/blender/car.py: cowl pivots at x = 0.34 / -0.20, blades parked toward -x),
    // projected into the plane coordinates
    const toPlane = (x, y, z) => { _a.set(x, y, z).sub(bl); return [_a.x - W / 2, _a.dot(upN)]; };
    const pa = toPlane(0.34, 1.072, 0.785), pb = toPlane(-0.20, 1.072, 0.785);
    const okPiv = (p) => Number.isFinite(p[0]) && Number.isFinite(p[1]) && p[1] > -0.25 && p[1] < 0.1;
    const PA = okPiv(pa) ? pa : [0.34, -0.05], PB = okPiv(pb) ? pb : [-0.20, -0.05];
    this.u = {
      uTime: { value: 0 }, uRain: { value: 0.5 }, uSpeed: { value: 0 },
      uWiper: { value: new THREE.Vector4(1.25, 1.85, 0.49, 0) },
      uPivots: { value: new THREE.Vector4(PA[0], PA[1], PB[0], PB[1]) },
      uCycle0: { value: 0 }, uSweepF: { value: 0.5 },
      uSky: { value: new THREE.Color(0.75, 0.78, 0.82) }, uGround: { value: new THREE.Color(0.05, 0.05, 0.05) },
      uCrackA: { value: new THREE.Vector4(0, 0, 0, 0) }, uCrackB: { value: new THREE.Vector4(0, 0, 0, 0) },
      tScene: { value: this.copy?.rt.texture ?? null }, uRes: { value: new THREE.Vector2(1, 1) }, uHasScene: { value: 0 },
      uFilm: { value: 0 }, uWH: { value: new THREE.Vector2(W, H) }, uPxRad: { value: 514 }, uStopT: { value: 1e9 },
    };
    this.W = W; this.H = H;
    const mat = new THREE.ShaderMaterial({ uniforms: this.u, vertexShader: WS_VERT, fragmentShader: WS_FRAG, transparent: true, depthWrite: false, side: THREE.DoubleSide });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.name = 'fx_windshield';
    this.mesh.frustumCulled = false; this.mesh.renderOrder = 20;
    this.mesh.visible = false;
    root.add(this.mesh);
    if (this.carSplash) root.add(this.carSplash.mesh);
  }
  /** Cracks follow car.damage: each boulder hit (a jump of > 0.05) adds / grows a crack; a repair (checkpoint) clears. */
  _updateCracks(dt) {
    const dmg = this.ctx.car?.damage ?? 0;
    const C = this.cracks || (this.cracks = []);
    if (this._dmgSeen === undefined) this._dmgSeen = dmg;
    if (dmg < this._dmgSeen - 0.02) C.length = 0;
    else if (dmg > this._dmgSeen + 0.05) {
      const hit = dmg - this._dmgSeen;
      const W = this.W || 1.3, H = this.H || 0.6;
      if (C.length >= 2) C.shift();
      const side = Math.random() < 0.6 ? 1 : -1;
      C.push({ x: side * W * (0.1 + Math.random() * 0.32), y: H * (0.3 + Math.random() * 0.45), size: 0, target: 0.16 + hit * 0.9 + Math.random() * 0.08, str: clamp(0.55 + hit * 1.5, 0, 1) });
    }
    this._dmgSeen = dmg;
    for (const c of C) c.size += (c.target - c.size) * Math.min(1, dt * 25); // it spreads in a blink
    if (!this.u) return;
    const set = (v, c) => (c ? v.set(c.x, c.y, Math.max(c.size, 0.001), c.str) : v.set(0, 0, 0, 0));
    set(this.u.uCrackA.value, C[0]); set(this.u.uCrackB.value, C[1]);
  }

  /** rain: local rain (0 under the tunnel roof); envRain: the weather (drops already on the glass stay) */
  update(dt, time, rain, envRain = rain) {
    const ctx = this.ctx;
    const want = ctx.cameraRig?.mode === 'car-cockpit' && !!ctx.car?.object;
    if (want && !this.mesh && !this._tried) this._build();
    this._updateCracks(dt);
    if (this.copy) this.copy.want = false;
    if (!this.mesh) return;
    const cracked = !!this.cracks?.length;
    this.mesh.visible = want && (envRain > 0.01 || cracked);
    if (this.carSplash) {
      this.carSplash.mesh.visible = want;
      if (want) this._hoodSplashes(dt, rain);
      this.carSplash.update(dt);
    }
    const u = this.u;
    const on = (ctx.car?.engineOn ?? true) && envRain > 0.08;
    // parked wipers: the glass slowly fogs up with fine droplets; running wipers clear it
    this.film = clamp(this.film + dt * (on ? -0.6 : rain * 0.012), 0, 0.28);
    if (!this.mesh.visible) return;
    u.uTime.value = time;
    u.uRain.value = clamp(envRain * 1.4, 0, 1);
    if (rain > 0.01) u.uStopT.value = 1e9; else if (u.uStopT.value > time) u.uStopT.value = time;
    u.uSpeed.value = Math.abs(ctx.car?.speed ?? 0);
    u.uFilm.value = this.film;
    if (on && !this.wiperOn) u.uCycle0.value = time;
    this.wiperOn = on;
    u.uWiper.value.w = on ? 1 : 0;
    // one sweep (up + down) takes ~1.1 s; in moderate rain the intermittent setting parks the blades for a few
    // seconds so the glass beads up between wipes, in heavy rain they run continuously
    const sweep = 1.1, pause = rain > 0.75 ? 0.15 : lerp(4.5, 1.2, clamp((rain - 0.1) / 0.6, 0, 1));
    const P = sweep + pause;
    if (Math.abs(u.uWiper.value.x - P) > 0.05) { u.uWiper.value.x = P; u.uSweepF.value = sweep / P; u.uCycle0.value = time; }
    const fl = ctx.env?.flashLevel ?? 0;
    // what the drops mirror / see above the screen: overcast sky outdoors, the dark sodium-lit vault in the tunnel
    const inT = clamp(+(ctx.terrain?.insideTunnel?.(ctx.camera.position) ?? 0) || 0, 0, 1);
    u.uSky.value.setRGB(lerp(0.62 + fl, 0.09, inT), lerp(0.66 + fl, 0.055, inT), lerp(0.72 + fl, 0.025, inT));
    u.uGround.value.setRGB(lerp(0.05, 0.02, inT), lerp(0.05, 0.015, inT), lerp(0.05, 0.01, inT));
    // refraction source: the opaque scene of this frame (post's composer input), half res with mips
    const post = ctx.post, ib = post?.composer?.inputBuffer;
    if (this.copy && ib && !ctx.flags?.nopost) {
      this.copy.want = true;
      u.uRes.value.set(ib.width, ib.height);
      u.uPxRad.value = ib.height / (2 * Math.tan(THREE.MathUtils.degToRad(ctx.camera.fov) / 2));
      u.uHasScene.value = this.copy.ok ? 1 : 0;
      u.tScene.value = this.copy.rt.texture;
    } else u.uHasScene.value = 0;
  }
}

// ================================================================================================= system
export default class Particles {
  constructor(ctx) {
    this.ctx = ctx;
    this.time = 0;
    this._pr = {}; this._prF = {};
    this._frontAcc = 0; this._sprayAcc = 0; this._splashAcc = 0; this._mistAcc = 0; this._wheelAcc = [0, 0, 0, 0];
    this._camPrev = new THREE.Vector3();
    this._impactsThisFrame = 0;
    this.enabled = { rain: true, splashes: true, windshield: true, veil: true, drips: true, wheelDrops: true, mist: true };
    // shared uniform objects (rain, drops, splashes, veil)
    this.shared = {
      skyL: { value: new THREE.Color(0.6, 0.68, 0.8) }, gndL: { value: new THREE.Color(0.06, 0.065, 0.06) }, flash: { value: 0 },
      exp: { value: 1 / 55 }, pix: { value: 0.0019 }, aperture: { value: 0.004 }, focus: { value: 8 },
      disp: { value: new THREE.Vector3() }, wind: { value: new THREE.Vector3(2, 0, 1.5) }, gust: { value: new THREE.Vector4(0.8, 0.6, 0.3, 0) },
      camVel: { value: new THREE.Vector3() },
    };
    this._camVel = this.shared.camVel.value;
    this.drips = [];
  }

  async init() {
    const { ctx } = this;
    const q = ctx.config?.quality ?? {};
    const key = q.key || 'high';
    try { const mod = await import('../render/impostor.js'); this.WIND = mod.WIND; } catch { this.WIND = null; }
    this.dustSys = new DustSystem(ctx, 1400);
    this.chips = new ChipSystem(ctx, 900);
    this.rain = new RainSystem(ctx, Math.round(30000 * (q.rain ?? 0.8)), this.shared);
    this.drops = new DropSystem(ctx, 1600, this.shared);
    this.splashes = new SplashSystem(ctx, 1600, this.shared);
    const veilLayers = key === 'ultra' || key === 'high' ? 3 : key === 'medium' ? 2 : 0;
    if (veilLayers && expectSoft(ctx)) this.veil = new RainVeil(ctx, this.shared, veilLayers);
    // ground splash mist: big, soft, low-alpha sprites (~1 screen of overdraw) -> high and ultra only
    this.mist = key === 'ultra' || key === 'high' ? new DustSystem(ctx, 160, 'fx_rain_mist') : null;
    this._mistRate = key === 'ultra' ? 16 : 10;
    if (this.mist) this.mist.mesh.renderOrder = 4;
    this.colorCopy = expectSoft(ctx) ? new SceneColorCopy(ctx) : null;
    this.carSplash = new SplashSystem(ctx, 256, this.shared, true);
    this.windshield = new Windshield(ctx, this.colorCopy, this.carSplash);
    this._off = ctx.events?.on?.('impact', (p) => this._onImpact(p));
    this._camPrev.copy(ctx.camera.position);
    // tree positions for canopy drips (scatter.json is already cached by terrain/vegetation)
    ctx.assets?.json?.('assets/world/scatter.json').then((s) => { this._indexTrees(s?.trees); }).catch(() => {});
  }

  // ---------------------------------------------------------------------------------------------- API
  /** Dust / spray burst. size ≈ radius of the initial cloud (m); energy ≈ KE/1e5 scales count, spread and lift. */
  dust(pos, size = 1, energy = 0.5) {
    if (!this.dustSys || !pos) return;
    const e = clamp(energy, 0, 20);
    // budget: a wet slope does not turn into a white-out, however many rocks bounce
    let n = Math.round(clamp(1.5 + Math.sqrt(e) * 2.5 + size * 0.6, 2, 9));
    n = Math.min(n, Math.floor(this._dustBudget));
    if (n <= 0) return;
    this._dustBudget -= n;
    const ground = this._ground(pos);
    const wet = this.ctx.env?.wetness ?? 0.75;
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2, r = Math.random();
      const sp = (1.2 + Math.sqrt(e) * 2.5) * (0.4 + Math.random() * 0.8);
      _v.set(Math.cos(a) * sp, (0.4 + Math.random() * 1.0) * sp * 0.8, Math.sin(a) * sp);
      _w.set(pos.x + Math.cos(a) * r * size * 0.5, Math.max(pos.y + Math.random() * size * 0.4, ground + size * 0.3), pos.z + Math.sin(a) * r * size * 0.5);
      const s0 = size * (0.45 + Math.random() * 0.45), s1 = size * (1.5 + Math.random() * 1.3) + Math.sqrt(e) * 0.8;
      // freshly crushed gneiss / granite powder, darkened by the rain (wet: darker, browner, shorter-lived)
      const t = Math.random();
      // (QA) x0.6: an optically thick cloud of rock powder looks like the bulk powder (albedo ~0.3-0.35 dry, ~0.2
      // wet) under the same sky, i.e. about as bright as the concrete headwall, well below the overcast sky; the old
      // values rendered it cotton-white
      const col = _col.setRGB(lerp(0.36, 0.3, wet) + t * 0.05, lerp(0.33, 0.26, wet) + t * 0.04, lerp(0.29, 0.21, wet) + t * 0.03).multiplyScalar(0.6);
      this.dustSys.emit(_w, _v, (2.5 + Math.random() * 3.5) * (0.7 + size * 0.15) * lerp(1.3, 0.8, wet), s0, s1, lerp(0.55, 0.36, wet) * (0.5 + Math.random() * 0.5), col, ground, 0.1 + Math.random() * 0.2);
    }
    // base surge: the densest part of the cloud rolls out along the ground first
    if (e > 0.25 && this._dustBudget >= 1) {
      const k = Math.min(Math.floor(this._dustBudget), e > 1.5 ? 3 : 2);
      this._dustBudget -= k;
      for (let i = 0; i < k; i++) {
        const a = Math.random() * Math.PI * 2, sp = (2.5 + Math.sqrt(e) * 2.2) * (0.7 + Math.random() * 0.6);
        _v.set(Math.cos(a) * sp, 0.25 + Math.random() * 0.4, Math.sin(a) * sp);
        _w.set(pos.x, Math.max(pos.y, ground + size * 0.25), pos.z);
        const t = Math.random();
        const col = _col.setRGB(lerp(0.33, 0.27, wet) + t * 0.04, lerp(0.3, 0.235, wet) + t * 0.03, lerp(0.26, 0.19, wet) + t * 0.03).multiplyScalar(0.6);
        this.dustSys.emit(_w, _v, (3 + Math.random() * 2.5) * lerp(1.2, 0.85, wet), size * 0.6, size * (2.2 + Math.random()) + Math.sqrt(e), lerp(0.45, 0.3, wet), col, ground, 0.05, 1, 1.35, 1.9);
      }
    }
  }

  /** Rock chips + mud splatter. opts: {speed=5, mud=0.35 (fraction of mud blobs), up=1, dir: Vector3 bias,
   *  wood: true -> small pale wood splinters and bark flakes instead (hatchet hits; no mud)} */
  debris(pos, count = 8, opts = {}) {
    if (!this.chips || !pos) return;
    const ground = this._ground(pos);
    const spd = opts.speed ?? 5, mudF = opts.mud ?? 0.35;
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2;
      const up = (0.5 + Math.random()) * (opts.up ?? 1);
      const s = spd * (0.35 + Math.random() * 0.8);
      _v.set(Math.cos(a) * s * 0.7, up * s * 0.8, Math.sin(a) * s * 0.7);
      if (opts.dir) _v.addScaledVector(opts.dir, s * 0.6);
      const mud = opts.wood ? 2 : (Math.random() < mudF ? 1 : 0);
      const sc = mud === 2 ? 0.012 + Math.pow(Math.random(), 2) * 0.022 : (mud ? 0.04 + Math.random() * 0.08 : 0.025 + Math.pow(Math.random(), 2) * 0.1);
      _w.set(pos.x + (Math.random() - 0.5) * 0.4, Math.max(pos.y, ground + 0.05) + Math.random() * 0.3, pos.z + (Math.random() - 0.5) * 0.4);
      this.chips.emit(_w, _v, 2.5 + Math.random() * 3.5, sc, mud, ground + sc * 0.3, (Math.random() - 0.5) * 18);
    }
    // water knocked off wet rock and mud: a spray of drops
    if (!opts.wood && this.drops) {
      const nd = Math.min(24, Math.round(count * 1.2));
      for (let i = 0; i < nd; i++) {
        const a = Math.random() * Math.PI * 2, s = spd * (0.4 + Math.random() * 0.9);
        _v.set(Math.cos(a) * s * 0.8, (0.6 + Math.random()) * s * 0.7, Math.sin(a) * s * 0.8);
        _w.set(pos.x, Math.max(pos.y, ground + 0.1), pos.z);
        this.drops.emit(_w, _v, 1.6, ground, 1 + Math.random() * 2.5, 0.6, 1.5);
      }
    }
  }

  /** Temporarily lift the dust budget (scripted big events). */
  dustBoost(seconds = 4) { this._boostT = Math.max(this._boostT || 0, seconds); }

  splash(pos) { this.splashes?.emit(pos, 0.06 + Math.random() * 0.05, 0.1 + Math.random() * 0.06); }

  /** One ballistic water drop (streak) that splashes where it lands (ground = y of the landing surface, optional). */
  drop(pos, vel, diameterMm = 2, ground = null) {
    if (!this.drops || !pos) return;
    const gy = ground ?? this._ground(pos);
    this.drops.emit(pos, vel || _v.set(0, 0, 0), 3, gy, diameterMm, 0.3, 1.5);
  }

  // ---------------------------------------------------------------------------------------------- internals
  _ground(pos) {
    const phys = this.ctx.physics;
    if (phys?.world) {
      const h = phys.raycast(_u.set(pos.x, pos.y + 2, pos.z), _down, 30, { groups: groups(G.ALL, G.STATIC | G.ROCK) });
      if (h) return h.point.y;
    }
    const road = this.ctx.road;
    if (road) { const pr = road.project(pos, this._pr); return pos.y - pr.dy; }
    return pos.y - 1;
  }

  _onImpact(p) {
    if (!p?.position || this._impactsThisFrame > 6) return;
    this._impactsThisFrame++;
    const cam = this.ctx.camera.position;
    const dist = p.position.distanceTo(cam);
    if (dist > 450) return;
    const e = p.energy ?? 0.5, r = p.radius ?? 0.5;
    if (p.source === 'car') {
      if (e > 0.05) this.debris(p.position, 6, { speed: 3, mud: 0.6 });
      return;
    }
    if (e < 0.03) return;
    const front = p.source === 'front';
    const repeat = 1 / Math.max(1, p.hits ?? 1);   // the first hit throws the most dust
    const size = clamp(r * 1.3 + Math.sqrt(e) * 0.9, 0.4, 7) * (front ? 0.6 : 1);
    if (repeat > 0.3 || e > 1) this.dust(p.position, size * (0.6 + 0.4 * repeat), (front ? e * 0.3 : e) * repeat);
    if (p.source !== 'front' && dist < 160) this.debris(p.position, Math.round(clamp(4 + e * 5, 4, 22)), { speed: 3 + Math.sqrt(e) * 3, mud: 0.4 });
  }

  _updateFront(dt) {
    const L = this.ctx.landslide, road = this.ctx.road;
    if (!L?.front?.active || !road || !this.dustSys) return;
    const cam = this.ctx.camera.position;
    const pr = road.project(cam, this._prF);
    const dist = Math.abs(pr.s - L.front.visS);
    if (dist > 600) return;
    const sp = L.front.speed ?? 0, h = L.front.height ?? 4;
    const growth = clamp(h / 4.4, 0, 1);
    // continuous churning cloud of spray and rock dust along the snout
    this._frontAcc += dt * (2.5 + sp * 2.5) * growth;
    const wet = this.ctx.env?.wetness ?? 0.75;
    let n = 0;
    while (this._frontAcc >= 1 && n++ < 12) {
      this._frontAcc -= 1;
      const d = lerp(-6.5, 3, Math.random());
      L.frontPoint(d, _w);
      _w.y += Math.random() * h * 0.5;
      _v.set(0, 0.6 + Math.random() * 1.2, 0).addScaledVector(road.tangentAt(L.front.visS, _u), sp * (0.9 + Math.random() * 0.5));
      const t = Math.random();
      const col = _col.setRGB(0.2 + t * 0.04, 0.19 + t * 0.035, 0.175 + t * 0.03);
      const ground = _w.y - Math.random() * h;
      this.dustSys.emit(_w, _v, 4 + Math.random() * 4, 1.6 + Math.random() * 1.8, 5 + Math.random() * 5 + sp * 0.4, (0.08 + Math.random() * 0.12) * lerp(1.2, 0.8, wet), col, ground, 0.04 + Math.random() * 0.1);
    }
    // mud + stones thrown off the rolling leading edge
    if (dist < 160) {
      this._sprayAcc += dt * (1.5 + sp * 2.2) * growth;
      let k = 0;
      while (this._sprayAcc >= 1 && k++ < 6) {
        this._sprayAcc -= 1;
        L.frontPoint(lerp(-7, 7, Math.random()), _w);
        _w.y += h * (0.2 + Math.random() * 0.6);
        const fw = road.tangentAt(L.front.visS, _u).clone();
        this.debris(_w, 3, { speed: 2 + sp * 0.6, mud: 0.7, up: 0.8, dir: fw });
      }
    }
  }

  /** Mean wind + gusts, shared with the trees (render/impostor.js WIND: xy direction, z strength, w gust). */
  _updateWind(dt, rain) {
    const S = this.shared;
    let dx = 0.8, dz = 0.6, str = 0.3 + 0.35 * rain, gust;
    const W = this.WIND?.uWind?.value;
    if (W && (this.ctx.vegetation || W.w > 0)) {
      const l = Math.hypot(W.x, W.y) || 1; dx = W.x / l; dz = W.y / l; str = W.z; gust = W.w;
    } else {
      const t = this.time;
      gust = Math.max(0, Math.sin(t * 0.13) * 0.6 + Math.sin(t * 0.37 + 1.3) * 0.4) * 0.8;
    }
    // the valley wind veers slowly (+-25 deg over tens of seconds), so the rain slant changes over time
    const t = this.time, veer = 0.3 * Math.sin(t * 0.047) + 0.15 * Math.sin(t * 0.13 + 2.1);
    const cv = Math.cos(veer), sv = Math.sin(veer);
    [dx, dz] = [dx * cv - dz * sv, dx * sv + dz * cv];
    // strength 0..1 -> metres per second at rain height (valley wind 2..7 m/s in gusts)
    const speed = 0.8 + 4.6 * str * (1 + gust);
    S.wind.value.set(dx * speed, 0, dz * speed);
    S.disp.value.addScaledVector(S.wind.value, dt);
    const g = S.gust.value;
    g.x = dx; g.y = dz; g.z = 0.25 + 0.5 * gust; g.w += dt * (0.9 + speed * 0.12);
    if (this.dustSys) this.dustSys.uniforms.uWind.value.copy(S.wind.value).multiplyScalar(0.35);
    if (this.mist) this.mist.uniforms.uWind.value.copy(S.wind.value).multiplyScalar(0.5);
  }

  _updateRain(dt) {
    const { ctx } = this;
    const R = this.rain;
    if (!R) return;
    const env = ctx.env;
    let rain = env?.rain ?? 0.35;
    const cam = ctx.camera.position;
    const tunnel = this._tunnelCut(cam);
    rain *= 1 - tunnel;
    this.rainLevel = rain;
    const S = this.shared;
    // photometry: horizon sky radiance from the fog colour (it is matched to the HDRI horizon), ground ~ wet forest
    const fl = env?.flashLevel ?? 0;
    S.flash.value = fl;
    if (ctx.scene.fog?.color) S.skyL.value.copy(ctx.scene.fog.color).multiplyScalar(1.0);
    const px = 2 * Math.tan(THREE.MathUtils.degToRad(ctx.camera.fov) / 2) / Math.max(ctx.renderer.domElement.height || 720, 1);
    S.pix.value = px;
    // Marshall-Palmer: R (mm/h) from env.rain (0.35 -> ~4 mm/h, 0.5 -> ~7 mm/h, 1 -> ~36 mm/h)
    const Rmm = 1.2 * Math.pow(30, rain);
    const lam = 4.1 * Math.pow(Rmm, -0.21);
    const u = R.u;
    u.uLambda.value = lam;
    u.uTime.value = this.time;
    u.uCam.value.copy(cam);
    // near box (5 m): ~40 % of the instances at ~1/3 of the real density of > 0.8 mm drops -> gain ~3 (each streak
    // is nearly one real drop); far box (14 m): the rest, gain ~ real/rendered density for the unresolved 2.5-7 m shell
    const used = this.enabled.rain && rain > 0.01 ? Math.round(R.maxN * clamp(0.25 + rain * 0.9, 0, 1)) : 0;
    u.uCount.value = used;
    u.uCountA.value = Math.round(used * 0.4);
    const nReal = 8000 / lam * Math.exp(-lam * 0.8);                       // drops > 0.8 mm per m^3
    const volA = 125, volB = 14 * 12 * 14;
    u.uGainA.value = clamp(nReal / Math.max(u.uCountA.value / volA, 1), 1, 6);
    u.uGain.value = clamp(0.45 * nReal / Math.max((used - u.uCountA.value) / volB, 0.5), 4, 40);
    // car cabin: no rain inside it
    const car = ctx.car?.object;
    const inCar = !!car && (ctx.cameraRig?.mode === 'car-cockpit' || (car.position.distanceToSquared(cam) < 100));
    u.uCarOn.value = inCar ? 1 : 0;
    if (inCar) { car.updateMatrixWorld(); u.uCarInv.value.copy(car.matrixWorld).invert(); }
    R.mesh.visible = u.uCount.value > 0;
    this._updateVeil(rain);
    // splashes: impacts of the larger drops (> ~2 mm make a visible crown), spread on the ground around the camera
    if (this.enabled.splashes && rain > 0.02 && this.splashes) {
      this._splashAcc += dt * 1600 * rain;
      _a.set(0, 0, -1).applyQuaternion(ctx.camera.quaternion); _a.y = 0;
      const fwdA = _a.lengthSq() > 1e-4 ? Math.atan2(_a.z, _a.x) : 0;
      let n = 0;
      while (this._splashAcc >= 1 && n++ < 40) {
        this._splashAcc -= 1;
        const a = Math.random() < 0.85 ? fwdA + (Math.random() - 0.5) * 2.4 : Math.random() * Math.PI * 2;
        const r = 0.7 + Math.sqrt(Math.random()) * 8;
        const x = cam.x + Math.cos(a) * r, z = cam.z + Math.sin(a) * r;
        const gy = this._groundCached(x, z, cam.y);
        if (gy == null) continue;
        if (inCar) { _v.set(x, gy, z).applyMatrix4(u.uCarInv.value); if (Math.abs(_v.x) < 1 && Math.abs(_v.z) < 2.2) continue; }
        const big = Math.random();
        this.splashes.emit(_w.set(x, gy + 0.004, z), 0.035 + big * big * 0.07, 0.07 + big * 0.1);
      }
      if (n >= 40) this._splashAcc = Math.min(this._splashAcc, 2);
    }
    this._updateMist(dt, rain, inCar);
    this._updateDrips(dt, rain);
  }

  _updateVeil(rain) {
    const V = this.veil, ctx = this.ctx;
    if (!V) return;
    const du = ctx.post?.depthUniforms;
    const on = this.enabled.veil && rain > 0.03 && !!du;
    V.mesh.visible = on;
    if (!on) return;
    const u = V.u;
    u.tSceneDepth.value = du.tSceneDepth.value; u.sceneDepthRes.value.copy(du.sceneDepthRes.value); u.sceneCamNF.value.copy(du.sceneCamNF.value);
    const cam = ctx.camera;
    u.uInvProj.value.copy(cam.projectionMatrixInverse);
    u.uCamWorld.value.copy(cam.matrixWorld);
    u.uCam.value.copy(cam.position);
    u.uTime.value = this.time;
    u.uRain.value = clamp(rain * 1.6, 0, 1);
    u.uWindRel.value.copy(this.shared.wind.value).sub(this._camVel);
  }

  /** Low splash mist over the wet road: fine droplets from shattered drops hang in a thin sheet above the surface. */
  _updateMist(dt, rain, inCar) {
    const M = this.mist, ctx = this.ctx;
    if (!M || !this.enabled.mist || rain < 0.2) return;
    const cam = ctx.camera.position;
    this._mistAcc += dt * this._mistRate * (rain - 0.15);
    _a.set(0, 0, -1).applyQuaternion(ctx.camera.quaternion); _a.y = 0;
    if (_a.lengthSq() < 1e-4) _a.set(1, 0, 0);
    _a.normalize();
    let n = 0;
    while (this._mistAcc >= 1 && n++ < 3) {
      this._mistAcc -= 1;
      const ahead = 6 + Math.random() * 40, side = (Math.random() - 0.5) * (6 + ahead * 0.8);
      const x = cam.x + _a.x * ahead - _a.z * side, z = cam.z + _a.z * ahead + _a.x * side;
      // only over the asphalt / shoulders (the splash layer of a paved surface); valley wisps belong to the fog
      if (ctx.road) { const pr = ctx.road.project(_w.set(x, cam.y, z), this._prM || (this._prM = {})); if (pr.d < -4.2 || pr.d > 3.6) continue; }
      const gy = this._groundAt(x, z, cam.y + 2);
      if (gy == null) continue;
      _w.set(x, gy + 0.15 + Math.random() * 0.2, z);
      if (ctx.terrain?.insideTunnel && ctx.terrain.insideTunnel(_w) > 0.05) continue;   // (QA) no rain splash mist in the tunnel
      _v.set((Math.random() - 0.5) * 0.3, 0.02, (Math.random() - 0.5) * 0.3);
      const c = this.shared.skyL.value;
      _col.setRGB(c.r * 0.95, c.g * 0.95, c.b * 0.95);
      const s = 1.3 + Math.random() * 1.8;
      // (QA) opacity x0.6: at ~4 mm/h the splash layer is barely there on real footage; each sheet read as a pale disc
      M.emit(_w, _v, 5 + Math.random() * 4, s * 0.8, s * 1.3, (0.065 + Math.random() * 0.05) * rain, _col, gy, 0, 0, 3.2, 0.8);
    }
  }

  /** Big drops falling from spruce crowns and rock ledges near the camera. Drip points are re-seeded as we move. */
  _updateDrips(dt, rain) {
    const ctx = this.ctx, road = ctx.road, D = this.drips;
    if (!this.drops || !this.enabled.drips || rain < 0.05 || !road) return;
    const cam = ctx.camera.position;
    const pr = road.project(cam, this._prD || (this._prD = {}));
    if (pr.dist > 60) return;
    // drop far drip points
    for (let i = D.length - 1; i >= 0; i--) if (D[i].p.distanceToSquared(cam) > 32 * 32) D.splice(i, 1);
    // seed up to 2 new points per frame (raycasts)
    for (let k = 0; k < 2 && D.length < 44; k++) {
      const p = this._seedDrip(pr.s, cam);
      if (p) D.push(p);
    }
    for (const d of D) {
      d.t -= dt;
      if (d.t > 0) continue;
      d.t += d.period * (0.6 + Math.random() * 0.8) / (0.5 + rain);
      const h = d.p.y - d.ground;
      const tHit = Math.sqrt(2 * Math.max(h, 0.05) / 9.81) * 1.05;
      this.drops.emit(d.p, _v.set((Math.random() - 0.5) * 0.1, 0, (Math.random() - 0.5) * 0.1), tHit + 0.3, d.ground, d.dmm, 0.08, 1.4);
      if (this.splashes && d.p.distanceToSquared(cam) < 20 * 20) this.splashes.emit(_w.set(d.p.x, d.ground + 0.004, d.p.z), 0.07 + d.dmm * 0.01, 0.14, tHit);
    }
  }

  _seedDrip(s0, cam) {
    const ctx = this.ctx, road = ctx.road, phys = ctx.physics;
    if (!phys?.world) return null;
    const near = this._treesNear(cam, 24);
    if (near.length && Math.random() < 0.55) {
      // a tree near the camera: drips fall from the lower branch tips (spruce crown radius ~2-3 m)
      for (let tries = 0; tries < 3; tries++) {
        const t = near[Math.floor(Math.random() * near.length)];
        const sc = t[3] || 1, a = Math.random() * Math.PI * 2, r = (0.7 + Math.random() * 1.9) * sc;
        const x = t[0] + Math.cos(a) * r, z = t[2] + Math.sin(a) * r;
        const y = t[1] + (2.5 + Math.random() * 6) * sc;
        const gy = this._groundAt(x, z, y - 4);
        if (gy == null || y - gy < 1) continue;
        return { p: new THREE.Vector3(x, y, z), ground: gy, period: 1.2 + Math.random() * 3.5, t: Math.random() * 3, dmm: 4 + Math.random() * 1.5 };
      }
    }
    // rock cut: find the face by a horizontal ray toward the slope, drip from just in front of it
    const s = s0 + (Math.random() - 0.3) * 50;
    const up = 2 + Math.random() * 9;
    const o = road.worldAt(s, 2.6, _b); o.y += up;
    const left = road.leftAt(s, _c3);
    const hit = phys.raycast(o, left, 8, { groups: groups(G.ALL, G.STATIC) });
    if (!hit || Math.abs(hit.normal.y) > 0.75) return null;
    const p = hit.point.clone().addScaledVector(hit.normal, 0.1);
    const gy = this._groundAt(p.x, p.z, p.y - 3.1);
    if (gy == null || p.y - gy < 0.8) return null;
    return { p, ground: gy, period: 0.8 + Math.random() * 3, t: Math.random() * 2, dmm: 4 + Math.random() * 1.5 };
  }

  /** Droplets flung off the wet tread (complements vehicle.js TyreSpray's mist puffs with resolved drops). */
  _updateWheelDrops(dt) {
    const ctx = this.ctx, car = ctx.car, vc = car?.controller;
    if (!this.drops || !this.enabled.wheelDrops || !vc || !car.body || !car.object || dt <= 0) return;
    const spd = Math.abs(car.speed || 0);
    const wet = ctx.env?.wetness ?? 0.75;
    if (spd < 2.5 || wet < 0.2 || (this._tunnel ?? 0) > 0.5) { this._wheelAcc.fill(0); return; }
    if (car.object.position.distanceToSquared(ctx.camera.position) > 60 * 60) return;
    const q = car.object.quaternion;
    const fwd = _a.set(0, 0, 1).applyQuaternion(q), left = _b.set(1, 0, 0).applyQuaternion(q);
    const lv = car.body.linvel();
    const dir = (car.speed || 0) >= 0 ? 1 : -1;
    const n = Math.min(4, car.wheels?.length ?? 4);
    for (let i = 0; i < n; i++) {
      if (!vc.wheelIsInContact?.(i)) { this._wheelAcc[i] = 0; continue; }
      const surf = car._surf?.[i];
      const sm = surf === 'asphalt' || surf == null ? 1 : surf === 'rock' || surf === 'wood' ? 0.6 : surf === 'gravel' ? 0.5 : 0.35;
      const w = car.wheels?.[i];
      const front = w?.front ?? i < 2, isLeft = w?.left ?? (i % 2 === 0);
      this._wheelAcc[i] += dt * (front ? 60 : 100) * sm * clamp((spd - 2.5) / 12, 0, 1.4) * wet;
      const cp = vc.wheelContactPoint?.(i);
      if (!cp) continue;
      let k = 0;
      while (this._wheelAcc[i] >= 1 && k++ < 5) {
        this._wheelAcc[i] -= 1;
        // release angle behind the contact patch: ground-frame velocity = car velocity + tread velocity
        const th = 0.15 + Math.random() * 0.9;
        const vs = spd * (0.7 + Math.random() * 0.3);
        const side = isLeft ? 1 : -1;
        _v.set(lv.x, 0, lv.z).addScaledVector(fwd, -dir * vs * Math.cos(th)).addScaledVector(left, side * (0.2 + Math.random() * 0.9));
        _v.y = vs * Math.sin(th) * 0.75;
        _w.set(cp.x, cp.y, cp.z).addScaledVector(fwd, -dir * (0.25 + 0.2 * Math.random())).addScaledVector(left, side * 0.05);
        _w.y += 0.05 + 0.25 * Math.sin(th);
        const dmm = 0.5 + Math.pow(Math.random(), 2) * 2.2;
        this.drops.emit(_w, _v, 1.2, cp.y - 0.02, dmm, 3.2 / dmm, 20);
      }
    }
  }

  /** 1 under the tunnel roof (sharp: the rain stops a few metres past the portal), 0 in the open. */
  _tunnelCut(cam) {
    const { ctx } = this;
    const road = ctx.road;
    const tun = ctx.terrain?.insideTunnel?.(cam) ?? null;
    if (!road) return tun ?? 0;
    const t0 = road.markers?.tunnel ?? 1150, t1 = road.markers?.tunnelEnd ?? 1260;
    const pr = road.project(cam, this._prT || (this._prT = {}));
    if (pr.s < t0 - 1 || pr.s > t1 + 5 || Math.abs(pr.d) > 5.5 || pr.dy > 6.8) return (this._tunnel = 0); // outside, or above the roof
    if (tun === 0) return (this._tunnel = 0);                    // terrain says daylight (not in the tube)
    return (this._tunnel = clamp((pr.s - (t0 + 0.5)) / 2.5, 0, 1)); // 0 at the portal plane, 1 at +3 m
  }

  _indexTrees(trees) {
    if (!trees?.length) return;
    const C = 24, grid = new Map();
    for (const t of trees) {
      const k = Math.floor(t[0] / C) * 4096 + Math.floor(t[2] / C);
      let a = grid.get(k); if (!a) grid.set(k, (a = [])); a.push(t);
    }
    this._treeGrid = { C, grid };
  }
  /** Trees within r (<= grid cell) of p, cached per 4 m of camera movement. */
  _treesNear(p, r) {
    const T = this._treeGrid;
    if (!T) return [];
    const kx = Math.floor(p.x / 4), kz = Math.floor(p.z / 4);
    if (this._tnKey === kx * 100003 + kz) return this._tn;
    this._tnKey = kx * 100003 + kz;
    const out = [], cx = Math.floor(p.x / T.C), cz = Math.floor(p.z / T.C);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
      const a = T.grid.get((cx + i) * 4096 + cz + j);
      if (a) for (const t of a) if ((t[0] - p.x) ** 2 + (t[2] - p.z) ** 2 < r * r) out.push(t);
    }
    return (this._tn = out);
  }
  /** Ground height on a 0.35 m lattice, cached (splashes need hundreds of heights per second). */
  _groundCached(x, z, fromY) {
    const C = 0.35, ix = Math.round(x / C), iz = Math.round(z / C);
    const k = ix * 1048576 + iz;
    const M = this._gc || (this._gc = new Map());
    let v = M.get(k);
    if (v === undefined) {
      if (M.size > 12000) M.clear();
      v = this._groundAt(ix * C, iz * C, fromY);
      M.set(k, v);
    }
    return v;
  }

  _groundAt(x, z, fromY) {
    const phys = this.ctx.physics;
    if (!phys?.world) {
      const road = this.ctx.road;
      if (!road) return null;
      const pr = road.project(_u.set(x, 0, z), this._pr);
      return Math.abs(pr.d) < 4 ? road.pointAt(pr.s, _u).y : null;
    }
    const h = phys.raycast(_u.set(x, fromY + 3, z), _down, 14, { groups: groups(G.ALL, G.STATIC | G.PROP | G.ROCK) });
    return h && h.normal.y > 0.6 ? h.point.y : null;
  }

  update(dt) {
    const { ctx } = this;
    this._impactsThisFrame = 0;
    this.time += dt;
    this._boostT = Math.max(0, (this._boostT || 0) - dt);
    const cap = this._boostT > 0 ? 160 : 60;
    this._dustBudget = Math.min((this._dustBudget ?? 40) + dt * (this._boostT > 0 ? 200 : 45), cap);
    const cam = ctx.camera.position;
    if (dt > 0) {
      _v.subVectors(cam, this._camPrev).divideScalar(dt);
      if (_v.lengthSq() > 60 * 60) _v.set(0, 0, 0);    // teleport
      this._camVel.lerp(_v, Math.min(1, dt * 6));
    }
    this._camPrev.copy(cam);
    this._updateWind(dt, this.rainLevel ?? ctx.env?.rain ?? 0.35);
    // splash height cache: forget it now and then (planks placed, tree cut, rocks settled)
    this._gcT = (this._gcT || 0) + dt;
    if (this._gcT > 6) { this._gcT = 0; this._gc?.clear(); }
    // sun direction / world up in view space for the puff lighting
    for (const D of [this.dustSys, this.mist]) {
      if (!D) continue;
      const sd = ctx.env?.sunDirection;
      if (sd) D.uniforms.uSunV.value.copy(sd).transformDirection(ctx.camera.matrixWorldInverse);
      D.uniforms.uUpV.value.set(0, 1, 0).transformDirection(ctx.camera.matrixWorldInverse);
      const fl = ctx.env?.flashLevel ?? 0;
      const inT = ctx.terrain?.insideTunnel?.(cam) ?? 0;
      D.uniforms.uAmb.value.setRGB(0.62 + fl, 0.66 + fl, 0.72 + fl).multiplyScalar(1 - 0.85 * inT);
      D.uniforms.uSun.value.setRGB(0.55, 0.52, 0.47).multiplyScalar(1 - inT);
      // soft particles: share post's opaque-depth copy (null with ?nopost)
      if (D.soft) {
        const du = ctx.post?.depthUniforms;
        if (du) { D.uniforms.tSceneDepth.value = du.tSceneDepth.value; D.uniforms.sceneDepthRes.value.copy(du.sceneDepthRes.value); D.uniforms.sceneCamNF.value.copy(du.sceneCamNF.value); }
      }
    }
    if (dt > 0) this._updateFront(dt);
    this.dustSys?.update(dt);
    this.mist?.update(dt);
    this.chips?.update(dt);
    this._updateRain(dt);
    if (dt > 0) this._updateWheelDrops(dt);
    this.drops?.update(dt);
    this.splashes?.update(dt);
    if (this.enabled.windshield) this.windshield?.update(dt, this.time, this.rainLevel ?? 0, ctx.env?.rain ?? this.rainLevel ?? 0);
    else if (this.windshield?.mesh) { this.windshield.mesh.visible = false; if (this.colorCopy) this.colorCopy.want = false; }
  }

  dispose() {
    this._off?.();
    for (const s of [this.dustSys, this.mist, this.chips, this.rain, this.drops, this.splashes, this.veil]) if (s?.mesh) this.ctx.scene.remove(s.mesh);
    this.colorCopy?.dispose();
    this.windshield?.mesh?.parent?.remove(this.windshield.mesh);
    this.carSplash?.mesh?.parent?.remove(this.carSplash.mesh);
  }
}

const _col = new THREE.Color();
