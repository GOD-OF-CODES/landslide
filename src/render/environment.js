// LANDSLIDE environment (RENDER workstream): sky dome, image-based light, sun + shadows, height fog, weather state.
//
// Public API (see DESIGN.md "env"):
//   env.sun            DirectionalLight (casts shadows; shadow frustum follows ctx.camera with texel snapping)
//   env.hemi           HemisphereLight (weak cool fill; boosted by lightning)
//   env.sky            the sky-dome Mesh (8k HDRI crop, horizon haze matched to the fog)
//   env.envMap         PMREM texture (also scene.environment), already rotated into world space
//   env.sunDirection   Vector3, unit vector pointing TOWARD the sun (world space)
//   env.wetness 0..1   (default 0.75)   env.rain 0..1 (default 0.35)   -> mirrored into fogUniforms.hEnv
//                      env.rain also sets the grey rain veil of the fog (fog.rainExtinction; faded out inside the tunnel)
//   env.look.skyScud   strength of the low ragged scud fragments drawn on the sky dome (skyUniforms.skyScud)
//   env.lightning(intensity=1)          flash: sky + light + fog boost, optional thunder via ctx.audio ('thunder')
//   env.autoLightning  0 = off (default), else mean seconds between random distant strikes
//   env.fogUniforms    shared height-fog uniforms (+ hEnv); see src/render/fog.js header for custom shaders
//   env.fog            live fog settings object (FOG_DEFAULTS shape); call env.applyFog() after editing it
//   env.materials      helpers from src/render/materials.js (pbrMaterial, triplanarPatch, applyWetness, applyRipples)
//   env.flashLevel     current lightning flash 0..1 (read-only)
//   env.envIntensity / env.sun.userData.baseIntensity   base values (lightning modulates around them every frame)
//   env.shadowCull(obj, rule)          distance-based castShadow toggling for obj and its descendants (see below)
//   Tunnel: all built-in lit materials lose sky light (IBL + hemi) inside the road tunnel (markers.tunnel..tunnelEnd),
//           so only the tunnel's own lamps light it. Opt out per material with defines.HTUNNEL_OFF.
import * as THREE from 'three';
import { floatRTSupport } from './materials.js';
import { liteFogChunk, installFog, createFogUniforms, writeFogUniforms, FOG_DEFAULTS, FOG_GLSL_UNIFORMS, FOG_GLSL_FUNCS, MIST_DEFAULTS, rainExtinction } from './fog.js';
import * as materials from './materials.js';

const DEG = Math.PI / 180;

// Look parameters (linear HDR units, same scale as the HDRI; exposure/tone mapping happen in post).
const LOOK = {
  sunAzimuthDeg: 28,          // world azimuth (atan2(z, x)) of the bright cloud hole: ahead (+X) and over the valley (+Z)
  sunColor: [1.0, 0.95, 0.88],
  sunIntensity: 0.6,                 // overcast: the "sun" is only the brighter cloud region (soft shaping, faint shadows)
  envIntensity: 0.95,
  envTint: [0.9, 0.96, 1.07],        // cool, bluish ambient
  envGround: [0.05, 0.054, 0.05],    // lower hemisphere radiance for IBL (wet dark forest floor, rock: albedo ~.05)
  // valley horizon in the IBL capture: the road sits in a deep forested valley, so the lower sky is hidden behind
  // dark wooded slopes (uphill, -Z, they rise ~30 deg; across the valley, +Z, the far range stands ~9 deg high).
  // tan(elevation) of the ridge line uphill / across the valley / along the road; distance (m) of those slopes.
  envRidge: [1, 0.58, 0.17, 0.11], envRidgeDist: [420, 2100, 900],
  envRidgeCol: [0.028, 0.034, 0.031],  // wet spruce forest: albedo ~.04 x (sky irradiance ~2.8 on a steep slope) / pi
  envCaptureHeight: 70,              // camera height used for the haze baked into the IBL
  skyIntensity: 1.2,                 // visible dome vs IBL: the valley walls hide the lower sky from the IBL, not from the eye
  skyTint: [0.97, 0.99, 1.03],
  skyGround: [0.3, 0.32, 0.34],      // below-horizon fill for the visible dome (terrain normally covers it)
  skyFogDist: 2600,                  // metres of fog applied to the sky (~ distance of the far ridge): horizon haze
  skyVar: 1.3,                       // procedural drifting cloud structure on the dome (0 = plain HDRI)
  skyCloudH: 420,                    // cloud deck height above the eye: upward rays stop there (less haze overhead)
  skyScud: 0.6,                      // low ragged scud fragments drifting under the deck (0 = off)
  fogColor: [0.6, 0.68, 0.8],        // haze inscatter (linear) ~ horizon sky radiance, cooled
  hemiSky: [0.62, 0.7, 0.82], hemiGround: [0.06, 0.065, 0.06], hemiIntensity: 0.07,
  shadowDistance: 150,               // how far toward the sun casters are captured (m). The sun (~35 deg up) sits over the
                                     // valley, so nothing tall stands up-sun of the road: 28 m trees throw ~40 m shadows.
  // distance-based caster culling (see shadowCull): an object stops casting once the camera is farther than
  // base + k * boundingRadius from its bounding sphere. Small props fade out of the shadow pass first.
  shadowCullBase: 30, shadowCullK: 18, shadowCullTerrain: 45,
};

const SKY_VERT = /* glsl */`
varying vec3 vDir;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vDir = wp.xyz - cameraPosition;
  gl_Position = projectionMatrix * viewMatrix * wp;
  #ifndef ENV_CAPTURE
    // pin to the far plane: the dome only fills pixels nothing else covered, whatever the terrain extent
    gl_Position.z = gl_Position.w;  // (same trick as three.js backgroundCube)
  #endif
}`;

const SKY_FRAG = /* glsl */`
uniform sampler2D tSky;
uniform mat3 skyRot;       // world dir -> texture dir
uniform vec4 skyEnc;       // x intensity/k (or plain intensity for HDR), y shoulder c, z v at bottom edge of crop, w lightning boost
uniform vec3 skyTint;
uniform vec3 skyGround;
uniform float skyFogDist;
uniform float skyCloudH;   // height of the cloud deck above the camera: limits the haze path for upward rays
uniform float skyGroundH;  // height above the ground seen by downward rays: limits their haze path
uniform float envCamY;
uniform float skyVar;      // strength of the procedural cloud-structure modulation
uniform float skyScud;     // coverage strength of the low scud fragments (0 = none)
uniform vec4 envRidge;     // (ENV_CAPTURE) x strength, y/z/w tan(elevation) of the ridge: uphill / valley / along road
uniform vec3 envRidgeD;    // (ENV_CAPTURE) distance of those slopes (m)
uniform vec3 envRidgeCol;  // (ENV_CAPTURE) forest radiance
uniform vec3 fogColor;
${FOG_GLSL_UNIFORMS}
${FOG_GLSL_FUNCS}
varying vec3 vDir;
void main() {
  vec3 d = normalize(vDir);
  vec3 ld = skyRot * d;
  float u = atan(ld.z, ld.x) * 0.15915494 + 0.5;
  float v = asin(clamp(ld.y, -1.0, 1.0)) * 0.31830989 + 0.5;
  #ifdef SKY_HDR
    vec3 L = texture2D(tSky, vec2(u, v)).rgb * skyEnc.x;
  #else
    vec2 uv = vec2(u, (v - skyEnc.z) / (1.0 - skyEnc.z));
    // seam-free mip selection (Tarini): use the derivative of whichever u parametrisation is continuous here
    vec2 dx = dFdx(uv), dy = dFdy(uv);
    float u2 = fract(u + 0.5);
    float dx2 = dFdx(u2), dy2 = dFdy(u2);
    if (abs(dx2) + abs(dy2) < abs(dx.x) + abs(dy.x)) { dx.x = dx2; dy.x = dy2; }
    vec3 e = textureGrad(tSky, uv, dx, dy).rgb;
    vec3 L = e / max(1.0 - skyEnc.y * e, 0.02) * skyEnc.x;   // undo the highlight shoulder, back to HDR units
  #endif
  L *= skyTint;
  float scud = 0.0;
  #ifndef ENV_CAPTURE
  // drifting rain-cloud structure over the (rather uniform) overcast HDRI: darker nimbostratus masses and brighter
  // thin patches, projected on a flat cloud deck so they converge to the horizon like real cloud
  if (ld.y > 0.0 && skyVar > 0.0) {
    vec2 pp = d.xz / (d.y + 0.06);
    vec2 w = hfogWind.xy * hfogMist2.z * 0.004;
    float n = 0.65 * hfog_fbm(pp * 0.9 + w) + 0.35 * hfog_fbm(pp * 3.3 - w * 1.8 + 7.0);
    float m = n - 0.5;
    L *= 1.0 + m * skyVar * smoothstep(0.015, 0.22, d.y) * (m < 0.0 ? 1.4 : 0.7);
    // scud (pannus): ragged low cloud fragments a few hundred metres up, under the nimbostratus deck; darker than the
    // deck (seen from below, lit only from above), fast-moving, frayed edges. The signature of a real rain sky.
    // Applied after the haze below: they are much nearer than the deck (only ~1/3 of the rain veil lies in front).
    if (skyScud > 0.0) {
      vec2 ps = d.xz / (d.y + 0.035) * 2.6;
      vec2 ws = hfogWind.xy * hfogMist2.z * 0.09;
      float s1 = hfog_fbm(ps + ws + 11.0);
      float s2 = hfog_fbm(ps * 3.6 + ws * 1.6 - 5.0);
      float sc = smoothstep(0.47, 0.62, s1 * 0.7 + s2 * 0.3 + 0.1 * (hfog_vnoise(ps * 11.0 + ws * 2.0) - 0.5));
      sc *= smoothstep(0.04, 0.16, d.y) * (1.0 - 0.5 * smoothstep(0.4, 0.85, d.y));
      scud = sc * skyScud;
    }
  }
  #endif
  // the HDRI below the horizon is a blurred fake ground: fade to our ground colour
  L = mix(L, skyGround, smoothstep(0.0, -0.14, ld.y));
  L *= 1.0 + skyEnc.w;
  vec3 camP = cameraPosition;
  #ifdef ENV_CAPTURE
    camP = vec3(0.0, envCamY, 0.0);
  #endif
  // identical fog function as every surface -> distant ridges melt into the horizon seamlessly
  float tSky = d.y > 0.0 ? min(skyFogDist, skyCloudH / max(d.y, 1e-3)) : min(skyFogDist, skyGroundH / max(-d.y, 1e-3));
  #ifdef ENV_CAPTURE
  // forested valley walls hide the lower sky (IBL + the reflection fallback of wet surfaces): a ridge line that is
  // high on the uphill side and low across the valley, with summit shapes and tree-top jaggies, at its own distance
  if (envRidge.x > 0.0 && d.y > -0.2) {
    float hl = length(d.xz) + 1e-4;
    vec2 az = d.xz / hl;
    float upH = smoothstep(-0.25, 0.9, -az.y), vaH = smoothstep(-0.25, 0.9, az.y);
    float base = mix(mix(envRidge.w, envRidge.y, upH), envRidge.z, vaH);
    float n = hfog_fbm(az * 2.3 + 5.0) - 0.5;
    float h = base * (1.0 + 1.1 * n) + 0.014 * (hfog_vnoise(az * 70.0) - 0.5);
    float e = d.y / hl;
    float cover = (1.0 - smoothstep(h - 0.01, h + 0.01, e)) * envRidge.x;
    if (cover > 0.0) {
      float tR = mix(mix(envRidgeD.z, envRidgeD.x, upH), envRidgeD.y, vaH);
      // rock bands and clearings between the trees; the upper slopes catch a little more sky
      vec3 rc = envRidgeCol * (0.75 + 0.9 * hfog_fbm(az * 17.0 + e * 9.0)) * (0.8 + 0.5 * clamp(e / max(h, 1e-3), 0.0, 1.0));
      vec3 R = hfog_apply(rc, d * tR, camP);
      L = mix(L, R, cover);
      tSky = mix(tSky, 0.0, cover);   // the ridge already carries its own aerial perspective
    }
  }
  #endif
  L = hfog_apply(L, d * tSky, camP);
  L *= mix(vec3(1.0), vec3(0.62, 0.64, 0.68), clamp(scud, 0.0, 1.0));
  gl_FragColor = vec4(L, 1.0);
}`;

// (LOWEND) Low preset sky: the same HDRI, haze and cloud structure with fewer noise taps (the deck structure's fine
// octave and the scud's detail layer are single value-noise taps; the fog functions use the lite fog chunk).
function skyFragLite() {
  let f = liteFogChunk(SKY_FRAG);
  for (const [a, b] of [
    ['float n = 0.65 * hfog_fbm(pp * 0.9 + w) + 0.35 * hfog_fbm(pp * 3.3 - w * 1.8 + 7.0);',
      'float n = 0.65 * hfog_fbm(pp * 0.9 + w) + 0.35 * hfog_vnoise(pp * 3.3 - w * 1.8 + 7.0);'],
    ['float s2 = hfog_fbm(ps * 3.6 + ws * 1.6 - 5.0);', 'float s2 = hfog_vnoise(ps * 3.6 + ws * 1.6 - 5.0);'],
    ['float sc = smoothstep(0.47, 0.62, s1 * 0.7 + s2 * 0.3 + 0.1 * (hfog_vnoise(ps * 11.0 + ws * 2.0) - 0.5));',
      'float sc = smoothstep(0.47, 0.62, s1 * 0.7 + s2 * 0.3);'],
  ]) { if (f.includes(a)) f = f.split(a).join(b); else console.warn('[env] lite sky substitution not found:', a.slice(0, 50)); }
  return f;
}

const _v = new THREE.Vector3(), _f = new THREE.Vector3(), _c = new THREE.Vector3(), _x = new THREE.Vector3(), _y = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _sph = new THREE.Sphere();

export default class Environment {
  constructor(ctx) {
    this.ctx = ctx;
    this._wetness = 0.75;
    this._rain = 0.35;
    this.flashLevel = 0;
    this._flashes = [];
    // mean seconds between random distant strikes (0 = off). On by default for the storm mood, off for fixed-camera
    // screenshots (?cam / ?camS) so A/B shots stay comparable. GAME may change it at any time.
    this.autoLightning = ctx.flags?.fixedCam ? 0 : 70;
    this._nextStrike = 0;
    this.look = { ...LOOK };
    // (LOWEND) shorter shadow-caster distances on the lighter presets (the shadow map also covers less: config)
    const qk0 = ctx.config?.quality?.key;
    if (qk0 === 'low') Object.assign(this.look, { shadowCullBase: 16, shadowCullK: 10, shadowCullTerrain: 28 });
    else if (qk0 === 'medium') Object.assign(this.look, { shadowCullBase: 24, shadowCullK: 14, shadowCullTerrain: 38 });
    this.fog = JSON.parse(JSON.stringify(FOG_DEFAULTS));
    this.fog.rainExtinction = rainExtinction(this._rain);   // grey veil of the falling rain (follows env.rain)
    this.mist = JSON.parse(JSON.stringify(MIST_DEFAULTS));   // volumetric mist settings (rendered by post MistPass)
    // the analytic valley mist sheets are only the fallback where post has no volumetric mist (low quality, no post);
    // otherwise they would cost two fBm sheets per pixel in every material for a flat "milk floor" look
    const qk = ctx.config?.quality?.key;
    if (qk === 'low' || ctx.flags?.nopost || /(^|[?&,=])nomist/.test(location.search)) this.fog.mistDensity = 0.008;
    // Fog chunks must be installed before any material compiles: do it right away.
    this.fogUniforms = installFog(createFogUniforms(), { lite: qk === 'low' });
    this.uniforms = this.fogUniforms; // alias
    this.materials = materials;
    materials.setEnvUniform(this.fogUniforms.hEnv);
    this.sunDirection = new THREE.Vector3(0.5, 0.6, 0.3).normalize();
    this.sun = null; this.hemi = null; this.sky = null; this.envMap = null;
    // shadow-caster culling state (see shadowCull)
    this._scRules = new Map(); this._scState = new WeakMap(); this._scList = []; this._scT = -1e9; this._scListAge = -1e9;
    this.applyFog();
  }

  get wetness() { return this._wetness; }
  set wetness(v) { this._wetness = THREE.MathUtils.clamp(+v || 0, 0, 1); this.fogUniforms.hEnv.value.y = this._wetness; }
  get rain() { return this._rain; }
  set rain(v) {
    this._rain = THREE.MathUtils.clamp(+v || 0, 0, 1);
    this.fogUniforms.hEnv.value.z = this._rain;
    this.fog.rainExtinction = rainExtinction(this._rain);
    this.fogUniforms.hfogRain.value.x = this.fog.rainExtinction * (1 - 0.9 * (this._tunnelIn || 0));
  }

  /** Push this.fog (settings) into the shared uniforms. */
  applyFog() {
    writeFogUniforms(this.fogUniforms, this.fog, this.ctx.time?.now ?? 0);
    const s = this.sunDirection, S = this.fogUniforms.hfogSun.value;
    S.x = s.x; S.y = s.y; S.z = s.z;
  }

  async init() {
    const { scene, renderer, config, assets } = this.ctx;
    const q = config.quality;
    const L = this.look;

    // ---- sky metadata (sun direction, encoding) --------------------------------------------------------------
    let meta = null;
    try { meta = await assets.json('assets/sky/sky.json'); } catch (e) { console.warn('[env] sky.json missing, using defaults', e); }
    this.skyMeta = meta || {
      files: { '8k': { file: 'sky_8k.jpg', width: 8192, height: 2276 }, '4k': { file: 'sky_4k.jpg', width: 4096, height: 1138 } },
      elevMinDeg: -10.02, encode: { k: 0.3133, c: 0.4029 }, sunDirTexSharp: [0.804, 0.571, -0.165],
    };
    const sd = this.skyMeta.sunDirTexSharp || this.skyMeta.sunDirTex;
    const texAz = Math.atan2(sd[2], sd[0]);
    // world = Ry(rotY) * tex; Ry(a) lowers atan2(z,x) by a  ->  rotY = texAz - targetAz
    this.rotY = texAz - L.sunAzimuthDeg * DEG;
    const rot4 = new THREE.Matrix4().makeRotationY(this.rotY);
    this.sunDirection.set(sd[0], sd[1], sd[2]).applyMatrix4(rot4).normalize();
    this.skyRot = new THREE.Matrix3().setFromMatrix4(rot4).transpose(); // world -> texture

    // ---- scene fog (FogExp2 instance: enables USE_FOG and keeps N8AO's fog-aware AO happy) ----------------------
    // The height-fog chunks ignore fogDensity; N8AO reads it as its AO distance fade (1 - exp(-(d*density)^2)):
    // 0.0095 keeps contact AO near the camera and fades it out by ~100 m (no dark halos on distant cliffs/slots).
    scene.fog = new THREE.FogExp2(new THREE.Color().setRGB(...L.fogColor), 0.0095);
    scene.background = null;
    this.applyFog();

    // ---- visible sky dome ------------------------------------------------------------------------------------
    const big = q.shadowMapSize >= 4096; // ultra/high
    const skyFile = this.skyMeta.files[big ? '8k' : '4k'] || this.skyMeta.files['8k'];
    let skyTex = null;
    try {
      skyTex = await assets.texture('assets/sky/' + skyFile.file, { srgb: true, repeat: true, anisotropy: 8 });
      skyTex.wrapS = THREE.RepeatWrapping; skyTex.wrapT = THREE.ClampToEdgeWrapping;
      skyTex.minFilter = THREE.LinearMipmapLinearFilter; skyTex.magFilter = THREE.LinearFilter;
      skyTex.generateMipmaps = true; skyTex.needsUpdate = true;
    } catch (e) { console.warn('[env] sky texture failed', e); }
    const enc = this.skyMeta.encode;
    const vMin = 0.5 + this.skyMeta.elevMinDeg / 180;
    this.skyUniforms = {
      tSky: { value: skyTex },
      skyRot: { value: this.skyRot },
      skyEnc: { value: new THREE.Vector4(L.skyIntensity / enc.k, enc.c, vMin, 0) },
      skyTint: { value: new THREE.Color().setRGB(...L.skyTint) },
      skyGround: { value: new THREE.Color().setRGB(...L.skyGround) },
      skyFogDist: { value: L.skyFogDist },
      skyCloudH: { value: L.skyCloudH },
      skyGroundH: { value: 450 },
      envCamY: { value: L.envCaptureHeight },
      skyVar: { value: L.skyVar },
      skyScud: { value: L.skyScud },
      fogColor: { value: scene.fog.color },
      ...this._sharedFogUniforms(),
    };
    const skyMat = new THREE.ShaderMaterial({
      name: 'SkyDome', uniforms: this.skyUniforms, vertexShader: SKY_VERT, fragmentShader: q.key === 'low' ? skyFragLite() : SKY_FRAG,
      side: THREE.BackSide, depthWrite: false, depthTest: true, depthFunc: THREE.LessEqualDepth, fog: false, toneMapped: false,
    });
    if (!skyTex) skyMat.defines = { SKY_HDR: '' };
    const far = this.ctx.camera.far || 6000;
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(Math.min(1000, far * 0.5), 64, 32), skyMat);
    this.sky.name = 'sky_dome';
    this.sky.frustumCulled = false;
    this.sky.renderOrder = 1e6; // after all opaques: depth test rejects every covered pixel
    this.sky.castShadow = this.sky.receiveShadow = false;
    this.sky.matrixAutoUpdate = true;
    scene.add(this.sky);

    // ---- image based lighting: PMREM captured from our own sky shader (HDR source, rotated, ground, haze) ----
    await this._buildEnvMap(q);

    // ---- lights ------------------------------------------------------------------------------------------------
    // no IBL (see _buildEnvMap): the hemisphere light carries the overcast sky's irradiance instead (pi x the mean
    // sky radiance x envIntensity, matched by eye against the IBL render)
    if (this._noIBL) this.look.hemiIntensity = L.hemiIntensity + Math.PI * 0.62 * L.envIntensity;
    this.hemi = new THREE.HemisphereLight(new THREE.Color().setRGB(...L.hemiSky), new THREE.Color().setRGB(...L.hemiGround), L.hemiIntensity);
    this.hemi.name = 'env_hemi';
    scene.add(this.hemi);

    this.sun = new THREE.DirectionalLight(new THREE.Color().setRGB(...L.sunColor), L.sunIntensity);
    this.sun.name = 'env_sun';
    this.sun.castShadow = true;
    const sh = this.sun.shadow;
    const size = q.sunShadowMapSize || q.shadowMapSize || 2048;   // (sunShadowMapSize: see config.js)
    sh.mapSize.set(size, size);
    this.shadowRadius = q.shadowRadius || 50;       // world half-extent of the shadow frustum (m)
    const c = sh.camera;
    c.left = c.bottom = -this.shadowRadius; c.right = c.top = this.shadowRadius;
    c.near = 1; c.far = L.shadowDistance + this.shadowRadius * 2.5;
    c.updateProjectionMatrix();
    const texel = (2 * this.shadowRadius) / size;
    // depth units: [near, far] -> [0, 1]; keep the constant bias to ~1 cm, rely on the normal offset (scaled by texel size)
    sh.bias = -0.012 / (c.far - c.near);
    sh.normalBias = Math.max(0.02, texel * 1.4);
    sh.radius = size >= 4096 ? 4 : 3;
    sh.intensity = 1.0;
    this.sun.userData.baseIntensity = L.sunIntensity;
    scene.add(this.sun, this.sun.target);
    this.envIntensity = L.envIntensity;
    scene.environmentIntensity = this.envIntensity;
    this._updateSunAndShadow(true);

    this._setupTunnel();
    this._pendingTest = new URLSearchParams(location.search).has('envtest');
  }

  /** Tunnel sky-light occlusion (see fog.js htunnel_occ): polyline along the road through markers.tunnel..tunnelEnd. */
  _setupTunnel() {
    const road = this.ctx.road, m = road?.markers;
    const u = this.fogUniforms;
    if (!m || m.tunnel == null) { u.hTunP.value.x = 0; return; }
    const s0 = m.tunnel, s1 = m.tunnelEnd ?? m.tunnel + 110;
    ['hTun0', 'hTun1', 'hTun2', 'hTun3'].forEach((k, i) => {
      const s = s0 + ((s1 - s0) * i) / 3, p = road.pointAt(s), o = u[k].value;
      o.x = p.x; o.y = p.y; o.z = p.z; o.w = s - s0;
    });
    Object.assign(u.hTunP.value, { x: 7.5, y: 24, z: 0.04, w: 9.5 });
  }

  /** 0 outside .. 1 deep inside the road tunnel (camera position). */
  _tunnelDepth() {
    const road = this.ctx.road, m = road?.markers;
    if (!m || m.tunnel == null) return 0;
    const p = road.project(this.ctx.camera.getWorldPosition(_v), this._projT || (this._projT = {}));
    if (p.dist > 10 || p.s > (m.tunnelEnd ?? m.tunnel + 110) + 40) return 0;
    return THREE.MathUtils.smoothstep(p.s, m.tunnel + 5, m.tunnel + 40);
  }

  _camInTunnel() {
    const road = this.ctx.road, m = road?.markers;
    if (!m || m.tunnel == null) return false;
    const p = road.project(this.ctx.camera.getWorldPosition(_v), this._projT || (this._projT = {}));
    return p.s > m.tunnel + 25 && p.s < (m.tunnelEnd ?? m.tunnel + 110) + 20 && p.dist < 10;
  }

  _sharedFogUniforms() {
    const u = {};
    for (const k of Object.keys(this.fogUniforms)) if (k !== 'hEnv') u[k] = this.fogUniforms[k];
    return u;
  }

  async _buildEnvMap(q) {
    const { renderer, scene, assets } = this.ctx;
    const L = this.look;
    // (LOWEND) PMREM renders into half-float targets: without them (no EXT_color_buffer_float/_half_float) there is
    // no IBL; the hemisphere light takes over the sky's diffuse share (see init) and specular falls back to the sun.
    if (!floatRTSupport(renderer).half) { this._noIBL = true; return; }
    let hdr = null;
    try { hdr = await assets.hdr('assets/sky/env_2k.hdr'); } catch (e) { console.warn('[env] env_2k.hdr failed', e); }
    const pmrem = new THREE.PMREMGenerator(renderer);
    try {
      const envScene = new THREE.Scene();
      const uni = {
        tSky: { value: hdr },
        skyRot: { value: this.skyRot },
        skyEnc: { value: new THREE.Vector4(1, 0, 0, 0) },
        skyTint: { value: new THREE.Color().setRGB(...L.envTint) },
        skyGround: { value: new THREE.Color().setRGB(...L.envGround) },
        skyFogDist: { value: L.skyFogDist },
        skyCloudH: { value: L.skyCloudH },
        skyGroundH: { value: 35 },
        envCamY: { value: L.envCaptureHeight },
        skyVar: { value: 0 },
        skyScud: { value: 0 },
        envRidge: { value: new THREE.Vector4(...L.envRidge) },
        envRidgeD: { value: new THREE.Vector3(...L.envRidgeDist) },
        envRidgeCol: { value: new THREE.Color().setRGB(...L.envRidgeCol) },
        fogColor: { value: scene.fog.color },
        ...this._sharedFogUniforms(),
      };
      const mat = new THREE.ShaderMaterial({
        uniforms: uni, vertexShader: SKY_VERT, fragmentShader: SKY_FRAG,
        side: THREE.BackSide, depthWrite: false, fog: false, toneMapped: false,
        defines: { SKY_HDR: '', ENV_CAPTURE: '' },
      });
      if (!hdr) { uni.tSky.value = this.skyUniforms.tSky.value; mat.defines = { ENV_CAPTURE: '' }; uni.skyEnc.value.copy(this.skyUniforms.skyEnc.value); }
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(50, 64, 32), mat);
      envScene.add(mesh);
      // mist sheets are irrelevant for the IBL; disable them during capture
      const m2 = this.fogUniforms.hfogMist2.value, md = m2.x; m2.x = 0;
      const rt = pmrem.fromScene(envScene, 0, 0.1, 100, { size: q.shadowMapSize >= 4096 ? 512 : 256 });
      m2.x = md;
      this.envRT = rt;
      this.envMap = rt.texture;
      scene.environment = this.envMap;
      mesh.geometry.dispose(); mat.dispose();
      hdr?.dispose?.();
    } catch (e) {
      console.error('[env] PMREM build failed', e);
    } finally {
      pmrem.dispose();
    }
  }

  /** Brief lightning flash. intensity ~0.3 (distant) .. 1.5 (close). */
  lightning(intensity = 1) {
    const now = this.ctx.time?.now ?? performance.now() / 1000;
    const n = 2 + Math.floor(Math.random() * 3);
    let t = now;
    for (let i = 0; i < n; i++) {
      this._flashes.push({ t0: t, amp: intensity * (i === 0 ? 1 : 0.35 + Math.random() * 0.65), tau: 0.035 + Math.random() * 0.07 });
      t += 0.05 + Math.random() * 0.16;
    }
    const delay = 700 + Math.random() * 2600 / Math.max(0.3, intensity);
    setTimeout(() => { try { this.ctx.audio?.play?.('thunder', { volume: Math.min(1, 0.4 + intensity * 0.5) }); } catch {} }, delay);
    this.ctx.events?.emit?.('env:lightning', { intensity });
  }

  _updateFlash(now) {
    let f = 0;
    this._flashes = this._flashes.filter((p) => now - p.t0 < p.tau * 12);
    for (const p of this._flashes) {
      const dt = now - p.t0;
      if (dt >= 0) f += p.amp * Math.exp(-dt / p.tau);
    }
    this.flashLevel = Math.min(f, 3);
    // light from a strike comes from the whole cloud base: mostly sky/hemisphere fill, the shadowed "sun" a little.
    // Soft-capped so a close strike (amp ~1.5, several return strokes stacking) does not hard-clip through AgX.
    const fl = this.flashLevel, fs = fl / (1 + fl * 0.45);
    if (this.sun) this.sun.intensity = this.sun.userData.baseIntensity * (1 + fs * 3);
    if (this.hemi) this.hemi.intensity = this.look.hemiIntensity * (1 + fs * 22);
    this.ctx.scene.environmentIntensity = this.envIntensity * (1 + fs * 1.4);
    if (this.skyUniforms) this.skyUniforms.skyEnc.value.w = fl * 2.5;
    this.fogUniforms.hfogSun.value.w = fl;
    this.fogUniforms.hEnv.value.w = fl;
  }

  _updateSunAndShadow(force = false) {
    const cam = this.ctx.camera;
    if (!this.sun) return;
    cam.getWorldPosition(_v);
    cam.getWorldDirection(_f);
    const R = this.shadowRadius;
    _f.y = 0;
    if (_f.lengthSq() < 1e-6) _f.set(1, 0, 0);
    _f.normalize();
    // centre the frustum ahead of the eye, where most visible shadow receivers are
    _c.copy(_v).addScaledVector(_f, R * 0.42);
    // snap in light space (fixed orientation: same basis as Object3D.lookAt for the shadow camera)
    const Ld = this.sunDirection;
    _x.crossVectors(_up, Ld).normalize();
    _y.crossVectors(Ld, _x);
    const texel = (2 * R) / this.sun.shadow.mapSize.x;
    const px = Math.round(_c.dot(_x) / texel) * texel;
    const py = Math.round(_c.dot(_y) / texel) * texel;
    const pz = _c.dot(Ld);
    _c.set(0, 0, 0).addScaledVector(_x, px).addScaledVector(_y, py).addScaledVector(Ld, pz);
    this.sun.target.position.copy(_c);
    this.sun.position.copy(_c).addScaledVector(Ld, this.look.shadowDistance);
    this.sun.target.updateMatrixWorld();
    this.sun.updateMatrixWorld();
  }

  // ---- shadow-caster culling ----------------------------------------------------------------------------------
  /**
   * Distance-based castShadow culling. Every caster in the scene is handled automatically (rule 'auto': cull when
   * the camera is more than look.shadowCullBase + look.shadowCullK * radius from the bounding sphere). Other systems
   * can override per object (applies to obj and all its meshes):
   *   env.shadowCull(obj, 40)                 cull beyond 40 m (distance from camera to the bounding-sphere surface)
   *   env.shadowCull(obj, {maxDist: 40})      same
   *   env.shadowCull(obj, false)              never touch it (also: obj.userData.shadowCull = false)
   *   env.shadowCull(obj, null)               back to the automatic rule
   * The owner's castShadow is respected: if a system changes castShadow itself, that becomes the new base value, and
   * culling only ever turns a caster off. Instanced meshes with frustumCulled = false (streamed instance subsets)
   * are left alone unless registered explicitly. Evaluated ~6x per second; the caster list refreshes every 2 s.
   * Caveat: if a system turns castShadow OFF at runtime on an object that culling has already switched off, the
   * change is invisible to env; such objects should set userData.shadowCull = false (or call shadowCull(obj, false)).
   */
  shadowCull(obj, rule = null) {
    if (!obj) return;
    const r = rule === false ? false : (typeof rule === 'number' ? { maxDist: rule } : rule);
    obj.traverse((o) => {
      if (!(o.isMesh || o.isPoints || o.isLine)) return;
      if (r === null || r === undefined) this._scRules.delete(o); else this._scRules.set(o, r);
      if (r === false) this._scRestore(o);
    });
    this._scListAge = 1e9; // rebuild the list on the next update
  }

  _scRestore(o) {
    const st = this._scState.get(o);
    if (st && o.castShadow === st.written && !st.written) o.castShadow = st.base;
    this._scState.delete(o);
  }

  _updateShadowCull(now) {
    if (!this._scState || this.ctx.flags?.noShadowCull) return;
    if (now - this._scT < 0.16 && now >= this._scT) return;
    this._scT = now;
    if (now - this._scListAge > 2 || now < this._scListAge) {
      this._scListAge = now;
      const list = this._scList; list.length = 0;
      this.ctx.scene.traverse((o) => {
        if (!(o.isMesh || o.isPoints || o.isLine) || o === this.sky) return;
        const st = this._scState.get(o);
        if (o.castShadow || (st && st.base)) list.push(o);
      });
    }
    const cam = this.ctx.camera.getWorldPosition(_v);
    const L = this.look;
    for (const o of this._scList) {
      let rule = this._scRules.get(o);
      if (rule === false || o.userData.shadowCull === false) { if (this._scState.has(o)) this._scRestore(o); continue; }
      let st = this._scState.get(o);
      if (!st) { st = { base: o.castShadow, written: o.castShadow, r: -1, cnt: -1 }; this._scState.set(o, st); }
      else if (o.castShadow !== st.written) st.base = o.castShadow;   // the owner changed it: that's the new base
      if (!st.base) { st.written = o.castShadow; continue; }
      if (!rule && o.isInstancedMesh && o.frustumCulled === false) { st.written = o.castShadow; continue; }
      // world bounding sphere (cached in object space; instanced: recomputed when the instance count changes)
      let bs;
      if (o.isInstancedMesh) {
        if (!o.boundingSphere || st.cnt !== o.count) { o.computeBoundingSphere(); st.cnt = o.count; }
        bs = o.boundingSphere;
      } else {
        const g = o.geometry;
        if (!g) continue;
        if (!g.boundingSphere) g.computeBoundingSphere();
        bs = g.boundingSphere;
      }
      _sph.copy(bs).applyMatrix4(o.matrixWorld);
      const dist = Math.max(0, _sph.center.distanceTo(cam) - _sph.radius);
      let maxDist;
      if (rule && rule.maxDist != null) maxDist = rule.maxDist;
      else if (/^terrain_near/.test(o.name)) maxDist = L.shadowCullTerrain;
      else if (o.isInstancedMesh) {
        // a scattered set: what matters is the size of ONE instance, measured to the nearest edge of the whole set
        const g = o.geometry;
        if (g && !g.boundingSphere) g.computeBoundingSphere();
        const ir = (g?.boundingSphere?.radius ?? 1) * o.matrixWorld.getMaxScaleOnAxis();
        maxDist = L.shadowCullBase + L.shadowCullK * Math.min(ir * 1.5, _sph.radius);
      } else maxDist = L.shadowCullBase + L.shadowCullK * _sph.radius;
      const want = dist <= maxDist;
      if (o.castShadow !== want) o.castShadow = want;
      st.written = want;
    }
  }

  update(dt) {
    const now = this.ctx.time?.now ?? 0;
    if (this._pendingTest) { this._pendingTest = false; this._addTestObjects(); }
    const u = this.fogUniforms;
    u.hfogMist2.value.z = now;
    const e = u.hEnv.value; e.x = now; e.y = this._wetness; e.z = this._rain;
    // no rain falls inside the tunnel: its grey veil fades out while the camera is in the tube
    this._tunnelIn = this._tunnelDepth();
    u.hfogRain.value.x = this.fog.rainExtinction * (1 - 0.9 * this._tunnelIn);
    if (this.sky) this.ctx.camera.getWorldPosition(this.sky.position);
    this._updateSunAndShadow();
    this._updateShadowCull(now);
    if (this.autoLightning > 0 && dt > 0) {
      if (!this._nextStrike) this._nextStrike = now + this.autoLightning * (0.5 + Math.random());
      if (now >= this._nextStrike) {
        // no strikes while the camera is deep in the tunnel (nothing would show; thunder alone is handled by audio)
        const inTunnel = (this.fogUniforms.hTunP.value.x > 0) && this._camInTunnel();
        if (!inTunnel) this.lightning(0.25 + Math.random() * 0.6);
        this._nextStrike = now + this.autoLightning * (0.4 + Math.random() * 1.2);
      }
    }
    this._updateFlash(now);
  }

  /** Scratch verification objects (?envtest): chrome ball, grey ball, 18% grey card in front of the camera. */
  _addTestObjects() {
    const { scene, camera } = this.ctx;
    const g = new THREE.Group(); g.name = 'envtest';
    const chrome = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), new THREE.MeshStandardMaterial({ metalness: 1, roughness: 0.02 }));
    const grey = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), new THREE.MeshStandardMaterial({ color: new THREE.Color(0.18, 0.18, 0.18), roughness: 0.9 }));
    const card = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(0.18, 0.18, 0.18), fog: false }));
    chrome.castShadow = grey.castShadow = true; chrome.receiveShadow = grey.receiveShadow = true;
    g.add(chrome, grey, card);
    camera.updateMatrixWorld();
    const fwd = camera.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
    const base = camera.getWorldPosition(new THREE.Vector3()).addScaledVector(fwd, 7);
    chrome.position.copy(base).addScaledVector(right, -1.6);
    grey.position.copy(base).addScaledVector(right, 1.2);
    card.position.copy(base).addScaledVector(right, 3.6).addScaledVector(camera.up, 1.2);
    card.lookAt(camera.getWorldPosition(new THREE.Vector3()));
    // compile coverage for every material family other systems use with the global fog/tunnel chunks
    const at = (obj, r, u) => { obj.position.copy(base).addScaledVector(right, r).addScaledVector(camera.up, u).addScaledVector(fwd, 6); g.add(obj); return obj; };
    const sph = new THREE.SphereGeometry(0.35, 24, 12);
    at(new THREE.Mesh(sph, new THREE.MeshLambertMaterial({ color: 0x777777 })), -3, 1.5);
    at(new THREE.Mesh(sph, new THREE.MeshPhongMaterial({ color: 0x777777 })), -2, 1.5);
    at(new THREE.Mesh(sph, new THREE.MeshToonMaterial({ color: 0x777777 })), -1, 1.5);
    at(new THREE.Mesh(sph, new THREE.MeshPhysicalMaterial({ transmission: 1, roughness: 0.1, thickness: 0.2 })), 0, 1.5);
    at(new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xaaaaaa })), 1, 1.5);
    const pts = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0.3, 0.2, 0, -0.3, 0.1, 0], 3));
    const ptsMat = new THREE.PointsMaterial({ size: 0.1, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false });
    ptsMat.defines = { HFOG_TRANSMITTANCE_ONLY: '' };
    at(new THREE.Points(pts, ptsMat), 2, 1.5);
    at(new THREE.Line(pts, new THREE.LineBasicMaterial({ color: 0xffffff })), 3, 1.5);
    at(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShadowMaterial({ opacity: 0.4 })), 0, -1.2).rotation.x = -Math.PI / 2;
    // custom ShaderMaterial exactly as documented in fog.js
    const custom = new THREE.ShaderMaterial({
      fog: true,
      uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, this.fogUniforms, { color: { value: new THREE.Color(0.3, 0.3, 0.3) } }]),
      vertexShader: '#include <common>\n#include <fog_pars_vertex>\nvoid main() { vec3 transformed = position; vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0); gl_Position = projectionMatrix * mvPosition;\n#include <fog_vertex>\n}',
      fragmentShader: 'uniform vec3 color;\n#include <common>\n#include <fog_pars_fragment>\nvoid main() { gl_FragColor = vec4(color, 1.0);\n#include <fog_fragment>\n}',
    });
    at(new THREE.Mesh(sph, custom), 4, 1.5);
    // instanced + triplanar + wetness + ripples, and a plain standard material with wetness + ripples
    materials.pbrMaterial(this.ctx, 'lichen_rock', { triplanar: true, wet: 0.5 }).then((m) => {
      materials.applyRipples(m);
      const im = new THREE.InstancedMesh(new THREE.BoxGeometry(0.4, 0.4, 0.4), m, 3);
      for (let i = 0; i < 3; i++) im.setMatrixAt(i, new THREE.Matrix4().makeTranslation(i * 0.6, 0, 0));
      im.castShadow = im.receiveShadow = true;
      at(im, -3, 0.6);
      const sm = materials.applyRipples(materials.applyWetness(new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.6 }), { porosity: 0.3 }));
      at(new THREE.Mesh(sph, sm), -1, 0.6);
    }).catch((e) => console.error('[env] envtest material failed', e));
    scene.add(g);
    this.testObjects = g;
  }

  dispose() {
    const { scene } = this.ctx;
    if (this.sky) { scene.remove(this.sky); this.sky.geometry.dispose(); this.sky.material.dispose(); }
    if (this.sun) scene.remove(this.sun, this.sun.target);
    this.sun?.shadow?.dispose?.();
    if (this.hemi) scene.remove(this.hemi);
    if (scene.environment === this.envMap) scene.environment = null;
    this.envRT?.dispose();
    this.testObjects && scene.remove(this.testObjects);
  }
}
