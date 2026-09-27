// Terrain splat material (TERRAIN workstream).
// MeshStandardMaterial + onBeforeCompile, so the global height fog, shadows and env lighting keep working.
//
// Seven CC0 PBR sets are packed at load time into two 1024² DataArrayTextures:
//   uAlb  (sRGB) : rgb = albedo, a = height proxy (AO + luminance) for height blending
//   uNrm  (lin)  : rg = normal.xy (OpenGL), b = roughness, a = AO
// Per pixel: masks from COLOR_0 (R=AO, G=gravel, B=mud/wet, A=vegetation) + slope + macro noise give seven
// weights; the two strongest layers are sampled biplanar (2 projections, strata kept horizontal on every
// face) with explicit gradients and height-blended. Beyond uLodDist the cheaper "far look" takes over: the
// same real textures at larger tiles (no flat averages), plus a canopy tint.
// Wetness (ctx.env.wetness) darkens porous albedo, lowers roughness, adds puddles and rock seepage streaks.
import * as THREE from 'three';
import { url } from '../core/assets.js';

// index is part of the contract with roadMaterial.js (4 = gravel, 5 = mud, 8 = angular stone debris). Layers are
// only ever appended.
export const LAYERS = ['aerial_rocks_02', 'lichen_rock', 'aerial_grass_rock', 'forest_ground_04', 'rocky_trail', 'brown_mud_rocks_01',
  'dark_rock_02', 'marble_cliff_03', 'gray_rocks'];
// world meters per texture repeat (near detail). Matched to each Poly Haven scan's real-world capture size
// (api.polyhaven.com/info: lichen_rock 2.0 m, forest_ground_04 3.15 m, rocky_trail 2.0 m, brown_mud_rocks_01 1.3 m,
// dark_rock_02 2.0 m, marble_cliff_03 5.75 m, gray_rocks 1.8 m) so grains, joints and fracture planes read at true
// scale (0.2-2 m blocks on the cut). The two aerial scans cover 50 m / 15 m: they are used as scree/meadow texture
// at 12 m / 9 m (their boulders then read as 0.2-1 m cobbles, which is what they stand in for).
const TILE = [12.0, 2.0, 9.0, 3.15, 2.0, 1.55, 2.0, 5.75, 1.8];
// normal map strength per layer
const NSTR = [1.0, 0.9, 0.75, 1.0, 1.0, 0.95, 1.15, 1.35, 1.1];
// per-layer grade: rgb multiplier (linear), a = desaturation. Pulls the warm/green scans toward the cold grey
// gneiss/limestone of an Alpine road cut; dark_rock_02 is lifted (it is a very dark scan).
// Scan albedos (linear luminance) are 0.036 (dark_rock_02) .. 0.45 (marble_cliff_03, a pale marble); weathered
// gneiss / limestone is ~0.2-0.25 dry (so the rock scans are graded toward that; wetness darkens them again in
// the shader). marble_cliff_03 keeps ~25% of its ochre (iron-oxide staining is real on Alpine cuts).
const TINT = [
  [1.15, 1.18, 1.24, 0.35], [1.45, 1.5, 1.6, 0.6], [0.8, 0.84, 0.76, 0.22], [0.92, 0.95, 0.92, 0.1],
  [0.85, 0.85, 0.87, 0.25], [1.0, 0.97, 0.94, 0.05], [3.9, 3.9, 4.1, 0.45], [0.66, 0.66, 0.68, 0.62],
  [0.95, 0.96, 1.0, 0.3],
];

let _arrays = null;

async function bitmap(path, size) {
  const r = await fetch(url(path));
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  const blob = await r.blob();
  return createImageBitmap(blob, {
    resizeWidth: size, resizeHeight: size, resizeQuality: 'high',
    colorSpaceConversion: 'none', premultiplyAlpha: 'none', imageOrientation: 'flipY',
  });
}

/** Builds (once) and returns {alb, nrm, avg: Float32Array(n*3) linear average albedo per layer}. */
export function loadTerrainArrays(ctx, size = 1024) {
  if (!_arrays) _arrays = buildArrays(ctx, size);
  return _arrays;
}

async function buildArrays(ctx, S) {
  const n = LAYERS.length;
  const alb = new Uint8Array(S * S * 4 * n);
  const nrm = new Uint8Array(S * S * 4 * n);
  const avg = new Float32Array(n * 3);
  const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(S, S) : Object.assign(document.createElement('canvas'), { width: S, height: S });
  const g = cv.getContext('2d', { willReadFrequently: true });
  const read = (bmp) => {
    g.clearRect(0, 0, S, S);
    g.drawImage(bmp, 0, 0, S, S);
    bmp.close?.();
    return g.getImageData(0, 0, S, S).data;
  };
  const toLin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  // load layer by layer (keeps peak memory low); the 3 maps of a layer load in parallel
  for (let l = 0; l < n; l++) {
    const name = LAYERS[l];
    const [bd, bn, ba] = await Promise.all(['diffuse.jpg', 'nor_gl.jpg', 'arm.jpg'].map((f) => bitmap(`assets/tex/${name}/${f}`, S)));
    const d = read(bd);
    const o = l * S * S * 4;
    const hp = new Float32Array(S * S);
    let sr = 0, sg = 0, sb = 0;
    for (let i = 0, k = 0; i < S * S; i++, k += 4) {
      alb[o + k] = d[k]; alb[o + k + 1] = d[k + 1]; alb[o + k + 2] = d[k + 2];
      hp[i] = d[k] * 0.3 + d[k + 1] * 0.59 + d[k + 2] * 0.11;
      if ((i & 15) === 0) { sr += d[k]; sg += d[k + 1]; sb += d[k + 2]; }
    }
    const cnt = (S * S) / 16;
    avg[l * 3] = toLin(sr / cnt); avg[l * 3 + 1] = toLin(sg / cnt); avg[l * 3 + 2] = toLin(sb / cnt);
    const nm = read(bn);
    const a = read(ba);
    let lo = 1e9, hi = -1e9;
    for (let i = 0, k = 0; i < S * S; i++, k += 4) {
      nrm[o + k] = nm[k]; nrm[o + k + 1] = nm[k + 1]; nrm[o + k + 2] = a[k + 1]; nrm[o + k + 3] = a[k];
      const h = a[k] * 0.65 + hp[i] * 0.35;
      hp[i] = h;
      if (h < lo) lo = h;
      if (h > hi) hi = h;
    }
    const sc = 255 / Math.max(hi - lo, 1);
    for (let i = 0, k = 0; i < S * S; i++, k += 4) alb[o + k + 3] = Math.max(0, Math.min(255, (hp[i] - lo) * sc));
  }
  const maxAniso = ctx.renderer.capabilities.getMaxAnisotropy();
  const mk = (data, srgb) => {
    const t = new THREE.DataArrayTexture(data, S, S, n);
    t.format = THREE.RGBAFormat;
    t.type = THREE.UnsignedByteType;
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.generateMipmaps = true;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.anisotropy = Math.min(ctx.config?.quality?.anisotropy ?? 8, maxAniso);
    t.unpackAlignment = 4;
    t.needsUpdate = true;
    // drop the CPU copy once uploaded
    t.onUpdate = () => { t.image.data = null; t.onUpdate = null; };
    return t;
  };
  return { alb: mk(alb, true), nrm: mk(nrm, false), avg };
}

// ---------------------------------------------------------------------------------------------
// GLSL
// ---------------------------------------------------------------------------------------------
export const GLSL_NOISE = /* glsl */`
float th_hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float th_vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(th_hash(i), th_hash(i + vec2(1.0, 0.0)), u.x), mix(th_hash(i + vec2(0.0, 1.0)), th_hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float th_fbm(vec2 p) {
  float a = 0.5, s = 0.0;
  for (int i = 0; i < 4; i++) { s += a * th_vnoise(p); p = mat2(1.6, 1.2, -1.2, 1.6) * p; a *= 0.5; }
  return s / 0.9375;
}
`;

const NL = LAYERS.length;

const FRAG_PARS = /* glsl */`
uniform sampler2DArray uAlb;
uniform sampler2DArray uNrm;
uniform float uTile[${NL}];
uniform float uNStr[${NL}];
uniform vec4 uTint[${NL}];
uniform vec3 uAvg[${NL}];
uniform float uWet;
uniform float uTime;
uniform vec2 uLodDist;
uniform vec4 uGap;      // washout: centreline point (xyz) + road grade
uniform vec2 uGapT;     // washout: horizontal unit tangent (xz)
uniform float uFx;      // 1 = realism-pass features on (A/B perf switch)
uniform vec4 uFxv;      // per-feature switches: x sheet streaks, y fall halo, z half-casts, w unused
uniform vec3 uFallMin;  // AABB of all fall segments (+ halo radius): cheap reject
uniform vec3 uFallMax;
uniform vec4 uFallA[16]; // gully falls (terrain.js water builder): top point xyz + jet half-width
uniform int uFallN;      // active falls (terrain.js selects the ones near the camera, nearest first)
uniform vec4 uFallB[16]; // bottom point xyz
varying vec4 vMasks;
varying vec3 vWPos;
varying vec3 vWNrm;
${GLSL_NOISE}

// biplanar: the dominant (ma) and the median (me) projection axes, with weights
struct THBP { int ma; int me; vec2 w; };
THBP th_bp(vec3 n) {
  vec3 an = abs(n);
  int ma = (an.x > an.y && an.x > an.z) ? 0 : (an.y > an.z) ? 1 : 2;
  int mi = (an.x < an.y && an.x < an.z) ? 0 : (an.y < an.z) ? 1 : 2;
  int me = 3 - mi - ma;
  vec2 w = vec2(an[ma], an[me]);
  w = clamp((w - 0.5773) / (1.0 - 0.5773), 0.0, 1.0);
  w = w * w;
  THBP b; b.ma = ma; b.me = me; b.w = w / max(w.x + w.y, 1e-5);
  return b;
}
// texture axes (u, v) for a projection axis k: side projections keep v = world Y, so layered rock
// (dark_rock_02 strata) stays horizontal on every face orientation
ivec2 th_ax(int k) { return k == 0 ? ivec2(2, 1) : (k == 1 ? ivec2(0, 2) : ivec2(0, 1)); }

// Water on a rock face, sampled in (horizontal face coordinate u, world height y). Sources sit at discrete points
// (a joint or ledge lip every ~2.6 m, in 7 m tall bands) and run 2-12 m down, widening and wandering as they go:
//   x = black "ink stripe" (Tintenstrich: cyanobacteria film where water runs for years; 0.2-1.2 m wide, dark, matt)
//   y = active seep (a live water film 3-20 cm wide, glossy, flowing)
//   z = iron-oxide staining (ochre halo under some seeps), w = pale calcite / efflorescence crust
vec4 th_seep(float u, float y, float seed) {
  vec4 r = vec4(0.0);
  const float CW = 2.6, RH = 7.0;
  float ci = floor(u / CW), ri = floor(y / RH);
  for (int j = 0; j <= 1; j++) {
    float row = ri + float(j);
    for (int k = -1; k <= 1; k++) {
      float c = ci + float(k);
      vec2 id = vec2(c, row) + seed;
      float h0 = th_hash(id);
      if (h0 > 0.7) continue;                                    // ~70% of joints drip in a rainstorm
      float h1 = th_hash(id + 7.13), h2 = th_hash(id + 3.71);
      float y0 = (row + 0.35 + 0.6 * h1) * RH;                   // source height
      float dy = y0 - y;                                          // distance below the source
      float len = 2.5 + 11.0 * h2;
      if (dy < -0.3 || dy > len + 2.0) continue;
      float h3 = th_hash(id + 11.9);
      // the trail follows the relief: wanders more the further it runs
      float wander = (th_vnoise(vec2(y * 0.45, c * 3.1 + seed)) - 0.5) * (0.35 + 0.05 * dy);
      float cx = (c + 0.15 + 0.7 * h3) * CW + wander;
      float wI = (0.1 + 0.5 * h1 * h1) * (1.0 + 0.09 * dy);      // ink stripe half-width
      float wS = (0.03 + 0.1 * h2) * (1.0 + 0.06 * dy);          // live film half-width (6-30 cm wide)
      float x = u - cx;
      // (PERF, QA) beyond the widest feature of this trail (the iron halo, ~2 wI) it adds < 2 %: skip the rest
      if (abs(x) > max(4.0 * wI, 3.0 * wS) + 0.05) continue;
      float fade = smoothstep(-0.3, 0.25, dy) * (1.0 - smoothstep(len, len + 2.0, dy));
      float ink = exp(-x * x / (wI * wI)) * fade * step(0.18, h0);
      // mottled, fibrous edges (the film follows micro-channels)
      ink *= smoothstep(0.15, 0.55, th_vnoise(vec2(x * 7.0 / max(wI, 0.1), y * 1.3 + c)) * 0.6 + 0.5 - abs(x) / max(wI, 0.05) * 0.35);
      float sp = exp(-x * x / (wS * wS)) * fade * step(h0, 0.55);
      r.x = max(r.x, ink * (0.45 + 0.55 * h2));
      r.y = max(r.y, sp);
      r.z = max(r.z, exp(-x * x / (4.0 * wI * wI)) * fade * step(0.8, fract(h3 * 7.3)) * 0.8);
      r.w = max(r.w, exp(-x * x / (0.6 * wI * wI)) * fade * step(0.9, fract(h1 * 5.7)) * smoothstep(0.5, 2.5, dy));
    }
  }
  return r;
}

// Pre-split blasting traces ("half-casts") on a road cut: half-barrels of the ~90 mm drill holes, 0.75-0.95 m
// apart, running down the face on the 10:1 batter wherever the pre-split plane survived (runs of 1-6 m; broken
// out elsewhere). Returns (across-groove slope -1..1, mask).
vec2 th_halfcast(float u, float y, float seed) {
  const float SP = 0.85;
  float uu = u + y * 0.1;
  float ci = floor(uu / SP);
  float h = th_hash(vec2(ci, seed));
  float x = uu - (ci + 0.5 + (h - 0.5) * 0.25) * SP;
  const float r = 0.045;
  float inG = 1.0 - smoothstep(r * 0.8, r, abs(x));
  float run = smoothstep(0.42, 0.5, th_vnoise(vec2(ci * 1.7 + seed, y * 0.3 + h * 5.0)));
  return vec2(clamp(x / r, -1.0, 1.0), inG * run);
}

vec4 th_grade(vec4 a, float L) {
  vec4 t = uTint[int(L)];
  vec3 c = a.rgb;
  c = mix(c, vec3(dot(c, vec3(0.3, 0.59, 0.11))), t.a) * t.rgb;
  return vec4(c, a.a);
}

void th_proj(float L, float t, float nstr, vec3 p, vec3 dpx, vec3 dpy, vec3 n, int k,
             out vec4 alb, out vec3 nw, out float rough, out float ao) {
  ivec2 ax = th_ax(k);
  vec2 uv = vec2(p[ax.x], p[ax.y]) / t;
  vec2 gx = vec2(dpx[ax.x], dpx[ax.y]) / t, gy = vec2(dpy[ax.x], dpy[ax.y]) / t;
  alb = textureGrad(uAlb, vec3(uv, L), gx, gy);
  vec4 d = textureGrad(uNrm, vec3(uv, L), gx, gy);
  vec3 tn = vec3((d.xy * 2.0 - 1.0) * nstr, 0.0); tn.z = sqrt(max(1.0 - dot(tn.xy, tn.xy), 0.0));
  // whiteout blend onto the geometric normal
  vec3 tw = vec3(tn.xy + vec2(n[ax.x], n[ax.y]), abs(tn.z) * n[k]);
  nw = vec3(0.0); nw[ax.x] = tw.x; nw[ax.y] = tw.y; nw[k] = tw.z;
  rough = d.b; ao = d.a;
}

// Samples layer L at tile size t. Returns albedo(rgb)+height(a) in alb; world normal, roughness, ao.
void th_layer(float L, float t, float nstr, vec3 p, vec3 dpx, vec3 dpy, vec3 n, THBP b,
              out vec4 alb, out vec3 nw, out float rough, out float ao) {
  th_proj(L, t, nstr, p, dpx, dpy, n, b.ma, alb, nw, rough, ao);
  if (b.w.y > 0.01) {
    vec4 a2; vec3 n2; float r2, o2;
    th_proj(L, t, nstr, p, dpx, dpy, n, b.me, a2, n2, r2, o2);
    alb = alb * b.w.x + a2 * b.w.y; nw = nw * b.w.x + n2 * b.w.y;
    rough = rough * b.w.x + r2 * b.w.y; ao = ao * b.w.x + o2 * b.w.y;
  }
  nw = normalize(nw);
  alb = th_grade(alb, L);
}

// far-look sampling: beyond ~450 m the tile grows 5x (a 10-20 m tile repeats visibly across a 2 km slope)
void th_layerF(float L, float t, float nstr, vec3 p, vec3 dpx, vec3 dpy, vec3 n, THBP b, float k,
               out vec4 alb, out vec3 nw, out float rough, out float ao) {
  if (k < 0.999) th_layer(L, t, nstr, p, dpx, dpy, n, b, alb, nw, rough, ao);
  if (k > 0.001) {
    vec4 a2; vec3 n2; float r2, o2;
    th_layer(L, t * 5.0, nstr, p, dpx, dpy, n, b, a2, n2, r2, o2);
    if (k >= 0.999) { alb = a2; nw = n2; rough = r2; ao = o2; }
    else { alb = mix(alb, a2, k); nw = normalize(mix(nw, n2, k)); rough = mix(rough, r2, k); ao = mix(ao, o2, k); }
  }
}

vec3 thAlbedo; vec3 thNormalW; float thRough; float thAO; float thGraze;

void th_splat() {
  vec3 p = vWPos;
  vec3 n = normalize(vWNrm);
  vec3 dpx = dFdx(p), dpy = dFdy(p);
  float dist = length(p - cameraPosition);
  float slope = 1.0 - n.y;
  float aoV = vMasks.r, grav = vMasks.g, mud = vMasks.b, veg = vMasks.a;
  // Projection axes follow the true (flat triangle) surface wherever the interpolated vertex normal disagrees with
  // it (decimated cut faces, overhangs, ledges): projecting along a smooth normal that is 30-60 deg off the real
  // face is what smeared the rock texture into horizontal streaks.
  vec3 fN0 = normalize(cross(dpx, dpy)); fN0 *= sign(dot(fN0, n) + 1e-4);
  n = normalize(mix(n, fN0, smoothstep(0.97, 0.8, dot(fN0, n))));
  slope = 1.0 - n.y;
  // terrain.py also writes the seepage streaks of the road cut into the mud/wet mask (B ~0.2-0.55 on steep, bare
  // faces). That is water running on rock, not mud: taken out of the mud weight (it painted brown mud texture and
  // the ochre scarp tint onto the cut) and used as soaked, glossy bands below. Slide scarps (B ~0.95) and gullies
  // (~0.75) stay mud.
  float seepB = mud * smoothstep(0.4, 0.7, slope) * (1.0 - smoothstep(0.55, 0.72, mud)) * (1.0 - smoothstep(0.2, 0.5, veg));
  mud -= seepB;
  THBP b = th_bp(n);
  float m1 = th_fbm(p.xz * 0.013);
  float m2 = th_vnoise(p.xz * 0.09 + 17.0);
  float m3 = th_vnoise(p.xz * 0.021 + 3.7);
  float mv = th_vnoise(vec2(p.x + p.z, p.y) * 0.05 + 9.1);   // varies up a face too
#ifdef TH_FAR_ONLY
  float lodT = 1.0;          // the far mesh has no near branch (it can be close: skirt, scar core, seams)
#else
  float lodT = smoothstep(uLodDist.x, uLodDist.y, dist);
#endif
  vec3 alb = vec3(0.0); vec3 nw = n; float rough = 0.9; float ao = 1.0;
  float snowW = 0.0;

  // ---------- weights (shared by both looks)
  float rock = smoothstep(0.33, 0.47, slope + (m2 - 0.5) * 0.16);
  rock = max(rock, smoothstep(0.38, 0.08, veg) * smoothstep(0.05, 0.2, slope) * (1.0 - grav));
  // steep faces inside the slide (head scarp, lateral scarps) are bare regolith / rock, not mud
  float scarp = mud * smoothstep(0.4, 0.75, slope);
  float mudW = mud * (1.0 - 0.45 * smoothstep(0.42, 0.78, slope));
  float gravW = grav * (1.0 - mudW);
  float rest = max(1.0 - mudW - gravW, 0.0);
  // cliff: steep, bare faces (road cut, fresh fractures) -> grey layered rock; moderate outcrops keep moss
  float cliff = smoothstep(0.4, 0.62, slope + (mv - 0.5) * 0.22) * (1.0 - 0.7 * smoothstep(0.35, 0.8, veg));
  float lich = smoothstep(0.45, 0.68, m1 * 0.6 + mv * 0.6 + (m2 - 0.5) * 0.25);
  // under the spruce canopy the floor is needle litter and moss; open grass only in clearings
  float forest = smoothstep(0.2, 0.65, veg) * smoothstep(0.18, 0.5, m3);
  // pebbles and cobbles carried in the slide mud
  float peb = smoothstep(0.5, 0.75, th_fbm(p.xz * 0.19 + 4.1)) * (1.0 - smoothstep(0.35, 0.7, slope));

#ifndef TH_FAR_ONLY
  if (lodT < 0.999) {
    float w[${NL}];
    float rk = rest * rock;
    // blasted cut faces: jointed rock with 0.2-2 m fracture blocks (marble_cliff_03, graded to grey gneiss), with
    // patches of the darker, bedded dark_rock_02; lichen only on the older, less steep outcrops and ledges
    float strat = smoothstep(0.58, 0.8, th_vnoise(vec2(p.x + p.z, p.y * 2.5) * 0.045 + 3.0));
    float lichC = lich * mix(1.0, 0.2, cliff);
    w[0] = rk * (1.0 - cliff) * (1.0 - lich);
    w[1] = rk * lichC;
    // (dark_rock_02 is a bedded scan whose curved layer edges read as rows of fish scales across a whole cut face:
    //  the cliffs use marble_cliff_03 only; the darker bedded zones are a tone shift, see 'strat' below)
    w[7] = rk * cliff * (1.0 - lichC);
    w[6] = 0.0;
    // ledges and benches of the cut (no soil mapped there): dark moss, needle litter and fines, not meadow
    float ledge = smoothstep(0.35, 0.05, veg);
    float fo = max(forest, ledge);
    w[2] = rest * (1.0 - rock) * (1.0 - fo);
    w[3] = rest * (1.0 - rock) * fo;
    // talus cones at the foot of the cut (sloping gravel) are angular fresh rock fragments, not a worn trail
    float talusW = smoothstep(0.07, 0.22, slope) * (0.6 + 0.4 * m2);
    w[8] = gravW * talusW + mudW * peb * 0.2;
    w[4] = gravW * (1.0 - talusW) + mudW * peb * 0.25;
    w[5] = mudW * (1.0 - peb * 0.45);
    int i1 = 0; float b1 = -1.0;
    for (int i = 0; i < ${NL}; i++) if (w[i] > b1) { b1 = w[i]; i1 = i; }
    int i2 = 0; float b2 = -1.0;
    for (int i = 0; i < ${NL}; i++) if (i != i1 && w[i] > b2) { b2 = w[i]; i2 = i; }
    float ws = max(b1 + b2, 1e-4); b1 /= ws; b2 /= ws;
    vec4 A1; vec3 N1; float R1, O1;
#ifdef TH_ANTITILE
    // Anti-tiling: a second, rotated + rescaled sample of the dominant layer takes over in patches. The switch is a
    // narrow, height-blended seam (the higher grains of either sample win), not a broad 50/50 cross-fade: averaging
    // two unrelated rock scans halves their contrast and reads as a smeared, blurry overlay on the cut face.
    // Most pixels need only one of the two samples.
    float kAT = smoothstep(0.4, 0.6, th_vnoise(p.xz * 0.11 + p.y * 0.05));
    if (kAT < 0.999) th_layer(float(i1), uTile[i1], uNStr[i1], p, dpx, dpy, n, b, A1, N1, R1, O1);
    vec4 A = A1; vec3 N = N1; float Rr = R1, Oo = O1;
    if (kAT > 0.001) {
      float c = 0.8, s = 0.6;
      vec3 pr = vec3(c * p.x - s * p.z, p.y, s * p.x + c * p.z) * 1.37 + 11.3;
      vec3 dxr = vec3(c * dpx.x - s * dpx.z, dpx.y, s * dpx.x + c * dpx.z) * 1.37;
      vec3 dyr = vec3(c * dpy.x - s * dpy.z, dpy.y, s * dpy.x + c * dpy.z) * 1.37;
      vec3 nr = vec3(c * n.x - s * n.z, n.y, s * n.x + c * n.z);
      vec4 Ab; vec3 Nb; float Rb, Ob;
      THBP br = th_bp(nr);
      th_layer(float(i1), uTile[i1] * 1.37, uNStr[i1], pr, dxr, dyr, nr, br, Ab, Nb, Rb, Ob);
      Nb = vec3(c * Nb.x + s * Nb.z, Nb.y, -s * Nb.x + c * Nb.z);
      if (kAT >= 0.999) { A = Ab; N = Nb; Rr = Rb; Oo = Ob; }
      else {
        float k = clamp((kAT - 0.5) * 3.0 + (Ab.a - A1.a) * 2.5 + 0.5, 0.0, 1.0);
        A = mix(A1, Ab, k); N = normalize(mix(N1, Nb, k)); Rr = mix(R1, Rb, k); Oo = mix(O1, Ob, k);
      }
    }
#else
    th_layer(float(i1), uTile[i1], uNStr[i1], p, dpx, dpy, n, b, A1, N1, R1, O1);
    vec4 A = A1; vec3 N = N1; float Rr = R1, Oo = O1;
#endif
    if (b2 > 0.02) {
      vec4 A2; vec3 N2; float R2, O2;
      th_layer(float(i2), uTile[i2], uNStr[i2], p, dpx, dpy, n, b, A2, N2, R2, O2);
      float ha = A.a + b1 * 1.2, hb = A2.a + b2 * 1.2;
      float hm = max(ha, hb) - 0.22;
      float ba = max(ha - hm, 0.0), bb = max(hb - hm, 0.0);
      float bs = max(ba + bb, 1e-4); ba /= bs; bb /= bs;
      A = A * ba + A2 * bb; N = normalize(N * ba + N2 * bb); Rr = Rr * ba + R2 * bb; Oo = Oo * ba + O2 * bb;
    }
    // crisp fracture facets on steep rock: add the flat triangle normal's deviation
    float facet = rock * rest * smoothstep(0.25, 0.55, slope) * (1.0 - lodT);
    N = normalize(N + (fN0 - n) * facet * 0.5);
    alb = A.rgb; nw = N; rough = Rr; ao = Oo;
    // darker, cooler bedded zones on the cut (dark schist bands in the gneiss), and warmer iron-stained ones
    alb *= mix(vec3(1.0), vec3(0.7, 0.72, 0.76), strat * rk * cliff);
    alb *= mix(vec3(1.0), vec3(1.12, 1.0, 0.86), smoothstep(0.62, 0.85, th_vnoise(vec2(p.x - p.z, p.y * 1.5) * 0.06 + 8.0)) * rk * cliff * 0.8);
  }
#endif
  if (lodT > 0.001) {
    // ---------- far look: the same scans at larger tiles (never flat averages)
    float kfar = smoothstep(450.0, 800.0, dist);
    float rk = clamp(rock * rest, 0.0, 1.0);
    vec4 Ag; vec3 Ng; float Rg, Og;
    th_layerF(2.0, 21.0, 0.8, p, dpx, dpy, n, b, kfar, Ag, Ng, Rg, Og);
    vec3 fa = Ag.rgb; vec3 fn = Ng; float fr = Rg, fo = Og;
    if (rk > 0.02) {
      vec4 Ar; vec3 Nr; float Rr2, Or2;
      float cl = cliff * (1.0 - lich * 0.4);
      if (cl > 0.5) th_layerF(7.0, 11.5, 1.1, p, dpx, dpy, n, b, kfar, Ar, Nr, Rr2, Or2);
      else th_layerF(0.0, 15.0, 0.9, p, dpx, dpy, n, b, kfar, Ar, Nr, Rr2, Or2);
      fa = mix(fa, Ar.rgb, rk); fn = normalize(mix(fn, Nr, rk)); fr = mix(fr, Rr2, rk); fo = mix(fo, Or2, rk);
    }
    // below the treeline the valley sides are closed spruce forest: from afar that is a dark canopy
    // (albedo ~0.03-0.05, bluish-green) mottled by crowns (~6-10 m) and their shadowed gaps, not the meadow scan
    float canopy = smoothstep(0.12, 0.5, veg) * (1.0 - rk) * (1.0 - gravW) * (1.0 - mudW);
    float crowns = th_vnoise(p.xz * 0.14) * 0.6 + th_vnoise(p.xz * 0.037 + 3.0) * 0.4;
    vec3 canC = vec3(0.03, 0.041, 0.03) * (0.55 + 0.9 * crowns) * mix(vec3(1.0), vec3(1.15, 1.05, 0.85), m3 * 0.5);
    // (no snow dusting below the cloud base: under an overcast sky snow sits at the haze colour and only washes the
    //  far slopes out; the dark canopy and wet rock are what carry the mountains through the haze)
    fa = mix(fa, canC, canopy * mix(0.55, 0.95, smoothstep(150.0, 600.0, dist)));
    fr = mix(fr, 0.85, canopy);
    if (mudW + gravW > 0.02) {
      vec4 Am; vec3 Nm; float Rm, Om;
      float isMud = step(gravW, mudW);
      if (isMud > 0.5) th_layerF(5.0, 9.0, 1.0, p, dpx, dpy, n, b, kfar, Am, Nm, Rm, Om);
      else th_layerF(4.0, 7.0, 1.0, p, dpx, dpy, n, b, kfar, Am, Nm, Rm, Om);
      float wm = clamp(mudW + gravW, 0.0, 1.0);
      fa = mix(fa, Am.rgb, wm); fn = normalize(mix(fn, Nm, wm)); fr = mix(fr, Rm, wm); fo = mix(fo, Om, wm);
    }
    // ---------- alpine zonation of the far mountains (what makes them read as real peaks through the haze).
    // Road ~1500 m a.s.l. -> world y 40-98; treeline ~2000 m (y ~560, as in terrain.py); fresh autumn-storm snow
    // above ~2150 m (y ~650). Above the treeline: brown autumn alpine turf on gentle ground, pale grey scree fans
    // (limestone/gneiss debris, albedo ~0.25) at 25-38 deg under the cliffs, dark wet rock bands (~0.08) on steep
    // ground streaked by couloirs, and snow (albedo ~0.8, fresh) on everything below ~40 deg, plus snow lines in
    // the couloirs of the steep faces.
    {
      float zn = th_fbm(p.xz * 0.0019 + 4.0) - 0.5;
      float yZ = p.y + 110.0 * zn;
      float alp = smoothstep(545.0, 620.0, yZ) * (1.0 - mudW);
      if (alp > 0.001) {
        // fall-line frame of the local slope: couloirs, rock ribs and debris cones run straight down it
        vec2 dnF = normalize(n.xz + vec2(1e-4, 0.0));
        float acr = dot(p.xz, vec2(-dnF.y, dnF.x));
        // couloirs every ~40-90 m across the face, sharp-floored (ridged noise), braiding slowly with height
        float cw = th_vnoise(vec2(acr * 0.016 + 1.6 * th_vnoise(vec2(p.y * 0.005, acr * 0.003 + 2.0)), p.y * 0.0022));
        float cou = 1.0 - abs(cw * 2.0 - 1.0);
        cou = cou * cou * cou;
        float cou2 = th_vnoise(vec2(acr * 0.07, p.y * 0.012) + 5.0);
        // cliff bands: bedded limestone / gneiss steps 40-70 m high that wrap round the massif at a near-constant
        // height (the tiered look of the Northern Limestone Alps / the Dolomites); risers steep and dark, the
        // ledges between them catch snow and scree
        float bq = (p.y + 60.0 * th_vnoise(p.xz * 0.0028 + 1.0)) / 58.0;
        float bf = fract(bq);
        float riser = smoothstep(0.04, 0.14, bf) * (1.0 - smoothstep(0.52, 0.66, bf)) * step(0.3, th_hash(vec2(floor(bq), 3.0)));
        float turf = alp * (1.0 - smoothstep(0.12, 0.26, slope));
        fa = mix(fa, vec3(0.105, 0.085, 0.055) * (0.75 + 0.5 * m1), turf * 0.85);
        // pale scree fans (albedo ~0.22-0.28) at 20-38 deg, widest under the couloir mouths
        float scree = alp * smoothstep(0.1, 0.2, slope) * (1.0 - smoothstep(0.32, 0.44, slope))
                    * smoothstep(0.25, 0.6, th_fbm(p.xz * 0.006 + 2.0) + 0.45 * cou);
        fa = mix(fa, vec3(0.24, 0.235, 0.23) * (0.8 + 0.4 * cou2), scree * 0.85);
        // steep ground: dark wet rock (~0.05-0.1), streaked along the fall line, darkest on the risers
        float band = alp * smoothstep(0.28, 0.44, slope + 0.1 * riser);
        vec3 rockC = vec3(0.075, 0.075, 0.08) * (0.55 + 0.8 * cou2) * mix(1.0, 0.7, riser);
        // iron-stained / paler limestone bands (strata colour, not shading)
        rockC *= mix(vec3(1.0), vec3(1.5, 1.4, 1.25), smoothstep(0.55, 0.8, th_vnoise(vec2(bq * 3.0, acr * 0.004))) * 0.6);
        fa = mix(fa, rockC, band * 0.88);
        fr = mix(fr, 0.5, band * uWet);
        float sl = 655.0 + 80.0 * zn;
        float hs = smoothstep(sl, sl + 110.0, p.y);
        // fresh autumn snow (albedo ~0.8): sticks below ~40 deg and on the ledges between the risers, never on
        // the risers themselves; fills the couloirs (lines of white down the dark faces) and thins out on the ribs
        float snow = hs * (1.0 - smoothstep(0.38, 0.56, slope + (cou2 - 0.5) * 0.25 + 0.35 * riser - 0.2 * cou));
        snow = max(snow, smoothstep(sl + 15.0, sl + 150.0, p.y) * smoothstep(0.35, 0.7, cou) * (1.0 - 0.6 * riser));
        snow *= smoothstep(0.25, 0.45, th_fbm(p.xz * 0.011 + 9.0) + 0.3);
        fa = mix(fa, vec3(0.78, 0.8, 0.84) * (0.9 + 0.1 * cou2), snow);
        fr = mix(fr, 0.55, snow);
        fn = normalize(mix(fn, n, snow * 0.7));
        snowW = snow * lodT;
      }
    }
    alb = mix(alb, fa, lodT); nw = normalize(mix(nw, fn, lodT)); rough = mix(rough, fr, lodT); ao = mix(ao, fo, lodT);
  }

#ifndef TH_FAR_ONLY
  // washout walls: the road structure in section -- 0.1 m asphalt (road mesh) over a pale compacted gravel
  // sub-base (~0.45 m), then brown fill
  {
    vec2 rel = p.xz - uGap.xz;
    float gs = dot(rel, uGapT), gd = dot(rel, vec2(-uGapT.y, uGapT.x));
    if (abs(gs) < 3.0 && abs(gd) < 7.5) {
      float depth = uGap.y + uGap.w * gs - 0.02 * abs(gd) - p.y;
      float wall = smoothstep(0.35, 0.7, slope);
      float base = smoothstep(0.08, 0.14, depth) * (1.0 - smoothstep(0.5, 0.62, depth + 0.06 * th_vnoise(p.xz * 3.0)));
      vec4 Ab; vec3 Nb; float Rb, Ob;
      th_layer(4.0, 1.1, 1.0, p, dpx, dpy, n, b, Ab, Nb, Rb, Ob);
      float k = base * wall;
      alb = mix(alb, Ab.rgb * vec3(1.05, 1.05, 1.08), k); nw = normalize(mix(nw, Nb, k)); rough = mix(rough, Rb, k);
      // a thin dark tack/binder line right under the asphalt
      alb *= 1.0 - 0.55 * wall * (1.0 - smoothstep(0.1, 0.16, depth)) * smoothstep(0.05, 0.1, depth);
    }
  }
#endif
  vec3 albBase = alb;
  // ---------- fresh slide mud: flow streaks and wet rills aligned with the fall line, warm/cold colour patches
  if (mudW > 0.02) {
    vec2 dn = normalize(n.xz + vec2(1e-4, 0.0));             // downslope (horizontal)
    float across = dot(p.xz, vec2(-dn.y, dn.x));
    float along = dot(p.xz, dn) - p.y * 0.5;
    float fl = th_fbm(vec2(across * 0.35, along * 0.035));
    float rill = smoothstep(0.62, 0.82, th_vnoise(vec2(across * 1.4, along * 0.08)) * 0.8 + th_vnoise(vec2(across * 4.0, along * 0.2)) * 0.3);
    float slopeM = smoothstep(0.08, 0.3, slope);
    alb = mix(alb, alb * (0.55 + 0.8 * smoothstep(0.28, 0.72, fl)), mudW * 0.9 * slopeM);
    // fresh, unweathered subsoil: patches of ochre/grey clay between the dark topsoil smears
    float clay = smoothstep(0.45, 0.8, th_fbm(p.xz * 0.045 + vec2(across, along) * 0.01 + 7.0));
    alb = mix(alb, alb * vec3(1.3, 1.12, 0.88), mudW * clay * 0.55);
    // raw scarp faces: fresh ochre / grey-brown regolith, banded by soil horizons
    float hor = th_vnoise(vec2(dot(p.xz, vec2(0.6, 0.8)) * 0.08, p.y * 0.9));
    alb = mix(alb, alb * mix(vec3(1.55, 1.3, 0.95), vec3(1.2, 1.15, 1.1), hor), scarp * 0.8);
    alb = mix(alb, vec3(dot(alb, vec3(0.3, 0.59, 0.11))) * vec3(0.95, 0.96, 1.0), mudW * (1.0 - clay) * smoothstep(0.4, 0.7, m3) * 0.35);
    alb *= 1.0 - 0.45 * rill * mudW * slopeM * uWet;
    rough = mix(rough, 0.22, rill * mudW * slopeM * uWet);
    // saturated mud: a soft sheen, never a mirror (micro-relief, grit and rain-pitting break the film)
    rough = mix(rough, max(0.8, rough), mudW * uWet * (1.0 - rill * slopeM));
  }

  // ---------- macro variation (kills tiling, adds natural patchiness)
  float mac = th_fbm(p.xz * 0.0045 + 5.0);
  alb *= mix(0.82 + 0.36 * m1, 1.0, snowW);
  alb = mix(alb, alb * vec3(1.06, 1.0, 0.9), smoothstep(0.35, 0.8, mac) * 0.5);
  alb = mix(alb, alb * vec3(0.9, 0.97, 1.05), smoothstep(0.6, 0.2, mac) * 0.5);
  // moss cushions on the cut's ledges and benches (wet: deep olive-green, low albedo ~0.05-0.08)
  float mossL = smoothstep(0.35, 0.05, veg) * (1.0 - rock) * rest;
  alb = mix(alb, alb * vec3(0.55, 0.72, 0.38), mossL * (0.5 + 0.4 * m2));
  // steep rock: large darker/lighter blotches up the face (weathering rind, recent spalls), vertical bias
  float blot = th_fbm(vec2(dot(p.xz, vec2(0.7, 0.7)) * 0.12, p.y * 0.05) + 2.0);
  alb *= mix(1.0, 0.75 + 0.5 * blot, rock * rest * cliff);

  // lifted rock scans must not push bright specks past a plausible albedo
  alb = min(alb, vec3(mix(0.6, 0.86, snowW)));
  // ---------- wetness
  float wet = uWet;
  float flatW = smoothstep(0.86, 0.97, n.y);
  // overhangs and the undersides of ledges stay dry in the rain: paler and matt next to the soaked face around
  // them (on real cuts this dry/wet patchwork is what tells you the rock is wet). Recesses (texture AO) too, a bit.
  float shelter = clamp(smoothstep(0.04, -0.3, n.y) + 0.35 * smoothstep(0.5, 0.15, ao) * smoothstep(0.6, 0.9, slope), 0.0, 1.0)
                * rock * rest * (1.0 - snowW);
  float wetR = wet * (1.0 - 0.8 * shelter);
  // wet rock darkens ~30%, soil/litter ~38%; near-vertical faces drain (wet in streaks, not soaked)
  float porous = mix(0.9, mix(0.9, 0.8, cliff), rock) * (1.0 - 0.8 * snowW);
  alb *= mix(1.0, 1.0 - 0.4 * porous, wetR);
  // wet rock gets glossy; litter, moss, soil and mud stay rough (water soaks in, micro-relief breaks the film)
  float glossy = clamp(rock * rest * (0.55 + 0.45 * cliff), 0.0, 1.0);
  // (a wet rock film is rain-pitted and gritty and the crystal grain pokes through it: the face stays fairly rough,
  //  else the bright overcast sky turns every 45 deg facet into a chalky whitish sheet. Only seepage is glossy.)
  rough = mix(rough, mix(max(0.55, rough * 0.9), max(0.45, rough * 0.72), glossy), wetR);
  // ---------- water on the steep rock: ink stripes, live seeps, iron staining, calcite (see th_seep)
  float face = smoothstep(0.42, 0.75, slope) * rock * (1.0 - gravW) * (1.0 - 0.8 * mudW * (1.0 - smoothstep(0.7, 0.9, slope)));
  // (QA) the washout walls are road fill and sub-base exposed by the storm hours ago: no years-old ink stripes,
  // calcite or iron staining, no blasting half-casts. Skipping the rock-face water there is also what kept the
  // plank-laying view (the trench wall fills the screen) under 50 fps.
  float washZ = 0.0;
#ifndef TH_FAR_ONLY
  {
    vec2 relW = p.xz - uGap.xz;
    washZ = step(abs(dot(relW, uGapT)), 3.2) * step(abs(dot(relW, vec2(-uGapT.y, uGapT.x))), 7.5);
    face *= 1.0 - washZ;
  }
#endif
  float streak = 0.0;
  if (face > 0.02 && dist < 260.0) {
    // two horizontal face coordinates (x for faces looking along z, z for faces looking along x), blended
    vec2 hw = abs(n.zx); hw = hw * hw; hw /= max(hw.x + hw.y, 1e-4);
    vec4 sw = vec4(0.0);
    if (hw.x > 0.02) sw += th_seep(p.x, p.y, 0.0) * hw.x;
    if (hw.y > 0.02) sw += th_seep(p.z, p.y, 17.0) * hw.y;
    sw *= face * (1.0 - smoothstep(160.0, 260.0, dist));
    // rain running off the face in sheets: vertical glossy bands 0.3-1.5 m wide wherever the relief gathers it
    // (sky glints along them are what a wet cut shows from a moving car), broken by ledges every few metres
    if (uFx * uFxv.x > 0.5 && dist < 150.0) {
      float fu = hw.x > hw.y ? p.x : p.z + 9.0;
      float sh = smoothstep(0.56, 0.8, th_vnoise(vec2(fu * 1.1, p.y * 0.07)) * 0.75 + th_vnoise(vec2(fu * 3.3, p.y * 0.35 + 5.0)) * 0.25);
      sh *= face * wetR * (1.0 - smoothstep(90.0, 150.0, dist));
      alb *= 1.0 - 0.18 * sh;
      rough = mix(rough, 0.2, sh * 0.8);
      streak = max(streak, sh * 0.5);
    }
    // ink stripes: blue-black film, matt-ish; iron: ochre-brown; calcite: pale grey-white crust
    alb = mix(alb, alb * vec3(0.16, 0.16, 0.17), min(sw.x * 1.7, 1.0));
    alb = mix(alb, alb * vec3(1.45, 1.05, 0.62) + vec3(0.02, 0.008, 0.0), sw.z * (1.0 - sw.x * 0.5));
    alb = mix(alb, vec3(0.34, 0.335, 0.32), sw.w * 0.45);
    rough = mix(rough, rough * 0.8, sw.x);
    // a live film: dark (wet), near-mirror, flowing (bands of thicker water slide down at ~0.4-0.8 m/s)
    float live = sw.y * wet;
    if (live > 0.01) {
      float flow = th_vnoise(vec2((p.x + p.z) * 9.0, p.y * 2.3 + uTime * 1.9));
      vec3 tg = normalize(vec3(-n.z, 0.0, n.x) + 1e-4);
      nw = normalize(nw + (tg * (flow - 0.5) * 0.5 + vec3(0.0, (flow - 0.5) * 0.25, 0.0)) * live);
      nw = normalize(mix(nw, n, live * 0.5));
      alb *= 1.0 - 0.5 * live;
      rough = mix(rough, 0.05 + 0.08 * flow, live);
    }
    streak = max(streak, max(sw.x * 0.6, sw.y));
  }
#ifndef TH_FAR_ONLY
  // pre-split half-casts on the steepest parts of the cut (near only; faded out before they can alias)
  if (uFx * uFxv.z > 0.5 && face > 0.3 && dist < 45.0 && slope > 0.75) {
    vec2 hwc = abs(n.zx); hwc = hwc * hwc; hwc /= max(hwc.x + hwc.y, 1e-4);
    vec2 gx = hwc.x > 0.02 ? th_halfcast(p.x, p.y, 0.0) : vec2(0.0);
    vec2 gz = hwc.y > 0.02 ? th_halfcast(p.z, p.y, 7.0) : vec2(0.0);
    // concave groove: the normal leans toward the groove axis, i.e. against the direction of increasing u
    vec3 pert = -(vec3(1.0, 0.0, 0.0) * gx.x * gx.y * hwc.x + vec3(0.0, 0.0, 1.0) * gz.x * gz.y * hwc.y);
    vec2 g = vec2(gx.x * hwc.x + gz.x * hwc.y, gx.y * hwc.x + gz.y * hwc.y);
    // only where the pre-split plane survived (large patches of the face), and only at a resolvable scale
    float keep = smoothstep(0.45, 0.6, th_vnoise(vec2(p.x + p.z, p.y) * 0.09 + 4.0)) * smoothstep(0.75, 0.9, slope);
    float aa = 1.0 - smoothstep(0.012, 0.03, length(vec2(length(dpx), length(dpy))));
    float hc = g.y * keep * aa * face;
    if (hc > 0.001) {
      nw = normalize(nw + (pert - n * dot(pert, n)) * 0.9 * keep * aa * face);
      alb *= 1.0 - 0.18 * hc * (1.0 - abs(g.x));
      // rain runs down the grooves: a glossy line in each
      rough = mix(rough, 0.16, hc * wet * (1.0 - abs(g.x)) * 0.8);
      streak = max(streak, hc * 0.5);
    }
  }
#endif
  // soaked bands from the mask (see seepB): darker, glossier, running down the face
  {
    float sb = seepB * wet * (0.6 + 0.4 * th_vnoise(vec2((p.x + p.z) * 0.8, p.y * 0.15)));
    alb *= 1.0 - 0.3 * sb;
    rough = mix(rough, 0.24, sb * 0.75);
    streak = max(streak, sb * 0.6);
  }
  // water trickles out along the bedding joints (texture cavities) too: the dark, glossy crevices of a wet cut
  float crev = face * wet * smoothstep(0.55, 0.2, ao) * 0.6;
  alb *= 1.0 - 0.35 * crev;
  rough = mix(rough, 0.2, crev);
#ifndef TH_FAR_ONLY
  // around each gully fall the rock is soaked black and glossy: spray and the film spreading from the jet, 0.5-2 m
  // either side, widest at the foot where the jet hits the ditch
  if (uFx * uFxv.y > 0.5 && washZ < 0.5 && dist < 200.0 && slope > 0.15 && all(greaterThan(p, uFallMin)) && all(lessThan(p, uFallMax))) {
    float fw = 0.0;
    float fj = 0.25 * (th_vnoise(p.xz * 3.0 + p.y * 2.0) - 0.5);
    for (int i = 0; i < 16; i++) {
      if (i >= uFallN) break;
      vec3 fa = uFallA[i].xyz, fb = uFallB[i].xyz;
      vec3 pa = p - fa, ba = fb - fa;
      if (dot(pa, pa) > 64.0 + dot(ba, ba) * 2.0) continue;
      float hh = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-4), 0.0, 1.0);
      float dd = length(pa - ba * hh);
      float r = uFallA[i].w * 2.0 + 0.45 + 0.5 * hh;
      fw = max(fw, 1.0 - smoothstep(r * 0.35, r, dd + fj));
    }
    fw *= wet * smoothstep(0.15, 0.4, slope) * (1.0 - snowW);
    if (fw > 0.001) {
      alb *= 1.0 - 0.5 * fw;
      rough = mix(rough, 0.1, fw * 0.85);
      float fl = th_vnoise(vec2((p.x + p.z) * 7.0, p.y * 2.0 + uTime * 2.2));
      nw = normalize(mix(nw, n, fw * 0.5) + vec3(0.0, (fl - 0.5) * 0.3, 0.0) * fw);
      streak = max(streak, fw);
    }
  }
#endif
  // splash / capillary zone at the foot of the cut: dark and damp for the first ~0.5-1 m above the ditch
  float footW = face * wet * smoothstep(0.5, 0.95, grav + mud * 0.5) * 0.5;
  alb *= 1.0 - 0.3 * footW;
  // puddles on flat gravel / mud / ditch
  float flatP = smoothstep(0.955, 0.99, n.y);   // water only pools on (nearly) level ground
  // gravel drains: only scattered pools in its low spots; mud / ditch floors hold more water
  float pud = smoothstep(0.64 - 0.08 * mud, 0.76 - 0.06 * mud, th_fbm(p.xz * 0.35) + (1.0 - ao) * 0.35 + mud * 0.18) * flatP * max(grav, mud) * wet;
  pud *= 1.0 - smoothstep(60.0, 120.0, dist);
  alb = mix(alb, alb * 0.45, pud);
  rough = mix(rough, 0.03, pud);
  nw = normalize(mix(nw, n, pud * 0.9));

  // the multiplicative passes above (streaks, rills, blotches, wetness, seepage) must not stack into black
  alb = max(alb, albBase * mix(mix(0.3, 0.14, streak), 0.14, pud));
  // ---------- baked AO (vertex) + texture AO
  // (vertex AO is 30 m Cycles AO at ~0.5-2 m vertex spacing: it only darkens the ambient term, floored, and never
  //  the albedo -- multiplied into both it reads as soft dark smudges)
  float aoTot = clamp(aoV, 0.35, 1.0) * mix(1.0, ao, 0.65);
  thAlbedo = alb; thNormalW = nw; thRough = clamp(rough, 0.03, 1.0); thAO = aoTot;
  // grazing-angle specular occlusion by grass blades, needles and pebbles (not on bare wet rock or puddles)
  thGraze = (1.0 - 0.2 * glossy) * (1.0 - pud) * (1.0 - 0.8 * streak);
#if defined(TH_DEBUG) && TH_DEBUG == 4
  thRough = 1.0; thGraze = 1.0;
#endif
}
`;

const VERT_PARS = /* glsl */`
attribute vec4 masks;
varying vec4 vMasks;
varying vec3 vWPos;
varying vec3 vWNrm;
`;

// near-look -> far-look blend distances per quality preset (m)
const LOD_DIST = { Ultra: [110, 260], High: [85, 210], Medium: [50, 125], Low: [35, 90] };

/**
 * Create the terrain material. opts: {far: bool (far-only look), antiTile: bool, lodNear, lodFar}
 * Returns a MeshStandardMaterial with .userData.update(dt) that tracks ctx.env.wetness.
 */
export async function createTerrainMaterial(ctx, opts = {}) {
  const arr = await loadTerrainArrays(ctx);
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 });
  const avg = [];
  for (let i = 0; i < NL; i++) avg.push(new THREE.Vector3(arr.avg[i * 3], arr.avg[i * 3 + 1], arr.avg[i * 3 + 2]));
  const q = ctx.config?.quality?.name;
  const ld = LOD_DIST[q] || LOD_DIST.High;
  const uniforms = {
    uAlb: { value: arr.alb }, uNrm: { value: arr.nrm },
    uTile: { value: TILE.slice() }, uNStr: { value: NSTR.slice() }, uAvg: { value: avg },
    uTint: { value: TINT.map((t) => new THREE.Vector4(...t)) },
    uWet: { value: ctx.env?.wetness ?? 0.75 }, uTime: { value: 0 },
    uLodDist: { value: new THREE.Vector2(opts.lodNear ?? ld[0], opts.lodFar ?? ld[1]) },
    uGap: { value: new THREE.Vector4(0, -1e4, 0, 0) }, uGapT: { value: new THREE.Vector2(1, 0) },
    uFx: { value: 1 }, uFxv: { value: new THREE.Vector4(1, 1, 1, 1) },
    uFallMin: { value: new THREE.Vector3(1e9, 1e9, 1e9) }, uFallMax: { value: new THREE.Vector3(-1e9, -1e9, -1e9) },
    uFallA: { value: Array.from({ length: 16 }, () => new THREE.Vector4(0, -1e4, 0, 0.1)) },
    uFallN: { value: 0 },
    uFallB: { value: Array.from({ length: 16 }, () => new THREE.Vector4(0, -1e4 - 1, 0, 0)) },
  };
  const road = ctx.road, gapS = road?.markers?.gap;
  if (road && gapS !== undefined) {
    const p = road.pointAt(gapS), t = road.tangentAt(gapS);
    const th = Math.hypot(t.x, t.z) || 1;
    uniforms.uGap.value.set(p.x, p.y, p.z, t.y / th);
    uniforms.uGapT.value.set(t.x / th, t.z / th);
  }
  const antiTile = opts.antiTile ?? (q === 'Ultra' || q === 'High');
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    if (opts.far) sh.defines = { ...(sh.defines || {}), TH_FAR_ONLY: '' };
    if (antiTile) sh.defines = { ...(sh.defines || {}), TH_ANTITILE: '' };
    const dbg = +(new URLSearchParams(location.search).get('terrainDebug') || 0);
    if (dbg) sh.defines = { ...(sh.defines || {}), TH_DEBUG: dbg };
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\n' + VERT_PARS)
      .replace('#include <fog_vertex>', `#include <fog_vertex>
        vMasks = masks;
        vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        vWNrm = normalize(mat3(modelMatrix) * objectNormal);`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + FRAG_PARS)
      .replace('#include <map_fragment>', 'th_splat();\n diffuseColor.rgb = thAlbedo;')
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = thRough;')
      .replace('#include <normal_fragment_maps>', 'normal = normalize((viewMatrix * vec4(thNormalW, 0.0)).xyz);')
      .replace('#include <dithering_fragment>', `#include <dithering_fragment>
      #ifdef TH_DEBUG
        #if TH_DEBUG == 1
          gl_FragColor = vec4(vMasks.g, vMasks.b, vMasks.a, 1.0);
        #elif TH_DEBUG == 2
          gl_FragColor = vec4(thAlbedo, 1.0);
        #elif TH_DEBUG == 4
        #else
          gl_FragColor = vec4(vec3(vMasks.r), 1.0);
        #endif
      #endif`)
      .replace('#include <aomap_fragment>', `{
        float ambientOcclusion = thAO;
        reflectedLight.indirectDiffuse *= ambientOcclusion;
        #if defined( USE_ENVMAP ) && defined( STANDARD )
          float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
          reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
        #endif
        reflectedLight.indirectSpecular *= mix(1.0, ambientOcclusion, 0.5);
        {
          float nv = saturate( dot( geometryNormal, geometryViewDir ) );
          float hocc = mix(1.0, smoothstep(0.0, 0.7, nv) * 0.55, thGraze);
          reflectedLight.indirectSpecular *= hocc;
          reflectedLight.directSpecular *= hocc;
        }
      }`);
  };
  m.customProgramCacheKey = () => `terrain3|${opts.far ? 'far' : 'near'}|${antiTile ? 'at' : ''}`;
  m.userData.uniforms = uniforms;
  m.userData.update = (dt) => {
    uniforms.uWet.value = ctx.env?.wetness ?? uniforms.uWet.value;
    uniforms.uTime.value += dt;
  };
  return m;
}

// ---------------------------------------------------------------------------------------------
// Running water (ditch, gully rivulets and falls): thin, soft-edged ribbons built by terrain.js.
// UV0 = (across, along) in meters (along = downstream distance); attribute wfx = (across01, foam, speed, turbidity).
// A shallow film over the ditch floor / rock: very dark turbid water (silt-laden runoff: albedo ~0.03-0.06, warm),
// near-mirror between the wavelets, white water (albedo ~0.5, rough) where the bed is steep.
// ---------------------------------------------------------------------------------------------
export function createWaterMaterial(ctx) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.06, metalness: 0, transparent: true, depthWrite: false, side: THREE.DoubleSide });
  const uniforms = { uTime: { value: 0 }, uWet: { value: ctx.env?.wetness ?? 0.75 } };
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec4 wfx;\nvarying vec4 vWfx;\nvarying vec2 vWuv;')
      .replace('#include <fog_vertex>', '#include <fog_vertex>\n vWfx = wfx; vWuv = uv;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
uniform float uTime; uniform float uWet;
varying vec4 vWfx; varying vec2 vWuv;
${GLSL_NOISE}
float wtH; float wtFoam; float wtSteep;`)
      .replace('#include <map_fragment>', `{
        float t = uTime, sp = vWfx.z;
        vec2 q = vWuv;
        // wavelets stretched along the flow, carried downstream; a faster, finer set on top
        float n1 = th_vnoise(vec2(q.x * 7.0, q.y * 2.2 - t * sp * 2.2));
        float n2 = th_vnoise(vec2(q.x * 15.0 + 3.0, q.y * 5.5 - t * sp * 3.4));
        float n3 = th_vnoise(vec2(q.x * 3.0 + 7.0, q.y * 0.9 - t * sp * 1.1));
        wtH = (n1 * 0.5 + n2 * 0.3 + n3 * 0.2);
        float foamA = vWfx.y;
        float steepF = smoothstep(0.35, 0.8, foamA);
        wtSteep = steepF;
        // across the ribbon: the flow is deepest in the middle; the margins are a thin film gliding over the rock
        float xc = abs(vWfx.x - 0.5) * 2.0;
        // A small gully fall (5-20 l/s) over a road cut is not a white sheet: it splits into a few aerated threads
        // (1-6 cm wide, white where they hit a step, drawn out along the fall), and between them a clear film glides
        // over black wet rock. Threads wander across the ribbon, braid and die out; clusters of drops break away.
        float lane = th_vnoise(vec2(q.x * 6.0 + th_vnoise(vec2(q.y * 0.9, 3.0)) * 2.5, q.y * 0.35 - t * sp * 0.5));
        float st = th_vnoise(vec2(q.x * 17.0 + lane * 3.0, q.y * 0.5 - t * sp * 1.7)) * 0.55
                 + th_vnoise(vec2(q.x * 41.0 + 5.0, q.y * 1.6 - t * sp * 2.8)) * 0.3
                 + th_vnoise(vec2(q.x * 9.0 + 2.0, q.y * 3.5 - t * sp * 4.0)) * 0.15;
        // steps in the bed (every ~0.6-1.5 m of fall) aerate the whole jet for a short stretch
        float step_ = smoothstep(0.62, 0.9, th_vnoise(vec2(q.x * 2.0, q.y * 1.1 - t * sp * 0.3)));
        // free-falling jet (speed code 2.8): aerated along most of its length, splitting into 2-4 threads
        float jet = smoothstep(2.3, 2.7, sp);
        float threads = smoothstep(0.48 - 0.06 * jet, 0.74 - 0.04 * jet, st + 0.18 * step_ - 0.25 * xc * xc);
        // the jet breaks up: gaps where it thins to a clear film, drop clusters in between
        threads *= mix(1.0, smoothstep(0.25, 0.55, th_vnoise(vec2(q.x * 5.0, q.y * 1.3 - t * sp * 1.2))), jet);
        float flecks = smoothstep(0.74, 0.93, wtH + foamA * 0.3);
        wtFoam = mix(flecks, threads, steepF) * min(foamA * 1.2, 1.0);
        wtFoam = clamp(wtFoam * (0.65 + 0.5 * n1), 0.0, 1.0);
        // silt-laden runoff (suspended fines from the fresh slide): warm, dark, turbid
        vec3 turb = mix(vec3(0.03, 0.029, 0.027), vec3(0.07, 0.052, 0.032), vWfx.w);
        // aerated water in a shaded gully under an overcast sky: pale grey, not paper white (albedo ~0.35-0.45)
        vec3 foamC = mix(vec3(0.3, 0.3, 0.29), vec3(0.44, 0.44, 0.43), n2) * mix(1.0, 1.12, steepF);
        // silt-laden runoff (the scar rills) foams tan-brown, not white
        foamC = mix(foamC, vec3(0.2, 0.16, 0.11), smoothstep(0.6, 0.9, vWfx.w));
        diffuseColor.rgb = mix(turb, foamC, wtFoam);
        float edge = smoothstep(0.0, 0.3, min(vWfx.x, 1.0 - vWfx.x));
        // a thin film over steep rock is mostly see-through (the dark wet rock shows), aerated threads are opaque
        float film = mix(0.78, 0.4 + 0.25 * (1.0 - xc), steepF);
        diffuseColor.a = edge * mix(film, 0.9, wtFoam) * mix(0.6, 1.0, uWet);
      }`)
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = mix(0.07 + 0.1 * wtH, 0.55, wtFoam);')
      .replace('#include <aomap_fragment>', `#include <aomap_fragment>
        // the ditch sits under a 6-14 m rock cut and the rills in V-gullies: most of the mirror hemisphere is dark
        // rock / forest, not sky (the env map has no local occlusion)
        // (a jet down the face of the cut looks out over the open valley instead: much less occluded)
        reflectedLight.indirectSpecular *= mix(mix(0.32, 0.75, wtSteep), 0.8, wtFoam) * mix(1.0, 0.55, smoothstep(0.6, 0.9, vWfx.w));`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
      {
        // bump from the wavelet height field (perturbNormalArb, surface-gradient form)
        float hb = (wtH + wtFoam * 0.6) * 0.012;
        vec2 dh = vec2(dFdx(hb), dFdy(hb));
        vec3 sp_ = -vViewPosition;
        vec3 sx = dFdx(sp_), sy = dFdy(sp_);
        vec3 R1 = cross(sy, normal), R2 = cross(normal, sx);
        float det = dot(sx, R1);
        vec3 grad = sign(det) * (dh.x * R1 + dh.y * R2);
        normal = normalize(abs(det) * normal - grad);
      }`);
  };
  m.customProgramCacheKey = () => 'terrain-water2';
  m.userData.update = (dt) => { uniforms.uTime.value += dt; uniforms.uWet.value = ctx.env?.wetness ?? uniforms.uWet.value; };
  return m;
}
