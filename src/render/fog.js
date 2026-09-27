/*
 * LANDSLIDE height fog / aerial perspective / valley mist  (RENDER workstream)
 * =============================================================================
 *
 * installFog(ctx) replaces three's fog ShaderChunks globally, so EVERY built-in material (Standard, Physical,
 * Basic, Lambert, Phong, Points, Sprite, Line, Shadow, Toon, Matcap) gets the height fog automatically, as long as
 * `material.fog === true` (the default for those materials) and `scene.fog` is set (env sets a THREE.FogExp2).
 *
 * The model (all per-pixel, analytic, no textures):
 *   1. Aerial perspective "haze": exponential height fog with a large scale height (~1 km). It uses chromatic
 *      extinction (blue is scattered more), so far mountains turn into blue-grey silhouettes. Its colour is `scene.fog.color`.
 *   2. Valley fog: a second exponential layer with a short scale height (~50 m). It is dense in the valley below the road
 *      and thins out above the road.
 *   3. Mist sheets: two noise-modulated horizontal slabs below road level (fBm in world XZ, drifting with the wind).
 *      They look like layered clouds in the valley. The cost is one ray/slab intersection plus 2 noise taps per slab,
 *      and only when the view ray actually crosses a slab before reaching the surface.
 *   4. Low cloud deck: density ramps up above ~560 m (the road is at y 30..100), with drifting fBm patchiness, so the
 *      far summits and high ridges dissolve into the overcast like real Alpine rain. The sky dome's rays stop below
 *      the deck (the HDRI already is cloud), so the summit/sky match comes from the haze + deck colour being tuned to
 *      the horizon sky radiance. Custom shaders can skip the deck with `#define HFOG_NO_CLOUD`.
 *   5. Sun inscatter: a soft forward-scatter glow toward the sun direction, plus a lightning boost.
 *   6. Falling rain: a grey, height-independent extinction below the cloud base (hfogRain.x), derived by env from
 *      env.rain with the visible extinction of rain, sigma = 0.25 R^0.63 km^-1 (see rainExtinction()).
 *
 * The shader works out the world position itself from mvPosition and viewMatrix (world = camera + transpose(R) * mv),
 * so it is correct for instancing, batching, skinning, morphs and displacement without any extra work.
 *
 * -----------------------------------------------------------------------------------------------------------------
 * HOW A CUSTOM ShaderMaterial GETS THE FOG
 * -----------------------------------------------------------------------------------------------------------------
 *   const mat = new THREE.ShaderMaterial({
 *     fog: true,                                            // required: three only defines USE_FOG when this is true
 *     uniforms: THREE.UniformsUtils.merge([
 *       THREE.UniformsLib.fog,                              // fogColor/fogDensity (+ the height-fog uniforms, already added)
 *       ctx.env?.fogUniforms ?? {},                         // explicit shared height-fog uniforms (same objects; harmless twice)
 *       { myUniform: { value: 1 } },
 *     ]),
 *     vertexShader: `
 *       #include <common>
 *       #include <fog_pars_vertex>
 *       void main() {
 *         vec3 transformed = position;                      // (object space; not needed by the fog, kept for convention)
 *         vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);   // include instanceMatrix yourself if instanced
 *         gl_Position = projectionMatrix * mvPosition;
 *         #include <fog_vertex>                             // AFTER mvPosition exists (vec4 or vec3 both work)
 *       }`,
 *     fragmentShader: `
 *       #include <common>
 *       #include <fog_pars_fragment>
 *       void main() {
 *         gl_FragColor = vec4(color, 1.0);                  // LINEAR colour (post does tone mapping + sRGB)
 *         #include <fog_fragment>                           // applies height fog to gl_FragColor.rgb
 *       }`,
 *   });
 *
 * Notes:
 *   - UniformsUtils.merge()/clone() keep the height-fog uniforms SHARED. Their values are plain {x,y,z,w} objects, which
 *     three copies by reference, so env updates reach every material. Never replace `.value` on them. Mutate the fields.
 *   - If a material ends up without the uniforms (all zero), the fog degrades to "no fog" (no NaNs).
 *   - ADDITIVE-BLENDED materials (rain streaks, sparks, glows) must NOT get inscatter, or they turn into fog-coloured
 *     sheets. Either set `fog: false`, or keep `fog: true` and set `material.defines = { HFOG_TRANSMITTANCE_ONLY: '' }`
 *     (assign it after construction for built-in materials; ShaderMaterial also accepts it in the constructor). That
 *     only attenuates the colour by the fog transmittance, which is correct for additive light.
 *   - Tunnel: fog.js also appends to three's `lights_fragment_maps`, so inside the road tunnel every built-in lit
 *     material (with fog) loses its sky light (IBL + hemisphere). Opt out with `material.defines.HTUNNEL_OFF`.
 *   - You can use the fog function yourself: `vec3 hfog_apply(vec3 col, vec3 worldPosMinusCamera, vec3 cameraPos)`
 *     is available inside `#ifdef USE_FOG` after `#include <fog_pars_fragment>`. Use `hfog_transmittance(...)`
 *     for the per-channel transmittance only.
 *   - Varyings added: `vHFogRel` (world position minus camera position). `vFogDepth` still exists for old code.
 *   - Shared uniforms (read only for other systems): see FOG_UNIFORM_DOC below or ctx.env.fogUniforms.
 *   - ctx.env.fogUniforms also contains `hEnv` = {x: time s, y: wetness 0..1, z: rain 0..1, w: lightning 0..1}.
 *     Declare `uniform vec4 hEnv;` in your own shader to use it (fog_pars_* do not declare it).
 * -----------------------------------------------------------------------------------------------------------------
 */
import * as THREE from 'three';

// ---------------------------------------------------------------------------------------------------------------
// GLSL library (self-contained: does not rely on <common> helpers). Uniform names are prefixed with hfog.
// ---------------------------------------------------------------------------------------------------------------
export const FOG_UNIFORM_DOC = `
  hfogA    x haze density at hfogC.x (1/m)   y haze falloff (1/m)   z valley density at hfogC.y (1/m)   w valley falloff (1/m)
  hfogB    rgb valley/mist inscatter colour (linear)   w global fog multiplier (0 = off)
  hfogC    x haze reference height (m)   y valley reference height (m)   z sun glow strength   w sun glow exponent
  hfogSun  xyz unit direction toward the sun (world)   w lightning flash 0..1
  hfogMist x sheet-1 centre y   y sheet-1 half thickness   z sheet-2 centre y   w sheet-2 half thickness (m)
  hfogMist2 x mist density (1/m)   y noise frequency (1/m)   z time (s)   w sheet-1 coverage 0..1
  hfogWind xy wind drift (m/s, world xz)   z mist-sheet brightness multiplier   w sheet-2 coverage 0..1
  hfogCloud x cloud-deck base y (m)   y soft ramp (m)   z density inside the deck (1/m, 0 = off)   w patchiness 0..1
  hfogCloud2 rgb cloud-deck inscatter colour (linear)   w noise frequency (1/m)
  hfogRain x grey extinction of the falling rain below the cloud base (1/m)   yzw unused
  hTun0..3 xyz road-centreline points through the tunnel, w distance from the portal (m)
  hTunP    x tunnel half width (0 = off)  y ramp length (m)  z residual sky light  w clear height (m)
`;

export const FOG_GLSL_UNIFORMS = /* glsl */`
uniform vec4 hfogA;
uniform vec4 hfogB;
uniform vec4 hfogC;
uniform vec4 hfogSun;
uniform vec4 hfogMist;
uniform vec4 hfogMist2;
uniform vec4 hfogWind;
uniform vec4 hfogCloud;
uniform vec4 hfogCloud2;
uniform vec4 hfogRain;
uniform vec4 hTun0;
uniform vec4 hTun1;
uniform vec4 hTun2;
uniform vec4 hTun3;
uniform vec4 hTunP;
`;

// requires: uniform vec3 fogColor; + FOG_GLSL_UNIFORMS
export const FOG_GLSL_FUNCS = /* glsl */`
float hfog_hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float hfog_vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hfog_hash(i), b = hfog_hash(i + vec2(1.0, 0.0));
  float c = hfog_hash(i + vec2(0.0, 1.0)), d = hfog_hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float hfog_fbm(vec2 p) {
  float s = 0.5 * hfog_vnoise(p);
  p = mat2(1.6, 1.2, -1.2, 1.6) * p + vec2(3.1, 1.7);
  s += 0.3 * hfog_vnoise(p);
  p = mat2(1.6, 1.2, -1.2, 1.6) * p + vec2(3.1, 1.7);
  s += 0.2 * hfog_vnoise(p);
  return s;
}
// optical depth of an exponential layer dens*exp(-fall*(y-href)) along a ray (origin y0, dir.y rdy, length t)
float hfog_layerOD(float dens, float fall, float href, float y0, float rdy, float t) {
  float base = dens * exp(clamp(-fall * (y0 - href), -80.0, 30.0));
  float x = fall * rdy * t;
  float g = abs(x) < 1e-3 ? 1.0 - 0.5 * x : (1.0 - exp(-clamp(x, -40.0, 80.0))) / x;
  return base * t * g;
}
// density of a mist sheet at p. Denser cloud also stands taller (billowy tops), bright tops, darker bases.
// returns (density, brightness)
vec2 hfog_mistDensity(vec3 p, float yc, float hh, float cov) {
  float fr = hfogMist2.y;
  float hs = (p.y - yc) / hh;
  vec2 q = (p.xz - hfogWind.xy * hfogMist2.z) * fr + vec2(yc * 0.0137, yc * 0.0071);
  q += vec2(hs * 0.35, -hs * 0.25);          // shear with height: stacked samples differ -> reads as volume
  float n = hfog_fbm(q);
  // fbm is ~N(0.5, 0.11): map coverage (fraction of the sheet that is cloud) to a threshold on n
  float t0 = 0.5 + (0.5 - clamp(cov, 0.0, 1.0)) * 0.45;
  float c = smoothstep(t0, t0 + 0.16, n);
  float top = 2.0 * hh * (0.3 + 0.7 * c);    // local cloud height above the sheet base
  float u = (p.y - (yc - hh)) / max(top, 1e-3);
  float prof = max(1.0 - (2.0 * u - 1.0) * (2.0 * u - 1.0), 0.0);
  return vec2(prof * c, 0.62 + 0.5 * clamp(u, 0.0, 1.0));
}
// returns (optical depth, brightness)
vec2 hfog_sheetOD(float yc, float hh, float cov, vec3 ro, vec3 rd, float t) {
  if (hh <= 0.0) return vec2(0.0, 1.0);
  float ta, tb;
  if (abs(rd.y) < 1e-4) {
    if (abs(ro.y - yc) > hh) return vec2(0.0, 1.0);
    ta = 0.0; tb = t;
  } else {
    float t1 = (yc - hh - ro.y) / rd.y;
    float t2 = (yc + hh - ro.y) / rd.y;
    ta = max(min(t1, t2), 0.0);
    tb = min(max(t1, t2), t);
  }
  float L = tb - ta;
  if (L <= 0.0) return vec2(0.0, 1.0);
  vec2 a = hfog_mistDensity(ro + rd * mix(ta, tb, 0.3), yc, hh, cov);
  vec2 b = hfog_mistDensity(ro + rd * mix(ta, tb, 0.75), yc, hh, cov);
  float d = a.x + b.x;
  float br = (a.x * a.y + b.x * b.y) / max(d, 1e-4);
  // sheets thin out with distance and at grazing view angles, so seen edge-on they never read as a flat milk
  // floor; they show when looking down into the valley (the smooth haze/valley layers own the horizon)
  float fade = (1.0 - smoothstep(700.0, 3500.0, ta)) * smoothstep(0.03, 0.16, abs(rd.y));
  return vec2(0.5 * d * L * hfogMist2.x * fade, br);
}
// Low cloud deck: density ramps linearly from 0 at the base to full over hfogCloud.y metres, constant above. The
// integral along the ray is analytic (piecewise quadratic in y). Patchiness: fbm at the ray's point where it enters
// the deck and at the end point, drifting with the wind, so summits and ridges drift in and out of the cloud.
float hfog_cloudG(float y) {
  float h = max(y - hfogCloud.x, 0.0), r = max(hfogCloud.y, 1.0);
  return h < r ? 0.5 * h * h / r : h - 0.5 * r;
}
float hfog_cloudOD(vec3 ro, vec3 rd, float t) {
  #ifdef HFOG_NO_CLOUD
    return 0.0;
  #else
  if (hfogCloud.z <= 0.0) return 0.0;
  float y1 = ro.y + rd.y * t;
  if (max(ro.y, y1) <= hfogCloud.x) return 0.0;
  float L;
  if (abs(rd.y) < 1e-4) L = clamp((ro.y - hfogCloud.x) / max(hfogCloud.y, 1.0), 0.0, 1.0) * t;
  else L = (hfog_cloudG(y1) - hfog_cloudG(ro.y)) / rd.y;
  // noise at the deck entry point (or the camera if already inside) and at the end point
  float te = abs(rd.y) < 1e-4 ? 0.0 : clamp((hfogCloud.x + 0.5 * hfogCloud.y - ro.y) / rd.y, 0.0, t);
  vec2 drift = hfogWind.xy * hfogMist2.z * 2.5;
  vec3 pe = ro + rd * te, p1 = ro + rd * t;
  float n = 0.5 * (hfog_fbm((pe.xz - drift) * hfogCloud2.w) + hfog_fbm((p1.xz - drift) * hfogCloud2.w + vec2(0.0, (p1.y - hfogCloud.x) * hfogCloud2.w)));
  float cpatch = mix(1.0, smoothstep(0.28, 0.62, n) * 1.9, hfogCloud.w);
  return max(L, 0.0) * hfogCloud.z * cpatch;
  #endif
}
// per-channel optical depth; also returns the inscatter colour through inCol
vec3 hfog_od(vec3 rel, vec3 camPos, out vec3 inCol) {
  inCol = fogColor;
  float mul = hfogB.w;
  float t = length(rel);
  if (mul <= 0.0 || t < 1e-3) return vec3(0.0);
  vec3 rd = rel / t;
  float odH = hfog_layerOD(hfogA.x, hfogA.y, hfogC.x, camPos.y, rd.y, t) * mul;
  float odV = hfog_layerOD(hfogA.z, hfogA.w, hfogC.y, camPos.y, rd.y, t) * mul;
  // patchy valley fog: low-frequency density variation around the end point, drifting with the wind
  vec2 vq = (camPos.xz + rel.xz - hfogWind.xy * hfogMist2.z * 0.6) * 0.0032;
  odV *= mix(0.4, 1.6, hfog_vnoise(vq));
  float odM = 0.0, mBr = 1.0;
  if (hfogMist2.x > 0.0) {
    vec2 m1 = hfog_sheetOD(hfogMist.x, hfogMist.y, hfogMist2.w, camPos, rd, t);
    vec2 m2 = hfog_sheetOD(hfogMist.z, hfogMist.w, hfogWind.w, camPos, rd, t);
    odM = (m1.x + m2.x) * mul;
    mBr = (m1.x * m1.y + m2.x * m2.y) / max(m1.x + m2.x, 1e-5);
  }
  float odC = hfog_cloudOD(camPos, rd, t) * mul;
  // falling rain: drops of 0.5-3 mm scatter geometrically (grey, height independent) up to the cloud base
  float odR = 0.0;
  if (hfogRain.x > 0.0) {
    float tR = (rd.y > 1e-4 && hfogCloud.x > 0.0) ? min(t, max(hfogCloud.x - camPos.y, 0.0) / rd.y) : t;
    odR = hfogRain.x * tR * mul;
  }
  float sumOD = odH + odV + odM + odC + odR + 1e-6;
  vec3 mistCol = hfogB.rgb * max(hfogWind.z, 0.0) * mBr;
  inCol = (fogColor * (odH + odR) + hfogB.rgb * odV + mistCol * odM + hfogCloud2.rgb * odC) / sumOD;
  float cs = max(dot(rd, hfogSun.xyz), 0.0);
  inCol *= (1.0 + hfogC.z * pow(cs, max(hfogC.w, 1.0))) * (1.0 + hfogSun.w * 2.5);
  // chromatic extinction: haze aerosols scatter blue more strongly (aerial perspective); mist, cloud and rain are grey
  return min(odH * vec3(0.8, 1.0, 1.25) + vec3(odV + odM + odC + odR), vec3(40.0));
}
// ---- tunnel occlusion of sky light (IBL + hemi) ------------------------------------------------------------
// hTun0..3: xyz points on the road centreline through the tunnel, w = distance from the portal (m)
// hTunP: x half width (m, 0 = off), y ramp length from the portal (m), z residual sky light inside, w clear height (m)
void htunnel_seg(vec3 p, vec4 a, vec4 b, bool first, inout float bestD, inout float bestS, inout float bestY) {
  vec2 ab = b.xz - a.xz;
  float tt = dot(p.xz - a.xz, ab) / max(dot(ab, ab), 1e-4);
  float t = clamp(tt, 0.0, 1.0);
  float d = length(p.xz - (a.xz + ab * t));
  if (d < bestD) {
    bestD = d;
    bestS = mix(a.w, b.w, first ? min(tt, 1.0) : t);
    bestY = mix(a.y, b.y, t);
  }
}
float htunnel_occ(vec3 p) {
  if (hTunP.x <= 0.0) return 1.0;
  vec3 mid = 0.5 * (hTun0.xyz + hTun3.xyz);
  if (distance(p, mid) > 0.5 * hTun3.w + 25.0) return 1.0;
  float bd = 1e9, bs = -1.0, by = 0.0;
  htunnel_seg(p, hTun0, hTun1, true, bd, bs, by);
  htunnel_seg(p, hTun1, hTun2, false, bd, bs, by);
  htunnel_seg(p, hTun2, hTun3, false, bd, bs, by);
  float lat = 1.0 - smoothstep(hTunP.x - 1.0, hTunP.x + 1.5, bd);
  float h = p.y - by;
  float vert = (1.0 - smoothstep(hTunP.w - 0.5, hTunP.w + 2.0, h)) * smoothstep(-4.0, -2.0, h);
  float along = smoothstep(0.0, hTunP.y, bs) * (1.0 - smoothstep(hTun3.w + 3.0, hTun3.w + 8.0, bs));
  return mix(1.0, hTunP.z, lat * vert * along);
}
vec3 hfog_transmittance(vec3 rel, vec3 camPos) {
  vec3 inCol;
  return exp(-hfog_od(rel, camPos, inCol));
}
vec3 hfog_apply(vec3 col, vec3 rel, vec3 camPos) {
  vec3 inCol;
  vec3 T = exp(-hfog_od(rel, camPos, inCol));
  #ifdef HFOG_TRANSMITTANCE_ONLY
    return col * T;
  #else
    return col * T + inCol * (1.0 - T);
  #endif
}
`;

const CHUNK_PARS_VERTEX = /* glsl */`
#ifdef USE_FOG
  varying float vFogDepth;
  varying vec3 vHFogRel;
#endif
`;
const CHUNK_VERTEX = /* glsl */`
#ifdef USE_FOG
  vFogDepth = - mvPosition.z;
  // world-space offset from the camera: transpose(R_view) * mv  (w = 0 drops the translation)
  vHFogRel = ( vec4( mvPosition.xyz, 0.0 ) * viewMatrix ).xyz;
#endif
`;
const CHUNK_PARS_FRAGMENT = /* glsl */`
#ifdef USE_FOG
  uniform vec3 fogColor;
  varying float vFogDepth;
  varying vec3 vHFogRel;
  #ifdef FOG_EXP2
    uniform float fogDensity;
  #else
    uniform float fogNear;
    uniform float fogFar;
  #endif
  ${FOG_GLSL_UNIFORMS}
  ${FOG_GLSL_FUNCS}
#endif
`;
const CHUNK_FRAGMENT = /* glsl */`
#ifdef USE_FOG
  gl_FragColor.rgb = hfog_apply( gl_FragColor.rgb, vHFogRel, cameraPosition );
#endif
`;

// Appended to three's lights_fragment_maps: inside the road tunnel the sky IBL / hemisphere light fades out
// (world position from the fog varying, so it only applies to materials with fog, i.e. all built-in lit ones).
const CHUNK_TUNNEL = /* glsl */`
#if defined( USE_FOG ) && !defined( HTUNNEL_OFF )
{
  float hTunOcc = htunnel_occ( cameraPosition + vHFogRel );
  #if defined( RE_IndirectDiffuse )
    irradiance *= hTunOcc;
    iblIrradiance *= hTunOcc;
  #endif
  #if defined( RE_IndirectSpecular )
    radiance *= hTunOcc;
    clearcoatRadiance *= hTunOcc;
  #endif
}
#endif
`;

// ---------------------------------------------------------------------------------------------------------------
// Shared uniforms (plain objects -> survive UniformsUtils.clone by reference)
// ---------------------------------------------------------------------------------------------------------------
const v4 = (x = 0, y = 0, z = 0, w = 0) => ({ x, y, z, w });

/** Default look parameters (world heights in metres; road runs y = 40..98, valley floor ~ -250). */
export const FOG_DEFAULTS = {
  enabled: 1,
  // Aerosol haze (chromatic). Rain-washed Alpine air is clean; what veils the valley in rain is the rain itself
  // (rainExtinction, set by env from env.rain) and the humid valley mist. Total near the road at env.rain 0.35
  // (~4 mm/h): ~0.00085/m -> meteorological visibility ~4.5 km; the far side of the valley (1.5 km) keeps ~25 %
  // of its contrast, the far range (3 km+) is a pale silhouette.
  hazeDensity: 0.00024,      // 1/m at hazeRef
  hazeFalloff: 1 / 1100,     // 1/m
  hazeRef: 0,
  valleyDensity: 0.00018,    // 1/m at valleyRef (~0.00012 at road level, ~0.0008 on the valley floor)
  valleyFalloff: 1 / 164,
  valleyRef: 0,
  valleyColor: [0.7, 0.74, 0.79],  // linear (brightness relative to fogColor matters)
  sunGlow: 0.2, sunGlowExp: 5,
  mist1Y: -35, mist1H: 20, mist2Y: -135, mist2H: 45,
  mistDensity: 0, mistFreq: 1 / 240, mistCoverage: 0.28, mistCoverage2: 0.42, mistBright: 1.0,
  wind: [1.2, 0.5],          // m/s drift of the mist noise
  // low cloud deck: summits and high ridges disappear into the overcast (road is at y 30..100)
  cloudBase: 420, cloudRamp: 300, cloudDensity: 0.0042, cloudPatch: 0.9, cloudFreq: 1 / 700,
  cloudColor: [0.62, 0.67, 0.74],
  // grey extinction of the falling rain (1/m) below the cloud base. env derives it from env.rain with the
  // visible-light extinction of rain, sigma = 0.25 R^0.63 km^-1 (Atlas 1953; R in mm/h, R = 1.2 * 30^rain as in
  // particles.js): 0.35 -> 4 mm/h -> 0.0006/m, 0.5 -> 6.6 mm/h -> 0.00083/m.
  rainExtinction: 0.0006,
};
/** Visible extinction coefficient (1/m) of rain for env.rain 0..1 (same rain-rate mapping as particles.js). */
export function rainExtinction(rain) {
  const R = 1.2 * Math.pow(30, Math.max(0, Math.min(1, rain)));
  return rain > 0.001 ? 0.00025 * Math.pow(R, 0.63) : 0;
}

export function createFogUniforms() {
  return {
    hfogA: { value: v4() },
    hfogB: { value: v4() },
    hfogC: { value: v4() },
    hfogSun: { value: v4(0, 1, 0, 0) },
    hfogMist: { value: v4() },
    hfogMist2: { value: v4() },
    hfogWind: { value: v4() },
    hfogCloud: { value: v4() },
    hfogCloud2: { value: v4() },
    hfogRain: { value: v4() },
    hTun0: { value: v4() }, hTun1: { value: v4() }, hTun2: { value: v4() }, hTun3: { value: v4() },
    hTunP: { value: v4(0, 22, 0.05, 9) },   // x = 0: tunnel occlusion disabled until env sets it from the road
    hEnv: { value: v4(0, 0.75, 0.35, 0) },   // x time, y wetness, z rain, w lightning (not declared by fog chunks)
  };
}

/** Write a settings object (FOG_DEFAULTS shape) into the shared uniform objects. */
export function writeFogUniforms(u, s, time = 0) {
  const A = u.hfogA.value, B = u.hfogB.value, C = u.hfogC.value, M = u.hfogMist.value, M2 = u.hfogMist2.value, W = u.hfogWind.value;
  A.x = s.hazeDensity; A.y = s.hazeFalloff; A.z = s.valleyDensity; A.w = s.valleyFalloff;
  B.x = s.valleyColor[0]; B.y = s.valleyColor[1]; B.z = s.valleyColor[2]; B.w = s.enabled;
  C.x = s.hazeRef; C.y = s.valleyRef; C.z = s.sunGlow; C.w = s.sunGlowExp;
  M.x = s.mist1Y; M.y = s.mist1H; M.z = s.mist2Y; M.w = s.mist2H;
  M2.x = s.mistDensity; M2.y = s.mistFreq; M2.z = time; M2.w = s.mistCoverage;
  W.x = s.wind[0]; W.y = s.wind[1]; W.z = s.mistBright; W.w = s.mistCoverage2 ?? s.mistCoverage;
  if (u.hfogCloud) {
    const K = u.hfogCloud.value, K2 = u.hfogCloud2.value, cc = s.cloudColor || [0.62, 0.67, 0.74];
    K.x = s.cloudBase ?? 400; K.y = s.cloudRamp ?? 260; K.z = s.cloudDensity ?? 0; K.w = s.cloudPatch ?? 0.7;
    K2.x = cc[0]; K2.y = cc[1]; K2.z = cc[2]; K2.w = s.cloudFreq ?? 1 / 700;
  }
  if (u.hfogRain) u.hfogRain.value.x = s.rainExtinction ?? 0;
}

let _installed = null;

/**
 * PERF (QA): three evaluates the full BRDF (RE_Direct) for every point / spot light on every lit fragment, even when
 * the light contributes nothing: zero intensity (the two tunnel PointLights sit at 0 for the whole road), outside a
 * spot cone (headlights, roadworks lamp) or beyond the light's cutoff distance. getPointLightInfo/getSpotLightInfo
 * already compute `directLight.visible = color != 0`; this guards RE_Direct with it. The result is bit-identical
 * (a zero-colour light adds exactly zero) and the branch is coherent (uniform- or distance-driven). It saved ~2-3 ms
 * per frame at 1280x720 on the M1 (escape drive). Light counts are unchanged, so nothing recompiles when lights fade in/out.
 */
function installLightSkip() {
  let c = THREE.ShaderChunk.lights_fragment_begin;
  if (c.includes('/*hlskip*/')) return;
  const call = 'RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );';
  const guard = (sec) => sec.replace(call, `/*hlskip*/ if ( directLight.visible ) { ${call} }`);
  const iP = c.indexOf('#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )');
  const iS = c.indexOf('#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )');
  const iE = c.indexOf('#if', iS + 10 + c.slice(iS + 10).indexOf('#pragma unroll_loop_end'));
  if (iP < 0 || iS < iP || iE < iS || c.split(call).length < 3) { console.warn('[fog] light-skip patch: unexpected lights_fragment_begin, skipped'); return; }
  c = c.slice(0, iP) + guard(c.slice(iP, iS)) + guard(c.slice(iS, iE)) + c.slice(iE);
  THREE.ShaderChunk.lights_fragment_begin = c;
}

/**
 * Overcast "sun" = a broad bright patch of cloud (~20 deg across), not a disc. A delta directional light would put
 * pin-sharp glints on puddles, wet asphalt and car paint, which is the classic CG tell in rain footage. For the
 * directional light only, the GGX lobe is widened like an area light (alpha' = alpha + HSUN_ALPHA, the Karis
 * sphere-light approximation with the source's angular radius): mirror-smooth water gets a soft sheen instead of a
 * glint, rough surfaces are unchanged. Diffuse is untouched. Only STANDARD/PHYSICAL materials (they have roughness).
 */
function installSunSoftSpec() {
  let c = THREE.ShaderChunk.lights_fragment_begin;
  if (c.includes('/*hsun*/')) return;
  const call = 'RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );';
  const iD = c.indexOf('#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )');
  const iC = iD < 0 ? -1 : c.indexOf(call, iD);
  const iE = iD < 0 ? -1 : c.indexOf('#pragma unroll_loop_end', iD);
  if (iD < 0 || iC < 0 || iE < iC) { console.warn('[fog] sun soft-spec patch: unexpected lights_fragment_begin, skipped'); return; }
  const wrapped = `/*hsun*/
		#ifdef STANDARD
		{ float hSunR0 = material.roughness; material.roughness = min( sqrt( hSunR0 * hSunR0 + 0.12 ), 1.0 );
		  ${call}
		  material.roughness = hSunR0; }
		#else
		${call}
		#endif`;
  THREE.ShaderChunk.lights_fragment_begin = c.slice(0, iC) + wrapped + c.slice(iC + call.length);
}

/**
 * Install the height fog globally. Idempotent. Returns the shared uniform set.
 * Must run before materials compile (env does it in its constructor).
 */
export function installFog(uniforms = createFogUniforms()) {
  if (_installed) return _installed;
  THREE.ShaderChunk.fog_pars_vertex = CHUNK_PARS_VERTEX;
  THREE.ShaderChunk.fog_vertex = CHUNK_VERTEX;
  THREE.ShaderChunk.fog_pars_fragment = CHUNK_PARS_FRAGMENT;
  THREE.ShaderChunk.fog_fragment = CHUNK_FRAGMENT;
  if (!THREE.ShaderChunk.lights_fragment_maps.includes('htunnel_occ')) THREE.ShaderChunk.lights_fragment_maps += CHUNK_TUNNEL;
  installLightSkip();
  installSunSoftSpec();
  // Shared uniform objects: add them to UniformsLib.fog (custom ShaderMaterials merging it get them) and to every
  // ShaderLib entry that already carries fog uniforms (built-in materials clone ShaderLib uniforms at compile time).
  const add = (target) => { for (const k in uniforms) if (k !== 'hEnv') target[k] = uniforms[k]; };
  add(THREE.UniformsLib.fog);
  for (const key of Object.keys(THREE.ShaderLib)) {
    const u = THREE.ShaderLib[key]?.uniforms;
    if (u && u.fogColor) add(u);
  }
  _installed = uniforms;
  return uniforms;
}

/** CPU mirror of the shader's transmittance (luminance-ish, green channel). Useful for gameplay/audio queries. */
export function fogTransmittanceCPU(s, camPos, worldPos) {
  const rx = worldPos.x - camPos.x, ry = worldPos.y - camPos.y, rz = worldPos.z - camPos.z;
  const t = Math.hypot(rx, ry, rz);
  if (t < 1e-3 || !s.enabled) return 1;
  const rdy = ry / t;
  const layer = (dens, fall, href) => {
    const base = dens * Math.exp(Math.max(-80, Math.min(30, -fall * (camPos.y - href))));
    const x = fall * rdy * t;
    const g = Math.abs(x) < 1e-3 ? 1 - 0.5 * x : (1 - Math.exp(-Math.max(-40, Math.min(80, x)))) / x;
    return base * t * g;
  };
  const tR = rdy > 1e-4 && s.cloudBase > 0 ? Math.min(t, Math.max(s.cloudBase - camPos.y, 0) / rdy) : t;
  const od = (layer(s.hazeDensity, s.hazeFalloff, s.hazeRef) + layer(s.valleyDensity, s.valleyFalloff, s.valleyRef) + (s.rainExtinction || 0) * tR) * s.enabled;
  return Math.exp(-Math.min(40, od));
}

// ---------------------------------------------------------------------------------------------------------------
// Volumetric mist (rendered by render/post.js MistPass: half-res ray march through three slabs, composited over the
// lit scene). This complements the analytic fog above with real 3D structure:
//   ground   thin wisps drifting across the road: 0..groundH m above a plane fitted to the road at the camera,
//            faded in from nearStart and out by farEnd (none right at the lens, none far away)
//   valley   the cloud sea in the valley below the road, with billowing tops and darker bases
//   band     low cloud banks at mid-slope height that drift through the valley and snag on the forested slopes
// Settings live on ctx.env.mist (MIST_DEFAULTS shape) and may be edited at runtime.
// ---------------------------------------------------------------------------------------------------------------
export const MIST_DEFAULTS = {
  enabled: 1,
  noiseFreq: 1 / 180,         // 1/m of the base noise (one 64^3 tile = 180 m)
  wind: [1.6, 0.7],           // m/s drift (same direction as the analytic mist)
  ground: { density: 0.007, height: 8, nearStart: 14, farEnd: 260, coverage: 0.34, freqMul: 4.5 },
  // valley: a broken cloud sea well below the road (the road is at y 40..98): dark forest shows through the gaps
  valley: { base: -120, top: -15, density: 0.01, coverage: 0.32, freqMul: 0.55 },
  band: { base: 140, top: 430, density: 0.009, coverage: 0.4, hug: 260, freqMul: 0.38 },
  // mist radiance relative to fog colour: tops lit by the overcast sky, bases self-shadowed (a cloud of optical
  // depth ~3 transmits ~40 % of the skylight to its base)
  light: { top: 1.08, base: 0.42, sunGlow: 0.35 },
};

/** Tileable 3D value-noise fBm (R8, size^3), for the volumetric mist. Deterministic. */
export function createMistNoise3D(size = 64) {
  const N = size, data = new Uint8Array(N * N * N);
  const octaves = [[4, 0.5], [8, 0.27], [16, 0.15], [32, 0.08]];
  const hash = (x, y, z, p) => {
    x = ((x % p) + p) % p; y = ((y % p) + p) % p; z = ((z % p) + p) % p;
    let h = (x * 374761393 + y * 668265263 + z * 2147483647 + p * 1013904223) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
  };
  const tables = octaves.map(([p]) => { const t = new Float32Array(p * p * p); for (let z = 0; z < p; z++) for (let y = 0; y < p; y++) for (let x = 0; x < p; x++) t[(z * p + y) * p + x] = hash(x, y, z, p); return t; });
  const sm = (t) => t * t * (3 - 2 * t);
  let min = Infinity, max = -Infinity;
  const f = new Float32Array(N * N * N);
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = 0;
    for (let o = 0; o < octaves.length; o++) {
      const [p, a] = octaves[o], t = tables[o], s = p / N;
      const fx = x * s, fy = y * s, fz = z * s;
      const x0 = Math.floor(fx), y0 = Math.floor(fy), z0 = Math.floor(fz);
      const ux = sm(fx - x0), uy = sm(fy - y0), uz = sm(fz - z0);
      const x1 = (x0 + 1) % p, y1 = (y0 + 1) % p, z1 = (z0 + 1) % p;
      const g = (xx, yy, zz) => t[(zz * p + yy) * p + xx];
      const a00 = g(x0, y0, z0) + (g(x1, y0, z0) - g(x0, y0, z0)) * ux, a10 = g(x0, y1, z0) + (g(x1, y1, z0) - g(x0, y1, z0)) * ux;
      const a01 = g(x0, y0, z1) + (g(x1, y0, z1) - g(x0, y0, z1)) * ux, a11 = g(x0, y1, z1) + (g(x1, y1, z1) - g(x0, y1, z1)) * ux;
      const b0 = a00 + (a10 - a00) * uy, b1 = a01 + (a11 - a01) * uy;
      v += a * (b0 + (b1 - b0) * uz);
    }
    const i = (z * N + y) * N + x; f[i] = v; if (v < min) min = v; if (v > max) max = v;
  }
  for (let i = 0; i < f.length; i++) data[i] = Math.round(((f[i] - min) / (max - min)) * 255);
  const tex = new THREE.Data3DTexture(data, N, N, N);
  tex.format = THREE.RedFormat; tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.RepeatWrapping;
  tex.unpackAlignment = 1; tex.needsUpdate = true;
  tex.name = 'mistNoise3D';
  return tex;
}
