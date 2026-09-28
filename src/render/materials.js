// LANDSLIDE material helpers (RENDER workstream).
//
//   await pbrMaterial(ctx, 'asphalt_02', {repeat: 4 | [x, y], triplanar?: true | {scale, sharpness, macro}, wet?: porosity,
//                    color?, roughness?, metalness?, normalScale?, aoIntensity?, envMapIntensity?, side?, tileSize?})
//       -> MeshStandardMaterial with map (sRGB) + normalMap + ARM (R = AO, G = roughness, B = metal) on aoMap/roughnessMap/
//          metalnessMap. Textures are cloned per material when repeat != 1 (the GPU image is shared).
//          `tileSize` (m) is the real-world size of one texture tile. It is used by `triplanar` (scale = 1/tileSize) and
//          defaults to TILE_SIZES[texName] (Poly Haven physical sizes).
//   triplanarPatch(material, {scale = 1/tileSize, sharpness = 4, macro = 0.25})
//       World-space triplanar mapping of map/normalMap/ARM. It uses a whiteout normal blend with the side flips corrected,
//       plus low-frequency macro variation of albedo/roughness to break up tiling. It works on instanced meshes.
//   applyWetness(material, {porosity = 0.5, strength = 1})
//       Rain wetness driven by ctx.env.wetness (the shared hEnv uniform). It darkens the albedo (porous surfaces more) and
//       lowers the roughness (non-porous surfaces toward a water film). Upward faces get wetter than overhangs. No puddles.
//   applyRipples(material, {strength = 1, scale = 1, minRoughness})
//       Animated raindrop ripple rings on upward-facing surfaces, scaled by ctx.env.rain. Meant for puddles / flat wet
//       surfaces. RIPPLE_GLSL exports the raw function `vec2 rain_ripples(vec2 worldXZ, float time, float rain)`, which
//       returns a height gradient for custom shaders.
//
// Every patch chains onto any existing onBeforeCompile and extends customProgramCacheKey, so the patches compose
// (for example triplanarPatch then applyWetness). Uniform values are shared objects, so runtime changes cost nothing.
import * as THREE from 'three';

/** Physical tile sizes (m) of the Poly Haven sets in public/assets/tex (for world-space mapping). */
export const TILE_SIZES = {
  asphalt_02: 2.0, aerial_rocks_02: 3.0, lichen_rock: 2.0, brown_mud_rocks_01: 2.5, aerial_grass_rock: 3.0,
  rocky_trail: 2.0, forest_ground_04: 2.0, precast_concrete_wall: 2.0, rough_wood: 1.0,
};

// Shared env uniform (time, wetness, rain, lightning). Resolved lazily from ctx.env when available.
const _fallbackEnv = { value: { x: 0, y: 0.75, z: 0.35, w: 0 } };
let _envUniform = null;
function envUniform(ctx) {
  if (ctx?.env?.fogUniforms?.hEnv) _envUniform = ctx.env.fogUniforms.hEnv;
  return _envUniform || _fallbackEnv;
}
/** Let the env register the shared uniform so patches made without ctx still animate. */
export function setEnvUniform(u) { _envUniform = u; }

/**
 * (LOWEND) Which floating-point colour buffers this GPU can render to: {half, full}. WebGL2 only renders to
 * RGBA16F / RGBA32F with EXT_color_buffer_float (or EXT_color_buffer_half_float); some old / mobile / software
 * drivers lack them. Probed once per renderer with a 4x4 target and checkFramebufferStatus (an extension can be
 * advertised and still give an incomplete framebuffer). ?nofloat=half|full simulates a missing extension (testing).
 */
export function floatRTSupport(renderer) {
  const ud = renderer.userData || (renderer.userData = {});
  if (ud.floatRT) return ud.floatRT;
  const sim = new URLSearchParams(location.search).get('nofloat');
  const probe = (type) => {
    const rt = new THREE.WebGLRenderTarget(4, 4, { type, depthBuffer: false, stencilBuffer: false });
    rt.texture.generateMipmaps = false;
    const prev = renderer.getRenderTarget();
    let ok = false;
    try {
      renderer.setRenderTarget(rt);
      const gl = renderer.getContext();
      ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    } catch { ok = false; }
    renderer.setRenderTarget(prev);
    rt.dispose();
    return ok;
  };
  const full = sim ? false : probe(THREE.FloatType);
  const half = sim === 'half' ? false : (full || probe(THREE.HalfFloatType));
  ud.floatRT = { half, full };
  if (!half || !full) console.warn('[render] float render targets:', JSON.stringify(ud.floatRT), '- using fallbacks');
  return ud.floatRT;
}

function chain(material, key, fn) {
  const prev = material.onBeforeCompile;
  const prevKey = material.customProgramCacheKey;
  material.onBeforeCompile = function (shader, renderer) {
    if (prev && prev !== THREE.Material.prototype.onBeforeCompile) prev.call(this, shader, renderer);
    fn(shader, renderer);
  };
  material.customProgramCacheKey = function () {
    const base = prevKey && prevKey !== THREE.Material.prototype.customProgramCacheKey ? prevKey.call(this) : '';
    return base + '|' + key;
  };
  material.needsUpdate = true;
  return material;
}

// ---------------------------------------------------------------------------------------------------------------
// pbrMaterial
// ---------------------------------------------------------------------------------------------------------------
export async function pbrMaterial(ctx, texName, opts = {}) {
  // default opts -> same cache key as everyone else calling ctx.assets.pbr(name): the GPU images are shared
  const set = await ctx.assets.pbr(texName);
  let rx = 1, ry = 1;
  if (Array.isArray(opts.repeat)) [rx, ry] = opts.repeat;
  else if (opts.repeat?.isVector2) { rx = opts.repeat.x; ry = opts.repeat.y; }
  else if (typeof opts.repeat === 'number') rx = ry = opts.repeat;
  const aniso = opts.anisotropy ?? ctx.config?.quality?.anisotropy;
  const prep = (t) => {
    if (!t) return null;
    if (rx === 1 && ry === 1 && (aniso == null || aniso === t.anisotropy)) return t;
    const c = t.clone(); c.repeat.set(rx, ry); c.wrapS = c.wrapT = THREE.RepeatWrapping;
    if (aniso != null) c.anisotropy = Math.min(aniso, ctx.assets.maxAnisotropy ?? aniso);
    return c;
  };
  const arm = prep(set.armMap);
  const mat = new THREE.MeshStandardMaterial({
    name: 'pbr_' + texName,
    map: prep(set.map),
    normalMap: prep(set.normalMap),
    aoMap: arm, roughnessMap: arm, metalnessMap: arm,
    roughness: opts.roughness ?? 1.0,
    metalness: opts.metalness ?? 1.0,       // multiplied by ARM.b (≈0 for these dielectrics)
    aoMapIntensity: opts.aoIntensity ?? 1.0,
    envMapIntensity: opts.envMapIntensity ?? 1.0,
    side: opts.side ?? THREE.FrontSide,
  });
  if (opts.color !== undefined) mat.color.set(opts.color);
  const ns = opts.normalScale ?? 1;
  if (Array.isArray(ns)) mat.normalScale.set(ns[0], ns[1]); else mat.normalScale.set(ns, ns);
  mat.userData.pbr = { texName };
  if (opts.triplanar) {
    const t = typeof opts.triplanar === 'object' ? opts.triplanar : {};
    const tile = opts.tileSize ?? TILE_SIZES[texName] ?? 2;
    triplanarPatch(mat, { scale: t.scale ?? 1 / tile, sharpness: t.sharpness ?? 4, macro: t.macro ?? 0.25 });
  }
  if (opts.wet !== undefined && opts.wet !== false) applyWetness(mat, { porosity: typeof opts.wet === 'number' ? opts.wet : 0.5 }, ctx);
  return mat;
}

// ---------------------------------------------------------------------------------------------------------------
// triplanar
// ---------------------------------------------------------------------------------------------------------------
const TP_COMMON = /* glsl */`
varying vec3 vTpPos;
varying vec3 vTpNormal;
`;
const TP_FRAG_HEAD = /* glsl */`
uniform vec4 tpParams; // x scale (1/m), y blend sharpness, z macro variation, w unused
float tp_hash(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float tp_noise(vec2 p) {
  vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(tp_hash(i), tp_hash(i + vec2(1.0, 0.0)), u.x), mix(tp_hash(i + vec2(0.0, 1.0)), tp_hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
`;
const TP_VERT = /* glsl */`
{
  vec4 tpP = vec4( transformed, 1.0 );
  vec3 tpNo = objectNormal;
  #ifdef USE_BATCHING
    tpP = batchingMatrix * tpP; tpNo = mat3( batchingMatrix ) * tpNo;
  #endif
  #ifdef USE_INSTANCING
    tpP = instanceMatrix * tpP; tpNo = mat3( instanceMatrix ) * tpNo;
  #endif
  tpP = modelMatrix * tpP;
  vTpPos = tpP.xyz;
  vTpNormal = normalize( mat3( modelMatrix ) * tpNo );
}
`;
// replaces <map_fragment>: sets up shared triplanar coordinates + samples albedo and ARM once
const TP_MAP = /* glsl */`
vec3 tpN = normalize( vTpNormal );
#ifdef DOUBLE_SIDED
  tpN *= gl_FrontFacing ? 1.0 : -1.0;
#endif
vec3 tpW = pow( abs( tpN ), vec3( tpParams.y ) );
tpW /= ( tpW.x + tpW.y + tpW.z + 1e-5 );
vec3 tpS = vec3( tpN.x < 0.0 ? -1.0 : 1.0, tpN.y < 0.0 ? -1.0 : 1.0, tpN.z < 0.0 ? -1.0 : 1.0 );
vec2 tpUvX = vec2( vTpPos.z * tpS.x, vTpPos.y ) * tpParams.x;
vec2 tpUvY = vec2( vTpPos.x * tpS.y, vTpPos.z ) * tpParams.x;
vec2 tpUvZ = vec2( -vTpPos.x * tpS.z, vTpPos.y ) * tpParams.x;
// macro variation (breaks tiling): two octaves of value noise in world space
float tpMacroN = tp_noise( vTpPos.xz * 0.045 + vTpPos.y * 0.02 ) * 0.65 + tp_noise( vTpPos.zx * 0.17 + 7.3 ) * 0.35;
float tpMacro = mix( 1.0, 0.72 + 0.56 * tpMacroN, tpParams.z );
#ifdef USE_MAP
  vec4 tpAlb = texture2D( map, tpUvX ) * tpW.x + texture2D( map, tpUvY ) * tpW.y + texture2D( map, tpUvZ ) * tpW.z;
  tpAlb.rgb *= tpMacro;
  diffuseColor *= tpAlb;
#endif
#if defined( USE_ROUGHNESSMAP )
  vec4 tpArm = texture2D( roughnessMap, tpUvX ) * tpW.x + texture2D( roughnessMap, tpUvY ) * tpW.y + texture2D( roughnessMap, tpUvZ ) * tpW.z;
#elif defined( USE_AOMAP )
  vec4 tpArm = texture2D( aoMap, tpUvX ) * tpW.x + texture2D( aoMap, tpUvY ) * tpW.y + texture2D( aoMap, tpUvZ ) * tpW.z;
#else
  vec4 tpArm = vec4( 1.0 );
#endif
tpArm.g = clamp( tpArm.g * mix( 1.0, 1.25 - 0.5 * tpMacroN, tpParams.z ), 0.0, 1.0 );
`;
const TP_NORMAL = /* glsl */`
#ifdef USE_NORMALMAP
{
  vec3 tnX = texture2D( normalMap, tpUvX ).xyz * 2.0 - 1.0;
  vec3 tnY = texture2D( normalMap, tpUvY ).xyz * 2.0 - 1.0;
  vec3 tnZ = texture2D( normalMap, tpUvZ ).xyz * 2.0 - 1.0;
  tnX.xy *= normalScale; tnY.xy *= normalScale; tnZ.xy *= normalScale;
  tnX.x *= tpS.x; tnY.x *= tpS.y; tnZ.x *= -tpS.z;
  // whiteout blend
  tnX = vec3( tnX.xy + tpN.zy, abs( tnX.z ) * tpN.x );
  tnY = vec3( tnY.xy + tpN.xz, abs( tnY.z ) * tpN.y );
  tnZ = vec3( tnZ.xy + tpN.xy, abs( tnZ.z ) * tpN.z );
  vec3 tpWorldN = normalize( tnX.zyx * tpW.x + tnY.xzy * tpW.y + tnZ.xyz * tpW.z );
  normal = normalize( ( viewMatrix * vec4( tpWorldN, 0.0 ) ).xyz );
}
#endif
`;

export function triplanarPatch(material, opts = {}) {
  const params = { value: new THREE.Vector4(opts.scale ?? 0.5, opts.sharpness ?? 4, opts.macro ?? 0.25, 0) };
  material.userData.triplanar = params;
  const rough = THREE.ShaderChunk.roughnessmap_fragment.replace('texture2D( roughnessMap, vRoughnessMapUv )', 'tpArm');
  const metal = THREE.ShaderChunk.metalnessmap_fragment.replace('texture2D( metalnessMap, vMetalnessMapUv )', 'tpArm');
  const ao = THREE.ShaderChunk.aomap_fragment.replace('texture2D( aoMap, vAoMapUv ).r', 'tpArm.r');
  return chain(material, 'tp', (shader) => {
    shader.uniforms.tpParams = params;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\n' + TP_COMMON)
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\n' + TP_VERT);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\n' + TP_COMMON + TP_FRAG_HEAD)
      .replace('#include <map_fragment>', TP_MAP)
      .replace('#include <roughnessmap_fragment>', rough)
      .replace('#include <metalnessmap_fragment>', metal)
      .replace('#include <normal_fragment_maps>', TP_NORMAL)
      .replace('#include <aomap_fragment>', ao);
  });
}

// ---------------------------------------------------------------------------------------------------------------
// wetness
// ---------------------------------------------------------------------------------------------------------------
const HENV_DECL = /* glsl */`
#ifndef HENV_DECLARED
#define HENV_DECLARED
uniform vec4 hEnv;      // x time, y wetness, z rain, w lightning
#endif
`;
const WET_HEAD = HENV_DECL + /* glsl */`
uniform vec4 wetParams; // x strength, y porosity, z min roughness when soaked, w unused
`;
const WET_CODE = /* glsl */`
{
  vec3 wetWN = ( vec4( normal, 0.0 ) * viewMatrix ).xyz;   // world-space shading normal
  float wetFacing = mix( 0.45, 1.0, smoothstep( -0.2, 0.75, wetWN.y ) );
  float wet = clamp( hEnv.y * wetParams.x, 0.0, 1.0 ) * wetFacing;
  float por = clamp( wetParams.y, 0.0, 1.0 );
  // porous materials soak up water and darken strongly; sealed ones barely darken but get a water film
  diffuseColor.rgb *= mix( 1.0, 1.0 - 0.62 * por, wet );
  #ifdef STANDARD
    float wetTarget = mix( wetParams.z, roughnessFactor * 0.7, por );
    roughnessFactor = mix( roughnessFactor, min( roughnessFactor, wetTarget ), wet );
  #endif
}
`;

export function applyWetness(material, opts = {}, ctx = null) {
  const params = { value: new THREE.Vector4(opts.strength ?? 1, opts.porosity ?? 0.5, opts.minRoughness ?? 0.08, 0) };
  material.userData.wetness = params;
  const env = envUniform(ctx);
  return chain(material, 'wet', (shader) => {
    shader.uniforms.hEnv = _envUniform || env;
    shader.uniforms.wetParams = params;
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\n' + WET_HEAD)
      .replace('#include <emissivemap_fragment>', WET_CODE + '\n#include <emissivemap_fragment>');
  });
}

// ---------------------------------------------------------------------------------------------------------------
// rain ripples
// ---------------------------------------------------------------------------------------------------------------
export const RIPPLE_GLSL = /* glsl */`
vec2 rp_hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}
// Height gradient (d/dx, d/dz) of expanding raindrop rings. p in metres (world xz), t in seconds, rain 0..1.
vec2 rain_ripples_layer(vec2 p, float t, float rain, float cellSize, float seed) {
  vec2 q = p / cellSize;
  vec2 cell = floor(q);
  vec2 g = vec2(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 c = cell + vec2(float(i), float(j));
      vec2 h = rp_hash22(c + seed);
      if (h.y > rain * 0.85 + 0.1) continue;          // fewer drops in light rain
      float ph = fract(t * (0.9 + 0.5 * h.x) + h.x * 7.0);
      vec2 ctr = c + 0.5 + (rp_hash22(c * 1.7 + seed + 11.0) - 0.5) * 0.8;
      vec2 d = q - ctr;
      float r = length(d);
      float x = r - ph * 0.85;                          // distance to the expanding ring
      float env = exp(-x * x * 90.0) * (1.0 - ph) * (1.0 - ph);
      float dh = cos(x * 38.0) * env;                   // d(height)/dr (up to a constant)
      g += dh * d / max(r, 1e-3);
    }
  }
  return g / cellSize;
}
vec2 rain_ripples(vec2 p, float t, float rain) {
  if (rain <= 0.001) return vec2(0.0);
  return rain_ripples_layer(p, t, rain, 0.32, 0.0) + 0.6 * rain_ripples_layer(p + 13.1, t * 1.13, rain, 0.21, 41.0);
}
`;
const RIPPLE_HEAD = HENV_DECL + /* glsl */`
uniform vec4 rippleParams; // x strength, y scale, z wetness gate, w unused
varying vec3 vRpPos;
` + RIPPLE_GLSL;
const RIPPLE_CODE = /* glsl */`
{
  vec3 rpWN = ( vec4( normal, 0.0 ) * viewMatrix ).xyz;
  float rpUp = smoothstep( 0.75, 0.95, rpWN.y );
  float rpAmt = rpUp * hEnv.z * mix( 1.0, hEnv.y, rippleParams.z ) * rippleParams.x;
  if ( rpAmt > 0.001 ) {
    vec2 rg = rain_ripples( vRpPos.xz / rippleParams.y, hEnv.x, hEnv.z ) * 0.03 * rpAmt;
    rpWN = normalize( rpWN + vec3( -rg.x, 0.0, -rg.y ) );
    normal = normalize( ( viewMatrix * vec4( rpWN, 0.0 ) ).xyz );
  }
}
`;
const RIPPLE_VERT = /* glsl */`
{
  vec4 rpP = vec4( transformed, 1.0 );
  #ifdef USE_INSTANCING
    rpP = instanceMatrix * rpP;
  #endif
  vRpPos = ( modelMatrix * rpP ).xyz;
}
`;

export function applyRipples(material, opts = {}, ctx = null) {
  const params = { value: new THREE.Vector4(opts.strength ?? 1, opts.scale ?? 1, opts.wetGate ?? 1, 0) };
  material.userData.ripples = params;
  const env = envUniform(ctx);
  return chain(material, 'rip', (shader) => {
    shader.uniforms.hEnv = _envUniform || env;
    shader.uniforms.rippleParams = params;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vRpPos;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\n' + RIPPLE_VERT);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\n' + RIPPLE_HEAD)
      .replace('#include <emissivemap_fragment>', RIPPLE_CODE + '\n#include <emissivemap_fragment>');
  });
}

// ---------------------------------------------------------------------------------------------------------------
// screen-space reflections for wet flat surfaces (road, puddles)
// ---------------------------------------------------------------------------------------------------------------
// applyScreenReflections(material, {steps = 20, maxRoughness = 0.5, strength = 1, film = 0, filmRoughness = 0.09})
//   Adds screen-space reflections to a MeshStandardMaterial's image-based specular ("radiance" after
//   lights_fragment_maps), so the material's own Fresnel/DFG, specular occlusion and fog still apply. The reflected
//   ray is marched in world space and projected with the PREVIOUS frame's view-projection into post's history buffer
//   (half-res linear HDR colour + distance to the camera in alpha), so it reflects the real scene: trees, the car,
//   lamps, mist, the actual rendered sky. Rays that leave the march without a hit take the far scene / sky straight
//   from the screen in the reflected direction ("infinity" sample, the tree line and sky that sell a wet road); rays
//   that leave the screen fall back to the env map. Roughness widens the fetch footprint (GGX lobe) and it is
//   stretched vertically at grazing angles like real wet asphalt. Gated to low roughness and upward-facing normals,
//   so it costs nothing elsewhere. steps = 0: infinity sample only (medium quality).
//   film > 0 adds the water-film lobe of a wet road (see SSR_CODE): patchy near-mirror reflections on top of the
//   rough lobe, strength film x rain wetness.
//   The shared uniforms (SSR_UNIFORMS) are driven by render/post.js; until the history exists ssrParams.x = 0 (off).
export const SSR_UNIFORMS = {
  tSSR: { value: null },
  ssrPrevVP: { value: new THREE.Matrix4() },
  ssrPrevCam: { value: new THREE.Vector3() },
  // x global strength (0 = off), y rain wetness 0..1 (water film; post refreshes it every frame while the SSR history
  // pass exists, i.e. medium/high/ultra), z history px per unit tan, w unused
  ssrParams: { value: new THREE.Vector4(0, 0.75, 256, 0) },
  ssrRes: { value: new THREE.Vector4(1, 1, 1, 1) },        // xy history size, zw 1/size
};

const SSR_PARS = /* glsl */`
uniform sampler2D tSSR;
uniform mat4 ssrPrevVP;
uniform vec3 ssrPrevCam;
uniform vec4 ssrParams;
uniform vec4 ssrRes;
uniform vec4 ssrMat;     // x strength, y max roughness
// history alpha = 1 / distance, box-filtered: foliage with sky holes reads as a semi-solid volume, not speckle
float ssr_dist(vec2 uv) { return 1.0 / max(textureLod(tSSR, uv, 0.0).a, 1e-4); }
float ssr_edge(vec2 uv) {
  vec2 a = smoothstep(vec2(0.0), vec2(0.05, 0.02), uv), b = smoothstep(vec2(0.0), vec2(0.05, 0.12), 1.0 - uv);
  return a.x * a.y * b.x * b.y;
}
// Traces the mirror direction once. Returns (hit uv, confidence, footprint scale): the history-pixel footprint of a
// lobe is cone(roughness) * footprint scale, so several lobes can share one trace (see ssr_fetch).
vec4 ssr_trace(vec3 P, vec3 R, bool nearOk) {
  vec2 hitUV = vec2(-1.0);
  float hitT = -1.0, conf = 0.0;
  vec3 pc = P - ssrPrevCam;
  float pc2 = dot(pc, pc), pcR = dot(pc, R);
  #if SSR_STEPS > 0
  // near reflections (car, trees, barriers, lamps) only where they can be resolved: within ~90 m.
  // Clip coordinates are linear along the ray (c(t) = cP + t cR) and the distance to the camera is a quadratic.
  if (pc2 < 8100.0 && nearOk) {
    vec4 cP = ssrPrevVP * vec4(P, 1.0), cR = ssrPrevVP * vec4(R, 0.0);
    // no per-pixel jitter: without temporal accumulation it turns thin occluders into speckle
    const float jit = 0.5;
    float tPrev = 0.0;
    for (int i = 0; i < SSR_STEPS; i++) {
      float t = 0.2 * pow(SSR_GROWTH, float(i) + jit);
      vec4 c = cP + t * cR;
      if (c.w <= 0.0) break;
      vec2 u = c.xy / c.w * 0.5 + 0.5;
      if (u.x < 0.0 || u.x > 1.0 || u.y < 0.0 || u.y > 1.0) break;
      float dd = sqrt(pc2 + t * (2.0 * pcR + t)) - ssr_dist(u);
      if (dd > 0.0 && dd < max(0.35, (t - tPrev) * 1.2)) {
        float a = tPrev, b = t;
        for (int k = 0; k < 4; k++) {
          float m = 0.5 * (a + b);
          vec4 cm = cP + m * cR;
          if (sqrt(pc2 + m * (2.0 * pcR + m)) - ssr_dist(cm.xy / cm.w * 0.5 + 0.5) > 0.0) b = m; else a = m;
        }
        vec4 cb = cP + b * cR;
        hitT = b; hitUV = cb.xy / cb.w * 0.5 + 0.5; conf = 1.0;
        break;
      }
      tPrev = t;
    }
  }
  #endif
  float foot;
  if (conf > 0.0) {
    foot = hitT / max(length(P + R * hitT - ssrPrevCam), 0.1) * ssrParams.z;
  } else {
    // no near hit: the far scene / sky in the reflected direction ("infinity" sample). Valid when what stands in that
    // pixel is far compared with the reflecting point (parallax is then small): the tree line and ridges behind a
    // wet road, the sky. Near things there (the car, a barrier) would reflect in the wrong place: env map instead.
    vec4 c = ssrPrevVP * vec4(R, 0.0);
    if (c.w <= 0.0) return vec4(0.0);
    hitUV = c.xy / c.w * 0.5 + 0.5;
    if (hitUV.x <= 0.0 || hitUV.x >= 1.0 || hitUV.y <= 0.0 || hitUV.y >= 1.0) return vec4(0.0);
    float od = ssr_dist(hitUV);
    conf = max(smoothstep(50.0, 140.0, od), smoothstep(2.0, 3.5, od / max(sqrt(pc2), 1.0)));
    foot = ssrParams.z;
  }
  conf *= ssr_edge(hitUV);
  return vec4(hitUV, conf, foot);
}
// history fetch for one GGX lobe: footprint = lobe cone x trace footprint, stretched vertically at grazing angles
vec3 ssr_fetch(vec4 tr, float rough, float stretch) {
  float f = (rough * rough * 1.6 + 0.0015) * tr.w;           // tan of the lobe half angle -> history pixels
  vec2 gx = vec2(max(f, 1.0) * ssrRes.z, 0.0), gy = vec2(0.0, max(f * stretch, 1.5) * ssrRes.w);
  return textureGrad(tSSR, tr.xy, gx, gy).rgb;
}
`;
// Two lobes. (1) The material's own GGX lobe (gated to smooth-ish surfaces). (2) The WATER FILM of a wet road: in rain
// the macrotexture of asphalt (0.5-1.5 mm deep) is partly flooded; where the film covers the aggregate the surface
// is a near-mirror (water, roughness ~0.08 once raindrop pits are included), elsewhere it keeps the rough wet-asphalt
// lobe. So a wet road shows recognisable, vertically smeared images of the tree line, the car, barriers and lamps
// on top of a broad sheen, instead of one uniform blur. Film coverage is patchy (noise in world xz) and scales with
// the rain wetness (ssrParams.y). The film sample is scaled by the ratio of the pre-integrated specular energy of
// the smooth and the rough lobe (dfgLUT), because three applies the rough lobe's DFG to 'radiance' afterwards.
const SSR_CODE = /* glsl */`
#if defined( RE_IndirectSpecular ) && defined( USE_ENVMAP ) && defined( USE_FOG )
if ( ssrParams.x > 0.0 && ssrMat.x > 0.0 ) {
  // mostly the geometric normal: pixel-scale normal-map detail would scatter the rays into speckle (it is roughness,
  // already in the blur); a quarter of the shading normal keeps the ripple / puddle-edge distortion
  vec3 ssrNv = normalize( mix( nonPerturbedNormal, geometryNormal, 0.25 ) );
  vec3 ssrN = normalize( ( vec4( ssrNv, 0.0 ) * viewMatrix ).xyz );
  float ssrUp = smoothstep( 0.6, 0.85, ssrN.y );
  float ssrGate = ( 1.0 - smoothstep( ssrMat.y * 0.6, ssrMat.y, material.roughness ) ) * ssrUp;
  vec3 ssrP = cameraPosition + vHFogRel;
  float ssrFilm = 0.0;
  if ( ssrMat.z > 0.0 && material.roughness > SSR_FILM_ROUGH ) {
    // flooded patches: ~0.5 m blotches inside ~3 m wetter / drier zones
    float fn = hfog_vnoise( ssrP.xz * 2.1 ) * 0.5 + hfog_vnoise( ssrP.xz * 0.33 + 3.7 ) * 0.5;
    ssrFilm = ssrMat.z * clamp( ssrParams.y * 1.3, 0.0, 1.0 ) * smoothstep( 0.3, 0.62, fn ) * ssrUp
            * ( 1.0 - smoothstep( 0.55, 0.8, material.roughness ) );
  }
  if ( ssrGate + ssrFilm > 0.001 ) {
    vec3 ssrV = normalize( -vHFogRel );
    float ssrNdV = saturate( dot( ssrN, ssrV ) );
    float ssrStretch = clamp( 0.3 / max( ssrNdV, 0.015 ), 1.0, 12.0 );   // grazing-angle elongation (vertical)
    vec4 ssrT = ssr_trace( ssrP, reflect( -ssrV, ssrN ), ssrFilm > 0.001 || material.roughness < 0.5 );
    float ssrC = ssrT.z * ssrParams.x;
    if ( ssrGate > 0.001 && ssrC > 0.0 ) radiance = mix( radiance, ssr_fetch( ssrT, material.roughness, ssrStretch ), ssrC * ssrGate * ssrMat.x );
    if ( ssrFilm > 0.001 ) {
      vec3 ssrFR = ssrC > 0.0 ? ssr_fetch( ssrT, SSR_FILM_ROUGH, ssrStretch ) : vec3( 0.0 );
      if ( ssrC < 0.999 ) {   // off-screen / unresolved: the sharp env-map lobe
        vec3 ssrFE = getIBLRadiance( geometryViewDir, ssrNv, SSR_FILM_ROUGH );
        #ifndef HTUNNEL_OFF
          ssrFE *= htunnel_occ( ssrP );
        #endif
        ssrFR = mix( ssrFE, ssrFR, ssrC );
      }
      float ssrDV = saturate( dot( geometryNormal, geometryViewDir ) );
      vec2 ssrDF = texture2D( dfgLUT, vec2( SSR_FILM_ROUGH, ssrDV ) ).rg;
      float ssrEf = material.specularColor.g * ssrDF.x + material.specularF90 * ssrDF.y;
      float ssrEr = material.specularColor.g * material.dfg.x + material.specularF90 * material.dfg.y;
      radiance = mix( radiance, ssrFR * clamp( ssrEf / max( ssrEr, 1e-3 ), 1.0, 3.0 ), ssrFilm * ssrMat.x );
    }
  }
}
#endif
`;

export function applyScreenReflections(material, opts = {}) {
  const steps = Math.max(0, Math.round(opts.steps ?? 20));
  const growth = steps > 0 ? Math.pow(150 / 0.2, 1 / steps) : 1.5;
  const matU = { value: new THREE.Vector4(opts.strength ?? 1, opts.maxRoughness ?? 0.5, opts.film ?? 0, 0) };
  const filmRough = opts.filmRoughness ?? 0.09;
  material.userData.ssr = matU;
  return chain(material, 'ssr' + steps + '_' + filmRough, (shader) => {
    const fs = shader.fragmentShader;
    if (!fs.includes('#include <lights_fragment_maps>') || !fs.includes('void main()')) {
      if (!material.userData.ssrWarned) { material.userData.ssrWarned = true; console.warn('[materials] SSR: shader has no lights_fragment_maps, skipped', material.name); }
      return;
    }
    Object.assign(shader.uniforms, SSR_UNIFORMS, { ssrMat: matU });
    shader.fragmentShader = fs
      .replace('void main()', `#define SSR_STEPS ${steps}\n#define SSR_GROWTH ${growth.toFixed(5)}\n#define SSR_FILM_ROUGH ${filmRough.toFixed(4)}\n${SSR_PARS}\nvoid main()`)
      .replace('#include <lights_fragment_maps>', '#include <lights_fragment_maps>\n' + SSR_CODE);
  });
}
