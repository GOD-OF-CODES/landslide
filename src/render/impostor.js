// Tree rendering (TREES workstream): instanced LOD0/LOD1 conifer meshes + a single-draw-call billboard impostor field.
//
//   const field = new TreeField(ctx, { gltf, meta, albedo, normal });   // gltf = trees.glb, meta = impostors.json
//   field.setTrees(Float32Array [x,y,z,scale,rotY,variant]*n)
//   field.update(dt)                                                      // every frame
//
// LOD scheme (distances from config.quality): LOD0 mesh (<= ~0.42*treeMeshDistance) -> LOD1 mesh (<= treeMeshDistance)
// -> impostor (<= impostorDistance). Every transition is a screen-space dithered crossfade: at a given distance each
// pixel belongs to exactly one LOD (complementary thresholds on the same interleaved-gradient noise), so there is no
// popping and no double-darkening. Meshes cast shadows; impostors do not (they start beyond the shadow range).
//
// Materials are MeshStandardMaterial + onBeforeCompile, so the scene lighting (IBL + sun + shadows) and the global
// height fog from render/fog.js apply automatically. Wind (trunk sway + branch flutter) is shared through WIND.
import * as THREE from 'three';

/** Shared wind + LOD uniforms (same objects in every tree/grass material). */
export const WIND = {
  uTime: { value: 0 },
  uWind: { value: new THREE.Vector4(0.8, 0.6, 0.35, 0) },   // xy = world xz direction, z = strength 0..1, w = gust
  uCamPos: { value: new THREE.Vector3() },                  // MAIN camera position (cameraPosition is the light in shadow passes)
  uWet: { value: 0.75 },
  // Crown self-shadowing of the sun for foliage. Foliage of every LOD ignores the shadow map (impostors and far LODs
  // cannot self-shadow) and scales direct light by this instead, so LOD transitions keep the same brightness.
  uSunVis: { value: 0.42 },
};
/** Foliage texture tuning (shared). */
// spec: sky-reflection strength. At 1.0 the outer (AO~1) cards each got a uniform grey sheen that outlined every card
// as a pale plate; the reflection is now also broken up per needle (see GLSL_MAP_BIAS) and kept low.
// The needle-card atlas itself carries the calibrated albedo (opaque-texel mean ~(0.04, 0.06, 0.024) linear), so the
// tint is neutral. trans: sun translucency / back-scatter strength through the needle mats.
export const FOLIAGE = {
  mapBias: { value: -0.75 }, spec: { value: 0.55 }, tint: { value: new THREE.Vector3(1, 1, 1) },
  trans: { value: 0.75 },
  // Impostor albedo scale that matches the LOD1 meshes as seen on screen. The deep, alpha-tested mesh crowns receive a
  // lot of N8AO while the flat billboards get almost none, so the match depends on the AO setting (measured with
  // scratch/trees/lodmatch.mjs, iterated on the final post-processed image; fog + tone mapping make it non-linear).
  impBright: { value: new THREE.Vector3(1, 1, 1) },
  // extra blend of the (already crown-bent) card normals toward a crown-spherical normal: neighbouring cards then
  // shade alike and no card outline reads as a separate leaf-shaped patch (the needle normal map keeps micro detail)
  sphere: { value: 0.35 },
  // crown-AO floor (albedo multiplier deep inside the crown) and the forward-scatter lift of the sky light. A real
  // spruce crown is nearly black inside (Cycles reference renders); the old 0.4 floor + 0.4 lift shaded every card as
  // a lit patch on a green ball. MUST match the impostor bake (tools/blender/trees.py impostor_material: 0.25 + 0.75 AO).
  aoMin: { value: 0.25 }, indirBoost: { value: 0.2 },
  // per-card colour spread (needle age / light: blue-green current-year shoots <-> olive, older inner needles)
  cardVar: { value: 1.0 },
  // grazing-angle coverage fade of the needle cards: smoothstep(edge.x, edge.y, |N.V|) (see GLSL_ALPHA_MIP)
  edge: { value: new THREE.Vector2(0.1, 0.45) },
};
// v2 (REALISM pass: sparser ragged cards, crown-AO floor 0.25): impostor re-baked at alpha 0.32, re-measured with
// scratch/trees/lodmatch.mjs (post on, 'high')
const IMP_BRIGHT = {
  off: [0.46, 0.57, 0.48], low: [0.39, 0.49, 0.41], medium: [0.34, 0.42, 0.37], high: [0.33, 0.41, 0.36],
};
// Crown AO -> albedo remap shared by the mesh LODs and the impostor bake (tools/blender/trees.py impostor_material).
// The raw analytic crown AO goes down to 0.14; applied at full strength on top of N8AO it made the crowns black.
const AO_REMAP = 'mix(uFolAOMin, 1.0, vTreeAO)';

// ------------------------------------------------------------------------------------------------------------------
// GLSL snippets
// ------------------------------------------------------------------------------------------------------------------
const GLSL_COMMON = /* glsl */`
uniform float uTime;
uniform vec4 uWind;
uniform vec3 uCamPos;
uniform float uWet;
float tr_hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float tr_ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
float tr_vn(vec3 p) {
  vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  float n = i.x + i.y * 57.0 + i.z * 113.0;
  vec4 a = fract(sin(vec4(n, n + 1.0, n + 57.0, n + 58.0)) * 43758.5453);
  vec4 b = fract(sin(vec4(n + 113.0, n + 114.0, n + 170.0, n + 171.0)) * 43758.5453);
  vec4 m = mix(a, b, f.z);
  return mix(mix(m.x, m.y, f.x), mix(m.z, m.w, f.x), f.y);
}
// per-tree needle colour: blue-green (Picea abies 'glauca'-ish, young stands) <-> dark olive (old, stressed), with a
// brightness spread. Same formula in TreeField.update (instanceColor of the mesh LODs) and the impostor shader.
vec3 tr_treeTint(float t) {
  return mix(vec3(0.84, 0.97, 1.12), vec3(1.12, 1.03, 0.78), t) * (0.8 + 0.32 * fract(t * 7.13));
}
// World-space wind displacement for a point at height h (m above the tree base) and horizontal distance r from the
// trunk axis. phase: per tree, bphase: per branch.
vec3 tr_wind(float h, float r, float phase, float bphase, vec3 wpos) {
  vec2 wd = uWind.xy;
  float S = uWind.z * (1.0 + uWind.w);
  float t = uTime;
  float sway = (0.55 + 0.45 * sin(t * 0.83 + phase * 6.28)) * (0.75 + 0.25 * sin(t * 2.1 + phase * 11.0));
  float hs = h / 20.0;
  vec3 d = vec3(wd.x, 0.0, wd.y) * (S * 0.32 * hs * hs * sway);
  // branch flutter: vertical bob + along-wind, grows with distance from the trunk
  float fl = sin(t * 2.7 + bphase * 6.28 + dot(wpos.xz, vec2(0.37, 0.29))) * 0.6
           + sin(t * 4.9 + bphase * 17.0 + wpos.y * 0.7) * 0.4;
  d += vec3(wd.x * 0.5, 1.0, wd.y * 0.5) * (S * 0.045 * r * fl);
  return d;
}
`;

// Instance helpers for InstancedMesh materials: per-instance distance -> LOD visibility band.
const GLSL_LOD_VERT = /* glsl */`
uniform vec4 uLod;          // x,y: inner transition (start,end); z,w: outer transition (start,end)
varying vec2 vLod;
varying float vTreeAO;
varying float vTreeRnd;
varying float vTreeRnd2;
attribute vec4 aTree;       // r = baked crown AO, g = branch phase, b = random, a = 1
`;
const GLSL_LOD_FRAG = /* glsl */`
uniform vec4 uLod;
varying vec2 vLod;
varying float vTreeAO;
varying float vTreeRnd;
varying float vTreeRnd2;
`;

function lodVertexCode(depthPass) {
  return /* glsl */`
  vec3 transformed = vec3( position );
  #ifdef USE_INSTANCING
    vec3 tr_ipos = instanceMatrix[3].xyz;
    float tr_s2 = dot(instanceMatrix[0].xyz, instanceMatrix[0].xyz);
    float tr_s = sqrt(tr_s2);
  #else
    vec3 tr_ipos = (modelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    float tr_s2 = 1.0; float tr_s = 1.0;
  #endif
  float tr_phase = tr_hash12(tr_ipos.xz * 0.173);
  float tr_h = max(position.y, 0.0) * tr_s;
  float tr_r = length(position.xz) * tr_s;
  vec3 tr_wpos = tr_ipos + vec3(0.0, tr_h, 0.0);
  vec3 tr_dw = tr_wind(tr_h, tr_r, tr_phase, aTree.g, tr_wpos);
  #ifdef USE_INSTANCING
    transformed += (transpose(mat3(instanceMatrix)) * tr_dw) / tr_s2;
  #else
    transformed += tr_dw;
  #endif
  vTreeAO = aTree.r;
  vTreeRnd = fract(aTree.b * 7.31 + aTree.g * 3.17);
  vTreeRnd2 = fract(aTree.b * 13.71 + aTree.g * 5.37 + tr_phase * 3.3);
  float tr_dist = distance(uCamPos, tr_ipos);
  vLod = vec2(uLod.y > 0.0 ? smoothstep(uLod.x, uLod.y, tr_dist) : 1.0, smoothstep(uLod.z, uLod.w, tr_dist));
  `;
}

// LOD selection. A LOD owns the interval [sOuter, sInner) of a selector value n in [0,1); the neighbouring LOD owns the
// complementary interval, so each card / texel block / trunk belongs to exactly one LOD at any distance.
// Foliage cards use their own random (whole cards thin out -> no screen-space dither pattern for post effects to
// smear), impostors use a tree-space block noise, trunks switch at the middle of the band.
const GLSL_DITHER = /* glsl */`
  {
    // whole cards thin out (a per-pixel dither turns into a 1-px checkerboard on the thin silhouettes that SMAA and
    // the chromatic aberration smear into magenta fringes)
    float tr_n = vTreeRnd;
    if (tr_n >= vLod.x || tr_n < vLod.y) discard;
  }
`;
const GLSL_LOD_MID = /* glsl */`
  if (0.5 >= vLod.x || 0.5 < vLod.y) discard;
`;

// Foliage alpha shaping:
//  1) grazing-angle fade: a card seen edge-on smears its texture into streaks (the #1 "CG tree" tell). Its coverage is
//     faded out with the true geometric facing (screen-space derivatives; the vertex normals are bent crown normals),
//     so edge-on cards thin to their needle tips instead and the crossing cards carry the silhouette.
//  2) mip coverage boost (Golus): keeps alpha-tested foliage from thinning out in the small mips.
const GLSL_ALPHA_MIP = /* glsl */`
  {
    vec2 tr_uvp = vMapUv * uMapSize;
    vec2 tr_dx = dFdx(tr_uvp), tr_dy = dFdy(tr_uvp);
    float tr_mip = max(0.0, 0.5 * log2(max(dot(tr_dx, tr_dx), dot(tr_dy, tr_dy))) + uMapBias);
    vec3 tr_gn = normalize(cross(dFdx(vViewPosition), dFdy(vViewPosition)));
    float tr_ndv = abs(dot(tr_gn, normalize(vViewPosition)));
    // fully fade edge-on cards: seen edge-on, a spray compresses to its brown axis twig and reads as a dark stroke
    float tr_edge = smoothstep(uFolEdge.x, uFolEdge.y, tr_ndv);
    diffuseColor.a *= tr_edge;
    // keep the lacy frond structure up close; only compensate coverage loss in the small mips
    diffuseColor.a *= 1.0 + max(0.0, tr_mip - 3.4) * 0.3;
  }
`;
// Sharper needle texture (negative LOD bias; anisotropic filtering keeps it stable).
const GLSL_MAP_BIAS = /* glsl */`
  float tr_needle = 1.0;
  float tr_green = 1.0;
  #ifdef USE_MAP
    vec4 sampledDiffuseColor = texture2D( map, vMapUv, uMapBias );
    diffuseColor *= sampledDiffuseColor;
    // needle-scale glint mask: only the lit, water-filmed needle surfaces reflect the sky, not the whole card
    float tr_lum = dot(sampledDiffuseColor.rgb, vec3(0.3, 0.59, 0.11));
    tr_needle = smoothstep(0.015, 0.09, tr_lum) * (0.35 + 0.65 * tr_hash12(floor(vMapUv * 700.0)));
    // needle (green) vs shoot/twig (brown) texels: only needles transmit light
    tr_green = clamp((sampledDiffuseColor.g - sampledDiffuseColor.r) * 28.0, 0.0, 1.0);
  #endif
`;

// ------------------------------------------------------------------------------------------------------------------
// Mesh materials
// ------------------------------------------------------------------------------------------------------------------
function baseUniforms(extra = {}) {
  return { uTime: WIND.uTime, uWind: WIND.uWind, uCamPos: WIND.uCamPos, uWet: WIND.uWet, uSunVis: WIND.uSunVis, ...extra };
}
const GLSL_SUNVIS = /* glsl */`
  reflectedLight.directDiffuse *= uSunVis;
  reflectedLight.directSpecular *= uSunVis;
`;
// Thin-needle translucency: diffuse transmission through the needle mat (lit from behind) plus forward scattering
// when looking toward the sun. Added before GLSL_SUNVIS so crown self-shadowing applies to it too.
// tr_mask: needle mask (1 = needles), tr_shell: outer-shell factor (crown AO).
function glslTranslucency(mask, shell) {
  return /* glsl */`
  #if NUM_DIR_LIGHTS > 0
  {
    vec3 tr_L = directionalLights[0].direction;
    vec3 tr_V = normalize(vViewPosition);
    float tr_back = pow(clamp(dot(-tr_V, tr_L), 0.0, 1.0), 4.0);
    float tr_thin = clamp(0.5 - 0.5 * dot(normal, tr_L), 0.0, 1.0);
    reflectedLight.directDiffuse += directionalLights[0].color * diffuseColor.rgb * (${mask})
      * (0.3 * tr_thin + 1.1 * tr_back) * (${shell}) * uFolTrans;
  }
  #endif
`;
}

/**
 * Foliage card material (alpha tested, double sided, bent normals, baked crown AO, wind, LOD dither).
 * opts: {map, normalMap, lod: Vector4 uniform value (shared per LOD), alphaTest}
 */
export function createFoliageMaterial(opts) {
  const m = new THREE.MeshStandardMaterial({
    map: opts.map, normalMap: opts.normalMap || null, side: THREE.DoubleSide,
    alphaTest: opts.alphaTest ?? 0.42, roughness: 0.8, metalness: 0.0,
    normalScale: new THREE.Vector2(0.8, 0.8), envMapIntensity: 1.0,
  });
  const lodU = { value: opts.lod || new THREE.Vector4(-2, -1, 1e9, 1e9 + 1) };
  const mapSize = { value: new THREE.Vector2(opts.map?.image?.width || 2048, opts.map?.image?.height || 2048) };
  m.userData.lod = lodU;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, baseUniforms({ uLod: lodU, uMapSize: mapSize, uMapBias: FOLIAGE.mapBias, uFolSpec: FOLIAGE.spec, uFolTint: FOLIAGE.tint, uFolTrans: FOLIAGE.trans, uFolSphere: FOLIAGE.sphere, uFolAOMin: FOLIAGE.aoMin, uFolIndir: FOLIAGE.indirBoost, uFolCardVar: FOLIAGE.cardVar, uFolEdge: FOLIAGE.edge }));
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\n' + GLSL_COMMON + GLSL_LOD_VERT + '\nuniform float uFolSphere;')
      .replace('#include <beginnormal_vertex>', /* glsl */`
        vec3 objectNormal = vec3( normal );
        {
          // crown-spherical normal: away from the trunk axis, tilted up (tree-local; origin at the trunk base)
          vec3 sphN = normalize(vec3(position.x, 0.0, position.z) + vec3(0.0, 0.55 * length(position.xz) + 0.05, 0.0));
          objectNormal = normalize(mix(objectNormal, sphN, uFolSphere));
        }
        #ifdef USE_TANGENT
          vec3 objectTangent = vec3( tangent.xyz );
        #endif
      `)
      .replace('#include <begin_vertex>', lodVertexCode(false));
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + GLSL_COMMON + GLSL_LOD_FRAG + '\nuniform vec2 uMapSize; uniform float uSunVis; uniform float uMapBias; uniform float uFolSpec; uniform vec3 uFolTint; uniform float uFolTrans; uniform float uFolAOMin; uniform float uFolIndir; uniform float uFolCardVar; uniform vec2 uFolEdge;')
      .replace('#include <map_fragment>', GLSL_MAP_BIAS)
      .replace('#include <alphatest_fragment>', GLSL_ALPHA_MIP + '#include <alphatest_fragment>\n' + GLSL_DITHER + /* glsl */`
        diffuseColor.rgb *= ${AO_REMAP} * uFolTint;
        {
          // per-card spread: current-year shoots bluer/brighter, older inner needles olive and duller
          float cv = uFolCardVar;
          vec3 hue = mix(vec3(0.9, 0.99, 1.1), vec3(1.08, 1.02, 0.82), vTreeRnd2);
          diffuseColor.rgb *= mix(vec3(1.0), hue * (0.86 + 0.26 * fract(vTreeRnd2 * 5.3)), cv);
          diffuseColor.rgb *= mix(vec3(1.0), mix(vec3(1.06, 0.96, 0.8), vec3(1.0), vTreeAO), cv * 0.6);
        }
        // wet needles darken only slightly (they gain a sheen instead)
        diffuseColor.rgb *= mix(1.0, 0.92, uWet);
      `)
      .replace('#include <roughnessmap_fragment>', /* glsl */`
        #include <roughnessmap_fragment>
        // wet needles: a water film on each needle (per-needle glint mask), twigs stay matte
        roughnessFactor = mix(roughnessFactor, 0.46, uWet * tr_needle);
      `)
      .replace('#include <aomap_fragment>', glslTranslucency('tr_green', '0.3 + 0.7 * vTreeAO') + GLSL_SUNVIS + /* glsl */`
        #include <aomap_fragment>
        // crown self-occlusion also blocks sky reflections deep inside the canopy
        reflectedLight.indirectSpecular *= vTreeAO * vTreeAO * uFolSpec * tr_needle;
        reflectedLight.directSpecular *= tr_needle;
        // thin-needle forward scattering: the overcast sky shines through the outer shell of the crown
        reflectedLight.indirectDiffuse *= 1.0 + uFolIndir * vTreeAO * vTreeAO * mix(0.5, 1.0, tr_green);
      `)
      // foliage: never flip the (bent) normal on back faces
      .replace('normal *= faceDirection;', '');
  };
  m.customProgramCacheKey = () => 'tree-foliage-v13';
  return m;
}

/** Bark material: albedo + normal + ARM (AO/rough), wetness streaks, wind, LOD dither. */
export function createBarkMaterial(opts) {
  const m = new THREE.MeshStandardMaterial({
    map: opts.map, normalMap: opts.normalMap || null, roughnessMap: opts.arm || null, aoMap: opts.arm || null,
    roughness: 1.0, metalness: 0.0, aoMapIntensity: 0.8, normalScale: new THREE.Vector2(1.2, 1.2),
  });
  if (opts.color) m.color.copy(opts.color);
  const lodU = { value: opts.lod || new THREE.Vector4(-2, -1, 1e9, 1e9 + 1) };
  m.userData.lod = lodU;
  const moss = opts.moss ?? 1.0;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, baseUniforms({ uLod: lodU, uMoss: { value: moss } }));
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\n' + GLSL_COMMON + GLSL_LOD_VERT + '\nvarying vec3 vBarkN; varying vec3 vBarkP;')
      .replace('#include <begin_vertex>', lodVertexCode(false) + '\n  vBarkN = normal; vBarkP = position;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + GLSL_COMMON + GLSL_LOD_FRAG + '\nvarying vec3 vBarkN; varying vec3 vBarkP; uniform float uMoss;')
      .replace('#include <alphatest_fragment>', '#include <alphatest_fragment>\n' + GLSL_LOD_MID + /* glsl */`
        diffuseColor.rgb *= mix(1.0, vTreeAO, 0.85);
        diffuseColor.rgb *= mix(1.0, 0.62, uWet);   // soaked bark is much darker (wet bark albedo ~0.5x dry)
        // Moss (Hypnum / Dicranum cushions): on the trunk's uphill side (tree-local -Z, see vegetation._orientTrees),
        // round the root flare and on the upper face of big branches. Patchy, fading out 1.5-3.5 m up the stem.
        // Wet moss is dark olive-green, linear albedo ~(0.035, 0.055, 0.015).
        float tr_mossM = 0.0;
        {
          vec3 bn = normalize(vBarkN);
          float n1 = tr_vn(vBarkP * vec3(4.0, 1.6, 4.0));
          float n2 = tr_vn(vBarkP * vec3(13.0, 7.0, 13.0));
          float up = smoothstep(0.0, 0.75, dot(bn, vec3(0.0, 0.0, -1.0)));
          float h = vBarkP.y;
          float trunk = up * (1.0 - smoothstep(0.6, 1.6 + 2.2 * n1, h)) + (1.0 - smoothstep(0.05, 0.55 + 0.4 * n1, h)) * 0.85;
          float limb = smoothstep(0.6, 0.95, bn.y) * step(0.8, h) * 0.55;
          tr_mossM = clamp(max(trunk, limb) * smoothstep(0.25, 0.6, n1 * 0.7 + n2 * 0.5) * uMoss, 0.0, 1.0);
          vec3 mossCol = mix(vec3(0.035, 0.055, 0.015), vec3(0.06, 0.08, 0.02), n2);
          diffuseColor.rgb = mix(diffuseColor.rgb, mossCol * mix(1.0, 0.8, uWet) * (0.5 + 0.5 * vTreeAO), tr_mossM);
        }
      `)
      .replace('#include <roughnessmap_fragment>', /* glsl */`
        #include <roughnessmap_fragment>
        roughnessFactor = mix(roughnessFactor, roughnessFactor * 0.55, uWet);
        roughnessFactor = mix(roughnessFactor, 0.62, tr_mossM);
      `)
      .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\n  normal = normalize(mix(normal, normalize(vNormal), tr_mossM * 0.7));');
  };
  m.customProgramCacheKey = () => 'tree-bark-v5' + (moss ? 'm' : '');
  return m;
}

/** Shadow depth material for instanced trees (wind + alpha test; no dither). */
export function createTreeDepthMaterial(map, alphaTest = 0.42) {
  const m = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: map || null, alphaTest: map ? alphaTest : 0 });
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, baseUniforms({ uLod: { value: new THREE.Vector4(-2, -1, 1e9, 1e9 + 1) } }));
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\n' + GLSL_COMMON + GLSL_LOD_VERT)
      .replace('#include <begin_vertex>', lodVertexCode(true));
  };
  m.customProgramCacheKey = () => 'tree-depth-v3' + (map ? 'm' : '');
  return m;
}

// ------------------------------------------------------------------------------------------------------------------
// Impostor material: view-aligned billboard, 4-frame blend (2 azimuths x 2 elevations), normal-atlas relighting
// ------------------------------------------------------------------------------------------------------------------
function createImpostorMaterial(meta, albedo, normal) {
  // alphaTest 0.22: the baked frames hold fractional sub-texel coverage; cutting at 0.5 left the crowns sparser and more
  // tiered than the (mip-filtered) mesh LODs they replace (crown pixel coverage matched with scratch/trees/lodmatch.mjs)
  const m = new THREE.MeshStandardMaterial({ roughness: 0.82, metalness: 0.0, alphaTest: 0.22, side: THREE.FrontSide });
  const vars = [];
  for (let i = 0; i < 8; i++) {
    const v = meta.variants[i % meta.variants.length];
    vars.push(new THREE.Vector4(v.x0 / meta.atlasW, v.width, v.height, v.pivotY));
  }
  const el = meta.elevations.map((e) => e * Math.PI / 180);
  const lodU = { value: new THREE.Vector4(60, 80, 1700, 1800) };
  m.userData.lod = lodU;
  const U = baseUniforms({
    uLod: lodU,
    uAtlas: { value: albedo }, uNAtlas: { value: normal },
    uVar: { value: vars },
    uGrid: { value: new THREE.Vector4(meta.frameW / meta.atlasW, meta.frameH / meta.atlasH, meta.azimuths, meta.elevations.length) },
    uElev: { value: new THREE.Vector2(el[0], el.length > 1 ? el[1] - el[0] : 1) },
    uAtlasSize: { value: new THREE.Vector2(meta.atlasW, meta.atlasH) },
    uTintAmt: { value: 1.0 },
    uImpDebug: { value: 0 },
    uFolSpec: FOLIAGE.spec, uFolTint: FOLIAGE.tint, uFolTrans: FOLIAGE.trans, uImpBright: FOLIAGE.impBright,
  });
  m.userData.uniforms = U;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, U);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\n' + GLSL_COMMON + /* glsl */`
        attribute vec4 iPos;      // xyz base, w scale
        attribute vec4 iData;     // x rotY, y variant, z tint seed, w unused
        uniform vec4 uLod;
        uniform vec4 uVar[8];
        uniform vec4 uGrid;
        uniform vec2 uElev;
        varying vec4 vUvA; varying vec4 vUvB; varying vec4 vW;
        varying vec2 vLod; varying vec3 vImpRot; varying float vTint; varying vec2 vQuv;
      `)
      .replace('#include <beginnormal_vertex>', /* glsl */`
        vec3 objectNormal = normalize(cameraPosition - iPos.xyz);
      `)
      .replace('#include <begin_vertex>', /* glsl */`
        float s = iPos.w;
        int vi = int(iData.y + 0.5);
        vec4 V = uVar[vi];
        vec3 center = iPos.xyz + vec3(0.0, V.w * s, 0.0);
        vec3 toCam = cameraPosition - center;
        float dist = length(toCam);
        vec3 vw = toCam / max(dist, 1e-3);
        vec3 f = -vw;
        vec3 right = cross(f, vec3(0.0, 1.0, 0.0));
        float rl = length(right);
        right = rl > 1e-4 ? right / rl : vec3(1.0, 0.0, 0.0);
        // Cylindrical (Y-aligned) quad through the trunk axis for normal viewing angles: the trunk base and the tree top
        // then project exactly in perspective. The old view-aligned quad leaned its top toward a camera below the crown
        // and the tree read ~10% taller and thinner than its mesh LOD at the switch. Steep views blend back to
        // view-aligned. The height is divided by the quad's projected-length factor so the (orthographic, elevation-
        // tilted) atlas frame keeps its proportions.
        vec3 upV = cross(right, f);
        float elvAbs = abs(asin(clamp(f.y, -1.0, 1.0)));
        vec3 up = normalize(mix(upV, vec3(0.0, 1.0, 0.0), 1.0 - smoothstep(0.52, 0.96, elvAbs)));
        float ud = dot(up, f);
        float hScale = 1.0 / sqrt(max(1.0 - ud * ud, 0.3));
        // camera direction in the tree's local frame (rotation about +Y by -rotY)
        float cr = cos(iData.x), sr = sin(iData.x);
        vec3 vl = vec3(cr * vw.x - sr * vw.z, vw.y, sr * vw.x + cr * vw.z);
        float az = atan(vl.x, vl.z);                       // from +Z toward +X
        float fa = fract(az / 6.2831853) * uGrid.z;
        float i0 = floor(fa); float wa = fa - i0; float i1 = mod(i0 + 1.0, uGrid.z);
        float elv = asin(clamp(vl.y, -1.0, 1.0));
        float fe = clamp((elv - uElev.x) / uElev.y, 0.0, uGrid.w - 1.0);
        float j0 = min(floor(fe), uGrid.w - 2.0); float wj = clamp(fe - j0, 0.0, 1.0);
        if (uGrid.w < 1.5) { j0 = 0.0; wj = 0.0; }
        vec2 c = uv;
        vQuv = c;
        vUvA = vec4(V.x + (i0 + c.x) * uGrid.x, (j0 + c.y) * uGrid.y, V.x + (i1 + c.x) * uGrid.x, (j0 + c.y) * uGrid.y);
        vUvB = vec4(V.x + (i0 + c.x) * uGrid.x, (j0 + 1.0 + c.y) * uGrid.y, V.x + (i1 + c.x) * uGrid.x, (j0 + 1.0 + c.y) * uGrid.y);
        vW = vec4((1.0 - wa) * (1.0 - wj), wa * (1.0 - wj), (1.0 - wa) * wj, wa * wj);
        vImpRot = vec3(cr, sr, 0.0);
        vTint = iData.z;
        float hq = (c.y - 0.5) * V.z * s * hScale;
        vec3 transformed = center + right * ((c.x - 0.5) * V.y * s) + up * hq;
        // pull the card toward the camera so terrain bumps in front of the trunk do not clip the crown, as a uniform
        // scale about the camera (projection unchanged; a plain offset made the impostor ~7% larger than the mesh)
        float pull = min(4.0, V.y * s * 0.3);
        transformed = cameraPosition + (transformed - cameraPosition) * (max(dist - pull, dist * 0.5) / dist);
        float hh = max(0.0, V.w * s + hq);
        transformed += tr_wind(hh, 0.0, tr_hash12(iPos.xz * 0.173), 0.0, center);
        // LOD distance measured exactly like the mesh LODs (main camera -> trunk base) so the complementary dither
        // thresholds line up pixel for pixel across the mesh/impostor band
        float lodDist = distance(uCamPos, iPos.xyz);
        vLod = vec2(smoothstep(uLod.x, uLod.y, lodDist), smoothstep(uLod.z, uLod.w, lodDist));
        if (vLod.x <= 0.0 || vLod.y >= 1.0) transformed = vec3(0.0, -1e5, 0.0); // fully handled by meshes / beyond range
      `);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + GLSL_COMMON + /* glsl */`
        uniform sampler2D uAtlas; uniform sampler2D uNAtlas; uniform vec2 uAtlasSize; uniform float uTintAmt; uniform int uImpDebug;
        uniform float uSunVis; uniform float uFolSpec; uniform vec3 uFolTint; uniform float uFolTrans; uniform vec3 uImpBright;
        varying vec4 vUvA; varying vec4 vUvB; varying vec4 vW;
        varying vec2 vLod; varying vec3 vImpRot; varying float vTint; varying vec2 vQuv;
        vec3 tr_nrm(vec2 uv) { return texture2D(uNAtlas, uv).xyz * 2.0 - 1.0; }
      `)
      .replace('#include <map_fragment>', /* glsl */`
        vec4 ia = texture2D(uAtlas, vUvA.xy) * vW.x + texture2D(uAtlas, vUvA.zw) * vW.y
                + texture2D(uAtlas, vUvB.xy) * vW.z + texture2D(uAtlas, vUvB.zw) * vW.w;
        {
          vec2 uvp = vUvA.xy * uAtlasSize;
          vec2 dx = dFdx(uvp), dy = dFdy(uvp);
          float mip = max(0.0, 0.5 * log2(max(dot(dx, dx), dot(dy, dy))));
          // coverage compensation only in the small mips (the old 1+0.3*mip bloated every silhouette into a blob)
          ia.a *= 1.0 + max(0.0, mip - 1.5) * 0.22;
        }
        diffuseColor.rgb = ia.rgb;
        diffuseColor.a = ia.a;
        // per-tree hue/brightness variation (same formula as the mesh LODs' instanceColor)
        vec3 tint = tr_treeTint(vTint);
        diffuseColor.rgb *= mix(vec3(1.0), tint, uTintAmt) * uFolTint * uImpBright;
        diffuseColor.rgb *= mix(1.0, 0.92, uWet);
      `)
      .replace('#include <alphatest_fragment>', /* glsl */`#include <alphatest_fragment>
        {
          // tree-locked selector at atlas-texel resolution (~1 px in the transition band): it stays fixed to the
          // tree while the camera moves and has none of the old 24x72 block look
          float tr_n = tr_hash12(floor(vQuv * vec2(128.0, 384.0)) + vTint * 17.0);
          if (tr_n >= vLod.x || tr_n < vLod.y) discard;
        }
      `)
      .replace('#include <normal_fragment_maps>', /* glsl */`
        {
          vec3 nl = tr_nrm(vUvA.xy) * vW.x + tr_nrm(vUvA.zw) * vW.y + tr_nrm(vUvB.xy) * vW.z + tr_nrm(vUvB.zw) * vW.w;
          nl = normalize(nl + vec3(0.0, 1e-4, 0.0));
          // tree-local -> world: rotation about +Y by rotY
          vec3 nw = vec3(vImpRot.x * nl.x + vImpRot.y * nl.z, nl.y, -vImpRot.y * nl.x + vImpRot.x * nl.z);
          normal = normalize((viewMatrix * vec4(nw, 0.0)).xyz);
          if (uImpDebug == 2) { gl_FragColor = vec4(nw * 0.5 + 0.5, 1.0); return; }
        }
      `)
      .replace('#include <aomap_fragment>', glslTranslucency('0.8', '0.72') + GLSL_SUNVIS + '#include <aomap_fragment>\n' +
        // average crown AO^2 x needle glint mask (the mesh LODs use per-vertex AO^2 and a per-texel mask)
        '  reflectedLight.indirectSpecular *= 0.16 * 0.45 * uFolSpec;\n' +
        '  reflectedLight.indirectDiffuse *= 1.12;\n')
      .replace('#include <dithering_fragment>', `#include <dithering_fragment>
        if (uImpDebug == 1) gl_FragColor = vec4(diffuseColor.rgb * 4.0, 1.0);`);
  };
  m.customProgramCacheKey = () => 'tree-impostor-v9';
  return m;
}

// ------------------------------------------------------------------------------------------------------------------
// Ground occlusion under each tree: slope-aligned soft disc, multiplicative (dst * src), fog-transmittance aware.
// Stands in for canopy shadow/AO on the forest floor where the shadow map does not reach (and for impostors).
// ------------------------------------------------------------------------------------------------------------------
function createGroundAOMaterial(variantRadius) {
  const m = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 });
  m.blending = THREE.CustomBlending;
  m.blendEquation = THREE.AddEquation;
  m.blendSrc = THREE.ZeroFactor;
  m.blendDst = THREE.SrcColorFactor;
  const U = { uVarR: { value: variantRadius }, uAOStr: { value: 0.55 }, uCamPos: WIND.uCamPos };
  m.userData.uniforms = U;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, U);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', /* glsl */`#include <common>
        attribute vec4 iPos; attribute vec4 iData; attribute vec3 iNrm;
        uniform float uVarR[8]; uniform vec3 uCamPos;
        varying vec2 vDisc; varying float vGDist;`)
      .replace('#include <begin_vertex>', /* glsl */`
        int vi = int(iData.y + 0.5);
        float R = uVarR[vi] * iPos.w * 1.25;
        vec3 n = normalize(iNrm);
        vec3 t = normalize(abs(n.y) < 0.99 ? cross(vec3(0.0, 1.0, 0.0), n) : vec3(1.0, 0.0, 0.0));
        vec3 b = cross(n, t);
        vDisc = position.xz;
        // stretch slightly downhill (crowns lean over the slope)
        vec3 down = normalize(vec3(n.x, 0.0, n.z) + 1e-4);
        vec3 transformed = iPos.xyz + n * 0.12 + (t * position.x + b * position.z) * R + down * R * 0.18 * (1.0 - n.y);
        if (iPos.w <= 0.0) transformed = vec3(0.0, -1e5, 0.0);
        vGDist = distance(uCamPos, iPos.xyz);`)
    ;
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vDisc; varying float vGDist; uniform float uAOStr;')
      .replace('#include <fog_fragment>', /* glsl */`
        float r = length(vDisc);
        float k = uAOStr * (1.0 - smoothstep(0.15, 1.0, r)) * (1.0 - smoothstep(500.0, 900.0, vGDist));
        #ifdef USE_FOG
          vec3 T = hfog_transmittance(vHFogRel, cameraPosition);
        #else
          vec3 T = vec3(1.0);
        #endif
        gl_FragColor = vec4(vec3(1.0) - k * T, 1.0);`);
  };
  m.customProgramCacheKey = () => 'tree-groundao-v2';
  return m;
}

// ------------------------------------------------------------------------------------------------------------------
// Geometry helpers
// ------------------------------------------------------------------------------------------------------------------
/** Collect {bark, foliage} geometries for a node (world matrix relative to the gltf root baked in). */
export function extractParts(node, root) {
  const out = {};
  root.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
  node.traverse((o) => {
    if (!o.isMesh) return;
    const g = dequantize(o.geometry);
    g.applyMatrix4(new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld));
    if (g.attributes.color) {
      g.setAttribute('aTree', g.attributes.color);
      g.deleteAttribute('color');
    } else {
      const n = g.attributes.position.count;
      const a = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) { a[i * 4] = 0.8; a[i * 4 + 1] = 0.5; a[i * 4 + 3] = 1; }
      g.setAttribute('aTree', new THREE.BufferAttribute(a, 4));
    }
    g.computeBoundingSphere();
    const name = o.material?.name || 'bark';
    const key = name.startsWith('foliage') ? 'foliage' : name;
    if (out[key]) out[key] = mergeTwo(out[key], g);
    else out[key] = g;
  });
  return out;
}

/** Copy of a geometry with every attribute as plain Float32 (meshopt/KHR_mesh_quantization data is normalized ints,
 *  which would clamp when a transform is baked in). */
export function dequantize(src) {
  const g = new THREE.BufferGeometry();
  for (const [k, A] of Object.entries(src.attributes)) {
    const n = A.count, s = A.itemSize, arr = new Float32Array(n * s);
    for (let i = 0; i < n; i++) for (let c = 0; c < s; c++) arr[i * s + c] = A.getComponent(i, c); // denormalized
    g.setAttribute(k, new THREE.BufferAttribute(arr, s));
  }
  if (src.index) g.setIndex(new THREE.BufferAttribute(new Uint32Array(src.index.array), 1));
  return g;
}

function mergeTwo(a, b) {
  // tiny merge for same-layout geometries (only needed if a node has two primitives with one material)
  const g = new THREE.BufferGeometry();
  for (const k of Object.keys(a.attributes)) {
    const A = a.attributes[k], B = b.attributes[k];
    if (!B) continue;
    const arr = new Float32Array((A.count + B.count) * A.itemSize);
    for (let i = 0; i < A.count; i++) for (let c = 0; c < A.itemSize; c++) arr[i * A.itemSize + c] = A.getComponent(i, c);
    for (let i = 0; i < B.count; i++) for (let c = 0; c < A.itemSize; c++) arr[(A.count + i) * A.itemSize + c] = B.getComponent(i, c);
    g.setAttribute(k, new THREE.BufferAttribute(arr, A.itemSize));
  }
  const ia = a.index ? Array.from(a.index.array) : [...Array(a.attributes.position.count).keys()];
  const ib = b.index ? Array.from(b.index.array) : [...Array(b.attributes.position.count).keys()];
  g.setIndex([...ia, ...ib.map((i) => i + a.attributes.position.count)]);
  return g;
}

// ------------------------------------------------------------------------------------------------------------------
// TreeField
// ------------------------------------------------------------------------------------------------------------------
const _m4 = new THREE.Matrix4(), _q = new THREE.Quaternion(), _p = new THREE.Vector3(), _s = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0), _c = new THREE.Color(), _sph = new THREE.Sphere(), _frus = new THREE.Frustum();
const _pm = new THREE.Matrix4(), _fwd = new THREE.Vector3();

/** Frustum test for shadow-only proxies: true only for orthographic frustums (directional-light shadow camera), whose
 *  left/right planes are antiparallel; perspective views (fov > 0) never are. three r186 calls
 *  object.intersectsFrustum() for frustum-culled objects in both the main and the shadow pass. */
function shadowOnlyFrustumTest(f) {
  return f.planes[0].normal.dot(f.planes[1].normal) < -0.9999;
}

export class TreeField {
  /**
   * @param ctx
   * @param {{gltf, meta, albedo, normal, variantNames?: string[]}} o
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    const THREE_ = THREE;
    const q = ctx.config?.quality || {};
    // Mesh trees end at 0.6 x treeMeshDistance (48 m on 'high'): beyond that a 25 m spruce is < 260 px tall and the
    // 384 px impostor frames are indistinguishable from the 4k-triangle LOD1, which cost ~350k triangles/pass there.
    const tmd = q.treeMeshDistance ?? 80;
    // 0.5 x treeMeshDistance (40 m on 'high'): with the calibrated impostor albedo the switch is invisible there and
    // it keeps the vegetation under ~0.55M triangles per frame at the start view
    this.meshDist = tmd * 0.5;
    // full-detail LOD0 out to ~24 m on 'high': the nearest two rows of trees along the road carry the realism
    this.lod0Dist = THREE.MathUtils.clamp(tmd * 0.22, 11, 30);
    this.farDist = q.impostorDistance ?? 1800;
    this.shadowDist = Math.min(26, tmd * 0.33);   // LOD1 trees beyond this do not cast shadows (soft overcast sun)
    this.band0 = 3.5;
    this.band1 = Math.max(6, this.meshDist * 0.09);
    this.group = new THREE_.Group();
    this.group.name = 'trees';
    ctx.scene.add(this.group);
    this.meta = o.meta;
    this.variants = [];
    this.count = 0;
    this._lastCam = new THREE.Vector3(1e9, 0, 0);
    this._frame = 0;

    const root = o.gltf.scene;
    const names = o.variantNames || ['conifer_0', 'conifer_1', 'conifer_2', 'conifer_3'];
    // shared textures from the GLB materials
    let folMap = null, folNrm = null, bark = null, barkGrey = null;
    root.traverse((m) => {
      if (!m.isMesh) return;
      const mat = m.material;
      if (mat.name === 'foliage' && !folMap) { folMap = mat.map; folNrm = mat.normalMap; }
      if (mat.name === 'bark' && !bark) bark = mat;
      if (mat.name === 'bark_grey' && !barkGrey) barkGrey = mat;
    });
    this.textures = { folMap, folNrm };
    const aniso = Math.min(8, ctx.renderer.capabilities.getMaxAnisotropy());
    for (const t of [folMap, folNrm, bark?.map, bark?.normalMap, barkGrey?.map]) if (t) t.anisotropy = aniso;

    // LOD band uniforms (shared by every material of a LOD)
    this.lodU = {
      l0: new THREE.Vector4(-2, -1, this.lod0Dist - this.band0, this.lod0Dist + this.band0),
      l1: new THREE.Vector4(this.lod0Dist - this.band0, this.lod0Dist + this.band0, this.meshDist - this.band1, this.meshDist + this.band1),
    };
    const mk = {
      fol: (lod) => createFoliageMaterial({ map: folMap, normalMap: folNrm, lod }),
      bark: (lod, src, grey) => createBarkMaterial({
        map: src?.map, normalMap: src?.normalMap, arm: src?.roughnessMap || src?.metalnessMap || src?.aoMap, lod,
        color: grey ? new THREE.Color(0.92, 0.93, 0.95) : null,
      }),
    };
    const matFol = { 0: mk.fol(this.lodU.l0), 1: mk.fol(this.lodU.l1) };
    const matBark = { 0: mk.bark(this.lodU.l0, bark), 1: mk.bark(this.lodU.l1, bark) };
    const matGrey = { 0: mk.bark(this.lodU.l0, barkGrey || bark, true), 1: mk.bark(this.lodU.l1, barkGrey || bark, true) };
    this.materials = [matFol[0], matFol[1], matBark[0], matBark[1], matGrey[0], matGrey[1]];
    this.depthFol = createTreeDepthMaterial(folMap);
    this.depthBark = createTreeDepthMaterial(null);

    // lod 0: LOD0 mesh; lod 1: LOD1 mesh casting shadows (near); lod 2: same LOD1 mesh without shadows (far)
    const cap = { 0: 120, 1: 260, 2: 900 };
    for (let vi = 0; vi < names.length; vi++) {
      const v = { name: names[vi], lods: {} };
      for (const lod of [0, 1, 2]) {
        const node = root.getObjectByName(lod === 0 ? names[vi] : names[vi] + '_lod1');
        if (!node) continue;
        const parts = extractParts(node, root);
        const meshes = [];
        let shared = null;
        for (const [key, geo] of Object.entries(parts)) {
          const isFol = key === 'foliage';
          const ml = Math.min(lod, 1);
          const mat = isFol ? matFol[ml] : (key === 'bark_grey' ? matGrey[ml] : matBark[ml]);
          const im = new THREE.InstancedMesh(geo, mat, cap[lod]);
          im.name = `${names[vi]}_lod${lod}_${key}`;
          // PERF (QA): every tree material discards (alpha test + dithered LOD crossfade), which defeats the M1's
          // hidden-surface removal. Drawing the distance bands near -> far (LOD0, LOD1, LOD2, then impostors) lets
          // early-z reject hidden needle cards: -12 % GPU time in the escape drive, no visual change.
          im.renderOrder = 1 + lod;
          im.count = 0;
          im.frustumCulled = false;
          im.castShadow = lod < 2;
          im.receiveShadow = !isFol;   // foliage uses uSunVis instead (same look for every LOD)
          im.customDepthMaterial = isFol ? this.depthFol : this.depthBark;
          im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
          if (shared) im.instanceMatrix = shared; else shared = im.instanceMatrix;
          if (isFol) {
            im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cap[lod] * 3), 3);
            im.instanceColor.setUsage(THREE.DynamicDrawUsage);
          }
          this.group.add(im);
          meshes.push(im);
        }
        v.lods[lod] = { meshes, cap: cap[lod] };
        // LOD0 trees cast their shadow with the LOD1 geometry (4x fewer triangles in the shadow pass; the shadow-map
        // texels are too coarse to show the difference). The proxy is culled from every perspective render (main
        // pass, transmission pass, N8AO) and only passes the orthographic frustum of the sun's shadow camera.
        if (lod === 0 && shared) {
          const n1 = root.getObjectByName(names[vi] + '_lod1');
          const p1 = n1 ? extractParts(n1, root) : null;
          if (p1) {
            for (const im of meshes) im.castShadow = false;
            v.shadowProxy = [];
            for (const [key, geo] of Object.entries(p1)) {
              const isFol = key === 'foliage';
              const pm = new THREE.InstancedMesh(geo, isFol ? matFol[0] : (key === 'bark_grey' ? matGrey[0] : matBark[0]), cap[0]);
              pm.name = `${names[vi]}_lod0_shadow_${key}`;
              pm.instanceMatrix = shared;
              pm.count = 0;
              pm.frustumCulled = true;
              pm.intersectsFrustum = shadowOnlyFrustumTest;
              pm.castShadow = true;
              pm.receiveShadow = false;
              pm.customDepthMaterial = isFol ? this.depthFol : this.depthBark;
              this.group.add(pm);
              v.shadowProxy.push(pm);
              meshes.push(pm);   // count/needsUpdate follow the LOD0 meshes
            }
          }
        }
      }
      this.variants.push(v);
    }

    // impostors
    this.impostor = null;
    if (o.meta && o.albedo && o.normal) {
      for (const t of [o.albedo, o.normal]) { t.anisotropy = Math.min(4, aniso); t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter; }
      o.albedo.colorSpace = THREE.SRGBColorSpace;
      o.normal.colorSpace = THREE.NoColorSpace;
      const ib = IMP_BRIGHT[ctx.flags?.nopost ? 'off' : (q.ao || 'medium')] || IMP_BRIGHT.medium;
      FOLIAGE.impBright.value.set(ib[0], ib[1], ib[2]);
      this.impMat = createImpostorMaterial(o.meta, o.albedo, o.normal);
      this.impMat.userData.lod.value.set(this.meshDist - this.band1, this.meshDist + this.band1, this.farDist * 0.94, this.farDist);
      this.materials.push(this.impMat);
    }
  }

  /** data: Float32Array [x,y,z,scale,rotY,variant] * n; normals (optional): Float32Array [nx,ny,nz] * n (ground) */
  setTrees(data, normals = null) {
    const n = Math.floor(data.length / 6);
    this.data = data;
    this.count = n;
    // spatial grid for mesh-LOD selection
    const C = this.cell = 40;
    this.grid = new Map();
    for (let i = 0; i < n; i++) {
      const k = Math.floor(data[i * 6] / C) + ',' + Math.floor(data[i * 6 + 2] / C);
      let a = this.grid.get(k);
      if (!a) this.grid.set(k, (a = []));
      a.push(i);
    }
    this.tint = new Float32Array(n);
    for (let i = 0; i < n; i++) this.tint[i] = fract(Math.sin(i * 12.9898 + data[i * 6] * 0.0137) * 43758.5453);

    if (this.impMat) {
      if (this.impostor) { this.group.remove(this.impostor); this.impostor.geometry.dispose(); }
      const g = new THREE.InstancedBufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], 3));
      g.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
      g.setAttribute('normal', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3));
      g.setIndex([0, 1, 2, 0, 2, 3]);
      const ip = new Float32Array(n * 4), id = new Float32Array(n * 4);
      const nv = this.meta.variants.length;
      for (let i = 0; i < n; i++) {
        ip[i * 4] = data[i * 6]; ip[i * 4 + 1] = data[i * 6 + 1]; ip[i * 4 + 2] = data[i * 6 + 2]; ip[i * 4 + 3] = data[i * 6 + 3];
        id[i * 4] = data[i * 6 + 4]; id[i * 4 + 1] = ((data[i * 6 + 5] | 0) % nv + nv) % nv; id[i * 4 + 2] = this.tint[i];
      }
      // full per-tree arrays; the GPU buffers hold a compacted, view-culled subset (see _compactImpostors)
      this._srcPos = ip; this._srcData = id;
      g.setAttribute('iPos', new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4).setUsage(THREE.DynamicDrawUsage));
      g.setAttribute('iData', new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4).setUsage(THREE.DynamicDrawUsage));
      g.instanceCount = 0;
      g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e7);
      const mesh = new THREE.Mesh(g, this.impMat);
      mesh.name = 'tree_impostors';
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      mesh.renderOrder = 4;   // after terrain + mesh trees (see the LOD renderOrder note above); sky dome is 1e6
      this.impostor = mesh;
      this.group.add(mesh);

      // ground occlusion decals (same instance buffers)
      if (this.groundAO) { this.group.remove(this.groundAO); this.groundAO.geometry.dispose(); }
      const gq = new THREE.InstancedBufferGeometry();
      gq.setAttribute('position', new THREE.Float32BufferAttribute([-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1], 3));
      gq.setAttribute('normal', new THREE.Float32BufferAttribute([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 3));
      gq.setIndex([0, 2, 1, 0, 3, 2]);
      gq.setAttribute('iPos', g.getAttribute('iPos'));
      gq.setAttribute('iData', g.getAttribute('iData'));
      const nrmA = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        if (normals) { nrmA[i * 3] = normals[i * 3]; nrmA[i * 3 + 1] = normals[i * 3 + 1]; nrmA[i * 3 + 2] = normals[i * 3 + 2]; }
        else nrmA[i * 3 + 1] = 1;
      }
      this._srcNrm = nrmA;
      gq.setAttribute('iNrm', new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage));
      gq.instanceCount = 0;
      gq.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e7);
      const radii = [];
      for (let i = 0; i < 8; i++) radii.push(this.meta.variants[i % nv].crownRadius || 3);
      this.groundAOMat = this.groundAOMat || createGroundAOMaterial(radii);
      const gm = new THREE.Mesh(gq, this.groundAOMat);
      gm.name = 'tree_ground_ao';
      gm.frustumCulled = false;
      gm.matrixAutoUpdate = false;
      gm.renderOrder = -2;
      this.groundAO = gm;
      this.group.add(gm);
    }
    this._lastCam.set(1e9, 0, 0);
    this._impCam = null;
  }

  /**
   * Impostor + ground-AO instance culling on the CPU: copies the trees that can be on screen into the (shared) GPU
   * instance buffers. Rebuilt only when the camera has moved > 4 m or turned > ~7 deg, so the view cone gets a wide
   * angular margin and every tree within 70 m is always kept. Drawing every one of the 25k trees cost 100k triangles
   * (mostly behind the camera) in vertex work for nothing.
   */
  _compactImpostors(force = false) {
    const cam = this.ctx.camera;
    const g = this.impostor?.geometry;
    if (!cam || !g || !this._srcPos) return;
    const fwd = _fwd.set(0, 0, -1).applyQuaternion(cam.quaternion);
    const cp = cam.position;
    if (!force && this._impCam && cp.distanceToSquared(this._impCam.p) < 16 && fwd.dot(this._impCam.f) > 0.9925) return;
    if (!this._impCam) this._impCam = { p: new THREE.Vector3(), f: new THREE.Vector3() };
    this._impCam.p.copy(cp); this._impCam.f.copy(fwd);
    // half-angle of the view cone: diagonal half-FOV + 16 deg margin
    const tanV = Math.tan(THREE.MathUtils.degToRad((cam.fov || 70) * 0.5));
    const half = Math.atan(tanV * Math.hypot(1, cam.aspect || 1.78)) + THREE.MathUtils.degToRad(16);
    const cosH = Math.cos(Math.min(half, Math.PI * 0.49));
    const far2 = this.farDist * this.farDist, near2 = 70 * 70;
    const SP = this._srcPos, SD = this._srcData, SN = this._srcNrm;
    const ip = g.getAttribute('iPos'), id = g.getAttribute('iData');
    const P = ip.array, D = id.array;
    const gq = this.groundAO?.geometry, inA = gq?.getAttribute('iNrm'), N = inA?.array;
    let k = 0;
    for (let i = 0; i < this.count; i++) {
      const o = i * 4;
      const s = SP[o + 3];
      if (s <= 0) continue;
      const dx = SP[o] - cp.x, dy = SP[o + 1] + 10 * s - cp.y, dz = SP[o + 2] - cp.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > far2) continue;
      if (d2 > near2) {
        // cone test against the tree's mid-height point, widened by its bounding radius
        const d = Math.sqrt(d2);
        const c = (dx * fwd.x + dy * fwd.y + dz * fwd.z) / d;
        if (c < cosH - 16 * s / d) continue;
      }
      const q = k * 4;
      P[q] = SP[o]; P[q + 1] = SP[o + 1]; P[q + 2] = SP[o + 2]; P[q + 3] = s;
      D[q] = SD[o]; D[q + 1] = SD[o + 1]; D[q + 2] = SD[o + 2]; D[q + 3] = SD[o + 3];
      if (N) { N[k * 3] = SN[i * 3]; N[k * 3 + 1] = SN[i * 3 + 1]; N[k * 3 + 2] = SN[i * 3 + 2]; }
      k++;
    }
    g.instanceCount = k;
    for (const a of [ip, id, inA]) {
      if (!a) continue;
      a.clearUpdateRanges?.();
      a.addUpdateRange?.(0, k * a.itemSize);
      a.needsUpdate = true;
    }
    if (gq) gq.instanceCount = k;
    this.impostorCount = k;
  }

  /** Mark tree i removed (scale 0) in both the LOD data and the impostor source (e.g. swept away by the slide). */
  hideTree(i) {
    if (!this.data || i < 0 || i >= this.count) return;
    this.data[i * 6 + 3] = 0;
    if (this._srcPos) this._srcPos[i * 4 + 3] = 0;
    this._impCam = null;
    this._lastCam.set(1e9, 0, 0);
  }

  update(dt) {
    const cam = this.ctx.camera;
    if (!cam || !this.data) return;
    this._frame++;
    this._compactImpostors();
    const cp = cam.position;
    const moved = cp.distanceToSquared(this._lastCam) > 0.25;
    const turned = !this._lastQ || Math.abs(cam.quaternion.dot(this._lastQ)) < 0.99985;
    if (!moved && !turned && this._frame % 30 !== 0) return;
    this._lastCam.copy(cp);
    (this._lastQ || (this._lastQ = new THREE.Quaternion())).copy(cam.quaternion);
    const data = this.data, C = this.cell;
    const R1 = this.meshDist + this.band1, R0 = this.lod0Dist + this.band0, L1in = this.lod0Dist - this.band0;
    const cx = Math.floor(cp.x / C), cz = Math.floor(cp.z / C), rc = Math.ceil(R1 / C);
    cam.updateMatrixWorld();
    _pm.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    _frus.setFromProjectionMatrix(_pm);
    const fwd = _fwd.set(0, 0, -1).applyQuaternion(cam.quaternion);
    const counts = this.variants.map(() => [0, 0, 0]);
    const SD = this.shadowDist;
    const nv = this.variants.length;
    for (let gx = cx - rc; gx <= cx + rc; gx++) {
      for (let gz = cz - rc; gz <= cz + rc; gz++) {
        const cell = this.grid.get(gx + ',' + gz);
        if (!cell) continue;
        for (const i of cell) {
          const o = i * 6;
          const dx = data[o] - cp.x, dy = data[o + 1] - cp.y, dz = data[o + 2] - cp.z;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 > R1 * R1) continue;
          const d = Math.sqrt(d2);
          const vi = ((data[o + 5] | 0) % nv + nv) % nv;
          const V = this.variants[vi];
          const s = data[o + 3];
          if (s <= 0) continue;
          // trees well behind the camera: their shadows fall away from the view (sun ahead) -> skip
          if (d > 9 && (dx * fwd.x + dz * fwd.z) < -0.35 * Math.hypot(dx, dz)) continue;
          let built = false;
          for (const lod0 of [0, 1]) {
            if (lod0 === 0 && d > R0) continue;
            if (lod0 === 1 && d < L1in) continue;
            const lod = lod0 === 1 && d > SD ? 2 : lod0;
            const L = V.lods[lod];
            if (!L) continue;
            if (lod >= 1) {
              // frustum cull mid-distance trees (generous sphere: whole tree)
              _sph.center.set(data[o], data[o + 1] + 12 * s, data[o + 2]);
              _sph.radius = 16 * s;
              if (!_frus.intersectsSphere(_sph)) continue;
            }
            const k = counts[vi][lod];
            if (k >= L.cap) continue;
            if (!built) {
              _p.set(data[o], data[o + 1], data[o + 2]);
              _q.setFromAxisAngle(_up, data[o + 4]);
              _s.setScalar(s);
              _m4.compose(_p, _q, _s);
              built = true;
            }
            const t = this.tint[i];
            treeTint(t, _c);
            for (const im of L.meshes) {
              if (im.instanceColor) im.setColorAt(k, _c);
            }
            L.meshes[0].setMatrixAt(k, _m4);
            counts[vi][lod] = k + 1;
          }
        }
      }
    }
    for (let vi = 0; vi < nv; vi++) {
      for (const lod of [0, 1, 2]) {
        const L = this.variants[vi].lods[lod];
        if (!L) continue;
        const n = counts[vi][lod];
        for (const im of L.meshes) {
          im.count = n;
          if (n) {
            im.instanceMatrix.clearUpdateRanges?.();
            im.instanceMatrix.addUpdateRange?.(0, n * 16);
            im.instanceMatrix.needsUpdate = true;
            if (im.instanceColor) { im.instanceColor.needsUpdate = true; }
          }
        }
      }
    }
    this.stats = { lod0: counts.reduce((a, c) => a + c[0], 0), lod1: counts.reduce((a, c) => a + c[1], 0),
      lod1far: counts.reduce((a, c) => a + c[2], 0), total: this.count };
  }

  /** Change LOD distances at runtime (debug / quality changes). */
  setDistances(lod0 = this.lod0Dist, mesh = this.meshDist, far = this.farDist) {
    this.lod0Dist = lod0; this.meshDist = mesh; this.farDist = far;
    this.band1 = Math.max(6, mesh * 0.09);
    this.shadowDist = Math.min(26, mesh * 0.66);
    this.lodU.l0.set(-2, -1, lod0 - this.band0, lod0 + this.band0);
    this.lodU.l1.set(lod0 - this.band0, lod0 + this.band0, mesh - this.band1, mesh + this.band1);
    this.impMat?.userData.lod.value.set(mesh - this.band1, mesh + this.band1, far * 0.94, far);
    this._lastCam.set(1e9, 0, 0);
    this._impCam = null;
  }

  dispose() {
    this.ctx.scene.remove(this.group);
    this.group.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    for (const m of this.materials) m.dispose();
  }
}

function fract(x) { return x - Math.floor(x); }
/** JS twin of GLSL tr_treeTint (per-tree needle colour, blue-green <-> dark olive). */
function treeTint(t, out) {
  const b = 0.8 + 0.32 * fract(t * 7.13);
  return out.setRGB((0.84 + 0.28 * t) * b, (0.97 + 0.06 * t) * b, (1.12 - 0.34 * t) * b);
}
