// LANDSLIDE system (SLIDE workstream): rockfall physics + the advancing debris front.
//
// Public API (DESIGN.md "landslide"):
//   frontS                 current s of the debris front (0 = no front yet); the mass advances toward +s
//   setFront(s)            place the front (small jumps are eased visually; big jumps snap)
//   setFrontSpeed(v)       m/s along +s (frontS advances by itself in update)
//   triggerIntroRockfall() scripted scar collapse at s≈135–155: big boulders + debris spray, then a mud/rock pile that
//                          blocks the road behind the player (parks the front at s≈150 if no front exists yet)
//   triggerGully(i, opts)  rockfall down gully i (0..3). opts: {count=7, duration=3, rMin=0.2, rMax=0.9, speed=6, aim}
//                          aim: true -> 2 extra ballistic rocks timed to cross the road near the car
//   startEscape()          escalating, timed rockfall ahead of/around the car (near misses; big hits are lethal)
//   stopEscape()
//   spawnBoulder(pos, vel, radius, opts) -> rock handle|null. opts: {angVel, variant, mud 0..1, lethal=true}
//   rumble 0..1            ground rumble from nearby rock activity + front proximity (audio / camera read it)
//   clear()                remove all rocks/debris and the front (checkpoint restart)
//   Extra for particles/audio: front {active, s, speed, height, visS}, frontPoint(d, out) (snout world point),
//   rockCount, surfaceY(s, d) (top of the debris mass or null)
// Events: 'impact' {position: Vector3, energy (≈KE/1e5), radius, source:'rock'|'front', hits, surface:'mud'|'asphalt'|
//         'gravel'|'rock' (what the rock hit: particles throw mud from soil, chips + water spray from asphalt), ground (y)},
//         'hazard:hit' {target:'player'|'car', cause:'boulder'|'front', energy}
import * as THREE from 'three';
import { G, groups } from '../physics/world.js';
import { PAL, palMix } from './particles.js';

const clamp = THREE.MathUtils.clamp;
const lerp = THREE.MathUtils.lerp;
const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };

const DENSITY = 2650;          // granite/gneiss kg/m^3
const GRAV = 9.81;
const DEBRIS_CAP = 140;        // settled rocks kept as static instances
const FRONT_SNOUT = 9.0;       // m of rounded snout (+ slurry apron) ahead of frontS
const FRONT_BACK = 150;        // m of mass drawn behind frontS
const FRONT_D0 = -11.5, FRONT_D1 = 9.5, FRONT_ND = 56;
const FRONT_H = 4.4;           // nominal thickness above the road (m)
const GROUND_DS = 1.0;         // ground-height cache row spacing (m)
const TOPO_L = 128, TOPO_R = 0.25;
// full-detail (2.9k tri, baked normal/AO/ORM) range for flying rocks and settled debris; LOD1 beyond
const LIVE_FULL = 62, LIVE_FULL_SMALL = 34, DEBRIS_FULL = 40;
const TRAIL_RANGE = 62;         // flying rocks shed mud / grit / water within this distance of the camera
const FRONT_PEBBLES = { ultra: 420, high: 360, medium: 240, low: 160 };   // cobbles in the snout / body (~290 tris each) // seamless flow-topography tile: length and sample spacing (m)

// ------------------------------------------------------------------------------------------------ noise (CPU)
const PERM = new Uint8Array(512);
{ let s = 1337; const p = [...Array(256).keys()]; for (let i = 255; i > 0; i--) { s = (s * 16807) % 2147483647; const j = s % (i + 1); [p[i], p[j]] = [p[j], p[i]]; } for (let i = 0; i < 512; i++) PERM[i] = p[i & 255]; }
function vnoise3(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), w = zf * zf * (3 - 2 * zf);
  const X = xi & 255, Y = yi & 255, Z = zi & 255;
  const A0 = PERM[X] + Y, A1 = PERM[X + 1] + Y;
  const p00 = PERM[A0], p10 = PERM[A1], p01 = PERM[A0 + 1], p11 = PERM[A1 + 1];
  const a0 = PERM[p00 + Z], b0 = PERM[p10 + Z], c0 = PERM[p01 + Z], d0 = PERM[p11 + Z];
  const a1 = PERM[p00 + Z + 1], b1 = PERM[p10 + Z + 1], c1 = PERM[p01 + Z + 1], d1 = PERM[p11 + Z + 1];
  const l0 = (a0 + (b0 - a0) * u) + ((c0 + (d0 - c0) * u) - (a0 + (b0 - a0) * u)) * v;
  const l1 = (a1 + (b1 - a1) * u) + ((c1 + (d1 - c1) * u) - (a1 + (b1 - a1) * u)) * v;
  return (l0 + (l1 - l0) * w) * (2 / 255) - 1;
}
function vnoise2(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const X = xi & 255, Y = yi & 255;
  const A0 = PERM[X] + Y, A1 = PERM[X + 1] + Y;
  const a = PERM[PERM[A0] + 17], b = PERM[PERM[A1] + 17], c = PERM[PERM[A0 + 1] + 17], d = PERM[PERM[A1 + 1] + 17];
  const l0 = a + (b - a) * u, l1 = c + (d - c) * u;
  return (l0 + (l1 - l0) * v) * (2 / 255) - 1;
}
const fbm3 = (x, y, z) => vnoise3(x, y, z) * 0.55 + vnoise3(x * 2.03 + 5.1, y * 2.03, z * 2.03) * 0.3 + vnoise3(x * 4.1, y * 4.1 + 3.3, z * 4.1) * 0.15;
const fbm2 = (x, y) => vnoise2(x, y) * 0.55 + vnoise2(x * 2.03 + 5.1, y * 2.03) * 0.3 + vnoise2(x * 4.1, y * 4.1 + 3.3) * 0.15;

// deterministic RNG
function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

const _v = new THREE.Vector3(), _w = new THREE.Vector3(), _u = new THREE.Vector3(), _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion();
const _m = new THREE.Matrix4(), _s = new THREE.Vector3(), _e = new THREE.Euler(), _p = new THREE.Vector3(), _l = new THREE.Vector3();
const _down = new THREE.Vector3(0, -1, 0);
const _col = new THREE.Color();
// Instanced meshes keep a running world box of what was written into them, so they can be frustum culled (camera and
// shadow passes) instead of always drawing every rock of the slide.
const _bbv = new THREE.Vector3();
function instReset(im) { im.count = 0; (im.userData.bb || (im.userData.bb = new THREE.Box3())).makeEmpty(); }
function instPush(im, m4, mud, vel = null) {
  const i = im.count, e = m4.elements;
  im.setMatrixAt(i, m4);
  im.geometry.attributes.aMud.array[i] = mud;
  if (im.userData.vel) { const a = im.geometry.attributes.aVel.array; if (vel) { a[i * 3] = vel.x; a[i * 3 + 1] = vel.y; a[i * 3 + 2] = vel.z; } else a[i * 3] = a[i * 3 + 1] = a[i * 3 + 2] = 0; }
  im.count = i + 1;
  const r = Math.hypot(e[0], e[1], e[2]) * 1.1;
  const bb = im.userData.bb || (im.userData.bb = new THREE.Box3());
  bb.expandByPoint(_bbv.set(e[12] + r, e[13] + r, e[14] + r)); bb.expandByPoint(_bbv.set(e[12] - r, e[13] - r, e[14] - r));
}
function instFinish(im) {
  im.instanceMatrix.needsUpdate = true;
  im.geometry.attributes.aMud.needsUpdate = true;
  if (im.userData.vel && im.count > 0) im.geometry.attributes.aVel.needsUpdate = true;
  if (im.count > 0 && im.userData.bb && !im.userData.bb.isEmpty()) im.boundingSphere = im.userData.bb.getBoundingSphere(im.boundingSphere || new THREE.Sphere());
  im.visible = im.count > 0;
}
const _gp = new THREE.Vector3(), _gl = new THREE.Vector3(), _cp = new THREE.Vector3(), _cv = new THREE.Vector3(), _cq = new THREE.Quaternion(), _rl = new THREE.Vector3();

// ------------------------------------------------------------------------------------------------ shader patches
// Rockfall boulders (REAL-WORLD MODEL): fresh gneiss/granite blocks torn out of a soaked slope.
//  * Wet crystalline rock darkens only ~30 % (low porosity: a thin water film, Lekner & Dorf 1988): the atlas albedo
//    (~0.11 linear, weathered dark gneiss) ends near 0.075. The film drains in patches, so roughness varies 0.36-0.55
//    (never below ~0.36: a film on a crystalline, micro-rough face is not a mirror, so the overcast sky never paints a
//    pale glossy highlight across the top faces).
//  * A block that has just broken out of the joint set shows 1-2 FRESH FRACTURE facets: unweathered feldspar/quartz,
//    neutral grey, ~1.75x brighter than the iron-stained joint faces (fresh granite ~0.30 dry / ~0.20 wet vs weathered
//    ~0.11-0.15). That facet-to-facet contrast is what keeps a tumbling block angular at 20-60 m.
//  * Rocks from a slide carry the soil they were bedded in: a clinging coat of wet forest earth (mud_forest graded to
//    linear albedo ~0.055/0.037/0.023, i.e. saturated dark-brown soil) with its own relief, thickest on the side that
//    was buried, and fine soil CAKED INTO EVERY CAVITY (baked AO = cavity) even on lightly coated blocks, plus a thin
//    brown silt stain around it. Coat roughness: matte clods 0.66, smeared mud 0.38.
//  * The shared 2048 atlas gives ~600 px per boulder, so crystal-scale grain (2-3 cm) is added procedurally at
//    close range (faded out with the screen-space derivative so it never aliases).
//  * Motion: a dashcam exposes ~1/60 s. The trailing half of a fast block is smeared back along its velocity relative
//    to the camera by a fraction of that exposure (aVel, capped): an in-camera motion blur hint at zero cost.
// aMud packs the coat amount (fract) and a per-rock seed (floor / 64) that orients and places the coat.
const packMud = (mud, seed) => Math.floor(clamp(seed, 0, 0.999) * 64) + clamp(mud, 0, 0.995);
const ROCK_MUD_VERT_HEAD = /* glsl */`
attribute float aMud;
attribute vec3 aVel;
uniform vec4 rkMB;         // x exposure fraction (s), y max smear (m)
varying float vMud;
varying float vSeed;
varying vec3 vRkObj;
varying vec3 vRkN;
varying vec3 vRkR0;
varying vec3 vRkR1;
varying vec3 vRkR2;
`;
const ROCK_MUD_VERT = /* glsl */`
{
  float rkSc = 1.0;
  mat3 rkM = mat3( modelViewMatrix );
  #ifdef USE_INSTANCING
    rkSc = length( instanceMatrix[0].xyz );
    rkM = rkM * mat3( instanceMatrix );
    // exposure smear: the trailing hemisphere (normal against the relative velocity) is pulled back along -v
    float rkSp = length( aVel );
    if ( rkSp > 0.8 && rkMB.x > 0.0 ) {
      mat3 im = mat3( instanceMatrix );
      vec3 dir = aVel / rkSp;
      float tr = max( 0.0, -dot( normalize( im * objectNormal ), dir ) );
      float len = min( rkSp * rkMB.x, rkMB.y );
      transformed += ( transpose( im ) * ( -dir * len * tr * tr ) ) / ( rkSc * rkSc );
    }
  #endif
  vRkObj = position * rkSc;
  vRkN = normal;
  vRkR0 = normalize( rkM[0] ); vRkR1 = normalize( rkM[1] ); vRkR2 = normalize( rkM[2] );
  vMud = fract( aMud ); vSeed = floor( aMud );
}
`;
const ROCK_MUD_FRAG_HEAD = /* glsl */`
varying float vMud;
varying float vSeed;
varying vec3 vRkObj;
varying vec3 vRkN;
varying vec3 vRkR0;
varying vec3 vRkR1;
varying vec3 vRkR2;
uniform sampler2D rkMudMap;
uniform sampler2D rkMudNor;
uniform vec4 rkP;          // x wetness 0..1, y coat texture frequency (1/m), z grain strength, w unused
float rk_h(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float rk_n(vec3 x) {
  vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(rk_h(i), rk_h(i + vec3(1,0,0)), f.x), mix(rk_h(i + vec3(0,1,0)), rk_h(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(rk_h(i + vec3(0,0,1)), rk_h(i + vec3(1,0,1)), f.x), mix(rk_h(i + vec3(0,1,1)), rk_h(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float rkMudMask = 0.0;
float rkFilm = 0.0;
float rkDrain = 0.0;
float rkRoot = 0.0;
float rkFresh = 0.0;
vec3 rkW = vec3(0.333);
vec3 rkNo = vec3(0.0, 1.0, 0.0);
`;
const ROCK_MUD_FRAG = /* glsl */`
{
  rkNo = normalize( vRkN );
  vec3 P = vRkObj;
  float sd = vSeed;
  vec3 sdir = normalize( vec3( fract( sd * 0.137 ) - 0.5, fract( sd * 0.311 ) - 0.5, fract( sd * 0.719 ) - 0.5 ) + 1e-3 );
  float side = dot( rkNo, sdir );
  // fresh fracture facets: one per block (two on half of them), never on the buried side
  vec3 fa = normalize( vec3( fract( sd * 0.531 + 0.21 ) - 0.5, fract( sd * 0.917 + 0.63 ) - 0.5, fract( sd * 0.273 + 0.37 ) - 0.5 ) + 1e-3 );
  fa = normalize( fa - sdir * ( max( dot( fa, sdir ), 0.0 ) * 1.6 + 0.1 ) );
  vec3 fb = normalize( cross( fa, sdir ) + fa * 0.25 );
  float fresh = smoothstep( 0.7, 0.9, dot( rkNo, fa ) );
  fresh = max( fresh, smoothstep( 0.78, 0.92, dot( rkNo, fb ) ) * step( 0.5, fract( sd * 0.377 + 0.1 ) ) );
  fresh *= 1.0 - smoothstep( 0.0, 0.45, side );
  float n1 = rk_n( P * 1.7 + sd * 1.37 );
  float n2 = rk_n( P * 4.9 + 3.1 + sd * 0.71 );
  float n3 = rk_n( P * 13.0 + 7.7 );
  float cav = 0.0;
  #ifdef USE_AOMAP
    cav = 1.0 - texture2D( aoMap, vAoMapUv ).r;
  #endif
  float on = step( 0.02, vMud );
  // the coat: thick blotches on the side that was buried, scraped off the fresh facets. Coverage grows with vMud from
  // ~15 % (a block that fell clean out of the joint set) to ~85 % (one that rode in the soil), so bare wet rock always
  // shows somewhere and keeps the block reading as stone
  float m = n1 * 0.5 + n2 * 0.3 + n3 * 0.2 + side * 0.6 - fresh * 0.8 + ( rk_n( P * 31.0 + 1.7 ) - 0.5 ) * 0.2;
  float thr = mix( 0.98, 0.12, vMud );
  rkMudMask = smoothstep( thr - 0.03, thr + 0.05, m ) * on;
  // fine soil packed into every crack, joint and hollow (baked AO: 82 % of the atlas is > 0.9, the creases 0.6-0.85)
  float cake = smoothstep( 0.06, 0.2, cav + ( n3 - 0.5 ) * 0.06 ) * ( 0.8 + 0.2 * vMud ) * on;
  rkMudMask = max( rkMudMask, cake );
  rkFilm = smoothstep( thr - 0.4, thr, m ) * on;
  rkDrain = rk_n( P * 3.3 + sd * 0.53 );
  rkW = pow( abs( rkNo ), vec3( 4.0 ) ); rkW /= dot( rkW, vec3( 1.0 ) );
  vec3 base = diffuseColor.rgb;
  float lum = dot( base, vec3( 0.2126, 0.7152, 0.0722 ) );
  vec3 rock = base * ( 1.0 - 0.3 * rkP.x );
  // fresh fracture: unweathered crystals, neutral grey, with the atlas's own grain kept as luminance variation
  vec3 fr = vec3( lum * 1.15 ) * vec3( 0.96, 0.99, 1.03 ) * ( 1.0 - 0.3 * rkP.x ) * ( 0.9 + 0.2 * n3 );
  rock = mix( rock, fr, fresh * 0.9 );
  rock = mix( rock, rock * vec3( 0.8, 0.69, 0.55 ), rkFilm * 0.75 * ( 1.0 - fresh * 0.7 ) );      // silt stain
  rkFresh = fresh;
  if ( rkMudMask > 0.002 ) {
    vec3 Q = P * rkP.y + sd * 0.173;
    vec3 mc = texture2D( rkMudMap, Q.zy ).rgb * rkW.x + texture2D( rkMudMap, Q.xz ).rgb * rkW.y + texture2D( rkMudMap, Q.xy ).rgb * rkW.z;
    // brown_mud_03 (scan mean linear 0.089/0.070/0.046) graded to saturated dark-brown soil: ~0.05/0.037/0.025 wet
    float ml = dot( mc, vec3( 0.3, 0.59, 0.11 ) );
    mc = mix( vec3( ml ), mc, 0.8 ) * vec3( 0.95, 0.82, 0.74 ) * ( 1.0 - 0.42 * rkP.x ) * ( 0.85 + 0.3 * n2 );
    // caked soil in the hollows is darker still (finer, wetter)
    mc *= 1.0 - 0.3 * smoothstep( 0.1, 0.3, cav );
    // fine roots torn out with the soil: a tangle of thin fibres (two warped stripe sets) inside the thick coat
    vec3 wp = P * 6.0 + ( rk_n( P * 2.3 + 4.0 ) * 2.6 - 1.3 ) * vec3( 1.0, -0.7, 0.45 ) + n3 * vec3( -0.6, 0.5, 0.9 );
    float f1 = abs( sin( dot( wp, vec3( 0.8, 0.35, -0.5 ) ) * 3.1 + sd ) );
    float f2 = abs( sin( dot( wp, vec3( -0.3, 0.9, 0.3 ) ) * 2.3 + sd * 0.7 ) );
    rkRoot = ( 1.0 - smoothstep( 0.035, 0.09, min( f1, f2 ) ) ) * smoothstep( 0.35, 0.8, rkMudMask ) * step( 0.3, rk_n( P * 1.1 + sd ) );
    mc = mix( mc, vec3( 0.075, 0.052, 0.036 ), rkRoot * 0.85 );
    rock = mix( rock, mc, rkMudMask );
  }
  diffuseColor.rgb = rock;
}
`;
const ROCK_ROUGH = /* glsl */`
#include <roughnessmap_fragment>
{
  // atlas roughness ~0.55-0.75 dry; a rain film on crystalline rock drains in patches: 0.36-0.5 wet. Never below
  // ~0.36 (a film on a micro-rough face is no mirror: no pale glossy highlight across the sky-facing faces)
  float r = roughnessFactor;
  r = mix( r, r * mix( 0.55, 0.8, rkDrain ), rkP.x );
  r = clamp( r, 0.4, 0.85 );
  r = mix( r, 0.42, rkFilm * 0.4 );
  r = mix( r, max( r, 0.5 ), rkFresh );                                 // fresh fracture: crystal facets, dull
  // wet soil: matte clods ~0.6, smeared slurry ~0.36 (Lekner & Dorf: a saturated fine soil is a film over grit)
  float mr = mix( 0.62, 0.38, smoothstep( 0.45, 0.8, rkDrain ) );
  roughnessFactor = mix( mix( r, mr, rkMudMask ), 0.6, rkRoot );
}
`;
const ROCK_NORMAL = /* glsl */`
#include <normal_fragment_maps>
{
  vec3 P = vRkObj;
  mat3 rkR = mat3( vRkR0, vRkR1, vRkR2 );
  vec3 G = P * 38.0;
  float fw = length( fwidth( G ) );
  float gs = rkP.z * ( 1.0 - smoothstep( 0.2, 0.6, fw ) ) * ( 1.0 - rkMudMask ) * ( 1.0 + rkFresh );
  if ( gs > 0.01 ) {
    float g0 = rk_n( G );
    vec3 gd = vec3( rk_n( G + vec3( 0.5, 0.0, 0.0 ) ), rk_n( G + vec3( 0.0, 0.5, 0.0 ) ), rk_n( G + vec3( 0.0, 0.0, 0.5 ) ) ) - g0;
    gd -= rkNo * dot( gd, rkNo );
    normal = normalize( normal - rkR * gd * gs );
  }
  if ( rkMudMask > 0.002 ) {
    vec3 Q = P * rkP.y + vSeed * 0.173;
    vec3 tx = texture2D( rkMudNor, Q.zy ).xyz * 2.0 - 1.0;
    vec3 ty = texture2D( rkMudNor, Q.xz ).xyz * 2.0 - 1.0;
    vec3 tz = texture2D( rkMudNor, Q.xy ).xyz * 2.0 - 1.0;
    vec3 N = rkNo;
    tx = vec3( tx.xy * 1.6 + N.zy, abs( tx.z ) * N.x );
    ty = vec3( ty.xy * 1.6 + N.xz, abs( ty.z ) * N.y );
    tz = vec3( tz.xy * 1.6 + N.xy, abs( tz.z ) * N.z );
    vec3 nm = normalize( tx.zyx * rkW.x + ty.xzy * rkW.y + tz.xyz * rkW.z );
    normal = normalize( mix( normal, normalize( rkR * nm ), rkMudMask ) );
  }
}
`;

function patchRockMaterial(mat, uniforms) {
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = function (shader, r) {
    if (prev && prev !== THREE.Material.prototype.onBeforeCompile) prev.call(this, shader, r);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\n' + ROCK_MUD_VERT_HEAD)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n' + ROCK_MUD_VERT);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\n' + ROCK_MUD_FRAG_HEAD)
      .replace('#include <map_fragment>', '#include <map_fragment>\n' + ROCK_MUD_FRAG)
      .replace('#include <roughnessmap_fragment>', ROCK_ROUGH)
      .replace('#include <normal_fragment_maps>', ROCK_NORMAL);
  };
  const prevKey = mat.customProgramCacheKey;
  mat.customProgramCacheKey = function () {
    const b = prevKey && prevKey !== THREE.Material.prototype.customProgramCacheKey ? prevKey.call(this) : '';
    return b + '|rkmud4';
  };
  mat.needsUpdate = true;
  return mat;
}

// Debris-flow surface (REAL-WORLD MODEL, Illgraben / Kedarnath footage): a bouldery, coarse snout followed by a
// finer, more liquid slurry body; everything is saturated, so the whole mass has a wet sheen (roughness 0.3-0.5 on
// clasts and clods, 0.08-0.18 on the liquid slurry film, which mirrors the overcast sky), colour a dark grey-brown
// (linear albedo ~0.06/0.045/0.03, wet silt + clay). The slurry surface moves faster in the middle than at the margins
// (plug flow with shear at the banks): a two-phase flow-map scroll of ripple normals, so speed can vary without the
// texture stretching over time.
const FLOW_VERT_HEAD = /* glsl */`
attribute vec3 aFlow;      // x: blend mask (0 rocky mud .. 1 smooth liquid mud), y: cavity/trough 0..1, z: surface speed factor
varying vec3 vFlow;
`;
const FLOW_FRAG_HEAD = /* glsl */`
varying vec3 vFlow;
uniform sampler2D mudB;
uniform sampler2D mudBn;
uniform vec4 flowP;        // x time, y front speed (m/s), z wetness, w unused
vec3 flowRipple( sampler2D tex, vec2 uv, float spd, float t ) {
  // two-phase flow map: each layer scrolls for one period, cross-faded so the reset never shows
  float per = 1.6;
  float p0 = fract( t / per ), p1 = fract( t / per + 0.5 );
  vec2 dir = vec2( 0.0, -1.0 ) * spd * per * 0.35;
  vec3 a = texture2D( tex, uv + dir * p0 ).xyz * 2.0 - 1.0;
  vec3 b = texture2D( tex, uv + dir * p1 + vec2( 0.37, 0.21 ) ).xyz * 2.0 - 1.0;
  return mix( a, b, abs( 1.0 - 2.0 * p0 ) );
}
`;
const FLOW_MAP = /* glsl */`
#ifdef USE_MAP
  vec4 fa = texture2D( map, vMapUv );
  float la = dot( fa.rgb, vec3( 0.3, 0.59, 0.11 ) );
  fa.rgb = mix( vec3( la ), fa.rgb, 0.85 );                               // grass bits in the scan -> plain mud
  vec4 fb = texture2D( mudB, vMapUv * 0.8 + vec2( 0.31, 0.17 ) );
  vec4 fc = mix( fa, fb * vec4( 0.82, 0.8, 0.8, 1.0 ), vFlow.x );
  fc.rgb *= mix( 1.0, 0.5, vFlow.y );                                   // troughs hold muddy water: darker
  diffuseColor *= fc;
#endif
`;
const FLOW_ROUGH = /* glsl */`
#include <roughnessmap_fragment>
roughnessFactor = mix( clamp( roughnessFactor * 0.7, 0.36, 0.72 ), 0.32, smoothstep( 0.35, 0.9, vFlow.x ) );  // saturated clods .. rippled liquid film
roughnessFactor = mix( roughnessFactor, 0.24, vFlow.y * 0.6 );
`;
const FLOW_NORMAL = /* glsl */`
#include <normal_fragment_maps>
#ifdef USE_NORMALMAP
{
  float t = flowP.x;
  float sp = ( 0.25 + flowP.y * 0.09 ) * vFlow.z;
  vec3 n1 = texture2D( mudBn, vNormalMapUv * 0.8 + vec2( 0.31, 0.17 ) ).xyz * 2.0 - 1.0;
  vec3 n2 = flowRipple( mudBn, vNormalMapUv * 2.2, sp, t );
  vec3 n3 = flowRipple( mudBn, vNormalMapUv * 4.1 + vec2( 0.5, 0.0 ), sp * 1.3, t + 0.8 );
  vec2 nxy = mix( n1.xy * 0.9, n1.xy * 0.3 + ( n2.xy * 0.65 + n3.xy * 0.35 ) * 0.45, vFlow.x );
  vec3 nbv = normalize( tbn * vec3( nxy * normalScale, 1.0 ) );
  normal = normalize( mix( normal, nbv, 0.3 + vFlow.x * 0.6 ) );
  // specular anti-aliasing: sub-pixel ripples widen the lobe instead of sparkling (Kaplanyan 2016, simplified)
  vec3 dn = fwidth( normal );
  roughnessFactor = max( roughnessFactor, clamp( dot( dn, dn ) * 6.0, 0.0, 0.55 ) );
}
#endif
`;

export default class Landslide {
  constructor(ctx) {
    this.ctx = ctx;
    this.frontS = 0;
    this.frontSpeed = 0;
    this.rumble = 0;
    this.rockCount = 0;
    this.front = { active: false, s: 0, speed: 0, height: 0, visS: 0 };
    this.rocks = [];            // active dynamic rocks
    this.debris = [];           // settled static rocks {variant, matrix, collider, mud}
    this.queue = [];            // scheduled actions {t, fn}
    this.t = 0;
    this.rng = mulberry(90210);
    this.escape = null;
    this._rockHandles = new Map(); // collider handle -> rock
    this._pr = {}; this._prCar = {}; this._prPl = {}; this._prCam = {}; this._prEsc = {};
    this._hazardT = -99;
    this._burst = 0;
    this._frontImpactT = 0;
    this.maxRocks = ctx.config?.quality?.maxRocks ?? 60;
    // set-piece dust through the particle pools: Ultra/High keep every puff, Medium/Low thin them out
    this.fxQ = { ultra: 1, high: 1, medium: 0.5, low: 0.3 }[ctx.config?.quality?.key] ?? 1;
    this._groundRows = new Map();
    this._frontDirty = true;
    this._visS = 0;
    this._grow = 1;       // front height factor (intro pile grows in)
    this._growTarget = 1;
    this._flowDisp = 0;
    this._surgeX = 0;
    this._churnT = 0;
  }

  // ============================================================================================ init
  async init() {
    const { ctx } = this;
    this.root = new THREE.Group();
    this.root.name = 'landslide';
    ctx.scene.add(this.root);
    let gltf = null;
    try { gltf = await ctx.assets.gltf('assets/models/rocks.glb'); } catch (e) { console.warn('[landslide] rocks.glb missing, using procedural rocks', e); }
    await this._buildRockAssets(gltf);
    try { this.decals = new SlideDecals(ctx, this._soil, ctx.config?.quality?.key === 'ultra' ? 320 : 220); this.root.add(this.decals.mesh); } catch (e) { console.warn('[landslide] decals failed', e); this.decals = null; }
    try { await this._buildFront(); } catch (e) { console.warn('[landslide] front mesh failed', e); }
    try { await this._buildFrontPassengers(); } catch (e) { console.warn('[landslide] front trees failed', e); }
    // player sensor reports strikes: convert hard hits by our rocks into hazards
    ctx.events?.on?.('player:struck', (p) => this._onPlayerStruck(p));
    this._spawns = ctx.terrain?.scatter?.slideSpawns || null;
    if (!this._spawns) {
      try { const sc = await ctx.assets.json('assets/world/scatter.json'); this._spawns = sc?.slideSpawns || null; } catch { /* fallback below */ }
    }
    // debug helpers
    if (ctx.flags?.debug) window.__slide = this;
  }

  async _buildRockAssets(gltf) {
    const { ctx } = this;
    const meshOf = (name) => { const o = gltf?.scene.getObjectByName(name); let m = null; o?.traverse((c) => { if (!m && c.isMesh) m = c; }); return m; };
    this.variants = [];
    let baseMat = null;
    for (let i = 0; i < 10; i++) {
      const m = meshOf('rock_' + i);
      const hull = meshOf('rock_hull_' + i);
      if (!m) continue;
      baseMat = baseMat || m.material;
      const pts = hull?.geometry?.attributes?.position ? Float32Array.from(hull.geometry.attributes.position.array) : null;
      m.geometry.computeBoundingSphere();
      this.variants.push({ geo: m.geometry, hull: pts, vol: pts ? hullVolume(pts) : 2.2 });
    }
    this.pebbles = [];
    for (let i = 0; i < 6; i++) { const m = meshOf('pebble_' + i); if (m) this.pebbles.push(m.geometry); }
    if (!this.variants.length) {
      // procedural fallback: fractured, displaced icosahedra
      const mat = await this._fallbackRockMaterial();
      baseMat = mat;
      for (let i = 0; i < 6; i++) {
        const g = proceduralRock(i * 7 + 3);
        const pts = Float32Array.from(g.attributes.position.array);
        this.variants.push({ geo: g, hull: pts, vol: 2.4 });
      }
      this.pebbles = [proceduralRock(99, 1)];
    }
    const mat = baseMat.clone();
    if (mat.metalness === 0) mat.metalnessMap = null;   // dielectric: skip the extra ORM fetch
    // clinging forest earth (CC0 mud_forest); flat fallbacks keep the shader valid when it is missing
    let soil = null, coat = null;
    try { soil = await ctx.assets.pbr('mud_forest'); } catch { /* fallback below */ }
    // the coat itself: brown_mud_03 (smeared wet clay with grit; shared with the debris front, so no extra memory).
    // mud_forest's leaf litter read as white speckle on a tumbling block
    try { coat = await ctx.assets.pbr('brown_mud_03'); } catch { coat = soil; }
    const flat = (r, g, b) => { const t = new THREE.DataTexture(new Uint8Array([r, g, b, 255]), 1, 1); t.needsUpdate = true; return t; };
    this._soil = soil;
    this.rockUniforms = {
      rkMudMap: { value: coat?.map || flat(40, 26, 14) },
      rkMudNor: { value: coat?.normalMap || flat(128, 128, 255) },
      rkP: { value: new THREE.Vector4(ctx.env?.wetness ?? 0.75, 1 / 1.1, 0.3, 0) },
      // exposure smear: 45 % of a 1/60 s dashcam exposure, capped at 0.3 m
      rkMB: { value: new THREE.Vector4(0.45 / 60, 0.3, 0, 0) },
    };
    patchRockMaterial(mat, this.rockUniforms);
    this.rockMat = mat;
    const cap = this.maxRocks;
    const mkInst = (geo, n, name, cast = true) => {
      const g = new THREE.BufferGeometry();
      for (const k in geo.attributes) g.setAttribute(k, geo.attributes[k]);
      g.setIndex(geo.index);
      g.boundingSphere = geo.boundingSphere;
      const mud = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
      mud.setUsage(THREE.DynamicDrawUsage);
      g.setAttribute('aMud', mud);
      // relative velocity for the exposure smear (written for flying rocks only; stays 0 elsewhere)
      const vel = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
      vel.setUsage(THREE.DynamicDrawUsage);
      g.setAttribute('aVel', vel);
      const im = new THREE.InstancedMesh(g, mat, n);
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      instReset(im); im.name = name;
      im.castShadow = cast; im.receiveShadow = true;
      im.frustumCulled = true;
      this.root.add(im);
      return im;
    };
    this.liveMeshes = this.variants.map((v, i) => mkInst(v.geo, cap, 'slide_rock_' + i));
    for (const im of this.liveMeshes) im.userData.vel = true;
    // LOD1 per variant (meshopt-simplified to ~30 % of the triangles, same UVs / atlas / baked maps), shared by flying
    // rocks beyond LIVE_FULL and settled debris beyond DEBRIS_FULL. A real block keeps its facets at any distance:
    // the old single pebble proxy turned every distant boulder into the same smooth potato.
    const simp = await loadSimplifier();
    this.loVar = this.variants.map((v, i) => {
      const lg = simp ? lodGeometry(v.geo, simp, 0.3, 0.02) : null;
      return lg ? mkInst(lg, cap + DEBRIS_CAP, 'slide_rock_lod1_' + i) : null;
    });
    this.debrisMeshes = this.variants.map((v, i) => mkInst(v.geo, DEBRIS_CAP, 'slide_debris_' + i));
    // rocks riding on the debris front (3 shapes, conveyor-belt motion)
    this.frontRockMeshes = [0, 3, 6].filter((i) => this.variants[i]).map((i) => mkInst(this.variants[i].geo, 48, 'front_rock_' + i, false));
    this._nPebbles = FRONT_PEBBLES[ctx.config?.quality?.key] ?? 360;
    if (this.pebbles.length) this.frontPebbleMesh = mkInst(this.pebbles[0], this._nPebbles, 'front_pebble', false);
    // distant LOD for every rock category (pebble proxy, ~290 tris)
    const loGeo = this.pebbles[2] || this.pebbles[0] || this.variants[0].geo;
    this.loMesh = mkInst(loGeo, cap + DEBRIS_CAP + 180, 'slide_rock_lo', true);
    this._loR2 = 26 * 26;   // front passengers only (see _updateFrontExtras)
  }

  /** Append an instance to the LOD proxy mesh. */
  _pushLo(m4, mud) {
    const im = this.loMesh;
    if (!im || im.count >= im.instanceMatrix.count) return;
    if (this._recLo) { const i = this._recLoN++; const a = this._recLo; if (a.length < (i + 1) * 17) a.length = (i + 1) * 17; m4.toArray(a, i * 17); a[i * 17 + 16] = mud; }
    instPush(im, m4, mud);
  }

  async _fallbackRockMaterial() {
    const { ctx } = this;
    try {
      const { pbrMaterial } = await import('../render/materials.js');
      return await pbrMaterial(ctx, 'lichen_rock', { triplanar: { scale: 0.5 } });
    } catch {
      return new THREE.MeshStandardMaterial({ color: 0x6a6660, roughness: 0.85 });
    }
  }

  // ============================================================================================ rocks
  /** Spawn a dynamic boulder. Returns the rock record or null. */
  spawnBoulder(pos, vel = null, radius = 0.6, opts = {}) {
    const { ctx } = this;
    const phys = ctx.physics, R = phys?.RAPIER, world = phys?.world;
    if (!world || !R || !this.variants?.length || !pos) return null;
    radius = clamp(radius, 0.08, 3);
    if (this.rocks.length >= this.maxRocks) {
      // recycle: the smallest / farthest-from-camera rock goes first
      const cam = ctx.camera.position;
      let worst = -1, ws = -Infinity;
      for (let i = 0; i < this.rocks.length; i++) {
        const r = this.rocks[i];
        const score = r.pos.distanceTo(cam) / (r.r + 0.3) + (r.still > 0.5 ? 200 : 0) + (r.frag ? 100 : 0) - (r.aimed ? 1e4 : 0);
        if (score > ws) { ws = score; worst = i; }
      }
      if (worst < 0) return null;
      this._retire(this.rocks[worst], false);
    }
    const vi = opts.variant ?? Math.floor(this.rng() * this.variants.length);
    const V = this.variants[vi % this.variants.length];
    const rot = opts.rotation || _q.setFromEuler(_e.set(this.rng() * 6.28, this.rng() * 6.28, this.rng() * 6.28)).clone();
    const bd = R.RigidBodyDesc.dynamic()
      .setTranslation(pos.x, pos.y, pos.z)
      .setRotation({ x: rot.x, y: rot.y, z: rot.z, w: rot.w })
      .setLinearDamping(0.04).setAngularDamping(0.35)
      .setCcdEnabled(true).setCanSleep(true);
    if (vel) bd.setLinvel(vel.x, vel.y, vel.z);
    let av = opts.angVel;
    if (!av) {
      // tumbling: spin about the horizontal axis across the direction of travel (rolling sense), ω ≈ 0.35 v/r
      // (field rockfall: 1-5 rad/s for metre-size blocks, 5-20 rad/s for cobbles), plus a random wobble
      const vx = vel?.x || 0, vz = vel?.z || 0, vh = Math.hypot(vx, vz);
      const w = clamp(Math.hypot(vx, vel?.y || 0, vz) / radius * 0.35, 0.8, 12);
      av = vh > 0.3
        ? { x: vz / vh * w + (this.rng() - 0.5) * 1.5, y: (this.rng() - 0.5) * 1.5, z: -vx / vh * w + (this.rng() - 0.5) * 1.5 }
        : { x: (this.rng() - 0.5) * 4, y: (this.rng() - 0.5) * 2, z: (this.rng() - 0.5) * 4 };
    }
    bd.setAngvel(av);
    if (radius < 0.5) bd.setDominanceGroup(-1);   // the car shoves small stones aside instead of hanging up on them
    const body = world.createRigidBody(bd);
    let cd = null;
    if (V.hull) {
      const pts = new Float32Array(V.hull.length);
      for (let i = 0; i < pts.length; i++) pts[i] = V.hull[i] * radius;
      cd = R.ColliderDesc.convexHull(pts);
    }
    if (!cd) cd = R.ColliderDesc.ball(radius * 0.8);
    cd.setDensity(DENSITY).setFriction(0.9).setRestitution(opts.restitution ?? (0.16 + this.rng() * 0.12))
      .setCollisionGroups(groups(G.ROCK, G.STATIC | G.CAR | G.PLAYER | G.ROCK | G.DEBRIS | (radius >= 0.55 ? 0 : G.PROP))); // big boulders flatten the guardrail and go over
    const collider = world.createCollider(cd, body);
    const rock = {
      body, collider, variant: vi % this.variants.length, r: radius, mass: body.mass() || DENSITY * V.vol * radius ** 3,
      pos: new THREE.Vector3(pos.x, pos.y, pos.z), quat: rot.clone(), prevV: new THREE.Vector3(vel?.x || 0, vel?.y || 0, vel?.z || 0),
      still: 0, cool: 0.1, age: 0, mud: opts.mud ?? (0.3 + this.rng() * 0.6), lethal: opts.lethal !== false, hits: 0,
      tag: opts.tag || null, scaleY: 1,
    };
    rock.mudP = packMud(rock.mud, this.rng());   // instance attribute value: coat amount + per-rock seed
    this.rocks.push(rock);
    this._rockHandles.set(collider.handle, rock);
    this.rockCount = this.rocks.length;
    return rock;
  }

  _retire(rock, toDebris) {
    const world = this.ctx.physics?.world;
    const i = this.rocks.indexOf(rock);
    if (i >= 0) this.rocks.splice(i, 1);
    this._rockHandles.delete(rock.collider?.handle);
    if (toDebris) this._addDebris(rock);
    try { if (world && rock.body) world.removeRigidBody(rock.body); } catch { /* already gone */ }
    rock.body = null;
    this.rockCount = this.rocks.length;
  }

  _addDebris(rock) {
    const { ctx } = this;
    const phys = ctx.physics, R = phys?.RAPIER;
    if (this.debris.length >= DEBRIS_CAP) {
      const old = this.debris.shift();
      if (old.collider) try { phys.world.removeCollider(old.collider, false); } catch { /* */ }
    }
    const m = new THREE.Matrix4().compose(rock.pos, rock.quat, _s.set(rock.r, rock.r, rock.r));
    let collider = null;
    const pr = ctx.road?.project(rock.pos, this._pr);
    if (R && phys.world && rock.r >= 0.35 && pr && Math.abs(pr.d) < 12 && !(rock.r < 0.45 && Math.abs(pr.d) < 3.2 && pr.dy < 1.2)) {
      const V = this.variants[rock.variant];
      let cd = null;
      if (V.hull) {
        const pts = new Float32Array(V.hull.length);
        for (let k = 0; k < pts.length; k++) pts[k] = V.hull[k] * rock.r;
        cd = R.ColliderDesc.convexHull(pts);
      }
      if (!cd) cd = R.ColliderDesc.ball(rock.r * 0.8);
      cd.setTranslation(rock.pos.x, rock.pos.y, rock.pos.z)
        .setRotation({ x: rock.quat.x, y: rock.quat.y, z: rock.quat.z, w: rock.quat.w })
        .setFriction(0.9).setCollisionGroups(groups(G.ROCK, G.ALL & ~G.SENSOR));
      collider = phys.world.createCollider(cd);
    }
    this.debris.push({ variant: rock.variant, matrix: m, collider, mud: rock.mud, mudP: rock.mudP, pos: rock.pos.clone(), r: rock.r });
    this._debrisDirty = true;
  }

  _removeDebrisAt(idx) {
    const d = this.debris[idx];
    if (d?.collider) try { this.ctx.physics.world.removeCollider(d.collider, false); } catch { /* */ }
    this.debris.splice(idx, 1);
    this._debrisDirty = true;
  }

  // ============================================================================================ scripted events
  _spawnSet(kind, sMin, sMax) {
    const sp = (this._spawns || []).filter((g) => g.kind === kind || g.id === kind);
    const pts = [];
    for (const g of sp) for (const p of g.points) {
      _v.set(p[0], p[1], p[2]);
      const pr = this.ctx.road.project(_v, this._pr);
      if (pr.s >= sMin && pr.s <= sMax) pts.push({ p: new THREE.Vector3(p[0], p[1], p[2]), s: pr.s, d: pr.d, dy: pr.dy, dir: new THREE.Vector3(...g.dir) });
    }
    return pts;
  }

  /** Fallback release points: 20–60 m up the slope via physics raycasts. */
  _fallbackSpawns(sMin, sMax, n) {
    const { ctx } = this;
    const out = [];
    for (let i = 0; i < n; i++) {
      const s = lerp(sMin, sMax, (i + 0.5) / n);
      const d = 16 + this.rng() * 26;
      const p = ctx.road.worldAt(s, d, new THREE.Vector3());
      const gy = ctx.physics?.groundHeight?.(p.x, p.z) ?? null;
      p.y = gy != null ? gy + 0.5 : p.y + d * 0.9;
      const l = ctx.road.leftAt(s, _l);
      out.push({ p, s, d, dy: p.y - ctx.road.pointAt(s, _v).y, dir: new THREE.Vector3(-l.x, -0.6, -l.z).normalize() });
    }
    return out;
  }

  _at(delay, fn) { this.queue.push({ t: this.t + delay, fn }); }

  triggerIntroRockfall() {
    const { ctx } = this;
    if (!ctx.road) return;
    const s0 = 132, s1 = 158;
    let pts = this._spawnSet('scar', s0 - 4, s1 + 4);
    if (pts.length < 4) pts = this._fallbackSpawns(s0, s1, 10);
    pts.sort((a, b) => a.dy - b.dy);
    const low = pts.slice(0, Math.max(3, Math.ceil(pts.length * 0.6)));
    const rnd = this.rng;
    const pick = (arr) => arr[Math.floor(rnd() * arr.length)];
    this._burst = 1;
    ctx.particles?.dustBoost?.(7);
    ctx.cameraRig?.shake?.(0.35);
    // initial crack: slab release + dust at the crown
    for (let i = 0; i < 4; i++) { const a = pick(pts); ctx.particles?.dust?.(a.p.clone(), 3.5, 1.2); }
    // the big ones
    const big = [1.35, 1.1, 0.95, 1.2, 0.85, 1.5, 0.75];
    big.forEach((r, i) => this._at(0.15 + i * 0.28 + rnd() * 0.2, () => {
      const a = pick(low);
      const p = a.p.clone(); p.y += r + 0.4;
      // aim: mostly across the road, landing between s0..s1
      const tgtS = lerp(s0 + 2, s1 - 2, (i + rnd()) / big.length);
      const tgt = ctx.road.worldAt(tgtS, lerp(-2.5, 2.5, rnd()), _w);
      const T = 1.9 + rnd() * 0.6;
      const vel = ballistic(p, tgt, T, new THREE.Vector3());
      vel.multiplyScalar(0.92);
      const l = ctx.road.leftAt(tgtS, _l);
      this.spawnBoulder(p, vel, r, { mud: 0.65, angVel: { x: -l.z * 3 + (rnd() - 0.5), y: rnd() - 0.5, z: l.x * 3 + (rnd() - 0.5) } });
      ctx.particles?.dust?.(p.clone(), 2.2 + r, 0.8);
    }));
    // the spray of smaller stuff
    for (let i = 0; i < 34; i++) this._at(0.3 + rnd() * 3.2, () => {
      const a = pick(pts);
      const r = 0.12 + Math.pow(rnd(), 2.2) * 0.55;
      const p = a.p.clone(); p.y += r + 0.2;
      const tgtS = lerp(s0 - 4, s1 + 4, rnd());
      const tgt = ctx.road.worldAt(tgtS, lerp(-4, 3.5, rnd()), _w);
      const vel = ballistic(p, tgt, 1.5 + rnd() * 1.2, new THREE.Vector3());
      this.spawnBoulder(p, vel, r, { mud: 0.4 + rnd() * 0.5 });
    });
    // the mud/rock pile that seals the road (rides the same mesh as the front)
    this._at(1.2, () => {
      if (!(this.frontS > 0)) {
        this._grow = 0.02; this._growTarget = 1;
        this.setFront(150, true);
        this.setFrontSpeed(0);
        this._introPile = true;
      }
      ctx.cameraRig?.shake?.(0.5);
      this._burst = 1;
    });
    for (let k = 0; k < 6; k++) this._at(1.4 + k * 0.45, () => {
      const p = ctx.road.worldAt(lerp(138, 156, rnd()), lerp(-3, 4, rnd()), new THREE.Vector3());
      p.y += 1.5;
      ctx.particles?.dust?.(p, 4 + rnd() * 2, 1.5);
      ctx.particles?.debris?.(p, 10);
    });
    // blocks toppling off the crest of the cut face: in full view of the look-back (the release points above are
    // hidden behind the crest). They fall ~10 m into the ditch / onto the road.
    for (let k = 0; k < 5; k++) this._at(0.25 + k * 0.4 + rnd() * 0.2, () => {
      const s = lerp(136, 162, (k + rnd()) / 5);
      const top = this._crestPoint(s);
      if (!top) return;
      const r = 0.6 + rnd() * 0.7;
      const l = ctx.road.leftAt(s, _l);
      const p = top.clone(); p.y += r + 0.2;
      p.addScaledVector(l, -r * 0.6);
      const vel = new THREE.Vector3(-l.x * (2.5 + rnd() * 2.5), 1 + rnd() * 1.5, -l.z * (2.5 + rnd() * 2.5));
      this.spawnBoulder(p, vel, r, { mud: 0.5 + rnd() * 0.4, lethal: false });
      this._impactCloud(top, 1.2, 2.5, top.y - 1);
    });
    // a few bounce on down the road toward the camera (non-lethal: the car is 40 m on by then)
    for (let k = 0; k < 3; k++) this._at(0.6 + k * 0.5, () => {
      const a = pick(low);
      const r = 0.45 + rnd() * 0.35;
      const p = a.p.clone(); p.y += r + 0.4;
      const tgt = ctx.road.worldAt(lerp(158, 168, rnd()), lerp(-2.5, 2, rnd()), _w);
      const vel = ballistic(p, tgt, 1.9 + rnd() * 0.4, new THREE.Vector3());
      this.spawnBoulder(p, vel, r, { mud: 0.6, restitution: 0.3, lethal: false });
    });
    // the collapse cloud: pulverised rock and soil boiling up from the scar and over the crest, then drifting
    for (let k = 0; k < 16; k++) this._at(0.15 + k * 0.16 + rnd() * 0.2, () => {
      const s = lerp(128, 168, rnd()), d = k < 6 ? lerp(5, 10, rnd()) : lerp(8, 34, rnd());
      const p = ctx.road.worldAt(s, d, new THREE.Vector3());
      const gy = this._groundY(p.x, p.z, p.y + 90) ?? p.y + d * 0.9;
      p.y = gy + 1.5 + rnd() * 3;
      const l = ctx.road.leftAt(s, _l);
      _v.set(-l.x * (1 + rnd() * 3), 0.8 + rnd() * 1.6, -l.z * (1 + rnd() * 3));
      // brownish grey, lighter where it billows higher over the scar (sky-lit), darker in the churn low down
      const hf = clamp((p.y - gy - 1.5) / 3, 0, 1);
      palMix(_col, PAL.slideBase, PAL.slideTop, 0.45 + hf * 0.4 + rnd() * 0.15, 0.07);
      if (!this._puff(p, _v, 8 + rnd() * 5, 4 + rnd() * 3, 13 + rnd() * 9, 0.28 + rnd() * 0.18, _col, gy, 0.2, 0.6, 1.2, 0.1 + (1 - hf) * 0.25)) ctx.particles?.dust?.(p, 8, 3);
    });
    // ground surge as the mud tongue crosses the road
    for (let k = 0; k < 7; k++) this._at(1.3 + k * 0.3, () => {
      const s = lerp(138, 162, rnd());
      const p = ctx.road.worldAt(s, lerp(-2, 5, rnd()), new THREE.Vector3());
      const gy = p.y; p.y += 1 + rnd() * 1.5;
      const l = ctx.road.leftAt(s, _l), tg = ctx.road.tangentAt(s, _u);
      const side = (rnd() - 0.5) * 4;
      _v.set(-l.x * (2 + rnd() * 3) + tg.x * side, 0.4 + rnd() * 0.6, -l.z * (2 + rnd() * 3) + tg.z * side);
      // the mud tongue's surge: mud spray + water, dense and dark at the base of the cloud
      palMix(_col, PAL.slideBase, PAL.slideTop, rnd() * 0.25, 0.07);
      this._puff(p, _v, 7 + rnd() * 4, 3 + rnd() * 2, 11 + rnd() * 7, 0.34 + rnd() * 0.15, _col, gy, 0.03, 1.2, 1.5, 0.6);
    });
  }

  /** Top of the cut face (crest lip) at s: the first ground point uphill of the ditch that is ≥ 6 m above the road. */
  _crestPoint(s) {
    const { ctx } = this;
    const c = ctx.road.pointAt(s, _gp);
    for (let d = 5.5; d <= 14; d += 0.75) {
      const p = ctx.road.worldAt(s, d, new THREE.Vector3());
      const y = this._groundY(p.x, p.z, c.y + 80);
      if (y != null && y > c.y + 6) { p.y = y; return p; }
    }
    return null;
  }

  triggerGully(i, opts = {}) {
    const { ctx } = this;
    if (!ctx.road) return;
    const gs = (this._spawns || []).filter((g) => g.kind === 'gully');
    const g = gs[i];
    const gullyS = g?.s ?? ctx.road.markers?.gullies?.[i];
    if (gullyS == null) return;
    let pts = g ? g.points.map((p) => ({ p: new THREE.Vector3(p[0], p[1], p[2]), dir: new THREE.Vector3(...g.dir) })) : this._fallbackSpawns(gullyS - 4, gullyS + 4, 6);
    const count = opts.count ?? 7, dur = opts.duration ?? 3;
    const rMin = opts.rMin ?? 0.2, rMax = opts.rMax ?? 0.9, speed = opts.speed ?? 6;
    const rnd = this.rng;
    for (let k = 0; k < count; k++) this._at(rnd() * dur, () => {
      const a = pts[Math.floor(rnd() * pts.length)];
      const r = lerp(rMin, rMax, Math.pow(rnd(), 1.6));
      const p = a.p.clone(); p.y += r + 0.3;
      const vel = a.dir.clone().multiplyScalar(speed * (0.7 + rnd() * 0.6));
      vel.x += (rnd() - 0.5) * 1.5; vel.z += (rnd() - 0.5) * 1.5;
      this.spawnBoulder(p, vel, r, { mud: 0.2 + rnd() * 0.4, tag: opts.tag || 'gully' });
      if (rnd() < 0.5) ctx.particles?.dust?.(p.clone(), 1.5 + r * 2, 0.5);
    });
    if (opts.aim) for (let k = 0; k < 2; k++) this._at(0.2 + k * 0.9, () => this._aimedRock(gullyS + (rnd() - 0.5) * 6, k === 0 ? 0.6 : 1.1, 1.2 + rnd() * 0.3, pts));
  }

  /**
   * Ballistic rock that lands on the road at s = tgtS at time T (from spawn). Release point from `pts` (gully) or from
   * the slope above the road. Returns the rock.
   */
  _aimedRock(tgtS, r, T, pts = null) {
    const { ctx } = this;
    const rnd = this.rng;
    let p;
    if (pts?.length) p = pts[Math.floor(rnd() * Math.min(pts.length, 9))].p.clone();
    else {
      const d = 13 + rnd() * 14;
      const sOff = (rnd() - 0.5) * 10;
      p = ctx.road.worldAt(tgtS + sOff, d, new THREE.Vector3());
      const gy = this._groundY(p.x, p.z, p.y + 80);
      p.y = (gy ?? p.y + d * 1.1) + 1.0;
    }
    p.y += r + 0.5;
    const tgt = ctx.road.worldAt(tgtS, lerp(-2.6, 1.8, rnd()), _w);
    tgt.y += r * 0.8;
    const vel = ballistic(p, tgt, T, new THREE.Vector3());
    const l = ctx.road.leftAt(tgtS, _l);
    const rock = this.spawnBoulder(p, vel, r, { mud: 0.3 + rnd() * 0.3, angVel: { x: -l.z * 4, y: (rnd() - 0.5) * 2, z: l.x * 4 } });
    if (rock) { rock.aimed = true; ctx.particles?.dust?.(p.clone(), 1.5 + r, 0.4); }
    return rock;
  }

  startEscape() {
    this.escape = { t: 0, next: 1.0, gullyDone: new Set(), gullyShow: new Set(), lastKind: null };
  }
  stopEscape() { this.escape = null; }

  /**
   * FAIR rockfall aimed relative to the car. The landing point is bounded by the car's *reachable interval* over the
   * flight time T (it can at most accelerate at A_MAX or brake at B_MAX), so:
   *   'ahead'  lands beyond anything the car can reach by then (plus a margin) and rolls on across the road toward the
   *            valley: slowing down only gives it more time to clear;
   *   'behind' lands short of where a hard-braking car would be (a stopped car is never under it): it crashes down in
   *            the mirrors / behind the chase camera.
   * Returns the rock (or null).
   */
  _fairRock(kind, carS, v, opts = {}) {
    const { ctx } = this;
    const rnd = this.rng;
    const A_MAX = 4.0, B_MAX = 10, CAR_HALF = 2.3; // B_MAX > tyre braking: also covers a car stopped short by a collision
    const T = opts.T ?? (1.55 + rnd() * 0.7);
    const r = opts.r ?? 0.5;
    let tgtS;
    if (kind === 'ahead') tgtS = carS + v * (T + 0.35) + 0.5 * A_MAX * T * T + CAR_HALF + r + (opts.margin ?? (6 + rnd() * 14));
    // (QA) from ~60 m before the portal (ahead targets lie 45-60 m out): an ahead rock would land in the portal throat or be aimed inside the
    // tunnel (its arc then hits the spur and it tumbles into the mouth, wedging the car against the headwall). Those
    // become 'behind' rocks (crashing down in the mirror), so the run-in to the tunnel stays open.
    if (kind === 'ahead' && tgtS > (ctx.road.markers?.tunnel ?? 1150) - 8) kind = 'behind';
    if (kind !== 'ahead') tgtS = carS + Math.max(0, v * T - 0.5 * B_MAX * T * T) - CAR_HALF - r - (opts.margin ?? (1.5 + rnd() * 6));
    // release: a gully release point or the slope above the road. The arc must be clear of the terrain (a rock that
    // clips the cut-face crest arrives late and breaks the timing): try a few release points, lofting higher each time.
    // Along-road drift never carries the rock toward the car: ahead rocks are released short of the landing point
    // (drifting +s, away), behind rocks beyond it (drifting -s, away).
    const dLand = opts.d ?? (kind === 'ahead' ? lerp(-0.3, 1.8, rnd()) : lerp(-2.4, 2.4, rnd())); // mid-road: it rolls on across, clear of the cut face
    const tgt = ctx.road.worldAt(tgtS, dLand, new THREE.Vector3());
    tgt.y += r * 0.8;
    let okPts = null;
    if (opts.pts?.length) {
      okPts = opts.pts.filter((q) => {
        if (q.s === undefined) q.s = ctx.road.project(q.p, { _hint: -1e6 }).s;
        return kind === 'ahead' ? q.s <= tgtS - 1 : q.s >= tgtS + 1;
      });
    }
    let p = null, vel = new THREE.Vector3(), clear = false;
    for (let k = 0; k < 7 && !clear; k++) {
      if (okPts?.length && k < 4) {
        p = okPts[Math.floor(rnd() * okPts.length)].p.clone();
        p.y += r + 0.5 + k * 1.5;
      } else {
        const d = 12 + rnd() * 12;
        const sOff = kind === 'ahead' ? -(1 + rnd() * 5) : 1 + rnd() * 5;
        p = ctx.road.worldAt(tgtS + sOff, d, new THREE.Vector3());
        const gy = this._groundY(p.x, p.z, p.y + 80);
        p.y = (gy ?? p.y + d * 1.1) + 1.5 + r + (k % 4) * 2.2;
      }
      ballistic(p, tgt, T, vel);
      clear = this._arcClear(p, vel, T, r);
    }
    const st = this.fairStats || (this.fairStats = { n: 0, clear: 0 }); st.n++; if (clear) st.clear++;
    if (!clear && opts.lethal === undefined) opts = { ...opts, lethal: false }; // cannot guarantee its timing: harmless
    const l = ctx.road.leftAt(tgtS, _l);
    const rock = this.spawnBoulder(p, vel, r, { mud: 0.25 + rnd() * 0.4, angVel: { x: -l.z * 4, y: (rnd() - 0.5) * 2, z: l.x * 4 }, tag: 'fair', restitution: kind === 'ahead' ? 0.3 + rnd() * 0.12 : undefined });
    if (rock) {
      rock.aimed = true; rock.kind = kind; rock.lethal = opts.lethal ?? r >= 0.3;
      rock.plan = { carS: +carS.toFixed(1), v: +v.toFixed(1), tgtS: +tgtS.toFixed(1), T: +T.toFixed(2), t0: this.t, pts: !!opts.pts, clear };
      if (rnd() < 0.6) ctx.particles?.dust?.(p.clone(), 1.2 + r * 1.5, 0.35);
      // a spray of small stuff in the same safe band (never lethal)
      const n = opts.spray ?? 0;
      for (let k = 0; k < n; k++) this._at(rnd() * 0.5, () => {
        const rr = 0.08 + rnd() * 0.2;
        const q = p.clone().add(_u.set((rnd() - 0.5) * 6, rnd() * 2, (rnd() - 0.5) * 6));
        const ds = kind === 'ahead' ? rnd() * 10 : -rnd() * 10;
        const t2 = ctx.road.worldAt(tgtS + ds, lerp(-2.5, 3, rnd()), new THREE.Vector3());
        const sp = this.spawnBoulder(q, ballistic(q, t2, T * (0.85 + rnd() * 0.3), new THREE.Vector3()), rr, { mud: 0.3 + rnd() * 0.4 });
        if (sp) sp.lethal = false;
      });
    }
    return rock;
  }

  /** Is the ballistic arc from p (velocity vel, radius r) clear of static geometry until just before it lands? */
  _arcClear(p, vel, T, r) {
    const phys = this.ctx.physics;
    if (!phys?.world) return true;
    const N = 8, tEnd = T * 0.93;
    const a = _gp.copy(p), b = _gl, dir = _cv;
    const opts = { groups: groups(G.ALL, G.STATIC), excludeCollider: this.wallCollider };
    for (let k = 1; k <= N; k++) {
      const t = tEnd * k / N;
      b.set(p.x + vel.x * t, p.y + vel.y * t - 0.5 * GRAV * t * t, p.z + vel.z * t);
      dir.subVectors(b, a);
      const len = dir.length();
      if (len > 1e-3) {
        dir.multiplyScalar(1 / len);
        if (phys.raycast(a, dir, len, opts)) return false;
        _u.set(a.x, a.y - r * 0.75, a.z);                 // the underside of the rock
        if (k > 1 && phys.raycast(_u, dir, len, opts)) return false;
      }
      a.copy(b);
    }
    return true;
  }

  _gullyPts(i) {
    const g = (this._spawns || []).filter((q) => q.kind === 'gully')[i];
    return g ? g.points.map((p) => ({ p: new THREE.Vector3(p[0], p[1], p[2]), dir: new THREE.Vector3(...g.dir) })) : null;
  }

  _updateEscape(dt) {
    const E = this.escape, { ctx } = this;
    if (!E || !ctx.road) return;
    E.t += dt;
    E.next -= dt;
    const car = ctx.car;
    const driving = !!car?.body && ctx.control !== 'foot';
    const ref = driving ? _p.copy(car.body.translation()) : (ctx.player?.enabled ? _p.copy(ctx.player.feet || ctx.camera.position) : _p.copy(ctx.camera.position));
    const pr = ctx.road.project(ref, this._prEsc);
    const carS = pr.s;
    const v = driving ? Math.max(0, car.speed ?? 0) : 0;
    if (carS > (ctx.road.markers?.tunnel ?? 1150) - 15) return; // safe in the tunnel mouth
    const rnd = this.rng;
    const inten = clamp(E.t / 50, 0, 1);
    // gullies: the chute lets go well before the car gets there (the rocks need 6-12 s to come down), so the torrent
    // is crossing the road as the car approaches; a timed set piece (fair rocks from the chute) as it passes
    const gullies = ctx.road.markers?.gullies || [];
    gullies.forEach((gs, i) => {
      const dist = gs - carS;
      if (!E.gullyDone.has(i) && dist > 0 && dist < clamp(Math.max(v, 8) * 16, 90, 240)) {
        E.gullyDone.add(i);
        this.triggerGully(i, { count: 10, duration: 1.8, rMin: 0.18, rMax: 0.85, speed: 7, tag: 'gully' });
      }
      if (!E.gullyShow.has(i) && dist > 0 && dist < Math.max(v, 6) * 3.2 + 12) {
        E.gullyShow.add(i);
        const pts = this._gullyPts(i);
        const T = 2.3 + rnd() * 0.3;
        this._fairRock('ahead', carS, v, { r: 0.9 + rnd() * 0.45, T: T - 0.4, margin: 5 + rnd() * 6, spray: 5 }); // chute releases bounce unpredictably: only 'behind' rocks use them
        this._at(0.35, () => { const c = this._escRef(); if (c) this._fairRock('ahead', c.s, c.v, { r: 0.45 + rnd() * 0.35, T: T - 0.5, margin: 14 + rnd() * 6, spray: 3 }); });
        this._at(1.1, () => { const c = this._escRef(); if (c) this._fairRock('behind', c.s, c.v, { r: 0.8 + rnd() * 0.5, T: 2.0, pts, margin: 2 + rnd() * 3, spray: 4 }); });
        E.next = Math.max(E.next, 2.2);
      }
    });
    if (E.next <= 0) {
      E.next = lerp(2.5, 1.05, inten) * (0.7 + rnd() * 0.6);
      // alternate-ish between ahead (crossing the road in front: steer / lift) and behind (crashing down in the mirror)
      let kind = rnd() < 0.58 ? 'ahead' : 'behind';
      if (kind === E.lastKind && rnd() < 0.4) kind = kind === 'ahead' ? 'behind' : 'ahead';
      E.lastKind = kind;
      const big = rnd() < 0.36 + inten * 0.15;
      const r = big ? 0.85 + rnd() * 0.5 : 0.3 + rnd() * 0.45;
      this._fairRock(kind, carS, v, { r, spray: 1 + Math.floor(rnd() * 3 * (0.5 + inten)) });
      // now and then a double: a second one on the other side of the car
      if (inten > 0.35 && rnd() < 0.3) this._at(0.25, () => { const c = this._escRef(); if (c) this._fairRock(kind === 'ahead' ? 'behind' : 'ahead', c.s, c.v, { r: 0.35 + rnd() * 0.5, spray: 1 }); });
    }
  }

  /** Current escape reference (car or walker) s and forward speed. */
  _escRef() {
    const { ctx } = this;
    if (!ctx.road) return null;
    const car = ctx.car;
    const driving = !!car?.body && ctx.control !== 'foot';
    const ref = driving ? _p.copy(car.body.translation()) : (ctx.player?.feet ? _p.copy(ctx.player.feet) : _p.copy(ctx.camera.position));
    const pr = ctx.road.project(ref, this._prEsc);
    return { s: pr.s, v: driving ? Math.max(0, car.speed ?? 0) : 0 };
  }

  // ============================================================================================ front
  setFront(s, snap = false) {
    const was = this.frontS > 0;
    this.frontS = Math.max(0, +s || 0);
    this.front.active = this.frontS > 0;
    if (!was || snap || Math.abs(this.frontS - this._visS) > 25) this._visS = this.frontS;
    if (this.front.active && this._introPile && !snap && s > 151) this._introPile = false;
    this._frontDirty = true;
  }
  setFrontSpeed(v) { this.frontSpeed = Math.max(0, +v || 0); }

  async _buildFront() {
    const { ctx } = this;
    // rows: dense at the snout, sparse far behind
    const us = [];
    let u = FRONT_SNOUT, step = 0.28;
    while (u > -FRONT_BACK) { us.push(u); u -= step; step = Math.min(2.2, step * 1.045); }
    us.push(-FRONT_BACK);
    this._us = Float32Array.from(us);
    const NS = us.length, ND = FRONT_ND;
    this._ds = new Float32Array(ND);
    for (let i = 0; i < ND; i++) {
      const t = i / (ND - 1);
      this._ds[i] = lerp(FRONT_D0, FRONT_D1, t);
    }
    const n = NS * ND;
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), uv = new Float32Array(n * 2), flow = new Float32Array(n * 3);
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aFlow', new THREE.BufferAttribute(flow, 3).setUsage(THREE.DynamicDrawUsage));
    const idx = [];
    for (let j = 0; j < NS - 1; j++) for (let i = 0; i < ND - 1; i++) {
      const a = j * ND + i, b = a + 1, c = a + ND, d = c + 1;
      idx.push(a, b, c, b, d, c);
    }
    g.setIndex(idx);
    this._buildTopo();
    this._heights = new Float32Array(n);
    this._grounds = new Float32Array(n);
    let mat;
    try {
      const A = await ctx.assets.pbr('brown_mud_rocks_01');
      let B = null;
      try { B = await ctx.assets.pbr('brown_mud_03'); } catch { try { B = await ctx.assets.pbr('brown_mud_02'); } catch { /* optional */ } }
      mat = new THREE.MeshStandardMaterial({
        map: A.map, normalMap: A.normalMap, roughnessMap: A.armMap, aoMap: A.armMap,
        metalness: 0, roughness: 1, color: new THREE.Color(0.56, 0.4, 0.27), normalScale: new THREE.Vector2(2.0, 2.0),
        envMapIntensity: 0.62,
      });
      const uni = { mudB: { value: B?.map || A.map }, mudBn: { value: B?.normalMap || A.normalMap }, flowP: { value: new THREE.Vector4() } };
      this._flowUniforms = uni;
      mat.onBeforeCompile = (shader) => {
        Object.assign(shader.uniforms, uni);
        shader.vertexShader = shader.vertexShader
          .replace('#include <common>', '#include <common>\n' + FLOW_VERT_HEAD)
          .replace('#include <begin_vertex>', '#include <begin_vertex>\nvFlow = aFlow;');
        shader.fragmentShader = shader.fragmentShader
          .replace('#include <common>', '#include <common>\n' + FLOW_FRAG_HEAD)
          .replace('#include <map_fragment>', FLOW_MAP)
          .replace('#include <roughnessmap_fragment>', FLOW_ROUGH)
          .replace('#include <normal_fragment_maps>', FLOW_NORMAL);
      };
      mat.customProgramCacheKey = () => 'slideflow1';
    } catch (e) {
      console.warn('[landslide] mud textures missing', e);
      mat = new THREE.MeshStandardMaterial({ color: 0x3a2c20, roughness: 0.4 });
    }
    this.frontMat = mat;
    const mesh = new THREE.Mesh(g, mat);
    mesh.name = 'debris_front';
    mesh.castShadow = true; mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    mesh.visible = false;
    this.root.add(mesh);
    this.frontMesh = mesh;
    this.frontGeo = g;

    // front wall collider (kinematic, member STATIC so the KCC and the car are blocked)
    const phys = ctx.physics, R = phys?.RAPIER;
    if (R && phys.world) {
      const body = phys.world.createRigidBody(R.RigidBodyDesc.kinematicPositionBased().setTranslation(0, -500, 0));
      const cd = R.ColliderDesc.cuboid(9.5, 2.8, 1.2).setTranslation(0, 2.2, 0).setFriction(0.9)
        .setCollisionGroups(groups(G.STATIC, G.CAR | G.PLAYER | G.PROP | G.DEBRIS));
      this.wallBody = body;
      this.wallCollider = phys.world.createCollider(cd, body);
    }

    // embedded rocks: conveyor state
    this._frontRocks = [];
    const rnd = mulberry(4242);
    const nBig = this.frontRockMeshes.length * 42;
    for (let k = 0; k < nBig; k++) this._frontRocks.push(this._newFrontRock(rnd, k % this.frontRockMeshes.length, false, true));
    this._frontPebbles = [];
    if (this.frontPebbleMesh) for (let k = 0; k < this._nPebbles; k++) this._frontPebbles.push(this._newFrontRock(rnd, 0, true, true));
    this._frontRnd = rnd;
  }

  _newFrontRock(rnd, mesh, pebble, init) {
    const snout = rnd() < (pebble ? 0.45 : 0.55);
    const r = pebble ? 0.07 + Math.pow(rnd(), 1.5) * 0.3 : (snout ? 0.4 : 0.3) + Math.pow(rnd(), 1.6) * 1.0;
    // a boulder-rich snout (bouldery debris-flow front) + rocks spread over the body, denser near the front
    let u = init ? -Math.pow(rnd(), 2.2) * (FRONT_BACK - 10) + FRONT_SNOUT * 0.4 : -rnd() * 30 - 2;
    if (snout) u = lerp(-3, 5.5, rnd());
    const mud = snout ? 0.62 + rnd() * 0.35 : 0.7 + rnd() * 0.28;
    return {
      mud, mudP: packMud(mud, rnd()),
      mesh, pebble, r, u, d: lerp(FRONT_D0 + 2.5, FRONT_D1 - 1.5, rnd()),
      axis: new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize(),
      q: new THREE.Quaternion().setFromEuler(new THREE.Euler(rnd() * 6, rnd() * 6, rnd() * 6)),
      sink: snout ? 0.1 + rnd() * 0.3 : 0.2 + rnd() * 0.45, spin: (0.5 + rnd()) * (rnd() < 0.5 ? -1 : 1),
    };
  }

  async _buildFrontPassengers() {
    // What a debris flow carries (REAL-WORLD MODEL): spruce trunks snapped off or ripped out with their root plates
    // (root wad 2-3.5 m across, soil still packed between the roots), stripped of most branches by the tumbling, with
    // splintered pale fresh-wood ends; a few whole green trees near the snout; loose broken branches.
    // Logs are procedural (buildSnappedLog); green trees and branches come from trees.glb (conifer_k_lod1, deadwood_k).
    const { ctx } = this;
    let gltf = null;
    try { gltf = await ctx.assets.gltf('assets/models/trees.glb'); } catch { /* logs only */ }
    const take = (name) => { const o = gltf?.scene.getObjectByName(name); if (!o) return null; const c = o.clone(true); c.traverse((m) => { if (m.isMesh) { m.castShadow = /conifer/.test(name); m.receiveShadow = true; } }); return c; };
    const rnd = mulberry(777);
    this._frontTrees = [];
    const wet = await import('../render/materials.js').catch(() => null);
    const matCache = new Map();
    const wetClone = (o) => o.traverse((m) => {
      if (!m.isMesh) return;
      const mats = Array.isArray(m.material) ? m.material : [m.material];
      const out = mats.map((mm) => {
        if (!matCache.has(mm)) {
          const c = mm.clone();
          if (/foliage/.test(mm.name)) { c.alphaTest = Math.max(c.alphaTest || 0, 0.45); c.transparent = false; c.side = THREE.DoubleSide; c.color?.multiplyScalar(0.62); }
          else c.color?.multiplyScalar(0.6);
          try { wet?.applyWetness?.(c, { porosity: 0.5 }, ctx); } catch { /* */ }
          matCache.set(mm, c);
        }
        return matCache.get(mm);
      });
      m.material = Array.isArray(m.material) ? out : out[0];
    });
    // log materials: wet spruce bark, fresh splintered wood, the soil packed in the root plate
    let logMats = null;
    try {
      const [bark, wood] = await Promise.all([ctx.assets.pbr('pine_bark'), ctx.assets.pbr('rough_wood').catch(() => null)]);
      const barkM = new THREE.MeshStandardMaterial({ map: bark.map, normalMap: bark.normalMap, roughnessMap: bark.armMap, aoMap: bark.armMap, color: new THREE.Color(0.5, 0.45, 0.42), roughness: 0.85, normalScale: new THREE.Vector2(1.4, 1.4) });
      // fresh spruce wood: dry albedo ~0.45 cream, soaked + smeared ~0.2
      const woodM = new THREE.MeshStandardMaterial({ map: wood?.map || null, normalMap: wood?.normalMap || null, color: wood ? new THREE.Color(0.95, 0.72, 0.45) : new THREE.Color(0.24, 0.18, 0.1), roughness: 0.62 });
      const soilM = new THREE.MeshStandardMaterial({ map: this._soil?.map || null, normalMap: this._soil?.normalMap || null, color: this._soil ? new THREE.Color(1.25, 1.12, 1.0) : new THREE.Color(0.05, 0.035, 0.02), roughness: 0.6, normalScale: new THREE.Vector2(1.6, 1.6) });
      for (const m of [barkM, soilM]) { try { wet?.applyWetness?.(m, { porosity: 0.55 }, ctx); } catch { /* */ } }
      logMats = [barkM, woodM, soilM];
    } catch (e) { console.warn('[landslide] log textures missing', e); }
    const items = [];
    for (const n of ['conifer_1_lod1', 'conifer_3_lod1']) { const o = take(n); if (o) { wetClone(o); items.push({ o, kind: 'tree' }); } }
    if (logMats) for (let v = 0; v < 5; v++) {
      const mesh = new THREE.Mesh(buildSnappedLog(mulberry(31 + v * 17), v % 3), logMats);
      mesh.castShadow = true; mesh.receiveShadow = true; mesh.name = 'front_log_' + v;
      const o = new THREE.Group(); o.add(mesh);
      items.push({ o, kind: 'log' });
    }
    for (const n of ['deadwood_0', 'deadwood_1', 'deadwood_0']) { const o = take(n); if (o) { wetClone(o); items.push({ o, kind: 'branch' }); } }
    items.forEach(({ o, kind }, k) => {
      const tree = kind === 'tree', log = kind === 'log';
      const holder = new THREE.Group();
      const tilt = new THREE.Group();
      holder.add(tilt); tilt.add(o);
      // lying down: rotate the tree's up axis (+Y) into the horizontal plane (trunk along -X), then center it
      o.position.set(0, 0, 0);
      o.rotation.order = 'ZYX';   // spin about the trunk first, then lay it down
      o.rotation.set(0, rnd() * 6.28, Math.PI / 2);
      o.scale.multiplyScalar(tree ? 0.75 + rnd() * 0.2 : log ? 1 : 1.4 + rnd() * 0.8);
      o.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(o);
      o.position.x -= (box.min.x + box.max.x) * 0.5;
      o.position.z -= (box.min.z + box.max.z) * 0.5;
      o.position.y -= (box.min.y + box.max.y) * 0.5;
      const half = (box.max.x - box.min.x) * 0.5;
      holder.visible = false;
      this.root.add(holder);
      const u = tree ? -4 - k * 16 - rnd() * 6 : log ? -2 - rnd() * 40 : -rnd() * 70;
      this._frontTrees.push({ obj: holder, tilt, half, u, d: lerp(-4, 5, rnd()), yaw: (rnd() - 0.5) * 0.9 + (rnd() < 0.3 ? Math.PI : 0), yawRate: (rnd() - 0.5) * 0.08, roll: rnd() * 6, tree, log, sink: tree ? 0.55 : log ? -0.02 : 0.12 });
    });
  }

  _groundY(x, z, fromY) {
    const phys = this.ctx.physics;
    if (!phys?.world) return null;
    const h = phys.raycast(_u.set(x, fromY, z), _down, fromY + 600, { groups: groups(G.ALL, G.STATIC), excludeCollider: this.wallCollider });
    return h ? h.point.y : null;
  }

  /** Ground heights for a cached row every GROUND_DS m (all FRONT_ND columns). */
  _groundRow(si) {
    let row = this._groundRows.get(si);
    if (row) return row;
    const road = this.ctx.road;
    const s = si * GROUND_DS;
    const c = road.pointAt(s, _gp), l = road.leftAt(s, _gl);
    row = new Float32Array(FRONT_ND);
    const gap = road.markers?.gap ?? 560;
    const plane = Math.abs(s - gap) > 4;
    for (let i = 0; i < FRONT_ND; i++) {
      const d = this._ds[i];
      // the asphalt is analytic (road.json + 2% crown): no ray needed there
      if (plane && Math.abs(d) < 2.9) { row[i] = c.y - 0.02 * Math.abs(d) + 0.01; continue; }
      const x = c.x + l.x * d, z = c.z + l.z * d;
      let y = this._groundY(x, z, c.y + 45);
      if (y == null || y > c.y + 44) {
        // no collider (isolation) or blocked: approximate the cross-section
        y = d > 4.3 ? c.y + (d - 4.3) * 1.6 : d < -3.9 ? c.y - (-3.9 - d) * 0.9 : c.y;
      }
      row[i] = y;
    }
    this._groundRows.set(si, row);
    if (this._groundRows.size > 900) { const k = this._groundRows.keys().next().value; this._groundRows.delete(k); }
    return row;
  }

  /** Seamless (period TOPO_L in material coordinate sm) per-column tiles: static lumpy topography + liquid mask. */
  _buildTopo() {
    const NK = TOPO_L / TOPO_R, ND = FRONT_ND;
    this._topo = new Float32Array(ND * NK);
    this._liq = new Float32Array(ND * NK);
    for (let i = 0; i < ND; i++) {
      const d = this._ds[i];
      const topo = (x) => {
        let n = fbm2(x * 0.16, d * 0.2) * 1.2;
        const rg = 1 - Math.abs(vnoise2(x * 0.42 + 7.7, d * 0.45));
        n += rg * rg * rg * 0.75 - 0.2;
        const cl = 1 - Math.abs(vnoise2(x * 1.25 + 3.3, d * 1.3));
        return n + cl * cl * 0.34 + vnoise2(x * 2.6, d * 2.6) * 0.07;
      };
      const liq = (x) => vnoise2(x * 0.09 + 11, d * 0.12);
      for (let k = 0; k < NK; k++) {
        const x = k * TOPO_R, w = x / TOPO_L;
        this._topo[i * NK + k] = topo(x) * (1 - w) + topo(x - TOPO_L) * w;
        this._liq[i * NK + k] = (liq(x) * (1 - w) + liq(x - TOPO_L) * w) * 1.25;
      }
    }
  }

  _topoAt(arr, i, sm) {
    const NK = TOPO_L / TOPO_R;
    let k = sm / TOPO_R;
    k = ((k % NK) + NK) % NK;
    const k0 = Math.floor(k), t = k - k0, k1 = k0 + 1 === NK ? 0 : k0 + 1;
    const b = i * NK;
    return arr[b + k0] + (arr[b + k1] - arr[b + k0]) * t;
  }

  /** Per-column (lateral) values that only depend on d and time: nose length + apron reach. Once per frame. */
  _updateColumns(T) {
    if (this._colT === T && this._lobe) return;
    this._colT = T;
    const ND = FRONT_ND;
    if (!this._lobe) { this._lobe = new Float32Array(ND); this._apronEnd = new Float32Array(ND); }
    for (let i = 0; i < ND; i++) {
      const d = this._ds[i];
      const lobe = 4.2 + 2.6 * (vnoise2(d * 0.16 + 3.7, T * 0.05) * 0.5 + 0.5) + 0.5 * vnoise2(d * 0.6 + T * 0.3, 1.3);
      this._lobe[i] = lobe;
      this._apronEnd[i] = lobe + Math.min(FRONT_SNOUT - 0.5 - lobe, 0.9) * clamp(0.5 + 0.9 * vnoise2(d * 0.35 + 9.1, T * 0.08) + 0.45 * vnoise2(d * 1.7 + 2.3, T * 0.2), 0, 1);
    }
    this._H = (FRONT_H + Math.min(this.frontSpeed, 10) * 0.22) * this._grow;
  }

  /** Debris-mass surface height (world y) at (s, d) (column di), given the ground height g. */
  _surf(s, d, di, u, roadY, g, T) {
    const H = this._H;
    let A;
    // lobate nose: its length varies across the road and slowly in time; a thin slurry apron runs ahead of it
    const lobe = this._lobe[di];
    // the churning snout face: lumps of debris bulge out of and fall back into the face (horizontal jitter of the
    // profile, so it shows on the steep face), faster when the flow runs
    let uj = u;
    if (u > -4) {
      const C = this._churnT;
      uj += (vnoise3(d * 0.55 + 1.3, C * 0.7, s * 0.45) * 0.8 + vnoise3(d * 1.5, C * 1.5 + 4.1, s * 1.4) * 0.45 + vnoise3(d * 3.7, C * 2.3, s * 3.1 + 2) * 0.16) * smooth(-4, -0.5, u);
    }
    if (uj > 0) { const x = Math.min(uj / lobe, 1); A = Math.pow(Math.max(0, 1 - x * x), 0.6); }
    else A = lerp(1, 0.52, smooth(-4, -40, uj));   // thick bouldery snout, thinner slurry body (~2 m) behind
    const sm = s - this._flowDisp;
    // big lumpy topography that travels with the surface (precomputed seamless tile) + slow/fast churning
    let n = this._topoAt(this._topo, di, sm) + vnoise3(sm * 0.55, d * 0.6, T * 0.25) * 0.32;
    if (u > -8) {
      const snoutZone = smooth(-8, 0, u);
      n += (vnoise3(sm * 1.1, d * 0.9, T * 1.6) * 0.5 + vnoise3(sm * 2.3, d * 2.1, T * 2.2) * 0.18) * snoutZone;
    }
    const cross = 0.72 + 0.34 * smooth(-4, 7, d);
    let level = roadY + (H * cross + n * (H / FRONT_H)) * A;
    if (u < -5) {
      const k = H / FRONT_H;
      // lateral levees of coarse debris along the margins of the body
      level += smooth(-5, -22, u) * k * (0.85 * Math.exp(-(((d + 3.3) / 1.5) ** 2)) + 0.6 * Math.exp(-(((d - 3.0) / 1.3) ** 2)));
      // surges: roll waves that run down the body faster than the front and pile up into the snout
      const zone = smooth(-85, -14, u) * (1 - smooth(-9, -3, u));
      if (zone > 0) { const w = Math.max(0, Math.sin((s - this._surgeX + d * 0.35) * 0.2618)); level += w * w * w * 1.0 * k * zone; }
    }
    // valley side: the flow pours over the edge and thins down the slope
    if (d < -3.6) level -= (-3.6 - d) * 1.1;
    const spill = d < -3.9 ? (0.35 + 1.2 * Math.exp(-(-3.9 - d) / 4)) * A * (H / FRONT_H) : 0.2 * A;
    // slurry apron: muddy water running ahead of the nose; at speed it rises into a bow wave right at the nose
    const bow = Math.min(this.frontSpeed / 10, 1) * 0.3 * smooth(lobe * 1.25, lobe * 0.85, u);
    const apron = u > lobe * 0.7 ? (0.03 + bow) * smooth(this._apronEnd[di], lobe * 0.9, u) * (0.35 + 0.65 * (vnoise2(sm * 0.8, d * 0.8) * 0.5 + 0.5)) : 0;
    let y;
    // uphill: where the ground rises above the flow level (cut face, scar slope) the surface dives under it
    if (d > 2 && g > level - spill) y = g > level ? level : g + spill;
    else y = Math.max(level, g + spill + apron);
    // bury the lateral edges and the snout tip under the ground
    const ed = smooth(FRONT_D0, FRONT_D0 + 2.2, d) * smooth(FRONT_D1, FRONT_D1 - 1.2, d);
    y = lerp(g - 0.35, y, ed);
    if (u > FRONT_SNOUT - 0.4) y = lerp(y, g - 0.05, smooth(FRONT_SNOUT - 0.4, FRONT_SNOUT, u));
    if (this._introPile && this._grow < 0.995) {
      // the collapse pours off the slope: a mud tongue crossing the road from the uphill side
      const e = lerp(FRONT_D1 + 1.5, FRONT_D0 - 2.5, clamp(this._grow * 1.3, 0, 1));
      y = lerp(g - 0.3, y, smooth(e - 2.5, e + 1.2, d + vnoise2(s * 0.35, 5.5) * 1.6));
    }
    return y;
  }

  _updateFront(dt) {
    const { ctx } = this;
    const mesh = this.frontMesh;
    if (!mesh || !ctx.road) return;
    const F = this.front;
    if (!(this.frontS > 0)) { mesh.visible = false; F.active = false; this._hideFrontExtras(); return; }
    F.active = true;
    this.frontS += this.frontSpeed * dt;
    // visual front eases toward the logical one (never behind by more than a few meters)
    const lag = this.frontS - this._visS;
    this._visS += lag * Math.min(1, dt * 1.5);
    if (Math.abs(lag) > 25) this._visS = this.frontS;
    this._grow += (this._growTarget - this._grow) * Math.min(1, dt * 0.55);
    this._flowDisp += (this.frontSpeed * 1.25 + 0.15) * dt;   // surface creeps even when the front is parked
    this._surgeX += (this.frontSpeed * 1.6 + 0.3) * dt;
    this._churnT += (0.3 + Math.min(this.frontSpeed, 12) * 0.22) * dt;
    this._updateColumns(this.t);
    F.s = this.frontS; F.speed = this.frontSpeed; F.height = (FRONT_H + Math.min(this.frontSpeed, 10) * 0.22) * this._grow; F.visS = this._visS;

    const cam = ctx.camera.position;
    const cpr = ctx.road.project(cam, this._prCam);
    const dist = cpr.s - this._visS;
    mesh.visible = dist > -FRONT_BACK - 20 && dist < 900;
    if (!mesh.visible) { this._hideFrontExtras(); return; }
    // throttle: far away and slow -> update less often
    this._frontTick = (this._frontTick || 0) + 1;
    const every = Math.abs(dist) > 250 ? 6 : (this.frontSpeed < 0.05 && Math.abs(dist) > 60 ? 3 : (Math.abs(dist) > 32 ? 2 : 1)); // the chase keeps it ~45 m back: every other frame there
    if (this._frontTick % every !== 0 && !this._frontDirty) {
      // skipped frame: the front's distant-LOD rocks are re-pushed from last frame's record (the LOD mesh is rebuilt
      // every frame), everything else keeps its instances
      this._extraDt = (this._extraDt || 0) + dt;
      const a = this._frontLoRec;
      if (a) for (let i = 0; i < this._frontLoN; i++) { _m.fromArray(a, i * 17); this._pushLo(_m, a[i * 17 + 16]); }
      return;
    }
    this._frontDirty = false;
    const T = this.t;
    const road = ctx.road, us = this._us, ds = this._ds, ND = FRONT_ND, NS = us.length;
    const pos = this.frontGeo.attributes.position.array, uv = this.frontGeo.attributes.uv.array, flow = this.frontGeo.attributes.aFlow.array;
    const H = this._heights, Gs = this._grounds;
    const rowP = [], rowL = [];
    for (let j = 0; j < NS; j++) {
      const u = us[j], s = Math.max(0.5, this._visS + u);
      const c = road.pointAt(s, _p), l = road.leftAt(s, _l);
      const fi = s / GROUND_DS, si = Math.floor(fi), f = fi - si;
      const r0 = this._groundRow(si), r1 = this._groundRow(si + 1);
      const sm = s - this._flowDisp;
      for (let i = 0; i < ND; i++) {
        const d = ds[i], k = j * ND + i;
        const g = r0[i] + (r1[i] - r0[i]) * f;
        const y = this._surf(s, d, i, u, c.y, g, T);
        H[k] = y; Gs[k] = g;
        pos[k * 3] = c.x + l.x * d; pos[k * 3 + 1] = y; pos[k * 3 + 2] = c.z + l.z * d;
        uv[k * 2] = (d + (y - c.y) * 0.8 * (smooth(3, 6, d) - smooth(-3.5, -6.5, d))) / 3.4; uv[k * 2 + 1] = (sm - (y - c.y) * 0.9) / 3.4;
        const th = y - g;
        // bouldery, coarse snout; finer and more liquid slurry further back; the slurry apron ahead of the nose
        const liquid = Math.max(clamp(0.38 + this._topoAt(this._liq, i, sm) * 1.3 - 0.35 * smooth(-9, 1, u) + 0.25 * smooth(-12, -45, u), 0, 1) * smooth(0.2, 1.2, th), smooth(this._lobe[i] * 0.72, this._lobe[i] * 0.95, u) * 0.18);
        const spd = (0.45 + 0.55 * clamp(1 - ((d + 0.5) / 7.5) ** 2, 0, 1)) * (1 + 0.6 * smooth(-10, 0, u));
        flow[k * 3] = liquid;
        flow[k * 3 + 2] = spd;
      }
      rowP.push(c.x, c.y, c.z); rowL.push(l.x, l.z);
    }
    // normals by central differences on the grid + trough (cavity) mask
    const nor = this.frontGeo.attributes.normal.array;
    for (let j = 0; j < NS; j++) {
      const j0 = Math.max(0, j - 1), j1 = Math.min(NS - 1, j + 1);
      for (let i = 0; i < ND; i++) {
        const i0 = Math.max(0, i - 1), i1 = Math.min(ND - 1, i + 1);
        const a = (j * ND + i0) * 3, b = (j * ND + i1) * 3, c = (j0 * ND + i) * 3, d = (j1 * ND + i) * 3;
        // tangents along d (tx) and along s (ts)
        const tx = pos[b] - pos[a], ty = pos[b + 1] - pos[a + 1], tz = pos[b + 2] - pos[a + 2];
        const sx = pos[c] - pos[d], sy = pos[c + 1] - pos[d + 1], sz = pos[c + 2] - pos[d + 2];
        let nx = ty * sz - tz * sy, ny = tz * sx - tx * sz, nz = tx * sy - ty * sx;
        const len = Math.hypot(nx, ny, nz) || 1;
        if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
        const k = j * ND + i;
        nor[k * 3] = nx / len; nor[k * 3 + 1] = ny / len; nor[k * 3 + 2] = nz / len;
        const avg = (H[j * ND + i0] + H[j * ND + i1] + H[j0 * ND + i] + H[j1 * ND + i]) * 0.25;
        flow[k * 3 + 1] = clamp((avg - H[k]) * 2.2, 0, 1);
      }
    }
    for (const k of ['position', 'normal', 'uv', 'aFlow']) this.frontGeo.attributes[k].needsUpdate = true;
    this._frontLoRec = this._frontLoRec || [];
    this._recLo = this._frontLoRec; this._recLoN = 0;
    try { this._updateFrontExtras(dt + (this._extraDt || 0)); } finally { this._frontLoN = this._recLoN; this._recLo = null; }
    this._extraDt = 0;
  }

  /** Top of the debris mass at (s, d) or null outside it. */
  surfaceY(s, d) {
    if (!(this.frontS > 0) || !this.ctx.road || !this._ds) return null;
    const u = s - this._visS;
    if (u > FRONT_SNOUT || u < -FRONT_BACK || d < FRONT_D0 || d > FRONT_D1) return null;
    const c = this.ctx.road.pointAt(s, _w);
    const fi = s / GROUND_DS, si = Math.floor(fi), f = fi - si;
    const di = clamp(Math.round((d - FRONT_D0) / (FRONT_D1 - FRONT_D0) * (FRONT_ND - 1)), 0, FRONT_ND - 1);
    const r0 = this._groundRow(si), r1 = this._groundRow(si + 1);
    const g = r0[di] + (r1[di] - r0[di]) * f;
    this._updateColumns(this.t);
    return this._surf(s, d, di, u, c.y, g, this.t);
  }

  /** World point on the snout at lateral offset d (mid-height of the leading edge). */
  frontPoint(d, out = new THREE.Vector3()) {
    if (!this.ctx.road) return out.set(0, 0, 0);
    const s = this._visS + FRONT_SNOUT * 0.55;
    this.ctx.road.worldAt(s, d, out);
    const y = this.surfaceY(s, d);
    if (y != null) out.y = y;
    return out;
  }

  _hideFrontExtras() {
    for (const m of this.frontRockMeshes || []) { instReset(m); m.visible = false; }
    if (this.frontPebbleMesh) { instReset(this.frontPebbleMesh); this.frontPebbleMesh.visible = false; }
    for (const t of this._frontTrees || []) t.obj.visible = false;
  }

  _updateFrontExtras(dt) {
    const { ctx } = this;
    const road = ctx.road, fs = this.frontSpeed;
    const meshes = this.frontRockMeshes || [];
    for (const m of meshes) instReset(m);
    const place = (R, im) => {
      // surface moves ~1.25x faster than the front: rocks travel toward the snout, roll over it and get buried
      R.u += fs * 0.25 * dt + 0.02 * dt;
      if (R.u > 6.2) Object.assign(R, this._newFrontRock(this._frontRnd, R.mesh, R.pebble, false));
      const s = this._visS + R.u;
      const y = this.surfaceY(s, R.d);
      if (y == null) return;
      if (R.u > -1.5) {
        // on the nose: embed along the surface normal (the face is steep, a vertical offset would bury it)
        const y2 = this.surfaceY(s + 0.35, R.d) ?? y;
        const slope = (y - y2) / 0.35, nl = Math.hypot(slope, 1), off = R.r * (1 - 2 * R.sink);
        const t = road.tangentAt(s, _u);
        road.worldAt(s, R.d, _v);
        _v.x += t.x * slope / nl * off; _v.z += t.z * slope / nl * off; _v.y = y + off / nl;
      } else {
        road.worldAt(s, R.d, _v);
        _v.y = y - R.r * R.sink;
      }
      const roll = (fs * 1.25 + 0.05) * dt / Math.max(R.r, 0.1) * R.spin;
      const nearSnout = smooth(-2, FRONT_SNOUT * 0.7, R.u);
      _q2.setFromAxisAngle(road.leftAt(s, _l), -roll * (1 + nearSnout * 3));
      R.q.premultiply(_q2);
      _q2.setFromAxisAngle(R.axis, roll * 0.2); R.q.multiply(_q2);
      if (R.pebble) {                       // cobbles are sub-pixel beyond a few tens of metres
        const c2 = _v.distanceToSquared(ctx.camera.position);
        if (c2 > 48 * 48 || (c2 > 26 * 26 && R.r < 0.16)) return;
      }
      _m.compose(_v, R.q, _s.set(R.r, R.r * 0.85, R.r));
      if (!R.pebble) {
        const c2 = _v.distanceToSquared(ctx.camera.position);
        if (c2 > 85 * 85 && R.r < 0.6) return;   // a few pixels: not worth an instance
        if (R.r < 0.45 || c2 > 20 * 20 || im.count >= 11) { this._pushLo(_m, R.mudP); return; }
      }
      instPush(im, _m, R.mudP);
    };
    for (const R of this._frontRocks || []) { const im = meshes[R.mesh]; if (im && im.count < 48) place(R, im); }
    for (const m of meshes) instFinish(m);
    const pm = this.frontPebbleMesh;
    if (pm) {
      instReset(pm);
      for (const R of this._frontPebbles) if (pm.count < this._nPebbles) place(R, pm);
      instFinish(pm);
    }
    for (const T of this._frontTrees || []) {
      T.u += (fs * 0.2 + 0.01) * dt;
      if (T.u > -2.5) { T.u = -8 - this._frontRnd() * 50; T.d = lerp(-6, 6, this._frontRnd()); }
      const s = this._visS + T.u;
      const y = this.surfaceY(s, T.d);
      if (y == null) { T.obj.visible = false; continue; }
      road.worldAt(s, T.d, _v);
      T.obj.visible = _v.distanceToSquared(ctx.camera.position) < 110 * 110;
      if (!T.obj.visible) continue;
      road.worldAt(s, T.d, T.obj.position);
      T.obj.position.y = y - T.sink;
      T.yaw += T.yawRate * dt * (0.3 + fs * 0.3);
      T.roll += (fs * 0.35) * dt;
      // pitch the trunk to lie on the surface: sample the mass at both ends (trunk along local -X / +X)
      const ang = road.yawAt(s) + T.yaw;
      const ex = Math.sin(ang + Math.PI / 2), ez = Math.cos(ang + Math.PI / 2); // local +X in world (y-rot)
      const pA = ctx.road.project(_w.set(T.obj.position.x + ex * T.half, 0, T.obj.position.z + ez * T.half), this._pr);
      const yA = this.surfaceY(pA.s, pA.d) ?? ctx.road.pointAt(pA.s, _u).y;
      const pB = ctx.road.project(_w.set(T.obj.position.x - ex * T.half, 0, T.obj.position.z - ez * T.half), this._pr);
      const yB = this.surfaceY(pB.s, pB.d) ?? ctx.road.pointAt(pB.s, _u).y;
      T.obj.position.y = Math.max(y, (yA + yB) * 0.5) - T.sink;
      T.obj.rotation.set(0, ang, 0);
      T.tilt.rotation.set(0.05 * Math.sin(T.roll * 0.7), 0, clamp(Math.atan2(yA - yB, 2 * T.half), -0.3, 0.3));
    }
  }

  // ============================================================================================ hazards
  _onPlayerStruck(p) {
    if (!p?.collider) return;
    const rock = this._rockHandles.get(p.collider.handle);
    if (!rock || !rock.lethal) return;
    if ((p.energy ?? 0) > 0.012 && rock.r >= 0.22) this._hazard('player', 'boulder', p.energy);
  }

  _hazard(target, cause, energy, extra = null) {
    const st = this.ctx.game?.state;
    if (st === 'dead' || st === 'win' || st === 'title') return;
    if (this.t - this._hazardT < (cause === 'boulder' && target === 'car' ? 1.2 : 2.5)) return;
    this._hazardT = this.t;
    this.ctx.events?.emit('hazard:hit', { target, cause, energy, ...(extra || {}) });
  }

  /** Does a settled rock sit on the drivable asphalt (outside the intro pile zone)? */
  _blocksLane(R, pr) {
    if (!pr || R.r < 0.45) return false;
    if (pr.s < (this.ctx.road?.markers?.rockfallIntro ?? 150) + 35) return false; // the intro pile is meant to block
    if (pr.dy > R.r + 1.6 || pr.dy < -1.5) return false;
    return Math.abs(pr.d) - R.r * 0.7 < 2.7;
  }

  /** Keep a lane-blocking rock rolling toward the nearest road edge; after a few tries it breaks apart. */
  _clearLane(R, pr) {
    const { ctx } = this;
    const b = R.body;
    if (!b) return false;
    R.nudges = (R.nudges || 0) + 1;
    if (R.nudges <= 3) {
      const l = ctx.road.leftAt(pr.s, _l);
      const side = pr.d > 0.4 ? 1 : -1;          // uphill half: into the ditch; valley half: over the edge
      const dv = 2.6 + R.nudges * 0.7;
      try {
        b.wakeUp();
        b.applyImpulse({ x: l.x * side * dv * R.mass, y: 0.9 * R.mass, z: l.z * side * dv * R.mass }, true);
        const t = ctx.road.tangentAt(pr.s, _u);
        b.setAngvel({ x: t.x * -side * dv / R.r, y: 0, z: t.z * -side * dv / R.r }, true);
      } catch { return false; }
      R.still = 0;
      return true;
    }
    // stuck (against the cut face / guardrail): the boulder splits into fragments small enough to drive over
    const n = 3 + Math.floor(this.rng() * 2);
    const at = R.pos.clone(), rr = R.r;
    this._retire(R, false);
    for (let k = 0; k < n; k++) {
      const q = at.clone().add(_u.set((this.rng() - 0.5) * rr, this.rng() * rr * 0.5, (this.rng() - 0.5) * rr));
      const f = this.spawnBoulder(q, new THREE.Vector3((this.rng() - 0.5) * 2, 1 + this.rng(), (this.rng() - 0.5) * 2), clamp(rr * (0.28 + this.rng() * 0.12), 0.12, 0.4), { mud: 0.5 });
      if (f) { f.lethal = false; f.nudges = 9; }
    }
    ctx.particles?.dust?.(at.clone(), 1.2 + rr * 1.5, 0.5);
    ctx.events?.emit('impact', { position: at, energy: 0.6 * rr, radius: rr, source: 'rock', surface: 'rock' });
    return true;
  }

  // ============================================================================================ impact effects
  /**
   * A rock hit something hard (velocity jump dv, unit direction n = the push it received ≈ the surface normal):
   * a gouge / mud splat on the ground, a billowing cloud of crushed rock + soil for big blocks, a spray of mud, and
   * the first hard hit of a big block breaks pieces off it (fresh rockfall blocks fracture on the first impact).
   */
  _onRockImpact(R, pos, e, dv, hit, surface) {
    const { ctx } = this;
    const cam = ctx.camera.position;
    const d2 = R.pos.distanceToSquared(cam);
    if (d2 > 260 * 260) return;
    if (hit && this.decals && d2 < 160 * 160 && R.r >= 0.15 && e > 0.02) {
      const v = R.prevV, vh = Math.hypot(v.x, v.z);
      const w = R.r * (1.3 + Math.min(e, 4) * 0.12);
      const len = w * (1 + Math.min(vh * 0.1, 1.6));
      this.decals.add(hit.point, hit.normal, vh > 0.5 ? _w.set(v.x, 0, v.z) : null, len, w, R.hits <= 2 ? 0 : 2, clamp(0.55 + e * 0.2, 0.55, 1));
    }
    // big blocks: a rolling cloud of pulverised rock and soil (wet slope: brown-grey, settles within ~10 s). The
    // spray, clots / chips and the low mist of every hit come from particles.impact() via the 'impact' event.
    if (R.r >= 0.42 && R.hits <= 2 && e > 0.35) this._impactCloud(pos, R.r, e, hit ? hit.point.y : pos.y - R.r, surface);
    if (R.hits === 1 && R.r >= 0.5 && dv > 4.5 && !R.frag) return this._fragment(R, pos, e);
    return false;
  }

  /** Ground contact of a rock impact (the push points out of the surface: look back along it) and its surface class
   *  for the particles: 'mud' (the debris mass, the scar, soil), 'asphalt', 'gravel' or 'rock' (rock-on-rock). */
  _impactGround(R, pos, nx, ny, nz) {
    const { ctx } = this;
    const phys = ctx.physics;
    let hit = null;
    if (ny > 0.35 && phys?.raycast) hit = phys.raycast(R.pos, _u.set(-nx, -ny, -nz), R.r * 1.8 + 0.3, { groups: groups(G.ALL, G.STATIC), excludeCollider: this.wallCollider });
    // landed on the debris mass itself
    if (this.frontS > 0 && ctx.road) {
      const pr = ctx.road.project(pos, this._prImp || (this._prImp = {}));
      if (pr.s < this._visS + FRONT_SNOUT && pr.d > FRONT_D0 && pr.d < FRONT_D1) {
        const y = this.surfaceY(pr.s, pr.d);
        if (y != null && pos.y < y + R.r + 0.6) return { hit, surface: 'mud' };
      }
    }
    if (!hit) return { hit, surface: 'rock' };
    const t = ctx.terrain?.surfaceAt?.(hit.point);
    return { hit, surface: t === 'asphalt' || t === 'gravel' || t === 'rock' ? t : 'mud' };
  }

  /** Break a block on its first hard impact. Timed (aimed) rocks keep their size and path: they only shed pieces. */
  _fragment(R, pos, e) {
    const { ctx } = this;
    if (this.rocks.length > this.maxRocks - 10) return false;
    const rnd = this.rng;
    const v = R.body?.linvel?.();
    if (!v) return false;
    const split = !R.aimed && !R.plan && R.r >= 0.7 && rnd() < 0.45;
    const n = split ? 2 + Math.floor(rnd() * 2) : 1 + Math.floor(rnd() * 3);
    const at = R.pos.clone(), rr = R.r, lethal = R.lethal, mud = R.mud, q = R.quat.clone();
    if (split) {
      // the block splits: two big halves carry on (the second slightly off-line), plus the loose pieces below
      this._retire(R, false);
      const halves = [0.76, 0.6];
      halves.forEach((f, k) => {
        const off = _u.set((rnd() - 0.5), (rnd() - 0.2) * 0.5, (rnd() - 0.5)).normalize().multiplyScalar(rr * 0.45 * (k ? 1 : -1));
        const p = at.clone().add(off);
        const vel = new THREE.Vector3(v.x, v.y, v.z).multiplyScalar(0.85 + rnd() * 0.1).addScaledVector(off, 2.2 / rr);
        const h = this.spawnBoulder(p, vel, rr * f, { mud, rotation: q, lethal });
        if (h) h.frag = 1;
      });
    }
    for (let k = 0; k < n; k++) {
      const dir = _u.set(rnd() - 0.5, 0.4 + rnd() * 0.6, rnd() - 0.5).normalize();
      const p = at.clone().addScaledVector(dir, rr * 0.7);
      const vel = new THREE.Vector3(v.x * 0.55, Math.abs(v.y) * 0.3, v.z * 0.55).addScaledVector(dir, 2.5 + rnd() * 3 + Math.sqrt(e));
      const f = this.spawnBoulder(p, vel, clamp(rr * (0.18 + rnd() * 0.2), 0.08, 0.35), { mud: 0.15 + rnd() * 0.35, lethal: false });
      if (f) { f.frag = 1; f.nudges = 9; f.lethal = false; }
    }
    ctx.particles?.debris?.(at.clone(), Math.round(10 + rr * 10), { speed: 3 + Math.sqrt(e) * 2.5, mud: 0.25 });
    return split;
  }

  /** Mud streak left by a bouncing / rolling block: decals chained along its contact path. */
  _rollStreak(R, v, sp) {
    const { ctx } = this;
    const cam = ctx.camera.position;
    if (R.pos.distanceToSquared(cam) > 75 * 75) { R._stP = null; return; }
    const hit = ctx.physics?.raycast?.(R.pos, _down, R.r * 1.3 + 0.15, { groups: groups(G.ALL, G.STATIC), excludeCollider: this.wallCollider });
    if (!hit) { R._stP = null; return; }
    const last = R._stP;
    if (!last) { R._stP = hit.point; return; }
    const dist = last.distanceTo(hit.point);
    if (dist < R.r * 1.3 + 0.2) return;
    if (dist < R.r * 6) {
      const mid = _p.addVectors(last, hit.point).multiplyScalar(0.5);
      this.decals.add(mid, hit.normal, _w.subVectors(hit.point, last), dist + R.r * 0.8, R.r * (0.7 + Math.min(sp, 8) * 0.03), sp > 5 ? 3 : 1, clamp(0.35 + R.mud * 0.6, 0.35, 0.9));
    }
    R._stP = hit.point;
  }

  /** One lit soft puff through the particle system's dust pool (bypasses its budget for set pieces). */
  _puff(p, vel, life, s0, s1, op, col, ground, buoy = 0.15, drag = 1.2, aspect = 1, soft = 0) {
    const D = this.ctx.particles?.dustSys;
    if (D && typeof D.emit === 'function') {
      // Medium / Low: fewer puffs (Math.random, not this.rng: the seeded stream drives the rocks and must not differ by preset)
      if (this.fxQ < 1 && Math.random() > this.fxQ) return true;
      D.emit(p, vel, life, s0, s1, op, col, ground, buoy, 1, aspect, drag, soft);
      return true;
    }
    return false;
  }

  /** Crushed-rock + soil cloud of a big impact: a dark ground surge (mud + water) that rolls out first, then a thinner,
   *  lighter brownish-grey billow rising above it (lit from above). A block landing on asphalt or bare rock throws far
   *  less of it than one gouging into the soil. */
  _impactCloud(pos, r, e, groundY, surface = 'mud') {
    const { ctx } = this;
    const cam = ctx.camera.position;
    const dist = pos.distanceTo(cam);
    if (dist > 320) return;
    const rnd = this.rng;
    // a wet slope throws far less dust than a dry one: budget + no double clouds from the same spot
    const now = this.t;
    this._clouds = (this._clouds || []).filter((c) => now - c.t < 1.4);
    for (const c of this._clouds) if (c.p.distanceToSquared(pos) < 9) return;
    if ((this._cloudBudget ?? 20) < 2) return;
    this._clouds.push({ p: pos.clone(), t: now });
    const hard = surface === 'asphalt' || surface === 'rock';
    const k = clamp(Math.sqrt(e), 0.6, 3.5);
    const n = Math.min(Math.floor(this._cloudBudget ?? 20), Math.round(clamp(1.5 + r * 2 + k * 0.7, 2, 6) * (dist < 12 ? 0.5 : 1) * (hard ? 0.6 : 1)));
    this._cloudBudget = (this._cloudBudget ?? 20) - n;
    for (let i = 0; i < n; i++) {
      const a = rnd() * Math.PI * 2, sp = (1.5 + k * 1.6) * (0.5 + rnd() * 0.8);
      const surge = i < Math.max(1, n * 0.55);
      _v.set(Math.cos(a) * sp, surge ? 0.2 + rnd() * 0.4 : 1.0 + rnd() * 1.6, Math.sin(a) * sp);
      _w.set(pos.x + Math.cos(a) * r * 0.6, Math.max(pos.y, groundY + r * 0.6) + rnd() * r * (surge ? 0.4 : 1.2), pos.z + Math.sin(a) * r * 0.6);
      if (hard) palMix(_col, PAL.rockMist, PAL.slideTop, surge ? 0.1 : 0.5, 0.06);
      else palMix(_col, PAL.slideBase, PAL.slideTop, surge ? rnd() * 0.2 : 0.55 + rnd() * 0.45, 0.07);
      const s0 = r * (1.0 + rnd() * 0.8), s1 = (surge ? 3 : 4) + r * (1.5 + rnd() * 1.5) + k * 0.9;
      const op = surge ? 0.22 + rnd() * 0.12 : 0.12 + rnd() * 0.1;
      if (!this._puff(_w, _v, (surge ? 4 : 5) + rnd() * 4, s0, s1, op, _col, groundY, surge ? 0.02 : 0.18, surge ? 1.5 : 0.8, surge ? 1.5 : 1, surge ? 0.6 : 0.3)) {
        ctx.particles?.dust?.(pos, clamp(r * 3 + k, 1, 9), e);
        return;
      }
    }
  }

  /** Spray + dust plume above the moving debris front (billows up, drifts with the valley wind). */
  _updatePlume(dt) {
    const { ctx } = this;
    const F = this.front;
    if (!F.active || !ctx.road || !this.frontMesh?.visible || dt <= 0) return;
    const cam = ctx.camera.position;
    const pr = ctx.road.project(cam, this._prCam);
    const dist = Math.abs(pr.s - this._visS);
    if (dist > 500) return;
    const fs = this.frontSpeed;
    const rate = (fs > 0.6 ? 0.8 + Math.min(fs, 12) * 0.26 : 0.12 + fs * 0.3) * this._grow * (ctx.config?.quality?.key === 'ultra' ? 1.4 : 1);
    this._plumeAcc = (this._plumeAcc || 0) + dt * rate;
    const rnd = this.rng;
    let n = 0;
    while (this._plumeAcc >= 1 && n++ < 3) {
      this._plumeAcc -= 1;
      const d = lerp(-6, 6.5, rnd());
      const sb = this._visS - 1 - rnd() * 9;
      ctx.road.worldAt(sb, d, _w);
      const sy = this.surfaceY(sb, d) ?? _w.y + F.height;
      const gy = sy - 0.5;
      // height in the plume: the base (mud + water spray churned off the flow) is dark, dense and hugs the surface;
      // the fines that make it higher are lighter brownish grey and thinner, lit from above
      const hf = rnd();
      _w.y = sy + F.height * (0.2 + hf * 0.75);
      const near = dist < 22;
      palMix(_col, PAL.frontBase, PAL.slideTop, 0.15 + hf * 0.8, 0.07);
      const tg = ctx.road.tangentAt(this._visS, _u);
      _v.set(0, 0.6 + hf * 0.8 + rnd() * 1.2, 0).addScaledVector(tg, fs * (0.6 + rnd() * 0.4));
      const s1 = (near ? 5 : 7) + rnd() * (near ? 3 : 6) + Math.min(fs, 12) * 0.3;
      const op = lerp(near ? 0.2 : 0.3, near ? 0.09 : 0.14, hf) + rnd() * 0.08;
      this._puff(_w, _v, 5 + rnd() * 3.5, 2 + rnd() * 2, s1, op, _col, gy, 0.12 + hf * 0.35 + rnd() * 0.1, 0.7, 1 + (1 - hf) * 0.5, lerp(0.55, 0.25, hf));
    }
  }

  // ============================================================================================ loop
  fixedUpdate(h) {
    const { ctx } = this;
    // front wall follows the (logical) front
    if (this.wallBody && ctx.road) {
      if (this.frontS > 0) {
        const s = this.frontS + 1.5;   // solid core of the nose: frontS+0.3 .. frontS+2.7
        const c = ctx.road.worldAt(s, 0, _v);
        const q = ctx.road.frameQuat(s, _q);
        this.wallBody.setNextKinematicTranslation({ x: c.x, y: c.y, z: c.z });
        this.wallBody.setNextKinematicRotation({ x: q.x, y: q.y, z: q.z, w: q.w });
      } else this.wallBody.setNextKinematicTranslation({ x: 0, y: -500, z: 0 });
    }
    if (!this.rocks.length) return;
    const car = ctx.car, cb = car?.body;
    let cp = null, cq = null, cv = null;
    if (cb) { const t = cb.translation(), r = cb.rotation(), v = cb.linvel(); cp = _cp.set(t.x, t.y, t.z); cq = _cq.set(r.x, r.y, r.z, r.w).invert(); cv = _cv.set(v.x, v.y, v.z); }
    let impacts = 0;
    for (let i = this.rocks.length - 1; i >= 0; i--) {
      const R = this.rocks[i];
      const b = R.body;
      if (!b) continue;
      const t = b.translation(), v = b.linvel(), q = b.rotation();
      if (!isFinite(t.x) || !isFinite(t.y)) { this._retire(R, false); continue; }
      R.pos.set(t.x, t.y, t.z); R.quat.set(q.x, q.y, q.z, q.w);
      R.age += h; R.cool -= h;
      // impact detection from the velocity jump (minus gravity)
      const dvx = v.x - R.prevV.x, dvy = v.y - R.prevV.y + GRAV * h, dvz = v.z - R.prevV.z;
      const dv = Math.hypot(dvx, dvy, dvz);
      if (dv > 2.2 && R.cool <= 0 && impacts < 5) {
        R.cool = 0.12; R.hits++; impacts++; R.hitT = R.age;
        const e = 0.5 * R.mass * dv * dv / 1e5;
        const pos = new THREE.Vector3(t.x - dvx / dv * R.r * 0.8, t.y - dvy / dv * R.r * 0.8, t.z - dvz / dv * R.r * 0.8);
        // what did it hit? (ground contact + surface type: particles throw mud from soil, chips + water from asphalt)
        let hit = null, surface = 'rock';
        // (only within effect range: particles ignore hits > 450 m away, _onRockImpact > 260 m)
        if (pos.distanceToSquared(ctx.camera.position) < 450 * 450) { try { ({ hit, surface } = this._impactGround(R, pos, dvx / dv, dvy / dv, dvz / dv)); } catch { /* fx only */ } }
        ctx.events?.emit('impact', { position: pos, energy: e, radius: R.r, source: 'rock', hits: R.hits, surface, ground: hit ? hit.point.y : undefined });
        let gone = false;
        try { gone = this._onRockImpact(R, pos, e, dv, hit, surface) === true; } catch (err) { if (!this._impErr) { this._impErr = true; console.warn('[landslide] impact fx', err); } }
        if (gone || !R.body) continue;
      }
      R.prevV.set(v.x, v.y, v.z);
      const sp = Math.hypot(v.x, v.y, v.z);
      if (this.decals && sp > 1.2 && R.r >= 0.18 && ((R._stT = (R._stT ?? 0) - h) <= 0)) { R._stT = 0.1; this._rollStreak(R, v, sp); }
      // car hit: rock inside the chassis box (car local frame), with relative kinetic energy
      if (cp && R.lethal && R.r >= 0.25) {
        _rl.set(t.x - cp.x, t.y - cp.y, t.z - cp.z).applyQuaternion(cq);
        const m = R.r * 0.75;
        if (Math.abs(_rl.x) < 0.9 + m && _rl.y > -0.2 - m && _rl.y < 1.8 + m && Math.abs(_rl.z) < 2.0 + m) {
          const rel = Math.hypot(v.x - cv.x, v.y - cv.y, v.z - cv.z);
          const e = 0.5 * R.mass * rel * rel / 1e5;
          const inCar = ctx.control === 'car' || ctx.cameraRig?.mode?.startsWith?.('car');
          if (e > 1.1 && _rl.y > 0.4 && inCar) this._hazard('car', 'boulder', e, { rockSpeed: sp, radius: R.r, kind: R.kind || R.tag || null, plan: R.plan ? { ...R.plan, age: +(this.t - R.plan.t0).toFixed(2) } : null });
        }
      }
      // player fallback proximity check (in case the sensor misses a fast rock)
      const pl = ctx.player;
      if (pl?.enabled && pl.feet && R.lethal && R.r >= 0.25 && sp > 4) {
        const fy = pl.feet.y;
        const cy = clamp(t.y, fy + 0.3, fy + 1.5);
        const dd = Math.hypot(t.x - pl.feet.x, t.y - cy, t.z - pl.feet.z);
        if (dd < R.r * 0.8 + 0.3) { const e = 0.5 * R.mass * sp * sp / 1e5; if (e > 0.012) this._hazard('player', 'boulder', e); }
      }
      // settle / cull
      if (sp < 0.25 || b.isSleeping()) R.still += h; else R.still = 0;
      const pr = ctx.road?.project(R.pos, R._pr || (R._pr = {}));
      if (pr && (pr.dy < -60 || (this.frontS > 0 && pr.s < this._visS - 2 && Math.abs(pr.d) < 9 && t.y < (this.surfaceY(pr.s, pr.d) ?? -1e9) + R.r * 0.2))) { this._retire(R, false); continue; }
      // timed rocks keep their promise after landing: an 'ahead' rock may not roll back down the road toward the car,
      // a 'behind' one may not chase it (only the along-road component is limited; the roll across is untouched)
      if (R.plan && R.age > R.plan.T * 0.8 && pr && Math.abs(pr.d) < 6) {
        const tg = ctx.road.tangentAt(pr.s, _u);
        const vs = v.x * tg.x + v.y * tg.y + v.z * tg.z;
        const lim = R.kind === 'ahead' ? (vs < -0.6 ? vs + 0.6 : 0) : (vs > 0.6 ? vs - 0.6 : 0);
        if (lim !== 0) { try { b.applyImpulse({ x: -tg.x * lim * R.mass * 0.5, y: -tg.y * lim * R.mass * 0.5, z: -tg.z * lim * R.mass * 0.5 }, true); } catch { /* */ } }
      }
      // a boulder slowing to a stop on the asphalt gets moving again right away (not after it has settled)
      if (sp < 1.8 && R.age > 1.0 && R.r >= 0.45 && R.age - (R.nudgeT ?? -9) > 0.9 && R.still <= 1.0 && pr && this._blocksLane(R, pr)) {
        R.nudgeT = R.age;
        if (this._clearLane(R, pr)) continue;
      }
      if (R.still > 1.0 || R.age > 40) {
        const keep = pr && pr.dy > -25 && Math.abs(pr.d) < 30;
        // a boulder must never plug the lane for good: it keeps rolling off the nearest edge (or breaks up)
        if (keep && R.age <= 40 && this._blocksLane(R, pr) && this._clearLane(R, pr)) continue;
        this._retire(R, keep);
      }
    }
  }

  update(dt) {
    const { ctx } = this;
    this.t += dt;
    // scheduled actions
    if (this.queue.length) {
      const due = this.queue.filter((a) => a.t <= this.t);
      if (due.length) {
        this.queue = this.queue.filter((a) => a.t > this.t);
        for (const a of due) { try { a.fn(); } catch (e) { console.warn('[landslide] scheduled action failed', e); } }
      }
    }
    if (this.escape && dt > 0) this._updateEscape(dt);
    if (this.loMesh) instReset(this.loMesh);
    this._updateFront(dt);
    this._updatePlume(dt);
    this._cloudBudget = Math.min((this._cloudBudget ?? 20) + dt * 10, 22);
    if (dt > 0) this._updateTreeSnaps();
    this._updateCamVel(dt);
    this._updateRockVisuals();
    if (dt > 0) { try { this._updateTrails(dt); } catch (e) { if (!this._trErr) { this._trErr = true; console.warn('[landslide] rock trails', e); } } }
    this._updateHazards(dt);
    this._updateRumble(dt);
    if (this._flowUniforms) { const fp = this._flowUniforms.flowP.value; fp.x = this.t; fp.y = this.frontSpeed; fp.z = ctx.env?.wetness ?? 0.75; }
    if (this.rockUniforms) this.rockUniforms.rkP.value.x = ctx.env?.wetness ?? 0.75;
  }

  /** Smoothed camera velocity (m/s) for the rocks' exposure smear; ignores teleports / cuts. */
  _updateCamVel(dt) {
    const cam = this.ctx.camera.position;
    if (!this._camPrev) { this._camPrev = cam.clone(); this._camVel = new THREE.Vector3(); return; }
    if (dt > 0) {
      _u.subVectors(cam, this._camPrev).divideScalar(dt);
      if (_u.lengthSq() > 70 * 70) _u.set(0, 0, 0);
      this._camVel.lerp(_u, clamp(dt * 12, 0, 1));
    }
    this._camPrev.copy(cam);
  }

  /**
   * Flying blocks shed what they carry (REAL-WORLD MODEL, rockfall footage): a tumbling block (w ~ 0.35 v/r, 1-12
   * rad/s) flings clods of its soil coat and grit off its rim at the tangential speed w x r (2-6 m/s on top of its own
   * velocity), plus a spray of water drops (1.5-3.5 mm) and a faint smear of fine wet soil. The coat is shed within
   * seconds, so the rate decays with the block's age and scales with its coat and size. Everything goes through the
   * particle system's pools, rate-limited so it never evicts the impact splats (chips <= ~40/s, drops <= ~70/s).
   */
  _updateTrails(dt) {
    const P = this.ctx.particles;
    const chips = P?.chips, drops = P?.drops, dust = P?.dustSys;
    if (!chips && !drops) return;
    const cam = this.ctx.camera;
    this._chipB = Math.min((this._chipB ?? 0) + dt * 70 * this.fxQ, 14);
    this._dropB = Math.min((this._dropB ?? 0) + dt * 90 * this.fxQ, 18);
    this._wispB = Math.min((this._wispB ?? 0) + dt * 36, 4);
    if (!this.rocks.length) return;
    const fr = this._frustum || (this._frustum = new THREE.Frustum());
    fr.setFromProjectionMatrix(_m.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse));
    const sph = this._trSph || (this._trSph = new THREE.Sphere());
    const cp = cam.position, R2 = TRAIL_RANGE * TRAIL_RANGE, rnd = Math.random;
    const phys = this.ctx.physics;
    for (const R of this.rocks) {
      if (R.r < 0.2 || !R.body) continue;
      const v = R.prevV, sp = v.length();
      if (sp < 3) continue;
      const d2 = R.pos.distanceToSquared(cp);
      if (d2 > R2) continue;
      sph.center.copy(R.pos); sph.radius = R.r * 3 + 2;
      if (!fr.intersectsSphere(sph)) continue;
      const airborne = R.age - (R.hitT ?? -9) > 0.12;
      // what is left of the coat to shed (fresh blocks carry the most; rolling sheds more slowly than a free spin)
      const coat = (0.35 + R.mud) * Math.exp(-R.age / 7);
      const rate = coat * clamp(sp / 10, 0.4, 1.6) * (0.6 + R.r * 7) * (airborne ? 1 : 0.45);
      R._trAcc = (R._trAcc || 0) + dt * rate;
      // landing height below the rock (cached raycast; the road plane when that fails)
      if (!(R._gyT > R.age)) {
        R._gyT = R.age + 0.3;
        const h = phys?.raycast?.(R.pos, _down, 90, { groups: groups(G.ALL, G.STATIC), excludeCollider: this.wallCollider });
        R._gy = h ? h.point.y : (R._pr ? R.pos.y - R._pr.dy : R.pos.y - 30);
      }
      const gy = R._gy;
      let av = null;
      try { av = R.body.angvel(); } catch { av = null; }
      const near = d2 < 20 * 20;
      while (R._trAcc >= 1) {
        R._trAcc -= 1;
        // a point on the block's rim, flung off tangentially
        _u.set(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
        _p.copy(R.pos).addScaledVector(_u, R.r * 0.85);
        _w.set(v.x, v.y, v.z).multiplyScalar(0.9 + rnd() * 0.08);
        if (av) { _l.set(av.x, av.y, av.z).cross(_u).multiplyScalar(R.r * 0.85 * (0.7 + rnd() * 0.4)); _w.add(_l); }
        _w.x += (rnd() - 0.5) * 1.2; _w.y += (rnd() - 0.3) * 1.2; _w.z += (rnd() - 0.5) * 1.2;
        if (chips && this._chipB >= 1) {
          this._chipB -= 1;
          const mud = rnd() < 0.72 ? 1 : 0;
          // clods of the coat 3-12 cm (visible at 10-40 m: ~1 px per 2-3 cm there), grit 1-5 cm
          const sc = mud ? 0.03 + Math.pow(rnd(), 1.5) * 0.09 * Math.min(R.r, 1.2) : 0.012 + Math.pow(rnd(), 2) * 0.04;
          chips.emit(_p, _w, 2 + rnd() * 2, sc, mud, gy + sc * 0.3, (rnd() - 0.5) * 22);
        }
        if (drops && this._dropB >= 2) {
          const nd = near ? 3 : 2;
          for (let k = 0; k < nd; k++) {
            this._dropB -= 1;
            _l.copy(_w); _l.x += (rnd() - 0.5) * 1.5; _l.y += (rnd() - 0.4) * 1.5; _l.z += (rnd() - 0.5) * 1.5;
            drops.emit(_p, _l, 2.5, gy, 1.5 + rnd() * 2, 0.5, 1.4);
          }
        }
      }
      // the wake: a faint, continuous smear of wet fine soil and spray shed from the spinning rim of big airborne
      // blocks (field footage: a thin brown-grey haze trailing a tumbling block for ~1 s, not a dry white plume).
      // Albedo of saturated fine soil ~0.12, so it reads darker than the rain haze behind it.
      R._wkAcc = (R._wkAcc || 0) + (airborne && R.r >= 0.4 && coat > 0.15 ? dt * (6 + 10 * Math.min(coat, 1)) * Math.min(sp / 8, 1.5) : 0);
      while (dust && R._wkAcc >= 1 && this._wispB >= 1) {
        R._wkAcc -= 1; this._wispB -= 1;
        _u.set(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
        _p.copy(R.pos).addScaledVector(_u, R.r * 0.6).addScaledVector(v, -0.03 - rnd() * 0.03);
        _w.set(v.x, v.y, v.z).multiplyScalar(0.25 + rnd() * 0.15);
        _w.addScaledVector(_u, 0.8);
        palMix(_col, PAL.frontBase, PAL.soilMist, rnd() * 0.6, 0.08);
        dust.emit(_p, _w, 0.6 + rnd() * 0.5, R.r * 0.3, R.r * (0.9 + rnd() * 0.4), 0.16 + rnd() * 0.1, _col, gy, 0.03, 1, 1, 2.6, 0.5);
      }
      if (R._wkAcc > 3) R._wkAcc = 3;
    }
  }

  /** Trees standing in the path of the flow snap as the snout reaches them ('tree:snap' {position}); the instance is
   *  removed from the forest (the flow carries its own uprooted trunks). Built lazily from vegetation.trees. */
  _updateTreeSnaps() {
    const { ctx } = this;
    if (!(this.frontS > 0) || !ctx.road) return;
    if (!this._snapTrees) {
      const f = ctx.vegetation?.trees;
      if (!f?.data || !f.count) return;
      const list = [];
      const pr = {};
      const tmp = new THREE.Vector3();
      for (let i = 0; i < f.count; i++) {
        const o = i * 6;
        if (!(f.data[o + 3] > 0)) continue;
        tmp.set(f.data[o], f.data[o + 1], f.data[o + 2]);
        if (tmp.x < 60 || tmp.x > 1250) continue;
        pr._hint = -1e6;
        ctx.road.project(tmp, pr);
        if (pr.d > FRONT_D0 + 1 && pr.d < FRONT_D1 - 1.5 && pr.dy < 6 && pr.dy > -12 && pr.s > 120 && pr.s < 1150) list.push({ s: pr.s, pos: tmp.clone() });
      }
      list.sort((a, b) => a.s - b.s);
      this._snapTrees = list;
      this._snapIdx = 0;
      // skip trees already behind the front (checkpoint restart)
      while (this._snapIdx < list.length && list[this._snapIdx].s < this._visS - 5) this._snapIdx++;
    }
    const L = this._snapTrees;
    if (this._snapResync) { this._snapResync = false; this._snapIdx = 0; while (this._snapIdx < L.length && L[this._snapIdx].s < this._visS - 5) this._snapIdx++; }
    let n = 0;
    while (this._snapIdx < L.length && L[this._snapIdx].s < this._visS + FRONT_SNOUT * 0.45 && n++ < 3) {
      const T = L[this._snapIdx++];
      if (T.gone) continue;
      T.gone = true;
      if (T.s < this._visS - 30) { try { ctx.vegetation?.removeTreesNear?.(T.pos, 0.9); } catch { /* */ } continue; } // jumped over
      try { ctx.vegetation?.removeTreesNear?.(T.pos, 0.9); } catch { /* optional */ }
      if (T.pos.distanceToSquared(ctx.camera.position) < 320 * 320) {
        const p = T.pos.clone(); p.y += 1.2;
        ctx.events?.emit('tree:snap', { position: p });
        ctx.particles?.debris?.(p.clone(), 8);
        ctx.particles?.dust?.(p, 2.5, 0.5);
      }
    }
  }

  _updateRockVisuals() {
    const meshes = this.liveMeshes;
    if (!meshes) return;
    const cam = this.ctx.camera.position;
    const L = this.lodCfg || (this.lodCfg = { live: LIVE_FULL, small: LIVE_FULL_SMALL, debris: DEBRIS_FULL, lod1: true });
    const loV = L.lod1 ? (this.loVar || []) : [];
    for (const m of meshes) instReset(m);
    for (const m of this.loVar || []) if (m) instReset(m);
    const cv = this._camVel;
    const lf2 = L.live * L.live, lfs2 = L.small * L.small;
    for (const R of this.rocks) {
      const im = meshes[R.variant];
      _m.compose(R.pos, R.quat, _s.set(R.r, R.r, R.r));
      const d2 = R.pos.distanceToSquared(cam);
      if (d2 > (R.r < 0.3 ? lfs2 : lf2)) {
        const lm = loV[R.variant];
        if (lm) { if (lm.count < lm.instanceMatrix.count) instPush(lm, _m, R.mudP); } else this._pushLo(_m, R.mudP);
        continue;
      }
      // velocity relative to the (moving) camera, for the exposure smear
      const v = R.prevV;
      instPush(im, _m, R.mudP, cv ? _w.set(v.x - cv.x, v.y - cv.y, v.z - cv.z) : v);
    }
    for (const m of meshes) instFinish(m);
    for (const m of this.debrisMeshes) instReset(m);
    const df2 = L.debris * L.debris;
    for (const D of this.debris) {
      const im = this.debrisMeshes[D.variant];
      if (!im) continue;
      if (D.pos.distanceToSquared(cam) > df2 * (D.r > 0.8 ? 1.6 : 1)) {
        const lm = loV[D.variant];
        if (lm) { if (lm.count < lm.instanceMatrix.count) instPush(lm, D.matrix, D.mudP); } else this._pushLo(D.matrix, D.mudP);
        continue;
      }
      instPush(im, D.matrix, D.mudP);
    }
    for (const m of this.debrisMeshes) instFinish(m);
    for (const m of this.loVar || []) if (m) instFinish(m);
    if (this.loMesh) instFinish(this.loMesh);
    // debris swallowed by the advancing front disappears into it
    if (this.frontS > 0 && this.debris.length && this.ctx.road) {
      for (let i = this.debris.length - 1; i >= 0; i--) {
        const D = this.debris[i];
        const pr = this.ctx.road.project(D.pos, this._pr);
        if (pr.s < this._visS + 1 && Math.abs(pr.d) < 9) {
          const y = this.surfaceY(pr.s, pr.d);
          if (y != null && D.pos.y + D.r * 0.3 < y) this._removeDebrisAt(i);
        }
      }
    }
  }

  _updateHazards(dt) {
    const { ctx } = this;
    if (!(this.frontS > 0) || !ctx.road || dt <= 0) return;
    // the moving front reaches the player / car (a parked pile only blocks, it does not kill)
    const reach = this.frontS + 2.4;
    const moving = this.frontSpeed > 0.05;
    const pl = ctx.player;
    if (pl?.enabled && (ctx.control === 'foot' || ctx.control === 'none') && pl.feet) {
      const pr = ctx.road.project(pl.feet, this._prPl);
      if (moving && pr.s < reach && pr.s > this.frontS - FRONT_BACK && pr.d > FRONT_D0 && pr.d < FRONT_D1 && pr.dy < 12) this._hazard('player', 'front', 10);
    }
    const car = ctx.car;
    if (car?.body && (ctx.control === 'car' || ctx.cameraRig?.mode?.startsWith?.('car'))) {
      const t = car.body.translation();
      const pr = ctx.road.project(_v.set(t.x, t.y, t.z), this._prCar);
      if (moving && pr.s < reach + 1.8 && pr.s > this.frontS - FRONT_BACK && Math.abs(pr.d) < 10 && pr.dy < 12) this._hazard('car', 'front', 20);
    }
    // grinding/snapping at the snout -> impact events (dust, sound, camera) scaled by speed and proximity
    this._frontImpactT -= dt;
    if (this._frontImpactT <= 0) {
      const fs = this.frontSpeed;
      this._frontImpactT = (fs > 0.05 ? lerp(0.9, 0.18, clamp(fs / 10, 0, 1)) : 2.5) * (0.6 + this.rng() * 0.8);
      const p = this.frontPoint(lerp(-5, 7, this.rng()), new THREE.Vector3());
      const e = (0.15 + this.rng() * 0.6) * (0.4 + clamp(fs / 6, 0, 1.5)) * this._grow;
      ctx.events?.emit('impact', { position: p, energy: e, radius: 0.8 + this.rng() * 0.8, source: 'front' });
    }
  }

  _updateRumble(dt) {
    const { ctx } = this;
    if (dt <= 0) return;
    const cam = ctx.camera.position;
    let a = 0;
    for (const R of this.rocks) {
      const v = R.prevV, ke = 0.5 * R.mass * (v.x * v.x + v.y * v.y + v.z * v.z) / 1e5;
      const d2 = R.pos.distanceToSquared(cam);
      a += ke / (1 + d2 / 900);
    }
    let target = Math.min(0.55, 1 - Math.exp(-a * 0.08));
    if (this.frontS > 0 && ctx.road) {
      const pr = ctx.road.project(cam, this._prCam);
      const dist = Math.max(0, pr.s - this.frontS);
      const near = Math.exp(-dist / 70);
      target = Math.max(target, near * (0.3 + 0.7 * clamp(this.frontSpeed / 8, 0, 1)) * this._grow + near * 0.15);
    }
    this._burst = Math.max(0, this._burst - dt * 0.35);
    target = Math.max(target, this._burst * 0.9);
    const k = target > this.rumble ? Math.min(1, dt * 6) : Math.min(1, dt * 1.2);
    this.rumble = clamp(this.rumble + (target - this.rumble) * k, 0, 1);
  }

  clear() {
    for (const R of [...this.rocks]) this._retire(R, false);
    for (let i = this.debris.length - 1; i >= 0; i--) this._removeDebrisAt(i);
    this.queue = []; this.escape = null;
    this._snapResync = true;
    this.frontS = 0; this.frontSpeed = 0; this._visS = 0; this.front.active = false; this._introPile = false;
    this._grow = 1; this._growTarget = 1;
    this.decals?.clear();
  }

  dispose() {
    this.clear();
    const w = this.ctx.physics?.world;
    try { if (this.wallBody && w) w.removeRigidBody(this.wallBody); } catch { /* */ }
    this.ctx.scene.remove(this.root);
  }
}

// ------------------------------------------------------------------------------------------------ helpers
let _simp = null;
/** three's bundled meshoptimizer simplifier (WASM), or null. */
function loadSimplifier() {
  if (!_simp) {
    _simp = import('three/examples/jsm/libs/meshopt_simplifier.module.js')
      .then(async (m) => { const S = m.MeshoptSimplifier; await S.ready; return S; })
      .catch((e) => { console.warn('[landslide] mesh simplifier unavailable, LOD1 falls back to the pebble proxy', e); return null; });
  }
  return _simp;
}

/** LOD geometry sharing geo's attributes (UVs, tangents, atlas) with a simplified index. Null on failure. */
function lodGeometry(geo, S, ratio, err) {
  try {
    const P = geo.attributes.position, n = P.count;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { pos[i * 3] = P.getX(i); pos[i * 3 + 1] = P.getY(i); pos[i * 3 + 2] = P.getZ(i); }
    const idx = geo.index ? Uint32Array.from(geo.index.array) : Uint32Array.from({ length: n }, (_, i) => i);
    const target = Math.max(36, Math.floor((idx.length * ratio) / 3) * 3);
    const res = S.simplify(idx, pos, 3, target, err, []);
    const out = Array.isArray(res) ? res[0] : res;
    if (!out || out.length < 36 || out.length >= idx.length) return null;
    const g = new THREE.BufferGeometry();
    for (const k in geo.attributes) g.setAttribute(k, geo.attributes[k]);
    g.setIndex(new THREE.BufferAttribute(n < 65536 ? Uint16Array.from(out) : out, 1));
    g.boundingSphere = geo.boundingSphere;
    return g;
  } catch (e) {
    console.warn('[landslide] LOD1 simplify failed', e);
    return null;
  }
}

/** Velocity that takes a projectile from p to q in time T under gravity. */
function ballistic(p, q, T, out) {
  return out.set((q.x - p.x) / T, (q.y - p.y) / T + 0.5 * GRAV * T, (q.z - p.z) / T);
}

/** Approximate volume of a convex point cloud (unit rock): bounding-box based estimate. */
function hullVolume(pts) {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let i = 0; i < pts.length; i += 3) {
    x0 = Math.min(x0, pts[i]); x1 = Math.max(x1, pts[i]);
    y0 = Math.min(y0, pts[i + 1]); y1 = Math.max(y1, pts[i + 1]);
    z0 = Math.min(z0, pts[i + 2]); z1 = Math.max(z1, pts[i + 2]);
  }
  return (x1 - x0) * (y1 - y0) * (z1 - z0) * 0.5;
}

/** Procedural fallback rock: fractured, displaced icosahedron (flat-cut planes + noise). */
function proceduralRock(seed, detail = 2) {
  const rnd = mulberry(seed);
  const g = new THREE.IcosahedronGeometry(1, detail);
  const p = g.attributes.position;
  const planes = [];
  for (let i = 0; i < 7; i++) {
    const n = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
    planes.push({ n, d: 0.55 + rnd() * 0.35 });
  }
  const sc = new THREE.Vector3(0.8 + rnd() * 0.4, 0.6 + rnd() * 0.3, 0.8 + rnd() * 0.4);
  for (let i = 0; i < p.count; i++) {
    _v.fromBufferAttribute(p, i);
    _v.multiplyScalar(1 + vnoise3(_v.x * 2 + seed, _v.y * 2, _v.z * 2) * 0.12);
    for (const pl of planes) { const k = _v.dot(pl.n); if (k > pl.d) _v.addScaledVector(pl.n, pl.d - k); }
    _v.multiply(sc);
    p.setXYZ(i, _v.x, _v.y, _v.z);
  }
  g.computeVertexNormals();
  g.computeBoundingSphere();
  return g;
}

/**
 * Procedural snapped spruce trunk, built upright (+Y) from the base (y=0). Groups: 0 bark, 1 fresh wood, 2 soil.
 * variant 0: root plate + broken top (long), 1: big root plate + short broken top, 2: snapped at both ends.
 */
function buildSnappedLog(rnd, variant) {
  const P = [[], [], []], U = [[], [], []], I = [[], [], []];
  const add = (g, x, y, z, u, v) => { P[g].push(x, y, z); U[g].push(u, v); return P[g].length / 3 - 1; };
  const quad = (g, a, b, c, d) => I[g].push(a, b, c, b, d, c);
  const L = variant === 0 ? 10 + rnd() * 3 : variant === 1 ? 6.5 + rnd() * 2 : 5 + rnd() * 2;
  const r0 = variant === 1 ? 0.3 + rnd() * 0.06 : 0.22 + rnd() * 0.06;
  const wad = variant !== 2;
  const RS = 10, LS = Math.ceil(L / 0.7);
  const bendX = (rnd() - 0.5) * 0.5, bendZ = (rnd() - 0.5) * 0.5;
  const axis = (y) => { const t = y / L; return [bendX * t * t + vnoise2(y * 0.6, 3.3) * 0.05, bendZ * t * t + vnoise2(y * 0.6, 7.1) * 0.05]; };
  const rad = (y) => r0 * (1 - 0.5 * y / L) * (1 + (wad ? 0.55 : 0.1) * Math.exp(-y / 0.5));
  const circ = 2 * Math.PI * r0;
  // trunk rings (bark); the first ring at y0 (inside the root plate or at the lower break)
  const y0 = wad ? 0.1 : 0;
  const rings = [];
  for (let j = 0; j <= LS; j++) {
    const y = y0 + (L - y0) * j / LS;
    const [ax, az] = axis(y), r = rad(y);
    const ring = [];
    for (let i = 0; i <= RS; i++) {
      const a = i / RS * Math.PI * 2;
      const rr = r * (1 + vnoise2(a * 1.3 + j * 0.2, y * 1.7) * 0.06);
      ring.push(add(0, ax + Math.cos(a) * rr, y, az + Math.sin(a) * rr, i / RS * circ, y));
    }
    rings.push(ring);
  }
  for (let j = 0; j < LS; j++) for (let i = 0; i < RS; i++) quad(0, rings[j][i], rings[j][i + 1], rings[j + 1][i], rings[j + 1][i + 1]);
  // splintered break: jagged ring of fresh wood rising from the last bark ring, then a fan to a torn core
  const splinter = (y, dir) => {
    const [ax, az] = axis(y), r = rad(y);
    const jag = [], ring = [];
    for (let i = 0; i <= RS; i++) {
      const a = i / RS * Math.PI * 2;
      const up = (i % RS === 0 ? 0.3 : 0.08 + Math.pow(rnd(), 1.6) * 0.9) * dir;
      const rr = r * (0.55 + rnd() * 0.4);
      ring.push(add(1, ax + Math.cos(a) * r, y, az + Math.sin(a) * r, i / RS, 0));
      jag.push(add(1, ax + Math.cos(a) * rr, y + up, az + Math.sin(a) * rr, i / RS, Math.abs(up)));
    }
    const core = add(1, ax, y + 0.25 * dir, az, 0.5, 0.3);
    for (let i = 0; i < RS; i++) {
      if (dir > 0) { quad(1, ring[i], ring[i + 1], jag[i], jag[i + 1]); I[1].push(jag[i], jag[i + 1], core); }
      else { quad(1, ring[i + 1], ring[i], jag[i + 1], jag[i]); I[1].push(jag[i + 1], jag[i], core); }
    }
  };
  splinter(L, 1);
  if (!wad) splinter(0, -1);
  // branch stubs on the upper trunk (the flow strips branches, leaving torn stubs)
  const nStub = 6 + Math.floor(rnd() * 6);
  for (let k = 0; k < nStub; k++) {
    const y = L * (0.35 + rnd() * 0.6), a = rnd() * Math.PI * 2;
    const [ax, az] = axis(y), r = rad(y);
    const len = 0.25 + rnd() * 0.7, br = 0.018 + rnd() * 0.03;
    const dx = Math.cos(a), dz = Math.sin(a), dy = 0.35 + rnd() * 0.4;
    tube(0, [[ax + dx * r * 0.7, y, az + dz * r * 0.7], [ax + dx * (r + len), y + dy * len, az + dz * (r + len)]], br, br * 0.6, 4);
  }
  if (wad) {
    // root plate: a lumpy, flattened disc of soil around the trunk base, perpendicular to the trunk
    const R = variant === 1 ? 1.6 + rnd() * 0.35 : 1.25 + rnd() * 0.3, TH = 0.38 + rnd() * 0.18;
    const WS = 16, HS = 8, base = P[2].length / 3;
    for (let j = 0; j <= HS; j++) {
      const ph = j / HS * Math.PI;
      for (let i = 0; i <= WS; i++) {
        const th = i / WS * Math.PI * 2;
        const nx = Math.sin(ph) * Math.cos(th), ny = Math.cos(ph), nz = Math.sin(ph) * Math.sin(th);
        const n = 1 + vnoise3(nx * 2.1 + variant, ny * 2.1, nz * 2.1) * 0.28 + vnoise3(nx * 5 + 3, ny * 5, nz * 5) * 0.1;
        const edge = 1 - 0.25 * Math.pow(Math.abs(Math.cos(th * 3 + variant)), 4);
        add(2, nx * R * n * edge, 0.05 + ny * TH * 0.5 * n, nz * R * n * edge, i / WS * R * 3, j / HS * 2);
      }
    }
    for (let j = 0; j < HS; j++) for (let i = 0; i < WS; i++) { const a = base + j * (WS + 1) + i; quad(2, a, a + 1, a + WS + 1, a + WS + 2); }
    // roots: torn off at the rim, trailing out of the soil plate (radial, drooping), plus stubby sinker roots
    const nR = 22 + Math.floor(rnd() * 10);
    for (let k = 0; k < nR; k++) {
      const a = rnd() * Math.PI * 2, dx = Math.cos(a), dz = Math.sin(a);
      const s0 = R * (0.45 + rnd() * 0.4), len = R * (0.2 + Math.pow(rnd(), 1.5) * 0.7), rr = 0.035 + Math.pow(rnd(), 1.5) * 0.08;
      const yv = (rnd() - 0.5) * TH * 0.7;
      const pts = [];
      for (let q = 0; q <= 3; q++) {
        const t = q / 3, rad2 = s0 + len * t;
        pts.push([dx * rad2 + (rnd() - 0.5) * 0.12 * t, yv - t * t * (0.2 + rnd() * 0.4) + (rnd() - 0.5) * 0.1, dz * rad2 + (rnd() - 0.5) * 0.12 * t]);
      }
      tube(0, pts, rr, rr * 0.35, 5);
    }
    for (let k = 0; k < 5; k++) {
      const a = rnd() * Math.PI * 2, rr = R * (0.1 + rnd() * 0.4);
      tube(0, [[Math.cos(a) * rr, -TH * 0.3, Math.sin(a) * rr], [Math.cos(a) * rr * 1.2, -TH * 0.3 - 0.4 - rnd() * 0.5, Math.sin(a) * rr * 1.2]], 0.05, 0.015, 4);
    }
  }
  function tube(g, pts, rA, rB, sides) {
    const n = pts.length, rings2 = [];
    for (let q = 0; q < n; q++) {
      const p = pts[q], pn = pts[Math.min(n - 1, q + 1)], pp = pts[Math.max(0, q - 1)];
      _v.set(pn[0] - pp[0], pn[1] - pp[1], pn[2] - pp[2]).normalize();
      _w.set(0, 1, 0); if (Math.abs(_v.y) > 0.9) _w.set(1, 0, 0);
      _u.crossVectors(_v, _w).normalize(); _w.crossVectors(_u, _v).normalize();
      const r = lerp(rA, rB, q / (n - 1)), ring = [];
      for (let i = 0; i <= sides; i++) {
        const a = i / sides * Math.PI * 2, c = Math.cos(a) * r, s2 = Math.sin(a) * r;
        ring.push(add(g, p[0] + _u.x * c + _w.x * s2, p[1] + _u.y * c + _w.y * s2, p[2] + _u.z * c + _w.z * s2, i / sides * 0.3, q * 0.3));
      }
      rings2.push(ring);
    }
    for (let q = 0; q < n - 1; q++) for (let i = 0; i < sides; i++) quad(g, rings2[q][i], rings2[q][i + 1], rings2[q + 1][i], rings2[q + 1][i + 1]);
  }
  // merge the three parts into one geometry with material groups
  const pos = [], uv = [], idx = [];
  const geo = new THREE.BufferGeometry();
  let vb = 0;
  for (let g = 0; g < 3; g++) {
    const i0 = idx.length;
    for (const i of I[g]) idx.push(i + vb);
    pos.push(...P[g]); uv.push(...U[g]);
    vb += P[g].length / 3;
    if (idx.length > i0) geo.addGroup(i0, idx.length - i0, g);
  }
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
  return geo;
}

// ------------------------------------------------------------------------------------------------ ground decals
// Gouges and mud streaks (REAL-WORLD MODEL): a block landing on asphalt or turf crushes and scrapes a pale-edged,
// dark scar full of wet crushed soil and throws a spatter of mud forward (downslope); a rolling block paints a
// broken, glistening mud band along its contact path. Wet mud albedo ~0.05, roughness 0.25 (film) .. 0.55 (clods).
const DEC_VERT_HEAD = /* glsl */`
attribute vec4 aDec;        // x type (atlas quadrant), y opacity, z uv offset, w unused
varying vec4 vDec;
varying vec2 vDcUv;
`;
const DEC_FRAG_HEAD = /* glsl */`
uniform sampler2D dcSplat;
varying vec4 vDec;
varying vec2 vDcUv;
float dcGloss = 0.0;
`;
class SlideDecals {
  constructor(ctx, soil, cap = 220) {
    this.ctx = ctx; this.cap = cap; this.head = 0; this.used = 0;
    const g = new THREE.PlaneGeometry(1, 1);
    g.rotateX(-Math.PI / 2);                   // +Y normal, uv.y runs along -Z (the streak direction)
    this.aDec = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4);
    this.aDec.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aDec', this.aDec);
    const splat = makeSplatTexture();
    const mat = new THREE.MeshStandardMaterial({
      map: soil?.map || null, normalMap: soil?.normalMap || null, color: soil ? 0xffffff : 0x3a2a1c,
      roughness: 0.5, metalness: 0, transparent: true, depthWrite: false,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -6,
    });
    const uni = { dcSplat: { value: splat } };
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, uni);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\n' + DEC_VERT_HEAD)
        .replace('#include <uv_vertex>', `#include <uv_vertex>
          vDec = aDec; vDcUv = uv;
          #ifdef USE_INSTANCING
            vec2 dcSz = vec2( length( instanceMatrix[0].xyz ), length( instanceMatrix[2].xyz ) );
          #else
            vec2 dcSz = vec2( 1.0 );
          #endif
          #ifdef USE_MAP
            vMapUv = uv * dcSz / 0.7 + aDec.zz;
          #endif
          #ifdef USE_NORMALMAP
            vNormalMapUv = uv * dcSz / 0.7 + aDec.zz;
          #endif`);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\n' + DEC_FRAG_HEAD)
        .replace('#include <map_fragment>', `#include <map_fragment>
          {
            vec4 spl = texture2D( dcSplat, vDcUv * 0.5 + vec2( mod( vDec.x, 2.0 ), floor( vDec.x * 0.5 ) ) * 0.5 );
            // smeared mud on dark wet asphalt reads warm and a little lighter (saturated silt + clay, linear ~0.09-0.13,
            // the rain keeps it wet and glossy); the gouge core is darker churned soil mixed with pulverised rock, which
            // the rain wets down to a mid grey (dry scuffs would be near white)
            float ml = dot( diffuseColor.rgb, vec3( 0.3, 0.59, 0.11 ) );
            vec3 dMud = vec3( 0.12, 0.078, 0.045 ) * clamp( 0.45 + ml * 20.0, 0.4, 1.15 );
            vec3 dCrush = mix( dMud * 0.5, vec3( 0.15, 0.143, 0.133 ), 0.55 + 0.45 * spl.b );
            diffuseColor.rgb = mix( dMud, dCrush, spl.g );
            diffuseColor.a *= clamp( spl.r * vDec.y, 0.0, 1.0 );
            if ( diffuseColor.a < 0.02 ) discard;
            dcGloss = spl.b;
          }`)
        .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix( 0.6, 0.36, dcGloss );');
    };
    mat.customProgramCacheKey = () => 'slidedecal2';
    this.mesh = new THREE.InstancedMesh(g, mat, cap);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    this.mesh.receiveShadow = true; this.mesh.castShadow = false;
    this.mesh.renderOrder = 2;
    this.mesh.visible = false;
    this.mesh.name = 'slide_decals';
    this._m = new THREE.Matrix4(); this._q = new THREE.Quaternion(); this._s = new THREE.Vector3();
    this._x = new THREE.Vector3(); this._y = new THREE.Vector3(); this._z = new THREE.Vector3(); this._p = new THREE.Vector3();
  }
  /** pos on the ground, normal (unit), dir (streak direction, any length; null = random), length, width (m). */
  add(pos, normal, dir, length, width, type = 0, opacity = 1) {
    const i = this.head; this.head = (this.head + 1) % this.cap;
    this.used = Math.min(this.cap, this.used + 1);
    const y = this._y.copy(normal).normalize();
    let z = this._z;
    if (dir && dir.lengthSq() > 1e-6) z.copy(dir); else z.set(Math.random() - 0.5, 0, Math.random() - 0.5);
    z.addScaledVector(y, -z.dot(y));
    if (z.lengthSq() < 1e-6) z.set(1, 0, 0).addScaledVector(y, -y.x);
    z.normalize().negate();                       // uv.y (splat forward) runs along -Z of the plane
    const x = this._x.crossVectors(y, z).normalize();
    this._m.makeBasis(x, y, z);
    this._q.setFromRotationMatrix(this._m);
    this._p.copy(pos).addScaledVector(y, 0.012);
    this._m.compose(this._p, this._q, this._s.set(Math.max(width, 0.12), 1, Math.max(length, 0.12)));
    this.mesh.setMatrixAt(i, this._m);
    const a = this.aDec.array;
    a[i * 4] = type; a[i * 4 + 1] = opacity; a[i * 4 + 2] = Math.random() * 7; a[i * 4 + 3] = 0;
    this.mesh.count = this.used;
    this.mesh.visible = true;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.aDec.needsUpdate = true;
  }
  clear() { this.head = 0; this.used = 0; this.mesh.count = 0; this.mesh.visible = false; }
}

/** 512² RGBA atlas of 4 splats (R coverage, G gouge core, B wet gloss): 0 impact gouge, 1 roll streak, 2 splash, 3 skid. */
function makeSplatTexture() {
  const N = 256, W = N * 2;
  const data = new Uint8Array(W * W * 4);
  const rnd = mulberry(5150);
  const blobs = (n, cx, cy, sx, sy, r0, r1, fwd) => Array.from({ length: n }, () => {
    const a = rnd() * Math.PI * 2, d = Math.pow(rnd(), 0.7);
    return { x: cx + Math.cos(a) * d * sx, y: cy + Math.sin(a) * d * sy + (fwd ? Math.abs(Math.sin(a)) * d * fwd : 0), r: lerp(r0, r1, Math.pow(rnd(), 2)) };
  });
  const kinds = [
    { spat: blobs(46, 0.5, 0.45, 0.42, 0.5, 0.008, 0.05, 0.2) },
    { spat: blobs(22, 0.5, 0.5, 0.36, 0.5, 0.006, 0.03, 0), prints: Array.from({ length: 6 }, (_, k) => ({ x: 0.5 + (rnd() - 0.5) * 0.14, y: (k + 0.5) / 6 + (rnd() - 0.5) * 0.06, rx: 0.14 + rnd() * 0.12, ry: 0.05 + rnd() * 0.05 })).filter(() => rnd() < 0.85) },
    { spat: blobs(60, 0.5, 0.5, 0.46, 0.46, 0.006, 0.06, 0) },
    { spat: blobs(18, 0.5, 0.5, 0.3, 0.5, 0.006, 0.025, 0) },
  ];
  const sm = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  for (let q = 0; q < 4; q++) {
    const ox = (q % 2) * N, oy = Math.floor(q / 2) * N, K = kinds[q];
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const u = (i + 0.5) / N, v = (j + 0.5) / N;
      const n1 = fbm2(u * 7 + q * 3.1, v * 7), n2 = fbm2(u * 23 + 5, v * 23 + q);
      let cov = 0, core = 0, gloss = 0;
      if (q === 0) {            // crater: elongated scar, torn edge, spatter thrown forward
        const dx = (u - 0.5) / 0.3, dy = (v - 0.42) / 0.36, d = Math.hypot(dx, dy) + n1 * 0.35 + n2 * 0.1;
        cov = sm(1.0, 0.75, d); core = sm(0.85, 0.35, d) * (0.6 + 0.4 * n2 + 0.3);
        gloss = sm(0.9, 0.5, d) * (0.5 + 0.5 * n1);
      } else if (q === 1) {     // bounce / roll prints: a chain of irregular mud stamps with gaps, not a painted band
        let dm = 9;
        for (const b of K.prints) { const dd = Math.hypot((u - b.x) / b.rx, (v - b.y) / b.ry); dm = Math.min(dm, dd); }
        cov = sm(1.0, 0.72, dm + n1 * 0.45 + n2 * 0.15); core = cov * 0.35 * (0.4 + n2);
        gloss = cov * (0.4 + 0.6 * n1);
      } else if (q === 2) {     // splash: irregular blob
        const d = Math.hypot(u - 0.5, v - 0.5) / 0.27 + n1 * 0.45 + n2 * 0.12;
        cov = sm(1.0, 0.8, d); core = sm(0.8, 0.2, d) * 0.4; gloss = cov * (0.5 + 0.5 * n2);
      } else {                  // skid: 2-3 parallel smears
        const lanes = Math.abs(Math.sin((u - 0.5) * 18 + n1 * 2.5));
        const taper = sm(0.0, 0.15, v) * sm(1.0, 0.8, v) * sm(0.42, 0.2, Math.abs(u - 0.5));
        cov = sm(0.35, 0.8, lanes) * taper * (0.5 + 0.5 * sm(-0.1, 0.3, n2)); core = cov * 0.3; gloss = cov * 0.8;
      }
      for (const b of K.spat) {
        const d = Math.hypot(u - b.x, v - b.y);
        if (d < b.r) { const w = sm(b.r, b.r * 0.6, d); cov = Math.max(cov, w * 0.9); gloss = Math.max(gloss, w * 0.7); }
      }
      const o = ((oy + j) * W + ox + i) * 4;
      data[o] = cov * 255; data[o + 1] = clamp(core, 0, 1) * 255; data[o + 2] = clamp(gloss, 0, 1) * 255; data[o + 3] = 255;
    }
  }
  const t = new THREE.DataTexture(data, W, W, THREE.RGBAFormat);
  t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  return t;
}
