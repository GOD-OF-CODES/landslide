// Road + tunnel materials (TERRAIN workstream).
// Road: asphalt_02 (2048) with anti-tiling, worn edge/centre lines, cracks + tar snakes, repair patches,
// polished tire tracks, wet sheen, noise puddles (mirror-smooth, rain ripples), mud/gravel washed over the
// road near the scar, the gullies and the washout. UV0 = (d, s) in meters; COLOR_0: R = AO, G = puddle/damage.
// Tunnel: cast-concrete portal (formwork panels, tie holes, runoff, name plate, delineators), rubble wing walls,
// drainage channel, lining with grime + seepage, daylight falloff with depth, analytic sodium lamps.
import * as THREE from 'three';
import { GLSL_NOISE, loadTerrainArrays } from './terrainMaterial.js';

// s1: end cap (the tube runs on round the bend past markers.tunnelEnd and fades to black); lampEnd: last lamps.
// Must match tools/blender/terrain.py TUN_END / LAMP_END.
export const TUNNEL = { s0: 1150, s1: 1298, lampEnd: 1276 };
const MAX_LAMPS = 40;

/**
 * Uniforms shared by the road and tunnel materials, driven by terrain.js every frame:
 *  uCamIn   0..1 camera inside the tube (fog loses its daylight inscatter: nothing lights the haze in there)
 *  uPtSkip  world positions of the terrain's two tunnel PointLights (they light cars/props; these two materials
 *           already get every lamp analytically, so they ignore those point lights)
 */
export const TUNNEL_SHARED = {
  uCamIn: { value: 0 },
  uPtSkip: { value: [new THREE.Vector3(0, -1e5, 0), new THREE.Vector3(0, -1e5, 0)] },
};

/**
 * three's lights_fragment_begin with the sun / directional lights scaled by dayVar (daylight left at this
 * fragment) and the terrain's tunnel PointLights masked out. Spot lights (car headlights) and other point lights
 * are untouched, so they light the tunnel at full strength.
 */
function lightsChunk(dayVar) {
  let c = THREE.ShaderChunk.lights_fragment_begin;
  const inj = (a, b) => { if (c.includes(a)) c = c.replace(a, a + b); else console.warn('[roadMaterial] lights chunk: missing', a); };
  inj('getDirectionalLightInfo( directionalLight, directLight );', `\n\t\tdirectLight.color *= ${dayVar};`);
  inj('getSunLightInfo( sunLight, directLight );', `\n\t\tdirectLight.color *= ${dayVar};`);
  inj('getPointLightInfo( pointLight, geometryPosition, directLight );', '\n\t\tdirectLight.color *= tn_ptMask( pointLight.position );');
  return c;
}

/** Fog inside the tube: when the camera is in there too, the haze only absorbs (no daylight to scatter). */
function fogChunk(insideVar) {
  return `#ifdef USE_FOG
  {
    float fIn = uCamIn * ${insideVar};
    // one fog evaluation (hfog_od is the expensive part: mist sheets + cloud deck noise) for both terms
    // (PERF, QA: evaluating hfog_transmittance + hfog_apply separately cost ~0.7 ms/frame on the road)
    vec3 fInC;
    vec3 fT = exp( -hfog_od( vHFogRel, cameraPosition, fInC ) );
    #ifdef HFOG_TRANSMITTANCE_ONLY
      vec3 fogged = gl_FragColor.rgb * fT;
    #else
      vec3 fogged = gl_FragColor.rgb * fT + fInC * ( 1.0 - fT );
    #endif
    gl_FragColor.rgb = mix( fogged, gl_FragColor.rgb * fT, fIn );
  }
#endif`;
}


/** Lamp positions along the tunnel vault (matches tools/blender/terrain.py build_lamps). */
export function tunnelLampPositions(road) {
  const out = [];
  const Y = 1.9 + 5.3 * Math.sqrt(1 - (2.45 / 4.9) ** 2) - 0.2;
  for (const [side, off] of [[1, 0], [-1, 4]]) {
    for (let s = TUNNEL.s0 + 4 + off; s < TUNNEL.lampEnd - 1e-6; s += 8) {
      const p = road.worldAt(s, 2.45 * side, new THREE.Vector3());
      p.y += Y;
      out.push(p);
    }
  }
  return out;
}

const TUNNEL_GLSL = /* glsl */`
uniform vec3 uLampPos[${MAX_LAMPS}];
uniform int uLampCount;
uniform vec3 uLampColor;
uniform float uCamIn;
uniform vec3 uPtSkip[2];
float tn_ptMask(vec3 lightPosView) {
  for (int k = 0; k < 2; k++) {
    vec3 q = (viewMatrix * vec4(uPtSkip[k], 1.0)).xyz;
    if (distance(q, lightPosView) < 0.05) return 0.0;
  }
  return 1.0;
}
// sodium lamp irradiance (diffuse) at p with world normal n
vec3 tunnelLamps(vec3 p, vec3 n) {
  vec3 acc = vec3(0.0);
  for (int i = 0; i < ${MAX_LAMPS}; i++) {
    if (i >= uLampCount) break;
    vec3 l = uLampPos[i] - p;
    float d2 = dot(l, l);
    if (d2 > 900.0) continue;
    float nl = max(dot(n, l * inversesqrt(d2)), 0.0) * 0.85 + 0.15;
    acc += nl / (d2 + 1.5);
  }
  return acc * uLampColor;
}
`;

function tunnelUniforms(ctx) {
  const lamps = ctx.road ? tunnelLampPositions(ctx.road) : [];
  const arr = [];
  for (let i = 0; i < MAX_LAMPS; i++) arr.push(lamps[i] ? lamps[i].clone() : new THREE.Vector3(0, -1e4, 0));
  return {
    ...TUNNEL_SHARED,
    uLampPos: { value: arr }, uLampCount: { value: Math.min(MAX_LAMPS, lamps.length) },
    uLampColor: { value: new THREE.Color(1.0, 0.52, 0.16).multiplyScalar(14.0) },
  };
}

// ---------------------------------------------------------------------------------------------
// Road
// ---------------------------------------------------------------------------------------------
const ROAD_FRAG_PARS = /* glsl */`
uniform sampler2D uAsph;
uniform sampler2D uAsphN;
uniform sampler2D uAsphArm;
uniform sampler2DArray uAlb;
uniform sampler2DArray uNrm;
uniform float uWet;
uniform float uTime;
uniform vec4 uMudZones[6];  // (s0, s1, strength, sideBias)
varying vec2 vRoad;
varying vec4 vMasks;
varying vec3 vWPos;
${GLSL_NOISE}
${TUNNEL_GLSL}
vec3 rdTN; float rdRough; float rdAO; float rdTunnel; float rdDay = 1.0; float rdSpecBoost; vec3 rdAlb; float rdSpecOcc;

vec2 rd_vor(vec2 p) { // F1, F2-F1 (edges)
  vec2 i = floor(p), f = fract(p); float f1 = 8.0, f2 = 8.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec2 g = vec2(float(x), float(y));
    vec2 o = vec2(th_hash(i + g), th_hash(i + g + 17.1));
    float d = length(g + o - f);
    if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
  }
  return vec2(f1, f2 - f1);
}

// crack line along s: distance (m) of d from a wandering line centred at d0
float rd_crackLine(float d, float s, float d0, float seed, float amp) {
  float c = d0 + amp * (th_vnoise(vec2(s * 0.35, seed)) - 0.5) + 0.05 * (th_vnoise(vec2(s * 2.7, seed + 3.0)) - 0.5);
  return abs(d - c);
}

vec3 rd_ripple(vec2 p, float t) { // xy normal perturbation + ring mask from rain drops
  vec2 i = floor(p), f = fract(p); vec2 acc = vec2(0.0);
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec2 g = vec2(float(x), float(y));
    vec2 c = g + vec2(th_hash(i + g), th_hash(i + g + 5.3)) * 0.8 + 0.1;
    float ph = fract(t * 0.9 + th_hash(i + g + 9.7));
    vec2 dv = f - c; float r = length(dv);
    float ring = ph * 1.2;
    float w = exp(-pow((r - ring) * 14.0, 2.0)) * (1.0 - ph) * (1.0 - ph);
    acc += w * dv / max(r, 1e-3) * sin((r - ring) * 40.0);
  }
  return vec3(acc, 0.0);
}

void rd_shade() {
  float d = vRoad.x, s = vRoad.y, ad = abs(d);
  float aaD = fwidth(d) + 0.002;
  // ---- base asphalt with anti-tiling (second sample rotated + rescaled)
  vec2 uv1 = vec2(d, s) / 3.0;
  const float cr = 0.6, sr = 0.8;
  vec2 uv2 = vec2(cr * d - sr * s, sr * d + cr * s) / 3.35 + vec2(0.37, 0.71);
  float kt = smoothstep(0.3, 0.7, th_vnoise(vec2(d * 0.4, s * 0.07)));
  vec3 c1 = texture(uAsph, uv1).rgb, c2 = texture(uAsph, uv2).rgb;
  vec3 n1 = texture(uAsphN, uv1).xyz * 2.0 - 1.0, n2 = texture(uAsphN, uv2).xyz * 2.0 - 1.0;
  n2.xy = vec2(cr * n2.x + sr * n2.y, -sr * n2.x + cr * n2.y);
  vec3 a1 = texture(uAsphArm, uv1).rgb, a2 = texture(uAsphArm, uv2).rgb;
  vec3 alb = mix(c1, c2, kt); vec3 tn = normalize(mix(n1, n2, kt)); vec3 arm = mix(a1, a2, kt);
  float lum = dot(alb, vec3(0.3, 0.59, 0.11));
  float rough = arm.g; float ao = arm.r;
  // macro tone: sun-bleached / oily variation
  float mac = th_fbm(vec2(d * 0.25, s * 0.03));
  alb *= 0.82 + 0.36 * mac;
  alb = mix(alb, alb * vec3(1.05, 1.02, 0.95), smoothstep(0.5, 0.8, th_vnoise(vec2(d * 0.1, s * 0.013))) * 0.5);

  // ---- repairs. (1) saw-cut rectangular patches: newer, darker, finer-textured hot mix, straight edges sealed with
  //      a 5-8 cm glossy tar overband, a little proud of the old surface; (2) utility-trench strips; (3) small
  //      round cold-mix pothole fills in the wheel paths (rougher, greyer, edge ravelled)
  float cell = floor(s / 31.0);
  float hp = th_hash(vec2(cell, 7.7));
  float ps0 = cell * 31.0 + 3.0 + th_hash(vec2(cell, 1.3)) * 16.0;
  float plen = 2.5 + th_hash(vec2(cell, 2.9)) * 7.0;
  float lane = th_hash(vec2(cell, 4.1)) > 0.5 ? 1.0 : -1.0;
  float pd0 = lane > 0.0 ? 0.15 + th_hash(vec2(cell, 5.5)) * 0.6 : -2.9 + th_hash(vec2(cell, 6.6)) * 0.4;
  float pw = 1.6 + th_hash(vec2(cell, 8.8)) * 1.3;
  // edges are saw cuts: straight, very slightly skewed
  float sk = (th_hash(vec2(cell, 9.9)) - 0.5) * 0.06;
  float ls = s - ps0 + sk * (d - pd0);
  float inPatch = step(0.45, hp) * step(0.0, ls) * step(ls, plen) * step(pd0, d) * step(d, pd0 + pw);
  float pb = min(min(ls, plen - ls), min(d - pd0, pd0 + pw - d));
  // transverse utility trench (0.6 m wide, full lane) every ~90 m
  float tc = floor((s + 40.0) / 93.0);
  float ts0 = tc * 93.0 - 40.0 + 20.0 + 50.0 * th_hash(vec2(tc, 3.1));
  float tw = 0.55 + 0.2 * th_hash(vec2(tc, 4.4));
  float tlane = th_hash(vec2(tc, 6.2)) > 0.4 ? 1.0 : 0.0;
  float trench = step(ts0, s) * step(s, ts0 + tw) * mix(step(d, 0.05), 1.0, tlane) * step(0.3, th_hash(vec2(tc, 8.1)));
  float tb = min(s - ts0, ts0 + tw - s);
  inPatch = max(inPatch, trench);
  pb = trench > 0.5 ? tb : pb;
  float seal = 0.0;
  if (inPatch > 0.5) {
    vec3 c3 = texture(uAsph, uv1 * 1.7 + 0.5).rgb;
    alb = mix(c3 * 0.62, c3 * 0.62 * vec3(0.98, 1.0, 1.03), trench);
    rough = min(rough, 0.82) * 0.93;
    tn = normalize(mix(tn, texture(uAsphN, uv1 * 1.7 + 0.5).xyz * 2.0 - 1.0, 0.75));
    tn.xy *= 0.7;                                    // finer, denser mix: less macro-texture
  }
  // tar overband on the cut edge (inside and outside the saw cut)
  {
    float eb = (inPatch > 0.5) ? pb : -1.0;
    if (eb < 0.0) {
      // outside: distance to the patch rectangle
      float ox = max(max(-ls, ls - plen), 0.0), oy = max(max(pd0 - d, d - pd0 - pw), 0.0);
      eb = step(0.45, hp) > 0.5 ? length(vec2(ox, oy)) : 9.0;
      if (s > ts0 - 0.2 && s < ts0 + tw + 0.2 && step(0.3, th_hash(vec2(tc, 8.1))) > 0.5 && (tlane > 0.5 || d < 0.1))
        eb = min(eb, max(ts0 - s, s - ts0 - tw));
    }
    float bw = 0.035 + 0.02 * th_vnoise(vec2(s, d) * 3.0);
    seal = 1.0 - smoothstep(bw - 0.01, bw + aaD, abs(eb));
  }

  // ---- cracks. Longitudinal lane-joint crack at the centre line, wheel-path cracks, transverse thermal cracks
  //      every ~6-15 m; most are crack-sealed (hot bitumen overband ~5-8 cm wide, glossy black, feathered edges
  //      where the squeegee thinned it), the rest are open (dark, thin, with fines washed into them)
  float crack = 0.0, tar = 0.0;
  // centre joint: sealed along most of its length
  float cj = rd_crackLine(d, s, 0.08, 11.0, 0.12);
  float cjOn = smoothstep(0.5, 0.62, th_vnoise(vec2(s * 0.05, 1.0)));
  tar = max(tar, (1.0 - smoothstep(0.022, 0.034 + aaD, cj)) * cjOn * (0.55 + 0.45 * th_vnoise(vec2(s * 1.7, 2.0))));
  crack = max(crack, (1.0 - smoothstep(0.003, 0.009 + aaD, cj)) * (1.0 - cjOn) * smoothstep(0.3, 0.45, th_vnoise(vec2(s * 0.05, 1.0))));
  // wheel-path longitudinal cracks (wander a little, open or sealed)
  float g2 = smoothstep(0.5, 0.65, th_vnoise(vec2(s * 0.04, 5.0)));
  float c2l = rd_crackLine(d, s, -2.3, 23.0, 0.35);
  float sealed2 = step(0.5, th_vnoise(vec2(s * 0.02, 7.0)));
  crack = max(crack, (1.0 - smoothstep(0.003, 0.01 + aaD, c2l)) * g2 * (1.0 - sealed2));
  tar = max(tar, (1.0 - smoothstep(0.028, 0.042 + aaD, c2l)) * g2 * sealed2);
  float g3 = smoothstep(0.5, 0.66, th_vnoise(vec2(s * 0.035, 13.0)));
  float c3l = rd_crackLine(d, s, 0.75, 51.0, 0.3);
  tar = max(tar, (1.0 - smoothstep(0.028, 0.042 + aaD, c3l)) * g3);
  // transverse thermal cracks: across the carriageway, slightly wavy, some only across one lane
  {
    float tcs = s / 9.0;
    float ti = floor(tcs + 0.5);
    float th0 = th_hash(vec2(ti, 21.0));
    float sc = (ti + (th0 - 0.5) * 0.6) * 9.0 + 0.7 * (th_vnoise(vec2(d * 0.55, ti)) - 0.5) + 0.18 * (th_vnoise(vec2(d * 2.1, ti + 7.0)) - 0.5)
             + 0.035 * (th_vnoise(vec2(d * 9.0, ti + 3.0)) - 0.5);
    float dist = abs(s - sc) / (0.7 + 0.6 * th_vnoise(vec2(d * 1.3, ti + 11.0)));
    float span = mix(step(d, 0.1), 1.0, step(0.35, th_hash(vec2(ti, 22.0))));
    // broken into segments (the crack wanders out of the sealant, dies out, restarts)
    span *= smoothstep(0.25, 0.4, th_vnoise(vec2(d * 0.9, ti + 5.0)));
    float on = step(0.3, th0) * span * (1.0 - smoothstep(2.75, 3.0, ad));
    float isSeal = step(0.35, th_hash(vec2(ti, 23.0)));
    // fwidth of s for AA
    float aaS = fwidth(s) + 0.002;
    tar = max(tar, (1.0 - smoothstep(0.028, 0.042 + aaS, dist)) * on * isSeal);
    crack = max(crack, (1.0 - smoothstep(0.003, 0.009 + aaS, dist)) * on * (1.0 - isSeal));
  }
  // alligator cracking in the outer wheel paths
  float allig = smoothstep(0.68, 0.8, th_fbm(vec2(d * 0.5, s * 0.06) + 3.3)) * smoothstep(1.6, 2.5, ad) * (1.0 - smoothstep(2.7, 2.9, ad));
  if (allig > 0.01) {
    vec2 q = vec2(d, s) * 5.5 + vec2(th_vnoise(vec2(d, s) * 2.0), th_vnoise(vec2(s, d) * 2.0 + 4.0)) * 0.6;
    vec2 v = rd_vor(q);
    float cw = 0.035 + aaD * 5.5;
    crack = max(crack, (1.0 - smoothstep(cw * 0.4, cw, v.y)) * allig * smoothstep(0.3, 0.6, th_vnoise(q * 0.7)));
  }
  crack *= 1.0 - inPatch;
  tar = max(tar, seal);
  // sealant: feathered (thin, partly worn to grey where tyres run), slight ridge at the band edge
  float tarWear = smoothstep(0.35, 0.8, th_vnoise(vec2(d * 4.0, s * 4.0))) * 0.5;
  alb = mix(alb, alb * 0.3, crack * 0.85);
  alb = mix(alb, mix(vec3(0.014), alb * 0.5, tarWear), tar * 0.95);
  rough = mix(rough, mix(0.3, 0.6, tarWear), tar);
  tn = normalize(mix(tn, vec3(0.0, 0.0, 1.0), tar * 0.8));
  ao *= 1.0 - 0.6 * crack;

  // ---- tire tracks: polished + darker wheel paths, slight ruts, oil drip in lane centre
  float trk = exp(-pow((ad - 0.72) / 0.28, 2.0)) + exp(-pow((ad - 2.28) / 0.28, 2.0));
  trk *= 0.7 + 0.3 * th_vnoise(vec2(d * 2.0, s * 0.2));
  alb *= 1.0 - 0.1 * trk;
  rough -= 0.12 * trk;
  float oil = exp(-pow((ad - 1.5) / 0.25, 2.0)) * smoothstep(0.4, 0.8, th_vnoise(vec2(d, s * 0.15)));
  alb *= 1.0 - 0.18 * oil;

  // ---- painted lines (edge at |d|=2.85, dashed centre 4.5 m / 12 m): old thermoplastic, faded to a dirty
  //      grey-cream (albedo ~0.35-0.45, not the 0.7 of fresh paint), glass-bead sheen gone, aggregate showing
  //      through, worn away where tyres cross it, grimy toward the shoulder
  float edgeL = 1.0 - smoothstep(0.06 - aaD, 0.06 + aaD, abs(ad - 2.85));
  float ph = mod(s + 2.0, 12.0);
  float dash = smoothstep(0.0, 0.04, ph) * (1.0 - smoothstep(4.5, 4.54, ph));
  float cenL = (1.0 - smoothstep(0.06 - aaD, 0.06 + aaD, ad)) * dash;
  float paint = max(edgeL, cenL);
  float wear = smoothstep(0.2, 0.62, th_fbm(vec2(d * 2.0, s * 0.45) + 1.7));
  float wear2 = smoothstep(0.25, 0.6, th_vnoise(vec2(d * 30.0, s * 9.0)));    // grain-scale pitting
  float aggr = smoothstep(0.15, 0.45, lum / 0.15);
  paint *= mix(0.12, 1.0, wear) * mix(0.35, 1.0, aggr) * mix(0.55, 1.0, wear2) * (1.0 - inPatch * 0.85) * (1.0 - tar);
  vec3 pcol = mix(vec3(0.44, 0.43, 0.39), vec3(0.34, 0.33, 0.3), smoothstep(2.9, 2.8, ad) * edgeL * 0.0 + smoothstep(0.4, 0.8, th_vnoise(vec2(d * 3.0, s * 0.8))));
  alb = mix(alb, pcol, paint * 0.9);
  rough = mix(rough, 0.6, paint);
  tn = normalize(mix(tn, vec3(0.0, 0.0, 1.0), paint * 0.45));

  // ---- mud / gravel washed onto the road: lobate tongues (debris-flow lobes 1.5-6 m wide with a sharp,
  //      slightly thicker front), wet glossy film, tyre tracks cutting through them
  float mud = 0.0;
  float flowZ = 0.0;   // sheet-flow strength (water running across the road)
  float gritZ = 0.0;   // loose stone debris density
  for (int i = 0; i < 6; i++) {
    vec4 z = uMudZones[i];
    float inZ = smoothstep(z.x - 6.0, z.x + 4.0, s) * (1.0 - smoothstep(z.y - 4.0, z.y + 6.0, s));
    if (inZ <= 0.0) continue;
    float lob = th_fbm(vec2(s * 0.16, float(i) * 3.1));
    float reach = (0.4 + 7.5 * lob * lob) * z.z * inZ;
    // lobate, digitate front: 0.5-1 m lobes on top of the 5-8 m tongues
    float edgeN = 0.55 * (th_vnoise(vec2(s * 0.9, float(i) * 7.0)) - 0.5) + 0.2 * (th_vnoise(vec2(s * 3.1, d * 2.0 + float(i))) - 0.5);
    float fromUp = smoothstep(3.1 - reach - 0.08, 3.1 - reach + 0.1, d + edgeN);
    float fromDn = smoothstep(-3.1 + reach * 0.5 + 0.08, -3.1 + reach * 0.5 - 0.1, d - edgeN);
    float m = max(fromUp, fromDn * (1.0 - z.w));
    // thinner inside the tongue (patchy film), thick at the front
    float front = fromUp * (1.0 - smoothstep(0.0, 0.35, d + edgeN - (3.1 - reach)));
    m *= mix(smoothstep(0.25, 0.6, th_fbm(vec2(d * 0.9, s * 0.3) + float(i)) + m * 0.35), 1.0, front);
    mud = max(mud, m);
    gritZ = max(gritZ, inZ * z.z * smoothstep(3.1 - reach * 1.5 - 1.0, 3.1 - reach * 0.8, d + edgeN));
    flowZ = max(flowZ, inZ * z.w);
  }
  // mud carried by tires just downstream of the scar
  mud = max(mud, trk * 0.55 * smoothstep(185.0, 195.0, s) * (1.0 - smoothstep(205.0, 290.0, s)) * smoothstep(0.35, 0.6, th_vnoise(vec2(d * 3.0, s * 0.4))));
  mud = max(mud, vMasks.g * 0.25 * smoothstep(545.0, 553.0, s) * (1.0 - smoothstep(567.0, 575.0, s)));
  // tyres cut two clean-ish tracks through the mud in the player lane (and more faintly in the other)
  float cut = exp(-pow((d + 0.78) / 0.2, 2.0)) + exp(-pow((d + 2.22) / 0.2, 2.0)) + 0.5 * (exp(-pow((d - 0.78) / 0.2, 2.0)) + exp(-pow((d - 2.22) / 0.2, 2.0)));
  mud *= 1.0 - 0.55 * clamp(cut, 0.0, 1.0) * smoothstep(0.3, 0.6, th_vnoise(vec2(d * 5.0, s * 0.5)));
  float mudFilm = 0.0;
  if (mud > 0.01) {
    vec2 mp = vec2(d, s);
    vec3 ma = texture(uAlb, vec3(mp / 1.55, 5.0)).rgb;
    vec3 mn = texture(uNrm, vec3(mp / 1.55, 5.0)).xyz;
    vec3 ga = texture(uAlb, vec3(mp / 2.0, 4.0)).rgb;
    float gm = smoothstep(0.4, 0.7, th_vnoise(mp * 0.8));
    vec3 mc = mix(ma * 0.8, ga, gm * 0.45);
    alb = mix(alb, mc, mud);
    tn = normalize(mix(tn, vec3((mn.xy * 2.0 - 1.0) * 0.8, 1.0), mud * 0.8));
    rough = mix(rough, 0.45, mud);
    mudFilm = mud * smoothstep(0.35, 0.65, th_vnoise(mp * 1.7 + 3.0));
  }

  // ---- loose angular stones / grit spilled on the asphalt. Debris sorts by size like a real deposit (power-law):
  //      a haze of sand and pea gravel (3-8 mm) everywhere in the zone, fewer 1-4 cm chips, clustered in streaks
  //      along the flow and at the tongue fronts; cobbles and bigger stones are 3D instances (terrain.js).
  float grit = 0.0;
  gritZ = max(gritZ, mud * 0.7);
  if (gritZ > 0.02) {
    // clustering: streaks drawn out across the road (the sheet flow carries it toward the valley edge)
    float clus = smoothstep(0.35, 0.75, th_vnoise(vec2(d * 0.6, s * 1.3)) * 0.6 + th_vnoise(vec2(d * 1.7, s * 3.1) + 5.0) * 0.4);
    // (1) chips: ~9 cm cells, sizes 1-4 cm, only some cells occupied
    vec2 q = vec2(d, s) * 11.0;
    vec2 gi = floor(q), gf = fract(q);
    float best = 9.0; vec2 bv = vec2(0.0); float bh = 0.0;
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
      vec2 g = vec2(float(x), float(y));
      float h = th_hash(gi + g + 31.0);
      vec2 o = vec2(th_hash(gi + g + 5.1), th_hash(gi + g + 9.3)) * 0.7 + 0.15;
      vec2 dv = gf - (g + o);
      // angular: max-norm blend in a rotated frame; size skewed small (h^2)
      float a = h * 6.2832; vec2 rv = vec2(cos(a) * dv.x + sin(a) * dv.y, -sin(a) * dv.x + cos(a) * dv.y);
      float fh = fract(h * 13.1);
      float r = mix(length(dv), max(abs(rv.x) * 1.25, abs(rv.y)), 0.65) / (0.12 + 0.4 * fh * fh);
      if (r < best) { best = r; bv = rv; bh = h; }
    }
    float dens = gritZ * (0.12 + 0.45 * clus);
    float stone = step(bh, dens) * (1.0 - smoothstep(0.75, 1.0, best));
    if (stone > 0.0) {
      // fresh broken gneiss / limestone, wet: albedo ~0.12-0.25 (the gray_rocks scan graded), ~45% mud-coated
      vec3 sa = texture(uAlb, vec3(vec2(d, s) / 1.8, 8.0)).rgb * mix(0.55, 1.15, fract(bh * 7.7));
      sa = mix(sa, vec3(0.1, 0.078, 0.058) * (0.8 + 0.4 * fract(bh * 17.3)), step(0.55, fract(bh * 3.3)) * 0.85);
      alb = mix(alb, sa, stone);
      tn = normalize(mix(tn, vec3(bv * 1.6, 1.0), stone));
      rough = mix(rough, 0.5, stone);
      ao *= 1.0 - 0.25 * smoothstep(0.8, 1.0, best) * stone;
    }
    // contact shadow ring around each chip
    ao *= 1.0 - 0.6 * step(bh, dens) * smoothstep(0.85, 1.0, best) * (1.0 - smoothstep(1.0, 1.35, best));
    grit = stone;
    // (2) sand and pea gravel: a speckle of 3-8 mm grains that greys and roughens the surface (no normal)
    float fine = th_hash(floor(vec2(d, s) * 140.0));
    float cov = gritZ * (0.25 + 0.5 * clus);
    float fineM = step(1.0 - cov, fine);
    // below the pixel footprint the grains average out (no shimmer): blend to the coverage fraction
    fineM = mix(fineM, cov, smoothstep(0.004, 0.012, fwidth(s) + fwidth(d))) * (1.0 - stone);
    alb = mix(alb, mix(vec3(0.16, 0.155, 0.15), vec3(0.09, 0.075, 0.06), step(0.5, fract(fine * 9.1))), fineM * 0.7);
    rough = mix(rough, 0.6, fineM * 0.5);
    grit = max(grit, fineM * 0.4);
  }

  // ---- tunnel: no rain inside, daylight falls off with depth
  rdTunnel = smoothstep(${TUNNEL.s0 - 0.5}, ${TUNNEL.s0 + 14}.0, s);
  rdDay = mix(1.0, 0.02, rdTunnel);
  float wet = uWet * (1.0 - smoothstep(${TUNNEL.s0}.0, ${TUNNEL.s0 + 6}.0, s));

  // ---- wetness + standing water. Water sits where the road holds it: in the (slightly rutted) wheel paths as
  //      long narrow puddles, along the kerb-side edges (the crown drains there), in patch-edge dips; sheets of
  //      water run across the road below the gullies and the slide.
  // wet asphalt darkens to ~half; saturated mud and fresh stone chips much less (their film is already water)
  alb *= 0.85 * mix(1.0, mix(0.52, 0.72, max(mud, grit)), wet);
  rough = mix(rough, mix(0.36, 0.24, trk), wet);
  float pmask = vMasks.g;
  float rut = exp(-pow((ad - 0.74) / 0.2, 2.0)) + exp(-pow((ad - 2.26) / 0.22, 2.0));
  float rutN = th_vnoise(vec2(d * 1.4, s * 0.11)) * 0.65 + th_vnoise(vec2(d * 3.0, s * 0.45) + 7.0) * 0.35;
  float pn = rut * (rutN + 0.12) + pmask * 0.38;
  // edge water: the crown sheds water to both edges (strongest on the uphill/inside edge of the bend)
  float edgeW = smoothstep(2.55, 2.95, ad) * (0.55 + 0.45 * th_vnoise(vec2(d * 2.0, s * 0.2)));
  pn = max(pn, edgeW * 0.9);
  // rare low spots (settled patch edges, a dip)
  pn = max(pn, smoothstep(0.72, 0.9, th_fbm(vec2(d * 0.6, s * 0.2) + 4.0)) * 0.95);
  float puddle = smoothstep(0.66, 0.74, pn) * wet;
  float damp = smoothstep(0.5, 0.66, pn) * wet;
  // water flowing across the road (thin sheet, 2-10 mm): below gullies / slide zones (flowZ), in lobes
  float sheet = 0.0;
  if (flowZ > 0.01) {
    float lobeS = th_fbm(vec2(s * 0.22, 11.0));
    sheet = flowZ * smoothstep(0.35, 0.55, lobeS + 0.15 * th_vnoise(vec2(d * 1.5, s * 1.5)));
    sheet *= wet;
  }
  alb *= 1.0 - 0.15 * damp;
  rough = mix(rough, 0.12, damp * 0.7);
  rdSpecBoost = 0.0;
  float water = max(puddle, sheet);
  if (water > 0.001) {
    // muddy water near the slide / gullies (suspended silt: warm, turbid)
    vec3 turb = vec3(0.075, 0.058, 0.038);
    float muddy = clamp(mud * 1.5 + flowZ * 0.8, 0.0, 1.0);
    alb = mix(alb, mix(alb * 0.35, turb, muddy * 0.8), water);
    rough = mix(rough, mix(0.02, 0.06, muddy), water);
    tn = normalize(mix(tn, vec3(0.0, 0.0, 1.0), water));
    // rain ripples on standing water
    float deep = smoothstep(0.45, 0.9, puddle / max(wet, 1e-3));
    if (deep > 0.001) {
      vec3 rp = rd_ripple(vec2(d, s) * 2.2, uTime) * 0.22;
      rp += rd_ripple(vec2(d, s) * 3.1 + 7.0, uTime * 1.13 + 0.5) * 0.16;
      tn = normalize(tn + vec3(rp.xy, 0.0) * deep);
    }
    // sheet flow: capillary wavelets elongated along the flow (toward the valley, -d), carried at ~0.6 m/s
    if (sheet > 0.001) {
      float t = uTime;
      vec2 q1 = vec2(d * 1.1 + t * 0.65, s * 5.0);
      vec2 q2 = vec2(d * 2.3 + t * 0.9, s * 11.0 + 3.0);
      float e = 0.05;
      float f0 = th_vnoise(q1) * 0.6 + th_vnoise(q2) * 0.4;
      float fs = th_vnoise(q1 + vec2(0.0, e * 5.0)) * 0.6 + th_vnoise(q2 + vec2(0.0, e * 11.0)) * 0.4;
      float fd = th_vnoise(q1 + vec2(e * 1.1, 0.0)) * 0.6 + th_vnoise(q2 + vec2(e * 2.3, 0.0)) * 0.4;
      vec2 gr = vec2(fd - f0, fs - f0) / e;
      tn = normalize(tn + vec3(-gr * vec2(0.08, 0.02), 0.0) * sheet);
      // small standing waves / foam lines where the sheet crosses a crack or the crown
      float foam = smoothstep(0.78, 0.95, f0) * sheet * (0.3 + 0.7 * max(tar, crack));
      alb = mix(alb, vec3(0.2, 0.19, 0.17), foam * 0.35);
    }
  }
  // wet mud film: glossy streaks on the tongues
  rough = mix(rough, 0.18, mudFilm * wet * (1.0 - water));
  // aggregate micro-occlusion: exposed stone chips shadow the water film between them, except where water
  // stands (puddles), the tyres have polished the surface, or paint / tar seal it
  rdSpecOcc = mix(0.55, 1.0, clamp(water / max(wet, 1e-3) + trk * 0.35 + paint * 0.5 + tar + mudFilm * 0.5, 0.0, 1.0));
  rdSpecOcc = mix(1.0, rdSpecOcc, wet);
  rdSpecOcc *= 1.0 - 0.3 * grit;
  rdTN = tn; rdRough = clamp(rough, 0.02, 1.0);
  rdAO = clamp(vMasks.r, 0.3, 1.0) * mix(1.0, ao, 0.6);
  rdAlb = alb * mix(1.0, vMasks.r, 0.08);
}
`;

// (s0, s1, mud strength, w). w = sheet-flow strength: water from the gully / slide running across the road. Also
// suppresses mud coming in from the valley edge (1 - w).
const MUD_ZONES = [
  [88, 200, 1.0, 0.45], [556 - 14, 560 + 14, 0.8, 0.3], [643, 657, 0.6, 1.0], [763, 777, 0.6, 1.0], [883, 897, 0.6, 1.0], [1003, 1017, 0.6, 1.0],
];

export async function createRoadMaterial(ctx) {
  const [tex, arr] = await Promise.all([ctx.assets.pbr('asphalt_02'), loadTerrainArrays(ctx)]);
  const m = new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0, color: 0xffffff });
  // setting normalMap makes three compute the UV tangent frame (tbn) we reuse
  m.normalMap = tex.normalMap;
  const uniforms = {
    uAsph: { value: tex.map }, uAsphN: { value: tex.normalMap }, uAsphArm: { value: tex.armMap },
    uAlb: { value: arr.alb }, uNrm: { value: arr.nrm },
    uWet: { value: ctx.env?.wetness ?? 0.75 }, uTime: { value: 0 },
    uMudZones: { value: MUD_ZONES.map((z) => new THREE.Vector4(...z)) },
    ...tunnelUniforms(ctx),
  };
  const rdDbg = +(new URLSearchParams(location.search).get('roadDebug') || 0);
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec4 masks;\nvarying vec2 vRoad;\nvarying vec4 vMasks;\nvarying vec3 vWPos;')
      .replace('#include <fog_vertex>', '#include <fog_vertex>\n vRoad = uv; vMasks = masks; vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + ROAD_FRAG_PARS)
      .replace('#include <map_fragment>', 'rd_shade(); diffuseColor.rgb = rdAlb;')
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = rdRough;')
      .replace('#include <normal_fragment_maps>', 'normal = normalize( tbn * rdTN );')
      .replace('#include <lights_fragment_begin>', lightsChunk('rdDay'))
      .replace('#include <fog_fragment>', fogChunk('rdTunnel') + (rdDbg ? `\n gl_FragColor = vec4(${rdDbg === 1 ? 'rdAlb * 2.0' : 'vec3(rdRough)'}, 1.0);` : ''))
      .replace('#include <aomap_fragment>', `{
        float ambientOcclusion = rdAO;
        reflectedLight.indirectDiffuse *= ambientOcclusion;
        #if defined( USE_ENVMAP ) && defined( STANDARD )
          float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
          reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
        #endif
        reflectedLight.indirectSpecular *= rdSpecOcc;
        // sky light (IBL) fades with depth; the sun is scaled in the lights loop; headlights stay at full strength
        reflectedLight.indirectDiffuse *= rdDay; reflectedLight.indirectSpecular *= rdDay;
        if (rdTunnel > 0.0) {
          vec3 nW = normalize((vec4(normal, 0.0) * viewMatrix).xyz);
          reflectedLight.indirectDiffuse += tunnelLamps(vWPos, nW) * BRDF_Lambert(material.diffuseColor) * smoothstep(0.0, 0.2, rdTunnel);
        }
      }`);
  };
  m.customProgramCacheKey = () => 'road-v3' + rdDbg;
  m.userData.uniforms = uniforms;
  m.userData.update = (dt) => {
    uniforms.uWet.value = ctx.env?.wetness ?? uniforms.uWet.value;
    uniforms.uTime.value += dt;
  };
  return m;
}

// ---------------------------------------------------------------------------------------------
// Tunnel (portal headwall, wing walls, tube)
// ---------------------------------------------------------------------------------------------
// Portal geometry (must match tools/blender/terrain.py build_tunnel): the headwall front face is 0.95 m before
// markers.tunnel, spans d = +-15.2 m and h = -0.8..16.4 m above the portal road; a 0.85 m concrete ring projects
// 0.25 m around the opening (opening half-width 4.9 m, springing line 1.9 m, arch radius 5.3 m, so the ring runs
// r = 4.9..5.75 m about (d 0, h 1.9)); coping box h 16.4..17.0; pilasters at |d| 15.2..15.8.
export const PORTAL = { front: -0.95, halfW: 15.2, top: 16.4, spring: 1.9, ringIn: 4.9, ringOut: 5.75, archH: 5.3, collar: 0.85, collarC: 3.2 };
// name plate (world metres in the portal plane; x = screen-right for the approaching driver)
const PLATE = { w: 6.6, h: 1.5, y0: 8.75 };
const F = (x) => x.toFixed(4);   // GLSL float literal

const TUN_FRAG_PARS = /* glsl */`
uniform sampler2D uCast;
uniform sampler2D uCastN;
uniform sampler2D uCastArm;
uniform sampler2DArray uAlb;
uniform sampler2DArray uNrm;
uniform sampler2D uPlate;
uniform float uWet;
uniform float uTTime;
uniform float uPortalY;
uniform vec3 uPortalP;
uniform vec3 uPortalL;
uniform vec3 uPortalT;
varying vec2 vTU;
varying vec4 vMasks;
varying vec3 vWPos;
${GLSL_NOISE}
${TUNNEL_GLSL}
float tnRough; float tnAO; vec3 tnTN; float tnDay = 1.0; float tnIn = 0.0;
vec3 tnT; vec3 tnB; vec3 tnN; vec3 tnNWg; float tnUseW = 0.0;   // world tangent frame (portal parts)
// 1 at the centre of a thin line of half-width w (m) at distance x, antialiased over the pixel footprint aa
// (energy-preserving: a line thinner than a pixel fades instead of widening to a full-strength pixel)
float tn_line(float x, float w, float aa) { return (1.0 - smoothstep(w, w + aa, x)) * (2.0 * w + 0.25 * aa) / (2.0 * w + aa); }
`;

// Portal-parts shading (REAL-WORLD MODEL: cast-in-place Alpine portal, 1950s-60s, in steady rain):
//  * Headwall: grey cast concrete (dry ~0.21 linear, wet ~0.11: porous concrete darkens 45-50 % when saturated) in
//    formwork panels 2.7 x 1.35 m (pour lifts every 2.7 m), each panel its own tone (a different pour / face of
//    the ply), 8 mm panel-joint fins, cold joints at the lifts that leak calcite, form-tie holes on a 0.9 x 1.35 m
//    grid (26 mm hole, 56 mm mortar plug, rust bleed below 60 % of them), a dark drip curtain under the coping and
//    runoff streaks (biofilm + dirt) that stay wet and glossier (roughness ~0.3 vs ~0.5), splash zone and moss at
//    the base. The name plate is a recessed precast panel with raised, once-painted letters.
//  * Ring: smoother precast arch ring with retroreflective delineator tiles (0.5 m red / white) around the arch
//    and red/white hazard boards on the vertical sides.
//  * Wing walls: older hand-laid coursed rubble (0.44 m courses, 0.4-0.9 m granite blocks, 2.5 cm recessed lime
//    joints), pillowed stone faces, black cyanobacteria streaks from the coping, lichen, moss in the joints.
//  * Drainage channel (JS mesh, part id 0.3): precast U-channel, water in the bottom (B mask) with rain rings.
const TUN_MAP = /* glsl */`{
  float tube = step(0.5, vMasks.a);
  float wing = (1.0 - tube) * step(0.425, vMasks.g) * step(vMasks.g, 0.6);
  float chan = (1.0 - tube) * (1.0 - step(0.35, vMasks.g));
  float h = vWPos.y - uPortalY;                       // height above the portal road level (m)
  vec3 c; vec3 arm;
  vec2 tuv = vTU;
  float rOv = -1.0;        // roughness override (< 0: none)
  float keepWet = 0.0;     // runs / streaks that stay wet (glossier)
  float dryish = 0.0;      // sheltered (dries lighter)
  float mossM = 0.0;
  // (derivatives in uniform control flow: inside the varying-dependent branches they are undefined)
  vec3 wdx = dFdx(vWPos), wdy = dFdy(vWPos);
  float aaW = max(max(length(wdx), length(wdy)), 1e-4);   // world metres per pixel
  // interpolated vertex normal, not the derivative one: fp32 derivatives at x ~1100 m jitter ~1 % per pixel quad
  tnN = normalize((vec4(vNormal, 0.0) * viewMatrix).xyz);
  vec3 wp = vWPos - uPortalP;
  if (tube > 0.5) {
    // cast concrete lining: rotate the scan so its pour lines run horizontally on the walls
    vec2 uv = vTU / 4.4;
    c = texture(uCast, uv).rgb; arm = texture(uCastArm, uv).rgb;
    tnTN = texture(uCastN, uv).xyz * 2.0 - 1.0;
    c *= 1.3;
    float n1 = th_fbm(vWPos.xz * 0.3 + vWPos.y * 0.2);
    c *= 0.88 + 0.24 * n1;
    float jd = abs(fract((vTU.y - ${TUNNEL.s0}.0) / 11.0 + 0.5) - 0.5) * 11.0;      // m to the nearest ring joint
    float joint = 1.0 - smoothstep(0.012, 0.03 + aaW, jd);
    c *= 1.0 - 0.35 * joint * (0.6 + 0.4 * th_vnoise(vTU * vec2(1.5, 0.3)));
    tnTN = normalize(mix(tnTN, vec3(sign(fract((vTU.y - ${TUNNEL.s0}.0) / 11.0 + 0.5) - 0.5), 0.0, 0.6), joint * 0.6));
    // exhaust soot toward the crown, blotchy
    float crown = smoothstep(6.5, 9.5, vTU.x) * (1.0 - smoothstep(14.0, 17.0, vTU.x)); // profile arc ~23.4 m, crown ~11.7
    c *= 1.0 - 0.45 * crown * (0.6 + 0.4 * th_vnoise(vTU * vec2(0.8, 0.15)));
    // road spray grime on the lower walls, tide marks / efflorescence near the kerb
    c = mix(c, c * vec3(0.4, 0.38, 0.35), vMasks.g * (0.6 + 0.4 * n1));
    float kerb = 1.0 - smoothstep(0.4, 1.4, min(vTU.x, 23.4 - vTU.x));
    c = mix(c, c * vec3(1.1, 1.08, 1.02), kerb * 0.35 * th_vnoise(vTU * vec2(1.3, 0.4)));
    float seep = smoothstep(0.66, 0.88, th_vnoise(vec2(vTU.x * 2.2 + vTU.y * 0.03, vWPos.y * 0.1)) * 0.8 + n1 * 0.3);
    c *= 1.0 - 0.28 * seep;
    keepWet = seep * 0.6;
  } else {
    // world frame of this face: T horizontal along it, B = N x T (up a vertical face; into the wall on a top face)
    vec3 tt = cross(vec3(0.0, 1.0, 0.0), tnN);
    tnT = dot(tt, tt) > 1e-3 ? normalize(tt) : -uPortalL;
    tnB = cross(tnN, tnT);
    tnUseW = 1.0;
    tuv = vec2(dot(wp, tnT), dot(wp, tnB));
    vec2 tgx = vec2(dot(wdx, tnT), dot(wdx, tnB)), tgy = vec2(dot(wdy, tnT), dot(wdy, tnB));
    float n1 = th_fbm(vWPos.xz * 0.3 + vWPos.y * 0.2);
    vec2 n2p = tuv * vec2(0.9, 0.25) + 3.0;
    float n2 = th_vnoise(n2p) * 0.65 + th_vnoise(n2p * 2.3 + 1.7) * 0.35;
    if (wing > 0.5) {
      float cope = smoothstep(0.55, 0.65, abs(tnN.y));   // the faces themselves are battered ~17 deg (|N.y| ~0.3)
      if (cope < 0.5) {
        // ---- coursed rubble masonry, irregular hand-laid joints
        vec2 wq = tuv + (vec2(th_vnoise(tuv * 2.3), th_vnoise(tuv * 2.3 + 5.2)) - 0.5) * 0.07;
        const float CH = 0.44, SL = 0.62;
        float row = floor(wq.y / CH);
        float yIn = wq.y - row * CH;
        float xr = wq.x + th_hash(vec2(row, 3.7)) * 1.3;
        float k = floor(xr / SL);
        float xIn = xr - k * SL;                               // 0..SL in cell k; joints jittered +-13 cm
        float Jk = (th_hash(vec2(k, row)) - 0.5) * 0.26, Jk1 = (th_hash(vec2(k + 1.0, row)) - 0.5) * 0.26;
        float ck = k, jl = Jk, jr = SL + Jk1;
        if (xIn < Jk) { ck = k - 1.0; jr = Jk; jl = -SL + (th_hash(vec2(k - 1.0, row)) - 0.5) * 0.26; }
        else if (xIn > SL + Jk1) { ck = k + 1.0; jl = SL + Jk1; jr = 2.0 * SL + (th_hash(vec2(k + 2.0, row)) - 0.5) * 0.26; }
        float exL = xIn - jl, exR = jr - xIn, eyB = yIn, eyT = CH - yIn;
        float e = min(min(exL, exR), min(eyB, eyT));
        vec2 sid = vec2(ck, row);
        float hs = th_hash(sid + 0.37);
        // stone face: rock-face scans (lichen_rock 2.0 m / dark_rock_02 2.0 m capture), each block its own crop,
        // tone and iron staining; weathered granite / gneiss ashlar ~0.12-0.22 dry
        float lay = th_hash(sid + 6.6) < 0.7 ? 1.0 : 6.0;
        vec2 suv = (tuv + vec2(th_hash(sid) * 7.0, th_hash(sid + 1.3) * 7.0)) / 2.0;
        vec3 sa = textureGrad(uAlb, vec3(suv, lay), tgx / 2.0, tgy / 2.0).rgb;
        vec4 sn = textureGrad(uNrm, vec3(suv, lay), tgx / 2.0, tgy / 2.0);
        float sl = dot(sa, vec3(0.3, 0.59, 0.11));
        sa = mix(vec3(sl), sa, 0.4) * (lay < 2.0 ? 3.2 : 4.4) * (0.55 + 0.6 * hs);
        sa *= mix(vec3(1.0), vec3(1.15, 1.0, 0.8), step(0.75, th_hash(sid + 4.4)));
        sa *= mix(vec3(1.0), vec3(0.85, 0.88, 0.94), step(0.8, th_hash(sid + 9.1)));
        vec3 stn = vec3(sn.xy * 2.0 - 1.0, 1.0);
        // pillowed faces: the block rounds off into the joint (normal tilts toward the nearest edge)
        // beyond a pixel per joint the pattern is averaged instead of aliasing (grazing views of the flared walls)
        float far = smoothstep(0.015, 0.05, aaW);
        float pw = 0.07 + 0.06 * th_hash(sid + 8.8), pa = (0.2 + 0.25 * th_hash(sid + 3.9)) * (1.0 - far);
        stn.x += pa * ((1.0 - smoothstep(0.0, pw, exR)) - (1.0 - smoothstep(0.0, pw, exL)));
        stn.y += pa * ((1.0 - smoothstep(0.0, pw, eyT)) - (1.0 - smoothstep(0.0, pw, eyB)));
        // recessed lime mortar, dirty and wet (albedo ~0.14), 2.5 cm joints
        float mort = mix(1.0 - smoothstep(0.0125, 0.0125 + aaW * 1.5, e), 0.1, far);
        vec3 mc = vec3(0.15, 0.145, 0.13) * (0.7 + 0.6 * th_vnoise(tuv * 9.0));
        c = mix(sa, mc, mort);
        tnTN = normalize(mix(stn, vec3(0.0, 0.0, 1.0), mort));
        arm = vec3(mix(sn.a, 0.45, mort), mix(sn.b, 0.85, mort), 0.0);
        // ---- older weathering: black cyanobacteria streaks from the coping, lichen, moss in the joints
        float s = vTU.x;                                        // wing-wall UV: (s, height above the wall base)
        float hgt = 1.6 + 15.2 * pow(clamp((s - 1131.0) / 18.5, 0.0, 1.0), 1.2);
        float dTop = hgt - vTU.y;
        float stc = th_vnoise(vec2(tuv.x * 4.0, tuv.y * 0.06)) * 0.65 + th_vnoise(vec2(tuv.x * 13.0, tuv.y * 0.2)) * 0.35;
        float runs = smoothstep(0.5, 0.78, stc) * (1.0 - smoothstep(1.5, 2.5 + 7.0 * th_vnoise(vec2(tuv.x * 1.7, 2.0)), dTop));
        float curtain = 1.0 - smoothstep(0.0, 0.5 + 0.9 * th_vnoise(vec2(tuv.x * 1.3, 7.0)), dTop);
        float blk = max(runs * 0.85, curtain * 0.7);
        c *= 1.0 - 0.6 * blk;
        keepWet = max(runs, curtain);
        float lich = smoothstep(0.62, 0.78, th_vnoise(tuv * 1.6 + 11.0) * 0.65 + th_vnoise(tuv * 3.7 + 2.0) * 0.35) * (1.0 - mort) * smoothstep(0.8, 2.0, vTU.y) * (1.0 - blk);
        c = mix(c, vec3(0.2, 0.21, 0.17) * (0.8 + 0.4 * th_vnoise(tuv * 20.0)), lich * 0.7);
        float mv = th_vnoise(tuv * 2.1 + 4.0) * 0.65 + th_vnoise(tuv * 4.9 + 1.0) * 0.35;
        float mj = mort * smoothstep(0.35, 0.6, mv) * (1.0 - smoothstep(1.5, 4.0 + 3.0 * n2, vTU.y) * 0.6);
        float mb = (1.0 - smoothstep(0.3, 1.1 + 0.5 * n2, vTU.y)) * smoothstep(0.35, 0.65, mv);
        mossM = max(mj, mb);
        c = mix(c, vec3(0.028, 0.042, 0.018) * (0.7 + 0.6 * th_vnoise(tuv * 14.0)), mossM * 0.85);
        // splash / soil zone at the foot
        c *= mix(1.0, 0.55, 1.0 - smoothstep(0.2, 0.9 + 0.4 * n2, vTU.y));
      } else {
        vec2 uv = tuv / 3.8, gx = tgx / 3.8, gy = tgy / 3.8;
        // (QA) the sky-facing copings hold the most water: the wettest, darkest concrete on a real portal
        c = textureGrad(uCast, uv, gx, gy).rgb * 1.3; arm = textureGrad(uCastArm, uv, gx, gy).rgb;
        tnTN = textureGrad(uCastN, uv, gx, gy).xyz * 2.0 - 1.0;
        float lum = dot(c, vec3(0.3, 0.59, 0.11));
        c = mix(vec3(lum), c, 0.4) * (0.75 + 0.4 * n1);
        // the sloped copings carry the runoff of the slope above: dark wet runs down the fall line, lichen crusts,
        // moss cushions in the rough patches
        float fr = smoothstep(0.45, 0.75, th_vnoise(vec2(tuv.x * 3.5, tuv.y * 0.15)) * 0.7 + th_vnoise(vec2(tuv.x * 11.0, tuv.y * 0.4)) * 0.3);
        c *= 1.0 - 0.35 * fr;
        float lich = smoothstep(0.6, 0.75, th_fbm(tuv * 1.4 + 7.0)) * (1.0 - fr);
        c = mix(c, vec3(0.2, 0.21, 0.17) * (0.8 + 0.4 * th_vnoise(tuv * 20.0)), lich * 0.6);
        mossM = smoothstep(0.3, 0.58, th_fbm(tuv * 0.9 + 3.0));
        c = mix(c, vec3(0.03, 0.042, 0.02) * (0.7 + 0.6 * th_vnoise(tuv * 12.0)), mossM * 0.85);
        keepWet = fr * 0.6;
        rOv = mix(mix(0.62, 0.42, fr * uWet), 0.8, mossM);   // gritty, lichen-crusted, mossy: diffuse, not a sky mirror
      }
    } else if (chan > 0.5) {
      // ---- drainage channel: precast concrete U, water in the bottom
      vec2 uv = tuv / 1.3;
      c = textureGrad(uCast, uv, tgx / 1.3, tgy / 1.3).rgb * 1.25; arm = textureGrad(uCastArm, uv, tgx / 1.3, tgy / 1.3).rgb;
      tnTN = textureGrad(uCastN, uv, tgx / 1.3, tgy / 1.3).xyz * 2.0 - 1.0;
      float lum = dot(c, vec3(0.3, 0.59, 0.11));
      c = mix(vec3(lum), c, 0.35) * mix(1.0, 0.6, 1.0 - vMasks.r);
      // silt and algae in the wetted perimeter
      float alg = smoothstep(0.4, 0.7, th_fbm(tuv * 3.0)) * (1.0 - vMasks.r);
      c = mix(c, vec3(0.035, 0.04, 0.025), alg * 0.7);
      keepWet = 1.0 - vMasks.r;
      rOv = mix(0.62, 0.4, keepWet);                     // rims: gritty, trafficked by runoff silt
      if (vMasks.b > 0.5) {
        // shallow running water: near-black albedo (silt-laden), a smooth surface broken by rain rings
        c = vec3(0.022, 0.021, 0.018);
        vec2 rp = tuv * 7.0;
        vec2 rc = floor(rp);
        vec2 rf = fract(rp) - 0.5 - (vec2(th_hash(rc), th_hash(rc + 3.1)) - 0.5) * 0.5;
        float ph = fract(uTTime * 1.3 + th_hash(rc + 7.7));
        float ring = sin(clamp((length(rf) - ph * 0.45) * 40.0, -3.14, 3.14)) * (1.0 - ph) * step(length(rf), ph * 0.45 + 0.08);
        tnTN = normalize(vec3(rf * ring * 0.6 + vec2(sin(tuv.x * 3.0 + uTTime * 2.0), 0.0) * 0.03, 1.0));
        rOv = 0.12;
        arm = vec3(1.0, 0.12, 0.0);
      }
    } else {
      // ---- headwall, ring, coping, pilasters (cast concrete)
      float px = -dot(wp, uPortalL), pt = dot(wp, uPortalT), ph = wp.y;
      float front = smoothstep(0.75, 0.9, -dot(tnN, uPortalT));
      float rr = ph > ${F(PORTAL.spring)} ? length(vec2(px, ph - ${F(PORTAL.spring)})) : abs(px);
      // the ring front is the only face this far forward below the coping (terrain.py: collar = opening + 0.85 m
      // radially from (0, 3.2), so its outer edge is at ~1.165 x the opening's elliptical radius)
      float ring = front * step(pt, -1.08) * step(ph, 12.0);
      float er = ph > ${F(PORTAL.spring)} ? length(vec2(px / ${F(PORTAL.ringIn)}, (ph - ${F(PORTAL.spring)}) / 5.3)) : abs(px) / ${F(PORTAL.ringIn)};
      float face = front * (1.0 - ring) * step(abs(px), ${F(PORTAL.halfW - 0.02)}) * step(ph, ${F(PORTAL.top - 0.02)});
      // formwork panels 2.7 x 1.35 m: each samples its own crop of the scan (tone steps panel to panel)
      vec2 pc = floor(vec2(px / 2.7, ph / 1.35));
      vec2 off = face > 0.5 ? vec2(th_hash(pc + 0.5), th_hash(pc + 7.1)) * 3.0 : vec2(0.0);
      vec2 uv = (tuv + off) / 2.6, gx = tgx / 2.6, gy = tgy / 2.6;
      c = textureGrad(uCast, uv, gx, gy).rgb; arm = textureGrad(uCastArm, uv, gx, gy).rgb;
      tnTN = textureGrad(uCastN, uv, gx, gy).xyz * 2.0 - 1.0;
      // concrete_wall_006 is a brown scan (mean 0.107/0.083/0.060): neutral cement grey, ~0.21 dry
      float lum = dot(c, vec3(0.3, 0.59, 0.11));
      c = mix(vec3(lum), c, 0.3) * vec3(2.36, 2.3, 2.2);
      c *= mix(1.0, 0.97 + 0.06 * th_hash(pc + 3.3), face);
      // large-scale mottling: dirt, lime haze, patchy drying
      c *= 0.82 + 0.3 * n1 + 0.12 * (th_vnoise(tuv * 0.35 + 9.0) - 0.5);
      if (face > 0.5) {
        float ax = aaW * 1.3;
        float jx = abs(fract(px / 2.7 + 0.5) - 0.5) * 2.7, jy = abs(fract(ph / 1.35 + 0.5) - 0.5) * 1.35;
        float lift = abs(fract(ph / 2.7 + 0.5) - 0.5) * 2.7;
        float lv = tn_line(jx, 0.004, ax), lh = tn_line(jy, 0.004, ax);
        float lk = tn_line(lift, 0.007, ax);
        // joints only where the panels actually leaked grout: patchy along each line
        float leak = 0.4 + 0.6 * th_vnoise(vec2(px, ph) * 1.7);
        c *= 1.0 - (0.12 * lv + 0.07 * lh + 0.16 * lk) * leak;
        // panel-joint fins: a raised grout line with a shadowed side
        tnTN.x += 0.25 * tn_line(jx, 0.008, ax) * sign(fract(px / 2.7 + 0.5) - 0.5) * leak;
        tnTN.y += (0.12 * tn_line(jy, 0.008, ax) + 0.3 * lk) * sign(fract(ph / 1.35 + 0.5) - 0.5) * leak;
        // cold joints leak calcite: pale crusts bleeding down from the lift line in patches
        float dB = (floor(ph / 2.7) + 1.0) * 2.7 - ph;          // m below the lift joint above
        float calc = smoothstep(0.6, 0.85, th_vnoise(vec2(px * 2.4, floor(ph / 2.7) * 5.3))) * exp(-dB / (0.2 + 0.5 * th_vnoise(vec2(px * 7.0, 1.0))));
        calc *= step(ph, ${F(PORTAL.top - 0.4)});
        c = mix(c, vec3(0.34, 0.34, 0.325), calc * 0.55);
        // form-tie holes, 0.9 x 1.35 m grid (panel-local x 0.45 / 1.35 / 2.25, y 0.675)
        vec2 tcell = vec2(0.9, 1.35);
        vec2 tci = floor(vec2(px, ph) / tcell);
        vec2 tq = vec2(px, ph) - (tci + 0.5) * tcell;
        float td = length(tq);
        float hole = tn_line(td, 0.013, aaW);
        float plug = tn_line(td, 0.028, aaW) * (1.0 - tn_line(td, 0.013, aaW));
        float th = th_hash(tci + 2.9);
        // most ties were plugged with mortar cones (a slightly different grey); a few are open, dark holes
        c = mix(c, c * (th > 0.2 ? 1.07 : 0.75), plug * 0.7);
        c = mix(c, c * mix(0.3, 0.9, step(0.2, th)), hole);
        tnTN.xy += (tq / max(td, 1e-3)) * 0.8 * (tn_line(abs(td - 0.028), 0.004, aaW)) * (th > 0.5 ? 1.0 : -1.0);
        float below = -tq.y;
        float rust = step(0.4, th) * step(0.0, below) * tn_line(abs(tq.x), 0.008 + below * 0.05, aaW)
                   * exp(-below / (0.12 + 0.5 * th_hash(tci + 5.3))) * step(0.02, below + 0.02);
        c = mix(c, vec3(0.2, 0.085, 0.03), rust * 0.6);
        // name plate: recessed precast panel with raised letters (canvas: RG normal, B panel / letters)
        vec2 puv = vec2((px + ${F(PLATE.w / 2)}) / ${F(PLATE.w)}, (ph - ${F(PLATE.y0)}) / ${F(PLATE.h)});
        float inP = step(0.0, puv.x) * step(puv.x, 1.0) * step(0.0, puv.y) * step(puv.y, 1.0);
        if (inP > 0.5) {
          vec4 pl = textureGrad(uPlate, puv, tgx / vec2(${F(PLATE.w)}, ${F(PLATE.h)}), tgy / vec2(${F(PLATE.w)}, ${F(PLATE.h)}));
          float plP = min(1.0, pl.b * 2.0), plL = max(0.0, pl.b * 2.0 - 1.0);
          c = mix(c, c * 0.78, plP);                                        // denser, darker precast panel
          vec3 paint = vec3(0.36, 0.355, 0.33) * (0.55 + 0.45 * th_vnoise(vec2(px, ph) * 11.0));
          c = mix(c, paint, plL * 0.85);                                     // once-white paint, weathered
          tnTN.xy += (pl.rg * 2.0 - 1.0) * 1.4;
        }
      }
      if (ring > 0.5) {
        c *= 1.08;
        // retroreflective delineator tiles around the arch: 0.5 m red / white along the ring
        float arc = ph > ${F(PORTAL.spring)} ? atan(px, ph - ${F(PORTAL.spring)}) * 5.45 : sign(px) * (8.56 + ${F(PORTAL.spring)} - ph);
        float eaa = aaW / 5.0;
        float band = smoothstep(1.05 - eaa, 1.05, er) * (1.0 - smoothstep(1.115, 1.115 + eaa, er)) * step(0.3, ph);
        float gap = tn_line(abs(fract(arc / 0.5 + 0.5) - 0.5) * 0.5, 0.012, aaW);   // tile gaps
        float seg = step(0.5, fract(arc / 1.0));
        // hazard boards on the vertical sides: 45 deg red/white stripes over the full ring width
        float side = step(ph, ${F(PORTAL.spring)}) * step(0.3, ph) * smoothstep(${F(PORTAL.ringIn)} + 0.05, ${F(PORTAL.ringIn)} + 0.05 + aaW, rr);
        float diag = step(0.5, fract((ph + abs(px)) / 0.4));
        float isStrip = max(band, side);
        float red = mix(seg, diag, side);
        vec3 sc = mix(vec3(0.46, 0.46, 0.44), vec3(0.24, 0.028, 0.02), red);
        sc *= 0.62 + 0.38 * th_vnoise(vec2(arc, ph) * 6.0);                  // road film and grime
        sc = mix(sc, sc * vec3(0.6, 0.55, 0.5), smoothstep(0.55, 0.8, th_vnoise(vec2(arc * 0.8, ph * 3.0))));
        sc *= 1.0 - 0.5 * (1.0 - smoothstep(0.3, 1.2, ph));                 // spray at the foot
        c = mix(c, sc, isStrip * (1.0 - gap));
        rOv = mix(rOv, 0.3, isStrip);
        tnTN = mix(tnTN, vec3(0.0, 0.0, 1.0), isStrip * 0.8);
      }
      // coping top / pilaster tops: sky-facing, wettest, mossy
      float top = smoothstep(0.6, 0.8, tnN.y);
      if (top > 0.01) mossM = top * smoothstep(0.45, 0.7, th_fbm(tuv * 1.1 + 5.0));
      // runoff from the coping: a dark drip curtain with a ragged lower edge, then narrow biofilm runs
      float dTop = ${F(PORTAL.top)} - ph;
      float curtain = (1.0 - smoothstep(0.0, 0.8 + 2.4 * th_vnoise(vec2(px * 0.7, 3.1)) + 1.6 * th_vnoise(vec2(px * 3.1, 1.7)), dTop)) * step(0.0, dTop);
      float stc = th_vnoise(vec2(px * 5.0, ph * 0.07)) * 0.6 + th_vnoise(vec2(px * 16.0, ph * 0.22)) * 0.4;
      float sLen = 4.0 + 16.0 * th_vnoise(vec2(px * 1.9, 0.5));
      float runs = smoothstep(0.48, 0.78, stc) * (1.0 - smoothstep(sLen * 0.5, sLen, dTop)) * step(0.0, dTop);
      // water shed around the ring drips off its underside at the springing: dark wet fans beside the opening
      float fan = front * (1.0 - ring) * (1.0 - smoothstep(0.0, 1.2, abs(abs(px) - 6.3))) * (1.0 - smoothstep(1.0, 3.5, ph));
      float blk = max(max(curtain * 0.75, runs), fan * 0.6) * (1.0 - top);
      c *= 1.0 - 0.62 * blk;
      // (sky-facing ledges: gritty, not a sky mirror; a thin glossy ledge aliases into a dotted highlight)
      keepWet = max(blk, top * 0.35);
      // splash zone and moss / algae at the foot
      float base = 1.0 - smoothstep(0.15, 1.3 + 0.6 * n2, ph);
      c = mix(c, c * vec3(0.45, 0.46, 0.4), base * 0.8);
      mossM = max(mossM, base * smoothstep(0.42, 0.68, n2) * (1.0 - ring));
      c = mix(c, vec3(0.03, 0.045, 0.02) * (0.7 + 0.6 * th_vnoise(tuv * 12.0)), mossM * 0.75);
    }
  }
  float wet = uWet * (1.0 - smoothstep(${TUNNEL.s0 - 1}.0, ${TUNNEL.s0 + 6}.0, vTU.y) * tube);
  // porous concrete / mortar darkens ~45 % when saturated, the granite blocks ~30 %
  c *= mix(1.0, mix(0.55, 0.7, wing), wet);
  tnRough = mix(arm.g, 0.46, wet * 0.75);
  tnRough = mix(tnRough, 0.3, keepWet * wet);
  tnRough = mix(tnRough, 0.72, mossM);
  if (rOv >= 0.0) tnRough = rOv;
  tnTN = normalize(tnTN);
  if (tnUseW > 0.5) tnNWg = normalize(tnT * tnTN.x + tnB * tnTN.y + tnN * tnTN.z);
  // outside the tube the 30 m Cycles AO over-darkens the portal slot (no bounce light): floor it
  tnAO = chan > 0.5 ? arm.r * vMasks.r : arm.r * mix(mix(0.55, 1.0, vMasks.r), vMasks.r, tube);
  diffuseColor.rgb = c;
  // daylight left at this fragment (uv.y = s in the tube and on the end cap; headwall / wings are outside)
  tnIn = smoothstep(${TUNNEL.s0 - 0.5}, ${TUNNEL.s0 + 14}.0, vTU.y) * tube;
  tnDay = mix(1.0, 0.015, tnIn);
}`;

/**
 * Name plate texture, drawn procedurally: RG = tangent-space normal (x right, y up), B = 0.5 * recessed panel +
 * 0.5 * raised letters, A = 1. Fictional name; bilingual as in South Tyrol / Ticino.
 */
function makePlateTexture() {
  const W = 1024, H = Math.round(1024 * PLATE.h / PLATE.w);
  const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const g = cv.getContext('2d', { willReadFrequently: true });
  const hm = (draw, blur) => {
    g.save(); g.filter = 'none'; g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    g.filter = blur ? `blur(${blur}px)` : 'none';
    draw(); g.restore();
    const d = g.getImageData(0, 0, W, H).data, o = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) o[i] = d[i * 4] / 255;
    return o;
  };
  const px = W / PLATE.w;   // pixels per metre
  const inset = 0.07 * px;
  const panel = hm(() => { g.fillStyle = '#fff'; g.fillRect(inset, inset, W - 2 * inset, H - 2 * inset); }, 1.2);
  const text = (fill) => {
    g.fillStyle = fill; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    const f1 = Math.round(0.5 * px), f2 = Math.round(0.34 * px);
    g.font = `700 ${f1}px "DIN Condensed", "Arial Narrow", "Helvetica Neue", Arial, sans-serif`;
    try { g.letterSpacing = `${Math.round(0.06 * px)}px`; } catch { /* older canvas */ }
    g.fillText('GALLERIA ROCCANERA', W / 2, H * 0.56);
    g.font = `600 ${f2}px "DIN Condensed", "Arial Narrow", "Helvetica Neue", Arial, sans-serif`;
    try { g.letterSpacing = `${Math.round(0.08 * px)}px`; } catch { /* older canvas */ }
    g.fillText('TUNNEL  ·  1961', W / 2, H * 0.86);
  };
  const letters = hm(() => text('#fff'), 1.4);
  const out = g.createImageData(W, H), o = out.data;
  // height: frame 0.02 m proud, panel recessed 0.02 m, letters 0.015 m proud of the panel
  const hgt = (i) => (1 - panel[i]) * 0.02 + letters[i] * 0.015;
  const k = px * 0.5;   // slope per pixel difference, in metres per metre
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const xl = y * W + Math.max(0, x - 1), xr = y * W + Math.min(W - 1, x + 1);
      const yu = Math.max(0, y - 1) * W + x, yd = Math.min(H - 1, y + 1) * W + x;
      const dx = (hgt(xr) - hgt(xl)) * k, dy = (hgt(yu) - hgt(yd)) * k;   // canvas y runs down, texture v up
      let nx = -dx, ny = -dy;
      const l = Math.hypot(nx, ny, 1); nx /= l; ny /= l;
      o[i * 4] = Math.round((nx * 0.5 + 0.5) * 255);
      o[i * 4 + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      // B packs both masks (letters only occur inside the panel): 0.5 * panel + 0.5 * letters. A stays opaque:
      // canvas 2D is premultiplied, so A = 0 texels would lose their normal (RG) on readback / upload
      o[i * 4 + 2] = Math.round((0.5 * panel[i] + 0.5 * Math.min(1, letters[i] * 1.6)) * 255);
      o[i * 4 + 3] = 255;
    }
  }
  g.putImageData(out, 0, 0);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.NoColorSpace;
  t.premultiplyAlpha = false;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.anisotropy = 8;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.needsUpdate = true;
  return t;
}

/**
 * Tunnel material: portal headwall + ring + coping (cast concrete in formwork panels, tie holes, calcite and rust,
 * runoff streaks, a name plate, delineator tiles), wing walls (older coursed rubble masonry), the drainage channel
 * (part id 0.3) and the tube lining. COLOR_0: R = AO, G = grime (tube kerb) / part id (0.3 channel, 0.4 headwall,
 * 0.45 wing walls), B = water (channel), A = 1 in the tube.
 */
export async function createTunnelMaterial(ctx) {
  const [cast, arr] = await Promise.all([ctx.assets.pbr('concrete_wall_006'), loadTerrainArrays(ctx)]);
  const m = new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0, color: 0xffffff });
  m.normalMap = cast.normalMap;
  let plate = null;
  try { plate = makePlateTexture(); } catch (e) { console.warn('[tunnel] name plate', e); }
  if (!plate) { plate = new THREE.DataTexture(new Uint8Array([128, 128, 0, 255]), 1, 1); plate.needsUpdate = true; }
  const road = ctx.road;
  const uniforms = {
    uCast: { value: cast.map }, uCastN: { value: cast.normalMap }, uCastArm: { value: cast.armMap },
    uAlb: { value: arr.alb }, uNrm: { value: arr.nrm }, uPlate: { value: plate },
    uWet: { value: ctx.env?.wetness ?? 0.75 },
    uTTime: { value: 0 },
    uPortalY: { value: road ? road.pointAt(TUNNEL.s0).y : 0 },
    uPortalP: { value: road ? road.pointAt(TUNNEL.s0) : new THREE.Vector3() },
    uPortalL: { value: road ? road.leftAt(TUNNEL.s0) : new THREE.Vector3(0, 0, -1) },
    uPortalT: { value: road ? (() => { const t = road.tangentAt(TUNNEL.s0); t.y = 0; return t.normalize(); })() : new THREE.Vector3(1, 0, 0) },
    ...tunnelUniforms(ctx),
  };
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec4 masks;\nvarying vec2 vTU;\nvarying vec4 vMasks;\nvarying vec3 vWPos;')
      .replace('#include <fog_vertex>', '#include <fog_vertex>\n vTU = uv; vMasks = masks; vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + TUN_FRAG_PARS)
      .replace('#include <map_fragment>', TUN_MAP)
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = tnRough;')
      .replace('#include <normal_fragment_maps>', 'normal = tnUseW > 0.5 ? normalize( ( viewMatrix * vec4( tnNWg, 0.0 ) ).xyz ) : normalize( tbn * tnTN );')
      .replace('#include <lights_fragment_begin>', lightsChunk('tnDay'))
      .replace('#include <fog_fragment>', fogChunk('tnIn'))
      .replace('#include <aomap_fragment>', `{
        reflectedLight.indirectDiffuse *= tnAO;
        reflectedLight.indirectSpecular *= mix(1.0, tnAO, 0.7);
        // sky light (IBL) fades with depth; the sun is scaled in the lights loop; headlights stay at full strength
        reflectedLight.indirectDiffuse *= tnDay; reflectedLight.indirectSpecular *= tnDay;
        if (vMasks.a > 0.5) {
          vec3 nW = normalize((vec4(normal, 0.0) * viewMatrix).xyz);
          reflectedLight.indirectDiffuse += tunnelLamps(vWPos, nW) * BRDF_Lambert(material.diffuseColor);
        }
      }`);
  };
  m.customProgramCacheKey = () => 'tunnel-v5';
  m.userData.uniforms = uniforms;
  m.userData.update = (dt = 0) => {
    uniforms.uWet.value = ctx.env?.wetness ?? uniforms.uWet.value;
    uniforms.uTTime.value = (uniforms.uTTime.value + (dt || 0)) % 1000;
  };
  return m;
}
