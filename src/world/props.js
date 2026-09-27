// Props system (PROPS workstream): roadworks site at the pull-off, guardrails along the valley edge, delineator
// posts, warning signs, and the plank bridge over the washed-out trench. Every mesh comes from
// public/assets/models/props.glb (built from scratch by tools/blender/props.py).
//
// API (DESIGN.md "props"):
//   items: { jerrycan, hatchet, planks }   Object3Ds in the world that GAME registers with interact (always defined,
//                                          even if the GLB failed to load: then they are empty Groups at the right spot)
//   hideItem(id)                           hides an item (and removes its collider, e.g. the plank stack)
//   showItem(id)                           (extra) undo hideItem, for checkpoint restarts
//   placePlanks() -> Object3D              builds the plank bridge over the trench at markers.gap (11 boards, ~2.5 m
//                                          deck centred on d = -1.5, spanning along +s, end ramps) + PROP colliders.
//                                          Idempotent: a second call returns the same bridge.
//   removePlanks()                         (extra) removes the bridge again (checkpoint restarts)
//   bridge: Object3D | null                the placed bridge
//   lightTower: Object3D, lampLight: SpotLight, setLampOn(bool)   (extra)
//   sites: { roadworksS, jerrycan: {s,d}, hatchet: {s,d}, planks: {s,d} }  (extra, road-space item spots)
//
// Guardrails: scatter.json `guardrails` runs at the valley edge (fallback: everything except the fallen-tree crown,
// the washout and the tunnel). The pull-off (markers.pulloff) edge is generated here from the same bench shape the
// terrain uses, so the rail swings out around the gravel pull-off continuously. Rails are instanced in ~64 m chunks
// (frustum-culled) with a hi/lo LOD swap by camera distance; thin static box colliders per segment.
//
// Colliders: barriers, guardrail, light tower, crates, tarp pile and the plank stack are STATIC boxes. The plank bridge
// is G.PROP (so the car wheels and the player walk on it, and footsteps report 'wood'). Traffic cones on the road are
// light dynamic bodies with membership bit G.SENSOR (non-sensor colliders): the car chassis knocks them over, but the
// wheel ray casts (STATIC|ROCK|DEBRIS|PROP) never climb onto them.
import * as THREE from 'three';
import { G, groups } from '../physics/world.js';

const Y = new THREE.Vector3(0, 1, 0);
const _v = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3();
const _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _m = new THREE.Matrix4(), _s1 = new THREE.Vector3(1, 1, 1);

const clamp = THREE.MathUtils.clamp;
const ss = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// guardrail geometry constants (tools/blender/props.py: RAIL_X back plane of the W-beam, RAIL_Z beam centre height)
const RAIL_X = 0.20, RAIL_H = 0.595, SEG = 4.0;
const RAIL_POST_D = -3.8;   // post line on the valley shoulder (beam face ends up at d ~ -3.5)
const LOD_DIST = 30;         // guardrail chunks whose centre is closer than LOD_DIST + chunk radius draw the full mesh

export default class Props {
  constructor(ctx) {
    this.ctx = ctx;
    this.root = new THREE.Group();
    this.root.name = 'props';
    this.items = { jerrycan: new THREE.Group(), hatchet: new THREE.Group(), planks: new THREE.Group() };
    for (const [k, o] of Object.entries(this.items)) { o.name = 'item_' + k; o.userData.itemId = k; }
    this.itemColliders = {};
    this.colliders = [];
    this.bridge = null;
    this.bridgeColliders = [];
    this.protos = {};
    this.railChunks = [];
    this.cones = [];          // {body, index}
    this.coneMesh = null;
    this.lampLight = null;
    this.lightTower = null;
    this._frame = 0;
    this.sites = null;
  }

  // ------------------------------------------------------------------------------------------------------------
  async init() {
    const { ctx } = this;
    ctx.scene.add(this.root);
    const road = ctx.road;
    if (!road) { console.warn('[props] no road: nothing placed'); return; }
    const mk = road.markers || {};
    this.P0 = mk.pulloff?.[0] ?? 365;
    this.P1 = mk.pulloff?.[1] ?? 425;
    this.RW = mk.roadworks ?? 395;
    this.GAP = mk.gap ?? 560;
    this._layoutSites();
    // items exist in the world even without the GLB (placeholders at the right spot)
    for (const [k, o] of Object.entries(this.items)) {
      const p = this.sites[k];
      this._placeOnGround(o, p.s, p.d, p.yaw || 0, p.lift || 0);
      this.root.add(o);
    }

    // ?propsSrc=scratch/props/x  loads x.glb instead (for testing a build without publishing it)
    const src = new URLSearchParams(location.search).get('propsSrc');
    let gltf = null;
    try { gltf = await ctx.assets.gltf(src ? src + '.glb' : 'assets/models/props.glb'); } catch (e) { console.warn('[props] props.glb failed to load', e); }
    if (!gltf) return;
    this._extractProtos(gltf.scene);
    this._prepMaterials();

    const step = (name, fn) => { try { fn(); } catch (e) { console.error(`[props] ${name} failed`, e); } };
    step('items', () => this._buildItems());
    step('site', () => this._buildSite());
    step('guardrails', () => this._buildGuardrails());
    step('delineators', () => this._buildDelineators());
    step('signs', () => this._buildSigns());
    step('roadside', () => this._buildRoadside());
    this.root.updateMatrixWorld(true);
  }

  // ------------------------------------------------------------------------------------------------------------
  // helpers
  // ------------------------------------------------------------------------------------------------------------
  /** Pull-off bench weight (0 on the normal road, 1 across the full-width pull-off), same shape as terrain.py. */
  pullW(s) { return ss(this.P0 - 10, this.P0 + 1, s) * (1 - ss(this.P1 - 1, this.P1 + 10, s)); }

  /** Valley edge d of the drivable/gravel shoulder at s. */
  edgeD(s) { return -3.9 - 10.1 * this.pullW(s); }

  /** Ground height at (s, d): ray cast into the terrain when available, else the road cross-section. */
  groundY(s, d, fallbackOnly = false) {
    const road = this.ctx.road;
    const c = road.pointAt(s, _v3);
    const cy = c.y;
    const ad = Math.abs(d);
    const formula = ad <= 3.0 ? cy - 0.02 * ad : cy - 0.10 - 0.018 * Math.max(0, ad - 3.9);
    if (fallbackOnly) return formula;
    const p = road.worldAt(s, d, _v2);
    const t = this.ctx.terrain;
    let h = null;
    try { h = t?.heightAt ? t.heightAt(p.x, p.z, cy + 4) : this.ctx.physics?.groundHeight?.(p.x, p.z, cy + 4, 60); } catch { h = null; }
    if (h === null || h === undefined || !isFinite(h) || Math.abs(h - formula) > 3) return formula;
    return h;
  }

  /** Upright transform at (s, d) with extra yaw (radians, relative to "+Z along +s"). */
  matAt(s, d, yaw = 0, lift = 0, y = null, out = new THREE.Matrix4(), tilt = null) {
    const road = this.ctx.road;
    const p = road.worldAt(s, d, _v);
    p.y = (y ?? this.groundY(s, d)) + lift;
    _q.setFromAxisAngle(Y, road.yawAt(s) + yaw);
    if (tilt) _q.multiply(_q2.setFromEuler(tilt));
    return out.compose(p, _q, _s1);
  }

  _placeOnGround(obj, s, d, yaw = 0, lift = 0, tilt = null) {
    this.matAt(s, d, yaw, lift, null, _m, tilt);
    _m.decompose(obj.position, obj.quaternion, obj.scale);
    obj.updateMatrixWorld(true);
    return obj;
  }

  _extractProtos(scene) {
    scene.updateMatrixWorld(true);
    const protoNames = new Set();
    scene.traverse((o) => { if (o.name) protoNames.add(o.name); });
    const collect = (node) => {
      const parts = [];
      const inv = new THREE.Matrix4().copy(node.matrixWorld).invert();
      const visit = (o, isRoot) => {
        // named child nodes that are protos of their own (lamp_head under light_tower) are not part of the parent
        if (o.isMesh) parts.push({ geometry: o.geometry, material: o.material, matrix: new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld), name: o.name });
        for (const c of o.children) {
          if (c.name === 'lamp_head' && node.name === 'light_tower') continue;
          visit(c, false);
        }
      };
      visit(node, true);
      return parts;
    };
    for (const name of ['jerrycan', 'hatchet', 'plank', 'plank_stack', 'plank_wedge', 'barrier', 'cone', 'crate', 'toolbox',
      'light_tower', 'lamp_head', 'sign_rockfall', 'sign_roadworks', 'guardrail', 'guardrail_bent', 'guardrail_lod',
      'guardrail_end', 'delineator', 'rail_reflector', 'sandbag', 'tarp_pile', 'km_post', 'snow_pole', 'culvert_grate',
      'sign_bend', 'sign_chains']) {
      const node = scene.getObjectByName(name);
      if (!node) continue;
      const parts = collect(node);
      if (parts.length) this.protos[name] = parts;
    }
  }

  _prepMaterials() {
    const seen = new Set();
    // The tarp and the scaffold boards share the baked 'wood' atlas but not its behaviour: woven PE barely darkens
    // when wet and turns glossy; the boards get a world-space dirt pass (every board in the stack shares one baked
    // UV region, so the dirt that depends on where a board sits -- splash-back near the ground, mud on the top
    // board -- is added here).
    const swap = (protoName, fromName, make) => {
      const parts = this.protos[protoName];
      if (!parts) return;
      for (const p of parts) if (p.material?.name === fromName) p.material = make(p.material);
    };
    let tarpMat = null, plankMat = null;
    // (blue woven PE is ~0.08 linear albedo new, ~0.18 UV-faded; the shared wood-atlas bake reads brighter than
    //  that under the overcast sky, so the tarp is toned down and cooled a little)
    swap('tarp_pile', 'props_wood', (m) => (tarpMat ||= Object.assign(m.clone(), { name: 'props_tarp', color: new THREE.Color(0.5, 0.55, 0.66) })));
    const mkPlank = (m) => (plankMat ||= this._plankDirt(Object.assign(m.clone(), { name: 'props_planks' })));
    swap('plank_stack', 'props_wood', mkPlank);
    swap('plank', 'props_wood', mkPlank);
    const porosity = { props_wood: 0.75, props_road: 0.35, props_metal: 0.12, props_rail: 0.1, props_tarp: 0.06, props_planks: 0.75 };
    // roughness floor when soaked: a water film on zinc/paint is glossy, but never a mirror (rails read as chrome)
    const minRough = { props_wood: 0.3, props_road: 0.22, props_metal: 0.24, props_rail: 0.34, props_tarp: 0.14, props_planks: 0.3 };
    const mats = [];
    for (const parts of Object.values(this.protos)) for (const p of parts) {
      const list = Array.isArray(p.material) ? p.material : [p.material];
      for (const m of list) if (m && !seen.has(m)) { seen.add(m); mats.push(m); }
    }
    this.materials = mats;
    const aniso = Math.min(8, this.ctx.renderer?.capabilities?.getMaxAnisotropy?.() || 4);
    for (const m of mats) {
      for (const k of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap']) if (m[k]) m[k].anisotropy = aniso;
      if (m.name === 'lamp') {
        m.emissive?.setRGB(1.0, 0.86, 0.66);
        m.emissiveIntensity = 26;
        m.toneMapped = true;
        continue;
      }
      if (m.name === 'tarp_water') {
        // rain water standing in the tarp's pockets: dark, clear film over the blue PE, near-mirror
        m.color.setRGB(0.012, 0.016, 0.02);
        m.roughness = 0.03; m.metalness = 0;
        m.envMapIntensity = 1.0;
        m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -1;
        continue;
      }
      m.envMapIntensity = 1.0;
      if (m.aoMap) m.aoMapIntensity = 1.0;
    }
    // rain wetness (RENDER helper); optional
    import('../render/materials.js').then(({ applyWetness, applyRipples }) => {
      if (!applyWetness) return;
      for (const m of mats) {
        if (m.name === 'lamp') continue;
        if (m.name === 'tarp_water') { try { applyRipples?.(m, { strength: 1.3, scale: 0.8, wetGate: 0 }, this.ctx); } catch (e) { /* optional */ } continue; }
        try { applyWetness(m, { porosity: porosity[m.name] ?? 0.3, strength: 1, minRoughness: minRough[m.name] ?? 0.2 }, this.ctx); } catch (e) { /* optional */ }
      }
    }).catch(() => {});
  }

  /**
   * World-space dirt for the scaffold boards (object space of the stack / plank; y = height above the ground):
   * rain splash-back from the gravel as dense mud speckles on the lowest ~25 cm of the vertical faces, dried and
   * fresh boot-mud smears and grit on the top faces, dark run-off streaks down the sides under the top board, green
   * algae on the shaded lower faces, wet dark end grain.
   */
  _plankDirt(m) {
    m.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vPlO;\nvarying vec3 vPlN;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n vPlO = position; vPlN = normal;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>
varying vec3 vPlO; varying vec3 vPlN;
float pl_h(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float pl_n(vec2 p) { vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(pl_h(i), pl_h(i + vec2(1, 0)), u.x), mix(pl_h(i + vec2(0, 1)), pl_h(i + vec2(1, 1)), u.x), u.y); }
float plDirtRough;`)
        .replace('#include <map_fragment>', `#include <map_fragment>
        {
          vec3 o = vPlO; vec3 nO = normalize(vPlN);
          float side = 1.0 - abs(nO.y);
          float top = smoothstep(0.75, 0.95, nO.y);
          vec2 q = vec2(o.x + o.z, o.y);
          vec3 mudC = vec3(0.085, 0.062, 0.04);
          // every board in the stack has its own history: silver-grey old boards next to yellower, newer ones, some
          // darker (they spent a winter on the ground). Board id from its layer (38 mm + gap) and column.
          float bId = floor((o.y - 0.074) / 0.0395) * 2.0 + step(0.0, o.z);
          float hb = pl_h(vec2(bId, 3.7)), hb2 = pl_h(vec2(bId, 9.1));
          diffuseColor.rgb *= mix(vec3(0.8, 0.82, 0.85), vec3(1.14, 1.05, 0.9), hb) * mix(0.78, 1.08, hb2);
          // dried mortar / concrete splatter (scaffold boards live on building sites): pale grey drops and blobs with
          // hard edges, mostly on the upper faces, some flicked onto the edges; ~15-30% of the boards carry it
          float spl = pl_n(o.xz * vec2(7.0, 11.0) + bId * 3.1) * 0.6 + pl_n(o.xz * 31.0 + bId) * 0.4;
          float mort = smoothstep(0.72, 0.745, spl) * step(0.7, pl_h(vec2(bId, 5.3))) * (0.4 + 0.6 * top);
          mort = max(mort, smoothstep(0.86, 0.88, pl_n(q * 45.0 + bId * 1.7)) * 0.8 * side * step(0.55, hb2));
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.32, 0.315, 0.3) * (0.85 + 0.3 * pl_n(o.xz * 90.0)), mort * 0.85);
          // splash-back speckles (drops thrown up from the gravel), densest at the bottom
          float spk = smoothstep(0.62, 0.8, pl_n(q * 95.0) * 0.6 + pl_n(q * 230.0 + 7.0) * 0.4);
          float low = exp(-max(o.y, 0.0) / 0.075);
          float splash = clamp(low * (0.35 + 0.65 * side) * (0.55 * spk + 0.45 * low), 0.0, 1.0);
          diffuseColor.rgb = mix(diffuseColor.rgb, mudC, splash * 0.9);
          // boot mud on the top faces: smears (elongated along the boards) + grit
          float smear = smoothstep(0.5, 0.78, pl_n(vec2(o.x * 1.3, o.z * 5.0) + 3.0) * 0.65 + pl_n(vec2(o.x * 4.0, o.z * 11.0)) * 0.35);
          float grit = smoothstep(0.78, 0.9, pl_n(o.xz * 160.0));
          diffuseColor.rgb = mix(diffuseColor.rgb, mudC * 1.25, top * (smear * 0.75 + grit * 0.5));
          // run-off streaks on the sides (from the top board down), algae low on the sides
          float stk = smoothstep(0.55, 0.8, pl_n(vec2((o.x + o.z) * 26.0, o.y * 1.5)));
          diffuseColor.rgb *= 1.0 - 0.3 * stk * side;
          float alg = smoothstep(0.45, 0.8, pl_n(vec2(o.x + o.z, o.y) * 7.0)) * smoothstep(0.2, 0.05, o.y) * side;
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.05, 0.065, 0.03), alg * 0.55);
          plDirtRough = max(max(splash, top * smear * 0.8), mort);
        }`)
        .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = mix(roughnessFactor, 0.62, plDirtRough);`);
    };
    m.customProgramCacheKey = () => 'props-planks-dirt';
    m.needsUpdate = true;
    return m;
  }

  /**
   * Alpine roadside furniture: orange GRP snow poles (Schneestangen) set out for the winter along the inside of
   * the road and where the valley edge has no rail, a kilometre stone, culvert inlet grates in the uphill ditch
   * (at every gully + every ~200 m), and two old rusty signs.
   */
  _buildRoadside() {
    const road = this.ctx.road;
    const mk = road.markers || {};
    const tun = mk.tunnel ?? 1150, scar = mk.scar || [90, 190], ft = mk.fallenTree ?? 305;
    const gul = mk.gullies || [650, 770, 890, 1010];
    const rnd = mulberry32(911);
    const near = (s, list, r) => list.some((g) => Math.abs(s - g) < r);
    // ---- snow poles
    if (this.protos.snow_pole) {
      const mats = [];
      for (let s = 6; s < tun - 18; s += 20) {
        const inScar = s > scar[0] - 4 && s < scar[1] + 4;
        const lean = new THREE.Euler((rnd() - 0.5) * 0.07, 0, (rnd() - 0.5) * 0.07);
        if (!inScar && !near(s, gul, 7) && Math.abs(s - this.GAP) > 10 && Math.abs(s - ft) > 8) {
          mats.push(this.matAt(s + (rnd() - 0.5) * 1.5, 4.72 + rnd() * 0.15, rnd() * 6.28, 0, null, new THREE.Matrix4(), lean));
        } else if (inScar && rnd() < 0.3) {
          // one snapped off / pushed over by the mud, lying on the slide toe
          mats.push(this.matAt(s, 4.2 + rnd() * 1.5, rnd() * 6.28, 0.05, null, new THREE.Matrix4(), new THREE.Euler(1.35 + rnd() * 0.2, 0, rnd() - 0.5)));
        }
        if (!this._hasRail(s) && !(s > this.P0 - 8 && s < this.P1 + 8) && Math.abs(s - this.GAP) > 6) {
          mats.push(this.matAt(s + 3, this.edgeD(s + 3) + 0.25, rnd() * 6.28, 0, null, new THREE.Matrix4(), lean));
        }
      }
      this.instanced('snow_pole', mats, { cast: true });
    }
    // ---- kilometre stone (faces along the road, read by traffic in both directions)
    if (this.protos.km_post) {
      const o = this.instance('km_post');
      this._placeOnGround(o, 268, 4.62, 0.04, -0.01, new THREE.Euler(0.02, 0, -0.03));
      this.root.add(o);
    }
    // ---- culvert inlets in the uphill ditch (the grate sits on the ditch floor, 3-6 cm proud of the gravel;
    //      snapped to the real floor once the physics queries work, see update())
    if (this.protos.culvert_grate) {
      const spots = [...gul.map((g) => g + 2.5), 58, 236, 468, 842, 1092].filter((s) => s < tun - 20 && Math.abs(s - this.GAP) > 8);
      this._culverts = spots.map((s) => ({ s, d: 3.66, yaw: (rnd() - 0.5) * 0.04 }));
      const mats = this._culverts.map((c) => this.matAt(c.s, c.d, c.yaw, 0, road.pointAt(c.s, _v).y - 0.31));
      this._culvertMeshes = this.instanced('culvert_grate', mats, { cast: false });
      for (const im of this._culvertMeshes) im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this._culvertSnapped = false;
    }
    // ---- old rusty signs (valley side, facing oncoming traffic, leaning a little)
    const place = (name, s, d, yaw, tilt) => {
      if (!this.protos[name]) return;
      const o = this.instance(name);
      this._placeOnGround(o, s, d, yaw, 0, tilt);
      this.root.add(o);
    };
    place('sign_bend', 452, -4.2, Math.PI - 0.08, new THREE.Euler(0.03, 0, 0.045));
    place('sign_chains', 12, -4.25, Math.PI - 0.1, new THREE.Euler(-0.02, 0, -0.03));
    place('sign_chains', 700, 4.6, 0.12, new THREE.Euler(0.025, 0, 0.04));
  }

  /** One-time: set the culvert grates onto the actual ditch floor (terrain ray casts work after the first step). */
  _snapCulverts() {
    const road = this.ctx.road, t = this.ctx.terrain;
    if (!this._culverts || !this._culvertMeshes?.length || !t?.heightAt) return;
    const m = new THREE.Matrix4();
    let moved = 0;
    this._culverts.forEach((c, i) => {
      let floor = null;
      const cy = road.pointAt(c.s, _v).y;
      for (const d of [3.5, 3.58, 3.66, 3.74, 3.82]) {
        const w = road.worldAt(c.s, d, _v2);
        const h = t.heightAt(w.x, w.z, cy + 2);
        if (h !== null && isFinite(h) && Math.abs(h - cy) < 2) floor = floor === null ? h : Math.min(floor, h);
      }
      if (floor === null) return;
      this.matAt(c.s, c.d, c.yaw, 0, floor + 0.045, m);
      for (const im of this._culvertMeshes) {
        // instanced() bakes the part offset into each instance: re-apply it
        const part = this.protos.culvert_grate.find((p) => p.geometry === im.geometry);
        im.setMatrixAt(i, _m.multiplyMatrices(m, part ? part.matrix : new THREE.Matrix4()));
        im.instanceMatrix.needsUpdate = true;
      }
      moved++;
    });
    for (const im of this._culvertMeshes) { im.computeBoundingSphere(); im.computeBoundingBox?.(); }
    return moved;
  }

  /** A plain Group of meshes for a proto (shares geometry and materials). */
  instance(name, { cast = true, recv = true } = {}) {
    const parts = this.protos[name];
    const g = new THREE.Group();
    g.name = name;
    if (!parts) return g;
    for (const p of parts) {
      const m = new THREE.Mesh(p.geometry, p.material);
      m.name = name + (parts.length > 1 ? '_' + p.name : '');
      p.matrix.decompose(m.position, m.quaternion, m.scale);
      m.castShadow = cast; m.receiveShadow = recv;
      g.add(m);
    }
    return g;
  }

  /** InstancedMeshes (one per part) for a proto with the given world matrices. */
  instanced(name, matrices, { cast = true, recv = true, parent = this.root } = {}) {
    const parts = this.protos[name];
    if (!parts || !matrices.length) return [];
    const out = [];
    for (const p of parts) {
      const im = new THREE.InstancedMesh(p.geometry, p.material, matrices.length);
      im.name = name + '_inst';
      for (let i = 0; i < matrices.length; i++) im.setMatrixAt(i, _m.multiplyMatrices(matrices[i], p.matrix));
      im.instanceMatrix.needsUpdate = true;
      im.computeBoundingSphere();
      im.computeBoundingBox?.();
      im.castShadow = cast; im.receiveShadow = recv;
      parent.add(im);
      out.push(im);
    }
    return out;
  }

  _addBox(center, half, quat, opts = {}) {
    const ph = this.ctx.physics;
    if (!ph?.world) return null;
    try {
      const c = ph.addStaticBox(center, half, quat, opts);
      this.colliders.push(c);
      return c;
    } catch (e) { console.warn('[props] collider failed', e); return null; }
  }

  /** Static box in an object's local frame: localCenter/half in the object's space. */
  _boxFor(matrix, localCenter, half, opts) {
    const p = _v.copy(localCenter).applyMatrix4(matrix);
    const q = new THREE.Quaternion().setFromRotationMatrix(_m.extractRotation(matrix));
    return this._addBox(p.clone(), half, q, opts);
  }

  // ------------------------------------------------------------------------------------------------------------
  // the roadworks site
  // ------------------------------------------------------------------------------------------------------------
  _layoutSites() {
    const rw = this.RW;
    // road-space (s, d) spots. The pull-off gravel reaches d = -14 for s in [P0+1, P1-1].
    this.sites = {
      roadworksS: rw,
      // jerrycan: by the light tower (the generator fuel can), on the road side of the tripod in the lamp's pool of
      // light, so it catches the eye from the road through the barrier gap
      jerrycan: { s: rw + 1.4, d: -7.0, yaw: 2.2 },
      // hatchet: lying on the crate by the toolbox at the back of the site
      hatchet: { s: rw + 10.4, d: -11.55, yaw: 0.35, lift: 0.0, onCrate: true },
      // plank stack: on two bearers along the road, near the site entry
      planks: { s: rw - 9.5, d: -6.9, yaw: -Math.PI / 2 + 0.03 },
    };
  }

  _buildItems() {
    const S = this.sites;
    // jerrycan
    const jc = this.instance('jerrycan');
    this.items.jerrycan.add(jc);
    // hatchet lying on its flat side: handle along +X, head flat faces up/down (rotate about local Z by -90 deg)
    const ht = this.instance('hatchet');
    ht.rotation.set(0, 0, -Math.PI / 2);
    ht.position.y = 0.021;
    this.items.hatchet.add(ht);
    if (S.hatchet.onCrate) {
      // re-seat on top of the crate (crate height 0.496 m)
      this._placeOnGround(this.items.hatchet, S.hatchet.s, S.hatchet.d, S.hatchet.yaw, 0.497);
    }
    // plank stack
    const st = this.instance('plank_stack');
    this.items.planks.add(st);
    // plank stack collider (player can climb it; removed when picked up)
    const pm = this.items.planks.matrixWorld;
    this.itemColliders.planks = this._boxFor(pm, new THREE.Vector3(0, 0.16, 0), new THREE.Vector3(2.0, 0.16, 0.37));
  }

  _buildSite() {
    const { ctx } = this;
    const rw = this.RW;
    const rnd = mulberry32(395);
    const jit = (a) => (rnd() * 2 - 1) * a;

    // ---- concrete barriers: lining the lane edge of the work zone (with two walk-through gaps) and a short row
    //      shielding the stockpile at the outer edge. Colliders: base slab + upper wall box.
    const barrierSpots = [];
    const lineS = [370.4, 372.6, 374.8, /* gap */ 379.6, 381.8, 384.0, 386.2, 388.4, /* gap */ 393.2, 395.4, 397.6, 399.8, 402.0, 404.2,
      /* gap */ 409.0, 411.2, 413.4, 415.6, 417.8].map((x) => x - 395 + rw);
    // (QA) the barrier's worn paint and scuffs are baked into one texture: turned all the same way, every barrier
    // showed the same scuff in the same place. A jersey barrier is symmetric, and on a real site they are set down
    // either way round, so about half are turned 180 deg (own RNG: the other props keep their placement).
    const flip = mulberry32(4711);
    for (const s of lineS) barrierSpots.push({ s: s + jit(0.05), d: -3.55 + jit(0.04), yaw: Math.PI / 2 + jit(0.025) + (flip() < 0.5 ? Math.PI : 0) });
    // one knocked askew by a truck
    barrierSpots[5].yaw += 0.16; barrierSpots[5].d -= 0.25;
    // outer row by the drop (behind the tarp pile) and an L at the downstream end
    for (const [s, d, y] of [[rw + 16.6, -12.6, Math.PI / 2 + 0.05], [rw + 18.8, -12.55, Math.PI / 2 - 0.02], [rw + 20.3, -11.3, 0.08]]) {
      barrierSpots.push({ s, d, yaw: y });
    }
    const bm = barrierSpots.map((b) => this.matAt(b.s, b.d, b.yaw));
    this.instanced('barrier', bm);
    for (const m of bm) {
      this._boxFor(m, new THREE.Vector3(0, 0.08, 0), new THREE.Vector3(1.05, 0.08, 0.29));
      this._boxFor(m, new THREE.Vector3(0, 0.46, 0), new THREE.Vector3(1.05, 0.34, 0.12));
    }

    // ---- light tower (on): lamps face local +Z; turn them toward the road (+d) and aim along the site
    const tw = { s: rw + 0.6, d: -8.8, yaw: Math.PI / 2 + 0.35 };
    const tower = this.instance('light_tower');
    this._placeOnGround(tower, tw.s, tw.d, tw.yaw);
    const head = this.instance('lamp_head', { cast: true });
    head.name = 'lamp_head';
    head.position.set(0, 2.36, 0);
    head.rotation.y = 0.0;
    tower.add(head);
    this.root.add(tower);
    tower.updateMatrixWorld(true);
    this.lightTower = tower;
    this._boxFor(tower.matrixWorld, new THREE.Vector3(0, 1.2, 0), new THREE.Vector3(0.12, 1.2, 0.12));
    // spot light from the lamp head down into the site toward the road
    const L = new THREE.SpotLight(0xffe2bd, 42, 38, THREE.MathUtils.degToRad(46), 0.55, 2);
    L.position.set(0, 2.30, 0.12);
    head.add(L);
    const tgt = new THREE.Object3D();
    tgt.position.set(0, -2.2, 6.0);
    head.add(tgt);
    L.target = tgt;
    // a spot shadow re-renders every caster in its frustum (terrain + trees too: ~400k tris, 30 calls): ultra only
    const hq = ctx.config?.quality?.name === 'Ultra';
    L.castShadow = !!hq;
    if (L.castShadow) {
      L.shadow.mapSize.set(512, 512);
      L.shadow.bias = -0.0008;
      L.shadow.normalBias = 0.03;
      L.shadow.camera.near = 0.3; L.shadow.camera.far = 16;
      L.shadow.radius = 3;
    }
    this.lampLight = L;
    head.updateMatrixWorld(true);
    // sandbags weighing down the tripod feet (feet at r = 0.78, 120 deg apart; three coords)
    const feet = [[0, -0.78], [-0.675, 0.39], [0.675, 0.39]];
    const sb = [];
    for (let i = 0; i < 3; i++) {
      const [fx, fz] = feet[i];
      if (i === 2) continue; // one foot left bare
      const lp = new THREE.Vector3(fx * 1.05, 0.0, fz * 1.05).applyMatrix4(tower.matrixWorld);
      const a = Math.atan2(fx, fz);
      _q.setFromAxisAngle(Y, a + tw.yaw + ctx.road.yawAt(tw.s) + Math.PI / 2 + jit(0.3));
      sb.push(new THREE.Matrix4().compose(new THREE.Vector3(lp.x, lp.y + 0.045, lp.z), _q.clone(), _s1));
    }

    // ---- crates + toolbox + hatchet spot (back of the site)
    const cr = [
      { s: rw + 10.4, d: -11.55, yaw: 0.35 },   // hatchet lies on this one
      { s: rw + 11.3, d: -12.55, yaw: 0.05 },
      { s: rw + 9.3, d: -12.7, yaw: -0.12 },
    ];
    const cm = cr.map((c) => this.matAt(c.s, c.d, c.yaw));
    // a third crate stacked on the second, slightly rotated
    const top = this.matAt(cr[1].s + 0.05, cr[1].d + 0.03, cr[1].yaw + 0.22, 0.497);
    cm.push(top);
    this.instanced('crate', cm);
    for (const m of cm) this._boxFor(m, new THREE.Vector3(0, 0.25, 0), new THREE.Vector3(0.42, 0.25, 0.27));
    const tb = this.instance('toolbox');
    this._placeOnGround(tb, rw + 9.6, -10.75, 1.25);
    this.root.add(tb);
    // the jerrycan's hatch side: a second (empty, dented) jerrycan is NOT placed: exactly one can in the world

    // ---- plank-stack surroundings: a loose plank on the gravel and wedges
    const lp = this.instance('plank');
    this._placeOnGround(lp, rw - 8.6, -8.3, -Math.PI / 2 + 0.28, 0.0);
    this.root.add(lp);

    // ---- tarp-covered pile (stockpile) at the downstream end + sandbags holding the tarp
    const tp = this.instance('tarp_pile');
    this._placeOnGround(tp, rw + 17.2, -9.4, 0.4);
    this.root.add(tp);
    this._boxFor(tp.matrixWorld, new THREE.Vector3(0, 0.3, 0), new THREE.Vector3(1.15, 0.3, 0.9));
    for (const [lx, lz, a] of [[-1.3, 0.2, 0.3], [1.25, -0.6, 1.4], [0.2, 1.05, 0.1], [-0.6, -1.0, -0.4]]) {
      const p = new THREE.Vector3(lx, 0, lz).applyMatrix4(tp.matrixWorld);
      const r = this.ctx.road.project(p, {});
      sb.push(this.matAt(r.s, r.d, a + jit(0.2), 0.0));
    }
    // a small stack of sandbags behind the barrier line near the site entry
    const base = { s: rw - 13.5, d: -4.6 };
    const stackPts = [[0, 0, 0, 0.05], [0.58, 0.02, 0, -0.08], [1.16, -0.03, 0, 0.1], [0.3, 0.02, 0.135, 0.3], [0.88, 0.0, 0.135, -0.2]];
    for (const [ds, dd, h, a] of stackPts) sb.push(this.matAt(base.s + ds, base.d + dd, Math.PI / 2 + a, h));
    // a low sandbag wall along the drop behind the crates (edge protection) + a couple of loose ones
    for (let k = 0; k < 6; k++) sb.push(this.matAt(rw + 6.2 + k * 0.6 + jit(0.05), -13.25 + jit(0.06), Math.PI / 2 + jit(0.15)));
    for (let k = 0; k < 3; k++) sb.push(this.matAt(rw + 6.5 + k * 0.6 + jit(0.05), -13.25 + jit(0.05), Math.PI / 2 + jit(0.2), 0.135));
    for (const [ds, dd, a] of [[-5.6, -10.4, 0.9], [7.1, -7.4, -0.4], [-1.2, -12.9, 1.7]]) sb.push(this.matAt(rw + ds, dd, a));
    this.instanced('sandbag', sb, { cast: false });

    // ---- a second, smaller covered pile (cement bags under a tarp) at the upstream end of the site
    const tp2 = this.instance('tarp_pile');
    this._placeOnGround(tp2, rw - 4.8, -12.0, -1.1);
    tp2.scale.set(0.72, 0.62, 0.8);
    this.root.add(tp2);
    tp2.updateMatrixWorld(true);
    this._boxFor(tp2.matrixWorld, new THREE.Vector3(0, 0.3, 0), new THREE.Vector3(1.15 * 0.72, 0.3 * 0.62, 0.9 * 0.8));
    // two more loose planks: one dropped by the crates, one across two sandbags as a makeshift bench
    const lp2 = this.instance('plank');
    this._placeOnGround(lp2, rw + 6.8, -11.6, 0.12, 0.0);
    this.root.add(lp2);

    // ---- traffic cones: lane taper on the road (dynamic, knockable) + a stack + a toppled one in the site
    const taper = [];
    const P0 = this.P0, P1 = this.P1;
    const coneLine = -2.6;
    for (let s = P0 - 12; s <= P1 - 4; s += 0) {
      const t = ss(P0 - 12, P0 + 4, s) * (1 - ss(P1 - 10, P1 - 2, s));
      const d = -3.05 + (coneLine + 3.05) * t;
      taper.push({ s: s + jit(0.15), d: d + jit(0.04), yaw: jit(3), dynamic: true });
      s += t > 0.98 ? 6.0 : 3.2;
    }
    // nested stack of 3 cones and a toppled cone in the site
    const stackBase = { s: rw - 3.2, d: -5.0 };
    const coneMats = [];
    const dyn = [];
    for (const c of taper) { coneMats.push(this.matAt(c.s, c.d, c.yaw)); dyn.push(true); }
    for (let k = 0; k < 3; k++) { coneMats.push(this.matAt(stackBase.s, stackBase.d, jit(3), 0.085 * k)); dyn.push(false); }
    coneMats.push(this.matAt(rw + 4.2, -6.2, 1.1, 0.19, null, new THREE.Matrix4(), new THREE.Euler(Math.PI / 2 - 0.25, 0, 0)));
    dyn.push(false);
    const [coneIM] = this.instanced('cone', coneMats);
    if (coneIM) {
      this.coneMesh = coneIM;
      coneIM.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      // dynamic bodies for the road cones
      const ph = ctx.physics, R = ph?.RAPIER;
      if (ph?.world && R) {
        const cg = groups(G.SENSOR, G.STATIC | G.CAR | G.ROCK | G.DEBRIS | G.SENSOR);
        for (let i = 0; i < coneMats.length; i++) {
          if (!dyn[i]) continue;
          const p = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
          coneMats[i].decompose(p, q, sc);
          try {
            const body = ph.world.createRigidBody(R.RigidBodyDesc.dynamic().setTranslation(p.x, p.y + 0.004, p.z)
              .setRotation({ x: q.x, y: q.y, z: q.z, w: q.w }).setLinearDamping(0.2).setAngularDamping(0.4).setSleeping(true)
              .setCcdEnabled(false));
            ph.world.createCollider(R.ColliderDesc.cone(0.33, 0.16).setTranslation(0, 0.37, 0).setDensity(90).setFriction(0.7)
              .setRestitution(0.15).setCollisionGroups(cg), body);
            ph.world.createCollider(R.ColliderDesc.cuboid(0.19, 0.02, 0.19).setTranslation(0, 0.02, 0).setDensity(260).setFriction(0.8)
              .setCollisionGroups(cg), body);
            this.cones.push({ body, index: i });
          } catch (e) { console.warn('[props] cone body failed', e); break; }
        }
      }
    }
  }

  // ------------------------------------------------------------------------------------------------------------
  // guardrails
  // ------------------------------------------------------------------------------------------------------------
  _railRuns() {
    const road = this.ctx.road;
    const L = road.length ?? 1300;
    const mk = road.markers || {};
    const sc = this.ctx.terrain?.scatter;
    let runs = Array.isArray(sc?.guardrails) && sc.guardrails.length ? sc.guardrails.map((r) => [+r[0], +r[1]]) : null;
    if (!runs) {
      const ft = mk.fallenTree ?? 305, gap = this.GAP, tun = mk.tunnel ?? 1150;
      runs = [[0, ft - 7], [ft + 7, gap - 12], [gap + 12, tun - 45]];
    }
    runs = runs.filter((r) => r[1] > r[0] + 2).sort((a, b) => a[0] - b[0]);
    // PROPS owns the pull-off edge: bridge a break around the pull-off into one continuous run
    const merged = [];
    for (const r of runs) {
      const prev = merged[merged.length - 1];
      if (prev && prev[1] >= this.P0 - 16 && prev[1] <= this.P0 + 2 && r[0] >= this.P1 - 2 && r[0] <= this.P1 + 16) prev[1] = r[1];
      else merged.push([r[0], Math.min(r[1], L)]);
    }
    return merged;
  }

  _buildGuardrails() {
    const road = this.ctx.road;
    if (!this.protos.guardrail) return;
    const runs = this._railRuns();
    this.railRuns = runs;
    const railD = (s) => RAIL_POST_D - 10.1 * this.pullW(s);
    const rnd = mulberry32(77);
    const segs = [];   // {m: Matrix4, len, bent}
    const ends = [];   // Matrix4 (may be mirrored)
    const P = new THREE.Vector3(), Q = new THREE.Vector3();
    for (const [s0, s1] of runs) {
      // posts: walk along the rail line so that consecutive posts are SEG apart in 3D
      const posts = [];
      let s = s0;
      const pt = (ss_) => { const d = railD(ss_); const w = road.worldAt(ss_, d, new THREE.Vector3()); w.y = this.groundY(ss_, d); return w; };
      posts.push({ s, p: pt(s) });
      while (s < s1) {
        let a = s, b = s + SEG * 1.2;
        const p0 = posts[posts.length - 1].p;
        // widen when the rail swings out (pull-off transitions) so the search bracket covers SEG
        while (pt(b).distanceTo(p0) < SEG && b < s + 40) b += 2;
        for (let k = 0; k < 18; k++) {
          const m = (a + b) / 2;
          if (pt(m).distanceTo(p0) < SEG) a = m; else b = m;
        }
        const sn = (a + b) / 2;
        if (sn > s1 + 1.0) break;
        posts.push({ s: sn, p: pt(sn) });
        s = sn;
      }
      if (posts.length < 2) continue;
      // smooth the post heights (the beam follows the road, not every gravel bump)
      const ys = posts.map((q) => q.p.y);
      for (let i = 0; i < posts.length; i++) {
        const a = ys[Math.max(0, i - 1)], b = ys[i], c = ys[Math.min(posts.length - 1, i + 1)];
        let y = (a + 2 * b + c) / 4;
        // stay with the real ground at this post: never float (posts would hang in the air where the terrain falls
        // away) and never sink the beam more than 20 cm into a bump
        y = THREE.MathUtils.clamp(y, b - 0.2, b + 0.06);
        posts[i].p.y = y;
      }
      for (let i = 0; i < posts.length - 1; i++) {
        P.copy(posts[i].p); Q.copy(posts[i + 1].p);
        const z = _v.subVectors(Q, P);
        const len = z.length();
        z.normalize();
        const x = _v2.crossVectors(Y, z).normalize();
        const y = _v3.crossVectors(z, x).normalize();
        const m = new THREE.Matrix4().makeBasis(x, y, z).scale(new THREE.Vector3(1, 1, len / SEG)).setPosition(P);
        segs.push({ m, len, s: posts[i].s, bent: false });
      }
      // terminals: run end continues along +s; run start is mirrored so it curls away from the road too
      const first = segs[segs.length - (posts.length - 1)];
      const last = segs[segs.length - 1];
      if (s0 > 1) {
        const x = new THREE.Vector3(), y = new THREE.Vector3(), z = new THREE.Vector3();
        first.m.extractBasis(x, y, z);
        z.normalize().negate();
        ends.push(new THREE.Matrix4().makeBasis(x.normalize(), y.normalize(), z).setPosition(posts[0].p));
      }
      if (s1 < (road.length ?? 1300) - 1) {
        const x = new THREE.Vector3(), y = new THREE.Vector3(), z = new THREE.Vector3();
        last.m.extractBasis(x, y, z);
        ends.push(new THREE.Matrix4().makeBasis(x.normalize(), y.normalize(), z.normalize()).setPosition(posts[posts.length - 1].p));
      }
    }
    // a few bent (hit) segments: next to the fallen tree crown, and some older impacts
    const ft = this.ctx.road.markers?.fallenTree ?? 305;
    const bentS = [ft - 9, ft + 9, 128, 612, 834];
    for (const bs of bentS) {
      let best = null, bd = 1e9;
      for (const sg of segs) { const dd = Math.abs(sg.s + 2 - bs); if (dd < bd) { bd = dd; best = sg; } }
      if (best && bd < 5) best.bent = true;
    }
    this.railSegs = segs;

    // ---- chunked instancing with hi/lo swap
    const CH = 8;
    const hasBent = !!this.protos.guardrail_bent, hasLod = !!this.protos.guardrail_lod, hasRefl = !!this.protos.rail_reflector;
    for (let i = 0; i < segs.length; i += CH) {
      const part = segs.slice(i, i + CH);
      const grp = new THREE.Group();
      grp.name = 'guardrail_chunk';
      this.root.add(grp);
      const hi = new THREE.Group(), lo = new THREE.Group();
      grp.add(hi, lo);
      const normal = part.filter((sg) => !(sg.bent && hasBent)).map((sg) => sg.m);
      const bent = part.filter((sg) => sg.bent && hasBent).map((sg) => sg.m);
      this.instanced('guardrail', normal, { parent: hi, cast: true });
      this.instanced('guardrail_bent', bent, { parent: hi, cast: true });
      if (hasRefl) {
        // reflector on every second post; un-scale (the reflector sits on the post, not stretched)
        const rm = part.filter((_, k) => (i + k) % 2 === 0).map((sg) => sg.m);
        this.instanced('rail_reflector', rm, { parent: hi, cast: false });
      }
      if (hasLod) this.instanced('guardrail_lod', part.map((sg) => sg.m), { parent: lo, cast: false });
      else this.instanced('guardrail', part.map((sg) => sg.m), { parent: lo, cast: false });
      const c = new THREE.Vector3();
      for (const sg of part) c.add(_v.setFromMatrixPosition(sg.m));
      c.multiplyScalar(1 / part.length);
      hi.visible = true; lo.visible = false;
      this.railChunks.push({ grp, hi, lo, center: c, near: true });
    }
    // terminals: separate meshes (mirrored ones need a negative-determinant Mesh, which three handles per object)
    if (this.protos.guardrail_end) {
      for (const m of ends) {
        const o = this.instance('guardrail_end');
        m.decompose(o.position, o.quaternion, o.scale);
        // makeBasis with a flipped z gives det < 0: decompose() puts it into scale.x = -1 (three flips the winding)
        this.root.add(o);
      }
    }
    // colliders: thin boxes along each beam (local frame: beam face at x ~ RAIL_X..RAIL_X+0.08, y ~ RAIL_H)
    const ph = this.ctx.physics;
    if (ph?.world) {
      for (const sg of segs) {
        const x = new THREE.Vector3(), y = new THREE.Vector3(), z = new THREE.Vector3();
        sg.m.extractBasis(x, y, z);
        const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x.normalize(), y.normalize(), z.normalize()));
        const pos = new THREE.Vector3().setFromMatrixPosition(sg.m)
          .addScaledVector(x, RAIL_X + 0.03).addScaledVector(y, RAIL_H - 0.08).addScaledVector(z, sg.len / 2);
        this._addBox(pos, new THREE.Vector3(0.06, 0.26, sg.len / 2 + 0.05), q);
      }
    }
  }

  // ------------------------------------------------------------------------------------------------------------
  // delineators + signs
  // ------------------------------------------------------------------------------------------------------------
  _hasRail(s) {
    for (const [a, b] of this.railRuns || []) if (s >= a - 1 && s <= b + 1) return true;
    return false;
  }

  _buildDelineators() {
    if (!this.protos.delineator) return;
    const road = this.ctx.road;
    const mk = road.markers || {};
    const tun = mk.tunnel ?? 1150;
    const scar = mk.scar || [90, 190];
    const rnd = mulberry32(12);
    const mats = [];
    for (let s = 12.5; s < tun - 12; s += 25) {
      if (Math.abs(s - this.GAP) < 9) continue;
      if (Math.abs(s - (mk.fallenTree ?? 305)) < 9) continue;
      // uphill side: at the foot of the cut face beyond the ditch; buried or missing in the landslide scar
      const inScar = s > scar[0] - 3 && s < scar[1] + 3;
      if (!inScar) {
        const lean = new THREE.Euler((rnd() - 0.5) * 0.06, 0, (rnd() - 0.5) * 0.06);
        mats.push(this.matAt(s + (rnd() - 0.5), 4.45, Math.PI + (rnd() - 0.5) * 0.1, 0, null, new THREE.Matrix4(), lean));
      } else if (Math.abs(s - 140) < 13) {
        // one knocked flat by mud, lying in the ditch
        mats.push(this.matAt(s, 3.9, Math.PI + 0.4, 0.06, null, new THREE.Matrix4(), new THREE.Euler(1.35, 0, 0.2)));
      }
      // valley side: only where there is no guardrail (the rail carries its own reflectors)
      if (!this._hasRail(s) && !(s > this.P0 - 8 && s < this.P1 + 8)) {
        mats.push(this.matAt(s, this.edgeD(s) + 0.15, Math.PI + (rnd() - 0.5) * 0.1));
      }
    }
    // extra valley-side posts marking the washout edges (the rail is missing there)
    for (const s of [this.GAP - 10, this.GAP - 5.5, this.GAP + 5.5, this.GAP + 10]) mats.push(this.matAt(s, -3.75, Math.PI + (rnd() - 0.5) * 0.2));
    this.instanced('delineator', mats, { cast: true });
  }

  _buildSigns() {
    const road = this.ctx.road;
    const place = (name, s, d, yaw) => {
      if (!this.protos[name]) return;
      const o = this.instance(name);
      this._placeOnGround(o, s, d, yaw);
      this.root.add(o);
      return o;
    };
    // signs face local +Z: yaw PI turns them toward traffic coming along +s. Right-hand side = valley side.
    place('sign_rockfall', 60, -4.15, Math.PI - 0.12);
    place('sign_roadworks', 330, -4.15, Math.PI - 0.12);
    // for traffic in the other direction (uphill side, facing +s)
    place('sign_roadworks', this.P1 + 22, 4.5, 0.1);
    place('sign_rockfall', (road.markers?.scar?.[1] ?? 190) + 30, 4.5, 0.1);
  }

  // ------------------------------------------------------------------------------------------------------------
  // items
  // ------------------------------------------------------------------------------------------------------------
  hideItem(id) {
    const o = this.items[id];
    if (o) o.visible = false;
    const c = this.itemColliders[id];
    if (c && this.ctx.physics?.world) {
      try { this.ctx.physics.world.removeCollider(c, false); } catch { /* already gone */ }
      this.itemColliders[id] = null;
      this.colliders = this.colliders.filter((x) => x !== c);
    }
  }

  showItem(id) {
    const o = this.items[id];
    if (!o) return;
    o.visible = true;
    if (id === 'planks' && !this.itemColliders.planks) {
      o.updateMatrixWorld(true);
      this.itemColliders.planks = this._boxFor(o.matrixWorld, new THREE.Vector3(0, 0.16, 0), new THREE.Vector3(2.0, 0.16, 0.37));
    }
  }

  /** Plank bridge over the trench at markers.gap: 11 boards side by side, centred on the player lane (d = -1.5). */
  placePlanks() {
    if (this.bridge) return this.bridge;
    const { ctx } = this;
    const road = ctx.road;
    const bridge = new THREE.Group();
    bridge.name = 'plank_bridge';
    const gap = this.GAP;
    const N = 11, W = 0.225, pitch = 0.2265, DC = -1.5, HL = 1.95; // plank half-length
    const halfW = (N - 1) * pitch / 2 + W / 2;
    // bearing height at each end: the highest ground under the plank ends (broken asphalt edge), sampled across
    const bearing = (sgn) => {
      let best = -1e9;
      for (let k = 0; k <= 6; k++) {
        const d = DC - halfW + (2 * halfW) * k / 6;
        for (const u of [1.45, 1.6, 1.75, 1.9]) best = Math.max(best, this.groundY(gap + sgn * u, d));
      }
      return best;
    };
    const yA = bearing(-1), yB = bearing(1);
    const sA = gap - HL, sB = gap + HL;
    const pA = road.worldAt(sA, DC, new THREE.Vector3()); pA.y = yA;
    const pB = road.worldAt(sB, DC, new THREE.Vector3()); pB.y = yB;
    const axis = new THREE.Vector3().subVectors(pB, pA).normalize();          // along +s (with slope)
    const left = road.leftAt(gap, new THREE.Vector3());
    const up = new THREE.Vector3().crossVectors(axis, left).normalize();
    if (up.y < 0) up.negate();
    const across = new THREE.Vector3().crossVectors(up, axis).normalize();    // toward +d
    const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
    // deck frame: X = along s (plank length axis), Y = up, Z = -d... use basis (axis, up, -across) (right-handed)
    const deckM = new THREE.Matrix4().makeBasis(axis, up, _v.copy(axis).cross(up).normalize()).setPosition(mid);
    // _v = axis x up = -across*(...) : z points to -d. Local z offsets therefore use -dd.
    const rnd = mulberry32(560);
    const planks = [];
    if (this.protos.plank) {
      for (let i = 0; i < N; i++) {
        const dd = (i - (N - 1) / 2) * pitch + (rnd() - 0.5) * 0.008;
        const o = this.instance('plank');
        o.position.set((rnd() - 0.5) * 0.12, 0.002 * rnd(), -dd);
        o.rotation.set(0, (rnd() - 0.5) * 0.02 + (rnd() < 0.5 ? Math.PI : 0), (rnd() - 0.5) * 0.004);
        planks.push(o);
      }
    }
    const deck = new THREE.Group();
    deck.name = 'plank_deck';
    deckM.decompose(deck.position, deck.quaternion, deck.scale);
    for (const o of planks) deck.add(o);
    bridge.add(deck);
    // end ramps: plank_wedge rises toward local +Z (0.55 across, 0.32 along, 0.042 high); 5 per end
    const wedges = [];
    if (this.protos.plank_wedge) {
      for (const sgn of [-1, 1]) {
        for (let k = 0; k < 5; k++) {
          const w = this.instance('plank_wedge');
          const dd = -halfW + 0.275 + k * ((2 * halfW - 0.55) / 4);
          // wedge local +Z (thick end) must point toward the deck: at the -s end that's +X of the deck frame
          w.position.set(sgn * (HL + 0.16 + 0.005), -0.002, -dd + (rnd() - 0.5) * 0.02);
          w.rotation.set(0, sgn < 0 ? Math.PI / 2 : -Math.PI / 2, 0);
          w.rotation.y += (rnd() - 0.5) * 0.06;
          deck.add(w);
          wedges.push(w);
        }
      }
    }
    bridge.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    this.root.add(bridge);
    bridge.updateMatrixWorld(true);
    this.bridge = bridge;

    // colliders (G.PROP: car wheels + player walk on it, footsteps report 'wood')
    const ph = ctx.physics, R = ph?.RAPIER;
    if (ph?.world && R) {
      const gp = groups(G.PROP, G.ALL);
      const q = new THREE.Quaternion().setFromRotationMatrix(_m.extractRotation(deckM));
      const deckC = new THREE.Vector3(0, 0.019, 0).applyMatrix4(deckM);
      const c1 = ph.addStaticBox(deckC, new THREE.Vector3(HL, 0.019, halfW), q, { groups: gp, friction: 1.0 });
      this.bridgeColliders.push(c1);
      // ramps: convex wedges in deck space
      for (const sgn of [-1, 1]) {
        const x0 = sgn * HL, x1 = sgn * (HL + 0.33);
        const pts = [];
        for (const z of [-halfW, halfW]) {
          for (const [x, y] of [[x0, 0.0], [x0, 0.04], [x1, -0.004]]) {
            const p = new THREE.Vector3(x, y, z).applyMatrix4(deckM);
            pts.push(p.x, p.y, p.z);
          }
        }
        try {
          const desc = R.ColliderDesc.convexHull(new Float32Array(pts));
          if (desc) {
            desc.setFriction(1.0).setCollisionGroups(gp);
            this.bridgeColliders.push(ph.world.createCollider(desc));
          }
        } catch (e) { console.warn('[props] ramp collider failed', e); }
      }
    }
    bridge.userData = { deckWidth: 2 * halfW, span: 2 * HL, s: gap, d: DC };
    return bridge;
  }

  removePlanks() {
    if (!this.bridge) return;
    this.root.remove(this.bridge);
    const w = this.ctx.physics?.world;
    for (const c of this.bridgeColliders) { try { w?.removeCollider(c, false); } catch { /* */ } }
    this.bridgeColliders = [];
    this.bridge = null;
  }

  setLampOn(on) {
    if (this.lampLight) this.lampLight.visible = !!on;
    for (const m of this.materials || []) if (m.name === 'lamp') m.emissiveIntensity = on ? 26 : 0;
  }

  // ------------------------------------------------------------------------------------------------------------
  update() {
    const cam = this.ctx.camera;
    // culvert grates: snap to the ditch floor once the terrain ray casts work (a few frames after boot)
    if (this._culverts && !this._culvertSnapped && this._frame > 12) {
      this._culvertSnapped = true;
      try { this._snapCulverts(); } catch (e) { console.warn('[props] culvert snap failed', e); }
    }
    // guardrail LOD swap (cheap: ~20 chunks, every 8 frames)
    if (cam && (this._frame++ & 7) === 0) {
      for (const c of this.railChunks) {
        const near = c.center.distanceTo(cam.position) < LOD_DIST + 16;
        if (near !== c.near) { c.near = near; c.hi.visible = near; c.lo.visible = !near; }
      }
    }
    // dynamic cones -> instance matrices (only while awake)
    if (this.coneMesh && this.cones.length) {
      let dirty = false;
      for (const c of this.cones) {
        const b = c.body;
        if (!b || b.isSleeping()) continue;
        const t = b.translation(), r = b.rotation();
        _m.compose(_v.set(t.x, t.y, t.z), _q.set(r.x, r.y, r.z, r.w), _s1);
        this.coneMesh.setMatrixAt(c.index, _m);
        dirty = true;
        // fell off the world: park it
        if (t.y < -2000) b.sleep();
      }
      if (dirty) { this.coneMesh.instanceMatrix.needsUpdate = true; this.coneMesh.computeBoundingSphere(); }
    }
  }

  dispose() {
    const w = this.ctx.physics?.world;
    for (const c of [...this.colliders, ...this.bridgeColliders]) { try { w?.removeCollider(c, false); } catch { /* */ } }
    for (const c of this.cones) { try { w?.removeRigidBody(c.body); } catch { /* */ } }
    this.ctx.scene.remove(this.root);
  }
}
