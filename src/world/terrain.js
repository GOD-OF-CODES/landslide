// Terrain system (TERRAIN workstream): loads assets/world/terrain.glb (built by tools/blender/terrain.py),
// assigns the terrain / road / tunnel materials, creates static trimesh colliders and exposes the terrain API.
//
// API (DESIGN.md "terrain"):
//   meshes: { near: Group, far: Group, road: Group, tunnel: Mesh, lamps: Mesh,
//             nearChunks: Mesh[], farChunks: Mesh[], roadChunks: Mesh[] }
//   heightAt(x, z, fromY?) -> y | null   (ray cast into static geometry; inside the tunnel casts from the vault)
//   surfaceAt(pos) -> 'asphalt'|'gravel'|'dirt'|'rock'|'grass'|'mud'
//   grassAt(x, z) -> 0..1 vegetation density (scatter.json grassMask, road (s,d) space)
//   insideTunnel(pos) -> 0..1 (1 = deep inside the tunnel, 0 = daylight)
//   scatter: parsed scatter.json (trees, rocks, slideSpawns, guardrails, grassMask)
import * as THREE from 'three';
import { createTerrainMaterial, createWaterMaterial } from '../render/terrainMaterial.js';
import { createRoadMaterial, createTunnelMaterial, tunnelLampPositions, TUNNEL, TUNNEL_SHARED, PORTAL } from '../render/roadMaterial.js';
import { G, groups } from '../physics/world.js';

const _v = new THREE.Vector3(), _d = new THREE.Vector3(0, -1, 0);
const _ss = THREE.MathUtils.smoothstep;
const _fLo = new THREE.Vector3(), _fHi = new THREE.Vector3();
/** Pull-off bench weight (tools/blender/terrain.py pull_w): bench edge at d = -3.9 - 11 w. */
function pullW(s, road) {
  const p = road?.markers?.pulloff || [365, 425];
  return _ss(s, p[0] - 12, p[0] - 1) * (1 - _ss(s, p[1] + 1, p[1] + 12));
}

export default class Terrain {
  constructor(ctx) {
    this.ctx = ctx;
    this.meshes = {};
    this.colliders = [];
    this.materials = [];
    this.scatter = null;
    this._pr = {};
    this._grass = null;
  }

  async init() {
    const { ctx } = this;
    // ?terrainSrc=scratch/terrain/x  loads x.glb + x.json instead (for testing a build without publishing it)
    const src = new URLSearchParams(location.search).get('terrainSrc');
    const [gltf, scatter] = await Promise.all([
      ctx.assets.gltf(src ? src + '.glb' : 'assets/world/terrain.glb'),
      ctx.assets.json(src ? src + '.json' : 'assets/world/scatter.json').catch((e) => { console.warn('[terrain] scatter.json missing', e); return null; }),
    ]);
    this.scatter = scatter;
    this._decodeGrass();
    const root = gltf.scene;
    root.name = 'terrain';
    root.updateMatrixWorld(true);
    const byName = (n) => root.getObjectByName(n);
    const near = byName('terrain_near'), far = byName('terrain_far'), road = byName('road');
    const tunnel = byName('tunnel'), lamps = byName('tunnel_lamps');
    const meshesOf = (o) => { const a = []; o?.traverse((m) => { if (m.isMesh) a.push(m); }); return a; };

    const [matNear, matFar, matRoad, matTunnel] = await Promise.all([
      createTerrainMaterial(ctx, { far: false }),
      createTerrainMaterial(ctx, { far: true }),
      createRoadMaterial(ctx),
      createTunnelMaterial(ctx),
    ]);
    this.materials.push(matNear, matFar, matRoad, matTunnel);
    const prep = (m, mat, cast, recv) => {
      const g = m.geometry;
      // COLOR_0 stays readable as 'color' (vegetation/grass sample the masks on the CPU); the shaders read the
      // same BufferAttribute as 'masks' (vertexColors stays off, so three never multiplies it into the albedo)
      if (g.attributes.color) g.setAttribute('masks', g.attributes.color);
      if (!g.attributes.masks) {
        const n = g.attributes.position.count, a = new Float32Array(n * 4).fill(1);
        g.setAttribute('masks', new THREE.BufferAttribute(a, 4));
      }
      m.material = mat;
      m.castShadow = cast; m.receiveShadow = recv;
      m.matrixAutoUpdate = false; m.updateMatrix();
    };
    const nearChunks = meshesOf(near), farChunks = meshesOf(far), roadChunks = meshesOf(road);
    nearChunks.forEach((m) => prep(m, matNear, true, true));
    farChunks.forEach((m) => prep(m, matFar, false, true));
    roadChunks.forEach((m) => prep(m, matRoad, false, true));
    // PERF (QA): the terrain and road shaders are the most expensive in the scene. Every tree, grass and impostor
    // material discards (alpha test), which defeats the M1's hidden-surface removal for whatever was drawn before
    // it, so the terrain goes last among the opaques (after the impostors at 4; the sky dome is 1e6) and early-z
    // rejects the ground hidden behind trees, rocks, props and the car: -2.2 ms/frame at s=240, no visual change.
    // Within that: road first (the terrain runs on under the asphalt), then near, then far (its skirt lies under
    // the near mesh).
    roadChunks.forEach((m) => (m.renderOrder = 4.4));
    nearChunks.forEach((m) => (m.renderOrder = 4.5));
    farChunks.forEach((m) => (m.renderOrder = 4.6));
    const tunnelMesh = meshesOf(tunnel)[0];
    if (tunnelMesh) prep(tunnelMesh, matTunnel, true, true);
    if (tunnelMesh && ctx.road) {
      try { const collar = this._portalCollar(tunnelMesh, matTunnel, root); if (collar) root.add(collar); }
      catch (e) { console.warn('[terrain] portal collar', e); }
      try { const seal = this._portalRingSeal(matTunnel, root); if (seal) root.add(seal); }
      catch (e) { console.warn('[terrain] portal ring seal', e); }
      try { const ul = this._tunnelUnderlay(root); if (ul) root.add(ul); }
      catch (e) { console.warn('[terrain] tunnel underlay', e); }
      // the portal drainage channel needs ground heights (Rapier queries work after the first step): built in update()
      this._chanPending = true; this._matTunnel = matTunnel;
    }
    const lampMesh = meshesOf(lamps)[0];
    if (lampMesh) {
      lampMesh.material = new THREE.MeshStandardMaterial({
        color: 0x222018, roughness: 0.4, metalness: 0.2, emissive: new THREE.Color(1.0, 0.55, 0.18), emissiveIntensity: 9,
      });
      lampMesh.castShadow = false; lampMesh.receiveShadow = false;
    }
    // ---- near LODs: terrain_near_lod1_XX (decimated, same AO/masks, chunk seams vertex-identical to LOD0)
    // takes over from terrain_near_XX beyond ~lodDist. LOD0 stays in nearChunks (colliders, vegetation).
    const lod1Root = byName('terrain_near_lod1');
    const lod1 = new Map();
    meshesOf(lod1Root).forEach((m) => lod1.set(m.name.replace('terrain_near_lod1_', ''), m));
    const lodDist = { Ultra: 230, High: 175, Medium: 130, Low: 100 }[ctx.config?.quality?.name] || 175;
    this.lods = [];
    if (lod1.size && near) {
      const box = new THREE.Box3(), c = new THREE.Vector3();
      for (const m of nearChunks) {
        const lo = lod1.get(m.name.replace('terrain_near_', ''));
        if (!lo) continue;
        m.geometry.computeBoundingBox();
        box.copy(m.geometry.boundingBox).applyMatrix4(m.matrixWorld);
        box.getCenter(c);
        // only the full-res chunks near the camera cast sun shadows (the shadow map covers ~60 m anyway)
        prep(lo, matNear, false, true);
        lo.renderOrder = 4.5;   // see the renderOrder note above
        const L = new THREE.LOD();
        L.name = m.name + '_lod';
        L.position.copy(c);
        near.add(L);
        for (const mm of [m, lo]) {
          mm.removeFromParent();
          mm.position.sub(c); mm.updateMatrix();
        }
        L.addLevel(m, 0, 0.04);
        L.addLevel(lo, lodDist, 0.04);
        L.updateMatrix();
        this.lods.push(L);
      }
      lod1Root.removeFromParent();
    }
    root.traverse((o) => { if (!o.isMesh) { o.matrixAutoUpdate = false; o.updateMatrix(); } });
    ctx.scene.add(root);
    root.updateMatrixWorld(true);
    this.root = root;
    this.meshes = { near, far, road, tunnel: tunnelMesh || tunnel, lamps: lampMesh || lamps, nearChunks, farChunks, roadChunks, lods: this.lods };

    // ---- physics: static trimeshes (terrain near, road, tunnel). Far terrain has no collider.
    const phys = ctx.physics;
    if (phys?.world) {
      const t0 = performance.now();
      for (const m of nearChunks) this.colliders.push(phys.addTrimesh(m, { friction: 0.85, groups: groups(G.STATIC) }));
      for (const m of roadChunks) this.colliders.push(phys.addTrimesh(m, { friction: 1.0, groups: groups(G.STATIC) }));
      if (tunnelMesh) this.colliders.push(phys.addTrimesh(tunnelMesh, { friction: 0.9, groups: groups(G.STATIC) }));
      this._groundHandles = new Set(this.colliders.map((c) => c.handle));
      if (ctx.flags?.debug) console.log(`[terrain] colliders built in ${(performance.now() - t0).toFixed(0)} ms`);
    }

    // ---- static decorative rocks (scatter.rocks: talus at the cut toe, scar/gully boulders, trench rubble)
    try { await this._buildRocks(); } catch (e) { console.warn('[terrain] static rocks skipped', e); }
    // ---- snapped / uprooted trunks carried by the slide (scatter.deadwood)
    try { await this._buildDeadwood(); } catch (e) { console.warn('[terrain] deadwood skipped', e); }
    // ---- running water: ditch, gully rivulets + falls, the washout ravine (ribbons draped on the ground)
    // (Rapier's scene queries only work after the first physics step: the ribbons are built time-sliced in update();
    //  the mesh + material exist now so the shader compiles during the boot warm-up)
    try { this._initWater(); } catch (e) { console.warn('[terrain] water skipped', e); }

    // ---- tunnel lights: two non-shadow sodium point lights that hop between the lamps nearest the camera
    this.lampPos = ctx.road ? tunnelLampPositions(ctx.road) : [];
    this.tunnelLights = [];
    for (let i = 0; i < 2; i++) {
      // matched to the analytic lamps in roadMaterial.js (linear (1, .52, .16) x 14): these light cars / props /
      // the player; the road + tunnel materials mask them out and sum every lamp themselves
      const L = new THREE.PointLight(0xffffff, 0, 22, 2);
      L.color.setRGB(1.0, 0.52, 0.16, THREE.LinearSRGBColorSpace);
      L.castShadow = false;
      ctx.scene.add(L);
      this.tunnelLights.push(L);
    }
  }

  /**
   * Instanced static rocks from scatter.rocks using ROCKS' rocks.glb (rock_0..9 boulders, pebble_0..5 for small
   * stones). Only instances within ~140 m (boulders) / 70 m (stones) of the camera are drawn: the instance buffers
   * are refilled when the camera has moved a few meters (cheap: < 1k rocks). Big rocks near the road get ball colliders.
   */
  async _buildRocks() {
    const { ctx } = this;
    const list = this.scatter?.rocks;
    if (!list?.length) return;
    const gltf = await ctx.assets.gltf('assets/models/rocks.glb');
    const meshOf = (name) => { const o = gltf.scene.getObjectByName(name); let m = null; o?.traverse((c) => { if (!m && c.isMesh) m = c; }); return m; };
    // buckets 0..9: rock_k (~2.9k tris), 10..19: rock_k_lod1 (~720), 20..29: rock_k_lod2 (~200), 30..35: pebble_k.
    // A boulder keeps its own shape through all three LODs (same transform), so switching does not pop; stones
    // (small scale) always use a pebble.
    const buckets = [];
    const add = (k, proto, lvl) => { if (proto) buckets[k] = { proto, lvl, items: [], mesh: null }; };
    for (let i = 0; i < 10; i++) {
      add(i, meshOf('rock_' + i), 0);
      add(10 + i, meshOf(`rock_${i}_lod1`) || meshOf('rock_' + i), 1);
      add(20 + i, meshOf(`rock_${i}_lod2`) || meshOf(`rock_${i}_lod1`) || meshOf('rock_' + i), 2);
    }
    for (let i = 0; i < 6; i++) add(30 + i, meshOf('pebble_' + i), 3);
    const hiKeys = [];
    for (let i = 0; i < 10; i++) if (buckets[i]) hiKeys.push(i);
    const stoneKeys = [];
    for (let i = 30; i < 36; i++) if (buckets[i]) stoneKeys.push(i);
    if (!hiKeys.length && !stoneKeys.length) return;
    let applyWetness = null;
    try { ({ applyWetness } = await import('../render/materials.js')); } catch { /* optional */ }
    const mats = new Map();
    const matFor = (m) => {
      if (!mats.has(m.material)) {
        const mm = m.material.clone();
        // rock is not very porous: darkens moderately, glossy but never a mirror (rain-pitted, gritty film)
        try { applyWetness?.(mm, { porosity: 0.6, strength: 1, minRoughness: 0.32 }, ctx); } catch { /* optional */ }
        mats.set(m.material, mm);
      }
      return mats.get(m.material);
    };
    const road = ctx.road;
    const pr = {};
    const phys = ctx.physics;
    const RAPIER = phys?.RAPIER;
    const q = new THREE.Quaternion(), e = new THREE.Euler(), sc = new THREE.Vector3(), pos = new THREE.Vector3();
    const items = [];
    for (const r of list) {
      const [x, y, z, s, ry, rx, v] = r;
      const vi = Math.abs(v | 0);
      const small = s < 0.32 || !hiKeys.length;
      let hi = -1, stone = -1;
      if (small) { if (!stoneKeys.length) continue; stone = stoneKeys[vi % stoneKeys.length]; }
      else hi = buckets[vi % 10] ? vi % 10 : hiKeys[vi % hiKeys.length];
      pos.set(x, y, z);
      e.set(rx, ry, 0, 'YXZ'); q.setFromEuler(e);
      sc.set(s, small ? s * 0.75 : s * 0.9, s); // stones lie on their broad side
      // per-instance tint: boulders in the fresh slide, the gullies and the trench are coated in wet mud;
      // talus and slope boulders are weathered grey with some variation
      let col;
      const hv = ((Math.sin(x * 12.9898 + z * 78.233) * 43758.5453) % 1 + 1) % 1;
      if (road) {
        road.project(pos, pr);
        const inScar = pr.s > 86 && pr.s < 194 && pr.d > 3.5;
        const inGully = (road.markers?.gullies || []).some((g) => Math.abs(pr.s - g) < 9 && pr.d > 3.5);
        const inGap = Math.abs(pr.s - 560) < 3 && Math.abs(pr.d) < 9;
        if (inScar || inGully || inGap) col = new THREE.Color(0.17 + 0.07 * hv, 0.14 + 0.05 * hv, 0.11 + 0.04 * hv);
      }
      if (!col) { const g = 0.24 + 0.14 * hv; col = new THREE.Color(g, g, g * 1.05); }
      items.push({ x, y, z, m: new THREE.Matrix4().compose(pos, q, sc), hi, stone, big: s >= 0.6, col });
      // collider for rocks you could walk or drive into
      if (RAPIER && phys.world && s >= 0.3 && road) {
        if (Math.abs(pr.d) < 22) {
          const desc = RAPIER.ColliderDesc.ball(s * 0.72).setTranslation(x, y + s * 0.05, z)
            .setFriction(0.9).setCollisionGroups(groups(G.STATIC));
          this.colliders.push(phys.world.createCollider(desc));
        }
      }
    }
    // ---- loose stones spilled onto the asphalt by the slide, the gullies and the washout (2-20 cm, no colliders):
    //      densest at the uphill edge where the mud tongues come in, thinning across the carriageway; most are
    //      fresh broken grey rock, some mud-coated
    if (stoneKeys.length && road) {
      let a = 0x2545f491;
      const rnd = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      const mk = road.markers || {};
      const zones = [[(mk.scar?.[0] ?? 90) + 4, (mk.scar?.[1] ?? 190) + 6, 2.6], [(mk.gap ?? 560) - 14, (mk.gap ?? 560) - 2.2, 1.6], [(mk.gap ?? 560) + 2.2, (mk.gap ?? 560) + 14, 1.6]];
      for (const g of mk.gullies || []) zones.push([g - 9, g + 9, 2.4]);
      const cp = new THREE.Vector3();
      for (const [s0, s1, dens] of zones) {
        const n = Math.round((s1 - s0) * dens);
        for (let i = 0; i < n; i++) {
          const sa = s0 + rnd() * (s1 - s0);
          const u = (sa - s0) / (s1 - s0);
          if (rnd() > Math.sin(Math.PI * u) * 1.3) continue;                 // thin out toward the zone ends
          const dd = 3.0 - Math.min(5.9, -Math.log(1 - rnd() * 0.995) * 1.5);    // exponential from the uphill edge
          const size = 0.025 + 0.16 * Math.pow(rnd(), 2.6);
          road.worldAt(sa, dd, cp);
          const cy = road.pointAt(sa, pos).y - 0.02 * Math.abs(dd);
          cp.y = cy + size * 0.75 * 0.35;
          e.set((rnd() - 0.5) * 0.5, rnd() * Math.PI * 2, (rnd() - 0.5) * 0.5, 'YXZ'); q.setFromEuler(e);
          sc.set(size * (0.8 + 0.4 * rnd()), size * 0.62, size * (0.8 + 0.4 * rnd()));
          const hv = rnd();
          const col = rnd() < 0.3 ? new THREE.Color(0.16 + 0.05 * hv, 0.13 + 0.04 * hv, 0.1 + 0.03 * hv)
            : new THREE.Color(0.55 + 0.3 * hv, 0.55 + 0.29 * hv, 0.57 + 0.29 * hv);
          items.push({ x: cp.x, y: cp.y, z: cp.z, m: new THREE.Matrix4().compose(cp, q, sc), hi: -1, stone: stoneKeys[(i * 7 + (hv * 5 | 0)) % stoneKeys.length], big: false, col, r2: 24 * 24 });
        }
      }
    }
    const cap = new Map();
    const inc = (k) => { if (k >= 0 && buckets[k]) cap.set(k, (cap.get(k) || 0) + 1); };
    for (const it of items) { if (it.hi >= 0) { inc(it.hi); inc(it.hi + 10); inc(it.hi + 20); } else inc(it.stone); }
    const root = new THREE.Group();
    root.name = 'terrain_rocks';
    for (const [k, n] of cap) {
      const b = buckets[k];
      const mesh = new THREE.InstancedMesh(b.proto.geometry, matFor(b.proto), n);
      mesh.name = 'rocks_' + ['hi', 'lod1', 'lod2', 'stone'][b.lvl] + '_' + k;
      mesh.castShadow = b.lvl === 0; mesh.receiveShadow = true;
      mesh.frustumCulled = false; // only nearby instances are uploaded
      mesh.count = 0;
      mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3).fill(1), 3);
      b.mesh = mesh;
      root.add(mesh);
    }
    ctx.scene.add(root);
    this.rockRoot = root;
    this.rockItems = items;
    this.rockBuckets = buckets.filter((b) => b && b.mesh);
    this._rockBuckets = buckets;
    this._rockCam = new THREE.Vector3(1e9, 0, 0);
    this.meshes.rocks = root;
  }

  /**
   * Procedural spruce trunks (scatter.deadwood: [x, y, z, length, rotY, pitch, radius]) merged into one mesh:
   * irregular tapered section, root flare on uprooted ones, splintered breaks with pale fresh wood, bark UVs in
   * meters. Logs near the road get capsule colliders.
   */
  async _buildDeadwood() {
    const { ctx } = this;
    const list = this.scatter?.deadwood;
    if (!list?.length) return;
    const tex = await ctx.assets.pbr('pine_bark');
    const RAD = 12, SEG = 10;
    const pos = [], nrm = [], uv = [], col = [], idxBig = [], idxSmall = [];
    const rnd = (() => { let a = 0x9e3779b9; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), v = new THREE.Vector3(), n3 = new THREE.Matrix3();
    const phys = ctx.physics, RAPIER = phys?.RAPIER, road = ctx.road, pr = {};
    for (const [x, y, z, L, ry, pitch, r0] of list) {
      const uprooted = rnd() < 0.4;
      const base = pos.length / 3;
      e.set(0, ry, pitch, 'YZX'); q.setFromEuler(e);
      m4.compose(v.set(x, y, z), q, new THREE.Vector3(1, 1, 1));
      n3.getNormalMatrix(m4);
      const bump = []; for (let k = 0; k < RAD; k++) bump.push(0.9 + 0.2 * rnd());
      const jag = []; for (let k = 0; k <= RAD; k++) jag.push(rnd());
      for (let i = 0; i <= SEG; i++) {
        const t = i / SEG;
        let xl = (t - 0.5) * L;
        let r = r0 * (1 - 0.38 * t);
        if (uprooted && t < 0.2) r *= 1 + 1.6 * Math.pow(1 - t / 0.2, 2.2);
        for (let k = 0; k <= RAD; k++) {
          const a = (k / RAD) * Math.PI * 2;
          const kk = k % RAD;
          let rr = r * bump[kk] * (uprooted && t < 0.12 ? 0.75 + 0.5 * jag[k] : 1);
          let xx = xl;
          let fresh = 0;
          // splintered break at the top end (and the bottom end of snapped, not uprooted, trunks)
          if (i === SEG || (i === 0 && !uprooted)) {
            const sgn = i === SEG ? 1 : -1;
            xx += sgn * (0.15 + 1.1 * jag[kk]) * r0 * 1.6;
            rr *= 0.35 + 0.5 * jag[(kk + 3) % RAD];
            fresh = 1;
          }
          const cy = Math.cos(a), sz = Math.sin(a);
          v.set(xx, cy * rr, sz * rr).applyMatrix4(m4);
          pos.push(v.x, v.y, v.z);
          v.set(0, cy, sz).applyMatrix3(n3).normalize();
          nrm.push(v.x, v.y, v.z);
          uv.push((k / RAD) * 2 * Math.PI * r0 / 0.9, xl / 0.9);
          // COLOR: r = fresh pale wood at breaks, g = mud smear on the underside / root end
          const mud = Math.max(0, -cy) * 0.7 + (uprooted && t < 0.15 ? 0.8 : 0);
          col.push(fresh, Math.min(1, mud), 0);
        }
      }
      // only real trunks cast sun shadows; branches / thin snapped stems are not worth a shadow-map pass
      const idx = r0 >= 0.18 || L >= 5 ? idxBig : idxSmall;
      for (let i = 0; i < SEG; i++) for (let k = 0; k < RAD; k++) {
        const a = base + i * (RAD + 1) + k, b = a + RAD + 1;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
      // colliders for trunks you could walk into
      if (RAPIER && phys?.world && road) {
        road.project(v.set(x, y, z), pr);
        if (Math.abs(pr.d) < 16) {
          const qa = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(1, 0, 0).applyQuaternion(q));
          const desc = RAPIER.ColliderDesc.capsule(L * 0.45, r0 * 0.9).setTranslation(x, y, z)
            .setRotation({ x: qa.x, y: qa.y, z: qa.z, w: qa.w }).setFriction(0.8).setCollisionGroups(groups(G.STATIC));
          this.colliders.push(phys.world.createCollider(desc));
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setAttribute('logc', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idxBig);
    g.computeBoundingSphere();
    const gS = new THREE.BufferGeometry();
    for (const k of ['position', 'normal', 'uv', 'logc']) gS.setAttribute(k, g.getAttribute(k));
    gS.setIndex(idxSmall);
    gS.boundingSphere = g.boundingSphere.clone();
    const mat = new THREE.MeshStandardMaterial({
      map: tex.map, normalMap: tex.normalMap, roughnessMap: tex.armMap, aoMap: tex.armMap, roughness: 1, metalness: 0,
      color: new THREE.Color(0.34, 0.32, 0.3),
    });
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uWet = { value: ctx.env?.wetness ?? 0.75 };
      mat.userData.sh = sh;
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute vec3 logc;\nvarying vec3 vLog;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvLog = logc;');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uWet;\nvarying vec3 vLog;')
        .replace('#include <map_fragment>', `#include <map_fragment>
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.42, 0.3, 0.18), vLog.r * 0.85);   // fresh splintered wood
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.09, 0.07, 0.05), vLog.g * 0.8);    // slide mud
          diffuseColor.rgb *= mix(1.0, 0.62, uWet);`)
        .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
          roughnessFactor = mix(roughnessFactor, max(0.55, roughnessFactor * 0.85), uWet);`);
    };
    mat.customProgramCacheKey = () => 'terrain-deadwood';
    const mesh = new THREE.Mesh(g, mat);
    mesh.name = 'terrain_deadwood';
    mesh.castShadow = idxBig.length > 0; mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    if (idxSmall.length) {
      const small = new THREE.Mesh(gS, mat);
      small.name = 'terrain_deadwood_small';
      small.castShadow = false; small.receiveShadow = true;
      small.matrixAutoUpdate = false;
      mesh.add(small);
    }
    ctx.scene.add(mesh);
    this.deadwood = mesh;
    this.meshes.deadwood = mesh;
    this.materials.push(mat);
    mat.userData.update = () => { if (mat.userData.sh) mat.userData.sh.uniforms.uWet.value = ctx.env?.wetness ?? 0.75; };
  }

  /** Terrain-only ray hit point along dir (unit, world) from o, or null. */
  _faceHit(o, dir, maxD) {
    const phys = this.ctx.physics;
    const set = this._groundHandles;
    const h = phys.raycast(o, dir, maxD, { groups: groups(G.ALL, G.STATIC), predicate: set ? (c) => set.has(c.handle) : undefined });
    return h ? h.point : null;
  }

  /** Terrain-only ground height (ignores rock / prop colliders); null where nothing is hit. */
  _groundY(x, z, y0) {
    const phys = this.ctx.physics;
    const set = this._groundHandles;
    const h = phys.raycast(_v.set(x, y0, z), _d, 300, { groups: groups(G.ALL, G.STATIC), predicate: set ? (c) => set.has(c.handle) : undefined });
    return h ? h.point.y : null;
  }

  /**
   * Running water as soft-edged ribbons draped on the ground (one merged mesh, no colliders):
   *  - the uphill ditch: a 2-6 cm deep film on the ditch floor wherever the ditch still has a channel (not where the
   *    slide filled it), flowing down-grade; width found from the ditch cross-section at the water level
   *  - each gully (markers.gullies) and the washout ravine: a rivulet down the thalweg, 0.2-0.7 m wide, white water
   *    where the bed is steep (the step down the rock cut into the ditch is a small fall)
   *  - two muddy rills down the slide scar
   */
  _initWater() {
    const { ctx } = this;
    const mat = createWaterMaterial(ctx);
    const g = new THREE.BufferGeometry();
    // placeholder: one tiny triangle under the road start (replaced once the ribbons are built)
    const p0 = ctx.road ? ctx.road.pointAt(0) : new THREE.Vector3();
    g.setAttribute('position', new THREE.Float32BufferAttribute([p0.x, p0.y - 5, p0.z, p0.x + 0.01, p0.y - 5, p0.z, p0.x, p0.y - 5, p0.z + 0.01], 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute([0, 1, 0, 0, 1, 0, 0, 1, 0], 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1], 2));
    g.setAttribute('wfx', new THREE.Float32BufferAttribute(new Array(12).fill(0), 4));
    const mesh = new THREE.Mesh(g, mat);
    mesh.name = 'terrain_water';
    mesh.castShadow = false; mesh.receiveShadow = true;
    mesh.renderOrder = 1;
    mesh.matrixAutoUpdate = false;
    mesh.frustumCulled = false;
    ctx.scene.add(mesh);
    this.water = mesh;
    this.meshes.water = mesh;
    this.materials.push(mat);
    this._waterJob = null;
    this._waterPending = !!(ctx.road && ctx.physics?.world);
  }

  /** Runs the water build generator for up to `budgetMs` per call. */
  _stepWater(budgetMs) {
    if (!this._waterJob) this._waterJob = this._buildWater();
    const t0 = performance.now();
    while (performance.now() - t0 < budgetMs) {
      const r = this._waterJob.next();
      if (r.done) { this._waterPending = false; this._waterJob = null; return; }
    }
  }

  *_buildWater() {
    const { ctx } = this;
    const road = ctx.road, phys = ctx.physics;
    if (!road || !phys?.world) return;
    const self = this;
    const pos = [], uvs = [], fx = [], idx = [];
    const P = new THREE.Vector3(), Q = new THREE.Vector3(), L = new THREE.Vector3();
    const hash = (a) => { const x = Math.sin(a * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
    // strip helper: rows = [{pts: [Vector3...], u01: [..], across: [..], along, foam, speed, turb}]
    const strip = (rows) => {
      if (rows.length < 2) return;
      const n = rows[0].pts.length;
      const base = pos.length / 3;
      for (const r of rows) {
        for (let k = 0; k < n; k++) {
          const p = r.pts[k];
          pos.push(p.x, p.y, p.z);
          uvs.push(r.across[k], r.along);
          fx.push(r.u01[k], r.foam, r.speed, r.turb);
        }
      }
      for (let i = 0; i < rows.length - 1; i++) for (let k = 0; k < n - 1; k++) {
        const a = base + i * n + k, b = a + n;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    };
    const mk = road.markers || {};
    const gap = mk.gap ?? 560, tun = mk.tunnel ?? 1150, scar = mk.scar || [90, 190];
    // ---------------- ditch
    {
      const DS = [];
      for (let d = 2.95; d <= 4.75; d += 0.1) DS.push(d);
      let rows = [];
      let along = 0, prevS = null;
      const flush = () => { strip(rows); rows = []; };
      for (let s = 3; s < tun - 16; s += 0.75) {
        yield;
        if (Math.abs(s - gap) < 3.2) { flush(); continue; }
        const c = road.pointAt(s, P);
        const cy = c.y;
        const hs = DS.map((d) => { const w = road.worldAt(s, d, Q); return self._groundY(w.x, w.z, cy + 3); });
        if (hs.some((h) => h === null)) { flush(); continue; }
        let kmin = 0;
        for (let k = 1; k < hs.length; k++) if (hs[k] < hs[kmin]) kmin = k;
        const floor = hs[kmin];
        const bankIn = Math.max(...hs.slice(0, Math.max(1, kmin))), bankOut = Math.max(...hs.slice(kmin));
        const depth = Math.min(bankIn, bankOut) - floor;
        // no channel (filled by the slide, the pull-off bench, the portal apron): no ditch water
        if (depth < 0.12 || floor > cy - 0.12 || kmin === 0 || kmin === hs.length - 1) { flush(); continue; }
        // shallow near the top of a run, deeper where the gullies feed it
        let wd = 0.02 + 0.03 * (0.5 + 0.5 * Math.sin(s * 0.37 + 2.0 * Math.sin(s * 0.11))) * (0.5 + hash(Math.floor(s / 5)));
        for (const g of mk.gullies || []) wd += 0.03 * Math.max(0, 1 - Math.abs(s - g - 12) / 40);
        const lvl = floor + wd;
        // wetted width: walk out from the floor to where the section rises above the water level
        let kl = kmin, kr = kmin;
        while (kl > 0 && hs[kl - 1] < lvl) kl--;
        while (kr < hs.length - 1 && hs[kr + 1] < lvl) kr++;
        const interp = (k0, k1) => { const h0 = hs[k0], h1 = hs[k1]; const f = (lvl - h0) / ((h1 - h0) || 1e-4); return DS[k0] + (DS[k1] - DS[k0]) * Math.min(Math.max(f, 0), 1); };
        const dl = kl > 0 ? interp(kl, kl - 1) : DS[0];
        const dr = kr < hs.length - 1 ? interp(kr, kr + 1) : DS[hs.length - 1];
        const dA = dl - 0.07, dB = dr + 0.07;
        if (dB - dA < 0.18) { flush(); continue; }
        if (prevS !== null && s - prevS > 1.0) flush();
        along += prevS === null ? 0 : (s - prevS);
        prevS = s;
        const pts = [], u01 = [], across = [];
        for (let k = 0; k < 4; k++) {
          const f = k / 3;
          const d = dA + (dB - dA) * f;
          const w = road.worldAt(s, d, new THREE.Vector3());
          w.y = lvl;
          pts.push(w); u01.push(f); across.push(d);
        }
        // the ditch drains toward lower road grade (the road climbs with s almost everywhere)
        const grade = road.tangentAt(s, L).y;
        rows.push({ pts, u01, across, along: grade > 0 ? -s : s, foam: 0.0, speed: 0.55, turb: 0.35 });
      }
      flush();
    }
    // ---------------- rivulets down thalwegs
    const rivulet = function* (s0, halfWin, dTop, turb, flowMul = 1) {
      // thalweg: lowest ground across [s0 - halfWin, s0 + halfWin] at each distance uphill
      const pts = [];
      let sPrev = s0;
      for (let d = dTop; d >= 3.6; d -= 0.5) {
        yield;
        let best = null, bs = sPrev;
        const win = Math.min(halfWin, 2.5 + Math.abs(d) * 0.08);
        for (let s = sPrev - win; s <= sPrev + win; s += 0.25) {
          const w = road.worldAt(s, d, Q);
          const y = self._groundY(w.x, w.z, road.pointAt(s, P).y + d * 1.6 + 40);
          if (y !== null && (best === null || y < best)) { best = y; bs = s; }
        }
        if (best === null) continue;
        sPrev = bs * 0.6 + sPrev * 0.4;
        pts.push({ s: sPrev, d });
      }
      if (pts.length < 4) return;
      // smooth the path along s
      for (let it = 0; it < 2; it++) for (let i = 1; i < pts.length - 1; i++) pts[i].s = (pts[i - 1].s + 2 * pts[i].s + pts[i + 1].s) / 4;
      const rows = [];
      const fall = [];   // steep rows (the jet down the rock cut): drives the wet halo in the terrain shader
      let along = 0, prev = null;
      let prevS = null;
      for (let i = 0; i < pts.length; i++) {
        let { s, d } = pts[i];
        const c = road.worldAt(s, d, new THREE.Vector3());
        const cy0 = road.pointAt(s, P).y + d * 1.6 + 40;
        let yc = self._groundY(c.x, c.z, cy0);
        if (yc === null) continue;
        // water falls straight down: below a drop, keep the lip's s (the thalweg search drifts sideways here)
        if (prev && prev.y - yc > 0.9 && prevS !== null) {
          const s2 = prevS;
          road.worldAt(s2, d, c);
          const y2 = self._groundY(c.x, c.z, road.pointAt(s2, P).y + d * 1.6 + 40);
          if (y2 !== null && prev.y - y2 > 0.9) { s = s2; yc = y2; for (let j = i; j < pts.length; j++) pts[j].s += (s2 - pts[j].s) * Math.max(0, 1 - (j - i) / 3); }
          else road.worldAt(s, d, c);
        }
        prevS = s;
        c.y = yc;
        // a drop over the rock cut / a bed step: the water leaves the lip as a free-falling jet (exit ~0.9 m/s:
        // ~0.8 m out from the face after a 4-5 m fall) instead of a quad cutting through the convex lip rock
        if (prev && prev.y - c.y > 0.9) {
          const Hd = prev.y - c.y;
          const nj = Math.min(9, Math.ceil(Hd / 0.5));
          const out = road.leftAt(s, new THREE.Vector3()).negate();
          const hd = Math.max(0.3, (c.x - prev.x) * out.x + (c.z - prev.z) * out.z);   // lip -> landing, toward the road
          const tj = road.tangentAt(s, L).setY(0).normalize();
          let last = prev.clone();
          for (let k = 1; k < nj; k++) {
            const f = k / nj, drop = Hd * f;
            const reach = Math.min(hd * 0.9, 0.9 * Math.sqrt(2 * drop / 9.81) + 0.06);
            const q = prev.clone().addScaledVector(out, reach);
            q.y = prev.y - drop;
            // a small flow clings to the face (and splashes off its ledges): find the face at this height with a
            // horizontal ray from the road side and keep the film 12 cm in front of it
            const lf = road.leftAt(s, Q);
            const o = new THREE.Vector3(q.x - lf.x * 3, q.y, q.z - lf.z * 3);
            const hit = self._faceHit(o, lf, 6);
            if (hit) {
              const dq = (q.x - hit.x) * lf.x + (q.z - hit.z) * lf.z;   // > 0: q is inside the rock
              if (dq > -0.12) q.addScaledVector(lf, -(dq + 0.12));
            }
            along += q.distanceTo(last); last = q.clone();
            const hwj = (0.14 + 0.12 * f) * flowMul * (0.8 + 0.4 * hash(k + s0));
            const jr = { pts: [], u01: [], across: [], along, foam: 1.0, speed: 2.8, turb };
            for (let kk = 0; kk < 4; kk++) {
              const ff = kk / 3, a = (ff - 0.5) * 2 * hwj;
              jr.pts.push(q.clone().addScaledVector(tj, a)); jr.u01.push(ff); jr.across.push(a);
            }
            rows.push(jr);
            fall.push({ p: q.clone(), hw: hwj, d });
          }
          along += c.distanceTo(last);
        } else if (prev) along += c.distanceTo(prev);
        // steepness of the bed -> white water
        let steep = 0;
        if (prev) steep = Math.min(1, Math.max(0, ((prev.y - c.y) / Math.max(0.05, Math.hypot(prev.x - c.x, prev.z - c.z)) - 0.6) / 1.2));
        prev = c.clone();
        // width: grows downstream, varies smoothly (pools and narrows every few metres), and a fall over a step
        // contracts to a narrow jet (0.1-0.25 m) clinging to the rock
        const wv = 0.75 + 0.25 * Math.sin(along * 0.9 + s0) + 0.15 * Math.sin(along * 2.3 + 1.7 * s0);
        const hw = (0.1 + 0.2 * Math.min(1, (dTop - d) / dTop)) * wv * flowMul * (1 - 0.55 * steep);
        const t = road.tangentAt(s, L).setY(0).normalize();
        const row = { pts: [], u01: [], across: [], along, foam: Math.min(1, 0.15 + steep * 0.9), speed: 1.0 + steep * 1.8, turb };
        for (let k = 0; k < 4; k++) {
          const f = k / 3, a = (f - 0.5) * 2 * hw;
          const p = c.clone().addScaledVector(t, a);
          const y = self._groundY(p.x, p.z, yc + 3);
          // drape: follow the V of the bed, a few cm proud; the centre rides a little higher (flowing body)
          p.y = (y === null ? yc : Math.max(y, yc - 0.3)) + 0.03 + (k === 1 || k === 2 ? 0.015 : 0);
          row.pts.push(p); row.u01.push(f); row.across.push(a);
        }
        rows.push(row);
        fall.push(steep > 0.3 ? { p: c.clone(), hw, d } : null);
      }
      strip(rows);
      // steep runs -> short segments (<= ~3 m) along the jet
      let run = [];
      const flushRun = () => {
        for (let k = 0; k + 1 < run.length; k += 3) {
          const a = run[k], b = run[Math.min(k + 3, run.length - 1)];
          if (a.p.y - b.p.y > 0.4) self._falls.push({ top: a.p, bot: b.p, hw: (a.hw + b.hw) / 2, d: b.d });
        }
        run = [];
      };
      for (const f of fall) { if (f) run.push(f); else flushRun(); }
      flushRun();
    };
    this._falls = [];
    for (const g of mk.gullies || []) yield* rivulet(g, 9, 44, 0.3);
    yield* rivulet(gap, 5, 30, 0.55, 1.2);
    // muddy rills down the slide scar (from the head of the lobes to the road)
    yield* rivulet(scar[0] + 26, 5, 30, 0.9, 0.8);
    yield* rivulet(scar[0] + 71, 5, 34, 0.9, 0.7);
    // wet halo around the falls (terrain_near material)
    const fu = this.materials[0]?.userData?.uniforms;
    if (fu?.uFallA) {
      // the jets down the road cut (nearest the road) matter most
      this._falls.sort((a, b) => a.d - b.d);
      this._fallT = 0;              // (QA) re-select the falls near the camera on the next update (_updateFalls)
    }
    if (!idx.length) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    g.setAttribute('wfx', new THREE.Float32BufferAttribute(fx, 4));
    g.setIndex(idx);
    g.computeVertexNormals();
    g.computeBoundingSphere();
    const old = this.water.geometry;
    this.water.geometry = g;
    this.water.frustumCulled = true;
    old.dispose();
    if (ctx.flags?.debug) console.log(`[terrain] water: ${idx.length / 3} tris`);
  }

  _updateRocks() {
    const cam = this.ctx.camera;
    if (!this.rockItems || !cam) return;
    const c = cam.position;
    if (c.distanceToSquared(this._rockCam) < 9) return;
    this._rockCam.copy(c);
    // LOD radii (m): big boulders hi/lod1/lod2 22/50/130, mid boulders 15/32/60, stones 34.
    // (QA perf: was 28/60, 20/38, 40; the scar view at s=150 drew 410k rock triangles + 150k in the shadow pass.
    // A boulder keeps its silhouette and baked normal map through the LODs, so the earlier switch is not visible.)
    const R_BIG = [22 * 22, 50 * 50, 130 * 130], R_MID = [15 * 15, 32 * 32, 60 * 60], STONE_R2 = 34 * 34;
    const B = this._rockBuckets;
    for (const b of this.rockBuckets) b.mesh.count = 0;
    for (const it of this.rockItems) {
      const dx = it.x - c.x, dz = it.z - c.z, dy = it.y - c.y;
      const d2 = dx * dx + dz * dz + dy * dy;
      let k = -1;
      if (it.hi >= 0) {
        const R = it.big ? R_BIG : R_MID;
        if (d2 < R[0]) k = it.hi; else if (d2 < R[1]) k = it.hi + 10; else if (d2 < R[2]) k = it.hi + 20;
      } else if (d2 < (it.r2 || STONE_R2)) k = it.stone;
      if (k < 0 || !B[k]?.mesh) continue;
      const m = B[k].mesh;
      m.setColorAt(m.count, it.col);
      m.setMatrixAt(m.count++, it.m);
    }
    for (const b of this.rockBuckets) {
      b.mesh.instanceMatrix.needsUpdate = true;
      if (b.mesh.instanceColor) b.mesh.instanceColor.needsUpdate = true;
    }
  }

  _decodeGrass() {
    const gm = this.scatter?.grassMask;
    if (!gm?.data) return;
    try {
      const bin = atob(gm.data);
      const a = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
      this._grass = { ...gm, arr: a };
    } catch (e) { console.warn('[terrain] grassMask decode failed', e); }
  }

  /** Vegetation density 0..1 at world (x, z) (0 outside the near corridor). */
  grassAt(x, z) {
    const g = this._grass, road = this.ctx.road;
    if (!g || !road) return 0;
    const pr = road.project(_v.set(x, 0, z), this._pr);
    const i = Math.round((pr.s - g.s0) / g.ds), j = Math.round((pr.d - g.d0) / g.dd);
    if (i < 0 || j < 0 || i >= g.ns || j >= g.nd) return 0;
    return g.arr[i * g.nd + j] / 255;
  }

  /**
   * (QA) Portal collar. The headwall's opening polygon and the tube's first profile ring have different segment
   * counts, so between them there are slivers a few cm wide that look straight through to the sky: a bright, broken
   * white line traced around the tunnel mouth. This adds a thin concrete arch ring (the lining's end face, as on a
   * real portal) 2 cm in front of both, from 3 cm inside the tube profile to 45 cm outside it. Built from the tube's
   * own first ring, so it always matches the asset. Returns a Mesh in `root`'s space, or null.
   */
  _portalCollar(tunnelMesh, mat, root) {
    const road = this.ctx.road, g = tunnelMesh.geometry;
    const P = g.attributes.position, M = g.attributes.masks, UV = g.attributes.uv;
    if (!P || !M || !UV) return null;
    let s0 = Infinity;
    for (let i = 0; i < P.count; i++) if (M.getW(i) > 0.5 && UV.getY(i) > TUNNEL.s0 - 30 && UV.getY(i) < s0) s0 = UV.getY(i);
    if (!isFinite(s0)) return null;
    tunnelMesh.updateWorldMatrix(true, false);
    const c = road.pointAt(s0), l = road.leftAt(s0), t = road.tangentAt(s0); t.y = 0; t.normalize();
    const pts = [];
    const w = new THREE.Vector3();
    for (let i = 0; i < P.count; i++) {
      if (M.getW(i) < 0.5 || Math.abs(UV.getY(i) - s0) > 0.05) continue;
      w.fromBufferAttribute(P, i).applyMatrix4(tunnelMesh.matrixWorld);
      const d = (w.x - c.x) * l.x + (w.z - c.z) * l.z, h = w.y - c.y;
      if (h < 0.12) continue;                                          // not on the road / kerb
      if (pts.some((q) => Math.abs(q.d - d) < 0.02 && Math.abs(q.h - h) < 0.02)) continue;   // uv-seam duplicates
      pts.push({ d, h });
    }
    if (pts.length < 6) return null;
    const crown = Math.max(...pts.map((q) => q.h)), half = Math.max(...pts.map((q) => Math.abs(q.d)));
    const hc = Math.max(0.5, crown - half);                             // centre of the vault arc
    pts.forEach((q) => { q.a = Math.atan2(q.d, q.h - hc); });
    pts.sort((a, b) => a.a - b.a);
    const IN = 0.03, OUT = 0.45, FRONT = 0.02;
    const n = pts.length, pos = new Float32Array(n * 2 * 3), uv = new Float32Array(n * 2 * 2), nor = new Float32Array(n * 2 * 3);
    const masks = new Float32Array(n * 2 * 4);
    const base = c.clone().addScaledVector(t, -FRONT);
    const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
    let arc = 0;
    for (let k = 0; k < n; k++) {
      const q = pts[k];
      if (k) arc += Math.hypot(q.d - pts[k - 1].d, q.h - pts[k - 1].h);
      // outward in the portal plane: radial on the vault, horizontal below the springing line
      let od = q.d, oh = q.h - hc;
      if (oh < 0) oh = 0;
      const ol = Math.hypot(od, oh) || 1; od /= ol; oh /= ol;
      for (let j = 0; j < 2; j++) {
        const r = j ? OUT : -IN;
        w.copy(base).addScaledVector(l, q.d + od * r); w.y = c.y + q.h + oh * r;
        w.applyMatrix4(inv);
        const vi = k * 2 + j;
        pos.set([w.x, w.y, w.z], vi * 3);
        nor.set([-t.x, 0, -t.z], vi * 3);
        uv.set([arc, r + q.h], vi * 2);
        masks.set([0.75, 0.4, 0, 0], vi * 4);                           // AO, part id 0.4 = headwall concrete
      }
    }
    const idx = [];
    for (let k = 0; k < n - 1; k++) {
      const a = k * 2, b = a + 1, cc = a + 2, d = a + 3;
      idx.push(a, cc, b, b, cc, d);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setAttribute('masks', new THREE.BufferAttribute(masks, 4));
    geo.setIndex(idx);
    // make the winding agree with the -t normal (front faces toward the approaching road)
    const A = new THREE.Vector3().fromBufferAttribute(geo.attributes.position, idx[0]);
    const B = new THREE.Vector3().fromBufferAttribute(geo.attributes.position, idx[1]);
    const C = new THREE.Vector3().fromBufferAttribute(geo.attributes.position, idx[2]);
    const fn = B.sub(A).cross(C.sub(A));
    if (fn.x * -t.x + fn.z * -t.z < 0) for (let i = 0; i < idx.length; i += 3) { const x = idx[i + 1]; idx[i + 1] = idx[i + 2]; idx[i + 2] = x; }
    geo.setIndex(idx);
    geo.computeBoundingSphere();
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = 'tunnel_portal_collar';
    mesh.receiveShadow = true; mesh.castShadow = false;
    mesh.matrixAutoUpdate = false;
    return mesh;
  }

  /**
   * (QA) Inside the tube nothing lies under the road strip: its edge (d = +-3.049, 6 cm below the crown, 0.5 m
   * vertices) meets the tunnel kerb face (d = +-3.05, 1 m vertices) along T-junction cracks, and the road's skirt
   * faces away from the camera (culled), so the sky dome showed through as a row of single-pixel blue-white sparkles
   * along the foot of both kerbs. A dark unlit strip 25 cm below the centreline, wide enough to catch grazing rays
   * that run on under the kerb and walkway (d = +-5.2), closes every such crack; it is hidden everywhere else.
   */
  _tunnelUnderlay(root) {
    const road = this.ctx.road;
    if (!road) return null;
    const s0 = TUNNEL.s0 - 1.0, s1 = TUNNEL.s1, W = 5.2, DY = -0.25;
    const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
    const pos = [], idx = [], c = new THREE.Vector3(), l = new THREE.Vector3(), w = new THREE.Vector3();
    const n = Math.ceil(s1 - s0);
    for (let k = 0; k <= n; k++) {
      const s = s0 + ((s1 - s0) * k) / n;
      road.pointAt(s, c); road.leftAt(s, l);
      for (const d of [W, -W]) {
        w.copy(c).addScaledVector(l, d); w.y = c.y + DY;
        w.applyMatrix4(inv);
        pos.push(w.x, w.y, w.z);
      }
      if (k) { const a = (k - 1) * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx);
    geo.computeBoundingSphere();
    const mat = new THREE.MeshBasicMaterial({ color: 0x0b0907, side: THREE.DoubleSide });
    this.materials?.push(mat);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = 'tunnel_underlay';
    mesh.castShadow = false; mesh.receiveShadow = false;
    mesh.matrixAutoUpdate = false;
    return mesh;
  }

  /**
   * The headwall's arch ring meets the face along a line with T-junction cracks (terrain.glb): a dotted bright line
   * traced along the top of the ring. A thin strip of headwall face, 1.5 cm proud of it, from just inside the ring's
   * outer side (hidden in the ring) to 20 cm outside, covers it; the tunnel material shades it in world space, so it
   * is continuous with the face. Returns a Mesh in root's space, or null.
   */
  _portalRingSeal(mat, root) {
    const road = this.ctx.road;
    if (!road) return null;
    const s0 = TUNNEL.s0, SP = PORTAL.spring, W = PORTAL.ringIn, AH = PORTAL.archH, CC = PORTAL.collarC, CW = PORTAL.collar;
    const c = road.pointAt(s0), l = road.leftAt(s0), t = road.tangentAt(s0); t.y = 0; t.normalize();
    // the ring's outer edge exactly as terrain.py builds it: the opening (vertical legs d = +-4.9, then a half
    // ellipse 4.9 x 5.3 over the springing line) pushed out 0.85 m radially from (0, 3.2), never below the opening
    const opening = [];
    for (let h = 0.15; h < SP; h += 0.35) opening.push([-W, h]);
    for (let k = 0; k <= 64; k++) { const a = Math.PI - (Math.PI * k) / 64; opening.push([W * Math.cos(a), SP + AH * Math.sin(a)]); }
    for (let h = SP - 0.35; h >= 0.15; h -= 0.35) opening.push([W, h]);
    const path = opening.map(([d, h]) => {
      const ox = d, oy = h - CC, ol = Math.hypot(ox, oy) || 1;
      const cd = d + (ox / ol) * CW, ch = Math.max(h + (oy / ol) * CW, h);
      const od = cd - d, oh = ch - h, n = Math.hypot(od, oh) || 1;
      return [cd, ch, od / n, oh / n];
    });
    const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
    const pos = [], nor = [], uv = [], masks = [], idx = [];
    const w = new THREE.Vector3();
    path.forEach(([d, h, od, oh], k) => {
      for (const r of [-0.06, 0.2]) {
        w.copy(c).addScaledVector(l, d + od * r).addScaledVector(t, PORTAL.front - 0.015);
        w.y = c.y + h + oh * r;
        w.applyMatrix4(inv);
        pos.push(w.x, w.y, w.z); nor.push(-t.x, 0, -t.z); uv.push(k * 0.3, r); masks.push(0.85, 0.4, 0.1, 0);
      }
      if (k) { const a = (k - 1) * 2; idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setAttribute('masks', new THREE.Float32BufferAttribute(masks, 4));
    // wind toward the approaching road (-t)
    const A = new THREE.Vector3(pos[0], pos[1], pos[2]), B = new THREE.Vector3(pos[6], pos[7], pos[8]), C = new THREE.Vector3(pos[3], pos[4], pos[5]);
    const fn = B.sub(A).cross(C.sub(A)).transformDirection(root.matrixWorld);
    if (fn.x * -t.x + fn.z * -t.z < 0) for (let i = 0; i < idx.length; i += 3) { const x = idx[i + 1]; idx[i + 1] = idx[i + 2]; idx[i + 2] = x; }
    geo.setIndex(idx);
    geo.computeBoundingSphere();
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = 'tunnel_portal_ring_seal';
    mesh.receiveShadow = true; mesh.castShadow = false;
    mesh.matrixAutoUpdate = false;
    return mesh;
  }

  /**
   * Portal drainage channel (REAL-WORLD MODEL: precast concrete U-channel, 0.30 m clear width, 0.20 m deep, 8 cm
   * walls, set flush with the apron, carrying the headwall and wing-wall runoff back to the road ditch). Runs along
   * the headwall foot on both sides of the ring and along both wing-wall bases to s ~1133, with a shallow film of
   * running water in the bottom. One mesh in the tunnel material (masks: R = AO, G = 0.3 part id, B = water).
   */
  _buildPortalChannel() {
    const { ctx } = this;
    const road = ctx.road, phys = ctx.physics, mat = this._matTunnel;
    if (!road || !mat) return;
    const s0 = TUNNEL.s0, face = s0 + PORTAL.front;
    const prw = (s) => THREE.MathUtils.clamp((s - 1131.0) / 18.5, 0, 1);
    const wingD = (s, side) => (side > 0 ? 4.6 + 10.6 * prw(s) : 3.9 + 11.3 * prw(s)) - 0.75;   // wing face |d|
    const P0 = road.pointAt(s0), L0 = road.leftAt(s0), T0 = road.tangentAt(s0); T0.y = 0; T0.normalize();
    const runs = [];
    for (const side of [1, -1]) {
      const hw = [], dEnd = wingD(face, side) - 0.02;
      for (let d = PORTAL.ringOut + 0.3; ; d += 0.5) {
        const dd = Math.min(d, dEnd);
        hw.push(P0.clone().addScaledVector(L0, side * dd).addScaledVector(T0, PORTAL.front - 0.26));
        if (dd >= dEnd) break;
      }
      runs.push(hw);
      const wr = [];
      for (let s = face - 0.3; s >= 1133; s -= 0.5) wr.push(road.worldAt(s, side * (wingD(s, side) - 0.27)));
      runs.push(wr);
    }
    const yRoad = P0.y;
    const down = new THREE.Vector3(0, -1, 0), o = new THREE.Vector3();
    const ground = (p) => {
      const pr = road.project(p, {});
      const top = road.pointAt(pr.s).y + 3.0;
      const hit = phys?.world ? phys.raycast(o.set(p.x, top, p.z), down, 9, { groups: groups(G.ALL, G.STATIC) }) : null;
      return hit ? hit.point.y : yRoad;
    };
    // cross-section (lateral offset, height below the rim top, AO, water)
    const prof = [[-0.23, -0.22, 0.75], [-0.23, 0, 1.0], [-0.15, 0, 1.0], [-0.15, -0.2, 0.55], [0.15, -0.2, 0.5], [0.15, 0, 0.55], [0.23, 0, 1.0], [0.23, -0.22, 0.75]];
    const pos = [], nor = [], uv = [], msk = [], idx = [];
    const a = new THREE.Vector3(), b = new THREE.Vector3(), n = new THREE.Vector3(), t = new THREE.Vector3(), lat = new THREE.Vector3();
    for (const run of runs) {
      if (run.length < 2) continue;
      // the terrain is not cut: the channel stands 14 cm proud and its water surface lies 3 cm above the ground,
      // so the apron never shows through inside the U
      const ys = run.map((p) => ground(p) + 0.14);
      // strips: each profile edge k -> k+1 (flat normals), then the water surface
      const strips = [];
      for (let k = 0; k < prof.length - 1; k++) strips.push([prof[k], prof[k + 1], 0]);
      strips.push([[-0.15, -0.11, 1.0], [0.15, -0.11, 1.0], 1]);
      let arc = 0;
      const base0 = pos.length / 3;
      for (let i = 0; i < run.length; i++) {
        const p = run[i], pp = run[Math.max(0, i - 1)], pn = run[Math.min(run.length - 1, i + 1)];
        t.subVectors(pn, pp); t.y = 0; t.normalize();
        lat.set(-t.z, 0, t.x);
        if (i) arc += p.distanceTo(run[i - 1]);
        for (const [p0, p1, water] of strips) {
          a.copy(p).addScaledVector(lat, p0[0]); a.y = ys[i] + p0[1];
          b.copy(p).addScaledVector(lat, p1[0]); b.y = ys[i] + p1[1];
          // the section is traversed outer-left -> over the rims and through the U -> outer-right, so the air side
          // is always to the left of travel: n = (-dh, dl) in the (lateral, up) plane
          const dl = p1[0] - p0[0], dh = p1[1] - p0[1];
          n.copy(lat).multiplyScalar(-dh); n.y += dl;
          if (water) n.set(0, 1, 0);
          n.normalize();
          for (const [q, pr] of [[a, p0], [b, p1]]) {
            pos.push(q.x, q.y, q.z); nor.push(n.x, n.y, n.z); uv.push(arc, pr[0]);
            // water: AO doubles as specular occlusion. In this walled cut the water mirrors the dark walls and rock,
            // not the open sky that the IBL probe sees
            msk.push(water ? 0.12 : pr[2], 0.3, water, 0);
          }
        }
      }
      const per = strips.length * 2;
      for (let i = 0; i < run.length - 1; i++) {
        for (let k = 0; k < strips.length; k++) {
          const v0 = base0 + i * per + k * 2, v1 = v0 + 1, v2 = v0 + per, v3 = v2 + 1;
          idx.push(v0, v2, v1, v1, v2, v3);
        }
      }
      // end caps: the two wall sections (and the floor slab) of the U, facing out of each end of the run
      for (const end of [0, run.length - 1]) {
        const p = run[end], q = run[end ? end - 1 : 1];
        t.subVectors(p, q); t.y = 0; t.normalize();          // outward along the run
        // same lateral as the section at this end (built from the neighbour span, as above)
        const lt = new THREE.Vector3().subVectors(run[Math.min(run.length - 1, end + 1)], run[Math.max(0, end - 1)]); lt.y = 0; lt.normalize();
        lat.set(-lt.z, 0, lt.x);
        for (const quad of [[[-0.23, -0.22], [-0.23, 0], [-0.15, 0], [-0.15, -0.22]], [[0.15, -0.22], [0.15, 0], [0.23, 0], [0.23, -0.22]], [[-0.15, -0.22], [-0.15, -0.2], [0.15, -0.2], [0.15, -0.22]]]) {
          const b0 = pos.length / 3;
          for (const [ol, oh] of quad) {
            a.copy(p).addScaledVector(lat, ol); a.y = ys[end] + oh;
            pos.push(a.x, a.y, a.z); nor.push(t.x, 0, t.z); uv.push(ol, oh); msk.push(0.8, 0.3, 0, 0);
          }
          idx.push(b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3);
        }
      }
    }
    if (!idx.length) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setAttribute('masks', new THREE.Float32BufferAttribute(msk, 4));
    geo.setIndex(idx);
    // wind every triangle so its face agrees with its vertex normal
    const P = geo.attributes.position, N = geo.attributes.normal, ix = geo.index.array;
    const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), fn = new THREE.Vector3();
    for (let i = 0; i < ix.length; i += 3) {
      a.fromBufferAttribute(P, ix[i]); e1.fromBufferAttribute(P, ix[i + 1]).sub(a); e2.fromBufferAttribute(P, ix[i + 2]).sub(a);
      fn.crossVectors(e1, e2);
      n.fromBufferAttribute(N, ix[i]);
      if (fn.dot(n) < 0) { const x = ix[i + 1]; ix[i + 1] = ix[i + 2]; ix[i + 2] = x; }
    }
    geo.computeBoundingSphere();
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = 'tunnel_portal_channel';
    mesh.receiveShadow = true; mesh.castShadow = false;
    mesh.matrixAutoUpdate = false;
    ctx.scene.add(mesh);
    this.channel = mesh;
  }

  /** 0 = outside, 1 = deep in the tunnel (daylight gone). */
  insideTunnel(pos) {
    const road = this.ctx.road;
    if (!road) return 0;
    const pr = road.project(pos, this._pr);
    if (pr.s < TUNNEL.s0 - 2 || pr.s > TUNNEL.s1 + 5 || Math.abs(pr.d) > 5.2 || pr.dy > 7.5) return 0;
    // rain and daylight are gone ~6 m past the portal (headwall front face is at s0 - 0.95)
    return THREE.MathUtils.smoothstep(pr.s, TUNNEL.s0 - 0.5, TUNNEL.s0 + 6);
  }

  /** Ground height below (x, z). fromY: cast origin (default: above the local surface; tunnel-aware). */
  heightAt(x, z, fromY) {
    const phys = this.ctx.physics;
    if (!phys?.world) return null;
    let y0 = fromY;
    if (y0 === undefined || y0 === null) {
      y0 = 2000;
      const road = this.ctx.road;
      if (road) {
        const pr = road.project(_v.set(x, 0, z), this._pr);
        if (pr.s > TUNNEL.s0 - 0.5 && pr.s < TUNNEL.s1 + 2 && Math.abs(pr.d) < 5.5) y0 = road.pointAt(pr.s, _v).y + 6.5;
      }
    }
    const h = phys.raycast(_v.set(x, y0, z), _d, y0 + 3000, { groups: groups(G.ALL, G.STATIC) });
    return h ? h.point.y : null;
  }

  surfaceAt(pos) {
    const road = this.ctx.road;
    if (!road || !pos) return 'dirt';
    const pr = road.project(pos, this._pr);
    const s = pr.s, d = pr.d, ad = Math.abs(d);
    const roadY = road.pointAt(s, _v).y;
    if (s > TUNNEL.s0 - 1 && s < TUNNEL.s1 + 2 && ad < 5) return 'asphalt';
    if (Math.abs(s - 560) < 1.9 && ad < 7 && pos.y < roadY - 0.3) return 'mud';
    if (ad <= 3.02) {
      if (s > 95 && s < 195 && d > 1.2) return 'mud';
      return 'asphalt';
    }
    if (d < -3.0 && d > -4.0) return 'gravel';
    if (d < 0 && d > -3.9 - 11.0 * pullW(s, road) - 0.1) return 'gravel';
    if (d > 3.0 && d < 4.7) return pos.y < roadY - 0.15 ? 'mud' : 'gravel';
    if (s > 88 && s < 192 && d > 4) return 'mud';
    for (const g of road.markers?.gullies || []) if (Math.abs(s - g) < 5 && d > 4) return 'rock';
    // slope from the collision normal
    const phys = this.ctx.physics;
    if (phys?.world) {
      const h = phys.raycast(_v.set(pos.x, pos.y + 1.5, pos.z), _d, 4, { groups: groups(G.ALL, G.STATIC) });
      if (h && h.normal.y < 0.62) return 'rock';
    }
    const gd = this.grassAt(pos.x, pos.z);
    return gd > 0.45 ? 'grass' : 'dirt';
  }

  fixedUpdate() { this._stepped = true; }

  /**
   * (QA, PERF) The wet halo around the gully falls looped over all 16 fall segments for every terrain pixel inside
   * the AABB of ALL falls, which spans the whole road from the washout to the last gully (0.4 ms at the plank-laying
   * view). Every 0.25 s pick only the falls within ~220 m of the camera (the halo fades out at 200 m), nearest
   * first, pass their count (the shader loop breaks there) and their own AABB, so most of the road rejects the block
   * outright and at a fall only 1-4 segments are tested. Same image.
   */
  _updateFalls(dt) {
    const fu = this.materials[0]?.userData?.uniforms, cam = this.ctx.camera;
    if (!fu?.uFallA || !fu.uFallN || !this._falls?.length || !cam) return;
    this._fallT = (this._fallT ?? 0) - dt;
    if (this._fallT > 0) return;
    this._fallT = 0.25;
    const c = cam.position, N = fu.uFallA.value.length;
    const sel = [];
    for (const f of this._falls) {
      const mx = (f.top.x + f.bot.x) * 0.5 - c.x, mz = (f.top.z + f.bot.z) * 0.5 - c.z;
      const d2 = mx * mx + mz * mz;
      if (d2 < 220 * 220) sel.push([d2, f]);
    }
    sel.sort((a, b) => a[0] - b[0]);
    const lo = _fLo.set(1e9, 1e9, 1e9), hi = _fHi.set(-1e9, -1e9, -1e9);
    const n = Math.min(N, sel.length);
    for (let i = 0; i < N; i++) {
      if (i >= n) { fu.uFallA.value[i].set(0, -1e4, 0, 0.1); fu.uFallB.value[i].set(0, -1e4 - 1, 0, 0); continue; }
      const f = sel[i][1];
      fu.uFallA.value[i].set(f.top.x, f.top.y + 0.2, f.top.z, f.hw);
      fu.uFallB.value[i].set(f.bot.x, f.bot.y, f.bot.z, 0);
      const r = f.hw * 2 + 1.5;
      lo.set(Math.min(lo.x, f.top.x - r, f.bot.x - r), Math.min(lo.y, f.bot.y - r), Math.min(lo.z, f.top.z - r, f.bot.z - r));
      hi.set(Math.max(hi.x, f.top.x + r, f.bot.x + r), Math.max(hi.y, f.top.y + r), Math.max(hi.z, f.top.z + r, f.bot.z + r));
    }
    fu.uFallN.value = n;
    if (fu.uFallMin) { fu.uFallMin.value.copy(lo); fu.uFallMax.value.copy(hi); }
  }

  update(dt) {
    for (const m of this.materials) m.userData.update?.(dt);
    this._updateFalls(dt);
    if (this._waterPending && this._stepped) {
      try { this._stepWater(4); } catch (e) { this._waterPending = false; console.warn('[terrain] water build failed', e); }
    }
    if (this._chanPending && this._stepped) {
      this._chanPending = false;
      try { this._buildPortalChannel(); } catch (e) { console.warn('[terrain] portal channel', e); }
    }
    this._updateRocks();
    // tunnel point lights follow the camera through the tunnel
    const cam = this.ctx.camera;
    if (!cam || !this.tunnelLights?.length || !this.lampPos.length) return;
    const pr = this.ctx.road?.project(cam.position, this._prCam || (this._prCam = {}));
    const near = pr && pr.s > TUNNEL.s0 - 25 && pr.s < TUNNEL.s1 && Math.abs(pr.d) < 12;
    TUNNEL_SHARED.uCamIn.value = this.insideTunnel(cam.position);
    if (!near) {
      for (let k = 0; k < this.tunnelLights.length; k++) { this.tunnelLights[k].intensity = 0; TUNNEL_SHARED.uPtSkip.value[k]?.set(0, -1e5, 0); }
      return;
    }
    const cp = cam.position;
    const sorted = this.lampPos.map((p, i) => [p.distanceToSquared(cp), i]).sort((a, b) => a[0] - b[0]);
    for (let k = 0; k < this.tunnelLights.length; k++) {
      const L = this.tunnelLights[k];
      const p = this.lampPos[sorted[k][1]];
      L.position.copy(p);
      TUNNEL_SHARED.uPtSkip.value[k]?.copy(p);
      L.intensity = 16 * THREE.MathUtils.smoothstep(pr.s, TUNNEL.s0 - 25, TUNNEL.s0 + 2);
    }
  }

  dispose() {
    const phys = this.ctx.physics;
    for (const c of this.colliders) { try { phys?.world?.removeCollider(c, false); } catch {} }
    this.colliders = [];
    if (this.root) this.ctx.scene.remove(this.root);
    if (this.rockRoot) this.ctx.scene.remove(this.rockRoot);
    if (this.deadwood) this.ctx.scene.remove(this.deadwood);
    if (this.water) this.ctx.scene.remove(this.water);
    if (this.channel) this.ctx.scene.remove(this.channel);
    for (const L of this.tunnelLights || []) this.ctx.scene.remove(L);
  }
}
