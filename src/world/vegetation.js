// Vegetation system (TREES workstream): forest (instanced LOD meshes + impostors), grass/ferns/flowers near the camera,
// shrubs + deadwood scatter, and the fallen tree across the road at markers.fallenTree.
//
// API (DESIGN.md "vegetation" / "Trees"):
//   vegetation.fallenTree = {
//     object: Group            // the whole tree in the world (hidden once cut)
//     cut: boolean             // true after split()
//     split(): Promise         // swap to the two cut halves (fallen_tree_a/b), make them dynamic Rapier bodies and push
//                              // them apart off the road; resolves once the lane (d in [-3, 0.5]) is clear (<= ~8 s)
//     chopPoint: Vector3       // world position on top of the trunk at the cut (d ~ 0), for GAME's interact prompt
//     hit(strength=1)          // optional visual feedback for a hatchet hit (trunk shake + wood chips)
//     halves: {a, b}           // Object3Ds of the halves (after split)
//   }
//   vegetation.trees          TreeField (src/render/impostor.js)
//   vegetation.grass          GrassField (src/render/grass.js)
//   vegetation.removeTreesNear(pos: Vector3, radius)  // e.g. for trees swept away by the slide (extra)
//   (ROUND 6) the fallen spruce also has two static nodes: fallen_debris (sprays, needles, torn-off branches, bark
//   flakes, soil spilled from the root plate) and fallen_chips (hatchet chips, revealed blow by blow via uReveal).
//   vegetation.ready          Promise resolved when everything is placed
import * as THREE from 'three';
import { TreeField, extractParts, createFoliageMaterial, createBarkMaterial, createTreeDepthMaterial, WIND, FOLIAGE } from '../render/impostor.js';
import { GrassField, corridorVegetation } from '../render/grass.js';
import { url } from '../core/assets.js';
import { G, groups } from '../physics/world.js';

const _v = new THREE.Vector3(), _v2 = new THREE.Vector3(), _q = new THREE.Quaternion(), _m = new THREE.Matrix4();
const Y = new THREE.Vector3(0, 1, 0);

async function probeJSON(path) {
  try {
    const r = await fetch(url(path), { method: 'GET' });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default class Vegetation {
  constructor(ctx) {
    this.ctx = ctx;
    this.group = new THREE.Group();
    this.group.name = 'vegetation';
    this.trees = null;
    this.grass = null;
    this.fallenTree = null;
    this._dyn = [];
    this._time = 0;
    this.foliage = FOLIAGE;   // shared foliage tuning uniforms (debug)
  }

  async init() {
    const { ctx } = this;
    ctx.scene.add(this.group);
    let gltf = null, meta = null, albedo = null, normal = null;
    try {
      [gltf, meta] = await Promise.all([
        ctx.assets.gltf('assets/models/trees.glb'),
        probeJSON('assets/models/impostors/impostors.json'),
      ]);
      if (meta) {
        [albedo, normal] = await Promise.all([
          ctx.assets.texture('assets/models/impostors/' + meta.albedo, { srgb: true, repeat: false, anisotropy: 4 }),
          ctx.assets.texture('assets/models/impostors/' + meta.normal, { srgb: false, repeat: false, anisotropy: 4 }),
        ]);
      }
    } catch (e) {
      console.warn('[vegetation] tree assets unavailable, forest disabled:', e?.message || e);
    }
    this.gltf = gltf;
    this._queriesReady = this._ensureSceneQueries();

    // ---- forest ------------------------------------------------------------------------------------------------
    if (gltf) {
      // conifer_0..3 living spruce/fir, conifer_4 (if present) a dead snag
      const variantNames = [];
      for (let i = 0; i < 8 && gltf.scene.getObjectByName('conifer_' + i); i++) variantNames.push('conifer_' + i);
      this.variantCount = variantNames.length || 4;
      this.trees = new TreeField(ctx, { gltf, meta, albedo, normal, variantNames: variantNames.length ? variantNames : undefined });
      const data = await this._treeData();
      const normals = this._treeNormals(data);
      this._orientTrees(data, normals);
      this.trees.setTrees(data, normals);
      this._buildFallenTree(gltf);
      if (this._queriesReady) this._buildScatter(gltf);
      else this._scatterPending = true;   // placed on the first update, once the physics queries work
    }

    // ---- grass / ferns / flowers ----------------------------------------------------------------------------------
    try {
      this._buildCanopy();
      const gm = this.gltf ? this._srcMat(this.gltf, 'grass') : null;
      if (gm?.map) { gm.map.anisotropy = Math.min(8, ctx.renderer.capabilities.getMaxAnisotropy()); }
      this.grass = new GrassField(ctx, { foliageMap: this.trees?.textures?.folMap, foliageNormal: this.trees?.textures?.folNrm,
        grassMap: gm?.map || null, grassNormal: gm?.normalMap || null, canopyAt: (x, z) => this.canopyAt(x, z) });
      this.group.add(this.grass.group);
    } catch (e) {
      console.warn('[vegetation] grass disabled:', e);
    }

    // chopping feedback from GAME
    ctx.events?.on?.('tree:chop', (p) => {
      if (!this.fallenTree) return;
      this.fallenTree.hit(1);
      if (p?.done && !this.fallenTree.cut) this.fallenTree.split();
    });
    this.ready = Promise.resolve();
  }

  /** Rapier only refreshes its scene-query BVH inside step(). At init nothing has stepped yet, so ray casts against
   *  the freshly built terrain colliders miss. If only fixed bodies exist, one step is side-effect free. */
  _ensureSceneQueries() {
    const phys = this.ctx.physics, road = this.ctx.road;
    if (!phys?.world || !road) return false;
    const test = () => {
      const p = road.pointAt(road.markers?.fallenTree ?? 305, _v);
      return !!phys.raycast(_v2.set(p.x, p.y + 50, p.z), new THREE.Vector3(0, -1, 0), 200, { groups: groups(G.ALL, G.STATIC) });
    };
    if (test()) return true;
    let dynamic = 0;
    try { phys.world.bodies.forEach((b) => { if (!b.isFixed()) dynamic++; }); } catch { dynamic = 1; }
    if (dynamic) return false;
    try { phys.world.step(); } catch { return false; }
    return test();
  }

  // ------------------------------------------------------------------------------------------------------------
  // Tree placement data
  // ------------------------------------------------------------------------------------------------------------
  async _treeData() {
    const { ctx } = this;
    let trees = ctx.terrain?.scatter?.trees;
    if (!trees) {
      const sc = await probeJSON('assets/world/scatter.json');
      trees = sc?.trees;
    }
    const road = ctx.road;
    const out = [];
    const pr = {};
    const ft = road?.markers?.fallenTree ?? 305;
    if (trees && trees.length) {
      for (const t of trees) {
        const [x, y, z, s, r, v] = t;
        if (road) {
          road.project(_v.set(x, y, z), pr);
          // keep the road, ditch and shoulder clear; keep a gap where the fallen tree came down
          if (pr.dist < 5.0 && pr.s > -30 && pr.s < 1310 && Math.abs(pr.dy) < 12) continue;
          if (Math.abs(pr.s - ft) < 10 && pr.d > -8 && pr.d < 16 && Math.abs(pr.dy) < 15) continue;
        }
        out.push(x, y, z, s, r, v);
      }
      const extra = this._densify();
      if (extra) for (let i = 0; i < extra.length; i++) out.push(extra[i]);
      return new Float32Array(out);
    }
    const extra = this._densify();
    if (extra && extra.length) return new Float32Array(extra);
    return this._fallbackScatter();
  }

  /**
   * Closed-canopy densification of the forest seen from the road: area-weighted random points on the terrain
   * triangles (TERRAIN's meshes), weighted by the vegetation mask (COLOR_0.a), slope and a clumping noise.
   * TERRAIN's scatter.json is ~40 trees/ha; a real montane spruce stand is several hundred.
   */
  _densify() {
    const { ctx } = this;
    const t = ctx.terrain, road = ctx.road;
    const chunks = [...(t?.meshes?.nearChunks || []), ...(t?.meshes?.farChunks || [])];
    if (!chunks.length || !road) return null;
    const t0 = performance.now();
    const m = road.markers || {};
    // (s, d) lookup grid around the road
    const P = road.p;
    let xmin = Infinity, xmax = -Infinity, zmin = Infinity, zmax = -Infinity;
    for (let i = 0; i < road.count; i++) {
      xmin = Math.min(xmin, P[i * 3]); xmax = Math.max(xmax, P[i * 3]);
      zmin = Math.min(zmin, P[i * 3 + 2]); zmax = Math.max(zmax, P[i * 3 + 2]);
    }
    const MAXD = 470, CELL = 8;
    xmin -= MAXD; zmin -= MAXD; xmax += MAXD; zmax += MAXD;
    const nx = Math.ceil((xmax - xmin) / CELL), nz = Math.ceil((zmax - zmin) / CELL);
    const GS = new Float32Array(nx * nz), GD = new Float32Array(nx * nz);
    const pr = {};
    for (let j = 0; j < nz; j++) {
      for (let i = 0; i < nx; i++) {
        const ii = j % 2 ? nx - 1 - i : i; // serpentine keeps the projection hint coherent
        road.project(_v.set(xmin + (ii + 0.5) * CELL, 0, zmin + (j + 0.5) * CELL), pr);
        GS[j * nx + ii] = pr.s; GD[j * nx + ii] = pr.d;
      }
    }
    const rnd = mulberry32(4242);
    const hash = (x, z) => { const h = Math.sin(x * 127.1 + z * 311.7) * 43758.5453; return h - Math.floor(h); };
    const vnoise = (x, z) => {
      const ix = Math.floor(x), iz = Math.floor(z), fx = x - ix, fz = z - iz;
      const ux = fx * fx * (3 - 2 * fx), uz = fz * fz * (3 - 2 * fz);
      const a = hash(ix, iz), b = hash(ix + 1, iz), c = hash(ix, iz + 1), d = hash(ix + 1, iz + 1);
      return (a + (b - a) * ux) * (1 - uz) + (c + (d - c) * ux) * uz;
    };
    const smooth = (a, b, x) => { const u = Math.min(1, Math.max(0, (x - a) / (b - a))); return u * u * (3 - 2 * u); };
    const out = [], nrmOut = [];
    const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3(), E1 = new THREE.Vector3(), E2 = new THREE.Vector3(), N = new THREE.Vector3();
    const ft = m.fallenTree ?? 305;
    const MAX_TREES = 38000;   // impostors are 2 triangles each; the cap only bounds the CPU LOD scan
    for (const mesh of chunks) {
      mesh.updateWorldMatrix(true, false);
      const g = mesh.geometry, pos = g.attributes.position, col = g.attributes.color, idx = g.index;
      if (!pos) continue;
      const mw = mesh.matrixWorld;
      const tris = idx ? idx.count / 3 : pos.count / 3;
      for (let tI = 0; tI < tris; tI++) {
        const i0 = idx ? idx.getX(tI * 3) : tI * 3, i1 = idx ? idx.getX(tI * 3 + 1) : tI * 3 + 1, i2 = idx ? idx.getX(tI * 3 + 2) : tI * 3 + 2;
        A.fromBufferAttribute(pos, i0).applyMatrix4(mw);
        B.fromBufferAttribute(pos, i1).applyMatrix4(mw);
        C.fromBufferAttribute(pos, i2).applyMatrix4(mw);
        const cx = (A.x + B.x + C.x) / 3, cz = (A.z + B.z + C.z) / 3;
        const gi = Math.floor((cx - xmin) / CELL), gj = Math.floor((cz - zmin) / CELL);
        if (gi < 0 || gj < 0 || gi >= nx || gj >= nz) continue;
        const s = GS[gj * nx + gi], d = GD[gj * nx + gi], ad = Math.abs(d);
        if (ad < 6.5 || ad > MAXD || s < -60 || s > 1320) continue;
        if (s > (m.tunnel ?? 1150) - 25 && ad < 60) continue;
        if (Math.abs(s - ft) < 12 && d > -9 && d < 18) continue;
        E1.subVectors(B, A); E2.subVectors(C, A); N.crossVectors(E1, E2);
        const len = N.length();
        if (len < 1e-6) continue;
        const ny = Math.abs(N.y) / len;
        if (ny < 0.62) continue;
        const areaH = Math.abs(N.y) * 0.5;
        let veg = col ? (col.getW(i0) + col.getW(i1) + col.getW(i2)) / 3 : 1;
        if (ad < 70) veg = Math.max(veg, corridorVegetation(m, s, d, ny) * (d < 0 && d > -16 ? 0.55 : 1));
        if (veg < 0.04) continue;
        // closed-canopy montane spruce stand (~350 stems/ha near the road, thinning with distance where the
        // impostors merge into a continuous canopy anyway)
        const base = ad < 130 ? 0.035 : ad < 300 ? 0.035 - (ad - 130) / 170 * 0.015 : 0.02 - (ad - 300) / 170 * 0.008;
        const clump = 0.25 + 1.1 * Math.max(0, vnoise(cx / 55, cz / 55) * 0.75 + vnoise(cx / 17, cz / 17) * 0.45 - 0.2);
        const dens = base * veg * smooth(0.62, 0.8, ny) * clump;
        const exp = areaH * dens;
        let k = Math.floor(exp);
        if (rnd() < exp - k) k++;
        for (let q = 0; q < k; q++) {
          let u = rnd(), v = rnd();
          if (u + v > 1) { u = 1 - u; v = 1 - v; }
          const px = A.x + E1.x * u + E2.x * v, py = A.y + E1.y * u + E2.y * v, pz = A.z + E1.z * u + E2.z * v;
          const r = rnd();
          const variant = r < 0.28 ? 0 : r < 0.54 ? 1 : r < 0.76 ? 2 : 3;
          const sc = 0.72 + rnd() * 0.5 + (ad > 150 ? 0.1 : 0);
          out.push(px, py - 0.25, pz, sc, rnd() * Math.PI * 2, variant);
          nrmOut.push(N.x / len * Math.sign(N.y || 1), Math.abs(N.y) / len, N.z / len * Math.sign(N.y || 1));
        }
        if (out.length / 6 > MAX_TREES) break;
      }
      if (out.length / 6 > MAX_TREES) break;
    }
    this._densifyInfo = { trees: out.length / 6, ms: Math.round(performance.now() - t0) };
    this._densifyNormals = new Float32Array(nrmOut);
    return out;
  }

  /** Scatter trees along the road corridor when TERRAIN's scatter.json is not available. */
  _fallbackScatter() {
    const { ctx } = this;
    const road = ctx.road;
    if (!road) return new Float32Array(0);
    const rnd = mulberry32(1234);
    const out = [];
    const t = ctx.terrain;
    const m = road.markers || {};
    const bad = (s, d) => {
      if (m.scar && s > m.scar[0] - 5 && s < m.scar[1] + 5 && d > 0 && d < 220) return true;
      if (m.gullies) for (const g of m.gullies) if (Math.abs(s - g) < 12 && d > 0) return true;
      if (Math.abs(s - (m.gap ?? 560)) < 8) return true;
      if (Math.abs(s - (m.fallenTree ?? 305)) < 10 && d > -8 && d < 16) return true;
      if (m.pulloff && s > m.pulloff[0] && s < m.pulloff[1] && d > -18 && d < 0) return true;
      if (s > (m.tunnel ?? 1150) - 40) return true;
      return false;
    };
    for (let s = -40; s < 1300; s += 3.2) {
      const base = road.pointAt(s, _v2).clone();
      for (const side of [1, -1]) {
        for (let k = 0; k < 14; k++) {
          const d = side * (side > 0 ? 13 : 6.5) + side * Math.pow(rnd(), 1.4) * 230;
          const ss = s + (rnd() - 0.5) * 3;
          if (bad(ss, d)) continue;
          if (rnd() > 0.55) continue;
          const p = road.worldAt(ss, d, _v);
          let y = null;
          if (t?.heightAt && Math.abs(d) < 44) y = t.heightAt(p.x, p.z);
          if (y === null || y === undefined) {
            y = side > 0 ? base.y + 9 + (d - 10) * 0.75 : base.y - (Math.abs(d) - 4) * 0.85;
          }
          out.push(p.x, y - 0.2, p.z, 0.7 + rnd() * 0.65, rnd() * Math.PI * 2, Math.floor(rnd() * 4));
        }
      }
    }
    return new Float32Array(out);
  }

  /** Ground normal under every tree (densified trees remember their triangle; others ray cast; default up). */
  _treeNormals(data) {
    const n = data.length / 6;
    const out = new Float32Array(n * 3);
    const phys = this.ctx.physics;
    const known = this._densifyNormals;
    const offset = n - (known ? known.length / 3 : 0);
    const dn = new THREE.Vector3(0, -1, 0);
    for (let i = 0; i < n; i++) {
      if (known && i >= offset) {
        const k = (i - offset) * 3;
        out[i * 3] = known[k]; out[i * 3 + 1] = known[k + 1]; out[i * 3 + 2] = known[k + 2];
        continue;
      }
      out[i * 3 + 1] = 1;
      if (this._queriesReady && phys?.world) {
        const h = phys.raycast(_v.set(data[i * 6], data[i * 6 + 1] + 6, data[i * 6 + 2]), dn, 14, { groups: groups(G.ALL, G.STATIC) });
        if (h) { out[i * 3] = h.normal.x; out[i * 3 + 1] = h.normal.y; out[i * 3 + 2] = h.normal.z; }
      }
    }
    return out;
  }

  /**
   * Real trees on a slope are not randomly oriented: the crown grows longer toward the open valley (light) and the
   * trunk's uphill side stays wet and mossy. The Blender trees put the long crown side at local +Z and the mossy side
   * at local -Z, so each tree is turned to face its local -Z uphill (+-35 deg jitter). Flat ground keeps the random yaw.
   * Also turns ~3.5 % of the living trees into dead snags (variant 4) when the GLB has one.
   */
  _orientTrees(data, normals) {
    const n = data.length / 6;
    const rnd = mulberry32(9151);
    const snag = this.variantCount > 4;
    for (let i = 0; i < n; i++) {
      const o = i * 6;
      const r0 = rnd(), r1 = rnd();
      if (normals) {
        const nx = normals[i * 3], nz = normals[i * 3 + 2];
        const h = Math.hypot(nx, nz);
        if (h > 0.08) {
          const ux = -nx / h, uz = -nz / h;          // uphill (horizontal)
          data[o + 4] = Math.atan2(-ux, -uz) + (r0 - 0.5) * 1.2;
        }
      }
      if (snag && r1 < 0.035 && data[o + 5] < 4) data[o + 5] = 4;
    }
  }

  /** Canopy cover grid (2 m cells, 0..1) from the tree positions: the ground flora thins and turns to moss, bilberry
   *  and ferns under the crowns (a closed spruce stand passes only ~5-15 % of the light to the floor). */
  _buildCanopy() {
    const f = this.trees;
    this._canopy = null;
    if (!f?.data || !f.count) return;
    const C = 2, grid = new Map(), d = f.data, radii = [3.1, 2.7, 3.3, 3.3, 1.2];
    for (let i = 0; i < f.count; i++) {
      const o = i * 6, s = d[o + 3];
      if (s <= 0) continue;
      const R = (radii[(d[o + 5] | 0) % radii.length] || 3) * s * 1.2;
      const x = d[o], z = d[o + 2];
      const i0 = Math.floor((x - R) / C), i1 = Math.floor((x + R) / C), j0 = Math.floor((z - R) / C), j1 = Math.floor((z + R) / C);
      for (let ii = i0; ii <= i1; ii++) {
        for (let jj = j0; jj <= j1; jj++) {
          const dx = (ii + 0.5) * C - x, dz = (jj + 0.5) * C - z;
          const q = 1 - Math.hypot(dx, dz) / R;
          if (q <= 0) continue;
          const k = ii * 73856093 ^ jj * 19349663;
          const v = Math.min(1, (grid.get(k) || 0) + q * 0.8);
          grid.set(k, v);
        }
      }
    }
    this._canopy = { C, grid };
  }

  canopyAt(x, z) {
    const c = this._canopy;
    if (!c) return 0;
    return c.grid.get(Math.floor(x / c.C) * 73856093 ^ Math.floor(z / c.C) * 19349663) || 0;
  }

  /** Hide trees within radius of pos (e.g. swept away by the debris flow). */
  removeTreesNear(pos, radius) {
    const f = this.trees;
    if (!f?.data) return 0;
    const r2 = radius * radius;
    let n = 0;
    for (let i = 0; i < f.count; i++) {
      const o = i * 6;
      const dx = f.data[o] - pos.x, dz = f.data[o + 2] - pos.z;
      if (dx * dx + dz * dz < r2 && f.data[o + 3] > 0) {
        f.hideTree(i);   // LOD data + impostor source; the compacted impostor buffer is rebuilt next frame
        n++;
      }
    }
    return n;
  }

  // ------------------------------------------------------------------------------------------------------------
  // Shrubs + deadwood scatter (instanced, distance-limited)
  // ------------------------------------------------------------------------------------------------------------
  _buildScatter(gltf) {
    const { ctx } = this;
    const road = ctx.road, t = ctx.terrain;
    if (!road) return;
    const rnd = mulberry32(777);
    const kinds = ['shrub_0', 'shrub_1', 'shrub_2', 'deadwood_0', 'deadwood_1'];
    const lists = kinds.map(() => []);
    const phys = ctx.physics;
    const m = road.markers || {};
    const pr = {};
    for (let s = -30; s < 1140; s += 1.6) {
      for (let k = 0; k < 5; k++) {
        const side = rnd() < 0.55 ? 1 : -1;
        const d = side * (side > 0 ? 6 + rnd() * 36 : 4.8 + rnd() * 34);
        const ss = s + rnd() * 2.2;
        if (m.scar && ss > m.scar[0] && ss < m.scar[1] && d > 0) continue;
        if (Math.abs(ss - (m.gap ?? 560)) < 4) continue;
        if (m.pulloff && ss > m.pulloff[0] - 3 && ss < m.pulloff[1] + 3 && d < 0 && d > -16) continue;
        if (Math.abs(ss - (m.fallenTree ?? 305)) < 8 && Math.abs(d) < 10) continue;
        const p = road.worldAt(ss, d, _v);
        const hit = phys?.world ? phys.raycast(_v2.set(p.x, p.y + 150, p.z), new THREE.Vector3(0, -1, 0), 450, { groups: groups(G.ALL, G.STATIC) }) : null;
        if (!hit || hit.normal.y < 0.62) continue;
        road.project(hit.point, pr);
        if (pr.dist < 4.8 || (pr.d > 0 && pr.d < 7.0) || (pr.d < 0 && pr.d > -5.5)) continue;
        const dens = Math.max(t?.grassAt ? t.grassAt(p.x, p.z) : 0.6, corridorVegetation(m, pr.s, pr.d, hit.normal.y));
        if (rnd() > dens * 0.85) continue;
        const r = rnd();
        const kind = r < 0.3 ? 0 : r < 0.52 ? 1 : r < 0.72 ? 2 : r < 0.88 ? 3 : 4;
        const sc = kind >= 3 ? 0.8 + rnd() * 0.5 : 0.6 + rnd() * 0.7;
        lists[kind].push(hit.point.x, hit.point.y - (kind >= 3 ? 0.03 : 0.05), hit.point.z, sc, rnd() * Math.PI * 2);
      }
    }
    this.scatter = [];
    const radius = this.scatterRadius = 60;
    for (let ki = 0; ki < kinds.length; ki++) {
      const node = gltf.scene.getObjectByName(kinds[ki]);
      if (!node || !lists[ki].length) continue;
      const parts = extractParts(node, gltf.scene);
      const lod = new THREE.Vector4(-2, -1, radius - 25, radius);
      const n = lists[ki].length / 5;
      const cap = Math.min(n, 400);
      const meshes = [];
      let shared = null;
      for (const [key, geo] of Object.entries(parts)) {
        const src = this._srcMat(gltf, key);
        const mat = key === 'foliage'
          ? createFoliageMaterial({ map: src?.map, normalMap: src?.normalMap, lod })
          : createBarkMaterial({ map: src?.map, normalMap: src?.normalMap, arm: src?.roughnessMap, lod, moss: 0.5,
            color: key === 'bark_grey' ? new THREE.Color(0.9, 0.9, 0.92) : null });
        const im = new THREE.InstancedMesh(geo, mat, cap);
        im.name = kinds[ki] + '_' + key;
        im.count = 0;
        im.frustumCulled = false;
        im.castShadow = false;
        im.receiveShadow = key !== 'foliage';
        im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        if (shared) im.instanceMatrix = shared; else shared = im.instanceMatrix;
        this.group.add(im);
        meshes.push(im);
      }
      this.scatter.push({ kind: kinds[ki], meshes, count: n, data: new Float32Array(lists[ki]), cap });
    }
    this._scatterCam = new THREE.Vector3(1e9, 0, 0);
  }

  /** Select shrub/deadwood instances near the camera (they are spread along the whole road). */
  _updateScatter() {
    const cam = this.ctx.camera;
    if (!cam || !this.scatter?.length) return;
    const q = cam.quaternion;
    const qd = this._scatterQ ? Math.abs(q.dot(this._scatterQ)) : 0;
    if (cam.position.distanceToSquared(this._scatterCam) < 4 && qd > 0.9995) return;
    this._scatterCam.copy(cam.position);
    (this._scatterQ || (this._scatterQ = new THREE.Quaternion())).copy(q);
    const cp = cam.position, R2 = this.scatterRadius * this.scatterRadius;
    cam.updateMatrixWorld();
    const frus = this._frus || (this._frus = new THREE.Frustum());
    frus.setFromProjectionMatrix(_m.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse));
    const sph = this._sph || (this._sph = new THREE.Sphere());
    for (const S of this.scatter) {
      const d = S.data, lead = S.meshes[0];
      let k = 0;
      for (let o = 0; o < d.length && k < S.cap; o += 5) {
        const dx = d[o] - cp.x, dy = d[o + 1] - cp.y, dz = d[o + 2] - cp.z;
        if (dx * dx + dy * dy + dz * dz > R2) continue;
        sph.center.set(d[o], d[o + 1] + 1, d[o + 2]); sph.radius = 2.5 * d[o + 3];
        if (!frus.intersectsSphere(sph)) continue;
        _q.setFromAxisAngle(Y, d[o + 4]);
        _m.compose(_v.set(d[o], d[o + 1], d[o + 2]), _q, _v2.setScalar(d[o + 3]));
        lead.setMatrixAt(k++, _m);
      }
      for (const im of S.meshes) im.count = k;
      lead.instanceMatrix.needsUpdate = true;
    }
  }

  _srcMat(gltf, name) {
    let found = null;
    gltf.scene.traverse((o) => { if (!found && o.isMesh && o.material?.name === name) found = o.material; });
    return found;
  }

  // ------------------------------------------------------------------------------------------------------------
  // Fallen tree
  // ------------------------------------------------------------------------------------------------------------
  _buildFallenTree(gltf) {
    const { ctx } = this;
    const road = ctx.road;
    if (!road) return;
    const s0 = road.markers?.fallenTree ?? 305;
    const whole = gltf.scene.getObjectByName('fallen_tree');
    if (!whole) return;
    const info = (() => {
      try { return JSON.parse(whole.userData?.collider || 'null'); } catch { return null; }
    })() || {
      axis: [[-6.2, 1.0, 0, 0.26], [0, 0.34, 0, 0.18], [4.5, 0.2, 0, 0.12], [10.5, 0.02, 0, 0.05]],
      rootPlate: { center: [-6.5, 1.0, 0], radius: 1.3, halfThickness: 0.25 }, crown: { x0: 1, x1: 11, radius: 2.2 },
      cutRadius: 0.19, topAtCut: 0.53,
    };
    this.fallenInfo = info;

    // Orientation: local +X toward the valley (-left), yawed toward +s so the root plate sits at the foot of the cut face.
    const P0 = road.worldAt(s0, 0.0, new THREE.Vector3());
    const roadY = P0.y;
    const left = road.leftAt(s0, new THREE.Vector3());
    const tan = road.tangentAt(s0, new THREE.Vector3()); tan.y = 0; tan.normalize();
    const rootX = info.rootPlate.center[0];
    // pick the smallest yaw for which the root plate (and the ground just behind it) is not inside the rock cut
    let yaw = 0.95, dirX = new THREE.Vector3();
    const groundAt = (p) => (ctx.terrain?.heightAt ? ctx.terrain.heightAt(p.x, p.z, roadY + 80) : null);
    // v2 model: the ground under the tree was surveyed at a fixed yaw (the crushed crown is draped over that ground)
    for (let a = 0.35; a <= 1.1 && !info.yaw; a += 0.05) {
      dirX.copy(left).multiplyScalar(-Math.cos(a)).addScaledVector(tan, Math.sin(a)).normalize();
      const rp = _v.copy(P0).addScaledVector(dirX, rootX);
      const rp2 = _v2.copy(P0).addScaledVector(dirX, rootX - 1.0);
      const h1 = groundAt(rp), h2 = groundAt(rp2);
      (this._fallenProbe || (this._fallenProbe = [])).push([+a.toFixed(2), h1 === null ? null : +(h1 - roadY).toFixed(2), h2 === null ? null : +(h2 - roadY).toFixed(2), +rootX.toFixed(2)]);
      yaw = a;
      const ok = (h, lim) => h === null || h === undefined || h < roadY + lim;
      if (ok(h1, 1.0) && ok(h2, 2.4)) break;
    }
    if (info.yaw) yaw = info.yaw;
    this._fallenYaw = yaw;
    dirX.copy(left).multiplyScalar(-Math.cos(yaw)).addScaledVector(tan, Math.sin(yaw)).normalize();
    const rotY = Math.atan2(-dirX.z, dirX.x); // rotation about +Y that maps local +X to dirX
    const root = new THREE.Group();
    root.name = 'fallen_tree_root';
    root.position.set(P0.x, roadY + 0.02, P0.z);
    root.quaternion.setFromAxisAngle(Y, rotY);
    this.group.add(root);
    root.updateMatrixWorld(true);

    // (ROUND 6) the fallen spruce has its own materials: 'fallen' (unique baked atlas: stem bark, root plate, end
    // grain, fresh wood; rain-soaked albedo, ARM roughness), and wind-patched foliage/bark (a lying crown does not sway
    // like a standing tree: cards resting on the road are still, cards hanging in the air flutter a little).
    const fmats = this._fallenMaterials(gltf);
    const makeObj = (name) => {
      const node = gltf.scene.getObjectByName(name);
      if (!node) return null;
      const parts = extractParts(node, gltf.scene);
      const g = new THREE.Group();
      g.name = name;
      for (const [key, geo] of Object.entries(parts)) {
        let mat = fmats[key];
        if (!mat) {
          const src = this._srcMat(gltf, key);
          mat = src ? src.clone() : new THREE.MeshStandardMaterial({ color: 0x6b5a45, roughness: 0.9 });
          mat.vertexColors = false; // COLOR_0 was renamed to aTree by extractParts
        }
        if (name === 'fallen_chips' && key === 'fallen') mat = fmats.chips;
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = name !== 'fallen_chips' && name !== 'fallen_debris';
        mesh.receiveShadow = key !== 'foliage';
        if (key === 'foliage') mesh.customDepthMaterial = fmats.foliageDepth;
        g.add(mesh);
      }
      return g;
    };
    const wholeObj = makeObj('fallen_tree');
    root.add(wholeObj);
    const halfA = makeObj('fallen_tree_a'), halfB = makeObj('fallen_tree_b');
    if (halfA) { halfA.visible = false; root.add(halfA); }
    if (halfB) { halfB.visible = false; root.add(halfB); }
    // static road debris (broken sprays, needles, torn-off branch pieces, bark flakes) and the hatchet chips, which
    // appear blow by blow (uReveal against the per-chip order in aTree.g)
    const debris = makeObj('fallen_debris');
    if (debris) root.add(debris);
    const chips = makeObj('fallen_chips');
    if (chips) root.add(chips);

    // (ROUND 6) the crown of the v2 model droops over the drop-off: the crown boxes follow the stem axis, lifted so they
    // never start inside the slope (a dynamic half that spawns interpenetrating gets launched at split). Probed once,
    // before the tree's own fixed body exists.
    if (info.v >= 2 && info.crown) {
      const axisY = (x) => { const a = info.axis; for (let i = 0; i + 1 < a.length; i++) if (x <= a[i + 1][0]) { const f = (x - a[i][0]) / (a[i + 1][0] - a[i][0] || 1); return a[i][1] + (a[i + 1][1] - a[i][1]) * f; } return a[a.length - 1][1]; };
      const c = info.crown, x0c = Math.max(c.x0, 4.2);
      info._crownY = [];
      for (let k = 0; k < 3; k++) {
        const xm = x0c + (c.x1 - x0c) * (k + 0.5) / 3;
        const hh = c.radius * (1 - 0.25 * k) * 0.3;
        let y = axisY(xm);
        for (const z of [-1, 0, 1]) {
          const p = root.localToWorld(_v.set(xm, 0, z * c.radius * 0.5));
          const gy = ctx.terrain?.heightAt ? ctx.terrain.heightAt(p.x, p.z, p.y + 30) : null;
          if (gy !== null && gy !== undefined) y = Math.max(y, gy - root.position.y + hh + 0.08);
        }
        info._crownY.push(y);
      }
    }
    // ---- static colliders (compound capsules + root plate box) on a fixed body
    const phys = ctx.physics;
    let body = null;
    const R = phys?.RAPIER;
    if (R && phys.world) {
      const rq = root.quaternion;
      body = phys.world.createRigidBody(R.RigidBodyDesc.fixed()
        .setTranslation(root.position.x, root.position.y, root.position.z)
        .setRotation({ x: rq.x, y: rq.y, z: rq.z, w: rq.w }));
      for (const cd of this._fallenColliderDescs(info, null)) phys.world.createCollider(cd.desc.setCollisionGroups(groups(G.STATIC)), body);
    }

    const chopLocal = new THREE.Vector3(0, info.topAtCut ?? 0.53, 0);
    const self = this;
    const ft = {
      object: root,
      whole: wholeObj,
      halves: { a: halfA, b: halfB },
      cut: false,
      body,
      rotY, yaw,
      chopPoint: root.localToWorld(chopLocal.clone()),
      _shake: 0,
      _hits: 0,
      hit(strength = 1) {
        this._shake = Math.min(1, this._shake + 0.6 * strength);
        const p = this.chopPoint;
        // (QA) a hatchet blow on a ~38 cm softwood trunk throws a few chips and splinters, not a shower (with the
        // 6 from game/sequence.js this was 20 per blow: confetti)
        try { ctx.particles?.debris?.(p.clone(), 5, { wood: true, speed: 3.2, up: 0.9 }); } catch {}
        this._hits++;
        if (fmats.reveal) fmats.reveal.value = Math.min(1, this._hits / 3.4);
        try { self._chopScar(this, root, info); } catch (e) { console.warn('[vegetation] chop scar', e); }
      },
      split() {
        if (this.cut) return this._splitPromise || Promise.resolve();
        if (this._scar) this._scar.visible = false;
        if (fmats.reveal) fmats.reveal.value = 1;
        this.cut = true;
        this._splitPromise = self._split(this);
        return this._splitPromise;
      },
    };
    this.fallenTree = ft;
  }

  /** Materials of the fallen spruce (see _buildFallenTree). Cached per gltf. */
  _fallenMaterials(gltf) {
    if (this._fmats) return this._fmats;
    const src = (k) => this._srcMat(gltf, k);
    // lying crown: no whole-tree sway (tr_h = 0), a small constant flutter radius instead of the distance from the
    // trunk base, scaled per vertex by aTree.a (0 on cards resting on the road); normals lean up and away from the stem
    const windPatch = (vs, weight) => vs
      .replace('float tr_h = max(position.y, 0.0) * tr_s;', 'float tr_h = 0.0;')
      .replace('float tr_r = length(position.xz) * tr_s;', 'float tr_r = 0.6;')
      .replace('    transformed += tr_dw;', weight ? '    transformed += tr_dw * aTree.a;' : '')
      .replace('vec3 sphN = normalize(vec3(position.x, 0.0, position.z) + vec3(0.0, 0.55 * length(position.xz) + 0.05, 0.0));',
        'vec3 sphN = normalize(vec3(0.0, 0.75, 0.0) + vec3(0.0, 0.0, position.z) * 0.6);');
    const patch = (m, key, weight) => {
      const ob = m.onBeforeCompile;
      m.onBeforeCompile = (sh, r) => { ob?.call(m, sh, r); sh.vertexShader = windPatch(sh.vertexShader, weight); };
      m.customProgramCacheKey = () => key;
      return m;
    };
    const fs = src('foliage');
    // backFix: sprays pressed flat on the road are seen from either side; never flip their (up-bent) normal
    const foliage = patch(createFoliageMaterial({ map: fs?.map, normalMap: fs?.normalMap, backFix: true }), 'fallen-foliage-v3', true);
    const bs = src('bark'), gs = src('bark_grey');
    const bark = patch(createBarkMaterial({ map: bs?.map, normalMap: bs?.normalMap, arm: bs?.roughnessMap, moss: 0 }), 'fallen-bark-v2', false);
    const barkGrey = patch(createBarkMaterial({ map: gs?.map, normalMap: gs?.normalMap, arm: gs?.roughnessMap, moss: 0 }), 'fallen-barkg-v2', false);
    const foliageDepth = patch(createTreeDepthMaterial(fs?.map), 'fallen-depth-f-v2', true);
    // 'fallen': unique atlas. Albedo is baked in the rain-soaked state; roughness (G) and AO (R) from the ARM map;
    // aTree.r carries baked contact occlusion (underside of the stem, inside the root plate, kerf bottom).
    const fsrc = src('fallen');
    const reveal = { value: 0 };
    const mkFallen = (chips) => {
      const m = new THREE.MeshStandardMaterial({
        map: fsrc?.map || null, normalMap: fsrc?.normalMap || null,
        roughnessMap: fsrc?.roughnessMap || null, aoMap: fsrc?.aoMap || null, aoMapIntensity: 1.0,
        roughness: 1.0, metalness: 0.0, color: fsrc?.map ? 0xffffff : 0x4a3c30,
      });
      // bark relief: the baked tangent normals are from a 1.7 m scan filtered to ~2 mm/texel; x1.5 restores the depth
      // of the scale edges that the filtering flattened
      if (fsrc?.normalScale) m.normalScale.copy(fsrc.normalScale).multiplyScalar(1.5);
      m.onBeforeCompile = (sh) => {
        if (chips) sh.uniforms.uReveal = reveal;
        sh.vertexShader = sh.vertexShader
          .replace('#include <common>', '#include <common>\nattribute vec4 aTree;\nvarying vec2 vFAo;')
          .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vFAo = aTree.rg;');
        sh.fragmentShader = sh.fragmentShader
          .replace('#include <common>', '#include <common>\nvarying vec2 vFAo;' + (chips ? '\nuniform float uReveal;' : ''))
          .replace('#include <map_fragment>', (chips ? '  if (vFAo.y > uReveal) discard;\n' : '') + '#include <map_fragment>\n  diffuseColor.rgb *= vFAo.x;');
      };
      m.customProgramCacheKey = () => (chips ? 'fallen-chips-v1' : 'fallen-v2');
      return m;
    };
    this._fmats = { foliage, bark, bark_grey: barkGrey, fallen: mkFallen(false), chips: mkFallen(true), foliageDepth, reveal };
    return this._fmats;
  }

  /**
   * (QA) The notch the hatchet opens in the trunk: a patch wrapped on the bark over the cut (the trunk mesh itself is
   * not carved). Fresh spruce sapwood is pale cream (linear albedo ~0.45/0.33/0.17) against the dark wet bark; the
   * kerf is a dark line across the grain with fibres torn out along it. It widens with every blow.
   */
  _chopScar(ft, root, info) {
    const R = info.cutRadius ?? 0.19;
    const r = R + 0.007;
    const n = Math.min(ft._hits, 3);
    // a V notch runs across the grain: ~2x as wide across the trunk as it is long along it (after 2 blows: ~10 cm
    // along the trunk, ~30 cm across the top)
    const len = 0.04 + 0.03 * n;                         // along the trunk (m)
    const arc = Math.min(1.9, (0.12 + 0.09 * n) / r);    // around it (rad)
    if (!ft._scarMat) {
      const S = 128, cv = document.createElement('canvas'); cv.width = cv.height = S;
      const g = cv.getContext('2d');
      const rnd = mulberry32(305);
      const img = g.createImageData(S, S), d = img.data;
      const edge = Array.from({ length: 64 }, () => 0.82 + 0.18 * rnd());
      const fib = Array.from({ length: S }, () => 0.85 + 0.3 * rnd());
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
        const u = (x + 0.5) / S * 2 - 1, v = (y + 0.5) / S * 2 - 1;     // u around the trunk, v along it
        const k = Math.min(63, Math.floor((u * 0.5 + 0.5) * 64));
        // lens outline: pointed where the V runs out at both sides of the trunk, ragged (torn fibres)
        const w = 0.95 * Math.pow(Math.max(0, 1 - u * u), 0.7) * edge[k];
        const rr = Math.abs(v) / Math.max(w, 1e-3);
        const i = (y * S + x) * 4;
        if (rr > 1) { d[i + 3] = 0; continue; }
        // two faces of the V: the one tilted to the sky lit, the other in its own shadow; fibres along the trunk
        let l = (v < 0 ? 1.0 : 0.6) * (0.8 + 0.2 * rr) * fib[x];
        let cr = 176 * l, cg = 150 * l, cb = 106 * l;
        if (rr < 0.12) { cr = 46; cg = 34; cb = 24; }                            // the bottom of the kerf
        if (rr > 0.84) { const t = (rr - 0.84) / 0.16; cr = cr * (1 - t) + 62 * t; cg = cg * (1 - t) + 44 * t; cb = cb * (1 - t) + 30 * t; } // torn bark lip
        d[i] = cr; d[i + 1] = cg; d[i + 2] = cb; d[i + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      const tex = new THREE.CanvasTexture(cv);
      tex.colorSpace = THREE.SRGBColorSpace;
      ft._scarMat = new THREE.MeshStandardMaterial({ map: tex, bumpMap: tex, bumpScale: 1.5, alphaTest: 0.5, roughness: 0.78, metalness: 0,
        polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
    }
    // open cylinder segment around local +X (the trunk axis), centred on top of the trunk
    const geo = new THREE.CylinderGeometry(r, r, len, 12, 1, true, Math.PI / 2 - arc / 2, arc);
    geo.rotateZ(Math.PI / 2);
    if (ft._scar) { ft._scar.geometry.dispose(); ft._scar.geometry = geo; return; }
    const m = new THREE.Mesh(geo, ft._scarMat);
    m.name = 'fallen_tree_notch';
    m.position.set(0, (info.topAtCut ?? 0.53) - R, 0);
    m.castShadow = false; m.receiveShadow = true;
    root.add(m);
    ft._scar = m;
  }

  /** Collider descriptors (in the tree's local frame). part: null (all), 'A' (x<0), 'B' (x>0). */
  _fallenColliderDescs(info, part) {
    const R = this.ctx.physics.RAPIER;
    const out = [];
    const ax = info.axis;
    const inPart = (x) => part === null || (part === 'A' ? x <= 0.02 : x >= -0.02);
    // split the axis polyline at x = 0 so the halves meet exactly at the cut
    const pts = [];
    for (let i = 0; i < ax.length; i++) {
      pts.push(ax[i]);
      if (i + 1 < ax.length && ax[i][0] < 0 && ax[i + 1][0] > 0) {
        const f = -ax[i][0] / (ax[i + 1][0] - ax[i][0]);
        pts.push(ax[i].map((v, k) => v + (ax[i + 1][k] - v) * f));
      }
    }
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1];
      const mx = (a[0] + b[0]) / 2;
      if (!inPart(mx)) continue;
      const A = new THREE.Vector3(a[0], a[1], a[2]), B = new THREE.Vector3(b[0], b[1], b[2]);
      const r = Math.max(0.05, (a[3] + b[3]) / 2);
      const len = A.distanceTo(B);
      const c = A.clone().add(B).multiplyScalar(0.5);
      const q = new THREE.Quaternion().setFromUnitVectors(Y, B.clone().sub(A).normalize());
      const hh = Math.max(0.01, len / 2 - (part ? r * 0.2 : 0));
      out.push({ desc: R.ColliderDesc.capsule(hh, r).setTranslation(c.x, c.y, c.z).setRotation({ x: q.x, y: q.y, z: q.z, w: q.w })
        .setFriction(0.9).setRestitution(0.05).setDensity(650) });
    }
    if (part !== 'B' && info.rootPlate) {
      const rp = info.rootPlate;
      // kept clear of the ground/cut face at spawn (interpenetration would push the body through the road trimesh)
      const hy = Math.max(0.3, Math.min(rp.radius * 0.9, rp.center[1] - 0.06));
      out.push({ desc: R.ColliderDesc.cuboid(rp.halfThickness ?? 0.25, hy, rp.radius * 0.7)
        .setTranslation(rp.center[0], rp.center[1], rp.center[2]).setFriction(1.0).setDensity(900) });
    }
    if (part !== 'A' && info.crown) {
      // dense crown mass (branches) beyond the road edge: a few boxes, heavier than the trunk so the half tips outward
      const c = info.crown;
      const segs = 3;
      for (let k = 0; k < segs; k++) {
        const x0 = Math.max(c.x0, 4.2) + (c.x1 - Math.max(c.x0, 4.2)) * k / segs;
        const x1 = Math.max(c.x0, 4.2) + (c.x1 - Math.max(c.x0, 4.2)) * (k + 1) / segs;
        const rad = c.radius * (1 - 0.25 * k);
        const cy = info._crownY ? info._crownY[k] : 0.3 + rad * 0.25;
        const hy = info._crownY ? rad * 0.3 : rad * 0.35;
        out.push({ desc: R.ColliderDesc.cuboid((x1 - x0) / 2, hy, rad * 0.6)
          .setTranslation((x0 + x1) / 2, cy, 0).setFriction(0.8).setDensity(part ? 160 : 60) });
      }
      // a couple of stiff branch stubs under the road part (so the half does not roll like a perfect cylinder)
      for (const x of [1.4, 3.0]) {
        out.push({ desc: R.ColliderDesc.cuboid(0.06, 0.12, 0.7).setTranslation(x, 0.12, 0).setFriction(1.0).setDensity(300) });
      }
    }
    return out;
  }

  async _split(ft) {
    const { ctx } = this;
    const phys = ctx.physics, R = phys?.RAPIER;
    const root = ft.object;
    ft.whole.visible = false;
    if (ft.halves.a) ft.halves.a.visible = true;
    if (ft.halves.b) ft.halves.b.visible = true;
    try { ctx.audio?.play?.('crack', { position: ft.chopPoint }); } catch {}
    try { ctx.particles?.debris?.(ft.chopPoint.clone(), 18, { wood: true, speed: 3.5, up: 0.9 }); ctx.particles?.dust?.(ft.chopPoint.clone(), 1.2, 0.4); } catch {}
    if (!R || !phys.world) {
      // no physics: animate the valley half away
      return this._animateAway(ft);
    }
    if (ft.body) { phys.world.removeRigidBody(ft.body); ft.body = null; }
    const road = ctx.road;
    const mkBody = (obj, part) => {
      if (!obj) return null;
      // re-parent the half to the vegetation group with the root's world transform (physics drives it from now on)
      root.updateMatrixWorld(true);
      obj.matrixWorld.decompose(_v, _q, _v2);
      this.group.add(obj);
      obj.position.copy(_v); obj.quaternion.copy(_q); obj.scale.set(1, 1, 1);
      const body = phys.world.createRigidBody(R.RigidBodyDesc.dynamic()
        .setTranslation(_v.x, _v.y, _v.z).setRotation({ x: _q.x, y: _q.y, z: _q.z, w: _q.w })
        .setLinearDamping(0.15).setAngularDamping(0.35).setCcdEnabled(true));
      for (const cd of this._fallenColliderDescs(this.fallenInfo, part)) {
        // the halves do not collide with each other (they start in contact at the cut)
        phys.world.createCollider(cd.desc.setCollisionGroups(groups(G.DEBRIS, G.STATIC | G.CAR | G.PLAYER | G.ROCK)), body);
      }
      phys.link(body, obj);
      return body;
    };
    const bodyA = mkBody(ft.halves.a, 'A');
    const bodyB = mkBody(ft.halves.b, 'B');
    const dirX = new THREE.Vector3(1, 0, 0).applyQuaternion(root.quaternion); // toward the valley
    const massB = bodyB?.mass?.() ?? 800, massA = bodyA?.mass?.() ?? 1500;
    if (bodyB) {
      bodyB.applyImpulse({ x: dirX.x * massB * 3.4, y: massB * 0.5, z: dirX.z * massB * 3.4 }, true);
      const ax = new THREE.Vector3().crossVectors(Y, dirX).normalize(); // tip the crown down over the edge
      bodyB.applyTorqueImpulse({ x: ax.x * massB * 0.8, y: 0, z: ax.z * massB * 0.8 }, true);
    }
    if (bodyA) {
      // swing the cut end uphill about the heavy root plate so the whole lane clears
      // the heavy root half just settles; if its cut end ends up on the valley lane, _updateDynamics shoves it
    }
    const dyn = { bodyA, bodyB, objA: ft.halves.a, objB: ft.halves.b, t: 0, dirX, roadY: root.position.y, cleared: false };
    this._dyn.push(dyn);
    ft.dyn = dyn;
    return new Promise((resolve) => { dyn.resolve = resolve; });
  }

  _laneBlocked(body, obj) {
    // true if any part of the half is still on the valley lane (car footprint d in [-2.9, -0.15]) at road height
    const road = this.ctx.road;
    if (!obj || !road) return false;
    const info = this.fallenInfo;
    const pr = {};
    for (const a of info.axis) {
      _v.set(a[0], a[1], a[2]);
      obj.localToWorld(_v);
      road.project(_v, pr);
      if (pr.d > -2.9 && pr.d < -0.15 && pr.dy > -0.8 && pr.dy < 4.5 && Math.abs(pr.s - (road.markers?.fallenTree ?? 305)) < 14) return true;
    }
    return false;
  }

  _updateDynamics(dt) {
    const phys = this.ctx.physics;
    for (let i = this._dyn.length - 1; i >= 0; i--) {
      const D = this._dyn[i];
      D.t += dt;
      const blockedB = D.bodyB && this._laneBlocked(D.bodyB, D.objB);
      const blockedA = D.bodyA && this._laneBlocked(D.bodyA, D.objA);
      // assisting shove if a half is still sitting on the lane after a while (keeps the level solvable)
      if (D.t > 1.5 && D.t < 12) {
        const left = this.ctx.road.leftAt(this.ctx.road.markers?.fallenTree ?? 305, _v);
        if (blockedB) {
          const m = D.bodyB.mass();
          D.bodyB.applyImpulse({ x: D.dirX.x * m * 7 * dt, y: 0, z: D.dirX.z * m * 7 * dt }, true);
        }
        if (blockedA && D.objA) {
          // push the cut end uphill: the half pivots about its heavy root plate
          const m = D.bodyA.mass();
          const cut = D.objA.localToWorld(_v2.set(0, 0.3, 0));
          D.bodyA.applyImpulseAtPoint({ x: left.x * m * 3 * dt, y: m * 0.8 * dt, z: left.z * m * 3 * dt }, { x: cut.x, y: cut.y, z: cut.z }, true);
        }
      }
      D.clearFor = (!blockedA && !blockedB) ? (D.clearFor || 0) + dt : 0;
      if (!D.cleared && ((D.clearFor > 0.5 && D.t > 1.0) || D.t > 9)) {
        D.cleared = true;
        D.resolve?.();
      }
      // remove halves that fell far down the valley; freeze the ones resting near the road into fixed bodies
      for (const k of ['A', 'B']) {
        const body = D['body' + k], obj = D['obj' + k];
        if (!body) continue;
        const t = body.translation();
        if (t.y < D.roadY - 60) {
          phys.unlink(body);
          phys.world.removeRigidBody(body);
          D['body' + k] = null;
          if (obj) obj.visible = false;
        } else if ((D.t > 6 && body.isSleeping()) || D.t > 25) {
          try { body.setBodyType(phys.RAPIER.RigidBodyType.Fixed, false); } catch {}
          phys.syncLinks?.();
          phys.unlink(body);
          D['body' + k] = null;   // keeps its colliders as static geometry
        }
      }
      if (!D.bodyA && !D.bodyB) this._dyn.splice(i, 1);
    }
  }

  _animateAway(ft) {
    // physics-less fallback: tip the valley half over the edge in ~2 s
    return new Promise((resolve) => {
      const b = ft.halves.b;
      if (!b) { resolve(); return; }
      const start = performance.now();
      const step = () => {
        const t = Math.min(1, (performance.now() - start) / 2000);
        b.position.set(t * 6, -t * t * 12, 0);
        b.rotation.set(t * 1.2, 0, -t * 0.8);
        if (t < 1) requestAnimationFrame(step); else { b.visible = false; resolve(); }
      };
      step();
    });
  }

  // ------------------------------------------------------------------------------------------------------------
  update(dt) {
    const { ctx } = this;
    this._time += dt;
    WIND.uTime.value = this._time;
    const env = ctx.env;
    const rain = env?.rain ?? 0.35;
    const w = WIND.uWind.value;
    // gusty wind: base strength follows the rain intensity a bit
    const gust = Math.max(0, Math.sin(this._time * 0.13) * 0.6 + Math.sin(this._time * 0.37 + 1.3) * 0.4);
    w.z = 0.3 + 0.35 * rain;
    w.w = gust * 0.8;
    WIND.uWet.value = env?.wetness ?? 0.75;
    if (ctx.camera) WIND.uCamPos.value.copy(ctx.camera.position);
    if (this._scatterPending && this.gltf && ctx.time?.frame > 2) {
      this._scatterPending = false;
      try { this._buildScatter(this.gltf); } catch (e) { console.warn('[vegetation] scatter failed', e); }
    }
    this.trees?.update(dt);
    this.grass?.update(dt);
    this._updateScatter();
    const ft = this.fallenTree;
    if (ft && !ft.cut && ft._shake > 0 && ft.whole) {
      ft._shake = Math.max(0, ft._shake - dt * 3);
      const a = ft._shake * 0.012;
      ft.whole.position.set(0, Math.sin(this._time * 60) * a, Math.sin(this._time * 47) * a);
    }
    if (this._dyn.length) this._updateDynamics(dt);
  }

  dispose() {
    this.trees?.dispose();
    this.grass?.dispose?.();
    this.ctx.scene.remove(this.group);
  }
}
