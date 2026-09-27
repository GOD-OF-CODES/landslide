// Near-camera ground vegetation (TREES workstream): rain-lodged mixed wild grass (2 near species mixes + a far LOD),
// seed-head panicles, a few late-season flowers, ferns, bilberry patches on the forest floor and coltsfoot /
// butterbur leaves in the wet uphill ditch.
//
//   const grass = new GrassField(ctx, { foliageMap, foliageNormal });   // foliage atlas from trees.glb (ground flora)
//   grass.update(dt)
//
// Placement: the world is split into 8 m chunks. A chunk is generated lazily (nearest first, a few per frame) by ray
// casting a 1 m grid into the static physics geometry (terrain) and interpolating; density comes from
// ctx.terrain.grassAt (TERRAIN's vegetation mask, 0 on road/rock/scar) and the slope. Instance buffers are rebuilt only
// when the camera enters a new chunk or new chunks finish. Blades shrink into the ground toward grassRadius (no edge).
//
// Real-world calibration (linear albedo, before the shader's rain darkening x0.82):
//   green grass blade tip ~(0.06, 0.09, 0.03), wet canopy reads ~0.06-0.08 in the green channel;
//   late-season straw (senescent leaves, seed stems) ~(0.27, 0.22, 0.12) -> ~0.2 wet; seed heads straw / purple-brown.
//   Blade bases and the soil between tussocks are near the terrain's wet soil albedo (~0.03-0.05), so the verge blends
//   into the ground instead of sitting on it as a green strip. Roughly 40-60 % straw by area on the drained verge.
import * as THREE from 'three';
import { WIND, createFoliageMaterial } from './impostor.js';
import { G, groups } from '../physics/world.js';

/**
 * Vegetation floor for the road corridor where TERRAIN's mask is 0 (it classifies the 35-50 deg natural slopes as rock):
 * spruce, shrubs and grass cling to those slopes in real Alpine road cuts. Keeps the road, ditch, landslide scar,
 * gullies, the washed-out gap, the pull-off and the tunnel spur clear. ny = terrain normal y.
 */
export function corridorVegetation(markers, s, d, ny) {
  const m = markers || {};
  const ad = Math.abs(d);
  if (ad < 5.2 || ad > 70) return 0;
  if (m.scar && s > m.scar[0] - 6 && s < m.scar[1] + 6 && d > 0) return 0;
  if (m.gullies) for (const g of m.gullies) if (Math.abs(s - g) < 13 && d > 0) return 0;
  if (Math.abs(s - (m.gap ?? 560)) < 9) return 0;
  if (s > (m.tunnel ?? 1150) - 30) return 0;
  if (m.pulloff && s > m.pulloff[0] - 2 && s < m.pulloff[1] + 2 && d < 0 && d > -16) return 0;
  const t = Math.min(1, Math.max(0, (ny - 0.66) / (0.84 - 0.66)));
  return t * t * (3 - 2 * t) * 0.7;
}

/** Shared grass tuning (debug: ctx.vegetation.grass.tune). */
export const GRASS_TUNE = { boost: { value: 1.4 } };

const CH = 8;          // chunk size (m)
const GRID = 9;        // ray grid per chunk (GRID x GRID)
const STRIDE = 7;      // grass point: x, y, z, scale, rotY, rank, dryness
const _v = new THREE.Vector3(), _d = new THREE.Vector3(0, -1, 0), _m = new THREE.Matrix4(), _q = new THREE.Quaternion();
const _s = new THREE.Vector3(), _p = new THREE.Vector3(), _c = new THREE.Color(), _q2 = new THREE.Quaternion();
const _p2 = new THREE.Vector3();
const Y = new THREE.Vector3(0, 1, 0);

function rng(seed) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function vnoise(x, z) {
  const h = (a, b) => { const v = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return v - Math.floor(v); };
  const ix = Math.floor(x), iz = Math.floor(z), fx = x - ix, fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx), uz = fz * fz * (3 - 2 * fz);
  const a = h(ix, iz), b = h(ix + 1, iz), c = h(ix, iz + 1), d = h(ix + 1, iz + 1);
  return (a + (b - a) * ux) * (1 - uz) + (c + (d - c) * ux) * uz;
}

// Atlas regions (mirror CARDS in tools/blender/trees.py; glTF uv convention: u = x/2048, v = y/2048, v down).
const A = 2048;
const FERN_UV = [
  { u0: 512 / A, u1: 768 / A, vBase: 1.0, vTip: 1536 / A, aspect: 0.5 },
  { u0: 768 / A, u1: 1024 / A, vBase: 1.0, vTip: 1536 / A, aspect: 0.5 },
];
// Wild-grass card atlas (trees.glb material 'grass', 1024 x 1024; mirrors GRASS_CARDS in tools/blender/trees.py).
const GA = 1024;
const GRASS_UV = {
  g_dry: { u0: 0, u1: 512 / GA, vBase: 512 / GA, vTip: 0, h: 0.72 },
  g_green: { u0: 512 / GA, u1: 1, vBase: 512 / GA, vTip: 0, h: 0.56 },
  g_straw: { u0: 0, u1: 512 / GA, vBase: 1, vTip: 512 / GA, h: 0.8 },
  g_mixed: { u0: 512 / GA, u1: 1, vBase: 1, vTip: 512 / GA, h: 0.62 },
};
const CARD_UV = {
  bilberry: { u0: 1024 / A, u1: 1536 / A, v0: 1536 / A, v1: 2048 / A },   // side view, base at v1
  coltsfoot: { u0: 1536 / A, u1: 1792 / A, v0: 1536 / A, v1: 1792 / A },  // top view, sinus toward v1
  butterbur: { u0: 1792 / A, u1: 2048 / A, v0: 1536 / A, v1: 1792 / A },
  litter: { u0: 1536 / A, u1: 2048 / A, v0: 1792 / A, v1: 2048 / A },     // top view, 1.2 x 0.6 m
};

// ------------------------------------------------------------------------------------------------------------------
// Geometry builders
// ------------------------------------------------------------------------------------------------------------------
// linear albedo palettes: [base, tip]
const PAL = {
  green: [[0.022, 0.03, 0.013], [0.058, 0.09, 0.03]],
  lush: [[0.02, 0.03, 0.012], [0.05, 0.082, 0.026]],
  yellow: [[0.03, 0.033, 0.014], [0.12, 0.122, 0.045]],
  straw: [[0.05, 0.042, 0.027], [0.27, 0.215, 0.12]],
  brown: [[0.035, 0.027, 0.018], [0.14, 0.09, 0.052]],
};

/**
 * Wild-grass clump. opts: {blades, segs, widthMul, straw (fraction of straw/brown blades), stems (seed stems),
 * lodge (how far rain has flattened the blades 0..1), tuft (tight low tuft)}.
 * Blades arch out of a tight tussock and flop over (rain-lodged); a few long seed stems carry drooping panicles.
 * Vertex colour = albedo x height AO (dark, soil-coloured bases).
 */
function buildClump(seed, o) {
  const r = rng(seed);
  const pos = [], nor = [], col = [], idx = [];
  const segs = o.segs;
  const pushStrip = (pts, widths, face, cbase, ctip, aoBase) => {
    const start = pos.length / 3;
    for (let i = 0; i < pts.length; i++) {
      const t = i / (pts.length - 1);
      const [px, py, pz] = pts[i];
      const f = face[i];
      const w = widths[i];
      pos.push(px - f[0] * w, py - f[1] * w, pz - f[2] * w, px + f[0] * w, py + f[1] * w, pz + f[2] * w);
      // blade normal: mostly up, a bit of the blade face (soft, so thin blades do not flicker)
      const tx = i ? px - pts[i - 1][0] : pts[1][0] - px, ty = i ? py - pts[i - 1][1] : pts[1][1] - py, tz = i ? pz - pts[i - 1][2] : pts[1][2] - pz;
      let nx = ty * f[2] - tz * f[1], ny = tz * f[0] - tx * f[2], nz = tx * f[1] - ty * f[0];
      if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx = nx / nl * 0.45; ny = ny / nl * 0.45 + 0.8; nz = nz / nl * 0.45;
      for (let s = 0; s < 2; s++) nor.push(nx, ny, nz);
      const g = Math.pow(t, 0.7);
      const ao = aoBase + (1 - aoBase) * Math.pow(t, 0.55);
      for (let s = 0; s < 2; s++) col.push((cbase[0] + (ctip[0] - cbase[0]) * g) * ao, (cbase[1] + (ctip[1] - cbase[1]) * g) * ao, (cbase[2] + (ctip[2] - cbase[2]) * g) * ao);
    }
    for (let i = 0; i < pts.length - 1; i++) {
      const a0 = start + i * 2;
      idx.push(a0, a0 + 1, a0 + 3, a0, a0 + 3, a0 + 2);
    }
  };
  const arch = (bx, bz, dirA, L, th0, bend, n) => {
    // polyline rising at angle th0 from vertical and curving over by `bend` radians along its length
    const dx = Math.cos(dirA), dz = Math.sin(dirA);
    const pts = [[bx, 0, bz]];
    let x = bx, y = 0, z = bz;
    for (let i = 1; i <= n; i++) {
      const t = i / n;
      const th = th0 + bend * t * t;
      const st = L / n;
      x += Math.sin(th) * dx * st; z += Math.sin(th) * dz * st; y += Math.cos(th) * st;
      pts.push([x, Math.max(y, 0.005), z]);
    }
    return pts;
  };
  for (let b = 0; b < o.blades; b++) {
    const a = r() * Math.PI * 2;
    const off = Math.sqrt(r()) * (o.tuft ? 0.05 : 0.09);
    const bx = Math.cos(a) * off, bz = Math.sin(a) * off;
    const L = o.tuft ? 0.1 + r() * 0.2 : 0.2 + r() * 0.42;
    const dirA = a + (r() - 0.5) * 1.2;
    const th0 = 0.12 + r() * 0.45 + off * 2;
    // long wet blades flop over (lodged by rain): up to ~110 deg of total bend
    const bend = (0.5 + r() * 1.3) * (0.4 + L * 1.6) * (0.6 + o.lodge * 0.8);
    const pts = arch(bx, bz, dirA, L, th0, bend, segs);
    const w = (0.005 + r() * 0.0065) * o.widthMul;
    const twist = (r() - 0.5) * 1.4;
    const widths = [], face = [];
    for (let i = 0; i <= segs; i++) {
      const t = i / segs;
      widths.push(w * (1 - t * 0.88));
      const fa = dirA + Math.PI / 2 + twist * t;
      face.push([Math.cos(fa), 0.15 * Math.sin(twist * t), Math.sin(fa)]);
    }
    const k = r();
    let p;
    if (k < o.straw * 0.8) p = PAL.straw;
    else if (k < o.straw) p = PAL.brown;
    else if (k < o.straw + 0.12) p = PAL.yellow;
    else p = r() < 0.5 ? PAL.green : PAL.lush;
    const j = 0.8 + r() * 0.4;
    pushStrip(pts, widths, face, p[0].map((v) => v * j), p[1].map((v) => v * j), 0.4);
  }
  // seed stems with panicles (Deschampsia / Agrostis / Dactylis-like), arched over by the rain
  for (let sI = 0; sI < o.stems; sI++) {
    const a = r() * Math.PI * 2;
    const bx = Math.cos(a) * 0.03 * r(), bz = Math.sin(a) * 0.03 * r();
    const L = 0.42 + r() * 0.5;
    const dirA = r() * Math.PI * 2;
    const n = Math.max(3, segs + 1);
    const pts = arch(bx, bz, dirA, L, 0.08 + r() * 0.3, (0.5 + r() * 1.1) * (0.6 + o.lodge * 0.7), n);
    const widths = [], face = [];
    const fa = dirA + Math.PI / 2;
    for (let i = 0; i <= n; i++) { widths.push(0.0022 * o.widthMul * (1 - 0.5 * i / n)); face.push([Math.cos(fa), 0, Math.sin(fa)]); }
    const sc = r() < 0.6 ? [[0.06, 0.06, 0.03], [0.2, 0.17, 0.095]] : [[0.04, 0.05, 0.022], [0.1, 0.11, 0.045]];
    pushStrip(pts, widths, face, sc[0], sc[1], 0.55);
    // panicle: spikelet quads hanging from the last third of the stem
    const head = r() < 0.55 ? [0.25, 0.2, 0.115] : [0.13, 0.085, 0.085];   // straw or purple-brown
    const ns = o.far ? 2 : 6 + Math.floor(r() * 4);
    for (let q = 0; q < ns; q++) {
      const f = 0.62 + 0.38 * (q + r() * 0.8) / ns;
      const fi = f * n, i0 = Math.min(n - 1, Math.floor(fi)), ft = fi - i0;
      const P0 = pts[i0], P1 = pts[i0 + 1];
      const px = P0[0] + (P1[0] - P0[0]) * ft, py = P0[1] + (P1[1] - P0[1]) * ft, pz = P0[2] + (P1[2] - P0[2]) * ft;
      const ha = r() * Math.PI * 2;
      const len = (o.far ? 0.05 : 0.022 + r() * 0.03) * (o.far ? 1.6 : 1);
      const ex = px + Math.cos(ha) * len * 0.6, ey = py - len * (0.5 + r() * 0.5), ez = pz + Math.sin(ha) * len * 0.6;
      const ww = (o.far ? 0.012 : 0.0055) * o.widthMul * 0.5;
      const fx = -Math.sin(ha), fz = Math.cos(ha);
      const s0 = pos.length / 3;
      pos.push(px - fx * ww, py, pz - fz * ww, px + fx * ww, py, pz + fz * ww, ex, ey, ez);
      for (let s = 0; s < 3; s++) { nor.push(0, 1, 0); }
      const j = 0.8 + r() * 0.4;
      for (let s = 0; s < 3; s++) col.push(head[0] * j, head[1] * j, head[2] * j);
      idx.push(s0, s0 + 1, s0 + 2);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

/**
 * Crossed-card grass clump: n vertical quads (segs rows each) around the clump centre, each showing one wild-grass card
 * (a 0.5-0.8 m strip of verge rendered from 3D blades). The quads lean (rain-lodged) and sink 3 cm into the ground so
 * the card's bottom edge never shows. aGr: x = height fraction (wind, AO), y = card random.
 */
function buildCardClump(seed, cards, n, segs, widthMul = 1) {
  const r = rng(seed);
  const pos = [], nor = [], uv = [], gr = [], idx = [];
  for (let k = 0; k < n; k++) {
    const U = GRASS_UV[cards[k % cards.length]];
    const a = k / n * Math.PI + (r() - 0.5) * 0.5;
    const h = U.h * (0.85 + r() * 0.3);
    const w = U.h * widthMul * (0.85 + r() * 0.3);   // square cards: width = card height
    const fx = Math.cos(a), fz = Math.sin(a);         // quad plane direction
    const nx = -fz, nz = fx;                           // quad normal (horizontal)
    const lean = (r() - 0.5) * 0.5;                    // top offset along the normal (x h)
    const ox = (r() - 0.5) * 0.2, oz = (r() - 0.5) * 0.2;
    const flip = r() < 0.5;
    const s0 = pos.length / 3;
    for (let i = 0; i <= segs; i++) {
      const t = i / segs;
      const y = -0.03 + (h + 0.03) * t;
      const off = lean * h * t * t;
      for (const sg of [-1, 1]) {
        pos.push(ox + fx * sg * w / 2 + nx * off, y, oz + fz * sg * w / 2 + nz * off);
        // soft, mostly-up normals: both faces of a card shade alike and the clump reads as a volume
        nor.push(nx * 0.35, 0.85, nz * 0.35);
        uv.push((sg < 0) !== flip ? U.u0 : U.u1, U.vBase + (U.vTip - U.vBase) * t);
        gr.push(t, r());
      }
    }
    for (let i = 0; i < segs; i++) { const q = s0 + i * 2; idx.push(q, q + 1, q + 3, q, q + 3, q + 2); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('aGr', new THREE.Float32BufferAttribute(gr, 2));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

/** Late-season flowers: few; thin stems with small heads (yarrow umbels, hawkweed, harebell, knapweed). */
function buildFlowers(seed) {
  const r = rng(seed);
  const pos = [], nor = [], col = [], idx = [];
  const palette = [[0.5, 0.49, 0.44], [0.45, 0.33, 0.04], [0.2, 0.12, 0.28], [0.34, 0.16, 0.2]];
  const stems = 2 + Math.floor(r() * 3);
  const pc = palette[Math.floor(r() * palette.length)];
  for (let b = 0; b < stems; b++) {
    const a = r() * Math.PI * 2, off = r() * 0.12;
    const bx = Math.cos(a) * off, bz = Math.sin(a) * off;
    const h = 0.22 + r() * 0.3;
    const lx = (r() - 0.5) * 0.14, lz = (r() - 0.5) * 0.14;
    const w = 0.0025;
    const s0 = pos.length / 3;
    for (let i = 0; i <= 2; i++) {
      const t = i / 2;
      pos.push(bx + lx * t * t - w, h * t * (1 - 0.15 * t), bz + lz * t * t, bx + lx * t * t + w, h * t * (1 - 0.15 * t), bz + lz * t * t);
      nor.push(0, 1, 0, 0, 1, 0);
      col.push(0.04, 0.06, 0.02, 0.04, 0.06, 0.02);
    }
    for (let i = 0; i < 2; i++) { const q = s0 + i * 2; idx.push(q, q + 1, q + 3, q, q + 3, q + 2); }
    const cx = bx + lx, cz = bz + lz, cy = h * 0.85;
    const hr = 0.01 + r() * 0.014;
    const c0 = pos.length / 3;
    pos.push(cx, cy + 0.004, cz); nor.push(0, 1, 0); col.push(pc[0] * 0.7, pc[1] * 0.7, pc[2] * 0.6);
    for (let k = 0; k < 6; k++) {
      const an = k / 6 * Math.PI * 2;
      pos.push(cx + Math.cos(an) * hr, cy - hr * 0.3, cz + Math.sin(an) * hr);
      nor.push(Math.cos(an) * 0.3, 1, Math.sin(an) * 0.3);
      const j = 0.8 + r() * 0.3;
      col.push(pc[0] * j, pc[1] * j, pc[2] * j);
    }
    for (let k = 0; k < 6; k++) idx.push(c0, c0 + 1 + ((k + 1) % 6), c0 + 1 + k);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

/** Fern clump: 6-9 arching frond strips textured with the fern cards of the foliage atlas. */
function buildFern(seed) {
  const r = rng(seed);
  const pos = [], nor = [], uv = [], tree = [], idx = [];
  const n = 6 + Math.floor(r() * 4);
  const segs = 4;
  for (let f = 0; f < n; f++) {
    const U = FERN_UV[f % 2];
    const a = f / n * Math.PI * 2 + (r() - 0.5) * 0.6;
    const L = 0.45 + r() * 0.35;
    const W = L * U.aspect;
    const rise = 0.8 + r() * 0.35;   // initial elevation angle
    const dx = Math.cos(a), dz = Math.sin(a);
    const px = -dz, pz = dx;          // frond width direction (horizontal)
    const roll = (r() - 0.5) * 0.5;
    let x = dx * 0.03, y = 0.0, z = dz * 0.03, el = rise;
    const s0 = pos.length / 3;
    for (let i = 0; i <= segs; i++) {
      const t = i / segs;
      const cw = Math.cos(roll), sw = Math.sin(roll);
      const wx = px * cw, wy = sw, wz = pz * cw;
      pos.push(x - wx * W / 2, y - wy * W / 2, z - wz * W / 2, x + wx * W / 2, y + wy * W / 2, z + wz * W / 2);
      const nx = -Math.sin(el) * dx * 0.6, nz = -Math.sin(el) * dz * 0.6;
      nor.push(nx, 1, nz, nx, 1, nz);
      const v = U.vBase + (U.vTip - U.vBase) * t;
      uv.push(U.u0, v, U.u1, v);
      const ao = 0.45 + 0.55 * t;
      tree.push(ao, f / n, r(), 1, ao, f / n, r(), 1);
      const step = L / segs;
      x += Math.cos(el) * dx * step; z += Math.cos(el) * dz * step; y += Math.sin(el) * step;
      el -= 0.46 + r() * 0.14;   // wet fronds hang lower
    }
    for (let i = 0; i < segs; i++) { const q = s0 + i * 2; idx.push(q, q + 1, q + 3, q, q + 3, q + 2); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('aTree', new THREE.Float32BufferAttribute(tree, 4));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

/**
 * Broad-leaved ditch forb clump built from the top-view leaf cards: coltsfoot (Tussilago farfara, 10-22 cm polygonal
 * heart-shaped leaves on short stalks: THE pioneer of wet road-cut clay and landslide mud) or butterbur (Petasites,
 * 30-60 cm kidney leaves on 0.3-0.6 m stalks, wet ditches and seeps). Each leaf is a slightly cupped/drooping 3x3 grid
 * on a stalk strip; the stalk samples the leaf's green centre.
 */
function buildForb(seed, kind) {
  const r = rng(seed);
  const U = CARD_UV[kind];
  const big = kind === 'butterbur';
  const pos = [], nor = [], uv = [], tree = [], idx = [];
  const n = big ? 3 + Math.floor(r() * 3) : 4 + Math.floor(r() * 4);
  const cu = (U.u0 + U.u1) / 2, cv = (U.v0 + U.v1) / 2;
  for (let k = 0; k < n; k++) {
    const a = k / n * Math.PI * 2 + (r() - 0.5) * 0.9;
    const dx = Math.cos(a), dz = Math.sin(a);
    const size = (big ? 0.48 + r() * 0.3 : 0.16 + r() * 0.1);          // card side (m)
    const stalk = big ? 0.28 + r() * 0.35 : 0.04 + r() * 0.1;
    const reach = big ? 0.12 + r() * 0.25 : 0.05 + r() * 0.12;
    const tx = dx * reach, tz = dz * reach, ty = stalk * (big ? 1 : 0.8);
    // stalk strip
    const s0 = pos.length / 3;
    const sw = big ? 0.01 : 0.004;
    const px = -dz * sw, pz = dx * sw;
    for (let i = 0; i <= 2; i++) {
      const t = i / 2;
      const x = tx * t * t, y = ty * Math.sin(t * Math.PI / 2), z = tz * t * t;
      pos.push(x - px, y, z - pz, x + px, y, z + pz);
      nor.push(dx * 0.3, 1, dz * 0.3, dx * 0.3, 1, dz * 0.3);
      uv.push(cu, cv, cu + 0.002, cv);
      tree.push(0.5 + 0.4 * t, k / n, r(), 1, 0.5 + 0.4 * t, k / n, r(), 1);
    }
    idx.push(s0, s0 + 1, s0 + 3, s0, s0 + 3, s0 + 2, s0 + 2, s0 + 3, s0 + 5, s0 + 2, s0 + 5, s0 + 4);
    // leaf grid: card -v axis (sinus side) toward the clump centre, card +u to the right; tilt up/down, droop at the rim
    const tilt = big ? (r() - 0.35) * 0.7 : 0.15 + r() * 0.5;   // rad, + = outer edge raised
    const l0 = pos.length / 3;
    const G = big ? 4 : 3;
    for (let j = 0; j <= G; j++) {
      for (let i = 0; i <= G; i++) {
        const fu = i / G - 0.5, fv = j / G - 0.5;               // fv: -0.5 at the sinus (inner) edge
        const along = (fv + 0.5 - 0.35) * size;                  // attachment ~35 % in from the inner edge
        const side = fu * size;
        const rim = Math.min(1, Math.hypot(fu, fv) * 2);
        // cupped centre, drooping + undulating rim (butterbur leaves sag under their own weight when wet)
        const h = Math.sin(tilt) * along - (big ? 0.16 : 0.04) * size * rim * rim + 0.06 * size * (1 - rim)
          + (big ? 0.025 : 0.01) * size * Math.sin(Math.atan2(fv, fu) * 5 + k) * rim;
        const x = tx + dx * along * Math.cos(tilt) - dz * side;
        const z = tz + dz * along * Math.cos(tilt) + dx * side;
        pos.push(x, ty + h, z);
        const nx = -dx * Math.sin(tilt) - (big ? 0.25 : 0.12) * fu * dz * -1, nz = -dz * Math.sin(tilt) + (big ? 0.25 : 0.12) * fu * dx * -1;
        nor.push(nx * 0.8 + dx * rim * 0.2, 1, nz * 0.8 + dz * rim * 0.2);
        uv.push(U.u0 + (fu + 0.5) * (U.u1 - U.u0), U.v1 - (fv + 0.5) * (U.v1 - U.v0));
        tree.push(0.65 + 0.35 * rim, k / n, r(), 1);
      }
    }
    for (let j = 0; j < G; j++) {
      for (let i = 0; i < G; i++) {
        const a0 = l0 + j * (G + 1) + i;
        idx.push(a0, a0 + 1, a0 + G + 2, a0, a0 + G + 2, a0 + G + 1);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('aTree', new THREE.Float32BufferAttribute(tree, 4));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

/** Forest-floor litter decal: one quad (1.2 x 0.6 m) showing spruce needle litter, twigs, a cone and feather-moss
 *  cushions with a ragged alpha outline; laid on the slope under the crowns near the road. */
function buildLitterQuad() {
  const U = CARD_UV.litter;
  const w = 1.2, h = 0.6, y = 0.02;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([-w / 2, y, -h / 2, w / 2, y, -h / 2, w / 2, y, h / 2, -w / 2, y, h / 2], 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute([U.u0, U.v1, U.u1, U.v1, U.u1, U.v0, U.u0, U.v0], 2));
  g.setAttribute('aTree', new THREE.Float32BufferAttribute([0.85, 0, 0, 1, 0.85, 0, 0, 1, 0.85, 0, 0, 1, 0.85, 0, 0, 1], 4));
  g.setIndex([0, 2, 1, 0, 3, 2]);
  g.computeBoundingSphere();
  return g;
}

/** Bilberry (Vaccinium myrtillus) clump: 3 crossed, slightly bent side-view sprig cards, 20-40 cm. */
function buildBilberry(seed) {
  const r = rng(seed);
  const U = CARD_UV.bilberry;
  const pos = [], nor = [], uv = [], tree = [], idx = [];
  const n = 3;
  for (let k = 0; k < n; k++) {
    const a = k / n * Math.PI + (r() - 0.5) * 0.4;
    const h = 0.26 + r() * 0.12, w = h * 1.0;   // card aspect 1:1 (512 x 512)
    const fx = Math.cos(a), fz = Math.sin(a);
    const ox = (r() - 0.5) * 0.12, oz = (r() - 0.5) * 0.12;
    const s0 = pos.length / 3;
    for (let i = 0; i <= 2; i++) {
      const t = i / 2;
      const lean = 0.06 * t * t;
      for (const sgn of [-1, 1]) {
        pos.push(ox + fx * sgn * w / 2 + fz * lean, h * t, oz + fz * sgn * w / 2 - fx * lean);
        nor.push(fz * 0.4, 1, -fx * 0.4);
        uv.push(sgn < 0 ? U.u0 : U.u1, U.v1 - t * (U.v1 - U.v0));
        tree.push(0.4 + 0.6 * t, k / n, r(), 1);
      }
    }
    for (let i = 0; i < 2; i++) { const q = s0 + i * 2; idx.push(q, q + 1, q + 3, q, q + 3, q + 2); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('aTree', new THREE.Float32BufferAttribute(tree, 4));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

// ------------------------------------------------------------------------------------------------------------------
// Grass material: vertex-coloured blades (x instance colour), wind bending + gust waves, distance shrink, rain wetness
// ------------------------------------------------------------------------------------------------------------------
function createGrassMaterial(radius) {
  // envMapIntensity < 1: thin blades with up-facing normals otherwise mirror the bright overcast sky at grazing angles
  // and the whole meadow turns silver-grey (grass occludes most of its own sky hemisphere)
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.8, metalness: 0, envMapIntensity: 0.45 });
  const U = {
    uTime: WIND.uTime, uWind: WIND.uWind, uCamPos: WIND.uCamPos, uWet: WIND.uWet,
    uFade: { value: new THREE.Vector2(radius * 0.72, radius) },
  };
  m.userData.uniforms = U;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, U);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', /* glsl */`#include <common>
        uniform float uTime; uniform vec4 uWind; uniform vec3 uCamPos; uniform vec2 uFade;
        float gr_noise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
          float a = fract(sin(dot(i, vec2(127.1, 311.7))) * 43758.55), b = fract(sin(dot(i + vec2(1, 0), vec2(127.1, 311.7))) * 43758.55);
          float c = fract(sin(dot(i + vec2(0, 1), vec2(127.1, 311.7))) * 43758.55), d = fract(sin(dot(i + vec2(1, 1), vec2(127.1, 311.7))) * 43758.55);
          return mix(mix(a, b, f.x), mix(c, d, f.x), f.y); }
      `)
      .replace('#include <begin_vertex>', /* glsl */`
        vec3 transformed = vec3(position);
        vec3 gr_base = instanceMatrix[3].xyz;
        float gr_dist = distance(uCamPos, gr_base);
        float gr_k = 1.0 - smoothstep(uFade.x, uFade.y, gr_dist);
        transformed *= vec3(mix(1.6, 1.0, gr_k), gr_k, mix(1.6, 1.0, gr_k));
        float gr_h = max(position.y, 0.0);
        vec2 wd = uWind.xy;
        // rain-soaked grass is heavy: it moves less and more slowly than dry grass
        float S = uWind.z * (1.0 + uWind.w) * 0.7;
        float wave = gr_noise(gr_base.xz * 0.12 - wd * uTime * 1.2);
        float flut = sin(uTime * 2.3 + dot(gr_base.xz, vec2(1.7, 2.3))) * 0.25;
        float bendAmt = S * (0.3 + 0.9 * wave + flut) * gr_h * gr_h * 2.0;
        vec3 gr_off = vec3(wd.x, 0.0, wd.y) * bendAmt;
        gr_off.y = -bendAmt * bendAmt * 0.6;
        float gr_s2 = dot(instanceMatrix[0].xyz, instanceMatrix[0].xyz);
        transformed += (transpose(mat3(instanceMatrix)) * gr_off) / gr_s2;
      `);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uWet;')
      .replace('#include <color_fragment>', /* glsl */`
        #include <color_fragment>
        diffuseColor.rgb *= mix(1.0, 0.82, uWet);
      `)
      .replace('#include <roughnessmap_fragment>', /* glsl */`
        #include <roughnessmap_fragment>
        // water film on the blades: glossier, with the sheen strongest on the lighter (straw) blades
        roughnessFactor = mix(roughnessFactor, 0.5, uWet * 0.6);
      `)
      .replace('normal *= faceDirection;', '');
  };
  m.customProgramCacheKey = () => 'grass-v4';
  return m;
}

/** Grass-card material: atlas albedo (x instance dryness tint), alpha test with mip coverage keep, wind, wetness. */
function createGrassCardMaterial(map, normalMap, radius) {
  const m = new THREE.MeshStandardMaterial({
    map, normalMap: normalMap || null, normalScale: new THREE.Vector2(0.5, 0.5), side: THREE.DoubleSide, alphaTest: 0.4,
    roughness: 0.85, metalness: 0, envMapIntensity: 0.3,
  });
  const mapSize = { value: new THREE.Vector2(map?.image?.width || 1024, map?.image?.height || 1024) };
  // uBoost: the post chain's N8AO (intensity 2, radius 2.2 m) sees the dense alpha-tested cards as deep crevices and
  // turns the tussocks near-black; this lifts the albedo back to the measured blade values seen through AO (same idea
  // as FOLIAGE.impBright for the impostors). 1.0 when rendering without post.
  const U = { uTime: WIND.uTime, uWind: WIND.uWind, uCamPos: WIND.uCamPos, uWet: WIND.uWet, uMapSize: mapSize,
    uFade: { value: new THREE.Vector2(radius * 0.72, radius) }, uBoost: GRASS_TUNE.boost };
  m.userData.uniforms = U;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, U);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', /* glsl */`#include <common>
        uniform float uTime; uniform vec4 uWind; uniform vec3 uCamPos; uniform vec2 uFade;
        attribute vec2 aGr; varying vec2 vGr;
        float gr_noise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
          float a = fract(sin(dot(i, vec2(127.1, 311.7))) * 43758.55), b = fract(sin(dot(i + vec2(1, 0), vec2(127.1, 311.7))) * 43758.55);
          float c = fract(sin(dot(i + vec2(0, 1), vec2(127.1, 311.7))) * 43758.55), d = fract(sin(dot(i + vec2(1, 1), vec2(127.1, 311.7))) * 43758.55);
          return mix(mix(a, b, f.x), mix(c, d, f.x), f.y); }`)
      .replace('#include <begin_vertex>', /* glsl */`
        vec3 transformed = vec3(position);
        vec3 gr_base = instanceMatrix[3].xyz;
        float gr_dist = distance(uCamPos, gr_base);
        float gr_k = 1.0 - smoothstep(uFade.x, uFade.y, gr_dist);
        transformed.y = mix(-0.05, transformed.y, gr_k);
        vGr = aGr;
        float gr_h = aGr.x;
        vec2 wd = uWind.xy;
        float S = uWind.z * (1.0 + uWind.w) * 0.7;   // rain-soaked grass is heavy: small, slow motion
        float wave = gr_noise(gr_base.xz * 0.12 - wd * uTime * 1.2);
        float flut = sin(uTime * 2.3 + dot(gr_base.xz, vec2(1.7, 2.3)) + aGr.y * 6.0) * 0.25;
        float bendAmt = S * (0.3 + 0.9 * wave + flut) * gr_h * gr_h * 0.35;
        vec3 gr_off = vec3(wd.x, 0.0, wd.y) * bendAmt;
        gr_off.y = -bendAmt * bendAmt * 0.8;
        float gr_s2 = dot(instanceMatrix[0].xyz, instanceMatrix[0].xyz);
        transformed += (transpose(mat3(instanceMatrix)) * gr_off) / gr_s2;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uWet; uniform vec2 uMapSize; uniform float uBoost; varying vec2 vGr;')
      .replace('#include <alphatest_fragment>', /* glsl */`
        {
          // keep coverage in the small mips (thin blades would otherwise thin out to nothing at 20-50 m)
          vec2 uvp = vMapUv * uMapSize;
          vec2 dx = dFdx(uvp), dy = dFdy(uvp);
          float mip = max(0.0, 0.5 * log2(max(dot(dx, dx), dot(dy, dy))));
          diffuseColor.a *= 1.0 + max(0.0, mip - 1.0) * 0.28;
        }
        #include <alphatest_fragment>
        diffuseColor.rgb *= mix(1.0, 0.82, uWet) * mix(uBoost, 1.0, smoothstep(0.0, 0.8, vGr.x));
      `)
      .replace('#include <roughnessmap_fragment>', /* glsl */`
        #include <roughnessmap_fragment>
        roughnessFactor = mix(roughnessFactor, 0.62, uWet * 0.6);   // water film on the blades
      `)
      .replace('#include <aomap_fragment>', '#include <aomap_fragment>\n  reflectedLight.indirectSpecular *= 0.45;   // blades occlude most of their own sky: no silvery grazing sheet')
      .replace('normal *= faceDirection;', '');
  };
  m.customProgramCacheKey = () => 'grass-card-v2';
  return m;
}

/** Ground-flora card material (bilberry, coltsfoot, butterbur): atlas albedo + normal, alpha test, flutter, wet sheen. */
function createFloraMaterial(map, normalMap, radius, opts = {}) {
  const m = new THREE.MeshStandardMaterial({
    map, normalMap: normalMap || null, side: THREE.DoubleSide, alphaTest: opts.alphaTest ?? 0.45,
    roughness: opts.roughness ?? 0.7, metalness: 0, normalScale: new THREE.Vector2(0.7, 0.7), envMapIntensity: 0.55,
  });
  if (opts.color) m.color.copy(opts.color);
  const U = { uTime: WIND.uTime, uWind: WIND.uWind, uCamPos: WIND.uCamPos, uWet: WIND.uWet,
    uFade: { value: new THREE.Vector2(radius * 0.75, radius) }, uWetRough: { value: opts.wetRough ?? 0.35 } };
  m.userData.uniforms = U;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, U);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uTime; uniform vec4 uWind; uniform vec3 uCamPos; uniform vec2 uFade;\nattribute vec4 aTree; varying float vFlAO;')
      .replace('#include <begin_vertex>', /* glsl */`
        vec3 transformed = vec3(position);
        vec3 fl_base = instanceMatrix[3].xyz;
        float fl_k = 1.0 - smoothstep(uFade.x, uFade.y, distance(uCamPos, fl_base));
        transformed *= fl_k;
        float fl_h = max(position.y, 0.0);
        float S = uWind.z * (1.0 + uWind.w);
        float ph = aTree.g * 6.28 + dot(fl_base.xz, vec2(0.7, 1.3));
        vec3 fl_off = vec3(uWind.x, 0.0, uWind.y) * S * 0.05 * fl_h * sin(uTime * 1.7 + ph)
                    + vec3(0.0, S * 0.015 * fl_h * sin(uTime * 3.1 + ph * 1.7), 0.0);
        float fl_s2 = dot(instanceMatrix[0].xyz, instanceMatrix[0].xyz);
        transformed += (transpose(mat3(instanceMatrix)) * fl_off) / fl_s2;
        vFlAO = aTree.r;
      `);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uWet; uniform float uWetRough; varying float vFlAO;')
      .replace('#include <map_fragment>', '#include <map_fragment>\n  diffuseColor.rgb *= vFlAO * mix(1.0, 0.85, uWet);')
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\n  roughnessFactor = mix(roughnessFactor, uWetRough, uWet);');
  };
  m.customProgramCacheKey = () => 'flora-v1';
  return m;
}

// ------------------------------------------------------------------------------------------------------------------
export class GrassField {
  constructor(ctx, opts = {}) {
    this.ctx = ctx;
    const q = ctx.config?.quality || {};
    this.radius = q.grassRadius ?? 55;
    this.density = q.grassDensity ?? 0.75;
    this.floraRadius = Math.min(this.radius, 36);
    this.group = new THREE.Group();
    this.group.name = 'grass';
    this.chunks = new Map();
    this.pending = [];
    this._camChunk = '';
    this._dirty = true;
    this._frame = 0;

    this.tune = GRASS_TUNE;
    this.canopyAt = typeof opts.canopyAt === 'function' ? opts.canopyAt : null;
    if (ctx.flags?.nopost || (q.ao && q.ao === 'off')) GRASS_TUNE.boost.value = 1.15;
    this.flowerMat = createGrassMaterial(Math.min(this.radius, 30));
    const dens = Math.max(0.3, this.density);
    if (opts.grassMap) {
      // crossed wild-grass cards (atlas rendered from 3D blades in tools/blender/trees.py): dense, mip-filtered, cheap
      this.cards = true;
      this.mat = createGrassCardMaterial(opts.grassMap, opts.grassNormal, this.radius);
      this.nearDry = this._mkInst(buildCardClump(11, ['g_dry', 'g_straw', 'g_dry'], 3, 2), this.mat, Math.round(9000 * dens), 'grass_near_dry', true);
      this.nearGreen = this._mkInst(buildCardClump(17, ['g_green', 'g_mixed', 'g_green'], 3, 2), this.mat, Math.round(9000 * dens), 'grass_near_green', true);
      this.far = this._mkInst(buildCardClump(23, ['g_mixed', 'g_green'], 2, 1, 1.15), this.mat, Math.round(26000 * dens), 'grass_far', true);
    } else {
      // fallback: blade geometry (a drier tussock mix with seed stems and a greener, lower tuft mix)
      this.mat = createGrassMaterial(this.radius);
      this.nearDry = this._mkInst(buildClump(11, { blades: 13, segs: 4, widthMul: 1.2, straw: 0.55, stems: 3, lodge: 0.8 }), this.mat, Math.round(16000 * dens), 'grass_near_dry', true);
      this.nearGreen = this._mkInst(buildClump(17, { blades: 14, segs: 3, widthMul: 1.1, straw: 0.18, stems: 1, lodge: 0.6, tuft: true }), this.mat, Math.round(14000 * dens), 'grass_near_green', true);
      this.far = this._mkInst(buildClump(23, { blades: 5, segs: 2, widthMul: 2.1, straw: 0.4, stems: 1, lodge: 0.7, far: true }), this.mat, Math.round(30000 * dens), 'grass_far', true);
    }
    this.flowers = this._mkInst(buildFlowers(5), this.flowerMat, 1500, 'flowers');
    this.ferns = null; this.bilberry = null; this.coltsfoot = null; this.butterbur = null; this.litter = null;
    if (opts.foliageMap) {
      const fm = createFoliageMaterial({ map: opts.foliageMap, normalMap: opts.foliageNormal,
        lod: new THREE.Vector4(-2, -1, this.radius * 0.8, this.radius), alphaTest: 0.4 });
      fm.color.setRGB(0.62, 0.6, 0.5);   // wet late-season fern: darker, less lime than the card
      this.ferns = this._mkInst(buildFern(3), fm, 2500, 'ferns');
      this.ferns.castShadow = false;   // knee-high fern shadows are invisible under the soft overcast sun
      const R = this.floraRadius;
      this.bilberry = this._mkInst(buildBilberry(31), createFloraMaterial(opts.foliageMap, opts.foliageNormal, R, { roughness: 0.65, color: new THREE.Color(0.8, 0.8, 0.75) }), 2600, 'bilberry');
      this.coltsfoot = this._mkInst(buildForb(41, 'coltsfoot'), createFloraMaterial(opts.foliageMap, opts.foliageNormal, R, { roughness: 0.8, wetRough: 0.55, color: new THREE.Color(0.58, 0.6, 0.58) }), 900, 'coltsfoot');
      this.butterbur = this._mkInst(buildForb(43, 'butterbur'), createFloraMaterial(opts.foliageMap, opts.foliageNormal, R, { roughness: 0.7, wetRough: 0.5, color: new THREE.Color(0.7, 0.74, 0.66) }), 400, 'butterbur');
      const lm = createFloraMaterial(opts.foliageMap, opts.foliageNormal, Math.min(R, 30), { roughness: 0.85, wetRough: 0.55, alphaTest: 0.5,
        color: new THREE.Color(0.62, 0.62, 0.52) });
      lm.polygonOffset = true; lm.polygonOffsetFactor = -2; lm.polygonOffsetUnits = -2; lm.side = THREE.FrontSide;
      this.litter = this._mkInst(buildLitterQuad(), lm, 1200, 'litter');
      this.litter.renderOrder = -1;
    }
  }

  _mkInst(geo, mat, cap, name, colored = false) {
    const im = new THREE.InstancedMesh(geo, mat, cap);
    im.name = name;
    im.count = 0;
    im.frustumCulled = false;
    im.castShadow = false;
    im.receiveShadow = true;
    im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (colored) {
      im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3).fill(1), 3);
      im.instanceColor.setUsage(THREE.DynamicDrawUsage);
    }
    this.group.add(im);
    return im;
  }

  // ---- density / placement -----------------------------------------------------------------------------------
  /** Valley verge (d -4.2..-6.5) is grassy except at the washed-out gap, the pull-off and the tunnel spur. */
  _vergeOK(m, s) {
    m = m || {};
    if (Math.abs(s - (m.gap ?? 560)) < 9) return false;
    if (m.pulloff && s > m.pulloff[0] - 2 && s < m.pulloff[1] + 2) return false;
    if (s > (m.tunnel ?? 1150) - 30 || s < -30) return false;
    return true;
  }

  /** Wet-ditch forbs: uphill ditch + foot of the cut (d 3.1..5.4), seeps near gullies / the gap / the scar edges. */
  _forbAt(m, s, d, ny, x, z, dy = 0) {
    m = m || {};
    if (s > (m.tunnel ?? 1150) - 25 || s < -30) return [0, 0];
    let wet = 0;
    if (m.gullies) for (const g of m.gullies) wet = Math.max(wet, 1 - Math.abs(s - g) / 28);
    wet = Math.max(wet, 1 - Math.abs(s - (m.gap ?? 560)) / 22);
    if (m.scar) wet = Math.max(wet, 0.6 * (1 - Math.min(Math.abs(s - m.scar[0]), Math.abs(s - m.scar[1])) / 20));
    const seep = vnoise(x * 0.07, z * 0.07);
    wet = Math.max(wet, Math.max(0, seep - 0.62) * 2.2);
    let col = 0, but = 0;
    if (d > 3.1 && d < 5.4 && ny > 0.5 && dy < 0.5) {
      if (m.gullies && m.gullies.some((g) => Math.abs(s - g) < 6)) return [0, 0];   // rock chute: bare
      if (Math.abs(s - (m.gap ?? 560)) < 1.8) return [0, 0];
      col = 0.35 + 0.5 * vnoise(x * 0.35 + 3.1, z * 0.35);
      but = wet * (d > 3.4 ? 1 : 0.5);
    } else if (d < -3.85 && d > -5.0 && this._vergeOK(m, s) && ny > 0.6) {
      col = 0.25 * vnoise(x * 0.3 + 7.7, z * 0.3);          // gravel shoulder edge
    } else if (d < -5 && d > -40 && ny > 0.55 && ny < 0.85 && wet > 0.3) {
      but = wet * 0.25; col = wet * 0.3;                      // seepy banks below the road
    }
    return [col, but];
  }

  _densityAt(x, z) {
    const t = this.ctx.terrain;
    if (t?.grassAt && t.meshes?.near) return t.grassAt(x, z);
    const road = this.ctx.road;
    if (!road) return 0;
    const pr = road.project(_v.set(x, 0, z), this._pr || (this._pr = {}));
    if (pr.dist < 4.8) return 0;
    return pr.dist < 40 ? 0.8 : 0;
  }

  _genChunk(key, cx, cz) {
    const phys = this.ctx.physics;
    const ch = { key, cx, cz, grass: null, flowers: null, ferns: null };
    const x0 = cx * CH, z0 = cz * CH;
    const road = this.ctx.road;
    let any = 0;
    for (const [ax, az] of [[0.5, 0.5], [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]]) {
      any = Math.max(any, this._densityAt(x0 + ax * CH, z0 + az * CH));
    }
    if (any < 0.05 && road) {
      const pr = road.project(_v.set(x0 + CH / 2, 0, z0 + CH / 2), this._pr4 || (this._pr4 = {}));
      if (pr.dist < 75 && pr.dist > 1) any = 0.1;
    }
    if (any < 0.05 || !phys?.world) { ch.empty = true; return ch; }
    const H = new Float32Array(GRID * GRID), N = new Float32Array(GRID * GRID), D = new Float32Array(GRID * GRID);
    const FC = new Float32Array(GRID * GRID), FB = new Float32Array(GRID * GRID), VG = new Float32Array(GRID * GRID);
    const CN = new Float32Array(GRID * GRID);
    const baseY = road ? road.project(_v.set(x0 + CH / 2, 0, z0 + CH / 2), this._pr2 || (this._pr2 = {})) : null;
    const fromY = road ? road.pointAt(baseY.s, _p).y + 400 : 2000;
    let hits = 0;
    const m = road?.markers;
    for (let j = 0; j < GRID; j++) {
      for (let i = 0; i < GRID; i++) {
        const x = x0 + i / (GRID - 1) * CH, z = z0 + j / (GRID - 1) * CH;
        const h = phys.raycast(_v.set(x, fromY, z), _d, 3000, { groups: groups(G.ALL, G.STATIC) });
        const k = j * GRID + i;
        if (h) { H[k] = h.point.y; N[k] = h.normal.y; hits++; } else { H[k] = NaN; N[k] = 0; }
        D[k] = this._densityAt(x, z);
        CN[k] = this.canopyAt ? this.canopyAt(x, z) : 0;
        if (h && road) {
          const pr = road.project(h.point, this._pr3 || (this._pr3 = {}));
          const [fc, fb] = this._forbAt(m, pr.s, pr.d, h.normal.y, x, z, pr.dy);
          FC[k] = fc; FB[k] = fb;
          if (pr.dist < 4.15 || (pr.d > 0 && pr.d < 5.6)) D[k] = 0;
          // valley verge just behind the guardrail: rank grass even where TERRAIN's mask calls it gravel/rock,
          // but patchy (bare gravel and soil between tussocks), never a continuous strip
          else if (pr.d < 0 && pr.d > -6.5) {
            const patch = vnoise(x * 0.45, z * 0.45);
            D[k] = Math.max(D[k] * 0.8, this._vergeOK(m, pr.s) ? 0.45 + 0.55 * patch : 0);
            VG[k] = 1;
          } else D[k] = Math.max(D[k], corridorVegetation(m, pr.s, pr.d, Math.min(1, h.normal.y + 0.08)) * 0.95);
        }
      }
    }
    if (!hits) { ch.empty = true; return ch; }
    const r = rng((cx * 73856093) ^ (cz * 19349663));
    const maxPer = Math.round(CH * CH * 5.5 * this.density * (this.cards ? 1.7 : 1));
    const pts = [], fl = [], fe = [], bb = [], cf = [], bu = [];
    const sample = (fx, fz, arr) => {
      const gx = fx * (GRID - 1), gz = fz * (GRID - 1);
      const i = Math.min(GRID - 2, Math.floor(gx)), j = Math.min(GRID - 2, Math.floor(gz));
      const tx = gx - i, tz = gz - j;
      const k00 = j * GRID + i, k10 = k00 + 1, k01 = k00 + GRID, k11 = k01 + 1;
      const a = arr[k00], b = arr[k10], c = arr[k01], d = arr[k11];
      return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz;
    };
    for (let n = 0; n < maxPer; n++) {
      const fx = r(), fz = r();
      const y = sample(fx, fz, H);
      if (!(y === y)) continue; // NaN
      const ny = sample(fx, fz, N);
      const verge = sample(fx, fz, VG);
      // grass clings to 50-60 deg road embankments (the verge below the guardrail); it only gives up on rock faces
      const nyMin = verge > 0.5 ? 0.45 : 0.6;
      if (ny < nyMin) continue;
      const can = Math.min(1, sample(fx, fz, CN) * 1.8) * (1 - verge);
      // shade under the spruce crowns: sparse grass (moss, litter, bilberry and ferns take over)
      const dens = sample(fx, fz, D) * THREE.MathUtils.smoothstep(ny, nyMin, nyMin + 0.2) * (1 - 0.9 * can);
      const wx = x0 + fx * CH, wz = z0 + fz * CH;
      // clumpy distribution: tussock patches with bare soil between (two octaves)
      const patch = 0.35 + 0.65 * Math.min(1, vnoise(wx * 0.5, wz * 0.5) * 0.8 + vnoise(wx * 1.7, wz * 1.7) * 0.5);
      if (r() > dens * patch) continue;
      // dryness: drained verge and steep banks are strawier; hollows / forest edge greener
      const dry = Math.min(1, Math.max(0, (vnoise(wx * 0.11 + 9.3, wz * 0.11) * 0.85 + verge * 0.3 + (1 - ny) * 0.3 - 0.3) * (1 - 0.85 * can)));
      pts.push(wx, y - 0.02, wz, (this.cards ? (0.6 + r() * 0.5) * (1 - 0.35 * can) : 0.8 + r() * 0.6), r() * 6.283, r(), dry);
      if (r() < 0.012 * dens) fl.push(wx, y - 0.01, wz, 0.8 + r() * 0.5, r() * 6.283, r());
      if (r() < 0.012 * (1.2 - dens)) fe.push(wx, y - 0.03, wz, 0.7 + r() * 0.6, r() * 6.283, r());
    }
    // forest floor under the crowns: ferns and dense clonal bilberry patches (independent of the grass mask)
    for (let n = 0; n < 70; n++) {
      const fx = r(), fz = r();
      const can = sample(fx, fz, CN);
      if (can < 0.15) continue;
      const y = sample(fx, fz, H), ny = sample(fx, fz, N);
      if (!(y === y) || ny < 0.6 || sample(fx, fz, VG) > 0.5) continue;
      const wx = x0 + fx * CH, wz = z0 + fz * CH;
      if (road) { const pr = road.project(_v.set(wx, y, wz), this._pr6 || (this._pr6 = {})); if (pr.dist < 6) continue; }
      const bp = vnoise(wx * 0.16 + 21.7, wz * 0.16);
      if (bp > 0.42 && r() < (bp - 0.42) * 3.0 * can) bb.push(wx, y - 0.02, wz, 0.75 + r() * 0.6, r() * 6.283, r());
      else if (r() < 0.12 * can) fe.push(wx, y - 0.03, wz, 0.7 + r() * 0.6, r() * 6.283, r());
    }
    // ferns and bilberry like the steeper, shadier forest floor (less grass)
    for (let n = 0; n < 40; n++) {
      const fx = r(), fz = r();
      const y = sample(fx, fz, H), ny = sample(fx, fz, N), dens = sample(fx, fz, D);
      if (!(y === y) || ny < 0.62 || ny > 0.9 || dens < 0.05 || sample(fx, fz, VG) > 0.5) continue;
      const wx = x0 + fx * CH, wz = z0 + fz * CH;
      if (n < 10) { if (r() < 0.35) fe.push(wx, y - 0.03, wz, 0.8 + r() * 0.6, r() * 6.283, r()); continue; }
      // bilberry grows in dense patches (clonal): noise-gated
      const bp = vnoise(wx * 0.16 + 21.7, wz * 0.16);
      if (bp > 0.52 && r() < (bp - 0.52) * 3.5) bb.push(wx, y - 0.02, wz, 0.75 + r() * 0.6, r() * 6.283, r());
    }
    // litter + moss decals under the crowns (slope-aligned: normal from the ray-grid height gradient)
    const lt = [];
    for (let n = 0; n < 40; n++) {
      const fx = 0.06 + r() * 0.88, fz = 0.06 + r() * 0.88;
      const can = sample(fx, fz, CN);
      if (can < 0.25 || r() > can) continue;
      const y = sample(fx, fz, H);
      if (!(y === y) || sample(fx, fz, VG) > 0.3) continue;
      const e = 0.05;
      const hx = (sample(fx + e, fz, H) - sample(fx - e, fz, H)) / (2 * e * CH);
      const hz = (sample(fx, fz + e, H) - sample(fx, fz - e, H)) / (2 * e * CH);
      if (!(hx === hx) || !(hz === hz) || hx * hx + hz * hz > 1.2) continue;      // steeper than ~48 deg: skip
      const wx = x0 + fx * CH, wz = z0 + fz * CH;
      if (road) { const pr = road.project(_v.set(wx, y, wz), this._pr7 || (this._pr7 = {})); if (pr.dist < 5.5) continue; }
      lt.push(wx, y, wz, 0.8 + r() * 0.7, r() * 6.283, hx, hz);
    }
    ch.litter = new Float32Array(lt);
    // ditch forbs
    for (let n = 0; n < 130; n++) {
      const fx = r(), fz = r();
      const y = sample(fx, fz, H);
      if (!(y === y)) continue;
      const wx = x0 + fx * CH, wz = z0 + fz * CH;
      const c = sample(fx, fz, FC), b = sample(fx, fz, FB);
      const isB = b > 0.05 && r() < b * 0.1;
      const isC = !isB && c > 0.05 && r() < c * 0.35;
      if (!isB && !isC) continue;
      // exact ground under the clump (the bilinear ray-grid height is wrong across the cut-face step)
      const hh = phys.raycast(_v.set(wx, fromY, wz), _d, 3000, { groups: groups(G.ALL, G.STATIC) });
      if (!hh || hh.normal.y < 0.5) continue;
      if (road) {
        const pr = road.project(hh.point, this._pr5 || (this._pr5 = {}));
        if (pr.d > 0 && pr.d < 6 && pr.dy > 0.5) continue;      // on a ledge of the rock cut
        if (pr.dist < 3.9) continue;
      }
      (isB ? bu : cf).push(wx, hh.point.y - 0.02, wz, isB ? 0.8 + r() * 0.5 : 0.75 + r() * 0.65, r() * 6.283, r());
    }
    // the uphill ditch (d 3.3-4.8, ~1.3 m wide) is too narrow for the 1 m ray grid: walk it in road coordinates
    if (road && this.coltsfoot) {
      const pc = road.project(_v.set(x0 + CH / 2, 0, z0 + CH / 2), this._pr8 || (this._pr8 = {}));
      if (pc.dist < 13) {
        for (let ss = pc.s - 7; ss < pc.s + 7; ss += 0.45) {
          const dd = 3.25 + r() * 1.6;
          const w = road.worldAt(ss, dd, _p2);
          if (w.x < x0 || w.x >= x0 + CH || w.z < z0 || w.z >= z0 + CH) continue;
          const [fc, fb] = this._forbAt(m, ss, dd, 0.9, w.x, w.z, -0.3);
          const patch = Math.max(0, vnoise(ss * 0.22, 5.5) * 1.4 - 0.35);
          const isB = fb > 0.1 && r() < fb * 0.3;
          const isC = !isB && r() < fc * patch * 0.8;
          if (!isB && !isC) continue;
          const hh = phys.raycast(_v.set(w.x, fromY, w.z), _d, 3000, { groups: groups(G.ALL, G.STATIC) });
          if (!hh || hh.normal.y < 0.5) continue;
          const pr = road.project(hh.point, this._pr5 || (this._pr5 = {}));
          if (pr.dy > 0.5 || pr.dist < 3.2) continue;
          (isB ? bu : cf).push(w.x, hh.point.y - 0.02, w.z, isB ? 0.8 + r() * 0.5 : 0.75 + r() * 0.65, r() * 6.283, r());
        }
      }
    }
    ch.grass = new Float32Array(pts);
    ch.flowers = new Float32Array(fl);
    ch.ferns = new Float32Array(fe);
    ch.bilberry = new Float32Array(bb);
    ch.coltsfoot = new Float32Array(cf);
    ch.butterbur = new Float32Array(bu);
    return ch;
  }

  // ---- per frame ---------------------------------------------------------------------------------------------
  update(dt) {
    const cam = this.ctx.camera;
    if (!cam) return;
    this._frame++;
    const cp = cam.position;
    const ccx = Math.floor(cp.x / CH), ccz = Math.floor(cp.z / CH);
    const key = ccx + ',' + ccz;
    const R = this.radius, rc = Math.ceil(R / CH) + 1;
    if (key !== this._camChunk) {
      this._camChunk = key;
      this._dirty = true;
      const want = [];
      for (let gz = ccz - rc; gz <= ccz + rc; gz++) {
        for (let gx = ccx - rc; gx <= ccx + rc; gx++) {
          const dx = (gx + 0.5) * CH - cp.x, dz = (gz + 0.5) * CH - cp.z;
          const d = Math.hypot(dx, dz);
          if (d > R + CH) continue;
          const k = gx + ',' + gz;
          if (!this.chunks.has(k)) want.push([d, k, gx, gz]);
        }
      }
      want.sort((a, b) => a[0] - b[0]);
      this.pending = want;
      for (const [k, c] of this.chunks) {
        const dx = (c.cx + 0.5) * CH - cp.x, dz = (c.cz + 0.5) * CH - cp.z;
        if (Math.hypot(dx, dz) > R + CH * 4) this.chunks.delete(k);
      }
    }
    let budget = this._frame < 5 ? 40 : 4;
    const t0 = performance.now();
    while (this.pending.length && budget-- > 0 && performance.now() - t0 < 4) {
      const [, k, gx, gz] = this.pending.shift();
      if (this.chunks.has(k)) continue;
      this.chunks.set(k, this._genChunk(k, gx, gz));
      this._dirty = true;
    }
    if (this._dirty && (this._frame % 3 === 0 || !this.pending.length)) {
      this._dirty = false;
      this._rebuild(cp);
    }
  }

  _rebuild(cp) {
    const R = this.radius;
    const nearR = Math.min(14, R * 0.3);
    const FR2 = this.floraRadius * this.floraRadius;
    let nD = 0, nG = 0, nF = 0, nFl = 0, nFe = 0, nB = 0, nC = 0, nU = 0, nL = 0;
    const capL = this.litter ? this.litter.instanceMatrix.count : 0;
    const cap = (im) => (im ? im.instanceMatrix.count : 0);
    const capD = cap(this.nearDry), capG = cap(this.nearGreen), capF = cap(this.far);
    const capFl = cap(this.flowers), capFe = cap(this.ferns), capB = cap(this.bilberry), capC = cap(this.coltsfoot), capU = cap(this.butterbur);
    const put = (im, k, arr, o, scaleMul = 1) => {
      _q.setFromAxisAngle(Y, arr[o + 4]);
      _s.setScalar(arr[o + 3] * scaleMul);
      _m.compose(_p.set(arr[o], arr[o + 1], arr[o + 2]), _q, _s);
      im.setMatrixAt(k, _m);
    };
    // dryness -> instance tint (straw patches warmer/lighter, lush patches cooler/darker)
    const tint = (im, k, dry, rank) => {
      const j = 0.88 + 0.24 * ((rank * 7.13) % 1);
      _c.setRGB((0.9 + 0.2 * dry) * j, (0.96 + 0.05 * dry) * j, (0.95 - 0.12 * dry) * j);
      im.setColorAt(k, _c);
    };
    const near = (arr, lim, im, capIm, n) => {
      if (!im || !arr) return n;
      for (let o = 0; o < arr.length && n < capIm; o += 6) {
        const ex = arr[o] - cp.x, ez = arr[o + 2] - cp.z;
        if (ex * ex + ez * ez < lim) put(im, n++, arr, o);
      }
      return n;
    };
    for (const c of this.chunks.values()) {
      if (c.empty || !c.grass) continue;
      const dx = (c.cx + 0.5) * CH - cp.x, dz = (c.cz + 0.5) * CH - cp.z;
      const dc = Math.hypot(dx, dz);
      if (dc > R + CH) continue;
      const g = c.grass;
      for (let o = 0; o < g.length; o += STRIDE) {
        const ex = g[o] - cp.x, ez = g[o + 2] - cp.z;
        const d = Math.sqrt(ex * ex + ez * ez);
        if (d > R) continue;
        const rank = g[o + 5], dry = g[o + 6];
        if (d < nearR) {
          // species mix follows dryness
          if (rank < 0.08 + 0.6 * dry) { if (nD < capD) { put(this.nearDry, nD, g, o); tint(this.nearDry, nD++, dry, rank); } }
          else if (nG < capG) { put(this.nearGreen, nG, g, o); tint(this.nearGreen, nG++, dry, rank); }
        } else {
          const keep = THREE.MathUtils.clamp(Math.pow(nearR / d, 1.5) * 1.1, 0.1, 1);
          if (rank < keep && nF < capF) { put(this.far, nF, g, o, 1.0 + (1 - keep) * 0.35); tint(this.far, nF++, dry, rank); }
        }
      }
      if (dc < 32 + CH) nFl = near(c.flowers, 30 * 30, this.flowers, capFl, nFl);
      nFe = near(c.ferns, R * R, this.ferns, capFe, nFe);
      if (dc < this.floraRadius + CH) {
        nB = near(c.bilberry, FR2, this.bilberry, capB, nB);
        nC = near(c.coltsfoot, FR2, this.coltsfoot, capC, nC);
        nU = near(c.butterbur, FR2, this.butterbur, capU, nU);
      }
      if (this.litter && c.litter && dc < 30 + CH) {
        const a = c.litter;
        for (let o = 0; o < a.length && nL < capL; o += 7) {
          const ex = a[o] - cp.x, ez = a[o + 2] - cp.z;
          if (ex * ex + ez * ez > 30 * 30) continue;
          _p.set(-a[o + 5], 1, -a[o + 6]).normalize();
          _q.setFromUnitVectors(Y, _p);
          _q2.setFromAxisAngle(Y, a[o + 4]);
          _q.multiply(_q2);
          _s.setScalar(a[o + 3]);
          _m.compose(_v.set(a[o], a[o + 1], a[o + 2]), _q, _s);
          this.litter.setMatrixAt(nL++, _m);
        }
      }
    }
    for (const [im, n] of [[this.nearDry, nD], [this.nearGreen, nG], [this.far, nF], [this.flowers, nFl], [this.ferns, nFe],
      [this.bilberry, nB], [this.coltsfoot, nC], [this.butterbur, nU], [this.litter, nL]]) {
      if (!im) continue;
      im.count = n;
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
    }
    this.stats = { nearDry: nD, nearGreen: nG, far: nF, flowers: nFl, ferns: nFe, bilberry: nB, coltsfoot: nC, butterbur: nU, litter: nL,
      chunks: this.chunks.size, pending: this.pending.length };
  }

  dispose() {
    this.group.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose?.(); });
    this.group.parent?.remove(this.group);
  }
}
