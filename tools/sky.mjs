// RENDER workstream: builds the visible sky-dome texture + sky metadata from the 8k HDRI.
//
// Usage: node tools/sky.mjs [--stats-only] [--below=10] [--q=88]
//
// Outputs (public/assets/sky/):
//   sky_8k.jpg  8192 x ~2276  equirect crop: elevation +90 deg (top row) .. -below deg (bottom row), sRGB
//   sky_4k.jpg  4096 x ~1138  same crop, for medium/low quality
//   sky.json    { width, height, elevMinDeg, encode: {k, c}, sunDirTex, meanDirTex, horizon, zenith, ... }
//
// Encoding: x = L * k (k chosen so the upper-hemisphere 99.9th luminance percentile ~ 1.0), then a soft
// highlight shoulder e = x / (1 + c*x) (c = 1 - 1/xmax so xmax maps to exactly 1), stored as sRGB.
// Decode in the shader (the texture is loaded as SRGBColorSpace so the GPU returns linear e):
//   x = e / (1 - c*e);  L = x / k.
// Directions use three.js equirect convention: u = atan(dir.z, dir.x)/(2pi) + 0.5, v = asin(dir.y)/pi + 0.5,
// with v = 1 at the top image row (flipY = true), so they can be fed straight into the engine.
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import sharp from 'sharp';

const args = Object.fromEntries(process.argv.slice(2).map((a) => { const [k, v] = a.replace(/^--/, '').split('='); return [k, v ?? true]; }));
const SRC = path.resolve('raw_assets/hdri/overcast_soil_puresky_8k.hdr');
const OUT = path.resolve('public/assets/sky');
const BELOW = +(args.below ?? 10);   // degrees below the horizon kept in the crop
const Q = +(args.q ?? 94);

console.time('parse');
const loader = new HDRLoader();
loader.setDataType(THREE.FloatType);
const buf = fs.readFileSync(SRC);
const tex = loader.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const W = tex.width, H = tex.height, D = tex.data; // RGBA float32, row 0 = top (zenith)
console.timeEnd('parse');
console.log('size', W, H, D.constructor.name);

const lum = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
// row r (0 = top) -> elevation (radians)
const elevOfRow = (r) => (0.5 - (r + 0.5) / H) * Math.PI;
// three.js equirect direction for (col, row)
function dirOf(c, r) {
  const u = (c + 0.5) / W, v = 1 - (r + 0.5) / H;
  const phi = (u - 0.5) * 2 * Math.PI, el = (v - 0.5) * Math.PI;
  return [Math.cos(phi) * Math.cos(el), Math.sin(el), Math.sin(phi) * Math.cos(el)];
}

// ---------- statistics on a 512x256 downsample ----------
const SW = 512, SH = 256, bx = W / SW, by = H / SH;
const small = new Float32Array(SW * SH * 3);
for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
  let r = 0, g = 0, b = 0;
  for (let j = 0; j < by; j++) for (let i = 0; i < bx; i++) {
    const o = (((y * by + j) * W) + x * bx + i) * 4;
    r += D[o]; g += D[o + 1]; b += D[o + 2];
  }
  const n = bx * by, o = (y * SW + x) * 3;
  small[o] = r / n; small[o + 1] = g / n; small[o + 2] = b / n;
}
const sDir = (x, y) => { const u = (x + 0.5) / SW, v = 1 - (y + 0.5) / SH; const phi = (u - 0.5) * 2 * Math.PI, el = (v - 0.5) * Math.PI; return [Math.cos(phi) * Math.cos(el), Math.sin(el), Math.sin(phi) * Math.cos(el)]; };
const sEl = (y) => (0.5 - (y + 0.5) / SH) * Math.PI;

// gaussian-blurred luminance (wrap in x) to find the brightest *region* (sun behind clouds)
function blurLum(sigma) {
  const L = new Float32Array(SW * SH);
  for (let i = 0; i < SW * SH; i++) L[i] = lum(small[i * 3], small[i * 3 + 1], small[i * 3 + 2]);
  const rad = Math.ceil(sigma * 3), k = [];
  for (let i = -rad; i <= rad; i++) k.push(Math.exp(-(i * i) / (2 * sigma * sigma)));
  const ks = k.reduce((a, b) => a + b, 0);
  const T = new Float32Array(SW * SH), O = new Float32Array(SW * SH);
  for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
    let s = 0; for (let i = -rad; i <= rad; i++) s += L[y * SW + ((x + i + SW) % SW)] * k[i + rad]; T[y * SW + x] = s / ks;
  }
  for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
    let s = 0; for (let i = -rad; i <= rad; i++) s += T[Math.min(SH - 1, Math.max(0, y + i)) * SW + x] * k[i + rad]; O[y * SW + x] = s / ks;
  }
  return O;
}
const stats = {};
for (const sigma of [3, 8]) {
  const B = blurLum(sigma);
  let best = -1, bi = 0;
  for (let y = 0; y < SH / 2; y++) for (let x = 0; x < SW; x++) { const l = B[y * SW + x]; if (sEl(y) > 0.02 && l > best) { best = l; bi = y * SW + x; } }
  const bxp = bi % SW, byp = Math.floor(bi / SW);
  stats['brightest_sigma' + sigma] = { dir: sDir(bxp, byp), elevDeg: sEl(byp) * 180 / Math.PI, azDeg: ((bxp + 0.5) / SW - 0.5) * 360, lum: best };
}
// luminance-weighted mean direction of the upper hemisphere (first-order SH ~ dominant light direction)
let md = [0, 0, 0], wsum = 0, E = [0, 0, 0], upperLum = [];
const hor = [0, 0, 0], zen = [0, 0, 0], low = [0, 0, 0]; let hn = 0, zn = 0, ln = 0;
for (let y = 0; y < SH; y++) {
  const el = sEl(y), dA = Math.cos(el);
  for (let x = 0; x < SW; x++) {
    const o = (y * SW + x) * 3, r = small[o], g = small[o + 1], b = small[o + 2], l = lum(r, g, b);
    if (el > 0) {
      const d = sDir(x, y), w = l * dA;
      md[0] += d[0] * w; md[1] += d[1] * w; md[2] += d[2] * w; wsum += w;
      // irradiance on an upward-facing plane (cosine weighted)
      E[0] += r * dA * Math.sin(el); E[1] += g * dA * Math.sin(el); E[2] += b * dA * Math.sin(el);
      upperLum.push(l);
    }
    const eDeg = el * 180 / Math.PI;
    if (eDeg > 0 && eDeg < 4) { hor[0] += r; hor[1] += g; hor[2] += b; hn++; }
    if (eDeg > 70) { zen[0] += r; zen[1] += g; zen[2] += b; zn++; }
    if (eDeg < -10 && eDeg > -40) { low[0] += r; low[1] += g; low[2] += b; ln++; }
  }
}
const dA = (2 * Math.PI / SW) * (Math.PI / SH);
const mlen = Math.hypot(...md);
stats.meanDir = md.map((v) => v / mlen);
stats.meanDirStrength = mlen / wsum; // 0 = isotropic, 1 = point light
stats.irradianceUp = E.map((v) => v * dA);
stats.horizon = hor.map((v) => v / hn);
stats.zenith = zen.map((v) => v / zn);
stats.lowerHemi = low.map((v) => v / ln);
upperLum.sort((a, b) => a - b);
const pct = (p) => upperLum[Math.min(upperLum.length - 1, Math.floor(p * upperLum.length))];
stats.upperLumPct = { p01: pct(0.01), p50: pct(0.5), p90: pct(0.9), p99: pct(0.99), p999: pct(0.999), max: upperLum[upperLum.length - 1] };
// full-res max in the upper hemisphere
let fmax = 0; for (let r = 0; r < H / 2; r++) for (let c = 0; c < W; c++) { const o = (r * W + c) * 4; const l = lum(D[o], D[o + 1], D[o + 2]); if (l > fmax) fmax = l; }
stats.upperLumMaxFull = fmax;
console.log(JSON.stringify(stats, null, 1));
if (args['stats-only']) process.exit(0);

// ---------- encode crop ----------
const rows = Math.round(((90 + BELOW) / 180) * H);
const k = 1 / stats.upperLumPct.p999;               // p99.9 -> 1.0
const xmax = Math.max(1.05, fmax * k);               // brightest texel maps to exactly e = 1 (no clipping)
const c = 1 - 1 / xmax;                              // e(xmax) = 1
const toSRGB = (v) => (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
// 1D LUT for speed: e in [0,1] -> sRGB byte (4096 steps)
const LUT = new Uint8Array(4097);
for (let i = 0; i <= 4096; i++) LUT[i] = Math.max(0, Math.min(255, Math.round(toSRGB(i / 4096) * 255)));
function encodeRows(srcW, srcRows, get) {
  const out = Buffer.alloc(srcW * srcRows * 3);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let r = 0; r < srcRows; r++) for (let x = 0; x < srcW; x++) {
    const [R, G, B] = get(x, r);
    for (let ch = 0; ch < 3; ch++) {
      const X = Math.max(0, [R, G, B][ch] * k);
      const e = Math.min(1, X / (1 + c * X));
      // tiny triangular dither in LUT space to fight banding in smooth cloud gradients
      const idx = Math.max(0, Math.min(4096, Math.round(e * 4096 + (rnd() - rnd()) * 6)));
      out[(r * srcW + x) * 3 + ch] = LUT[idx];
    }
  }
  return out;
}
fs.mkdirSync(OUT, { recursive: true });
console.time('encode8k');
const px8 = encodeRows(W, rows, (x, r) => { const o = (r * W + x) * 4; return [D[o], D[o + 1], D[o + 2]]; });
await sharp(px8, { raw: { width: W, height: rows, channels: 3 } }).jpeg({ quality: Q, mozjpeg: true, chromaSubsampling: '4:2:0' }).toFile(path.join(OUT, 'sky_8k.jpg'));
console.timeEnd('encode8k');
// 4k: 2x2 box downsample in linear light
const W4 = W / 2, rows4 = Math.floor(rows / 2);
const px4 = encodeRows(W4, rows4, (x, r) => {
  let R = 0, G = 0, B = 0;
  for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) { const o = (((r * 2 + j) * W) + x * 2 + i) * 4; R += D[o]; G += D[o + 1]; B += D[o + 2]; }
  return [R / 4, G / 4, B / 4];
});
await sharp(px4, { raw: { width: W4, height: rows4, channels: 3 } }).jpeg({ quality: Q, mozjpeg: true, chromaSubsampling: '4:2:0' }).toFile(path.join(OUT, 'sky_4k.jpg'));

const meta = {
  source: 'overcast_soil_puresky_8k.hdr (Poly Haven, CC0)',
  files: { '8k': { file: 'sky_8k.jpg', width: W, height: rows }, '4k': { file: 'sky_4k.jpg', width: W4, height: rows4 } },
  elevMinDeg: -((rows / H) * 180 - 90), // exact bottom-edge elevation of the crop
  encode: { k, c, note: 'linear e from sRGB texture; x = e/(1-c*e); L = x/k' },
  sunDirTex: stats.brightest_sigma8.dir, sunDirTexSharp: stats.brightest_sigma3.dir,
  sunElevDeg: stats.brightest_sigma8.elevDeg, sunAzDeg: stats.brightest_sigma8.azDeg,
  meanDirTex: stats.meanDir, meanDirStrength: stats.meanDirStrength,
  horizon: stats.horizon, zenith: stats.zenith, lowerHemi: stats.lowerHemi, irradianceUp: stats.irradianceUp,
  upperLumPct: stats.upperLumPct, upperLumMax: fmax,
};
fs.writeFileSync(path.join(OUT, 'sky.json'), JSON.stringify(meta, null, 1));
for (const f of ['sky_8k.jpg', 'sky_4k.jpg']) console.log(f, (fs.statSync(path.join(OUT, f)).size / 1e6).toFixed(2), 'MB');
console.log(JSON.stringify(meta, null, 1));
