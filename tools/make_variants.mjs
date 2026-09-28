// Builds the lighter asset variants used by the MEDIUM and LOW quality presets.
// Ultra/High always load the original full-quality files; nothing here touches them.
//
//   node tools/make_variants.mjs          (re-run after any asset changes)
//
// Output: public/assets/q/<tier>/<same relative path>, plus public/assets/q/variants.json listing every
// file that has a variant, e.g. "tex/asphalt_02/diffuse.jpg" -> "q/mid/tex/asphalt_02/diffuse.webp".
// core/assets.js rewrites requests through this manifest for medium ("mid") and low ("lo").
import fs from 'node:fs/promises';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import sharp from 'sharp';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression } from '@gltf-transform/extensions';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';

const PUB = path.resolve('public/assets');
const OUT = path.join(PUB, 'q');
const TIERS = { mid: 1024, lo: 512 };
// the most visible surfaces keep 1k on low (the road you drive on, the cut face beside it)
const LO_KEEP_1K = new Set(['asphalt_02', 'aerial_rocks_02', 'lichen_rock']);

const manifest = { version: 1, tiers: Object.keys(TIERS), files: {} };
const add = (rel, tier, outRel) => { (manifest.files[rel] ||= {})[tier] = outRel; };

async function textures() {
  const sets = await fs.readdir(path.join(PUB, 'tex'));
  for (const set of sets) {
    const dir = path.join(PUB, 'tex', set);
    if (!(await fs.stat(dir)).isDirectory()) continue;
    for (const f of await fs.readdir(dir)) {
      if (!/\.(jpg|jpeg|png)$/i.test(f)) continue;
      const src = path.join(dir, f);
      const meta = await sharp(src).metadata();
      for (const [tier, size0] of Object.entries(TIERS)) {
        const size = tier === 'lo' && LO_KEEP_1K.has(set) ? 1024 : size0;
        const target = Math.min(size, meta.width);
        const rel = `tex/${set}/${f}`;
        const outRel = `q/${tier}/tex/${set}/${f.replace(/\.\w+$/, '.webp')}`;
        const isNormal = /nor/i.test(f);
        await fs.mkdir(path.dirname(path.join(PUB, outRel)), { recursive: true });
        await sharp(src).resize(target, target, { kernel: 'lanczos3' })
          .webp({ quality: isNormal ? 90 : 84, effort: 5, smartSubsample: !isNormal })
          .toFile(path.join(PUB, outRel));
        add(rel, tier, outRel);
      }
    }
  }
}

async function images(relList) {
  for (const rel of relList) {
    const src = path.join(PUB, rel);
    try { await fs.access(src); } catch { continue; }
    const meta = await sharp(src).metadata();
    for (const [tier, size] of Object.entries(TIERS)) {
      const scale = tier === 'mid' ? 0.75 : 0.5;
      const w = Math.round(meta.width * scale), h = Math.round(meta.height * scale);
      const outRel = `q/${tier}/${rel.replace(/\.\w+$/, '.webp')}`;
      await fs.mkdir(path.dirname(path.join(PUB, outRel)), { recursive: true });
      await sharp(src).resize(w, h, { kernel: 'lanczos3' }).webp({ quality: /normal/.test(rel) ? 90 : 86, effort: 5 }).toFile(path.join(PUB, outRel));
      add(rel, tier, outRel);
      void size;
    }
  }
}

// Assets whose texture detail matters too much to shrink on low (the conifer needle atlas turns blotchy at 512)
const KEEP_FULL = { lo: new Set(['models/trees.glb']), mid: new Set() };

// `gltf-transform resize` decodes EXT_meshopt_compression and writes the geometry uncompressed (car.glb: +2 MB), which
// ate most of the texture saving and left hands.glb without a variant. Re-encode it with meshopt WITHOUT quantization:
// lossless, every vertex attribute stays bit-identical to the original (QA-checked on car/hands), only smaller.
let _io = null;
async function gltfIO() {
  if (!_io) {
    await MeshoptDecoder.ready; await MeshoptEncoder.ready;
    _io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder, 'meshopt.encoder': MeshoptEncoder });
  }
  return _io;
}
async function recompress(file) {
  const io = await gltfIO(), doc = await io.read(file);
  doc.createExtension(EXTMeshoptCompression).setRequired(true)
    .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE }); // no quantize() pass: no filters
  await io.write(file, doc);
}
/** Largest texture edge in a GLB (px). */
async function maxTextureSize(file) {
  const doc = await (await gltfIO()).read(file);
  return Math.max(0, ...doc.getRoot().listTextures().map((t) => Math.max(...(t.getSize() || [0, 0]))));
}

async function models(relList) {
  for (const rel of relList) {
    const src = path.join(PUB, rel);
    try { await fs.access(src); } catch { continue; }
    const maxTex = await maxTextureSize(src);
    for (const [tier, size] of Object.entries(TIERS)) {
      if (KEEP_FULL[tier]?.has(rel)) continue;
      // nothing to shrink: no variant (resize would still re-encode every image lossily at the same size)
      if (maxTex <= size) continue;
      const outRel = `q/${tier}/${rel}`;
      const out = path.join(PUB, outRel);
      await fs.mkdir(path.dirname(out), { recursive: true });
      // resize only shrinks textures larger than the limit; the geometry is re-encoded losslessly (see recompress)
      execFileSync('npx', ['gltf-transform', 'resize', src, out, '--width', String(size), '--height', String(size)], { stdio: 'pipe' });
      await recompress(out);
      const [a, b] = [(await fs.stat(src)).size, (await fs.stat(out)).size];
      if (b >= a * 0.97) { await fs.rm(out); continue; } // no gain (no big textures inside)
      add(rel, tier, outRel);
    }
  }
}

async function hdr() {
  // 1k version of the same Poly Haven HDRI for the lighting capture on medium/low
  const rel = 'sky/env_2k.hdr';
  const outRel = 'q/lo/sky/env_1k.hdr';
  const out = path.join(PUB, outRel);
  // cached in raw_assets/ (git-ignored) so a re-run does not need the network (the q/ folder is wiped above)
  const cache = path.resolve('raw_assets/hdri/overcast_soil_puresky_1k.hdr');
  try { await fs.access(out); } catch {
    let buf;
    try { buf = await fs.readFile(cache); } catch {
      const f = await (await fetch('https://api.polyhaven.com/files/overcast_soil_puresky')).json();
      buf = Buffer.from(await (await fetch(f.hdri['1k'].hdr.url)).arrayBuffer());
      try { await fs.mkdir(path.dirname(cache), { recursive: true }); await fs.writeFile(cache, buf); } catch { /* optional cache */ }
    }
    await fs.mkdir(path.dirname(out), { recursive: true });
    await fs.writeFile(out, buf);
  }
  add(rel, 'mid', outRel);
  add(rel, 'lo', outRel);
}

await fs.rm(OUT, { recursive: true, force: true });
await textures();
await images(['models/impostors/albedo.webp', 'models/impostors/normal.webp', 'sky/sky_4k.jpg']);
await models(['models/trees.glb', 'models/props.glb', 'models/car.glb', 'models/rocks.glb', 'models/hands.glb']);
await hdr();
await fs.writeFile(path.join(OUT, 'variants.json'), JSON.stringify(manifest));

const du = async (d) => { let t = 0; for (const e of await fs.readdir(d, { withFileTypes: true })) { const p = path.join(d, e.name); t += e.isDirectory() ? await du(p) : (await fs.stat(p)).size; } return t; };
for (const t of Object.keys(TIERS)) console.log(t, (await du(path.join(OUT, t)) / 1e6).toFixed(1), 'MB');
console.log('entries', Object.keys(manifest.files).length);
