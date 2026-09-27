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

async function models(relList) {
  for (const rel of relList) {
    const src = path.join(PUB, rel);
    try { await fs.access(src); } catch { continue; }
    for (const [tier, size] of Object.entries(TIERS)) {
      const outRel = `q/${tier}/${rel}`;
      const out = path.join(PUB, outRel);
      await fs.mkdir(path.dirname(out), { recursive: true });
      // resize only shrinks textures larger than the limit; geometry is untouched
      execFileSync('npx', ['gltf-transform', 'resize', src, out, '--width', String(size), '--height', String(size)], { stdio: 'pipe' });
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
  try { await fs.access(out); } catch {
    const f = await (await fetch('https://api.polyhaven.com/files/overcast_soil_puresky')).json();
    const buf = Buffer.from(await (await fetch(f.hdri['1k'].hdr.url)).arrayBuffer());
    await fs.mkdir(path.dirname(out), { recursive: true });
    await fs.writeFile(out, buf);
  }
  add(rel, 'mid', outRel);
  add(rel, 'lo', outRel);
}

await fs.rm(OUT, { recursive: true, force: true });
await textures();
await images(['models/impostors/albedo.webp', 'models/impostors/normal.webp', 'sky/sky_4k.jpg']);
await models(['models/trees.glb', 'models/props.glb', 'models/car.glb', 'models/rocks.glb']);
await hdr();
await fs.writeFile(path.join(OUT, 'variants.json'), JSON.stringify(manifest));

const du = async (d) => { let t = 0; for (const e of await fs.readdir(d, { withFileTypes: true })) { const p = path.join(d, e.name); t += e.isDirectory() ? await du(p) : (await fs.stat(p)).size; } return t; };
for (const t of Object.keys(TIERS)) console.log(t, (await du(path.join(OUT, t)) / 1e6).toFixed(1), 'MB');
console.log('entries', Object.keys(manifest.files).length);
