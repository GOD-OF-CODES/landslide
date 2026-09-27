// Downloads CC0 assets from Poly Haven (resolved through the /files API) into raw_assets/.
// Usage: node tools/fetch_assets.mjs
import fs from 'node:fs/promises';
import path from 'node:path';

const OUT = path.resolve('raw_assets');
const API = 'https://api.polyhaven.com/files/';

const HDRIS = [['overcast_soil_puresky', '2k'], ['overcast_soil_puresky', '8k']];
// CC0 texture sets (albedo, OpenGL normal, AO/rough/metal, displacement). The engine's resized copies live in
// public/assets/tex/ (committed); these raw sources feed the Blender bakes (tools/blender/*.py) and tools/sky.mjs.
// Rule 1 of DESIGN.md: NO 3D models are downloaded. Every mesh is built from scratch (Blender Python or JS).
const TEXTURES = [
  // terrain / road / tunnel (engine)
  ['asphalt_02', '2k'], ['aerial_rocks_02', '2k'], ['lichen_rock', '2k'], ['brown_mud_rocks_01', '2k'],
  ['aerial_grass_rock', '2k'], ['dark_rock_02', '2k'], ['rocky_trail', '1k'], ['forest_ground_04', '1k'],
  ['precast_concrete_wall', '1k'], ['concrete_wall_006', '1k'], ['brown_mud_02', '1k'],
  // trees, rocks, props (Blender bakes) + a few engine materials
  ['pine_bark', '2k'], ['mossy_rock', '2k'], ['quarry_wall', '2k'], ['rock_01', '2k'], ['rough_wood', '1k'],
  ['ash_veneer', '1k'], ['hessian_230', '1k'], ['rust_coarse_01', '1k'],
];

async function get(url, dest) {
  try { await fs.access(dest); return; } catch {}
  await fs.mkdir(path.dirname(dest), { recursive: true });
  for (let i = 0; i < 5; i++) {
    try {
      const r = await fetch(url);
      if (r.ok) { await fs.writeFile(dest, Buffer.from(await r.arrayBuffer())); console.log('  ✓', path.relative(OUT, dest)); return; }
      console.warn('  retry', url, r.status);
    } catch (e) { console.warn('  retry', url, e.message); }
  }
  throw new Error('failed ' + url);
}
const files = async (id) => (await fetch(API + id)).json();

async function main() {
  for (const [id, res] of HDRIS) {
    const f = await files(id);
    await get(f.hdri[res].hdr.url, path.join(OUT, 'hdri', `${id}_${res}.hdr`));
    if (f.tonemapped) await get(f.tonemapped.url, path.join(OUT, 'hdri', `${id}_tonemapped.jpg`));
  }
  for (const [id, res] of TEXTURES) {
    const f = await files(id);
    for (const map of ['Diffuse', 'nor_gl', 'arm', 'Displacement']) {
      const entry = f[map]?.[res];
      if (!entry) continue;
      const fmt = entry.jpg || entry.png;
      const ext = entry.jpg ? 'jpg' : 'png';
      await get(fmt.url, path.join(OUT, 'tex', id, `${map.toLowerCase()}.${ext}`));
    }
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
