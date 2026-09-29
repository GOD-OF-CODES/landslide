// Builds the CrazyGames package: only the Low and Medium quality tiers, and without the full-quality files that only
// High/Ultra load. Then zips the CONTENTS of the build (index.html at the top level of the zip).
//
//   node tools/package_crazygames.mjs [out.zip]        (default: ./landslide-crazygames-build.zip)
//   npm run build:crazygames -- [out.zip]
//
// What is left out (Low/Medium never request these; the check at the end fails the build if anything they need is gone):
//   - every original asset that has BOTH a 'mid' and a 'lo' variant in public/assets/q/variants.json
//     (full-resolution textures, full-size models, the 2k lighting HDR, the 4k sky and impostor atlases originals);
//   - sky/sky_8k.jpg (only chosen when shadowMapSize >= 4096, i.e. High/Ultra).
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const OUT_DIR = path.resolve('dist-crazygames');
const zipPath = path.resolve(process.argv[2] || 'landslide-crazygames-build.zip');

execFileSync('npx', ['vite', 'build', '--mode', 'crazygames', '--outDir', OUT_DIR, '--emptyOutDir'], { stdio: 'inherit' });

const assets = path.join(OUT_DIR, 'assets');
const manifest = JSON.parse(fs.readFileSync(path.join(assets, 'q', 'variants.json'), 'utf8'));
const removed = [];
const drop = (rel) => {
  const f = path.join(assets, rel);
  if (fs.existsSync(f)) { removed.push([rel, fs.statSync(f).size]); fs.rmSync(f); }
};
for (const [rel, tiers] of Object.entries(manifest.files)) if (tiers.mid && tiers.lo) drop(rel);
drop('sky/sky_8k.jpg');
// remove directories left empty
const prune = (d) => { for (const e of fs.readdirSync(d, { withFileTypes: true })) if (e.isDirectory()) prune(path.join(d, e.name)); if (d !== OUT_DIR && fs.readdirSync(d).length === 0) fs.rmdirSync(d); };
prune(OUT_DIR);

// sanity: every variant file Low/Medium rely on is present, and index.html is at the top level
const missing = [];
for (const tiers of Object.values(manifest.files)) for (const t of ['mid', 'lo']) if (tiers[t] && !fs.existsSync(path.join(assets, tiers[t]))) missing.push(tiers[t]);
if (missing.length) { console.error('missing variant files:', missing); process.exit(1); }
if (!fs.existsSync(path.join(OUT_DIR, 'index.html'))) { console.error('index.html missing'); process.exit(1); }

fs.rmSync(zipPath, { force: true });
execFileSync('zip', ['-q', '-r', '-X', zipPath, '.', '-x', '*.DS_Store', '-x', '__MACOSX/*'], { cwd: OUT_DIR, stdio: 'inherit' });

let files = 0; const walk = (d) => { for (const e of fs.readdirSync(d, { withFileTypes: true })) e.isDirectory() ? walk(path.join(d, e.name)) : files++; };
walk(OUT_DIR);
const size = fs.statSync(zipPath).size;
console.log(`removed ${removed.length} High/Ultra-only files (${(removed.reduce((a, [, s]) => a + s, 0) / 1e6).toFixed(1)} MB)`);
console.log(`zip: ${zipPath}\n  ${(size / 1024 / 1024).toFixed(1)} MB, ${files} files (limits: 250 MB, 1500 files)`);
