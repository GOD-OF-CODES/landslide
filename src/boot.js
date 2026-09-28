// Tiny entry module: shows the quality screen immediately (it only needs config + gate, a few KB), while the
// main game bundle (three.js, Rapier physics, all systems: several MB) downloads in the background. When the
// player presses Start, the chosen preset is stored and the game boots with it.
import { config, PARAMS, storedQuality } from './core/config.js';
import { showQualityGate } from './ui/gate.js';

const mainModule = import('./main.js'); // starts downloading now; main.js does nothing until start() is called
mainModule.catch(() => {});

async function run() {
  let gateDone = false;
  try { gateDone = sessionStorage.getItem('landslide.gateDone') === '1'; } catch {}
  // Test/debug URLs skip the gate (same rules as core/debug.js flags)
  const skip = PARAMS.has('quality') || PARAMS.has('autostart') || PARAMS.has('cam') || PARAMS.has('camS');
  if (!skip && !gateDone) {
    const key = await showQualityGate(storedQuality() ?? 'low');
    config.setQuality(key);
    try { sessionStorage.setItem('landslide.gateDone', '1'); } catch {}
  }
  const main = await mainModule;
  await main.start();
}

run().catch((e) => {
  console.error('[boot] fatal', e);
  document.body.insertAdjacentHTML('beforeend', `<pre style="position:fixed;inset:20px;color:#f88;background:#000c;padding:20px;z-index:99999;white-space:pre-wrap">Fatal error: ${e?.stack || e}</pre>`);
  window.__READY = true;
});
