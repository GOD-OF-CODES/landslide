// Tiny entry module: shows the quality screen immediately (it only needs config + gate, a few KB), while the
// main game bundle (three.js, Rapier physics, all systems: several MB) downloads in the background. When the
// player presses Start, the chosen preset is stored and the game boots with it.
import { config, PARAMS, storedQuality } from './core/config.js';
import { showQualityGate } from './ui/gate.js';
import { isStaleChunkError, recoverFromStaleBuild, showFatal } from './core/recover.js';

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
  let main;
  try { main = await mainModule; } catch (e) {
    // stale deployment or dropped connection: one automatic reload (the quality choice is kept)
    if (isStaleChunkError(e) && recoverFromStaleBuild(e)) return;
    throw e;
  }
  await main.start();
}

run().catch((e) => {
  console.error('[boot] fatal', e);
  showFatal(e);
});
