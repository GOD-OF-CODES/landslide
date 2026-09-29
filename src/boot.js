// Tiny entry module: shows the quality screen immediately (it only needs config + gate, a few KB), while the
// main game bundle (three.js, Rapier physics, all systems: several MB) downloads in the background. When the
// player presses Start, the chosen preset is stored and the game boots with it.
import { config, PARAMS, storedQuality } from './core/config.js';
import { showQualityGate } from './ui/gate.js';
import { isStaleChunkError, recoverFromStaleBuild, showFatal } from './core/recover.js';
import { platform, installIframeGuards } from './core/platform.js';
import { installFpsCounter } from './ui/fps.js';

// CrazyGames: iframe input guards and SDK init as early as possible (init never throws; standalone = 'disabled')
installIframeGuards();
platform.init();
installFpsCounter(); // on-screen FPS (Settings > Show FPS), off by default

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
  // the SDK must be initialised before the loading screen reports loadingStart; don't wait on a hung SDK forever
  await Promise.race([platform.ready, new Promise((r) => setTimeout(r, 4000))]);
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
