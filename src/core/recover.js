// Recovery from stale builds. Every deploy produces new hashed chunk names; a tab that loaded the previous
// deployment (left open across a deploy, restored from a cached page, or a slow connection racing a deploy) then
// asks for chunks that no longer exist -> "Failed to fetch dynamically imported module". The fix is one reload,
// which fetches the current index.html and chunks. A timestamp guard prevents reload loops; the quality choice
// survives (it is in localStorage, and sessionStorage marks the gate as done).

const KEY = 'landslide.chunkReload';
const STALE = /dynamically imported module|Importing a module script failed|error loading dynamically imported module|Failed to fetch|ChunkLoadError|Loading chunk/i;

export function isStaleChunkError(e) {
  return STALE.test(String(e?.message || e));
}

/** Reload once to pick up the current deployment. Returns true if a reload was started. */
export function recoverFromStaleBuild(reason) {
  let last = 0;
  try { last = +(sessionStorage.getItem(KEY) || 0); } catch {}
  if (Date.now() - last < 60000) return false; // already reloaded in the last minute: don't loop
  try { sessionStorage.setItem(KEY, String(Date.now())); } catch {}
  console.warn('[boot] a game file failed to load (likely a newer version was deployed); reloading', reason);
  location.reload();
  return true;
}

/** Shows a friendly error with a reload button instead of a raw stack trace. */
export function showFatal(e) {
  const stale = isStaleChunkError(e);
  const msg = stale
    ? 'A game file could not be downloaded. This usually means a newer version was just published, or the connection dropped.'
    : 'Something went wrong while starting the game.';
  const el = document.createElement('div');
  el.style.cssText = 'position:fixed;inset:0;z-index:99999;display:flex;align-items:center;justify-content:center;background:#0b0d0e;color:#eceee9;font:15px/1.5 system-ui,sans-serif;padding:24px';
  el.innerHTML = `<div style="max-width:560px;text-align:center">
    <div style="font:500 28px/1 'Barlow Condensed','Arial Narrow',sans-serif;letter-spacing:.3em;margin-bottom:18px">LANDSLIDE</div>
    <p style="color:rgba(236,238,233,.75);margin:0 0 22px">${msg}</p>
    <button style="all:unset;cursor:pointer;padding:12px 28px;background:#eaa53f;color:#0b0d0e;font-weight:600;letter-spacing:.18em;text-transform:uppercase;border-radius:4px">Reload</button>
    <pre style="margin-top:22px;color:rgba(236,238,233,.35);font-size:11px;white-space:pre-wrap;text-align:left">${String(e?.message || e).slice(0, 400).replace(/</g, '&lt;')}</pre></div>`;
  el.querySelector('button').onclick = () => { try { sessionStorage.removeItem(KEY); } catch {} location.reload(); };
  document.body.appendChild(el);
  window.__READY = true;
}

// Vite fires this when a preloaded dependency of a dynamic import fails to load.
window.addEventListener('vite:preloadError', (ev) => {
  if (recoverFromStaleBuild(ev?.payload)) ev.preventDefault?.();
});
