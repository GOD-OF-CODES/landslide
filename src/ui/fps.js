// On-screen FPS counter, toggled from Settings ("Show FPS"). Stored with the other HUD settings in
// localStorage 'landslide.settings' as `showFps` (default off).
//
// Self-contained: it measures real presented frames with requestAnimationFrame (what the player sees), so it needs
// no access to the game loop. Updates the text twice a second; costs nothing measurable. Colour: green >= 50 fps,
// amber >= 30, red below. Shows the average frame time and the worst frame of the last interval (stutter).

const KEY = 'landslide.settings';
let el = null, on = false, raf = 0;
let frames = 0, acc = 0, worst = 0, last = 0;

function readSetting() {
  try { return !!(JSON.parse(localStorage.getItem(KEY) || '{}') || {}).showFps; } catch { return false; }
}

function ensureEl() {
  if (el) return el;
  el = document.createElement('div');
  el.id = 'fps-counter';
  el.setAttribute('aria-hidden', 'true');
  el.style.cssText = [
    'position:fixed', 'z-index:9000', 'pointer-events:none', 'user-select:none',
    'top:calc(env(safe-area-inset-top, 0px) + 6px)', 'left:calc(env(safe-area-inset-left, 0px) + 8px)',
    'font:600 11px/1.25 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace', 'letter-spacing:.04em',
    'color:#9be38a', 'background:rgba(8,10,11,.55)', 'padding:3px 7px', 'border-radius:4px',
    'text-shadow:0 1px 1px rgba(0,0,0,.6)', 'display:none', 'white-space:nowrap',
  ].join(';');
  el.textContent = '-- FPS';
  document.body.appendChild(el);
  return el;
}

function tick(t) {
  raf = requestAnimationFrame(tick);
  if (last) {
    const dt = Math.min(t - last, 1000);
    frames++; acc += dt; if (dt > worst) worst = dt;
    if (acc >= 500) {
      const fps = (frames * 1000) / acc;
      const ms = acc / frames;
      el.textContent = `${Math.round(fps)} FPS  ${ms.toFixed(1)} ms  max ${worst.toFixed(0)}`;
      el.style.color = fps >= 50 ? '#9be38a' : fps >= 30 ? '#eaa53f' : '#ff6b57';
      frames = 0; acc = 0; worst = 0;
    }
  }
  last = t;
}

/** Show or hide the counter (and persist nothing: the HUD settings own persistence). */
export function setFpsVisible(visible) {
  on = !!visible;
  ensureEl().style.display = on ? 'block' : 'none';
  cancelAnimationFrame(raf);
  last = 0; frames = 0; acc = 0; worst = 0;
  if (on) raf = requestAnimationFrame(tick);
}

export function isFpsVisible() { return on; }

/** Apply the saved setting at startup. */
export function installFpsCounter() {
  setFpsVisible(readSetting());
  // keep in sync when the setting changes in another tab
  window.addEventListener('storage', (e) => { if (e.key === KEY) setFpsVisible(readSetting()); });
}
