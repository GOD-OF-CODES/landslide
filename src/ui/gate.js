// Pre-load quality screen: shown before the renderer or any heavy asset is created, so the chosen preset decides
// what gets downloaded (ultra/high = original full-quality assets, medium/low = lighter variants).
// Low is preselected unless the player chose something before. Skipped with ?quality=, ?autostart and ?cam flags.
//
// It is built from the same parts as the loading screen that follows it (src/ui/style.css: the .bd backdrop, the
// centred wordmark and kicker, the .load-foot row), so the hand-over is seamless: on Continue the choices fade out and
// a "Preparing" bar takes their place in the exact spot where the HUD's loading bar appears; the gate then stays up
// until the HUD has built its loading screen underneath, and fades away over it.
//
// Touch UI (RESPONSIVE): this module is the first UI code to run (src/boot.js imports it), so it also decides the touch
// UI mode for every screen: `html.touch` is set here before any screen exists (style.css scopes every touch rule under
// it, so desktop is untouched), the "Rotate your device" overlay is installed (CSS shows it in portrait), and iOS
// pinch/double-tap zoom is blocked. The HUD re-syncs the mode with ctx.input.touch once the input exists.
import { QUALITY_PRESETS } from '../core/config.js';
import * as CFG from '../core/config.js'; // namespace import: DEVICE may be absent (then the local fallback is used)

const INFO = {
  low: { title: 'Low', blurb: 'Runs on most laptops and integrated graphics. Lighter textures, fewer effects.' },
  medium: { title: 'Medium', blurb: 'For recent laptops. Sharper textures, soft shadows and ambient occlusion.' },
  high: { title: 'High', blurb: 'Full-quality assets and effects. For Apple Silicon or a dedicated GPU.' },
  ultra: { title: 'Ultra', blurb: 'Everything at maximum, at native Retina resolution. For powerful GPUs.' },
};
const ORDER = ['low', 'medium', 'high', 'ultra'];
/** Download per preset in MB, measured on the production build: the asset files each preset requests at boot, each
 *  file counted once (QA: brown_mud_rocks_01 and car.glb are requested twice, the second time a 304 from the cache,
 *  which the earlier count included: High read 95). Code (JS, wasm, CSS) adds about 2.3 MB more on the wire.
 *  Shared with the loading screen, which counts the same bytes. config.js `download` is only a fallback.
 *  Re-measured after round 6 (hands.glb, fallen spruce v2, variants re-encoded with lossless meshopt geometry):
 *  low 23.4, medium 32.6, high/ultra 89.1 MB. */
export const DOWNLOAD_MB = { low: 23, medium: 33, high: 89, ultra: 89 };
export const QUALITY_INFO = INFO;
const BACKDROP = `<div class="bd"><div class="bd-topo"></div>
  <div class="bd-ridge far"></div><div class="bd-fog f1"></div><div class="bd-ridge near"></div><div class="bd-fog f2"></div>
  <div class="bd-rain r1"></div><div class="bd-rain r2"></div><div class="bd-grain"></div></div>`;

// ------------------------------------------------------------------------------------------------ touch UI mode
/** Touch-primary device (phone, tablet)? ?touch=1 / ?touch=0 force either mode. Uses the shared detection
 *  (core/config.js DEVICE.touch) when it exists; the fallback mirrors it: coarse primary pointer + touch points/events
 *  (a touchscreen laptop driven by a mouse or trackpad stays in desktop mode). */
export function detectTouchUI() {
  const d = CFG.DEVICE;
  if (d && typeof d.touch === 'boolean') return d.touch;
  try {
    const p = new URLSearchParams(location.search).get('touch');
    if (p === '1' || p === 'true' || p === 'on' || p === '') return true;
    if (p === '0' || p === 'false' || p === 'off') return false;
  } catch {}
  try {
    const tp = (navigator.maxTouchPoints || 0) > 0 || 'ontouchstart' in window;
    return tp && !!window.matchMedia?.('(pointer: coarse)').matches;
  } catch { return false; }
}
let touchUI = false;
export const isTouchUI = () => touchUI;
/** Switch the touch UI on/off (html.touch + the rotate overlay). The HUD calls this with ctx.input.touch. */
export function setTouchUI(on) {
  touchUI = !!on;
  const html = document.documentElement;
  html.classList.toggle('touch', touchUI);
  if (touchUI) { installRotateOverlay(); installZoomGuards(); }
}

const PHONE_SVG = `<svg class="ro-phone" viewBox="0 0 160 160" aria-hidden="true">
  <path class="ro-arrow" d="M122 52 A52 52 0 0 0 76 28" fill="none" stroke-width="1.6" stroke-linecap="round"/>
  <path class="ro-arrow" d="M83 23.5 L75.5 28 L82.5 33" fill="none" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
  <g class="ro-body">
    <rect x="61" y="44" width="38" height="72" rx="7" fill="rgba(10,12,13,.6)" stroke-width="2"/>
    <rect x="65.5" y="53" width="29" height="54" rx="1.5" class="ro-screen"/>
    <line x1="75" y1="48.5" x2="85" y2="48.5" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="74" y1="111.5" x2="86" y2="111.5" stroke-width="1.4" stroke-linecap="round" opacity=".6"/>
  </g>
  <path class="ro-glyph" d="M66 90 L76 71 L82 80 L86 75 L94 90 Z"/>
</svg>`;

/** "Rotate your device": built once, on touch devices only; style.css shows it in portrait (above the gate and every
 *  screen, below the WebGL "tap to reload" screen). The HUD pauses the game while it is up. */
function installRotateOverlay() {
  if (document.getElementById('rotate-overlay') || !document.body) return;
  const el = document.createElement('div');
  el.id = 'rotate-overlay';
  el.setAttribute('role', 'alertdialog');
  el.setAttribute('aria-label', 'Rotate your device to landscape');
  el.innerHTML = `${BACKDROP}
    <div class="ro-center">
      <div class="brand-mark">LANDSLIDE</div>
      <div class="load-kicker">A mountain road. Rain. No way back.</div>
      ${PHONE_SVG}
      <div class="ro-title">Rotate your device</div>
      <p class="ro-sub">Landslide is played in landscape. Turn your device sideways to continue.</p>
    </div>`;
  // nothing under the overlay reacts to a stray touch
  const stop = (e) => { if (e.cancelable) e.preventDefault(); e.stopPropagation(); };
  for (const t of ['touchstart', 'touchmove', 'touchend', 'pointerdown', 'click']) el.addEventListener(t, stop, { passive: false });
  document.body.appendChild(el);
}

/** iOS Safari ignores user-scalable=no: block its pinch gesture events and double-tap zoom (touch-action in style.css
 *  covers the rest). */
let zoomGuards = false;
function installZoomGuards() {
  if (zoomGuards) return;
  zoomGuards = true;
  const no = (e) => { if (e.cancelable) e.preventDefault(); };
  for (const t of ['gesturestart', 'gesturechange', 'gestureend']) document.addEventListener(t, no, { passive: false });
  document.addEventListener('dblclick', no, { passive: false });
}

// Test hook: ?safe=l,t,r,b fakes the safe-area insets in px (emulators report 0), e.g. ?safe=47,0,47,21 (iPhone).
try {
  const s = new URLSearchParams(location.search).get('safe');
  if (s) {
    const [l = 0, t = 0, r = 0, b = 0] = s.split(',').map((v) => Math.max(0, +v || 0));
    const st = document.documentElement.style;
    st.setProperty('--sa-l', l + 'px'); st.setProperty('--sa-t', t + 'px'); st.setProperty('--sa-r', r + 'px'); st.setProperty('--sa-b', b + 'px');
  }
} catch {}
setTouchUI(detectTouchUI());
// phones and tablets: the preset blurbs speak about the device in hand (the numbers come from core/config.js)
const TOUCH_BLURB = {
  low: 'Best for most phones. Lighter textures and fewer effects for a smooth frame rate.',
  medium: 'For recent phones and tablets. Sharper textures and softer shadows.',
  high: 'Full-quality textures and effects. For the newest high-end phones and tablets.',
  ultra: 'Everything at maximum. Very demanding on a phone or tablet.',
};
if (touchUI) for (const k of ORDER) INFO[k].blurb = TOUCH_BLURB[k];

/** Rough device hint from the WebGL renderer string. It's only a suggestion; the player decides. */
export function recommendQuality() {
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return 'low';
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const r = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)).toLowerCase();
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    const mem = navigator.deviceMemory || 8;
    const mobile = /android|iphone|ipad|mobile/i.test(navigator.userAgent);
    if (mobile || mem <= 4 || /swiftshader|llvmpipe|software|mali|adreno|powervr|intel.*(hd|uhd)/.test(r)) return 'low';
    if (/rtx|radeon rx|rx [5-9]\d{3}|m[1-9] (pro|max|ultra)|m[2-9]\b|apple m[2-9]/.test(r)) return 'high';
    if (/gtx|radeon|iris|apple m1|apple gpu|apple/.test(r)) return 'medium';
    return 'low';
  } catch { return 'low'; }
}

export function showQualityGate(initial = 'low') {
  const rec = recommendQuality();
  let sel = ORDER.includes(initial) ? initial : 'low';
  const mb = (k) => DOWNLOAD_MB[k] ?? QUALITY_PRESETS[k]?.download ?? '?';
  const root = document.createElement('div');
  root.id = 'quality-gate';
  root.innerHTML = `${BACKDROP}
  <div class="load-center qg-center" aria-hidden="true">
    <div class="brand-mark">LANDSLIDE</div>
    <div class="load-kicker">A mountain road. Rain. No way back.</div>
  </div>
  <div class="qg-foot" role="dialog" aria-label="Choose graphics quality">
    <div class="load-row"><span class="load-label" id="qg-label">Graphics quality</span><span class="qg-sugg">Suggested for this device: <b>${INFO[rec].title}</b></span></div>
    <div class="qg-seg" role="radiogroup" aria-labelledby="qg-label">
      ${ORDER.map((k) => `<button type="button" role="radio" data-q="${k}" aria-checked="false" tabindex="-1">
        ${k === rec ? '<span class="qg-rec">Suggested</span>' : ''}<span class="qg-name">${INFO[k].title}</span><span class="qg-mb">${mb(k)} MB</span></button>`).join('')}
    </div>
    <p class="qg-blurb" aria-live="polite"></p>
    <div class="qg-actions">
      <button type="button" class="m-item primary qg-go"><span class="m-bar"></span><span class="m-label">Continue</span></button>
      <span class="qg-note">${touchUI ? 'You can change this later in Settings. The game lowers its resolution by itself if it stutters.' : 'You can change this later in Settings. Low and Medium also lower the resolution by themselves if the game stutters.'}</span>
    </div>
  </div>
  <div class="load-foot qg-prep" aria-hidden="true">
    <div class="load-row"><span class="load-label">Preparing</span><span class="load-pct">0%</span></div>
    <div class="load-bar busy"><i></i><b></b></div>
    <div class="load-meta"><span class="load-q"></span><span class="load-eta"></span></div>
    <div class="load-tip" style="visibility:hidden">&nbsp;</div>
  </div>`;
  document.body.appendChild(root);

  // reveal the type once the web font is in (or after a short timeout offline), as the HUD does
  const ready = () => root.classList.add('ready');
  try {
    const fl = document.fonts?.load?.('500 40px "Barlow Condensed"');
    if (fl) Promise.race([fl, new Promise((r) => setTimeout(r, 1200))]).then(ready, ready); else ready();
  } catch { ready(); }

  const opts = [...root.querySelectorAll('.qg-seg button')];
  const blurb = root.querySelector('.qg-blurb');
  const paint = (focus) => {
    opts.forEach((b) => {
      const on = b.dataset.q === sel;
      b.setAttribute('aria-checked', String(on));
      b.tabIndex = on ? 0 : -1;
      if (on && focus) b.focus({ preventScroll: true });
    });
    blurb.textContent = INFO[sel].blurb;
  };
  paint(false);
  // keyboard users start on the selected option (touch: no focus ring, the selection itself is the highlight)
  if (!touchUI) setTimeout(() => { if (!root.classList.contains('leaving')) opts.find((b) => b.dataset.q === sel)?.focus({ preventScroll: true }); }, 60);

  return new Promise((resolve) => {
    let finished = false;
    const done = () => {
      if (finished) return;
      finished = true;
      window.removeEventListener('keydown', onKey, true);
      root.classList.add('leaving');
      root.querySelector('.qg-prep .load-q').textContent = `${INFO[sel].title} quality · ${mb(sel)} MB`;
      document.activeElement?.blur?.();
      handOver(root);
      resolve(sel);
    };
    const onKey = (e) => {
      const i = ORDER.indexOf(sel);
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { sel = ORDER[Math.min(3, i + 1)]; paint(true); e.preventDefault(); }
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { sel = ORDER[Math.max(0, i - 1)]; paint(true); e.preventDefault(); }
      else if (e.key === 'Home') { sel = ORDER[0]; paint(true); e.preventDefault(); }
      else if (e.key === 'End') { sel = ORDER[3]; paint(true); e.preventDefault(); }
      else if (e.key === 'Enter' || e.key === 'NumpadEnter') { e.preventDefault(); done(); }
      else if (e.key === ' ' && document.activeElement?.classList?.contains('qg-go')) { e.preventDefault(); done(); }
    };
    window.addEventListener('keydown', onKey, true);
    opts.forEach((b) => {
      b.addEventListener('click', () => { sel = b.dataset.q; paint(true); });
      b.addEventListener('dblclick', () => { sel = b.dataset.q; done(); });
    });
    root.querySelector('.qg-go').addEventListener('click', done);
  });
}

/** Keep the gate (now a "Preparing" screen) up until the HUD's own loading screen exists under it, then fade it out. */
function handOver(root) {
  const ui = document.getElementById('ui');
  let gone = false;
  const leave = () => {
    if (gone) return;
    gone = true;
    mo?.disconnect();
    clearTimeout(fallback);
    root.classList.add('gone');
    setTimeout(() => root.remove(), 700);
  };
  const check = () => { if (ui?.querySelector('.screen.loading')) setTimeout(leave, 60); };
  let mo = null;
  try { mo = new MutationObserver(check); if (ui) mo.observe(ui, { childList: true }); } catch { mo = null; }
  const fallback = setTimeout(leave, 45000);
  check();
  if (!ui) setTimeout(leave, 400);
}
