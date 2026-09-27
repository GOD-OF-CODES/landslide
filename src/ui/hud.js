// LANDSLIDE: all DOM UI (loading, title, HUD, pause, death, win). Owned by the UI workstream.
// API (DESIGN.md section 3, "hud"):
//   setPrompt(text|null, progress=null)  setObjective(text)  toast(text, seconds)  subtitle(text, seconds)
//   setDriving(bool)  setFuel(0..1)  setSpeed(kmh)  setInventory(list)  damageVignette(0..1)
//   fade(toBlack, seconds) -> Promise   showScreen('title'|'pause'|'dead'|'win'|null, data)
// Extras: letterbox(bool|null), hideHud(bool), pause(), resume(), settings (object), playTime (s).
// Emits: ui:start {auto}, ui:pause, ui:resume, ui:retry, ui:quit, ui:settings {settings}.
// Every method is safe to call at any time (before boot, with a missing ctx, with other systems absent).
import { ICONS, ITEM_NAMES } from './icons.js';
import { Cluster } from './gauges.js';

const SETTINGS_KEY = 'landslide.settings';
const DEFAULTS = { sensitivity: 1, invertY: false, volume: 0.8, subtitles: true };
const BASE_SENS = 0.0022;
const QUALITY_KEYS = ['low', 'medium', 'high', 'ultra'];

const TIPS = [
  'The debris flow does not stop. Every second you stand still, it gets closer.',
  'Roadworks crews leave their tools behind. Look for the site beside the road.',
  'Hold the key to use a tool. Let go and you start the job over.',
  'Wet asphalt has half the grip. Brake before the bend, not in it.',
  'Listen. Rockfall is heard before it is seen.',
  'Hold Shift to sprint. Stamina runs out fast on a climb.',
  'Press C in the car to switch between the cockpit and the chase camera.',
  'Headphones recommended.',
  'A 20 litre jerrycan is heavy. A few litres will still get an old engine running.',
];

const LOAD_LABELS = {
  env: 'Hanging the clouds', terrain: 'Carving the mountain', vegetation: 'Growing the forest',
  props: 'Setting up the roadworks', landslide: 'Loosening the slope', particles: 'Making it rain',
  car: 'Warming up the engine', player: 'Finding your feet', cameraRig: 'Mounting the camera',
  audio: 'Recording the rain', interact: 'Leaving tools behind', game: 'Writing the day',
  post: 'Developing the film',
};
const LOAD_ORDER = ['hud', 'env', 'terrain', 'vegetation', 'props', 'landslide', 'particles', 'car', 'player',
  'cameraRig', 'audio', 'interact', 'game', 'post'];

const CAUSES = {
  boulder: ['Crushed by rockfall', 'A boulder the size of a car came down the gully. There was no time to hear it.'],
  front: ['Buried by the landslide', 'The slope came down behind you and did not stop.'],
  fall: ['Fell into the ravine', 'The edge was softer than it looked.'],
  car: ['The car was crushed', 'The mountain caught up with the car.'],
  rollover: ['The car rolled over', 'A rock caught a wheel and the car went onto its roof. The slope did the rest.'],
  default: ['You did not make it', 'The mountain does not wait.'],
};

const EPILOGUE = `Behind you the mountain kept moving for another hour. By nightfall four hundred metres of road
had gone into the valley, along with the forest that had held it in place for a century.
The rescue crews reached the far portal at first light. You were still there, listening to the rain.`;

const CREDITS_HTML = `
  <p class="cred-lead">All 3D models, terrain and audio built from scratch.</p>
  <p>Textures &amp; HDRI: <a href="https://polyhaven.com" target="_blank" rel="noopener">Poly Haven</a> (CC0)</p>
  <p class="dim">Built with three.js, Rapier, postprocessing and N8AO. Models authored procedurally in Blender.</p>`;

const CONTROLS = [
  ['On foot', [['W A S D', 'Move'], ['Mouse', 'Look'], ['Shift', 'Sprint'], ['Space', 'Jump'], ['E', 'Interact / hold to use']]],
  ['Driving', [['W / S', 'Throttle / brake, reverse'], ['A / D', 'Steer'], ['Space', 'Handbrake'], ['C', 'Cockpit / chase camera'], ['E', 'Get in / out']]],
  ['General', [['Tab', 'Show objective'], ['Esc', 'Pause']]],
];

const clamp01 = (v) => Math.max(0, Math.min(1, v));
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function fmtTime(sec) {
  sec = Math.max(0, Math.round(+sec || 0));
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export default class Hud {
  constructor(ctx) {
    this.ctx = ctx || {};
    this.el = null;
    this.started = false;
    this.screen = null;
    this.playTime = 0;
    this.settings = this._loadSettings();
    this._driving = false; this._drivingSet = false;
    this._vig = 0; this._vigTarget = 0; this._vigShown = -1;
    this._invKey = ''; this._invExplicit = false;
    this._prompt = { text: null, key: null, prog: null };
    this._objective = ''; this._objTimer = 0;
    this._subTimer = 0;
    this._hudHidden = false; this._letterbox = null;
    this._wasLocked = false; this._selfUnlock = false;
    this._offs = [];
    try { this._build(); } catch (e) { console.error('[hud] build failed', e); this.el = null; }
    try { this._bind(); } catch (e) { console.error('[hud] bind failed', e); }
    this._applySettings();
  }

  // ------------------------------------------------------------------------------------------ build
  _build() {
    let root = document.getElementById('ui');
    if (!root) { root = el('div'); root.id = 'ui'; document.body.appendChild(root); }
    root.innerHTML = '';
    this.root = root;
    // reveal display type once the web font is in (or after a timeout offline), so nothing reflows
    const ready = () => root.classList.add('fonts-ready');
    try {
      const fl = document.fonts?.load?.('500 40px "Barlow Condensed"');
      if (fl) Promise.race([fl, new Promise((r) => setTimeout(r, 1500))]).then(ready, ready); else ready();
    } catch { ready(); }

    // --- in-game HUD layer
    const hud = this.hudEl = el('div', 'hud');
    hud.innerHTML = `
      <div class="vignette"></div>
      <div class="lb lb-top"></div><div class="lb lb-bot"></div>
      <div class="objective" aria-live="polite"><div class="obj-label">Objective</div><div class="obj-text"></div></div>
      <div class="toasts" aria-live="polite"></div>
      <div class="reticle"><div class="dot"></div>
        <svg class="stamina" viewBox="0 0 40 40"><circle cx="20" cy="20" r="16" class="st-bg"/><circle cx="20" cy="20" r="16" class="st-fg"/></svg>
      </div>
      <div class="prompt">
        <div class="key"><svg class="ring" viewBox="0 0 44 44"><circle cx="22" cy="22" r="19.5" class="r-bg"/><circle cx="22" cy="22" r="19.5" class="r-fg"/></svg><span class="key-cap">E</span></div>
        <div class="prompt-text"></div>
      </div>
      <div class="inventory"></div>
      <div class="subtitle" aria-live="polite"><span class="sub-speaker"></span><span class="sub-text"></span></div>`;
    this.cluster = new Cluster();
    hud.appendChild(this.cluster.el);
    root.appendChild(hud);
    const q = (s) => hud.querySelector(s);
    this.$ = {
      vignette: q('.vignette'), objective: q('.objective'), objText: q('.obj-text'), toasts: q('.toasts'),
      reticle: q('.reticle'), stamina: q('.stamina'), stFg: q('.st-fg'), prompt: q('.prompt'),
      keyCap: q('.key-cap'), ring: q('.ring'), ringFg: q('.r-fg'), promptText: q('.prompt-text'),
      inventory: q('.inventory'), subtitle: q('.subtitle'), subSpeaker: q('.sub-speaker'), subText: q('.sub-text'),
    };

    // --- fade (above HUD, below menus)
    this.fadeEl = el('div', 'fade');
    root.appendChild(this.fadeEl);

    // --- shared panels (moved between title and pause)
    this.panels = {
      settings: this._buildSettings(),
      controls: this._buildControls(),
      credits: this._buildCredits(),
    };
    // every panel ends with a real Back button (mouse users), Esc does the same
    for (const P of Object.values(this.panels)) {
      const b = el('button', 'p-back', '<kbd>Esc</kbd><span>Back</span>');
      b.type = 'button';
      b.addEventListener('click', (e) => { e.stopPropagation(); if (this._panelScreen) this._closePanel(this._panelScreen); });
      P.appendChild(b);
    }

    // --- screens
    this.screens = {
      title: this._buildTitle(),
      pause: this._buildPause(),
      dead: this._buildDead(),
      win: this._buildWin(),
    };
    for (const s of Object.values(this.screens)) root.appendChild(s);

    // --- loading (top)
    this.loadEl = this._buildLoading();
    root.appendChild(this.loadEl);
  }

  _buildLoading() {
    const L = el('div', 'screen loading on');
    L.innerHTML = `
      <div class="bd"><div class="bd-topo"></div>
        <div class="bd-ridge far"></div><div class="bd-fog f1"></div><div class="bd-ridge near"></div><div class="bd-fog f2"></div>
        <div class="bd-rain r1"></div><div class="bd-rain r2"></div><div class="bd-grain"></div></div>
      <div class="load-center">
        <div class="brand-mark">LANDSLIDE</div>
        <div class="load-kicker">A mountain road. Rain. No way back.</div>
      </div>
      <div class="load-foot">
        <div class="load-row"><span class="load-label">Preparing</span><span class="load-pct">0%</span></div>
        <div class="load-bar"><i></i></div>
        <div class="load-tip"></div>
      </div>`;
    this.$load = { bar: L.querySelector('.load-bar i'), pct: L.querySelector('.load-pct'), label: L.querySelector('.load-label'), tip: L.querySelector('.load-tip') };
    this._tipIdx = Math.floor(Math.random() * TIPS.length);
    this.$load.tip.textContent = TIPS[this._tipIdx];
    return L;
  }

  _menu(items) {
    const nav = el('nav', 'menu');
    for (const [id, label, fn, cls] of items) {
      const b = el('button', 'm-item' + (cls ? ' ' + cls : ''), `<span class="m-bar"></span><span class="m-label">${label}</span>`);
      b.type = 'button'; b.dataset.id = id;
      b.addEventListener('click', (e) => { e.stopPropagation(); try { fn(b); } catch (err) { console.error('[hud] menu', err); } });
      nav.appendChild(b);
    }
    return nav;
  }

  _buildTitle() {
    const S = el('div', 'screen title');
    S.innerHTML = `<div class="scrim-left"></div>
      <div class="title-col">
        <div class="t-kicker"><span>A survival short</span></div>
        <h1 class="brand-mark big">LANDSLIDE</h1>
        <p class="t-tag">Somewhere above the valley, the mountain has started to move.</p>
      </div>
      <div class="panel-host"></div>
      <div class="t-foot"><span>Headphones recommended</span><span class="sep"></span><span class="q-note"></span></div>`;
    const col = S.querySelector('.title-col');
    col.appendChild(this._menu([
      ['start', 'Start', () => this._startClicked(), 'primary'],
      ['settings', 'Settings', (b) => this._togglePanel(S, 'settings', b)],
      ['controls', 'Controls', (b) => this._togglePanel(S, 'controls', b)],
      ['credits', 'Credits', (b) => this._togglePanel(S, 'credits', b)],
    ]));
    S.querySelector('.q-note').textContent = `Quality: ${this.ctx.config?.quality?.name ?? 'High'}`;
    return S;
  }

  _buildPause() {
    const S = el('div', 'screen pause');
    S.innerHTML = `<div class="scrim-full"></div>
      <div class="pause-col">
        <div class="p-label">Paused</div>
        <div class="p-obj"><div class="obj-label">Objective</div><div class="p-obj-text"></div></div>
      </div>
      <div class="panel-host"></div>
      <div class="p-foot"><span class="p-time"></span></div>`;
    S.querySelector('.pause-col').appendChild(this._menu([
      ['resume', 'Resume', () => this.resume(), 'primary'],
      ['retry', 'Retry from checkpoint', () => this._retry()],
      ['settings', 'Settings', (b) => this._togglePanel(S, 'settings', b)],
      ['controls', 'Controls', (b) => this._togglePanel(S, 'controls', b)],
      ['quit', 'Quit to title', () => this._quit()],
    ]));
    return S;
  }

  _buildDead() {
    const S = el('div', 'screen dead');
    S.innerHTML = `<div class="scrim-dead"></div>
      <div class="end-col">
        <div class="e-kicker">Signal lost</div>
        <h2 class="e-title"></h2>
        <p class="e-text"></p>
      </div>`;
    S.querySelector('.end-col').appendChild(this._menu([
      ['retry', 'Retry from checkpoint', () => this._retry(), 'primary'],
      ['quit', 'Quit to title', () => this._quit()],
    ]));
    return S;
  }

  _buildWin() {
    const S = el('div', 'screen win');
    S.innerHTML = `<div class="scrim-win"></div>
      <div class="end-col">
        <div class="e-kicker">Escaped</div>
        <div class="w-time"><span class="w-num">00:00</span><span class="w-cap">on the mountain</span></div>
        <p class="w-epi"></p>
        <div class="w-cred">${CREDITS_HTML}</div>
      </div>`;
    S.querySelector('.w-epi').textContent = EPILOGUE.replace(/\s*\n\s*/g, ' ');
    S.querySelector('.end-col').appendChild(this._menu([
      ['again', 'Return to title', () => this._quit(), 'primary'],
    ]));
    return S;
  }

  _buildSettings() {
    const P = el('section', 'panel settings');
    const cur = this.ctx.config?.qualityKey ?? 'high';
    P.innerHTML = `<h3>Settings</h3>
      <div class="row"><label>Graphics quality</label>
        <div class="seg" role="radiogroup">${QUALITY_KEYS.map((k) => `<button type="button" role="radio" data-q="${k}" aria-checked="${k === cur}">${k[0].toUpperCase() + k.slice(1)}</button>`).join('')}</div>
      </div>
      <div class="apply-row"><span class="apply-note"></span><button type="button" class="btn-apply">Apply and reload</button></div>
      <div class="row"><label for="ls-sens">Mouse sensitivity</label><div class="rng"><input id="ls-sens" type="range" min="0.25" max="3" step="0.05"><output class="val-sens"></output></div></div>
      <div class="row"><label>Invert vertical look</label><button type="button" class="tog" role="switch" data-k="invertY"><i></i></button></div>
      <div class="row"><label for="ls-vol">Master volume</label><div class="rng"><input id="ls-vol" type="range" min="0" max="100" step="1"><output class="val-vol"></output></div></div>
      <div class="row"><label>Subtitles</label><button type="button" class="tog" role="switch" data-k="subtitles"><i></i></button></div>`;
    let pending = null;
    const applyRow = P.querySelector('.apply-row'), note = P.querySelector('.apply-note');
    P.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
      pending = b.dataset.q === cur ? null : b.dataset.q;
      P.querySelectorAll('.seg button').forEach((o) => o.setAttribute('aria-checked', String(o === b)));
      applyRow.classList.toggle('on', !!pending);
      note.textContent = this.started ? 'Reloading restarts the game.' : 'The game reloads to apply this.';
    }));
    P.querySelector('.btn-apply').addEventListener('click', () => {
      if (!pending) return;
      try { this.ctx.config?.setQuality?.(pending); } catch {}
      try { localStorage.setItem('landslide.quality', pending); } catch {}
      const u = new URL(location.href); u.searchParams.delete('quality'); u.searchParams.delete('autostart');
      location.href = u.toString();
    });
    const sens = P.querySelector('#ls-sens'), vol = P.querySelector('#ls-vol');
    const sync = () => {
      sens.value = this.settings.sensitivity; P.querySelector('.val-sens').textContent = (+this.settings.sensitivity).toFixed(2) + '×';
      vol.value = Math.round(this.settings.volume * 100); P.querySelector('.val-vol').textContent = Math.round(this.settings.volume * 100);
      P.querySelectorAll('.tog').forEach((t) => t.setAttribute('aria-checked', String(!!this.settings[t.dataset.k])));
      for (const r of [sens, vol]) r.style.setProperty('--p', ((r.value - r.min) / (r.max - r.min)) * 100 + '%');
    };
    this._syncSettingsUI = sync;
    sens.addEventListener('input', () => this._setSetting('sensitivity', +sens.value));
    vol.addEventListener('input', () => this._setSetting('volume', +vol.value / 100));
    P.querySelectorAll('.tog').forEach((t) => t.addEventListener('click', () => this._setSetting(t.dataset.k, !this.settings[t.dataset.k])));
    sync();
    return P;
  }

  _buildControls() {
    const P = el('section', 'panel controls');
    P.innerHTML = '<h3>Controls</h3>' + CONTROLS.map(([grp, rows]) =>
      `<div class="c-grp"><div class="c-head">${grp}</div>${rows.map(([k, v]) =>
        `<div class="c-row"><span class="keys">${k.split(' / ').map((x) => x.split(' ').map((kk) => `<kbd>${esc(kk)}</kbd>`).join('')).join('<em>/</em>')}</span><span class="c-act">${esc(v)}</span></div>`).join('')}</div>`).join('');
    return P;
  }

  _buildCredits() {
    const P = el('section', 'panel credits');
    P.innerHTML = `<h3>Credits</h3>${CREDITS_HTML}`;
    return P;
  }

  // ------------------------------------------------------------------------------------------ events
  _bind() {
    const ev = this.ctx.events;
    const on = (n, f) => { if (ev?.on) this._offs.push(ev.on(n, f)); };

    // loading progress
    const only = this.ctx.flags?.only;
    this._sysTotal = only ? Math.max(1, only.filter((k) => k !== 'hud').length) : LOAD_ORDER.length - 1;
    this._sysDone = 0; this._loadShown = 0; this._booted = false;
    on('boot:progress', (p) => {
      if (p?.key === 'hud') return;
      this._sysDone++;
      const idx = LOAD_ORDER.indexOf(p?.key);
      const next = LOAD_ORDER.slice(idx + 1).find((k) => !only || only.includes(k));
      this._loadLabel = this._sysDone >= this._sysTotal || !next ? 'Compiling shaders' : (LOAD_LABELS[next] || 'Loading');
    });
    on('boot:done', () => this._onBootDone());
    on('ui:start', (p) => this._onStart(p));
    on('objective', (p) => { if (p?.text) this.setObjective(p.text); });
    on('dead', (p) => this._autoEnd('dead', p, 2200));
    on('win', (p) => this._autoEnd('win', p, 2600));
    this._loadLabel = LOAD_LABELS.env;
    this._loadStart = performance.now();
    this._loadTick = this._loadTick.bind(this);
    requestAnimationFrame(this._loadTick);
    this._tipTimer = setInterval(() => this._nextTip(), 6500);

    // keyboard (menus + pause); registered in capture so we see keys before gameplay listeners
    this._onKey = (e) => this._key(e);
    window.addEventListener('keydown', this._onKey, true);
    this._onKeyUp = (e) => { if (e.code === 'Tab') this._tabHeld = false; };
    window.addEventListener('keyup', this._onKeyUp, true);
    // pointer lock loss => pause
    this._onLock = () => {
      const locked = this._isLocked();
      if (this._wasLocked && !locked && !this._selfUnlock && this.started && !this.screen) this.pause();
      this._selfUnlock = false;
      this._wasLocked = locked;
    };
    document.addEventListener('pointerlockchange', this._onLock);
    // click on the canvas while playing re-acquires pointer lock (Chrome refuses right after an exit)
    this._onCanvasClick = () => { if (this.started && !this.screen && !this._isLocked()) this.ctx.input?.requestLock?.(); };
    this.ctx.renderer?.domElement?.addEventListener('click', this._onCanvasClick);
  }

  _isLocked() { try { return !!this.ctx.input?.locked; } catch { return false; } }

  _loadTick(now) {
    if (!this.loadEl || this._loadGone) return;
    const pr = this.ctx.assets?.progress;
    const assetFrac = pr && pr.total > 0 ? pr.loaded / pr.total : 0;
    const sys = this._sysDone / this._sysTotal;
    const target = this._booted ? 1 : Math.min(0.97, 0.04 + 0.66 * sys + 0.27 * assetFrac);
    const t = Math.max(this._loadShown, target);
    // ease toward target so the bar never jumps
    this._loadShown += (t - this._loadShown) * (this._booted ? 0.2 : 0.08);
    if (this._booted && this._loadShown > 0.995) this._loadShown = 1;
    const pct = Math.round(this._loadShown * 100);
    this.$load.bar.style.transform = `scaleX(${this._loadShown.toFixed(4)})`;
    if (this.$load.pct.textContent !== pct + '%') this.$load.pct.textContent = pct + '%';
    const label = this._booted ? 'Ready' : this._loadLabel;
    if (this.$load.label.textContent !== label) this.$load.label.textContent = label;
    requestAnimationFrame(this._loadTick);
  }

  _nextTip() {
    if (this._loadGone || !this.$load) { clearInterval(this._tipTimer); return; }
    const tip = this.$load.tip;
    tip.classList.add('out');
    setTimeout(() => { this._tipIdx = (this._tipIdx + 1) % TIPS.length; tip.textContent = TIPS[this._tipIdx]; tip.classList.remove('out'); }, 450);
  }

  _onBootDone() {
    this._booted = true;
    this._applySettings(); // audio exists now
    const auto = !!this.ctx.flags?.autostart;
    setTimeout(() => {
      if (!this.started && !auto) this.showScreen('title');
      this._hideLoading();
    }, auto ? 150 : 650);
  }

  _hideLoading() {
    if (!this.loadEl || this._loadGone) return;
    this.loadEl.classList.remove('on');
    this.loadEl.classList.add('leaving');
    clearInterval(this._tipTimer);
    setTimeout(() => { this._loadGone = true; this.loadEl?.remove(); }, 1400);
  }

  _startClicked() {
    try { this.ctx.audio?.unlock?.(); } catch (e) { console.warn('[hud] audio unlock', e); }
    try { this.ctx.input?.requestLock?.(); } catch {}
    this._wasLocked = false;
    this.ctx.events?.emit?.('ui:start', { auto: false });
    if (!this.ctx.events?.emit) this._onStart({ auto: false });
  }

  _onStart() {
    if (this.started) return;
    this.started = true;
    this.playTime = 0;
    this._hideLoading();
    this.showScreen(null);
    this._applySettings();
    if (this._objective) this._showObjective();
  }

  _autoEnd(name, data, delay) {
    clearTimeout(this._autoEndT);
    this._autoEndT = setTimeout(() => {
      if (this.screen === name) return;
      const st = this.ctx.game?.state;
      if (st && st !== name) return; // game already moved on (retry)
      this.showScreen(name, data);
    }, delay);
  }

  _key(e) {
    const code = e.code;
    const menu = this.screen ? this.screens[this.screen] : null;
    if (code === 'Escape') {
      if (this.screen === 'pause') {
        if (this._openPanel) this._closePanel(this.screens.pause); else this.resume();
        e.preventDefault();
      } else if (this.screen === 'title' && this._openPanel) {
        this._closePanel(this.screens.title); e.preventDefault();
      } else if (!this.screen && this.started) {
        this.pause(); e.preventDefault();
      }
      return;
    }
    if (menu && !this._loadGoneBlocking()) {
      if (['Tab', 'ArrowDown', 'ArrowUp'].includes(code)) {
        e.preventDefault();
        this._moveFocus(menu, code === 'ArrowUp' || (code === 'Tab' && e.shiftKey) ? -1 : 1);
      } else if (code === 'Space' || code === 'Enter' || code === 'NumpadEnter') {
        const a = document.activeElement;
        if (a && menu.contains(a) && (a.tagName === 'BUTTON')) { e.preventDefault(); a.click(); }
      }
      return;
    }
    if (code === 'Tab' && this.started && !this.screen && !this._tabHeld) {
      this._tabHeld = true;
      if (this._objective) this._showObjective();
    }
  }
  _loadGoneBlocking() { return this.loadEl && !this._loadGone && this.loadEl.classList.contains('on'); }

  _focusables(root) {
    return [...root.querySelectorAll('button, input, a[href]')].filter((n) => n.offsetParent !== null && !n.disabled);
  }
  _moveFocus(root, dir) {
    const list = this._focusables(root);
    if (!list.length) return;
    const i = list.indexOf(document.activeElement);
    const n = list[(i < 0 ? (dir > 0 ? 0 : list.length - 1) : (i + dir + list.length) % list.length)];
    n.focus({ preventScroll: true });
  }

  _togglePanel(screen, id, btn) {
    if (this._openPanel === id && this._panelScreen === screen) { this._closePanel(screen); return; }
    const host = screen.querySelector('.panel-host');
    const P = this.panels[id];
    if (!host || !P) return;
    host.innerHTML = '';
    host.appendChild(P);
    this._syncSettingsUI?.();
    screen.querySelectorAll('.m-item').forEach((b) => b.classList.toggle('active', b === btn));
    host.classList.remove('on'); void host.offsetWidth; host.classList.add('on');
    this._openPanel = id; this._panelScreen = screen;
    screen.classList.add('panel-open');
    // keyboard users land inside the panel; the opener keeps its active marker
    const first = P.querySelector('.seg [aria-checked="true"]') || P.querySelector('button:not(.p-back), input, a[href]');
    if (first) setTimeout(() => { if (this._openPanel === id) first.focus({ preventScroll: true }); }, 30);
    else btn?.focus?.({ preventScroll: true });
  }
  _closePanel(screen) {
    screen?.classList.remove('panel-open');
    const host = screen?.querySelector('.panel-host');
    host?.classList.remove('on');
    screen?.querySelectorAll('.m-item').forEach((b) => b.classList.remove('active'));
    const btn = screen?.querySelector(`.m-item[data-id="${this._openPanel}"]`);
    this._openPanel = null; this._panelScreen = null;
    btn?.focus({ preventScroll: true });
  }

  _retry() {
    const ev = this.ctx.events;
    const listeners = ev?.map?.get?.('ui:retry')?.size ?? 0;
    this.showScreen(null);
    try { this.ctx.input?.requestLock?.(); } catch {}
    if (listeners > 0) ev.emit('ui:retry', {});
    else if (this.ctx.game?.restartFromCheckpoint) { ev?.emit?.('ui:retry', {}); this.ctx.game.restartFromCheckpoint(); }
    else location.reload();
  }

  _quit() {
    this.ctx.events?.emit?.('ui:quit', {});
    const u = new URL(location.href);
    for (const k of ['autostart', 'skip', 'cam', 'camS', 'time']) u.searchParams.delete(k);
    this.fade(true, 0.5).then(() => { location.href = u.toString(); });
  }

  // ------------------------------------------------------------------------------------------ settings
  _loadSettings() {
    try { return { ...DEFAULTS, ...(JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') || {}) }; } catch { return { ...DEFAULTS }; }
  }
  _setSetting(k, v) {
    this.settings[k] = v;
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(this.settings)); } catch {}
    this._applySettings();
    this._syncSettingsUI?.();
    this.ctx.events?.emit?.('ui:settings', { settings: { ...this.settings } });
  }
  _applySettings() {
    const s = this.settings, inp = this.ctx.input;
    if (inp) { inp.sensitivity = BASE_SENS * (+s.sensitivity || 1); inp.invertY = !!s.invertY; }
    try { this.ctx.audio?.setMasterVolume?.(s.volume); } catch {}
    if (!s.subtitles) this.$?.subtitle?.classList.remove('on');
  }

  // ------------------------------------------------------------------------------------------ pause
  pause() {
    if (!this.started || this.screen) return;
    this.showScreen('pause');
  }
  resume() {
    if (this.screen !== 'pause') return;
    this.showScreen(null);
    try { this.ctx.input?.requestLock?.(); } catch {}
  }

  // ------------------------------------------------------------------------------------------ public API
  showScreen(name, data = {}) {
    if (!this.root || !this.screens) return;
    name = name && this.screens?.[name] ? name : null;
    const prev = this.screen;
    data = data || {};
    if (this._openPanel && this._panelScreen) this._closePanel(this._panelScreen);
    for (const [k, s] of Object.entries(this.screens)) s.classList.toggle('on', k === name);
    this.screen = name;
    const ctx = this.ctx;

    if (name === 'dead') {
      const [title, text] = CAUSES[data.cause] || (data.cause ? [String(data.cause), CAUSES.default[1]] : CAUSES.default);
      const S = this.screens.dead;
      S.querySelector('.e-title').textContent = data.title || title;
      S.querySelector('.e-text').textContent = data.text || text;
    } else if (name === 'win') {
      this.screens.win.querySelector('.w-num').textContent = fmtTime(data.time ?? this.playTime);
      if (data.epilogue) this.screens.win.querySelector('.w-epi').textContent = data.epilogue;
    } else if (name === 'pause') {
      const S = this.screens.pause;
      S.querySelector('.p-obj-text').textContent = this._objective || '';
      S.querySelector('.p-obj').style.display = this._objective ? '' : 'none';
      S.querySelector('.p-time').textContent = this.started ? `Time ${fmtTime(this.playTime)}` : '';
      const st = ctx.game?.state;
      S.querySelector('.m-item[data-id="retry"]').style.display = st && ['title', 'intro'].includes(st) ? 'none' : '';
    } else if (name === 'title') {
      this.started = false;
    }

    // pause/input state
    const menuOpen = !!name;
    if (name === 'pause') { ctx.paused = true; if (prev !== 'pause') ctx.events?.emit?.('ui:pause', {}); }
    else if (prev === 'pause') { ctx.paused = false; ctx.events?.emit?.('ui:resume', {}); }
    if (ctx.input) ctx.input.enabled = !menuOpen;
    if (menuOpen && name !== 'title' && this._isLocked()) { this._selfUnlock = true; try { ctx.input.exitLock(); } catch {} }
    this.root?.classList.toggle('menu-open', menuOpen);

    if (name) {
      const first = this.screens[name].querySelector('.m-item.primary') || this.screens[name].querySelector('button');
      setTimeout(() => { if (this.screen === name) first?.focus({ preventScroll: true }); }, name === 'title' ? 900 : 80);
    } else if (document.activeElement && this.root?.contains(document.activeElement)) {
      document.activeElement.blur();
    }
    this._refreshVisibility();
  }

  setPrompt(text, progress = null) {
    const $ = this.$; if (!$) return;
    if (!text) {
      if (this._prompt.text !== null) { this._prompt.text = null; $.prompt.classList.remove('on'); this.hudEl?.classList.remove('prompting'); }
      return;
    }
    text = String(text);
    if (text !== this._prompt.text) {
      this._prompt.text = text;
      let key = 'E', label = text;
      const m = /\[([^\]]{1,8})\]/.exec(text);
      if (m) { key = m[1]; label = (text.slice(0, m.index) + ' ' + text.slice(m.index + m[0].length)).replace(/\s+/g, ' ').trim(); }
      $.keyCap.textContent = key;
      $.keyCap.classList.toggle('wide', key.length > 2);
      $.promptText.textContent = label;
      $.prompt.classList.add('on');
      this.hudEl?.classList.add('prompting');
    }
    const hasProg = progress != null && isFinite(progress);
    const p = hasProg ? clamp01(+progress) : null;
    if (p !== this._prompt.prog) {
      this._prompt.prog = p;
      $.prompt.classList.toggle('holding', hasProg);
      $.prompt.classList.toggle('done', hasProg && p >= 1);
      const C = 2 * Math.PI * 19.5;
      $.ringFg.style.strokeDasharray = `${C}`;
      $.ringFg.style.strokeDashoffset = `${(C * (1 - (p ?? 0))).toFixed(2)}`;
    }
  }

  setObjective(text) {
    if (!this.$) return;
    text = text ? String(text) : '';
    if (text === this._objective) { if (text) this._showObjective(); return; }
    this._objective = text;
    const o = this.$.objective;
    if (!text) { o.classList.remove('on'); return; }
    o.classList.remove('on', 'new'); void o.offsetWidth;
    this.$.objText.textContent = text;
    o.classList.add('new');
    this._showObjective();
  }
  _showObjective() {
    const o = this.$?.objective; if (!o || !this._objective) return;
    o.classList.add('on');
    clearTimeout(this._objTimer);
    this._objTimer = setTimeout(() => o.classList.remove('on', 'new'), 9000);
  }

  toast(text, seconds = 3.5) {
    const box = this.$?.toasts; if (!box || !text) return;
    const t = el('div', 'toast');
    // "Picked up: Hatchet" -> small-caps kicker + body
    const m = /^([^:]{2,22}):\s+(.+)$/s.exec(String(text));
    if (m) t.innerHTML = `<span class="t-k">${esc(m[1])}</span><span class="t-b">${esc(m[2])}</span>`;
    else t.textContent = String(text);
    box.appendChild(t);
    while (box.children.length > 3) box.firstChild.remove();
    requestAnimationFrame(() => t.classList.add('on'));
    setTimeout(() => { t.classList.remove('on'); t.classList.add('off'); setTimeout(() => t.remove(), 600); }, Math.max(0.8, +seconds || 3.5) * 1000);
  }

  subtitle(text, seconds) {
    const $ = this.$; if (!$) return;
    clearTimeout(this._subTimer);
    if (!text) { $.subtitle.classList.remove('on'); return; }
    text = String(text);
    let speaker = '', body = text;
    const m = /^([A-Z][\w .'-]{0,24}):\s+(.+)$/s.exec(text);
    if (m) { speaker = m[1]; body = m[2]; }
    $.subSpeaker.textContent = speaker;
    $.subSpeaker.style.display = speaker ? '' : 'none';
    $.subText.textContent = body;
    if (this.settings.subtitles !== false) {
      $.subtitle.classList.remove('on'); void $.subtitle.offsetWidth; $.subtitle.classList.add('on');
    }
    const dur = seconds ?? Math.min(9, 1.8 + body.length * 0.055);
    this._subTimer = setTimeout(() => $.subtitle.classList.remove('on'), dur * 1000);
  }

  setDriving(b) { this._driving = !!b; this._drivingSet = true; this._refreshVisibility(); }
  setFuel(f) { try { this.cluster?.setFuel(f); } catch {} }
  setSpeed(kmh) { try { this.cluster?.setSpeed(kmh); } catch {} }

  // setInventory(ids, details?) — details (optional) = [{id, label, short, icon}] as sent by game/inventory.js
  setInventory(list, details = null) {
    this._invExplicit = true;
    this._setInv(list, details);
  }
  _setInv(list, details) {
    const box = this.$?.inventory; if (!box) return;
    let arr = [];
    try { arr = list ? [...list].map((x) => (typeof x === 'object' && x ? x.id : x)).filter((x) => x && (ICONS[x] || ITEM_NAMES[x])) : []; } catch { arr = []; }
    const info = {};
    if (Array.isArray(details)) for (const d of details) if (d?.id) info[d.id] = d;
    const key = arr.join(',');
    if (key === this._invKey) return;
    const prev = new Set(this._invKey ? this._invKey.split(',') : []);
    this._invKey = key;
    box.innerHTML = '';
    for (const id of arr) {
      const s = el('div', 'slot' + (prev.has(id) ? '' : ' fresh'), `<div class="ico">${ICONS[id] || ''}</div><div class="nm">${esc(info[id]?.short || ITEM_NAMES[id] || id)}</div>`);
      s.title = info[id]?.label || ITEM_NAMES[id] || id;
      box.appendChild(s);
    }
    box.classList.toggle('on', arr.length > 0);
  }

  damageVignette(v) { this._vigTarget = clamp01(+v || 0); }

  fade(toBlack = true, seconds = 1) {
    const f = this.fadeEl;
    if (!f) return Promise.resolve();
    const s = Math.max(0, +seconds || 0);
    f.style.transitionDuration = s + 's';
    void f.offsetWidth;
    f.classList.toggle('on', !!toBlack);
    return new Promise((res) => setTimeout(res, s * 1000 + 20));
  }

  letterbox(on) { this._letterbox = on == null ? null : !!on; this._refreshVisibility(); }
  hideHud(b) { this._hudHidden = !!b; this._refreshVisibility(); }

  // ------------------------------------------------------------------------------------------ per frame
  _refreshVisibility() {
    const hud = this.hudEl; if (!hud) return;
    const ctx = this.ctx;
    // "Cinematic" only hides the HUD during a real cutscene: a running camera path, or a held cinematic view while
    // the game has taken control away (control 'none' with a live game state). The rig starts in 'cinematic' mode,
    // so keying on mode alone would hide the HUD forever whenever the game sequence is missing.
    const rig = ctx.cameraRig;
    const cinRunning = !!(rig?._cin && !rig._cin.done);
    const cine = cinRunning || (rig?.mode === 'cinematic' && ctx.control === 'none' && !!ctx.game?.state
      && !['title', 'win', 'dead'].includes(ctx.game.state));
    const playing = this.started && !this.screen;
    const show = playing && !this._hudHidden && !cine;
    const driving = this._drivingSet ? this._driving : ctx.control === 'car';
    const lb = this._letterbox ?? (this.started && cine && !this.screen);
    hud.classList.toggle('show', show);
    hud.classList.toggle('driving', show && driving);
    hud.classList.toggle('cockpit', show && driving && rig?.mode === 'car-cockpit');
    hud.classList.toggle('foot', show && !driving && ctx.control === 'foot');
    hud.classList.toggle('letterbox', !!lb);
    hud.classList.toggle('subs', this.started && this.screen !== 'title');
  }

  update(dt) {
    if (!this.hudEl) return;
    const now = performance.now();
    const ctx = this.ctx;
    if (this.started && !this.screen) {
      const st = ctx.game?.state;
      if (st !== 'win' && st !== 'dead') this.playTime += dt || 0;
    }
    this._refreshVisibility();
    this.cluster?.update(now);

    // inventory sync from game (unless game drives it explicitly)
    if (!this._invExplicit && ctx.game?.inventory) {
      const inv = ctx.game.inventory;
      const order = ['jerrycan', 'hatchet', 'planks'].filter((k) => inv.has?.(k));
      if (order.join(',') !== this._invKey) this._setInv(order, ctx.game.inventory.describe?.());
    }

    // damage vignette (smooth, with a heartbeat pulse at high values)
    const rawDt = Math.min(0.1, (now - (this._lastNow || now)) / 1000); this._lastNow = now;
    this._vig += (this._vigTarget - this._vig) * Math.min(1, rawDt * 6);
    let v = this._vig;
    if (v > 0.45) v *= 0.88 + 0.12 * Math.pow(Math.max(0, Math.sin(now * 0.0072)), 6);
    if (Math.abs(v - this._vigShown) > 0.004) { this._vigShown = v; this.$.vignette.style.opacity = v.toFixed(3); }

    // stamina ring around the dot
    const p = ctx.player;
    const stam = ctx.control === 'foot' && p ? clamp01(p.stamina ?? 1) : 1;
    const showSt = stam < 0.985;
    if (showSt !== this._stShown) { this._stShown = showSt; this.$.stamina.classList.toggle('on', showSt); }
    if (showSt) {
      const C = 2 * Math.PI * 16, key = Math.round(stam * 200);
      if (key !== this._stKey) {
        this._stKey = key;
        this.$.stFg.style.strokeDasharray = `${C}`;
        this.$.stFg.style.strokeDashoffset = (C * (1 - stam)).toFixed(2);
        this.$.stamina.classList.toggle('low', !!p?.exhausted || stam < 0.2);
      }
    }
  }

  dispose() {
    for (const off of this._offs) try { off(); } catch {}
    window.removeEventListener('keydown', this._onKey, true);
    window.removeEventListener('keyup', this._onKeyUp, true);
    document.removeEventListener('pointerlockchange', this._onLock);
    this.ctx.renderer?.domElement?.removeEventListener('click', this._onCanvasClick);
    clearInterval(this._tipTimer); clearTimeout(this._objTimer); clearTimeout(this._subTimer); clearTimeout(this._autoEndT);
    this._loadGone = true;
    if (this.root) this.root.innerHTML = '';
    this.hudEl = null; this.$ = null;
  }
}
