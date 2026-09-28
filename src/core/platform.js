// CrazyGames SDK v3 integration + iframe input guards.
//
// The SDK script is loaded by index.html. Its `environment` is 'crazygames' on CrazyGames, 'local' on
// localhost (demo ads), and 'disabled' everywhere else (e.g. the standalone Vercel build), where every SDK
// method throws. So every call below goes through `call()`, which checks the environment and swallows errors.
//
// What the game reports:
//   loadingStart/loadingStop  around the loading screen (main.js boot -> 'boot:done')
//   gameplayStart/Stop        whenever active play begins or pauses/ends (title menu, pause menu, death, win, ads)
//   happytime                 on escaping the landslide (the one big achievement)
//   midgame ads               only at natural breaks: retry after death, quit to title. Never during play.
// A "break" starts the moment Retry/Quit is pressed: the game is held (paused, input off, pointer released,
// faded to black, menus/pause ignored) until the ad has finished or failed; only then does the restart/quit run.
// Audio is suspended while an ad plays and while the platform's `muteAudio` setting is on.
// Rewarded ads: not used. The game is a short linear run with generous checkpoints, so there is nothing
// to grant in exchange that would not either break the pacing or feel like paying to skip the challenge.

import { PARAMS } from './config.js';

const sdk = () => window.CrazyGames?.SDK;
let initPromise = null;

export const platform = {
  env: 'none',              // 'none' (script missing) | 'disabled' | 'local' | 'crazygames'
  ready: Promise.resolve(false),
  initDone: false,
  adActive: false,          // an ad is on screen (adStarted .. adFinished/adError)
  busy: false,              // a break is in progress (Retry/Quit pressed .. restart/quit proceeds)
  playing: false,           // desired gameplay state (what the game is doing)
  ctx: null,
  _sentPlaying: false,      // gameplay state the SDK was last told
  _loading: false, _loadingSent: false,
  _muteReasons: new Set(),
  _adPromise: null,

  get enabled() { return this.env === 'local' || this.env === 'crazygames'; },

  /** Ads are skipped in automated-test URLs (the headless test harness uses them) unless ?adtest is given. */
  get adsAllowed() {
    if (!this.enabled) return false;
    if (PARAMS.has('adtest')) return true;
    return !(PARAMS.has('autostart') || PARAMS.has('cam') || PARAMS.has('camS') || PARAMS.has('skip'));
  },

  /** Starts SDK initialisation (idempotent). Resolves true when the SDK is usable. Never rejects. */
  init() {
    if (initPromise) return initPromise;
    initPromise = (async () => {
      const S = sdk();
      if (!S?.init) { this.env = 'none'; this.initDone = true; return false; }
      try {
        await S.init();
        this.env = S.environment || 'disabled';
      } catch (e) {
        this.env = S.environment || 'disabled';
        if (this.env !== 'disabled') console.warn('[platform] CrazyGames SDK init failed', e);
      }
      this.initDone = true;
      if (this.enabled) {
        const apply = (s) => this._setMute('platform', !!s?.muteAudio);
        this.call((S2) => { apply(S2.game.settings); S2.game.addSettingsChangeListener(apply); });
        // replay state that changed while init was still running (e.g. a slow parent handshake)
        if (this._loading) this._sendLoading(true);
        this._syncPlaying();
      }
      return this.enabled;
    })();
    this.ready = initPromise;
    return initPromise;
  },

  /** Runs fn(SDK) only when the SDK is initialised and enabled; never throws. */
  call(fn) {
    const S = sdk();
    if (!this.initDone || !S || !S.environment || S.environment === 'disabled' || !this.enabled) return undefined;
    try { return fn(S); } catch (e) { console.warn('[platform] SDK call failed', e); return undefined; }
  },

  // ---- loading: loadingStop is only sent if loadingStart went out (events always pair up)
  loadingStart() { this._loading = true; this._sendLoading(true); },
  loadingStop() { this._loading = false; if (this._loadingSent) this._sendLoading(false); },
  _sendLoading(on) {
    if (!this.enabled || !this.initDone) return;
    if (on && !this._loadingSent) { this.call((S) => S.game.loadingStart()); this._loadingSent = true; }
    else if (!on && this._loadingSent) { this.call((S) => S.game.loadingStop()); this._loadingSent = false; }
  },

  happytime() { this.call((S) => S.game.happytime()); },

  // ---- gameplay: only transitions reach the SDK; replayed after a late init
  setPlaying(on) { this.playing = !!on; this._syncPlaying(); },
  _syncPlaying() {
    if (!this.enabled || !this.initDone || this.playing === this._sentPlaying) return;
    this._sentPlaying = this.playing;
    const on = this.playing;
    this.call((S) => (on ? S.game.gameplayStart() : S.game.gameplayStop()));
  },

  /** Per-frame: derive "actively playing" from the game state. Cheap; only transitions reach the SDK. */
  update(ctx) {
    const st = ctx.game?.state;
    const active = st === 'intro' || st === 'stalled' || st === 'onfoot' || st === 'driving' || st === 'escape';
    this.setPlaying(!!active && !ctx.paused && !this.adActive && !this.busy);
  },

  /**
   * Midgame ad. Resolves when the ad finished, failed, or was skipped (never rejects). While a request is in
   * flight, further calls return the same promise (no duplicate requests).
   */
  midgame() {
    if (!this.adsAllowed) return Promise.resolve(false);
    if (this._adPromise) return this._adPromise;
    this._adPromise = new Promise((resolve) => {
      let resolved = false, showing = false;
      const settle = (ok) => { if (!resolved) { resolved = true; clearTimeout(guard); this._adPromise = null; resolve(ok); } };
      const end = (ok) => { if (showing) { showing = false; this._adEnd(); } settle(ok); };
      // The SDK always answers (adError covers unfilled/cooldown/adblock); this guard only protects against a
      // hung ad server. A late adStarted is still handled safely (game paused + audio muted until it ends).
      const guard = setTimeout(() => settle(false), 30000);
      try {
        sdk().ad.requestAd('midgame', {
          adStarted: () => { showing = true; clearTimeout(guard); this._adStart(); },
          adFinished: () => end(true),
          adError: (err) => {
            if (err?.code && err.code !== 'adCooldown' && err.code !== 'unfilled') console.info('[platform] ad', err.code, err.message || '');
            end(false);
          },
        });
      } catch (e) { console.warn('[platform] requestAd failed', e); end(false); }
    });
    return this._adPromise;
  },

  _adStart() {
    const ctx = this.ctx;
    this.adActive = true;
    this._pausedBefore = !!ctx?.paused;
    if (ctx) ctx.paused = true;
    this._releasePointer();
    this.setPlaying(false);
    this._setMute('ad', true);
  },
  _adEnd() {
    const ctx = this.ctx;
    this.adActive = false;
    // during a break the break owns the paused state; otherwise respect an open HUD menu
    if (ctx && !this.busy) ctx.paused = !!ctx.hud?.screen || this._pausedBefore;
    this._setMute('ad', false);
  },
  _releasePointer() {
    const ctx = this.ctx;
    try {
      if (ctx?.input?.locked) { if (ctx.hud) ctx.hud._selfUnlock = true; ctx.input.exitLock(); }
    } catch {}
  },

  /**
   * Natural break (retry after death, quit to title): hold the game, show the ad, then proceed().
   * Repeat calls while a break is running return the same promise.
   */
  _break(proceed, resumeAfter = true) {
    if (this.busy) return this._breakPromise;
    if (!this.adsAllowed) return proceed();
    const ctx = this.ctx, hud = ctx?.hud;
    this.busy = true;
    try { clearTimeout(hud?._autoEndT); } catch {}          // don't let the death/win screen pop back up
    try { hud?.showScreen?.(null); } catch {}               // close menus (this may un-pause: re-held below)
    if (ctx) { ctx.paused = true; if (ctx.input) ctx.input.enabled = false; }
    this._releasePointer();
    try { hud?.fade?.(true, 0.25); } catch {}
    this.setPlaying(false);
    this._breakPromise = this.midgame()
      .catch(() => false)
      .then(() => {
        // quitting keeps the game held until the page navigates away (no gameplayStart in between)
        if (!resumeAfter) return proceed();
        this.busy = false;
        if (ctx?.input) ctx.input.enabled = true;
        if (ctx) ctx.paused = false;
        return proceed();
      });
    return this._breakPromise;
  },

  /** Audio is silenced while any reason is active (ad playing, platform mute setting). */
  _setMute(reason, on) {
    if (on) this._muteReasons.add(reason); else this._muteReasons.delete(reason);
    this._applyMute();
  },
  _applyMute() {
    const ac = this.ctx?.audio?.ac;
    if (!ac) return;
    if (this._muteBusy) { this._muteDirty = true; return; } // re-checked when the pending transition resolves
    const want = this._muteReasons.size > 0;
    let p = null;
    try {
      if (want && ac.state === 'running') p = ac.suspend();
      else if (!want && ac.state === 'suspended') p = ac.resume();
    } catch {}
    if (p?.finally) {
      this._muteBusy = true;
      p.catch(() => {}).finally(() => {
        this._muteBusy = false;
        if (this._muteDirty) { this._muteDirty = false; }
        this._applyMute(); // reconcile with the latest wanted state
      });
    }
  },

  /** Called by main.js once systems exist: hooks audio unlock, retry/quit ad breaks, pause, and win. */
  attach(ctx) {
    this.ctx = ctx;
    // audio context is created lazily on the first gesture: re-apply the mute state when it appears
    const audio = ctx.audio;
    if (audio?.unlock) {
      const unlock = audio.unlock.bind(audio);
      audio.unlock = (...a) => unlock(...a).then((v) => { this._applyMute(); return v; });
    }
    // retry after death / from the pause menu -> midgame ad first, then restart (the SDK enforces its own cooldown)
    const game = ctx.game;
    if (game?.restartFromCheckpoint) {
      const restart = game.restartFromCheckpoint.bind(game);
      game.restartFromCheckpoint = (...a) => this._break(() => restart(...a));
    }
    const hud = ctx.hud;
    if (hud) {
      // quit to title (pause menu, death or win screen) is a natural break too
      if (hud._quit) { const quit = hud._quit.bind(hud); hud._quit = (...a) => this._break(() => quit(...a), false); }
      // no pause menu, death/win screens or menu actions while a break or an ad is running
      for (const k of ['pause', '_retry', '_autoEnd']) {
        if (typeof hud[k] !== 'function') continue;
        const f = hud[k].bind(hud);
        hud[k] = (...a) => (this.busy || this.adActive ? undefined : f(...a));
      }
    }
    ctx.events?.on?.('win', () => this.happytime());
    this._applyMute(); // a muteAudio setting may have arrived before the audio context existed
  },
};

/**
 * Stop the host page from scrolling or navigating when the game uses Space, arrows or the wheel.
 * The game runs inside an iframe on CrazyGames. Keys typed into form controls, and wheel over UI panels that can
 * still scroll in that direction, keep their normal behaviour.
 */
export function installIframeGuards() {
  const KEYS = new Set(['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End']);
  const editable = (el) => !!el?.closest?.('input, textarea, select, [contenteditable=""], [contenteditable="true"]');
  const activates = (el) => !!el?.closest?.('button, a[href], [role="button"], [role="slider"], [role="radio"]');
  window.addEventListener('keydown', (e) => {
    if (!KEYS.has(e.code)) return;
    if (editable(e.target)) return;
    if (e.code === 'Space' && activates(e.target)) return; // Space presses the focused button
    e.preventDefault();
  });
  // true if some ancestor of el can still scroll in the wheel direction (then the wheel is its to use)
  const canScroll = (el, dy) => {
    for (let n = el; n && n !== document.body && n !== document.documentElement; n = n.parentElement) {
      if (n.scrollHeight <= n.clientHeight + 1) continue;
      const oy = getComputedStyle(n).overflowY;
      if (oy !== 'auto' && oy !== 'scroll') continue;
      if (dy < 0 ? n.scrollTop > 0 : n.scrollTop + n.clientHeight < n.scrollHeight - 1) return true;
    }
    return false;
  };
  window.addEventListener('wheel', (e) => { if (!canScroll(e.target, e.deltaY)) e.preventDefault(); }, { passive: false });
}

// exposed for debugging and automated tests (read-only use)
window.__platform = platform;
