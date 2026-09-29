// Keyboard / mouse input with per-frame edge detection and pointer lock, plus the touch layer's hooks.
// Key identifiers are KeyboardEvent.code values ('KeyW', 'Space', 'ShiftLeft', 'ArrowUp', ...).
//
// Touch (phones/tablets, or ?touch=1): src/ui/touch.js draws the on-screen controls and feeds this object:
//   - virtual keys: hold(source, code, on) / pulse(code, seconds). Ref-counted per source, kept apart from the
//     keyboard's `keys` set; down()/pressed()/released() see both, so every system keeps reading key codes.
//   - axes {moveX, moveY, steer, throttle, brake}: analog -1..1 (0..1 for the pedals). moveX/moveY below combine them
//     with WASD; vehicle.js reads steer (+ = right) / throttle / brake. Keyboard-only play leaves them at 0.
//   - addLook(dxPx, dyPx): drag deltas, smoothed (frame-rate independent) into dx/dy at the start of the next frame.
//   - lookHeld: a finger is resting on the look area (the camera rig does not recentre under it).
// In touch mode pointer lock is never requested, so the HUD's pause-on-lock-loss never fires.
import { TouchControls } from '../ui/touch.js';

/** Touch-primary device? ?touch=1 / ?touch=0 force it either way (tests, desktop debugging). */
export function detectTouch() {
  try {
    const p = new URLSearchParams(location.search).get('touch');
    if (p === '1' || p === 'true' || p === 'on') return true;
    if (p === '0' || p === 'false' || p === 'off') return false;
  } catch { /* no location (worker/test) */ }
  try {
    const hasTouch = (navigator.maxTouchPoints || 0) > 0 || 'ontouchstart' in window;
    if (!hasTouch) return false;
    const mm = window.matchMedia;
    if (mm) return mm('(pointer: coarse)').matches || (mm('(any-pointer: coarse)').matches && !mm('(any-pointer: fine)').matches);
    return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || '');
  } catch { return false; }
}

export class Input {
  constructor(dom) {
    this.dom = dom;
    this.keys = new Set();
    this._pressed = new Set();
    this._released = new Set();
    this.mouseButtons = new Set();
    this._mousePressed = new Set();
    this.dx = 0; this.dy = 0;          // mouse movement accumulated this frame (pixels)
    this.wheel = 0;
    this.enabled = true;               // false while menus are open (game code should respect `input.active`)
    this.sensitivity = 0.0022;         // radians per pixel, used by look controllers
    this.invertY = false;
    // --- touch hooks (inert on desktop)
    this.touch = detectTouch();
    this.axes = { moveX: 0, moveY: 0, steer: 0, throttle: 0, brake: 0 };
    this.lookHeld = false;
    this._virt = new Map();            // code -> Set(source) : virtual key holds (touch buttons)
    this._pulses = new Map();          // code -> expiry (performance.now ms)
    this._lookPend = { x: 0, y: 0 };   // touch look not yet released into dx/dy
    this._lookTau = 0.03;              // s: smoothing time constant of the touch look
    this._lastBegin = 0;
    this._beginQueued = false;
    this._begin = this._begin.bind(this);

    window.addEventListener('keydown', (e) => {
      if (['Space', 'ArrowUp', 'ArrowDown', 'Tab'].includes(e.code)) e.preventDefault();
      if (!this.keys.has(e.code)) this._pressed.add(e.code);
      this.keys.add(e.code);
    });
    window.addEventListener('keyup', (e) => { this.keys.delete(e.code); this._released.add(e.code); });
    window.addEventListener('blur', () => { this.keys.clear(); this.mouseButtons.clear(); this.releaseVirtual(); });
    window.addEventListener('mousemove', (e) => {
      // touch mode: taps also fire compatibility mouse events; look comes from the touch layer only
      if (this.touch && !this.locked) return;
      if (this.locked || this.mouseButtons.size) { this.dx += e.movementX || 0; this.dy += e.movementY || 0; }
    });
    window.addEventListener('mousedown', (e) => { this.mouseButtons.add(e.button); this._mousePressed.add(e.button); });
    window.addEventListener('mouseup', (e) => this.mouseButtons.delete(e.button));
    window.addEventListener('wheel', (e) => { this.wheel += Math.sign(e.deltaY); }, { passive: true });
    document.addEventListener('pointerlockchange', () => this.onLockChange?.(this.locked));

    this.touchUI = null;
    if (this.touch) {
      document.addEventListener('visibilitychange', () => { if (document.hidden) this.releaseVirtual(); });
      try { this.touchUI = new TouchControls(this); } catch (e) { console.error('[input] touch controls failed', e); }
    }
  }
  get locked() { return document.pointerLockElement === this.dom; }
  get active() { return this.enabled; }
  requestLock() {
    if (this.touch) return; // no pointer lock on touch devices (and so no pause-on-lock-loss)
    try { const p = this.dom.requestPointerLock?.(); p?.catch?.(() => {}); } catch {}
  }
  exitLock() { if (this.locked) document.exitPointerLock(); }
  _vdown(code) { return this._virt.has(code); }
  down(code) { return this.enabled && (this.keys.has(code) || this._virt.has(code)); }
  pressed(code) { return this.enabled && this._pressed.has(code); }
  released(code) { return this._released.has(code); }
  mouseDown(b = 0) { return this.enabled && this.mouseButtons.has(b); }
  mousePressed(b = 0) { return this.enabled && this._mousePressed.has(b); }
  // -1..1 axes; WASD + arrows (+ the touch stick)
  get moveX() {
    const k = (this.down('KeyD') || this.down('ArrowRight') ? 1 : 0) - (this.down('KeyA') || this.down('ArrowLeft') ? 1 : 0);
    const a = this.enabled ? this.axes.moveX : 0;
    return a ? Math.max(-1, Math.min(1, k + a)) : k;
  }
  get moveY() {
    const k = (this.down('KeyW') || this.down('ArrowUp') ? 1 : 0) - (this.down('KeyS') || this.down('ArrowDown') ? 1 : 0);
    const a = this.enabled ? this.axes.moveY : 0;
    return a ? Math.max(-1, Math.min(1, k + a)) : k;
  }
  // Synthetic input for automated tests: input.simulate({KeyW:true}) holds keys until cleared
  simulate(state) {
    for (const [k, v] of Object.entries(state)) {
      if (v) { if (!this.keys.has(k)) this._pressed.add(k); this.keys.add(k); }
      else { this.keys.delete(k); this._released.add(k); }
    }
  }

  // ---------------------------------------------------------------------------------------- touch hooks
  /** Virtual key hold by `source` (a touch button, the stick's auto-sprint...). Edges fire on the first hold / last release. */
  hold(source, code, on) {
    let s = this._virt.get(code);
    if (on) {
      if (!s) { s = new Set(); this._virt.set(code, s); if (!this.keys.has(code)) this._pressed.add(code); }
      s.add(source);
    } else if (s && s.delete(source) && s.size === 0) {
      this._virt.delete(code);
      if (!this.keys.has(code)) this._released.add(code);
    }
  }
  /** Hold a virtual key for a short time (at least one full frame). */
  pulse(code, seconds = 0.06) {
    this.hold('pulse', code, true);
    this._pulses.set(code, performance.now() + seconds * 1000);
    this._queueBegin();
  }
  /** Release every virtual key and analog axis (menus, blur, controls hidden). */
  releaseVirtual() {
    for (const [code] of this._virt) { if (!this.keys.has(code)) this._released.add(code); }
    this._virt.clear(); this._pulses.clear();
    const a = this.axes; a.moveX = a.moveY = a.steer = a.throttle = a.brake = 0;
    this._lookPend.x = this._lookPend.y = 0;
    this.lookHeld = false;
  }
  /** Touch look drag, in touch-scaled pixels (the look controllers multiply by `sensitivity`). */
  addLook(dx, dy) {
    this._lookPend.x += dx; this._lookPend.y += dy;
    this._queueBegin();
  }
  _queueBegin() {
    if (this._beginQueued) return;
    this._beginQueued = true;
    requestAnimationFrame(this._begin);
  }
  // Runs at the start of an animation frame, before the game's frame callback (endFrame queues it right before
  // main.js requests its own frame): releases the smoothed touch look into dx/dy and expires pulses.
  _begin(now) {
    this._beginQueued = false;
    const t = performance.now();
    const dt = Math.min(0.1, Math.max(0.001, (t - (this._lastBegin || t - 16.7)) / 1000));
    this._lastBegin = t;
    const P = this._lookPend;
    if (P.x || P.y) {
      const k = 1 - Math.exp(-dt / this._lookTau);
      let rx = P.x * k, ry = P.y * k;
      if (Math.abs(P.x - rx) < 0.05) rx = P.x; // no endless decay tail (the camera's idle timer must see zero)
      if (Math.abs(P.y - ry) < 0.05) ry = P.y;
      this.dx += rx; this.dy += ry; P.x -= rx; P.y -= ry;
    }
    for (const [code, until] of this._pulses) {
      if (t >= until) { this._pulses.delete(code); this.hold('pulse', code, false); }
    }
    try { this.touchUI?.frame(dt, now); } catch (e) { console.error('[touch] frame', e); }
  }
  endFrame() {
    this._pressed.clear(); this._released.clear(); this._mousePressed.clear();
    this.dx = 0; this.dy = 0; this.wheel = 0;
    if (this.touch) this._queueBegin();
  }
}
