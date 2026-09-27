// Keyboard / mouse input with per-frame edge detection and pointer lock.
// Key identifiers are KeyboardEvent.code values ('KeyW', 'Space', 'ShiftLeft', 'ArrowUp', ...).
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
    window.addEventListener('keydown', (e) => {
      if (['Space', 'ArrowUp', 'ArrowDown', 'Tab'].includes(e.code)) e.preventDefault();
      if (!this.keys.has(e.code)) this._pressed.add(e.code);
      this.keys.add(e.code);
    });
    window.addEventListener('keyup', (e) => { this.keys.delete(e.code); this._released.add(e.code); });
    window.addEventListener('blur', () => { this.keys.clear(); this.mouseButtons.clear(); });
    window.addEventListener('mousemove', (e) => {
      if (this.locked || this.mouseButtons.size) { this.dx += e.movementX || 0; this.dy += e.movementY || 0; }
    });
    window.addEventListener('mousedown', (e) => { this.mouseButtons.add(e.button); this._mousePressed.add(e.button); });
    window.addEventListener('mouseup', (e) => this.mouseButtons.delete(e.button));
    window.addEventListener('wheel', (e) => { this.wheel += Math.sign(e.deltaY); }, { passive: true });
    document.addEventListener('pointerlockchange', () => this.onLockChange?.(this.locked));
  }
  get locked() { return document.pointerLockElement === this.dom; }
  get active() { return this.enabled; }
  requestLock() { try { const p = this.dom.requestPointerLock?.(); p?.catch?.(() => {}); } catch {} }
  exitLock() { if (this.locked) document.exitPointerLock(); }
  down(code) { return this.enabled && this.keys.has(code); }
  pressed(code) { return this.enabled && this._pressed.has(code); }
  released(code) { return this._released.has(code); }
  mouseDown(b = 0) { return this.enabled && this.mouseButtons.has(b); }
  mousePressed(b = 0) { return this.enabled && this._mousePressed.has(b); }
  // -1..1 axes; WASD + arrows
  get moveX() { return (this.down('KeyD') || this.down('ArrowRight') ? 1 : 0) - (this.down('KeyA') || this.down('ArrowLeft') ? 1 : 0); }
  get moveY() { return (this.down('KeyW') || this.down('ArrowUp') ? 1 : 0) - (this.down('KeyS') || this.down('ArrowDown') ? 1 : 0); }
  // Synthetic input for automated tests: input.simulate({KeyW:true}) holds keys until cleared
  simulate(state) {
    for (const [k, v] of Object.entries(state)) {
      if (v) { if (!this.keys.has(k)) this._pressed.add(k); this.keys.add(k); }
      else { this.keys.delete(k); this._released.add(k); }
    }
  }
  endFrame() {
    this._pressed.clear(); this._released.clear(); this._mousePressed.clear();
    this.dx = 0; this.dy = 0; this.wheel = 0;
  }
}
