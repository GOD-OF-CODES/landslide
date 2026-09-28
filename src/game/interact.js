// GAME workstream: interaction targeting + prompts (DESIGN.md "interact").
//
//   interact.register({
//     id, object | position,      // Object3D (world position of its origin, or of `offset` in its local frame), a Vector3,
//                                  // or a function () => Vector3|null (resolved every frame, so late-loading systems work)
//     radius = 2,                  // max distance from the eye (or feet) in meters
//     prompt,                      // string or (ctx) => string   shown as "[E] prompt" / "Hold [E] prompt"
//     hint,                        // optional string or fn: shown (without a key) when canUse() is false
//     canUse(ctx) -> bool,         // default: always
//     use(ctx),                    // called on E (instant) or when the hold completes
//     hold = 0,                    // seconds of hold-E (0 = instant press)
//     mode = 'foot',               // 'foot': proximity + look-at targeting; 'car': applies while seated (no proximity)
//     angle = 38,                  // max look-off angle in degrees for 'foot' entries
//     priority = 0,
//     onHold(progress, dt),        // optional: called every frame while holding (e.g. glug audio, hatchet wind-up)
//     onCancel(),                  // optional: hold released before completing
//     repeat = false,              // hold again without releasing E (chopping rhythm)
//     focus = false,               // inspection: while targeted on foot, the camera racks focus onto it (cameraRig.focusOn)
//   })
//   interact.unregister(id)
//   interact.current            // id of the targeted entry or null
//   interact.progress           // 0..1 hold progress
//   interact.lock(seconds)      // suppress interactions briefly (animations, transitions)
//
// It is the ONLY system calling hud.setPrompt (so prompts never flicker between owners).
import * as THREE from 'three';

const _p = new THREE.Vector3(), _dir = new THREE.Vector3(), _to = new THREE.Vector3();
const DEG = Math.PI / 180;

export default class Interact {
  constructor(ctx) {
    this.ctx = ctx;
    this.entries = new Map();
    this.current = null;
    this.progress = 0;
    this.enabled = true;
    this._lockT = 0;
    this._latch = false;          // E must be released before the next use
    this._lastPrompt = undefined;
    this._lastProg = null;
    this._holding = null;
    this._dwell = 0; this._dwellId = null;
  }

  register(e) {
    if (!e || !e.id) throw new Error('interact.register: id required');
    const entry = { radius: 2, hold: 0, mode: 'foot', angle: 38, priority: 0, ...e };
    this.entries.set(e.id, entry);
    return () => this.unregister(e.id);
  }

  unregister(id) {
    if (this.current === id) { this.current = null; this.progress = 0; }
    this.entries.delete(id);
  }

  has(id) { return this.entries.has(id); }

  lock(seconds = 0.4) { this._lockT = Math.max(this._lockT, seconds); this.progress = 0; }

  /** World position of an entry (or null when its object is missing/hidden). */
  positionOf(e, out = _p) {
    let src = e.object ?? e.position;
    if (typeof src === 'function') src = src(this.ctx);
    if (!src) return null;
    if (src.isObject3D) {
      if (!src.parent && src !== this.ctx.scene) return null;
      if (e.requireVisible !== false && !isVisible(src)) return null;
      src.updateWorldMatrix(true, false);
      if (e.offset) return out.copy(e.offset).applyMatrix4(src.matrixWorld);
      return out.setFromMatrixPosition(src.matrixWorld);
    }
    if (src.isVector3) return out.copy(src);
    if (Array.isArray(src)) return out.set(src[0], src[1], src[2]);
    return null;
  }

  _text(v) { return typeof v === 'function' ? v(this.ctx) : v; }

  _hintText(e) {
    if (!e.hint) return null;
    try { return this._text(e.hint) || null; } catch { return null; }
  }

  _usable(e) {
    try { return e.canUse ? !!e.canUse(this.ctx) : true; } catch (err) { console.warn('[interact] canUse', e.id, err); return false; }
  }

  _pick() {
    const { ctx } = this;
    const control = ctx.control;
    if (control === 'car') {
      let best = null, bestP = -Infinity;
      for (const e of this.entries.values()) {
        if (e.mode !== 'car') continue;
        const usable = this._usable(e);
        if (!usable && !this._hintText(e)) continue;
        const pr = e.priority + (usable ? 10 : 0);
        if (pr > bestP) { bestP = pr; best = e; }
      }
      return best ? { e: best, usable: this._usable(best) } : null;
    }
    if (control !== 'foot') return null;
    const pl = ctx.player;
    if (!pl?.eye) return null;
    const eye = pl.eye;
    const look = pl.lookDir ? pl.lookDir(_dir) : _dir.set(0, 0, -1);
    const feet = pl.feet || pl.position || eye;
    let best = null, bestScore = -Infinity, bestUsable = false, bestDist = 0;
    for (const e of this.entries.values()) {
      if (e.mode !== 'foot') continue;
      const p = this.positionOf(e);
      if (!p) continue;
      const dEye = eye.distanceTo(p);
      const dFeet = Math.hypot(p.x - feet.x, p.z - feet.z) + Math.max(0, feet.y - p.y - 0.3) * 0.5;
      const dist = Math.min(dEye, dFeet + 0.25);
      if (dist > e.radius) continue;
      _to.subVectors(p, eye);
      const len = _to.length() || 1;
      const cos = _to.dot(look) / len;
      // very close items only need a rough look; farther ones must be looked at
      const maxAng = (dEye < 1.0 ? 70 : e.angle) * DEG;
      if (cos < Math.cos(maxAng)) continue;
      const usable = this._usable(e);
      if (!usable && !this._hintText(e)) continue;
      const score = cos * 1.2 - dist / e.radius * 0.5 + e.priority + (usable ? 0.3 : 0);
      if (score > bestScore) { bestScore = score; best = e; bestUsable = usable; bestDist = dEye; }
    }
    return best ? { e: best, usable: bestUsable, dist: bestDist } : null;
  }

  update(dt) {
    const { ctx } = this;
    const input = ctx.input;
    const state = ctx.game?.state;
    const inactive = !this.enabled || ctx.paused || !input || state === 'title' || state === 'dead' || state === 'win' || ctx.control === 'none';
    if (this._lockT > 0) this._lockT -= dt;
    const eDown = input?.down?.('KeyE');
    if (!eDown) this._latch = false;

    if (inactive || this._lockT > 0) {
      this._cancelHold();
      this.current = null;
      this._setPrompt(null, null);
      return;
    }

    const pick = this._pick();
    const e = pick?.e || null;
    if (!e || (this._holding && this._holding !== e)) this._cancelHold();
    this.current = e ? e.id : null;
    if (!e) { this._setPrompt(null, null); return; }
    // inspecting something up close: the lens racks focus onto it (subtle background blur; post DOF, Ultra/High only).
    // Brief by design: from 0.3 s to ~2 s after the player stops to look at it, and while a hold-to-use action runs on
    // it; the rig then eases the blur out and switches the DOF pass off again.
    if (e.focus && pick.dist > 0) {
      this._dwell = this._dwellId === e.id ? this._dwell + dt : 0;
      this._dwellId = e.id;
      const v = ctx.player?.velocity;
      const still = !v || Math.hypot(v.x, v.z) < 1.6;
      if ((still && this._dwell > 0.3 && this._dwell < 2.2) || (this._holding === e && this.progress > 0)) {
        try { ctx.cameraRig?.focusOn?.(pick.dist, { hold: 0.2 }); } catch {}
      }
    } else this._dwellId = null;

    if (!pick.usable) {
      this._setPrompt(this._hintText(e), null);
      return;
    }

    const label = this._text(e.prompt) || 'Use';
    if (e.hold > 0) {
      if (eDown && !this._latch) {
        this._holding = e;
        this.progress = Math.min(1, this.progress + dt / e.hold);
        try { e.onHold?.(this.progress, dt, ctx); } catch (err) { console.warn('[interact] onHold', err); }
        if (this.progress >= 1) {
          this.progress = 0; this._holding = null;
          if (!e.repeat) this._latch = true;
          this._use(e);
          return;
        }
        this._setPrompt(`Hold [E]  ${label}`, this.progress);
      } else {
        if (this._holding) this._cancelHold();
        this._setPrompt(`Hold [E]  ${label}`, null);
      }
    } else {
      this._setPrompt(`[E]  ${label}`, null);
      if (input.pressed('KeyE') && !this._latch) {
        this._latch = true;
        this._use(e);
      }
    }
  }

  _use(e) {
    try { e.use?.(this.ctx); } catch (err) { console.error('[interact] use', e.id, err); }
    // the entry may have changed/unregistered itself: refresh the prompt next frame
    this._lastPrompt = undefined;
  }

  _cancelHold() {
    if (this._holding) {
      const h = this._holding;
      this._holding = null;
      try { h.onCancel?.(this.ctx); } catch {}
    }
    this.progress = 0;
  }

  _setPrompt(text, progress) {
    const q = progress === null ? null : Math.round(progress * 100) / 100;
    if (text === this._lastPrompt && q === this._lastProg) return;
    this._lastPrompt = text; this._lastProg = q;
    try { this.ctx.hud?.setPrompt?.(text, progress); } catch (err) { console.warn('[interact] hud.setPrompt', err); }
  }

  /** Force the prompt to refresh (e.g. after the HUD rebuilt its DOM). */
  refresh() { this._lastPrompt = undefined; }

  dispose() { this.entries.clear(); this._setPrompt(null, null); }
}

function isVisible(o) {
  for (let n = o; n; n = n.parent) if (n.visible === false) return false;
  return true;
}
