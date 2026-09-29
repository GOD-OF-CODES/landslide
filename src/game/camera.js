// GAME workstream: camera rig (DESIGN.md "cameraRig").
//
//   mode: 'foot' | 'car-chase' | 'car-cockpit' | 'cinematic';  setMode(m, {blend = 0.55 s})
//   shake(trauma 0..1)                  trauma model: decays; offsets/rotation noise ∝ trauma² (+ landslide.rumble floor)
//   cinematic(pathFn, duration, opts) -> Promise
//        pathFn(t01, outPos, outTarget, ctx) sets the pose; may return {fov, roll}. opts: {loop, hold, blend, ease}
//        (hold = stay in 'cinematic' at the end; otherwise the previous mode is restored)
//   carMode: the preferred seated view ('car-cockpit' | 'car-chase'); C toggles it while ctx.control === 'car'
//   lookBack 0..1: hold Q while driving to look back (cockpit: over the shoulder; chase: camera swings round)
//   viewmodel: first-person gloved hands + jacket forearms (assets/models/hands.glb) holding the hatchet, the jerrycan or
//     the planks: setHeld('hatchet'|'jerrycan'|'planks'|null), windup(0..1), strike() -> Promise, pour(p|null),
//     place(p|null), reach(worldPos) -> Promise (see the Viewmodel section below)
//   kick(pitch, yaw, roll)             camera kick impulse (rad/s) into a stiff spring (the hatchet blow)
//   focusOn(dist, {ref, lens, range, hold, attack, release, force})   brief focus pull through post.setDOF (Ultra/High
//     only); without `range` the blur is calibrated to a thin-lens camera (lensRange) at the reference distance(s)
//
// Foot: FPS from player.eye/yaw/pitch, stride-synced head-bob, landing dip spring, breathing sway, strafe roll.
// Chase: spring-damped follow with look-ahead, mouse orbit that recenters, collision against static geometry
// (never inside the cut face), ground clamp. Cockpit: at car.seatCam, mouse look that recenters, G-force head sway.
// FOV kick with speed (and sprint). Mode changes blend from the current pose.
import * as THREE from 'three';
import { G, groups } from '../physics/world.js';

const clamp = THREE.MathUtils.clamp;
const lerp = THREE.MathUtils.lerp;
const smooth01 = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
const dampK = (rate, dt) => 1 - Math.exp(-rate * dt);
const wrapAngle = (a) => { while (a > Math.PI) a -= Math.PI * 2; while (a < -Math.PI) a += Math.PI * 2; return a; };

const _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3(), _v4 = new THREE.Vector3();
const _q1 = new THREE.Quaternion(), _q2 = new THREE.Quaternion();
const _e = new THREE.Euler(0, 0, 0, 'YXZ');
const _m = new THREE.Matrix4();
const Y = new THREE.Vector3(0, 1, 0);
const Q_FLIP = new THREE.Quaternion().setFromAxisAngle(Y, Math.PI); // car +Z forward -> camera looks down -Z

// depth of field (see CameraRig.lensRange)
const LENSES = {
  phone: { sensor: 0.036, N: 1.8 },
  cine: { sensor: 0.054, N: 1.4 },
};
const DOF_BLUR_PX = 5;
// inverse of smoothstep on [0,1]: u with 3u^2 - 2u^3 = m
const invSmooth = (m) => 0.5 - Math.sin(Math.asin(clamp(1 - 2 * m, -1, 1)) / 3);

// cheap smooth pseudo-noise (sum of incommensurate sines), roughly -1..1
function noise(t, seed) {
  return Math.sin(t * 1.0 + seed * 12.9898) * 0.5 + Math.sin(t * 2.31 + seed * 78.233) * 0.3 + Math.sin(t * 4.97 + seed * 37.719) * 0.2;
}

export default class CameraRig {
  constructor(ctx) {
    this.ctx = ctx;
    this.camera = ctx.camera;
    this.mode = 'cinematic';
    this.carMode = 'car-cockpit';
    this.fovBase = ctx.config?.camera?.fov ?? 70;
    this.fov = this.fovBase;
    this.trauma = 0;
    this.enabled = true;
    this.pos = new THREE.Vector3().copy(this.camera.position);
    this.quat = new THREE.Quaternion().copy(this.camera.quaternion);
    // blending between modes
    this._blendT = 1; this._blendDur = 0.001;
    this._fromPos = new THREE.Vector3(); this._fromQuat = new THREE.Quaternion();
    // foot
    this._bobPhase = 0; this._bobAmt = 0; this._dip = 0; this._dipV = 0; this._roll = 0; this._breathT = 0;
    // cockpit look
    this.lookYaw = 0; this.lookPitch = -0.04; this._mouseIdle = 99;
    this._headOff = new THREE.Vector3();
    // chase
    this.orbitYaw = 0; this.orbitPitch = 0; this._chaseYaw = null; this._chasePos = new THREE.Vector3(); this._chaseVel = new THREE.Vector3();
    this._chaseInit = false; this._chaseDist = 5.6;
    // cinematic
    this._cin = null; this._cinPos = new THREE.Vector3(); this._cinTarget = new THREE.Vector3();
    this._prevMode = 'foot';
    this._t = 0;
    // focus pull (post DOF): see focusOn()
    this._focus = { dist: 10, cur: 10, range: 10, until: -1, amt: 0, on: false, attack: 0.35, release: 0.7 };
    this._kick = { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
    // viewmodel
    this.viewmodel = new Viewmodel(ctx);

    ctx.events?.on?.('player:land', (p) => { this._dipV -= clamp((p?.speed ?? 3) * 0.055, 0.05, 0.6); this.shake(clamp(((p?.speed ?? 0) - 4) * 0.06, 0, 0.3)); });
    ctx.events?.on?.('car:impact', (p) => { if (this.mode.startsWith('car')) this.shake(clamp((p?.impulse ?? 0) / 14000, 0.05, 0.6)); });
    ctx.events?.on?.('impact', (p) => {
      if (!p?.position || p.source === 'car') return;
      const d = p.position.distanceTo(this.camera.position);
      const e = p.energy ?? 1;
      this.shake(clamp(Math.sqrt(e) * 0.5 / (1 + d * d * 0.004), 0, 0.5));
    });
  }

  async init() {
    this.ctx.camera.fov = this.fovBase;
    this.ctx.camera.updateProjectionMatrix();
    // the hands + held items are built now (not lazily) so the boot shader warm-up compiles their programs
    try { await this.viewmodel.init(); } catch (e) { console.warn('[cameraRig] viewmodel init', e); }
  }

  /** Camera kick (radians): an impulse into a stiff damped spring (a hatchet blow jarring the view). */
  kick(pitch = 0, yaw = 0, roll = 0) {
    const K = this._kick;
    K.vx += pitch; K.vy += yaw; K.vz += roll;
  }

  // --------------------------------------------------------------------------------------------- API
  setMode(m, opts = {}) {
    if (!['foot', 'car-chase', 'car-cockpit', 'cinematic'].includes(m)) return;
    if (m === 'car-chase' || m === 'car-cockpit') this.carMode = m;
    if (m === this.mode && !opts.force) return;
    if (this.mode !== 'cinematic') this._prevMode = this.mode;
    this.mode = m;
    const blend = opts.blend ?? 0.55;
    this._fromPos.copy(this.camera.position); this._fromQuat.copy(this.camera.quaternion);
    this._blendT = 0; this._blendDur = Math.max(0.001, blend);
    if (m === 'car-chase') this._chaseInit = false;
    if (m === 'car-cockpit') { this.lookYaw = 0; this.lookPitch = -0.04; }
    if (m !== 'cinematic' && this._cin && !this._cin.done) this._endCinematic(false);
  }

  /** Snap (no blend) — e.g. after a teleport or checkpoint restart. */
  snap() { this._blendT = this._blendDur = 1; this._chaseInit = false; }

  shake(t) { if (t > 0 && isFinite(t)) this.trauma = Math.min(1, this.trauma + t); }

  /**
   * Brief, subtle focus pull, like a phone camera refocusing on something close (post DOF; only on presets with
   * config.quality.dof, a no-op elsewhere). Call it every frame while it applies (it lapses `hold` s after the last
   * call) or once with a longer `hold`. The focus racks to `dist` metres; the background blur eases in over `attack` s
   * and out over `release` s, then the DOF pass is switched off again (it costs nothing while off).
   * `range` = metres beyond the focus plane over which the blur ramps to full: bigger is subtler.
   */
  focusOn(dist, opts = {}) {
    if (!(dist > 0) || !isFinite(dist)) return;
    const F = this._focus;
    let range = opts.range;
    if (range == null) {
      // thin-lens calibration (see lensRange): the blur at the reference distance(s) matches a real lens; when even
      // that would be sub-visible the DOF pass is not switched on at all (it costs ~3 ms at 720p)
      const L = this.lensRange(dist, opts.ref ?? Math.max(30, dist * 12), opts.lens);
      if (!L || L.peak < 0.04) return;
      range = L.range;
    }
    F.dist = dist;
    F.range = range;
    F.until = this._t + (opts.hold ?? 0.15);
    F.attack = opts.attack ?? 0.35; F.release = opts.release ?? 0.7;
    F.force = !!opts.force;
  }
  focusOff() { this._focus.until = -1; }

  /**
   * Thin-lens depth of field for the post DOF. The post library's circle of confusion is a linear ramp,
   * smoothstep(0, range, |d - F|), scaled to a ~2.2 px radius blur (half-res) at full magnitude (DOF_BLUR_PX diameter).
   * A real lens gives c = f^2 / (N (F - f)) * |d - F| / d on the sensor (f from the current FOV and the sensor width).
   * `range` is solved so the blur at each reference distance `ref` (number or array) is at most the real lens's blur
   * in pixels (the largest range wins, i.e. the subtlest match), so nothing is ever blurrier than the real camera
   * would make it. Returns {range, peak} (peak = largest CoC magnitude 0..1 at the refs) or null.
   *   lens: 'phone' (full-frame equivalent f/1.8: an eye/phone refocusing on something at arm's length) or 'cine'
   *   (large-format cinema prime wide open, 54 mm sensor, f/1.4) or {sensor (m), N}.
   */
  lensRange(F, ref, lens = 'phone') {
    const L = typeof lens === 'object' && lens ? lens : (LENSES[lens] || LENSES.phone);
    const cam = this.camera;
    const w = this.ctx.renderer?.domElement?.width || 1280;
    const vf = THREE.MathUtils.degToRad(cam.fov || 70);
    const hf = 2 * Math.atan(Math.tan(vf / 2) * (cam.aspect || 16 / 9));
    const f = (L.sensor / 2) / Math.tan(hf / 2);
    const refs = Array.isArray(ref) ? ref : [ref];
    let range = 0, peak = 0;
    for (const d of refs) {
      const x = Math.abs(d - F);
      if (!(d > 0) || x < 1e-3) continue;
      const c = (f * f) / (L.N * Math.max(F - f, 1e-3)) * x / d;          // CoC on the sensor (m)
      const m = clamp((c / L.sensor) * w / DOF_BLUR_PX, 1e-3, 0.95);       // -> fraction of the post's full blur
      peak = Math.max(peak, m);
      range = Math.max(range, x / invSmooth(m));
    }
    return range > 0 ? { range, peak } : null;
  }

  _updateFocus(dt) {
    const F = this._focus, post = this.ctx.post;
    const q = this.ctx.config?.quality;
    const can = !!q?.dof && typeof post?.setDOF === 'function';
    // frame-time budget: the DOF pass costs ~3 ms at 720p on the M1, so on High a gameplay focus pull only starts when
    // the game is running with headroom (smoothed real frame time under 1/52 s); Ultra and cinematics (force) always
    const now = performance.now(), fdt = (now - (this._pfPrev || now)) / 1000;
    this._pfPrev = now;
    if (fdt > 0 && fdt < 0.25) this._frameT = this._frameT ? this._frameT + (fdt - this._frameT) * 0.05 : fdt;
    const headroom = F.force || F.on || q?.key === 'ultra' || !(this._frameT > 1 / 52);
    const want = can && headroom && this._t < F.until ? 1 : 0;
    const rate = 1 / Math.max(0.05, want ? F.attack : F.release);
    F.amt = want > F.amt ? Math.min(1, F.amt + dt * rate) : Math.max(0, F.amt - dt * rate);
    if (F.amt <= 0.001) {
      if (F.on) { F.on = false; try { post?.setDOF?.(null); } catch {} }
      return;
    }
    if (!F.on) F.cur = F.dist; else F.cur = lerp(F.cur, F.dist, dampK(6, dt));
    F.on = true;
    // the blur fades in by narrowing the ramp from "nothing within 600 m" to the requested range (log space: even)
    const k = smooth01(0, 1, F.amt);
    const range = Math.exp(lerp(Math.log(600), Math.log(Math.max(0.5, F.range)), k));
    try { post.setDOF(F.cur, range); } catch {}
  }

  cinematic(pathFn, duration = 5, opts = {}) {
    if (this._cin && !this._cin.done) this._endCinematic(false);
    return new Promise((resolve) => {
      const prev = this.mode === 'cinematic' ? this._prevMode : this.mode;
      this.setMode('cinematic', { blend: opts.blend ?? 0.8, force: true });
      this._prevMode = prev;
      this._cin = { fn: pathFn, dur: Math.max(0.01, duration), t: 0, resolve, loop: !!opts.loop, hold: !!opts.hold, ease: opts.ease, done: false, fov: null, roll: 0 };
    });
  }

  stopCinematic(restore = true) { if (this._cin && !this._cin.done) this._endCinematic(restore); }

  _endCinematic(restore) {
    const c = this._cin;
    if (!c || c.done) return;
    c.done = true;
    this._cin = null;
    if (restore && !c.hold) this.setMode(this._prevMode || 'foot', { blend: 0.6 });
    c.resolve?.();
  }

  // --------------------------------------------------------------------------------------------- frame
  update(dt) {
    if (!this.enabled) return;
    const { ctx } = this;
    const cam = this.camera;
    if (!(dt > 0)) return; // paused: freeze the view
    this._t += dt;
    this._updateFocus(dt);
    const input = ctx.input;

    // C toggles the seated view
    if (ctx.control === 'car' && input?.pressed?.('KeyC') && (this.mode === 'car-chase' || this.mode === 'car-cockpit')) {
      this.setMode(this.mode === 'car-chase' ? 'car-cockpit' : 'car-chase', { blend: 0.35 });
    }

    // hold Q: look back over the shoulder (cockpit) / swing the chase camera round (see what is coming down behind)
    const lbWant = ctx.control === 'car' && (this.mode === 'car-chase' || this.mode === 'car-cockpit') && !!input?.down?.('KeyQ') ? 1 : 0;
    this.lookBack = lerp(this.lookBack || 0, lbWant, dampK(lbWant ? 7 : 5, dt));
    let fovTarget = this.fovBase;
    let ok = false;
    switch (this.mode) {
      case 'foot': ok = this._foot(dt); if (ok) fovTarget += this._footFov; break;
      case 'car-cockpit': ok = this._cockpit(dt, input); if (ok) fovTarget += this._speedFov(); break;
      case 'car-chase': ok = this._chase(dt, input); if (ok) fovTarget += this._speedFov() * 1.2; break;
      case 'cinematic': ok = this._cinematic(dt); if (ok && this._cinFov) fovTarget = this._cinFov; break;
    }
    if (!ok) { this.pos.copy(cam.position); this.quat.copy(cam.quaternion); }

    // blend from the previous pose
    if (this._blendT < this._blendDur) {
      this._blendT += dt;
      const k = smooth01(0, 1, this._blendT / this._blendDur);
      this.pos.lerpVectors(this._fromPos, this.pos, k);
      this.quat.slerpQuaternions(this._fromQuat, this.quat, k);
    }

    // trauma shake + continuous ground rumble
    this.trauma = Math.max(0, this.trauma - dt * 0.75);
    const rumble = clamp(ctx.landslide?.rumble ?? 0, 0, 1);
    const bump = this.mode.startsWith('car') ? clamp((ctx.car?.bump ?? 0) * 0.35, 0, 0.35) : 0;
    const amt = this.trauma * this.trauma + rumble * rumble * 0.22 + bump * bump;
    cam.position.copy(this.pos);
    cam.quaternion.copy(this.quat);
    if (amt > 1e-4) {
      const t = this._t * 16;
      const posScale = this.mode === 'car-cockpit' ? 0.25 : 1;
      _e.set(0.05 * amt * noise(t, 1), 0.035 * amt * noise(t, 2), 0.07 * amt * noise(t, 3), 'YXZ');
      cam.quaternion.multiply(_q1.setFromEuler(_e));
      _v1.set(noise(t, 4), noise(t, 5), noise(t, 6)).multiplyScalar(0.14 * amt * posScale).applyQuaternion(cam.quaternion);
      cam.position.add(_v1);
      fovTarget += amt * 2;
    }

    // kick spring (critically damped-ish, ~0.2 s settle); velocities are impulses in rad/s x 60
    const K = this._kick;
    if (K.vx || K.vy || K.vz || K.x || K.y || K.z) {
      const w = 26, z = 0.55;
      for (const a of ['x', 'y', 'z']) {
        const va = 'v' + a;
        K[va] += (-w * w * K[a] - 2 * z * w * K[va]) * dt;
        K[a] += K[va] * dt;
        if (Math.abs(K[a]) < 1e-5 && Math.abs(K[va]) < 1e-4) { K[a] = 0; K[va] = 0; }
      }
      _e.set(K.x, K.y, K.z, 'YXZ');
      cam.quaternion.multiply(_q1.setFromEuler(_e));
    }

    // FOV
    this.fov = lerp(this.fov, fovTarget, dampK(this.mode === 'cinematic' ? 20 : 2.5, dt));
    if (Math.abs(cam.fov - this.fov) > 0.01) { cam.fov = this.fov; cam.updateProjectionMatrix(); }
    cam.updateMatrixWorld(true);

    this.viewmodel.update(dt, this.mode === 'foot' && this._blendT >= this._blendDur * 0.7 && ctx.control === 'foot', this);
  }

  _speedFov() {
    const v = Math.abs(this.ctx.car?.speed ?? 0);
    return smooth01(6, 36, v) * 9;
  }

  // --------------------------------------------------------------------------------------------- foot
  _foot(dt) {
    const pl = this.ctx.player;
    if (!pl?.eye) return false;
    const vel = pl.velocity || _v2.set(0, 0, 0);
    const spd = Math.hypot(vel.x, vel.z);
    const grounded = pl.onGround !== false;
    // stride-synced bob: one half-cycle of the phase per footstep (character.js strides 0.62 + 0.14 v meters)
    const stride = 0.62 + 0.14 * spd;
    if (grounded && spd > 0.2) this._bobPhase += (spd * dt / stride) * Math.PI;
    const target = grounded ? smooth01(0.2, 1.4, spd) : 0;
    this._bobAmt = lerp(this._bobAmt, target, dampK(8, dt));
    const run = smooth01(2.0, 5.0, spd);
    const Av = (0.028 + 0.03 * run) * this._bobAmt, Ah = (0.018 + 0.012 * run) * this._bobAmt;
    const ph = this._bobPhase;
    const bobY = Av * (0.5 - 0.5 * Math.cos(2 * ph)) - Av * 0.5;
    const bobX = Ah * Math.sin(ph);
    // landing dip (spring)
    this._dipV += (-this._dip * 90 - this._dipV * 13) * dt;
    this._dip += this._dipV * dt;
    this._dip = clamp(this._dip, -0.35, 0.1);
    // breathing (heavier when out of stamina)
    this._breathT += dt * (0.24 + (1 - (pl.stamina ?? 1)) * 0.35) * Math.PI * 2;
    const breath = (0.0035 + (1 - (pl.stamina ?? 1)) * 0.006) * Math.sin(this._breathT);
    // strafe roll + step roll
    const yaw = pl.yaw ?? 0;
    const rx = Math.cos(yaw), rz = -Math.sin(yaw);
    const side = vel.x * rx + vel.z * rz;
    this._roll = lerp(this._roll, -side * 0.006, dampK(6, dt));
    const roll = this._roll + 0.004 * Math.sin(ph) * this._bobAmt;

    _e.set((pl.pitch ?? 0) + breath + this._dip * 0.25 + bobY * 0.15, yaw, roll, 'YXZ');
    this.quat.setFromEuler(_e);
    this.pos.copy(pl.eye);
    this.pos.y += bobY + this._dip + breath * 0.4;
    this.pos.x += rx * bobX; this.pos.z += rz * bobX;
    this._footFov = (pl.sprinting ? 4 : 0) * smooth01(3, 5, spd);
    return true;
  }

  // --------------------------------------------------------------------------------------------- cockpit
  _cockpit(dt, input) {
    const car = this.ctx.car;
    if (!car?.object) return false;
    const ctrl = this.ctx.control === 'car';
    const sens = input?.sensitivity ?? 0.0022;
    const mdx = ctrl ? (input?.dx || 0) : 0, mdy = ctrl ? (input?.dy || 0) : 0;
    if (mdx || mdy || (ctrl && input?.lookHeld)) this._mouseIdle = 0; else this._mouseIdle += dt; // lookHeld: a finger resting on the touch look area
    this.lookYaw = clamp(this.lookYaw - mdx * sens, -2.1, 2.1);
    this.lookPitch = clamp(this.lookPitch - mdy * sens * (input?.invertY ? -1 : 1), -0.95, 0.75);
    const v = Math.abs(car.speed || 0);
    if (this._mouseIdle > 1.4 && v > 2.5) {
      const k = dampK(1.6, dt);
      this.lookYaw = lerp(this.lookYaw, 0, k); this.lookPitch = lerp(this.lookPitch, -0.04, k);
    }
    const seat = car.seatCam;
    if (seat?.parent) seat.getWorldPosition(this.pos);
    else this.pos.set(0.36, 1.32, -0.15).applyMatrix4(car.object.matrixWorld);
    // head sway from G-forces (head lags the body), in car-local axes
    const lat = clamp(car.lateralG || 0, -1.2, 1.2), lon = clamp(car.longG || 0, -1.2, 1.2);
    _v1.set(lat * 0.035, -Math.min(0.02, (car.bump || 0) * 0.01), -lon * 0.04);
    this._headOff.lerp(_v1, dampK(5, dt));
    this.pos.add(_v2.copy(this._headOff).applyQuaternion(car.object.quaternion));
    // keep the horizon a bit steadier than the body (neck reflex): remove 35 % of the body roll
    const cq = car.object.quaternion;
    _v3.set(0, 0, 1).applyQuaternion(cq);
    const bodyYaw = Math.atan2(_v3.x, _v3.z);
    _q2.setFromAxisAngle(Y, bodyYaw);
    const levelled = _q2.slerp(cq, 0.65); // yaw-only -> body: 65 % of the body pitch/roll remains
    const lb = this.lookBack || 0;
    _e.set(lerp(this.lookPitch, 0.02, lb), lerp(this.lookYaw, 2.55, lb), 0, 'YXZ');
    this.quat.copy(levelled).multiply(Q_FLIP).multiply(_q1.setFromEuler(_e));
    return true;
  }

  // --------------------------------------------------------------------------------------------- chase
  _chase(dt, input) {
    const { ctx } = this;
    const car = ctx.car;
    if (!car?.object) return false;
    const ctrl = ctx.control === 'car';
    const sens = input?.sensitivity ?? 0.0022;
    const mdx = ctrl ? (input?.dx || 0) : 0, mdy = ctrl ? (input?.dy || 0) : 0;
    if (mdx || mdy || (ctrl && input?.lookHeld)) this._mouseIdle = 0; else this._mouseIdle += dt; // lookHeld: a finger resting on the touch look area
    this.orbitYaw = wrapAngle(this.orbitYaw - mdx * sens * 1.2);
    this.orbitPitch = clamp(this.orbitPitch - mdy * sens, -0.25, 0.9);
    const v = car.speed || 0;
    if (this._mouseIdle > 1.2) {
      const k = dampK(Math.abs(v) > 2 ? 1.8 : 0.4, dt);
      this.orbitYaw = lerp(this.orbitYaw, 0, k); this.orbitPitch = lerp(this.orbitPitch, 0, k);
    }
    const cp = car.object.position;
    const fwd = _v1.set(0, 0, 1).applyQuaternion(car.object.quaternion);
    fwd.y = 0; if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, 1); fwd.normalize();
    let heading = Math.atan2(fwd.x, fwd.z);
    // follow the velocity direction when sliding
    const vel = car.velocity ? car.velocity(_v4) : null;
    if (vel && Math.hypot(vel.x, vel.z) > 4 && v > 0) heading = wrapAngle(heading + wrapAngle(Math.atan2(vel.x, vel.z) - heading) * 0.35);
    if (this._chaseYaw === null || !this._chaseInit) this._chaseYaw = heading;
    this._chaseYaw = wrapAngle(this._chaseYaw + wrapAngle(heading - this._chaseYaw) * dampK(3.2, dt));
    const lb = this.lookBack || 0;
    const yaw = this._chaseYaw + lerp(this.orbitYaw, Math.PI, lb);
    const pitch = 0.16 + this.orbitPitch;
    const dist = 5.4 + clamp(Math.abs(v), 0, 30) * 0.035;
    const pivot = _v2.set(cp.x, cp.y + 1.35, cp.z);
    const desired = _v3.set(-Math.sin(yaw) * Math.cos(pitch) * dist, Math.sin(pitch) * dist + 0.35, -Math.cos(yaw) * Math.cos(pitch) * dist).add(pivot);
    // critically-damped spring toward the desired point
    if (!this._chaseInit) { this._chasePos.copy(desired); this._chaseVel.set(0, 0, 0); this._chaseInit = true; }
    const w = 9;
    _v4.subVectors(desired, this._chasePos).multiplyScalar(w * w).addScaledVector(this._chaseVel, -2 * w);
    this._chaseVel.addScaledVector(_v4, dt);
    this._chasePos.addScaledVector(this._chaseVel, dt);
    // never lag more than ~3 m from the ideal point (teleports, big hits)
    if (this._chasePos.distanceTo(desired) > 3) this._chasePos.lerp(desired, 0.5);
    this.pos.copy(this._chasePos);
    // collision: pull in toward the pivot so the camera never sits inside the cut face / terrain / tunnel
    const phys = ctx.physics;
    if (phys?.world) {
      _v4.subVectors(this.pos, pivot);
      const len = _v4.length();
      if (len > 0.01) {
        _v4.multiplyScalar(1 / len);
        let hit = null;
        try { hit = phys.raycast(pivot, _v4, len + 0.35, { groups: groups(G.ALL, G.STATIC), excludeBody: car.body }); } catch { hit = null; }
        if (hit) {
          const d = Math.max(0.6, hit.distance - 0.35);
          this.pos.copy(pivot).addScaledVector(_v4, d);
          this._chasePos.lerp(this.pos, 0.5);
        }
      }
      // ground clamp
      let g = null;
      try { g = phys.raycast(_v4.set(this.pos.x, this.pos.y + 1.5, this.pos.z), _v1.set(0, -1, 0), 3.0, { groups: groups(G.ALL, G.STATIC) }); } catch { g = null; }
      if (g && this.pos.y < g.point.y + 0.45) this.pos.y = g.point.y + 0.45;
    }
    // look-ahead target
    const ahead = (1.5 + clamp(v, 0, 30) * 0.22) * (1 - lb * 1.4);
    const fw = _v1.set(Math.sin(this._chaseYaw), 0, Math.cos(this._chaseYaw));
    const target = _v4.set(cp.x, cp.y + 1.05, cp.z).addScaledVector(fw, ahead);
    _m.lookAt(this.pos, target, Y);
    this.quat.setFromRotationMatrix(_m);
    return true;
  }

  // --------------------------------------------------------------------------------------------- cinematic
  _cinematic(dt) {
    const c = this._cin;
    this._cinFov = null;
    if (!c) return false; // hold the last pose
    c.t += dt;
    let k = c.t / c.dur;
    if (c.loop) k = k % 1;
    else if (k >= 1) k = 1;
    const e = c.ease ? c.ease(k) : k;
    let r = null;
    try { r = c.fn(e, this._cinPos, this._cinTarget, this.ctx); } catch (err) { console.warn('[cameraRig] cinematic path', err); this._endCinematic(true); return false; }
    this.pos.copy(this._cinPos);
    _m.lookAt(this._cinPos, this._cinTarget, Y);
    this.quat.setFromRotationMatrix(_m);
    if (r?.roll) this.quat.multiply(_q1.setFromAxisAngle(_v1.set(0, 0, 1), r.roll));
    if (r?.fov) this._cinFov = r.fov;
    if (!c.loop && c.t >= c.dur) this._endCinematic(true);
    return true;
  }

  dispose() { this.viewmodel.dispose(); }
}


// ================================================================================================= viewmodel
// First-person hands (assets/models/hands.glb, built by tools/blender/hands.py): gloved hands + rain-jacket forearms,
// one skinned mesh per arm. Bones <S>_forearm -> <S>_hand -> fingers; bone-local frames are Blender's (Y along the
// bone, Z = back of the hand, X = lateral), so hand frames below are given as (finger direction, back-of-hand normal).
// Finger poses come from the armature extras (userData.poses: pose -> bone -> pose-basis quaternion, applied as
// rest * basis) and the grip sockets (userData.sockets: the cylinder each grip pose was closed onto, in hand-bone
// space), so held items sit exactly where the fingers wrap.
//
// Rendering: the viewmodel group is a camera child scaled toward the eye by VM.k (a uniform scale about the eye is
// invisible in perspective but keeps everything within ~0.3 m, so it never clips into the trunk, the car or a wall),
// and x/y by tan(camFov/2)/tan(VM.fov/2) so it is drawn with its own narrower, steady FOV (no stretching toward the
// screen edge, no breathing with the sprint FOV kick). Items are the props' own prototypes (same materials).
//
// API (sequence.js drives it):
//   setHeld('hatchet' | 'jerrycan' | 'planks' | null)   what the hands carry (swapped with a lower/raise)
//   windup(0..1), strike() -> Promise (resolves when the blade meets the wood), pour(p 0..1 | null),
//   place(p 0..1 | null), reach(worldPos) -> Promise (resolves when the hand closes on the item), ready, debug()
const VM = {
  k: 0.4,                 // anti-clip scale toward the eye
  fov: 56,                // viewmodel vertical FOV (deg)
  upper: 0.30,            // upper arm (m); only used to aim the forearm, the upper arm is never drawn
  shoulderR: new THREE.Vector3(0.175, -0.215, 0.09),   // body space (eye at the origin, yaw frame)
  shoulderL: new THREE.Vector3(-0.175, -0.215, 0.09),
};
const V3 = () => new THREE.Vector3();
const Q4 = () => new THREE.Quaternion();
const _hq = { a: V3(), b: V3(), c: V3(), m: new THREE.Matrix4() };
const ease = (t) => t * t * (3 - 2 * t);
const easeIn = (t) => t * t * t;
const ONE = new THREE.Vector3(1, 1, 1);
const ZAX = new THREE.Vector3(0, 0, 1);

/** Quaternion of a hand bone whose fingers point along `fwd`, back of the hand toward `dorsal`. */
function handQuat(fwd, dorsal, out) {
  const y = _hq.a.copy(fwd).normalize();
  const z = _hq.b.copy(dorsal).addScaledVector(y, -dorsal.dot(y));
  if (z.lengthSq() < 1e-8) z.set(0, 1, 0).addScaledVector(y, -y.y);
  z.normalize();
  const x = _hq.c.crossVectors(y, z);
  _hq.m.makeBasis(x, y, z);
  return out.setFromRotationMatrix(_hq.m);
}
/** Quaternion whose local +Y lies along `ay` and local +Z toward `az` (orthogonalised). */
function axesQuat(ay, az, out) {
  const y = _hq.a.copy(ay).normalize();
  const z = _hq.b.copy(az).addScaledVector(y, -az.dot(y)).normalize();
  const x = _hq.c.crossVectors(y, z);
  _hq.m.makeBasis(x, y, z);
  return out.setFromRotationMatrix(_hq.m);
}

class VMHand {
  constructor(side, armRoot, data) {
    this.side = side;
    this.root = armRoot;
    this.bones = {};
    armRoot.traverse((o) => { if (o.isBone) this.bones[o.name] = o; if (o.isSkinnedMesh) this.mesh = o; });
    this.fore = this.bones[side + '_forearm'];
    this.hand = this.bones[side + '_hand'];
    this.foreLen = this.hand ? this.hand.position.length() : 0.27;
    this.rest = {};
    this.cur = {};
    this.fingers = [];
    for (const [n, b] of Object.entries(this.bones)) {
      this.rest[n] = b.quaternion.clone();
      if (n !== side + '_forearm' && n !== side + '_hand') { this.fingers.push(n); this.cur[n] = Q4(); }
    }
    this.poses = {};
    for (const [pn, tbl] of Object.entries(data.poses || {})) {
      const m = {};
      for (const [bn, q] of Object.entries(tbl)) m[bn] = new THREE.Quaternion(q[0], q[1], q[2], q[3]);
      this.poses[pn] = m;
    }
    this.sockets = {};
    for (const [sn, S] of Object.entries(data.sockets || {})) {
      this.sockets[sn] = { p: new THREE.Vector3().fromArray(S.p), d: new THREE.Vector3().fromArray(S.d).normalize(), r: S.r };
    }
    this.pose = 'rest';
    this.poseRate = 14;
    this.pos = new THREE.Vector3(0, -0.7, -0.3);      // hand-bone (wrist) position in viewmodel space
    this.quat = Q4();
    this.vis = 0;
    this.t = { a: V3(), b: V3(), c: V3(), d: V3(), q: Q4(), m: new THREE.Matrix4() };
  }

  /** Matrix of an item held in this hand, in hand-bone space: the item's local +Y lies along the socket cylinder
   *  toward the thumb side (-d), its +Z toward the finger direction, and the item point (0, -slide, 0) sits on the
   *  socket centre (i.e. the item slides `slide` m toward the thumb side). */
  itemInHand(socket, slide = 0) {
    const S = this.sockets[socket] || { p: new THREE.Vector3(0, 0.08, -0.033), d: new THREE.Vector3(this.side === 'R' ? 1 : -1, 0, 0) };
    const up = V3().copy(S.d).negate();
    const q = axesQuat(up, V3().set(0, 1, 0), Q4());
    const o = V3().copy(S.p).addScaledVector(up, slide);
    return new THREE.Matrix4().compose(o, q, ONE);
  }

  applyFingers(dt) {
    const P = this.poses[this.pose] || this.poses.rest;
    const k = dampK(this.poseRate, dt);
    for (const n of this.fingers) {
      const t = P?.[n];
      if (t) this.cur[n].slerp(t, k);
      this.bones[n].quaternion.copy(this.rest[n]).multiply(this.cur[n]);
    }
  }

  /** Aims the forearm (two-bone IK through an invisible upper arm) so the wrist lands on this.pos with the hand at
   *  this.quat. The sleeve is rigid on the forearm, which takes the hand's roll (the cuff hides the wrist). */
  solve(shoulder, pole) {
    if (!this.fore || !this.hand) return;
    const T = this.t;
    const W = this.pos, L1 = VM.upper, L2 = this.foreLen;
    const d = T.a.subVectors(W, shoulder);
    const dist = d.length();
    const E = T.b;
    if (dist >= L1 + L2 - 1e-4 || dist < 1e-4) E.copy(W).addScaledVector(d.normalize(), -L2);
    else {
      const cosA = clamp((L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist), -1, 1);
      const sinA = Math.sqrt(1 - cosA * cosA);
      d.multiplyScalar(1 / dist);
      const pp = T.c.copy(pole).addScaledVector(d, -pole.dot(d)).normalize();
      E.copy(shoulder).addScaledVector(d, L1 * cosA).addScaledVector(pp, L1 * sinA);
    }
    const hy = T.c.set(0, 1, 0).applyQuaternion(this.quat);
    const fy = T.d.subVectors(W, E).normalize();
    // the wrist bends at most ~41 deg (the glove gauntlet stays inside the cuff); beyond that the elbow gives
    const cb = clamp(fy.dot(hy), -1, 1), ang = Math.acos(cb), lim = 0.72;
    if (ang > lim) {
      fy.lerp(hy, (ang - lim) / ang).normalize();
      E.copy(W).addScaledVector(fy, -L2);
    }
    const fz = T.a.set(0, 0, 1).applyQuaternion(this.quat);
    fz.addScaledVector(fy, -fz.dot(fy));
    if (fz.lengthSq() < 1e-6) fz.set(0, 1, 0).addScaledVector(fy, -fy.y);
    fz.normalize();
    const fx = T.c.crossVectors(fy, fz);
    T.m.makeBasis(fx, fy, fz);
    this.fore.position.copy(E);
    this.fore.quaternion.setFromRotationMatrix(T.m);
    this.hand.quaternion.copy(this.fore.quaternion).invert().multiply(this.quat);
  }
}

class Viewmodel {
  constructor(ctx) {
    this.ctx = ctx;
    this.group = new THREE.Group();
    this.group.name = 'viewmodel';
    this.group.visible = false;
    this.root = new THREE.Group();          // true camera-space metres inside the scaled group
    this.root.name = 'viewmodel_root';
    this.group.add(this.root);
    ctx.camera.add(this.group);
    this.hands = {};
    this.items = {};
    this.held = null;       // requested by the game
    this.shown = null;      // currently in the hands
    this.vis = 0;           // 0 lowered .. 1 up (swap)
    this._wind = 0; this._windT = 0;
    this._strike = null;
    this._pouring = false; this._pourT = 0; this._pourW = 0;
    this._placing = false; this._placeT = 0; this._placeW = 0;
    this._reach = null;
    this._sway = V3(); this._swayV = V3();
    this._t = 0;
    this.ready = false;
    this._attach = null;
    this._bodyQ = Q4();
    this._sh = { R: V3(), L: V3() };
    // elbow pole (body space): the elbows hang down, a little out and back beside the torso
    this._poleBR = new THREE.Vector3(0.40, -0.80, 0.45).normalize();
    this._poleBL = new THREE.Vector3(-0.40, -0.80, 0.45).normalize();
    this._poleR = V3(); this._poleL = V3();
    // private temporaries (never shared with the module helpers)
    this.v = Array.from({ length: 10 }, V3);
    this.q = Array.from({ length: 6 }, Q4);
    this.m = Array.from({ length: 3 }, () => new THREE.Matrix4());
    this.e = new THREE.Euler();
  }

  async init() {
    const { ctx } = this;
    let gltf = null;
    try { gltf = await ctx.assets.gltf('assets/models/hands.glb'); } catch (e) { console.warn('[viewmodel] hands.glb', e); }
    if (gltf?.scene) {
      for (const S of ['R', 'L']) {
        const arm = gltf.scene.getObjectByName('hand_' + S);
        if (!arm) continue;
        const data = {};
        try { data.poses = JSON.parse(arm.userData?.poses || '{}'); } catch { data.poses = {}; }
        try { data.sockets = JSON.parse(arm.userData?.sockets || '{}'); } catch { data.sockets = {}; }
        arm.position.set(0, 0, 0); arm.quaternion.identity(); arm.scale.set(1, 1, 1);
        this.root.add(arm);
        const h = new VMHand(S, arm, data);
        if (!h.fore || !h.hand || !h.mesh) { console.warn('[viewmodel] hand rig incomplete', S); this.root.remove(arm); continue; }
        arm.traverse((o) => {
          if (!o.isMesh) return;
          o.frustumCulled = false; o.castShadow = false; o.receiveShadow = true;
          const m = o.material;
          if (m) { m.envMapIntensity = 1.0; for (const k of ['map', 'normalMap', 'roughnessMap']) if (m[k]) m[k].anisotropy = 4; }
        });
        arm.visible = false;
        this.hands[S] = h;
      }
    }
    // items: the props' own prototypes (materials shared with the world copies)
    const props = ctx.props;
    const mk = (name) => {
      let o = null;
      try { o = props?.instance?.(name, { cast: false, recv: true }); } catch { o = null; }
      if (!o || !o.children?.length) return null;
      o.traverse((m) => { if (m.isMesh) { m.frustumCulled = false; m.castShadow = false; m.receiveShadow = true; } });
      return o;
    };
    const hat = mk('hatchet') || await buildHatchet(ctx).catch(() => null);
    if (hat) { hat.visible = false; this.root.add(hat); this.items.hatchet = hat; }
    const jc = mk('jerrycan');
    if (jc) { jc.visible = false; this.root.add(jc); this.items.jerrycan = jc; }
    const pl = new THREE.Group();
    for (let i = 0; i < 3; i++) {
      const b = mk('plank');
      if (!b) break;
      // three boards stacked flat, a little out of line
      b.position.set(i * 0.035 - 0.03, i * 0.0385, (i === 1 ? 0.004 : -0.003));
      b.rotation.set(0, (i - 1) * 0.004, 0);
      pl.add(b);
    }
    if (pl.children.length) { pl.visible = false; this.root.add(pl); this.items.planks = pl; }
    this.ready = true;
  }

  // ------------------------------------------------------------------------------------------- API
  setHeld(id) { this.held = id && this.items[id] ? id : null; }
  windup(p) { this._windT = clamp(p, 0, 1); }
  strike() {
    return new Promise((resolve) => {
      if (!this.ready || this.shown !== 'hatchet' || !this.hands.R) { resolve(); return; }
      if (this._strike) this._strike.resolve?.();
      this._strike = { t: 0, resolve, hit: false, from: null };
    });
  }
  pour(p) { this._pouring = p != null; this._pourT = p == null ? 0 : clamp(p, 0, 1); }
  place(p) { this._placing = p != null; this._placeT = p == null ? 0 : clamp(p, 0, 1); }
  reach(worldPos) {
    return new Promise((resolve) => {
      if (!this.ready || !worldPos || !this.group.parent) { resolve(); return; }
      if (this._reach) this._reach.resolve?.();
      const side = this.shown ? 'L' : 'R';
      if (!this.hands[side]) { resolve(); return; }
      this._reach = { t: 0, target: worldPos.clone(), side, resolve, closed: false };
    });
  }
  debug() {
    return { ready: this.ready, held: this.held, shown: this.shown, vis: +this.vis.toFixed(2), hands: Object.keys(this.hands), items: Object.keys(this.items),
      R: this.hands.R ? { vis: this.hands.R.vis, pos: this.hands.R.pos.toArray().map((x) => +x.toFixed(3)), pose: this.hands.R.pose } : null,
      L: this.hands.L ? { vis: this.hands.L.vis, pose: this.hands.L.pose } : null };
  }

  _endAll() {
    if (this._strike) { this._strike.resolve?.(); this._strike = null; }
    if (this._reach) { this._reach.resolve?.(); this._reach = null; }
  }

  // ------------------------------------------------------------------------------------------- frame
  update(dt, active, rig) {
    if (!this.ready) return;
    this._t += dt;
    const { ctx } = this;
    const cam = ctx.camera;
    if (!active) {
      if (this.group.visible || this.shown) {
        this.group.visible = false;
        for (const it of Object.values(this.items)) it.visible = false;
        for (const H of Object.values(this.hands)) { H.root.visible = false; H.vis = 0; }
      }
      this.vis = 0; this.shown = null;
      this._endAll();
      return;
    }
    // anti-clip scale + own FOV
    const s = Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2) / Math.tan(THREE.MathUtils.degToRad(VM.fov) / 2);
    this.group.scale.set(VM.k * s, VM.k * s, VM.k);
    // body frame (yaw only) -> viewmodel space: the arms hang from the torso, not from the head
    const yaw = ctx.player?.yaw ?? 0;
    this.q[0].setFromAxisAngle(Y, yaw);
    this._bodyQ.copy(cam.quaternion).invert().multiply(this.q[0]);
    this._sh.R.copy(VM.shoulderR).applyQuaternion(this._bodyQ);
    this._sh.L.copy(VM.shoulderL).applyQuaternion(this._bodyQ);
    this._poleR.copy(this._poleBR).applyQuaternion(this._bodyQ);
    this._poleL.copy(this._poleBL).applyQuaternion(this._bodyQ);

    // swap: lower the current item, switch, raise the new one (never while a reach or a blow is under way)
    const busy = !!(this._strike || this._reach);
    if (this.held !== this.shown) {
      if (!busy) {
        this.vis = Math.max(0, this.vis - dt / 0.22);
        if (this.vis <= 0) {
          if (this.shown && this.items[this.shown]) this.items[this.shown].visible = false;
          this.shown = this.held;
          this._wind = 0; this._attach = null;
          this._pouring = false; this._pourW = 0; this._placing = false; this._placeW = 0;
        }
      }
    } else if (this.shown) this.vis = Math.min(1, this.vis + dt / 0.36);
    const R = this.hands.R, L = this.hands.L;
    const item = this.shown ? this.items[this.shown] : null;
    // mouse sway: the arms lag the view a little (damped spring), like real arms with inertia
    const input = ctx.input;
    const mdx = ctx.control === 'foot' ? (input?.dx || 0) : 0, mdy = ctx.control === 'foot' ? (input?.dy || 0) : 0;
    const sv = this.v[0].set(clamp(-mdx * 0.0005, -0.03, 0.03), clamp(mdy * 0.0005, -0.03, 0.03), 0);
    this._swayV.addScaledVector(sv, 18).addScaledVector(this._sway, -140 * dt);
    this._swayV.multiplyScalar(Math.exp(-16 * dt));
    this._sway.addScaledVector(this._swayV, dt);
    const c = { bob: rig._bobAmt || 0, ph: rig._bobPhase || 0, breath: Math.sin(this._t * 1.35) * 0.0025, lower: (1 - ease(this.vis)) * 0.42 };
    let rOn = false, lOn = false;
    this._lUsed = false;
    if (R) R.pose = 'rest';
    if (L) L.pose = 'rest';
    if (item) item.visible = this.vis > 0.01;
    if (R && item) {
      if (this.shown === 'hatchet') this._hatchet(dt, R, L, c);
      else if (this.shown === 'jerrycan') this._jerrycan(dt, R, L, c);
      else if (this.shown === 'planks') this._planks(dt, R, L, item, c);
      rOn = true; lOn = this._lUsed;
    }
    if (this._strike && this.shown !== 'hatchet') { this._strike.resolve?.(); this._strike = null; }
    // pickup reach with the free hand
    if (this._reach) {
      const H = this._reach.side === 'L' ? L : R;
      if (H && !(H === R && rOn) && !(H === L && lOn)) { this._reachAnim(dt, H, this._reach); if (H === R) rOn = true; else lOn = true; }
      else { this._reach.resolve?.(); this._reach = null; }
    }
    // hands: visibility, fingers, IK
    for (const [S, H] of Object.entries(this.hands)) {
      const on = S === 'R' ? rOn : lOn;
      H.vis = on ? Math.min(1, H.vis + dt * 6) : Math.max(0, H.vis - dt * 4);
      if (!on && H.vis <= 0) { H.root.visible = false; continue; }
      if (!on) H.pos.y -= dt * 1.6;                     // an idle hand drops out of view
      H.root.visible = true;
      H.applyFingers(dt);
      H.solve(S === 'R' ? this._sh.R : this._sh.L, S === 'R' ? this._poleR : this._poleL);
    }
    // an item carried in a hand follows the hand frame (the IK lands the hand bone exactly on H.pos / H.quat)
    if (item && item.visible && this._attach) {
      const A = this._attach;
      this.m[0].compose(A.hand.pos, A.hand.quat, ONE).multiply(A.m);
      this.m[0].decompose(item.position, item.quaternion, item.scale);
    }
    this.group.visible = !!(R?.root.visible || L?.root.visible || item?.visible);
  }

  // ---- helpers ------------------------------------------------------------------------------------------------
  /** Hand H placed so that the item (root-space matrix itemM) sits in its grip (m = item matrix in hand space). */
  _handFromItem(H, itemM, m) {
    this.m[1].copy(m).invert();
    this.m[2].copy(itemM).multiply(this.m[1]);
    this.m[2].decompose(H.pos, H.quat, this.v[9]);
  }

  /** walk bob (figure-eight), breathing, mouse sway, swap lowering */
  _layer(pos, quat, c, amt = 1) {
    pos.x += (Math.sin(c.ph) * 0.010 * c.bob + this._sway.x * 0.6) * amt;
    pos.y += (-Math.abs(Math.cos(c.ph)) * 0.014 * c.bob + 0.007 * c.bob + c.breath + this._sway.y * 0.6) * amt - c.lower;
    this.e.set(this._sway.y * 1.4 * amt + c.breath * 0.8, this._sway.x * 1.6 * amt, Math.sin(c.ph) * 0.02 * c.bob * amt, 'YXZ');
    quat.premultiply(this.q[5].setFromEuler(this.e));
  }

  // ---- hatchet: idle hold, two-handed wind-up over the right shoulder, strike, recoil ---------------------------
  _hatchet(dt, R, L, c) {
    const [P0, F0, D0, P1, F1, D1, PI, tmp] = this.v;
    const [qIdle, qWind, qImp] = this.q;
    if (!this._hatM) this._hatM = R.itemInHand('fist', 0.043);      // butt ~1 cm past the little finger
    this._attach = { hand: R, m: this._hatM };
    R.pose = 'fist';
    this._wind = lerp(this._wind, this._windT, dampK(this._windT > this._wind ? 7 : 12, dt));
    const w = ease(clamp(this._wind, 0, 1));
    // key poses (viewmodel space): idle low at the right with the head up and forward; wind-up raised by the ear
    // (the blade faces the hand's finger direction and the head lies along back-of-hand x fingers)
    P0.set(0.18, -0.175, -0.38); F0.set(-0.12, -0.60, -1.0); D0.set(1.0, 0.05, 0.12);
    P1.set(0.265, 0.035, -0.40); F1.set(0.08, 1.0, 0.18); D1.set(1.0, -0.05, -0.22);
    handQuat(F0, D0, qIdle); handQuat(F1, D1, qWind);
    R.pos.lerpVectors(P0, P1, w);
    R.quat.slerpQuaternions(qIdle, qWind, w);
    let lUse = w;
    const S = this._strike;
    if (S) {
      S.t += dt;
      const tS = 0.105, tR = 0.17, tE = 0.62;
      if (!S.from) S.from = { p: R.pos.clone(), q: R.quat.clone() };
      // impact (the player looks at the cut): forearm pointing away from the eye, handle upright on screen, the blade
      // driven into the screen toward the notch, the head at the centre
      PI.set(0.05, -0.30, -0.56);
      handQuat(tmp.set(-0.05, 0.10, -1.0), this.v[8].set(1.0, 0.0, 0.05), qImp);
      if (S.t < tS) {
        const u = easeIn(S.t / tS) * 0.85 + (S.t / tS) * 0.15;
        R.pos.lerpVectors(S.from.p, PI, u); R.quat.slerpQuaternions(S.from.q, qImp, u);
        lUse = 1;
      } else {
        if (!S.hit) {
          S.hit = true;
          this.ctx.cameraRig?.kick?.(-0.55, 0.12, -0.22);   // the blow jars the view
          S.resolve?.();
        }
        // recoil: the blade bounces back a few cm, then the arm recovers to the idle hold
        const r = S.t < tR ? Math.sin(((S.t - tS) / (tR - tS)) * Math.PI) : 0;
        const back = smooth01(tR, tE, S.t);
        tmp.copy(PI); tmp.y += r * 0.02; tmp.z += r * 0.035;
        R.pos.lerpVectors(tmp, P0, back); R.quat.slerpQuaternions(qImp, qIdle, back);
        lUse = 1 - smooth01(tR, tE * 0.8, S.t);
        if (S.t >= tE) { this._strike = null; this._windT = 0; this._wind = 0; }
      }
    }
    this._layer(R.pos, R.quat, c, 1 - w * 0.6);
    // supporting left hand on the handle just above the right fist (toward the head) while winding / striking
    this._lUsed = !!L && lUse > 0.02;
    if (this._lUsed) {
      if (!this._hatML) this._hatML = L.itemInHand('fist', 0.043 - 0.095);   // one hand-width up the handle
      L.pose = 'fist';
      this.m[0].compose(R.pos, R.quat, ONE).multiply(this._hatM);
      this._handFromItem(L, this.m[0], this._hatML);
      L.pos.y -= (1 - ease(clamp(lUse * 1.6, 0, 1))) * 0.35;
    }
  }

  // ---- jerrycan: carried at the side (body space), pour pose while refuelling ------------------------------------
  _jerrycan(dt, R, L, c) {
    const [cp, cf, cd, tgt, canPos, hx, hz, fwd] = this.v;
    const [cq, qa, qb] = this.q;
    if (!this._jcM) {
      // the middle carrying bar (item +X, at y 0.478) lies in the hook; the can hangs on the finger-tip side
      const S = R.sockets.hook || { p: new THREE.Vector3(0, 0.1125, -0.0215), d: new THREE.Vector3(1, -0.22, 0).normalize() };
      const xb = V3().copy(S.d).normalize();
      const yb = V3().set(0, -1, 0).addScaledVector(xb, xb.y).normalize();
      const zb = V3().crossVectors(xb, yb);
      const q = Q4().setFromRotationMatrix(new THREE.Matrix4().makeBasis(xb, yb, zb));
      this._jcM = new THREE.Matrix4().compose(V3().copy(S.p).addScaledVector(yb, -0.478), q, ONE);
      // left hand under the rear bottom corner (item space), palm up, fingers round the edge
      const hq = handQuat(V3().set(-1, 0.3, 0), V3().set(0, -1, 0), Q4());
      this._jcML = new THREE.Matrix4().compose(V3().set(-0.12, -0.024, 0.0), hq, ONE).invert();
    }
    this._attach = { hand: R, m: this._jcM };
    this._pourW = lerp(this._pourW, this._pouring ? 1 : 0, dampK(this._pouring ? 4.5 : 3, dt));
    const pw = ease(this._pourW);
    R.pose = pw > 0.5 ? 'grip' : 'hook';
    // carry (body space): arm hanging at the right side, the can swinging a little with the stride
    const sw = Math.sin(c.ph) * 0.05 * c.bob;
    cp.set(0.24, -0.53 + Math.abs(Math.cos(c.ph)) * 0.01 * c.bob, -0.26 + sw).applyQuaternion(this._bodyQ);
    handQuat(cf.set(0.05, -1, -0.1 - sw * 0.8), cd.set(1, 0.02, 0.05), cq);
    cq.premultiply(this._bodyQ);
    R.pos.copy(cp); R.quat.copy(cq);
    this._lUsed = false;
    if (pw > 0.001) {
      // pour: can raised in front, the spout turned toward the filler neck and tipped with the progress
      tgt.set(-0.25, -0.55, -0.9);
      const fc = this.ctx.car?.fuelCap;
      if (fc?.getWorldPosition) {
        fc.getWorldPosition(fwd);
        // (camera metres, not root space: aiming at the filler's true offset turns the can three-quarters to the
        //  viewer with its handles and the gripping hand in view, which reads better than an exact on-screen aim)
        this.ctx.camera.worldToLocal(fwd);
        if (fwd.z < -0.05) tgt.copy(fwd);
      }
      // the can is seen from above: its top (handles) toward the eye, the spout end aimed at the filler neck; as the
      // pour goes on the rear end is lifted (the can tips about its depth axis) and it trembles a little
      canPos.set(0.17, -0.32, -0.74);
      hx.subVectors(tgt, canPos);
      if (hx.lengthSq() < 1e-6) hx.set(-0.2, 0.2, -1);
      hx.normalize();
      // world up in viewmodel space: the can stays upright in the world (tipped only by the pour)
      const upv = this.v[8].set(0, 1, 0).applyQuaternion(this.q[4].copy(this.ctx.camera.quaternion).invert());
      hz.crossVectors(hx, upv).normalize();
      upv.crossVectors(hz, hx).normalize();
      this.m[0].makeBasis(hx, upv, hz);
      const tip = lerp(0.42, -0.12, ease(this._pourT)) + Math.sin(this._t * 9) * 0.015 * this._pourT;
      qa.setFromRotationMatrix(this.m[0]).multiply(qb.setFromAxisAngle(ZAX, tip));
      // the spout (item +X end, top) sits ~0.2 m from the can centre: slide the can so the spout tip reaches the aim line
      this.m[0].compose(canPos, qa, ONE);
      this._handFromItem(R, this.m[0], this._jcM);
      R.pos.lerpVectors(cp, R.pos, pw);
      qb.copy(R.quat); R.quat.slerpQuaternions(cq, qb, pw);
      if (L && this._jcML) {
        // the left hand follows the can as it is (blended) now
        this.m[0].compose(R.pos, R.quat, ONE).multiply(this._jcM);
        L.pose = 'flat';
        this._handFromItem(L, this.m[0], this._jcML);
        L.pos.y -= (1 - pw) * 0.35;
        this._lUsed = pw > 0.02;
      }
    }
    this._layer(R.pos, R.quat, c, 0.35 + pw * 0.3);
    if (this._lUsed) this._layer(L.pos, L.quat, c, 0);
  }

  // ---- planks: on the right shoulder (body space), swung forward and lowered while placing ----------------------
  _planks(dt, R, L, item, c) {
    const [pos] = this.v;
    const [qa, qb] = this.q;
    this._placeW = lerp(this._placeW, this._placing ? this._placeT : 0, dampK(8, dt));
    const u = this._placeW;
    const lift = ease(clamp(u / 0.45, 0, 1)), down = ease(clamp((u - 0.4) / 0.6, 0, 1));
    // bundle frame (body space): +X along the boards (forward), resting on the shoulder
    // (carried well out on the shoulder, the front end raised a little: the boards stay out of the line of sight)
    pos.set(lerp(0.235, 0.06, lift), lerp(-0.17, -0.55, lift) - down * 0.55, lerp(0.42, -0.25, lift) - down * 0.25);
    pos.x += Math.sin(c.ph) * 0.006 * c.bob; pos.y += -Math.abs(Math.cos(c.ph)) * 0.012 * c.bob;
    const pitch = lerp(0.075, -0.02, lift) - down * 0.35 + Math.sin(c.ph * 2) * 0.004 * c.bob;
    this.e.set(0, Math.PI / 2 + lerp(-0.06, 0.0, lift), 0, 'YXZ');
    qa.setFromEuler(this.e).multiply(qb.setFromAxisAngle(ZAX, pitch)).premultiply(this._bodyQ);
    pos.applyQuaternion(this._bodyQ);
    pos.y -= c.lower;
    this.m[0].compose(pos, qa, ONE);
    this._attach = null;
    item.position.copy(pos); item.quaternion.copy(qa); item.scale.set(1, 1, 1);
    if (!this._plRM) {
      // right hand under the bundle ~0.6 m ahead of the shoulder, palm up, fingers hooked round the outer edge
      const hq = handQuat(V3().set(0, 0.55, 1), V3().set(0, -1, 0.2), Q4());
      this._plRM = new THREE.Matrix4().compose(V3().set(0.62, -0.03, 0.035), hq, ONE).invert();
      const hqL = handQuat(V3().set(0, 0.55, -1), V3().set(0, -1, -0.2), Q4());
      this._plLM = new THREE.Matrix4().compose(V3().set(1.05, -0.03, -0.03), hqL, ONE).invert();
    }
    R.pose = 'grip';
    this._handFromItem(R, this.m[0], this._plRM);
    this._lUsed = !!L && lift > 0.02;
    if (this._lUsed) {
      L.pose = 'grip';
      this._handFromItem(L, this.m[0], this._plLM);
      L.pos.y -= (1 - lift) * 0.4;
    }
  }

  // ---- pickup reach -------------------------------------------------------------------------------------------
  _reachAnim(dt, H, Rc) {
    const [tp, d, start, dors] = this.v;
    Rc.t += dt;
    const tOut = 0.30, tClose = 0.40, tEnd = 0.78;
    // target in viewmodel space: the farthest point on the eye->item line the hand can reach (so on screen the palm
    // lands on the item), wrist 8 cm short of the palm
    tp.copy(Rc.target);
    this.root.worldToLocal(tp);
    const sh = H.side === 'L' ? this._sh.L : this._sh.R;
    const dist = tp.length();
    d.copy(tp).multiplyScalar(1 / Math.max(dist, 1e-4));
    const us = d.dot(sh), disc = us * us - sh.lengthSq() + 0.62 * 0.62;
    const t = disc > 0 ? Math.min(dist, us + Math.sqrt(disc)) : Math.min(dist, 0.45);
    tp.copy(d).multiplyScalar(t);
    d.subVectors(tp, sh).normalize();
    tp.addScaledVector(d, -0.08);
    start.set(H.side === 'L' ? -0.22 : 0.22, -0.60, -0.25);
    if (Rc.t < tOut) {
      H.pos.lerpVectors(start, tp, ease(Rc.t / tOut));
      H.pose = 'open';
    } else if (Rc.t < tClose) {
      H.pos.copy(tp);
      H.pose = 'grip';
      if (!Rc.closed) { Rc.closed = true; Rc.resolve?.(); }
    } else {
      H.pos.lerpVectors(tp, start, ease((Rc.t - tClose) / (tEnd - tClose)));
      H.pose = 'grip';
    }
    // fingers toward the item, the back of the hand up and a little outward
    handQuat(d, dors.set(H.side === 'L' ? -0.35 : 0.35, 1, 0.2), H.quat);
    if (Rc.t >= tEnd) { if (!Rc.closed) Rc.resolve?.(); this._reach = null; }
  }

  dispose() { this.ctx.camera.remove(this.group); }
}

async function buildHatchet(ctx) {
  const g = new THREE.Group();
  let wood = null;
  try { wood = await ctx.assets.pbr('rough_wood'); } catch { wood = null; }
  const handleMat = new THREE.MeshStandardMaterial({ color: 0x9a7a58, roughness: 0.62, metalness: 0 });
  if (wood) {
    handleMat.map = wood.map; handleMat.normalMap = wood.normalMap; handleMat.roughnessMap = wood.armMap; handleMat.aoMap = wood.armMap;
    handleMat.normalScale.set(0.6, 0.6); handleMat.needsUpdate = true;
  }
  // slightly curved handle (lathe profile along +Y), 0.44 m, grip at the origin (the props hatchet's convention)
  const pts = [];
  for (let i = 0; i <= 12; i++) { const t = i / 12; pts.push(new THREE.Vector2(0.0135 + 0.004 * Math.sin(t * Math.PI) - (t > 0.9 ? 0.003 : 0) + (t < 0.08 ? 0.004 : 0), t * 0.44)); }
  const handle = new THREE.Mesh(new THREE.LatheGeometry(pts, 14), handleMat);
  handle.scale.set(1, 1, 1.3);
  handle.position.y = -0.098;
  g.add(handle);
  const steel = new THREE.MeshStandardMaterial({ color: 0x55585c, roughness: 0.42, metalness: 0.95 });
  const headGeo = new THREE.BoxGeometry(0.034, 0.075, 0.155, 1, 2, 6);
  const p = headGeo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const t = (p.getZ(i) + 0.0775) / 0.155;              // 0 = poll, 1 = blade edge (+Z)
    p.setX(i, p.getX(i) * (1 - 0.88 * smooth01(0.35, 1, t)));
    p.setY(i, p.getY(i) * (1 + 0.9 * smooth01(0.55, 1, t)));
  }
  headGeo.computeVertexNormals();
  const head = new THREE.Mesh(headGeo, steel);
  head.position.set(0, 0.31, 0.035);
  g.add(head);
  g.traverse((o) => { if (o.isMesh) { o.castShadow = false; o.receiveShadow = true; o.frustumCulled = false; } });
  return g;
}
