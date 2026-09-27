// GAME workstream: camera rig (DESIGN.md "cameraRig").
//
//   mode: 'foot' | 'car-chase' | 'car-cockpit' | 'cinematic';  setMode(m, {blend = 0.55 s})
//   shake(trauma 0..1)                  trauma model: decays; offsets/rotation noise ∝ trauma² (+ landslide.rumble floor)
//   cinematic(pathFn, duration, opts) -> Promise
//        pathFn(t01, outPos, outTarget, ctx) sets the pose; may return {fov, roll}. opts: {loop, hold, blend, ease}
//        (hold = stay in 'cinematic' at the end; otherwise the previous mode is restored)
//   carMode: the preferred seated view ('car-cockpit' | 'car-chase'); C toggles it while ctx.control === 'car'
//   lookBack 0..1: hold Q while driving to look back (cockpit: over the shoulder; chase: camera swings round)
//   viewmodel: setHeld('hatchet' | null), windup(0..1), strike() -> Promise (the hand-held hatchet on foot)
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

  init() {
    this.ctx.camera.fov = this.fovBase;
    this.ctx.camera.updateProjectionMatrix();
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
    if (mdx || mdy) this._mouseIdle = 0; else this._mouseIdle += dt;
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
    if (mdx || mdy) this._mouseIdle = 0; else this._mouseIdle += dt;
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
// The hatchet held in the right hand while on foot. It clones props.items.hatchet when PROPS provides it (the prop's
// longest bbox axis becomes the handle, grip at the origin), else a small procedural hatchet (rough_wood + steel).
class Viewmodel {
  constructor(ctx) {
    this.ctx = ctx;
    this.group = new THREE.Group();
    this.group.name = 'viewmodel';
    this.group.visible = false;
    ctx.camera.add(this.group);
    this.held = null;
    this.models = {};
    this._wind = 0; this._windT = 0;
    this._strike = null;
    this._sway = new THREE.Vector2();
    this._vis = 0;
    this._t = 0;
    this._building = false;
  }

  setHeld(id) { this.held = id || null; }
  windup(p) { this._windT = clamp(p, 0, 1); }
  strike() {
    return new Promise((resolve) => { this._strike = { t: 0, resolve }; });
  }

  _ensure(id) {
    if (this.models[id] !== undefined || this._building) return this.models[id] || null;
    if (id !== 'hatchet') { this.models[id] = null; return null; }
    const src = this.ctx.props?.items?.hatchet;
    const pivot = new THREE.Group();
    let obj = null;
    if (src && src.isObject3D) {
      obj = src.clone(true);
      obj.position.set(0, 0, 0); obj.quaternion.identity(); obj.scale.copy(src.getWorldScale(new THREE.Vector3()));
      obj.visible = true;
      obj.traverse((o) => { o.visible = true; });
      obj.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(obj);
      const size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
      // longest axis = handle -> +Y (head up), with the bulk of the tool on +Y (grip at the origin)
      const ax = size.x >= size.y && size.x >= size.z ? 'x' : (size.y >= size.z ? 'y' : 'z');
      const dir = new THREE.Vector3(); dir[ax] = c[ax] >= 0 ? 1 : -1;
      obj.quaternion.setFromUnitVectors(dir, Y);
      const holder = new THREE.Group(); holder.add(obj);
      // grip a bit above the handle end
      holder.position.y = -0.04;
      pivot.add(holder);
    } else {
      this._building = true;
      buildHatchet(this.ctx).then((g) => { pivot.add(g); this._building = false; }).catch(() => { this._building = false; });
    }
    pivot.traverse((o) => { if (o.isMesh) { o.castShadow = false; o.receiveShadow = false; o.frustumCulled = false; } });
    this.group.add(pivot);
    pivot.visible = false;
    this.models[id] = pivot;
    return pivot;
  }

  update(dt, active, rig) {
    this._t += dt;
    const want = active && this.held ? this.held : null;
    for (const [id, m] of Object.entries(this.models)) if (m) m.visible = id === want;
    const m = want ? this._ensure(want) : null;
    this._vis = lerp(this._vis, m ? 1 : 0, dampK(10, dt));
    this.group.visible = !!m;
    if (!m) return;
    m.visible = true;
    // pose: rest -> wind-up (raised back over the shoulder) -> strike (down and forward)
    this._wind = lerp(this._wind, this._windT, dampK(14, dt));
    const w = this._wind;
    const input = this.ctx.input;
    this._sway.x = lerp(this._sway.x, clamp(-(input?.dx || 0) * 0.0006, -0.04, 0.04), dampK(8, dt));
    this._sway.y = lerp(this._sway.y, clamp((input?.dy || 0) * 0.0006, -0.04, 0.04), dampK(8, dt));
    const bob = rig._bobAmt || 0, ph = rig._bobPhase || 0;
    let px = lerp(0.23, 0.26, w), py = lerp(-0.34, -0.06, w), pz = lerp(-0.42, -0.26, w);
    let rx = lerp(-0.25, 1.15, w), ry = lerp(0.15, 0.05, w), rz = lerp(-0.35, -0.15, w);
    if (this._strike) {
      const s = this._strike;
      s.t += dt;
      const a = smooth01(0, 0.11, s.t), b = smooth01(0.16, 0.55, s.t);
      const k = a * (1 - b);
      px = lerp(px, 0.08, k); py = lerp(py, -0.30, k); pz = lerp(pz, -0.62, k);
      rx = lerp(rx, -1.35, k); rz = lerp(rz, -0.05, k);
      if (s.t > 0.12 && !s.hit) { s.hit = true; s.resolve?.(); }
      if (s.t > 0.6) { this._strike = null; this._windT = 0; this._wind = 0.0; }
    }
    // out of view while (dis)appearing
    py -= (1 - this._vis) * 0.35;
    px += this._sway.x + Math.sin(ph) * 0.006 * bob;
    py += this._sway.y + Math.abs(Math.cos(ph)) * 0.008 * bob + Math.sin(this._t * 1.4) * 0.002;
    m.position.set(px, py, pz);
    m.rotation.set(rx, ry, rz, 'YXZ');
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
  // slightly curved hickory handle (lathe profile swept along Y), 0.40 m
  const pts = [];
  for (let i = 0; i <= 12; i++) { const t = i / 12; pts.push(new THREE.Vector2(0.0135 + 0.004 * Math.sin(t * Math.PI) - (t > 0.9 ? 0.003 : 0) + (t < 0.08 ? 0.004 : 0), t * 0.40)); }
  const handle = new THREE.Mesh(new THREE.LatheGeometry(pts, 14), handleMat);
  handle.scale.set(1, 1, 1.35);
  handle.position.y = -0.06;
  g.add(handle);
  // forged head: a wedge (tapered box) with a slightly darker, worn-steel look
  const steel = new THREE.MeshStandardMaterial({ color: 0x55585c, roughness: 0.42, metalness: 0.95 });
  const headGeo = new THREE.BoxGeometry(0.155, 0.075, 0.034, 6, 2, 1);
  const p = headGeo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i);
    const t = (x + 0.0775) / 0.155;              // 0 = poll, 1 = blade edge
    p.setZ(i, p.getZ(i) * (1 - 0.88 * smooth01(0.35, 1, t)));
    p.setY(i, p.getY(i) * (1 + 0.9 * smooth01(0.55, 1, t)));
  }
  headGeo.computeVertexNormals();
  const head = new THREE.Mesh(headGeo, steel);
  head.position.set(0.035, 0.31, 0);
  g.add(head);
  // bright honed edge
  const edge = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.14, 0.006), new THREE.MeshStandardMaterial({ color: 0xb8bcc0, roughness: 0.2, metalness: 1 }));
  edge.position.set(0.112, 0.31, 0);
  g.add(edge);
  g.traverse((o) => { if (o.isMesh) { o.castShadow = false; o.receiveShadow = false; o.frustumCulled = false; } });
  return g;
}
