// VEHICLE workstream — first-person walking (DESIGN.md "player").
//
// Kinematic capsule (r 0.30, half-height 0.60, feet at centre-0.90, eye at feet+1.65) moved by Rapier's
// KinematicCharacterController in the fixed step, interpolated for rendering.
//  - WASD walk, Shift sprint (stamina), Space jump (coyote time + buffer), gravity, snap-to-ground
//  - autostep <= 0.68 m plus a manual ledge-climb fallback (for round obstacles like the fallen trunk);
//    the guardrail (~0.75 m) stays too high to step over
//  - max climbable slope 50 deg; steeper ground makes you slide down
//  - pushes small dynamic bodies (KCC impulses, mass 80 kg)
//  - footsteps by surface (ctx.terrain.surfaceAt) with stride timing; 'player:land' on landing
//  - a sensor capsule reports boulder/debris strikes as 'player:struck' {speed, energy, collider, position}
// Collider groups: the solid capsule only interacts with PROP|SENSOR in the solver (so it never shoves the car or boulders
// with infinite kinematic mass); the KCC itself is blocked by STATIC|CAR|ROCK|DEBRIS|PROP.
import * as THREE from 'three';
import { G, groups, RAPIER } from './world.js';

const DEG = Math.PI / 180;
const clamp = THREE.MathUtils.clamp;
const smooth01 = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };

const RADIUS = 0.30, HALF = 0.60, CENTER = RADIUS + HALF, EYE = 1.65;
const STEP_MAX = 0.68;           // max climbable ledge (trunk top <= 0.65; guardrail ~0.75 stays a barrier)
const AUTOSTEP = 0.36;           // Rapier autostep for kerbs/stones; higher ledges use the deterministic ledge climb
const MAX_SLOPE = 50 * DEG;
const GRAVITY = 9.81;
const JUMP_V = 3.3;              // ~0.55 m jump
const SURF_SPEED = { asphalt: 1, gravel: 0.97, dirt: 0.93, rock: 0.9, grass: 0.94, mud: 0.78, wood: 1 };

const _v = new THREE.Vector3(), _w = new THREE.Vector3(), _f = new THREE.Vector3(), _r = new THREE.Vector3();

export default class Character {
  constructor(ctx) {
    this.ctx = ctx;
    this.object = new THREE.Group(); this.object.name = 'player';
    this.eye = new THREE.Vector3();
    this.feet = new THREE.Vector3();
    this.yaw = 0; this.pitch = 0;
    this.velocity = new THREE.Vector3();
    this.onGround = false; this.sprinting = false; this.stamina = 1; this.exhausted = false;
    this.enabled = false; this.speedScale = 1; this.surface = 'asphalt'; this.sliding = false;
    this.body = null; this.collider = null; this.kcc = null;
    this._explicit = false;
    this._acc = 0; this._vy = 0; this._hv = new THREE.Vector3();
    this._prev = new THREE.Vector3(); this._cur = new THREE.Vector3();
    this._jumpBuf = 0; this._coyote = 0; this._sinceJump = 1; this._skip = 0;
    this._stride = 0; this._vault = 0; this._vaultTop = 0; this._blocked = 0; this._prog = 1; this._mantle = 0;
    this._vaultDir = new THREE.Vector3();
    this._groundCollider = null; this._groundNormal = new THREE.Vector3(0, 1, 0);
    this._proj = {};
  }

  async init() {
    const { physics, road, scene } = this.ctx;
    const world = physics.world;
    const start = road ? road.worldAt((road.markers?.carStart ?? 40), -4.2) : new THREE.Vector3();
    this.body = world.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(start.x, start.y + CENTER + 0.05, start.z));
    this.collider = world.createCollider(
      RAPIER.ColliderDesc.capsule(HALF, RADIUS).setFriction(0).setRestitution(0)
        .setCollisionGroups(groups(G.PLAYER, G.PROP | G.SENSOR))
        .setActiveCollisionTypes(RAPIER.ActiveCollisionTypes.DEFAULT | RAPIER.ActiveCollisionTypes.KINEMATIC_FIXED),
      this.body);
    this.sensor = world.createCollider(
      RAPIER.ColliderDesc.capsule(HALF + 0.05, RADIUS + 0.1).setSensor(true)
        .setCollisionGroups(groups(G.PLAYER, G.ROCK | G.DEBRIS)),
      this.body);
    physics.onCollision(this.sensor, (other, started) => { if (started) this._onStruck(other); });

    const kcc = world.createCharacterController(0.03);
    kcc.setUp({ x: 0, y: 1, z: 0 });
    kcc.setMaxSlopeClimbAngle(MAX_SLOPE);
    kcc.setMinSlopeSlideAngle(MAX_SLOPE);
    kcc.enableAutostep(AUTOSTEP, 0.15, false);
    kcc.enableSnapToGround(0.32);
    kcc.setSlideEnabled(true);
    kcc.setApplyImpulsesToDynamicBodies(true);
    kcc.setCharacterMass(80);
    this.kcc = kcc;
    this.moveGroups = groups(G.PLAYER, G.STATIC | G.CAR | G.ROCK | G.DEBRIS | G.PROP);

    this.collider.setEnabled(false); this.sensor.setEnabled(false);
    const t = this.body.translation();
    this._prev.set(t.x, t.y, t.z); this._cur.copy(this._prev);
    this._syncObject(1);
    scene.add(this.object);
    if (road) this.yaw = road.yawAt(road.markers?.carStart ?? 40) + Math.PI;
  }

  // ------------------------------------------------------------------------------------------------ API
  setEnabled(b) {
    b = !!b;
    this._explicit = true;
    if (b === this.enabled) return;
    this.enabled = b;
    this.collider?.setEnabled(b); this.sensor?.setEnabled(b);
    this._hv.set(0, 0, 0); this._vy = 0; this.velocity.set(0, 0, 0);
    this._skip = 1;
  }

  /** pos = FEET position (world). yaw optional (radians; 0 looks toward -Z). */
  teleport(pos, yaw) {
    if (!this.body) return;
    const c = { x: pos.x, y: pos.y + CENTER + 0.03, z: pos.z };
    this.body.setTranslation(c, true);
    this.body.setNextKinematicTranslation(c);
    this._prev.set(c.x, c.y, c.z); this._cur.copy(this._prev);
    this._hv.set(0, 0, 0); this._vy = 0; this.velocity.set(0, 0, 0);
    this._vault = 0; this._skip = 1; this._stride = 0;
    if (typeof yaw === 'number') { this.yaw = yaw; this.pitch = 0; }
    this._syncObject(1);
  }

  lookDir(out = new THREE.Vector3()) {
    const cp = Math.cos(this.pitch);
    return out.set(-Math.sin(this.yaw) * cp, Math.sin(this.pitch), -Math.cos(this.yaw) * cp);
  }

  get position() { return this.feet; }

  // ------------------------------------------------------------------------------------------------ fixed step
  fixedUpdate(h) {
    if (!this.body) return;
    const { ctx } = this;
    this._acc -= h;
    const t = this.body.translation();
    this._prev.set(t.x, t.y, t.z);

    if (!this.enabled) {
      // auto-enable when the game hands control to the player's feet without calling setEnabled
      if (ctx.control === 'foot' && !this._explicit) {
        const dp = ctx.car?.doorPoint;
        if (dp) { dp.getWorldPosition(_v); this.teleport(_v); }
        this.setEnabled(true); this._explicit = false;
      }
      return;
    }
    if (this._skip > 0) { this._skip--; this.body.setNextKinematicTranslation(t); return; }

    // fell out of the world (isolated test without terrain): stop quietly
    if (ctx.road) {
      const pr = ctx.road.project(this._prev, this._proj);
      if (pr.dy < -120 || !isFinite(t.y)) { this.enabled = false; this.collider.setEnabled(false); this.sensor.setEnabled(false); console.info('[player] no ground: disabled'); return; }
    }

    const input = ctx.input;
    const active = ctx.control === 'foot' && !ctx.paused && input?.active;
    const cfg = ctx.config?.game || {};
    const walk = cfg.walkSpeed ?? 1.6, run = cfg.runSpeed ?? 5.2, staminaSec = cfg.stamina ?? 9;

    // --- intent
    let mx = 0, my = 0, wantSprint = false;
    if (active) {
      mx = input.moveX; my = input.moveY;
      wantSprint = input.down('ShiftLeft') || input.down('ShiftRight');
    }
    _f.set(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
    _r.set(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
    const wish = _w.set(0, 0, 0).addScaledVector(_f, my * (my < 0 ? 0.7 : 1)).addScaledVector(_r, mx * 0.85);
    let wishLen = wish.length();
    if (wishLen > 1) { wish.multiplyScalar(1 / wishLen); wishLen = 1; }

    // --- stamina / sprint
    const moving = wishLen > 0.1;
    this.sprinting = wantSprint && my > 0.1 && !this.exhausted && this.onGround !== false && moving;
    if (this.sprinting) this.stamina = Math.max(0, this.stamina - h / staminaSec);
    else this.stamina = Math.min(1, this.stamina + h / (staminaSec * (moving ? 1.6 : 1.0)));
    if (this.stamina <= 0) this.exhausted = true;
    if (this.exhausted && this.stamina > 0.35) this.exhausted = false;
    if (this.exhausted) this.sprinting = false;

    // --- target speed with surface + slope effects
    let target = (this.sprinting ? run : walk) * this.speedScale * (SURF_SPEED[this.surface] ?? 1);
    if (this.onGround && moving) {
      const n = this._groundNormal;
      const slope = Math.acos(clamp(n.y, -1, 1));
      const uphill = -(n.x * wish.x + n.z * wish.z); // >0 when moving against the slope normal (uphill)
      if (uphill > 0) target *= 1 - 0.55 * smooth01(8 * DEG, 45 * DEG, slope) * clamp(uphill / Math.max(Math.sin(slope), 1e-3), 0, 1);
    }
    const desiredV = _v.copy(wish).multiplyScalar(target);
    const accel = this.onGround ? (moving ? (this.sprinting ? 9 : 11) : 14) : 2.0;
    const dvx = desiredV.x - this._hv.x, dvz = desiredV.z - this._hv.z;
    const dl = Math.hypot(dvx, dvz), maxDv = accel * h;
    if (dl > maxDv) { this._hv.x += dvx / dl * maxDv; this._hv.z += dvz / dl * maxDv; } else { this._hv.x = desiredV.x; this._hv.z = desiredV.z; }

    // --- vertical
    this._sinceJump += h;
    if (this._jumpBuf > 0) this._jumpBuf -= h;
    if (this.onGround) this._coyote = 0.12; else this._coyote -= h;
    if (this.onGround && this._vy < 0) this._vy = 0;
    if (active && this._jumpBuf > 0 && this._coyote > 0 && this._sinceJump > 0.3 && this.stamina > 0.05) {
      this._vy = JUMP_V; this._jumpBuf = 0; this._coyote = 0; this._sinceJump = 0;
      this.stamina = Math.max(0, this.stamina - 0.04);
    }
    this._vy -= GRAVITY * h;
    if (this._vy < -45) this._vy = -45;

    // --- manual ledge climb (fallback for round/irregular obstacles the autostep misses)
    if (this._vault > 0) {
      this._vault -= h;
      const feetY = t.y - CENTER;
      const vd = this._vaultDir;
      if (this._mantle <= 0 && feetY < this._vaultTop + 0.05) { // rise
        this._vy = 2.6; this._hv.set(vd.x * 0.15, 0, vd.z * 0.15);
      } else { // mantle: hold height and move over the top (past the apex of round obstacles)
        if (this._mantle <= 0) this._mantle = 0.5;
        this._mantle -= h;
        this._vy = feetY < this._vaultTop + 0.03 ? 0.6 : 0;
        this._hv.set(vd.x * 1.4, 0, vd.z * 1.4);
        if (this._mantle <= 0) { this._vault = 0; this._mantle = 0; }
      }
      if (this._vault <= 0) { this._mantle = 0; this._vy = Math.min(this._vy, 0); }
    }

    const desired = { x: this._hv.x * h, y: this._vy * h, z: this._hv.z * h };
    // walk along the ground plane (keeps horizontal pace on walkable slopes, no hopping when going downhill)
    const gn = this._groundNormal;
    if (this.onGround && this._vy <= 0 && this._vault <= 0 && gn.y > Math.cos(MAX_SLOPE) && gn.y < 0.9995) {
      desired.y += -(gn.x * desired.x + gn.z * desired.z) / gn.y;
    }
    const kcc = this.kcc;
    kcc.computeColliderMovement(this.collider, desired, RAPIER.QueryFilterFlags.EXCLUDE_SENSORS, this.moveGroups);
    const mv = kcc.computedMovement();
    let grounded = kcc.computedGrounded();

    // ground info + steep-slope detection from the collisions of this move
    let groundCol = null, steep = false, bestNy = -1, hitCeil = false, wall = false;
    const n = kcc.numComputedCollisions();
    for (let i = 0; i < n; i++) {
      const c = kcc.computedCollision(i);
      if (!c) continue;
      const ny = c.normal1.y;
      if (ny > bestNy) { bestNy = ny; groundCol = c.collider; this._groundNormal.set(c.normal1.x, c.normal1.y, c.normal1.z); }
      if (ny > 0.05 && ny < Math.cos(MAX_SLOPE)) steep = true;
      if (ny < -0.5) hitCeil = true;
      if (Math.abs(ny) < 0.45) wall = true;
    }
    // Rapier reports "grounded" on slopes steeper than the climb limit too: treat those as sliding ground
    if (grounded && n > 0 && bestNy < Math.cos(MAX_SLOPE) - 0.01 && this._vault <= 0) grounded = false;
    if (grounded && bestNy < Math.cos(MAX_SLOPE)) this._groundNormal.set(0, 1, 0);
    this._groundCollider = groundCol;

    const wasGround = this.onGround;
    const prevVy = this._vy;
    this.onGround = grounded;
    this.sliding = !grounded && steep;
    if (hitCeil && this._vy > 0) this._vy = 0;
    if (!grounded && this._vy < 0 && this._vault <= 0) {
      if (this.sliding) this._vy = Math.max(this._vy, -5.5); // sliding down a steep wet bank (KCC projects it on the slope)
      else this._vy = Math.max(this._vy, mv.y / h - GRAVITY * h); // can't fall faster than we actually moved (stuck in a crease)
    }

    // blocked while pushing forward? (smoothed progress along the wish direction; contacts with round obstacles jitter)
    const wantH = Math.hypot(desired.x, desired.z);
    if (wantH > 1e-5 && my > 0.1 && this._vault <= 0) {
      const along = (mv.x * desired.x + mv.z * desired.z) / (wantH * wantH);
      this._prog = this._prog * 0.8 + along * 0.2;
      if ((grounded || this._coyote > 0) && this._prog < 0.35 && (wall || this._blocked > 0)) {
        this._blocked += h;
        if (this._blocked > 0.1) { if (this._tryLedge(t)) this._prog = 1; this._blocked = 0; }
      } else this._blocked = 0;
    } else { this._prog = 1; this._blocked = 0; }

    if (this.debugLog) this.debugLog.push({ y: +(t.y - CENTER).toFixed(3), des: [+desired.x.toFixed(4), +desired.y.toFixed(4), +desired.z.toFixed(4)], mv: [+mv.x.toFixed(4), +mv.y.toFixed(4), +mv.z.toFixed(4)], g: grounded ? 1 : 0, n, bestNy: +bestNy.toFixed(2), vault: +this._vault.toFixed(2), blk: +this._blocked.toFixed(2), steep: steep ? 1 : 0 });
    const nx = t.x + mv.x, ny = t.y + mv.y, nz = t.z + mv.z;
    this.body.setNextKinematicTranslation({ x: nx, y: ny, z: nz });
    this.velocity.set(mv.x / h, mv.y / h, mv.z / h);

    // landing
    if (!wasGround && grounded && prevVy < -2.2) this.ctx.events.emit('player:land', { speed: -prevVy });

    // footsteps by stride
    if (grounded) {
      const hs = Math.hypot(mv.x, mv.z);
      const spd = hs / h;
      if (spd > 0.25) {
        this._stride += hs;
        const stride = 0.62 + 0.14 * spd; // m per step: walk ~0.8 m, run ~1.35 m
        if (this._stride > stride) {
          this._stride -= stride;
          this._footstep(spd > walk * 1.35, nx, ny - CENTER, nz);
        }
      } else this._stride = Math.min(this._stride, 0.4);
    }
    if (!wasGround && grounded && prevVy < -1.5) this._footstep(false, nx, ny - CENTER, nz);
  }

  _tryLedge(t) {
    const phys = this.ctx.physics;
    const dir = _v.set(this._hv.x, 0, this._hv.z);
    if (dir.lengthSq() < 1e-4) dir.set(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
    dir.normalize();
    const feetY = t.y - CENTER;
    const filt = groups(G.PLAYER, G.STATIC | G.PROP | G.ROCK | G.CAR);
    // probe down in front of the capsule for a walkable top between 0.2 and STEP_MAX above the feet
    for (const reach of [RADIUS + 0.12, RADIUS + 0.3]) {
      const ox = t.x + dir.x * reach, oz = t.z + dir.z * reach;
      const hit = phys.raycast({ x: ox, y: feetY + STEP_MAX + 0.05, z: oz }, { x: 0, y: -1, z: 0 }, STEP_MAX + 0.05, { groups: filt, excludeBody: this.body });
      if (!hit) continue;
      const top = hit.point.y - feetY;
      if (top < 0.2 || top > STEP_MAX + 0.01 || hit.normal.y < Math.cos(MAX_SLOPE)) continue;
      // head room above the ledge
      const up = phys.raycast({ x: ox, y: hit.point.y + 0.05, z: oz }, { x: 0, y: 1, z: 0 }, 1.8, { groups: filt, excludeBody: this.body });
      if (up) continue;
      this._vault = 1.4; this._mantle = 0; this._vaultTop = hit.point.y; this._vaultDir.copy(dir);
      return true;
    }
    return false;
  }

  _footstep(run, x, y, z) {
    let surface = 'asphalt';
    const col = this._groundCollider;
    const mem = col ? (col.collisionGroups() >>> 16) & 0xffff : 0;
    if (mem & G.PROP) surface = 'wood';
    else if (mem & (G.ROCK | G.CAR)) surface = 'rock';
    else if (this.ctx.terrain?.surfaceAt) {
      try { surface = this.ctx.terrain.surfaceAt(_v.set(x, y, z)) || 'asphalt'; } catch { /* terrain not ready */ }
    }
    this.surface = surface;
    this.ctx.events.emit('footstep', { surface, run, position: new THREE.Vector3(x, y, z) });
  }

  _onStruck(other) {
    if (!this.enabled || !other) return;
    const b = other.parent?.();
    if (!b || !b.isDynamic?.()) return;
    const v = b.linvel();
    const rel = Math.hypot(v.x - this.velocity.x, v.y - this.velocity.y, v.z - this.velocity.z);
    if (rel < 2.5) return;
    const m = b.mass?.() ?? 100;
    const p = b.translation();
    this.ctx.events.emit('player:struck', { collider: other, speed: rel, energy: 0.5 * m * rel * rel / 1e5, position: new THREE.Vector3(p.x, p.y, p.z) });
  }

  // ------------------------------------------------------------------------------------------------ frame
  update(dt) {
    if (!this.body) return;
    const { ctx } = this;
    const H = ctx.config?.physics?.step || 1 / 60;
    this._acc = clamp(this._acc + dt, 0, H * 0.999);
    const t = this.body.translation();
    this._cur.set(t.x, t.y, t.z);
    this._syncObject(this.enabled ? this._acc / H : 1);

    const input = ctx.input;
    const active = this.enabled && ctx.control === 'foot' && !ctx.paused && input?.active;
    if (!active || dt <= 0) return;
    const sens = input.sensitivity ?? 0.0022;
    this.yaw -= input.dx * sens;
    this.pitch -= input.dy * sens * (input.invertY ? -1 : 1);
    this.pitch = clamp(this.pitch, -1.48, 1.48);
    if (this.yaw > Math.PI) this.yaw -= Math.PI * 2; else if (this.yaw < -Math.PI) this.yaw += Math.PI * 2;
    if (input.pressed('Space')) this._jumpBuf = 0.15;
    if (input.mousePressed(0) && !input.locked) input.requestLock();
  }

  _syncObject(alpha) {
    _v.lerpVectors(this._prev, this._cur, alpha);
    this.feet.set(_v.x, _v.y - CENTER, _v.z);
    this.eye.set(_v.x, _v.y - CENTER + EYE, _v.z);
    this.object.position.copy(this.feet);
    this.object.rotation.set(0, this.yaw, 0);
    this.object.updateMatrixWorld(true);
  }

  dispose() {
    const { physics, scene } = this.ctx;
    try {
      physics.removeHandlers(this.sensor);
      if (this.kcc) physics.world.removeCharacterController(this.kcc);
      if (this.body) physics.world.removeRigidBody(this.body);
    } catch { /* world gone */ }
    scene?.remove(this.object);
    this.body = null;
  }
}
