import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';

// Collision membership bits (see DESIGN.md "Physics").
export const G = { STATIC: 1, CAR: 2, PLAYER: 4, ROCK: 8, DEBRIS: 16, PROP: 32, SENSOR: 64, ALL: 0xffff };
/** Rapier InteractionGroups: 16 high bits = membership, 16 low bits = filter (what it collides with). */
export const groups = (member, filter = G.ALL) => ((member & 0xffff) << 16) | (filter & 0xffff);

const _m = new THREE.Matrix4(), _v = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3();

/**
 * Physics wrapper around a Rapier world. Access as ctx.physics.
 *  - ctx.physics.RAPIER / ctx.physics.world for raw access
 *  - fixed step driven by main.js (config.physics.step)
 */
export default class Physics {
  constructor(ctx) {
    this.ctx = ctx;
    this.links = [];               // {body, object, offset?}
    this.collisionHandlers = new Map(); // collider.handle -> fn(otherCollider, started)
    this.forceHandlers = new Map();     // collider.handle -> fn(otherCollider, totalForceMagnitude, event)
  }

  async init() {
    await RAPIER.init();
    this.RAPIER = RAPIER;
    this.world = new RAPIER.World({ x: 0, y: this.ctx.config.physics.gravity, z: 0 });
    this.world.timestep = this.ctx.config.physics.step;
    this.events = new RAPIER.EventQueue(true);
  }

  /** Called by main.js once per fixed step. */
  step() {
    this.world.step(this.events);
    this.events.drainCollisionEvents((h1, h2, started) => {
      const c1 = this.world.getCollider(h1), c2 = this.world.getCollider(h2);
      this.collisionHandlers.get(h1)?.(c2, started);
      this.collisionHandlers.get(h2)?.(c1, started);
    });
    this.events.drainContactForceEvents((ev) => {
      const h1 = ev.collider1(), h2 = ev.collider2(), f = ev.totalForceMagnitude();
      const c1 = this.world.getCollider(h1), c2 = this.world.getCollider(h2);
      this.forceHandlers.get(h1)?.(c2, f, ev);
      this.forceHandlers.get(h2)?.(c1, f, ev);
    });
  }

  /** Copies body transforms onto linked Object3Ds. Called by main.js after the physics steps of a frame. */
  syncLinks() {
    for (const L of this.links) {
      if (L.body.isValid && !L.body.isValid()) continue;
      const t = L.body.translation(), r = L.body.rotation();
      L.object.position.set(t.x, t.y, t.z);
      L.object.quaternion.set(r.x, r.y, r.z, r.w);
      if (L.offset) L.object.quaternion.multiply(L.offset);
    }
  }
  link(body, object, offset = null) { this.links.push({ body, object, offset }); return () => this.unlink(body); }
  unlink(body) { this.links = this.links.filter((l) => l.body !== body); }

  /** Register a handler for collision start/stop on a collider (enables COLLISION_EVENTS). */
  onCollision(collider, fn) {
    collider.setActiveEvents(collider.activeEvents() | RAPIER.ActiveEvents.COLLISION_EVENTS);
    this.collisionHandlers.set(collider.handle, fn);
  }
  /** Register a handler for contact forces above `threshold` newtons (enables CONTACT_FORCE_EVENTS). */
  onContactForce(collider, fn, threshold = 0) {
    collider.setActiveEvents(collider.activeEvents() | RAPIER.ActiveEvents.CONTACT_FORCE_EVENTS);
    collider.setContactForceEventThreshold(threshold);
    this.forceHandlers.set(collider.handle, fn);
  }
  removeHandlers(collider) { this.collisionHandlers.delete(collider.handle); this.forceHandlers.delete(collider.handle); }

  /**
   * Static triangle-mesh collider from a THREE.Mesh (world matrix baked in) or from an Object3D subtree (all meshes merged).
   * opts: {groups, friction=0.9, restitution=0}
   */
  addTrimesh(object, opts = {}) {
    object.updateWorldMatrix(true, true);
    const verts = [], idx = [];
    object.traverse((m) => {
      if (!m.isMesh || !m.geometry?.attributes?.position) return;
      const g = m.geometry, pos = g.attributes.position, base = verts.length / 3;
      for (let i = 0; i < pos.count; i++) {
        _v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
        verts.push(_v.x, _v.y, _v.z);
      }
      if (g.index) for (let i = 0; i < g.index.count; i++) idx.push(base + g.index.getX(i));
      else for (let i = 0; i < pos.count; i++) idx.push(base + i);
    });
    const desc = RAPIER.ColliderDesc.trimesh(new Float32Array(verts), new Uint32Array(idx))
      .setFriction(opts.friction ?? 0.9).setRestitution(opts.restitution ?? 0)
      .setCollisionGroups(opts.groups ?? groups(G.STATIC));
    return this.world.createCollider(desc);
  }

  /** Static oriented box. center: Vector3, half: Vector3, quat?: Quaternion */
  addStaticBox(center, half, quat = null, opts = {}) {
    const desc = RAPIER.ColliderDesc.cuboid(half.x, half.y, half.z)
      .setTranslation(center.x, center.y, center.z)
      .setFriction(opts.friction ?? 0.8)
      .setCollisionGroups(opts.groups ?? groups(G.STATIC));
    if (quat) desc.setRotation({ x: quat.x, y: quat.y, z: quat.z, w: quat.w });
    if (opts.sensor) desc.setSensor(true);
    return this.world.createCollider(desc);
  }

  /**
   * Ray cast. Returns {point: Vector3, normal: Vector3, distance, collider} or null.
   * opts: {groups (filter InteractionGroups), excludeBody, excludeCollider, solid=true}
   */
  raycast(origin, dir, maxDist = 100, opts = {}) {
    const ray = new RAPIER.Ray({ x: origin.x, y: origin.y, z: origin.z }, { x: dir.x, y: dir.y, z: dir.z });
    const hit = this.world.castRayAndGetNormal(ray, maxDist, opts.solid ?? true, undefined, opts.groups, opts.excludeCollider, opts.excludeBody, opts.predicate);
    if (!hit) return null;
    const d = hit.timeOfImpact;
    return {
      point: new THREE.Vector3(origin.x + dir.x * d, origin.y + dir.y * d, origin.z + dir.z * d),
      normal: new THREE.Vector3(hit.normal.x, hit.normal.y, hit.normal.z),
      distance: d, collider: hit.collider,
    };
  }

  /** Ground height below (x,z) from static geometry, or null. */
  groundHeight(x, z, fromY = 2000, maxDist = 4000) {
    const h = this.raycast(_s.set(x, fromY, z), _v.set(0, -1, 0), maxDist, { groups: groups(G.ALL, G.STATIC) });
    return h ? h.point.y : null;
  }
}

export { RAPIER };
