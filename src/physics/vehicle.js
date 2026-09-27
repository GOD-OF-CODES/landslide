// VEHICLE workstream — driving physics for the old mountain 4x4 (DESIGN.md "car").
//
//  - Visual: public/assets/models/car.glb (+ car.json) from the CAR workstream. If missing, a procedural placeholder with the
//    same node/material names is built, and (in dev) the real model is swapped in automatically as soon as it appears.
//  - Physics: dynamic chassis (compound of cuboids, density 0, explicit mass/CoG/inertia from car.json) driven by Rapier's
//    DynamicRayCastVehicleController (4 ray-cast wheels). On top of Rapier's Bullet-style model we add:
//      * per-axle spring rates from static load (equal sag), progressive bump stops, a soft front anti-roll bar
//      * tyre-limited braking (no ABS: wheels lock and lose lateral grip), handbrake, rolling resistance, aero drag
//      * surface grip from ctx.terrain.surfaceAt (wet asphalt / gravel / dirt / mud / grass / rock / wood) and slope derating
//      * engine: torque curve, torque converter, 5-speed automatic with shift delay + torque interruption, kick-down,
//        engine braking, reverse, rev limiter, idle, cranking, fuel consumption, sputter + stall
//  - `object` is the INTERPOLATED visual (render state, one physics step behind); `body` is the physics truth.
//  - Visual FX (self-contained, 1 draw call each): ContactShadow (ground AO footprint + tyre contact cores under the car),
//    TyreSpray (wet-road mist plumes + droplets behind the tyres, cold-weather exhaust vapour).
//  - Render cost: car.glb ships ALPHA glass (never transmission: it re-renders the whole scene), inset shadow-caster
//    proxy, interior hidden when far (see _setupRenderCost). Cabin bounce fill for the cockpit view (see _setupCabinFill).
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { G, groups, RAPIER } from './world.js';
import { url } from '../core/assets.js';

const DEG = Math.PI / 180;
const clamp = THREE.MathUtils.clamp;
const lerp = THREE.MathUtils.lerp;
const smooth01 = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
const approach = (v, target, maxDelta) => (v < target ? Math.min(target, v + maxDelta) : Math.max(target, v - maxDelta));

// Geometry defaults = DESIGN.md car contract (overridden by car.json and by the model's wheel node positions).
const CAR_DEFAULTS = {
  mass: 1350, wheelbase: 2.30, track: 1.42, wheelRadius: 0.36, wheelWidth: 0.22,
  suspensionRest: 0.28, suspensionTravel: 0.24, cgHeight: 0.60, frontAxleZ: 1.15, rearAxleZ: -1.15,
  length: 3.9, width: 1.70, height: 1.75, comZ: 0.08,
};

// Wet-weather tyre friction coefficient (mu), rolling-resistance coefficient, and how much mud it throws onto the car.
export const SURFACES = {
  asphalt: { mu: 0.74, rr: 0.014, mud: 0.0 },
  gravel: { mu: 0.56, rr: 0.030, mud: 0.25 },
  dirt: { mu: 0.48, rr: 0.040, mud: 0.6 },
  rock: { mu: 0.60, rr: 0.020, mud: 0.05 },
  grass: { mu: 0.40, rr: 0.050, mud: 0.4 },
  mud: { mu: 0.30, rr: 0.085, mud: 1.0 },
  wood: { mu: 0.52, rr: 0.016, mud: 0.0 },
};

// ~1.7 L carburettor petrol engine (Niva-class): 140 Nm @ 3000, ~60 kW @ 5400.
const ENGINE = {
  idle: 820, redline: 5600, limiter: 5900, convStall: 2250,
  torque: [[0, 0], [500, 60], [1000, 96], [1500, 114], [2000, 127], [2500, 135], [3000, 140], [3500, 139], [4000, 134], [4500, 126], [5000, 116], [5500, 104], [6000, 86], [6500, 60]],
  gears: [3.67, 2.10, 1.36, 1.00, 0.82], reverse: 3.53, final: 4.68, eff: 0.84, // final incl. transfer case (3.9 x 1.2)
  frontSplit: 0.45, // permanent 4x4 torque split (front share)
  shiftTime: 0.42,
};
const BRAKES = { total: 8600, front: 0.62, handbrake: 2800 /* N per rear wheel */, hold: 3500 /* N per wheel, auto hold at standstill */ };
const AERO = 0.5 * 1.2 * 0.52 * 2.65; // 0.5 rho Cd A  (boxy 4x4)
const STEER = { max: 33 * DEG, ratio: 15.5 };
const SPUTTER_L = 0.06;    // liters left when the engine starts to cough
const SPUTTER_T = 7;      // seconds from the first cough to the stall (reserve phase is time based): coughing starts just after the intro rockfall (s≈175-190)
const FUEL_CAL = 0.50;     // calibration of config.game.fuelPerMeterBase (see _updateFuel)
const REVERSE_MAX = 7.0;   // m/s (~25 km/h) soft limit in reverse

const _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3(), _v4 = new THREE.Vector3();
// fixed-step private frame vectors (never used by helpers)
const _fwd = new THREE.Vector3(), _up = new THREE.Vector3(), _left = new THREE.Vector3(), _vel = new THREE.Vector3(), _hp = new THREE.Vector3();
const _q1 = new THREE.Quaternion(), _q2 = new THREE.Quaternion();
const _m1 = new THREE.Matrix4();
const AX_X = new THREE.Vector3(1, 0, 0), AX_Y = new THREE.Vector3(0, 1, 0), AX_Z = new THREE.Vector3(0, 0, 1);
const WHEEL_NAMES = ['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr'];

function torqueAt(rpm) {
  const T = ENGINE.torque;
  if (rpm <= T[0][0]) return T[0][1];
  for (let i = 1; i < T.length; i++) {
    if (rpm <= T[i][0]) { const a = T[i - 1], b = T[i]; return a[1] + (b[1] - a[1]) * (rpm - a[0]) / (b[0] - a[0]); }
  }
  return T[T.length - 1][1];
}

/** True if a real GLB exists at path (vite's SPA fallback answers missing files with index.html + 200). */
async function probeGlb(path) {
  try {
    const r = await fetch(url(path), { cache: 'no-store', headers: { Range: 'bytes=0-3' } });
    const b = new Uint8Array(await r.arrayBuffer());
    return r.ok && b.length >= 4 && b[0] === 0x67 && b[1] === 0x6c && b[2] === 0x54 && b[3] === 0x46; // 'glTF'
  } catch { return false; }
}
/** Parsed JSON at path, or null if missing (no console noise). */
async function fetchJson(path) {
  try {
    const r = await fetch(url(path), { cache: 'no-store' });
    const txt = await r.text();
    if (!r.ok || !(r.headers.get('content-type') || '').includes('json')) return null;
    return JSON.parse(txt);
  } catch { return null; }
}

const _hlQ = new THREE.Quaternion();   // headlight beam axis (see _setupBeamFalloff)
export default class Vehicle {
  constructor(ctx) {
    this.ctx = ctx;
    this.object = new THREE.Group();
    this.object.name = 'car';
    this.body = null;
    this.controller = null;
    this.colliders = [];
    this.spec = { ...CAR_DEFAULTS };

    // public state (DESIGN.md)
    this.speed = 0; this.rpm = 0; this.throttle = 0; this.brake = 0; this.steer = 0; this.steerAngle = 0; this.handbrake = false;
    this.gear = 1; this.engineOn = true; this.stalling = false; this.cranking = 0; this.damage = 0;
    const g = ctx.config?.game || {};
    this.fuelCapacity = g.fuelCapacity ?? 45;
    this.fuel = g.fuelStart ?? 0.35;
    this.fuelScale = 1;                 // GAME may tune consumption without touching config.js
    this.fuelFlow = 0;                  // current consumption (L/s)
    this.sputter = 0;                   // 0..1 progress of the fuel-starvation phase (0 = normal)
    this.lights = { head: true, brake: false, hazard: false };
    this.headlightIntensity = 70;       // candela (three.js physical units); x3.5 inside the tunnel (see _updateMaterials)
    this.controlsEnabled = true;
    this.aiInput = null;                // {throttle, brake, steer(-1..1), handbrake} overrides player input
    this.autopilot = null;              // {speed m/s | null (steer only, coast), d lane offset, brake?}  road follower for cinematics/tests
    this.gripScale = 1;                 // global grip multiplier (e.g. heavier rain)
    this.rollMomentScale = 1;           // 0 = Rapier's default (almost no body roll), 1 = physical roll moment
    this.mud = 0;                       // 0..1 accumulated mud on wheels/lower body
    this.rolledOver = false; this.upright = 1;
    this.frozen = false;
    this.lateralG = 0; this.longG = 0;
    this.skid = 0;                      // 0..1 tyre sliding (for tyre-squeal / gravel-scrub audio)
    this.bump = 0;                      // suspension jolt speed (m/s, decays) for camera shake / thumps
    this.surface = 'asphalt';           // dominant surface under the wheels

    // anchors (always exist; re-parented when the model arrives)
    this.seatCam = new THREE.Object3D(); this.seatCam.name = 'seat_cam';
    this.fuelCap = new THREE.Object3D(); this.fuelCap.name = 'fuel_cap';
    this.doorPoint = new THREE.Object3D(); this.doorPoint.name = 'door_point';

    // internal
    this._acc = 0;
    this._prevPos = new THREE.Vector3(); this._prevQuat = new THREE.Quaternion();
    this._curPos = new THREE.Vector3(); this._curQuat = new THREE.Quaternion();
    this._shiftT = 0; this._sinceShift = 0; this._revHold = 0; this._fwdHold = 0;
    this._starve = 0; this._sputterTimer = 0; this._stallT = 0; this._limiterCut = 0; this._crankFail = false;
    this._surf = ['asphalt', 'asphalt', 'asphalt', 'asphalt'];
    this._surfIdx = 0; this._stepCount = 0;
    this._locked = [false, false, false, false];
    this._spin = [0, 0, 0, 0];
    this._steerW = [0, 0, 0, 0];
    this._stepImpulse = 0; this._impAcc = 0; this._impCooldown = 0; this._lastVel = new THREE.Vector3();
    this._rollT = 0; this._hudT = 0; this._blinkT = 0; this._shakeT = 0;
    this._proj = {};
    this.wheels = [];
    this.mats = {};
  }

  /** Public: true while fuel starvation has cut the engine (sputter phase). Read by AUDIO with cranking and sputter. */
  get fuelCut() { return this._starve > 0 && this.engineOn; }

  // ------------------------------------------------------------------------------------------------ init / model
  async init() {
    const { ctx } = this;
    let gltf = null, json = null;
    const [hasGlb, j] = await Promise.all([probeGlb('assets/models/car.glb'), fetchJson('assets/models/car.json')]);
    json = j;
    if (hasGlb) { try { gltf = await ctx.assets.gltf('assets/models/car.glb'); } catch { gltf = null; } }
    if (json) this._applySpec(json);
    const root = gltf ? gltf.scene : buildPlaceholderCar(this.spec);
    this.isPlaceholder = !gltf;
    this._setModel(root);
    this._buildPhysics();
    this._buildLights();
    ctx.scene.add(this.object);
    // ground contact shadow + wet-road tyre spray (self-contained, 1 draw call each)
    try {
      this.contactShadow = new ContactShadow(this); ctx.scene.add(this.contactShadow.mesh);
      this.spray = new TyreSpray(this); ctx.scene.add(this.spray.mesh);
    } catch (e) { console.warn('[car] fx init failed', e); }

    const start = ctx.road?.markers?.carStart ?? 40;
    this.teleport(start, -(ctx.road?.json?.laneCenter ?? 1.5));

    this._offs = [
      ctx.events.on('car:exit', () => { this.throttle = 0; this.brake = 0; }),
    ];
    // Dev convenience: swap in the real model as soon as the CAR workstream exports it.
    if (this.isPlaceholder && import.meta.env?.DEV) this._pollModel();
  }

  _applySpec(j) {
    this.json = j;
    const s = this.spec;
    for (const k of Object.keys(CAR_DEFAULTS)) if (typeof j[k] === 'number' && isFinite(j[k])) s[k] = j[k];
    if (typeof j.comZ !== 'number' && typeof j.frontAxleZ === 'number' && typeof j.rearAxleZ === 'number') {
      s.comZ = (j.frontAxleZ + j.rearAxleZ) / 2 + 0.07 * (j.frontAxleZ - j.rearAxleZ) / 2.3; // slightly front-heavy
    }
  }

  _pollModel() {
    let tries = 0;
    const tick = async () => {
      if (!this.isPlaceholder || tries++ > 400) return;
      if (await probeGlb('assets/models/car.glb')) {
        try {
          const json = await fetchJson('assets/models/car.json');
          const gltf = await this.ctx.assets.gltfLoader.loadAsync(url('assets/models/car.glb') + '?t=' + Date.now());
          if (json) this._applySpec(json);
          this._setModel(gltf.scene);
          this._rebuildWheels();
          this._applyMassProps();
          this.contactShadow?.rebuild();
          this.isPlaceholder = false;
          console.info('[car] real car.glb loaded (hot swap)');
          return;
        } catch (e) { /* not ready yet (partially written file); retry */ }
      }
      this._pollTimer = setTimeout(tick, 8000);
    };
    this._pollTimer = setTimeout(tick, 8000);
  }

  /** Install a visual root (gltf scene or placeholder) with the DESIGN node names. */
  _setModel(root) {
    // remove old visual
    if (this.model) { this.object.remove(this.model); disposeTree(this.model); }
    for (const w of this.wheels) w.node?.parent?.remove(w.node);
    this.model = root;
    this.object.add(root);
    root.updateMatrixWorld(true);

    const find = (n) => root.getObjectByName(n);
    this.bodyNode = find('car_body') || root;
    // wheels: re-parent directly under the car root (keeps transforms; we drive them in car space)
    const s = this.spec;
    this.wheels = WHEEL_NAMES.map((name, i) => {
      let node = find(name);
      const front = i < 2, left = i % 2 === 0;
      if (!node) {
        node = new THREE.Group(); node.name = name;
        node.position.set((left ? 1 : -1) * s.track / 2, s.wheelRadius, front ? s.frontAxleZ : s.rearAxleZ);
        root.add(node);
      }
      this.object.attach(node);
      return { node, rest: node.position.clone(), q0: node.quaternion.clone(), front, left };
    });
    // derive geometry from actual wheel nodes
    const fl = this.wheels[0].rest, fr = this.wheels[1].rest, rl = this.wheels[2].rest;
    s.track = Math.abs(fl.x - fr.x) || s.track;
    s.frontAxleZ = (fl.z + fr.z) / 2; s.rearAxleZ = (rl.z + this.wheels[3].rest.z) / 2;
    s.wheelbase = s.frontAxleZ - s.rearAxleZ || s.wheelbase;

    // steering wheel: rotation sign so that a left turn turns the rim counter-clockwise as seen by the driver
    this.steeringWheel = find('steering_wheel');
    if (this.steeringWheel) {
      this._swQ0 = this.steeringWheel.quaternion.clone();
      this.object.updateMatrixWorld(true);
      const wq = this.object.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(this.steeringWheel.getWorldQuaternion(new THREE.Quaternion()));
      const zc = new THREE.Vector3(0, 0, 1).applyQuaternion(wq); // column axis in car space
      this._swSign = zc.z < 0 ? 1 : -1;
    }
    // steering ratio: full lock of the road wheels = the model's max steering-wheel angle
    const swMax = this.json?.steeringWheel?.maxAngle;
    this.steerRatio = (typeof swMax === 'number' && swMax > 1) ? swMax / STEER.max : STEER.ratio;
    // dashboard needles (car.json "needles": rotation.z in degrees, 0 = straight up)
    const nd = this.json?.needles || {};
    this.needles = {};
    for (const k of ['speedo', 'tacho', 'fuel', 'temp']) {
      const n = find('needle_' + k);
      if (n) this.needles[k] = { node: n, cfg: nd[k] || null, val: 0 };
    }
    this.engineTemp = 0.5;
    // anchors: use model empties when present (keep our Object3D instances so references stay valid)
    // (an empty exported without a translation sits at the car origin: then car.json's position wins)
    this.object.updateMatrixWorld(true);
    const anchor = (obj, name, jsonKey, fallback) => {
      const n = find(name);
      obj.removeFromParent();
      let usable = false;
      if (n) { const p = this.object.worldToLocal(n.getWorldPosition(new THREE.Vector3())); usable = p.lengthSq() > 0.01; }
      if (usable) { n.add(obj); obj.position.set(0, 0, 0); obj.quaternion.identity(); return; }
      const jp = this.json?.[jsonKey];
      (this.bodyNode || this.object).add(obj);
      obj.position.copy(Array.isArray(jp) && jp.length === 3 ? new THREE.Vector3().fromArray(jp) : fallback);
      obj.quaternion.identity();
    };
    anchor(this.seatCam, 'seat_cam', 'seatCam', new THREE.Vector3(0.37, 1.34, -0.18));
    anchor(this.fuelCap, 'fuel_cap', 'fuelCap', new THREE.Vector3(0.88, 0.95, -1.38));
    anchor(this.doorPoint, 'door_point', 'doorPoint', new THREE.Vector3(1.75, 0.0, -0.15));
    // tailpipe exit (rear right, under the bumper; car.py builds it at Blender (-0.45, -1.90, 0.265))
    const ep = this.json?.exhaust;
    this.exhaustPoint = Array.isArray(ep) && ep.length === 3 ? new THREE.Vector3().fromArray(ep) : new THREE.Vector3(-0.45, 0.265, -this.spec.length / 2 + 0.02);

    // materials by name prefix (exporters append .001 etc.)
    this.mats = {};
    const want = ['paint', 'glass', 'chrome', 'trim_black', 'rubber', 'interior', 'headlight', 'brakelight', 'indicator', 'plate', 'gauge'];
    root.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true; o.receiveShadow = true;
      const ms = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of ms) {
        if (!m?.name) continue;
        const key = want.find((w) => m.name === w || m.name.startsWith(w + '.') || m.name.startsWith(w + '_'));
        if (!key) continue;
        (this.mats[key] ||= new Set()).add(m);
      }
      if (ms.some((m) => m?.name?.startsWith('glass'))) o.castShadow = false;
    });
    for (const k of Object.keys(this.mats)) this.mats[k] = [...this.mats[k]];
    // cache base values for wetness / mud modulation, make sure emissive materials have an emissive colour
    for (const m of this.mats.paint || []) { m.userData.r0 = m.roughness; }
    for (const m of this.mats.rubber || []) { m.userData.c0 = m.color.clone(); m.userData.r0 = m.roughness; }
    const emis = { headlight: 0xfff1d8, brakelight: 0xff1a0a, indicator: 0xff8a14, gauge: 0x9fe8ff };
    for (const [k, col] of Object.entries(emis)) {
      for (const m of this.mats[k] || []) {
        if (!m.emissive) continue;
        if (m.emissive.getHex() === 0) m.emissive.setHex(col);
        m.userData.e0 = m.emissive.clone();
        m.emissiveIntensity = 0;
      }
    }
    // (v4) instrument backlight: 12 V incandescent bulbs (~2700 K) behind the dial film -> warm white-amber glow
    for (const m of this.mats.gauge || []) { m.emissive.setRGB(1.0, 0.74, 0.46); m.userData.e0 = m.emissive.clone(); }
    this._lastEmis = {};
    this._setupRenderCost(root);
  }

  /**
   * Render-cost + realism tweaks applied to whatever model is installed:
   *  - glass: car.glb exports alpha-blended glass (opacity in the material). The shading tweak keeps full-strength
   *    Fresnel reflections (specular is not scaled by alpha, like real glass) and makes dirty/rough areas of the baked
   *    glass maps more opaque (dirt film). Transmission is never enabled (it re-renders the whole scene every frame).
   *  - shadows: the 16 real casters (~78k tris) are replaced by a ~400-tri inset proxy (soft overcast sun: identical).
   *  - interior meshes (dash, gauges, needles, steering wheel, seats) are hidden when the camera is far from the car.
   */
  /**
   * (QA) A reflector headlamp throws its light forward: seen inside the beam the lens is blinding (it clips in any
   * camera), seen from 40 deg off-axis it is a far dimmer glow in which the reflector bowl and lens flutes still read.
   * With a uniform emission the lamps were flat white discs from every angle (the rockfall cinematic films the car
   * from the front quarter). Emission is scaled by the angle between the view ray and the car's forward axis:
   * 1 within ~15 deg of the axis, ~0.12 beyond ~40 deg.
   */
  _setupBeamFalloff() {
    this._hlFwd = { value: new THREE.Vector3(0, 0, 1) };
    for (const m of this.mats.headlight || []) {
      const prev = m.onBeforeCompile, prevKey = m.customProgramCacheKey;
      m.onBeforeCompile = (sh, r) => {
        if (prev && prev !== THREE.Material.prototype.onBeforeCompile) prev.call(m, sh, r);
        sh.uniforms.uHlFwd = this._hlFwd;
        sh.fragmentShader = sh.fragmentShader
          .replace('#include <common>', '#include <common>\nuniform vec3 uHlFwd;')
          .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          {
            vec3 hlF = normalize( ( viewMatrix * vec4( uHlFwd, 0.0 ) ).xyz );
            float hlC = dot( normalize( vViewPosition ), hlF );
            totalEmissiveRadiance *= mix( 0.12, 1.0, smoothstep( 0.76, 0.965, hlC ) );
          }`);
      };
      const base = prevKey && prevKey !== THREE.Material.prototype.customProgramCacheKey ? prevKey.call(m) : '';
      m.customProgramCacheKey = () => base + '|hlbeam';
      m.needsUpdate = true;
    }
  }

  _setupRenderCost(root) {
    this.interiorMeshes = [];
    root.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = false;
      const ms = Array.isArray(o.material) ? o.material : [o.material];
      let inside = false;
      for (let p = o; p && p !== root; p = p.parent) if (/^(steering_wheel|needle_|air_freshener|mirror_charm|key_ring)/.test(p.name)) inside = true;
      if (ms.every((m) => /^(interior|gauge)/.test(m?.name || ''))) inside = true;
      if (inside) this.interiorMeshes.push(o);
    });
    const U = this._cabinUniforms();
    for (const m of this.mats.glass || []) alphaGlass(m, false, false, U);
    for (const m of this.mats.headlight || []) if (m.transparent) alphaGlass(m, true); // clear lens: facets reflect at full strength
    try { this._setupBeamFalloff(); } catch (e) { console.warn('[car] headlight beam falloff', e); }
    try { this._setupLens(); } catch (e) { console.warn('[car] gauge lens', e); }
    this._setupCabinFill();
    try { this._setupMirror(); } catch (e) { console.warn('[car] mirror', e); this.mirror = null; }
    this._setupDangles();
    // shadow proxy (rebuilt on hot swap)
    if (this.shadowProxy) { this.object.remove(this.shadowProxy); this.shadowProxy.geometry.dispose(); }
    this.shadowProxy = buildShadowProxy(this.spec, this.json, this.wheels);
    this.object.add(this.shadowProxy);
    this._interiorShown = true;
  }

  _buildLights() {
    // One spot for both low beams (cheaper than two); added now so shaders compile with it (toggle intensity only).
    const L = new THREE.SpotLight(0xffe8c8, 0, 70, 0.62, 0.55, 1.6);
    L.name = 'car_headlights';
    const hl = this.json?.headlights;
    const hy = Array.isArray(hl) && hl.length ? hl.reduce((a, p) => a + p[1], 0) / hl.length : 0.82;
    const hz = Array.isArray(hl) && hl.length ? hl.reduce((a, p) => a + p[2], 0) / hl.length : this.spec.length / 2 - 0.05;
    L.position.set(0, hy, hz + 0.05);
    L.target.position.set(0, -0.5, hz + 18);
    L.castShadow = false;
    this.object.add(L, L.target);
    this.headlightSource = L;
  }

  // ------------------------------------------------------------------------------------------------ physics build
  _buildPhysics() {
    const { physics } = this.ctx;
    const world = physics.world;
    const s = this.spec;
    const bd = RAPIER.RigidBodyDesc.dynamic()
      .setCanSleep(false).setCcdEnabled(true)
      .setLinearDamping(0.0).setAngularDamping(0.08);
    this.body = world.createRigidBody(bd);
    this._applyMassProps();

    // Compound body shape (density 0: mass comes from _applyMassProps). Bottom edges sit ~0.30-0.40 m above ground.
    const L = s.length, W = s.width, H = s.height;
    let boxes = [
      // [hx, hy, hz, cx, cy, cz]
      [W / 2 - 0.02, 0.31, L / 2 - 0.16, 0, 0.71, -0.02],           // tub: sills, floor, hood, boot (0.40..1.02)
      [W / 2 - 0.12, (H - 1.02) / 2, 1.08, 0, (H + 1.02) / 2, -0.68], // cabin + roof
    ];
    const jb = this.json?.chassisBoxes;
    if (Array.isArray(jb) && jb.length) boxes = jb.map((b) => [...b.halfExtents, ...b.center]);
    // low bumpers / skid plates: the first thing that meets a log, kerb or trench wall (bottom ~0.31 m)
    boxes.push([W / 2 - 0.05, 0.22, 0.10, 0, 0.53, L / 2 - 0.12], [W / 2 - 0.05, 0.18, 0.08, 0, 0.56, -L / 2 + 0.10]);
    const cg = groups(G.CAR, G.STATIC | G.CAR | G.ROCK | G.DEBRIS | G.PROP | G.SENSOR);
    for (const [hx, hy, hz, cx, cy, cz] of boxes) {
      const d = RAPIER.ColliderDesc.cuboid(hx, hy, hz).setTranslation(cx, cy, cz)
        .setDensity(0).setFriction(0.35).setRestitution(0.05).setCollisionGroups(cg);
      const c = world.createCollider(d, this.body);
      this.colliders.push(c);
      physics.onContactForce(c, (other, f) => { this._stepImpulse += f * (this.ctx.config.physics.step || 1 / 60); }, 25000);
    }
    // Tyre "faces": frictionless balls at the wheels, bottom ~0.11 m above the static ground. Rays can't see vertical
    // faces, so these stop the wheels at walls/steps (trench far wall, kerbs, rocks) and act as the hard bottoming stop.
    this.tyreColliders = this.wheels.map((w) => {
      const d = RAPIER.ColliderDesc.ball(s.wheelRadius * 0.92).setTranslation(w.rest.x, w.rest.y + 0.075, w.rest.z)
        .setDensity(0).setFriction(0).setFrictionCombineRule(RAPIER.CoefficientCombineRule.Min)
        .setRestitution(0.05).setCollisionGroups(cg);
      const c = world.createCollider(d, this.body);
      physics.onContactForce(c, (other, f) => { this._stepImpulse += f * (this.ctx.config.physics.step || 1 / 60); }, 25000);
      this.colliders.push(c);
      return c;
    });
    this.controller = world.createVehicleController(this.body);
    this.controller.indexUpAxis = 1;
    this.controller.setIndexForwardAxis = 2; // (setter property in the Rapier d.ts)
    this.wheelGroups = groups(G.CAR, G.STATIC | G.ROCK | G.DEBRIS | G.PROP);
    this._rebuildWheels(true);
  }

  _applyMassProps() {
    if (!this.body) return;
    const s = this.spec, m = s.mass;
    const l = s.length, w = s.width, h = 1.35;
    const I = { x: m / 12 * (h * h + l * l) * 0.72, y: m / 12 * (w * w + l * l) * 0.78, z: m / 12 * (w * w + h * h) * 0.80 };
    this.body.setAdditionalMassProperties(m, { x: 0, y: s.cgHeight, z: s.comZ }, I, { x: 0, y: 0, z: 0, w: 1 }, true);
  }

  /** (Re)create the 4 ray-cast wheels from the wheel node rest positions. */
  _rebuildWheels(first = false) {
    const vc = this.controller, s = this.spec;
    const R = s.wheelRadius;
    // Suspension: rest length = full droop length (Rapier rays only reach rest+radius). Static sag chosen, spring rates
    // derived per axle so both axles sag equally (Rapier force = k * x * chassisMass).
    const sag = 0.105, travel = 0.24;
    const staticLen = 0.17, rest = staticLen + sag;
    const a = s.frontAxleZ - s.comZ, b = s.comZ - s.rearAxleZ, L = a + b;
    const frac = [b / L / 2, b / L / 2, a / L / 2, a / L / 2];
    this.susp = { sag, travel, staticLen, rest, min: rest - travel };
    if (!first) { // hot swap: update in place
      for (let i = 0; i < 4; i++) {
        const w = this.wheels[i];
        vc.setWheelChassisConnectionPointCs(i, { x: w.rest.x, y: w.rest.y + staticLen, z: w.rest.z });
        vc.setWheelRadius(i, R);
        this.tyreColliders?.[i]?.setTranslationWrtParent({ x: w.rest.x, y: w.rest.y + 0.075, z: w.rest.z });
      }
      return;
    }
    for (let i = 0; i < 4; i++) {
      const w = this.wheels[i];
      vc.addWheel({ x: w.rest.x, y: w.rest.y + staticLen, z: w.rest.z }, { x: 0, y: -1, z: 0 }, { x: -1, y: 0, z: 0 }, rest, R);
      const k = 9.81 * frac[i] / sag;                 // 1/s^2 per unit chassis mass
      const zc = 2 * Math.sqrt(k * frac[i]);          // critical damping coefficient (Rapier units)
      w.k = k; w.kCur = k;
      vc.setWheelSuspensionStiffness(i, k);
      vc.setWheelSuspensionCompression(i, 0.30 * zc); // old worn dampers: soft bump,
      vc.setWheelSuspensionRelaxation(i, 0.52 * zc);  // firmer rebound
      vc.setWheelMaxSuspensionTravel(i, travel);
      vc.setWheelMaxSuspensionForce(i, 80000);
      vc.setWheelFrictionSlip(i, SURFACES.asphalt.mu);
      vc.setWheelSideFrictionStiffness(i, 0.92);
      vc.setWheelBrake(i, 0); vc.setWheelEngineForce(i, 0); vc.setWheelSteering(i, 0);
    }
  }

  // ------------------------------------------------------------------------------------------------ public API
  start() {
    if (this.engineOn) return true;
    if (this.fuel <= 0.0005) {
      this.cranking = 1.1; this._crankFail = true;
      this.ctx.events.emit('car:startFail', { reason: 'fuel' });
      return false;
    }
    this.engineOn = true; this.stalling = false; this._stallT = 0; this._starve = 0;
    this.cranking = 0.55; this._crankFail = false; this.rpm = Math.max(this.rpm, 300);
    if (this.gear === 0) this.gear = 1;
    this.ctx.events.emit('car:start', {});
    return true;
  }

  stop() {
    if (!this.engineOn) return;
    this.engineOn = false; this.stalling = false; this._starve = 0;
  }

  refuel(liters) {
    this.fuel = clamp(this.fuel + (liters || 0), 0, this.fuelCapacity);
    return this.fuel;
  }

  setControlsEnabled(b) { this.controlsEnabled = !!b; if (!b) { this.throttle = 0; this.brake = 0; } }

  /** Place the car on the road at (s, d), facing +s (rotated by yawOffset), at rest (or with opts.speed m/s). */
  teleport(s, d = -1.5, yawOffset = 0, opts = {}) {
    const { road, physics } = this.ctx;
    if (!road || !this.body) return;
    const p = road.worldAt(s, d, _v1);
    const q = road.frameQuat(s, _q1);
    if (yawOffset) q.premultiply(_q2.setFromAxisAngle(AX_Y, yawOffset));
    // ground under the car centre (static geometry only)
    let y = p.y - Math.abs(d) * 0.02;
    const gh = physics.raycast(_v2.set(p.x, p.y + 4, p.z), _v3.set(0, -1, 0), 12, { groups: groups(G.ALL, G.STATIC | G.PROP), excludeBody: this.body });
    if (gh) y = gh.point.y;
    this.body.setTranslation({ x: p.x, y: y + 0.03, z: p.z }, true);
    this.body.setRotation({ x: q.x, y: q.y, z: q.z, w: q.w }, true);
    const v0 = opts.speed || 0;
    const fwd = _v4.set(0, 0, 1).applyQuaternion(q);
    this.body.setLinvel({ x: fwd.x * v0, y: fwd.y * v0, z: fwd.z * v0 }, true);
    this.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    if (this.frozen) { this.body.setEnabled(true); this.frozen = false; }
    this.speed = v0; this.steer = 0; this.steerAngle = 0; this.throttle = 0; this.brake = 0;
    this._shiftT = 0; this._sinceShift = 0; this.gear = 1;
    if (v0 > 0) { let g = 1; while (g < 5 && this._coupledRpm(v0, g) > 3000) g++; this.gear = g; }
    this.rpm = this.engineOn ? Math.max(ENGINE.idle, this._coupledRpm(v0, this.gear)) : 0;
    this.rolledOver = false; this._rollT = 0; this._impAcc = 0; this._stepImpulse = 0;
    this._proj = {};
    // reset interpolation so the first rendered frame doesn't lerp across the map
    this._prevPos.set(p.x, y + 0.03, p.z); this._prevQuat.copy(q);
    this._curPos.copy(this._prevPos); this._curQuat.copy(q);
    this.object.position.copy(this._prevPos); this.object.quaternion.copy(q);
    this.object.updateMatrixWorld(true);
  }

  /** Rapier disables a body whose state became non-finite: zero the velocities and switch it back on. */
  _recoverBody() {
    const b = this.body;
    const v = b.linvel(), w = b.angvel(), t = b.translation();
    if (!isFinite(v.x + v.y + v.z)) b.setLinvel({ x: 0, y: 0, z: 0 }, false);
    if (!isFinite(w.x + w.y + w.z)) b.setAngvel({ x: 0, y: 0, z: 0 }, false);
    b.setEnabled(true);
    if (!isFinite(t.x + t.y + t.z)) { this.teleport(isFinite(this._proj?.s) ? this._proj.s : (this.ctx.road?.markers?.carStart ?? 40), -1.5); }
    b.wakeUp();
    if (!this._recoverWarned) { this._recoverWarned = true; console.info('[car] physics body was disabled (non-finite state): re-enabled'); }
  }

  /** Put the car back on its wheels on the nearest road position (debug / recovery). */
  resetUpright() {
    const t = this.body.translation();
    const pr = this.ctx.road.project(_v1.set(t.x, t.y, t.z), {});
    this.teleport(pr.s, clamp(pr.d, -2.5, 2.5));
  }

  /** World-space velocity (m/s) of the chassis. */
  velocity(out = new THREE.Vector3()) { const v = this.body?.linvel(); return v ? out.set(v.x, v.y, v.z) : out.set(0, 0, 0); }
  get kmh() { return this.speed * 3.6; }
  get position() { return this.object.position; }

  // ------------------------------------------------------------------------------------------------ fixed step
  fixedUpdate(h) {
    if (!this.body || !this.controller) return;
    this._acc -= h;
    const t = this.body.translation(), r = this.body.rotation();
    this._prevPos.set(t.x, t.y, t.z); this._prevQuat.set(r.x, r.y, r.z, r.w);
    if (this.frozen) return;
    // Rapier applies setAdditionalMassProperties at the next world step: until then mass() is 0 and any impulse (vehicle
    // controller suspension, anti-roll bar) turns the velocity into NaN, and Rapier then silently disables the body.
    // (Happens whenever the query pipeline is already up to date before the first step, e.g. after another system stepped.)
    if (!(this.body.mass() > 0)) return;
    if (!this.body.isEnabled()) this._recoverBody();
    this._stepCount++;

    const { ctx } = this;
    const vc = this.controller, s = this.spec, R = s.wheelRadius;
    const q = this._prevQuat;
    const fwd = _fwd.set(0, 0, 1).applyQuaternion(q);
    const up = _up.set(0, 1, 0).applyQuaternion(q);
    const leftAxis = _left.set(1, 0, 0).applyQuaternion(q);
    const lv = this.body.linvel();
    const vel = _vel.set(lv.x, lv.y, lv.z);
    const v = vel.dot(fwd);
    this.speed = v;
    const vAbs = Math.abs(v);
    this.upright = up.y;

    // --- falling out of the world (e.g. ?only=car with no terrain): freeze quietly once
    if (ctx.road) {
      const pr = ctx.road.project(this._prevPos, this._proj);
      if (pr.dy < -80 || !isFinite(t.y)) {
        this.frozen = true; this.body.setEnabled(false);
        console.info('[car] no ground under the car: body frozen');
        return;
      }
    }

    // --- impacts from the previous step's contact forces
    this._processImpacts(h, vel);

    // --- driver input
    let thrIn = 0, brkIn = 0, steerIn = 0, hb = false, driven = false;
    const input = ctx.input;
    const human = ctx.control === 'car' && this.controlsEnabled && !ctx.paused && input?.active;
    if (this.aiInput) {
      const a = this.aiInput; driven = true;
      thrIn = clamp(a.throttle || 0, 0, 1); brkIn = clamp(a.brake || 0, 0, 1); hb = !!a.handbrake;
      steerIn = clamp(a.steer || 0, -1, 1);
      if (a.gear !== undefined && a.gear !== this.gear && this._shiftT <= 0) this._shift(a.gear);
    } else if (this.autopilot) {
      driven = true;
      const a = this._autopilot(v);
      thrIn = a.throttle; brkIn = a.brake; steerIn = a.steer;
    } else if (human) {
      driven = true;
      const w = input.down('KeyW') || input.down('ArrowUp');
      const sDown = input.down('KeyS') || input.down('ArrowDown');
      steerIn = (input.down('KeyA') || input.down('ArrowLeft') ? 1 : 0) - (input.down('KeyD') || input.down('ArrowRight') ? 1 : 0);
      hb = input.down('Space');
      if (this.gear >= 0) {
        thrIn = w ? 1 : 0; brkIn = sDown ? 1 : 0;
        if (sDown && !w && v < 0.7) { this._revHold += h; if (this._revHold > 0.22) { this._shift(-1); this._revHold = 0; } } else this._revHold = 0;
      } else {
        thrIn = sDown ? 1 : 0; brkIn = w ? 1 : 0;
        if (w && !sDown && v > -0.7) { this._fwdHold += h; if (this._fwdHold > 0.15) { this._shift(1); this._fwdHold = 0; } } else this._fwdHold = 0;
      }
    }
    this.handbrake = hb;

    // pedals (keyboard is digital: ramp them like a foot would)
    this.throttle = approach(this.throttle, thrIn, (thrIn > this.throttle ? 3.2 : 6.0) * h);
    this.brake = approach(this.brake, brkIn, (brkIn > this.brake ? 4.0 : 7.0) * h);

    // steering: speed-sensitive lock, return-to-centre (self-aligning) that grows with speed
    const maxSteer = STEER.max / (1 + Math.pow(vAbs / 10.5, 1.5));
    if (this.aiInput || this.autopilot) {
      this.steer = approach(this.steer, steerIn, 3.0 * h);
    } else if (steerIn !== 0) {
      const opposite = this.steer !== 0 && Math.sign(this.steer) !== steerIn;
      const rateIn = 2.1 / (1 + vAbs / 16);
      this.steer = approach(this.steer, steerIn, (opposite ? rateIn + 2.2 : rateIn) * h);
    } else {
      this.steer = approach(this.steer, 0, (1.4 + vAbs * 0.12) * h);
    }
    this.steerAngle = this.steer * maxSteer;

    // --- fuel, sputter, stall, cranking
    this._updateFuel(h, vAbs);

    // --- drivetrain
    const drive = this._updateDrivetrain(h, v);

    // --- per-wheel forces
    const a = this.steerAngle, WB = s.wheelbase, TR = s.track;
    // Ackermann: inner wheel steers more
    let dl = a, dr = a;
    if (Math.abs(a) > 1e-4) {
      const Rt = WB / Math.tan(Math.abs(a));
      const inner = Math.atan(WB / (Rt - TR / 2)), outer = Math.atan(WB / (Rt + TR / 2));
      if (a > 0) { dl = inner; dr = outer; } else { dl = -outer; dr = -inner; }
    }
    this._steerW[0] = dl; this._steerW[1] = dr; this._steerW[2] = 0; this._steerW[3] = 0;

    // surface sampling: one wheel per step (15 Hz per wheel)
    const si = this._surfIdx = (this._surfIdx + 1) & 3;
    if (vc.wheelIsInContact(si)) this._surf[si] = this._surfaceUnder(si);

    const hold = this.throttle < 0.04 && vAbs < 0.35; // hill-hold / parking brake at standstill (also with nobody driving)
    const dir = this.gear >= 0 ? 1 : -1;
    let contacts = 0;
    for (let i = 0; i < 4; i++) {
      const w = this.wheels[i];
      vc.setWheelSteering(i, this._steerW[i]);
      const inContact = vc.wheelIsInContact(i);
      if (inContact) contacts++;
      const N = Math.max(0, vc.wheelSuspensionForce(i) || 0);
      const sf = SURFACES[this._surf[i]] || SURFACES.asphalt;
      // slope derating: tyres can't climb steep banks / the cut face
      let slopeF = 1;
      if (inContact) {
        const n = vc.wheelContactNormal(i);
        if (n) slopeF = 1 - 0.88 * smooth01(27 * DEG, 44 * DEG, Math.acos(clamp(n.y, -1, 1)));
      }
      const mu = sf.mu * slopeF * this.gripScale;
      // brakes (tyre limited, no ABS)
      let Fb = this.brake * BRAKES.total * (w.front ? BRAKES.front : 1 - BRAKES.front) / 2;
      if (hb && !w.front) Fb += BRAKES.handbrake;
      if (hold) Fb = Math.max(Fb, BRAKES.hold);
      const grip = mu * N;
      const locked = inContact && vAbs > 1.0 && ((Fb > grip * 1.02 && Fb > 400) || (hb && !w.front));
      this._locked[i] = locked;
      Fb = Math.min(Fb, grip * 0.92);
      const Frr = sf.rr * N;
      const split = (w.front ? ENGINE.frontSplit : 1 - ENGINE.frontSplit) / 2;
      const Fd = (hb && !w.front) ? 0 : drive.force * split;
      const Feb = drive.engineBrake * split;
      if (Fb > 1 || Fd <= 0) {
        vc.setWheelEngineForce(i, 0);
        vc.setWheelBrake(i, (Fb + Frr + Feb) * h);
      } else if (Fd > Frr) {
        vc.setWheelEngineForce(i, dir * (Fd - Frr));
        vc.setWheelBrake(i, 0);
      } else {
        vc.setWheelEngineForce(i, 0);
        vc.setWheelBrake(i, (Frr - Fd) * h);
      }
      // lateral grip: locked/handbraked wheels slide
      vc.setWheelFrictionSlip(i, Math.max(0.05, mu * (locked ? 0.55 : 1)));
      // progressive bump stop in the last 40% of compression travel
      const len = vc.wheelSuspensionLength(i) ?? this.susp.staticLen;
      const bs0 = this.susp.min + 0.05;
      let k = w.k;
      if (inContact && len < bs0) {
        const pen = (bs0 - len) / 0.05;
        k = w.k * (1 + 6 * pen * pen) ;
      }
      if (Math.abs(k - w.kCur) > 0.01) { vc.setWheelSuspensionStiffness(i, k); w.kCur = k; }
      w.len = len; w.contact = inContact;
    }
    this.wheelsInContact = contacts;
    // audio / camera cues from the last step: tyre sliding (0..1), suspension jolts (m/s), dominant surface
    let skid = 0, bump = 0; const cnt = {};
    for (let i = 0; i < 4; i++) {
      const w = this.wheels[i];
      if (!w.contact) { w.lenPrev = w.len; continue; }
      const N = vc.wheelSuspensionForce(i) || 0, fs = vc.wheelFrictionSlip(i) || 1;
      const lim = N * h * fs;
      if (lim > 1e-3) skid = Math.max(skid, Math.abs(vc.wheelSideImpulse(i) || 0) / lim);
      if (this._locked[i]) skid = Math.max(skid, Math.min(1, vAbs / 4));
      bump = Math.max(bump, Math.abs((w.len - (w.lenPrev ?? w.len)) / h));
      w.lenPrev = w.len;
      cnt[this._surf[i]] = (cnt[this._surf[i]] || 0) + 1;
    }
    this.skid = lerp(this.skid || 0, clamp((skid - 0.75) / 0.25, 0, 1) * smooth01(1.5, 4, vAbs), 0.25);
    this.bump = Math.max(bump, (this.bump || 0) * 0.85);
    let best = 'asphalt', bc = 0; for (const k in cnt) if (cnt[k] > bc) { bc = cnt[k]; best = k; }
    this.surface = best;

    // front anti-roll bar (soft: old 4x4). Pure roll torque from the left/right compression difference.
    const w0 = this.wheels[0], w1 = this.wheels[1];
    if (w0.contact && w1.contact) {
      const diff = (w1.len - w0.len); // >0: left compressed more
      const F = clamp(diff * 16000, -6000, 6000);
      if (Math.abs(F) > 1) {
        const I = F * h;
        const hpL = _hp.copy(w0.rest).applyQuaternion(q).add(this._prevPos);
        this.body.applyImpulseAtPoint({ x: up.x * I, y: up.y * I, z: up.z * I }, { x: hpL.x, y: hpL.y, z: hpL.z }, true);
        const hpR = _hp.copy(w1.rest).applyQuaternion(q).add(this._prevPos);
        this.body.applyImpulseAtPoint({ x: -up.x * I, y: -up.y * I, z: -up.z * I }, { x: hpR.x, y: hpR.y, z: hpR.z }, true);
      }
    }

    // aero drag
    const sp = vel.length();
    if (sp > 0.5) {
      const k = -AERO * sp * h;
      this.body.applyImpulse({ x: vel.x * k, y: vel.y * k, z: vel.z * k }, true);
    }

    vc.updateVehicle(h, RAPIER.QueryFilterFlags.EXCLUDE_SENSORS, this.wheelGroups);
    this._restoreRollMoment(q, up);

    // telemetry: accelerations in the body frame (g)
    const dv = _v4.set(lv.x - this._lastVel.x, lv.y - this._lastVel.y, lv.z - this._lastVel.z).divideScalar(h);
    this.lateralG = lerp(this.lateralG, dv.dot(leftAxis) / 9.81, 0.15);
    this.longG = lerp(this.longG, dv.dot(fwd) / 9.81, 0.15);
    this._lastVel.set(lv.x, lv.y, lv.z);

    // rollover
    if (this.upright < 0.3) { this._rollT += h; if (this._rollT > 1.2 && !this.rolledOver) { this.rolledOver = true; ctx.events.emit('car:rollover', {}); } }
    else { this._rollT = 0; if (this.upright > 0.8) this.rolledOver = false; }

    // mud accumulation
    let mudRate = 0;
    for (let i = 0; i < 4; i++) mudRate += (SURFACES[this._surf[i]]?.mud || 0) * 0.25;
    this.mud = clamp(this.mud + mudRate * Math.min(vAbs, 15) * h * 0.004 - h * 0.0004 * (ctx.env?.rain ?? 0.35), 0, 1);
  }

  /**
   * Rapier (like Bullet) applies each tyre's side impulse at a point scaled toward the CoM height (rollInfluence = 0.1,
   * not exposed in JS), which removes ~90% of the cornering roll moment. Put the missing roll torque back so the body
   * leans like a real soft 4x4 (roll gradient ~5-6 deg/g).
   */
  _restoreRollMoment(q, up) {
    const vc = this.controller;
    const com = this.body.worldCom();
    let tx = 0, ty = 0, tz = 0;
    for (let i = 0; i < 4; i++) {
      if (!vc.wheelIsInContact(i)) continue;
      const si = vc.wheelSideImpulse(i);
      if (!si) continue;
      const cp = vc.wheelContactPoint(i), n = vc.wheelContactNormal(i);
      if (!cp || !n) continue;
      // steered axle in world space, projected onto the contact plane (same construction Rapier uses)
      const ax = _v4.set(-Math.cos(this._steerW[i]), 0, Math.sin(this._steerW[i])).applyQuaternion(q);
      const dn = ax.x * n.x + ax.y * n.y + ax.z * n.z;
      ax.set(ax.x - n.x * dn, ax.y - n.y * dn, ax.z - n.z * dn).normalize();
      const Fx = ax.x * si, Fy = ax.y * si, Fz = ax.z * si;
      const rx = cp.x - com.x, ry = cp.y - com.y, rz = cp.z - com.z;
      const k = (up.x * rx + up.y * ry + up.z * rz) * 0.9;
      const mx = up.x * k, my = up.y * k, mz = up.z * k; // missing lever arm
      tx += my * Fz - mz * Fy; ty += mz * Fx - mx * Fz; tz += mx * Fy - my * Fx;
    }
    if (tx || ty || tz) this.body.applyTorqueImpulse({ x: tx * this.rollMomentScale, y: ty * this.rollMomentScale, z: tz * this.rollMomentScale }, true);
  }

  _coupledRpm(v, gear) {
    const ratio = gear === 0 ? 0 : (gear > 0 ? ENGINE.gears[gear - 1] : ENGINE.reverse) * ENGINE.final;
    return Math.abs(v) / this.spec.wheelRadius * ratio * 60 / (2 * Math.PI);
  }

  _shift(g) {
    if (g === this.gear) return;
    const prev = this.gear;
    this.gear = g;
    this._shiftT = (prev > 0 && g > 0) ? ENGINE.shiftTime : 0.5; // engaging D/R takes a moment too
    this._sinceShift = 0;
    this.ctx.events.emit('car:gear', { gear: g, prev });
  }

  _updateDrivetrain(h, v) {
    const E = ENGINE;
    const R = this.spec.wheelRadius;
    if (this._shiftT > 0) this._shiftT -= h;
    this._sinceShift += h;
    const ratio = this.gear === 0 ? 0 : (this.gear > 0 ? E.gears[this.gear - 1] : E.reverse) * E.final;
    // wheel speed in the direction of the selected gear (rolling backwards in D -> 0)
    const vGear = this.gear >= 0 ? v : -v;
    const coupled = Math.max(0, vGear) / R * ratio * 60 / (2 * Math.PI);
    this.coupledRpm = coupled;

    // automatic shifting
    if (this.engineOn && this.gear > 0 && this._shiftT <= 0 && this._sinceShift > 0.9) {
      const thr = this.throttle, g = this.gear;
      const upAt = lerp(2350, 5300, Math.pow(thr, 0.85));
      const downAt = lerp(1150, 2500, thr);
      if (g < 5 && coupled > upAt) this._shift(g + 1);
      else if (g > 1) {
        const lower = coupled * E.gears[g - 2] / E.gears[g - 1];
        if ((coupled < downAt || (thr > 0.95 && lower < 4300)) && lower < 5000) this._shift(g - 1);
      }
    }

    let thrEff = this.engineOn ? this.throttle : 0;
    if (this.gear < 0) thrEff *= 1 - smooth01(REVERSE_MAX - 1.5, REVERSE_MAX, -v);
    if (this._starve > 0) thrEff = 0;                          // fuel cut: engine coughs
    else if (this.stalling) thrEff = Math.min(thrEff, 0.3 * Math.pow(1 - this.sputter, 2.5)); // running lean: power ceiling falls as the bowl empties
    if (this._limiterCut > 0) { this._limiterCut -= h; thrEff = 0; }
    if (this.rpm > E.limiter) this._limiterCut = 0.05;
    // idle controller keeps it running
    const idleThr = this.engineOn && this._starve <= 0 ? clamp((E.idle - this.rpm) / 400, 0, 0.25) : 0;

    let target, mult = 1, couple = 0;
    const shifting = this._shiftT > 0;
    if (!this.engineOn) {
      target = this.cranking > 0 ? 240 + 60 * Math.sin(this._stepCount * 0.9) : 0;
    } else if (ratio === 0 || shifting) {
      target = E.idle + thrEff * (E.limiter - E.idle) * (shifting ? 0.35 : 1);
      if (shifting) target = Math.max(target, coupled * 0.9);
    } else {
      const conv = E.idle + thrEff * (E.convStall - E.idle);
      target = Math.max(coupled, conv);
      const sr = coupled / Math.max(conv, 1);
      mult = sr < 0.85 ? lerp(1.9, 1.0, sr / 0.85) : 1;
      couple = smooth01(0.35, 0.95, sr);
    }
    if (this._starve > 0 && this.engineOn) target = Math.min(target, E.idle * 0.75); // stumbling
    const tau = !this.engineOn ? 0.35 : (target > this.rpm ? (couple > 0.9 ? 0.05 : 0.10) : (couple > 0.9 ? 0.05 : 0.20));
    this.rpm += (target - this.rpm) * (1 - Math.exp(-h / tau));
    if (!this.engineOn && this.cranking <= 0 && this.rpm < 30) this.rpm = 0;

    const Tmax = torqueAt(this.rpm);
    const Tfric = 16 + this.rpm * 0.0058; // pumping + friction losses (engine braking)
    let Te = this.engineOn ? (thrEff + idleThr * 0.3) * Tmax - (1 - thrEff) * Tfric : 0;
    let force = 0, engineBrake = 0;
    if (ratio > 0 && this.engineOn) {
      const shiftF = shifting ? 0.3 : 1;
      if (Te > 0) force = Te * mult * ratio * E.eff / R * shiftF;
      else engineBrake = -Te * couple * ratio / R * shiftF;
      // engine braking only resists forward motion in gear direction
      if (vGear < 0.5) engineBrake = 0;
    }
    this.engineTorque = Te;
    return { force, engineBrake };
  }

  _updateFuel(h, vAbs) {
    const g = this.ctx.config?.game || {};
    if (this.cranking > 0) {
      this.cranking -= h;
      if (this.cranking <= 0 && this._crankFail) { this._crankFail = false; }
    }
    if (!this.engineOn) {
      this.stalling = false;
      return;
    }
    // Consumption per meter ∝ load (throttle × rpm): config.game.fuelPerMeterBase at light throttle, up to
    // fuelThrottleFactor × that at full throttle / high rpm (carburettor power enrichment), plus a small idle flow.
    // FUEL_CAL calibrates the model so config.game.fuelStart from markers.carStart runs dry at markers.stall at a
    // natural 40-60 km/h pace up the 4-6 % cold-open grade (measured in the full game; see scratch/vehicle).
    const base = (g.fuelPerMeterBase ?? 0.0028) * FUEL_CAL, factor = g.fuelThrottleFactor ?? 1.8;
    const load = clamp(this.throttle * (0.3 + 0.7 * this.rpm / ENGINE.redline) * 1.25, 0, 1);
    const perMeter = base * (1 + (factor - 1) * load);
    const idlePerSec = base * 0.3 * (this.rpm / ENGINE.idle);
    const flow = perMeter * vAbs + idlePerSec; // L/s
    this.fuelFlow = flow;
    if (this.fuel > SPUTTER_L) {
      this.fuel = Math.max(SPUTTER_L * 0.999, this.fuel - flow * h * this.fuelScale);
    } else {
      // Reserve phase: the pickup draws air, the carburettor bowl runs dry. Time-based so the drama always lasts
      // ~SPUTTER_T s (engine coughs, power fades, the car loses speed on the grade and finally dies).
      this.fuel = Math.max(0, this.fuel - SPUTTER_L / SPUTTER_T * (0.7 + 0.3 * this.throttle) * h);
    }

    // fuel starvation: sputter when nearly empty (engine cuts in and out, cuts get longer), stall at 0
    if (this.fuel < SPUTTER_L) {
      this.stalling = true;
      const e = this.sputter = 1 - this.fuel / SPUTTER_L; // 0..1 (1 = empty)
      if (this._starve > 0) {
        this._starve -= h;
      } else {
        this._sputterTimer -= h;
        if (this._sputterTimer <= 0) {
          this._starve = 0.08 + Math.random() * (0.15 + 0.6 * e);
          this._sputterTimer = this._starve + 0.15 + Math.random() * (1.1 - 0.9 * e);
        }
      }
    } else { this.stalling = false; this._starve = 0; this.sputter = 0; }
    if (this.fuel <= 0) {
      this.engineOn = false; this.stalling = false; this._starve = 0; this.sputter = 0; this.fuelFlow = 0;
      this.ctx.events.emit('car:stall', { reason: 'fuel' });
    }
  }

  _surfaceUnder(i) {
    const vc = this.controller;
    const col = vc.wheelGroundObject(i);
    if (col) {
      const mem = (col.collisionGroups() >>> 16) & 0xffff;
      const parent = col.parent?.();
      if (mem & G.PROP) return 'wood';
      if (parent && parent.isDynamic?.()) return 'rock';
    }
    const t = this.ctx.terrain;
    if (t?.surfaceAt) {
      try {
        const p = vc.wheelContactPoint(i);
        if (p) { const s = t.surfaceAt(_v4.set(p.x, p.y, p.z)); if (SURFACES[s]) return s; }
      } catch { /* terrain not ready */ }
    }
    return 'asphalt';
  }

  _autopilot(v) {
    const ap = this.autopilot, road = this.ctx.road;
    const pos = this._prevPos;
    const pr = road.project(pos, this._proj);
    const look = clamp(4.5 + Math.abs(v) * 0.75, 5, 20);
    const tgt = road.worldAt(pr.s + look, ap.d ?? -1.5, _v4);
    const q = this._prevQuat;
    const dx = tgt.x - pos.x, dz = tgt.z - pos.z;
    const left = _v2.set(1, 0, 0).applyQuaternion(q), fwd = _v1.set(0, 0, 1).applyQuaternion(q);
    const lx = dx * left.x + dz * left.z, lz = dx * fwd.x + dz * fwd.z;
    const kappa = 2 * lx / Math.max(1, lx * lx + lz * lz);
    const ang = Math.atan(this.spec.wheelbase * kappa);
    const maxSteer = STEER.max / (1 + Math.pow(Math.abs(v) / 10.5, 1.5));
    const steer = clamp(ang / maxSteer, -1, 1);
    if (ap.speed === null || ap.speed === undefined) return { steer, throttle: 0, brake: ap.brake || 0 }; // steer-only (coasting)
    const err = ap.speed - v;
    return { steer, throttle: clamp(err * 0.45, 0, 1), brake: clamp(-err * 0.3 - 0.05, 0, 1) };
  }

  _processImpacts(h, vel) {
    const imp = this._stepImpulse; this._stepImpulse = 0;
    this._impAcc = this._impAcc * 0.55 + imp;
    if (this._impCooldown > 0) this._impCooldown -= h;
    if (this._impAcc > 1600 && this._impCooldown <= 0) {
      const m = this.spec.mass;
      const impulse = this._impAcc;
      const energy = 0.5 * impulse * impulse / m / 1e5;
      const dir = _v4.copy(vel); const sp = dir.length();
      if (sp > 0.3) dir.multiplyScalar(1 / sp); else dir.set(0, 0, 1).applyQuaternion(this._prevQuat);
      const position = new THREE.Vector3().copy(this._prevPos).addScaledVector(dir, 1.6).add(_v1.set(0, 0.6, 0));
      this.damage = clamp(this.damage + Math.max(0, impulse - 1600) / 45000, 0, 1);
      this.ctx.events.emit('car:impact', { impulse, energy, position, speed: sp });
      this.ctx.events.emit('impact', { position, energy, radius: 1.2, source: 'car' });
      this._impCooldown = 0.35; this._impAcc = 0;
    }
  }

  // ------------------------------------------------------------------------------------------------ per frame
  update(dt) {
    if (!this.body) return;
    const H = this.ctx.config?.physics?.step || 1 / 60;
    this._acc = clamp(this._acc + dt, 0, H * 0.999);
    const alpha = this.frozen ? 1 : this._acc / H;
    const t = this.body.translation(), r = this.body.rotation();
    this._curPos.set(t.x, t.y, t.z); this._curQuat.set(r.x, r.y, r.z, r.w);
    if (this.frozen) { this._prevPos.copy(this._curPos); this._prevQuat.copy(this._curQuat); }
    this.object.position.lerpVectors(this._prevPos, this._curPos, alpha);
    this.object.quaternion.slerpQuaternions(this._prevQuat, this._curQuat, alpha);
    if (dt <= 0) { this.object.updateMatrixWorld(true); return; }

    this._updateWheelsVisual(dt);
    this._updateMaterials(dt);
    this._updateGauges(dt);

    // engine shake on the body shell (idle rumble, sputter shudder)
    if (this.bodyNode && this.bodyNode !== this.model) {
      this._shakeT += dt;
      const on = this.engineOn || this.cranking > 0;
      const idleAmt = on ? (this.rpm < 1400 ? 0.0012 : 0.0005) : 0;
      const sput = this._starve > 0 ? 0.004 : 0;
      const crank = this.cranking > 0 ? 0.0025 : 0;
      const a = idleAmt + sput + crank;
      const f = Math.max(this.rpm, 200) / 60 * 2 * Math.PI * 0.5; // 2nd order of a 4-cylinder (per crank rev / 2)
      this.bodyNode.rotation.z = a * Math.sin(this._shakeT * Math.min(f, 90));
      this.bodyNode.rotation.x = a * 0.5 * Math.sin(this._shakeT * Math.min(f, 90) * 0.71 + 1.3);
    }

    this._updateDangles(dt);
    // world matrices current now, so the camera rig (updated after us) can read seatCam/doorPoint this frame
    this.object.updateMatrixWorld(true);
    this._updateCabin();
    this._updateMirror();
    this.contactShadow?.update(dt);
    this.spray?.update(dt);
    this._updateLod();

    // HUD
    this._hudT -= dt;
    if (this._hudT <= 0 && this.ctx.control === 'car') {
      this._hudT = 0.1;
      this.ctx.hud?.setSpeed?.(Math.abs(this.speed * 3.6));
      this.ctx.hud?.setFuel?.(this.fuel / this.fuelCapacity);
    }
  }

  _updateWheelsVisual(dt) {
    const vc = this.controller;
    const lv = this.body.linvel(), av = this.body.angvel();
    const q = this._curQuat;
    const fwd = _v1.set(0, 0, 1).applyQuaternion(q);
    const com = this.body.worldCom?.() || this.body.translation();
    for (let i = 0; i < 4; i++) {
      const w = this.wheels[i];
      const len = vc.wheelSuspensionLength(i) ?? this.susp.staticLen;
      w.node.position.set(w.rest.x, w.rest.y + this.susp.staticLen - len, w.rest.z);
      // ground speed at the wheel: v + w x r
      const wp = _v2.copy(w.node.position).applyQuaternion(q).add(this._curPos);
      const rx = wp.x - com.x, ry = wp.y - com.y, rz = wp.z - com.z;
      const vx = lv.x + av.y * rz - av.z * ry, vy = lv.y + av.z * rx - av.x * rz, vz = lv.z + av.x * ry - av.y * rx;
      let vl = vx * fwd.x + vy * fwd.y + vz * fwd.z;
      const contact = vc.wheelIsInContact(i);
      if (this._locked[i] || (this.handbrake && !w.front)) vl = 0;
      if (!contact) { // airborne wheel: coasts down slowly, or is spun by the engine
        vl = (w.freeV ?? 0) * Math.exp(-dt * 0.6);
        if (this.engineOn && this.throttle > 0.2 && this.gear !== 0 && !(this.handbrake && !w.front)) {
          vl = (this.gear > 0 ? 1 : -1) * this.rpm / this._ratio() * 2 * Math.PI / 60 * this.spec.wheelRadius;
        }
      }
      w.freeV = vl;
      this._spin[i] = (this._spin[i] + vl / this.spec.wheelRadius * dt) % (Math.PI * 2);
      _q1.setFromAxisAngle(AX_Y, this._steerW[i]);
      _q2.setFromAxisAngle(AX_X, this._spin[i]);
      w.node.quaternion.copy(_q1).multiply(_q2).multiply(w.q0);
    }
    if (this.steeringWheel) {
      _q1.setFromAxisAngle(AX_Z, this._swSign * this.steerAngle * (this.steerRatio || STEER.ratio));
      this.steeringWheel.quaternion.copy(this._swQ0).multiply(_q1);
    }
  }

  /** Interior (dash, gauges, seats, steering wheel) only when the camera is near the car or in the cockpit. */
  _updateLod() {
    const cam = this.ctx.camera;
    if (!cam || !this.interiorMeshes?.length) return;
    const d2 = cam.position.distanceToSquared(this.object.position);
    const show = d2 < 22 * 22 || this.ctx.cameraRig?.mode === 'car-cockpit';
    if (show === this._interiorShown) return;
    this._interiorShown = show;
    for (const m of this.interiorMeshes) m.visible = show;
  }

  _updateGauges(dt) {
    const N = this.needles;
    if (!N) return;
    const ign = this.engineOn || this.cranking > 0;
    this.engineTemp = clamp(this.engineTemp + dt * (this.engineOn ? 0.004 : -0.002), 0, 0.62);
    const set = (k, frac, zeroDeg, maxDeg, rate) => {
      const n = N[k]; if (!n) return;
      n.val += (clamp(frac, 0, 1.05) - n.val) * (1 - Math.exp(-dt * rate));
      n.node.rotation.z = (zeroDeg + (maxDeg - zeroDeg) * n.val) * DEG;
    };
    const c = (k, a, b, d0, d1) => { const g = N[k]?.cfg; return g ? [g[a] ?? d0, g[b] ?? d1] : [d0, d1]; };
    const sp = N.speedo?.cfg?.maxKmh ?? 160, tm = N.tacho?.cfg?.maxRpm ?? 7000;
    set('speedo', Math.abs(this.speed * 3.6) / sp, ...c('speedo', 'zeroDeg', 'maxDeg', 135, -135), 6);          // cable-driven: works without ignition
    set('tacho', ign ? this.rpm / tm : 0, ...c('tacho', 'zeroDeg', 'maxDeg', 135, -135), 10);
    set('fuel', ign ? Math.max(this.fuel / this.fuelCapacity, this.fuel > 0.001 ? 0.015 : 0) : 0, ...c('fuel', 'emptyDeg', 'fullDeg', 60, -60), 1.5);
    set('temp', ign ? this.engineTemp : 0, ...c('temp', 'coldDeg', 'hotDeg', 60, -60), 0.8);
  }

  _ratio() {
    if (this.gear === 0) return 1;
    return (this.gear > 0 ? ENGINE.gears[this.gear - 1] : ENGINE.reverse) * ENGINE.final;
  }

  _updateMaterials(dt) {
    const M = this.mats;
    const head = !!this.lights.head;
    const braking = this.brake > 0.08;
    this.lights.brake = braking;
    this._blinkT += dt;
    const blink = this.lights.hazard && (this._blinkT % 0.75) < 0.4;
    const ign = this.engineOn || this.cranking > 0;
    const dim = this.cranking > 0 ? 0.55 : 1; // lamps dip while cranking
    const set = (key, intensity) => {
      if (this._lastEmis[key] === intensity) return;
      this._lastEmis[key] = intensity;
      for (const m of M[key] || []) m.emissiveIntensity = intensity;
    };
    set('headlight', head ? 2.1 * dim : 0); // lens emission is not scaled by the 0.12 alpha (alphaGlass)
    if (this._hlFwd && this.object) this._hlFwd.value.set(0, 0, 1).applyQuaternion(this.object.getWorldQuaternion(_hlQ));
    set('brakelight', (braking ? 5.5 : 0) + (head ? 0.9 : 0));
    set('indicator', blink ? 5 : 0);
    set('gauge', ign ? 0.9 * dim : 0);
    if (this.headlightSource) {
      // daylight: the dipped beams barely register (real ratio to overcast daylight); in the tunnel they dominate the
      // sodium lamps like real low beams do
      const mk = this.ctx.road?.markers, s = this._proj?.s;
      const tun = mk && isFinite(s) ? smooth01(mk.tunnel - 20, mk.tunnel + 8, s) * (1 - smooth01(mk.tunnelEnd - 4, mk.tunnelEnd + 6, s)) : 0;
      this.headlightSource.intensity = head ? this.headlightIntensity * (1 + 2.5 * tun) * dim : 0;
    }

    // wetness on paint (rain beads make the finish glossier), mud on the tyres
    const wet = this.ctx.env?.wetness ?? 0.75;
    if (this._wetLast === undefined || Math.abs(this._wetLast - wet) > 0.01) {
      this._wetLast = wet;
      for (const m of M.paint || []) m.roughness = (m.userData.r0 ?? 0.5) * (1 - 0.35 * wet);
    }
    if (this._mudLast === undefined || Math.abs(this._mudLast - this.mud) > 0.01) {
      this._mudLast = this.mud;
      for (const m of M.rubber || []) {
        if (!m.userData.c0) continue;
        m.color.copy(m.userData.c0).lerp(_mudColor, this.mud * 0.7);
        m.roughness = lerp(m.userData.r0 ?? 0.9, 1, this.mud);
      }
    }
  }

  dispose() {
    clearTimeout(this._pollTimer);
    for (const off of this._offs || []) off();
    const { physics, scene } = this.ctx;
    try {
      for (const c of this.colliders) physics.removeHandlers(c);
      if (this.controller) physics.world.removeVehicleController(this.controller);
      if (this.body) physics.world.removeRigidBody(this.body);
    } catch { /* world gone */ }
    scene?.remove(this.object);
    if (this.contactShadow) { scene?.remove(this.contactShadow.mesh); this.contactShadow.dispose(); }
    if (this.spray) { scene?.remove(this.spray.mesh); this.spray.dispose(); }
    this.mirror?.rt?.dispose();
    this.body = null; this.controller = null;
  }
}

const _mudColor = new THREE.Color(0.16, 0.12, 0.085);
const _sodium = new THREE.Color(1.0, 0.55, 0.18);

function disposeTree(root) {
  root.traverse((o) => { if (o.isMesh) { o.geometry?.dispose?.(); } });
}

// ------------------------------------------------------------------------------------------------ placeholder car
// Procedural stand-in with the exact DESIGN node + material names; replaced by public/assets/models/car.glb when present.
function buildPlaceholderCar(spec) {
  const root = new THREE.Group(); root.name = 'car_placeholder';
  const body = new THREE.Group(); body.name = 'car_body'; root.add(body);
  const M = {
    paint: new THREE.MeshPhysicalMaterial({ name: 'paint', color: 0x4a1612, roughness: 0.62, metalness: 0.0, clearcoat: 0.25, clearcoatRoughness: 0.5 }),
    glass: new THREE.MeshPhysicalMaterial({ name: 'glass', color: 0x1a2226, roughness: 0.04, metalness: 0, transparent: true, opacity: 0.38, depthWrite: false }),
    chrome: new THREE.MeshStandardMaterial({ name: 'chrome', color: 0xb8b8b8, roughness: 0.28, metalness: 1 }),
    trim_black: new THREE.MeshStandardMaterial({ name: 'trim_black', color: 0x121212, roughness: 0.75 }),
    rubber: new THREE.MeshStandardMaterial({ name: 'rubber', color: 0x151515, roughness: 0.92 }),
    interior: new THREE.MeshStandardMaterial({ name: 'interior', color: 0x26231f, roughness: 0.85 }),
    headlight: new THREE.MeshStandardMaterial({ name: 'headlight', color: 0xdddddd, roughness: 0.1, emissive: 0xfff1d8, emissiveIntensity: 0 }),
    brakelight: new THREE.MeshStandardMaterial({ name: 'brakelight', color: 0x5a0805, roughness: 0.25, emissive: 0xff1a0a, emissiveIntensity: 0 }),
    indicator: new THREE.MeshStandardMaterial({ name: 'indicator', color: 0x6a3a08, roughness: 0.25, emissive: 0xff8a14, emissiveIntensity: 0 }),
    plate: new THREE.MeshStandardMaterial({ name: 'plate', color: 0xd8d8cc, roughness: 0.5 }),
    gauge: new THREE.MeshStandardMaterial({ name: 'gauge', color: 0x111111, roughness: 0.4, emissive: 0x9fe8ff, emissiveIntensity: 0 }),
  };
  const L = spec.length, W = spec.width, H = spec.height;
  const add = (geo, mat, x, y, z, parent = body, rx = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.rotation.x = rx; parent.add(m); return m; };
  // lower tub
  add(new RoundedBoxGeometry(W, 0.62, L - 0.12, 3, 0.07), M.paint, 0, 0.72, 0);
  // cabin greenhouse
  add(new RoundedBoxGeometry(W - 0.2, 0.62, 2.05, 2, 0.05), M.glass, 0, 1.36, -0.7);
  // pillars + roof
  const pill = new THREE.BoxGeometry(0.07, 0.62, 0.08);
  for (const x of [-1, 1]) for (const z of [0.33, -0.6, -1.7]) add(pill, M.paint, x * (W / 2 - 0.12), 1.36, z);
  add(new RoundedBoxGeometry(W - 0.16, 0.07, 2.12, 2, 0.03), M.paint, 0, H - 0.04, -0.7);
  // hood bulge, grille, bumpers
  add(new RoundedBoxGeometry(W - 0.3, 0.06, 1.2, 2, 0.03), M.paint, 0, 1.04, 1.25);
  add(new THREE.BoxGeometry(W - 0.5, 0.26, 0.04), M.trim_black, 0, 0.82, L / 2 - 0.04);
  add(new RoundedBoxGeometry(W + 0.02, 0.17, 0.14, 2, 0.03), M.trim_black, 0, 0.46, L / 2 - 0.02);
  add(new RoundedBoxGeometry(W + 0.02, 0.17, 0.14, 2, 0.03), M.trim_black, 0, 0.48, -L / 2 + 0.02);
  // wheel arch flares
  const flare = new THREE.BoxGeometry(0.06, 0.12, 0.95);
  for (const x of [-1, 1]) for (const z of [spec.frontAxleZ, spec.rearAxleZ]) add(flare, M.trim_black, x * (W / 2 + 0.01), 0.76, z);
  // lamps
  const hl = new THREE.CylinderGeometry(0.085, 0.085, 0.04, 20);
  for (const x of [-1, 1]) add(hl, M.headlight, x * 0.58, 0.84, L / 2 - 0.02, body, Math.PI / 2);
  const tl = new THREE.BoxGeometry(0.12, 0.2, 0.03);
  for (const x of [-1, 1]) add(tl, M.brakelight, x * (W / 2 - 0.1), 0.82, -L / 2 + 0.02);
  const ind = new THREE.BoxGeometry(0.1, 0.05, 0.03);
  for (const x of [-1, 1]) add(ind, M.indicator, x * 0.62, 0.7, L / 2 - 0.01);
  add(new THREE.BoxGeometry(0.52, 0.12, 0.01), M.plate, 0, 0.62, -L / 2 - 0.005);
  // spare wheel on the tailgate
  add(new THREE.CylinderGeometry(0.33, 0.33, 0.2, 20), M.rubber, 0, 1.0, -L / 2 - 0.1, body, Math.PI / 2);
  // interior: dash, gauges, seats
  add(new THREE.BoxGeometry(W - 0.25, 0.2, 0.35), M.interior, 0, 1.02, 0.2);
  add(new THREE.BoxGeometry(0.3, 0.1, 0.02), M.gauge, 0.37, 1.1, 0.03);
  const seat = new THREE.BoxGeometry(0.48, 0.5, 0.12);
  for (const x of [-1, 1]) { add(seat, M.interior, x * 0.37, 1.08, -0.52, body, -0.2); add(new THREE.BoxGeometry(0.48, 0.12, 0.5), M.interior, x * 0.37, 0.8, -0.3); }
  // steering wheel: local +Z along the column toward the driver
  const sw = new THREE.Group(); sw.name = 'steering_wheel';
  sw.position.set(0.37, 1.13, 0.3);
  sw.quaternion.setFromUnitVectors(AX_Z, new THREE.Vector3(0, 0.45, -0.89).normalize());
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.19, 0.017, 8, 28), M.trim_black); sw.add(rim);
  const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.03, 0.015), M.trim_black); sw.add(spoke);
  const spoke2 = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.18, 0.015), M.trim_black); spoke2.position.y = -0.09; sw.add(spoke2);
  body.add(sw);
  // anchors
  const e = (name, x, y, z) => { const o = new THREE.Object3D(); o.name = name; o.position.set(x, y, z); body.add(o); };
  e('seat_cam', 0.37, 1.34, -0.2);
  e('fuel_cap', W / 2 + 0.02, 0.95, spec.rearAxleZ - 0.25);
  e('door_point', W / 2 + 0.9, 0.0, -0.2);
  // wheels: tyre + steel rim + hub, spin axis local X
  const tyre = new THREE.CylinderGeometry(spec.wheelRadius, spec.wheelRadius, spec.wheelWidth, 28, 1);
  tyre.rotateZ(Math.PI / 2);
  const rimG = new THREE.CylinderGeometry(0.2, 0.2, spec.wheelWidth + 0.01, 18, 1); rimG.rotateZ(Math.PI / 2);
  const hubG = new THREE.CylinderGeometry(0.06, 0.06, spec.wheelWidth + 0.04, 8, 1); hubG.rotateZ(Math.PI / 2);
  const lugG = new THREE.BoxGeometry(spec.wheelWidth + 0.03, 0.03, 0.03);
  WHEEL_NAMES.forEach((name, i) => {
    const w = new THREE.Group(); w.name = name;
    const front = i < 2, left = i % 2 === 0;
    w.position.set((left ? 1 : -1) * spec.track / 2, spec.wheelRadius, front ? spec.frontAxleZ : spec.rearAxleZ);
    w.add(new THREE.Mesh(tyre, M.rubber));
    const rm = new THREE.Mesh(rimG, M.chrome); rm.material = M.chrome; w.add(rm);
    w.add(new THREE.Mesh(hubG, M.trim_black));
    const lug = new THREE.Mesh(lugG, M.trim_black); lug.position.y = 0.13; w.add(lug); // visible spin cue
    root.add(w);
  });
  root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return root;
}

// ------------------------------------------------------------------------------------------------ render helpers
/** Alpha-glass shading (see _setupRenderCost): Fresnel specular is not scaled by the alpha, the baked dirt film adds
 *  opacity. Base opacity comes from the asset (material.opacity; 0.2 in car.glb).
 *  (v4) Seen from inside, the glass no longer mirrors the sky: it reflects what is really in front of it, i.e. the
 *  dash top (the classic dashcam ghost of the dashboard in the lower windshield: defroster slots, a map, a receipt)
 *  with the Fresnel reflectance of both faces of the sheet (~8-10 % at the 45-50 deg a windshield is seen at), and
 *  the dim cabin elsewhere. `lens` = clear cover without a dirt film (gauge cluster). */
function alphaGlass(m, emissive = false, lens = false, U = null) {
  if (m.userData.glassPatched) return;
  m.userData.glassPatched = true;
  if (m.transmission > 0) console.warn('[car] glass material "' + m.name + '" arrived with transmission; car.glb should export alpha glass');
  if (!emissive && !lens && (!m.transparent || !(m.opacity < 1))) { m.transparent = true; m.opacity = 0.2; } // older assets: opaque/transmissive
  m.depthWrite = false;
  m.premultipliedAlpha = false;
  const alpha = { value: lens ? 0.0 : 0.5 }; // extra opacity of the dirt film (rough areas of the baked glass maps)
  m.userData.glassDirt = alpha;
  const inside = !!U && !emissive && !lens;
  const prev = m.onBeforeCompile;
  m.onBeforeCompile = (sh, r) => {
    prev?.call(m, sh, r);
    sh.uniforms.uGlassDirt = alpha;
    let insideCode = 'if (!gl_FrontFacing) { glSpec *= 0.1; glDirt *= 0.6; glA *= 0.6; }';
    if (inside) {
      sh.uniforms.uWorldToCar = U.uWorldToCar; sh.uniforms.uDashTex = U.uDashTex; sh.uniforms.uDashK = U.uDashK;
      sh.vertexShader = 'uniform mat4 uWorldToCar;\nvarying vec3 vGCarPos;\nvarying vec3 vGCarNrm;\n' + sh.vertexShader.replace('#include <project_vertex>', `#include <project_vertex>
        vGCarPos = (uWorldToCar * modelMatrix * vec4(transformed, 1.0)).xyz;
        vGCarNrm = mat3(uWorldToCar) * (mat3(modelMatrix) * objectNormal);`);
      sh.fragmentShader = 'uniform mat4 uWorldToCar;\nuniform sampler2D uDashTex;\nuniform float uDashK;\nvarying vec3 vGCarPos;\nvarying vec3 vGCarNrm;\n' + sh.fragmentShader;
      // inside/outside from the camera position (robust to the winding of the exported glass)
      insideCode = `vec3 gCam = (uWorldToCar * vec4(cameraPosition, 1.0)).xyz;
        if (abs(gCam.x) < 0.78 && gCam.y > 0.45 && gCam.y < 1.66 && gCam.z > -1.8 && gCam.z < 0.8) {
          vec3 gI = normalize(vGCarPos - gCam);
          vec3 gN = normalize(vGCarNrm); if (dot(gN, gI) > 0.0) gN = -gN;
          vec3 gR = reflect(gI, gN);
          float gF = 0.04 + 0.96 * pow(1.0 - clamp(-dot(gI, gN), 0.0, 1.0), 5.0);
          gF = 2.0 * gF / (1.0 + gF);
          #ifdef USE_ENVMAP
            vec3 gSky = getIBLIrradiance(normalize((viewMatrix * vec4(0.0, 1.0, 0.0, 0.0)).xyz)) / PI;
          #else
            vec3 gSky = vec3(1.0);
          #endif
          vec3 gL = vec3(0.012) * gSky;
          if (gR.y < -0.02) {
            vec3 gH = vGCarPos + gR * ((1.046 - vGCarPos.y) / gR.y);
            vec2 gUv = vec2((gH.x + 0.75) / 1.5, (gH.z - 0.40) / 0.35);
            if (gUv.x > 0.0 && gUv.x < 1.0 && gUv.y > 0.0 && gUv.y < 1.0) gL = texture2D(uDashTex, gUv).rgb * uDashK * gSky;   // dash radiance calibrated to its on-screen value (sky through the glass + soft sun)
          }
          glSpec = gF * gL * (1.0 - 0.5 * glDirt);
          glDirt *= 0.6; glA *= 0.6;
        }`;
    }
    sh.fragmentShader = 'uniform float uGlassDirt;\n' + sh.fragmentShader
      .replace('#include <opaque_fragment>', `
        float glDirt = uGlassDirt * smoothstep(0.25, 0.9, roughnessFactor);
        float glA = clamp(diffuseColor.a + glDirt, 0.0, 1.0);   // clean glass only absorbs (tint); the dirt film scatters
        vec3 glSpec = outgoingLight - totalDiffuse - totalEmissiveRadiance;   // specular (+ clearcoat) part
        ${insideCode}
        // un-premultiplied for normal alpha blending, so the height fog applied afterwards scales with alpha too
        gl_FragColor = vec4((totalDiffuse * glDirt + glSpec + ${emissive ? 'totalEmissiveRadiance' : 'vec3(0.0)'}) / max(glA, 1e-3), glA);`);
  };
  m.customProgramCacheKey = () => 'car_glass_v6' + (emissive ? 'e' : '') + (inside ? 'i' : '') + (lens ? 'l' : '');
  m.needsUpdate = true;
}

/** Low-poly inset shadow caster: body boxes + wheels + spare wheel + roof rack. Invisible in the main pass. */
function buildShadowProxy(spec, json, wheels) {
  const geos = [];
  const box = (hx, hy, hz, cx, cy, cz) => { const g = new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2); g.translate(cx, cy, cz); geos.push(g); };
  const inset = 0.035;
  const jb = json?.chassisBoxes;
  if (Array.isArray(jb) && jb.length) {
    for (const b of jb) box(b.halfExtents[0] - inset, b.halfExtents[1] - inset, b.halfExtents[2] - inset, b.center[0], b.center[1], b.center[2]);
  } else {
    const W = spec.width, L = spec.length, H = spec.height;
    box(W / 2 - 0.02 - inset, 0.31 - inset, L / 2 - 0.16 - inset, 0, 0.71, -0.02);
    box(W / 2 - 0.12 - inset, (H - 1.02) / 2 - inset, 1.08 - inset, 0, (H + 1.02) / 2, -0.68);
  }
  // roof rack (thin slab just above the roof) and the spare wheel on the tailgate
  const roofY = spec.height;
  box(spec.width / 2 - 0.2, 0.02, 0.85, 0, roofY + 0.06, -0.55);
  const R = spec.wheelRadius;
  const spare = new THREE.CylinderGeometry(R * 0.9, R * 0.9, 0.2, 14); spare.rotateX(Math.PI / 2); spare.translate(0, 0.98, -spec.length / 2 - 0.07); geos.push(spare);
  for (const w of wheels) {
    const g = new THREE.CylinderGeometry(R * 0.97, R * 0.97, spec.wheelWidth * 0.9, 14); g.rotateZ(Math.PI / 2);
    g.translate(w.rest.x, w.rest.y, w.rest.z); geos.push(g);
  }
  for (const g of geos) { g.deleteAttribute('uv'); g.deleteAttribute('normal'); }
  const merged = mergeGeometries(geos, false);
  for (const g of geos) g.dispose();
  const mat = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false });
  const mesh = new THREE.Mesh(merged, mat);
  mesh.name = 'car_shadow_proxy';
  mesh.castShadow = true; mesh.receiveShadow = false;
  mesh.renderOrder = -10;
  return mesh;
}

// ------------------------------------------------------------------------------------------------ contact shadow
/**
 * Ambient-occlusion "contact shadow" under the car: the dark, soft footprint every real car has on an overcast day
 * (sky light is blocked by the body), with darker cores where the tyres touch. One alpha-blended decal quad, placed on
 * the plane through the wheel contact points (or a ray-cast when airborne) and faded out with height above ground.
 */
class ContactShadow {
  constructor(vehicle) {
    this.v = vehicle;
    const s = vehicle.spec;
    this.sx = s.width + 1.3; this.sz = s.length + 1.3;
    const geo = new THREE.PlaneGeometry(this.sx, this.sz); geo.rotateX(-Math.PI / 2);
    this.tex = makeContactTexture(s, vehicle.wheels, this.sx, this.sz);
    this.mat = new THREE.MeshBasicMaterial({
      color: 0x000000, alphaMap: this.tex, transparent: true, opacity: 0.92, depthWrite: false,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8,
    });
    this.mesh = new THREE.Mesh(geo, this.mat);
    this.mesh.name = 'car_contact_shadow';
    this.mesh.renderOrder = -5;           // before glass/particles among transparents
    this.mesh.castShadow = false; this.mesh.receiveShadow = false;
    this.mesh.matrixAutoUpdate = false;
    this._n = new THREE.Vector3(); this._c = new THREE.Vector3(); this._f = new THREE.Vector3(); this._x = new THREE.Vector3();
    this._m = new THREE.Matrix4(); this._inv = new THREE.Matrix4();
    this.fade = 1;
  }

  rebuild() { // wheel positions changed (hot swap)
    this.tex.dispose();
    this.tex = makeContactTexture(this.v.spec, this.v.wheels, this.sx, this.sz);
    this.mat.alphaMap = this.tex; this.mat.needsUpdate = true;
  }

  /** Called after the car object got its interpolated transform for this frame. */
  update(dt) {
    const v = this.v, vc = v.controller, obj = v.object;
    if (!vc || !v.body) { this.mesh.visible = false; return; }
    // contact points (world, latest physics step) -> body-local (latest body pose) -> world via the interpolated pose
    this._inv.compose(v._curPos, v._curQuat, _one).invert();
    const pts = [];
    for (let i = 0; i < 4; i++) {
      if (!vc.wheelIsInContact(i)) continue;
      const p = vc.wheelContactPoint(i);
      if (!p) continue;
      pts.push(new THREE.Vector3(p.x, p.y, p.z).applyMatrix4(this._inv));
    }
    let target = 0;
    const c = this._c, n = this._n;
    if (pts.length >= 3) {
      c.set(0, 0, 0); for (const p of pts) c.add(p); c.multiplyScalar(1 / pts.length);
      if (pts.length === 4) {
        // plane normal from the diagonals fl->rr and fr->rl (wheel order fl, fr, rl, rr) in body space
        const a = _t1.subVectors(pts[3], pts[0]), b = _t2.subVectors(pts[2], pts[1]);
        n.crossVectors(b, a).normalize(); if (n.y < 0) n.negate();
      } else {
        n.crossVectors(_t1.subVectors(pts[1], pts[0]), _t2.subVectors(pts[2], pts[0])).normalize(); if (n.y < 0) n.negate();
      }
      c.applyMatrix4(obj.matrixWorld);
      n.transformDirection(obj.matrixWorld);
      target = 1;
    } else {
      // airborne / tipped: ray-cast straight down from the chassis centre
      this._rayT = (this._rayT || 0) - dt;
      if (this._rayT <= 0) {
        this._rayT = 0.05;
        const o = _t1.set(0, 0.8, 0).applyMatrix4(obj.matrixWorld);
        const hit = v.ctx.physics?.raycast?.(o, _t2.set(0, -1, 0), 4, { groups: groups(G.ALL, G.STATIC | G.PROP), excludeBody: v.body });
        this._hit = hit ? { p: hit.point.clone(), n: hit.normal.clone(), h: o.y - 0.8 - hit.point.y } : null;
      }
      if (this._hit) {
        c.copy(this._hit.p); n.copy(this._hit.n);
        target = (1 - smooth01(0.15, 1.4, this._hit.h)) * clamp(v.upright * 1.5, 0, 1);
      }
    }
    this.fade += (target - this.fade) * (1 - Math.exp(-dt * 10));
    if (this.fade < 0.02 || n.lengthSq() < 0.5) { this.mesh.visible = false; return; }
    this.mesh.visible = true;
    this.mat.opacity = 0.92 * this.fade;
    // basis: Y = ground normal, Z = car forward projected onto the ground plane
    const f = this._f.set(0, 0, 1).applyQuaternion(obj.quaternion);
    f.addScaledVector(n, -f.dot(n)).normalize();
    const x = this._x.crossVectors(n, f);
    this._m.makeBasis(x, n, f).setPosition(c.x + n.x * 0.012, c.y + n.y * 0.012, c.z + n.z * 0.012);
    this.mesh.matrix.copy(this._m);
    this.mesh.matrixWorldNeedsUpdate = true;
  }

  dispose() { this.mesh.geometry.dispose(); this.mat.dispose(); this.tex.dispose(); }
}
const _one = new THREE.Vector3(1, 1, 1), _t1 = new THREE.Vector3(), _t2 = new THREE.Vector3();

/** Alpha texture: soft rounded-rectangle footprint of the body + dark ellipses at the 4 tyre contact patches. */
function makeContactTexture(spec, wheels, sx, sz) {
  const W = 128, H = 256;
  const data = new Uint8Array(W * H * 4);
  const hx = spec.width / 2 - 0.12, hz = spec.length / 2 - 0.2, rr = 0.35;
  const wp = wheels.map((w) => [w.rest.x, w.rest.z]);
  for (let j = 0; j < H; j++) {
    const z = (0.5 - (j + 0.5) / H) * sz; // row 0 (v=0) maps to +z (front) after the PlaneGeometry rotateX(-90deg)
    for (let i = 0; i < W; i++) {
      const x = ((i + 0.5) / W - 0.5) * sx;
      // rounded-rect signed distance (m)
      const qx = Math.abs(x) - (hx - rr), qz = Math.abs(z) - (hz - rr);
      const sd = Math.hypot(Math.max(qx, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qz), 0) - rr;
      let body = 1 - smooth01(-0.45, 0.5, sd);
      body = 0.62 * body * body;
      let tyre = 0;
      for (const [wx, wz] of wp) {
        const dx = (x - wx) / 0.16, dz = (z - wz) / 0.3;
        tyre = Math.max(tyre, Math.exp(-(dx * dx + dz * dz) * 1.4));
      }
      const a = 1 - (1 - body) * (1 - 0.9 * tyre);
      const k = (j * W + i) * 4, b = Math.round(clamp(a, 0, 1) * 255);
      data[k] = data[k + 1] = data[k + 2] = b; data[k + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
  tex.flipY = false; tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter; tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

// ------------------------------------------------------------------------------------------------ tyre spray
/**
 * Wet-road tyre spray: the mist "rooster tail" every car throws on a wet road, plus a few heavier droplets. One draw
 * call (instanced camera-facing quads, simulated analytically in the vertex shader from spawn state: drag + gravity),
 * a small CPU ring buffer only writes newly spawned particles. Lit as sky-lit water mist, alpha blended, height fog.
 */
const SPRAY_VERT = /* glsl */`
  #include <common>
  #include <fog_pars_vertex>
  attribute vec3 aP0; attribute vec3 aV0; attribute vec4 aT; // aT: birth time, life, size0, size1
  attribute vec4 aK;                                           // aK: drag, gravity scale (<0 rises), seed, opacity
  uniform float uTime; uniform vec3 uWind; uniform mat4 uCarInv;
  varying vec2 vUv; varying float vA; varying float vSeed;
  void main() {
    float age = uTime - aT.x;
    float life = aT.y;
    vUv = uv; vSeed = aK.z;
    if (age < 0.0 || age > life) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vA = 0.0; return; }
    float k = aK.x;
    float e = (1.0 - exp(-k * age)) / k;
    float g = 9.81 * aK.y;
    // velocity decays toward the wind; gravity pulls against drag (terminal speed g/k)
    vec3 wp = aP0 + (aV0 - uWind) * e + uWind * age;
    wp.y -= g / k * (age - e);
    float t = age / life;
    float size = mix(aT.z, aT.w, 1.0 - (1.0 - t) * (1.0 - t));
    vec4 mvPosition = viewMatrix * vec4(wp, 1.0);
    float ang = aK.z * 6.2831 + age * (aK.z - 0.5) * 0.8;
    vec2 c = (uv - 0.5) * size;
    mvPosition.xy += vec2(c.x * cos(ang) - c.y * sin(ang), c.x * sin(ang) + c.y * cos(ang));
    gl_Position = projectionMatrix * mvPosition;
    vA = smoothstep(0.0, 0.08, t) * (1.0 - smoothstep(0.15, 1.0, t)) * aK.w;
    // (v4) vapour/mist that drifts into the body shell (wind pushing exhaust forward) must not appear in the cabin
    vec3 cp = (uCarInv * vec4(wp, 1.0)).xyz;
    vec3 dq = abs(cp - vec3(0.0, 1.05, -0.05)) - vec3(0.80, 0.62, 1.80);
    vA *= smoothstep(-0.05, 0.25, max(dq.x, max(dq.y, dq.z)));
    #include <fog_vertex>
  }`;
const SPRAY_FRAG = /* glsl */`
  #include <common>
  #include <fog_pars_fragment>
  uniform sampler2D uTex; uniform vec3 uLight; uniform float uOpacity;
  varying vec2 vUv; varying float vA; varying float vSeed;
  void main() {
    vec2 uv = vUv;
    // two quadrants of the puff atlas (2x2) chosen by seed
    float q = floor(vSeed * 3.999);
    uv = uv * 0.5 + vec2(mod(q, 2.0), floor(q / 2.0)) * 0.5;
    float d = texture2D(uTex, uv).r;
    float a = d * vA * uOpacity;
    if (a < 0.003) discard;
    gl_FragColor = vec4(uLight * (0.85 + 0.3 * d), a);
    #include <fog_fragment>
  }`;

class TyreSpray {
  constructor(vehicle, cap = 700) {
    this.v = vehicle; this.cap = cap; this.head = 0; this.time = 0;
    const base = new THREE.PlaneGeometry(1, 1);
    const g = new THREE.InstancedBufferGeometry();
    g.index = base.index; g.setAttribute('position', base.attributes.position); g.setAttribute('uv', base.attributes.uv);
    this.aP0 = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3), 3);
    this.aV0 = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3), 3);
    this.aT = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4).fill(-1000), 4);
    this.aK = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4).fill(1), 4);
    for (const a of [this.aP0, this.aV0, this.aT, this.aK]) a.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aP0', this.aP0); g.setAttribute('aV0', this.aV0); g.setAttribute('aT', this.aT); g.setAttribute('aK', this.aK);
    g.instanceCount = cap;
    const ctx = vehicle.ctx;
    this.u = { uTime: { value: 0 }, uWind: { value: new THREE.Vector3(0.6, 0, 0.25) }, uTex: { value: makePuffTexture() },
      uLight: { value: new THREE.Color(0.4, 0.42, 0.45) }, uOpacity: { value: 0.085 },
      uCarInv: vehicle._cabinUniforms?.().uWorldToCar ?? { value: new THREE.Matrix4().makeTranslation(0, -1e4, 0) } };
    this.mat = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, ctx.env?.fogUniforms ?? {}, {}]),
      vertexShader: SPRAY_VERT, fragmentShader: SPRAY_FRAG, fog: true, transparent: true, depthWrite: false,
    });
    Object.assign(this.mat.uniforms, this.u); // share our uniform objects (merge() clones them)
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.name = 'car_tyre_spray';
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 5;
    this._acc = [0, 0, 0, 0];
    this._dirty = false; this._lo = cap; this._hi = -1;
    this.alive = 0;
  }

  _spawn(p, vel, life, s0, s1, drag, grav, alpha = 1) {
    const i = this.head; this.head = (this.head + 1) % this.cap;
    this.aP0.array.set([p.x, p.y, p.z], i * 3);
    this.aV0.array.set([vel.x, vel.y, vel.z], i * 3);
    this.aT.array.set([this.time, life, s0, s1], i * 4);
    this.aK.array.set([drag, grav, Math.random(), alpha], i * 4);
    this._lo = Math.min(this._lo, i); this._hi = Math.max(this._hi, i);
    this._dirty = true;
  }

  update(dt) {
    const v = this.v, ctx = v.ctx, vc = v.controller;
    this.time += dt;
    this.u.uTime.value = this.time;
    if (!vc || !v.body || dt <= 0) return;
    const wet = ctx.env?.wetness ?? 0.75, rain = ctx.env?.rain ?? 0.35;
    // sky-lit mist colour follows the sky light (dimmer in the tunnel / under lightning flashes brighter)
    const pr = v._proj;
    const inTunnel = pr && ctx.road?.markers && pr.s > ctx.road.markers.tunnel + 4 && pr.s < ctx.road.markers.tunnelEnd;
    if (inTunnel) this.u.uLight.value.setRGB(0.07, 0.05, 0.03); // sodium-lit
    else this.u.uLight.value.setRGB(0.4, 0.42, 0.45);
    const q = v.object.quaternion;
    const fwd = _s1.set(0, 0, 1).applyQuaternion(q), left = _s2.set(1, 0, 0).applyQuaternion(q);
    const lv = v.body.linvel();
    const spd = Math.abs(v.speed);
    const water = clamp(wet * 1.1 - 0.15, 0, 1) * (inTunnel ? 0.25 : 1);
    for (let i = 0; i < 4; i++) {
      const w = v.wheels[i];
      if (!vc.wheelIsInContact(i) || water <= 0.02) { this._acc[i] = 0; continue; }
      const surf = v._surf[i];
      const sm = surf === 'asphalt' ? 1 : surf === 'rock' || surf === 'wood' ? 0.7 : surf === 'gravel' ? 0.45 : 0.25;
      const I = water * sm * smooth01(3.5, 18, spd);
      if (I <= 0.01) { this._acc[i] = 0; continue; }
      // rear wheels run in the wet track of the front ones and throw the bigger plume
      // (QA) twice the puffs at ~0.55x the opacity each: the plume integrates into a continuous low-contrast
      // haze behind the tyres (as on dashcam footage) instead of separate round blobs on the dark asphalt
      const rate = (w.front ? 52 : 76) * I * (0.7 + 0.6 * rain);
      this._acc[i] += rate * dt;
      const cp = vc.wheelContactPoint(i);
      if (!cp) continue;
      const dir = v.speed >= 0 ? 1 : -1;
      let n = 0;
      while (this._acc[i] >= 1 && n++ < 10) {
        this._acc[i] -= 1;
        const side = (w.left ? 1 : -1);
        const lat = (Math.random() - 0.3) * 0.2 * side;
        const p = _s3.set(cp.x, cp.y, cp.z).addScaledVector(fwd, -dir * (0.3 + Math.random() * 0.25)).addScaledVector(left, lat);
        p.y += 0.12 + Math.random() * 0.2;
        const heavy = Math.random() < 0.18;
        // ground-frame launch velocity: carried along with the car, thrown up/back off the tread and out sideways
        const up = (heavy ? 1.2 : 0.7) + Math.random() * 1.1 + spd * 0.05;
        const back = -dir * spd * (0.15 + Math.random() * 0.2);
        const out = side * (0.3 + Math.random() * 0.8) * (0.5 + spd * 0.03);
        const vel = _s4.set(lv.x * 0.55, 0, lv.z * 0.55).addScaledVector(fwd, back).addScaledVector(left, out);
        vel.y = up;
        // heavy droplets: 2-6 mm drops, a few cm across even motion-blurred (a 35 cm puff read as a white blob)
        if (heavy) this._spawn(p, vel, 0.4 + Math.random() * 0.3, 0.03, 0.06, 2.2, 0.55, 0.6);
        // (QA) a small car at 40-50 km/h on a ~1 mm water film throws a plume ~1 m high that is gone in ~0.5 s (the
        // 2.4 m, 1.2 s puffs hung on the road behind the car as discrete pale blobs in the chase view)
        else this._spawn(p, vel, 0.38 + Math.random() * 0.35 * I, 0.25 + 0.12 * Math.random(), 0.6 + 0.6 * I + Math.random() * 0.3, 3.2, 0.04);
      }
    }
    this._exhaust(dt, fwd, left, lv, spd, rain);
    if (this._dirty) {
      for (const a of [this.aP0, this.aV0, this.aT, this.aK]) {
        a.clearUpdateRanges();
        a.addUpdateRange(this._lo * a.itemSize, (this._hi - this._lo + 1) * a.itemSize);
        a.needsUpdate = true;
      }
      this._dirty = false; this._lo = this.cap; this._hi = -1;
    }
  }

  /** Cold, wet air: the old carburettor engine's exhaust condenses into white vapour (strongest at idle / standstill). */
  _exhaust(dt, fwd, left, lv, spd, rain) {
    const v = this.v;
    const on = v.engineOn && !(v._starve > 0);
    const crankPuff = v.cranking > 0 ? 1 : 0;
    if (!on && !crankPuff) { this._exAcc = 0; return; }
    const load = clamp(v.rpm / 3000, 0.25, 2) * (0.6 + 0.8 * v.throttle);
    const vis = 1 / (1 + spd / 4);               // at speed the plume is stretched thin and dilutes quickly
    this._exAcc = (this._exAcc || 0) + dt * (6 + 10 * load) * (0.5 + 0.5 * vis);
    const ex = v.exhaustPoint;
    let n = 0;
    while (this._exAcc >= 1 && n++ < 4) {
      this._exAcc -= 1;
      const p = _s3.copy(ex).applyMatrix4(v.object.matrixWorld);
      const vel = _s4.set(lv.x, lv.y, lv.z).addScaledVector(fwd, -(1.2 + 1.6 * load) + Math.random() * 0.4)
        .addScaledVector(left, (Math.random() - 0.5) * 0.4);
      vel.y += 0.1 + Math.random() * 0.25;
      // x2.45: compensates the spray's lower uOpacity (0.21 -> 0.085, QA) so the exhaust vapour keeps its density
      const a = 2.45 * (0.55 + 0.35 * vis) * (0.8 + 0.4 * rain) * (v.stalling ? 1.3 : 1);
      this._spawn(p, vel, 1.6 + Math.random() * 1.4, 0.07, 0.7 + Math.random() * 0.5 + 0.3 * load, 1.6, -0.012, a);
    }
  }

  dispose() { this.mesh.geometry.dispose(); this.mat.dispose(); this.u.uTex.value.dispose(); }
}
const _s1 = new THREE.Vector3(), _s2 = new THREE.Vector3(), _s3 = new THREE.Vector3(), _s4 = new THREE.Vector3();

/** 2x2 atlas of soft, lumpy mist puffs (R channel = density), generated procedurally. */
function makePuffTexture() {
  const N = 128, S = N * 2;
  const data = new Uint8Array(S * S * 4);
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let qy = 0; qy < 2; qy++) for (let qx = 0; qx < 2; qx++) {
    const blobs = [];
    for (let b = 0; b < 14; b++) { const a = rnd() * 6.283, r = Math.sqrt(rnd()) * 0.26; blobs.push([0.5 + Math.cos(a) * r, 0.5 + Math.sin(a) * r, 0.1 + rnd() * 0.16, 0.4 + rnd() * 0.6]); }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const u = (x + 0.5) / N, w = (y + 0.5) / N;
      let d = 0;
      for (const [bx, by, br, bs] of blobs) { const dd = ((u - bx) ** 2 + (w - by) ** 2) / (br * br); d += bs * Math.exp(-dd * 2.2); }
      const r = Math.hypot(u - 0.5, w - 0.5) / 0.5;
      d = clamp(d * 0.55, 0, 1) * (1 - smooth01(0.55, 1.0, r));
      const k = ((qy * N + y) * S + qx * N + x) * 4, b = Math.round(d * 255);
      data[k] = data[k + 1] = data[k + 2] = b; data[k + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
  tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter; tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

// ------------------------------------------------------------------------------------------------ cabin light (v4)
// The IBL knows nothing about the cabin. car.glb's interior occlusion map (ORM red) holds the fraction of the overcast
// sky irradiance that reaches each texel THROUGH THE GLASS (computed analytically in car.py from the window outlines,
// times the baked local AO); here we (1) add a small uniform bounce term on top (seats/dash/headliner inter-reflection,
// which the map deliberately leaves out), (2) replace three's specular occlusion by a per-pixel test of the reflection
// ray against the same window outlines, so dash tops, wheel rims and knobs mirror the bright glass while faces that
// look back into the cabin stay dark, and (3) add micro-detail (pebble grain / weave / leather grain) selected by the
// material class stored in ORM blue (interior metalness is forced to 0).
const CABIN_FILL = 0.12;  // bounce irradiance / pi relative to the IBL (dark early-90s cabin: albedo ~0.03-0.06)
const CABIN_PARS = /* glsl */`
uniform vec3 uCabinFill;
uniform mat4 uWorldToCar;
uniform sampler2D uCabDetail;
uniform float uCabDetailOn;
uniform float uCabNoAO;
varying vec3 vCarPos;
varying vec3 vObjPos;
varying vec3 vObjNrm;
float cabinWindow(vec3 p, vec3 d, float soft) {
  const float ROOF = 1.596, FLR = 0.52, REAR = -1.755, K = 1.255556, WC = 1.594212, BSL = 0.108475, CS = 0.880356;
  float t = 1e9; float face = -1.0;
  if (d.y > 1e-4) { float tt = (ROOF - p.y) / d.y; if (tt < t) { t = tt; face = 0.0; } }
  if (d.y < -1e-4) { float tt = (FLR - p.y) / d.y; if (tt < t) { t = tt; face = 1.0; } }
  if (d.z < -1e-4) { float tt = (REAR - p.z) / d.z; if (tt < t) { t = tt; face = 2.0; } }
  float dw = d.z + d.y / K; if (dw > 1e-4) { float tt = (WC - (p.z + p.y / K)) / dw; if (tt < t) { t = tt; face = 3.0; } }
  float dL = d.x + BSL * d.y; if (dL > 1e-4) { float tt = (CS - (p.x + BSL * p.y)) / dL; if (tt < t) { t = tt; face = 4.0; } }
  float dR = -d.x + BSL * d.y; if (dR > 1e-4) { float tt = (CS - (-p.x + BSL * p.y)) / dR; if (tt < t) { t = tt; face = 4.0; } }
  vec3 e = p + d * max(t, 0.0);
  float S = 0.012 + max(t, 0.0) * soft;   // glossy lobe footprint on the glass
  if (face == 3.0) {
    float wsw = 0.613 - 0.0825 * (e.y - 1.10);
    return smoothstep(1.112 - S, 1.112 + S, e.y) * smoothstep(1.573 + S, 1.573 - S, e.y) * smoothstep(wsw + S, wsw - S, abs(e.x));
  }
  if (face == 2.0) return smoothstep(0.588 + S, 0.588 - S, abs(e.x)) * smoothstep(1.152 - S, 1.152 + S, e.y) * smoothstep(1.543 + S, 1.543 - S, e.y);
  if (face == 4.0) {
    float dff = (e.y <= 1.06 ? 0.655 : 0.655 - (e.y - 1.06) / K) - 0.046;
    float hh = smoothstep(1.107 - S, 1.107 + S, e.y) * smoothstep(1.556 + S, 1.556 - S, e.y);
    float door = smoothstep(dff + S, dff - S, e.z) * smoothstep(-0.398 - S, -0.398 + S, e.z);
    float qtr = smoothstep(-0.542 + S, -0.542 - S, e.z) * smoothstep(-1.616 - S, -1.616 + S, e.z);
    return hh * max(door, qtr);
  }
  return 0.0;
}
`;

const CABIN_AO_INT = /* glsl */`{
  #ifdef USE_AOMAP
    float cabAO = (texture2D(aoMap, vAoMapUv).r - 1.0) * aoMapIntensity + 1.0;
  #else
    float cabAO = uCabNoAO;
  #endif
  reflectedLight.indirectDiffuse *= cabAO;
  reflectedLight.indirectDiffuse += uCabinFill * diffuseColor.rgb * (0.3 + 0.7 * sqrt(cabAO));
  vec3 cabRw = transformDirectionByInverseViewMatrix(reflect(-geometryViewDir, normal), viewMatrix);
  vec3 cabRc = normalize(mat3(uWorldToCar) * cabRw);
  float cabR2 = material.roughness * material.roughness;
  float cabWin = cabinWindow(vCarPos, cabRc, cabR2 * 1.4);
  float cabSV = mix(cabWin, cabAO, smoothstep(0.35, 0.9, material.roughness));
  reflectedLight.indirectSpecular *= mix(0.035, 1.0, cabSV);
}`;
// chrome & co: standard AO outside, cabin light inside the cabin volume (below the windshield plane)
const CABIN_AO_EXT = /* glsl */`{
  #ifdef USE_AOMAP
    float cabAO0 = (texture2D(aoMap, vAoMapUv).r - 1.0) * aoMapIntensity + 1.0;
  #else
    float cabAO0 = 1.0;
  #endif
  float cabIn = step(abs(vCarPos.x), 0.765) * step(0.48, vCarPos.y) * step(vCarPos.y, 1.62) * step(-1.76, vCarPos.z)
              * step(vCarPos.z + vCarPos.y / 1.255556, 1.6142);
  if (cabIn > 0.5) {
    float cabAO = uCabNoAO;
    reflectedLight.indirectDiffuse *= cabAO;
    reflectedLight.indirectDiffuse += uCabinFill * diffuseColor.rgb * 0.6;
    vec3 cabRw = transformDirectionByInverseViewMatrix(reflect(-geometryViewDir, normal), viewMatrix);
    vec3 cabRc = normalize(mat3(uWorldToCar) * cabRw);
    float cabR2 = material.roughness * material.roughness;
    float cabWin = cabinWindow(vCarPos, cabRc, cabR2 * 1.4);
    float cabSV = mix(cabWin, cabAO, smoothstep(0.35, 0.9, material.roughness));
    reflectedLight.indirectSpecular *= mix(0.03, 1.0, cabSV);
  } else {
    reflectedLight.indirectDiffuse *= cabAO0;
    #if defined( USE_ENVMAP ) && defined( STANDARD )
      float dotNV = saturate(dot(geometryNormal, geometryViewDir));
      reflectedLight.indirectSpecular *= computeSpecularOcclusion(dotNV, cabAO0, material.roughness);
    #endif
  }
}`;

function patchCabinVertex(sh) {
  sh.vertexShader = 'uniform mat4 uWorldToCar;\nvarying vec3 vCarPos;\nvarying vec3 vObjPos;\nvarying vec3 vObjNrm;\n' + sh.vertexShader
    .replace('#include <project_vertex>', `#include <project_vertex>
      vCarPos = (uWorldToCar * modelMatrix * vec4(transformed, 1.0)).xyz;
      vObjPos = transformed;
      vObjNrm = objectNormal;`);
}

Vehicle.prototype._cabinUniforms = function () {
  if (this._cabU) return this._cabU;
  const q = this.ctx.config?.quality?.key || 'high';
  this._cabU = {
    uCabinFill: { value: new THREE.Color(0, 0, 0) },
    uWorldToCar: { value: new THREE.Matrix4() },
    uCabDetail: { value: makeCabinDetailTexture() },
    uCabDetailOn: { value: q === 'low' ? 0.0 : 1.0 },
    uCabNoAO: { value: 0.22 },
    uDashTex: { value: makeDashTopTexture() },
    uDashK: { value: 1.2 },
  };
  this.cabinFill = this._cabU.uCabinFill; // (kept for older code paths)
  return this._cabU;
};

/** Interior + gauge materials: cabin light, window-tested reflections, micro-detail (see the block comment above). */
Vehicle.prototype._setupCabinFill = function () {
  const U = this._cabinUniforms();
  const list = [...(this.mats.interior || []), ...(this.mats.gauge || [])];
  if (this.gaugeLens?.material) list.push(this.gaugeLens.material);
  // exterior-atlas materials that also appear inside (knobs, sliders, gear stick, screws, keys): gated by position
  const extList = [...(this.mats.chrome || [])];
  for (const m of [...list, ...extList]) {
    if (m.userData.cabinV4) continue;
    m.userData.cabinV4 = true;
    const ext = extList.includes(m);
    const prev = m.onBeforeCompile;
    m.onBeforeCompile = (sh, r) => {
      prev?.call(m, sh, r);
      if (ext) sh.fragmentShader = '#define CAB_EXT\n' + sh.fragmentShader;
      Object.assign(sh.uniforms, { uCabinFill: U.uCabinFill, uWorldToCar: U.uWorldToCar, uCabDetail: U.uCabDetail, uCabDetailOn: U.uCabDetailOn, uCabNoAO: U.uCabNoAO });
      patchCabinVertex(sh);
      sh.fragmentShader = 'uniform mat4 modelMatrix;\n' + sh.fragmentShader
        .replace('#include <common>', '#include <common>\n' + CABIN_PARS)
        .replace('#include <metalnessmap_fragment>', `#include <metalnessmap_fragment>
          vec2 cabDN = vec2(0.0); float cabAxis = 2.0;
          #if defined( USE_METALNESSMAP ) && !defined( CAB_EXT )
          {
            float cls = floor(texelMetalness.b * 4.0 + 0.5);
            metalnessFactor = 0.0;
            vec3 an = abs(vObjNrm);
            vec2 duv;
            if (an.x > an.y && an.x > an.z) { duv = vObjPos.zy; cabAxis = 0.0; }
            else if (an.y > an.z) { duv = vObjPos.xz; cabAxis = 1.0; }
            else { duv = vObjPos.xy; cabAxis = 2.0; }
            float sc = cls < 1.5 ? 45.0 : (cls < 2.5 ? 70.0 : 32.0);
            vec4 dtx = texture2D(uCabDetail, duv * sc);
            vec2 dn = (cls > 1.5 && cls < 2.5) ? dtx.ba * 2.0 - 1.0 : dtx.rg * 2.0 - 1.0;
            float str = cls < 0.5 ? 0.0 : (cls < 1.5 ? 0.42 : (cls < 2.5 ? 0.35 : 0.5));
            cabDN = dn * str * uCabDetailOn;
            roughnessFactor = clamp(roughnessFactor + (length(dn) - 0.35) * 0.30 * str * uCabDetailOn, 0.04, 1.0);
          }
          #endif`)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
          if (cabDN.x != 0.0 || cabDN.y != 0.0) {
            vec3 dObj = cabAxis < 0.5 ? vec3(0.0, cabDN.y, cabDN.x) : (cabAxis < 1.5 ? vec3(cabDN.x, 0.0, cabDN.y) : vec3(cabDN.x, cabDN.y, 0.0));
            normal = normalize(normal + mat3(viewMatrix) * (mat3(modelMatrix) * dObj));
          }`)
        .replace('#include <aomap_fragment>', ext ? CABIN_AO_EXT : CABIN_AO_INT);
    };
    const key = m.customProgramCacheKey?.bind(m);
    m.customProgramCacheKey = () => 'car_cabin_v5' + (ext ? 'x' : '') + '|' + (key ? key() : '');
    m.needsUpdate = true;
  }
};

/** Per frame: car pose for the cabin shaders + bounce level (daylight through the glass, sodium in the tunnel). */
Vehicle.prototype._updateCabin = function () {
  const U = this._cabU;
  if (!U) return;
  U.uWorldToCar.value.copy(this.object.matrixWorld).invert();
  const envI = this.ctx.scene?.environmentIntensity ?? 1;
  let tunIn = 0;
  try { tunIn = this.ctx.terrain?.insideTunnel?.(this.object.position) ?? 0; } catch { tunIn = 0; }
  // the IBL is not occluded by the portal cut / tunnel: follow them by road position (sun and sky blocked by rock)
  const mk = this.ctx.road?.markers, s = this._proj?.s;
  const slot = mk?.tunnel != null && isFinite(s) ? smooth01(mk.tunnel - 70, mk.tunnel - 40, s) * (1 - smooth01(mk.tunnelEnd + 2, mk.tunnelEnd + 10, s)) : 0;
  const k = CABIN_FILL * envI * (1 - 0.85 * tunIn) * (1 - 0.35 * slot);
  U.uDashK.value = 1.2 * (1 - 0.7 * slot) * (1 - 0.8 * tunIn);
  const sd = 0.05 * tunIn;
  U.uCabinFill.value.setRGB(0.88 * k + _sodium.r * sd, 0.94 * k + _sodium.g * sd, k + _sodium.b * sd);
};

/** 256^2 periodic micro-detail normals: RG = pebble grain (moulded PP / leather-look PU), BA = velour pile + twill. */
function makeCabinDetailTexture(size = 256) {
  let seed = 1234567;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const cells = 24, cs = size / cells;
  const pts = [];
  for (let j = 0; j < cells; j++) for (let i = 0; i < cells; i++) pts.push([(i + 0.15 + 0.7 * rnd()) * cs, (j + 0.15 + 0.7 * rnd()) * cs, 0.6 + 0.4 * rnd()]);
  const H1 = new Float32Array(size * size), H2 = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const ci = Math.floor(x / cs), cj = Math.floor(y / cs);
      let f1 = 1e9, f2 = 1e9, hh = 1;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const ii = (ci + di + cells) % cells, jj = (cj + dj + cells) % cells;
        const p = pts[jj * cells + ii];
        const px = p[0] + (ci + di - ii) * cs, py = p[1] + (cj + dj - jj) * cs;
        const d = Math.hypot(x + 0.5 - px, y + 0.5 - py);
        if (d < f1) { f2 = f1; f1 = d; hh = p[2]; } else if (d < f2) f2 = d;
      }
      const e = Math.min(1, (f2 - f1) / (cs * 0.32));
      H1[y * size + x] = e * e * (3 - 2 * e) * hh;
      const tw = Math.sin((x + y) * Math.PI * 2 * 16 / size) * 0.5 + 0.5;
      H2[y * size + x] = 0.35 * tw + 0.65 * rnd();
    }
  }
  // blur the pile noise a little
  const B = new Float32Array(size * size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let s = 0;
    for (let d = -1; d <= 1; d++) s += H2[y * size + ((x + d + size) % size)] + H2[((y + d + size) % size) * size + x];
    B[y * size + x] = s / 6;
  }
  const data = new Uint8Array(size * size * 4);
  const at = (A, x, y) => A[((y + size) % size) * size + ((x + size) % size)];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const o = (y * size + x) * 4;
    let gx = (at(H1, x + 1, y) - at(H1, x - 1, y)) * 2.2, gy = (at(H1, x, y + 1) - at(H1, x, y - 1)) * 2.2;
    let l = Math.hypot(gx, gy, 1);
    data[o] = Math.round((-gx / l * 0.5 + 0.5) * 255); data[o + 1] = Math.round((-gy / l * 0.5 + 0.5) * 255);
    gx = (at(B, x + 1, y) - at(B, x - 1, y)) * 2.5; gy = (at(B, x, y + 1) - at(B, x, y - 1)) * 2.5;
    l = Math.hypot(gx, gy, 1);
    data[o + 2] = Math.round((-gx / l * 0.5 + 0.5) * 255); data[o + 3] = Math.round((-gy / l * 0.5 + 0.5) * 255);
  }
  const t = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearMipmapLinearFilter; t.generateMipmaps = true;
  t.anisotropy = 4; t.colorSpace = THREE.NoColorSpace; t.needsUpdate = true;
  return t;
}

/** Albedo of the dash top as seen in the windshield (linear): x = l (-0.75..0.75), y = f (0.40..0.75). Dusty black
 *  plastic, the defroster grilles along the glass, the instrument hood, the road map and the fuel receipt. */
function makeDashTopTexture() {
  const W = 512, H = 128;
  const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const g = cv.getContext('2d');
  const X = (l) => (l + 0.75) / 1.5 * W, Y = (f) => (1 - (f - 0.40) / 0.35) * H;   // canvas row 0 = far edge (v = 1)
  const lin = (v) => Math.round(Math.min(1, v) * 255);
  const grey = (v) => `rgb(${lin(v)},${lin(v * 0.95)},${lin(v * 0.88)})`;
  g.fillStyle = grey(0.072); g.fillRect(0, 0, W, H);
  for (let i = 0; i < 900; i++) { g.fillStyle = grey(0.05 + Math.random() * 0.05); g.fillRect(Math.random() * W, Math.random() * H, 2 + Math.random() * 6, 1 + Math.random() * 3); }
  g.fillStyle = grey(0.035); g.fillRect(X(0.165), Y(0.60), X(0.555) - X(0.165), Y(0.42) - Y(0.60));           // instrument hood
  for (const s of [1, -1]) {
    const l0 = s > 0 ? 0.15 : -0.45;
    g.fillStyle = grey(0.012); g.fillRect(X(l0), Y(0.701), X(l0 + 0.30) - X(l0), Y(0.665) - Y(0.701));
    g.fillStyle = grey(0.05);
    for (let k = 0; k < 6; k++) { const f = 0.6695 + k * 0.0054; g.fillRect(X(l0 + 0.004), Y(f + 0.001), X(l0 + 0.296) - X(l0 + 0.004), Math.max(1, Y(f - 0.001) - Y(f + 0.001))); }
  }
  g.save(); g.translate(X(-0.42), Y(0.535)); g.rotate(11 * Math.PI / 180);
  g.fillStyle = `rgb(${lin(0.55)},${lin(0.53)},${lin(0.45)})`; g.fillRect(-0.135 / 1.5 * W, -0.075 / 0.35 * H, 0.27 / 1.5 * W, 0.15 / 0.35 * H);
  g.fillStyle = `rgb(${lin(0.30)},${lin(0.40)},${lin(0.24)})`;
  for (let i = 0; i < 14; i++) g.fillRect((Math.random() - 0.5) * 0.25 / 1.5 * W, (Math.random() - 0.5) * 0.13 / 0.35 * H, 6 + Math.random() * 14, 4 + Math.random() * 8);
  g.restore();
  g.save(); g.translate(X(0.16), Y(0.668)); g.rotate(-22 * Math.PI / 180);
  g.fillStyle = grey(0.62); g.fillRect(-0.0275 / 1.5 * W, -0.0525 / 0.35 * H, 0.055 / 1.5 * W, 0.105 / 0.35 * H);
  g.restore();
  g.filter = 'blur(1px)'; g.drawImage(cv, 0, 0);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.NoColorSpace; t.minFilter = THREE.LinearMipmapLinearFilter; t.needsUpdate = true;
  return t;
}

// ------------------------------------------------------------------------------------------------ rear-view mirror (v4)
/** Mirror glass: an off-axis planar-reflection render target (the eye reflected in the mirror plane, frustum = the
 *  mirror outline), low res and low rate, on high/ultra while in the cockpit; a static dim blur on medium/low.
 *  Day-mode prism mirror reflectance ~0.8 (silvered back face) plus a slight grey tint of the glass. */
Vehicle.prototype._setupMirror = function () {
  const T = THREE;
  if (this.mirror) { this.mirror.mesh.removeFromParent(); this.mirror.mesh.geometry.dispose(); this.mirror.rt?.dispose(); }
  const fr = this.json?.rearMirror;
  const c = Array.isArray(fr?.center) ? new T.Vector3().fromArray(fr.center) : new T.Vector3(0, 1.522, 0.2243);
  const n = Array.isArray(fr?.normal) ? new T.Vector3().fromArray(fr.normal).normalize() : new T.Vector3(0, 0, -1);
  const up = Array.isArray(fr?.up) ? new T.Vector3().fromArray(fr.up).normalize() : new T.Vector3(0, 1, 0);
  const w = fr?.width ?? 0.19, h = fr?.height ?? 0.042;
  const xAx = new T.Vector3().crossVectors(up, n).normalize();
  const upO = new T.Vector3().crossVectors(n, xAx).normalize();
  const geo = new T.PlaneGeometry(w, h);
  const q = this.ctx.config?.quality?.key || 'high';
  const live = q === 'ultra' || q === 'high';
  let rt = null, tex;
  if (live) {
    rt = new T.WebGLRenderTarget(q === 'ultra' ? 384 : 288, q === 'ultra' ? 86 : 64, { type: T.HalfFloatType, depthBuffer: true });
    rt.texture.colorSpace = T.NoColorSpace; rt.texture.generateMipmaps = false;
    tex = rt.texture;
  } else tex = makeStaticMirrorTexture();
  tex.wrapS = T.ClampToEdgeWrapping; tex.repeat.x = -1; tex.offset.x = 1;   // a mirror flips left/right
  const mat = new T.MeshBasicMaterial({ name: 'rear_mirror', map: tex, color: new T.Color(0.78, 0.79, 0.80) });
  const mesh = new T.Mesh(geo, mat);
  mesh.name = 'rear_mirror_glass';
  mesh.quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(xAx, upO, n));
  mesh.position.copy(c).addScaledVector(n, 0.0007);
  mesh.castShadow = false; mesh.receiveShadow = false;
  (this.bodyNode || this.object).add(mesh);
  this.interiorMeshes?.push(mesh);
  const cam = new T.PerspectiveCamera(20, w / h, 0.05, 600);
  cam.layers.mask = this.ctx.camera?.layers?.mask ?? 1;
  this.mirror = { mesh, rt, cam, live, c, n, xAx, up: upO, w, h, frame: 0, staticK: -1 };
};

Vehicle.prototype._updateMirror = function () {
  const M = this.mirror, ctx = this.ctx;
  if (!M) return;
  const cockpit = ctx.cameraRig?.mode === 'car-cockpit';
  if (!M.live) {                               // static: brightness follows the scene IBL
    const k = 0.9 * (ctx.scene?.environmentIntensity ?? 1);
    if (Math.abs(k - M.staticK) > 0.02) { M.staticK = k; M.mesh.material.color.setRGB(0.78 * k, 0.79 * k, 0.80 * k); }
    return;
  }
  const r = ctx.renderer, cam = ctx.camera;
  if (!r || !cam || !M.mesh.visible) return;
  // ~1.1 ms per update on the M1 (~0.7 M tris: the instanced forest/terrain can't be culled by the narrow frustum):
  // 30 Hz on ultra, 10 Hz on high while driving in the cockpit; ~1 Hz when the glass is merely seen from outside
  // (chase camera, on foot next to the car); always once so it never shows an unrendered target
  const every = cockpit ? (ctx.config?.quality?.key === 'ultra' ? 2 : 6) : 60;
  if (M.rendered && (M.frame++ % every) !== 0) return;
  if (M.rendered && !cockpit && cam.position.distanceToSquared(this.object.position) > 12 * 12) return;
  // mirror frame in world space
  M.mesh.updateWorldMatrix(true, false);
  const mw = M.mesh.matrixWorld;
  const C = _v1.setFromMatrixPosition(mw);
  _mirNdc.copy(C).project(cam);
  if (_mirNdc.z > 1 || Math.abs(_mirNdc.x) > 1.25 || Math.abs(_mirNdc.y) > 1.25) return;
  const N = _v2.set(0, 0, 1).transformDirection(mw);
  const X = _v3.set(1, 0, 0).transformDirection(mw);
  const Y = _v4.set(0, 1, 0).transformDirection(mw);
  let E = cam.getWorldPosition(_mirE);
  let dist = _mirD.subVectors(E, C).dot(N);
  if (dist < 0.05) {                         // viewer in front of the glass (e.g. walking round the car): driver's eye
    if (M.rendered) return;
    E = this.seatCam.getWorldPosition(_mirE); dist = _mirD.subVectors(E, C).dot(N);
    if (dist < 0.05) return;
  }
  const Ep = _mirEp.copy(E).addScaledVector(N, -2 * dist);
  // camera looks along +N from the reflected eye; right = -X (mirror flip), up = Y
  _m1.makeBasis(_mirX.copy(X).negate(), Y, _mirZ.copy(N).negate());
  M.cam.position.copy(Ep); M.cam.quaternion.setFromRotationMatrix(_m1); M.cam.updateMatrixWorld(true);
  // off-axis frustum through the mirror outline, near plane = mirror plane
  const rel = _mirD.subVectors(C, Ep);
  const cx = rel.dot(_mirX), cy = rel.dot(Y);
  const near = dist, far = 600;
  M.cam.projectionMatrix.makePerspective(cx - M.w / 2, cx + M.w / 2, cy + M.h / 2, cy - M.h / 2, near, far);
  M.cam.projectionMatrixInverse.copy(M.cam.projectionMatrix).invert();
  const prevRT = r.getRenderTarget(), prevAuto = r.shadowMap.autoUpdate;
  const hidden = [];
  // skip what a 288x64 mirror can't resolve (grass, pebbles, deadwood, droplets) and our own wheels / FX (~40 % of the
  // pass's triangles); the forest, terrain, landslide and dust stay
  if (!M.hideList || (M.frame & 255) === 1) {
    M.hideList = ['fx_windshield', 'fx_rain', 'fx_rain_veil', 'fx_rain_mist', 'fx_drops', 'fx_splash', 'fx_chips', 'vegetation',
      'terrain_rocks', 'terrain_deadwood', 'car_contact_shadow', 'car_tyre_spray'].map((nm) => ctx.scene.getObjectByName(nm)).filter(Boolean);
    for (const w of this.wheels) if (w.node) M.hideList.push(w.node);
  }
  for (const o of M.hideList) if (o.visible) { o.visible = false; hidden.push(o); }
  M.mesh.visible = false;
  try {
    r.shadowMap.autoUpdate = false;
    r.setRenderTarget(M.rt);
    r.clear();
    r.render(ctx.scene, M.cam);
    M.rendered = true;
  } catch (e) {
    console.warn('[car] mirror render failed; using the static mirror', e);
    M.live = false; M.mesh.material.map = makeStaticMirrorTexture(); M.mesh.material.map.repeat.x = -1; M.mesh.material.map.offset.x = 1; M.mesh.material.needsUpdate = true;
  } finally {
    r.setRenderTarget(prevRT); r.shadowMap.autoUpdate = prevAuto;
    M.mesh.visible = true;
    for (const o of hidden) o.visible = true;
  }
};

/** medium/low: what a 90s mirror mostly shows on a grey day: the dark cabin frame around the bright rear window. */
function makeStaticMirrorTexture() {
  const W = 128, H = 32;
  const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const g = cv.getContext('2d');
  g.fillStyle = 'rgb(10,10,10)'; g.fillRect(0, 0, W, H);
  const gr = g.createLinearGradient(0, 4, 0, 26);
  gr.addColorStop(0, 'rgb(150,152,156)'); gr.addColorStop(0.45, 'rgb(104,108,110)'); gr.addColorStop(0.55, 'rgb(40,44,40)'); gr.addColorStop(1, 'rgb(34,34,32)');
  g.fillStyle = gr; g.fillRect(18, 3, 92, 24);
  g.fillStyle = 'rgb(22,26,22)';                                   // tree line along the horizon
  for (let x = 18; x < 110; x += 3) g.fillRect(x, 12 - Math.round(Math.abs(Math.sin(x * 1.7)) * 5), 3, 6);
  g.fillStyle = 'rgb(62,62,60)'; g.fillRect(52, 17, 24, 10);        // the road falling away behind
  g.fillStyle = 'rgb(14,14,14)'; g.fillRect(0, 26, W, 6);          // tailgate / rear bench
  g.filter = 'blur(2px)'; g.drawImage(cv, 0, 0);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.NoColorSpace;
  return t;
}

// ------------------------------------------------------------------------------------------------ cluster lens (v4)
/** Clear acrylic lens over the gauges: nearly invisible, but it catches the window-tested cabin reflections. */
Vehicle.prototype._setupLens = function () {
  if (this.gaugeLens) { this.gaugeLens.removeFromParent(); this.gaugeLens.geometry.dispose(); this.gaugeLens.material.dispose(); }
  const L = this.json?.gaugeLens;
  if (!L || !Array.isArray(L.center)) return;
  const t = (L.tilt ?? 15) * DEG;
  const n = new THREE.Vector3(0, Math.sin(t), -Math.cos(t));
  const m = new THREE.MeshStandardMaterial({ name: 'gauge_lens', color: 0x050505, roughness: 0.14, metalness: 0, transparent: true, opacity: 0.03, depthWrite: false });
  alphaGlass(m, false, true);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(L.width ?? 0.345, L.height ?? 0.10), m);
  mesh.name = 'gauge_lens';
  mesh.position.fromArray(L.center);
  mesh.quaternion.setFromUnitVectors(AX_Z, n);
  mesh.renderOrder = 5; mesh.castShadow = false; mesh.receiveShadow = false;
  (this.bodyNode || this.object).add(mesh);
  this.interiorMeshes?.push(mesh);
  this.gaugeLens = mesh;
};

// ------------------------------------------------------------------------------------------------ dangling bits (v4)
/** Prayer beads under the mirror and the key ring swing like damped pendulums driven by the car's accelerations. */
Vehicle.prototype._setupDangles = function () {
  const J = this.json?.dangling || {};
  this.dangles = [];
  for (const [name, L, damp] of [['mirror_charm', J.mirror_charm?.length ?? 0.12, 0.7], ['key_ring', J.key_ring?.length ?? 0.035, 2.2]]) {
    const node = this.model?.getObjectByName(name);
    if (!node) continue;
    this.dangles.push({ node, q0: node.quaternion.clone(), L, damp, ax: 0, vx: 0, az: 0, vz: 0 });
  }
};

Vehicle.prototype._updateDangles = function (dt) {
  if (!this.dangles?.length || dt <= 0) return;
  const g = 9.81, aL = (this.lateralG || 0) * g, aF = (this.longG || 0) * g;
  const vib = (this.engineOn ? 0.6 : 0) + (this._starve > 0 ? 3 : 0) + Math.min(4, (this.bump || 0) * 3);
  const n = Math.min(8, Math.ceil(dt / (1 / 120)));
  const h = dt / n;
  for (const d of this.dangles) {
    const w2 = g / d.L;
    for (let i = 0; i < n; i++) {
      const jx = (Math.random() - 0.5) * vib, jz = (Math.random() - 0.5) * vib;
      d.vx += (-w2 * Math.sin(d.ax) - (aL + jx) / d.L * Math.cos(d.ax) - d.damp * d.vx) * h;
      d.vz += (-w2 * Math.sin(d.az) - (aF + jz) / d.L * Math.cos(d.az) - d.damp * d.vz) * h;
      d.ax = clamp(d.ax + d.vx * h, -1.1, 1.1); d.az = clamp(d.az + d.vz * h, -1.1, 1.1);
    }
    _q1.setFromEuler(_dE.set(-d.az, 0, d.ax));
    d.node.quaternion.copy(d.q0).multiply(_q1);
  }
};

const _mirNdc = new THREE.Vector3(), _mirE = new THREE.Vector3(), _mirD = new THREE.Vector3(), _mirEp = new THREE.Vector3(), _mirX = new THREE.Vector3(), _mirZ = new THREE.Vector3();
const _dE = new THREE.Euler();
