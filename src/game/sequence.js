// GAME workstream: the gameplay state machine, pacing, objectives, checkpoints (DESIGN.md "game").
//
//   game.state        'title' | 'intro' | 'stalled' | 'onfoot' | 'driving' | 'escape' | 'win' | 'dead'
//   game.checkpoint   name of the checkpoint a retry restores ('intro' | 'stall' | 'onfoot' | 'refuel' | 'escape' | 'gap' | 'tunnel')
//   game.inventory    Inventory (a Set of 'jerrycan' | 'hatchet' | 'planks'; pushes itself to the HUD)
//   game.fail(cause)  cause: 'front' | 'boulder' | 'fall' | 'car'
//   game.restartFromCheckpoint()
//   game.enterCar() / game.exitCar()          (extra; used by the interactions and by automated tests)
//   game.applyCheckpoint(name)                (extra; ?skip=<name> does this on ui:start)
//   game.flags        {treeSeen, roadworksSeen, refueled, startAttempts, stuck, ...} (extra, read-only)
//   game.tune         pacing values (config.game merged with the defaults below)
//
// Flow: title (cinematic fly-over) -> intro (cold-open drive, radio, rumble, rockfall at s≈175, fuel runs dry near
// markers.stall) -> stalled ([E] Get out; the debris front starts creeping) -> onfoot (fuel at the roadworks, hatchet
// for the fallen tree, planks for the washout, refuel, get in, start: the first try coughs) -> driving/escape (the
// front chases, gully rockfall; stop at the gap and lay the planks) -> win (car or player past markers.win).
import * as THREE from 'three';
import Inventory from './inventory.js';

const clamp = THREE.MathUtils.clamp;
const smooth01 = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
const _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3();

// Pacing defaults (config.game.* overrides; everything here is owned by GAME).
const DEFAULTS = {
  fuelStart: 0.35,
  jerrycanLiters: 5,
  chopHitsRequired: 3,
  introStartSpeed: 9,          // m/s the cold open starts at (dashcam feel)
  rumbleS: 120,                // first distant rumble
  rockfallS: 175,              // scripted rockfall behind the car
  sputterHoldS: 168,           // fuel cannot run into reserve before this s ...
  sputterForceS: 200,          // ... and is forced into reserve after it (the 11 s reserve phase ends ~235-250)
  footTimeLimit: 360,          // s on foot before the front stops being held behind the car
  frontCarMargin: 24,          // m: while on foot, the front crawls to a stop this far behind the car
  refuelHold: 4.0, chopHold: 0.75, planksHold: 2.2,
  gapWarnDist: 48,             // m before the gap: "the road's gone"
  frontSlowDist: 24,           // escape: the front slows to frontFloor when this close behind the reference point
  frontFloor: 1.6,             // m/s (a car waiting at the gap has ~20 s)
  frontFloorStuck: 4.5,        // m/s once the car is stuck in the trench
  frontMaxFactor: 1.8,         // × chaseSpeed when far behind (rubber band)
  winSlowmo: 0.35,
  lethalRockSpeed: 6.0,        // m/s: a rock must itself be moving this fast to count as a hit (ramming a resting rock is a crash)
  lethalCarEnergy: 34,         // hazard energy of a single hit that crushes a MOVING car outright (a 1.3 m boulder at ~24 m/s)
  lethalCarEnergyStopped: 16,  // ... and a stopped / crawling car (< 3 m/s): sitting under the rockfall is fatal sooner
  carHitPoints: 40,            // lesser hits add clamp(energy / carHitPoints, .08, .45) to car.damage; wrecked at 1
  introCruise: 12,             // m/s: the cold open holds the lane at this pace until the player steers
  rockResumeAfterGap: 22,      // m past the washout (and > 5 m/s) before the escape rockfall resumes
  rockfallCinematic: true,     // cut to a short look-back shot when the intro slope comes down
  debrisFront: { startS: 160, speed0: 0.22, accel: 0.0045, chaseSpeed: 11, chaseLag: 55 },
};
// GAME's own escape tuning, applied over config.game.debrisFront (config.js is the lead's; these values were balanced
// with scratch/game/playthrough.mjs): the front sits ~45-55 m behind a car doing 40-55 km/h (a looming wall in the
// look-back / chase view), gives a stopped car ~12 s, and is beatable without flat-out driving.
const ESCAPE_OVERRIDES = { debrisFront: { chaseSpeed: 11, chaseLag: 46 }, frontSlowDist: 18, frontFloor: 1.5 };

/** Tank fraction (0..1) -> gauge needle fraction: linear over the sender's travel, pinned at E below ~0.1 L. */
export function fuelGauge(f) {
  f = +f || 0;
  if (f <= 0.002) return 0;
  return clamp(f / 0.9, 0, 1);
}

const ITEM_OFFSETS = { jerrycan: new THREE.Vector3(0, 0.26, 0), hatchet: new THREE.Vector3(0, 0.04, 0), planks: new THREE.Vector3(0, 0.22, 0) };
const PLAYING = ['intro', 'stalled', 'onfoot', 'driving', 'escape'];

export default class Sequence {
  constructor(ctx) {
    this.ctx = ctx;
    const g = ctx.config?.game || {};
    this.tune = { ...DEFAULTS, ...g, ...ESCAPE_OVERRIDES, debrisFront: { ...DEFAULTS.debrisFront, ...(g.debrisFront || {}), ...ESCAPE_OVERRIDES.debrisFront } };
    this.state = 'title';
    this.checkpoint = 'intro';
    this.inventory = new Inventory(ctx);
    this.flags = {};
    this.playTime = 0;
    this.deaths = 0;
    this._timers = [];
    this._rt = 0;             // real (unscaled, pause-aware) clock
    this._stateT = 0;         // real seconds in the current state
    this._pendingSkip = null;
    this._slowmo = null;
    this._prCar = {}; this._prPl = {}; this._prRef = {};
    this._offs = [];
    this._resetFlags();
  }

  _resetFlags() {
    this.flags = {
      treeSeen: false, roadworksSeen: false, gapSeen: false, refueled: false, startAttempts: 0, stuck: false,
      rumbled: false, rockfall: false, sputterSaid: false, stallEvt: false, stillT: 0, chops: 0,
      planksHinted: false, footT: 0, stuckSaid: false, buriedWarned: false, escapeStarted: false, tunnelSaid: false,
    };
  }

  // =============================================================================================== init
  init() {
    const { ctx } = this;
    const on = (n, f) => { if (ctx.events?.on) this._offs.push(ctx.events.on(n, f)); };
    on('ui:start', (p) => this._onStart(p));
    on('ui:retry', () => this.restartFromCheckpoint());
    on('hazard:hit', (p) => this._onHazard(p));
    on('car:stall', () => { this.flags.stallEvt = true; if (this.state === 'intro') this._say('No... no, no, no. Come on!', 3); });
    ctx.control = 'none';
    this.car?.setControlsEnabled?.(false);
    this._registerInteractions();
    this._installFuelGauge();
    this._startTitleCinematic();
    if (ctx.flags?.debug) window.__game = this;
  }

  /** The fuel gauge reads the tank through a float sender whose travel ends ~10% above the true bottom (a real "E"
   *  still sloshes a few litres), so the jerrycan's 5 L of 45 shows as a visible ~1/8 tank. vehicle.js calls
   *  hud.setFuel(liters / capacity); GAME remaps that value here (see requestsForLead: a car.fuelGauge hook). */
  _installFuelGauge() {
    const hud = this.ctx.hud;
    if (!hud?.setFuel || hud.__gameFuelMap) return;
    const raw = hud.setFuel.bind(hud);
    hud.__gameFuelMap = true;
    hud.setFuel = (f) => raw(fuelGauge(f));
  }

  get car() { return this.ctx.car; }
  get player() { return this.ctx.player; }
  get road() { return this.ctx.road; }
  get markers() { return this.ctx.road?.markers || {}; }

  // =============================================================================================== helpers
  _after(sec, fn) { this._timers.push({ t: this._rt + sec, fn }); }
  _say(text, sec) { try { this.ctx.hud?.subtitle?.(text, sec); } catch (e) { console.warn('[game] subtitle', e); } }
  _toast(text, sec) { try { this.ctx.hud?.toast?.(text, sec); } catch {} }
  _objective(text) {
    if (text === this._objText) return;
    this._objText = text;
    try { this.ctx.hud?.setObjective?.(text); } catch {}
    this.ctx.events?.emit?.('objective', { text });
  }
  _setState(s) {
    if (s === this.state) return;
    const prev = this.state;
    this.state = s;
    this._stateT = 0;
    this.ctx.events?.emit?.('game:state', { state: s, prev });
  }
  _play(name, opts) { try { return this.ctx.audio?.play?.(name, opts || {}) || null; } catch { return null; } }

  /** Road-projected s/d/dy of the car body (physics truth) or null. */
  carProj() {
    const car = this.car;
    if (!car?.body || !this.road) return null;
    const t = car.body.translation();
    return this._proj(_v1.set(t.x, t.y, t.z), this._prCar);
  }
  /** road.project with a guard: its windowed search can lock onto the wrong stretch after a teleport
   *  (accepts a local minimum up to 30 m away), so re-run it globally when the result looks off-road. */
  _proj(p, out) {
    const r = this.road.project(p, out);
    if (r.dist > 9) { out._hint = -1e6; this.road.project(p, out); }
    return out;
  }
  playerProj() {
    const pl = this.player;
    if (!pl?.feet || !this.road) return null;
    return this._proj(pl.feet, this._prPl);
  }
  /** s of whoever the game is following (car when driving, feet when walking). */
  refProj() { return this.ctx.control === 'foot' ? this.playerProj() : (this.carProj() || this.playerProj()); }

  _roadAhead(s, out) {
    const t = this.road.tangentAt(s, out);
    return t;
  }

  // =============================================================================================== car in/out
  enterCar(opts = {}) {
    const { ctx } = this;
    const car = this.car;
    if (!car) return false;
    this.player?.setEnabled?.(false);
    ctx.control = 'car';
    car.aiInput = null;
    car.setControlsEnabled?.(true);
    const rig = ctx.cameraRig;
    rig?.setMode?.(rig.carMode || 'car-cockpit', { blend: opts.snap ? 0 : 0.7 });
    if (opts.snap) rig?.snap?.();
    rig?.viewmodel?.setHeld?.(null);
    ctx.hud?.setDriving?.(true);
    ctx.interact?.lock?.(0.6);
    if (!opts.silent) ctx.events?.emit?.('car:enter', {});
    return true;
  }

  exitCar(opts = {}) {
    const { ctx } = this;
    const car = this.car, pl = this.player;
    if (!pl) return false;
    // feet at the door point (on the ground), facing along the car
    const pos = _v2;
    if (car?.doorPoint) {
      car.object?.updateMatrixWorld?.(true);
      car.doorPoint.getWorldPosition(pos);
    } else if (this.road) {
      const pr = this.carProj();
      this.road.worldAt(pr ? pr.s : (this.markers.stall ?? 235), 0.2, pos);
    }
    const gy = ctx.physics?.groundHeight?.(pos.x, pos.z, pos.y + 1.6, 4);
    if (gy != null && isFinite(gy)) pos.y = gy;
    let yaw = pl.yaw;
    if (car?.object) {
      const f = _v3.set(0, 0, 1).applyQuaternion(car.object.quaternion);
      yaw = Math.atan2(-f.x, -f.z) + 0.35; // a glance outward (toward the uphill side) as you step out
    }
    pl.teleport?.(pos, yaw);
    pl.setEnabled?.(true);
    if (car) { car.setControlsEnabled?.(false); car.aiInput = { throttle: 0, brake: 1, steer: 0, handbrake: true }; }
    ctx.control = 'foot';
    const rig = ctx.cameraRig;
    rig?.setMode?.('foot', { blend: opts.snap ? 0 : 0.65 });
    if (opts.snap) rig?.snap?.();
    ctx.hud?.setDriving?.(false);
    ctx.interact?.lock?.(0.6);
    if (!opts.silent) ctx.events?.emit?.('car:exit', {});
    return true;
  }

  // =============================================================================================== interactions
  _registerInteractions() {
    const ia = this.ctx.interact;
    if (!ia?.register) return;
    const self = this;
    const inv = this.inventory;
    const onFootPlaying = () => this.ctx.control === 'foot' && (this.state === 'onfoot' || this.state === 'driving' || this.state === 'escape');

    // ---- pick-ups at the roadworks
    const pickup = (id, prompt, radius) => ia.register({
      id: 'pickup_' + id, radius, angle: 34,
      object: () => this.ctx.props?.items?.[id] || null,
      offset: ITEM_OFFSETS[id],
      prompt,
      canUse: () => onFootPlaying() && !inv.has(id),
      use: () => self._pickup(id),
    });
    pickup('jerrycan', 'Take the jerrycan', 2.3);
    pickup('hatchet', 'Take the hatchet', 2.2);
    pickup('planks', 'Take the scaffold planks', 2.6);

    // ---- chop the fallen tree (3 hold-E chops)
    ia.register({
      id: 'chop', radius: 2.5, angle: 42, hold: this.tune.chopHold, repeat: true, priority: 0.2,
      position: () => {
        const ft = this.ctx.vegetation?.fallenTree;
        return ft && !ft.cut && ft.chopPoint ? ft.chopPoint : null;
      },
      prompt: () => `Chop the trunk  (${Math.min(this.flags.chops + 1, this.tune.chopHitsRequired)}/${this.tune.chopHitsRequired})`,
      hint: () => (onFootPlaying() && !inv.has('hatchet') ? 'Too thick to move. You need something to cut it' : null),
      canUse: () => onFootPlaying() && inv.has('hatchet') && !this.ctx.vegetation?.fallenTree?.cut,
      onHold: (p) => { this.ctx.cameraRig?.viewmodel?.windup?.(p); },
      onCancel: () => { this.ctx.cameraRig?.viewmodel?.windup?.(0); },
      use: () => self._chop(),
    });

    // ---- refuel at the filler cap
    ia.register({
      id: 'refuel', radius: 1.9, angle: 50, hold: this.tune.refuelHold, priority: 0.5,
      object: () => this.car?.fuelCap || null,
      requireVisible: false,
      prompt: 'Refuel',
      hint: () => (onFootPlaying() && !this.flags.refueled && !inv.has('jerrycan') ? 'Fuel filler. The tank is empty' : null),
      canUse: () => onFootPlaying() && inv.has('jerrycan'),
      onHold: (p) => {
        // glug-glug: re-touching the pour voice keeps it looping
        if (!this._pourT || this._rt - this._pourT > 0.45) { this._pourT = this._rt; this._pour = this._play('fuelPour', { position: this.car?.fuelCap }) || this._pour; }
      },
      onCancel: () => { try { this._pour?.stop?.(0.25); } catch {} this._pour = null; this._pourT = 0; },
      use: () => self._refuel(),
    });

    // ---- get in (door)
    ia.register({
      id: 'get_in', radius: 2.3, angle: 50, priority: 0.1,
      position: () => {
        const car = this.car;
        if (!car?.doorPoint || !car.object) return null;
        car.doorPoint.getWorldPosition(_v3);
        const c = car.object.position;
        _v3.x += (c.x - _v3.x) * 0.45; _v3.z += (c.z - _v3.z) * 0.45; _v3.y += 1.0;
        return _v3;
      },
      prompt: () => (this.flags.refueled ? 'Get in' : 'Get in the car'),
      canUse: () => onFootPlaying(),
      use: () => self._getIn(),
    });

    // ---- lay the planks over the washout
    ia.register({
      id: 'place_planks', radius: 2.9, angle: 55, hold: this.tune.planksHold, priority: 0.4,
      position: () => {
        if (!this.road || this.ctx.props?.bridge) return null;
        const p = this.road.worldAt((this.markers.gap ?? 560) - 1.9, -1.5, _v3);
        p.y += 0.1;
        return p;
      },
      prompt: 'Lay the planks across',
      hint: () => (onFootPlaying() ? 'Far too wide to drive across. It needs a bridge' : null),
      canUse: () => onFootPlaying() && inv.has('planks'),
      onHold: (p) => { if (p > 0.05 && !this._plankSfx) { this._plankSfx = true; this._play('thud', { position: this.road.worldAt((this.markers.gap ?? 560) - 1.6, -1.5, new THREE.Vector3()) }); } },
      onCancel: () => { this._plankSfx = false; },
      use: () => self._placePlanks(),
    });

    // ---- seated: start engine / get out
    ia.register({
      id: 'start_engine', mode: 'car', priority: 5,
      prompt: 'Start the engine',
      hint: () => {
        if (this.ctx.control !== 'car' || this.car?.engineOn || !(this.car?.fuel > 0.001)) return null;
        if (!this.ctx.vegetation?.fallenTree?.cut && this._carBeforeTree()) return 'The fallen tree still blocks the road';
        if (!inv.has('planks') && !this.ctx.props?.bridge && this._carBeforeGap()) return 'Get the planks first: the road ahead is washing out';
        return null;
      },
      canUse: () => {
        const car = this.car;
        if (this.ctx.control !== 'car' || !car || car.engineOn || !(car.fuel > 0.001) || (car.cranking > 0)) return false;
        if (!this.ctx.vegetation?.fallenTree?.cut && this._carBeforeTree()) return false;
        if (!inv.has('planks') && !this.ctx.props?.bridge && this._carBeforeGap()) return false;
        return true;
      },
      use: () => self._tryStart(),
    });
    ia.register({
      id: 'get_out', mode: 'car', priority: 1,
      prompt: 'Get out',
      canUse: () => this.ctx.control === 'car' && ['stalled', 'driving', 'escape'].includes(this.state) && Math.abs(this.car?.speed ?? 0) < 1.2,
      use: () => self._getOut(),
    });
  }

  _carBeforeTree() { const pr = this.carProj(); return !pr || pr.s < (this.markers.fallenTree ?? 305) + 3; }
  _carBeforeGap() { const pr = this.carProj(); return !pr || pr.s < (this.markers.gap ?? 560); }

  _pickup(id) {
    const { ctx } = this;
    ctx.props?.hideItem?.(id);
    this.inventory.add(id);
    ctx.events?.emit?.('item:pickup', { id });
    ctx.interact?.lock?.(0.35);
    if (id === 'jerrycan') {
      this._say(this.flags.refueled ? 'Empty now.' : 'Heavy. Sloshing. Maybe five litres. It will do.', 3.5);
      this._toast('Picked up: Jerrycan (5 L)', 3);
    } else if (id === 'hatchet') {
      this._say(this.ctx.vegetation?.fallenTree?.cut ? 'Might come in handy.' : 'A hatchet. That trunk is not going to like this.', 3.5);
      this._toast('Picked up: Hatchet', 3);
    } else if (id === 'planks') {
      this._say('Heavy boards. If the road really is washed out up there, these might get me across.', 4);
      this._toast('Picked up: Scaffold planks', 3);
    }
    this._afterPickupHints(id);
    this._refreshObjective();
  }

  _afterPickupHints(id) {
    const inv = this.inventory;
    if (!this.flags.planksHinted && id !== 'planks' && !inv.has('planks') && !this.ctx.props?.bridge) {
      this.flags.planksHinted = true;
      this._after(4.2, () => {
        if (!inv.has('planks') && this.state === 'onfoot') this._say('RADIO: ...the road above the roadworks is reported washed out. Do not attempt...', 5);
      });
    }
  }

  _chop() {
    const ft = this.ctx.vegetation?.fallenTree;
    const req = this.tune.chopHitsRequired;
    if (!ft || ft.cut || this._chopping) return;
    this._chopping = true;
    const vm = this.ctx.cameraRig?.viewmodel;
    try { vm?.strike?.(); } catch {}
    this.ctx.interact?.lock?.(0.55);
    // the blade meets the wood ~0.13 s into the swing (independent of the viewmodel promise)
    this._after(0.13, () => {
      this._chopping = false;
      this.flags.chops++;
      const done = this.flags.chops >= req;
      this.ctx.cameraRig?.shake?.(0.16);
      try { this.ctx.particles?.debris?.(ft.chopPoint.clone(), 2, { wood: true, speed: 3.2, up: 0.9 }); } catch {}   // wood splinters (not rock chips; vegetation's hit() adds a few more)
      this.ctx.events?.emit?.('tree:chop', { hits: this.flags.chops, done });
      if (done) {
        try { ft.split?.(); } catch (e) { console.warn('[game] fallenTree.split', e); } // idempotent (vegetation also splits on tree:chop done)
        this._say('Come on... there!', 2.5);
        this._after(1.6, () => { if (this.state === 'onfoot') this._say('The lane is clear. Now fuel.', 3); });
        this._refreshObjective();
      } else if (this.flags.chops === 1) {
        this._say('Soaked through. This will take a few.', 2.8);
      }
    });
  }

  _refuel() {
    const car = this.car;
    try { this._pour?.stop?.(0.4); } catch {}
    this._pour = null; this._pourT = 0;
    const L = this.tune.jerrycanLiters ?? 5;
    car?.refuel?.(L);
    this.inventory.delete('jerrycan');
    this.flags.refueled = true;
    this._say('Five litres. That will have to be enough.', 3.2);
    this._refreshObjective();
  }

  _getIn() {
    if (!this.enterCar()) return;
    if (this.state === 'onfoot') {
      this._refreshObjective();
      if (!this.flags.refueled) this._after(0.9, () => this._say('Still empty. I need fuel.', 2.5));
    }
  }

  _getOut() {
    if (!this.exitCar()) return;
    if (this.state === 'stalled') {
      this._setState('onfoot');
      this.checkpoint = this.checkpoint === 'stall' ? 'stall' : this.checkpoint;
      this._after(1.2, () => this._say('Roadworks signs back there said one kilometre. There has to be fuel somewhere up ahead.', 5));
      this._refreshObjective();
    } else if (this.state === 'driving' || this.state === 'escape') {
      this._refreshObjective();
    }
  }

  _tryStart() {
    const car = this.car;
    if (!car) return;
    this.flags.startAttempts++;
    this.ctx.interact?.lock?.(1.6);
    if (this.flags.startAttempts === 1) {
      // flooded carburettor: it catches, coughs and dies (for drama)
      car.cranking = 1.3;
      this.ctx.events?.emit?.('car:startFail', { reason: 'flooded' });
      this.ctx.cameraRig?.shake?.(0.06);
      this._say('Come on... come on!', 2.2);
      return;
    }
    const ok = car.start?.();
    if (ok === false) { this._say('Nothing...', 2); return; }
    this.ctx.cameraRig?.shake?.(0.1);
    this._onEngineStarted();
  }

  _onEngineStarted() {
    this._setState('driving');
    this.checkpoint = 'escape';
    this._saveCheckpoint('escape');
    this._say('YES! Go, go, go!', 2.5);
    this._objective('Drive! Get to the tunnel before the mountain comes down');
    this._after(1.2, () => this._beginEscape());
  }

  _beginEscape() {
    if (this.state !== 'driving') return;
    this._setState('escape');
    this.flags.escapeStarted = true;
    try { this.ctx.landslide?.startEscape?.(); } catch (e) { console.warn('[game] startEscape', e); }
    this.ctx.cameraRig?.shake?.(0.35);
    this._play('crack', { volume: 1 });
  }

  _placePlanks() {
    const { ctx } = this;
    this._plankSfx = false;
    let bridge = null;
    try { bridge = ctx.props?.placePlanks?.(); } catch (e) { console.warn('[game] placePlanks', e); }
    this.inventory.delete('planks');
    ctx.events?.emit?.('planks:placed', {});
    ctx.cameraRig?.shake?.(0.08);
    this._say(bridge ? 'That should hold. I hope.' : 'That will have to do.', 2.8);
    this._refreshObjective();
  }

  _refreshObjective() {
    const st = this.state;
    const inv = this.inventory;
    const F = this.flags;
    const ft = this.ctx.vegetation?.fallenTree;
    const treeCut = !ft || ft.cut;
    const bridge = !!this.ctx.props?.bridge;
    let t = '';
    if (st === 'stalled') t = 'Get out and find fuel';
    else if (st === 'onfoot') {
      if (!inv.has('jerrycan') && !F.refueled) t = F.treeSeen && !treeCut && !inv.has('hatchet') ? 'Find fuel at the roadworks, and something to cut the tree' : 'Find fuel. The roadworks up ahead might have some';
      else if (F.treeSeen && !treeCut && !inv.has('hatchet')) t = 'Find something to cut through the fallen tree';
      else if (F.roadworksSeen && !inv.has('planks') && !bridge) t = 'Take the scaffold planks from the roadworks';
      else if (!treeCut && inv.has('hatchet')) t = 'Cut through the fallen tree';
      else if (inv.has('jerrycan')) t = 'Refuel the car';
      else if (this.ctx.control === 'car') t = 'Start the engine';
      else t = 'Get in the car';
    } else if (st === 'driving' || st === 'escape') {
      const pr = this.carProj();
      const gap = this.markers.gap ?? 560;
      if (F.stuck) t = 'The car is stuck';
      else if (!bridge && pr && pr.s > gap - this.tune.gapWarnDist && pr.s < gap + 2) t = this.ctx.control === 'foot' ? 'Lay the planks across the washout' : 'Stop! Get out and lay the planks across the washout';
      else if (bridge && pr && pr.s < gap + 2) t = this.ctx.control === 'foot' ? 'Get back in the car' : 'Drive across the planks';
      else t = 'Drive! Get to the tunnel before the mountain comes down';
    }
    if (t) this._objective(t);
  }

  // =============================================================================================== start / title
  _startTitleCinematic() {
    const rig = this.ctx.cameraRig, road = this.road;
    if (!rig?.cinematic || !road) return;
    this.ctx.control = 'none';
    rig.cinematic(titlePath(road), 72, { loop: true, blend: 0.001 });
  }

  _onStart(p) {
    if (this.state !== 'title') return;
    this.playTime = 0;
    const skip = this.ctx.flags?.skip;
    // Rapier scene queries are only valid after a physics step: apply once a few frames have run
    this._pendingSkip = skip && CHECKPOINTS[skip] ? skip : 'intro';
    this._pendingAuto = !!p?.auto;
    this._pendingAt = this._rt + (this._pendingAuto ? 0 : 0.45);
    if (!this._pendingAuto) { try { this.ctx.hud?.fade?.(true, 0.4); } catch {} }
    this._setState('intro'); // hides the title; the real setup happens in update()
    if (this._pendingAuto) this.ctx.cameraRig?.stopCinematic?.(false);
  }

  _beginIntro() {
    const { ctx } = this;
    this.applyCheckpoint('intro', { quiet: true });
    if (ctx.env) { try { ctx.env.rain = Math.max(ctx.env.rain ?? 0.35, 0.5); } catch {} }
    this._objective('Keep driving. Find fuel before the pass');
    // hands-off: the car holds its lane at a steady pace until the player first steers (W/S still change the pace)
    const car = this.car;
    if (car?.body) { this._introAP = { speed: Math.max(this.tune.introCruise, 1), d: -1.5 }; car.autopilot = this._introAP; }
    const T = [
      [1.5, 'Fuel light has been on since the last village...'],
      [6.0, 'Twenty more kilometres over the pass. Come on, old girl.'],
      [12.0, 'RADIO: ...heavy rain on the southern slopes... risk of landslides... drivers are advised...'],
      [18.5, 'RADIO: ...the pass road above the roadworks is reported washed out...'],
    ];
    for (const [t, s] of T) this._after(t, () => { if (this.state === 'intro') this._say(s, 4.8); });
  }

  // =============================================================================================== checkpoints
  _saveCheckpoint(name) {
    const pr = this.carProj();
    if (!pr) return;
    if (name === 'stall' || name === 'escape') this._stallS = clamp(pr.s, 180, 290);
  }

  /** Restore a named game situation (checkpoint restart and ?skip=). */
  applyCheckpoint(name, opts = {}) {
    const C = CHECKPOINTS[name];
    if (!C) { console.warn('[game] unknown checkpoint', name); return; }
    const { ctx } = this;
    const m = this.markers;
    const tune = this.tune;
    const car = this.car, pl = this.player, props = ctx.props, slide = ctx.landslide;
    this._timers = [];
    this._endSlowmo(true);
    this._pour = null;
    this._resetFlags();
    this._dmgSaid = this._dmgSaid2 = false;
    this._introAP = null;
    const stallS = this._stallS ?? ((m.stall ?? 235) + 7);
    const c = C(this, { stallS, m, tune });
    this.checkpoint = c.checkpoint ?? name;
    // world
    try { slide?.clear?.(); } catch (e) { console.warn('[game] landslide.clear', e); }
    if (c.bridge) { try { props?.placePlanks?.(); } catch {} } else { try { props?.removePlanks?.(); } catch {} }
    const inv = c.inv || [];
    this.inventory.set(inv);
    for (const id of ['jerrycan', 'hatchet', 'planks']) {
      const used = (id === 'jerrycan' && c.refueled) || (id === 'planks' && c.bridge);
      if (inv.includes(id) || used) props?.hideItem?.(id); else props?.showItem?.(id);
    }
    const ft = ctx.vegetation?.fallenTree;
    if (c.treeCut && ft && !ft.cut) { try { ft.split?.(); } catch {} }
    this.flags.refueled = !!c.refueled;
    this.flags.startAttempts = c.startAttempts ?? 0;
    this.flags.treeSeen = !!c.treeCut || !!(ft && ft.cut);
    this.flags.roadworksSeen = !!c.roadworksSeen;
    this.flags.rumbled = this.flags.rockfall = name !== 'intro';
    this.flags.escapeStarted = !!c.escape;
    this.flags.footT = c.footT ?? 0;
    this.flags.chops = ft?.cut ? tune.chopHitsRequired : 0;
    // car
    if (car?.body) {
      car.aiInput = null; car.autopilot = null;
      car.damage = 0;
      car.fuel = c.fuel;
      if (c.engine) { if (!car.engineOn) { car.engineOn = true; car.stalling = false; } }
      else car.stop?.();
      car.cranking = 0;
      car.teleport?.(c.carS, c.carD ?? -1.5, 0, { speed: c.speed || 0 });
      if (c.engine) car.rpm = Math.max(car.rpm || 0, 850);
    }
    // front
    if (slide) {
      if (c.frontS > 0) { slide.setFront?.(c.frontS, true); slide.setFrontSpeed?.(c.frontSpeed ?? 0); }
      if (c.escape && c.carS < (m.gap ?? 560) + 1 + tune.rockResumeAfterGap && c.carS > (m.gap ?? 560) - 30) this.flags.rockPaused = true;
      else if (c.escape) { try { slide.startEscape?.(); } catch {} }
    }
    // player / control
    if (c.onFoot && pl) {
      ctx.control = 'foot';
      if (car?.object) car.object.updateMatrixWorld(true);
      if (c.playerAt === 'fuelCap' && car?.fuelCap) {
        this.exitCar({ snap: true, silent: true });
        car.fuelCap.getWorldPosition(_v2);
        const cc = car.object.position;
        const dir = _v3.set(_v2.x - cc.x, 0, _v2.z - cc.z).normalize();
        const feet = _v1.copy(_v2).addScaledVector(dir, 0.9);
        const gy = ctx.physics?.groundHeight?.(feet.x, feet.z, feet.y + 2, 5);
        if (gy != null) feet.y = gy;
        pl.teleport?.(feet, Math.atan2(dir.x, dir.z));
      } else this.exitCar({ snap: true, silent: true });
    } else {
      this.enterCar({ snap: true, silent: true });
    }
    ctx.cameraRig?.snap?.();
    ctx.interact?.lock?.(0.5);
    ctx.hud?.damageVignette?.(0);
    this._setState(c.state);
    if (car && c.state === 'intro') car.setControlsEnabled?.(true);
    this._refreshObjective();
    if (c.state === 'driving') this._after(1.2, () => this._beginEscape());
    if (!opts.quiet && c.say) this._after(0.8, () => this._say(c.say, 3.5));
  }

  restartFromCheckpoint() {
    const { ctx } = this;
    if (this.state === 'title') return;
    const name = this.checkpoint || 'stall';
    try { ctx.hud?.showScreen?.(null); } catch {}
    try { ctx.hud?.fade?.(true, 0); } catch {}
    if (name === 'intro') this._beginIntro(); else this.applyCheckpoint(name);
    try { ctx.hud?.fade?.(false, 1.4); } catch {}
  }

  // =============================================================================================== fail / win
  _onHazard(p) {
    if (!PLAYING.includes(this.state)) return;
    this.lastHazard = { ...p, s: this.refProj()?.s };
    if (p?.cause === 'boulder' && p?.target === 'car') {
      // Damage model: a boulder hit dents the car (car.damage), cracks the windscreen (particles read car.damage), jolts
      // the camera; it only kills outright on a very large impact (lower threshold if the car is sitting still), or
      // once the accumulated damage wrecks the car.
      const car = this.car;
      const e = p?.energy ?? 0;
      const fell = p.rockSpeed != null ? (this.lastHazard.rockSpeed = +p.rockSpeed.toFixed(2)) > this.tune.lethalRockSpeed : this._rockFellOnCar(p);
      const vCar = Math.abs(car?.speed ?? 0);
      const crushE = vCar < 3 ? this.tune.lethalCarEnergyStopped : this.tune.lethalCarEnergy;
      if (!(fell && e >= crushE)) {
        const dmg = fell ? clamp(e / this.tune.carHitPoints, 0.08, 0.45) : 0.03;
        if (car) car.damage = clamp((car.damage || 0) + dmg, 0, 1);
        this.ctx.cameraRig?.shake?.(clamp(0.4 + dmg * 1.2, 0, 1));
        try { this.ctx.post?.impactBlur?.(0.4 + dmg); this.ctx.hud?.damageVignette?.(0.35 + dmg); } catch {}
        this.ctx.events?.emit?.('car:damage', { damage: car?.damage ?? 0, hit: dmg, energy: e });
        this.lastHazard.ignored = true; this.lastHazard.dmg = +dmg.toFixed(3); this._ignored = (this._ignored || 0) + 1;
        this._after(0.7, () => { if (PLAYING.includes(this.state)) this.ctx.hud?.damageVignette?.(clamp((car?.damage || 0) * 0.35, 0, 0.3)); });
        if (car && car.damage >= 1) { this.fail('boulder'); return; }
        if (fell) {
          if (!this._dmgSaid) { this._dmgSaid = true; this._say('Argh! Keep going... keep going!', 2); }
          else if (car && car.damage > 0.6 && !this._dmgSaid2) { this._dmgSaid2 = true; this._say("She can't take another one of those...", 2.6); }
        }
        return;
      }
    }
    const cause = p?.cause === 'front' && p?.target === 'car' ? 'car' : (p?.cause || 'boulder');
    this.fail(cause);
  }

  /** True when a fast-moving rock (its own speed, not the car's) is at the car: a real hit from above/the side. */
  _rockFellOnCar(p) {
    const rocks = this.ctx.landslide?.rocks, car = this.car;
    if (!rocks?.length || !car?.body) return true; // can't tell: trust the hazard
    const t = car.body.translation();
    let best = 0;
    for (const R of rocks) {
      if (!R?.pos || !R.prevV) continue;
      const d = Math.hypot(R.pos.x - t.x, R.pos.y - t.y - 0.8, R.pos.z - t.z);
      if (d > 3.2 + (R.r || 0)) continue;
      best = Math.max(best, R.prevV.length());
    }
    this.lastHazard.rockSpeed = +best.toFixed(2);
    return best > this.tune.lethalRockSpeed;
  }

  fail(cause = 'front') {
    if (!PLAYING.includes(this.state)) return;
    const { ctx } = this;
    this._setState('dead');
    this.deaths++;
    ctx.control = 'none';
    this.deathCause = cause;
    const car = this.car;
    if (car) { car.aiInput = { throttle: 0, brake: 1, steer: 0 }; }
    try { this._pour?.stop?.(0.2); } catch {}
    ctx.cameraRig?.shake?.(cause === 'fall' ? 0.3 : 0.9);
    ctx.hud?.damageVignette?.(1);
    try { ctx.post?.impactBlur?.(0.8); } catch {}
    this._startSlowmo(0.3, 1.3);
    ctx.events?.emit?.('dead', { cause });
    this._after(1.5, () => {
      if (this.state !== 'dead') return;
      this._endSlowmo(true);
      ctx.control = 'none';
      try { ctx.hud?.showScreen?.('dead', { cause }); } catch {}
    });
  }

  _win() {
    const { ctx } = this;
    this._setState('win');
    try { ctx.landslide?.stopEscape?.(); ctx.landslide?.setFrontSpeed?.(0); } catch {}
    const car = this.car;
    if (car && ctx.control === 'car') car.aiInput = { throttle: 0, brake: 0.45, steer: 0 };
    this._startSlowmo(this.tune.winSlowmo, 2.4);
    const time = ctx.hud?.playTime ?? this.playTime;
    this.winTime = time;
    this._say('...', 1.5);
    this._objective('');
    ctx.events?.emit?.('win', { time });
    this._after(1.6, () => {
      Promise.resolve(ctx.hud?.fade?.(true, 2.2)).then(() => {
        if (this.state !== 'win') return;
        this._endSlowmo(true);
        ctx.control = 'none';
        try { ctx.hud?.showScreen?.('win', { time }); } catch {}
        try { ctx.hud?.fade?.(false, 1.0); } catch {}
      });
    });
  }

  _startSlowmo(scale, seconds) { this._slowmo = { scale, t: 0, dur: seconds }; this.ctx.time.scale = scale; }
  _endSlowmo(force) { if (this._slowmo || force) { this._slowmo = null; this.ctx.time.scale = 1; } }

  // =============================================================================================== frame
  update(dt) {
    const { ctx } = this;
    const scale = ctx.time?.scale || 1;
    const rdt = ctx.paused ? 0 : (dt > 0 ? dt / scale : 0);
    if (rdt <= 0) return;
    this._rt += rdt;
    this._stateT += rdt;
    if (PLAYING.includes(this.state)) this.playTime += rdt;

    // timers (real time)
    if (this._timers.length && this._timers.some((a) => a.t <= this._rt)) {
      const due = this._timers.filter((a) => a.t <= this._rt);
      {
        this._timers = this._timers.filter((a) => a.t > this._rt);
        for (const a of due) { try { a.fn(); } catch (e) { console.error('[game] timer', e); } }
      }
    }
    // slow-mo ramp back (real time)
    if (this._slowmo) {
      const S = this._slowmo;
      S.t += rdt;
      const k = smooth01(S.dur * 0.45, S.dur, S.t);
      ctx.time.scale = S.scale + (1 - S.scale) * k;
      if (S.t >= S.dur) this._endSlowmo(true);
    }

    if (this._pendingSkip && (ctx.time.frame ?? 99) > 3 && this._rt >= (this._pendingAt || 0)) {
      const name = this._pendingSkip; this._pendingSkip = null;
      ctx.cameraRig?.stopCinematic?.(false);
      if (name === 'intro') this._beginIntro();
      else { this.applyCheckpoint(name); }
      if (!this._pendingAuto) { try { ctx.hud?.fade?.(false, 1.2); } catch {} }
      return;
    }
    if (this._pendingSkip) return;

    // hatchet in hand while it is useful; planks slow you down
    const inv = this.inventory;
    const ft = ctx.vegetation?.fallenTree;
    ctx.cameraRig?.viewmodel?.setHeld?.(inv.has('hatchet') && ft && !ft.cut ? 'hatchet' : null);
    if (this.player) this.player.speedScale = inv.has('planks') ? 0.84 : 1;

    switch (this.state) {
      case 'intro': this._updateIntro(rdt); break;
      case 'stalled': this._updateFront(rdt, 'foot'); this._checkFalls(); break;
      case 'onfoot': this._updateOnFoot(rdt); break;
      case 'driving': case 'escape': this._updateEscape(rdt); break;
      default: break;
    }
  }

  _updateIntro(dt) {
    const car = this.car;
    const pr = this.carProj();
    if (!car || !pr) return;
    const F = this.flags, T = this.tune;
    this._updateIntroAssist(dt);
    // soft fuel script: reserve cannot start before sputterHoldS and must start by sputterForceS
    if (!F.stallEvt && car.engineOn) {
      if (pr.s < T.sputterHoldS && car.fuel < 0.066) car.fuel = 0.066;
      if (pr.s > T.sputterForceS && car.fuel > 0.0599) car.fuel = 0.0599;
    }
    if (!F.rumbled && pr.s > T.rumbleS) {
      F.rumbled = true;
      this.ctx.cameraRig?.shake?.(0.18);
      this._play('thunder', { volume: 0.7, close: false });
      this._after(1.4, () => { if (this.state === 'intro') this._say('What was that? Thunder?', 2.6); });
    }
    if (!F.rockfall && pr.s > T.rockfallS) {
      F.rockfall = true;
      try { this.ctx.landslide?.triggerIntroRockfall?.(); } catch (e) { console.warn('[game] triggerIntroRockfall', e); }
      this.ctx.cameraRig?.shake?.(0.75);
      if (T.rockfallCinematic) this._rockfallShot();
      this._after(0.9, () => this.ctx.cameraRig?.shake?.(0.5));
      this._after(1.2, () => { if (this.state === 'intro') this._say('Jesus! The whole slope is coming down back there!', 3.2); });
      this._after(4.8, () => { if (this.state === 'intro') this._say('No way back now. Keep going, keep going...', 3); });
    }
    if (!F.sputterSaid && car.stalling) {
      F.sputterSaid = true;
      this._after(0.6, () => { if (this.state === 'intro') this._say('Not now... please, not now.', 2.6); });
    }
    // coast to a halt after the stall
    if (F.stallEvt || (!car.engineOn && F.rockfall)) {
      if (Math.abs(car.speed) < 0.25) F.stillT += dt; else F.stillT = 0;
      if (F.stillT > 0.6) this._enterStalled();
    }
    this._checkFalls();
  }

  /** Lane-hold for the cold open: releases on the first steering key; throttle/brake keys trim the pace meanwhile. */
  _updateIntroAssist(dt) {
    const ap = this._introAP, car = this.car, inp = this.ctx.input;
    if (!ap || !car) return;
    if (car.autopilot !== ap) { if (car.autopilot == null && this.ctx.cameraRig?.mode !== 'cinematic') this._introAP = null; return; } // someone else drives (cinematic, tests)
    if (this.ctx.control !== 'car' || !inp?.down || this.ctx.cameraRig?.mode === 'cinematic') return;
    const k = (c) => { try { return !!inp.down(c); } catch { return false; } };
    if (k('KeyA') || k('KeyD') || k('ArrowLeft') || k('ArrowRight')) { car.autopilot = null; this._introAP = null; return; }
    const v = Math.max(0, car.speed || 0);
    if (k('KeyW') || k('ArrowUp')) ap.speed = Math.min(Math.max(ap.speed, v) + 3.5 * dt, 24);
    else if (k('KeyS') || k('ArrowDown') || k('Space')) ap.speed = Math.max(0, Math.min(ap.speed, v) - 7 * dt);
  }

  /** 3.4 s look-back while the car keeps going on autopilot: the camera rides ahead of the car, looking past it at the
   *  slope collapsing onto the road behind. Control returns to the player afterwards. */
  _rockfallShot() {
    const { ctx } = this;
    const car = this.car, rig = ctx.cameraRig, road = this.road;
    if (!car?.object || !rig?.cinematic || !road) return;
    const hadAuto = !!car.autopilot;
    const prev = car.autopilot;
    if (!hadAuto) car.autopilot = { speed: clamp(car.speed, 8, 13), d: -1.5 };
    const back = new THREE.Vector3(), pr = {};
    const scarS = this.markers.rockfallIntro ?? 150;
    rig.cinematic((t, pos, target) => {
      const cp = car.object.position;
      road.project(cp, pr);
      road.worldAt(pr.s + 4.5 - t * 1.5, -3.3 + t * 0.4, pos);
      pos.y = cp.y + 1.25 + t * 0.35;
      road.worldAt(scarS + 2, 2.5, back);
      back.y = road.pointAt(scarS, _v3).y + 3.5 - t * 1.5;
      target.copy(back);
      return { fov: 58 };
    }, 3.4, { blend: 0.25 }).then(() => {
      if (!hadAuto && car.autopilot && this.state === 'intro') car.autopilot = prev || null;
      if (this.state === 'intro') rig.setMode(rig.carMode || 'car-cockpit', { blend: 0.35 });
    });
  }

  _enterStalled() {
    const { ctx } = this;
    const car = this.car;
    if (!this.flags.rockfall) { this.flags.rockfall = true; try { ctx.landslide?.triggerIntroRockfall?.(); } catch {} }
    if (car && this._introAP && car.autopilot === this._introAP) car.autopilot = null;
    this._introAP = null;
    this._setState('stalled');
    this.checkpoint = 'stall';
    this._saveCheckpoint('stall');
    if (car) { car.aiInput = { throttle: 0, brake: 0.6, steer: 0, handbrake: true }; }
    const slide = ctx.landslide;
    if (slide) {
      const s0 = this.tune.debrisFront.startS;
      slide.setFront?.(Math.max(s0, slide.frontS || 0));
      slide.setFrontSpeed?.(this.tune.debrisFront.speed0);
    }
    this.flags.footT = 0;
    this._say('Empty. Bone dry.', 2.5);
    this._after(2.8, () => { if (this.state === 'stalled') this._say('And that sound... the slope behind me is still moving.', 3.6); });
    this._refreshObjective();
  }

  /** The creeping debris front while the player is stranded (held behind the car until footTimeLimit). */
  _updateFront(dt) {
    const slide = this.ctx.landslide;
    const F = this.flags, D = this.tune.debrisFront;
    F.footT += dt;
    const cp = this.carProj();
    if (!slide || !cp) return;
    let v = D.speed0 + D.accel * F.footT;
    const fs = slide.frontS || 0;
    if (F.footT < this.tune.footTimeLimit) {
      const cap = cp.s - this.tune.frontCarMargin;
      v *= clamp((cap - fs) / 18, 0.0, 1);
      v = Math.max(v, 0.06); // keeps grinding forward (and stays lethal)
      if (fs > cap) v = 0.06;
    }
    slide.setFrontSpeed?.(v);
    if (fs > 0 && fs > cp.s - 2.5) {
      // the car is buried: no way out
      this._hazard(this.ctx.control === 'car' ? 'car' : 'player', 'front', 20);
      return;
    }
    if (!F.buriedWarned && fs > cp.s - 45 && F.footT > this.tune.footTimeLimit * 0.75) {
      F.buriedWarned = true;
      this._say('The mud is almost at the car. Hurry!', 3);
    }
  }

  _updateOnFoot(dt) {
    const F = this.flags;
    const m = this.markers;
    this._updateFront(dt);
    if (this.state !== 'onfoot') return;
    const pp = this.ctx.control === 'foot' ? this.playerProj() : this.carProj();
    if (pp) {
      if (!F.treeSeen && pp.s > (m.fallenTree ?? 305) - 16 && Math.abs(pp.d) < 12) {
        F.treeSeen = true;
        const ft = this.ctx.vegetation?.fallenTree;
        if (ft && !ft.cut) this._say(this.inventory.has('hatchet') ? 'Time to use that hatchet.' : 'A whole tree across the road. The car will never get past that.', 3.8);
        this._refreshObjective();
      }
      if (!F.roadworksSeen && pp.s > (m.roadworks ?? 395) - 28) {
        F.roadworksSeen = true;
        this._say('Roadworks. Nobody here... but there might be fuel for that generator.', 4);
        this._refreshObjective();
      }
    }
    this._checkFalls();
  }

  /** Escape: rubber-banded chase, gap warnings, stuck detection, win. */
  _updateEscape(dt) {
    const { ctx } = this;
    const F = this.flags, D = this.tune.debrisFront, T = this.tune;
    const m = this.markers;
    const slide = ctx.landslide;
    const cp = this.carProj();
    const pp = ctx.control === 'foot' ? this.playerProj() : null;
    const ref = pp && cp ? Math.min(pp.s, cp.s) : (cp ? cp.s : pp?.s);
    const gap = m.gap ?? 560;
    const bridge = !!ctx.props?.bridge;
    // stuck in the trench (evaluated fresh every frame: reversing out of a dipped wheel un-sticks you)
    const stuck = !!(cp && !bridge && Math.abs(cp.s - gap) < 3.2 && cp.dy < -0.28 && Math.abs(cp.d) < 4.5);
    if (stuck !== F.stuck) { F.stuck = stuck; this._refreshObjective(); }
    if (stuck && !F.stuckSaid) {
      F.stuckSaid = true;
      this._say('No! It is stuck! It is not coming out!', 3);
      ctx.cameraRig?.shake?.(0.3);
    }
    // front
    if (slide && ref != null) {
      const fs = slide.frontS || 0;
      if (!(fs > 0)) slide.setFront?.(ref - D.chaseLag, true);
      const dist = ref - (slide.frontS || 0);
      let v = D.chaseSpeed * clamp((dist - T.frontSlowDist) / Math.max(1, D.chaseLag - T.frontSlowDist), 0, T.frontMaxFactor);
      v = Math.max(v, F.stuck ? T.frontFloorStuck : T.frontFloor);
      if (this.state === 'driving') v = Math.min(v, D.chaseSpeed * 0.7);
      if (cp && cp.s > (m.tunnel ?? 1150) - 5) v = Math.min(v, 3); // the portal: it spills out and stops
      slide.setFrontSpeed?.(v);
      // our own safety net for a car the front has swallowed (landslide only checks while the player is inside)
      if (cp && (slide.frontS || 0) > cp.s + 0.5 && ctx.control === 'foot') { this._hazard('car', 'front', 20); return; }
    }
    // the washout is a breather from the rockfall (the front is the threat there): rockfall pauses while the car
    // waits at the trench and resumes once it is across
    if (cp && slide && F.escapeStarted) {
      const atGap = !bridge ? cp.s > gap - 30 && cp.s < gap + 3 : cp.s < gap + 2 && cp.s > gap - 30;
      if (atGap && !F.rockPaused) { F.rockPaused = true; try { slide.stopEscape?.(); } catch {} }
      else if (!atGap && F.rockPaused && cp.s > gap + this.tune.rockResumeAfterGap && (this.car?.speed ?? 0) > 5) { F.rockPaused = false; try { slide.startEscape?.(); } catch {} }
    }
    // gap warnings
    if (cp && !bridge && !F.gapSeen && cp.s > gap - T.gapWarnDist && cp.s < gap) {
      F.gapSeen = true;
      this._say('The road is gone! Stop, STOP!', 2.6);
      this._after(2.8, () => { if (this.state === 'escape' || this.state === 'driving') this._say('The planks. Lay them across the trench.', 3); });
      this._refreshObjective();
    }
    if (cp && bridge && ctx.control === 'car' && cp.s > gap + 3 && cp.s < gap + 60 && !F.crossed) {
      F.crossed = true;
      this._say('Made it across!', 2);
      this._refreshObjective();
      this.checkpoint = 'gap_after';
    }
    if (cp && !F.tunnelSaid && cp.s > (m.tunnel ?? 1150) - 140) {
      F.tunnelSaid = true;
      this._say('The tunnel! Almost there!', 2.5);
    }
    // periodic objective refresh (depends on position)
    this._objT = (this._objT || 0) - dt;
    if (this._objT <= 0) { this._objT = 0.5; this._refreshObjective(); }
    // win
    const wp = ctx.control === 'foot' ? pp : cp;
    if (wp && wp.s > (m.win ?? 1175) && Math.abs(wp.d) < 9 && wp.dy > -4) { this._win(); return; }
    this._checkFalls();
  }

  _checkFalls() {
    const { ctx } = this;
    if (ctx.control === 'foot') {
      const pp = this.playerProj();
      if (pp && pp.dy < -7 && pp.d < -3.5) { this._hazard('player', 'fall', 5); return; }
    }
    const cp = this.carProj();
    if (cp && ctx.control === 'car' && cp.dy < -6 && cp.d < -3.5) { this._hazard('car', 'fall', 5); return; }
    // (QA) a car a boulder has rolled onto its roof cannot be driven out: without this the player sat upside down
    // for ~30 s until the debris front arrived. Fail after 2.5 s on the roof (car.rolledOver already needs 1.2 s).
    const car = ctx.car;
    const dt = ctx.time?.dt ?? 0;
    if (car && ctx.control === 'car' && car.rolledOver) this._rollT = (this._rollT || 0) + dt; else this._rollT = 0;
    if (this._rollT > 2.5) { this._rollT = 0; this._hazard('car', 'rollover', 5); }
  }

  /** Game-detected hazards go through the bus like the landslide's (audio/hud listen to hazard:hit). */
  _hazard(target, cause, energy) {
    if (!PLAYING.includes(this.state)) return;
    this.ctx.events?.emit?.('hazard:hit', { target, cause, energy });
    if (PLAYING.includes(this.state)) this.fail(cause === 'front' && target === 'car' ? 'car' : cause); // no bus / no listener
  }

  dispose() {
    for (const off of this._offs) { try { off(); } catch {} }
    this._offs = [];
    const ia = this.ctx.interact;
    for (const id of ['pickup_jerrycan', 'pickup_hatchet', 'pickup_planks', 'chop', 'refuel', 'get_in', 'place_planks', 'start_engine', 'get_out']) ia?.unregister?.(id);
    this.ctx.time.scale = 1;
  }
}

// ================================================================================================= checkpoints
// Each returns the situation to restore. carS etc. in road space.
const CHECKPOINTS = {
  intro: (g, { m, tune }) => ({
    state: 'intro', checkpoint: 'intro', carS: m.carStart ?? 40, speed: tune.introStartSpeed, engine: true,
    fuel: tune.fuelStart ?? 0.35, inv: [], frontS: 0,
  }),
  stall: (g, { stallS, tune }) => ({
    state: 'stalled', carS: stallS, engine: false, fuel: 0, inv: [],
    frontS: tune.debrisFront.startS, frontSpeed: tune.debrisFront.speed0, say: 'Out of fuel. I need to find some.',
  }),
  onfoot: (g, { stallS, tune }) => ({
    state: 'onfoot', checkpoint: 'stall', carS: stallS, engine: false, fuel: 0, inv: [], onFoot: true,
    frontS: tune.debrisFront.startS, frontSpeed: tune.debrisFront.speed0,
  }),
  refuel: (g, { stallS, tune }) => ({
    state: 'onfoot', checkpoint: 'stall', carS: stallS, engine: false, fuel: 0, inv: ['jerrycan', 'hatchet', 'planks'],
    onFoot: true, playerAt: 'fuelCap', treeCut: true, roadworksSeen: true, footT: 120,
    frontS: Math.max(tune.debrisFront.startS, stallS - 60), frontSpeed: 0.3,
  }),
  escape: (g, { stallS, tune }) => ({
    state: 'driving', carS: stallS, engine: true, fuel: tune.jerrycanLiters ?? 5, inv: ['hatchet', 'planks'],
    refueled: true, treeCut: true, roadworksSeen: true, startAttempts: 2,
    frontS: stallS - 40, frontSpeed: 2, say: 'Go!',
  }),
  gap: (g, { m, tune }) => ({
    state: 'escape', checkpoint: 'gap', carS: (m.gap ?? 560) - 70, speed: 11, engine: true, fuel: (tune.jerrycanLiters ?? 5) - 1,
    inv: ['hatchet', 'planks'], refueled: true, treeCut: true, roadworksSeen: true, startAttempts: 2, escape: true,
    frontS: (m.gap ?? 560) - 70 - tune.debrisFront.chaseLag, frontSpeed: tune.debrisFront.chaseSpeed,
  }),
  gap_after: (g, { m, tune }) => ({
    state: 'escape', checkpoint: 'gap_after', carS: (m.gap ?? 560) + 8, speed: 6, engine: true, fuel: (tune.jerrycanLiters ?? 5) - 1.5,
    inv: ['hatchet'], bridge: true, refueled: true, treeCut: true, roadworksSeen: true, startAttempts: 2, escape: true,
    frontS: (m.gap ?? 560) + 8 - tune.debrisFront.chaseLag, frontSpeed: tune.debrisFront.chaseSpeed,
  }),
  tunnel: (g, { m, tune }) => ({
    state: 'escape', checkpoint: 'tunnel', carS: (m.tunnel ?? 1150) - 60, speed: 12, engine: true, fuel: 2,
    inv: ['hatchet'], bridge: true, refueled: true, treeCut: true, roadworksSeen: true, startAttempts: 2, escape: true,
    frontS: (m.tunnel ?? 1150) - 60 - tune.debrisFront.chaseLag, frontSpeed: tune.debrisFront.chaseSpeed,
  }),
};

// ================================================================================================= title camera
// Three slow shots behind the title UI (hard cuts): over the drop looking at the scar, low along the wet road toward
// the fallen tree, and a high wide over the valley toward the tunnel spur.
function titlePath(road) {
  const shots = [
    { a: [68, -3.6, 4.4], b: [96, -3.4, 5.0], ta: [150, 16, 20], tb: [158, 14, 17] },
    { a: [262, -2.4, 2.6], b: [286, -2.0, 2.2], ta: [320, 0.5, 1.2], tb: [330, 0.5, 1.0] },
    { a: [640, -70, 42], b: [720, -66, 46], ta: [980, 10, 10], tb: [1060, 10, 10] },
  ];
  const pa = new THREE.Vector3(), pb = new THREE.Vector3();
  return (t, outPos, outTarget) => {
    const n = shots.length;
    const f = clamp(t, 0, 0.99999) * n;
    const i = Math.floor(f), u = f - i;
    const S = shots[i];
    const k = u * u * (3 - 2 * u) * 0.35 + u * 0.65;
    const lerp3 = (a, b) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
    const P = lerp3(S.a, S.b), Tt = lerp3(S.ta, S.tb);
    road.worldAt(P[0], P[1], pa); pa.y = road.pointAt(P[0], pb).y + P[2];
    road.worldAt(Tt[0], Tt[1], outTarget); outTarget.y = road.pointAt(Tt[0], pb).y + Tt[2];
    outPos.copy(pa);
    // slight handheld drift
    outPos.x += Math.sin(t * 40) * 0.15; outPos.y += Math.sin(t * 57 + 1) * 0.1;
    return { fov: 55 };
  };
}
