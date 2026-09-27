// AUDIO workstream: fully procedural Web Audio for LANDSLIDE (no sample files).
//
// API (DESIGN.md "audio"):
//   unlock()                    create/resume the AudioContext (call on a user gesture; also hooked to the first
//                               pointerdown/keydown). Idempotent. Returns a Promise.
//   setMasterVolume(v 0..1)
//   play(name, {position, volume, rate, delay, open, heavy, close}) -> handle {stop(fade)} | null
//       names: chop crack pickup door fuelPour plankDrop thud splash ignition ignitionFail thunder heartbeat
//              (+ rock crash clunk drip keyClick footstep breath)
//   stats                       {ready, voices, rms, peak, tunnel, inCar}
//
// Continuous layers read every frame: ctx.car (rpm, throttle, engineOn, fuelCut, cranking, sputter, speed, gear, surface,
// skid, bump), ctx.landslide (rumble, frontS), ctx.env.rain, ctx.player.stamina, camera position (listener),
// ctx.cameraRig.mode (cockpit muffling). Events: impact, footstep, player:land, player:struck, car:*, item:pickup,
// tree:chop, tree:snap, planks:placed, hazard:hit.
//
// Graph:
//   spatial voice: src -> gain -> air-absorption lowpass -> HRTF panner -> ext bus  (+ send -> reverb)
//   ext bus -> cabin muffle lowpass -> mix;  int bus (cabin: roof rain, engine boom) -> mix;  self bus -> mix
//   reverb: send -> [mountain IR convolver | tunnel IR convolver] -> ext
//   mix -> compressor -> limiter -> master volume -> pause gain -> destination
// Sound banks are rendered by src/audio/dsp.js in a module worker (bank.worker.js) after unlock.
import * as THREE from 'three';
import { flags } from '../core/debug.js';

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const SOUND_SPEED = 343;
const GEARS = [3.67, 2.10, 1.36, 1.00, 0.82], REVERSE = 3.53;
const MAX_SPATIAL = 24;
const FOOT_SURF = ['asphalt', 'gravel', 'mud', 'rock', 'dirt', 'grass', 'wood'];

// bank: [key, generator, sampleRate (0 = context rate), variants, arg]
function bankList(sr) {
  const L = [
    ['irMountain', 'irMountain', 0, 1], ['irTunnel', 'irTunnel', 0, 1],
    ['rain', 'rainLoop', 0, 1], ['roof', 'roofLoop', 0, 1], ['wind', 'windLoop', 32000, 1],
    ['tyreWet', 'tyreWetLoop', 0, 1], ['gravelLoop', 'gravelLoop', 0, 1], ['mudLoop', 'mudLoop', 32000, 1],
    ['keyClick', 'keyClick', 0, 2],
  ];
  for (const s of FOOT_SURF) { L.push([`step_${s}`, 'footstep', 0, 4, { surface: s }]); L.push([`run_${s}`, 'footstep', 0, 3, { surface: s, run: true }]); }
  L.push(
    ['door_open', 'door', 0, 1, { open: true }], ['door_close', 'door', 0, 2],
    ['clunk', 'clunk', 0, 3], ['clunk_soft', 'clunk', 0, 2, { soft: true }],
    ['rock0', 'rockImpact', 0, 4, { size: 0 }], ['rock1', 'rockImpact', 0, 4, { size: 1 }], ['rock2', 'rockImpact', 0, 3, { size: 2 }],
    ['crash', 'carCrash', 0, 2], ['thud', 'thud', 0, 3], ['thud_heavy', 'thud', 0, 2, { heavy: true }],
    ['breath_in', 'breath', 0, 3, { inhale: true }], ['breath_out', 'breath', 0, 3],
    ['heartbeat', 'heartbeat', 0, 1], ['pickup', 'pickup', 0, 2], ['chop', 'chop', 0, 4],
    ['crack', 'crack', 0, 2], ['crack_short', 'crack', 0, 3, { short: true }],
    ['fuelPour', 'fuelPour', 0, 1, { dur: 6 }], ['plankDrop', 'plankDrop', 0, 2], ['splash', 'splash', 0, 2],
    ['drip', 'drip', 0, 4],
    ['rumble', 'rumbleLoop', 16000, 1], ['rockRoll', 'rockRollLoop', 22050, 1], ['mudFlow', 'mudFlowLoop', 22050, 1],
    ['thunder', 'thunder', 22050, 2], ['thunder_close', 'thunder', 32000, 1, { close: true }],
  );
  return L.map(([key, gen, rate, variants, arg]) => ({ key, gen, sr: rate || sr, variants, arg: arg || {} }));
}

export default class Audio {
  constructor(ctx) {
    this.ctx = ctx;
    this.ac = null;
    this.muted = !!flags.mute;
    this.volume = 0.8;
    this.bufs = {};            // key -> AudioBuffer[]
    this.voices = new Set();   // spatial one-shot voices
    this.stats = { ready: 0, total: 0, voices: 0, rms: 0, peak: 0, tunnel: 0, inCar: false, engine: 'none' };
    this._last = {};           // dedupe: name -> time
    this._unsub = [];
    this._pending = new Map();
    this._jobId = 0;
    this._t = 0;
    this._proj = {};
    this._v = new THREE.Vector3(); this._v2 = new THREE.Vector3(); this._q = new THREE.Quaternion();
    this._fwd = new THREE.Vector3(); this._up = new THREE.Vector3();
    this._gust = { g: 0.3, target: 0.3, next: 0 };
    this._breath = { level: 0, next: 0, phase: 0 };
    this._heartNext = 0;
    this._crackNext = 5; this._dripNext = 0; this._clunkCd = 0;
    this._script = null; this._pendingScript = null;
    this._coughUntil = 0;
    this._pour = null;
    this._stepSide = 1;

    // gesture unlock (autostart path has no Start click)
    if (!this.muted) {
      const onGesture = () => { this.unlock(); };
      window.addEventListener('pointerdown', onGesture, { once: true, capture: true });
      window.addEventListener('keydown', onGesture, { once: true, capture: true });
      this._unsub.push(() => { window.removeEventListener('pointerdown', onGesture, { capture: true }); window.removeEventListener('keydown', onGesture, { capture: true }); });
    }
    this._listen();
  }

  // ------------------------------------------------------------------------------------------------ public API
  setMasterVolume(v) {
    this.volume = clamp(+v || 0, 0, 1);
    if (this.master) this.master.gain.setTargetAtTime(this.volume, this.ac.currentTime, 0.05);
  }

  unlock() {
    if (this.muted) return Promise.resolve(false);
    if (this._unlocking) return this._unlocking;
    this._unlocking = (async () => {
      try {
        if (!this.ac) {
          const AC = window.AudioContext || window.webkitAudioContext;
          if (!AC) return false;
          this.ac = new AC({ latencyHint: 'interactive' });
          this._buildGraph();
          this._startBanks();
          await this._initEngine();
        }
        if (this.ac.state !== 'running') await this.ac.resume();
        return true;
      } catch (e) {
        console.error('[audio] unlock failed', e);
        return false;
      }
    })();
    const p = this._unlocking;
    p.then(() => { if (this.ac && this.ac.state !== 'running') this._unlocking = null; }); // allow retry on a later gesture
    return p;
  }

  play(name, opts = {}) {
    if (!this.ac || this.muted) return null;
    try { return this._play(name, opts); } catch (e) { console.error('[audio] play', name, e); return null; }
  }

  dispose() {
    for (const f of this._unsub) { try { f(); } catch {} }
    this._unsub = [];
    try { this.worker?.terminate(); } catch {}
    try { this.ac?.close(); } catch {}
    this.ac = null;
  }

  // ------------------------------------------------------------------------------------------------ graph
  _buildGraph() {
    const ac = this.ac, t = ac.currentTime;
    const G = (v = 1) => { const g = ac.createGain(); g.gain.value = v; return g; };
    this.pauseGain = G(1);
    this.master = G(this.volume);
    this.limiter = ac.createDynamicsCompressor();
    this.limiter.threshold.value = -2.5; this.limiter.knee.value = 0; this.limiter.ratio.value = 20;
    this.limiter.attack.value = 0.002; this.limiter.release.value = 0.12;
    this.comp = ac.createDynamicsCompressor();
    this.comp.threshold.value = -20; this.comp.knee.value = 8; this.comp.ratio.value = 3;
    this.comp.attack.value = 0.008; this.comp.release.value = 0.3;
    this.mix = G(1);
    this.mix.connect(this.comp); this.comp.connect(this.limiter); this.limiter.connect(this.master);
    this.master.connect(this.pauseGain); this.pauseGain.connect(ac.destination);
    this.analyser = ac.createAnalyser(); this.analyser.fftSize = 2048;
    this.limiter.connect(this.analyser);
    this._abuf = new Float32Array(this.analyser.fftSize);
    this._busMeters = null; // per-bus meters (debug only), created after the buses below

    // exterior bus (muffled when sitting in the closed cabin)
    this.ext = G(1);
    this.extLp = ac.createBiquadFilter(); this.extLp.type = 'lowpass'; this.extLp.frequency.value = 20000; this.extLp.Q.value = 0.5;
    this.extGain = G(1);
    this.ext.connect(this.extLp); this.extLp.connect(this.extGain); this.extGain.connect(this.mix);
    this.int = G(0); this.int.connect(this.mix);       // cabin-only layers
    this.self = G(1); this.self.connect(this.mix);     // player body sounds (non-spatial)

    // reverb
    this.send = G(1);
    this.convM = ac.createConvolver(); this.convT = ac.createConvolver();
    this.wetM = G(0.22); this.wetT = G(0);
    this.send.connect(this.convM); this.send.connect(this.convT);
    this.convM.connect(this.wetM); this.convT.connect(this.wetT);
    this.wetM.connect(this.ext); this.wetT.connect(this.ext);
    this.selfSend = G(0.35); this.self.connect(this.selfSend); this.selfSend.connect(this.send);

    // ---- continuous layers (sources attach when their banks arrive)
    // rain outside (stereo bed)
    this.rainLp = ac.createBiquadFilter(); this.rainLp.type = 'lowpass'; this.rainLp.frequency.value = 16000;
    this.rainGain = G(0); this.rainLp.connect(this.rainGain); this.rainGain.connect(this.ext);
    this.rainSend = G(0); this.rainGain.connect(this.rainSend); this.rainSend.connect(this.send); // portal rain echoing in the tunnel
    // tunnel room tone: low air movement / distant traffic-less hum of a long concrete tube
    this.tunLp = ac.createBiquadFilter(); this.tunLp.type = 'lowpass'; this.tunLp.frequency.value = 160; this.tunLp.Q.value = 0.9;
    this.tunGain = G(0); this.tunLp.connect(this.tunGain); this.tunGain.connect(this.ext);
    // roof drumming (cabin)
    this.roofGain = G(0); this.roofGain.connect(this.int);
    // wind: broadband + whistle
    this.windBp = ac.createBiquadFilter(); this.windBp.type = 'bandpass'; this.windBp.frequency.value = 400; this.windBp.Q.value = 0.6;
    this.windGain = G(0); this.windBp.connect(this.windGain); this.windGain.connect(this.ext);
    this.whistleBp = ac.createBiquadFilter(); this.whistleBp.type = 'bandpass'; this.whistleBp.frequency.value = 1300; this.whistleBp.Q.value = 9;
    this.whistleGain = G(0); this.whistleBp.connect(this.whistleGain); this.whistleGain.connect(this.ext);
    // deep landslide rumble (non-spatial, felt)
    this.rumbleLp = ac.createBiquadFilter(); this.rumbleLp.type = 'lowpass'; this.rumbleLp.frequency.value = 900;
    this.rumbleGain = G(0); this.rumbleLp.connect(this.rumbleGain); this.rumbleGain.connect(this.mix);
    // spatial loops: rolling rocks + mud-flow front
    this.rollPan = this._panner(30, 1); this.rollGain = G(0); this.rollGain.connect(this.rollPan); this._outSpatial(this.rollPan, 0.35);
    this.flowPan = this._panner(25, 1); this.flowGain = G(0); this.flowGain.connect(this.flowPan); this._outSpatial(this.flowPan, 0.3);
    // tyres
    this.tyrePan = this._panner(3, 1.2);
    this._outSpatial(this.tyrePan, 0.12);
    this.tyreWetGain = G(0); this.gravelGain = G(0); this.mudGain = G(0);
    this.skidBp = ac.createBiquadFilter(); this.skidBp.type = 'bandpass'; this.skidBp.frequency.value = 850; this.skidBp.Q.value = 5;
    this.skidGain = G(0); this.skidBp.connect(this.skidGain); this.skidGain.connect(this.tyrePan);
    for (const g of [this.tyreWetGain, this.gravelGain, this.mudGain]) g.connect(this.tyrePan);
    this.carWindGain = G(0); this.carWindGain.connect(this.ext);
    if (flags.debug) {
      this._busMeters = {};
      for (const [k, node] of Object.entries({ ext: this.extGain, int: this.int, self: this.self, rumble: this.rumbleGain, wet: this.send })) {
        const an = ac.createAnalyser(); an.fftSize = 1024; node.connect(an); this._busMeters[k] = an;
      }
    }
    void t;
  }

  _panner(ref = 3, rolloff = 1, model = 'HRTF') {
    const p = this.ac.createPanner();
    p.panningModel = model; p.distanceModel = 'inverse';
    p.refDistance = ref; p.rolloffFactor = rolloff; p.maxDistance = 10000;
    return p;
  }
  _outSpatial(node, send = 0.2) {
    node.connect(this.ext);
    if (send > 0) { const g = this.ac.createGain(); g.gain.value = send; node.connect(g); g.connect(this.send); node._send = g; }
  }
  _setPos(p, x, y, z) {
    if (p.positionX) { const t = this.ac.currentTime; p.positionX.setTargetAtTime(x, t, 0.015); p.positionY.setTargetAtTime(y, t, 0.015); p.positionZ.setTargetAtTime(z, t, 0.015); }
    else p.setPosition(x, y, z);
  }

  _loopSource(key, dest, rate = 1) {
    const b = this.bufs[key]?.[0];
    if (!b || !dest) return null;
    const s = this.ac.createBufferSource();
    s.buffer = b; s.loop = true; s.playbackRate.value = rate;
    s.connect(dest);
    s.start(this.ac.currentTime + 0.01, Math.random() * b.duration);
    return s;
  }

  // ------------------------------------------------------------------------------------------------ banks
  _startBanks() {
    const list = bankList(this.ac.sampleRate);
    this.stats.total = list.reduce((a, b) => a + b.variants, 0);
    const jobs = [];
    for (const it of list) for (let v = 0; v < it.variants; v++) jobs.push({ ...it, seed: v + 1, v });
    const onResult = (job, sr, channels) => {
      if (!this.ac || !channels?.length) return;
      const b = this.ac.createBuffer(channels.length, channels[0].length, sr);
      channels.forEach((c, i) => b.copyToChannel(c, i));
      (this.bufs[job.key] ||= [])[job.v] = b;
      this.stats.ready++;
      this._onBank(job.key, b);
    };
    try {
      this.worker = new Worker(new URL('./bank.worker.js', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e) => {
        const { id, sr, channels, error } = e.data;
        const job = this._pending.get(id); this._pending.delete(id);
        if (!job) return;
        if (error) { console.error('[audio] bank', job.key, error); return; }
        onResult(job, sr, channels);
        if (!this._pending.size) { try { this.worker.terminate(); } catch {} this.worker = null; }
      };
      this.worker.onerror = (e) => { console.error('[audio] bank worker', e.message || e); };
      for (const j of jobs) { const id = ++this._jobId; this._pending.set(id, j); this.worker.postMessage({ id, gen: j.gen, sr: j.sr, seed: j.seed, arg: j.arg }); }
    } catch (e) {
      // fallback: render on the main thread, one job per idle slot
      import('./dsp.js').then(({ GENERATORS }) => {
        const next = () => {
          const j = jobs.shift(); if (!j || !this.ac) return;
          try { onResult(j, j.sr, GENERATORS[j.gen](j.sr, j.seed, j.arg)); } catch (err) { console.error('[audio] bank', j.key, err); }
          setTimeout(next, 30);
        };
        next();
      });
    }
  }

  _onBank(key) {
    const ac = this.ac;
    if (key === 'irMountain') this.convM.buffer = this.bufs.irMountain[0];
    else if (key === 'irTunnel') this.convT.buffer = this.bufs.irTunnel[0];
    else if (key === 'rain') this._loopSource('rain', this.rainLp);
    else if (key === 'roof') this._loopSource('roof', this.roofGain);
    else if (key === 'wind') {
      this._loopSource('wind', this.windBp); this._loopSource('wind', this.whistleBp, 0.93); this._loopSource('wind', this.tunLp, 0.6);
      const lp = ac.createBiquadFilter(); lp.type = 'bandpass'; lp.frequency.value = 700; lp.Q.value = 0.5; lp.connect(this.carWindGain);
      this._loopSource('wind', lp, 1.07);
    } else if (key === 'tyreWet') {
      this.tyreWetSrc = this._loopSource('tyreWet', this.tyreWetGain);
      this.skidSrc = this._loopSource('tyreWet', this.skidBp, 0.8);
    } else if (key === 'gravelLoop') this.gravelSrc = this._loopSource('gravelLoop', this.gravelGain);
    else if (key === 'mudLoop') this.mudSrc = this._loopSource('mudLoop', this.mudGain);
    else if (key === 'rumble') this._loopSource('rumble', this.rumbleLp);
    else if (key === 'rockRoll') this._loopSource('rockRoll', this.rollGain);
    else if (key === 'mudFlow') this._loopSource('mudFlow', this.flowGain);
  }

  // ------------------------------------------------------------------------------------------------ engine
  async _initEngine() {
    const ac = this.ac;
    this.engExPan = this._panner(2.2, 1); this._outSpatial(this.engExPan, 0.25);
    this.engBayPan = this._panner(2.2, 1); this._outSpatial(this.engBayPan, 0.15);
    this.engExGain = ac.createGain(); this.engBayGain = ac.createGain();
    this.engExGain.connect(this.engExPan); this.engBayGain.connect(this.engBayPan);
    // cabin boom: structure-borne low end heard inside the car
    this.cabinLp = ac.createBiquadFilter(); this.cabinLp.type = 'lowpass'; this.cabinLp.frequency.value = 380; this.cabinLp.Q.value = 1.2;
    this.cabinGain = ac.createGain(); this.cabinGain.gain.value = 0;
    this.cabinLp.connect(this.cabinGain); this.cabinGain.connect(this.int);
    try {
      await ac.audioWorklet.addModule(new URL('./engine.worklet.js', import.meta.url));
      const node = new AudioWorkletNode(ac, 'landslide-engine', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2] });
      const split = ac.createChannelSplitter(2);
      node.connect(split);
      split.connect(this.engExGain, 0); split.connect(this.engBayGain, 1);
      split.connect(this.cabinLp, 0); split.connect(this.cabinLp, 1);
      this.engine = node;
      this.engP = node.parameters;
      this.stats.engine = 'worklet';
    } catch (e) {
      console.error('[audio] engine worklet unavailable, using oscillator fallback', e);
      // fallback: two detuned sawtooth oscillators at the firing frequency through a low-pass
      const o1 = ac.createOscillator(), o2 = ac.createOscillator(), lp = ac.createBiquadFilter(), g = ac.createGain();
      o1.type = 'sawtooth'; o2.type = 'square'; lp.type = 'lowpass'; lp.frequency.value = 500; g.gain.value = 0;
      o1.connect(lp); o2.connect(lp); lp.connect(g); g.connect(this.engExGain); g.connect(this.cabinLp);
      o1.start(); o2.start();
      this._fallbackEngine = { o1, o2, lp, g };
      this.stats.engine = 'fallback';
    }
  }

  _setEngine(rpm, thr, fire, crank, misfire, cough, whF, whA, gain) {
    const t = this.ac.currentTime;
    if (this.engP) {
      const P = this.engP;
      P.get('rpm').setValueAtTime(clamp(rpm, 0, 9000), t); P.get('throttle').setValueAtTime(clamp(thr, 0, 1), t);
      P.get('fire').setValueAtTime(fire, t); P.get('crank').setValueAtTime(crank, t);
      P.get('misfire').setValueAtTime(clamp(misfire, 0, 1), t); P.get('cough').setValueAtTime(clamp(cough, 0, 1), t);
      P.get('whineFreq').setValueAtTime(clamp(whF, 20, 8000), t); P.get('whineAmp').setValueAtTime(clamp(whA, 0, 1), t);
      P.get('gain').setValueAtTime(clamp(gain, 0, 4), t);
    } else if (this._fallbackEngine) {
      const f = this._fallbackEngine, ff = Math.max(1, rpm / 30);
      f.o1.frequency.setTargetAtTime(ff, t, 0.03); f.o2.frequency.setTargetAtTime(ff * 0.5, t, 0.03);
      f.lp.frequency.setTargetAtTime(300 + 1500 * thr, t, 0.05);
      f.g.gain.setTargetAtTime(fire && rpm > 100 ? gain * (0.08 + 0.12 * thr) : 0, t, 0.03);
    }
  }

  _updateEngine(dt) {
    const car = this.ctx.car, now = this._t;
    // scripted start/fail sequence (play('ignition'|'ignitionFail') when no vehicle drives the starter)
    if (this._pendingScript && now >= this._pendingScript.at) {
      const ps = this._pendingScript; this._pendingScript = null;
      if (!car || !(car.cranking > 0)) this._script = { mode: ps.mode, t: 0, rpm: 0 };
    }
    let rpm = 0, thr = 0, fire = 0, crank = 0, mis = 0, cough = 0, whF = 400, whA = 0, on = false;
    const sc = this._script;
    if (sc) {
      sc.t += dt;
      const T = sc.t;
      if (sc.mode === 'ignition') {
        crank = T < 0.75 ? 1 : 0;
        fire = T > 0.55 ? 1 : 0;
        const tgt = T < 0.55 ? 260 : T < 1.1 ? 1500 : 820 + 700 * Math.exp(-(T - 1.1) * 2.5);
        sc.rpm += (tgt - sc.rpm) * Math.min(1, dt * (T < 0.55 ? 10 : 6));
        rpm = sc.rpm; thr = T > 0.55 && T < 1.0 ? 0.35 : 0.02;
        if (T > 6) this._script = null;
      } else {
        crank = T < 1.3 ? 1 : 0; cough = 1;
        sc.rpm += ((T < 1.3 ? 250 : 0) - sc.rpm) * Math.min(1, dt * 8);
        rpm = sc.rpm;
        if (T > 2.2) this._script = null;
      }
      on = true;
    } else if (car) {
      rpm = +car.rpm || 0;
      const running = !!car.engineOn;
      crank = car.cranking > 0 ? 1 : 0;
      this._crankT = crank ? (this._crankT || 0) + dt : 0;
      thr = running ? clamp(+car.throttle || 0, 0, 1) : 0;
      // a cold carburetted engine needs a few compressions before it catches
      const cut = typeof car.fuelCut === 'boolean' ? car.fuelCut : car._starve > 0; // fuel starvation cuts the ignition
      fire = running && !cut && !(crank && this._crankT < 0.3) ? 1 : 0;
      mis = running && car.sputter > 0 ? 0.04 + 0.3 * car.sputter : 0;
      cough = crank && !running ? 1 : (now < this._coughUntil ? 1 : 0);
      const g = car.gear | 0;
      const ratio = g > 0 ? GEARS[Math.min(g, 5) - 1] : g < 0 ? REVERSE : 0;
      if (ratio) {
        whF = (rpm / 60) / ratio * 29; // transfer-case / diff pinion mesh
        whA = clamp(Math.abs(car.speed || 0) / 8, 0, 1) * (0.25 + 0.75 * thr) * (g < 0 ? 1.8 : g === 1 ? 1.2 : 0.8);
      }
      on = rpm > 5 || crank > 0;
    }
    if (!this.engine && !this._fallbackEngine) return;
    this._setEngine(rpm, thr, fire, crank, mis, cough, whF, whA, on ? 0.5 : 0);
    // positions: exhaust at the rear, bay at the front
    const obj = car?.object;
    if (obj) {
      obj.updateWorldMatrix?.(true, false);
      const ex = this._v.set(0.35, 0.3, -2.0).applyMatrix4(obj.matrixWorld);
      this._setPos(this.engExPan, ex.x, ex.y, ex.z);
      const bay = this._v.set(0, 0.8, 1.4).applyMatrix4(obj.matrixWorld);
      this._setPos(this.engBayPan, bay.x, bay.y, bay.z);
    } else {
      const c = this.ctx.camera.position; // no car: put it right in front of the listener
      this._setPos(this.engExPan, c.x, c.y - 1, c.z); this._setPos(this.engBayPan, c.x, c.y - 1, c.z);
    }
    const t = this.ac.currentTime;
    this.engExGain.gain.setTargetAtTime(1.0, t, 0.05);
    this.engBayGain.gain.setTargetAtTime(this._inCar ? 0.75 : 0.55, t, 0.05);
    this.cabinGain.gain.setTargetAtTime(this._inCar ? 0.7 : 0, t, 0.1);
  }

  // ------------------------------------------------------------------------------------------------ voices
  _pick(key) { const a = this.bufs[key]; if (!a?.length) return null; const list = a.filter(Boolean); return list[Math.floor(Math.random() * list.length)] || null; }

  _posOf(p) {
    if (!p) return null;
    if (p.isObject3D) return p.getWorldPosition(new THREE.Vector3());
    if (typeof p.x === 'number') return p;
    return null;
  }

  /**
   * Start a buffer voice. opts: position (spatial), volume, rate, delay (s), bus ('ext'|'int'|'self'),
   * ref (panner refDistance), send (reverb), loop, pan (self bus stereo pan), soundDelay (apply speed of sound).
   */
  _voice(key, o = {}) {
    const ac = this.ac;
    const b = o.buffer || this._pick(key);
    if (!b) return null;
    const src = ac.createBufferSource(); src.buffer = b;
    const rate = (o.rate ?? 1) * (o.jitter === false ? 1 : 1 + (Math.random() - 0.5) * 0.08);
    src.playbackRate.value = rate;
    if (o.loop) { src.loop = true; src.loopStart = o.loopStart ?? 0; src.loopEnd = o.loopEnd ?? b.duration; }
    const g = ac.createGain(); g.gain.value = o.volume ?? 1;
    src.connect(g);
    let delay = o.delay || 0;
    const pos = this._posOf(o.position);
    const nodes = [src, g];
    let spatial = false;
    if (pos) {
      const cam = this.ctx.camera.position;
      const d = Math.hypot(pos.x - cam.x, pos.y - cam.y, pos.z - cam.z);
      if (d > (o.maxDist ?? 2500)) return null;
      if (o.soundDelay !== false) delay += Math.max(0, d - 15) / SOUND_SPEED;
      const lp = ac.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 0.5;
      lp.frequency.value = clamp(20000 / (1 + d / 70), 350, 20000); // air absorption + ground effect
      const p = this._panner(o.ref ?? 3, o.rolloff ?? 1, d > 80 ? 'equalpower' : 'HRTF');
      if (p.positionX) { p.positionX.value = pos.x; p.positionY.value = pos.y; p.positionZ.value = pos.z; } else p.setPosition(pos.x, pos.y, pos.z);
      g.connect(lp); lp.connect(p);
      p.connect(o.bus === 'int' ? this.int : this.ext);
      const sg = ac.createGain(); sg.gain.value = o.send ?? 0.3; p.connect(sg); sg.connect(this.send);
      nodes.push(lp, p, sg);
      spatial = true;
    } else {
      let out = g;
      if (o.pan) { const sp = ac.createStereoPanner(); sp.pan.value = o.pan; g.connect(sp); out = sp; nodes.push(sp); }
      const bus = o.bus === 'int' ? this.int : o.bus === 'self' ? this.self : o.bus === 'mix' ? this.mix : this.ext;
      out.connect(bus);
      if (o.send && o.bus !== 'self') { const sg = ac.createGain(); sg.gain.value = o.send; out.connect(sg); sg.connect(this.send); nodes.push(sg); }
    }
    const v = { src, g, nodes, spatial, t0: ac.currentTime + delay, key, stopped: false };
    v.stop = (fade = 0.08) => {
      if (v.stopped) return; v.stopped = true;
      try { const t = this.ac.currentTime; g.gain.cancelScheduledValues(t); g.gain.setTargetAtTime(0, t, fade / 3); src.stop(t + fade + 0.05); } catch {}
    };
    src.onended = () => { v.stopped = true; this.voices.delete(v); for (const n of nodes) { try { n.disconnect(); } catch {} } };
    if (spatial) {
      if (this.voices.size >= MAX_SPATIAL) { // steal the oldest
        let old = null; for (const x of this.voices) { if (!old || x.t0 < old.t0) old = x; }
        old?.stop(0.03); this.voices.delete(old);
      }
      this.voices.add(v);
    }
    src.start(ac.currentTime + delay, o.offset || 0);
    return v;
  }

  _dedupe(name, win = 0.12) {
    const now = this.ac.currentTime;
    if (now - (this._last[name] ?? -9) < win) return true;
    this._last[name] = now;
    return false;
  }

  _play(name, o) {
    const vol = o.volume ?? 1;
    const pos = o.position;
    switch (name) {
      case 'chop': if (this._dedupe('chop', 0.15)) return null;
        return this._voice('chop', { position: pos || this._ahead(0.9), volume: 0.9 * vol, ref: 2, send: 0.35 });
      case 'crack': if (this._dedupe('crack', 0.3)) return null;
        return this._voice(o.short ? 'crack_short' : 'crack', { position: pos || this._ahead(3), volume: vol, ref: 6, send: 0.45 });
      case 'pickup': if (this._dedupe('pickup', 0.3)) return null;
        return this._voice('pickup', { bus: 'self', volume: 0.55 * vol });
      case 'door': if (this._dedupe('door', 0.25)) return null;
        return this._voice(o.open ? 'door_open' : 'door_close', { position: pos || this.ctx.car?.doorPoint || null, volume: vol, ref: 2, bus: this._inCar ? 'int' : 'ext', send: 0.25 });
      case 'fuelPour': return this._fuelPour(o);
      case 'plankDrop': if (this._dedupe('plankDrop', 0.3)) return null;
        return this._voice('plankDrop', { position: pos || this._ahead(2), volume: vol, ref: 5, send: 0.4 });
      case 'thud': if (this._dedupe('thud', 0.1)) return null;
        return pos ? this._voice(o.heavy ? 'thud_heavy' : 'thud', { position: pos, volume: vol, ref: 3 }) : this._voice(o.heavy ? 'thud_heavy' : 'thud', { bus: 'self', volume: 0.8 * vol });
      case 'splash': if (this._dedupe('splash', 0.15)) return null;
        return pos ? this._voice('splash', { position: pos, volume: vol, ref: 4 }) : this._voice('splash', { bus: 'self', volume: 0.7 * vol });
      case 'ignition': case 'ignitionFail': {
        if (this._dedupe('keyClick', 0.4)) return null;
        this._pendingScript = { mode: name, at: this._t + 0.15 };
        if (name === 'ignitionFail') this._coughUntil = this._t + 1.4;
        return this._voice('keyClick', { bus: 'int', volume: 0.6 * vol, forceInt: true, ...(this._inCar ? {} : { position: this.ctx.car?.seatCam || null, ref: 1, bus: 'ext' }) });
      }
      case 'keyClick': return this._voice('keyClick', { bus: 'int', volume: 0.6 * vol });
      case 'thunder': {
        const close = o.close ?? vol > 0.85;
        const tun = 1 - 0.85 * this._tunnel;
        return this._voice(close ? 'thunder_close' : 'thunder', { volume: (close ? 1.0 : 0.75) * vol * tun, send: 0.15, jitter: true, rate: 0.9 + Math.random() * 0.2 });
      }
      case 'heartbeat': return this._voice('heartbeat', { bus: 'self', volume: 0.6 * vol, jitter: false });
      case 'breath': return this._voice(o.inhale ? 'breath_in' : 'breath_out', { bus: 'self', volume: 0.4 * vol });
      case 'rock': return this._rockImpact(pos, o.energy ?? 1, o.radius);
      case 'crash': return this._voice('crash', { position: pos, volume: vol, ref: 4 });
      case 'clunk': return this._voice(o.soft ? 'clunk_soft' : 'clunk', { position: pos || this.ctx.car?.object, volume: vol, ref: 2 });
      case 'drip': return this._voice('drip', { position: pos || this._ahead(2), volume: 0.4 * vol, ref: 1.5 });
      case 'footstep': return this._footstep({ surface: o.surface, run: o.run, position: pos });
      default: return null;
    }
  }

  _ahead(dist = 1) {
    const cam = this.ctx.camera;
    const f = this._v2.set(0, 0, -1).applyQuaternion(cam.quaternion);
    return new THREE.Vector3().copy(cam.position).addScaledVector(f, dist).add(new THREE.Vector3(0, -0.4, 0));
  }

  _fuelPour(o) {
    const p = this._pour;
    const now = this._t;
    if (p && !p.v.stopped) {
      p.touches++; p.last = now;
      if (p.touches === 2) { p.v.src.loop = true; p.v.src.loopStart = 1.2; p.v.src.loopEnd = 5.2; }
      return p.v;
    }
    const pos = o.position || this.ctx.car?.fuelCap || this._ahead(0.7);
    const v = this._voice('fuelPour', { position: pos, volume: 0.8 * (o.volume ?? 1), ref: 1.2, jitter: false, send: 0.2 });
    if (!v) return null;
    this._pour = { v, touches: 1, last: now };
    return v;
  }

  _rockImpact(pos, energy = 1, radius) {
    if (!pos) return null;
    const size = energy < 0.12 ? 0 : energy < 2.5 ? 1 : 2;
    const vol = clamp(0.25 + 0.55 * Math.pow(energy, 0.4), 0.2, 1.6);
    const rate = clamp(radius ? 1.25 / Math.pow(Math.max(radius, 0.2), 0.25) : 1, 0.7, 1.4);
    return this._voice('rock' + size, { position: pos, volume: vol, rate, ref: [2, 6, 14][size], send: 0.45, maxDist: 2500 });
  }

  _footstep(p) {
    const s = FOOT_SURF.includes(p?.surface) ? p.surface : 'asphalt';
    const key = (p?.run ? 'run_' : 'step_') + s;
    this._stepSide = -this._stepSide;
    const vol = (p?.run ? 0.75 : 0.55) * (s === 'mud' ? 0.9 : 1);
    return this._voice(key, { bus: 'self', volume: vol, pan: 0.12 * this._stepSide, rate: p?.run ? 1.03 : 1 });
  }

  // ------------------------------------------------------------------------------------------------ events
  _listen() {
    const ev = this.ctx.events;
    if (!ev?.on) return;
    const on = (n, f) => this._unsub.push(ev.on(n, (p) => { if (!this.ac || this.muted) return; try { f(p || {}); } catch (e) { console.error('[audio] event', n, e); } }));
    on('impact', (p) => {
      if (!p.position) return;
      if (p.source === 'car') { if (!this._dedupe('crash', 0.25)) this._voice('crash', { position: p.position, volume: clamp(0.35 + 0.25 * Math.sqrt(p.energy || 0), 0.3, 1.4), ref: 4, bus: this._inCar ? 'int' : 'ext' }); return; }
      // limit dense rockfall to ~14 impacts/s
      const now = this._t;
      this._impWin = (this._impWin || []).filter((x) => now - x < 1);
      if (this._impWin.length > 14 && (p.energy || 0) < 1) return;
      this._impWin.push(now);
      this._rockImpact(p.position, p.energy ?? 1, p.radius);
    });
    on('footstep', (p) => this._footstep(p));
    on('player:land', (p) => {
      const sp = p.speed || 3;
      this._voice(sp > 6 ? 'thud_heavy' : 'thud', { bus: 'self', volume: clamp(0.3 + sp * 0.08, 0.3, 1) });
      this._footstep({ surface: this.ctx.player?.surface, run: true });
    });
    on('player:struck', (p) => this._voice('thud_heavy', { bus: 'self', volume: clamp(0.5 + (p.energy || 0), 0.5, 1.2) }));
    on('hazard:hit', () => this._voice('thud_heavy', { bus: 'self', volume: 1 }));
    on('car:start', () => { if (!this._dedupe('keyClick', 0.4)) this._voice('keyClick', { bus: this._inCar ? 'int' : 'ext', position: this._inCar ? null : this.ctx.car?.object, volume: 0.6, ref: 1.5 }); });
    on('car:startFail', () => {
      this._coughUntil = this._t + 1.4;
      if (!this._dedupe('keyClick', 0.4)) this._voice('keyClick', { bus: this._inCar ? 'int' : 'ext', position: this._inCar ? null : this.ctx.car?.object, volume: 0.6, ref: 1.5 });
    });
    on('car:stall', () => this._voice('clunk_soft', { position: this.ctx.car?.object, volume: 0.5, ref: 2, bus: this._inCar ? 'int' : 'ext' }));
    on('car:gear', () => { if (this._inCar) this._voice('clunk_soft', { bus: 'int', volume: 0.18 }); });
    on('car:rollover', () => this._voice('crash', { position: this.ctx.car?.object, volume: 1.2, ref: 4 }));
    on('car:enter', () => { this._play('door', { open: true, volume: 0.8 }); setTimeout(() => { if (this.ac) { this._last.door = -9; this._play('door', { volume: 1 }); } }, 900); });
    on('car:exit', () => { this._play('door', { open: true, volume: 0.9 }); setTimeout(() => { if (this.ac) { this._last.door = -9; this._play('door', { volume: 0.9 }); } }, 1300); });
    on('item:pickup', () => this._play('pickup', {}));
    on('tree:chop', () => this._play('chop', {}));
    on('tree:snap', (p) => this._voice('crack_short', { position: p.position, volume: 1, ref: 8, send: 0.5 }));
    on('planks:placed', () => {
      let pos = null;
      try { const s = this.ctx.road?.markers?.gap ?? 560; pos = this.ctx.road.worldAt(s, 0, new THREE.Vector3()); } catch {}
      this._play('plankDrop', { position: pos });
    });
    on('ui:start', () => { if (navigator.userActivation?.hasBeenActive) this.unlock(); });
    on('dead', () => { this._muffle = 1; });
    on('game:state', (p) => { if (p.state !== 'dead') this._muffle = 0; });
    on('ui:retry', () => { this._muffle = 0; });
  }

  // ------------------------------------------------------------------------------------------------ per frame
  update(dt) {
    if (!this.ac || this.muted) return;
    const ctx = this.ctx, ac = this.ac, t = ac.currentTime;
    const rdt = ctx.paused ? 0 : Math.min(0.1, dt || 0);
    this._t += rdt;
    // pause / resume
    const paused = !!ctx.paused;
    if (paused !== this._paused) { this._paused = paused; this.pauseGain.gain.setTargetAtTime(paused ? 0 : 1, t, 0.06); }
    if (paused) return;

    // listener from the camera
    const cam = ctx.camera;
    cam.updateMatrixWorld?.();
    const e = cam.matrixWorld.elements;
    const px = e[12], py = e[13], pz = e[14];
    this._fwd.set(-e[8], -e[9], -e[10]).normalize(); this._up.set(e[4], e[5], e[6]).normalize();
    const L = ac.listener;
    if (L.positionX) {
      L.positionX.setTargetAtTime(px, t, 0.01); L.positionY.setTargetAtTime(py, t, 0.01); L.positionZ.setTargetAtTime(pz, t, 0.01);
      L.forwardX.setTargetAtTime(this._fwd.x, t, 0.01); L.forwardY.setTargetAtTime(this._fwd.y, t, 0.01); L.forwardZ.setTargetAtTime(this._fwd.z, t, 0.01);
      L.upX.setTargetAtTime(this._up.x, t, 0.01); L.upY.setTargetAtTime(this._up.y, t, 0.01); L.upZ.setTargetAtTime(this._up.z, t, 0.01);
    } else { L.setPosition(px, py, pz); L.setOrientation(this._fwd.x, this._fwd.y, this._fwd.z, this._up.x, this._up.y, this._up.z); }

    // where is the listener: in the cabin? in the tunnel?
    const car = ctx.car;
    let inCar = ctx.cameraRig?.mode === 'car-cockpit';
    if (!inCar && car?.seatCam && ctx.control === 'car' && ctx.cameraRig?.mode !== 'car-chase') {
      car.seatCam.getWorldPosition(this._v); inCar = this._v.distanceTo(cam.position) < 0.9;
    }
    this._inCar = inCar;
    let tunnel = 0, depth = 0, s = 0, d = 0;
    try {
      const pr = ctx.road?.project(cam.position, this._proj);
      if (pr) {
        s = pr.s; d = pr.d;
        const m = ctx.road.markers || {};
        const t0 = m.tunnel ?? 1150, t1 = m.tunnelEnd ?? 1260;
        tunnel = smooth(t0 - 6, t0 + 8, s) * (1 - smooth(t1 + 2, t1 + 10, s)) * (1 - smooth(6, 10, Math.abs(d))) * (1 - smooth(9, 14, pr.dy));
        depth = Math.max(0, s - t0);
      }
    } catch {}
    this._tunnel = tunnel;
    this.stats.tunnel = +tunnel.toFixed(2); this.stats.inCar = inCar;

    // buses
    const muf = this._muffle ? 1 : 0; // dead: the world drops away behind the death screen
    this.extLp.frequency.setTargetAtTime(muf ? 480 : inCar ? 1300 : 20000, t, muf ? 0.6 : 0.08);
    this.extGain.gain.setTargetAtTime((inCar ? 0.6 : 1) * (muf ? 0.55 : 1), t, muf ? 0.8 : 0.08);
    this.int.gain.setTargetAtTime(inCar ? 1 : 0, t, 0.08);
    this.wetM.gain.setTargetAtTime(0.2 * (1 - tunnel), t, 0.2);
    this.wetT.gain.setTargetAtTime(0.75 * tunnel, t, 0.2);

    // rain (outside / roof / tunnel)
    const rain = clamp(ctx.env?.rain ?? 0.35, 0, 1);
    const leak = 0.08 + 0.3 * Math.exp(-depth / 25); // rain heard through the portal
    const rainOut = (0.25 + 0.75 * rain) * (1 - tunnel * (1 - leak));
    this.rainGain.gain.setTargetAtTime(0.42 * rainOut, t, 0.2);
    this.rainLp.frequency.setTargetAtTime(tunnel > 0.3 ? 1200 + 9000 * Math.exp(-depth / 12) : 16000, t, 0.2);
    this.rainSend.gain.setTargetAtTime(0.9 * tunnel, t, 0.2);
    this.tunGain.gain.setTargetAtTime(0.5 * tunnel, t, 0.3);
    this.roofGain.gain.setTargetAtTime(inCar ? (0.2 + 0.6 * rain) * (1 - tunnel) : 0, t, 0.15);

    // wind with gusts
    const G = this._gust;
    if (this._t > G.next) { G.target = Math.random() < 0.25 ? 0.6 + Math.random() * 0.5 : 0.12 + Math.random() * 0.3; G.next = this._t + 1.5 + Math.random() * 5; }
    G.g += (G.target - G.g) * Math.min(1, rdt * 0.7);
    const shelter = 1 - 0.9 * tunnel;
    this.windGain.gain.setTargetAtTime(0.35 * G.g * shelter, t, 0.3);
    this.windBp.frequency.setTargetAtTime(250 + 500 * G.g, t, 0.3);
    this.whistleGain.gain.setTargetAtTime(0.1 * Math.max(0, G.g - 0.45) * shelter, t, 0.3);
    this.whistleBp.frequency.setTargetAtTime(1000 + 700 * G.g, t, 0.4);

    // engine + tyres
    this._updateEngine(rdt);
    this._updateTyres(rdt, car, inCar);

    // landslide
    this._updateSlide(rdt, cam.position);

    // body: breathing + heartbeat
    this._updateBody(rdt);

    // distant thunder rolling beyond the clouds (no visible flash) when the environment isn't scheduling strikes
    if (!(ctx.env?.autoLightning > 0)) {
      if (!this._thunderNext) this._thunderNext = this._t + 25 + Math.random() * 40;
      if (this._t > this._thunderNext) {
        this._thunderNext = this._t + 50 + Math.random() * 90;
        this._play('thunder', { volume: 0.35 + Math.random() * 0.25, close: false });
      }
    }

    // tunnel ambience: drips
    if (tunnel > 0.5 && this._t > this._dripNext) {
      this._dripNext = this._t + 0.4 + Math.random() * 2.2;
      const p = new THREE.Vector3(px + (Math.random() - 0.5) * 16, py + 2 + Math.random() * 3, pz + (Math.random() - 0.5) * 16);
      this._voice('drip', { position: p, volume: 0.25 + Math.random() * 0.3, ref: 1.5, send: 0.9 });
    }

    // sustained fuel pour ends when interact stops re-triggering it
    if (this._pour && !this._pour.v.stopped && this._pour.touches > 1 && this._t - this._pour.last > 0.7) { this._pour.v.stop(0.25); this._pour = null; } // GAME re-touches every ~0.45 s

    // meters (every 6th frame)
    if (((this._frame = (this._frame || 0) + 1) % 6) === 0) {
      this.analyser.getFloatTimeDomainData(this._abuf);
      let sum = 0, pk = 0; for (const v of this._abuf) { sum += v * v; const a = Math.abs(v); if (a > pk) pk = a; }
      this.stats.rms = +Math.sqrt(sum / this._abuf.length).toFixed(4); this.stats.peak = +pk.toFixed(3);
      this.stats.voices = this.voices.size;
      if (this._busMeters) {
        const b = new Float32Array(1024), out = {};
        for (const [k, an] of Object.entries(this._busMeters)) { an.getFloatTimeDomainData(b); let q = 0; for (const v of b) q += v * v; out[k] = +(20 * Math.log10(Math.sqrt(q / b.length) + 1e-6)).toFixed(1); }
        this.stats.busDb = out;
      }
      if (flags.debug) window.__AUDIO = this.stats;
    }
  }

  _updateTyres(dt, car, inCar) {
    const t = this.ac.currentTime;
    if (!car?.object) { for (const g of [this.tyreWetGain, this.gravelGain, this.mudGain, this.skidGain, this.carWindGain]) g.gain.setTargetAtTime(0, t, 0.1); return; }
    const p = this._v.set(0, 0.25, 0).applyMatrix4(car.object.matrixWorld);
    this._setPos(this.tyrePan, p.x, p.y, p.z);
    const v = Math.abs(car.speed || 0);
    const surf = car.surface || 'asphalt';
    const loose = surf === 'gravel' || surf === 'dirt' || surf === 'rock' ? 1 : 0;
    const mud = surf === 'mud' || surf === 'grass' ? 1 : 0;
    const hard = 1 - loose - mud;
    const k = clamp(v / 20, 0, 1.6);
    const wet = 0.4 + 0.6 * clamp(this.ctx.env?.wetness ?? 0.75, 0, 1);
    this.tyreWetGain.gain.setTargetAtTime(hard * Math.pow(k, 1.3) * 0.55 * wet, t, 0.08);
    if (this.tyreWetSrc) this.tyreWetSrc.playbackRate.setTargetAtTime(0.7 + 0.4 * k, t, 0.1);
    this.gravelGain.gain.setTargetAtTime(loose * clamp(v / 12, 0, 1.3) * 0.6, t, 0.08);
    if (this.gravelSrc) this.gravelSrc.playbackRate.setTargetAtTime(0.55 + 0.5 * clamp(v / 15, 0, 1.4), t, 0.1);
    this.mudGain.gain.setTargetAtTime(mud * clamp(v / 10, 0, 1.3) * 0.55, t, 0.08);
    if (this.mudSrc) this.mudSrc.playbackRate.setTargetAtTime(0.7 + 0.3 * clamp(v / 12, 0, 1.3), t, 0.1);
    const skid = clamp(car.skid || 0, 0, 1) * (hard ? 1 : 0.3);
    this.skidGain.gain.setTargetAtTime(skid * 0.35, t, 0.05);
    this.skidBp.frequency.setTargetAtTime(700 + 300 * skid, t, 0.1);
    this.carWindGain.gain.setTargetAtTime(Math.pow(clamp(v / 30, 0, 1.2), 2) * (inCar ? 0.25 : 0.35), t, 0.1);
    // suspension clunks
    this._clunkCd -= dt;
    if ((car.bump || 0) > 1.1 && this._clunkCd <= 0) {
      this._clunkCd = 0.28;
      this._voice('clunk', { position: car.object, volume: clamp((car.bump - 0.8) * 0.45, 0.15, 1), ref: 2, bus: inCar ? 'int' : 'ext', send: 0.1 });
    }
  }

  _updateSlide(dt, camPos) {
    const t = this.ac.currentTime, ls = this.ctx.landslide, road = this.ctx.road;
    const rumble = clamp(+(ls?.rumble ?? 0) || 0, 0, 1);
    this.rumbleGain.gain.setTargetAtTime(Math.pow(rumble, 1.4) * 1.1, t, 0.15);
    this.rumbleLp.frequency.setTargetAtTime(300 + 900 * rumble, t, 0.2);
    const frontS = +(ls?.frontS ?? 0) || 0;
    if (frontS > 0 && road) {
      try {
        const p = road.worldAt(frontS, 0, this._v2); this._setPos(this.flowPan, p.x, p.y + 1, p.z);
        const q = road.worldAt(frontS - 10, 14, this._v2); this._setPos(this.rollPan, q.x, q.y + 8, q.z);
      } catch {}
      this.flowGain.gain.setTargetAtTime(0.9 * (0.35 + 0.65 * rumble), t, 0.2);
      this.rollGain.gain.setTargetAtTime(0.9 * rumble, t, 0.2);
    } else {
      this.flowGain.gain.setTargetAtTime(0, t, 0.3);
      // rolling rocks still audible from the scar when the slope is moving
      if (rumble > 0 && road) { try { const q = road.worldAt(150, 30, this._v2); this._setPos(this.rollPan, q.x, q.y + 25, q.z); } catch {} }
      this.rollGain.gain.setTargetAtTime(0.8 * rumble, t, 0.3);
    }
    // trees snapping in the moving mass
    if (rumble > 0.3 && this._t > this._crackNext && road) {
      this._crackNext = this._t + (2 + Math.random() * 6) / rumble;
      const s0 = frontS > 0 ? frontS - Math.random() * 30 : 90 + Math.random() * 100;
      try {
        const p = road.worldAt(s0, 12 + Math.random() * 40, new THREE.Vector3()); p.y += 10 + Math.random() * 20;
        this._voice('crack_short', { position: p, volume: 0.8 + Math.random() * 0.4, ref: 10, send: 0.5 });
      } catch {}
    }
  }

  _updateBody(dt) {
    const pl = this.ctx.player, B = this._breath;
    const foot = this.ctx.control === 'foot' || (!this.ctx.car && pl);
    let target = 0;
    if (pl && foot) {
      const st = clamp(pl.stamina ?? 1, 0, 1);
      target = pl.exhausted ? 1 : clamp((0.55 - st) / 0.5, 0, 1);
      if (pl.sprinting) target = Math.max(target, 0.18 + 0.4 * (1 - st));
    }
    B.level += (target - B.level) * Math.min(1, dt * (target > B.level ? 1.5 : 0.25));
    if (B.level > 0.15 && this._t > B.next) {
      const period = 2.4 - 1.3 * B.level;
      const vol = 0.2 + 0.7 * B.level;
      this._voice('breath_in', { bus: 'self', volume: vol * 0.8 });
      const outAt = period * 0.42;
      setTimeout(() => { if (this.ac && !this.ctx.paused) this._voice('breath_out', { bus: 'self', volume: vol }); }, outAt * 1000);
      B.next = this._t + period * (0.9 + Math.random() * 0.2);
    }
    // heartbeat when exhausted and the slide is close
    const rumble = +(this.ctx.landslide?.rumble ?? 0) || 0;
    const hb = B.level > 0.7 ? B.level : (foot && rumble > 0.75 ? 0.5 : 0);
    if (hb > 0 && this._t > this._heartNext) {
      this._heartNext = this._t + 60 / (95 + 50 * hb);
      this._voice('heartbeat', { bus: 'self', volume: 0.35 * hb, jitter: false });
    }
  }
}
