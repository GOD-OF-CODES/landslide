// AUDIO workstream: AudioWorkletProcessor that synthesises an old carburetted 4-cylinder petrol engine.
//
// Physical model (cheap, per sample):
//   crank angle -> one combustion event every 180 deg (4-stroke inline-4) -> per-cylinder exhaust blowdown pulse
//   (+ gas turbulence noise) -> exhaust system: header waveguide + main pipe waveguide (open-end reflections) ->
//   muffler cavity resonators -> tail-pipe low-pass (brighter under load) -> soft saturation.
//   A second channel carries the engine bay: intake roar (throttle), valve-train ticks, block radiation,
//   starter motor (commutator buzz + pinion mesh, loaded by each compression stroke) and gearbox / transfer-case whine.
// Output: 2 channels. ch0 = exhaust (tail pipe, rear of the car), ch1 = engine bay (front of the car).
//
// AudioParams (k-rate, smoothed inside):
//   rpm        crankshaft rpm (from the vehicle)               throttle 0..1
//   fire       1 = ignition + fuel available, 0 = no combustion (stall, starved, no fuel)
//   crank      1 = starter motor engaged
//   misfire    0..1 probability that a firing event misfires (fuel starvation sputter)
//   cough      0..1 probability of a weak catch while cranking with fire=0 (failed start: cranks, coughs, dies)
//   whineFreq  Hz, whineAmp 0..1   (gear / transfer-case whine)
//   gain       output gain
const TAU = Math.PI * 2;

class SVF { // Chamberlin state-variable filter (band / low outputs)
  constructor() { this.low = 0; this.band = 0; this.f = 0.1; this.q = 1; }
  set(fc, Q, sr) { this.f = 2 * Math.sin(Math.PI * Math.min(fc, sr / 6.5) / sr); this.q = 1 / Q; }
  tick(x) {
    this.low += this.f * this.band;
    const high = x - this.low - this.q * this.band;
    this.band += this.f * high;
    return this.band;
  }
}

class Delay {
  constructor(n) { this.buf = new Float32Array(n); this.i = 0; this.n = n; }
  read(d) { let j = this.i - d; if (j < 0) j += this.n; return this.buf[j]; }
  write(x) { this.buf[this.i] = x; this.i = (this.i + 1) % this.n; }
}

class EngineProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    const k = 'k-rate';
    return [
      { name: 'rpm', defaultValue: 0, minValue: 0, maxValue: 9000, automationRate: k },
      { name: 'throttle', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: k },
      { name: 'fire', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: k },
      { name: 'crank', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: k },
      { name: 'misfire', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: k },
      { name: 'cough', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: k },
      { name: 'whineFreq', defaultValue: 400, minValue: 20, maxValue: 8000, automationRate: k },
      { name: 'whineAmp', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: k },
      { name: 'gain', defaultValue: 1, minValue: 0, maxValue: 4, automationRate: k },
    ];
  }

  constructor() {
    super();
    const sr = this.sr = sampleRate;
    this.seed = 0x1234567;
    this.ph = 0; this.ev = 0;             // crank angle (revolutions), last event index
    this.rpmS = 0; this.thrS = 0; this.crankS = 0; this.whS = 0; this.gainS = 1; this.whF = 400;
    this.pulseI = 1e9; this.pulseW = 100; this.pulseA = 0; this.pulseN = 0; this.pulseBright = 0;
    this.pendingDelay = -1; this.pendingW = 100; this.pendingA = 0; this.pendingN = 0; this.pendingBright = 0;
    this.skipped = 0;
    this.cylGain = [1.0, 0.84, 0.95, 0.88];
    this.cylDelay = [0, 0.00035, 0.0001, 0.0006]; // s (worn timing/valves -> lumpy idle)
    // exhaust: header (~0.7 m) and main pipe (~2.6 m) waveguides with hot-gas speed of sound ~ 500 m/s
    this.hdr = new Delay(Math.ceil(sr * 0.01)); this.hdrD = Math.round(sr * 2 * 0.7 / 510);
    this.pipe = new Delay(Math.ceil(sr * 0.03)); this.pipeD = Math.round(sr * 2 * 2.6 / 500);
    this.hdrLp = 0; this.pipeLp = 0;
    this.muf = [new SVF(), new SVF(), new SVF(), new SVF()];
    this.mufSpec = [[88, 2.5, 0.9], [176, 3.5, 0.55], [310, 4, 0.35], [560, 3, 0.18]];
    for (let i = 0; i < 4; i++) this.muf[i].set(this.mufSpec[i][0], this.mufSpec[i][1], sr);
    this.tailLp1 = 0; this.tailLp2 = 0;
    // intake / bay
    this.intake = new SVF(); this.intake2 = new SVF(); this.valve = new SVF(); this.valve.set(2100, 5, sr);
    this.blockLp = 0; this.blockBp = new SVF(); this.blockBp.set(240, 1.5, sr);
    this.valveEv = 0;
    // starter
    this.stPh = 0; this.stPh2 = 0; this.stBuzz = new SVF(); this.stLp = 0;
    this.whPh = 0; this.whPh2 = 0;
    // dc blockers
    this.dc0x = 0; this.dc0y = 0; this.dc1x = 0; this.dc1y = 0;
    this.bayLp1 = 0; this.bayLp2 = 0;
    this.crackle = 0; this.crackBp = new SVF(); this.crackBp.set(2400, 1.2, sr);
  }

  rnd() { // xorshift32 -> [0,1)
    let x = this.seed; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this.seed = x >>> 0;
    return this.seed / 4294967296;
  }

  process(inputs, outputs, P) {
    const out = outputs[0];
    const o0 = out[0], o1 = out[1] || out[0];
    const n = o0.length, sr = this.sr;
    const rpmT = P.rpm[0], thrT = P.throttle[0], fire = P.fire[0] > 0.5, crankT = P.crank[0];
    const misfire = P.misfire[0], cough = P.cough[0], whF = P.whineFreq[0], whA = P.whineAmp[0], gainT = P.gain[0];
    const a = 1 - Math.exp(-1 / (0.03 * sr)); // ~30 ms parameter smoothing (params arrive at frame rate)
    // block-rate filter updates
    const load = this.thrS;
    const rpmN = Math.min(1, this.rpmS / 6000);
    this.intake.set(380 + this.rpmS * 0.09, 2.2, sr);
    this.intake2.set(1500 + this.rpmS * 0.25, 3, sr);
    const tailFc = 750 + 2200 * load + 900 * rpmN;
    const kTail = 1 - Math.exp(-TAU * tailFc / sr);
    const kHdr = 1 - Math.exp(-TAU * 3200 / sr), kPipe = 1 - Math.exp(-TAU * 1700 / sr);
    const kBlock = 1 - Math.exp(-TAU * 700 / sr);
    const kSt = 1 - Math.exp(-TAU * 900 / sr);
    const kBay = 1 - Math.exp(-TAU * (2200 + 1800 * load) / sr);
    const overrun = this.crankS < 0.05 && thrT < 0.05 && this.rpmS > 1400;

    for (let i = 0; i < n; i++) {
      this.rpmS += (rpmT - this.rpmS) * a;
      this.thrS += (thrT - this.thrS) * a;
      this.crankS += (crankT - this.crankS) * a;
      this.whS += (whA - this.whS) * a;
      this.whF += (whF - this.whF) * a;
      this.gainS += (gainT - this.gainS) * a;
      let rpm = this.rpmS;
      // while cranking, each compression stroke slows the crank (chugga-chugga)
      const comp = Math.cos(TAU * 2 * this.ph);
      if (this.crankS > 0.05) rpm *= 1 + 0.28 * this.crankS * comp;
      const revPerS = rpm / 60;
      this.ph += revPerS / sr;
      if (this.ph > 1e6) this.ph -= 1e6;

      // ---------------------------------------------------------------- combustion events (every 0.5 rev)
      const evIdx = Math.floor(this.ph * 2);
      if (evIdx !== this.ev && rpm > 30) {
        this.ev = evIdx;
        const cyl = ((evIdx % 4) + 4) % 4;
        const interval = 0.5 / Math.max(revPerS, 1); // s between events
        let A, N, bright = 0;
        let fires = fire && this.rnd() >= misfire;
        let weakCatch = false;
        if (!fire && this.crankS > 0.3 && cough > 0 && this.rnd() < cough * 0.3) { fires = true; weakCatch = true; }
        if (fires) {
          const idleLump = 1 + (this.rnd() - 0.5) * (0.18 + 0.2 * (1 - rpmN)); // cycle-to-cycle variation
          A = (overrun ? 0.26 : 0.34 + 0.66 * load) * this.cylGain[cyl] * idleLump;
          N = 0.25 + 0.55 * load;
          if (this.skipped >= 2 && this.rnd() < 0.55) { A *= 2.0; bright = 1; } // cough / pop after a misfire run
          if (weakCatch) { A = 0.55 + this.rnd() * 0.4; bright = 0.6; }
          if (overrun && this.rnd() < 0.02) { this.crackle = 0.5 + this.rnd() * 0.6; }
          this.skipped = 0;
        } else {
          A = 0.1 * this.cylGain[cyl]; N = 0.1; // pumping: compression released through the exhaust valve
          this.skipped++;
        }
        this.pendingDelay = Math.round((this.cylDelay[cyl] + this.rnd() * 0.0004) * sr);
        this.pendingA = A; this.pendingN = N; this.pendingBright = bright;
        this.pendingW = Math.max(0.0011, Math.min(0.0055, interval * 0.28)) * sr;
      }
      if (this.pendingDelay >= 0) {
        if (this.pendingDelay-- === 0) {
          this.pulseI = 0; this.pulseW = this.pendingW; this.pulseA = this.pendingA; this.pulseN = this.pendingN;
          this.pulseBright = this.pendingBright;
        }
      }
      // exhaust blowdown pulse: fast rise, decay, rarefaction undershoot
      let ex = 0, env = 0;
      if (this.pulseI < this.pulseW * 1.8) {
        const x = this.pulseI / this.pulseW;
        if (x < 1) { env = 0.5 - 0.5 * Math.cos(TAU * Math.min(1, x * 1.6) * 0.5); env *= Math.exp(-x * 2.2); }
        const under = x > 0.55 ? -0.28 * Math.sin(Math.PI * Math.min(1, (x - 0.55) / 1.25)) : 0;
        ex = this.pulseA * (env + under);
        ex += this.pulseA * this.pulseN * env * (this.rnd() * 2 - 1) * 1.1;
        if (this.pulseBright > 0) ex += this.pulseA * this.pulseBright * env * (this.rnd() * 2 - 1) * 1.6;
        this.pulseI++;
      }
      // over-run crackle (unburnt mixture popping in the hot exhaust)
      if (this.crackle > 0.001) {
        ex += this.crackBp.tick((this.rnd() * 2 - 1) * this.crackle) * 1.5;
        this.crackle *= 0.9985;
      }

      // ---------------------------------------------------------------- exhaust system
      const hFb = this.hdr.read(this.hdrD);
      this.hdrLp += (hFb - this.hdrLp) * kHdr;
      const h = ex + 0.32 * this.hdrLp;
      this.hdr.write(h);
      const pFb = this.pipe.read(this.pipeD);
      this.pipeLp += (pFb - this.pipeLp) * kPipe;
      const p = h - 0.55 * this.pipeLp;           // open end: inverted reflection
      this.pipe.write(p);
      let m = p * 0.35;
      for (let k = 0; k < 4; k++) m += this.muf[k].tick(p) * this.mufSpec[k][2];
      this.tailLp1 += (m - this.tailLp1) * kTail;
      this.tailLp2 += (this.tailLp1 - this.tailLp2) * kTail;
      let yEx = Math.tanh(this.tailLp2 * 1.6) * 0.8;

      // ---------------------------------------------------------------- engine bay
      // intake: roar modulated by induction strokes, stronger with throttle
      const indPh = (this.ph * 2) % 1;
      const indEnv = Math.max(0, Math.sin(Math.PI * indPh));
      const wn = this.rnd() * 2 - 1;
      const intake = (this.intake.tick(wn) * 0.7 + this.intake2.tick(wn) * 0.08) * (0.05 + 0.6 * this.thrS) * (0.25 + 0.75 * indEnv) * (0.2 + rpmN);
      // block radiation: combustion knock through the block (lowpassed pulses + a body resonance)
      this.blockLp += (ex - this.blockLp) * kBlock;
      const block = this.blockLp * 0.45 + this.blockBp.tick(ex) * 0.25;
      // valve train: 8 valve events per 2 revs -> tick every quarter rev
      const vIdx = Math.floor(this.ph * 4);
      let vt = 0;
      if (vIdx !== this.valveEv) { this.valveEv = vIdx; vt = (0.4 + this.rnd() * 0.6) * (0.02 + 0.03 * rpmN) * (rpm > 50 ? 1 : 0); }
      const valve = this.valve.tick(vt) * 0.9;
      // starter motor (pinion 9 teeth into a ~110-tooth ring gear -> ~12x crank speed)
      let st = 0;
      if (this.crankS > 0.01) {
        const load2 = 0.5 + 0.5 * comp; // compression loading
        const fSt = Math.max(5, (this.rpmS / 60) * 12 * (1 - 0.1 * load2));
        this.stPh += fSt / sr; if (this.stPh > 1) this.stPh -= 1;
        this.stPh2 += fSt * 9 / sr; if (this.stPh2 > 1) this.stPh2 -= 1;
        const saw = this.stPh * 2 - 1;
        this.stLp += (saw - this.stLp) * kSt;
        this.stBuzz.set(fSt * 12, 6, sr);
        const buzz = this.stBuzz.tick(this.rnd() * 2 - 1) * 1.6;
        const mesh = Math.sin(TAU * this.stPh2);
        st = (this.stLp * 0.35 + buzz * 0.5 + mesh * 0.12) * this.crankS * (0.55 + 0.45 * load2);
      }
      // gearbox / transfer-case whine
      this.whPh += this.whF / sr; if (this.whPh > 1) this.whPh -= 1;
      this.whPh2 += this.whF * 2.02 / sr; if (this.whPh2 > 1) this.whPh2 -= 1;
      const whine = (Math.sin(TAU * this.whPh) + 0.35 * Math.sin(TAU * this.whPh2)) * this.whS * 0.05;
      let yBay = intake * 0.9 + block + valve + st * 0.55;
      this.bayLp1 += (yBay - this.bayLp1) * kBay; this.bayLp2 += (this.bayLp1 - this.bayLp2) * kBay;
      yBay = this.bayLp2 + whine;

      // dc blockers + output gain
      const g = this.gainS;
      this.dc0y = yEx - this.dc0x + 0.9985 * this.dc0y; this.dc0x = yEx;
      this.dc1y = yBay - this.dc1x + 0.9985 * this.dc1y; this.dc1x = yBay;
      let v0 = this.dc0y * g, v1 = this.dc1y * g;
      if (!(v0 === v0)) { v0 = 0; this.reset(); }
      if (!(v1 === v1)) { v1 = 0; this.reset(); }
      o0[i] = v0;
      if (o1 !== o0) o1[i] = v1;
    }
    return true;
  }

  reset() {
    this.hdr.buf.fill(0); this.pipe.buf.fill(0);
    this.hdrLp = this.pipeLp = this.tailLp1 = this.tailLp2 = this.blockLp = this.stLp = 0;
    for (const f of [...this.muf, this.intake, this.intake2, this.valve, this.blockBp, this.stBuzz, this.crackBp]) { f.low = 0; f.band = 0; }
    this.dc0x = this.dc0y = this.dc1x = this.dc1y = 0;
  }
}

registerProcessor('landslide-engine', EngineProcessor);
