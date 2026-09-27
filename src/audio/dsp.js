// AUDIO workstream: pure procedural sound generators (no AudioContext dependency).
// Every generator is deterministic: gen(sr, seed, opts) -> Float32Array[] (1 or 2 channels).
// They are used by audio.js (pre-rendered banks, generated lazily) and by scratch/audio/test.html.
//
// Techniques: modal synthesis (sums of exponentially damped sinusoids with inharmonic ratios) for solid impacts,
// Minnaert bubble chirps for water, granular micro-impacts for gravel / rain / debris, stochastic N-wave trains for
// thunder, filtered noise with physical envelopes for everything else. Loops are made seamless by wrapping event times
// and running recursive filters twice over a periodic input (so the filter state at the loop point is continuous).

const TAU = Math.PI * 2;

// ------------------------------------------------------------------------------------------------ basics
export function makeRng(seed = 1) {
  let a = (seed * 2654435761) >>> 0 || 1;
  const r = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  r.range = (lo, hi) => lo + (hi - lo) * r();
  r.log = (lo, hi) => lo * Math.pow(hi / lo, r());
  r.gauss = () => { let s = 0; for (let i = 0; i < 4; i++) s += r(); return (s - 2) * 1.732; };
  r.expo = (mean) => -Math.log(1 - r() * 0.999999) * mean;
  r.pick = (arr) => arr[Math.floor(r() * arr.length)];
  return r;
}

/** RBJ biquad (direct form I). type: lp | hp | bp | peak | lowshelf | highshelf | notch */
export class Biquad {
  constructor(type, f, q, sr, gainDb = 0) { this.x1 = this.x2 = this.y1 = this.y2 = 0; this.set(type, f, q, sr, gainDb); }
  set(type, f, q, sr, gainDb = 0) {
    f = Math.min(Math.max(f, 5), sr * 0.49);
    const w = TAU * f / sr, cw = Math.cos(w), sw = Math.sin(w), al = sw / (2 * q), A = Math.pow(10, gainDb / 40);
    let b0, b1, b2, a0, a1, a2;
    switch (type) {
      case 'hp': b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = b0; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al; break;
      case 'bp': b0 = al; b1 = 0; b2 = -al; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al; break;
      case 'notch': b0 = 1; b1 = -2 * cw; b2 = 1; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al; break;
      case 'peak': b0 = 1 + al * A; b1 = -2 * cw; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * cw; a2 = 1 - al / A; break;
      case 'lowshelf': { const s = 2 * Math.sqrt(A) * al;
        b0 = A * ((A + 1) - (A - 1) * cw + s); b1 = 2 * A * ((A - 1) - (A + 1) * cw); b2 = A * ((A + 1) - (A - 1) * cw - s);
        a0 = (A + 1) + (A - 1) * cw + s; a1 = -2 * ((A - 1) + (A + 1) * cw); a2 = (A + 1) + (A - 1) * cw - s; break; }
      case 'highshelf': { const s = 2 * Math.sqrt(A) * al;
        b0 = A * ((A + 1) + (A - 1) * cw + s); b1 = -2 * A * ((A - 1) + (A + 1) * cw); b2 = A * ((A + 1) + (A - 1) * cw - s);
        a0 = (A + 1) - (A - 1) * cw + s; a1 = 2 * ((A - 1) - (A + 1) * cw); a2 = (A + 1) - (A - 1) * cw - s; break; }
      default: b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = b0; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al; // lp
    }
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0; this.a1 = a1 / a0; this.a2 = a2 / a0;
    return this;
  }
  tick(x) {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1; this.x1 = x; this.y2 = this.y1; this.y1 = y;
    return y;
  }
  run(a, from = 0, to = a.length) { for (let i = from; i < to; i++) a[i] = this.tick(a[i]); return a; }
}

/** Apply a filter chain in place (fresh state). */
export function filt(a, sr, specs) {
  for (const [type, f, q = 0.707, g = 0] of specs) new Biquad(type, f, q, sr, g).run(a);
  return a;
}
/** Same as filt but loop-seamless: runs each filter twice over the periodic signal and keeps the 2nd pass. */
export function filtLoop(a, sr, specs) {
  for (const [type, f, q = 0.707, g = 0] of specs) {
    const b = new Biquad(type, f, q, sr, g);
    const tmp = a.slice();
    b.run(tmp);
    b.run(a);
  }
  return a;
}

/** Damped sinusoid (mode). tau = 1/e decay time in s. glide: relative frequency change over tau (negative = pitch drop). */
export function addMode(buf, sr, t0, f, tau, amp, phase = 0, glide = 0, attack = 0) {
  const i0 = Math.max(0, Math.round(t0 * sr));
  const n = Math.min(buf.length - i0, Math.ceil(tau * 7 * sr));
  if (n <= 0 || f >= sr * 0.48) return;
  let ph = phase;
  const k = Math.exp(-1 / (tau * sr));
  let env = amp;
  const atk = Math.max(1, attack * sr);
  for (let i = 0; i < n; i++) {
    const fi = glide ? f * (1 + glide * (1 - Math.exp(-i / (tau * sr)))) : f;
    ph += TAU * fi / sr;
    const a = i < atk ? i / atk : 1;
    buf[i0 + i] += env * a * Math.sin(ph);
    env *= k;
  }
}

/** Minnaert bubble: damped sinusoid with a rising chirp (bubble shrinking / surface rise). */
export function addBubble(buf, sr, t0, f, tau, amp, rise = 0.5) {
  addMode(buf, sr, t0, f, tau, amp, 0, rise, 0.0004);
}

/**
 * Filtered-noise burst. env: 'exp' (instant attack, exp decay tau), or {a, d} attack (s) / decay tau (s).
 * filter: array of biquad specs, applied to the burst only.
 */
export function addNoise(buf, sr, t0, tau, amp, rng, filter = null, attack = 0, len = 0) {
  const i0 = Math.max(0, Math.round(t0 * sr));
  const n = Math.min(buf.length - i0, Math.ceil((len || (attack + tau * 7)) * sr));
  if (n <= 0) return;
  const tmp = new Float32Array(n);
  const k = Math.exp(-1 / (tau * sr)), atk = Math.max(1, attack * sr);
  let env = 1;
  for (let i = 0; i < n; i++) {
    let e;
    if (i < atk) e = i / atk; else { e = env; env *= k; }
    tmp[i] = (rng() * 2 - 1) * e;
  }
  if (filter) filt(tmp, sr, filter);
  for (let i = 0; i < n; i++) buf[i0 + i] += tmp[i] * amp;
}

/** Tiny micro-impact (grain of gravel / drop / debris): click + short resonance. */
export function addGrain(buf, sr, t0, f, tau, amp, rng) {
  const i0 = Math.round(t0 * sr);
  if (i0 < 0 || i0 >= buf.length) return;
  addMode(buf, sr, t0, f, tau, amp, rng() * TAU);
  buf[i0] += amp * (rng() - 0.5) * 0.8;
}

export function peakOf(chs) { let p = 0; for (const c of chs) for (let i = 0; i < c.length; i++) { const v = Math.abs(c[i]); if (v > p) p = v; } return p; }
export function normalize(chs, peak = 0.9) {
  const p = peakOf(chs); if (p < 1e-9) return chs;
  const g = peak / p; for (const c of chs) for (let i = 0; i < c.length; i++) c[i] *= g;
  return chs;
}
function fadeEdges(a, sr, fin = 0.002, fout = 0.02) {
  const ni = Math.floor(fin * sr), no = Math.floor(fout * sr);
  for (let i = 0; i < ni && i < a.length; i++) a[i] *= i / ni;
  for (let i = 0; i < no && i < a.length; i++) a[a.length - 1 - i] *= i / no;
  return a;
}
function softClip(a, drive = 1) { for (let i = 0; i < a.length; i++) a[i] = Math.tanh(a[i] * drive) / Math.tanh(drive); return a; }
const buf = (sr, sec) => new Float32Array(Math.max(1, Math.ceil(sr * sec)));

// ------------------------------------------------------------------------------------------------ noise loops
/** Stereo noise loop. color: white | pink | brown */
export function noiseLoop(sr, seed, color = 'white', sec = 4) {
  const r = makeRng(seed);
  const out = [];
  for (let c = 0; c < 2; c++) {
    const a = buf(sr, sec);
    for (let i = 0; i < a.length; i++) a[i] = r() * 2 - 1;
    if (color === 'pink') pinkify(a);
    else if (color === 'brown') brownify(a, sr);
    out.push(a);
  }
  return normalize(out, 0.7);
}
function pinkify(a) {
  // Paul Kellet's refined pink filter, run twice for a periodic steady state
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  const src = a.slice();
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < a.length; i++) {
      const w = src[i];
      b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.969 * b2 + w * 0.153852;
      b3 = 0.8665 * b3 + w * 0.3104856; b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898;
      a[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
      b6 = w * 0.115926;
    }
  }
}
function brownify(a, sr) {
  const src = a.slice();
  let y = 0; const leak = Math.exp(-TAU * 12 / sr);
  for (let pass = 0; pass < 2; pass++) for (let i = 0; i < a.length; i++) { y = y * leak + src[i] * 0.02; a[i] = y; }
}
// ------------------------------------------------------------------------------------------------ footsteps
/** surface: asphalt | gravel | mud | rock | dirt | grass | wood */
export function footstep(sr, seed, surface = 'asphalt', run = false) {
  const r = makeRng(seed * 7 + 3);
  const a = buf(sr, 0.5);
  const heel = 0.004;
  const toe = heel + (run ? r.range(0.045, 0.065) : r.range(0.075, 0.11));
  const v = run ? 1.25 : 1;
  const thump = (t, amp, fc = 500, tau = 0.012) => addNoise(a, sr, t, tau, amp * v, r, [['lp', fc, 0.8], ['lp', fc * 1.2, 0.7]], 0.0015);
  switch (surface) {
    case 'gravel': {
      thump(heel, 0.9, 420, 0.014);
      const grains = (t0, n, spread, gain) => {
        for (let i = 0; i < n; i++) {
          const t = t0 + r.expo(spread * 0.4) + r() * 0.004;
          addGrain(a, sr, t, r.log(900, 7000), r.range(0.0004, 0.0025), gain * r.expo(0.12) * v, r);
        }
      };
      grains(heel, run ? 150 : 110, 0.06, 1.0);
      grains(toe, run ? 90 : 70, 0.07, 0.8);
      addNoise(a, sr, heel, 0.03, 0.12 * v, r, [['bp', 2600, 0.6]], 0.004);
      addNoise(a, sr, toe, 0.04, 0.09 * v, r, [['bp', 3200, 0.6]], 0.006);
      break;
    }
    case 'mud': {
      thump(heel, 1.0, 260, 0.02);
      // squelch: resonant noise sweeping upward as the boot sinks, then a suction pop as it lifts
      const n = Math.floor(0.22 * sr), i0 = Math.floor(0.01 * sr);
      const bq = new Biquad('bp', 250, 3, sr);
      for (let i = 0; i < n && i0 + i < a.length; i++) {
        if ((i & 31) === 0) bq.set('bp', 240 + 520 * (i / n), 3.2, sr);
        const t = i / sr;
        const e = Math.min(1, t / 0.025) * Math.exp(-t / 0.08);
        a[i0 + i] += bq.tick(r() * 2 - 1) * e * 0.9 * v;
      }
      addBubble(a, sr, r.range(0.14, 0.2), r.range(220, 380), 0.011, 0.2 * v, 0.35); // suction pop
      for (let i = 0; i < 6; i++) addBubble(a, sr, r.range(0.02, 0.2), r.log(500, 1400), r.range(0.006, 0.015), r.range(0.03, 0.1) * v, 0.6);
      addNoise(a, sr, 0.16, 0.03, 0.12 * v, r, [['hp', 1500], ['lp', 5000]], 0.01);
      break;
    }
    case 'rock': {
      addNoise(a, sr, heel, 0.0015, 0.9 * v, r, [['hp', 1800]]);
      for (const f of [1700, 2650, 4100]) addMode(a, sr, heel, f * r.range(0.85, 1.15), r.range(0.003, 0.008), 0.18 * v, r() * TAU);
      thump(heel, 0.7, 420, 0.01);
      for (let i = 0; i < 18; i++) addGrain(a, sr, heel + r.expo(0.02), r.log(1500, 6000), 0.001, 0.12 * r() * v, r);
      addNoise(a, sr, toe, 0.0012, 0.45 * v, r, [['hp', 2000]]);
      thump(toe, 0.4, 500, 0.008);
      addNoise(a, sr, toe + 0.005, 0.025, 0.05 * v, r, [['bp', 3500, 0.7]], 0.005); // scrape
      break;
    }
    case 'wood': {
      for (const [f, tau, g] of [[165, 0.05, 0.5], [390, 0.035, 0.35], [830, 0.02, 0.22], [1550, 0.01, 0.12]]) addMode(a, sr, heel, f * r.range(0.93, 1.07), tau, g * v, r() * TAU);
      addNoise(a, sr, heel, 0.002, 0.4 * v, r, [['hp', 1200]]);
      for (const [f, tau, g] of [[165, 0.04, 0.3], [390, 0.03, 0.2], [830, 0.015, 0.12]]) addMode(a, sr, toe, f * r.range(0.93, 1.07), tau, g * v, r() * TAU);
      break;
    }
    case 'dirt': case 'grass': {
      thump(heel, 0.85, 330, 0.016);
      thump(toe, 0.45, 380, 0.012);
      const hp = surface === 'grass' ? 1800 : 1100;
      addNoise(a, sr, heel + 0.01, 0.05, 0.1 * v, r, [['hp', hp], ['lp', 7000]], 0.015); // wet vegetation swish
      for (let i = 0; i < 20; i++) addGrain(a, sr, heel + r.expo(0.03), r.log(700, 3000), 0.0015, 0.1 * r() * v, r);
      if (surface === 'grass') for (let i = 0; i < 5; i++) addBubble(a, sr, r.range(0.01, 0.12), r.log(1200, 3200), 0.008, 0.04 * v, 0.5);
      break;
    }
    default: { // wet asphalt: heel click + rubber slap + water film squish/splash
      thump(heel, 0.75, 520, 0.01);
      addNoise(a, sr, heel, 0.0025, 0.55 * v, r, [['hp', 1600], ['lp', 9000]]);
      addNoise(a, sr, heel + 0.003, 0.035, 0.28 * v, r, [['bp', 2400, 0.6]], 0.004);
      for (let i = 0; i < 5; i++) addBubble(a, sr, heel + r.range(0.008, 0.07), r.log(1100, 3600), r.range(0.006, 0.016), r.range(0.04, 0.13) * v, 0.5);
      thump(toe, 0.45, 600, 0.008);
      addNoise(a, sr, toe, 0.025, 0.18 * v, r, [['bp', 3200, 0.7]], 0.004); // toe scuff + spray
      for (let i = 0; i < 3; i++) addBubble(a, sr, toe + r.range(0.005, 0.05), r.log(1400, 3800), 0.008, 0.06 * v, 0.5);
    }
  }
  fadeEdges(a, sr);
  return normalize([a], 0.85);
}

// ------------------------------------------------------------------------------------------------ impacts
/** Rock impact. size 0 small (pebbles), 1 medium (0.3-0.6 m rock), 2 large (boulder). */
export function rockImpact(sr, seed, size = 1) {
  const r = makeRng(seed * 13 + size * 101);
  const dur = [0.45, 1.4, 2.8][size];
  const a = buf(sr, dur);
  if (size === 0) {
    const hits = 1 + (r() < 0.6 ? 1 : 0) + (r() < 0.3 ? 1 : 0);
    let t = 0.002;
    for (let h = 0; h < hits; h++) {
      const g = h === 0 ? 1 : r.range(0.25, 0.55);
      addNoise(a, sr, t, 0.001, 0.8 * g, r, [['hp', 1200]]);
      for (let m = 0; m < 5; m++) addMode(a, sr, t, r.log(1300, 5200), r.range(0.004, 0.02), r.range(0.1, 0.35) * g, r() * TAU);
      addNoise(a, sr, t, 0.006, 0.35 * g, r, [['lp', 900]]);
      t += r.range(0.04, 0.14);
    }
  } else if (size === 1) {
    addMode(a, sr, 0.002, r.range(75, 115), 0.04, 0.9, 0, -0.35, 0.001); // thump with pitch drop
    addNoise(a, sr, 0.002, 0.022, 1.0, r, [['lp', 2600], ['hp', 60]]); // crunch
    addNoise(a, sr, 0.002, 0.008, 0.45, r, [['hp', 2500]]); // crack transient
    for (let m = 0; m < 7; m++) addMode(a, sr, 0.002, r.log(230, 1900), r.range(0.012, 0.06), r.range(0.08, 0.3), r() * TAU);
    // secondary bounce
    const tb = r.range(0.18, 0.45);
    addNoise(a, sr, tb, 0.015, 0.35, r, [['lp', 2200]]);
    for (let m = 0; m < 4; m++) addMode(a, sr, tb, r.log(300, 2000), r.range(0.01, 0.04), 0.09, r() * TAU);
    // debris scatter: pebbles and grit knocked loose
    for (let i = 0; i < 45; i++) addGrain(a, sr, 0.01 + r.expo(0.18), r.log(1000, 6500), r.range(0.0008, 0.004), 0.14 * r.expo(0.5), r);
    addNoise(a, sr, 0.01, 0.12, 0.08, r, [['bp', 1800, 0.5]], 0.01);
  } else {
    addMode(a, sr, 0.003, r.range(36, 52), 0.14, 1.0, 0, -0.4, 0.002); // ground shake
    addMode(a, sr, 0.003, r.range(70, 95), 0.07, 0.6, 0, -0.3, 0.001);
    addNoise(a, sr, 0.003, 0.06, 1.2, r, [['lp', 1300], ['lp', 1600], ['hp', 30]]); // massive crunch
    addNoise(a, sr, 0.003, 0.02, 0.5, r, [['hp', 1500], ['lp', 9000]]); // fracture crack
    for (let m = 0; m < 8; m++) addMode(a, sr, 0.003, r.log(110, 900), r.range(0.03, 0.12), r.range(0.1, 0.3), r() * TAU);
    // fragments and crumble tail
    const n = Math.floor(0.9 * sr), tmp = new Float32Array(n);
    for (let i = 0; i < n; i++) { const t = i / sr; tmp[i] = (r() * 2 - 1) * Math.exp(-t / 0.35) * Math.min(1, t / 0.03); }
    filt(tmp, sr, [['lp', 2200], ['hp', 120]]);
    for (let i = 0; i < n && i < a.length; i++) a[i + Math.floor(0.02 * sr)] += tmp[i] * 0.22;
    for (let i = 0; i < 110; i++) addGrain(a, sr, 0.02 + r.expo(0.45), r.log(600, 6000), r.range(0.001, 0.006), 0.2 * r.expo(0.45), r);
    for (let i = 0; i < 4; i++) { // secondary chunks
      const t = r.range(0.1, 1.0), g = r.range(0.12, 0.35);
      addNoise(a, sr, t, 0.02, g, r, [['lp', 1800]]);
      addMode(a, sr, t, r.range(90, 160), 0.035, g * 0.8, 0, -0.2);
    }
  }
  softClip(a, 1.2);
  fadeEdges(a, sr, 0.0005, 0.05);
  return normalize([a], 0.9);
}

/** Car body crash: crumpling sheet steel, panel booms, rattles. */
export function carCrash(sr, seed) {
  const r = makeRng(seed * 17 + 5);
  const a = buf(sr, 1.8);
  addMode(a, sr, 0.002, 58, 0.09, 1.0, 0, -0.3, 0.002);
  // jagged crumple: noise with random 2-8 ms amplitude segments
  const n = Math.floor(0.35 * sr);
  const tmp = new Float32Array(n);
  let seg = 0, g = 1;
  for (let i = 0; i < n; i++) {
    if (--seg <= 0) { seg = Math.floor(r.range(0.002, 0.009) * sr); g = r.expo(0.6); }
    tmp[i] = (r() * 2 - 1) * g * Math.exp(-i / sr / 0.09);
  }
  filt(tmp, sr, [['bp', 1400, 0.45], ['lowshelf', 300, 0.7, 6]]);
  for (let i = 0; i < n; i++) a[i] += tmp[i] * 0.9;
  for (const f of [86, 139, 228, 371, 604, 985, 1610]) addMode(a, sr, 0.003, f * r.range(0.9, 1.1), r.range(0.12, 0.4), r.range(0.1, 0.3), r() * TAU);
  for (let i = 0; i < 60; i++) addGrain(a, sr, 0.01 + r.expo(0.25), r.log(1800, 7000), r.range(0.003, 0.015), 0.1 * r.expo(0.6), r);
  softClip(a, 1.3);
  fadeEdges(a, sr, 0.0005, 0.1);
  return normalize([a], 0.9);
}

/** Suspension bottoming / chassis clunk. soft=true for gear-shift thunk. */
export function clunk(sr, seed, soft = false) {
  const r = makeRng(seed * 19 + (soft ? 7 : 1));
  const a = buf(sr, 0.4);
  addMode(a, sr, 0.001, r.range(60, 80), soft ? 0.03 : 0.035, 1.0, 0, -0.2, 0.0015);
  addNoise(a, sr, 0.001, 0.008, soft ? 0.3 : 0.6, r, [['lp', 900]]);
  for (const [f, tau, g] of [[320, 0.04, 0.35], [545, 0.025, 0.25], [915, 0.015, 0.18], [2300, 0.006, 0.1]]) addMode(a, sr, 0.002, f * r.range(0.9, 1.1), tau, g * (soft ? 0.5 : 1), r() * TAU);
  if (!soft) for (let i = 0; i < 12; i++) addGrain(a, sr, 0.01 + r.expo(0.05), r.log(1500, 5000), 0.004, 0.06 * r(), r);
  fadeEdges(a, sr);
  return normalize([a], 0.9);
}

// ------------------------------------------------------------------------------------------------ one-shots
export function chop(sr, seed) {
  const r = makeRng(seed * 23 + 11);
  const a = buf(sr, 0.9);
  addNoise(a, sr, 0.001, 0.0012, 1.0, r, [['hp', 900]]); // blade bite
  addNoise(a, sr, 0.001, 0.004, 0.45, r, [['bp', 3200, 0.8]]);
  // log body: large softwood log, highly damped low modes
  for (const [f, tau, g] of [[118, 0.07, 0.75], [262, 0.05, 0.5], [455, 0.035, 0.35], [790, 0.022, 0.25], [1380, 0.012, 0.18], [2350, 0.006, 0.1]]) addMode(a, sr, 0.002, f * r.range(0.92, 1.08), tau, g, r() * TAU);
  addNoise(a, sr, 0.002, 0.02, 0.6, r, [['lp', 420]]); // thunk
  // steel head ring (weak, the blade is buried in wood)
  addMode(a, sr, 0.002, r.range(2700, 3000), 0.12, 0.035, 0);
  addMode(a, sr, 0.002, r.range(4100, 4500), 0.08, 0.025, 0);
  // splinters/wood fibres tearing + chips
  for (let i = 0; i < 25; i++) addGrain(a, sr, 0.004 + r.expo(0.03), r.log(1200, 6000), r.range(0.0005, 0.002), 0.12 * r.expo(0.5), r);
  for (let i = 0; i < 4; i++) addGrain(a, sr, r.range(0.25, 0.6), r.log(800, 2500), 0.004, 0.05, r); // chips landing
  softClip(a, 1.4);
  fadeEdges(a, sr, 0.0003, 0.05);
  return normalize([a], 0.92);
}

/** Tree trunk splitting and breaking (short=true: a single sharp snap for landslide trees). */
export function crack(sr, seed, short = false) {
  const r = makeRng(seed * 29 + (short ? 3 : 0));
  const dur = short ? 1.4 : 3.2;
  const a = buf(sr, dur);
  const snap = (t, g) => {
    addNoise(a, sr, t, 0.003, 1.0 * g, r, [['hp', 350]]);
    addNoise(a, sr, t, 0.02, 0.45 * g, r, [['lp', 1600]]);
    for (const [f, tau, gg] of [[170, 0.07, 0.55], [410, 0.045, 0.4], [930, 0.025, 0.3], [1850, 0.012, 0.2], [3400, 0.005, 0.12]]) addMode(a, sr, t, f * r.range(0.85, 1.15), tau, gg * g, r() * TAU);
  };
  const fibre = (t, g) => {
    addNoise(a, sr, t, r.range(0.0008, 0.003), g, r, [['hp', 700]]);
    addMode(a, sr, t, r.log(380, 3200), r.range(0.004, 0.015), g * 0.5, r() * TAU);
  };
  const creak = (t0, len, g, f0, f1) => {
    // stick-slip friction: a pulse train with drifting rate, each pulse ringing the trunk briefly
    let t = t0;
    while (t < t0 + len) {
      const u = (t - t0) / len;
      const rate = f0 + (f1 - f0) * u + r.gauss() * 4;
      const env = Math.sin(Math.PI * u) * g * (0.6 + 0.4 * r());
      addMode(a, sr, t, r.range(260, 520), 0.004, env * 0.25, r() * TAU);
      addMode(a, sr, t, r.range(900, 1400), 0.002, env * 0.1, r() * TAU);
      t += 1 / Math.max(15, rate);
    }
  };
  if (short) {
    snap(0.005, 1);
    let t = 0.02;
    for (let i = 0; i < 18; i++) { t += r.expo(0.025); fibre(t, 0.35 * r()); }
    creak(0.05, 0.4, 0.5, 60, 35);
    addNoise(a, sr, 0.3, 0.25, 0.12, r, [['bp', 1500, 0.4]], 0.08); // crown swish
  } else {
    snap(0.01, 0.9);
    creak(0.1, 1.9, 0.9, 38, 95);
    let t = 0.05;
    while (t < 2.1) { const rate = 6 + 40 * Math.pow(t / 2.1, 2); t += r.expo(1 / rate); fibre(t, (0.12 + 0.4 * t / 2.1) * r()); }
    snap(2.12, 1.0);
    for (let i = 0; i < 25; i++) fibre(2.14 + r.expo(0.08), 0.3 * r());
    addMode(a, sr, 2.13, 55, 0.12, 0.6, 0, -0.3, 0.003);
    addNoise(a, sr, 2.2, 0.35, 0.1, r, [['bp', 1200, 0.4]], 0.1); // needles / branches swish
  }
  softClip(a, 1.3);
  fadeEdges(a, sr, 0.0005, 0.1);
  return normalize([a], 0.92);
}

export function pickup(sr, seed) {
  const r = makeRng(seed * 31 + 1);
  const a = buf(sr, 0.6);
  // jacket/glove rustle: noise with irregular amplitude
  const n = Math.floor(0.3 * sr), tmp = new Float32Array(n);
  let g = 0, seg = 0;
  for (let i = 0; i < n; i++) {
    if (--seg <= 0) { seg = Math.floor(r.range(0.004, 0.02) * sr); g = r(); }
    const t = i / sr;
    tmp[i] = (r() * 2 - 1) * g * Math.min(1, t / 0.03) * Math.exp(-t / 0.12);
  }
  filt(tmp, sr, [['hp', 1500], ['bp', 3500, 0.5]]);
  for (let i = 0; i < n; i++) a[i] += tmp[i] * 0.6;
  addNoise(a, sr, 0.09, 0.015, 0.4, r, [['lp', 350]]); // grab
  for (const f of [1850, 2950, 4400]) addMode(a, sr, 0.11, f * r.range(0.95, 1.05), 0.03, 0.08, r() * TAU); // handle/buckle clink
  fadeEdges(a, sr);
  return normalize([a], 0.8);
}

/** Car door. open=true: handle pull, latch release, check-strap detents. Otherwise a heavy slam. */
export function door(sr, seed, open = false) {
  const r = makeRng(seed * 37 + (open ? 5 : 9));
  const a = buf(sr, open ? 0.8 : 0.9);
  if (open) {
    addNoise(a, sr, 0.005, 0.002, 0.5, r, [['hp', 1500]]);
    addMode(a, sr, 0.005, 2450, 0.004, 0.25, 0);
    // latch release clack
    addNoise(a, sr, 0.09, 0.006, 0.6, r, [['lp', 1200]]);
    for (const [f, tau, g] of [[310, 0.02, 0.4], [820, 0.012, 0.25], [2100, 0.005, 0.15]]) addMode(a, sr, 0.09, f, tau, g, r() * TAU);
    addNoise(a, sr, 0.11, 0.04, 0.12, r, [['bp', 1500, 0.8]], 0.01); // seal peel
    // check strap detents
    for (const t of [0.3, 0.46]) { addMode(a, sr, t, r.range(850, 950), 0.012, 0.25, 0); addMode(a, sr, t, 2200, 0.005, 0.12, 0); addNoise(a, sr, t, 0.003, 0.15, r, [['hp', 800]]); }
  } else {
    addNoise(a, sr, 0.004, 0.0015, 0.6, r, [['hp', 2000]]); // latch striker
    addMode(a, sr, 0.004, 3200, 0.003, 0.2, 0); addMode(a, sr, 0.016, 4700, 0.003, 0.15, 0);
    addNoise(a, sr, 0.006, 0.028, 1.0, r, [['lp', 380], ['lp', 450]]); // air/body thump
    for (const [f, tau, g] of [[62, 0.09, 0.9], [96, 0.07, 0.6], [151, 0.05, 0.4], [243, 0.035, 0.3], [420, 0.06, 0.12], [690, 0.05, 0.1], [1130, 0.04, 0.07]]) addMode(a, sr, 0.006, f * r.range(0.95, 1.05), tau, g, r() * TAU);
    for (let i = 0; i < 16; i++) addGrain(a, sr, 0.02 + r.expo(0.03), r.log(1800, 4500), 0.006, 0.05 * r(), r); // glass/trim rattle
  }
  softClip(a, 1.2);
  fadeEdges(a, sr);
  return normalize([a], 0.9);
}

/** Pouring from a jerrycan into the filler neck: stream + venting glugs (bubbles) + rising cavity resonance. */
export function fuelPour(sr, seed, dur = 5) {
  const r = makeRng(seed * 41 + 3);
  const a = buf(sr, dur);
  const n = a.length;
  // stream: splashy liquid noise, fluctuating, resonating in the filler neck (rising as the tank fills)
  const bq = new Biquad('bp', 1200, 0.7, sr), res = new Biquad('peak', 420, 4, sr, 8);
  let fl = 0.5;
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    if ((i & 63) === 0) { res.set('peak', 380 + 260 * t / dur, 5, sr, 9); fl += (r() - 0.5) * 0.15; fl = Math.min(1, Math.max(0.2, fl)); }
    const env = Math.min(1, t / 0.35) * Math.min(1, (dur - t) / 0.3);
    a[i] = res.tick(bq.tick(r() * 2 - 1)) * env * fl * 0.22;
  }
  // glugs: the can vents by gulping air through the spout
  let t = r.range(0.25, 0.4);
  while (t < dur - 0.3) {
    const u = t / dur;
    addBubble(a, sr, t, r.range(170, 260) * (1 + 0.3 * u), r.range(0.025, 0.04), r.range(0.35, 0.55), 0.7);
    addNoise(a, sr, t, 0.012, 0.18, r, [['lp', 700]]); // flow restarts
    for (let k = 0; k < 3; k++) addBubble(a, sr, t + r.range(0.01, 0.07), r.log(450, 1500), r.range(0.006, 0.018), r.range(0.05, 0.15), 0.5);
    t += r.range(0.2, 0.36);
  }
  // metallic can resonance knocks
  for (let i = 0; i < 3; i++) addMode(a, sr, r.range(0.1, dur - 0.5), r.range(640, 720), 0.08, 0.04, 0);
  fadeEdges(a, sr, 0.01, 0.1);
  return normalize([a], 0.8);
}

/** A 3.9 m scaffold board dropped flat: first end, other end, clatter; steel end bands. */
export function plankDrop(sr, seed) {
  const r = makeRng(seed * 43 + 7);
  const a = buf(sr, 1.8);
  const hit = (t, g) => {
    addNoise(a, sr, t, 0.0015, 0.7 * g, r, [['hp', 900]]);
    addNoise(a, sr, t, 0.012, 0.5 * g, r, [['lp', 500]]);
    for (const [f, tau, gg] of [[78, 0.18, 0.5], [182, 0.12, 0.45], [312, 0.09, 0.35], [521, 0.06, 0.28], [784, 0.04, 0.2], [1160, 0.025, 0.14], [1900, 0.012, 0.08]]) addMode(a, sr, t, f * r.range(0.95, 1.05), tau, gg * g, r() * TAU);
  };
  hit(0.004, 1);
  addMode(a, sr, 0.004, 2380, 0.06, 0.05, 0); addMode(a, sr, 0.004, 3710, 0.04, 0.035, 0);
  hit(r.range(0.05, 0.09), 0.85);
  let t = 0.15;
  for (let i = 0; i < 4; i++) { t += r.range(0.06, 0.16) * (1 - i * 0.15); hit(t, 0.4 * Math.pow(0.6, i)); }
  softClip(a, 1.2);
  fadeEdges(a, sr, 0.0005, 0.1);
  return normalize([a], 0.9);
}

export function thud(sr, seed, heavy = false) {
  const r = makeRng(seed * 47 + (heavy ? 1 : 0));
  const a = buf(sr, 0.7);
  addMode(a, sr, 0.002, r.range(48, 62), heavy ? 0.09 : 0.06, 1.0, 0, -0.35, 0.003);
  addNoise(a, sr, 0.002, heavy ? 0.045 : 0.03, 0.9, r, [['lp', 280], ['lp', 320]]);
  addNoise(a, sr, 0.004, 0.02, 0.18, r, [['hp', 1200], ['lp', 6000]], 0.002); // clothes, gravel
  for (let i = 0; i < 12; i++) addGrain(a, sr, 0.01 + r.expo(0.04), r.log(1200, 4500), 0.002, 0.05 * r(), r);
  fadeEdges(a, sr);
  return normalize([a], 0.9);
}

export function splash(sr, seed) {
  const r = makeRng(seed * 53 + 3);
  const a = buf(sr, 1.2);
  addNoise(a, sr, 0.002, 0.03, 0.9, r, [['hp', 250], ['lp', 7000]], 0.002);
  addNoise(a, sr, 0.01, 0.15, 0.35, r, [['bp', 2800, 0.6]], 0.02); // spray
  addNoise(a, sr, 0.005, 0.08, 0.45, r, [['lp', 450]], 0.005); // sloosh
  for (let i = 0; i < 40; i++) addBubble(a, sr, 0.01 + r.expo(0.15), r.log(450, 3200), r.range(0.008, 0.035), r.range(0.05, 0.25) * r(), r.range(0.3, 0.9));
  for (let i = 0; i < 30; i++) addGrain(a, sr, 0.15 + r.expo(0.25), r.log(1500, 6000), 0.002, 0.06 * r(), r); // droplets falling back
  fadeEdges(a, sr, 0.001, 0.1);
  return normalize([a], 0.9);
}

export function heartbeat(sr, seed) {
  const r = makeRng(seed * 59 + 1);
  const a = buf(sr, 0.9);
  const beat = (t, f, g) => {
    addMode(a, sr, t, f, 0.045, g, 0, -0.25, 0.012);
    addNoise(a, sr, t, 0.03, g * 0.25, r, [['lp', 110], ['lp', 130]], 0.01);
  };
  beat(0.01, 52, 1.0);
  beat(0.29, 63, 0.7);
  fadeEdges(a, sr);
  return normalize([a], 0.9);
}

/** Ignition key: key detent clicks + starter solenoid clunk. */
export function keyClick(sr, seed) {
  const r = makeRng(seed * 61 + 5);
  const a = buf(sr, 0.35);
  for (const t of [0.004, 0.05, 0.1]) { addNoise(a, sr, t, 0.0008, 0.35, r, [['hp', 2500]]); addMode(a, sr, t, r.range(3800, 4600), 0.003, 0.12, 0); }
  addNoise(a, sr, 0.14, 0.006, 0.8, r, [['lp', 700]]); // solenoid
  addMode(a, sr, 0.14, 185, 0.02, 0.6, 0); addMode(a, sr, 0.14, 650, 0.012, 0.35, 0); addMode(a, sr, 0.14, 1900, 0.004, 0.15, 0);
  fadeEdges(a, sr);
  return normalize([a], 0.8);
}

export function drip(sr, seed) {
  const r = makeRng(seed * 67 + 2);
  const a = buf(sr, 0.18);
  addBubble(a, sr, 0.002, r.log(900, 2300), r.range(0.008, 0.02), 0.8, r.range(0.4, 1.0));
  addNoise(a, sr, 0.001, 0.0008, 0.3, r, [['hp', 2000]]);
  fadeEdges(a, sr);
  return normalize([a], 0.8);
}

/** Heavy breath. inhale=true: sharper, higher hiss. Mouth breathing after sprinting. */
export function breath(sr, seed, inhale = false) {
  const r = makeRng(seed * 71 + (inhale ? 9 : 2));
  const dur = inhale ? r.range(0.42, 0.55) : r.range(0.5, 0.65);
  const a = buf(sr, dur);
  const n = a.length;
  const src = new Float32Array(n);
  for (let i = 0; i < n; i++) src[i] = r() * 2 - 1;
  const F = inhale ? [[420, 5, 0.5], [1850, 6, 0.9], [2700, 7, 0.7], [4200, 3, 0.4]] : [[650, 4, 1.0], [1150, 5, 0.8], [2500, 6, 0.4], [3500, 3, 0.2]];
  for (const [f, q, g] of F) {
    const bq = new Biquad('bp', f * r.range(0.95, 1.05), q, sr);
    for (let i = 0; i < n; i++) a[i] += bq.tick(src[i]) * g;
  }
  filt(a, sr, [['hp', inhale ? 500 : 250]]);
  for (let i = 0; i < n; i++) {
    const u = i / n;
    const env = inhale ? Math.pow(Math.sin(Math.PI * Math.pow(u, 0.7)), 1.5) : Math.min(1, u / 0.1) * Math.pow(1 - u, 1.6);
    a[i] *= env;
  }
  if (!inhale) addNoise(a, sr, 0.002, 0.03, 0.08, r, [['lp', 300]], 0.01); // chest push
  fadeEdges(a, sr, 0.005, 0.03);
  return normalize([a], inhale ? 0.6 : 0.8);
}

/** Thunder: stochastic N-wave train from a tortuous lightning channel. close=true adds the ripping crack. */
export function thunder(sr, seed, close = false) {
  const r = makeRng(seed * 73 + (close ? 1 : 0));
  const dur = close ? r.range(9, 12) : r.range(10, 15);
  const out = [];
  // cluster structure shared by both channels (the envelope), individual N-waves decorrelated per channel
  const clusters = [];
  const nC = 3 + Math.floor(r() * 4);
  for (let c = 0; c < nC; c++) clusters.push({ t: (close ? 0.05 : 0.3) + r.expo(close ? 1.2 : 2.0) + c * r.range(0.2, 0.8), w: r.range(0.3, 1.2), g: c === 0 ? 1 : r.range(0.3, 0.9) });
  const env = (t) => {
    let e = 0;
    for (const c of clusters) { const x = (t - c.t) / c.w; e += c.g * (x < 0 ? Math.exp(-x * x * 8) : Math.exp(-x * 0.9)); }
    return e;
  };
  for (let ch = 0; ch < 2; ch++) {
    const rc = makeRng(seed * 79 + ch * 1000 + (close ? 7 : 0));
    const a = buf(sr, dur);
    const n = a.length;
    // N-waves: arrival rate follows the envelope
    let t = 0;
    while (t < dur - 0.2) {
      const e = env(t);
      const rate = 20 + 260 * e;
      t += rc.expo(1 / rate);
      const w = rc.range(0.004, 0.03) * (close && t < 0.6 ? 0.4 : 1); // N-wave duration
      const amp = rc.expo(1) * e;
      const i0 = Math.floor(t * sr), wn = Math.max(4, Math.floor(w * sr));
      for (let i = 0; i < wn && i0 + i < n; i++) a[i0 + i] += amp * (1 - 2 * i / wn);
    }
    // low rumble bed (brown noise with the envelope)
    let y = 0;
    for (let i = 0; i < n; i++) { y = y * 0.9985 + (rc() * 2 - 1) * 0.05; a[i] += y * env(i / sr) * 2.0; }
    // atmospheric absorption: distance lowpass
    const fc = close ? 2800 : rc.range(180, 320);
    filt(a, sr, [['lp', fc, 0.6], ['lp', fc * 1.3, 0.6], ['hp', 22], ['lowshelf', 90, 0.7, 5]]);
    if (close) { // the rip: bright crack at the start
      addNoise(a, sr, 0.02, 0.08, 0.5, rc, [['hp', 600], ['lp', 7000]], 0.005);
      addNoise(a, sr, 0.03, 0.4, 0.3, rc, [['bp', 1200, 0.5]], 0.02);
    }
    for (let i = 0; i < n; i++) { const u = i / n; a[i] *= Math.min(1, u * 60) * Math.min(1, (1 - u) * 6); }
    out.push(a);
  }
  return normalize(out, 0.9);
}

// ------------------------------------------------------------------------------------------------ loops
/**
 * Outdoor rain (stereo loop): dense stochastic droplets on asphalt, leaves and puddles over a steady hiss,
 * plus a distant forest wash. density ~ drops per second per channel.
 */
export function rainLoop(sr, seed, sec = 7, density = 1400) {
  const r = makeRng(seed * 83 + 1);
  const out = [];
  for (let ch = 0; ch < 2; ch++) {
    const a = buf(sr, sec);
    const n = a.length;
    // base hiss (millions of far drops)
    const h = new Float32Array(n);
    for (let i = 0; i < n; i++) h[i] = r() * 2 - 1;
    filtLoop(h, sr, [['hp', 500], ['lp', 9000], ['peak', 3500, 0.8, 3]]);
    // forest wash (rain on needles, low-mid roar)
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = r() * 2 - 1;
    filtLoop(w, sr, [['bp', 900, 0.5], ['lp', 2500]]);
    for (let i = 0; i < n; i++) a[i] = h[i] * 0.16 + w[i] * 0.1;
    // near drops: clicks on hard surfaces + bubbles on puddles (wrapped for a seamless loop)
    const drops = Math.floor(density * sec);
    const tmp = new Float32Array(Math.ceil(0.03 * sr));
    for (let d = 0; d < drops; d++) {
      const t = r() * sec;
      const g = r.expo(1) * 0.05;
      tmp.fill(0);
      const kind = r();
      if (kind < 0.55) { // hard surface tick
        const f = r.log(2500, 9000), tau = r.range(0.0004, 0.0015);
        addMode(tmp, sr, 0, f, tau, g, r() * TAU); tmp[0] += g * 0.8 * (r() - 0.5);
      } else if (kind < 0.8) { // puddle bubble
        addBubble(tmp, sr, 0, r.log(1400, 4800), r.range(0.002, 0.006), g * 0.6, r.range(0.2, 0.8));
      } else { // leaf / vegetation patter (lower, soft)
        addMode(tmp, sr, 0, r.log(700, 2200), r.range(0.001, 0.003), g * 0.7, r() * TAU);
      }
      const i0 = Math.floor(t * sr);
      for (let i = 0; i < tmp.length; i++) a[(i0 + i) % n] += tmp[i];
    }
    // a few louder close drops (gutter, guardrail, rock)
    for (let d = 0; d < 6 * sec; d++) {
      tmp.fill(0);
      addMode(tmp, sr, 0, r.log(900, 3500), r.range(0.002, 0.006), r.range(0.05, 0.14), r() * TAU, 0.3);
      const i0 = Math.floor(r() * n);
      for (let i = 0; i < tmp.length; i++) a[(i0 + i) % n] += tmp[i];
    }
    out.push(a);
  }
  return normalize(out, 0.6);
}

/** Rain drumming on the steel roof, heard from inside the car (stereo loop). */
export function roofLoop(sr, seed, sec = 6, density = 700) {
  const r = makeRng(seed * 89 + 3);
  const out = [];
  for (let ch = 0; ch < 2; ch++) {
    const n = Math.ceil(sec * sr);
    const ex = new Float32Array(n); // excitation (drop impacts)
    const drops = Math.floor(density * sec);
    for (let d = 0; d < drops; d++) {
      const i0 = Math.floor(r() * n), g = r.expo(1);
      const w = Math.floor(r.range(0.0002, 0.0006) * sr) + 1;
      for (let i = 0; i < w; i++) ex[(i0 + i) % n] += g * Math.sin(Math.PI * i / w);
    }
    const a = new Float32Array(n);
    // panel modes (roof skin + headliner damping)
    const modes = [[190, 9, 0.8], [285, 10, 0.7], [410, 12, 0.6], [640, 12, 0.45], [980, 10, 0.3], [1650, 6, 0.2]];
    for (const [f, q, g] of modes) {
      const tmp = ex.slice();
      filtLoop(tmp, sr, [['bp', f * r.range(0.95, 1.05), q]]);
      for (let i = 0; i < n; i++) a[i] += tmp[i] * g;
    }
    // direct tick of each drop (bright) + muffled outside rain through glass
    const tick = ex.slice(); filtLoop(tick, sr, [['hp', 2500], ['lp', 8000]]);
    const hiss = new Float32Array(n); for (let i = 0; i < n; i++) hiss[i] = r() * 2 - 1;
    filtLoop(hiss, sr, [['lp', 1400], ['hp', 150]]);
    for (let i = 0; i < n; i++) a[i] += tick[i] * 0.25 + hiss[i] * 0.05;
    out.push(a);
  }
  return normalize(out, 0.6);
}

/** Tyre-on-gravel crunch texture (mono loop). Played with playbackRate ~ speed. */
export function gravelLoop(sr, seed, sec = 3) {
  const r = makeRng(seed * 97 + 1);
  const n = Math.ceil(sec * sr);
  const a = new Float32Array(n);
  const tmp = new Float32Array(Math.ceil(0.012 * sr));
  for (let d = 0; d < 2600 * sec; d++) {
    tmp.fill(0);
    addMode(tmp, sr, 0, r.log(700, 6500), r.range(0.0004, 0.002), r.expo(0.1), r() * TAU);
    const i0 = Math.floor(r() * n);
    for (let i = 0; i < tmp.length; i++) a[(i0 + i) % n] += tmp[i];
  }
  // stones pinging off the wheel arches
  for (let d = 0; d < 5 * sec; d++) {
    tmp.fill(0);
    addMode(tmp, sr, 0, r.range(1800, 3200), 0.003, r.range(0.2, 0.4), 0);
    const i0 = Math.floor(r() * n);
    for (let i = 0; i < tmp.length; i++) a[(i0 + i) % n] += tmp[i];
  }
  const low = new Float32Array(n); for (let i = 0; i < n; i++) low[i] = r() * 2 - 1;
  filtLoop(low, sr, [['lp', 350], ['lp', 400]]);
  for (let i = 0; i < n; i++) a[i] += low[i] * 1.2;
  return normalize([a], 0.7);
}

/** Tyres through mud and standing water (mono loop). */
export function mudLoop(sr, seed, sec = 3) {
  const r = makeRng(seed * 101 + 1);
  const n = Math.ceil(sec * sr);
  const a = new Float32Array(n);
  const low = new Float32Array(n); for (let i = 0; i < n; i++) low[i] = r() * 2 - 1;
  filtLoop(low, sr, [['lp', 600], ['peak', 250, 1.5, 6]]);
  // slow random AM (slosh)
  let g = 0.5, tg = 0.5;
  for (let i = 0; i < n; i++) { if (i % 800 === 0) tg = r(); g += (tg - g) * 0.002; a[i] = low[i] * (0.3 + g); }
  const tmp = new Float32Array(Math.ceil(0.08 * sr));
  for (let d = 0; d < 26 * sec; d++) {
    tmp.fill(0);
    addBubble(tmp, sr, 0, r.log(180, 900), r.range(0.01, 0.03), r.range(0.1, 0.35), 0.7);
    const i0 = Math.floor(r() * n);
    for (let i = 0; i < tmp.length; i++) a[(i0 + i) % n] += tmp[i];
  }
  const spr = new Float32Array(n); for (let i = 0; i < n; i++) spr[i] = r() * 2 - 1;
  filtLoop(spr, sr, [['hp', 1800], ['lp', 7000]]);
  for (let i = 0; i < n; i++) a[i] += spr[i] * 0.12 * (0.4 + g);
  return normalize([a], 0.7);
}

// ------------------------------------------------------------------------------------------------ impulse responses
/** Open mountain road: nearby rock cut slap, forest diffusion, a few late valley echoes. Stereo. */
export function irMountain(sr, seed = 1) {
  const r = makeRng(seed * 103 + 1);
  const sec = 3.2, out = [];
  for (let ch = 0; ch < 2; ch++) {
    const rc = makeRng(seed * 107 + ch * 31);
    const a = buf(sr, sec), n = a.length;
    // diffuse forest/slope tail (dark), starts after the ground reflection
    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) { const t = i / sr; d[i] = (rc() * 2 - 1) * Math.exp(-t / 0.32) * Math.min(1, t / 0.02); }
    filt(d, sr, [['lp', 3200], ['hp', 120]]);
    for (let i = 0; i < n; i++) a[i] += d[i] * 0.22;
    // discrete reflections: rock cut face (near), road surface, far cliffs across the ravine
    const taps = [[0.006, 0.35, 8000], [0.021 + ch * 0.003, 0.5, 6000], [0.034, 0.3, 5000], [0.052 + ch * 0.004, 0.25, 4000]];
    for (let k = 0; k < 6; k++) taps.push([0.25 + k * r.range(0.18, 0.35) + ch * 0.01, 0.12 * Math.pow(0.7, k), 1800 - k * 180]);
    for (const [t, g, fc] of taps) {
      const len = Math.floor((t > 0.2 ? 0.06 : 0.004) * sr);
      const tmp = new Float32Array(len + 64);
      for (let i = 0; i < len; i++) tmp[i] = (rc() * 2 - 1) * Math.sin(Math.PI * i / len);
      filt(tmp, sr, [['lp', fc]]);
      const i0 = Math.floor(t * sr);
      for (let i = 0; i < tmp.length && i0 + i < n; i++) a[i0 + i] += tmp[i] * g * (t > 0.2 ? 0.8 : 3);
    }
    out.push(a);
  }
  return normalize(out, 0.6);
}

/** Concrete road tunnel: flutter between walls, long bright dense tail (RT60 ~3.4 s). Stereo. */
export function irTunnel(sr, seed = 1) {
  const sec = 4.4, out = [];
  for (let ch = 0; ch < 2; ch++) {
    const rc = makeRng(seed * 109 + ch * 17);
    const a = buf(sr, sec), n = a.length;
    const lo = new Float32Array(n), hi = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / sr, on = Math.min(1, t / 0.03);
      lo[i] = (rc() * 2 - 1) * Math.exp(-t / 0.5) * on;
      hi[i] = (rc() * 2 - 1) * Math.exp(-t / 0.33) * on;
    }
    filt(lo, sr, [['lp', 1200]]);
    filt(hi, sr, [['hp', 1200], ['lp', 9000]]);
    for (let i = 0; i < n; i++) a[i] = lo[i] * 0.3 + hi[i] * 0.22;
    // flutter echo between the walls (~9 m wide) + vault
    const per = 2 * 9.2 / 343;
    for (let k = 1; k < 30; k++) {
      const t = k * per * (1 + (rc() - 0.5) * 0.02) + ch * 0.0015;
      const g = 0.5 * Math.pow(0.8, k);
      const i0 = Math.floor(t * sr), len = Math.floor(0.002 * sr * (1 + k * 0.3));
      for (let i = 0; i < len && i0 + i < n; i++) a[i0 + i] += (rc() * 2 - 1) * g * Math.sin(Math.PI * i / len);
    }
    out.push(a);
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ landslide loops
/** Wrap-add a short buffer into a loop (seamless). */
function wrapAdd(a, tmp, i0, g = 1) { const n = a.length; for (let i = 0; i < tmp.length; i++) a[(i0 + i) % n] += tmp[i] * g; }

/** Deep ground rumble of a moving slope (stereo loop): infrasonic-ish roar, slow surges, grinding mid band. */
export function rumbleLoop(sr, seed, sec = 8) {
  const r = makeRng(seed * 113 + 1);
  const out = [];
  // shared slow surge envelope (periodic: sum of integer-cycle sines)
  const nS = Math.ceil(sec * sr);
  const surge = new Float32Array(nS);
  const comps = [[1, r() * TAU, 0.35], [2, r() * TAU, 0.25], [3, r() * TAU, 0.2], [5, r() * TAU, 0.12], [8, r() * TAU, 0.08]];
  for (let i = 0; i < nS; i++) { let s = 0; for (const [k, p, g] of comps) s += g * Math.sin(TAU * k * i / nS + p); surge[i] = 0.7 + s; }
  for (let ch = 0; ch < 2; ch++) {
    const a = buf(sr, sec), n = a.length;
    const lo = new Float32Array(n), mid = new Float32Array(n);
    for (let i = 0; i < n; i++) { lo[i] = r() * 2 - 1; mid[i] = r() * 2 - 1; }
    filtLoop(lo, sr, [['lp', 70, 0.8], ['lp', 90, 0.7], ['hp', 14]]);
    filtLoop(mid, sr, [['bp', 260, 0.6], ['lp', 700]]);
    for (let i = 0; i < n; i++) a[i] = lo[i] * 6.0 * surge[i] + mid[i] * 0.5 * surge[i] * surge[i];
    // grinding: dense low knocks of rock against rock
    const tmp = new Float32Array(Math.ceil(0.12 * sr));
    for (let d = 0; d < 40 * sec; d++) {
      tmp.fill(0);
      addMode(tmp, sr, 0, r.log(45, 240), r.range(0.015, 0.05), r.expo(0.12), r() * TAU, -0.2, 0.002);
      wrapAdd(a, tmp, Math.floor(r() * n));
    }
    out.push(a);
  }
  return normalize(out, 0.8);
}

/** Rocks tumbling and rolling down a slope (stereo loop): trains of bounces with decreasing interval, grit. */
export function rockRollLoop(sr, seed, sec = 6) {
  const r = makeRng(seed * 127 + 3);
  const out = [];
  for (let ch = 0; ch < 2; ch++) {
    const a = buf(sr, sec), n = a.length;
    const tmp = new Float32Array(Math.ceil(0.25 * sr));
    const rocks = Math.floor(7 * sec);
    for (let k = 0; k < rocks; k++) {
      let t = r() * sec;
      const size = r.log(0.15, 1.2);
      const bounces = 3 + Math.floor(r() * 8);
      for (let b = 0; b < bounces; b++) {
        tmp.fill(0);
        const g = r.range(0.3, 1) * Math.pow(size, 0.8);
        addMode(tmp, sr, 0.001, r.range(60, 140) / Math.sqrt(size), r.range(0.02, 0.05), g, r() * TAU, -0.2, 0.001);
        for (let m = 0; m < 3; m++) addMode(tmp, sr, 0.001, r.log(250, 1600) / Math.sqrt(size), r.range(0.006, 0.02), g * 0.25, r() * TAU);
        addNoise(tmp, sr, 0.001, 0.006, g * 0.5, r, [['lp', 1500]]);
        wrapAdd(a, tmp, Math.floor(t * sr));
        t += r.range(0.08, 0.35) * Math.sqrt(size);
      }
    }
    for (let d = 0; d < 300 * sec; d++) { // grit and small stones
      tmp.fill(0);
      addGrain(tmp, sr, 0, r.log(700, 5000), r.range(0.0006, 0.003), 0.05 * r.expo(0.6), r);
      wrapAdd(a, tmp, Math.floor(r() * n));
    }
    const bed = new Float32Array(n); for (let i = 0; i < n; i++) bed[i] = r() * 2 - 1;
    filtLoop(bed, sr, [['bp', 500, 0.5], ['lp', 1800]]);
    for (let i = 0; i < n; i++) a[i] += bed[i] * 0.08;
    out.push(a);
  }
  return normalize(out, 0.8);
}

/** Debris / mud flow front (stereo loop): churning slurry roar, glops, crushed wood and stone. */
export function mudFlowLoop(sr, seed, sec = 6) {
  const r = makeRng(seed * 131 + 5);
  const out = [];
  for (let ch = 0; ch < 2; ch++) {
    const a = buf(sr, sec), n = a.length;
    const lo = new Float32Array(n), hiss = new Float32Array(n);
    for (let i = 0; i < n; i++) { lo[i] = r() * 2 - 1; hiss[i] = r() * 2 - 1; }
    filtLoop(lo, sr, [['lp', 220, 0.7], ['peak', 90, 1, 6], ['hp', 20]]);
    filtLoop(hiss, sr, [['bp', 1100, 0.5], ['lp', 3500]]);
    // periodic random amplitude churn
    const nk = 24, keys = []; for (let k = 0; k < nk; k++) keys.push(r());
    for (let i = 0; i < n; i++) {
      const u = i / n * nk, k0 = Math.floor(u), f = u - k0, s = f * f * (3 - 2 * f);
      const churn = 0.55 + 0.9 * (keys[k0 % nk] * (1 - s) + keys[(k0 + 1) % nk] * s);
      a[i] = lo[i] * 2.2 * churn + hiss[i] * 0.22 * churn;
    }
    const tmp = new Float32Array(Math.ceil(0.15 * sr));
    for (let d = 0; d < 30 * sec; d++) { // glops (large bubbles in the slurry)
      tmp.fill(0); addBubble(tmp, sr, 0, r.log(70, 320), r.range(0.02, 0.06), r.range(0.1, 0.4), 0.6);
      wrapAdd(a, tmp, Math.floor(r() * n));
    }
    for (let d = 0; d < 25 * sec; d++) { // crushed stones / wood knocks
      tmp.fill(0);
      if (r() < 0.3) { addNoise(tmp, sr, 0, 0.002, 0.25, r, [['hp', 600]]); addMode(tmp, sr, 0, r.log(300, 1400), 0.01, 0.12, r() * TAU); }
      else addMode(tmp, sr, 0, r.log(90, 500), r.range(0.01, 0.04), r.range(0.08, 0.3), r() * TAU, -0.15, 0.001);
      wrapAdd(a, tmp, Math.floor(r() * n));
    }
    out.push(a);
  }
  return normalize(out, 0.8);
}

/** Wind bed (stereo loop): broadband pink noise; the runtime filters/gusts it. */
export function windLoop(sr, seed, sec = 6) {
  const out = noiseLoop(sr, seed * 137 + 1, 'pink', sec);
  for (const a of out) filtLoop(a, sr, [['hp', 40], ['lp', 5000]]);
  return normalize(out, 0.7);
}

/** Wet asphalt tyre hiss (mono loop): water film peeling + spray, broadband. */
export function tyreWetLoop(sr, seed, sec = 3) {
  const r = makeRng(seed * 139 + 7);
  const n = Math.ceil(sec * sr), a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = r() * 2 - 1;
  filtLoop(a, sr, [['hp', 300], ['peak', 1800, 0.7, 4], ['lp', 8000]]);
  const tmp = new Float32Array(Math.ceil(0.02 * sr));
  for (let d = 0; d < 500 * sec; d++) { tmp.fill(0); addBubble(tmp, sr, 0, r.log(900, 4500), r.range(0.002, 0.006), r.range(0.05, 0.3), 0.5); wrapAdd(a, tmp, Math.floor(r() * n)); }
  return normalize([a], 0.7);
}

/** Generator registry: name -> (sr, seed, arg) => channels. Used by the bank worker and the test page. */
export const GENERATORS = {
  footstep: (sr, seed, a) => footstep(sr, seed, a?.surface, a?.run),
  rockImpact: (sr, seed, a) => rockImpact(sr, seed, a?.size ?? 1),
  carCrash, clunk: (sr, seed, a) => clunk(sr, seed, !!a?.soft), chop,
  crack: (sr, seed, a) => crack(sr, seed, !!a?.short), pickup,
  door: (sr, seed, a) => door(sr, seed, !!a?.open), fuelPour: (sr, seed, a) => fuelPour(sr, seed, a?.dur ?? 5),
  plankDrop, thud: (sr, seed, a) => thud(sr, seed, !!a?.heavy), splash, heartbeat, keyClick, drip,
  breath: (sr, seed, a) => breath(sr, seed, !!a?.inhale), thunder: (sr, seed, a) => thunder(sr, seed, !!a?.close),
  rainLoop: (sr, seed) => rainLoop(sr, seed), roofLoop: (sr, seed) => roofLoop(sr, seed),
  gravelLoop: (sr, seed) => gravelLoop(sr, seed), mudLoop: (sr, seed) => mudLoop(sr, seed),
  tyreWetLoop: (sr, seed) => tyreWetLoop(sr, seed), windLoop: (sr, seed) => windLoop(sr, seed),
  rumbleLoop: (sr, seed) => rumbleLoop(sr, seed), rockRollLoop: (sr, seed) => rockRollLoop(sr, seed),
  mudFlowLoop: (sr, seed) => mudFlowLoop(sr, seed),
  irMountain: (sr, seed) => irMountain(sr, seed), irTunnel: (sr, seed) => irTunnel(sr, seed),
};
