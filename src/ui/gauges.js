// Analog instrument cluster (speedometer + fuel gauge with low-fuel lamp), drawn as SVG.
// Styled after early-90s 4x4 clusters: black faces, warm-white backlit markings, orange needles,
// a chrome-ish bezel and a glass reflection. Needles are driven by a damped spring (timed from
// performance.now(), so they keep settling even though game dt is 0 while paused).
import { ICONS } from './icons.js';

const NS = 'http://www.w3.org/2000/svg';
const rad = (deg) => (deg * Math.PI) / 180;
// angle measured clockwise from 12 o'clock
const polar = (cx, cy, r, deg) => [cx + r * Math.sin(rad(deg)), cy - r * Math.cos(rad(deg))];
const f1 = (v) => v.toFixed(1);

function arcPath(cx, cy, r, a0, a1) {
  const [x0, y0] = polar(cx, cy, r, a0), [x1, y1] = polar(cx, cy, r, a1);
  const large = Math.abs(a1 - a0) > 180 ? 1 : 0;
  return `M${f1(x0)} ${f1(y0)}A${r} ${r} 0 ${large} 1 ${f1(x1)} ${f1(y1)}`;
}

const DEFS = `
<defs>
  <radialGradient id="ls-face" cx="50%" cy="42%" r="62%">
    <stop offset="0" stop-color="#1b1e1f"/><stop offset=".7" stop-color="#0d0f10"/><stop offset="1" stop-color="#040505"/>
  </radialGradient>
  <linearGradient id="ls-bezel" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#6a6f73"/><stop offset=".18" stop-color="#2a2d30"/><stop offset=".55" stop-color="#121416"/>
    <stop offset=".85" stop-color="#34383b"/><stop offset="1" stop-color="#8a8f92"/>
  </linearGradient>
  <linearGradient id="ls-glass" x1="0" y1="0" x2=".35" y2="1">
    <stop offset="0" stop-color="#fff" stop-opacity=".16"/><stop offset=".45" stop-color="#fff" stop-opacity=".03"/><stop offset="1" stop-color="#fff" stop-opacity="0"/>
  </linearGradient>
  <radialGradient id="ls-cap" cx="40%" cy="35%" r="70%">
    <stop offset="0" stop-color="#4a4e52"/><stop offset=".6" stop-color="#16181a"/><stop offset="1" stop-color="#050606"/>
  </radialGradient>
  <linearGradient id="ls-needle" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#c2421c"/><stop offset=".5" stop-color="#ff7a3d"/><stop offset="1" stop-color="#b83a17"/>
  </linearGradient>
  <radialGradient id="ls-inner" cx="50%" cy="58%" r="52%">
    <stop offset=".78" stop-color="#000" stop-opacity="0"/><stop offset=".94" stop-color="#000" stop-opacity=".55"/><stop offset="1" stop-color="#000" stop-opacity=".9"/>
  </radialGradient>
  <radialGradient id="ls-backlight" cx="50%" cy="50%" r="50%">
    <stop offset="0" stop-color="#ffcf8a" stop-opacity=".055"/><stop offset=".55" stop-color="#ffb870" stop-opacity=".02"/><stop offset="1" stop-color="#ffb870" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="ls-spec" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#fff" stop-opacity=".22"/><stop offset=".5" stop-color="#fff" stop-opacity=".05"/><stop offset="1" stop-color="#fff" stop-opacity="0"/>
  </linearGradient>
  <radialGradient id="ls-lamp" cx="50%" cy="50%" r="50%">
    <stop offset="0" stop-color="#ffb347" stop-opacity=".55"/><stop offset="1" stop-color="#ff8c1a" stop-opacity="0"/>
  </radialGradient>
</defs>`;

function dialShell(extra = '') {
  return `
  <circle cx="100" cy="100" r="99" fill="#000" fill-opacity=".55"/>
  <circle cx="100" cy="100" r="97" fill="url(#ls-bezel)"/>
  <circle cx="100" cy="100" r="90.5" fill="#030404"/>
  <circle cx="100" cy="100" r="88" fill="url(#ls-face)"/>
  <circle cx="100" cy="100" r="88" fill="url(#ls-backlight)"/>
  <circle cx="100" cy="100" r="88" fill="none" stroke="#000" stroke-opacity=".8" stroke-width="3"/>
  ${extra}`;
}

// inner bezel shadow (the ring sits proud of the face), then the curved glass: a broad soft sheen, a thin
// specular arc along the upper-left rim, and a faint bounce on the lower rim.
const glass = `
  <circle cx="100" cy="100" r="88.5" fill="url(#ls-inner)"/>
  <path d="M22 78 A80 80 0 0 1 170 58 C130 50 70 58 22 78Z" fill="url(#ls-glass)"/>
  <path d="${arcPath(100, 100, 85, -78, -8)}" fill="none" stroke="url(#ls-spec)" stroke-width="2.2" stroke-linecap="round"/>
  <path d="${arcPath(100, 100, 85.5, 128, 200)}" fill="none" stroke="#fff" stroke-opacity=".07" stroke-width="1.4" stroke-linecap="round"/>
  <circle cx="100" cy="100" r="88" fill="none" stroke="#fff" stroke-opacity=".05" stroke-width="1"/>`;

function speedoSVG() {
  const A0 = -125, A1 = 125, MAX = 160;
  let ticks = '', nums = '';
  for (let v = 0; v <= MAX; v += 5) {
    const a = A0 + (v / MAX) * (A1 - A0);
    const major = v % 20 === 0, mid = v % 10 === 0;
    const r2 = major ? 67 : mid ? 72 : 75;
    const [x1, y1] = polar(100, 100, 80, a), [x2, y2] = polar(100, 100, r2, a);
    ticks += `<line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(x2)}" y2="${f1(y2)}" stroke-width="${major ? 2.6 : mid ? 1.5 : 0.9}" stroke-opacity="${major ? 1 : mid ? 0.85 : 0.55}"/>`;
    if (major) {
      const [tx, ty] = polar(100, 100, 55, a);
      nums += `<text x="${f1(tx)}" y="${f1(ty + 4.8)}">${v}</text>`;
    }
  }
  return `<svg class="dial dial-speed" viewBox="0 0 200 200" aria-hidden="true">${DEFS}
  ${dialShell()}
  <path d="${arcPath(100, 100, 81.5, A0, A1)}" fill="none" stroke="#e9e1d2" stroke-opacity=".25" stroke-width=".8"/>
  <g class="ticks" stroke="#efe7d8">${ticks}</g>
  <g class="nums" fill="#efe7d8">${nums}</g>
  <text class="unit" x="100" y="72" fill="#efe7d8" fill-opacity=".62">km/h</text>
  <g class="odo" transform="translate(76 126)">
    <rect width="48" height="13" rx="1.5" fill="#000" stroke="#2c2f31" stroke-width=".8"/>
    <text class="odo-t" x="21" y="9.9" fill="#d9d2c4" fill-opacity=".85">00000</text>
    <rect x="39.6" y="1.2" width="7.4" height="10.6" fill="#e2dccf"/>
    <text class="odo-d" x="43.3" y="9.9" fill="#111">0</text>
  </g>
  <g class="needle-sh" opacity=".45"><path d="M-1.4 -77 L1.4 -77 L3.2 16 L-3.2 16 Z" fill="#000"/></g>
  <g class="needle">
    <path d="M-1.1 -78 L1.1 -78 L2.6 16 L-2.6 16 Z" fill="url(#ls-needle)"/>
    <path d="M-1.1 -78 L1.1 -78 L2.6 16 L-2.6 16 Z" fill="none" stroke="#000" stroke-opacity=".35" stroke-width=".6"/>
  </g>
  <circle cx="100" cy="100" r="10.5" fill="url(#ls-cap)" stroke="#000" stroke-width="1"/>
  <circle cx="100" cy="100" r="3" fill="#0a0b0c" stroke="#575b5e" stroke-width=".6"/>
  ${glass}
  </svg>`;
}

function fuelSVG() {
  const A0 = -58, A1 = 58, CX = 100, CY = 122;
  let ticks = '';
  for (let i = 0; i <= 8; i++) {
    const a = A0 + (i / 8) * (A1 - A0);
    const major = i % 4 === 0, mid = i % 2 === 0;
    const [x1, y1] = polar(CX, CY, 78, a), [x2, y2] = polar(CX, CY, major ? 64 : mid ? 68 : 72, a);
    ticks += `<line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(x2)}" y2="${f1(y2)}" stroke-width="${major ? 3.2 : mid ? 2.2 : 1.4}"/>`;
  }
  const lab = (t, frac) => { const [x, y] = polar(CX, CY, 51, A0 + frac * (A1 - A0)); return `<text x="${f1(x)}" y="${f1(y + 7)}">${t}</text>`; };
  return `<svg class="dial dial-fuel" viewBox="0 0 200 200" aria-hidden="true">${DEFS}
  ${dialShell()}
  <path d="${arcPath(CX, CY, 74, A0, A0 + 0.065 * (A1 - A0))}" fill="none" stroke="#d63a22" stroke-width="9"/>
  <g class="ticks" stroke="#efe7d8">${ticks}</g>
  <g class="nums" fill="#efe7d8">${lab('E', 0)}${lab('½', 0.5)}${lab('F', 1)}</g>
  <g class="lamp" transform="translate(84 136)">
    <circle cx="11" cy="12" r="22" fill="url(#ls-lamp)" class="lamp-glow"/>
    <svg class="lamp-icon" x="0" y="0" width="24" height="24" viewBox="0 0 24 24">${ICONS.pump}</svg>
  </g>
  <g class="needle-sh" opacity=".45"><path d="M-1.7 -75 L1.7 -75 L3.4 12 L-3.4 12 Z" fill="#000"/></g>
  <g class="needle">
    <path d="M-1.4 -76 L1.4 -76 L3 12 L-3 12 Z" fill="url(#ls-needle)"/>
  </g>
  <circle cx="${CX}" cy="${CY}" r="11" fill="url(#ls-cap)" stroke="#000" stroke-width="1"/>
  ${glass}
  </svg>`;
}

class Needle {
  constructor(el, cx, cy, a0, a1, shadow = null) {
    this.el = el; this.sh = shadow; this.cx = cx; this.cy = cy; this.a0 = a0; this.a1 = a1;
    this.target = 0; this.v = 0; this.x = 0; this.last = -1;
  }
  step(dt) {
    // lightly under-damped spring: realistic needle overshoot/settle
    const k = 70, c = 13;
    const acc = k * (this.target - this.x) - c * this.v;
    this.v += acc * dt; this.x += this.v * dt;
    const a = this.a0 + Math.max(-0.02, Math.min(1.02, this.x)) * (this.a1 - this.a0);
    if (Math.abs(a - this.last) > 0.02) {
      this.last = a;
      this.el.setAttribute('transform', `translate(${this.cx} ${this.cy}) rotate(${a.toFixed(2)})`);
      // the needle floats a few mm above the face: its shadow falls down-right (light from the upper left)
      this.sh?.setAttribute('transform', `translate(${this.cx + 1.6} ${this.cy + 2.4}) rotate(${a.toFixed(2)})`);
    }
  }
}

export class Cluster {
  constructor() {
    const el = document.createElement('div');
    el.className = 'cluster';
    el.innerHTML = `<div class="gauge gauge-fuel">${fuelSVG()}</div><div class="gauge gauge-speed">${speedoSVG()}</div>`;
    this.el = el;
    this.speed = new Needle(el.querySelector('.dial-speed .needle'), 100, 100, -125, 125, el.querySelector('.dial-speed .needle-sh'));
    this.fuel = new Needle(el.querySelector('.dial-fuel .needle'), 100, 122, -58, 58, el.querySelector('.dial-fuel .needle-sh'));
    this.lamp = el.querySelector('.dial-fuel .lamp');
    this.odoEl = el.querySelector('.odo-t'); this.odoDec = el.querySelector('.odo-d');
    this.kmh = 0; this.fuelFrac = 0; this.lowFuel = null;
    this.odo = 41873.2 + Math.random() * 30; // km, old car
    this._odoShown = '';
    this._t = performance.now();
    this.speed.step(0); this.fuel.step(0);
    this._renderOdo();
  }
  setSpeed(kmh) { this.kmh = Math.max(0, +kmh || 0); this.speed.target = Math.min(this.kmh, 168) / 160; }
  setFuel(f) {
    this.fuelFrac = Math.max(0, Math.min(1, +f || 0));
    this.fuel.target = this.fuelFrac;
    // reserve lamp: lit on the empty tank of the cold open, dark on the jerrycan's ~1/8 (5 L of 45) for the whole escape
    const low = this.fuelFrac < 0.045;
    if (low !== this.lowFuel) { this.lowFuel = low; this.el.classList.toggle('low-fuel', low); }
  }
  _renderOdo() {
    const s = Math.floor(this.odo * 10).toString().padStart(6, '0').slice(-6);
    if (s !== this._odoShown) { this._odoShown = s; this.odoEl.textContent = s.slice(0, 5); this.odoDec.textContent = s[5]; }
  }
  update(now = performance.now()) {
    const dt = Math.min(0.05, Math.max(0, (now - this._t) / 1000));
    this._t = now;
    this.speed.step(dt); this.fuel.step(dt);
    this.odo += (this.kmh / 3600) * dt;
    this._renderOdo();
  }
}
