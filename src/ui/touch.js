// LANDSLIDE: on-screen touch controls for phones and tablets (TOUCH workstream).
//
// Created by core/input.js when the device is touch-primary (or with ?touch=1); never built on desktop.
// It only feeds the Input object, so every system keeps reading the same things it reads on desktop:
//   on foot  left ~40 %: floating stick -> input.axes.moveX/moveY (past ~85 % it sprints: virtual ShiftLeft)
//            right side: drag to look -> input.addLook (smoothed into input.dx/dy)
//            buttons: Interact (virtual KeyE, hold for chop/refuel/planks; mirrors interact.prompt), Jump (Space),
//            Sprint (toggle, ShiftLeft)
//   in car   left ~40 %: steering slider -> input.axes.steer (+ = right); right side: drag to look / orbit
//            buttons: Gas (KeyW), Brake / reverse (KeyS), Handbrake (Space), Camera (KeyC), Interact (KeyE)
//   always   Pause (top right) -> hud.pause()
// Pointer events with touch-action: none; every pointer is tracked by id, so move + look + a button work together.
// The layer sits under the HUD (#ui, z 10) and over the canvas: HUD text stays on top and passes touches through.
// The controls hide during cinematics, menus, fades, ads / ad breaks and the win and death screens.
import { platform } from '../core/platform.js';
import { ICONS } from './icons.js';

const PLAYING = new Set(['intro', 'stalled', 'onfoot', 'driving', 'escape']);
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth01 = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
const LEFT_ZONE = 0.4;          // fraction of the width that belongs to the stick / steering
const STICK_DEAD = 0.1, STICK_WALK = 0.72, SPRINT_ON = 0.86, SPRINT_OFF = 0.76;
const STEER_DEAD = 0.05;
const LOOK_FULL_WIDTH = 3.4;    // radians turned by a drag across the whole (long) screen side at sensitivity 1
const BASE_SENS = 0.0022;       // hud.js BASE_SENS: input.sensitivity at setting 1

const svg = (body, sw = 2.4) =>
  `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
const IC = {
  pause: svg('<path d="M18.5 14v20M29.5 14v20" stroke-width="3.6"/>'),
  jump: svg('<path d="M14 23.5 24 13.5l10 10"/><path d="M14 34.5 24 24.5l10 10" stroke-opacity=".55"/>', 3),
  sprint: svg('<circle cx="29.5" cy="10" r="3.4"/><path d="M19.5 19.5 27 16.5l5 5.5 6 1.5"/><path d="M27 16.5 22.5 28l6.5 5.5V42"/><path d="M22.5 28 18 35.5l-8 .5"/>'),
  hand: svg('<path d="M18.5 26V12.5a2.6 2.6 0 0 1 5.2 0V23M23.7 21.5v-3a2.6 2.6 0 0 1 5.2 0V23M28.9 20a2.6 2.6 0 0 1 5.2 0v3.5M34.1 22a2.6 2.6 0 0 1 5.2 0V31c0 6.5-4.3 10.5-10.5 10.5h-2.3c-4.2 0-6.8-1.9-8.9-5.2L11.4 29a2.5 2.5 0 0 1 3.9-3.1l3.2 3.1"/>', 2.2),
  door: svg('<path d="M11 23 19 12.5h18V39H11z"/><path d="M12.5 23H37"/><path d="M29 28.5h4.5"/>', 2.2),
  key: svg('<circle cx="15.5" cy="24" r="6.5"/><circle cx="15.5" cy="24" r="1.6" fill="currentColor" stroke="none"/><path d="M22 24h18M34 24v5.5M39.5 24v4"/>', 2.2),
  cam: svg('<rect x="7" y="15" width="25" height="18" rx="3"/><path d="M32 21.5 41 17v14l-9-4.5"/>', 2.2),
  hb: svg('<circle cx="24" cy="24" r="11.5"/><path d="M20.5 30.5V17.5h5a4 4 0 0 1 0 8h-5"/><path d="M8.5 14a19 19 0 0 0 0 20M39.5 14a19 19 0 0 1 0 20"/>', 2.2),
  steer: svg('<path d="M15 16.5 7.5 24l7.5 7.5M33 16.5l7.5 7.5-7.5 7.5"/>', 2.6),
  grip: '<svg class="tc-grip" viewBox="0 0 40 30" aria-hidden="true"><path d="M8 6h24M8 15h24M8 24h24" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" fill="none"/></svg>',
};
const ACT_ICON = {
  pickup_jerrycan: ICONS.jerrycan, refuel: ICONS.jerrycan, pickup_hatchet: ICONS.hatchet, chop: ICONS.hatchet,
  pickup_planks: ICONS.planks, place_planks: ICONS.planks, get_in: IC.door, get_out: IC.door, start_engine: IC.key,
};

const CSS = `
#touch-ui { position: fixed; inset: 0; z-index: 9; pointer-events: none; opacity: 0; visibility: hidden;
  transition: opacity .35s cubic-bezier(.2,.7,.1,1), visibility 0s linear .35s;
  --u: clamp(48px, 13vmin, 68px);
  --sl: max(12px, env(safe-area-inset-left, 0px)); --sr: max(12px, env(safe-area-inset-right, 0px));
  --st: max(10px, env(safe-area-inset-top, 0px)); --sb: max(10px, env(safe-area-inset-bottom, 0px));
  --tc-ink: var(--ink, #eceee9); --tc-acc: var(--accent, #eaa53f);
  --tc-fill: rgba(12, 14, 15, .42); --tc-fill-dn: rgba(236, 238, 233, .26);
  --tc-edge: rgba(236, 238, 233, .38); --tc-edge-dn: rgba(236, 238, 233, .8);
  font-family: var(--f-disp, 'Barlow Condensed', 'Arial Narrow', Arial, sans-serif); color: var(--tc-ink);
  -webkit-user-select: none; user-select: none; -webkit-touch-callout: none; -webkit-tap-highlight-color: transparent;
  touch-action: none; }
#touch-ui.on { opacity: 1; visibility: visible; transition: opacity .35s cubic-bezier(.2,.7,.1,1), visibility 0s; }
#touch-ui * { -webkit-tap-highlight-color: transparent; -webkit-touch-callout: none; box-sizing: border-box; }
#touch-ui .tc-pad { position: absolute; inset: 0; touch-action: none; }
#touch-ui.on .tc-pad { pointer-events: auto; }

/* buttons */
#touch-ui .tc-btn { position: absolute; display: grid; place-items: center; width: var(--s); height: var(--s); border-radius: 50%;
  background: var(--tc-fill); border: 1.5px solid var(--tc-edge); color: var(--tc-ink); touch-action: none;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, .2), 0 6px 20px rgba(0, 0, 0, .28), inset 0 1px 0 rgba(255, 255, 255, .06);
  transition: transform .14s cubic-bezier(.2,.7,.1,1), background-color .14s, border-color .14s, color .14s, opacity .3s; }
#touch-ui .tc-btn > svg { width: 50%; height: 50%; filter: drop-shadow(0 1px 1.5px rgba(0, 0, 0, .55)); pointer-events: none; }
#touch-ui .tc-btn.dn { transform: scale(.92); background: var(--tc-fill-dn); border-color: var(--tc-edge-dn); }
#touch-ui.on .tc-grp.act .tc-btn, #touch-ui.on .tc-pause { pointer-events: auto; }
#touch-ui .tc-lbl { font: 500 11px/1 var(--f-disp, sans-serif); letter-spacing: .24em; text-indent: .24em; text-transform: uppercase;
  color: var(--tc-ink); text-shadow: 0 1px 2px rgba(0, 0, 0, .6); pointer-events: none; }

/* mode groups: the foot and car sets slide / fade across when the player gets in or out */
#touch-ui .tc-grp { position: absolute; inset: 0; pointer-events: none; opacity: 0; transform: translateY(18px);
  transition: opacity .4s cubic-bezier(.2,.7,.1,1), transform .5s cubic-bezier(.2,.7,.1,1); }
#touch-ui .tc-grp.act { opacity: 1; transform: none; }

#touch-ui .tc-pause { --s: max(48px, calc(var(--u) * .9)); right: var(--sr); top: var(--st); }
#touch-ui .tc-pause > svg { width: 44%; height: 44%; }

/* on foot */
#touch-ui .tc-jump { --s: calc(var(--u) * 1.45); right: calc(var(--sr) + var(--u) * .3); bottom: calc(var(--sb) + var(--u) * .3); }
#touch-ui .tc-sprint { --s: calc(var(--u) * 1.08); right: calc(var(--sr) + var(--u) * 2.05); bottom: calc(var(--sb) + var(--u) * .5); }
#touch-ui .tc-sprint.on { color: var(--tc-acc); border-color: var(--tc-acc); background: rgba(234, 165, 63, .16); }
#touch-ui .tc-sprint .tc-stam { position: absolute; left: -6px; top: -6px; width: calc(100% + 12px); height: calc(100% + 12px); transform: rotate(-90deg); opacity: 0; transition: opacity .4s; filter: none; }
#touch-ui .tc-sprint .tc-stam.show { opacity: 1; }
#touch-ui .tc-stam circle { fill: none; stroke-width: 2.2; }
#touch-ui .tc-stam .sb { stroke: rgba(255, 255, 255, .14); }
#touch-ui .tc-stam .sf { stroke: rgba(255, 255, 255, .8); stroke-linecap: round; }
#touch-ui .tc-stam.low .sf { stroke: var(--danger, #d4452c); }

/* in the car */
#touch-ui .tc-ped { border-radius: calc(var(--u) * .28); width: var(--w); height: var(--h); display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: calc(var(--u) * .14); }
#touch-ui .tc-ped .tc-grip { width: 46%; height: auto; opacity: .7; filter: drop-shadow(0 1px 1px rgba(0, 0, 0, .5)); pointer-events: none; }
#touch-ui .tc-gas { --w: calc(var(--u) * 1.5); --h: calc(var(--u) * 2.25); right: calc(var(--sr) + var(--u) * .3); bottom: calc(var(--sb) + var(--u) * .3); }
#touch-ui .tc-brake { --w: calc(var(--u) * 1.5); --h: calc(var(--u) * 1.55); right: calc(var(--sr) + var(--u) * 2.1); bottom: calc(var(--sb) + var(--u) * .3); }
#touch-ui .tc-gas.dn { border-color: var(--tc-acc); background: rgba(234, 165, 63, .22); }
#touch-ui .tc-brake.dn { border-color: rgba(235, 120, 95, .95); background: rgba(212, 69, 44, .24); }
#touch-ui .tc-brake .tc-rev { opacity: .6; font-size: 9px; letter-spacing: .2em; margin-top: -4px; }
#touch-ui .tc-hb { --s: calc(var(--u) * 1.08); right: calc(var(--sr) + var(--u) * 3.85); bottom: calc(var(--sb) + var(--u) * .5); }
#touch-ui .tc-cam { --s: max(48px, calc(var(--u) * .9)); right: calc(var(--sr) + max(48px, var(--u) * .9) + var(--u) * .3); top: var(--st); }

/* interact: a pill that mirrors the prompt, with a hold ring; appears only when there is something to do */
#touch-ui .tc-act { position: absolute; right: calc(var(--sr) + var(--u) * .3); display: flex; align-items: center; gap: calc(var(--u) * .22);
  height: calc(var(--u) * 1.3); padding: 0 calc(var(--u) * .12) 0 calc(var(--u) * .36); border-radius: 999px; max-width: 62vw;
  background: rgba(12, 14, 15, .5); border: 1.5px solid rgba(234, 165, 63, .8); color: var(--tc-ink);
  box-shadow: 0 8px 26px rgba(0, 0, 0, .35); touch-action: none; pointer-events: none;
  opacity: 0; transform: translateX(14px) scale(.94); transform-origin: 100% 50%;
  transition: opacity .25s cubic-bezier(.2,.7,.1,1), transform .3s cubic-bezier(.2,.7,.1,1), background-color .14s, border-color .14s; }
#touch-ui .tc-foot .tc-act { bottom: calc(var(--sb) + var(--u) * 2.15); }
#touch-ui .tc-car .tc-act { bottom: calc(var(--sb) + var(--u) * 2.9); }
#touch-ui .tc-act.show { opacity: 1; transform: none; }
#touch-ui.on .tc-grp.act .tc-act.show { pointer-events: auto; }
#touch-ui .tc-act.dn { background: rgba(234, 165, 63, .26); transform: scale(.96); }
#touch-ui .tc-act-txt { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; min-width: 0; pointer-events: none; }
#touch-ui .tc-act-k { display: none; font: 500 10px/1 var(--f-disp, sans-serif); letter-spacing: .3em; text-transform: uppercase; color: var(--tc-acc); }
#touch-ui .tc-act.hold .tc-act-k { display: block; }
#touch-ui .tc-act-l { font: 500 15px/1.1 var(--f-disp, sans-serif); letter-spacing: .12em; text-transform: uppercase; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; max-width: 100%; text-shadow: 0 1px 2px rgba(0, 0, 0, .6); }
#touch-ui .tc-act-b { position: relative; flex: none; width: calc(var(--u) * 1.06); height: calc(var(--u) * 1.06); border-radius: 50%;
  display: grid; place-items: center; background: rgba(236, 238, 233, .92); color: #0d0f10; pointer-events: none; }
#touch-ui .tc-act-i { width: 62%; height: 62%; display: grid; place-items: center; }
#touch-ui .tc-act-i svg { width: 100%; height: 100%; }
#touch-ui .tc-ring { position: absolute; left: -6px; top: -6px; width: calc(100% + 12px); height: calc(100% + 12px); transform: rotate(-90deg); opacity: 0; transition: opacity .2s; }
#touch-ui .tc-act.holding .tc-ring { opacity: 1; }
#touch-ui .tc-ring circle { fill: none; stroke-width: 3; }
#touch-ui .tc-ring .rb { stroke: rgba(255, 255, 255, .2); }
#touch-ui .tc-ring .rf { stroke: var(--tc-acc); stroke-linecap: round; }

/* floating stick */
#touch-ui .tc-stick, #touch-ui .tc-steer { position: absolute; left: 0; top: 0; pointer-events: none; opacity: 0; transition: opacity .18s; }
#touch-ui .tc-stick.on, #touch-ui .tc-steer.on { opacity: 1; transition: none; }
#touch-ui .tc-base { position: absolute; width: calc(var(--R) * 2px); height: calc(var(--R) * 2px); margin: calc(var(--R) * -1px) 0 0 calc(var(--R) * -1px);
  border-radius: 50%; border: 1.5px solid rgba(236, 238, 233, .42);
  background: radial-gradient(closest-side, rgba(12, 14, 15, .08), rgba(12, 14, 15, .3) 80%, rgba(236, 238, 233, .1) 100%);
  box-shadow: 0 0 0 1px rgba(0, 0, 0, .12), 0 8px 30px rgba(0, 0, 0, .25); transition: border-color .15s; }
#touch-ui .tc-base::after { content: ''; position: absolute; inset: 14%; border-radius: 50%; border: 1px dashed rgba(236, 238, 233, .2); }
#touch-ui .tc-stick.run .tc-base { border-color: var(--tc-acc); }
#touch-ui .tc-knob { position: absolute; width: calc(var(--u) * .92); height: calc(var(--u) * .92); margin: calc(var(--u) * -.46) 0 0 calc(var(--u) * -.46);
  border-radius: 50%; background: rgba(236, 238, 233, .86); box-shadow: 0 4px 16px rgba(0, 0, 0, .4), inset 0 -2px 0 rgba(0, 0, 0, .12); }
#touch-ui .tc-stick.run .tc-knob { background: var(--tc-acc); }
#touch-ui .tc-track { position: absolute; height: calc(var(--u) * .5); width: calc(var(--Rs) * 2px + var(--u) * .5);
  margin: calc(var(--u) * -.25) 0 0 calc(var(--Rs) * -1px - var(--u) * .25); border-radius: 999px;
  background: rgba(12, 14, 15, .3); border: 1.5px solid rgba(236, 238, 233, .42); box-shadow: 0 8px 30px rgba(0, 0, 0, .25); }
#touch-ui .tc-track::before { content: ''; position: absolute; left: 50%; top: 22%; bottom: 22%; width: 1.5px; margin-left: -.75px; background: rgba(236, 238, 233, .35); }
#touch-ui .tc-track b { position: absolute; top: 30%; bottom: 30%; left: 50%; width: 0; background: var(--tc-acc); border-radius: 999px; opacity: .85; }

/* idle hints where the thumb goes */
#touch-ui .tc-ghost { position: absolute; left: calc(var(--sl) + var(--u) * .7); bottom: calc(var(--sb) + var(--u) * 1.7);
  display: grid; place-items: center; pointer-events: none; opacity: 0; transition: opacity .5s; }
#touch-ui .tc-ghost span { position: absolute; top: calc(100% + 8px); left: 0; right: 0; text-align: center; opacity: .8; }
#touch-ui .tc-ghost-stick { width: calc(var(--R) * 2px); height: calc(var(--R) * 2px); border-radius: 50%; border: 1.5px dashed rgba(236, 238, 233, .34); }
#touch-ui .tc-ghost-stick i { width: calc(var(--u) * .5); height: calc(var(--u) * .5); border-radius: 50%; background: rgba(236, 238, 233, .3); }
#touch-ui .tc-ghost-steer { width: calc(var(--Rs) * 2px + var(--u) * .5); height: calc(var(--u) * .5); border-radius: 999px; border: 1.5px dashed rgba(236, 238, 233, .34); }
#touch-ui .tc-ghost-steer > svg { width: calc(var(--Rs) * 2px); height: calc(var(--u) * .9); opacity: .55; }
#touch-ui.foot.idle-l .tc-ghost-stick, #touch-ui.car.idle-l .tc-ghost-steer { opacity: .75; }

@media (prefers-reduced-motion: reduce) { #touch-ui, #touch-ui * { transition-duration: .01s !important; } }
`;

export class TouchControls {
  constructor(input) {
    this.input = input;
    this.visible = false;
    this.mode = 'foot';
    this.ptr = new Map();         // pointerId -> handler {move(e), end(e)}
    this.dead = new Set();        // pointers that were down when the controls hid: ignored until lifted
    this.stick = null; this.steer = null; this.look = null;
    this.sprintToggle = false;
    this._W = 0; this._H = 0;
    this._act = { text: undefined, prog: undefined, id: undefined };
    this._stickIdle = 0;
    this._introPulse = false;
    document.documentElement.classList.add('touch');
    this._build();
    this._bindGlobal();
    this._layout();
  }

  // ------------------------------------------------------------------------------------------------ DOM
  _build() {
    if (!document.getElementById('touch-ui-style')) {
      const st = document.createElement('style');
      st.id = 'touch-ui-style';
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    const L = this.el = document.createElement('div');
    L.id = 'touch-ui';
    L.className = 'foot';
    L.setAttribute('aria-hidden', 'true');
    const act = `<div class="tc-act" role="button" aria-label="Interact">
        <div class="tc-act-txt"><span class="tc-act-k">Hold</span><span class="tc-act-l"></span></div>
        <div class="tc-act-b"><svg class="tc-ring" viewBox="0 0 48 48"><circle class="rb" cx="24" cy="24" r="21.5"/><circle class="rf" cx="24" cy="24" r="21.5"/></svg><span class="tc-act-i"></span></div>
      </div>`;
    L.innerHTML = `
      <div class="tc-pad"></div>
      <div class="tc-ghost tc-ghost-stick"><i></i><span class="tc-lbl">Move</span></div>
      <div class="tc-ghost tc-ghost-steer">${IC.steer}<span class="tc-lbl">Steer</span></div>
      <div class="tc-stick"><div class="tc-base"></div><div class="tc-knob"></div></div>
      <div class="tc-steer"><div class="tc-track"><b></b></div><div class="tc-knob"></div></div>
      <div class="tc-grp tc-foot act">
        <div class="tc-btn tc-sprint" role="button" aria-label="Sprint">${IC.sprint}
          <svg class="tc-stam" viewBox="0 0 48 48"><circle class="sb" cx="24" cy="24" r="22.5"/><circle class="sf" cx="24" cy="24" r="22.5"/></svg></div>
        <div class="tc-btn tc-jump" role="button" aria-label="Jump">${IC.jump}</div>
        ${act}
      </div>
      <div class="tc-grp tc-car">
        <div class="tc-btn tc-hb" role="button" aria-label="Handbrake">${IC.hb}</div>
        <div class="tc-btn tc-ped tc-brake" role="button" aria-label="Brake and reverse">${IC.grip}<span class="tc-lbl">Brake</span><span class="tc-lbl tc-rev">Reverse</span></div>
        <div class="tc-btn tc-ped tc-gas" role="button" aria-label="Accelerate">${IC.grip}${IC.grip}<span class="tc-lbl">Gas</span></div>
        <div class="tc-btn tc-cam" role="button" aria-label="Camera">${IC.cam}</div>
        ${act}
      </div>
      <div class="tc-btn tc-pause" role="button" aria-label="Pause">${IC.pause}</div>`;
    const q = (s) => L.querySelector(s);
    this.$ = {
      pad: q('.tc-pad'), foot: q('.tc-foot'), car: q('.tc-car'),
      stick: q('.tc-stick'), base: q('.tc-stick .tc-base'), knob: q('.tc-stick .tc-knob'),
      steer: q('.tc-steer'), track: q('.tc-track'), sknob: q('.tc-steer .tc-knob'), sbar: q('.tc-track b'),
      sprint: q('.tc-sprint'), stam: q('.tc-stam'), stamFg: q('.tc-stam .sf'),
      acts: [...L.querySelectorAll('.tc-act')].map((a) => ({
        el: a, k: a.querySelector('.tc-act-k'), l: a.querySelector('.tc-act-l'), i: a.querySelector('.tc-act-i'), rf: a.querySelector('.rf'),
      })),
    };
    const C = 2 * Math.PI * 21.5;
    for (const a of this.$.acts) { a.rf.style.strokeDasharray = `${C}`; a.rf.style.strokeDashoffset = `${C}`; }
    this.$.stamFg.style.strokeDasharray = `${2 * Math.PI * 22.5}`;
    (document.body || document.documentElement).appendChild(L);

    // pad: stick / steering on the left, look everywhere else
    this.$.pad.addEventListener('pointerdown', (e) => this._padDown(e));
    // buttons
    this._holdBtn(q('.tc-jump'), 'Space', 'jump');
    this._holdBtn(q('.tc-hb'), 'Space', 'hb');
    this._holdBtn(q('.tc-gas'), 'KeyW', 'gas');
    this._holdBtn(q('.tc-brake'), 'KeyS', 'brake');
    this._holdBtn(q('.tc-cam'), 'KeyC', 'cam');
    for (const a of this.$.acts) this._holdBtn(a.el, 'KeyE', 'act');
    this._tapBtn(this.$.sprint, () => { this.sprintToggle = !this.sprintToggle; this._syncSprint(); }, true);
    this._tapBtn(q('.tc-pause'), () => this._pause(), false);
  }

  _bindGlobal() {
    const opts = { passive: false };
    window.addEventListener('pointermove', (e) => { const h = this.ptr.get(e.pointerId); if (h) { h.move?.(e); if (e.cancelable) e.preventDefault(); } }, opts);
    const up = (e) => {
      this.dead.delete(e.pointerId);
      const h = this.ptr.get(e.pointerId);
      if (h) { this.ptr.delete(e.pointerId); h.end?.(e); }
    };
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
    // no long-press menus, text selection, pinch zoom (iOS gesture events) or double-tap zoom over the game
    this.el.addEventListener('contextmenu', (e) => e.preventDefault());
    this.el.addEventListener('selectstart', (e) => e.preventDefault());
    for (const t of ['gesturestart', 'gesturechange', 'gestureend']) document.addEventListener(t, (e) => { if (this.visible) e.preventDefault(); }, opts);
    // iOS: stop the page rubber-banding under a drag while playing (menus keep their own scrolling)
    document.addEventListener('touchmove', (e) => { if (this.visible && e.cancelable) e.preventDefault(); }, opts);
    window.addEventListener('blur', () => this._releaseAll());
    document.addEventListener('visibilitychange', () => { if (document.hidden) this._releaseAll(); });
  }

  _claim(e, h) {
    this.ptr.set(e.pointerId, h);
    try { e.target.setPointerCapture?.(e.pointerId); } catch { /* synthetic / already gone */ }
  }

  _holdBtn(el, code, name) {
    let n = 0;
    el.addEventListener('pointerdown', (e) => {
      if (!this.visible || this.dead.has(e.pointerId) || this.ptr.has(e.pointerId)) return;
      e.preventDefault(); e.stopPropagation();
      if (n++ === 0) { el.classList.add('dn'); this.input.hold('tc:' + name, code, true); }
      this._claim(e, { end: () => { if (--n <= 0) { n = 0; el.classList.remove('dn'); this.input.hold('tc:' + name, code, false); } } });
    });
  }

  _tapBtn(el, fn, onDown) {
    el.addEventListener('pointerdown', (e) => {
      if (!this.visible || this.dead.has(e.pointerId) || this.ptr.has(e.pointerId)) return;
      e.preventDefault(); e.stopPropagation();
      el.classList.add('dn');
      if (onDown) fn();
      this._claim(e, { end: (ev) => { el.classList.remove('dn'); if (!onDown && ev?.type === 'pointerup' && this.visible) fn(); } });
    });
  }

  _pause() {
    const hud = platform.ctx?.hud;
    if (hud?.pause) { hud.pause(); return; }
    window.dispatchEvent(new KeyboardEvent('keydown', { code: 'Escape', key: 'Escape', bubbles: true }));
  }

  // ------------------------------------------------------------------------------------------------ pad
  _padDown(e) {
    if (!this.visible || this.dead.has(e.pointerId) || this.ptr.has(e.pointerId)) return;
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    e.preventDefault();
    const x = e.clientX, y = e.clientY;
    const left = x < this._W * LEFT_ZONE;
    if (left && this.mode === 'foot' && !this.stick) return this._stickStart(e, x, y);
    if (left && this.mode === 'car' && !this.steer) return this._steerStart(e, x, y);
    if (left) return; // a second finger on the stick side: ignored
    this._lookStart(e, x, y);
  }

  // ---- floating stick (on foot)
  _stickStart(e, x, y) {
    const S = this.stick = { id: e.pointerId, ox: x, oy: y, x, y, run: false };
    this.$.base.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    this.$.stick.classList.add('on');
    this._stickApply();
    this._claim(e, {
      move: (ev) => { S.x = ev.clientX; S.y = ev.clientY; this._stickApply(); },
      end: () => this._stickEnd(),
    });
  }
  _stickApply() {
    const S = this.stick; if (!S) return;
    const R = this._R;
    let dx = S.x - S.ox, dy = S.y - S.oy, d = Math.hypot(dx, dy);
    // the base follows a thumb that slides past the rim (so it never has to come back to find the centre)
    const lim = R * 1.3;
    if (d > lim) {
      const k = (d - lim) / d;
      S.ox += dx * k; S.oy += dy * k; dx = S.x - S.ox; dy = S.y - S.oy; d = lim;
      this.$.base.style.transform = `translate3d(${S.ox}px, ${S.oy}px, 0)`;
    }
    const m = Math.min(1, d / R);
    const kx = d > 1e-3 ? dx / d : 0, ky = d > 1e-3 ? dy / d : 0;
    const vis = Math.min(d, R);
    this.$.knob.style.transform = `translate3d(${S.ox + kx * vis}px, ${S.oy + ky * vis}px, 0)`;
    const mm = m < STICK_DEAD ? 0 : (m - STICK_DEAD) / (1 - STICK_DEAD);
    const walk = Math.min(1, mm / STICK_WALK);
    const A = this.input.axes;
    A.moveX = kx * walk; A.moveY = -ky * walk;
    const run = S.run ? m > SPRINT_OFF : m > SPRINT_ON;
    if (run !== S.run) { S.run = run; this.$.stick.classList.toggle('run', run); this.input.hold('tc:stickrun', 'ShiftLeft', run); }
  }
  _stickEnd() {
    const A = this.input.axes;
    A.moveX = 0; A.moveY = 0;
    this.input.hold('tc:stickrun', 'ShiftLeft', false);
    this.$.stick.classList.remove('on', 'run');
    this.stick = null;
    if (this.sprintToggle) { this.sprintToggle = false; this._syncSprint(); } // sprint lock ends when you stop
  }

  // ---- steering slider (car)
  _steerStart(e, x, y) {
    const S = this.steer = { id: e.pointerId, ox: x, oy: y, x };
    this.$.track.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    this.$.steer.classList.add('on');
    this._steerApply();
    this._claim(e, {
      move: (ev) => { S.x = ev.clientX; this._steerApply(); },
      end: () => this._steerEnd(),
    });
  }
  _steerApply() {
    const S = this.steer; if (!S) return;
    const Rs = this._Rs;
    let dx = S.x - S.ox;
    const lim = Rs * 1.15;
    if (Math.abs(dx) > lim) { S.ox += dx - Math.sign(dx) * lim; dx = Math.sign(dx) * lim; this.$.track.style.transform = `translate3d(${S.ox}px, ${S.oy}px, 0)`; }
    const s = clamp(dx / Rs, -1, 1);
    this.$.sknob.style.transform = `translate3d(${S.ox + s * Rs}px, ${S.oy}px, 0)`;
    const bar = this.$.sbar;
    bar.style.left = s < 0 ? `${50 + s * 50 * (Rs * 2) / (Rs * 2 + this._u * 0.5)}%` : '50%';
    bar.style.width = `${Math.abs(s) * 50 * (Rs * 2) / (Rs * 2 + this._u * 0.5)}%`;
    const a = Math.abs(s) < STEER_DEAD ? 0 : (Math.abs(s) - STEER_DEAD) / (1 - STEER_DEAD);
    // a little expo: fine corrections near the centre, full lock at the end of the travel
    this.input.axes.steer = Math.sign(s) * a * (0.45 + 0.55 * a);
  }
  _steerEnd() {
    this.input.axes.steer = 0;
    this.$.steer.classList.remove('on');
    this.steer = null;
  }

  // ---- look drag (right side)
  _lookStart(e, x, y) {
    if (this.look) { this.dead.add(this.look.id); this.ptr.delete(this.look.id); } // the newest finger takes over
    const P = this.look = { id: e.pointerId, x, y, t: e.timeStamp || performance.now(), v: 0 };
    this.input.lookHeld = true;
    this._claim(e, {
      move: (ev) => {
        const nx = ev.clientX, ny = ev.clientY;
        const dx = nx - P.x, dy = ny - P.y;
        P.x = nx; P.y = ny;
        const t = ev.timeStamp || performance.now();
        const dtE = Math.max(4, t - P.t); P.t = t;
        P.v = P.v * 0.6 + (Math.hypot(dx, dy) / dtE) * 1000 * 0.4;
        // mild acceleration for fast flicks (turning round on a small screen), linear for slow aiming
        const gain = 1 + 0.45 * smooth01(700, 2600, P.v);
        const k = this._lookK * gain;
        this.input.addLook(dx * k, dy * k * 0.85);
      },
      end: () => { if (this.look === P) { this.look = null; this.input.lookHeld = false; } },
    });
  }

  // ------------------------------------------------------------------------------------------------ state
  _releaseAll() {
    for (const [id, h] of this.ptr) { this.dead.add(id); try { h.end?.(null); } catch { /* ignore */ } }
    this.ptr.clear();
    this.stick = null; this.steer = null; this.look = null;
    this.$.stick.classList.remove('on', 'run'); this.$.steer.classList.remove('on');
    this.sprintToggle = false; this._syncSprint();
    this.input.releaseVirtual();
  }

  _syncSprint() {
    this.$.sprint.classList.toggle('on', this.sprintToggle);
    this.input.hold('tc:sprint', 'ShiftLeft', this.sprintToggle);
  }

  _layout() {
    const W = window.innerWidth, H = window.innerHeight;
    if (W === this._W && H === this._H) return;
    this._W = W; this._H = H;
    const mn = Math.min(W, H);
    this._u = clamp(mn * 0.13, 48, 68);
    this._R = clamp(mn * 0.155, 50, 84);           // stick travel radius (px)
    this._Rs = clamp(mn * 0.2, 64, 120);           // steering half-travel (px)
    this._lookK = LOOK_FULL_WIDTH / (BASE_SENS * Math.max(W, H, 1));
    this.el.style.setProperty('--R', this._R.toFixed(1));
    this.el.style.setProperty('--Rs', this._Rs.toFixed(1));
  }

  _visibleFor(ctx) {
    if (!ctx || platform.busy || platform.adActive || ctx.paused) return false;
    const hud = ctx.hud;
    if (hud && (!hud.started || hud.screen || hud._hudHidden)) return false;
    if (hud?.fadeEl?.classList?.contains('on')) return false;
    if (ctx.control !== 'foot' && ctx.control !== 'car') return false;
    const st = ctx.game?.state;
    if (st && !PLAYING.has(st)) return false;
    const rig = ctx.cameraRig;
    if (rig && ((rig._cin && !rig._cin.done) || rig.mode === 'cinematic')) return false;
    return true;
  }

  /** Per frame, at the start of the animation frame (core/input.js). */
  frame(dt) {
    const ctx = platform.ctx;
    this._layout();
    const vis = this._visibleFor(ctx);
    const mode = ctx?.control === 'car' ? 'car' : 'foot';
    if (vis !== this.visible) {
      this.visible = vis;
      this.el.classList.toggle('on', vis);
      if (!vis) this._releaseAll();
    }
    if (mode !== this.mode) {
      this._releaseAll();
      this.mode = mode;
      this.el.classList.toggle('foot', mode === 'foot');
      this.el.classList.toggle('car', mode === 'car');
      this.$.foot.classList.toggle('act', mode === 'foot');
      this.$.car.classList.toggle('act', mode === 'car');
      this._act.text = undefined; this._act.id = undefined;
    }
    if (!vis) return;

    // idle hint where the left thumb goes (fades once it has been used a few times)
    const leftBusy = !!(this.stick || this.steer);
    this._stickIdle = leftBusy ? 0 : this._stickIdle + dt;
    this.el.classList.toggle('idle-l', !leftBusy && this._stickIdle > 0.6);

    this._updateAct(ctx);

    if (mode === 'foot') {
      // stamina around the sprint button
      const p = ctx.player, stam = clamp(p?.stamina ?? 1, 0, 1);
      const show = stam < 0.985;
      if (show !== this._stShow) { this._stShow = show; this.$.stam.classList.toggle('show', show); }
      if (show) {
        const key = Math.round(stam * 100);
        if (key !== this._stKey) {
          this._stKey = key;
          this.$.stamFg.style.strokeDashoffset = `${(2 * Math.PI * 22.5 * (1 - stam)).toFixed(2)}`;
          this.$.stam.classList.toggle('low', !!p?.exhausted || stam < 0.2);
        }
      }
    } else {
      // The cold open's lane-hold (game/sequence.js) hands the wheel over on the first steering KEY; the analog
      // slider mimics that with a one-frame virtual A/D press when it first leaves the centre.
      const s = this.input.axes.steer;
      if (Math.abs(s) > 0.25) {
        if (!this._introPulse && ctx.game?.state === 'intro' && ctx.car?.autopilot) {
          this._introPulse = true;
          this.input.pulse(s < 0 ? 'KeyA' : 'KeyD', 0.01);
        }
      } else if (Math.abs(s) < 0.1) this._introPulse = false;
    }
  }

  _updateAct(ctx) {
    const ia = ctx.interact;
    const P = ia?.prompt;
    const text = P?.text || null;
    const A = this._act;
    const a = this.$.acts[this.mode === 'car' ? 1 : 0];
    if (text !== A.text || ia?.current !== A.id) {
      A.text = text; A.id = ia?.current;
      const m = text ? /\[([^\]]{1,8})\]/.exec(text) : null;
      let label = '', hold = false;
      if (m) {
        label = (text.slice(0, m.index) + ' ' + text.slice(m.index + m[0].length)).replace(/\s+/g, ' ').trim();
        const hm = /^hold\s+/i.exec(label);
        if (hm) { hold = true; label = label.slice(hm[0].length); }
      }
      for (const b of this.$.acts) if (b !== a) b.el.classList.remove('show', 'holding');
      a.el.classList.toggle('show', !!m);
      a.el.classList.toggle('hold', hold);
      if (m) {
        a.l.textContent = label;
        a.el.setAttribute('aria-label', label);
        const icon = ACT_ICON[A.id] || IC.hand;
        if (a.i.__icon !== icon) { a.i.innerHTML = icon; a.i.__icon = icon; }
      }
      A.prog = undefined;
    }
    const prog = P?.progress != null && isFinite(P.progress) ? clamp(P.progress, 0, 1) : null;
    const key = prog == null ? null : Math.round(prog * 200);
    if (key !== A.prog) {
      A.prog = key;
      a.el.classList.toggle('holding', prog != null);
      const C = 2 * Math.PI * 21.5;
      a.rf.style.strokeDashoffset = `${(C * (1 - (prog ?? 0))).toFixed(2)}`;
    }
  }
}
