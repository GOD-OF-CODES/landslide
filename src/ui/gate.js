// Pre-load quality screen: shown before the renderer or any heavy asset is created, so the chosen preset decides
// what gets downloaded (ultra/high = original full-quality assets, medium/low = lighter variants).
// Low is preselected unless the player chose something before. Skipped with ?quality=, ?autostart and ?cam flags.
import { QUALITY_PRESETS } from '../core/config.js';

const INFO = {
  low: { title: 'Low', blurb: 'Runs on most laptops and integrated graphics. Lighter textures, fewer effects.' },
  medium: { title: 'Medium', blurb: 'For recent laptops. Sharper textures, soft shadows and ambient occlusion.' },
  high: { title: 'High', blurb: 'Full-quality assets and effects. For Apple Silicon or a dedicated GPU.' },
  ultra: { title: 'Ultra', blurb: 'Everything at maximum, native Retina resolution. For powerful GPUs.' },
};
const ORDER = ['low', 'medium', 'high', 'ultra'];

/** Rough device hint from the WebGL renderer string. It's only a suggestion; the player decides. */
export function recommendQuality() {
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return 'low';
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const r = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)).toLowerCase();
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    const mem = navigator.deviceMemory || 8;
    const mobile = /android|iphone|ipad|mobile/i.test(navigator.userAgent);
    if (mobile || mem <= 4 || /swiftshader|llvmpipe|software|mali|adreno|powervr|intel.*(hd|uhd)/.test(r)) return 'low';
    if (/rtx|radeon rx|rx [5-9]\d{3}|m[1-9] (pro|max|ultra)|m[2-9]\b|apple m[2-9]/.test(r)) return 'high';
    if (/gtx|radeon|iris|apple m1|apple gpu|apple/.test(r)) return 'medium';
    return 'low';
  } catch { return 'low'; }
}

export function showQualityGate(initial = 'low') {
  const rec = recommendQuality();
  let sel = ORDER.includes(initial) ? initial : 'low';
  const root = document.createElement('div');
  root.id = 'quality-gate';
  root.innerHTML = `
  <style>
    #quality-gate{position:fixed;inset:0;z-index:10000;display:flex;align-items:center;justify-content:center;
      background:radial-gradient(120% 90% at 50% 20%,#1d2226 0%,#0b0d0e 70%);font-family:var(--f-body,system-ui,sans-serif);color:#eceee9}
    #quality-gate .qg{width:min(920px,calc(100vw - 32px));}
    #quality-gate h1{font:500 clamp(34px,5vw,64px)/1 var(--f-disp,'Arial Narrow',sans-serif);letter-spacing:.42em;text-indent:.42em;text-align:center;margin:0 0 6px;text-transform:uppercase}
    #quality-gate .sub{text-align:center;color:rgba(236,238,233,.6);letter-spacing:.18em;text-transform:uppercase;font-size:12px;margin-bottom:34px}
    #quality-gate .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    @media (max-width:760px){#quality-gate .grid{grid-template-columns:repeat(2,1fr)}}
    #quality-gate button.opt{all:unset;box-sizing:border-box;cursor:pointer;padding:18px 16px 16px;border:1px solid rgba(236,238,233,.14);
      background:rgba(255,255,255,.03);border-radius:6px;min-height:168px;display:flex;flex-direction:column;gap:8px;transition:border-color .2s,background .2s}
    #quality-gate button.opt:hover{border-color:rgba(236,238,233,.35)}
    #quality-gate button.opt.on{border-color:#eaa53f;background:rgba(234,165,63,.08);box-shadow:inset 0 0 0 1px #eaa53f}
    #quality-gate button.opt:focus-visible{outline:2px solid #eaa53f;outline-offset:2px}
    #quality-gate .t{font:500 22px/1 var(--f-disp,'Arial Narrow',sans-serif);letter-spacing:.14em;text-transform:uppercase}
    #quality-gate .b{font-size:13px;line-height:1.45;color:rgba(236,238,233,.72);flex:1}
    #quality-gate .m{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:rgba(236,238,233,.5)}
    #quality-gate .tag{align-self:flex-start;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#0b0d0e;background:#eaa53f;padding:3px 6px;border-radius:3px}
    #quality-gate .row{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-top:26px;flex-wrap:wrap}
    #quality-gate .note{font-size:12px;color:rgba(236,238,233,.55);max-width:560px;line-height:1.5}
    #quality-gate .go{all:unset;cursor:pointer;padding:14px 30px;background:#eaa53f;color:#0b0d0e;font:600 15px/1 var(--f-disp,'Arial Narrow',sans-serif);letter-spacing:.24em;text-transform:uppercase;border-radius:4px}
    #quality-gate .go:hover{filter:brightness(1.08)}
    #quality-gate .go:focus-visible{outline:2px solid #fff;outline-offset:3px}
  </style>
  <div class="qg" role="dialog" aria-label="Choose graphics quality">
    <h1>Landslide</h1>
    <div class="sub">Choose graphics quality</div>
    <div class="grid" role="radiogroup">
      ${ORDER.map((k) => `
        <button class="opt" role="radio" data-q="${k}" aria-checked="false">
          <span class="t">${INFO[k].title}</span>
          ${k === rec ? '<span class="tag">Suggested for this device</span>' : ''}
          <span class="b">${INFO[k].blurb}</span>
          <span class="m">≈ ${QUALITY_PRESETS[k].download} MB download</span>
        </button>`).join('')}
    </div>
    <div class="row">
      <div class="note">You can change this later from the title screen. If the game stutters, pick a lower setting; Low and Medium also adjust resolution automatically to keep it smooth.</div>
      <button class="go" type="button">Start</button>
    </div>
  </div>`;
  document.body.appendChild(root);
  const opts = [...root.querySelectorAll('button.opt')];
  const paint = () => opts.forEach((b) => { const on = b.dataset.q === sel; b.classList.toggle('on', on); b.setAttribute('aria-checked', String(on)); });
  paint();
  opts.find((b) => b.dataset.q === sel)?.focus();
  return new Promise((resolve) => {
    const done = () => {
      window.removeEventListener('keydown', onKey);
      root.style.transition = 'opacity .35s'; root.style.opacity = '0';
      setTimeout(() => root.remove(), 380);
      resolve(sel);
    };
    const onKey = (e) => {
      const i = ORDER.indexOf(sel);
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { sel = ORDER[Math.min(3, i + 1)]; paint(); e.preventDefault(); }
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { sel = ORDER[Math.max(0, i - 1)]; paint(); e.preventDefault(); }
      else if (e.key === 'Enter') { e.preventDefault(); done(); }
    };
    window.addEventListener('keydown', onKey);
    opts.forEach((b) => {
      b.addEventListener('click', () => { sel = b.dataset.q; paint(); });
      b.addEventListener('dblclick', () => { sel = b.dataset.q; done(); });
    });
    root.querySelector('.go').addEventListener('click', done);
  });
}
