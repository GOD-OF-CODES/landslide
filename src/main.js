import * as THREE from 'three';
import { config, DEVICE } from './core/config.js';
import { Events } from './core/events.js';
import { Input } from './core/input.js';
import { Assets, setVariantTier, BASE } from './core/assets.js';
import { RoadPath } from './core/road.js';
import { Debug, flags } from './core/debug.js';
import Physics from './physics/world.js';
import { isStaleChunkError, recoverFromStaleBuild, showFatal } from './core/recover.js';
import { platform } from './core/platform.js';

// ---------------------------------------------------------------------------------------------
// System registry. Each module default-exports a class: new X(ctx); optional async init();
// optional fixedUpdate(h) (before each physics step), update(dt) (every frame), dispose().
// Instances are stored on ctx[key]. A failing module is logged and skipped so others keep working.
// See DESIGN.md "Systems" for ownership and APIs.
// ---------------------------------------------------------------------------------------------
const SYSTEMS = [
  // key          module loader                                  init order
  ['hud',        () => import('./ui/hud.js')],
  ['env',        () => import('./render/environment.js')],
  ['terrain',    () => import('./world/terrain.js')],
  ['vegetation', () => import('./world/vegetation.js')],
  ['props',      () => import('./world/props.js')],
  ['landslide',  () => import('./world/landslide.js')],
  ['particles',  () => import('./world/particles.js')],
  ['car',        () => import('./physics/vehicle.js')],
  ['player',     () => import('./physics/character.js')],
  ['cameraRig',  () => import('./game/camera.js')],
  ['audio',      () => import('./audio/audio.js')],
  ['interact',   () => import('./game/interact.js')],
  ['game',       () => import('./game/sequence.js')],
  ['post',       () => import('./render/post.js')],
];
// Per-frame update order (differs from init order: camera before env/post, game logic first).
const UPDATE_ORDER = ['game', 'interact', 'car', 'player', 'landslide', 'particles', 'vegetation', 'props', 'terrain',
  'cameraRig', 'env', 'audio', 'hud', 'post'];

// Every big asset the systems request, fetched up front in parallel. The keys match the systems' own calls
// (same path + options), so they receive these shared promises instead of downloading one after another.
function prefetch(assets) {
  const P = [
    assets.gltf('assets/world/terrain.glb'), assets.json('assets/world/scatter.json'),
    assets.gltf('assets/models/trees.glb'), assets.gltf('assets/models/props.glb'),
    assets.gltf('assets/models/rocks.glb'), assets.gltf('assets/models/car.glb'),
    assets.gltf('assets/models/hands.glb'),
    assets.json('assets/sky/sky.json'), assets.hdr('assets/sky/env_2k.hdr'),
    ...['asphalt_02', 'concrete_wall_006', 'pine_bark', 'rough_wood', 'mud_forest', 'brown_mud_03',
      'brown_mud_rocks_01', 'brown_mud_02'].map((n) => assets.pbr(n)),
  ];
  for (const p of P) p?.catch?.(() => {}); // failures surface in the owning system
}

async function boot() {
  const container = document.getElementById('app');
  platform.loadingStart(); // CrazyGames: the loading screen starts here and ends at 'boot:done'

  // Quality was already chosen by src/boot.js (pre-load gate) before this module started.
  if (config.quality.assetTier) {
    try {
      const manifest = await fetch(BASE + 'assets/q/variants.json').then((r) => (r.ok ? r.json() : null));
      setVariantTier(manifest, config.quality.assetTier);
    } catch (e) { console.warn('[boot] no asset variants, using full-quality files', e); }
  }
  // start fetching every system module now (dynamic imports are cached, init still runs in order below)
  for (const [, loader] of SYSTEMS) loader().catch(() => {});

  // (MOBILEPERF) WebGL2 is required (three r186). Say so plainly instead of failing somewhere inside the renderer.
  if (typeof WebGL2RenderingContext === 'undefined') {
    throw new FriendlyError('LANDSLIDE needs WebGL 2, which this browser does not support. Please update your browser '
      + '(Safari 15 or later on iPhone and iPad, or a recent Chrome, Firefox or Edge) and open the game again.');
  }
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance', stencil: false, depth: true });
  } catch (e) {
    console.warn('[boot] WebGL renderer', e);
    throw new FriendlyError('Your browser could not start 3D graphics (WebGL 2). Close other tabs or apps, check that '
      + 'hardware acceleration is turned on in the browser settings, and reload.');
  }
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping; // tone mapping happens in render/post.js
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap; // r18x: PCFSoft removed; PCF is filtered (use light.shadow.radius)
  const q = config.quality;
  // render resolution: CSS size x min(devicePixelRatio, maxDpr) x the preset's pixelRatio (x the adaptive scale below)
  const presetRatio = () => Math.min(window.devicePixelRatio || 1, q.maxDpr) * q.pixelRatio;
  const vp0 = viewportSize();
  renderer.setPixelRatio(presetRatio());
  renderer.setSize(vp0.w, vp0.h);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(config.camera.fov, vp0.w / vp0.h, config.camera.near, config.camera.far);
  camera.position.set(0, 50, 0);
  scene.add(camera); // so camera-attached objects (rain, viewmodel) render

  const ctx = {
    THREE, renderer, scene, camera, config, flags,
    events: new Events(),
    input: new Input(renderer.domElement),
    assets: new Assets(renderer),
    physics: new Physics(null),
    road: null,
    control: 'none',          // 'car' | 'foot' | 'none'  (who consumes gameplay input)
    paused: false,
    time: { now: 0, dt: 0, frame: 0, scale: 1 },
    systems: {},
  };
  ctx.physics.ctx = ctx;
  ctx.debug = new Debug(ctx);
  ctx.device = DEVICE;
  const gl = watchContextLoss(renderer, ctx);
  // (MOBILEPERF) a hidden tab/app: silence the audio (through the platform's mute reasons, so it reconciles with ads
  // and the CrazyGames mute setting); the frame loop below skips all work while hidden.
  const onVisibility = () => { try { platform._setMute?.('hidden', !!document.hidden); } catch {} };
  document.addEventListener('visibilitychange', onVisibility);
  onVisibility();
  if (!flags.only) prefetch(ctx.assets);

  // Core data first
  ctx.road = new RoadPath(await ctx.assets.json('assets/world/road.json'));
  await ctx.physics.init();

  // Instantiate + init systems in order
  const only = flags.only ? new Set(['hud', ...flags.only]) : null;
  for (const [key, loader] of SYSTEMS) {
    if (only && !only.has(key)) continue;
    try {
      const mod = await loader();
      const Cls = mod.default;
      if (typeof Cls !== 'function') throw new Error('module has no default export class');
      const sys = new Cls(ctx);
      ctx[key] = sys; ctx.systems[key] = sys;
      if (sys.init) await sys.init();
      ctx.events.emit('boot:progress', { key, done: true });
    } catch (e) {
      // a system chunk from an older deployment is gone: reload once to get the current build
      if (isStaleChunkError(e) && recoverFromStaleBuild(e)) return;
      console.error(`[boot] system "${key}" failed to load/init:`, e);
      ctx.events.emit('boot:progress', { key, error: String(e) });
      delete ctx[key]; delete ctx.systems[key];
    }
  }

  const fixedCam = ctx.debug.applyFixedCamera(camera);
  // Resize. applySize() always resizes (adaptive resolution calls it after changing the pixel ratio); onResize() is the
  // event handler: it re-reads the devicePixelRatio (it changes when a window moves to another display, with browser
  // zoom, and on some phones on rotation) and only does work when the size or the ratio really changed. On touch
  // devices the size comes from visualViewport (iOS Safari: the visible area, not the 100vh layout box) and is checked
  // again shortly after each event, because iOS reports stale sizes at the moment it fires resize/orientationchange.
  let lastW = vp0.w, lastH = vp0.h, lastRatio = presetRatio();
  let ares = null;
  const applySize = () => {
    const { w, h } = viewportSize();
    lastW = w; lastH = h;
    renderer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    ctx.events.emit('resize', { width: w, height: h });
  };
  const onResize = () => {
    const { w, h } = viewportSize();
    const ratio = presetRatio();
    if (w === lastW && h === lastH && ratio === lastRatio) return;
    if (ratio !== lastRatio) { lastRatio = ratio; renderer.setPixelRatio(ratio * (ares ? ares.scale : 1)); }
    applySize();
  };
  const onResizeEvent = () => {
    onResize();
    if (DEVICE.touch) { setTimeout(onResize, 250); setTimeout(onResize, 700); }
  };
  window.addEventListener('resize', onResizeEvent);
  if (DEVICE.touch) {
    window.addEventListener('orientationchange', onResizeEvent);
    window.visualViewport?.addEventListener?.('resize', onResizeEvent);
  }

  // Warm up: compile shaders before first visible frame. compileAsync alone only builds the main-pass programs of
  // objects that are visible right now, and no shadow-map depth programs at all: every caster type (instanced,
  // alpha-tested, custom depth material) then compiled its depth variant the first time it entered the sun's
  // shadow frustum, a 50-300 ms hitch in the middle of the rockfall (QA). So hidden / empty meshes are forced
  // visible, unculled and non-empty for one compile + one throwaway render (which builds the shadow programs)
  // while the loading screen is still up, then restored.
  const unforce = forceRenderable(scene);
  try { await renderer.compileAsync(scene, camera); } catch (e) { console.warn('[boot] compileAsync', e); }
  try {
    // through the post chain when it exists: the programs then also meet the HDR render targets they draw into
    renderer.shadowMap.needsUpdate = true;
    if (ctx.post?.render && !flags.nopost) ctx.post.render(0); else renderer.render(scene, camera);
  } catch (e) { console.warn('[boot] shader warm-up render', e); }
  unforce();

  platform.attach(ctx);
  ctx.events.emit('boot:done', {});
  platform.loadingStop();
  if (flags.autostart) ctx.events.emit('ui:start', { auto: true });

  const timer = new THREE.Timer();
  timer.connect?.(document);
  const H = config.physics.step;
  let acc = 0;
  const order = UPDATE_ORDER.map((k) => k);
  const call = (key, fn, arg) => {
    const s = ctx.systems[key];
    if (!s || !s[fn]) return;
    try { s[fn](arg); } catch (e) {
      console.error(`[loop] ${key}.${fn} threw`, e);
      s.__errors = (s.__errors || 0) + 1;
      if (s.__errors > 20) { console.error(`[loop] disabling ${key}.${fn}`); s[fn] = null; }
    }
  };

  // Adaptive resolution (desktop: medium/low only; ultra/high always render at their full preset resolution; mobile:
  // every preset, see config.js MOBILE_ADJUST). Lowers the pixel ratio in 10% steps when the frame rate stays under
  // ~40 fps (mobile ~27 fps), and restores it when there is headroom. The floor is adaptiveMin (desktop 0.55).
  const aMin = q.adaptiveMin ?? 0.55, aDown = q.adaptiveDown ?? 1 / 40, aUp = q.adaptiveUp ?? 1 / 57;
  ares = { on: !!config.quality.adaptiveRes && !flags.fixedCam, scale: 1, t: 0, n: 0, sum: 0, min: aMin };
  ctx.adaptiveRes = ares;
  function adaptResolution(rawDt) {
    if (!ares.on || ctx.paused || document.hidden) return;
    ares.sum += rawDt; ares.n++; ares.t += rawDt;
    if (ares.t < 2) return;
    const avg = ares.sum / ares.n;
    ares.t = ares.sum = ares.n = 0;
    let next = ares.scale;
    if (avg > aDown) next = Math.max(aMin, ares.scale * 0.9);
    else if (avg < aUp) next = Math.min(1, ares.scale * 1.06);
    if (Math.abs(next - ares.scale) > 0.005) {
      ares.scale = next;
      renderer.setPixelRatio(lastRatio * next);
      applySize();
    }
  }

  let readyFrames = 0;
  function frame() {
    // (MOBILEPERF) no work while the page is hidden (browsers usually stop rAF then anyway; some iframes only throttle
    // it) or while the WebGL context is lost (the "tap to reload" overlay is up; see watchContextLoss)
    if (document.hidden || gl.lost) {
      if (gl.lost) platform.update(ctx);
      ctx.input.endFrame();
      requestAnimationFrame(frame);
      return;
    }
    timer.update();
    const rawDt = Math.min(timer.getDelta(), 0.1);
    const dt = ctx.paused ? 0 : rawDt * ctx.time.scale;
    ctx.time.dt = dt; ctx.time.now += dt; ctx.time.frame++;

    if (!ctx.paused) {
      acc += dt;
      let n = 0;
      while (acc >= H && n < config.physics.maxSubSteps) {
        for (const k of order) call(k, 'fixedUpdate', H);
        ctx.physics.step();
        acc -= H; n++;
      }
      if (n === config.physics.maxSubSteps) acc = 0;
      ctx.physics.syncLinks();
    }
    for (const k of order) {
      if (k === 'post') continue;
      if (k === 'cameraRig' && (fixedCam || flags.freecam)) continue;
      call(k, 'update', dt);
    }
    if (flags.freecam) ctx.debug.updateFreeCam(camera, rawDt);

    if (ctx.post && !flags.nopost) call('post', 'render', rawDt);
    else renderer.render(scene, camera);

    platform.update(ctx);
    ctx.debug.update(rawDt, renderer);
    if (readyFrames > 60) adaptResolution(rawDt);
    ctx.input.endFrame();
    if (++readyFrames === 5) window.__READY = true; // headless tests wait for this
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/** Canvas size in CSS pixels. Touch devices: the visual viewport (the area really visible on iOS Safari, where the
 *  layout viewport / 100vh can extend under the toolbars); pinch-zoomed (scale > 1) or missing: the window. */
function viewportSize() {
  const vv = DEVICE.touch ? window.visualViewport : null;
  if (vv && vv.scale <= 1.01 && vv.width > 0 && vv.height > 0) return { w: Math.floor(vv.width), h: Math.floor(vv.height) };
  return { w: window.innerWidth, h: window.innerHeight };
}

/** An error whose message is written for the player (shown by showFatal). */
class FriendlyError extends Error {}

/**
 * WebGL context loss (MOBILEPERF). Mobile browsers drop the context under memory pressure or after the app was in the
 * background (iOS Safari especially), and a GPU reset can do it on desktop. Without handling, the last frame just
 * freezes. Instead: the game is held (paused, so the platform reports gameplayStop and the loop skips all work) and a
 * "Tap to reload" screen comes up. It stays even if the context is restored: render-target contents (the PMREM sky
 * lighting, the reflection history, baked arrays) are gone, so a reload (quality kept, no quality screen) is the
 * reliable way back. Returns {lost} (read every frame).
 */
function watchContextLoss(renderer, ctx) {
  const state = { lost: false, restored: false };
  const canvas = renderer.domElement;
  let el = null;
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault(); // allows the browser to restore it (and stops some browsers from showing their own error)
    if (state.lost) return;
    state.lost = true;
    console.warn('[boot] WebGL context lost');
    ctx.paused = true;
    try { if (ctx.input) { ctx.input.enabled = false; ctx.input.exitLock?.(); } } catch {}
    try { platform._setMute?.('glLost', true); } catch {}
    try { ctx.events?.emit?.('gl:lost', {}); } catch {}
    if (el) return;
    el = document.createElement('div');
    el.id = 'gl-lost';
    el.setAttribute('role', 'alertdialog');
    el.style.cssText = 'position:fixed;inset:0;z-index:99998;display:flex;align-items:center;justify-content:center;'
      + 'background:rgba(11,13,14,.94);color:#eceee9;font:15px/1.5 system-ui,sans-serif;padding:24px;cursor:pointer;'
      + 'touch-action:manipulation;-webkit-tap-highlight-color:transparent;text-align:center';
    const verb = DEVICE.touch ? 'Tap' : 'Click';
    el.innerHTML = `<div style="max-width:520px">
      <div style="font:500 28px/1 'Barlow Condensed','Arial Narrow',sans-serif;letter-spacing:.3em;margin-bottom:18px">LANDSLIDE</div>
      <p style="color:rgba(236,238,233,.75);margin:0 0 22px">The graphics were reset by the device (this can happen when the game was
      in the background or memory ran low). Your quality setting is kept.</p>
      <button type="button" style="all:unset;cursor:pointer;padding:14px 30px;background:#eaa53f;color:#0b0d0e;font-weight:600;
      letter-spacing:.18em;text-transform:uppercase;border-radius:4px">${verb} to reload</button></div>`;
    const reload = (ev) => { ev?.preventDefault?.(); try { sessionStorage.setItem('landslide.gateDone', '1'); } catch {} location.reload(); };
    el.addEventListener('click', reload);
    el.addEventListener('touchend', reload);
    document.body.appendChild(el);
  }, false);
  canvas.addEventListener('webglcontextrestored', () => { state.restored = true; console.warn('[boot] WebGL context restored; reload to continue'); }, false);
  return state;
}

/** Temporarily makes every mesh renderable (visible incl. ancestors, not frustum-culled, instance count >= 1).
 *  Returns a function that restores the previous state. Used only for the boot shader warm-up. */
function forceRenderable(scene) {
  const undo = [];
  scene.traverse((o) => {
    if (!(o.isMesh || o.isPoints || o.isLine || o.isSprite)) return;
    for (let p = o; p && p !== scene; p = p.parent) {
      if (!p.visible) { const q = p; q.visible = true; undo.push(() => { q.visible = false; }); }
    }
    if (o.frustumCulled) { o.frustumCulled = false; undo.push(() => { o.frustumCulled = true; }); }
    if (o.isInstancedMesh && o.count === 0 && o.instanceMatrix?.count >= 1) { o.count = 1; undo.push(() => { o.count = 0; }); }
  });
  return () => { for (let i = undo.length - 1; i >= 0; i--) undo[i](); };
}

/** Entry point, called by src/boot.js once the quality is chosen. */
export function start() {
  return boot().catch((e) => {
    if (e instanceof FriendlyError) { console.warn('[boot]', e.message); showFatal(e); return; }
    console.error('[boot] fatal', e);
    if (isStaleChunkError(e) && recoverFromStaleBuild(e)) return;
    showFatal(e);
  });
}
