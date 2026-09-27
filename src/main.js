import * as THREE from 'three';
import { config } from './core/config.js';
import { Events } from './core/events.js';
import { Input } from './core/input.js';
import { Assets } from './core/assets.js';
import { RoadPath } from './core/road.js';
import { Debug, flags } from './core/debug.js';
import Physics from './physics/world.js';

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

async function boot() {
  const container = document.getElementById('app');
  const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance', stencil: false, depth: true });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping; // tone mapping happens in render/post.js
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap; // r18x: PCFSoft removed; PCF is filtered (use light.shadow.radius)
  const q = config.quality;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, q.maxDpr) * q.pixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(config.camera.fov, window.innerWidth / window.innerHeight, config.camera.near, config.camera.far);
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
      console.error(`[boot] system "${key}" failed to load/init:`, e);
      ctx.events.emit('boot:progress', { key, error: String(e) });
      delete ctx[key]; delete ctx.systems[key];
    }
  }

  const fixedCam = ctx.debug.applyFixedCamera(camera);
  const onResize = () => {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    ctx.events.emit('resize', { width: w, height: h });
  };
  window.addEventListener('resize', onResize);

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

  ctx.events.emit('boot:done', {});
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

  let readyFrames = 0;
  function frame() {
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

    ctx.debug.update(rawDt, renderer);
    ctx.input.endFrame();
    if (++readyFrames === 5) window.__READY = true; // headless tests wait for this
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
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

boot().catch((e) => {
  console.error('[boot] fatal', e);
  document.body.insertAdjacentHTML('beforeend', `<pre style="position:fixed;inset:20px;color:#f88;background:#000c;padding:20px;z-index:99999;white-space:pre-wrap">Fatal error: ${e?.stack || e}</pre>`);
  window.__READY = true;
});
