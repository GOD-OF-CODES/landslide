import * as THREE from 'three';
import { PARAMS } from './config.js';

// URL flags (see DESIGN.md "Debug & testing"):
//  ?debug            stats overlay + window.__ctx
//  ?autostart        skip title screen (needed for headless screenshots)
//  ?only=a,b         only init these system keys (plus core); for isolated testing
//  ?skip=<checkpoint> start at a gameplay checkpoint: stall | onfoot | refuel | escape | gap | tunnel
//  ?cam=x,y,z,tx,ty,tz   fixed camera (position, look-at target); implies free camera, no game camera
//  ?camS=s,d,h,lookAhead camera placed relative to the road: at road (s,d) height h above road, looking lookAhead meters ahead
//  ?freecam          WASD/QE + right-drag fly camera
//  ?quality=ultra|high|medium|low
//  ?nopost           render without post-processing
//  ?mute             no audio
//  ?time=seconds     advance scripted timeline (e.g. landslide) by this much at start (for screenshots)
export const flags = {
  debug: PARAMS.has('debug'),
  autostart: PARAMS.has('autostart') || PARAMS.has('cam') || PARAMS.has('camS'),
  only: PARAMS.get('only') ? PARAMS.get('only').split(',') : null,
  skip: PARAMS.get('skip'),
  cam: PARAMS.get('cam') ? PARAMS.get('cam').split(',').map(Number) : null,
  camS: PARAMS.get('camS') ? PARAMS.get('camS').split(',').map(Number) : null,
  freecam: PARAMS.has('freecam'),
  nopost: PARAMS.has('nopost'),
  mute: PARAMS.has('mute'),
  time: PARAMS.get('time') ? Number(PARAMS.get('time')) : 0,
};
flags.fixedCam = !!(flags.cam || flags.camS);

export class Debug {
  constructor(ctx) {
    this.ctx = ctx;
    this.frames = 0; this.acc = 0; this.fps = 0;
    this.stats = { fps: 0, ms: 0, calls: 0, triangles: 0, geometries: 0, textures: 0, programs: 0, frame: 0 };
    window.__STATS = this.stats;
    if (flags.debug) {
      window.__ctx = ctx;
      this.el = document.createElement('div');
      this.el.style.cssText = 'position:fixed;left:8px;bottom:8px;z-index:9999;font:11px/1.35 ui-monospace,monospace;color:#cfe;background:rgba(0,0,0,.55);padding:6px 8px;border-radius:4px;pointer-events:none;white-space:pre';
      document.body.appendChild(this.el);
    }
    window.__ctx = ctx; // always available for automated testing
    this.yaw = 0; this.pitch = 0;
  }

  /** Apply ?cam / ?camS once the road is known. Returns true if a fixed camera was applied. */
  applyFixedCamera(camera) {
    const { road } = this.ctx;
    if (flags.cam) {
      const [x, y, z, tx, ty, tz] = flags.cam;
      camera.position.set(x, y, z);
      camera.lookAt(tx ?? x, ty ?? y, tz ?? z - 1);
      return true;
    }
    if (flags.camS && road) {
      const [s, d = -1.5, h = 1.6, ahead = 30] = flags.camS;
      const p = road.worldAt(s, d);
      camera.position.set(p.x, p.y + h, p.z);
      const t = road.worldAt(s + ahead, d);
      camera.lookAt(t.x, t.y + h * 0.6, t.z);
      return true;
    }
    return false;
  }

  updateFreeCam(camera, dt) {
    const input = this.ctx.input;
    if (input.mouseDown(2) || input.locked) {
      this.yaw -= input.dx * 0.003; this.pitch -= input.dy * 0.003;
      this.pitch = THREE.MathUtils.clamp(this.pitch, -1.5, 1.5);
      camera.quaternion.setFromEuler(new THREE.Euler(this.pitch, this.yaw, 0, 'YXZ'));
    }
    const sp = (input.keys.has('ShiftLeft') ? 60 : 12) * dt;
    const f = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const r = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    if (input.keys.has('KeyW')) camera.position.addScaledVector(f, sp);
    if (input.keys.has('KeyS')) camera.position.addScaledVector(f, -sp);
    if (input.keys.has('KeyD')) camera.position.addScaledVector(r, sp);
    if (input.keys.has('KeyA')) camera.position.addScaledVector(r, -sp);
    if (input.keys.has('KeyE')) camera.position.y += sp;
    if (input.keys.has('KeyQ')) camera.position.y -= sp;
  }

  update(dt, renderer) {
    this.frames++; this.acc += dt;
    const info = renderer.info;
    this.stats.frame++;
    if (this.acc >= 0.5) {
      this.fps = this.frames / this.acc; this.frames = 0; this.acc = 0;
      Object.assign(this.stats, {
        fps: Math.round(this.fps), ms: +(1000 / Math.max(this.fps, 1)).toFixed(1),
        calls: info.render.calls, triangles: info.render.triangles,
        geometries: info.memory.geometries, textures: info.memory.textures, programs: info.programs?.length ?? 0,
        state: this.ctx.game?.state ?? null,
      });
      if (this.el) {
        const c = this.ctx.camera.position;
        const pr = this.ctx.road ? this.ctx.road.project(c, {}) : null;
        this.el.textContent = `${this.stats.fps} fps  ${this.stats.ms} ms\ncalls ${info.render.calls}  tris ${(info.render.triangles / 1e6).toFixed(2)}M\ncam ${c.x.toFixed(1)}, ${c.y.toFixed(1)}, ${c.z.toFixed(1)}` + (pr ? `\nroad s=${pr.s.toFixed(1)} d=${pr.d.toFixed(1)} dy=${pr.dy.toFixed(1)}` : '') + `\nstate ${this.stats.state}`;
      }
    }
  }
}
