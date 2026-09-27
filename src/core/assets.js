import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

// Base URL that works in dev and in a relative-path production build.
export const BASE = import.meta.env.BASE_URL || './';

// Lighter asset variants for the MEDIUM ('mid') and LOW ('lo') presets (built by tools/make_variants.mjs).
// Ultra/High never set a tier, so they always load the original full-quality files.
let _variants = null, _tier = null;
export function setVariantTier(manifest, tier) { _variants = manifest?.files || null; _tier = tier || null; }
function variantOf(p) {
  if (!_tier || !_variants) return p;
  const m = /^assets\/(.+)$/.exec(p);
  const v = m && _variants[m[1]]?.[_tier];
  return v ? 'assets/' + v : p;
}
export const url = (p) => {
  if (p.startsWith('http') || p.startsWith('data:') || p.startsWith('blob:')) return p;
  return BASE + variantOf(p.replace(/^\.?\//, ''));
};

/**
 * Cached asset loading with a global progress counter (ctx.assets.progress).
 * Paths are relative to /public, e.g. 'assets/models/car.glb'.
 */
export class Assets {
  constructor(renderer) {
    this.renderer = renderer;
    this.cache = new Map();
    this.progress = { loaded: 0, total: 0, get fraction() { return this.total ? this.loaded / this.total : 1; } };
    this.manager = new THREE.LoadingManager();
    this.gltfLoader = new GLTFLoader(this.manager);
    this.gltfLoader.setMeshoptDecoder(MeshoptDecoder);
    this.ktx2 = new KTX2Loader(this.manager).setTranscoderPath(url('libs/basis/')).detectSupport(renderer);
    this.gltfLoader.setKTX2Loader(this.ktx2);
    this.texLoader = new THREE.TextureLoader(this.manager);
    this.hdrLoader = new HDRLoader(this.manager);
    this.maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
  }

  _track(key, promiseFactory) {
    if (this.cache.has(key)) return this.cache.get(key);
    this.progress.total++;
    const p = promiseFactory().then(
      (v) => { this.progress.loaded++; return v; },
      (e) => { this.progress.loaded++; console.error('[assets] failed', key, e); throw e; },
    );
    this.cache.set(key, p);
    return p;
  }

  /** Resolves to the parsed GLTF ({scene, animations, ...}). Clone gltf.scene yourself if you need several copies. */
  gltf(path) { return this._track('gltf:' + path, () => this.gltfLoader.loadAsync(url(path))); }

  json(path) { return this._track('json:' + path, () => fetch(url(path)).then((r) => { if (!r.ok) throw new Error(r.status + ' ' + path); return r.json(); })); }

  /**
   * Texture with sensible defaults. opts: {srgb=false, repeat=true, anisotropy=max(8), flipY=true(default of TextureLoader)}
   */
  texture(path, opts = {}) {
    const key = 'tex:' + path + JSON.stringify(opts);
    return this._track(key, () => this.texLoader.loadAsync(url(path)).then((t) => {
      t.colorSpace = opts.srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      if (opts.repeat !== false) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
      t.anisotropy = Math.min(opts.anisotropy ?? 8, this.maxAnisotropy);
      if (opts.flipY === false) t.flipY = false;
      t.needsUpdate = true;
      return t;
    }));
  }

  /** PBR set from public/assets/tex/<name>/: {map (sRGB), normalMap, armMap (R=AO, G=rough, B=metal)} */
  async pbr(name, opts = {}) {
    const base = `assets/tex/${name}/`;
    const [map, normalMap, armMap] = await Promise.all([
      this.texture(base + 'diffuse.jpg', { ...opts, srgb: true }),
      this.texture(base + 'nor_gl.jpg', opts),
      this.texture(base + 'arm.jpg', opts),
    ]);
    return { map, normalMap, armMap };
  }

  hdr(path) {
    return this._track('hdr:' + path, () => this.hdrLoader.loadAsync(url(path)).then((t) => { t.mapping = THREE.EquirectangularReflectionMapping; return t; }));
  }
}
