import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { config } from './config.js';

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

// (MOBILEPERF) GPU-memory cap for the textures embedded in GLBs on phones and tablets (config DEVICE.mobile). iOS Safari
// kills a tab at ~1-1.5 GB, and on Low the originals of trees.glb alone (no Low variant) bring four 2048 px maps
// (foliage albedo/normal, fallen-spruce atlas) = ~85 MB with mipmaps. Halving them to 1024 px saves ~65 MB, and at a
// phone's render resolution (Low: ~630x290) the difference is invisible. Desktop: no cap (0), nothing changes.
function glbTextureCap() {
  const q = config.quality;
  if (!q.mobile) return 0;
  return q.key === 'low' || q.key === 'medium' ? 1024 : 2048;
}

/** Downscales every texture of a parsed glTF whose larger side exceeds `cap` px (aspect kept), before first upload. */
async function capGltfTextures(gltf, cap) {
  const seen = new Set(), jobs = [];
  gltf?.scene?.traverse((o) => {
    const ms = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
    for (const m of ms) for (const k in m) {
      const t = m[k];
      if (!t || !t.isTexture || seen.has(t)) continue;
      seen.add(t);
      const img = t.image, w = img?.width || 0, h = img?.height || 0;
      if (Math.max(w, h) > cap) jobs.push(resample(t, img, w, h, cap / Math.max(w, h)));
    }
  });
  await Promise.all(jobs);
}
async function resample(t, img, w, h, s) {
  const nw = Math.max(1, Math.round(w * s)), nh = Math.max(1, Math.round(h * s));
  let out = null;
  try {
    // straight (non-premultiplied) alpha like GLTFLoader's own ImageBitmaps, so cut-out foliage keeps its fringe colour
    if (typeof createImageBitmap === 'function') {
      const b = await createImageBitmap(img, { resizeWidth: nw, resizeHeight: nh, resizeQuality: 'high', premultiplyAlpha: 'none' });
      if (b.width === nw && b.height === nh) out = b; else b.close?.();
    }
  } catch { out = null; }
  if (!out) {
    try { // (older Safari ignores the resize options) canvas fallback
      const c = document.createElement('canvas'); c.width = nw; c.height = nh;
      const g = c.getContext('2d'); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      g.drawImage(img, 0, 0, nw, nh);
      out = c;
    } catch (e) { console.warn('[assets] texture downscale failed', e); return; }
  }
  if (img !== out) { try { img.close?.(); } catch {} }
  t.image = out;
  t.needsUpdate = true;
}

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
  gltf(path) {
    return this._track('gltf:' + path, () => this.gltfLoader.loadAsync(url(path)).then(async (g) => {
      const cap = glbTextureCap();
      if (cap) { try { await capGltfTextures(g, cap); } catch (e) { console.warn('[assets] texture cap', path, e); } }
      return g;
    }));
  }

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
