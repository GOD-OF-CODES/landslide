"""
LANDSLIDE - TREES workstream. Procedural conifers, fallen tree, shrubs, deadwood and impostor atlases.

Everything is built from scratch here (no downloaded models). The only external inputs are CC0 textures:
  raw_assets/tex/pine_bark (Poly Haven, CC0), public/assets/tex/brown_mud_rocks_01 (Poly Haven, CC0).

Usage (headless, one Blender process at a time):
  /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/blender/trees.py -- [options]
Options:
  --stages cards,models,impostors,export,preview   (default: all; 'models' is implied by the later ones)
  --force-cards        re-render the needle/twig card atlas even if it is cached in scratch/trees/build
  --samples N          Cycles samples for card / impostor renders (default 48)
  --preview-samples N  Cycles samples for the preview renders (default 64)
  --gpu                render with Metal
  --cards a,b          (with --force-cards) re-render only these cards into the cached atlas

Outputs:
  scratch/trees/build/foliage_albedo.png, foliage_normal.png, cards.json  (card atlas, rendered from 3D needles)
  scratch/trees/build/grass_albedo.png, grass_normal.png, grass_cards.json (wild-grass card atlas, 3D blades)
  public/assets/models/trees.glb                                           (meshopt + webp)
  public/assets/models/impostors/{impostors.json, albedo.webp, normal.webp}
  scratch/trees/preview_*.png

Nodes in trees.glb: conifer_0..3 (+ _lod1) living Norway spruce / silver fir, conifer_4 (+ _lod1) a dead snag,
fallen_tree, fallen_tree_a/_b, shrub_0..2, deadwood_0..1, grass_cards (1-triangle carrier of the 'grass' material).
Materials: bark, bark_grey (fir bark, dead branches, snags), foliage (card atlas), grass (grass atlas), soil, wood_cut.

REALISM v2 (tree look matched to real Picea abies / Abies alba seen from a road at 3-30 m):
  - ragged, open spray cards (irregular / missing / snapped laterals, drooping shoots) instead of ovate fronds;
  - comb-spruce curtains (pendulous branchlets 0.2-0.8 m) as the dominant lower/mid-crown texture;
  - crown asymmetry (long "light" side at three +Z; the engine turns it downhill), gaps and near-empty whorls;
  - grey dead branches with beard lichen (Bryoria dark, Usnea pale) below the live crown; a broken-top variant
    (flat ragged top + upturned side leader) and a snag;
  - needle albedo ~(0.03, 0.07, 0.03) linear per needle, card mean ~(0.04, 0.06, 0.023).

Coordinates: built in Blender Z-up. The glTF exporter maps Blender (x, y, z) -> three (x, z, -y).
"""
import bpy
import math
import os
import sys
import json
import time
import subprocess
import numpy as np
from mathutils import Vector, Matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
BUILD = os.path.join(ROOT, 'scratch', 'trees', 'build')
SCRATCH = os.path.join(ROOT, 'scratch', 'trees')
OUT_MODELS = os.path.join(ROOT, 'public', 'assets', 'models')
OUT_IMP = os.path.join(OUT_MODELS, 'impostors')
RAWTEX = os.path.join(ROOT, 'raw_assets', 'tex')
PUBTEX = os.path.join(ROOT, 'public', 'assets', 'tex')
HDRI = os.path.join(ROOT, 'raw_assets', 'hdri', 'overcast_soil_puresky_2k.hdr')
for d in (BUILD, SCRATCH, OUT_MODELS, OUT_IMP):
    os.makedirs(d, exist_ok=True)


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    a = {'stages': 'cards,models,impostors,export,preview', 'force_cards': False, 'samples': 48,
         'preview_samples': 64, 'gpu': False}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == '--stages': a['stages'] = argv[i + 1]; i += 1
        elif k == '--force-cards': a['force_cards'] = True
        elif k == '--samples': a['samples'] = int(argv[i + 1]); i += 1
        elif k == '--preview-samples': a['preview_samples'] = int(argv[i + 1]); i += 1
        elif k == '--gpu': a['gpu'] = True
        elif k == '--preview': a['preview'] = argv[i + 1]; i += 1
        elif k == '--cards': a['cards'] = argv[i + 1].split(','); i += 1
        i += 1
    a['stages'] = set(a['stages'].split(','))
    return a


ARGS = parse_args()
T0 = time.time()


def log(*a):
    print('[trees %6.1fs]' % (time.time() - T0), *a, flush=True)


# ---------------------------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------------------------
def nrm(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def rot_axis(v, axis, ang):
    """Rotate vector(s) v about unit axis by ang (Rodrigues)."""
    axis = nrm(axis)
    c, s = math.cos(ang), math.sin(ang)
    v = np.asarray(v, np.float64)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(v, axis) if v.ndim == 1 else (v @ axis)[:, None]) * (1 - c)


def perp(v):
    v = nrm(v)
    a = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0, 0])
    return nrm(np.cross(v, a))


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def hexlin(h, jitter=None, rng=None):
    """sRGB hex -> linear rgb (+ optional multiplicative jitter)."""
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    c = srgb_decode(c)
    if jitter is not None and rng is not None:
        c = c * (1 + rng.uniform(-jitter, jitter, 3)) * (1 + rng.uniform(-jitter, jitter))
    return c


# ---------------------------------------------------------------------------------------------
# Mesh builder (numpy accumulation -> one bpy mesh). Triangles only.
# Per-vertex: position, custom normal, uv, color (RGBA float, linear). Per-face: material index.
# ---------------------------------------------------------------------------------------------
class MB:
    def __init__(self):
        self.P, self.N, self.UV, self.C, self.F, self.M = [], [], [], [], [], []
        self.n = 0

    def add(self, P, N, UV, C, F, mat=0):
        P = np.asarray(P, np.float64).reshape(-1, 3)
        k = len(P)
        N = np.asarray(N, np.float64).reshape(-1, 3) if N is not None else np.tile([0, 0, 1.0], (k, 1))
        UV = np.asarray(UV, np.float64).reshape(-1, 2) if UV is not None else np.zeros((k, 2))
        C = np.asarray(C, np.float64)
        if C.ndim == 1: C = np.tile(C, (k, 1))
        if C.shape[1] == 3: C = np.concatenate([C, np.ones((k, 1))], 1)
        F = np.asarray(F, np.int64).reshape(-1, 3) + self.n
        self.P.append(P); self.N.append(N); self.UV.append(UV); self.C.append(C)
        self.F.append(F); self.M.append(np.full(len(F), mat, np.int32))
        self.n += k

    def arrays(self):
        if not self.P:
            return (np.zeros((0, 3)),) * 2 + (np.zeros((0, 2)), np.zeros((0, 4)), np.zeros((0, 3), np.int64), np.zeros(0, np.int32))
        return (np.concatenate(self.P), np.concatenate(self.N), np.concatenate(self.UV), np.concatenate(self.C),
                np.concatenate(self.F), np.concatenate(self.M))

    def tri_count(self):
        return sum(len(f) for f in self.F)

    def transform(self, fn_p, fn_n=None):
        """Apply a function to all positions (and normals)."""
        self.P = [fn_p(p) for p in self.P]
        if fn_n is not None:
            self.N = [fn_n(n) for n in self.N]

    def build(self, name, materials, color_name='Col'):
        P, N, UV, C, F, M = self.arrays()
        me = bpy.data.meshes.new(name)
        me.from_pydata(P.tolist(), [], F.tolist())
        me.update()
        nl = len(F) * 3
        lv = np.empty(nl, np.int64)
        me.loops.foreach_get('vertex_index', lv)
        uvl = me.uv_layers.new(name='UVMap')
        uvl.data.foreach_set('uv', UV[lv].astype(np.float32).ravel())
        ca = me.color_attributes.new(color_name, 'FLOAT_COLOR', 'POINT')
        ca.data.foreach_set('color', C.astype(np.float32).ravel())
        me.color_attributes.active_color = ca
        try:
            me.color_attributes.render_color_index = me.color_attributes.active_color_index
        except Exception:
            pass
        me.polygons.foreach_set('material_index', M)
        for m in materials:
            me.materials.append(m)
        me.shade_smooth()
        me.normals_split_custom_set_from_vertices(nrm(N).astype(np.float32).tolist())
        me.update()
        if me.validate(verbose=False, clean_customdata=False):
            log('  mesh %s: validate() fixed problems' % name)
            me.normals_split_custom_set_from_vertices(nrm(N).astype(np.float32).tolist()) if len(me.vertices) == len(N) else None
        ob = bpy.data.objects.new(name, me)
        return ob


def link(ob, coll=None):
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


# ---------------------------------------------------------------------------------------------
# Primitive generators
# ---------------------------------------------------------------------------------------------
def transport_frames(pts):
    """Parallel-transport frames along a polyline. Returns tangents T and normals Nn (k,3)."""
    pts = np.asarray(pts, np.float64)
    k = len(pts)
    T = np.zeros((k, 3))
    T[1:-1] = pts[2:] - pts[:-2]
    T[0] = pts[1] - pts[0]
    T[-1] = pts[-1] - pts[-2]
    T = nrm(T)
    Nn = np.zeros((k, 3))
    Nn[0] = perp(T[0])
    for i in range(1, k):
        a, b = T[i - 1], T[i]
        ax = np.cross(a, b)
        s = np.linalg.norm(ax)
        if s < 1e-9:
            Nn[i] = Nn[i - 1]
        else:
            ang = math.atan2(s, float(np.dot(a, b)))
            Nn[i] = rot_axis(Nn[i - 1], ax / s, ang)
        Nn[i] = nrm(Nn[i] - T[i] * np.dot(Nn[i], T[i]))
    return T, Nn


def tube(mb, pts, radii, sides, color, mat=0, u_rep=1.0, v_scale=1.0, v0=0.0, cap=False,
         radial_fn=None, color_fn=None, seam_twist=0.0):
    """Tapered tube along pts. radial_fn(i, theta) -> radius multiplier (bumps, buttresses).
    UV: u = theta/2pi * u_rep, v = arclength * v_scale + v0. Seam vertices duplicated."""
    pts = np.asarray(pts, np.float64)
    T, Nn = transport_frames(pts)
    B = np.cross(T, Nn)
    k = len(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    th = np.linspace(0, 2 * math.pi, sides + 1)
    P, N, UV = [], [], []
    for i in range(k):
        for j, t in enumerate(th):
            tt = t + seam_twist
            d = Nn[i] * math.cos(tt) + B[i] * math.sin(tt)
            r = radii[i] * (radial_fn(i, t) if radial_fn else 1.0)
            P.append(pts[i] + d * r)
            N.append(d)
            UV.append((t / (2 * math.pi) * u_rep, arc[i] * v_scale + v0))
    P, N, UV = np.array(P), np.array(N), np.array(UV)
    F = []
    w = sides + 1
    for i in range(k - 1):
        for j in range(sides):
            a, b, c, d = i * w + j, i * w + j + 1, (i + 1) * w + j + 1, (i + 1) * w + j
            F.append((a, b, c)); F.append((a, c, d))
    if color_fn is not None:
        C = np.array([color_fn(P[i], i // w) for i in range(len(P))])
    else:
        C = color
    if cap:
        base = len(P)
        P = np.vstack([P, pts[-1] + T[-1] * radii[-1] * 0.3])
        N = np.vstack([N, T[-1]])
        UV = np.vstack([UV, [0.5, arc[-1] * v_scale + v0]])
        if not np.isscalar(C) and np.ndim(C) == 2:
            C = np.vstack([C, C[-1]])
        for j in range(sides):
            F.append(((k - 1) * w + j, (k - 1) * w + j + 1, base))
    mb.add(P, N, UV, C, F, mat)
    return arc[-1]


# ---------------------------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------------------------
def new_mat(name):
    m = bpy.data.materials.get(name)
    if m:
        bpy.data.materials.remove(m)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes.clear()
    return m


def load_image(path, name=None, colorspace='sRGB'):
    img = bpy.data.images.load(path, check_existing=True)
    if name: img.name = name
    img.colorspace_settings.name = colorspace
    return img


def np_to_image(arr, name, path, colorspace='Non-Color', fmt='PNG'):
    """arr: (h, w, 4) float in [0,1], row 0 = TOP of the image. Saves exactly these values."""
    h, w = arr.shape[:2]
    img = bpy.data.images.get(name)
    if img: bpy.data.images.remove(img)
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).astype(np.float32).ravel())
    img.filepath_raw = path
    img.file_format = fmt
    img.save()
    img.colorspace_settings.name = colorspace
    return img


def image_to_np(path):
    """Load an image file -> (h, w, 4) float array, row 0 = TOP. Values as stored (no colour conversion)."""
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = 'Non-Color'
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(h, w, 4)[::-1].copy()


# ---------------------------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    if ARGS['gpu']:
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'METAL'
            prefs.get_devices()
            for d in prefs.devices: d.use = True
            sc.cycles.device = 'GPU'
        except Exception as e:
            log('GPU unavailable', e)
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.cycles.use_denoising = False
    return sc


def render_exr(sc, cam_obj, res_x, res_y, path, samples):
    sc.camera = cam_obj
    sc.render.resolution_x, sc.render.resolution_y = res_x, res_y
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.cycles.samples = samples
    sc.cycles.max_bounces = 0
    sc.cycles.transparent_max_bounces = 128
    sc.cycles.filter_width = 1.2
    sc.render.image_settings.file_format = 'OPEN_EXR'
    sc.render.image_settings.color_depth = '32'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return image_to_np(path)


def ortho_cam(name, center, width, height, look='-Z'):
    cam = bpy.data.cameras.new(name)
    cam.type = 'ORTHO'
    cam.ortho_scale = max(width, height)
    cam.clip_start = 0.001
    cam.clip_end = 500
    ob = bpy.data.objects.new(name, cam)
    if look == '-Z':
        ob.location = (center[0], center[1], center[2] + 50)
        ob.rotation_euler = (0, 0, 0)
    elif look == '+Y':  # camera at -Y looking toward +Y, up = +Z
        ob.location = (center[0], center[1] - 200, center[2])
        ob.rotation_euler = (math.pi / 2, 0, 0)
    cam.sensor_fit = 'HORIZONTAL' if width >= height else 'VERTICAL'
    bpy.context.scene.collection.objects.link(ob)
    return ob


def unpremultiply(img):
    a = img[..., 3:4]
    rgb = np.where(a > 1e-4, img[..., :3] / np.maximum(a, 1e-4), 0)
    return rgb, a[..., 0]


def dilate(rgb, mask, iters=24):
    """Bleed colours from covered pixels into uncovered neighbours (prevents dark mip fringes)."""
    rgb = rgb.copy()
    m = mask.astype(bool).copy()
    for _ in range(iters):
        if m.all(): break
        acc = np.zeros_like(rgb); cnt = np.zeros(m.shape)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            sm = np.roll(np.roll(m, dy, 0), dx, 1)
            sc = np.roll(np.roll(rgb, dy, 0), dx, 1)
            acc += sc * sm[..., None]; cnt += sm
        new = (~m) & (cnt > 0)
        rgb[new] = acc[new] / cnt[new][:, None]
        m |= new
    if not m.all():
        mean = rgb[m].mean(0) if m.any() else np.array([0.2, 0.25, 0.15])
        rgb[~m] = mean
    return rgb


# =============================================================================================
# 1) CARD ATLAS: needle / twig / leaf / fern / lichen geometry rendered to RGBA albedo + normal
# =============================================================================================
ATLAS = 2048
# name: (x0, y0, x1, y1) pixels from TOP-left, world height (m), anchor ('bottom' = base at bottom centre,
# axis up; 'top' = attached along top edge, hangs down; 'center' = top-view leaf, stalk at the centre)
# Layout v2 (REALISM pass): ragged/open spruce sprays, a longer comb curtain, lichen-covered dead twigs, and the
# ground-flora cards used by grass.js (fern, bilberry, coltsfoot, butterbur, forest litter).
CARDS = {
    'spruce':   dict(px=(0, 0, 768, 1024), h=0.90, anchor='bottom'),
    'spruce2':  dict(px=(768, 0, 1280, 1024), h=0.80, anchor='bottom'),
    'fir':      dict(px=(1280, 0, 2048, 1024), h=0.84, anchor='bottom'),
    'hang':     dict(px=(0, 1024, 1024, 1536), h=0.60, anchor='top'),
    'dead':     dict(px=(1024, 1024, 1536, 1536), h=0.60, anchor='bottom'),
    'twigs':    dict(px=(1536, 1024, 2048, 1536), h=0.55, anchor='bottom'),
    'leaf':     dict(px=(0, 1536, 512, 2048), h=0.42, anchor='bottom'),
    'fern_a':   dict(px=(512, 1536, 768, 2048), h=0.72, anchor='bottom'),
    'fern_b':   dict(px=(768, 1536, 1024, 2048), h=0.62, anchor='bottom'),
    'bilberry': dict(px=(1024, 1536, 1536, 2048), h=0.36, anchor='bottom'),
    'coltsfoot': dict(px=(1536, 1536, 1792, 1792), h=0.24, anchor='center'),
    'butterbur': dict(px=(1792, 1536, 2048, 1792), h=0.56, anchor='center'),
    'litter':   dict(px=(1536, 1792, 2048, 2048), h=0.60, anchor='center'),
}
for _k, _c in CARDS.items():
    x0, y0, x1, y1 = _c['px']
    _c['w'] = _c['h'] * (x1 - x0) / (y1 - y0)
    _c['uv'] = (x0 / ATLAS, 1 - y1 / ATLAS, x1 / ATLAS, 1 - y0 / ATLAS)  # Blender UV (u0, v0, u1, v1)


def poly_sample(pts, s):
    """Point and unit tangent at arc length s along polyline pts."""
    pts = np.asarray(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    s = float(np.clip(s, 0, arc[-1]))
    i = int(max(0, min(np.searchsorted(arc, s) - 1, len(pts) - 2)))
    f = (s - arc[i]) / max(seg[i], 1e-9)
    return pts[i] + (pts[i + 1] - pts[i]) * f, nrm(pts[i + 1] - pts[i]), arc[-1]


def poly_len(pts):
    return float(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1).sum())


ZUP = np.array([0, 0, 1.0])


class CardGeo:
    """Accumulates small render geometry (twigs, needles, leaves) with per-element colours."""
    def __init__(self, seed):
        self.mb = MB()
        self.rng = np.random.default_rng(seed)

    def twig(self, pts, r0, r1, col, sides=5):
        pts = np.asarray(pts)
        radii = np.linspace(r0, r1, len(pts))
        tube(self.mb, pts, radii, sides, col, cap=True)

    def needle(self, base, direc, length, width, col, flat=0.0, up=None, tipcol=None, curve=0.0):
        """4-angled (rhombic) needle, slightly curved; flat>0 flattens it (fir)."""
        d = nrm(direc)
        u = nrm(np.cross(d, up)) if up is not None else perp(d)
        v = nrm(np.cross(u, d))
        w1 = width * 0.5
        w2 = width * 0.5 * (1 - flat * 0.75)
        ring = [u * w1, v * w2, -u * w1, -v * w2]
        mid = base + d * length * 0.5 + v * curve * length
        tip = base + d * length + v * curve * length * 1.4
        P = [base + r * 0.75 for r in ring] + [mid + r for r in ring] + [tip]
        N = [nrm(r) for r in ring] * 2 + [d]
        F = []
        for j in range(4):
            a, b = j, (j + 1) % 4
            F += [(a, b, 4 + b), (a, 4 + b, 4 + a), (4 + a, 4 + b, 8)]
        C = np.tile(np.append(col, 1), (9, 1))
        if tipcol is not None:
            C[8, :3] = tipcol
            C[4:8, :3] = col * 0.6 + tipcol * 0.4
        C[0:4, :3] *= 0.8
        self.mb.add(P, N, None, C, F)

    def leaf(self, base, direc, normal, length, width, col, curl=0.2, serr=True, K=12):
        """Ovate leaf/pinnule with midrib fold and (optionally) serrated margin."""
        d = nrm(direc)
        n = nrm(normal - d * np.dot(normal, d))
        s = nrm(np.cross(n, d))
        P = [base]; N = [n]
        for i in range(1, K + 1):
            t = i / K
            P.append(base + d * length * t + n * (-curl * length * t * t)); N.append(n)
        rows = {}
        for side in (1, -1):
            idx = []
            for i in range(1, K):
                t = i / K
                wv = width * 0.5 * (math.sin(math.pi * t ** 0.75) ** 0.8) * (1 + (0.1 * (i % 2) if serr else 0))
                p = base + d * length * t + s * side * wv + n * (0.2 * wv - curl * length * t * t)
                idx.append(len(P)); P.append(p); N.append(nrm(n - s * side * 0.3))
            rows[side] = idx
        F = []
        for side in (1, -1):
            r = rows[side]
            for i in range(K - 1):
                m0, m1 = i + 1, i + 2
                a = r[i]
                b = r[i + 1] if i + 1 < len(r) else K
                tri1 = (m0, a, m1) if side == 1 else (m0, m1, a)
                F.append(tri1)
                if b != K:
                    F.append((a, b, m1) if side == 1 else (a, m1, b))
            F.append((0, r[0], 1) if side == 1 else (0, 1, r[0]))
        C = np.tile(np.append(col, 1), (len(P), 1))
        C[1:K + 1, :3] *= 1.18  # midrib
        for side in (1, -1):
            C[rows[side], :3] *= 0.9
        self.mb.add(P, N, None, C, F)

    def round_leaf(self, center, radius, col, kind='coltsfoot', rot=0.0, cup=0.12, nlobe=None):
        """Top-view palmate leaf (coltsfoot: polygonal-cordate, toothed; butterbur: kidney, scalloped), lying in the
        XY plane facing +Z, stalk attachment at `center`. Radial palmate veins lighter; felted margin paler."""
        rng = self.rng
        K = 72
        th = np.linspace(0, 2 * math.pi, K, endpoint=False) + rot
        nl = nlobe or (7 if kind == 'coltsfoot' else 11)
        rel = th - rot - math.pi * 1.5          # sinus (basal notch) pointing -Y
        rel = np.angle(np.exp(1j * rel))
        if kind == 'coltsfoot':
            teeth = 0.06 * np.abs(np.sin(th * nl * 1.5)) ** 3 + 0.035 * np.sin(th * nl * 4.5)
            base_r = 1.0 + 0.1 * np.cos(rel * 2) + teeth
            notch = 1 - 0.55 * np.exp(-(rel / 0.33) ** 2)
        else:
            teeth = 0.035 * np.abs(np.sin(th * nl)) ** 0.5 + 0.02 * np.sin(th * nl * 3.1)
            base_r = 1.0 + 0.06 * np.cos(rel * 2) + teeth
            notch = 1 - 0.62 * np.exp(-(rel / 0.4) ** 2)
        R = radius * base_r * notch * (1 + rng.normal(0, 0.012, K))
        # rings from centre to margin (cupped: margin raised a little; tears/holes on some butterbur leaves)
        rings = 7
        P = [np.array(center) + np.array([0, radius * 0.18, radius * 0.02])]
        N = [np.array([0, 0, 1.0])]
        C = [np.append(col * 1.25, 1)]
        for ri in range(1, rings + 1):
            f = ri / rings
            for j in range(K):
                r = R[j] * f
                x, y = math.cos(th[j]) * r, math.sin(th[j]) * r + radius * 0.18 * (1 - f)
                z = cup * radius * f * f + 0.02 * radius * math.sin(th[j] * nl) * f + 0.035 * radius * math.sin(th[j] * 3 + 1.3) * f * f
                P.append(np.array(center) + np.array([x, y, z]))
                N.append(nrm(np.array([-x * cup * 2 / radius, -y * cup * 2 / radius, 1.0])))
                # palmate veins: lighter along nl radial lines; interveinal darker; margin slightly paler/yellower
                vein = math.exp(-(math.sin((th[j] - rot) * nl / 2) / 0.07) ** 2)
                c = col * (0.9 + 0.16 * vein * (1 - 0.6 * f) - 0.06 * f) * (1 + 0.06 * rng.normal())
                if kind == 'butterbur' and f > 0.8 and rng.random() < 0.08:
                    c = c * 0.6 + hexlin('6b5a2c') * 0.4     # browning, slug-eaten margin
                C.append(np.append(c, 1))
        F = []
        for j in range(K):
            F.append((0, 1 + j, 1 + (j + 1) % K))
        for ri in range(1, rings):
            for j in range(K):
                a = 1 + (ri - 1) * K + j; b = 1 + (ri - 1) * K + (j + 1) % K
                cc = 1 + ri * K + (j + 1) % K; d = 1 + ri * K + j
                F += [(a, d, cc), (a, cc, b)]
        self.mb.add(P, N, None, np.array(C), F)
        # raised midrib + main veins as thin tubes (visible relief)
        for v in range(-(nl // 2), nl // 2 + 1):
            a = rot + math.pi / 2 + v * (2 * math.pi * 0.42 / max(1, nl // 2))
            L = radius * 0.85 * (1 - 0.35 * abs(v) / max(1, nl // 2))
            p0 = np.array(center) + np.array([0, radius * 0.16, radius * 0.03])
            pts = [p0 + np.array([math.cos(a) * L * t, math.sin(a) * L * t, cup * radius * t * t * 0.9 + 0.004]) for t in (0, 0.5, 1)]
            self.twig(np.array(pts), radius * 0.011, radius * 0.003, col * 1.22, sides=3)

    def shoot_curve(self, base, direc, length, bend_axis=None, bend=0.0, droop=0.0, n=6, wobble=0.0):
        d = nrm(direc)
        pts = [np.asarray(base, float)]
        seg = length / (n - 1)
        for i in range(1, n):
            if bend_axis is not None and bend != 0:
                d = nrm(rot_axis(d, bend_axis, bend / (n - 1)))
            d = nrm(d + np.array([0, 0, -droop / (n - 1)]) + (self.rng.normal(0, wobble, 3) if wobble else 0))
            pts.append(pts[-1] + d * seg)
        return np.array(pts)

    def hang_curve(self, base, length, n=8, sway=0.25, wobble=0.05):
        """Pendulous branchlet in the card plane: hangs toward -Y with a lazy S-curve and a slight outward flick."""
        rng = self.rng
        d = nrm([rng.uniform(-sway, sway), -1, rng.uniform(-0.15, 0.15)])
        pts = [np.asarray(base, float)]
        seg = length / (n - 1)
        for i in range(1, n):
            d = nrm(d + rng.normal(0, wobble, 3) * np.array([1, 0.3, 0.6]) + np.array([0, -0.08, 0]))
            pts.append(pts[-1] + d * seg)
        return np.array(pts)

    def needled_shoot(self, pts, r0, twigcol, needle_len, density, colfn, flat=0.0, plane_up=None,
                      spread=(0.85, 1.3), fwd=0.3, top_bias=0.0, twig_sides=5, tip_new=0.0, width=0.0013, bare=0.0):
        """Needles spirally around a shoot (density = needles per cm). flat>0: fir-like two ranks.
        bare: fraction of the shoot base without needles (old, shed inner shoots)."""
        rng = self.rng
        pts = np.asarray(pts)
        self.twig(pts, r0, r0 * 0.5, twigcol, sides=twig_sides)
        L = poly_len(pts)
        count = int(L * 100 * density * (1 - bare))
        golden = math.pi * (3 - math.sqrt(5))
        up = plane_up if plane_up is not None else ZUP
        for k in range(count):
            s = (bare + (1 - bare) * (k + rng.uniform(0, 0.9)) / max(count, 1)) * L
            p, t, _ = poly_sample(pts, s)
            rel = s / L
            col, tipc = colfn(rel, rng)
            if tip_new > 0 and rel > 1 - tip_new:
                col = col * 0.5 + hexlin('5f7a30') * 0.5
                tipc = col * 1.2
            if flat > 0:
                side = nrm(np.cross(t, up))
                sgn = 1 if k % 2 == 0 else -1
                if rng.random() < 0.3:  # upper-surface needles pointing forward/up, shorter
                    dirn = nrm(t * 0.7 + up * 0.55 + side * sgn * rng.uniform(0.1, 0.7))
                    nl = needle_len * rng.uniform(0.5, 0.75)
                else:
                    ang = rng.uniform(1.1, 1.5)
                    dirn = nrm(t * math.cos(ang) + side * sgn * math.sin(ang) + up * rng.uniform(-0.05, 0.15))
                    nl = needle_len * rng.uniform(0.8, 1.15)
                nl *= (0.7 + 0.3 * math.sin(math.pi * min(rel * 1.25 + 0.1, 1)))
                self.needle(p, dirn, nl, width * 1.9, col, flat=flat, up=up, tipcol=tipc, curve=0.05)
            else:
                perp0 = perp(t)
                b0 = np.cross(t, perp0)
                phi = k * golden + rng.uniform(-0.4, 0.4)
                radial = nrm(perp0 * math.cos(phi) + b0 * math.sin(phi))
                radial = nrm(radial + up * top_bias)
                ang = rng.uniform(*spread)
                dirn = nrm(radial * math.sin(ang) + t * (math.cos(ang) + fwd))
                nl = needle_len * rng.uniform(0.7, 1.2) * (0.65 + 0.35 * min(1, rel * 4)) * (1 - 0.3 * rel ** 5)
                self.needle(p + radial * r0 * 0.5, dirn, nl, width, col, tipcol=tipc, curve=rng.uniform(0.02, 0.1))
        return L


def spruce_needle_col(rel, rng):
    r = rng.random()
    # Norway spruce needles: dark blue-green, not black. Measured needle reflectance ~0.03 (R), 0.07-0.08 (G, 550 nm),
    # ~0.03 (B); the rendered card mean with twigs and self-occlusion lands near (0.035, 0.055, 0.025).
    # Older (inner, rel small) needles duller/olive, current-year needles at the shoot tips brighter.
    if r < 0.035:
        base = hexlin('6b5a2c', 0.2, rng)  # old, browning needle
    elif r < 0.3 + 0.2 * rel:
        base = hexlin('3f5838', 0.14, rng)
    elif r < 0.55:
        base = hexlin('2f4633', 0.12, rng)   # blue-green
    else:
        base = hexlin('354a2c', 0.14, rng)   # olive
    return base, base * 1.3


def fir_needle_col(rel, rng):
    base = hexlin('334b2e', 0.13, rng) if rng.random() > 0.03 else hexlin('6b5a2c', 0.2, rng)
    return base, base * 1.25


def spruce_spray(g, base, direc, length, width, n_lat, new_growth=0.06, lat_density=8.0, droop=0.25):
    """Norway spruce branch end seen from above/below: axis + ragged alternate laterals (irregular lengths, missing,
    snapped, drooping) with sparse 3rd-order shoots. The envelope must NOT be a smooth ovate frond: from the road the
    sky shows through the gaps and the outline is spiky."""
    rng = g.rng
    twig_brown, twig_old = hexlin('4e3c2b'), hexlin('3e3329')
    axis = g.shoot_curve(base, direc, length, bend_axis=ZUP, bend=rng.uniform(-0.25, 0.25), n=12, wobble=0.025)
    g.needled_shoot(axis, 0.0034, twig_old, 0.017, 6.5, spruce_needle_col, top_bias=0.25, tip_new=new_growth * 1.5,
                    bare=0.06)
    L = poly_len(axis)
    s = L * rng.uniform(0.04, 0.09)
    j = 0
    while s < L * 0.93:
        p, t, _ = poly_sample(axis, s)
        rel = s / L
        side = 1 if (j % 2 == 0) != (rng.random() < 0.18) else -1
        j += 1
        s += L / n_lat * rng.uniform(0.55, 1.6)
        if rng.random() < 0.13:
            continue                                            # missing lateral: gap
        ang = rng.uniform(0.55, 1.05)
        d = nrm(rot_axis(t, ZUP, side * ang))
        reach = width * 0.47 * (0.25 + 0.75 * (1 - rel) ** 0.7) * rng.uniform(0.62, 1.15)
        snapped = rng.random() < 0.1
        if snapped:
            reach *= rng.uniform(0.25, 0.45)
        Ll = reach / max(math.sin(ang), 0.55)
        lat = g.shoot_curve(p, d, Ll, bend_axis=ZUP, bend=-side * rng.uniform(0.1, 0.55),
                            droop=droop * rng.uniform(0.4, 1.6), n=8, wobble=0.035)
        g.needled_shoot(lat, 0.0018, twig_brown, 0.015, lat_density, spruce_needle_col, top_bias=0.2,
                        tip_new=0 if snapped else new_growth, width=0.0012, bare=0.1 if rel < 0.4 else 0.0)
        Ll = poly_len(lat)
        s3 = rng.uniform(0.01, 0.025)
        q = 0
        while s3 < Ll * 0.9:
            p3, t3, _ = poly_sample(lat, s3)
            s3 += rng.uniform(0.011, 0.024)
            q += 1
            if rng.random() < 0.14:
                continue
            sd = 1 if (q % 2) != (rng.random() < 0.25) else -1
            d3 = nrm(rot_axis(t3, ZUP, sd * rng.uniform(0.45, 1.0)) + np.array([0, 0, rng.uniform(-0.45, 0.15)]))
            L3 = rng.uniform(0.04, 0.11) * (1 - 0.45 * s3 / Ll)
            sh = g.shoot_curve(p3, d3, L3, n=3, droop=rng.uniform(0.0, 0.25))
            g.needled_shoot(sh, 0.0011, twig_brown, 0.012, lat_density * 1.1, spruce_needle_col, top_bias=0.15,
                            twig_sides=4, tip_new=new_growth * 2, width=0.0011)


def spruce_curtain(g, w, h):
    """Comb spruce: a 2nd-order branch along the top edge with a curtain of pendulous branchlets of very uneven
    length (0.2-1.0 x card height), each with splayed, drooping short shoots. Gaps between strands."""
    rng = g.rng
    y_top = h / 2 - 0.03
    twig_brown, twig_old = hexlin('4e3c2b'), hexlin('3e3329')
    axis = g.shoot_curve([-w / 2 + 0.015, y_top, 0], [1, 0, 0], w * 0.97, n=14, wobble=0.012)
    axis[:, 1] = np.clip(axis[:, 1] - 0.02 * np.sin(np.linspace(0, math.pi, len(axis))), -h / 2, y_top)
    g.needled_shoot(axis, 0.0036, twig_old, 0.016, 6.0, spruce_needle_col, top_bias=0.1, bare=0.03)
    x = -w / 2 + rng.uniform(0.01, 0.04)
    while x < w / 2 - 0.03:
        x += rng.uniform(0.016, 0.05)
        if rng.random() < 0.1:
            continue
        p = np.array([x, y_top - 0.004 - 0.02 * math.sin(math.pi * (x / w + 0.5)), rng.uniform(-0.02, 0.02)])
        L = (h - 0.05) * (rng.uniform(0.18, 1.0) ** 0.8)
        sh = g.hang_curve(p, L, n=9, sway=0.22, wobble=0.045)
        sh[:, 1] = np.maximum(sh[:, 1], -h / 2 + 0.01)
        g.needled_shoot(sh, 0.0017, twig_brown, 0.015, 8.0, spruce_needle_col, top_bias=0.0, tip_new=0.12, bare=0.05)
        Ls = poly_len(sh)
        s3 = rng.uniform(0.02, 0.05)
        q = 0
        while s3 < Ls * 0.88:
            p3, t3, _ = poly_sample(sh, s3)
            s3 += rng.uniform(0.02, 0.045)
            q += 1
            if rng.random() < 0.15:
                continue
            sd = 1 if q % 2 else -1
            d3 = nrm(rot_axis(t3, ZUP, sd * rng.uniform(0.35, 0.85)) + np.array([0, 0, rng.uniform(-0.3, 0.3)]))
            s3c = g.shoot_curve(p3, d3, rng.uniform(0.025, 0.075), n=3, droop=0.1)
            g.needled_shoot(s3c, 0.0011, twig_brown, 0.013, 8.5, spruce_needle_col, twig_sides=4, tip_new=0.15)
        # occasionally a strand forks
        if rng.random() < 0.3:
            p2, _, _ = poly_sample(sh, Ls * rng.uniform(0.25, 0.5))
            sh2 = g.hang_curve(p2, L * rng.uniform(0.3, 0.6), n=6, sway=0.4)
            sh2[:, 1] = np.maximum(sh2[:, 1], -h / 2 + 0.01)
            g.needled_shoot(sh2, 0.0014, twig_brown, 0.014, 8.0, spruce_needle_col, tip_new=0.12)


def fir_frond(g, base, direc, length, width):
    rng = g.rng
    twig_brown, twig_old = hexlin('6a5238'), hexlin('4a4038')
    axis = g.shoot_curve(base, direc, length, bend_axis=ZUP, bend=rng.uniform(-0.18, 0.18), n=10, wobble=0.015)
    g.needled_shoot(axis, 0.0038, twig_old, 0.025, 6.0, fir_needle_col, flat=1.0)
    L = poly_len(axis)
    npair = 9
    for j in range(npair):
        s = (j + 0.45 + rng.uniform(-0.2, 0.2)) / (npair + 0.15) * L * 0.9
        p, t, _ = poly_sample(axis, s)
        rel = s / L
        for side in (1, -1):
            if rng.random() < 0.12:
                continue
            ang = rng.uniform(0.7, 1.05)
            d = nrm(rot_axis(t, ZUP, side * ang))
            reach = width * 0.46 * (0.35 + 0.65 * (1 - rel) ** 0.6) * rng.uniform(0.55, 1.0)
            lat = g.shoot_curve(p, d, reach / math.sin(ang), bend_axis=ZUP, bend=-side * rng.uniform(0.05, 0.35), n=6,
                                wobble=0.02)
            g.needled_shoot(lat, 0.0024, twig_brown, 0.021, 7.0, fir_needle_col, flat=1.0)
            Ll = poly_len(lat)
            q = 0
            while True:
                s3 = (q + 0.9) * rng.uniform(0.036, 0.05)
                if s3 > Ll * 0.85: break
                p3, t3, _ = poly_sample(lat, s3)
                for sd in (1, -1):
                    if rng.random() < 0.15:
                        continue
                    d3 = nrm(rot_axis(t3, ZUP, sd * rng.uniform(0.7, 0.95)))
                    L3 = rng.uniform(0.035, 0.08) * (1 - 0.45 * s3 / Ll)
                    sh = g.shoot_curve(p3, d3, L3, n=3)
                    g.needled_shoot(sh, 0.0015, twig_brown, 0.017, 7.5, fir_needle_col, flat=1.0, twig_sides=4)
                q += 1


def lichen_tuft(g, p, kind, scale=1.0):
    """Beard lichen hanging from a twig: Usnea (pale grey-green, branched) or Bryoria (dark brown-black horsehair)."""
    rng = g.rng
    # Usnea filaments ~0.5-1 mm, Bryoria ~0.3 mm; drawn ~2x thicker so a tuft survives the card mips as a wisp
    col = hexlin('7f8672', 0.12, rng) if kind == 'usnea' else hexlin('302820', 0.18, rng)
    n = int(rng.integers(8, 16))
    for _ in range(n):
        sp = [np.asarray(p, float).copy()]
        dd = nrm([rng.uniform(-0.4, 0.4), -1, rng.uniform(-0.3, 0.3)])
        L = rng.uniform(0.04, 0.12) * scale
        for _i in range(7):
            dd = nrm(dd + rng.normal(0, 0.35, 3) * np.array([1, 0.3, 1]) + np.array([0, -0.35, 0]))
            sp.append(sp[-1] + dd * L / 7)
        g.twig(np.array(sp), 0.0016 if kind == 'usnea' else 0.0011, 0.0005, col, sides=3)


def dead_branchlet(g, base, direc, L, r0, lichen=0.0):
    """Dead grey spruce branch end: a dense bottle-brush of fine bare twigs (needles long shed), knobbly, with grey
    crust/foliose lichen colour on the thicker parts and beard-lichen tufts (lower crown of closed montane stands)."""
    rng = g.rng

    def branch(p, d, L, r0, depth):
        pts = g.shoot_curve(p, d, L, bend_axis=ZUP, bend=rng.uniform(-0.7, 0.7), droop=rng.uniform(-0.1, 0.15),
                            n=6, wobble=0.07)
        # weathered dead spruce twigs: grey-brown, linear albedo ~0.08-0.14 (wet); lichen crust a little lighter
        grey = hexlin(['5a524a', '4e433b', '645b53', '6e6b62'][rng.integers(4)], 0.15, rng)
        if depth <= 1 and rng.random() < 0.35:
            grey = grey * 0.7 + hexlin('7c8272') * 0.3      # grey-green foliose lichen crust
        g.twig(pts, r0, r0 * 0.35, grey, sides=5 if depth == 0 else 4)
        Lb = poly_len(pts)
        if depth < 3:
            nk = int(Lb / (0.035 + 0.025 * depth)) + 1
            for q in range(nk):
                p3, t3, _ = poly_sample(pts, (q + 0.6) / (nk + 0.3) * Lb)
                sd = 1 if q % 2 else -1
                d3 = nrm(rot_axis(t3, ZUP, sd * rng.uniform(0.5, 1.1)) + np.array([0, 0, rng.uniform(-0.3, 0.3)]))
                if rng.random() < 0.85:
                    branch(p3, d3, L * rng.uniform(0.25, 0.5), r0 * 0.55, depth + 1)
        elif rng.random() < 0.2:
            for _ in range(rng.integers(2, 6)):
                g.needle(pts[-1], nrm(rng.normal(size=3) + d), 0.013, 0.0011, hexlin('7a5a36', 0.25, rng))
        if lichen > 0 and depth in (1, 2) and rng.random() < lichen:
            pp, _, _ = poly_sample(pts, Lb * rng.uniform(0.2, 0.8))
            lichen_tuft(g, pp, 'bryoria' if rng.random() < 0.7 else 'usnea', 1.0 + 0.6 * (depth == 1))

    branch(np.asarray(base, float), direc, L, r0, 0)


def bilberry_sprig(g, base, h):
    """Vaccinium myrtillus: green angular (4-ridged) stems, zig-zag branching, alternate small ovate leaves 1-3 cm
    (mid-green, some reddening in autumn), 20-40 cm tall. Side view."""
    rng = g.rng
    stem_col = hexlin('4a6a2a')
    for k in range(3):
        d0 = nrm([rng.uniform(-0.5, 0.5), 1, rng.uniform(-0.3, 0.3)])
        L0 = h * rng.uniform(0.6, 0.95)
        stem = g.shoot_curve(base + np.array([rng.uniform(-0.03, 0.03), 0, 0]), d0, L0, bend_axis=ZUP,
                             bend=rng.uniform(-0.5, 0.5), n=8, wobble=0.06)
        g.twig(stem, 0.0022, 0.0009, stem_col * rng.uniform(0.85, 1.1), sides=4)
        Ls = poly_len(stem)
        nb = int(rng.integers(3, 6))
        for b in range(nb):
            p, t, _ = poly_sample(stem, Ls * (0.3 + 0.6 * (b + rng.uniform(0, 0.8)) / nb))
            side = 1 if b % 2 else -1
            d = nrm(rot_axis(t, ZUP, side * rng.uniform(0.5, 1.0)))
            br = g.shoot_curve(p, d, Ls * rng.uniform(0.25, 0.45), bend_axis=ZUP, bend=-side * 0.5, n=5, wobble=0.05)
            g.twig(br, 0.0014, 0.0007, stem_col, sides=3)
            for pts in (br, stem):
                Lb = poly_len(pts)
                nleaf = max(2, int(Lb / 0.022))
                for q in range(nleaf):
                    if rng.random() < 0.25:
                        continue
                    pq, tq, _ = poly_sample(pts, Lb * (q + 0.5) / nleaf)
                    sd = 1 if q % 2 else -1
                    dd = nrm(rot_axis(tq, ZUP, sd * rng.uniform(0.6, 1.2)) + np.array([0, 0, rng.uniform(-0.3, 0.3)]))
                    Llf = rng.uniform(0.012, 0.028)
                    col = hexlin('3f6326', 0.18, rng)
                    rr = rng.random()
                    if rr < 0.12: col = hexlin('7a3322', 0.2, rng)      # autumn red
                    elif rr < 0.2: col = hexlin('6e6a2a', 0.2, rng)     # yellowing
                    g.leaf(pq, dd, nrm(ZUP + rng.normal(0, 0.4, 3)), Llf, Llf * 0.62, col, curl=0.1, serr=True, K=6)


def litter_patch(g, w, h):
    """Top view of the spruce forest floor: needle litter (brown/grey-brown), bits of twig, a cone, moss cushions.
    Patchy alpha so it breaks up the terrain texture where it is laid."""
    rng = g.rng
    # ragged patch outline: a noisy ellipse (so the decal never shows its rectangle)
    ph = rng.uniform(0, 6.28, 4)

    def inside(x, y):
        a = math.atan2(y / (h / 2), x / (w / 2))
        rr = 0.78 + 0.12 * math.sin(3 * a + ph[0]) + 0.08 * math.sin(5 * a + ph[1]) + 0.05 * math.sin(9 * a + ph[2])
        return (x / (w / 2)) ** 2 + (y / (h / 2)) ** 2 < (rr * rng.uniform(0.85, 1.05)) ** 2
    for _ in range(6500):
        x, y = rng.uniform(-w / 2, w / 2), rng.uniform(-h / 2, h / 2)
        if not inside(x, y):
            continue
        a = rng.uniform(0, 2 * math.pi)
        col = hexlin(['6b4a2e', '5a4636', '7a5c3c', '4a3b2e', '8a7a60'][rng.integers(5)], 0.15, rng)
        g.needle(np.array([x, y, rng.uniform(0, 0.006)]), np.array([math.cos(a), math.sin(a), rng.uniform(-0.1, 0.1)]),
                 rng.uniform(0.012, 0.02), 0.0012, col)
    for _ in range(26):
        x, y = rng.uniform(-w / 2.4, w / 2.4), rng.uniform(-h / 2.4, h / 2.4)
        a = rng.uniform(0, 2 * math.pi)
        L = rng.uniform(0.04, 0.16)
        pts = np.array([[x, y, 0.004], [x + math.cos(a) * L * 0.5, y + math.sin(a) * L * 0.5, 0.006],
                        [x + math.cos(a + 0.2) * L, y + math.sin(a + 0.2) * L, 0.004]])
        g.twig(pts, 0.0025, 0.001, hexlin('5e5048', 0.15, rng), sides=4)
    # moss cushions (Hylocomium/Pleurozium: yellow-green feathery) as dense tiny leaves
    for _ in range(9):
        cx, cy = rng.uniform(-w / 3, w / 3), rng.uniform(-h / 3.2, h / 3.2)
        rad = rng.uniform(0.04, 0.12)
        # wet feather moss: olive to yellow-green, linear albedo ~(0.05-0.09, 0.08-0.12, 0.02-0.03)
        col0 = hexlin(['4a6424', '586e26', '3d5620'][rng.integers(3)], 0.1, rng)
        for _i in range(int(900 * rad / 0.08)):
            a = rng.uniform(0, 2 * math.pi); r = rad * math.sqrt(rng.random())
            p = np.array([cx + math.cos(a) * r, cy + math.sin(a) * r, 0.008 * (1 - r / rad)])
            b = rng.uniform(0, 2 * math.pi)
            g.leaf(p, np.array([math.cos(b), math.sin(b), 0.3]), ZUP, rng.uniform(0.006, 0.012), 0.004, col0 * rng.uniform(0.8, 1.2),
                   curl=0.0, serr=False, K=3)
    # a spruce cone (elongated, scaly)
    cx, cy = rng.uniform(-w / 4, w / 4), rng.uniform(-h / 4, h / 4)
    a = rng.uniform(0, 2 * math.pi)
    ax = np.array([math.cos(a), math.sin(a), 0.0])
    for k in range(40):
        t = k / 40
        pc = np.array([cx, cy, 0.015]) + ax * (t - 0.5) * 0.12
        r = 0.016 * math.sin(math.pi * min(1, 0.1 + t)) ** 0.6
        phi = k * GOLDEN_C
        d = nrm(perp(ax) * math.cos(phi) + np.cross(ax, perp(ax)) * math.sin(phi) + ax * 0.4)
        g.leaf(pc, d, nrm(np.cross(d, ax) + ZUP * 0.2), r * 1.4, r * 1.2, hexlin('6a4a30', 0.15, rng), curl=0.1, serr=False, K=4)


GOLDEN_C = math.pi * (3 - math.sqrt(5))


def build_card_geometry(name, seed):
    """Card geometry laid out in the XY plane inside the card's world rect (x in [-w/2, w/2], y in [-h/2, h/2]).
    The render camera looks down -Z."""
    c = CARDS[name]
    w, h = c['w'], c['h']
    g = CardGeo(seed)
    rng = g.rng
    if name == 'spruce':
        spruce_spray(g, np.array([0, -h / 2 + 0.012, 0]), [rng.uniform(-0.05, 0.05), 1, 0], h * 0.9, w, 31,
                     lat_density=10.0)
    elif name == 'spruce2':  # upper crown: ascending brush shoot, denser, more new growth
        spruce_spray(g, np.array([0, -h / 2 + 0.012, 0]), [0, 1, 0], h * 0.92, w, 30, new_growth=0.16,
                     lat_density=11.0, droop=0.1)
    elif name == 'fir':
        fir_frond(g, np.array([0, -h / 2 + 0.012, 0]), [0, 1, 0], h * 0.95, w)
    elif name == 'hang':
        spruce_curtain(g, w, h)
    elif name in ('dead', 'twigs'):
        base = np.array([0, -h / 2 + 0.012, 0])
        dead_branchlet(g, base, [rng.uniform(-0.1, 0.1), 1, 0], h * 0.95, 0.0055 if name == 'dead' else 0.0045,
                       lichen=0.8 if name == 'dead' else 0.0)
        for _ in range(2 if name == 'dead' else 1):
            dead_branchlet(g, base + np.array([rng.uniform(-0.02, 0.02), 0.02, 0]), [rng.uniform(-0.9, 0.9), 1, 0],
                           h * 0.65, 0.004, lichen=0.6 if name == 'dead' else 0.0)
    elif name == 'leaf':
        base = np.array([0, -h / 2 + 0.012, 0])
        stem = g.shoot_curve(base, [rng.uniform(-0.1, 0.1), 1, 0], h * 0.8, bend_axis=ZUP, bend=0.25, n=8)
        g.twig(stem, 0.0035, 0.0015, hexlin('5b4a3c'))
        Ls = poly_len(stem)
        nleaf = 12
        for j in range(nleaf):
            p, t, _ = poly_sample(stem, (j + 0.8) / (nleaf + 0.5) * Ls)
            rel = j / nleaf
            side = 1 if j % 2 else -1
            d = nrm(rot_axis(t, ZUP, side * rng.uniform(0.6, 1.0)) + np.array([0, 0, rng.uniform(-0.2, 0.2)]))
            L = rng.uniform(0.07, 0.10) * (1 - 0.35 * rel)
            pet = p + d * 0.012
            g.twig(np.array([p, pet]), 0.0012, 0.001, hexlin('6a6a3a'), sides=3)
            col = hexlin('3a5420', 0.2, rng)
            if rng.random() < 0.12: col = hexlin('7a7a2e', 0.2, rng)
            g.leaf(pet, d, nrm(ZUP + rng.normal(0, 0.35, 3)), L, L * rng.uniform(0.7, 0.85), col, curl=rng.uniform(0.05, 0.25))
        g.leaf(stem[-1], nrm(stem[-1] - stem[-2]), ZUP, 0.09, 0.07, hexlin('45612a', 0.15, rng))
    elif name in ('fern_a', 'fern_b'):
        base = np.array([0, -h / 2 + 0.012, 0])
        rach = g.shoot_curve(base, [0, 1, 0], h * 0.96, bend_axis=ZUP, bend=rng.uniform(-0.2, 0.2), n=16)
        g.twig(rach, 0.0026, 0.0008, hexlin('4f5a2a'))
        L = poly_len(rach)
        npin = 24 if name == 'fern_a' else 19
        for j in range(npin):
            s = (0.1 + 0.88 * (j + 0.5) / npin) * L
            p, t, _ = poly_sample(rach, s)
            rel = s / L
            for side in (1, -1):
                if rng.random() < 0.06:
                    continue
                d = nrm(rot_axis(t, ZUP, side * rng.uniform(1.0, 1.25)))
                pl = (w * 0.47) * math.sin(math.pi * min(1, 0.22 + rel * 0.92)) ** 0.8 * rng.uniform(0.75, 1.0)
                pin = g.shoot_curve(p + (t * 0.004 * side), d, pl, bend_axis=ZUP, bend=side * 0.3, n=6)
                g.twig(pin, 0.0009, 0.0004, hexlin('4f6a2a'), sides=3)
                Lp = poly_len(pin)
                nlob = max(3, int(Lp / 0.0085))
                for q in range(nlob):
                    pq, tq, _ = poly_sample(pin, (q + 0.4) / nlob * Lp * 0.95)
                    lw = 0.017 * (1 - 0.65 * q / nlob) * (1 - 0.35 * rel) + 0.003
                    # wet late-season bracken/male fern: darker, less lime than fresh fronds; some browning pinnae
                    col = hexlin('33561f', 0.14, rng)
                    if rng.random() < (0.05 if name == 'fern_a' else 0.12): col = hexlin('5e5a2e', 0.2, rng)
                    for sd in (1, -1):
                        dd = nrm(rot_axis(tq, ZUP, sd * 1.15) + tq * 0.5)
                        g.leaf(pq, dd, ZUP, lw, lw * 0.62, col, curl=0.08, serr=False, K=6)
                g.leaf(pin[-1], nrm(pin[-1] - pin[-2]), ZUP, 0.012, 0.006, hexlin('33561f', 0.14, rng), serr=False, K=6)
    elif name == 'bilberry':
        bilberry_sprig(g, np.array([0, -h / 2 + 0.01, 0]), h * 0.95)
    elif name == 'coltsfoot':
        # grey-green upper surface (young leaves felted), pale veins
        g.round_leaf(np.array([0, -h * 0.02, 0]), h * 0.42, hexlin('4f6236'), kind='coltsfoot', rot=0.0, cup=0.1)
    elif name == 'butterbur':
        g.round_leaf(np.array([0, -h * 0.02, 0]), h * 0.44, hexlin('3d5a26'), kind='butterbur', rot=0.0, cup=0.16)
    elif name == 'litter':
        litter_patch(g, w, h)
    return g


# ---------------------------------------------------------------------------------------------
# Wild-grass card atlas (1024 x 1024, 4 side-view clump cards), rendered from 3D blade geometry like the needle cards.
# Exported as material 'grass' on the tiny node 'grass_cards'; the engine's GrassField draws crossed-card clumps with it.
# Linear albedo palette (before the engine's rain darkening x0.82): green blade tip ~(0.06, 0.09, 0.03), straw
# ~(0.27, 0.215, 0.12), bases near wet soil (~0.03-0.05).
# ---------------------------------------------------------------------------------------------
GRASS_ATLAS = 1024
GRASS_CARDS = {
    'g_dry':   dict(px=(0, 0, 512, 512), h=0.72, anchor='bottom'),
    'g_green': dict(px=(512, 0, 1024, 512), h=0.56, anchor='bottom'),
    'g_straw': dict(px=(0, 512, 512, 1024), h=0.8, anchor='bottom'),
    'g_mixed': dict(px=(512, 512, 1024, 1024), h=0.62, anchor='bottom'),
}
for _k, _c in GRASS_CARDS.items():
    x0, y0, x1, y1 = _c['px']
    _c['w'] = _c['h'] * (x1 - x0) / (y1 - y0)
    _c['uv'] = (x0 / GRASS_ATLAS, 1 - y1 / GRASS_ATLAS, x1 / GRASS_ATLAS, 1 - y0 / GRASS_ATLAS)

GPAL = {
    'green': (np.array([0.026, 0.036, 0.015]), np.array([0.065, 0.1, 0.032])),
    'lush': (np.array([0.024, 0.036, 0.014]), np.array([0.055, 0.09, 0.028])),
    'yellow': (np.array([0.03, 0.033, 0.014]), np.array([0.12, 0.122, 0.045])),
    'straw': (np.array([0.05, 0.042, 0.027]), np.array([0.27, 0.215, 0.12])),
    'brown': (np.array([0.035, 0.027, 0.018]), np.array([0.14, 0.09, 0.052])),
}


def grass_ribbon(g, pts, w0, cb, ct, twist=0.0, ao0=0.5):
    """Tapered blade ribbon along pts (3D), colour base->tip x height AO."""
    pts = np.asarray(pts)
    n = len(pts)
    P, N, C, F = [], [], [], []
    for i in range(n):
        t = i / (n - 1)
        tg = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        tg = nrm(tg)
        side = nrm(np.cross(tg, np.array([0, 0, 1.0])) + np.array([0, 0, 0.0001]))
        side = rot_axis(side, tg, twist * t)
        w = w0 * (1 - 0.9 * t ** 1.2)
        nn = nrm(np.cross(side, tg))
        if nn[2] < 0: nn = -nn
        P += [pts[i] - side * w, pts[i] + side * w]
        N += [nn, nn]
        ao = ao0 + (1 - ao0) * t ** 0.55
        c = (cb + (ct - cb) * t ** 0.7) * ao
        C += [np.append(c, 1), np.append(c, 1)]
    for i in range(n - 1):
        a = 2 * i
        F += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
    g.mb.add(P, N, None, np.array(C), F)


def grass_arch(base, dirA, L, th0, bend, n, zdepth=1.0):
    """Blade/stem polyline: rises at th0 from vertical toward azimuth dirA (in the X/Z plane of the card: x = right,
    z = depth), curving over by `bend` rad (rain-lodged). Card y = up."""
    dx, dz = math.cos(dirA), math.sin(dirA) * zdepth
    pts = [np.array(base, float)]
    for i in range(1, n + 1):
        t = i / n
        th = th0 + bend * t * t
        st = L / n
        p = pts[-1] + np.array([math.sin(th) * dx * st, math.cos(th) * st, math.sin(th) * dz * st])
        p[1] = max(p[1], base[1] + 0.004)
        pts.append(p)
    return np.array(pts)


def grass_clump(g, h, w, blades, straw, stems, lodge, short=1.0, forbs=0):
    rng = g.rng
    y0 = -h / 2 + 0.006
    for b in range(blades):
        # several tussocks per card (the card is a strip of verge, not one plant)
        cxs = [-w * 0.22, 0.0, w * 0.21]
        cx = cxs[b % 3] + rng.normal(0, w * 0.05)
        off = abs(rng.normal(0, 0.035))
        a = rng.uniform(0, 2 * math.pi)
        base = np.array([cx + math.cos(a) * off, y0, math.sin(a) * off])
        L = (0.16 + rng.random() * 0.36) * short * (h / 0.7)
        dirA = a + rng.uniform(-0.8, 0.8)
        th0 = 0.1 + rng.random() * 0.45 + off * 3
        bend = (0.4 + rng.random() * 1.4) * (0.4 + L * 1.7) * (0.55 + lodge * 0.9)
        pts = grass_arch(base, dirA, L, th0, bend, 7, zdepth=0.6)
        k = rng.random()
        if k < straw * 0.8: pal = GPAL['straw']
        elif k < straw: pal = GPAL['brown']
        elif k < straw + 0.12: pal = GPAL['yellow']
        else: pal = GPAL['green'] if rng.random() < 0.5 else GPAL['lush']
        j = rng.uniform(0.8, 1.2)
        grass_ribbon(g, pts, rng.uniform(0.0022, 0.004), pal[0] * j, pal[1] * j, twist=rng.uniform(-1.5, 1.5))
    for s in range(stems):
        cx = rng.uniform(-w * 0.35, w * 0.35)
        base = np.array([cx, y0, rng.uniform(-0.03, 0.03)])
        L = (0.4 + rng.random() * 0.4) * (h / 0.7)
        dirA = rng.uniform(0, 2 * math.pi)
        pts = grass_arch(base, dirA, L, rng.uniform(0.05, 0.3), (0.3 + rng.random() * 1.0) * (0.5 + lodge * 0.7), 8, zdepth=0.5)
        sc = (np.array([0.06, 0.06, 0.03]), np.array([0.2, 0.17, 0.095])) if rng.random() < 0.6 else \
             (np.array([0.04, 0.05, 0.022]), np.array([0.1, 0.11, 0.045]))
        grass_ribbon(g, pts, 0.0011, sc[0], sc[1], ao0=0.5)
        head = hexlin('8a7a55', 0.12, rng) if rng.random() < 0.55 else hexlin('5e4a4a', 0.15, rng)
        Ls = poly_len(pts)
        feathery = rng.random() < 0.5
        for q in range(int(rng.integers(10, 22))):
            p, t, _ = poly_sample(pts, Ls * rng.uniform(0.7, 1.0))
            d = nrm(np.array([rng.uniform(-1, 1), -rng.uniform(0.2, 1.0), rng.uniform(-0.5, 0.5)]) + t * 0.3)
            if feathery:     # hair-grass panicle: fine branches with tiny spikelets
                e = p + d * rng.uniform(0.02, 0.05)
                g.twig(np.array([p, (p + e) / 2 + np.array([0, 0.004, 0]), e]), 0.0004, 0.0003, head * 0.9, sides=3)
                g.leaf(e, d, np.array([0, 0, 1.0]), rng.uniform(0.004, 0.007), 0.0018, head, curl=0.0, serr=False, K=3)
            else:            # dense spikelet clusters (cocksfoot / fescue)
                g.leaf(p, d, np.array([0, 0, 1.0]), rng.uniform(0.008, 0.014), 0.0035, head, curl=0.05, serr=False, K=4)
    for f in range(forbs):
        # low broad leaves among the grass: ribwort / greater plantain rosettes, lady's mantle, sorrel
        cx = rng.uniform(-w * 0.35, w * 0.35)
        kind = rng.integers(3)
        for l in range(int(rng.integers(3, 7))):
            a = rng.uniform(0, 2 * math.pi)
            d = nrm(np.array([math.cos(a), rng.uniform(0.25, 0.8), math.sin(a) * 0.6]))
            base = np.array([cx, y0 + 0.004, 0.0])
            col = hexlin(['36521f', '415a24', '2f4a1c'][kind], 0.15, rng)
            if rng.random() < 0.12: col = hexlin('6a5a2a', 0.2, rng)
            if kind == 0:
                g.leaf(base, d, nrm(np.array([0, 1.0, 0.3])), rng.uniform(0.07, 0.14), 0.018, col, curl=0.15, serr=False, K=8)
            elif kind == 1:
                g.leaf(base, d, nrm(np.array([0, 1.0, 0.3])), rng.uniform(0.05, 0.08), 0.045, col, curl=0.2, serr=True, K=8)
            else:
                g.leaf(base, d, nrm(np.array([0, 1.0, 0.3])), rng.uniform(0.05, 0.09), 0.03, col, curl=0.1, serr=False, K=8)


def build_grass_geometry(name, seed):
    c = GRASS_CARDS[name]
    w, h = c['w'], c['h']
    g = CardGeo(seed)
    if name == 'g_dry':
        grass_clump(g, h, w, 150, straw=0.55, stems=9, lodge=0.8)
    elif name == 'g_green':
        grass_clump(g, h, w, 190, straw=0.15, stems=3, lodge=0.6, short=0.75, forbs=3)
    elif name == 'g_straw':
        grass_clump(g, h, w, 130, straw=0.8, stems=8, lodge=1.0, short=1.1)
    elif name == 'g_mixed':
        grass_clump(g, h, w, 150, straw=0.35, stems=6, lodge=0.7, forbs=4)
    return g


def render_card_atlas(force=False, grass=False):
    CS, SIZE, builder, pre = (GRASS_CARDS, GRASS_ATLAS, build_grass_geometry, 'grass') if grass else \
        (CARDS, ATLAS, build_card_geometry, 'foliage')
    albedo_path = os.path.join(BUILD, pre + '_albedo.png')
    normal_path = os.path.join(BUILD, pre + '_normal.png')
    meta_path = os.path.join(BUILD, 'cards.json' if not grass else 'grass_cards.json')
    if not force and all(os.path.exists(p) for p in (albedo_path, normal_path, meta_path)):
        log(pre, 'card atlas cached')
        return
    log('rendering', pre, 'card atlas')
    only = ARGS.get('cards')
    if only and not any(k in CS for k in only):
        log('  nothing to re-render in', pre)
        return
    if only and os.path.exists(albedo_path) and os.path.exists(normal_path):
        albedo = image_to_np(albedo_path)
        normal = image_to_np(normal_path)
        log('  re-rendering only', only)
    else:
        only = None
        albedo = np.zeros((SIZE, SIZE, 4))
        normal = np.zeros((SIZE, SIZE, 4))
    for ci, (name, c) in enumerate(CS.items()):
        if only and name not in only:
            continue
        sc = reset_scene()
        g = builder(name, 1000 + ci * 17)
        # material: emission = vertex colour * AO  (albedo pass) / geometry normal (normal pass)
        mat = new_mat('card')
        nt = mat.node_tree
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        em = nt.nodes.new('ShaderNodeEmission')
        attr = nt.nodes.new('ShaderNodeVertexColor'); attr.layer_name = 'Col'
        ao = nt.nodes.new('ShaderNodeAmbientOcclusion'); ao.inputs['Distance'].default_value = 0.035; ao.samples = 8
        nt.links.new(attr.outputs['Color'], ao.inputs['Color'])
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.inputs['Factor'].default_value = 0.3
        nt.links.new(attr.outputs['Color'], mix.inputs[6]); nt.links.new(ao.outputs['Color'], mix.inputs[7])
        nt.links.new(mix.outputs[2], em.inputs['Color'])
        nt.links.new(em.outputs[0], out.inputs['Surface'])
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        vm = nt.nodes.new('ShaderNodeVectorMath'); vm.operation = 'MULTIPLY_ADD'
        vm.inputs[1].default_value = (0.5, 0.5, 0.5); vm.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(geo.outputs['Normal'], vm.inputs[0])
        em2 = nt.nodes.new('ShaderNodeEmission')
        nt.links.new(vm.outputs[0], em2.inputs['Color'])
        ob = link(g.mb.build('card_' + name, [mat]))
        log('  card %-8s tris=%d' % (name, g.mb.tri_count()))
        x0, y0, x1, y1 = c['px']
        W, H = x1 - x0, y1 - y0
        cam = ortho_cam('cam', (0, 0, 0), c['w'], c['h'])
        a = render_exr(sc, cam, W, H, os.path.join(BUILD, 'tmp_card_a.exr'), ARGS['samples'])
        nt.links.new(em2.outputs[0], out.inputs['Surface'])
        n = render_exr(sc, cam, W, H, os.path.join(BUILD, 'tmp_card_n.exr'), max(8, ARGS['samples'] // 3))
        rgb, alpha = unpremultiply(a)
        nrgb, _ = unpremultiply(n)
        nv = nrgb * 2 - 1
        nv[..., 2] = np.abs(nv[..., 2])  # camera-facing hemisphere
        nv = nrm(nv)
        # thicken sub-pixel needles a little so an alpha test at ~0.45 keeps them (x1.25: the old x1.6 filled the gaps
        # between laterals and every spray read as a solid leaf-shaped plate in the engine)
        alpha_s = np.clip(alpha * 1.25, 0, 1)
        op = alpha > 0.5
        if op.any():
            log('  card %-8s linear mean %s coverage %.3f' % (name, np.round(rgb[op].mean(0), 4), float(alpha_s.mean())))
        if grass:
            # soft side edges: blades arching out of the card are faded instead of cut by a hard vertical line
            xf = (np.arange(W) + 0.5) / W
            alpha_s = alpha_s * (smoothstep(0.0, 0.1, xf) * smoothstep(1.0, 0.9, xf))[None, :]
        mask = alpha > 0.02
        rgb = dilate(rgb, mask, 12)
        nd = dilate(nv * 0.5 + 0.5, mask, 12)
        albedo[y0:y1, x0:x1, :3] = srgb_encode(rgb)
        albedo[y0:y1, x0:x1, 3] = alpha_s
        normal[y0:y1, x0:x1, :3] = nd
        normal[y0:y1, x0:x1, 3] = 1
    np_to_image(albedo, pre + '_albedo', albedo_path, 'sRGB')
    np_to_image(normal, pre + '_normal', normal_path, 'Non-Color')
    meta = {k: {'px': v['px'], 'w': v['w'], 'h': v['h'], 'anchor': v['anchor'], 'uv': v['uv']} for k, v in CS.items()}
    json.dump({'size': SIZE, 'cards': meta}, open(meta_path, 'w'), indent=1)
    log(pre, 'card atlas done')




# =============================================================================================
# 2) TREE SKELETONS + LOD EMISSION
# =============================================================================================
GOLDEN = math.pi * (3 - math.sqrt(5))
# Beard-lichen cards read as opaque pale rectangles in the engine (too little alpha coverage variation): disabled.
LICHEN_CARDS = False

VARIANTS = {
    # H: height (m), dbh: diameter at breast height, cb: crown base as fraction of H, R: max crown radius.
    # light: azimuth (Blender, rad) of the long "light" side of the crown; the engine turns every tree so that this side
    # faces downhill (valley light) and the opposite side (Blender +Y = three -Z) faces uphill, where the trunk carries
    # moss. asym: reach +/- on the light/shade side. broken: live crown ends at this fraction of H (dead spike above,
    # a side branch turned up as the new leader). kind 'snag': dead standing stem, all branches dead.
    'conifer_0': dict(seed=11, kind='spruce', H=24.0, dbh=0.44, cb=0.34, R=3.1, whorl=0.50, per=(4, 6), inter=(1, 2),
                      comb=0.9, droop=0.66, lean=0.10, gaps=0.14, lichen=0.35, cs=1.0, station=0.2, asym=0.3),
    'conifer_1': dict(seed=23, kind='spruce', H=18.5, dbh=0.34, cb=0.18, R=2.7, whorl=0.45, per=(4, 6), inter=(1, 2),
                      comb=0.55, droop=0.48, lean=0.05, gaps=0.1, lichen=0.2, cs=0.95, station=0.2, asym=0.25),
    'conifer_2': dict(seed=37, kind='spruce', H=28.0, dbh=0.56, cb=0.45, R=3.3, whorl=0.52, per=(4, 6), inter=(1, 2),
                      comb=1.0, droop=0.85, lean=0.14, gaps=0.2, lichen=0.55, cs=1.05, station=0.21, asym=0.35),
    # (QA) conifer_2 was the broken-top variant (broken=0.9): its candelabra leader clump sat ~1.5-2 m above the flat
    # crown on a thin dead spike that vanishes beyond ~40 m, so every fourth spruce on the skyline carried a dark
    # 'hat' floating above its top. It is a normal spire now; conifer_4 (snag) keeps a broken stem.
    'conifer_3': dict(seed=51, kind='fir', H=22.0, dbh=0.46, cb=0.3, R=3.3, whorl=0.56, per=(4, 5), inter=(0, 1),
                      comb=0.0, droop=0.1, lean=0.06, gaps=0.12, lichen=0.3, cs=1.0, station=0.17, asym=0.25),
    'conifer_4': dict(seed=63, kind='snag', H=16.5, dbh=0.4, cb=0.95, R=1.3, whorl=0.55, per=(3, 5), inter=(0, 1),
                      comb=0.0, droop=0.2, lean=0.12, gaps=0.3, lichen=0.8, cs=1.0, station=0.2, asym=0.2,
                      broken=0.84),
}
FALLEN = dict(seed=77, kind='spruce', H=18.2, dbh=0.56, cb=0.33, R=2.9, whorl=0.48, per=(4, 6), inter=(1, 2),
              comb=0.35, droop=0.5, lean=0.0, gaps=0.08, lichen=0.3, cs=1.0)
FALLEN_SPLIT = 6.2          # standing height of the cut point (= local x 0 once lying)
SAPLING = dict(seed=91, kind='spruce', H=1.9, dbh=0.045, cb=0.05, R=0.75, whorl=0.2, per=(4, 5), inter=(0, 1),
               comb=0.0, droop=0.25, lean=0.04, gaps=0.05, lichen=0.0, cs=0.42, base=-0.15)


class Skel:
    pass


def make_skeleton(V):
    rng = np.random.default_rng(V['seed'])
    H, R = V['H'], V['R']
    zc = V['cb'] * H
    sk = Skel()
    sk.V, sk.H, sk.R, sk.zc, sk.kind = V, H, R, zc, V['kind']
    sk.light = V.get('light', -math.pi / 2) + rng.uniform(-0.5, 0.5)
    sk.zbreak = V['broken'] * H if V.get('broken') else None
    # crown envelope height: a broken living tree keeps the crown width it had at the break (flat, ragged top
    # formed by the upper whorls and an upturned side leader) instead of tapering to a spire
    sk.Henv = H
    if sk.zbreak is not None and V['kind'] != 'snag':
        sk.Henv = zc + (sk.zbreak - zc) / 0.62
    step = max(0.2, H / 48)
    zb = V.get('base', -0.7)
    zs = np.concatenate([np.linspace(zb, min(1.2, H * 0.15), 7), np.arange(min(1.2, H * 0.15) + step, H - 0.05, step), [H]])
    la = rng.uniform(0, 2 * math.pi)
    ph = rng.uniform(0, 6.28, 4)
    fz = np.clip(zs / H, 0, 1)
    amp = min(0.05, H * 0.003)
    cx = V['lean'] * math.cos(la) * fz ** 1.6 + amp * np.sin(zs * 0.33 + ph[0]) * fz + amp * 0.4 * np.sin(zs * 1.1 + ph[1])
    cy = V['lean'] * math.sin(la) * fz ** 1.6 + amp * np.sin(zs * 0.29 + ph[2]) * fz + amp * 0.4 * np.sin(zs * 0.95 + ph[3])
    sk.tz = zs
    sk.tc = np.stack([cx, cy, zs], 1)
    rbh = V['dbh'] / 2
    zbh = min(1.3, H * 0.1)
    r = np.where(zs >= zbh, rbh * np.clip((H - zs) / (H - zbh), 0, 1) ** 0.85,
                 rbh * (1 + 0.42 * ((zbh - np.clip(zs, -5, zbh)) / zbh) ** 2.2))
    sk.tr = np.maximum(r, 0.005)
    sk.rbh = rbh
    sk.flare_phase = rng.uniform(0, 6.28)
    sk.flare_n = int(rng.integers(5, 7))
    sk.branches = []

    def add_branch(z, az, scale):
        h = (z - zc) / (H - zc)
        alive = z > zc - rng.uniform(0, 0.9) and sk.kind != 'snag'
        if sk.zbreak is not None and z > sk.zbreak:
            alive = False
        hh = min(max(h, 0.0), 1.0)
        ctr = sk.center(z)
        rad = sk.radius(z)
        if alive:
            side = 1 + V.get('asym', 0.0) * math.cos(az - sk.light)
            reach = env(sk, max(z, zc)) * scale * rng.uniform(0.72, 1.18) * side
            if rng.random() < V['gaps']:
                reach *= rng.uniform(0.2, 0.5)
            if sk.kind == 'spruce':
                pitch0 = -0.55 * V['droop'] * (1 - hh) ** 1.5 + 0.95 * hh ** 1.8 + rng.normal(0, 0.08)
                sag = 0.5 * V['droop'] * (1 - hh)
                tip = 0.3 + 0.5 * V['droop'] * (1 - hh)
            else:
                pitch0 = -0.1 + 0.5 * hh ** 2.2 + rng.normal(0, 0.05)
                sag, tip = 0.06, 0.12
            L = reach / max(0.5, math.cos(pitch0 - sag * 0.5))
        else:
            # dead branches: longest just below the live crown, short stubs further down (self-pruning in a closed
            # stand); snags: short broken stubs
            L = rng.uniform(0.25, 1.7) * min(1.0, 0.35 + z / max(zc, 0.5))
            if sk.kind == 'snag':
                L = rng.uniform(0.2, 1.3) * (1 - 0.5 * z / H)
            if rng.random() < 0.45:
                L = rng.uniform(0.08, 0.35)
            pitch0, sag, tip = rng.uniform(-0.5, 0.1), 0.1, 0.0
        L = max(L, 0.18 * V['cs'] if alive else 0.06)
        n = 6
        pts = [ctr + np.array([math.cos(az), math.sin(az), 0]) * rad * 0.8]
        azd = az
        for i in range(1, n):
            t = i / (n - 1)
            pitch = pitch0 - sag * math.sin(math.pi * t * 0.8) + tip * t ** 2.5
            azd += rng.normal(0, 0.05)
            d = np.array([math.cos(azd) * math.cos(pitch), math.sin(azd) * math.cos(pitch), math.sin(pitch)])
            pts.append(pts[-1] + d * L / (n - 1))
        r0 = (0.010 + 0.03 * min(1, L / 3.5)) if alive else (0.005 + 0.011 * min(1, L / 1.5))
        r0 = min(r0, rad * 0.55) * (V['dbh'] / 0.45) ** 0.3
        radii = r0 * (1 - np.linspace(0, 1, n) ** 0.8 * (0.75 if alive else 0.85))
        sk.branches.append(dict(pts=np.array(pts), radii=radii, L=L, alive=alive, h=h, az=az, z=z, scale=scale,
                                phase=float(rng.random()), seed=int(rng.integers(1 << 30)), stub=(not alive and L < 0.4),
                                trunc=None))

    z = min(0.9, H * 0.2) + rng.uniform(0, 0.25) * min(1, H / 10)
    wi = 0
    while z < H - 0.25 * V['cs']:
        nb = int(rng.integers(V['per'][0], V['per'][1] + 1))
        if rng.random() < V['gaps'] * 0.6:
            nb = int(rng.integers(1, 3))          # a nearly empty whorl: a horizontal gap through the crown
        az0 = wi * GOLDEN * 1.3 + rng.uniform(0, 1)
        for k in range(nb):
            add_branch(z + rng.uniform(-0.03, 0.03), az0 + 2 * math.pi * k / nb + rng.normal(0, 0.22), 1.0)
        nxt = V['whorl'] * rng.uniform(0.75, 1.25) * (0.85 + 0.35 * z / H)
        for m in range(int(rng.integers(V['inter'][0], V['inter'][1] + 1))):
            add_branch(z + nxt * rng.uniform(0.25, 0.8), rng.uniform(0, 2 * math.pi), rng.uniform(0.35, 0.6))
        z += nxt
        wi += 1
    sk.top_z = z - nxt
    sk.flag = None
    if sk.zbreak is not None and sk.kind != 'snag':
        cand = [b for b in sk.branches if b['alive'] and b['z'] < sk.zbreak and b['z'] > sk.zbreak - 2.0]
        if cand:
            b = max(cand, key=lambda b: b['L'])
            # re-shape it: short horizontal elbow then vertical (candelabra leader), foliage along it
            p0 = b['pts'][0]
            az = b['az']
            elbow = p0 + np.array([math.cos(az), math.sin(az), 0.15]) * 0.4
            Lup = min(1.3, H - sk.zbreak + 0.4)
            pts = [p0, elbow] + [elbow + np.array([math.cos(az) * 0.08 * t, math.sin(az) * 0.08 * t, Lup * t]) for t in (0.33, 0.66, 1.0)]
            pts.insert(2, elbow + np.array([0, 0, 0.05]))
            b['pts'] = np.array(pts[:6]) if len(pts) >= 6 else np.array(pts)
            b['radii'] = np.linspace(b['radii'][0] * 1.2, b['radii'][0] * 0.3, len(b['pts']))
            b['L'] = poly_len(b['pts'])
            b['leader'] = True
            sk.flag = b
    return sk


def _center(self, z):
    return np.array([np.interp(z, self.tz, self.tc[:, 0]), np.interp(z, self.tz, self.tc[:, 1]), z])


def _radius(self, z):
    return float(np.interp(z, self.tz, self.tr))


Skel.center = _center
Skel.radius = _radius


def env(sk, z):
    """Crown envelope: horizontal branch reach at height z."""
    h = (z - sk.zc) / (getattr(sk, 'Henv', sk.H) - sk.zc)
    if h < 0:
        return 0.0
    h = min(h, 1.0)
    if sk.kind == 'spruce':
        e = sk.R * (1 - h) ** 0.92 * (0.8 + 0.2 * smoothstep(0, 0.15, h))
    else:
        e = sk.R * (1 - h) ** 0.68 * (0.85 + 0.15 * smoothstep(0, 0.15, h))
    return max(float(e), 0.2 * sk.V['cs'])


def ao_of(sk, P):
    """Analytic crown occlusion per vertex (0..1): dark deep inside the crown, bright at the outer shell/top."""
    P = np.asarray(P)
    z = P[:, 2]
    cx = np.interp(z, sk.tz, sk.tc[:, 0]); cy = np.interp(z, sk.tz, sk.tc[:, 1])
    r = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    h = np.clip((z - sk.zc) / (sk.H - sk.zc), 0, 1)
    e = np.array([max(env(sk, zz), 0.35) if zz >= sk.zc else 0.0 for zz in z])
    inside = (0.3 + 0.7 * smoothstep(0.05, 1.08, r / np.maximum(e, 0.3)) ** 0.85) * (0.74 + 0.26 * h)
    inside = np.where(h > 0.9, np.minimum(1, inside * 1.12), inside)
    below = 0.42 + 0.12 * smoothstep(0, 1.8, r)
    ao = np.where(z >= sk.zc, inside, below)
    return np.clip(ao, 0.14, 1.0)


def outward_of(sk, P, up=0.55):
    P = np.asarray(P)
    z = P[:, 2]
    cx = np.interp(z, sk.tz, sk.tc[:, 0]); cy = np.interp(z, sk.tz, sk.tc[:, 1])
    d = np.stack([P[:, 0] - cx, P[:, 1] - cy, np.zeros(len(P))], 1)
    d = nrm(d + 1e-4)
    d[:, 2] += up
    return nrm(d)


def emit_card(mb, sk, name, base, vdir, udir, length, segs, droop, phase, rnd, mat=1, bend=0.62):
    c = CARDS[name]
    wid = length * c['w'] / c['h']
    u0, v0, u1, v1 = c['uv']
    vdir = nrm(vdir)
    udir = nrm(udir - vdir * np.dot(udir, vdir))
    ncard = nrm(np.cross(udir, vdir))
    sgn = -1.0 if ncard[2] >= 0 else 1.0
    rows = []
    cur = np.asarray(base, float).copy()
    d = vdir.copy()
    rows.append(cur.copy())
    for i in range(1, segs + 1):
        d = nrm(rot_axis(d, udir, sgn * droop / segs))
        cur = cur + d * length / segs
        rows.append(cur.copy())
    P, UV = [], []
    for i, rp in enumerate(rows):
        t = i / segs
        vv = v0 + t * (v1 - v0) if c['anchor'] == 'bottom' else v1 - t * (v1 - v0)
        P.append(rp - udir * wid / 2); UV.append((u0, vv))
        P.append(rp + udir * wid / 2); UV.append((u1, vv))
    P = np.array(P)
    out = outward_of(sk, P)
    nf = np.where((out @ ncard)[:, None] >= 0, ncard, -ncard)
    N = nrm(nf * (1 - bend) + out * bend)
    ao = ao_of(sk, P)
    C = np.stack([ao, np.full(len(P), phase), np.full(len(P), rnd), np.ones(len(P))], 1)
    F = []
    for i in range(segs):
        a, b, cc, dd = 2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2
        F += [(a, b, cc), (a, cc, dd)]
    mb.add(P, N, UV, C, F, mat)


def hdir(t):
    h = np.array([t[0], t[1], 0.0])
    n = np.linalg.norm(h)
    return h / n if n > 1e-6 else np.array([1.0, 0, 0])


def emit_tree(sk, lod, part=None, split=None, excl=0.0, mat_dead=2):
    """Emit mesh for a skeleton. lod 0/1. part in (None,'A','B') splits at standing height `split`
    (A: below, B: above); branches with |z-split|<excl are removed. mat_dead: material index of dead grey wood
    (dead branches, the dead spike of a broken top, a snag's stem)."""
    V = sk.V
    mb = MB()
    cs = V['cs']
    snag = sk.kind == 'snag'
    spruce = sk.kind == 'spruce'
    # ---- trunk ----------------------------------------------------------------------------
    sides = 10 if lod == 0 else 5
    zs, cs_, rs = sk.tz, sk.tc, sk.tr
    cuts = [c for c in (split, sk.zbreak) if c is not None]
    if cuts:
        zs = np.sort(np.unique(np.concatenate([zs, cuts])))
        cs_ = np.stack([np.interp(zs, sk.tz, sk.tc[:, 0]), np.interp(zs, sk.tz, sk.tc[:, 1]), zs], 1)
        rs = np.interp(zs, sk.tz, sk.tr)
    sel = np.ones(len(zs), bool)
    if lod == 1:
        sel[:] = False
        sel[::5] = True
        sel[-1] = True
        sel[np.argmin(np.abs(zs - 0))] = True
        for c in cuts:
            sel[np.argmin(np.abs(zs - c))] = True
    if part == 'A': sel &= zs <= split + 1e-6
    if part == 'B': sel &= zs >= split - 1e-6
    if sk.zbreak is not None:
        # broken stem: a short dead stub above the break (0.6 m on a living tree), a snag ends at the break
        sel &= zs <= sk.zbreak + (1e-6 if snag else 0.6)
    idx = np.nonzero(sel)[0]
    tp, tr, tz = cs_[idx], rs[idx], zs[idx]
    n_fl = sk.flare_n

    def radial(i, th, _tz=None):
        z = (_tz if _tz is not None else tz)[i]
        fl = math.exp(-max(z + 0.15, 0) / 0.45) * max(0.0, math.cos(n_fl * th + sk.flare_phase)) ** 2
        return 1 + 0.38 * fl + 0.025 * math.sin(3 * th + z * 1.7) + 0.015 * math.sin(7 * th - z * 3.1)
    u_rep = max(1, round(2 * math.pi * sk.rbh / 0.85))

    def tcol(p, ring):
        a = ao_of(sk, p[None])[0]
        return (min(1.0, a * 1.05), 0.0, 0.0, 1.0)

    def trunk_tube(mask, mat):
        ii = np.nonzero(mask)[0]
        if len(ii) < 2: return
        tzz = tz[ii]
        tube(mb, tp[ii], tr[ii], sides, None, mat=mat, u_rep=u_rep, v_scale=1 / 1.3, v0=float(tzz[0]) / 1.3,
             radial_fn=lambda i, th, _t=tzz: radial(i, th, _t), color_fn=tcol, cap=(mat == mat_dead and snag))
    if len(tp) >= 2:
        if snag:
            trunk_tube(np.ones(len(tz), bool), mat_dead)
        elif sk.zbreak is not None:
            trunk_tube(tz <= sk.zbreak + 1e-6, 0)
            trunk_tube(tz >= sk.zbreak - 1e-6, mat_dead)
        else:
            trunk_tube(np.ones(len(tz), bool), 0)
    # jagged broken top of a snag: a few splinters
    if sk.zbreak is not None and part in (None, 'B'):
        r = np.random.default_rng(V['seed'] + 9)
        ztop = sk.zbreak if snag else min(sk.H, sk.zbreak + 0.6)
        top = sk.center(ztop)
        rt = max(sk.radius(ztop), 0.03)
        for k in range(5 if lod == 0 else 2):
            a = r.uniform(0, 2 * math.pi)
            b0 = top + np.array([math.cos(a), math.sin(a), 0]) * rt * 0.6
            tube(mb, np.array([b0 - ZUP * 0.05, b0 + np.array([0, 0, r.uniform(0.15, 0.6)]) + r.normal(0, 0.04, 3)]),
                 np.array([rt * 0.35, rt * 0.04]), 3, (0.8, 0, 0, 1), mat=mat_dead, cap=True)
    # ---- branches + foliage ----------------------------------------------------------------
    for bi, b in enumerate(sk.branches):
        if split is not None:
            if abs(b['z'] - split) < excl: continue
            if part == 'A' and b['z'] >= split: continue
            if part == 'B' and b['z'] < split: continue
        if snag and sk.zbreak is not None and b['z'] > sk.zbreak - 0.2:
            continue
        r = np.random.default_rng(b['seed'])
        pts, radii, L = b['pts'], b['radii'], b['L']
        Lvis = L if b['trunc'] is None else L * b['trunc']
        phase = b['phase']
        hh = min(max(b['h'], 0.0), 1.0)
        leader = b.get('leader', False)
        # branch wood
        if lod == 0:
            if b['alive']:
                use = [0, 2, 5] if L > 1.1 else ([0, 5] if L > 0.55 else None)
                sides_b = 3
            else:
                use = [0, 2, 5] if L > 0.7 else [0, 5]
                sides_b = 4 if L > 0.7 else 3
        else:
            if b['alive']: use = [0, 5] if leader else None
            elif L < 0.8: use = None
            else: use, sides_b = [0, 5], 3
        if leader:
            use, sides_b = list(range(len(pts))), (5 if lod == 0 else 3)
        if use is not None:
            bp = pts[use]
            br = radii[use]
            if b['trunc'] is not None:
                k = max(2, int(round(len(use) * b['trunc'])))
                bp, br = bp[:k], br[:k] * np.linspace(1, 0.8, k)
            ao_b = ao_of(sk, bp)
            tube(mb, bp, br, sides_b, None, mat=0 if b['alive'] else mat_dead, u_rep=1, v_scale=1 / 0.6,
                 color_fn=lambda p, ring, _a=ao_b, _ph=phase: (float(_a[min(ring, len(_a) - 1)]), _ph, 0.0, 1.0),
                 cap=not b['alive'])
        segs = 2 if lod == 0 else 1
        csz = 0.78 if lod == 0 else 0.92
        if b['alive'] and leader:
            # new leader of a broken top: short ascending sprays all round it + an upright tip
            nk = 9 if lod == 0 else 4
            for k in range(nk):
                s = L * (0.2 + 0.75 * k / nk)
                p, t, _ = poly_sample(pts, s)
                a = r.uniform(0, 2 * math.pi)
                vdir = nrm(np.array([math.cos(a), math.sin(a), r.uniform(0.2, 0.8)]))
                emit_card(mb, sk, 'spruce2', p, vdir, rot_axis(nrm(np.cross(vdir, ZUP)), vdir, r.uniform(-0.4, 0.4)),
                          cs * r.uniform(0.5, 0.8) * (1 if lod == 0 else 1.25), segs, 0.35, phase, r.random())
            p = pts[-1]
            emit_card(mb, sk, 'spruce2', p - ZUP * 0.35, ZUP, np.array([1.0, 0, 0]), 0.6 * cs, segs, 0.05, phase,
                      r.random(), bend=0.4)
            continue
        if b['alive']:
            card = 'spruce' if spruce else 'fir'
            bare = float(np.clip(0.04 + 0.3 * (L / sk.R) * (1 - hh) ** 0.8, 0.03, 0.38))
            stations = np.arange(bare * L, Lvis - 0.1 * cs, V.get('station', 0.21) * cs * csz)
            # comb spruce: pendulous 2nd-order branchlets hang in curtains under the lower and middle crown branches
            comb = V['comb'] * (0.25 + 0.75 * (1 - hh) ** 0.7) if spruce else 0.0
            for k, s in enumerate(stations):
                p, t, _ = poly_sample(pts, s)
                th = hdir(t)
                rel = s / L
                Lf = cs * (0.5 + 0.42 * (1 - rel)) * (0.55 + 0.45 * min(1, L / 2.2)) * r.uniform(0.8, 1.2)
                if sk.kind == 'fir': Lf *= 1.15
                Lf = min(Lf, 0.35 * cs + 0.9 * L) * csz
                for side in (-1, 1):
                    keep = (lod == 0) or ((k + (side > 0)) % 2 == 0)
                    if not keep:
                        continue
                    if spruce and r.random() < 0.1 * (1 - hh):
                        continue                   # an empty station: sky shows through the branch
                    yaw = side * r.uniform(0.25, 1.0) if spruce else side * r.uniform(0.35, 1.1)
                    roll = r.uniform(-0.6, 0.6) if spruce else r.uniform(-0.55, 0.55)
                    # steeply rolled side sprays: seen from the road (below/side) the crown must not read as stacked
                    # horizontal shelves. More of them toward the drooping outer, lower crown.
                    p_roll = (0.3 + 0.25 * rel) * (1.2 - 0.5 * hh) if spruce else 0.18
                    if r.random() < p_roll:
                        roll = r.choice([-1, 1]) * r.uniform(0.85, 1.45)
                    pitch = math.asin(np.clip(t[2], -1, 1)) * (0.8 if spruce else 0.5) - \
                        (r.uniform(0.12, 0.45) + 0.3 * rel * (1 - hh) if spruce else r.uniform(-0.12, 0.2))
                    dH = rot_axis(th, ZUP, yaw)
                    vdir = dH * math.cos(pitch) + ZUP * math.sin(pitch)
                    udir = rot_axis(nrm(np.cross(vdir, ZUP)), vdir, roll)
                    sc = 1.0 if lod == 0 else 1.25   # LOD1: half the stations, larger cards (~LOD0 crown volume)
                    dr = (0.72 if spruce else 0.22) * r.uniform(0.7, 1.3)
                    emit_card(mb, sk, card, p, vdir, udir, Lf * sc / 0.94, segs if (lod == 0 or Lf > 0.5) else 1,
                              dr if lod == 0 else (0.5 if spruce else 0.16), phase, r.random())
            # hanging curtains along the outer 2/3 of the branch (the dominant texture of comb spruce from the road)
            if comb > 0 and L > 0.7:
                s = max(bare * L, 0.3 * L)
                ci = 0
                while s < Lvis - 0.2 * cs:
                    step = cs * r.uniform(0.3, 0.55)
                    if r.random() < comb and (lod == 0 or ci % 2 == 0):
                        p, t, _ = poly_sample(pts, min(s + step * 0.5, Lvis - 0.1))
                        th = rot_axis(hdir(t), ZUP, r.uniform(-0.35, 0.35))
                        side = r.choice([-1, 1])
                        off = nrm(np.cross(ZUP, th)) * side * r.uniform(0.0, 0.18)
                        vdir = rot_axis(-ZUP, th, r.uniform(-0.3, 0.3))
                        Lh = cs * r.uniform(0.38, 0.75) * (0.6 + 0.4 * (1 - hh)) * (1 if lod == 0 else 1.2)
                        emit_card(mb, sk, 'hang', p + off - ZUP * 0.03, vdir, th, Lh, segs, side * 0.25, phase,
                                  r.random(), bend=0.45)
                    s += step
                    ci += 1
            # tip frond (upturned: comb spruce branch tips curve up)
            if b['trunc'] is None:
                p, t, _ = poly_sample(pts, L - 0.06 * cs)
                pitch = math.asin(np.clip(t[2], -1, 1)) + (0.25 if spruce else 0.08)
                vdir = hdir(t) * math.cos(pitch) + ZUP * math.sin(pitch)
                udir = rot_axis(nrm(np.cross(vdir, ZUP)), vdir, r.uniform(-0.3, 0.3))
                Lt = cs * (0.5 + 0.28 * min(1, L / 2)) * r.uniform(0.85, 1.1) * (1 if lod == 0 else 1.3)
                emit_card(mb, sk, 'spruce2' if spruce else 'fir', p - vdir * 0.08, vdir, udir,
                          min(Lt, 0.3 * cs + L), segs, 0.2, phase, r.random())
            # twigs + dead shoots on the bare inner part (lower crown: needles shed in the shade)
            if lod == 0 and hh < 0.5 and L > 1.0 and r.random() < 0.55:
                p, t, _ = poly_sample(pts, L * r.uniform(0.06, max(0.1, bare)))
                vdir = nrm(t + r.normal(0, 0.4, 3))
                emit_card(mb, sk, 'twigs', p, vdir,
                          rot_axis(perp(vdir), vdir, r.uniform(0, 6.28)), r.uniform(0.35, 0.6), 1, 0.0, phase, r.random())
        else:
            # dead branch: grey lichen-hung twig cards (the lichen lives on the 'dead' card)
            if Lvis > 0.35 and (lod == 0 or Lvis > 0.9):
                n_dead = 1 if (lod == 1 or Lvis < 0.9) else 2
                for q in range(n_dead):
                    p, t, _ = poly_sample(pts, (0.08 + 0.45 * q) * L)
                    vdir = nrm(t + r.normal(0, 0.18, 3))
                    udir = rot_axis(perp(vdir), vdir, r.uniform(0, 6.28))
                    ln = min(Lvis * (0.95 - 0.35 * q), 1.0) * (1 if lod == 0 else 1.2)
                    emit_card(mb, sk, 'dead' if (r.random() < 0.3 + V['lichen'] * 0.7) else 'twigs', p, vdir, udir,
                              ln, 1, 0.0, phase, r.random(), bend=0.3)
                if lod == 0 and r.random() < 0.4:
                    emit_card(mb, sk, 'twigs', p, nrm(t + r.normal(0, 0.4, 3)),
                              rot_axis(perp(vdir), vdir, r.uniform(0, 6.28)), min(Lvis, 0.6), 1, 0.0, phase, r.random())
    # ---- leader / top ---------------------------------------------------------------------
    if part in (None, 'B') and not snag and sk.zbreak is None:
        r = np.random.default_rng(V['seed'] + 5)
        top = sk.center(sk.H)
        Ltop = min(1.5, max(0.5, sk.H - sk.top_z + 0.55)) * cs
        card = 'spruce2' if spruce else 'fir'
        n_top = 3 if lod == 0 else 2
        for k in range(n_top):
            a = k * math.pi / n_top + r.uniform(-0.2, 0.2)
            udir = np.array([math.cos(a), math.sin(a), 0])
            vdir = nrm(ZUP + r.normal(0, 0.06, 3))
            base = top - vdir * Ltop * 0.97
            emit_card(mb, sk, card, base, vdir, udir, Ltop, segs if lod == 0 else 1, 0.05, 0.5, r.random(), bend=0.4)
    return mb


# ---------------------------------------------------------------------------------------------
# Fallen tree (lying along +X in three coords = Blender +X; root plate at -X; cut at x=0)
# ---------------------------------------------------------------------------------------------
def fallen_axis_h(X):
    X = np.asarray(X, float)
    return np.where(X < 0, 0.34 - 0.03 * X + 0.012 * X * X, 0.34 - 0.03 * X - 0.002 * np.maximum(0, X - 4.5) ** 2)


def lay_down(mb, seed, flatten=True):
    """Standing coords -> lying: X = z - split, Y = y, Z = -x + axis_h(X). Flatten under-ground parts on the road."""
    rng = np.random.default_rng(seed)
    newP, newN = [], []
    for P, N in zip(mb.P, mb.N):
        X = P[:, 2] - FALLEN_SPLIT
        Q = np.stack([X, P[:, 1], -P[:, 0] + fallen_axis_h(X)], 1)
        M = np.stack([N[:, 2], N[:, 1], -N[:, 0]], 1)
        if flatten:
            floor = 0.02 + 0.025 * np.abs(np.sin(Q[:, 0] * 3.1 + Q[:, 1] * 2.3))
            m = (Q[:, 0] < 4.3) & (Q[:, 2] < floor)
            # squash: things pushed into the ground spread out a bit and lie flat
            depth = floor - Q[:, 2]
            Q[m, 2] = floor[m] + np.minimum(depth[m], 0.4) * 0.08
            M[m] = nrm(M[m] * 0.3 + np.array([0, 0, 1.0]))
        newP.append(Q); newN.append(M)
    mb.P, mb.N = newP, newN


def fallen_skeleton():
    sk = make_skeleton(FALLEN)
    rng = np.random.default_rng(FALLEN['seed'] + 1)
    for b in sk.branches:
        down = math.cos(b['az'])  # +x in standing frame == down after lying
        if b['alive']:
            if down > 0.55 and b['z'] < FALLEN_SPLIT + 5.0:
                b['trunc'] = rng.uniform(0.2, 0.45)   # crushed underside branches (road part)
            elif rng.random() < 0.3:
                b['trunc'] = rng.uniform(0.35, 0.7)   # snapped in the fall
    return sk


def root_plate(mb, seed):
    """Torn-out root plate: vertical soil disc at the root end with radiating roots (lying frame)."""
    rng = np.random.default_rng(seed)
    X0 = -FALLEN_SPLIT - 0.25
    c = np.array([X0, 0.0, float(fallen_axis_h(-FALLEN_SPLIT))])
    R0 = 1.3
    nseg, nring = 56, 6
    th = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
    outline = R0 * (1 + 0.16 * np.sin(3 * th + 1.3) + 0.1 * np.sin(5 * th + 0.4) + 0.07 * np.sin(9 * th + 2.1)
                    + 0.05 * np.sin(17 * th + 0.7) + rng.uniform(-0.12, 0.12, nseg))
    outline[np.sin(th) < -0.45] *= 0.8   # flatter, torn bottom
    outline[np.sin(th) > 0.6] *= 1.08    # grassy top lip overhangs a bit
    lump = rng.normal(0, 1, (nring + 1, nseg))
    for face, dx, sgn in (('front', -0.34, -1), ('back', 0.22, 1)):
        P, N, UV = [c + np.array([dx - 0.1 * sgn, 0, 0])], [np.array([sgn, 0, 0])], [(0.5, 0.5)]
        for ri in range(1, nring + 1):
            f = ri / nring
            for j in range(nseg):
                y = math.cos(th[j]) * outline[j] * f
                z = math.sin(th[j]) * outline[j] * f
                bul = (1 - f ** 2) * (0.3 if face == 'front' else 0.1) + lump[ri, j] * 0.07 * (1 - 0.5 * f)
                P.append(c + np.array([dx + sgn * bul, y, z]))
                N.append(nrm(np.array([sgn, 0, 0]) + np.array([0, y, z]) * 0.35 + rng.normal(0, 0.3, 3)))
                UV.append((0.5 + y / 1.3, 0.5 + z / 1.3))
        F = []
        for j in range(nseg):
            a, b = 1 + j, 1 + (j + 1) % nseg
            F.append((0, a, b) if sgn > 0 else (0, b, a))
        for ri in range(1, nring):
            for j in range(nseg):
                a = 1 + (ri - 1) * nseg + j
                b = 1 + (ri - 1) * nseg + (j + 1) % nseg
                cc = 1 + ri * nseg + (j + 1) % nseg
                d = 1 + ri * nseg + j
                F += ([(a, d, cc), (a, cc, b)] if sgn > 0 else [(a, cc, d), (a, b, cc)])
        ao = np.full(len(P), 0.75)
        mb.add(P, N, UV, np.stack([ao, np.zeros(len(P)), np.zeros(len(P)), np.ones(len(P))], 1), F, 2)
    # rim band connecting front and back
    P, N, UV, F = [], [], [], []
    for j in range(nseg + 1):
        jj = j % nseg
        y, z = math.cos(th[jj]) * outline[jj], math.sin(th[jj]) * outline[jj]
        for dx in (-0.34, 0.22):
            P.append(c + np.array([dx + rng.normal(0, 0.05), y * (1.0 + rng.normal(0, 0.03)), z * (1.0 + rng.normal(0, 0.03))]))
            N.append(nrm(np.array([0, y, z])))
            UV.append((j / nseg * 5, dx * 2))
    for j in range(nseg):
        a, b, cc, d = 2 * j, 2 * j + 1, 2 * j + 3, 2 * j + 2
        F += [(a, cc, b), (a, d, cc)]
    mb.add(P, N, UV, (0.6, 0, 0, 1), F, 2)
    # embedded stones (soil material, lumpy icosahedra)
    for k in range(9):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.2, 0.9) * R0
        cen = c + np.array([-0.42 - rng.uniform(0, 0.15), math.cos(a) * rr, math.sin(a) * rr])
        rad = rng.uniform(0.07, 0.2)
        ico = np.array([[0, 1, 1.618], [0, -1, 1.618], [0, 1, -1.618], [0, -1, -1.618], [1, 1.618, 0], [-1, 1.618, 0],
                        [1, -1.618, 0], [-1, -1.618, 0], [1.618, 0, 1], [-1.618, 0, 1], [1.618, 0, -1], [-1.618, 0, -1]], float)
        ico = nrm(ico)
        Pst = cen + ico * rad * rng.uniform(0.7, 1.2, (12, 1)) * np.array([1.0, 1.1, 0.8])
        Fst = [(0, 8, 1), (0, 1, 9), (0, 4, 8), (0, 9, 5), (0, 5, 4), (1, 8, 6), (1, 7, 9), (1, 6, 7), (2, 3, 10), (2, 11, 3),
               (2, 10, 4), (2, 5, 11), (2, 4, 5), (3, 6, 10), (3, 11, 7), (3, 7, 6), (4, 10, 8), (5, 9, 11), (6, 8, 10), (7, 11, 9)]
        mb.add(Pst, ico, (Pst[:, 1:3] - cen[1:3]) / 0.5 + 0.5, (0.8, 0, 0, 1), Fst, 2)
    # grassy lip on the top edge of the plate (the old forest-floor surface): small fern/grass cards
    sk_l = Skel(); sk_l.tz = np.array([-100, 100.0]); sk_l.tc = np.array([[0, 0, -100], [0, 0, 100.0]]); sk_l.zc = -100
    sk_l.H = 100; sk_l.R = 1; sk_l.V = {'cs': 1}; sk_l.kind = 'spruce'
    for k in range(10):
        a = rng.uniform(0.35, math.pi - 0.35)
        j = int((a / (2 * math.pi)) * nseg) % nseg
        top = c + np.array([rng.uniform(-0.1, 0.25), math.cos(a) * outline[j] * 0.97, math.sin(a) * outline[j] * 0.97])
        vdir = nrm(np.array([rng.uniform(-0.3, 0.6), math.cos(a) * 0.6, math.sin(a) * 0.6 + 0.5]))
        emit_card(mb, sk_l, 'fern_a' if k % 2 else 'fern_b', top, vdir, perp(vdir), rng.uniform(0.3, 0.5), 1, 0.3,
                  rng.random(), rng.random(), mat=1, bend=0.3)
    # roots
    for k in range(30):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.15, 1.02) * R0
        start = c + np.array([-0.36 - rng.uniform(0, 0.12), math.cos(a) * rr, math.sin(a) * rr])
        d = nrm(np.array([rng.uniform(-1.0, -0.2), math.cos(a) * rng.uniform(0.3, 1), math.sin(a) * rng.uniform(0.3, 1)]))
        L = rng.uniform(0.3, 1.4)
        pts = [start]
        for i in range(4):
            d = nrm(d + rng.normal(0, 0.3, 3) + np.array([0, 0, -0.25]))
            pts.append(pts[-1] + d * L / 4)
        rad0 = rng.uniform(0.012, 0.06) if k > 6 else rng.uniform(0.05, 0.09)
        tube(mb, np.array(pts), np.linspace(rad0, rad0 * 0.3, 5), 4, (0.55, 0.3, 0.0, 1.0), mat=0, v_scale=1 / 0.5, cap=True)


def cut_face(mb, X, sgn, r_cut, seed):
    """Hatchet-cut end face at x=X facing sgn*X, rough V-notch facets, 'wood_cut' material (index 3)."""
    rng = np.random.default_rng(seed)
    c = np.array([X, 0.0, float(fallen_axis_h(X))])
    n = 18
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    P, N, UV = [c + np.array([sgn * -0.03, 0, 0])], [np.array([sgn, 0, 0])], [(0.5, 0.5)]
    for ring, f in ((0, 0.55), (1, 1.0)):
        for j in range(n):
            y, z = math.cos(th[j]) * r_cut * f, math.sin(th[j]) * r_cut * f
            notch = 0.10 * abs(math.sin(th[j])) * f  # chopped from both sides -> faces angled
            P.append(c + np.array([sgn * (-notch + rng.normal(0, 0.008)), y, z]))
            N.append(nrm(np.array([sgn, 0, 0]) + rng.normal(0, 0.12, 3)))
            UV.append((0.5 + 0.48 * y / r_cut, 0.5 + 0.48 * z / r_cut))
    F = []
    for j in range(n):
        a, b = 1 + j, 1 + (j + 1) % n
        F.append((0, a, b) if sgn > 0 else (0, b, a))
        a2, b2 = 1 + n + j, 1 + n + (j + 1) % n
        F += ([(a, a2, b2), (a, b2, b)] if sgn > 0 else [(a, b2, a2), (a, b, b2)])
    mb.add(P, N, UV, (0.9, 0, 0, 1), F, 3)


def build_fallen(mats):
    sk = fallen_skeleton()
    parts = {}
    r_cut = sk.radius(FALLEN_SPLIT) * 1.02
    for key, part in (('fallen_tree', None), ('fallen_tree_a', 'A'), ('fallen_tree_b', 'B')):
        mb = emit_tree(sk, 0, part=part, split=FALLEN_SPLIT, excl=0.9, mat_dead=4)
        lay_down(mb, 3)
        if part in (None, 'A'):
            root_plate(mb, 21)
        if part == 'A': cut_face(mb, 0.0, 1, r_cut, 5)
        if part == 'B': cut_face(mb, 0.0, -1, r_cut, 6)
        ob = mb.build(key, mats)
        # collider hints for the engine (three.js local coords: x along trunk, y up, z = -Blender y)
        xs = np.linspace(-FALLEN_SPLIT, sk.H - FALLEN_SPLIT - 1.5, 14)
        axis = [[float(x), float(fallen_axis_h(x)), 0.0, float(sk.radius(x + FALLEN_SPLIT))] for x in xs]
        ob['collider'] = json.dumps({
            'axis': axis, 'split': 0.0,
            'rootPlate': {'center': [-FALLEN_SPLIT - 0.3, float(fallen_axis_h(-FALLEN_SPLIT)), 0.0], 'radius': 1.3, 'halfThickness': 0.25},
            'crown': {'x0': 1.0, 'x1': sk.H - FALLEN_SPLIT - 1.0, 'radius': 2.2},
            'cutRadius': float(r_cut), 'topAtCut': float(fallen_axis_h(0.0) + r_cut),
        })
        parts[key] = (ob, mb.tri_count())
    return parts


# ---------------------------------------------------------------------------------------------
# Shrubs and deadwood
# ---------------------------------------------------------------------------------------------
def build_alder(seed):
    """Green alder bush: multiple arching stems with leafy twig cards."""
    rng = np.random.default_rng(seed)
    mb = MB()
    sk = Skel(); sk.tz = np.array([-1, 10.0]); sk.tc = np.array([[0, 0, -1], [0, 0, 10.0]]); sk.zc = 0.0; sk.H = 1.9
    sk.R = 1.1; sk.V = {'cs': 1}; sk.kind = 'spruce'
    for k in range(8):
        a = k * 2 * math.pi / 8 + rng.uniform(-0.3, 0.3)
        lean = rng.uniform(0.25, 0.75)
        L = rng.uniform(1.4, 2.1)
        pts = [np.array([math.cos(a) * 0.05, math.sin(a) * 0.05, -0.05])]
        d = nrm(np.array([math.cos(a) * math.sin(lean), math.sin(a) * math.sin(lean), math.cos(lean)]))
        for i in range(6):
            d = nrm(d + np.array([math.cos(a), math.sin(a), -0.35]) * 0.12 + rng.normal(0, 0.05, 3))
            pts.append(pts[-1] + d * L / 6)
        pts = np.array(pts)
        tube(mb, pts, np.linspace(0.022, 0.005, 7), 5, (0.7, rng.random(), 0, 1), mat=0, v_scale=1 / 0.5)
        Ls = poly_len(pts)
        for q in range(9):
            s = Ls * (0.3 + 0.7 * (q + rng.uniform(0, 0.8)) / 9)
            p, t, _ = poly_sample(pts, s)
            side = 1 if q % 2 else -1
            vdir = nrm(rot_axis(t, ZUP, side * rng.uniform(0.4, 1.0)) + ZUP * rng.uniform(0.0, 0.5))
            udir = rot_axis(nrm(np.cross(vdir, ZUP)), vdir, rng.uniform(-0.6, 0.6))
            emit_card(mb, sk, 'leaf', p, vdir, udir, rng.uniform(0.38, 0.55), 2, 0.25, rng.random(), rng.random())
    # AO: darker toward the middle/bottom
    for i, P in enumerate(mb.P):
        ao = np.clip(0.35 + 0.65 * smoothstep(0.0, 1.0, np.hypot(P[:, 0], P[:, 1])) * (0.6 + 0.4 * smoothstep(0, 1.6, P[:, 2])), 0.2, 1)
        mb.C[i][:, 0] = ao
    return mb


def build_mound(seed):
    """Low prostrate conifer clump (dwarf mountain pine / juniper-like) from flat fir sprays."""
    rng = np.random.default_rng(seed)
    mb = MB()
    sk = Skel(); sk.tz = np.array([-1, 10.0]); sk.tc = np.array([[0, 0, -1], [0, 0, 10.0]]); sk.zc = 0.0; sk.H = 0.9
    sk.R = 1.0; sk.V = {'cs': 1}; sk.kind = 'fir'
    for k in range(13):
        a = k * GOLDEN * 2 + rng.uniform(-0.2, 0.2)
        L = rng.uniform(0.7, 1.15)
        rise = rng.uniform(0.25, 0.7)
        pts = [np.array([0, 0, -0.05])]
        d = nrm(np.array([math.cos(a), math.sin(a), rise]))
        for i in range(5):
            d = nrm(d + np.array([0, 0, -0.18]) + rng.normal(0, 0.06, 3))
            pts.append(pts[-1] + d * L / 5)
        pts = np.array(pts)
        tube(mb, pts, np.linspace(0.03, 0.008, 6), 4, (0.5, rng.random(), 0, 1), mat=0, v_scale=1 / 0.5)
        for q in range(4):
            p, t, _ = poly_sample(pts, L * (0.35 + 0.65 * q / 3))
            for side in (-1, 1):
                vdir = nrm(rot_axis(hdir(t), ZUP, side * rng.uniform(0.3, 0.9)) + ZUP * rng.uniform(0.0, 0.4))
                udir = rot_axis(nrm(np.cross(vdir, ZUP)), vdir, rng.uniform(-0.3, 0.3))
                emit_card(mb, sk, 'fir' if k % 3 else 'spruce', p, vdir, udir, rng.uniform(0.35, 0.5), 2, 0.25,
                          rng.random(), rng.random())
    for i, P in enumerate(mb.P):
        ao = np.clip(0.3 + 0.7 * smoothstep(0.0, 1.0, np.hypot(P[:, 0], P[:, 1])) * (0.6 + 0.4 * smoothstep(0, 0.7, P[:, 2])), 0.2, 1)
        mb.C[i][:, 0] = ao
    return mb


def build_deadwood(kind, seed):
    rng = np.random.default_rng(seed)
    mb = MB()
    if kind == 0:
        # dead log lying along X, broken ends, stubs, partly sunk
        L = 3.8
        xs = np.linspace(-L / 2, L / 2, 10)
        pts = np.stack([xs, 0.04 * np.sin(xs * 1.3), np.full_like(xs, 0.0)], 1)
        rad = 0.11 * (1 - 0.25 * (xs + L / 2) / L)
        pts[:, 2] = rad * 0.75
        def rfn(i, th):
            return 1 + 0.05 * math.sin(5 * th + i) + (0.1 if (i == 0 or i == len(xs) - 1) and math.sin(th * 3 + i) > 0.3 else 0)
        tube(mb, pts, rad, 10, (0.6, 0.2, 0, 1), mat=0, u_rep=2, v_scale=1 / 1.0, radial_fn=rfn, cap=True)
        for k in range(7):
            x = rng.uniform(-L / 2 + 0.3, L / 2 - 0.2)
            a = rng.uniform(0.3, math.pi - 0.3) * (1 if rng.random() < 0.7 else -1)
            base = np.array([x, 0, float(np.interp(x, xs, rad)) * 0.75])
            d = nrm(np.array([rng.uniform(-0.3, 0.6), math.cos(a), abs(math.sin(a))]))
            ln = rng.uniform(0.08, 0.5)
            tube(mb, np.array([base, base + d * ln]), np.array([0.025, 0.012]), 4, (0.6, 0.2, 0, 1), mat=0, cap=True)
    else:
        # fallen dead branch with side twigs lying on the ground
        L = 2.6
        pts = [np.array([-L / 2, 0, 0.05])]
        d = np.array([1.0, 0, 0])
        for i in range(6):
            d = nrm(d + rng.normal(0, 0.15, 3) * np.array([0.3, 1, 0.1]))
            pts.append(pts[-1] + d * L / 6)
        pts = np.array(pts)
        pts[:, 2] = np.linspace(0.06, 0.03, len(pts))
        tube(mb, pts, np.linspace(0.05, 0.012, len(pts)), 6, (0.6, 0.3, 0, 1), mat=0, v_scale=1 / 0.6, cap=True)
        sk = Skel(); sk.tz = np.array([-1, 10.0]); sk.tc = np.array([[0, 0, -1], [0, 0, 10.0]]); sk.zc = 0.0; sk.H = 1
        sk.R = 1.0; sk.V = {'cs': 1}; sk.kind = 'spruce'
        for q in range(8):
            p, t, _ = poly_sample(pts, L * (0.15 + 0.8 * q / 7))
            side = 1 if q % 2 else -1
            dd = nrm(rot_axis(t, ZUP, side * rng.uniform(0.5, 1.1)) + ZUP * rng.uniform(0.0, 0.5))
            ln = rng.uniform(0.25, 0.7)
            e = p + dd * ln
            e[2] = max(e[2], 0.02)
            tube(mb, np.array([p, (p + e) / 2 + ZUP * 0.03, e]), np.array([0.018, 0.01, 0.004]), 4, (0.6, 0.3, 0, 1), mat=0)
            if rng.random() < 0.6:
                emit_card(mb, sk, 'dead', p, nrm(dd * 0.8 + ZUP * 0.35), rot_axis(nrm(np.cross(dd, ZUP)), dd, rng.uniform(-0.5, 0.5)),
                          rng.uniform(0.4, 0.6), 1, 0.0, 0.3, rng.random())
        for i, P in enumerate(mb.P):
            mb.C[i][:, 0] = 0.7
    return mb


# =============================================================================================
# 3) MATERIALS + TEXTURES
# =============================================================================================
def gltf_output_group():
    ng = bpy.data.node_groups.get('glTF Material Output')
    if ng: return ng
    ng = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
    ng.interface.new_socket('Occlusion', in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket('Thickness', in_out='INPUT', socket_type='NodeSocketFloat')
    return ng


def resize_to(path_in, path_out, size, fn=None, colorspace='sRGB'):
    """Load image, optional numpy fn on (h,w,4) 0..1 stored values, resize to size, save PNG."""
    if os.path.exists(path_out) and fn is None and os.path.getmtime(path_out) > os.path.getmtime(path_in):
        return load_image(path_out, colorspace=colorspace)
    img = bpy.data.images.load(path_in, check_existing=False)
    img.colorspace_settings.name = 'Non-Color'
    img.scale(size, size)
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(h, w, 4)[::-1]
    if fn is not None: a = fn(a)
    bpy.data.images.remove(img)
    return np_to_image(a, os.path.basename(path_out), path_out, colorspace)


def make_wood_cut(path, size=512, seed=3):
    """Procedural cross-section of a freshly hatchet-cut spruce log: rings, heart/sapwood, bark rim, chop marks."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size - 0.5
    r = np.hypot(x, y) / 0.48
    a = np.arctan2(y, x)
    warp = 0.012 * np.sin(a * 3 + 1) + 0.008 * np.sin(a * 7 + 2) + 0.004 * rng.normal(0, 1, (size, size))
    rr = r + warp
    rings = 0.5 + 0.5 * np.sin(rr * 2 * math.pi * 26 + 3 * np.sin(rr * 9))
    rings = rings ** 3
    heart = srgb_decode(np.array([0.78, 0.62, 0.42]))
    sap = srgb_decode(np.array([0.90, 0.80, 0.62]))
    late = srgb_decode(np.array([0.62, 0.45, 0.28]))
    base = np.where((rr < 0.62)[..., None], heart, sap)
    base = base * (1 - 0.45 * rings[..., None]) + late * 0.45 * rings[..., None]
    # hatchet chop marks: diagonal bands of different shading
    chop = 0.5 + 0.5 * np.sin((x * 0.8 + y) * 55 + 4 * np.sin(x * 13))
    base *= (0.85 + 0.2 * chop[..., None])
    # radial drying cracks
    for _ in range(5):
        ca = rng.uniform(-math.pi, math.pi)
        m = (np.abs(np.angle(np.exp(1j * (a - ca)))) < 0.012 * (1.1 - r)) & (r > 0.25) & (r < 0.95)
        base[m] *= 0.35
    # wet dirty edges + bark rim
    bark = srgb_decode(np.array([0.32, 0.22, 0.16]))
    rim = smoothstep(0.93, 0.98, r)
    base = base * (1 - rim[..., None]) + bark * rim[..., None]
    base *= (1 - 0.25 * smoothstep(0.75, 0.95, r))[..., None]
    out = np.concatenate([srgb_encode(base), np.ones((size, size, 1))], -1)
    return np_to_image(out, 'wood_cut', path, 'sRGB')


def image_node(nt, img, colorspace=None):
    t = nt.nodes.new('ShaderNodeTexImage')
    t.image = img
    return t


def export_material(name, diffuse, normal=None, arm=None, alpha=False, rough=0.8):
    m = new_mat(name)
    nt = m.node_tree
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    t = image_node(nt, diffuse)
    nt.links.new(t.outputs['Color'], bsdf.inputs['Base Color'])
    if alpha:
        nt.links.new(t.outputs['Alpha'], bsdf.inputs['Alpha'])
    if normal is not None:
        tn = image_node(nt, normal)
        nm = nt.nodes.new('ShaderNodeNormalMap')
        nt.links.new(tn.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    if arm is not None:
        ta = image_node(nt, arm)
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(ta.outputs['Color'], sep.inputs['Color'])
        nt.links.new(sep.outputs['Green'], bsdf.inputs['Roughness'])
        nt.links.new(sep.outputs['Blue'], bsdf.inputs['Metallic'])
        g = nt.nodes.new('ShaderNodeGroup')
        g.node_tree = gltf_output_group()
        nt.links.new(sep.outputs['Red'], g.inputs['Occlusion'])
    else:
        bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = 0.0 if arm is None else bsdf.inputs['Metallic'].default_value
    nt.links.new(bsdf.outputs[0], out.inputs['Surface'])
    return m


def make_materials():
    tex = {}
    bark_dir = os.path.join(RAWTEX, 'pine_bark')
    tex['bark_d'] = resize_to(os.path.join(bark_dir, 'diffuse.jpg'), os.path.join(BUILD, 'bark_diffuse.png'), 1024)
    tex['bark_n'] = resize_to(os.path.join(bark_dir, 'nor_gl.jpg'), os.path.join(BUILD, 'bark_normal.png'), 1024, colorspace='Non-Color')
    tex['bark_a'] = resize_to(os.path.join(bark_dir, 'arm.jpg'), os.path.join(BUILD, 'bark_arm.png'), 1024, colorspace='Non-Color')

    def greyify(a):  # silver-fir / weathered deadwood: desaturate, lighten slightly, cooler
        lin = srgb_decode(a[..., :3])
        lum = (lin * np.array([0.3, 0.59, 0.11])).sum(-1, keepdims=True)
        g = lum * np.array([1.0, 1.0, 1.02]) * 1.35
        out = lin * 0.3 + g * 0.7
        return np.concatenate([srgb_encode(out), a[..., 3:]], -1)
    tex['grey_d'] = resize_to(os.path.join(bark_dir, 'diffuse.jpg'), os.path.join(BUILD, 'bark_grey_diffuse.png'), 1024, fn=greyify)
    soil = os.path.join(PUBTEX, 'brown_mud_rocks_01')
    tex['soil_d'] = resize_to(os.path.join(soil, 'diffuse.jpg'), os.path.join(BUILD, 'soil_diffuse.png'), 1024)
    tex['soil_n'] = resize_to(os.path.join(soil, 'nor_gl.jpg'), os.path.join(BUILD, 'soil_normal.png'), 1024, colorspace='Non-Color')
    tex['soil_a'] = resize_to(os.path.join(soil, 'arm.jpg'), os.path.join(BUILD, 'soil_arm.png'), 1024, colorspace='Non-Color')
    tex['wood'] = make_wood_cut(os.path.join(BUILD, 'wood_cut.png'))
    tex['fol_d'] = load_image(os.path.join(BUILD, 'foliage_albedo.png'), colorspace='sRGB')
    tex['fol_n'] = load_image(os.path.join(BUILD, 'foliage_normal.png'), colorspace='Non-Color')
    tex['grass_d'] = load_image(os.path.join(BUILD, 'grass_albedo.png'), colorspace='sRGB')
    tex['grass_n'] = load_image(os.path.join(BUILD, 'grass_normal.png'), colorspace='Non-Color')
    mats = {
        'bark': export_material('bark', tex['bark_d'], tex['bark_n'], tex['bark_a']),
        'bark_grey': export_material('bark_grey', tex['grey_d'], tex['bark_n'], tex['bark_a']),
        'foliage': export_material('foliage', tex['fol_d'], tex['fol_n'], None, alpha=True, rough=0.62),
        'grass': export_material('grass', tex['grass_d'], tex['grass_n'], None, alpha=True, rough=0.7),
        'soil': export_material('soil', tex['soil_d'], tex['soil_n'], tex['soil_a']),
        'wood_cut': export_material('wood_cut', tex['wood'], None, None, rough=0.75),
    }
    return mats, tex


# =============================================================================================
# 4) BUILD ALL MODELS
# =============================================================================================
def build_models():
    reset_scene()
    mats, tex = make_materials()
    objs = {}
    stats = {}
    skels = {}
    for name, V in VARIANTS.items():
        sk = make_skeleton(V)
        skels[name] = sk
        bark = mats['bark_grey'] if V['kind'] == 'fir' else mats['bark']
        for lod in (0, 1):
            mb = emit_tree(sk, lod)
            key = name if lod == 0 else name + '_lod1'
            objs[key] = mb.build(key, [bark, mats['foliage'], mats['bark_grey']])
            stats[key] = mb.tri_count()
            nf = int(sum((m == 1).sum() for m in mb.M))
            log('  %-15s foliage %6d  wood %6d' % (key, nf, stats[key] - nf))
    for key, (ob, tris) in build_fallen([mats['bark'], mats['foliage'], mats['soil'], mats['wood_cut'],
                                         mats['bark_grey']]).items():
        objs[key] = ob
        stats[key] = tris
    sap = make_skeleton(SAPLING)
    mb = emit_tree(sap, 0)
    objs['shrub_0'] = mb.build('shrub_0', [mats['bark'], mats['foliage'], mats['bark_grey']]); stats['shrub_0'] = mb.tri_count()
    mb = build_alder(301)
    objs['shrub_1'] = mb.build('shrub_1', [mats['bark_grey'], mats['foliage']]); stats['shrub_1'] = mb.tri_count()
    mb = build_mound(302)
    objs['shrub_2'] = mb.build('shrub_2', [mats['bark'], mats['foliage']]); stats['shrub_2'] = mb.tri_count()
    for k in (0, 1):
        mb = build_deadwood(k, 400 + k)
        objs['deadwood_%d' % k] = mb.build('deadwood_%d' % k, [mats['bark_grey'], mats['foliage']])
        stats['deadwood_%d' % k] = mb.tri_count()
    mb = MB()
    mb.add([[0, 0, 0], [0.1, 0, 0], [0.1, 0, 0.1]], [[0, -1, 0]] * 3, [[0, 0], [1, 0], [1, 1]], (1, 0, 0, 1), [(0, 1, 2)])
    objs['grass_cards'] = mb.build('grass_cards', [mats['grass']]); stats['grass_cards'] = 1
    coll = bpy.data.collections.new('export')
    bpy.context.scene.collection.children.link(coll)
    for k, ob in objs.items():
        coll.objects.link(ob)
        if k.startswith('conifer') and not k.endswith('lod1'):
            V = VARIANTS[k]
            ob['treeHeight'] = V['H']
            ob['crownRadius'] = V['R']
    for k, v in stats.items():
        log('  %-15s %6d tris' % (k, v))
    return objs, mats, tex, skels, stats


# =============================================================================================
# 5) EXPORT (glTF -> gltf-transform meshopt + webp)
# =============================================================================================
def run(cmd):
    log('$ ' + ' '.join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError('command failed')
    return r.stdout


def export_glb(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for ob in objs.values():
        ob.hide_set(False)
        ob.select_set(True)
    bpy.context.view_layer.objects.active = next(iter(objs.values()))
    raw = os.path.join(BUILD, 'trees_raw.glb')
    bpy.ops.export_scene.gltf(filepath=raw, export_format='GLB', use_selection=True, export_apply=False,
                              export_yup=True, export_texcoords=True, export_normals=True, export_tangents=False,
                              export_vertex_color='ACTIVE', export_all_vertex_colors=False, export_extras=True,
                              export_image_format='AUTO', export_materials='EXPORT', export_cameras=False,
                              export_lights=False)
    out = os.path.join(OUT_MODELS, 'trees.glb')
    run(['npx', 'gltf-transform', 'optimize', raw, out, '--compress', 'meshopt', '--texture-compress', 'webp',
         '--texture-size', '2048', '--join', 'false', '--flatten', 'false', '--instance', 'false', '--palette', 'false',
         '--simplify', 'false', '--prune-attributes', 'false', '--weld', 'false'])
    log('exported', out, '%.2f MB' % (os.path.getsize(out) / 1e6))


# =============================================================================================
# 6) IMPOSTOR ATLAS (8 azimuths x 4 elevations per conifer, albedo*AO + object-space normals)
# =============================================================================================
IMP_AZ = 8
IMP_EL = [-25.0, 0.0, 25.0, 50.0]
IMP_FW, IMP_FH = 128, 384


def impostor_material(kind, which, tex):
    m = new_mat('imp_%s_%s' % (kind, which))
    nt = m.node_tree
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    dimg = {'bark': tex['bark_d'], 'bark_grey': tex['grey_d'], 'foliage': tex['fol_d']}[kind]
    nimg = {'bark': tex['bark_n'], 'bark_grey': tex['bark_n'], 'foliage': tex['fol_n']}[kind]
    d = image_node(nt, dimg)
    em = nt.nodes.new('ShaderNodeEmission')
    if which == 'albedo':
        attr = nt.nodes.new('ShaderNodeVertexColor'); attr.layer_name = 'Col'
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(attr.outputs['Color'], sep.inputs['Color'])
        mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'
        mul.inputs['Factor'].default_value = 1.0
        # same crown-AO remap as the engine's mesh LODs (impostor.js AO_REMAP, FOLIAGE.aoMin): albedo * (0.25 + 0.75 * AO)
        aor = nt.nodes.new('ShaderNodeMath'); aor.operation = 'MULTIPLY_ADD'
        # (bark: 0.15 + 0.85 * AO, as createBarkMaterial)
        aor.inputs[1].default_value = 0.75 if kind == 'foliage' else 0.85
        aor.inputs[2].default_value = 0.25 if kind == 'foliage' else 0.15
        nt.links.new(sep.outputs['Red'], aor.inputs[0])
        nt.links.new(d.outputs['Color'], mul.inputs[6])
        nt.links.new(aor.outputs[0], mul.inputs[7])
        nt.links.new(mul.outputs[2], em.inputs['Color'])
    else:
        tn = image_node(nt, nimg)
        nm = nt.nodes.new('ShaderNodeNormalMap')
        nt.links.new(tn.outputs['Color'], nm.inputs['Color'])
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        flip = nt.nodes.new('ShaderNodeMath'); flip.operation = 'MULTIPLY_ADD'
        flip.inputs[1].default_value = -2.0; flip.inputs[2].default_value = 1.0
        nt.links.new(geo.outputs['Backfacing'], flip.inputs[0])
        sc = nt.nodes.new('ShaderNodeVectorMath'); sc.operation = 'SCALE'
        nt.links.new(nm.outputs['Normal'], sc.inputs[0]); nt.links.new(flip.outputs[0], sc.inputs['Scale'])
        vt = nt.nodes.new('ShaderNodeVectorTransform'); vt.vector_type = 'NORMAL'
        vt.convert_from = 'WORLD'; vt.convert_to = 'OBJECT'
        nt.links.new(sc.outputs[0], vt.inputs[0])
        ma = nt.nodes.new('ShaderNodeVectorMath'); ma.operation = 'MULTIPLY_ADD'
        ma.inputs[1].default_value = (0.5, 0.5, 0.5); ma.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(vt.outputs[0], ma.inputs[0])
        nt.links.new(ma.outputs[0], em.inputs['Color'])
    if kind == 'foliage':
        # card-alpha threshold: seen from 40+ m the engine's mip-filtered, coverage-boosted cards read fuller than the
        # full-res texture cut at 0.5; 0.32 matches the (sparser, ragged v2) LOD1 crowns at the switch (0.18 made the
        # impostors visibly denser and more shelf-like)
        gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.32
        nt.links.new(d.outputs['Alpha'], gt.inputs[0])
        tr = nt.nodes.new('ShaderNodeBsdfTransparent')
        mix = nt.nodes.new('ShaderNodeMixShader')
        nt.links.new(gt.outputs[0], mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1]); nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs['Surface'])
    else:
        nt.links.new(em.outputs[0], out.inputs['Surface'])
    return m


def swap_materials(objs_or_meshes, mapping):
    for me in objs_or_meshes:
        for i, m in enumerate(me.materials):
            if m is not None and m.name in mapping:
                me.materials[i] = mapping[m.name]


def mesh_positions(ob):
    me = ob.data
    P = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', P)
    return P.reshape(-1, 3).astype(np.float64)


def view_basis(az_deg, el_deg):
    """Blender-local basis for a view from azimuth az (from three +Z toward three +X) and elevation el."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    v = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])  # tree -> camera
    f = -v
    right = nrm(np.cross(f, ZUP))
    upc = np.cross(right, f)
    return right, f, upc


def render_impostors(objs, mats, tex, skels):
    sc = bpy.context.scene
    for ob in bpy.data.objects:
        ob.hide_render = True
    albedo_m = {k: impostor_material(k, 'albedo', tex) for k in ('bark', 'bark_grey', 'foliage')}
    normal_m = {k: impostor_material(k, 'normal', tex) for k in ('bark', 'bark_grey', 'foliage')}
    NE = len(IMP_EL)
    W, H = IMP_FW * IMP_AZ * len(VARIANTS), IMP_FH * NE
    atlas_a = np.zeros((H, W, 4))
    atlas_n = np.zeros((H, W, 4))
    variants = []
    for vi, name in enumerate(VARIANTS):
        src = objs[name]
        P = mesh_positions(src)
        Hh = VARIANTS[name]['H']
        best = None
        for cz in np.linspace(0.40, 0.60, 9) * Hh:
            piv = np.array([0, 0, cz])
            hw = hh = 0
            for el in IMP_EL:
                for i in range(IMP_AZ):
                    right, f, upc = view_basis(i * 360 / IMP_AZ, el)
                    Q = P - piv
                    hw = max(hw, np.abs(Q @ right).max())
                    hh = max(hh, np.abs(Q @ upc).max())
            if best is None or hh < best[2]:
                best = (cz, hw, hh)
        cz, hw, hh = best
        aspect = IMP_FH / IMP_FW
        if hh / hw > aspect: hw = hh / aspect
        else: hh = hw * aspect
        hw *= 1.02; hh *= 1.02
        fwW, fhW = 2 * hw, 2 * hh
        piv = Vector((0, 0, cz))
        copies = []
        for j, el in enumerate(IMP_EL):
            for i in range(IMP_AZ):
                right, f, upc = view_basis(i * 360 / IMP_AZ, el)
                R = Matrix((tuple(right), tuple(f), tuple(upc))).to_4x4()
                cx = (i + 0.5) * fwW - IMP_AZ * fwW / 2
                czw = (j + 0.5) * fhW - NE * fhW / 2
                ob = src.copy()
                sc.collection.objects.link(ob)
                ob.hide_render = False
                ob.matrix_world = Matrix.Translation((cx, 0, czw)) @ R @ Matrix.Translation(-piv)
                copies.append(ob)
        cam = ortho_cam('imp_cam', (0, 0, 0), IMP_AZ * fwW, NE * fhW, look='+Y')
        cam.data.ortho_scale = IMP_AZ * fwW
        cam.data.sensor_fit = 'HORIZONTAL'
        me = src.data
        orig = [m for m in me.materials]
        swap_materials([me], {'bark': albedo_m['bark'], 'bark_grey': albedo_m['bark_grey'], 'foliage': albedo_m['foliage']})
        a = render_exr(sc, cam, IMP_FW * IMP_AZ, IMP_FH * NE, os.path.join(BUILD, 'tmp_imp_a.exr'), ARGS['samples'])
        for k, m in enumerate(orig): me.materials[k] = m
        swap_materials([me], {'bark': normal_m['bark'], 'bark_grey': normal_m['bark_grey'], 'foliage': normal_m['foliage']})
        n = render_exr(sc, cam, IMP_FW * IMP_AZ, IMP_FH * NE, os.path.join(BUILD, 'tmp_imp_n.exr'), max(8, ARGS['samples'] // 2))
        for k, m in enumerate(orig): me.materials[k] = m
        for ob in copies:
            bpy.data.objects.remove(ob)
        bpy.data.objects.remove(cam)
        rgb, alpha = unpremultiply(a)
        nrgb, _ = unpremultiply(n)
        nb = nrgb * 2 - 1
        nt3 = nrm(np.stack([nb[..., 0], nb[..., 2], -nb[..., 1]], -1))  # Blender local -> three local
        x0 = vi * IMP_AZ * IMP_FW
        # image row 0 = TOP; elevation row j=0 (lowest) is at the BOTTOM of the image
        for j in range(NE):
            for i in range(IMP_AZ):
                ys = slice(H - (j + 1) * IMP_FH, H - j * IMP_FH)
                xs = slice(i * IMP_FW, (i + 1) * IMP_FW)
                m = alpha[ys, xs] > 0.02
                fr = rgb[ys, xs]
                w = alpha[ys, xs][..., None]
                mean = (fr * w).sum((0, 1)) / max(w.sum(), 1e-6)
                fr = dilate(fr, m, 2)
                m2 = dilate(m[..., None].astype(float) * np.ones(3), m, 2)[..., 0] > 0.5
                fr[~m2] = mean
                atlas_a[ys, x0 + i * IMP_FW: x0 + (i + 1) * IMP_FW, :3] = srgb_encode(fr)
                # coverage-preserving alpha: scale so the fraction of texels passing the engine's 0.5 alpha test
                # equals the true (fractional) coverage of the frame. A flat x1.15 lost the thin sub-texel sprays,
                # and the impostors read narrower and more tiered than the LOD1 meshes.
                fa = alpha[ys, xs]
                target = float(fa.mean())
                lo, hi = 1.0, 6.0
                for _ in range(24):
                    k = 0.5 * (lo + hi)
                    if float((fa * k > 0.5).mean()) > target: hi = k
                    else: lo = k
                k = 0.5 * (lo + hi)
                atlas_a[ys, x0 + i * IMP_FW: x0 + (i + 1) * IMP_FW, 3] = np.clip(fa * k, 0, 1)
                if i == 0 and j == 1:
                    log('    %s alpha scale %.2f coverage %.3f' % (name, k, target))
                nn = dilate(nt3[ys, xs] * 0.5 + 0.5, m, 16)
                atlas_n[ys, x0 + i * IMP_FW: x0 + (i + 1) * IMP_FW, :3] = nn
                atlas_n[ys, x0 + i * IMP_FW: x0 + (i + 1) * IMP_FW, 3] = 1
        variants.append({'name': name, 'index': vi, 'x0': x0, 'width': fwW, 'height': fhW, 'pivotY': cz,
                         'treeHeight': Hh, 'crownRadius': VARIANTS[name]['R'], 'kind': VARIANTS[name]['kind']})
        log('  impostor %s frame %.2fx%.2f m pivot %.2f' % (name, fwW, fhW, cz))
    # normal atlas at half resolution
    nh = atlas_n[0::2, 0::2] * 0.25 + atlas_n[1::2, 0::2] * 0.25 + atlas_n[0::2, 1::2] * 0.25 + atlas_n[1::2, 1::2] * 0.25
    v = nh[..., :3] * 2 - 1
    nh[..., :3] = nrm(v) * 0.5 + 0.5
    pa = os.path.join(BUILD, 'imp_albedo.png')
    pn = os.path.join(BUILD, 'imp_normal.png')
    np_to_image(atlas_a, 'imp_albedo', pa, 'sRGB')
    np_to_image(nh, 'imp_normal', pn, 'Non-Color')
    js = ("const sharp=require('sharp');(async()=>{"
          "await sharp(%r).webp({quality:90,alphaQuality:100,effort:5}).toFile(%r);"
          "await sharp(%r).removeAlpha().webp({quality:92,effort:5}).toFile(%r);})()"
          % (pa, os.path.join(OUT_IMP, 'albedo.webp'), pn, os.path.join(OUT_IMP, 'normal.webp')))
    run(['node', '-e', js])
    meta = {
        'version': 1,
        'note': 'Impostor atlas. uv origin bottom-left (three.js flipY). Frame (i=azimuth, j=elevation) of variant v '
                'covers u in [(x0 + i*frameW)/atlasW, (x0+(i+1)*frameW)/atlasW], v in [j*frameH/atlasH, (j+1)*frameH/atlasH]. '
                'Azimuth i*360/az deg is the direction from tree to camera in tree-local three.js coords, measured from +Z '
                'toward +X; elevation positive = camera above. Quad is view-aligned, centred on (0, pivotY, 0), size width x height. '
                'Albedo = sRGB albedo*AO, alpha coverage. Normal atlas (half res) = tree-local three.js normal*0.5+0.5.',
        'albedo': 'albedo.webp', 'normal': 'normal.webp', 'atlasW': W, 'atlasH': H, 'normalScale': 0.5,
        'frameW': IMP_FW, 'frameH': IMP_FH, 'azimuths': IMP_AZ, 'elevations': IMP_EL, 'variants': variants,
    }
    json.dump(meta, open(os.path.join(OUT_IMP, 'impostors.json'), 'w'), indent=1)
    log('impostors written', W, H)


# =============================================================================================
# 7) PREVIEW RENDERS (Cycles, overcast HDRI)
# =============================================================================================
def preview_materials(mats):
    pm = {}
    for k, m in mats.items():
        c = m.copy()
        c.name = k + '_pv'
        nt = c.node_tree
        bsdf = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
        src = bsdf.inputs['Base Color'].links[0].from_socket
        attr = nt.nodes.new('ShaderNodeVertexColor'); attr.layer_name = 'Col'
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(attr.outputs['Color'], sep.inputs['Color'])
        mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'
        mul.inputs['Factor'].default_value = 1.0
        nt.links.new(src, mul.inputs[6]); nt.links.new(sep.outputs['Red'], mul.inputs[7])
        nt.links.new(mul.outputs[2], bsdf.inputs['Base Color'])
        if k == 'foliage':
            bsdf.inputs['Roughness'].default_value = 0.5
            al = bsdf.inputs['Alpha'].links[0].from_socket
            gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.4
            nt.links.new(al, gt.inputs[0]); nt.links.new(gt.outputs[0], bsdf.inputs['Alpha'])
            try:
                bsdf.inputs['Subsurface Weight'].default_value = 0.0
            except Exception:
                pass
        pm[k] = c
    return pm


def setup_world(strength=1.0):
    w = bpy.data.worlds.new('pv_world')
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    env_t = nt.nodes.new('ShaderNodeTexEnvironment')
    env_t.image = bpy.data.images.load(HDRI, check_existing=True)
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = strength
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(env_t.outputs['Color'], bg.inputs['Color'])
    nt.links.new(bg.outputs[0], out.inputs['Surface'])


def ground_plane(size=120, tex_name='forest_ground_04', rep=40):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    g = bpy.context.active_object
    g.name = 'pv_ground'
    for loop in g.data.uv_layers.active.data:
        loop.uv = (loop.uv[0] * rep, loop.uv[1] * rep)
    d = os.path.join(PUBTEX, tex_name)
    m = export_material('pv_ground_' + tex_name, load_image(os.path.join(d, 'diffuse.jpg')),
                        load_image(os.path.join(d, 'nor_gl.jpg'), colorspace='Non-Color'),
                        load_image(os.path.join(d, 'arm.jpg'), colorspace='Non-Color'))
    g.data.materials.append(m)
    return g


def persp_cam(loc, target, lens=32):
    cam = bpy.data.cameras.new('pv_cam')
    cam.lens = lens
    cam.clip_start = 0.05
    cam.clip_end = 2000
    ob = bpy.data.objects.new('pv_cam', cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return ob


def render_preview(path, cam, samples, res=(1280, 720)):
    sc = bpy.context.scene
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.diffuse_bounces = 3
    sc.cycles.transparent_max_bounces = 64
    sc.cycles.filter_width = 1.5
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.exposure = 0.3
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_depth = '8'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    log('preview', os.path.basename(path), '%.0fs' % (time.time() - t))


def place(src, loc, rotz=0.0, coll=None):
    ob = src.copy()
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (0, 0, rotz)
    ob.hide_render = False
    return ob


def previews(objs, mats):
    sc = bpy.context.scene
    for ob in bpy.data.objects:
        ob.hide_render = True
    pm = preview_materials(mats)
    swap_materials([ob.data for ob in objs.values()], {k: v for k, v in pm.items()})
    setup_world(1.0)
    ground_plane()
    S = ARGS['preview_samples']
    which = ARGS.get('preview', 'all')
    tmp = []
    if which in ('all', 'lineup'):
        xs = [-16, -8, 0, 8, 16]
        for x, k in zip(xs, VARIANTS):
            tmp.append(place(objs[k], (x, 0, 0), 0.4 * x))
        for x, k in ((-7, 'shrub_0'), (-1.5, 'shrub_1'), (3.5, 'shrub_2'), (8.5, 'deadwood_0'), (0.5, 'deadwood_1')):
            tmp.append(place(objs[k], (x, -9, 0), 0.7))
        cam = persp_cam((0, -48, 1.7), (0, 0, 10.5), lens=28)
        render_preview(os.path.join(SCRATCH, 'preview_lineup.png'), cam, S)
        for ob in tmp: bpy.data.objects.remove(ob)
        tmp = []
    if which in ('all', 'lod'):
        tmp.append(place(objs['conifer_0'], (-4.5, 0, 0)))
        tmp.append(place(objs['conifer_0_lod1'], (4.5, 0, 0)))
        cam = persp_cam((0, -34, 1.7), (0, 0, 10.5), lens=32)
        render_preview(os.path.join(SCRATCH, 'preview_lod.png'), cam, S)
        for ob in tmp: bpy.data.objects.remove(ob)
        tmp = []
    if which in ('all', 'close'):
        tmp.append(place(objs['conifer_2'], (0, 0, 0)))
        tmp.append(place(objs['conifer_0'], (-6, 7, 0), 1.0))
        tmp.append(place(objs['shrub_2'], (1.8, -2.2, 0)))
        cam = persp_cam((1.2, -4.2, 1.6), (0, 0, 4.0), lens=24)
        render_preview(os.path.join(SCRATCH, 'preview_close.png'), cam, S)
        for ob in tmp: bpy.data.objects.remove(ob)
        tmp = []
    if which in ('all', 'fallen'):
        tmp.append(place(objs['fallen_tree'], (0, 0, 0)))
        cam = persp_cam((-2.5, -8.5, 1.65), (0.5, 0, 0.4), lens=26)
        render_preview(os.path.join(SCRATCH, 'preview_fallen.png'), cam, S)
        for ob in tmp: bpy.data.objects.remove(ob)
        tmp = [place(objs['fallen_tree_a'], (-0.6, 0.3, 0)), place(objs['fallen_tree_b'], (0.8, -0.5, 0), -0.12)]
        cam = persp_cam((1.5, -3.2, 1.4), (0, 0, 0.35), lens=30)
        render_preview(os.path.join(SCRATCH, 'preview_cut.png'), cam, S)
        for ob in tmp: bpy.data.objects.remove(ob)


# =============================================================================================
# main
# =============================================================================================
def main():
    st = ARGS['stages']
    if 'cards' in st:
        render_card_atlas(ARGS['force_cards'])
        render_card_atlas(ARGS['force_cards'], grass=True)
    if st & {'models', 'impostors', 'export', 'preview'}:
        objs, mats, tex, skels, stats = build_models()
        json.dump(stats, open(os.path.join(BUILD, 'tri_stats.json'), 'w'), indent=1)
        if 'export' in st:
            export_glb(objs)
        if 'impostors' in st:
            render_impostors(objs, mats, tex, skels)
        if 'preview' in st:
            previews(objs, mats)
    log('done')


if __name__ == '__main__':
    main()
