"""
LANDSLIDE - roadside & roadworks props (PROPS workstream).

Every mesh is modelled from scratch here (no downloaded models). CC0 Poly Haven scans (raw_assets/tex) are only
used as texture sources inside procedural Cycles materials that are baked into four shared 2048 atlases:

    metal : jerrycan, hatchet, toolbox, light_tower (+ lamp_head)
    wood  : plank, bearer, plank_wedge, crate, sandbag, tarp_pile
    road  : barrier, cone, delineator, rail_reflector, sign_rockfall, sign_roadworks
    rail  : guardrail, guardrail_end

Per atlas the bake writes albedo (sRGB), tangent-space normal (OpenGL, +Y) and ORM (R=AO, G=rough, B=metal).
Wear masks are procedural: Bevel/AO-inside edge masks (chipped paint, polished edges), AO cavity masks (grime),
height gradients (mud splash), stretched noise (rain/rust streaks). Sign faces are rasterised in numpy.

Nodes written to public/assets/models/props.glb (three.js space, +Y up; origin at the base centre unless noted):
    jerrycan, hatchet (origin at the grip, handle along +Y, blade edge toward +Z), plank (length along X),
    plank_stack (12 boards: 2 columns x 6 layers on 2 bearers), plank_wedge (ramp wedge, rises toward +Z),
    barrier (length along X), cone, crate, toolbox, light_tower (child lamp_head at the mast top; lamps face +Z;
    lens material 'lamp'), sign_rockfall / sign_roadworks (face toward +Z), guardrail (post at the origin, beam runs
    along +Z for 4.0 m, traffic face toward +X), guardrail_bent, guardrail_lod, guardrail_end (terminal that
    continues along +Z and curls toward -X), delineator (reflector toward +Z), rail_reflector (in guardrail space),
    sandbag, tarp_pile.

Usage:
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -P tools/blender/props.py -- [options]
    --only a,b         build only these props (look-dev); implies --no-export
    --bake all|none|metal,wood,road,rail    atlases to (re)bake (default all). Others reuse scratch/props/bake/*.png
                       (geometry + unwrap are deterministic, so cached bakes stay aligned)
    --size 2048        atlas size       --samples 12   bake AA samples
    --clay             clay preview renders only (geometry check, no bake, no export)
    --preview all|none|a,b   baked preview renders (default all)
    --no-export        skip the GLB
"""
import bpy
import bmesh
import sys
import os
import math
import time
import json
import subprocess
import numpy as np
from mathutils import Vector, Matrix, Euler

T0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SCR = os.path.join(ROOT, 'scratch', 'props')
BAKE = os.path.join(SCR, 'bake')
OUT_GLB = os.path.join(ROOT, 'public', 'assets', 'models', 'props.glb')
TEX = os.path.join(ROOT, 'raw_assets', 'tex')
HDRI = os.path.join(ROOT, 'raw_assets', 'hdri', 'overcast_soil_puresky_2k.hdr')
os.makedirs(BAKE, exist_ok=True)

ATLASES = {
    'metal': ['jerrycan', 'hatchet', 'toolbox', 'light_tower', 'lamp_head'],
    'wood': ['plank', 'bearer', 'plank_wedge', 'crate', 'sandbag', 'tarp_pile'],
    'road': ['barrier', 'cone', 'delineator', 'rail_reflector', 'sign_rockfall', 'sign_roadworks', 'km_post', 'snow_pole'],
    'rail': ['guardrail', 'guardrail_end', 'culvert_grate', 'sign_bend', 'sign_chains'],
}
ATLAS_OF = {p: a for a, ps in ATLASES.items() for p in ps}
AO_DIST = {'metal': 0.12, 'wood': 0.25, 'road': 0.2, 'rail': 0.2}


def log(*a):
    print('[props %6.1fs]' % (time.time() - T0), *a, flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    a = {'only': None, 'bake': 'all', 'size': 2048, 'samples': 12, 'clay': False, 'preview': 'all', 'export': True}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == '--only':
            a['only'] = argv[i + 1].split(','); i += 1
        elif k == '--bake':
            a['bake'] = argv[i + 1]; i += 1
        elif k == '--size':
            a['size'] = int(argv[i + 1]); i += 1
        elif k == '--samples':
            a['samples'] = int(argv[i + 1]); i += 1
        elif k == '--preview':
            a['preview'] = argv[i + 1]; i += 1
        elif k == '--clay':
            a['clay'] = True
        elif k == '--no-export':
            a['export'] = False
        i += 1
    if a['only'] or a['clay']:
        a['export'] = False
    return a


ARGS = parse_args()
RNG = np.random.default_rng(1234)


# ================================================================================================
# scene / image utilities
# ================================================================================================
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


_IMG = {}


def load_image(path, noncolor=False):
    key = (path, noncolor)
    if key in _IMG:
        return _IMG[key]
    im = bpy.data.images.load(path, check_existing=False)
    im.colorspace_settings.name = 'Non-Color' if noncolor else 'sRGB'
    _IMG[key] = im
    return im


def tex(name, kind='diffuse'):
    for ext in ('jpg', 'png'):
        p = os.path.join(TEX, name, f'{kind}.{ext}')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{name}/{kind}')


def srgb(c):
    """sRGB 0..255 or 0..1 tuple -> linear RGBA tuple for node defaults."""
    c = [v / 255.0 if max(c) > 1.0 else v for v in c[:3]]
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return (lin[0], lin[1], lin[2], 1.0)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def np_image(name, arr, path=None, noncolor=True):
    """arr (H,W,C) float 0..1, row 0 = TOP. Stored as a byte image (no colour transform when Non-Color)."""
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    H, W, C = arr.shape
    rgba = np.ones((H, W, 4), np.float32)
    rgba[..., :min(C, 3)] = arr[..., :min(C, 3)]
    if C == 1:
        rgba[..., 1] = rgba[..., 2] = arr[..., 0]
    if C == 4:
        rgba[..., 3] = arr[..., 3]
    im = bpy.data.images.new(name, W, H, alpha=True, float_buffer=False)
    im.colorspace_settings.name = 'Non-Color' if noncolor else 'sRGB'
    im.pixels.foreach_set(rgba[::-1].ravel())
    if path:
        im.filepath_raw = path
        im.file_format = 'PNG'
        im.save()
    return im


def read_pixels(im):
    a = np.empty(im.size[0] * im.size[1] * 4, np.float32)
    im.pixels.foreach_get(a)
    return a.reshape(im.size[1], im.size[0], 4)[::-1]   # row 0 = top


# ================================================================================================
# 2D raster helpers (sign faces, emboss maps)
# ================================================================================================
def raster(W, H, polys, extent, ss=3):
    """Even-odd fill of polygons (list of (N,2) arrays in metres, x right / y up) into an (H,W) coverage map.
    extent = (x0, x1, y0, y1) in metres mapped to the image (row 0 = y1)."""
    x0, x1, y0, y1 = extent
    Hs, Ws = H * ss, W * ss
    cov = np.zeros((Hs, Ws), bool)
    for P in polys:
        P = np.asarray(P, np.float64)
        px = (P[:, 0] - x0) / (x1 - x0) * Ws
        py = (y1 - P[:, 1]) / (y1 - y0) * Hs
        c0, c1 = int(max(0, np.floor(px.min()))), int(min(Ws, np.ceil(px.max()) + 1))
        r0, r1 = int(max(0, np.floor(py.min()))), int(min(Hs, np.ceil(py.max()) + 1))
        if c1 <= c0 or r1 <= r0:
            continue
        X, Y = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)
        ins = np.zeros(X.shape, bool)
        n = len(px)
        for i in range(n):
            ax, ay, bx, by = px[i], py[i], px[(i + 1) % n], py[(i + 1) % n]
            if ay == by:
                continue
            cond = ((ay > Y) != (by > Y)) & (X < (bx - ax) * (Y - ay) / (by - ay) + ax)
            ins ^= cond
        cov[r0:r1, c0:c1] |= ins
    return cov.reshape(H, ss, W, ss).mean(axis=(1, 3))


def circle(cx, cy, r, n=48, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r * (1 + jitter * (rng.random(n) - 0.5))
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], -1)


def rock_poly(cx, cy, r, seed):
    rng = np.random.default_rng(seed)
    n = 7 + int(rng.integers(0, 3))
    a = np.sort(rng.random(n)) * 2 * np.pi + rng.random() * 6
    rr = r * (0.7 + 0.45 * rng.random(n))
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], -1)


def thick_line(p0, p1, w):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12) * w / 2
    return np.array([p0 + n, p1 + n, p1 - n, p0 - n])


def rounded_triangle(side, r, cy=0.0, n=10):
    """Equilateral triangle (apex up) with rounded corners; centroid at (0, cy)."""
    h = side * math.sqrt(3) / 2
    V = np.array([[-side / 2, -h / 3], [side / 2, -h / 3], [0, 2 * h / 3]])
    pts = []
    for i in range(3):
        a, b, c = V[i - 1], V[i], V[(i + 1) % 3]
        u1 = (a - b) / np.linalg.norm(a - b)
        u2 = (c - b) / np.linalg.norm(c - b)
        bis = (u1 + u2) / np.linalg.norm(u1 + u2)
        ang = math.acos(np.clip(np.dot(u1, u2), -1, 1))
        dcen = r / math.sin(ang / 2)
        cen = b + bis * dcen
        t1 = b + u1 * r / math.tan(ang / 2)
        t2 = b + u2 * r / math.tan(ang / 2)
        a1 = math.atan2(*(t1 - cen)[::-1])
        a2 = math.atan2(*(t2 - cen)[::-1])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        for k in range(n + 1):
            aa = a1 + da * k / n
            pts.append(cen + r * np.array([math.cos(aa), math.sin(aa)]))
    P = np.array(pts)
    P[:, 1] += cy
    return P


def inset_poly(P, d):
    """Offset a convex CCW polygon inward by d (vertex normal offset, fine for rounded shapes)."""
    n = len(P)
    out = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e1 = b - a
        e2 = c - b
        n1 = np.array([-e1[1], e1[0]]) / (np.linalg.norm(e1) + 1e-12)
        n2 = np.array([-e2[1], e2[0]]) / (np.linalg.norm(e2) + 1e-12)
        m = n1 + n2
        m /= np.linalg.norm(m) + 1e-12
        cosh = max(np.dot(m, n1), 0.3)
        out.append(b + m * d / cosh)
    return np.array(out)


def poly_area(P):
    return 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])


# ================================================================================================
# geometry
# ================================================================================================
class MB:
    """Mesh builder: accumulate parts (vertex array + faces + material slot), then build one object."""

    def __init__(self):
        self.V, self.F, self.M, self.n = [], [], [], 0
        self.ranges = {}

    def add(self, V, F, mat=0, M=None, recalc=True, tag=None):
        V = np.asarray(V, np.float64).reshape(-1, 3)
        F = [list(f) for f in F]
        if recalc:
            V, F = _recalc(V, F)
        if M is not None:
            M = np.asarray(M, np.float64)
            V = V @ M[:3, :3].T + M[:3, 3]
        start = self.n
        self.V.append(V)
        self.F.extend([[i + self.n for i in f] for f in F])
        self.M.extend([mat] * len(F))
        self.n += len(V)
        if tag:
            r = self.ranges.get(tag, [])
            r.append((start, self.n))
            self.ranges[tag] = r
        return start

    def add_bm(self, bm, mat=0, M=None, recalc=True, tag=None):
        bm.verts.index_update()
        V = np.array([v.co[:] for v in bm.verts])
        F = [[v.index for v in f.verts] for f in bm.faces]
        bm.free()
        return self.add(V, F, mat, M, recalc, tag)

    def verts(self):
        return np.concatenate(self.V) if self.V else np.zeros((0, 3))

    def build(self, name, mats, sharp=40.0, V=None):
        me = bpy.data.meshes.new(name)
        VV = self.verts() if V is None else V
        me.from_pydata(VV.tolist(), [], self.F)
        me.validate(clean_customdata=False)
        me.update()
        if len(me.polygons) == len(self.M):
            me.polygons.foreach_set('material_index', self.M)
        for m in mats:
            me.materials.append(m)
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(sharp))
        obj = link(bpy.data.objects.new(name, me))
        return obj


def _recalc(V, F):
    bm = bmesh.new()
    vs = [bm.verts.new(v) for v in V]
    for f in F:
        try:
            bm.faces.new([vs[i] for i in f])
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.index_update()
    F2 = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    return V, F2


def T(x=0, y=0, z=0):
    return np.array(Matrix.Translation((x, y, z)))


def R(ax, ang_deg):
    return np.array(Matrix.Rotation(math.radians(ang_deg), 4, ax))


def S(sx, sy=None, sz=None):
    sy = sx if sy is None else sy
    sz = sx if sz is None else sz
    return np.diag([sx, sy, sz, 1.0])


def mm(*Ms):
    out = np.eye(4)
    for M in Ms:
        out = out @ M
    return out


def rbox(sx, sy, sz, r=0.0, segs=2, center=(0, 0, 0), sub=0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    if sub:
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=sub, use_grid_fill=True)
    if r > 0:
        bmesh.ops.bevel(bm, geom=list(bm.edges) + list(bm.verts) if False else list(bm.edges), offset=r,
                        offset_type='OFFSET', segments=segs, profile=0.5, affect='EDGES', clamp_overlap=True)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    return bm


def lathe(prof, segs=24, cap0=True, cap1=True, a0=0.0):
    """prof: [(r, z), ...] bottom->top. Returns V, F (outward winding)."""
    V, F = [], []
    n = len(prof)
    for (r, z) in prof:
        for j in range(segs):
            a = a0 + 2 * math.pi * j / segs
            V.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(n - 1):
        for j in range(segs):
            a, b = i * segs + j, i * segs + (j + 1) % segs
            F.append([a, b, b + segs, a + segs])
    if cap0 and prof[0][0] > 1e-6:
        F.append(list(range(segs))[::-1])
    if cap1 and prof[-1][0] > 1e-6:
        F.append(list(range((n - 1) * segs, n * segs)))
    return np.array(V), F


def frames(P, up=(0, 0, 1)):
    P = np.asarray(P, float)
    Tn = np.gradient(P, axis=0)
    Tn /= np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-12
    up = np.array(up, float)
    N = np.zeros_like(P)
    n0 = np.cross(up, Tn[0])
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.cross((1, 0, 0), Tn[0])
    N[0] = n0 / np.linalg.norm(n0)
    for i in range(1, len(P)):
        v = np.cross(Tn[i - 1], Tn[i])
        s = np.linalg.norm(v)
        if s < 1e-9:
            N[i] = N[i - 1]
        else:
            ang = math.atan2(s, np.dot(Tn[i - 1], Tn[i]))
            N[i] = np.array(Matrix.Rotation(ang, 3, Vector(v / s)) @ Vector(N[i - 1]))
    B = np.cross(Tn, N)
    return Tn, N, B


def sweep(P, prof, closed_path=False, caps=True, scale=None, up=(0, 0, 1), N=None, B=None):
    """Sweep a closed 2D profile (m,2) along path P (n,3). prof coords are (along N, along B)."""
    P = np.asarray(P, float)
    prof = np.asarray(prof, float)
    if N is None:
        _, N, B = frames(P, up)
    n, m = len(P), len(prof)
    sc = np.ones(n) if scale is None else np.asarray(scale, float)
    V = (P[:, None, :] + sc[:, None, None] * (prof[None, :, 0:1] * N[:, None, :] + prof[None, :, 1:2] * B[:, None, :]))
    V = V.reshape(-1, 3)
    F = []
    rows = n if closed_path else n - 1
    for i in range(rows):
        i2 = (i + 1) % n
        for j in range(m):
            j2 = (j + 1) % m
            F.append([i * m + j, i * m + j2, i2 * m + j2, i2 * m + j])
    if caps and not closed_path:
        F.append(list(range(m))[::-1])
        F.append(list(range((n - 1) * m, n * m)))
    return V, F


def circ(r, n=8, ry=None):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), (ry if ry else r) * np.sin(a)], -1)


def rrect(w, h, r, n=3):
    """Rounded rectangle (CCW) centred at 0, w x h."""
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def spline(pts, n=24):
    """Catmull-Rom through pts, resampled to n points by arc length."""
    P = np.asarray(pts, float)
    P2 = np.concatenate([P[:1] * 2 - P[1:2], P, P[-1:] * 2 - P[-2:-1]])
    out = []
    for i in range(1, len(P2) - 2):
        p0, p1, p2, p3 = P2[i - 1], P2[i], P2[i + 1], P2[i + 2]
        for t in np.linspace(0, 1, 16, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    out = np.array(out)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))])
    s = np.linspace(0, L[-1], n)
    return np.stack([np.interp(s, L, out[:, k]) for k in range(3)], -1)


def tube(pts, r, sides=8, n=24, caps=True, smooth=True):
    P = spline(pts, n) if smooth else np.asarray(pts, float)
    return sweep(P, circ(r, sides), caps=caps)


def cyl(r, h, segs=16, z0=0.0, r1=None):
    return lathe([(r, z0), (r if r1 is None else r1, z0 + h)], segs)


def icochunk(r, seed, sub=1):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=r)
    rng = np.random.default_rng(seed)
    for v in bm.verts:
        v.co *= 0.75 + 0.5 * rng.random()
    return bm


def hexnut(r=0.009, h=0.007):
    return lathe([(r, 0), (r, h * 0.8), (r * 0.8, h)], 6, a0=math.pi / 6)


def dome(r, h, segs=10):
    prof = [(r, 0)] + [(r * math.cos(t), h * math.sin(t)) for t in np.linspace(0.3, 1.35, 3)] + [(r * 0.15, h)]
    return lathe(prof, segs)


# ================================================================================================
# shader node builder (bake materials)
# ================================================================================================
def _in(n, ident):
    return next(s for s in n.inputs if s.identifier == ident)


def _out(n, ident):
    return next(s for s in n.outputs if s.identifier == ident)


class NB:
    def __init__(self, name):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        self.m = m
        self.nt = m.node_tree
        self.nt.nodes.clear()
        self.out = self.new('ShaderNodeOutputMaterial')
        self.bsdf = self.new('ShaderNodeBsdfPrincipled')
        self.emi = self.new('ShaderNodeEmission')
        self.tc = self.new('ShaderNodeTexCoord')
        self.geo = self.new('ShaderNodeNewGeometry')
        self.h = None
        self._cache = {}

    def new(self, t, **kw):
        n = self.nt.nodes.new(t)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def put(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
            return
        if sock.type == 'RGBA':
            if isinstance(v, (int, float)):
                v = (v, v, v, 1.0)
            elif len(v) == 3:
                v = (*v, 1.0)
        elif sock.type == 'VECTOR' and isinstance(v, (int, float)):
            v = (v, v, v)
        sock.default_value = v

    # --- coordinates
    @property
    def P(self):
        return self.tc.outputs['Object']

    @property
    def N(self):
        return self.geo.outputs['Normal']

    def objN(self):
        """object-space normal"""
        if 'objN' not in self._cache:
            vt = self.new('ShaderNodeVectorTransform', vector_type='NORMAL', convert_from='WORLD', convert_to='OBJECT')
            self.put(vt.inputs[0], self.N)
            self._cache['objN'] = self.vmath('NORMALIZE', vt.outputs[0])
        return self._cache['objN']

    def sep(self, v):
        n = self.new('ShaderNodeSeparateXYZ')
        self.put(n.inputs[0], v)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def comb(self, x, y, z):
        n = self.new('ShaderNodeCombineXYZ')
        self.put(n.inputs[0], x)
        self.put(n.inputs[1], y)
        self.put(n.inputs[2], z)
        return n.outputs[0]

    def rgb(self, r, g=None, b=None):
        n = self.new('ShaderNodeCombineColor')
        self.put(n.inputs[0], r)
        self.put(n.inputs[1], r if g is None else g)
        self.put(n.inputs[2], r if b is None else b)
        return n.outputs[0]

    def mapping(self, v, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        n = self.new('ShaderNodeMapping')
        self.put(n.inputs['Vector'], v)
        n.inputs['Location'].default_value = loc
        n.inputs['Rotation'].default_value = [math.radians(a) for a in rot]
        n.inputs['Scale'].default_value = scale if not isinstance(scale, (int, float)) else (scale,) * 3
        return n.outputs[0]

    # --- math
    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.new('ShaderNodeMath', operation=op, use_clamp=clamp)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        if c is not None:
            self.put(n.inputs[2], c)
        return n.outputs[0]

    def add(self, a, b, clamp=False):
        return self.math('ADD', a, b, clamp=clamp)

    def mul(self, a, b, clamp=False):
        return self.math('MULTIPLY', a, b, clamp=clamp)

    def sub(self, a, b, clamp=False):
        return self.math('SUBTRACT', a, b, clamp=clamp)

    def inv(self, a):
        return self.math('SUBTRACT', 1.0, a, clamp=True)

    def maxf(self, a, b):
        return self.math('MAXIMUM', a, b)

    def minf(self, a, b):
        return self.math('MINIMUM', a, b)

    def absf(self, a):
        return self.math('ABSOLUTE', a)

    def vmath(self, op, a, b=None, s=None):
        n = self.new('ShaderNodeVectorMath', operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        if s is not None:
            self.put(n.inputs['Scale'], s)
        return n.outputs['Value'] if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE') else n.outputs['Vector']

    def mr(self, v, a, b, c=0.0, d=1.0, smooth=False, clamp=True):
        n = self.new('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR', clamp=clamp)
        self.put(n.inputs['Value'], v)
        n.inputs['From Min'].default_value = a
        n.inputs['From Max'].default_value = b
        self.put(n.inputs['To Min'], c)
        self.put(n.inputs['To Max'], d)
        return n.outputs['Result']

    def mix(self, f, a, b, blend='MIX', clamp=True):
        n = self.new('ShaderNodeMix', data_type='RGBA', blend_type=blend, clamp_result=clamp)
        self.put(_in(n, 'Factor_Float'), f)
        self.put(_in(n, 'A_Color'), a)
        self.put(_in(n, 'B_Color'), b)
        return _out(n, 'Result_Color')

    def mixf(self, f, a, b):
        n = self.new('ShaderNodeMix', data_type='FLOAT')
        self.put(_in(n, 'Factor_Float'), f)
        self.put(_in(n, 'A_Float'), a)
        self.put(_in(n, 'B_Float'), b)
        return _out(n, 'Result_Float')

    def hsv(self, c, h=0.5, s=1.0, v=1.0):
        n = self.new('ShaderNodeHueSaturation')
        self.put(n.inputs['Color'], c)
        self.put(n.inputs['Hue'], h)
        self.put(n.inputs['Saturation'], s)
        self.put(n.inputs['Value'], v)
        return n.outputs['Color']

    def lum(self, c):
        n = self.new('ShaderNodeRGBToBW')
        self.put(n.inputs[0], c)
        return n.outputs[0]

    # --- textures
    def noise(self, v, scale, detail=4.0, rough=0.55, dist=0.0, lac=2.0):
        n = self.new('ShaderNodeTexNoise')
        self.put(n.inputs['Vector'], v if v is not None else self.P)
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        n.inputs['Roughness'].default_value = rough
        n.inputs['Distortion'].default_value = dist
        n.inputs['Lacunarity'].default_value = lac
        return n.outputs['Factor']

    def noisec(self, v, scale, detail=3.0, rough=0.5):
        n = self.new('ShaderNodeTexNoise')
        self.put(n.inputs['Vector'], v if v is not None else self.P)
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        n.inputs['Roughness'].default_value = rough
        return n.outputs['Color']

    def vor(self, v, scale, feature='F1', rand=1.0, out='Distance'):
        n = self.new('ShaderNodeTexVoronoi', feature=feature)
        self.put(n.inputs['Vector'], v if v is not None else self.P)
        n.inputs['Scale'].default_value = scale
        n.inputs['Randomness'].default_value = rand
        return n.outputs[out]

    def img(self, path, v, proj='BOX', blend=0.25, noncolor=False, ext='REPEAT', image=None):
        n = self.new('ShaderNodeTexImage')
        n.image = image if image is not None else load_image(path, noncolor)
        n.projection = proj
        if proj == 'BOX':
            n.projection_blend = blend
        n.extension = ext
        self.put(n.inputs['Vector'], v)
        return n.outputs['Color']

    def tri(self, name, kind, P, tile, axis=None, noncolor=False, sharp=4.0):
        """Triplanar sampling in object space; if `axis` is given the texture V axis (grain) runs along it on every
        face where possible. P: object coords (m). Returns colour."""
        im = load_image(tex(name, kind), noncolor)
        x, y, z = self.sep(P)
        comp = {'x': x, 'y': y, 'z': z}
        k = 1.0 / tile
        planes = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}
        nx, ny, nz = self.sep(self.objN())
        w = {}
        for ax, n in (('x', nx), ('y', ny), ('z', nz)):
            w[ax] = self.math('POWER', self.absf(n), sharp)
        cols = {}
        for pl, (a, b) in planes.items():
            if axis is not None and axis in (a, b):
                u, v = (b if axis == a else a), axis
            else:
                u, v = a, b
            uv = self.comb(self.mul(comp[u], k), self.mul(comp[v], k), 0.0)
            n = self.new('ShaderNodeTexImage')
            n.image = im
            n.extension = 'REPEAT'
            self.put(n.inputs['Vector'], uv)
            cols[pl] = n.outputs['Color']
        tot = self.add(self.add(w['x'], w['y']), w['z'])
        wxy = self.math('DIVIDE', w['y'], self.add(self.add(w['x'], w['y']), 1e-5))
        c = self.mix(wxy, cols['x'], cols['y'])
        c = self.mix(self.math('DIVIDE', w['z'], self.add(tot, 1e-5)), c, cols['z'])
        return c

    def wave(self, v, scale, distort=4.0, detail=3.0, dscale=1.0, axis='Z', profile='SIN'):
        n = self.new('ShaderNodeTexWave', wave_type='BANDS', bands_direction=axis, wave_profile=profile)
        self.put(n.inputs['Vector'], v if v is not None else self.P)
        n.inputs['Scale'].default_value = scale
        n.inputs['Distortion'].default_value = distort
        n.inputs['Detail'].default_value = detail
        n.inputs['Detail Scale'].default_value = dscale
        return n.outputs['Fac']

    def ramp(self, f, stops):
        n = self.new('ShaderNodeValToRGB')
        self.put(n.inputs[0], f)
        els = n.color_ramp.elements
        for i, (pos, col) in enumerate(stops):
            e = els[i] if i < 2 else els.new(pos)
            e.position = pos
            e.color = col if len(col) == 4 else (*col, 1)
        return n.outputs['Color']

    # --- masks
    def bevel(self, r, samples=8, normal=None):
        n = self.new('ShaderNodeBevel', samples=samples)
        n.inputs['Radius'].default_value = r
        if normal is not None:
            self.put(n.inputs['Normal'], normal)
        return n.outputs['Normal']

    def ao(self, dist, inside=False, local=True, samples=16):
        n = self.new('ShaderNodeAmbientOcclusion', samples=samples, inside=inside, only_local=local)
        n.inputs['Distance'].default_value = dist
        return n.outputs['AO']

    def edge(self, r=0.004, lo=0.05, hi=0.35, samples=8):
        """Convex-edge mask: bevel normal deviation + inside-AO."""
        d = self.vmath('DOT_PRODUCT', self.bevel(r, samples), self.N)
        e1 = self.mr(self.sub(1.0, d), lo * 0.2, hi * 0.2)
        e2 = self.mr(self.ao(r * 2.5, inside=True, samples=samples), 0.85, 0.45)
        return self.maxf(e1, e2)

    def cavity(self, dist=0.03, lo=0.95, hi=0.5, samples=12):
        return self.mr(self.ao(dist, samples=samples), lo, hi)

    def bump(self, h, strength=1.0, dist=0.001, normal=None):
        n = self.new('ShaderNodeBump')
        self.put(n.inputs['Height'], h)
        n.inputs['Strength'].default_value = strength
        n.inputs['Distance'].default_value = dist
        if normal is not None:
            self.put(n.inputs['Normal'], normal)
        return n.outputs['Normal']

    def finish(self, base, rough, metal=0.0, normal=None, lamp=False):
        self.put(self.bsdf.inputs['Base Color'], base)
        self.put(self.bsdf.inputs['Roughness'], rough)
        self.put(self.bsdf.inputs['Metallic'], metal)
        if normal is not None:
            self.put(self.bsdf.inputs['Normal'], normal)
        data = self.new('ShaderNodeCombineColor')
        self.put(data.inputs[0], rough)
        self.put(data.inputs[1], metal)
        self.put(data.inputs[2], 0.0)
        self.h = {'albedo': self.bsdf.inputs['Base Color'].links[0].from_socket if self.bsdf.inputs['Base Color'].links else None,
                  'data': data.outputs[0], 'base_const': None if self.bsdf.inputs['Base Color'].links else tuple(self.bsdf.inputs['Base Color'].default_value)}
        self.nt.links.new(self.bsdf.outputs[0], self.out.inputs['Surface'])
        self.m['bake'] = True
        _MATS.append(self)
        return self.m


_MATS = []


def set_pass(nb, which):
    nt = nb.nt
    for l in list(nt.links):
        if l.to_node == nb.out or l.to_node == nb.emi:
            nt.links.remove(l)
    if which in ('normal', 'ao'):
        nt.links.new(nb.bsdf.outputs[0], nb.out.inputs['Surface'])
        return
    if which == 'albedo':
        if nb.h['albedo'] is not None:
            nt.links.new(nb.h['albedo'], nb.emi.inputs['Color'])
        else:
            nb.emi.inputs['Color'].default_value = nb.h['base_const']
    else:
        nt.links.new(nb.h['data'], nb.emi.inputs['Color'])
    nb.emi.inputs['Strength'].default_value = 1.0
    nt.links.new(nb.emi.outputs[0], nb.out.inputs['Surface'])


# ------------------------------------------------------------------------------------------------
# material library
# ------------------------------------------------------------------------------------------------
def stretch(B, P, sx, sy, sz):
    return B.mapping(P, scale=(sx, sy, sz))


def M_paint(name, color, *, chip=0.5, rust=0.4, dirt=0.5, mud=0.3, mud_h=0.08, gloss=0.45, fade=0.3, emboss=None,
            edge_r=0.004, scratches=0.5, stain=None, seed=0, primer=(120, 70, 55), zspan=0.5, dents=0.4, edgebox=None):
    """Chipped painted steel. emboss: dict(image, x0, x1, z0, z1, axis='y', depth) planar height map."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed * 3.1, seed * 1.7, seed * 0.9))
    Pz = B.sep(B.P)[2]
    nzo = B.sep(B.objN())[2]
    paint = srgb(color)
    # paint colour variation + sun fade on upward faces
    v1 = B.noise(P, 3.0, 4, 0.6)
    v2 = B.noise(P, 18.0, 3, 0.5)
    col = B.hsv(paint, 0.5 + 0.015 * 0, 1.0, 1.0)
    col = B.mix(B.mr(v1, 0.35, 0.65, 0.0, 1.0), B.hsv(paint, 0.49, 0.9, 0.86), B.hsv(paint, 0.51, 1.08, 1.12))
    col = B.mix(B.mul(B.mr(v2, 0.4, 0.7), 0.35), col, B.hsv(paint, 0.5, 0.85, 1.2))
    up = B.mr(nzo, 0.2, 0.95)
    col = B.mix(B.mul(up, fade), col, B.hsv(col, 0.5, 0.55, 1.35))
    # chips: along convex edges (broken up by noise) + scattered impact chips
    e = B.edge(edge_r, samples=8)
    if edgebox is not None:
        # object-space proximity to the outer box edges (catches large-radius rounded corners the bevel misses)
        hx, hy, bz0, bz1, bw = edgebox
        ex_, ey_, ez_ = B.sep(B.P)
        px = B.mr(B.absf(ex_), hx - bw, hx)
        py = B.mr(B.absf(ey_), hy - bw * 0.7, hy)
        pz = B.maxf(B.mr(ez_, bz1 - bw, bz1), B.mr(ez_, bz0 + bw, bz0))
        eb = B.maxf(B.maxf(B.minf(px, py), B.minf(px, pz)), B.minf(py, pz))
        brk = B.mr(B.noise(P, 16.0, 6, 0.7), 0.42, 0.62)
        e = B.maxf(e, B.mul(eb, B.mul(brk, 0.95)))
    cn = B.noise(P, 26.0, 8, 0.72)
    ch_edge = B.mr(B.mul(e, B.mr(cn, 0.3, 0.75, 0.2, 1.6)), 0.55 - chip * 0.3, 0.6 - chip * 0.3)
    sc = B.noise(P, 38.0, 8, 0.7)
    ch_spot = B.mr(sc, 0.74 - chip * 0.06, 0.76 - chip * 0.06)
    ch_spot = B.mul(ch_spot, B.mr(B.noise(P, 4.0, 2), 0.45, 0.6))
    chips = B.maxf(ch_edge, ch_spot)
    prim_zone = B.maxf(B.mr(B.mul(e, B.mr(cn, 0.3, 0.75, 0.2, 1.6)), 0.4 - chip * 0.25, 0.46 - chip * 0.25), B.mr(sc, 0.69 - chip * 0.06, 0.71 - chip * 0.06))
    # scratches: long thin voronoi edges, random direction per patch
    scr = B.vor(B.mapping(P, rot=(20, 35, 10), scale=(90, 8, 90)), 1.0, 'DISTANCE_TO_EDGE')
    scr = B.mul(B.mr(scr, 0.02, 0.0), B.mr(B.noise(P, 6.0, 2), 0.5, 0.65))
    scr = B.mul(scr, scratches)
    # bare steel + rust inside chips
    steel = B.mix(B.mr(B.noise(P, 60.0, 4), 0.3, 0.7), srgb((92, 92, 95)), srgb((132, 131, 128)))
    rust_t = B.tri('rust_coarse_01', 'diffuse', P, 0.5)
    rn = B.noise(P, 14.0, 6, 0.65)
    rmask = B.mr(rn, 0.52 - rust * 0.3, 0.62 - rust * 0.3)
    bare = B.mix(rmask, steel, rust_t)
    primer_c = srgb(primer)
    col = B.mix(B.mul(prim_zone, 0.9), col, primer_c)
    col = B.mix(chips, col, bare)
    col = B.mix(B.mul(scr, 0.7), col, B.hsv(col, 0.5, 0.4, 1.6))
    # rust bloom/bleed around chips and at the bottom
    bleed = B.mul(B.mr(B.noise(P, 9.0, 5, 0.6), 0.55, 0.75), B.mul(rust, B.mr(Pz, 0.08, 0.0)))
    col = B.mix(B.mul(bleed, 0.7), col, rust_t)
    # grime in cavities, dust on top, mud at the bottom
    cav = B.cavity(0.025)
    grime = srgb((62, 55, 44))
    col = B.mix(B.mul(cav, dirt * 0.85), col, grime)
    dust = B.mul(B.mul(up, dirt * 0.35), B.mr(B.noise(P, 7.0, 4), 0.35, 0.7))
    col = B.mix(dust, col, srgb((150, 138, 118)))
    mudt = B.hsv(B.tri('brown_mud_rocks_01', 'diffuse', P, 0.7), 0.5, 0.8, 0.55)
    mz = B.mr(B.add(Pz, B.mul(B.sub(B.noise(P, 11.0, 5, 0.7), 0.5), 0.08)), mud_h, mud_h * 0.25)
    mudm = B.mul(mz, mud)
    col = B.mix(mudm, col, mudt)
    if stain is not None:
        # dark glossy fuel/oil runs below a point (x, z) in object space
        sx, sz, sw = stain
        x, y, z = B.sep(B.P)
        dx = B.absf(B.sub(x, sx))
        run = B.mul(B.mr(dx, sw, sw * 0.3), B.mr(z, sz, sz - 0.25))
        streak = B.mr(B.noise(B.mapping(P, scale=(60, 60, 4)), 1.0, 3), 0.45, 0.62)
        stm = B.mul(run, streak)
        col = B.mix(B.mul(stm, 0.6), col, B.hsv(col, 0.5, 1.1, 0.45))
    else:
        stm = 0.0
    # roughness
    r_paint = B.mr(v2, 0.3, 0.7, gloss - 0.08, gloss + 0.1)
    rough = B.mixf(chips, r_paint, B.mixf(rmask, 0.38, 0.85))
    rough = B.mixf(B.mul(up, fade * 0.8), rough, 0.8)
    rough = B.mixf(B.mul(cav, dirt), rough, 0.9)
    rough = B.mixf(mudm, rough, 0.95)
    rough = B.mixf(dust, rough, 0.92)
    if stain is not None:
        rough = B.mixf(stm, rough, 0.15)
    metal = B.mul(chips, B.inv(B.mul(rmask, 1.0)))
    metal = B.mul(metal, B.inv(mudm))
    # normal: rounded edges, dents, orange peel, chip steps, emboss
    nrm = B.bevel(0.0015, 6)
    if emboss:
        x, y, z = B.sep(B.P)
        a = emboss.get('axis', 'y')
        uu = B.mr(x, emboss['x0'], emboss['x1'], 0, 1, clamp=False)
        vv = B.mr(z, emboss['z0'], emboss['z1'], 0, 1, clamp=False)
        eh = B.img(None, B.comb(uu, vv, 0), proj='FLAT', noncolor=True, ext='EXTEND', image=emboss['image'])
        eh = B.sep(eh)[0]
        facing = B.mr(B.absf(B.sep(B.objN())[1]), 0.6, 0.85)
        nrm = B.bump(B.mul(eh, facing), 1.0, emboss['depth'], nrm)
    dent = B.noise(P, 2.2, 2, 0.5)
    nrm = B.bump(dent, dents, 0.004, nrm)
    peel = B.noise(P, 260.0, 2, 0.5)
    nrm = B.bump(peel, 0.05, 0.0004, nrm)
    nrm = B.bump(B.add(B.mul(chips, -1.0), B.mul(rmask, B.mul(chips, B.noise(P, 80.0, 4) ))), 0.6, 0.0004, nrm)
    nrm = B.bump(mudm, 0.4, 0.001, nrm)
    return B.finish(col, rough, metal, nrm)


def M_galv(name, *, rust=0.4, grime=0.5, seed=0, spray_h=0.35, zmax=0.8, white_rust=0.4, streak_axis='y'):
    """Weathered hot-dip galvanised steel: spangle, zinc patina, white rust, rust streaks, road-spray grime."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed * 2.3, seed * 0.7, seed * 1.1))
    Pz = B.sep(B.P)[2]
    spang = B.vor(P, 28.0, 'F1', 1.0, 'Color')
    sp = B.sep(spang)[0]
    base = B.mix(B.mr(sp, 0.0, 1.0), srgb((124, 127, 127)), srgb((150, 152, 150)))
    mott = B.noise(P, 5.0, 5, 0.6)
    base = B.mix(B.mr(mott, 0.35, 0.7), B.hsv(base, 0.5, 0.9, 0.78), base)
    # dull zinc patina (grey-white, dielectric)
    pat = B.mr(B.noise(P, 2.5, 4, 0.6), 0.35, 0.6)
    patc = srgb((136, 138, 134))
    base = B.mix(B.mul(pat, 0.7), base, patc)
    wr = B.mul(B.mr(B.noise(P, 16.0, 6, 0.7), 0.64 - white_rust * 0.1, 0.72 - white_rust * 0.1), white_rust)
    base = B.mix(wr, base, srgb((196, 196, 188)))
    # rust: streaks running down from holes/edges (stretched noise), edge rust
    streak = B.noise(B.mapping(P, scale=(55, 55, 2.2)), 1.0, 4, 0.55)
    sm = B.mul(B.mr(streak, 0.56 - rust * 0.15, 0.7 - rust * 0.15), B.mr(B.noise(P, 1.7, 2), 0.42, 0.62))
    e = B.edge(0.003, samples=6)
    er = B.mul(B.mul(e, B.mr(B.noise(P, 30.0, 6), 0.45, 0.7)), rust)
    rmask = B.maxf(B.mul(sm, rust * 1.4), er)
    rust_t = B.tri('rust_coarse_01', 'diffuse', P, 0.45)
    rust_c = B.hsv(rust_t, 0.5, 1.0, 0.9)
    base = B.mix(B.mul(rmask, 0.85), base, rust_c)
    # road spray grime: fine speckle, heavier toward the bottom
    spk = B.mr(B.noise(P, 90.0, 3, 0.6), 0.45, 0.75)
    low = B.mr(Pz, spray_h, 0.0)
    gm = B.mul(B.add(B.mul(low, 0.9), 0.15), B.mul(spk, grime))
    gcol = srgb((88, 80, 66))
    base = B.mix(gm, base, gcol)
    cav = B.cavity(0.03)
    base = B.mix(B.mul(cav, grime * 0.8), base, srgb((60, 55, 46)))
    rough = B.mr(mott, 0.3, 0.7, 0.5, 0.7)
    rough = B.mixf(pat, rough, 0.78)
    rough = B.mixf(rmask, rough, 0.88)
    rough = B.mixf(B.maxf(gm, wr), rough, 0.9)
    metal = B.mul(B.inv(B.maxf(B.mul(pat, 0.5), rmask)), B.inv(B.maxf(gm, wr)))
    metal = B.mul(metal, 0.8)
    nrm = B.bevel(0.0012, 6)
    nrm = B.bump(B.noise(P, 120.0, 3), 0.08, 0.0003, nrm)
    nrm = B.bump(B.mul(rmask, B.noise(P, 60.0, 5)), 0.5, 0.0005, nrm)
    return B.finish(base, rough, metal, nrm)


def M_wood(name, *, texname='rough_wood', tile=1.0, axis='x', tint=(1.0, 1.0, 1.0), grey=0.35, dirt=0.5,
           mud=0.4, mud_h=0.06, ends=None, splash=0.3, seed=0, val=1.0, sat=1.0, wet_ends=0.4, rough_base=0.82):
    """Weathered timber. ends: (x_min, x_max) for end grain + mud/wet stains near the ends."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed * 1.9, seed * 0.4, seed * 0.7))
    x, y, z = B.sep(B.P)
    w = B.tri(texname, 'diffuse', P, tile, axis=axis)
    w = B.hsv(w, 0.5, sat, val)
    w = B.mix(1.0, w, tint, 'MULTIPLY')
    # silver-grey weathering patches
    gm = B.mr(B.noise(P, 1.2, 4, 0.6), 0.35, 0.7)
    w = B.mix(B.mul(gm, grey), w, B.hsv(w, 0.5, 0.25, 1.15))
    # per-board colour shift
    w = B.mix(B.mr(B.noise(P, 0.35, 1), 0.3, 0.7), B.hsv(w, 0.49, 0.9, 0.85), B.hsv(w, 0.51, 1.1, 1.1))
    cav = B.cavity(0.02, 0.97, 0.6)
    w = B.mix(B.mul(cav, dirt), w, srgb((40, 33, 26)))
    e = B.edge(0.004, samples=6)
    w = B.mix(B.mul(B.mul(e, 0.35), B.mr(B.noise(P, 20.0, 4), 0.4, 0.7)), w, B.hsv(w, 0.5, 0.6, 1.35))
    # concrete / mortar splashes
    spl = B.mr(B.noise(P, 9.0, 6, 0.7), 0.7, 0.73)
    spl = B.mul(B.mul(spl, splash), B.mr(B.noise(P, 1.3, 2), 0.5, 0.65))
    w = B.mix(spl, w, srgb((150, 148, 140)))
    endm = 0.0
    if ends is not None:
        x0, x1 = ends
        d_end = B.minf(B.sub(x, x0), B.sub(x1, x))
        endm = B.mr(B.add(d_end, B.mul(B.sub(B.noise(P, 6.0, 4), 0.5), 0.12)), 0.35, 0.02)
        w = B.mix(B.mul(endm, wet_ends), w, B.hsv(w, 0.5, 1.1, 0.55))
    # mud: bottom + ends
    mudt = B.hsv(B.tri('brown_mud_rocks_01', 'diffuse', P, 0.7), 0.5, 0.8, 0.6)
    mz = B.mr(B.add(z, B.mul(B.sub(B.noise(P, 9.0, 5, 0.7), 0.5), 0.05)), mud_h, mud_h * 0.2)
    mm_ = B.mul(B.maxf(mz, B.mul(endm, 0.6)), mud)
    mm_ = B.mul(mm_, B.mr(B.noise(P, 5.0, 4), 0.3, 0.6))
    w = B.mix(mm_, w, mudt)
    rough = B.mixf(gm, rough_base, 0.9)
    rough = B.mixf(mm_, rough, 0.95)
    hgt = B.tri(texname, 'displacement', P, tile, axis=axis, noncolor=True)
    nrm = B.bevel(0.002, 6)
    nrm = B.bump(B.sep(hgt)[0], 1.0, 0.0025, nrm)
    nrm = B.bump(mm_, 0.5, 0.0015, nrm)
    return B.finish(w, rough, 0.0, nrm)


def M_plastic(name, color, *, rough=0.45, dirt=0.5, scuff=0.5, fade=0.3, mud=0.3, mud_h=0.12, seed=0,
              bands=None, band_color=(222, 224, 220), black=None, black_color=(22, 22, 22)):
    """Coloured plastic (cone, delineator) with optional retroreflective bands (z ranges) and a black z band."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed * 1.3, seed * 2.1, 0))
    x, y, z = B.sep(B.P)
    c = srgb(color)
    v1 = B.noise(P, 4.0, 3)
    col = B.mix(B.mr(v1, 0.3, 0.7), B.hsv(c, 0.5, 0.95, 0.9), B.hsv(c, 0.5, 1.05, 1.06))
    nzo = B.sep(B.objN())[2]
    up = B.mr(nzo, 0.0, 0.9)
    col = B.mix(B.mul(B.add(B.mul(up, 0.6), 0.4), fade), col, B.hsv(col, 0.5, 0.6, 1.2))
    bm = 0.0
    if black:
        bm = B.mul(B.mr(z, black[0] - 0.002, black[0] + 0.002), B.mr(z, black[1] + 0.002, black[1] - 0.002))
        col = B.mix(bm, col, srgb(black_color))
    band = 0.0
    if bands:
        for (za, zb) in bands:
            bb = B.mul(B.mr(z, za - 0.0015, za + 0.0015), B.mr(z, zb + 0.0015, zb - 0.0015))
            band = B.maxf(band, bb) if not isinstance(band, float) else bb
        # sheeting: faint microprism cells, scuffed/torn to show plastic
        cells = B.vor(P, 700.0, 'F1', 1.0, 'Distance')
        sheet = B.mix(B.mr(cells, 0.0, 0.5), srgb(band_color), B.hsv(srgb(band_color), 0.5, 1.0, 0.85))
        tear = B.mul(B.mr(B.noise(P, 30.0, 6, 0.7), 0.66, 0.7), band)
        band_eff = B.mul(band, B.inv(tear))
        col = B.mix(band_eff, col, sheet)
    # scuffs (light scratches), grime in cavities, road spray from below
    scr = B.vor(B.mapping(P, rot=(0, 0, 25), scale=(40, 40, 6)), 1.0, 'DISTANCE_TO_EDGE')
    scr = B.mul(B.mr(scr, 0.03, 0.0), B.mul(B.mr(B.noise(P, 5.0, 2), 0.45, 0.6), scuff))
    col = B.mix(B.mul(scr, 0.6), col, B.hsv(col, 0.5, 0.5, 1.3))
    cav = B.cavity(0.03)
    col = B.mix(B.mul(cav, dirt), col, srgb((50, 45, 38)))
    spk = B.mr(B.noise(P, 70.0, 3, 0.6), 0.5, 0.75)
    low = B.mr(z, mud_h * 3, 0.0)
    gm = B.mul(B.mul(B.add(low, 0.12), spk), dirt)
    col = B.mix(gm, col, srgb((95, 85, 70)))
    mudt = B.hsv(B.tri('brown_mud_rocks_01', 'diffuse', P, 0.6), 0.5, 0.8, 0.6)
    mz = B.mul(B.mr(B.add(z, B.mul(B.sub(B.noise(P, 10.0, 5, 0.7), 0.5), 0.06)), mud_h, mud_h * 0.3), mud)
    col = B.mix(mz, col, mudt)
    r = B.mixf(band if not isinstance(band, float) else 0.0, rough, 0.32)
    if black:
        r = B.mixf(bm, r, 0.5)
    r = B.mixf(B.maxf(gm, B.mul(cav, dirt)), r, 0.85)
    r = B.mixf(mz, r, 0.95)
    r = B.mixf(scr, r, 0.7)
    nrm = B.bevel(0.0015, 6)
    nrm = B.bump(B.noise(P, 180.0, 2), 0.05, 0.0003, nrm)
    nrm = B.bump(B.mul(scr, -1.0), 0.3, 0.0002, nrm)
    if bands:
        nrm = B.bump(B.mul(cells, band), 0.12, 0.0002, nrm)
    nrm = B.bump(mz, 0.4, 0.001, nrm)
    return B.finish(col, r, 0.0, nrm)


def M_rubber(name, *, color=(28, 28, 27), dust=0.5, rough=0.82, seed=0, grain=1.0):
    B = NB(name)
    P = B.mapping(B.P, loc=(seed, seed * 2, 0))
    c = srgb(color)
    g = B.noise(P, 300.0, 2)
    col = B.mix(B.mr(g, 0.3, 0.7), B.hsv(c, 0.5, 1, 0.8), B.hsv(c, 0.5, 1, 1.25))
    nzo = B.sep(B.objN())[2]
    up = B.mr(nzo, 0.2, 0.9)
    dm = B.mul(B.mul(up, dust), B.mr(B.noise(P, 8.0, 4), 0.3, 0.7))
    cav = B.cavity(0.02)
    col = B.mix(B.maxf(dm, B.mul(cav, dust * 0.6)), col, srgb((105, 96, 82)))
    r = B.mixf(dm, rough, 0.95)
    nrm = B.bump(g, 0.25 * grain, 0.0005)
    nrm = B.bevel(0.0015, 6, normal=None) if False else nrm
    return B.finish(col, r, 0.0, nrm)


def M_concrete_paint(name, *, bands_x=0.5, x0=-1.0, seed=0):
    """Precast concrete barrier with worn red/white paint bands along x."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed * 3.3, seed * 1.1, 0))
    x, y, z = B.sep(B.P)
    con = B.tri('concrete_wall_006', 'diffuse', P, 1.6)
    con = B.hsv(con, 0.5, 0.7, 1.05)
    con = B.mix(B.mr(B.noise(P, 1.0, 3), 0.3, 0.7), B.hsv(con, 0.5, 1, 0.85), B.hsv(con, 0.5, 1, 1.1))
    # paint bands
    u = B.math('DIVIDE', B.sub(x, x0), bands_x)
    band = B.math('MODULO', B.math('FLOOR', u), 2.0)
    red = srgb((128, 31, 27))
    white = srgb((168, 163, 150))
    pc = B.mix(band, red, white)
    pv = B.noise(P, 3.0, 4)
    pc = B.mix(B.mr(pv, 0.3, 0.7), B.hsv(pc, 0.5, 0.92, 0.9), B.hsv(pc, 0.5, 1.02, 1.04))
    nzo = B.sep(B.objN())[2]
    up = B.mr(nzo, 0.3, 0.95)
    pc = B.mix(B.mul(up, 0.45), pc, B.hsv(pc, 0.5, 0.6, 1.08))
    # paint wear: edges, top, lower toe (scraped by tyres/ploughs), random flaking
    e = B.edge(0.012, lo=0.05, hi=0.3, samples=8)
    fl = B.noise(P, 7.0, 7, 0.68)
    wear = B.maxf(B.mul(e, B.mr(fl, 0.3, 0.6, 0.5, 1.6)), B.mr(fl, 0.61, 0.66))
    toe = B.mr(z, 0.16, 0.05)
    wear = B.maxf(wear, B.mul(toe, B.mr(B.noise(P, 5.0, 5), 0.35, 0.55)))
    wear = B.mr(wear, 0.5, 0.56)
    col = B.mix(wear, pc, con)
    # tyre rub marks on the lower slope
    rub = B.mul(B.mr(B.noise(B.mapping(P, scale=(3, 30, 30)), 1.0, 3), 0.55, 0.7), B.mul(B.mr(z, 0.1, 0.2), B.mr(z, 0.4, 0.28)))
    rub = B.mul(rub, B.mr(B.noise(P, 0.8, 2), 0.45, 0.6))
    col = B.mix(B.mul(rub, 0.8), col, srgb((30, 29, 28)))
    # rain streaks from the top edge, grime in pores and cavities, mud splash at the foot
    st = B.noise(B.mapping(P, scale=(26, 26, 1.2)), 1.0, 4, 0.6)
    stm = B.mul(B.mr(st, 0.5, 0.7), B.mr(z, 0.2, 0.75))
    col = B.mix(B.mul(stm, 0.62), col, B.hsv(col, 0.5, 0.7, 0.55))
    grm = B.mul(B.mr(B.noise(P, 2.2, 6, 0.68), 0.4, 0.75), B.add(0.35, B.mul(B.mr(z, 0.6, 0.0), 0.65)))
    col = B.mix(B.mul(grm, 0.72), col, srgb((92, 84, 70)))
    cav = B.cavity(0.04)
    col = B.mix(B.mul(cav, 0.7), col, srgb((45, 42, 37)))
    mudt = B.hsv(B.tri('brown_mud_rocks_01', 'diffuse', P, 0.8), 0.5, 0.85, 0.55)
    mz = B.mr(B.add(z, B.mul(B.sub(B.noise(P, 8.0, 5, 0.7), 0.5), 0.12)), 0.2, 0.03)
    col = B.mix(mz, col, mudt)
    moss = B.mul(B.mr(B.noise(P, 3.0, 5), 0.6, 0.7), B.mr(z, 0.12, 0.0))
    col = B.mix(B.mul(moss, 0.6), col, srgb((58, 66, 30)))
    rough = B.mixf(wear, 0.62, 0.9)
    rough = B.mixf(mz, rough, 0.95)
    rough = B.mixf(rub, rough, 0.75)
    hc = B.tri('concrete_wall_006', 'displacement', P, 1.6, noncolor=True)
    nrm = B.bevel(0.006, 8)
    nrm = B.bump(B.sep(hc)[0], 0.8, 0.004, nrm)
    nrm = B.bump(B.mul(B.inv(wear), 1.0), 0.5, 0.0004, nrm)
    nrm = B.bump(mz, 0.6, 0.002, nrm)
    return B.finish(col, rough, 0.0, nrm)


def M_forged(name, *, seed=0):
    """Hatchet head: black forge scale, hammer peen, rust spots, bright honed bevel (thin region near the edge)."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed, 0, 0))
    x, y, z = B.sep(B.P)
    scale_c = srgb((42, 42, 44))
    peen = B.vor(P, 90.0, 'F1', 1.0, 'Distance')
    col = B.mix(B.mr(B.noise(P, 30.0, 4), 0.3, 0.7), B.hsv(scale_c, 0.5, 1, 0.7), B.hsv(scale_c, 0.5, 1, 1.4))
    rust_t = B.tri('rust_coarse_01', 'diffuse', P, 0.25)
    rm = B.mr(B.noise(P, 22.0, 7, 0.7), 0.6, 0.7)
    col = B.mix(rm, col, B.hsv(rust_t, 0.5, 1.05, 0.9))
    e = B.edge(0.0015, samples=8)
    col = B.mix(B.mul(e, 0.6), col, srgb((120, 118, 114)))
    thin = B.mr(B.absf(x), 0.0022, 0.0016)
    hone = B.mul(thin, B.mr(y, -0.085, -0.095))
    grind = B.noise(B.mapping(P, scale=(900, 20, 900)), 1.0, 2)
    steel = B.mix(B.mr(grind, 0.3, 0.7), srgb((150, 150, 152)), srgb((205, 205, 206)))
    col = B.mix(hone, col, steel)
    r = B.mixf(rm, 0.55, 0.85)
    r = B.mixf(e, r, 0.35)
    r = B.mixf(hone, r, 0.18)
    metal = B.mixf(rm, 0.75, 0.1)
    metal = B.mixf(hone, metal, 1.0)
    nrm = B.bevel(0.0008, 6)
    nrm = B.bump(B.mul(peen, B.inv(hone)), 0.35, 0.0006, nrm)
    nrm = B.bump(B.mul(rm, B.noise(P, 120.0, 4)), 0.4, 0.0003, nrm)
    return B.finish(col, r, metal, nrm)


def M_ash(name, *, seed=0, grip=(-0.08, 0.06)):
    """Worn ash handle: light grain, lacquer worn at the grip, hand grime."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed, 0, 0))
    x, y, z = B.sep(B.P)
    w = B.tri('ash_veneer', 'diffuse', P, 0.45, axis='z')
    w = B.hsv(w, 0.49, 1.45, 0.6)
    w = B.mix(B.mr(B.noise(P, 6.0, 3), 0.3, 0.7), B.hsv(w, 0.5, 1, 0.8), B.hsv(w, 0.5, 1, 1.08))
    # long fibre streaks along the handle (z) + darker weathered grain lines
    fib = B.noise(B.mapping(P, scale=(160, 160, 3.0)), 1.0, 5, 0.65)
    w = B.mix(B.mul(B.mr(fib, 0.45, 0.72), 0.55), w, B.hsv(w, 0.5, 1.15, 0.55))
    fib2 = B.noise(B.mapping(P, scale=(420, 420, 8.0)), 1.0, 3, 0.6)
    w = B.mix(B.mul(B.mr(fib2, 0.55, 0.75), 0.35), w, B.hsv(w, 0.5, 0.8, 1.2))
    # oily, grey-brown grime: blotchy everywhere, heavy near the head (top of the handle)
    blot = B.mr(B.noise(P, 9.0, 6, 0.7), 0.45, 0.75)
    headg = B.mr(z, 0.18, 0.30)
    w = B.mix(B.mul(B.maxf(B.mul(blot, 0.45), headg), 0.7), w, srgb((58, 46, 34)))
    gm = B.mul(B.mr(z, grip[0] - 0.03, grip[0] + 0.02), B.mr(z, grip[1] + 0.04, grip[1]))
    gn = B.mr(B.noise(P, 20.0, 4), 0.3, 0.7)
    w = B.mix(B.mul(gm, B.add(0.45, B.mul(gn, 0.35))), w, B.hsv(w, 0.5, 0.9, 0.45))
    e = B.edge(0.003, samples=6)
    w = B.mix(B.mul(e, 0.25), w, B.hsv(w, 0.5, 0.7, 1.25))
    scr = B.vor(B.mapping(P, scale=(60, 60, 5)), 1.0, 'DISTANCE_TO_EDGE')
    scr = B.mul(B.mr(scr, 0.02, 0.0), B.mr(B.noise(P, 8.0, 2), 0.45, 0.6))
    w = B.mix(B.mul(scr, 0.5), w, B.hsv(w, 0.5, 0.8, 0.6))
    # growth-ring grain: wavy bands across x/y stretched hugely along z, so the lines run down the handle
    gP = B.mapping(P, rot=(0, 0, 8), scale=(1.0, 1.0, 0.06))
    ring = B.wave(gP, 55.0, 9.0, 4.0, 1.2, axis='X', profile='SAW')
    ring = B.mr(ring, 0.6, 0.97)
    ring2 = B.wave(B.mapping(P, rot=(0, 0, -5), scale=(1.0, 1.0, 0.04)), 23.0, 6.0, 3.0, 0.8, axis='Y', profile='SAW')
    ring2 = B.mr(ring2, 0.8, 0.99)
    grain = B.maxf(B.mul(ring, 0.8), B.mul(ring2, 0.6))
    w = B.mix(B.mul(grain, 0.85), w, B.hsv(w, 0.5, 1.25, 0.42))
    # weathered: grey cast away from the grip, darker oily hand zone, knocks near the head
    w = B.hsv(w, 0.5, 0.82, 0.88)
    w = B.mix(B.mul(B.mr(B.noise(P, 3.0, 4), 0.4, 0.7), 0.35), w, B.hsv(w, 0.5, 0.35, 1.05))
    knock = B.mul(B.mr(B.noise(P, 45.0, 6, 0.7), 0.68, 0.74), B.mr(z, 0.12, 0.26))
    w = B.mix(B.mul(knock, 0.8), w, srgb((44, 34, 26)))
    cav = B.cavity(0.012)
    w = B.mix(B.mul(cav, 0.8), w, srgb((42, 32, 22)))
    r = B.mixf(gm, 0.62, 0.42)
    r = B.mixf(grain, r, 0.78)
    r = B.mixf(scr, r, 0.7)
    hgt = B.tri('ash_veneer', 'displacement', P, 0.45, axis='z', noncolor=True)
    nrm = B.bump(B.sep(hgt)[0], 0.4, 0.0008)
    nrm = B.bump(B.mul(grain, -1.0), 0.35, 0.0003, nrm)
    nrm = B.bump(B.mul(knock, -1.0), 0.6, 0.0006, nrm)
    nrm = B.bump(B.mul(scr, -1.0), 0.3, 0.0003, nrm)
    return B.finish(w, r, 0.0, nrm)


def M_fabric(name, *, texname='hessian_230', tile=0.35, tint=(1, 1, 1), val=1.0, sat=1.0, mud=0.5, mud_h=0.05,
             dirt=0.6, seed=0, rough=0.92, sand=0.3):
    B = NB(name)
    P = B.mapping(B.P, loc=(seed * 1.7, seed * 0.3, 0))
    x, y, z = B.sep(B.P)
    f = B.tri(texname, 'diffuse', P, tile)
    f = B.hsv(f, 0.5, sat, val)
    f = B.mix(1.0, f, tint, 'MULTIPLY')
    f = B.mix(B.mr(B.noise(P, 3.0, 3), 0.3, 0.7), B.hsv(f, 0.5, 1, 0.8), B.hsv(f, 0.5, 1, 1.1))
    cav = B.cavity(0.03)
    f = B.mix(B.mul(cav, dirt), f, srgb((42, 36, 28)))
    sd = B.mul(B.mr(B.noise(P, 12.0, 5, 0.7), 0.55, 0.7), sand)
    f = B.mix(sd, f, srgb((160, 145, 118)))
    mudt = B.hsv(B.tri('brown_mud_rocks_01', 'diffuse', P, 0.6), 0.5, 0.8, 0.55)
    mz = B.mul(B.mr(B.add(z, B.mul(B.sub(B.noise(P, 9.0, 5, 0.7), 0.5), 0.06)), mud_h, mud_h * 0.2), mud)
    f = B.mix(mz, f, mudt)
    hgt = B.tri(texname, 'displacement', P, tile, noncolor=True)
    nrm = B.bump(B.sep(hgt)[0], 0.7, 0.0015)
    nrm = B.bump(B.noise(P, 25.0, 3), 0.2, 0.002, nrm)
    return B.finish(f, B.mixf(mz, rough, 0.96), 0.0, nrm)


def M_tarp(name, color=(38, 88, 150), *, seed=0):
    """Woven PE tarp. Uses the mesh attributes 'rest' (flat-sheet coordinates, m) and 'pud' (0..1 puddle mask from
    the priority-flood in _tarp_puddles): weave, fold-memory creases (it was folded in quarters x thirds), a 3 cm
    rope hem with aluminium grommets every 0.5 m, UV fading on top, silt rings where puddles stand/dried, mud at
    the ground edge, needles and leaves."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed, 0, 0))
    x, y, z = B.sep(B.P)
    rv = B.new('ShaderNodeAttribute', attribute_name='rest').outputs['Vector']
    pud = B.new('ShaderNodeAttribute', attribute_name='pud').outputs['Fac']
    ru, rw, _ = B.sep(rv)
    c = srgb(color)
    nzo = B.sep(B.objN())[2]
    up = B.mr(nzo, 0.3, 1.0)
    col = B.mix(B.mr(B.noise(P, 1.5, 3), 0.3, 0.7), B.hsv(c, 0.5, 0.95, 0.85), B.hsv(c, 0.5, 1.0, 1.08))
    # UV-faded top (paler, greyer: PE loses colour in a season on site)
    col = B.mix(B.mul(up, 0.55), col, B.hsv(col, 0.52, 0.5, 1.28))
    # woven tapes (~2.5 mm), in the sheet frame
    wx = B.math('SINE', B.mul(ru, 2 * math.pi / 0.005))
    wy = B.math('SINE', B.mul(rw, 2 * math.pi / 0.005))
    weave = B.mul(B.add(wx, 1.0), B.add(wy, 1.0))
    # fold-memory creases: quarters across the width, thirds across the depth (whitened, stressed film)
    def crease(v, period, off):
        f = B.math('FRACT', B.add(B.math('DIVIDE', v, period), off))
        d = B.mul(B.absf(B.sub(f, 0.5)), period)
        return B.mr(d, 0.004, 0.0)
    fold = B.maxf(crease(ru, TARP_W / 4, 0.5), crease(rw, TARP_D / 3, 0.5))
    fold = B.mul(fold, B.mr(B.noise(P, 8.0, 3), 0.38, 0.62))
    col = B.mix(B.mul(fold, 0.25), col, B.hsv(col, 0.5, 0.6, 1.25))
    # hem (double layer + rope) and grommets
    ed = B.minf(B.sub(TARP_W / 2, B.absf(ru)), B.sub(TARP_D / 2, B.absf(rw)))
    hem = B.mr(ed, 0.036, 0.03)
    along = B.maxf(B.mul(ru, B.mr(B.absf(rw), TARP_D / 2 - 0.05, TARP_D / 2 - 0.04)), B.mul(rw, B.mr(B.absf(ru), TARP_W / 2 - 0.05, TARP_W / 2 - 0.04)))
    gpos = B.mul(B.absf(B.sub(B.math('FRACT', B.add(B.math('DIVIDE', B.add(ru, rw), 0.5), 0.5)), 0.5)), 0.5)
    gr = B.math('SQRT', B.add(B.mul(gpos, gpos), B.mul(B.sub(ed, 0.018), B.sub(ed, 0.018))))
    grom = B.mul(B.mr(gr, 0.0125, 0.011), B.mr(gr, 0.005, 0.0065))
    hole = B.mr(gr, 0.0065, 0.0055)
    col = B.mix(B.mul(hem, 0.35), col, B.hsv(col, 0.5, 1.1, 0.8))
    col = B.mix(grom, col, srgb((150, 150, 146)))
    col = B.mix(hole, col, srgb((20, 20, 20)))
    # grime in folds, silt rings where puddles stand and dried, mud at the ground edge
    cav = B.cavity(0.06, 0.97, 0.55)
    dm = B.mul(cav, B.mr(B.noise(P, 3.0, 4), 0.3, 0.7))
    col = B.mix(B.mul(dm, 0.85), col, srgb((70, 62, 50)))
    ring = B.mul(B.mr(pud, 0.05, 0.35), B.mr(pud, 0.95, 0.6))
    silt = B.maxf(B.mul(ring, 0.9), B.mul(B.mr(pud, 0.3, 0.8), 0.55))
    silt = B.mul(silt, B.mr(B.noise(P, 12.0, 4), 0.25, 0.6))
    col = B.mix(silt, col, srgb((96, 82, 62)))
    mudt = B.hsv(B.tri('brown_mud_rocks_01', 'diffuse', P, 0.7), 0.5, 0.75, 0.6)
    mz = B.mr(B.add(z, B.mul(B.sub(B.noise(P, 6.0, 5, 0.7), 0.5), 0.06)), 0.07, 0.015)
    col = B.mix(mz, col, mudt)
    # spruce needles and a few leaves blown onto it
    lv = B.mr(B.noise(P, 40.0, 3), 0.72, 0.75)
    ndl = B.mul(B.mr(B.vor(B.mapping(P, rot=(0, 0, 35), scale=(1, 12, 1)), 60.0, 'F1'), 0.08, 0.0), B.mr(B.noise(P, 3.0, 2), 0.55, 0.7))
    col = B.mix(B.mul(B.maxf(lv, ndl), 0.8), col, srgb((72, 52, 30)))
    r = B.mixf(up, 0.42, 0.55)
    r = B.mixf(B.maxf(dm, mz), r, 0.9)
    r = B.mixf(silt, r, 0.8)
    r = B.mixf(grom, r, 0.35)
    metal = B.mul(grom, 0.9)
    nrm = B.bump(weave, 0.08, 0.0002)
    nrm = B.bump(fold, 0.5, 0.0012, nrm)
    nrm = B.bump(hem, 0.6, 0.002, nrm)
    nrm = B.bump(B.sub(grom, hole), 0.8, 0.0015, nrm)
    nrm = B.bump(B.noise(P, 14.0, 4), 0.15, 0.002, nrm)
    return B.finish(col, r, metal, nrm)


def M_sign_face(name, image, extent, *, seed=0):
    """Retroreflective sheeting on aluminium: image mapped planar on object XZ (extent x0,x1,z0,z1)."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed, 0, 0))
    x, y, z = B.sep(B.P)
    x0, x1, z0, z1 = extent
    uv = B.comb(B.mr(x, x0, x1, 0, 1, clamp=False), B.mr(z, z0, z1, 0, 1, clamp=False), 0)
    face = B.img(None, uv, proj='FLAT', ext='EXTEND', image=image)
    # fading, grime running down from the top edge, spray from below
    col = B.hsv(face, 0.5, 0.82, 0.92)
    st = B.noise(B.mapping(P, scale=(30, 30, 1.5)), 1.0, 4, 0.6)
    stm = B.mul(B.mr(st, 0.5, 0.72), 0.55)
    col = B.mix(stm, col, B.hsv(col, 0.5, 0.8, 0.62))
    spk = B.mul(B.mr(B.noise(P, 60.0, 3), 0.55, 0.75), B.mr(z, z0 + 0.25, z0))
    col = B.mix(B.mul(spk, 0.8), col, srgb((95, 88, 74)))
    cells = B.vor(P, 500.0, 'F1', 1.0, 'Distance')
    dnt = B.noise(P, 3.0, 2)
    r = B.mixf(stm, 0.3, 0.55)
    nrm = B.bump(cells, 0.06, 0.0002)
    nrm = B.bump(dnt, 0.25, 0.002, nrm)
    return B.finish(col, r, 0.0, nrm)


def M_alu(name, *, seed=0, dirt=0.6):
    B = NB(name)
    P = B.mapping(B.P, loc=(seed, 0, 0))
    x, y, z = B.sep(B.P)
    c = B.mix(B.mr(B.noise(P, 5.0, 4), 0.3, 0.7), srgb((140, 142, 142)), srgb((175, 176, 175)))
    brushed = B.noise(B.mapping(P, scale=(300, 300, 3)), 1.0, 2)
    c = B.mix(B.mul(B.mr(brushed, 0.4, 0.6), 0.3), c, B.hsv(c, 0.5, 1, 0.85))
    st = B.noise(B.mapping(P, scale=(30, 30, 1.5)), 1.0, 4, 0.6)
    stm = B.mul(B.mr(st, 0.45, 0.72), dirt)
    c = B.mix(stm, c, srgb((80, 76, 66)))
    cav = B.cavity(0.03)
    c = B.mix(B.mul(cav, dirt), c, srgb((50, 46, 40)))
    r = B.mixf(stm, 0.45, 0.8)
    m = B.mixf(stm, 0.85, 0.2)
    nrm = B.bump(brushed, 0.05, 0.0002)
    return B.finish(c, r, m, nrm)


def M_simple(name, color, rough=0.5, metal=0.0, noise=0.1, seed=0):
    B = NB(name)
    P = B.mapping(B.P, loc=(seed, 0, 0))
    c = srgb(color)
    col = B.mix(B.mr(B.noise(P, 20.0, 3), 0.3, 0.7), B.hsv(c, 0.5, 1, 1 - noise), B.hsv(c, 0.5, 1, 1 + noise))
    cav = B.cavity(0.02)
    col = B.mix(B.mul(cav, 0.5), col, srgb((30, 28, 25)))
    return B.finish(col, rough, metal, B.bevel(0.001, 4))


def M_reflector(name, color=(235, 228, 205)):
    B = NB(name)
    P = B.P
    c = srgb(color)
    cells = B.vor(B.mapping(P, scale=(1, 1, 1)), 350.0, 'F1', 0.3, 'Distance')
    col = B.mix(B.mr(cells, 0.0, 0.6), c, B.hsv(c, 0.5, 1, 0.7))
    cav = B.cavity(0.01)
    col = B.mix(B.mul(cav, 0.5), col, srgb((60, 55, 45)))
    nrm = B.bump(cells, 0.5, 0.0006)
    return B.finish(col, 0.18, 0.0, nrm)


# ================================================================================================
# prop builders.  Each returns a list of objects; obj['atlas'] + obj['uvw'] (per material slot weights)
# Blender coords: Z up; three.js = (x, z, -y)  =>  three +Z (sign faces, lamp direction) is Blender -Y.
# ================================================================================================
def finish_obj(obj, uvw=None):
    obj['atlas'] = ATLAS_OF[obj.name]
    obj['uvw'] = json.dumps(uvw or {})
    return obj


# ---------------------------------------------------------------- jerrycan
def emboss_jerrycan(W, H0, H1):
    """Height map for the pressed NATO-can side: raised X ribs + perimeter bead, slightly domed panels."""
    res_x, res_z = 520, 700
    xs = np.linspace(-W / 2, W / 2, res_x)
    zs = np.linspace(H1, H0, res_z)                   # row 0 = top
    X, Z = np.meshgrid(xs, zs)
    zc = (H0 + H1) / 2
    a, b = W / 2 - 0.030, (H1 - H0) / 2 - 0.032
    u, v = X, Z - zc
    # X ribs from the panel corners
    L = math.hypot(a, b)
    d1 = np.abs(u * b - v * a) / L
    d2 = np.abs(u * b + v * a) / L
    inside = np.clip((a + 0.004 - np.abs(u)) / 0.01, 0, 1) * np.clip((b + 0.004 - np.abs(v)) / 0.01, 0, 1)
    rib = np.exp(-(np.minimum(d1, d2) / 0.0105) ** 2) * inside
    # perimeter bead (rounded-rect ring)
    qx = np.maximum(np.abs(u) - (a - 0.012), 0)
    qz = np.maximum(np.abs(v) - (b - 0.012), 0)
    rr = np.hypot(qx, qz) + np.minimum(np.maximum(np.abs(u) - (a - 0.012), np.abs(v) - (b - 0.012)), 0)
    ring = np.exp(-((rr - 0.012) / 0.0085) ** 2)
    dome = np.clip(1 - (u / a) ** 2, 0, 1) * np.clip(1 - (v / b) ** 2, 0, 1)
    h = 0.35 + 0.55 * np.maximum(rib, ring) + 0.12 * dome * (1 - np.maximum(rib, ring))
    h -= 0.08 * np.exp(-((rr - 0.028) / 0.01) ** 2) * (1 - rib)
    return np.clip(h, 0, 1)


def build_jerrycan():
    W, D, H = 0.345, 0.165, 0.448
    z0 = 0.004
    zc = (z0 + H) / 2
    rc = 0.02
    em_img = np_image('jc_emboss', emboss_jerrycan(W, z0, H), os.path.join(BAKE, '_jc_emboss.png'))
    paint = M_paint('m_jerrycan', (84, 90, 56), chip=0.9, rust=0.4, dirt=0.75, mud=0.6, mud_h=0.085, gloss=0.5, scratches=0.9,
                    fade=0.35, emboss=dict(image=em_img, x0=-W / 2, x1=W / 2, z0=z0, z1=H, depth=0.0045),
                    edge_r=0.008, stain=(0.13, 0.43, 0.05), seed=3, primer=(110, 72, 58),
                    edgebox=(W / 2, D / 2, z0, H, 0.015))
    mb = MB()
    # body: rounded box
    bm = rbox(W, D, H - z0, r=rc, segs=4, center=(0, 0, zc))
    mb.add_bm(bm, 0)
    # welded seam flange around the perimeter (plane y=0)
    hw, hh = W / 2 + 0.0005, (H - z0) / 2 + 0.0005
    path = []
    for cx, cz, a0 in ((hw - rc, hh - rc, 0), (-hw + rc, hh - rc, 90), (-hw + rc, -hh + rc, 180), (hw - rc, -hh + rc, 270)):
        for k in range(7):
            a = math.radians(a0 + 90 * k / 6)
            path.append((cx + rc * math.cos(a), 0.0, zc + cz + rc * math.sin(a)))
    path = np.array(path)
    n = len(path)
    Tn = np.roll(path, -1, 0) - np.roll(path, 1, 0)
    Tn /= np.linalg.norm(Tn, axis=1, keepdims=True)
    Bv = np.tile([0, 1.0, 0], (n, 1))
    Nv = np.cross(Tn, Bv)
    cen = np.array([0, 0, zc])
    if np.dot(Nv[0], path[0] - cen) < 0:
        Nv = -Nv
    prof = rrect(0.0075, 0.0034, 0.0012, 2) + np.array([0.0012, 0])
    V, F = sweep(path, prof, closed_path=True, N=Nv, B=Bv)
    mb.add(V, F, 0)
    # three carrying handles: tubular bars along x (so the can hangs flat against the leg), spaced across the depth,
    # legs welded onto small pads on the top
    for yh in (-0.052, 0.0, 0.052):
        pts = [(-0.066, yh, H - 0.004), (-0.061, yh, H + 0.017), (-0.048, yh, H + 0.029), (0.0, yh, H + 0.032),
               (0.048, yh, H + 0.029), (0.061, yh, H + 0.017), (0.066, yh, H - 0.004)]
        V, F = tube(pts, 0.0085, 10, 22)
        mb.add(V, F, 0)
    for xp in (-0.066, 0.066):
        mb.add_bm(rbox(0.03, 0.15, 0.005, r=0.002, segs=1, center=(xp, 0, H + 0.0005)), 0)
    # spout: tilted neck on the +X end of the top, bayonet cap with cam lever and locking pin
    Ms = mm(T(0.118, 0, H - 0.006), R('Y', 32))
    V, F = lathe([(0.024, -0.012), (0.024, 0.004), (0.021, 0.006), (0.021, 0.030), (0.024, 0.031)], 20)
    mb.add(V, F, 0, M=Ms)
    V, F = lathe([(0.0285, 0.028), (0.0295, 0.032), (0.0295, 0.046), (0.0275, 0.050), (0.012, 0.0515)], 24)
    mb.add(V, F, 0, M=Ms)
    for k in range(12):   # grip ribs on the cap
        a = 360.0 * k / 12
        rib = rbox(0.004, 0.003, 0.014, center=(0.0297, 0, 0.039))
        mb.add_bm(rib, 0, M=mm(Ms, R('Z', a)))
    # hinge bracket on the handle side of the neck + cam lever over the cap
    br = rbox(0.012, 0.022, 0.016, r=0.002, segs=1, center=(-0.028, 0, 0.022))
    mb.add_bm(br, 0, M=Ms)
    lever = [(-0.030, 0, 0.030), (-0.022, 0, 0.056), (0.0, 0, 0.060), (0.030, 0, 0.058), (0.052, 0, 0.050)]
    V, F = sweep(spline(lever, 14), rrect(0.0045, 0.018, 0.0015, 2), caps=True, up=(0, 1, 0))
    mb.add(V, F, 0, M=Ms)
    V, F = cyl(0.004, 0.03, 10, z0=-0.015)
    mb.add(V, F, 0, M=mm(Ms, T(-0.028, 0, 0.024), R('X', 90)))
    # locking pin on a short chain-stub toward the handles
    V, F = tube([(0.05, 0.0, 0.05), (0.06, 0.0, 0.04), (0.058, 0.0, 0.02)], 0.0025, 6, 10)
    mb.add(V, F, 0, M=Ms)
    obj = mb.build('jerrycan', [paint], sharp=50)
    return [finish_obj(obj)]


# ---------------------------------------------------------------- hatchet
def build_hatchet():
    """Handle along +Z (three +Y), head at the top, blade edge toward -Y (three +Z). Origin at the grip."""
    mb = MB()
    ash = M_ash('m_hatchet_handle', seed=5, grip=(-0.04, 0.07))
    steel = M_forged('m_hatchet_head', seed=2)
    wedge = M_paint('m_hatchet_wedge', (70, 70, 72), chip=0.2, rust=0.7, dirt=0.3, mud=0.0, gloss=0.4, fade=0.0, seed=9)
    # --- handle: oval sections along z with swell at the end and a gentle S-curve (bottom sweeps to +y)
    zs = np.concatenate([np.linspace(-0.098, -0.085, 3), np.linspace(-0.075, 0.345, 18)])
    secs = []
    for z in zs:
        t = (z + 0.098) / 0.443
        wy = 0.030 + 0.012 * np.exp(-((z + 0.085) / 0.018) ** 2) - 0.003 * t      # blade-direction width
        tx = 0.022 + 0.006 * np.exp(-((z + 0.085) / 0.02) ** 2) - 0.002 * t
        oy = 0.012 * (1 - t) ** 2 - 0.004 * math.sin(t * math.pi)
        if z < -0.088:
            k = (z + 0.098) / 0.010
            wy *= 0.55 + 0.45 * math.sqrt(max(k, 0))
            tx *= 0.55 + 0.45 * math.sqrt(max(k, 0))
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ca, sa = np.cos(a), np.sin(a)
        ex = 2.5
        px = tx / 2 * np.sign(ca) * np.abs(ca) ** (2 / ex)
        py = wy / 2 * np.sign(sa) * np.abs(sa) ** (2 / ex)
        secs.append(np.stack([px, py + oy, np.full(16, z)], -1))
    secs = np.array(secs)
    nS, m = secs.shape[:2]
    V = secs.reshape(-1, 3)
    F = [[i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j] for i in range(nS - 1) for j in range(m)]
    F.append(list(range(m))[::-1])
    F.append(list(range((nS - 1) * m, nS * m)))
    mb.add(V, F, 0)
    # --- head: parametric forged wedge. u: poll (y=+0.034) -> edge (y=-0.122); v: bottom -> top
    Nu, Nv = 30, 22
    us = np.linspace(0, 1, Nu)
    vs = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, Nv))
    zt = 0.338
    def head_shape(u):
        y = 0.034 - 0.156 * u
        # top line: flat, slight rise toward the toe; bottom line: flat at the eye, sweeping into the beard
        ztop = zt + 0.004 * max(0.0, (u - 0.6) / 0.4) ** 2
        eye = math.exp(-((u - 0.22) / 0.16) ** 2)
        zbot = zt - 0.048 - 0.036 * max(0.0, (u - 0.45) / 0.55) ** 1.6 + 0.004 * eye
        # thickness: thick poll/eye, thin cheeks, honed bevel
        if u < 0.42:
            t = 0.030 + 0.008 * eye
        else:
            k = (u - 0.42) / 0.5
            t = 0.030 * (1 - k) ** 1.35 + 0.0038 * k
        if u > 0.92:
            k = (u - 0.92) / 0.08
            t = 0.0038 * (1 - k) + 0.0006 * k
        return y, ztop, zbot, t
    verts = []
    for side in (1, -1):
        for u in us:
            y, ztop, zbot, t = head_shape(u)
            for v in vs:
                z = zbot + v * (ztop - zbot)
                w = 2 * v - 1
                rnd = (1 - abs(w) ** 7) ** (1 / 7)
                x = side * t / 2 * rnd
                yy = y - 0.010 * (1 - w * w) * max(0.0, (u - 0.75) / 0.25) ** 2   # convex cutting edge
                verts.append((x, yy, z))
    verts = np.array(verts)
    F = []
    blk = Nu * Nv
    for s in range(2):
        for i in range(Nu - 1):
            for j in range(Nv - 1):
                a = s * blk + i * Nv + j
                q = [a, a + Nv, a + Nv + 1, a + 1]
                F.append(q if s == 0 else q[::-1])
    # stitch the two sides along v=0 / v=1 by welding (x == 0 there) and cap poll/edge ends
    V2 = verts.copy()
    remap = np.arange(len(V2))
    for i in range(Nu):
        for j in (0, Nv - 1):
            remap[blk + i * Nv + j] = i * Nv + j
    F = [[int(remap[k]) for k in f] for f in F]
    poll = [i for i in range(Nv)] + [int(remap[blk + j]) for j in range(Nv - 2, 0, -1)]
    F.append(poll)
    edge = [(Nu - 1) * Nv + j for j in range(Nv)] + [int(remap[blk + (Nu - 1) * Nv + j]) for j in range(Nv - 2, 0, -1)]
    F.append(edge[::-1])
    used = sorted(set(k for f in F for k in f))
    idx = {k: n for n, k in enumerate(used)}
    V2 = V2[used]
    F = [[idx[k] for k in f] for f in F]
    mb.add(V2, F, 1)
    # wooden kerf wedge + steel ring wedge on the handle top
    wd = rbox(0.0035, 0.028, 0.012, center=(0, -0.001, 0.341))
    mb.add_bm(wd, 0)
    V, F = cyl(0.0042, 0.006, 10, z0=0.339)
    mb.add(V, F, 2, M=T(0.0, 0.004, 0))
    obj = mb.build('hatchet', [ash, steel, wedge], sharp=48)
    return [finish_obj(obj, {'1': 1.5})]


# ---------------------------------------------------------------- scaffold plank
PL_L, PL_W, PL_T = 3.9, 0.225, 0.038


def plank_deform(V, seed=0):
    x = V[:, 0]
    k = 2 * x / PL_L
    V = V.copy()
    V[:, 2] += 0.007 * (1 - k * k)
    th = 0.010 * k
    yc, zc = V[:, 1], V[:, 2] - PL_T / 2
    V[:, 1] = yc * np.cos(th) - zc * np.sin(th)
    V[:, 2] = yc * np.sin(th) + zc * np.cos(th) + PL_T / 2
    return V


def build_plank():
    mat = M_wood('m_plank', texname='rough_wood', tile=0.9, axis='x', tint=srgb((236, 228, 212))[:3], grey=0.45, dirt=0.6,
                 mud=0.55, mud_h=0.012, ends=(-PL_L / 2, PL_L / 2), splash=0.5, seed=1, val=1.05, sat=0.9)
    band = M_galv('m_plank_band', rust=0.75, grime=0.7, seed=4, spray_h=0.02, white_rust=0.3)
    mb = MB()
    r = 0.004
    ys_a, ys_b = 0.028, -0.041         # crack positions at the -X / +X ends
    dlt = 0.0026
    # profile: bottom edge left->right, top edge right->left (CCW seen from +X)
    bot = [(-PL_W / 2 + r, 0.0)] + sorted([(ys_a - dlt, 0.0), (ys_a, 0.0), (ys_a + dlt, 0.0), (ys_b - dlt, 0.0), (ys_b, 0.0), (ys_b + dlt, 0.0)]) + [(PL_W / 2 - r, 0.0)]
    corner = lambda cx, cz, a0: [(cx + r * math.cos(math.radians(a0 + 90 * k / 2)), cz + r * math.sin(math.radians(a0 + 90 * k / 2))) for k in range(1, 2)]
    prof = []
    prof += bot
    prof += corner(PL_W / 2 - r, r, 270)
    prof += [(PL_W / 2, r), (PL_W / 2, PL_T - r)]
    prof += corner(PL_W / 2 - r, PL_T - r, 0)
    prof += [(y, PL_T) for (y, _) in bot[::-1]]
    prof += corner(-PL_W / 2 + r, PL_T - r, 90)
    prof += [(-PL_W / 2, PL_T - r), (-PL_W / 2, r)]
    prof += corner(-PL_W / 2 + r, r, 180)
    prof = np.array(prof)
    m = len(prof)
    crack_ids = {}
    for key, ys in (('a', ys_a), ('b', ys_b)):
        ids = [i for i in range(m) if abs(prof[i, 0] - ys) < 1e-9]
        crack_ids[key] = ids                     # [bottom-center, top-center]
    rng = np.random.default_rng(7)
    for half in (0, 1):
        if half == 0:
            xs = np.concatenate([[-PL_L / 2], -PL_L / 2 + np.array([0.012, 0.03, 0.06, 0.1, 0.16, 0.24, 0.32]), np.linspace(-1.5, 0.0, 6)])
        else:
            xs = np.concatenate([np.linspace(0.0, 1.5, 6), PL_L / 2 - np.array([0.32, 0.24, 0.16, 0.1, 0.06, 0.03, 0.012]), [PL_L / 2]])
        secs = []
        for x in xs:
            S_ = np.zeros((m, 3))
            S_[:, 0] = x
            S_[:, 1] = prof[:, 0]
            S_[:, 2] = prof[:, 1]
            de = PL_L / 2 - abs(x)
            key = 'a' if x < 0 else 'b'
            depth = min(1.0, max(0.0, 1 - de / (0.30 if key == 'a' else 0.22))) ** 1.4 * PL_T * 0.5
            for i in crack_ids[key]:
                S_[i, 2] += depth if prof[i, 1] < PL_T / 2 else -depth
            # neighbours of the crack open up into a V near the end
            if de < 0.3:
                for i in range(m):
                    yv = prof[i, 0]
                    ysx = ys_a if key == 'a' else ys_b
                    if 0 < abs(yv - ysx) < dlt * 1.5:
                        S_[i, 1] += np.sign(yv - ysx) * 0.0015 * max(0, 1 - de / 0.3)
            if de < 1e-6:
                # sawn but uneven end + a broken corner
                jit = 0.003 * np.sin(prof[:, 0] * 60 + half) + 0.002 * (prof[:, 1] / PL_T)
                S_[:, 0] += -np.sign(x) * np.abs(jit)
            if half == 1:
                brk = np.clip((prof[:, 0] - 0.06) / 0.05, 0, 1) * np.clip(prof[:, 1] / PL_T, 0.3, 1)
                S_[:, 0] = np.minimum(S_[:, 0], PL_L / 2 - 0.045 * brk - (0.0 if de < 1e-6 else 0.0005))
            secs.append(S_)
        secs = np.array(secs)
        nS = len(secs)
        V = secs.reshape(-1, 3)
        F = [[i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j] for i in range(nS - 1) for j in range(m)]
        if half == 0:
            F.append(list(range(m))[::-1])
        else:
            F.append(list(range((nS - 1) * m, nS * m)))
        mb.add(V, F, 0, recalc=False)
    # galvanised hoop bands + nails near both ends
    rect = rrect(PL_W, PL_T, r, 2)
    for sgn in (-1, 1):
        xa = sgn * (PL_L / 2 - 0.06)
        xb = sgn * (PL_L / 2 - 0.03)
        outer = rrect(PL_W + 0.003, PL_T + 0.003, r + 0.0015, 2)
        inner = rrect(PL_W - 0.004, PL_T - 0.004, r, 2)
        mo = len(outer)
        V = []
        for x in (xa, xb):
            for p in outer:
                V.append((x, p[0], p[1] + PL_T / 2))
        for x in (xb, xa):
            for p in inner:
                V.append((x, p[0], p[1] + PL_T / 2))
        V = np.array(V)
        F = []
        for ring in range(4):
            r0, r1 = ring * mo, ((ring + 1) % 4) * mo
            for j in range(mo):
                F.append([r0 + j, r0 + (j + 1) % mo, r1 + (j + 1) % mo, r1 + j])
        mb.add(V, F, 1, recalc=True)
        for yn in (-0.07, 0.07):
            V, F = dome(0.0045, 0.0018, 8)
            mb.add(V, F, 1, M=T((xa + xb) / 2, yn, PL_T + 0.0014))
    V = plank_deform(mb.verts())
    obj = mb.build('plank', [mat, band], sharp=45, V=V)
    return [finish_obj(obj)]


def build_bearer():
    mat = M_wood('m_bearer', texname='rough_wood', tile=0.8, axis='y', tint=srgb((215, 200, 180))[:3], grey=0.3, dirt=0.7,
                 mud=0.8, mud_h=0.035, splash=0.2, seed=6, val=0.8)
    mb = MB()
    mb.add_bm(rbox(0.10, 0.72, 0.075, r=0.006, segs=2, center=(0, 0, 0.0375), sub=2))
    V = mb.verts()
    rng = np.random.default_rng(3)
    V[:, 2] += 0.002 * np.sin(V[:, 1] * 9)
    obj = mb.build('bearer', [mat], sharp=40, V=V)
    return [finish_obj(obj)]


def build_plank_wedge():
    """Timber ramp wedge: across X 0.55 m, along Y 0.32 m, rises from ~0 at -Y... (three: rises toward +Z)."""
    mat = M_wood('m_wedge', texname='rough_wood', tile=0.8, axis='x', tint=srgb((220, 206, 186))[:3], grey=0.4, dirt=0.6,
                 mud=0.7, mud_h=0.02, seed=8)
    mb = MB()
    Lx, Ly, Hh = 0.55, 0.32, 0.042
    # thick end at Blender -Y (three +Z), thin end at +Y
    prof = np.array([(-Ly / 2, 0.0), (Ly / 2, 0.0), (Ly / 2, 0.004), (-Ly / 2 + 0.01, Hh), (-Ly / 2, Hh - 0.003)])
    xs = np.linspace(-Lx / 2, Lx / 2, 5)
    secs = np.array([[(x, p[0], p[1]) for p in prof] for x in xs])
    nS, m = secs.shape[:2]
    V = secs.reshape(-1, 3)
    F = [[i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j] for i in range(nS - 1) for j in range(m)]
    F += [list(range(m))[::-1], list(range((nS - 1) * m, nS * m))]
    mb.add(V, F, 0)
    obj = mb.build('plank_wedge', [mat], sharp=30)
    return [finish_obj(obj)]


# ---------------------------------------------------------------- crate
def build_crate():
    mat = M_wood('m_crate', texname='rough_wood', tile=0.7, axis='x', tint=srgb((238, 222, 196))[:3], grey=0.25, dirt=0.55,
                 mud=0.5, mud_h=0.05, splash=0.2, seed=11, val=1.08, sat=1.05)
    mat2 = M_wood('m_crate_v', texname='rough_wood', tile=0.7, axis='z', tint=srgb((228, 212, 186))[:3], grey=0.3, dirt=0.6,
                  mud=0.5, mud_h=0.05, splash=0.2, seed=12, val=1.0)
    mat3 = M_wood('m_crate_y', texname='rough_wood', tile=0.7, axis='y', tint=srgb((238, 222, 196))[:3], grey=0.25, dirt=0.55,
                  mud=0.5, mud_h=0.05, splash=0.2, seed=13, val=1.05)
    mb = MB()
    Lx, Ly, Hz = 0.80, 0.50, 0.46
    t, post = 0.018, 0.045
    rng = np.random.default_rng(21)
    for sx in (-1, 1):
        for sy in (-1, 1):
            mb.add_bm(rbox(post, post, Hz - 0.02, r=0.003, segs=1, center=(sx * (Lx / 2 - post / 2), sy * (Ly / 2 - post / 2), 0.01 + (Hz - 0.02) / 2)), 1)
    slat_h, gap = 0.105, 0.03
    for k in range(3):
        zc = 0.035 + k * (slat_h + gap) + slat_h / 2
        for sy in (-1, 1):
            j = rng.normal(0, 0.002, 3)
            mb.add_bm(rbox(Lx, t, slat_h, r=0.002, segs=1, center=(j[0], sy * (Ly / 2 + t / 2), zc + j[1])), 0,
                      M=R('X', rng.normal(0, 0.3)))
        for sx in (-1, 1):
            j = rng.normal(0, 0.002, 3)
            mb.add_bm(rbox(t, Ly, slat_h, r=0.002, segs=1, center=(sx * (Lx / 2 + t / 2), j[0], zc + j[1])), 2)
    # bottom boards and lid boards
    for k in range(4):
        w = (Lx - 0.01) / 4
        xc = -Lx / 2 + w / 2 + k * w + 0.0025
        mb.add_bm(rbox(w - 0.006, Ly + 2 * t, t, r=0.002, segs=1, center=(xc, 0, t / 2)), 2)
        j = rng.normal(0, 0.0015, 2)
        mb.add_bm(rbox(w - 0.008, Ly + 2 * t, t, r=0.002, segs=1, center=(xc + j[0], j[1], Hz + t / 2)), 2,
                  M=R('Y', rng.normal(0, 0.4)))
    for sx in (-1, 1):
        mb.add_bm(rbox(0.07, Ly - 0.02, t, r=0.002, segs=1, center=(sx * (Lx / 2 - 0.08), 0, Hz + 1.5 * t)), 2)
    obj = mb.build('crate', [mat, mat2, mat3], sharp=40)
    return [finish_obj(obj)]


# ---------------------------------------------------------------- toolbox
def build_toolbox():
    paint = M_paint('m_toolbox', (150, 28, 24), chip=0.5, rust=0.45, dirt=0.6, mud=0.4, mud_h=0.04, gloss=0.35,
                    fade=0.3, edge_r=0.004, seed=7, primer=(90, 90, 88))
    black = M_rubber('m_toolbox_grip', color=(24, 24, 24), dust=0.4, rough=0.6)
    mb = MB()
    Lx, Ly, Hb, Hl = 0.52, 0.21, 0.16, 0.055
    mb.add_bm(rbox(Lx, Ly, Hb, r=0.008, segs=3, center=(0, 0, Hb / 2 + 0.004)), 0)
    # lid: slight dome, overlapping lip
    bm = rbox(Lx + 0.006, Ly + 0.006, Hl, r=0.012, segs=3, center=(0, 0, Hb + 0.004 + Hl / 2 + 0.002), sub=0)
    for v in bm.verts:
        if v.co.z > Hb + Hl * 0.8:
            v.co.z += 0.008 * (1 - (v.co.x / (Lx / 2)) ** 2)
    mb.add_bm(bm, 0)
    # rolled reinforcement bead at the box rim + feet
    rim = rrect(Lx + 0.004, Ly + 0.004, 0.009, 3)
    P3 = np.array([(p[0], p[1], Hb - 0.004) for p in rim])
    n = len(P3)
    Tn = np.roll(P3, -1, 0) - np.roll(P3, 1, 0)
    Tn /= np.linalg.norm(Tn, axis=1, keepdims=True)
    Bv = np.tile([0, 0, 1.0], (n, 1))
    Nv = np.cross(Tn, Bv)
    if np.dot(Nv[0], P3[0] * np.array([1, 1, 0])) < 0:
        Nv = -Nv
    V, F = sweep(P3, circ(0.003, 6), closed_path=True, N=Nv, B=Bv)
    mb.add(V, F, 0)
    # latches on the front (-Y) and hinge knuckles on the back
    for xl in (-0.17, 0.17):
        mb.add_bm(rbox(0.035, 0.008, 0.05, r=0.002, segs=1, center=(xl, -Ly / 2 - 0.006, Hb + 0.004)), 0)
        mb.add_bm(rbox(0.02, 0.006, 0.02, r=0.002, segs=1, center=(xl, -Ly / 2 - 0.011, Hb + 0.018)), 0)
    V, F = cyl(0.005, Lx - 0.06, 10, z0=-(Lx - 0.06) / 2)
    mb.add(V, F, 0, M=mm(T(0, Ly / 2 + 0.004, Hb + 0.006), R('Y', 90)))
    # carry handle: steel rod brackets + black grip
    for xs in (-0.075, 0.075):
        mb.add_bm(rbox(0.018, 0.03, 0.012, r=0.003, segs=1, center=(xs, 0, Hb + Hl + 0.014)), 0)
    pts = [(-0.075, 0, Hb + Hl + 0.016), (-0.07, 0, Hb + Hl + 0.045), (-0.05, 0, Hb + Hl + 0.056), (0.05, 0, Hb + Hl + 0.056),
           (0.07, 0, Hb + Hl + 0.045), (0.075, 0, Hb + Hl + 0.016)]
    V, F = tube(pts, 0.004, 8, 20)
    mb.add(V, F, 0)
    V, F = cyl(0.011, 0.1, 12, z0=-0.05)
    mb.add(V, F, 1, M=mm(T(0, 0, Hb + Hl + 0.056), R('Y', 90)))
    obj = mb.build('toolbox', [paint, black], sharp=45)
    return [finish_obj(obj)]


# ---------------------------------------------------------------- jersey barrier
def build_barrier():
    mat = M_concrete_paint('m_barrier', bands_x=0.5, x0=-1.0, seed=2)
    steel = M_galv('m_barrier_loop', rust=0.9, grime=0.7, seed=5)
    L = 2.0
    prof = np.array([(0.30, 0.0), (0.30, 0.065), (0.29, 0.08), (0.122, 0.33), (0.075, 0.78), (0.06, 0.80),
                     (-0.06, 0.80), (-0.075, 0.78), (-0.122, 0.33), (-0.29, 0.08), (-0.30, 0.065), (-0.30, 0.0)])
    prof = prof[::-1]
    rng = np.random.default_rng(5)
    xs = np.linspace(-L / 2, L / 2, 9)
    secs = []
    for x in xs:
        S_ = np.zeros((len(prof), 3))
        S_[:, 0] = x + (0.004 * rng.standard_normal(len(prof)) if abs(abs(x) - L / 2) < 1e-6 else 0)
        S_[:, 1] = prof[:, 0] * (1 + 0.004 * math.sin(x * 2.1))
        S_[:, 2] = prof[:, 1] + (0.003 * math.sin(x * 3.3) if True else 0) * (prof[:, 1] > 0.5)
        secs.append(S_)
    secs = np.array(secs)
    nS, m = secs.shape[:2]
    V = secs.reshape(-1, 3)
    F = [[i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j] for i in range(nS - 1) for j in range(m)]
    F += [list(range(m))[::-1], list(range((nS - 1) * m, nS * m))]
    me = bpy.data.meshes.new('barrier_tmp')
    V2, F2 = _recalc(V, F)
    me.from_pydata(V2.tolist(), [], F2)
    base = link(bpy.data.objects.new('barrier_tmp', me))
    # booleans: drain/fork slot, lifting holes, chipped arrises
    cutters = []
    def cutter(bm, name):
        mc = bpy.data.meshes.new(name)
        bm.to_mesh(mc)
        bm.free()
        o = link(bpy.data.objects.new(name, mc))
        o.hide_render = True
        cutters.append(o)
        return o
    cutter(rbox(0.42, 0.8, 0.14, r=0.01, segs=2, center=(0, 0, 0.0)), 'c_slot')
    for xh in (-0.55, 0.55):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.03, radius2=0.03, depth=0.5)
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, 'X'), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(xh, 0, 0.66), verts=bm.verts)
        cutter(bm, 'c_hole')
    chips = [(-0.8, 0.07, 0.80), (0.35, -0.07, 0.8), (0.95, 0.28, 0.07), (-0.2, -0.29, 0.07), (0.99, -0.06, 0.55),
             (-0.99, 0.11, 0.3), (0.6, 0.075, 0.79), (-0.45, 0.29, 0.07)]
    for i, (cx, cy, cz) in enumerate(chips):
        bm = icochunk(0.028 + 0.018 * rng.random(), 40 + i, 1)
        bmesh.ops.scale(bm, vec=(1.9 if abs(abs(cx) - 1.0) > 0.05 else 0.8, 0.8, 0.8 if abs(abs(cx) - 1.0) > 0.05 else 1.9), verts=bm.verts)
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Euler(tuple((rng.random(3) - 0.5) * 0.6)).to_matrix(), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(cx, cy, cz), verts=bm.verts)
        cutter(bm, f'c_chip{i}')
    for c in cutters:
        md = base.modifiers.new('b', 'BOOLEAN')
        md.operation = 'DIFFERENCE'
        md.object = c
        md.solver = 'EXACT'
    dg = bpy.context.evaluated_depsgraph_get()
    ev = base.evaluated_get(dg)
    me2 = bpy.data.meshes.new_from_object(ev)
    mb = MB()
    Vb = np.array([v.co[:] for v in me2.vertices])
    Fb = [list(p.vertices) for p in me2.polygons]
    mb.add(Vb, Fb, 0, recalc=False)
    for c in cutters:
        bpy.data.objects.remove(c)
    bpy.data.objects.remove(base)
    # steel connection loops at the ends
    for sx in (-1, 1):
        for zl in (0.22, 0.58):
            pts = [(sx * (L / 2 - 0.03), -0.05, zl), (sx * (L / 2 + 0.045), -0.045, zl), (sx * (L / 2 + 0.06), 0.0, zl),
                   (sx * (L / 2 + 0.045), 0.045, zl), (sx * (L / 2 - 0.03), 0.05, zl)]
            V, F = tube(pts, 0.009, 8, 14)
            mb.add(V, F, 1)
    obj = mb.build('barrier', [mat, steel], sharp=35)
    return [finish_obj(obj)]


# ---------------------------------------------------------------- traffic cone
def build_cone():
    body = M_plastic('m_cone', (226, 74, 22), rough=0.5, dirt=0.75, scuff=0.7, fade=0.12, mud=0.4, mud_h=0.09,
                     bands=[(0.465, 0.615), (0.29, 0.375)], seed=1)
    base = M_rubber('m_cone_base', color=(26, 26, 26), dust=0.6, rough=0.85)
    mb = MB()
    # square rubber base with rounded corners and chamfered top
    B = 0.38
    for (w, z) in ((B, 0.0), (B, 0.022), (B - 0.02, 0.034)):
        pass
    base_prof = [(0.0, 0.0)]
    sq = rrect(B, B, 0.045, 3)
    sq2 = rrect(B - 0.022, B - 0.022, 0.035, 3)
    ring_r = 0.155
    V = []
    for P, z in ((sq, 0.0), (sq, 0.024), (sq2, 0.036)):
        for p in P:
            V.append((p[0], p[1], z))
    V = np.array(V)
    m = len(sq)
    F = [[i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j] for i in range(2) for j in range(m)]
    F += [list(range(m))[::-1], list(range(2 * m, 3 * m))]
    mb.add(V, F, 1)
    # collar + cone body (lathe) with a rolled top lip and dark opening
    prof = [(0.17, 0.03), (0.172, 0.05), (0.158, 0.058), (0.148, 0.07), (0.043, 0.72), (0.045, 0.728), (0.042, 0.735),
            (0.030, 0.735), (0.028, 0.70)]
    V, F = lathe(prof, 32, cap0=False, cap1=True)
    mb.add(V, F, 0, recalc=False)
    obj = mb.build('cone', [body, base], sharp=35)
    return [finish_obj(obj)]


# ---------------------------------------------------------------- delineator + rail reflector
def delin_profile(w=0.12, d=0.1):
    # CCW in (x, y); front (reflector side) toward -y
    return np.array([(-w / 2 + 0.012, -d / 2), (w / 2 - 0.012, -d / 2), (w / 2, -d / 2 + 0.012), (0.022, d / 2),
                     (-0.022, d / 2), (-w / 2, -d / 2 + 0.012)])


def build_delineator():
    mat = M_plastic('m_delineator', (232, 232, 226), rough=0.4, dirt=0.6, scuff=0.4, fade=0.2, mud=0.5, mud_h=0.14,
                    black=(0.72, 1.2), seed=3)
    refl = M_reflector('m_delin_refl', (238, 190, 60))
    mb = MB()
    P2 = delin_profile()
    m = len(P2)
    zs = [-0.15, 0.3, 0.72, 0.99]
    V = []
    for z in zs:
        for p in P2:
            ztop = z if z < 0.9 else 1.0 - 0.02 * (p[1] + 0.05) / 0.1 * -1 - 0.02
            V.append((p[0], p[1], ztop))
    V = np.array(V)
    F = [[i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j] for i in range(len(zs) - 1) for j in range(m)]
    F += [list(range(m))[::-1], list(range((len(zs) - 1) * m, len(zs) * m))]
    mb.add(V, F, 0)
    mb.add_bm(rbox(0.045, 0.006, 0.17, r=0.0015, segs=1, center=(0, -0.052, 0.855)), 1)
    obj = mb.build('delineator', [mat, refl], sharp=35)
    return [finish_obj(obj)]


RAIL_X = 0.20        # back plane of the W-beam (post at x=0, traffic side +X)
RAIL_Z = 0.595       # beam centre height


def build_rail_reflector():
    """Small reflector block standing on the beam at the post (same local frame as 'guardrail'); reflector faces
    Blender +Y = three -Z, i.e. toward traffic approaching from lower s."""
    mat = M_plastic('m_railrefl', (230, 230, 224), rough=0.4, dirt=0.6, scuff=0.3, fade=0.2, mud=0.2, mud_h=0.3,
                    black=(0.8, 0.95), seed=5)
    refl = M_reflector('m_railrefl_r', (240, 200, 70))
    mb = MB()
    P2 = delin_profile(0.07, 0.05)
    P2 = np.stack([P2[:, 0], -P2[:, 1]], -1)[::-1]
    m = len(P2)
    V = []
    for z in (0.66, 0.95):
        for p in P2:
            V.append((p[0] + RAIL_X + 0.05, p[1], z))
    V = np.array(V)
    F = [[j, (j + 1) % m, m + (j + 1) % m, m + j] for j in range(m)]
    F += [list(range(m))[::-1], list(range(m, 2 * m))]
    mb.add(V, F, 0)
    mb.add_bm(rbox(0.03, 0.004, 0.075, r=0.001, segs=1, center=(RAIL_X + 0.05, 0.026, 0.875)), 1)
    mb.add_bm(rbox(0.012, 0.03, 0.09, r=0.002, segs=1, center=(RAIL_X + 0.012, 0, 0.7)), 0)
    obj = mb.build('rail_reflector', [mat, refl], sharp=35)
    return [finish_obj(obj)]


# ---------------------------------------------------------------- signs
SIGN_SIDE = 0.9
SIGN_CZ = 2.0      # centroid height of the triangle above the post base
SIGN_EXT = (-0.47, 0.47, SIGN_CZ - 0.47 + 0.07, SIGN_CZ + 0.47 + 0.07)


def sign_image(kind, res=1024):
    x0, x1, z0, z1 = SIGN_EXT
    ext = (x0, x1, z0 - SIGN_CZ, z1 - SIGN_CZ)       # relative to the centroid
    outer = rounded_triangle(SIGN_SIDE, 0.045, 0.0)
    red_o = inset_poly(outer, 0.010)
    red_i = inset_poly(outer, 0.075)
    c_rim = raster(res, res, [outer], ext, 2)
    c_red = raster(res, res, [red_o], ext, 2)
    c_fld = raster(res, res, [red_i], ext, 2)
    white = np.array([0.93, 0.93, 0.91])
    red = np.array([0.78, 0.10, 0.12])
    field = white if kind == 'rockfall' else np.array([0.97, 0.76, 0.12])
    black = np.array([0.07, 0.07, 0.075])
    img = np.zeros((res, res, 3))
    img[:] = white
    img = img * (1 - c_red[..., None]) + red * c_red[..., None]
    img = img * (1 - c_fld[..., None]) + field * c_fld[..., None]
    polys = []
    if kind == 'rockfall':
        cliff = np.array([(0.30, -0.33), (0.30, 0.25), (0.13, 0.25), (0.105, 0.19), (0.125, 0.13), (0.09, 0.07), (0.115, 0.0),
                          (0.075, -0.06), (0.10, -0.12), (0.06, -0.18), (0.075, -0.24), (0.03, -0.33)])
        polys.append(cliff)
        for (cx, cy, r, sd) in ((0.02, 0.1, 0.034, 1), (-0.055, 0.0, 0.043, 2), (-0.12, -0.13, 0.05, 3), (0.012, -0.05, 0.018, 4),
                                (-0.03, 0.155, 0.017, 5), (-0.1, 0.07, 0.014, 6), (-0.19, -0.2, 0.022, 7)):
            polys.append(rock_poly(cx, cy, r, sd))
    else:
        polys.append(np.array([(-0.02, -0.33), (0.02, -0.25), (0.08, -0.205), (0.14, -0.19), (0.2, -0.21), (0.26, -0.26), (0.3, -0.33)]))
        polys.append(circle(-0.075, 0.11, 0.034))
        polys.append(np.array([(-0.115, 0.07), (-0.05, 0.075), (0.0, -0.02), (-0.03, -0.06), (-0.075, -0.05)]))
        polys.append(thick_line((-0.05, -0.05), (-0.12, -0.24), 0.036))
        polys.append(thick_line((-0.02, -0.045), (0.04, -0.14), 0.034))
        polys.append(thick_line((0.04, -0.14), (0.03, -0.25), 0.032))
        polys.append(thick_line((-0.05, 0.05), (0.035, -0.02), 0.026))
        polys.append(thick_line((-0.03, 0.08), (0.15, -0.14), 0.013))
        polys.append(np.array([(0.13, -0.12), (0.175, -0.155), (0.16, -0.215), (0.12, -0.19)]))
    c_pic = raster(res, res, polys, ext) * c_fld
    img = img * (1 - c_pic[..., None]) + black * c_pic[..., None]
    img = img * c_rim[..., None] + np.array([0.6, 0.6, 0.6]) * (1 - c_rim[..., None])
    return img   # authored as sRGB display values


def build_sign(kind):
    name = 'sign_' + kind
    arr = sign_image(kind)
    im = np_image(name + '_face', arr, os.path.join(BAKE, f'_{name}_face.png'), noncolor=False)
    face = M_sign_face('m_' + name + '_face', im, SIGN_EXT, seed=3 if kind == 'rockfall' else 5)
    back = M_alu('m_' + name + '_back', seed=2, dirt=0.6)
    galv = M_galv('m_' + name + '_post', rust=0.35, grime=0.5, seed=7 if kind == 'rockfall' else 9, spray_h=0.6, white_rust=0.5)
    cap = M_rubber('m_' + name + '_cap', color=(20, 20, 20), dust=0.3, rough=0.5)
    mb = MB()
    outer = rounded_triangle(SIGN_SIDE, 0.045, 0.0)
    n = len(outer)
    yF, yB, yL = -0.0012, 0.0012, 0.02
    # face plate (front, facing -Y) + back plate + folded flange around the perimeter
    V = [(p[0], yF, p[1] + SIGN_CZ) for p in outer]
    mb.add(np.array(V), [list(range(n))], 0, recalc=False)
    # make sure the face points to -Y
    V = [(p[0], yB, p[1] + SIGN_CZ) for p in outer]
    mb.add(np.array(V), [list(range(n))[::-1]], 1, recalc=False)
    inner = inset_poly(outer, 0.012)
    V = []
    for P_, y in ((outer, yF), (outer, yL), (inner, yL), (inner, yB)):
        for p in P_:
            V.append((p[0], y, p[1] + SIGN_CZ))
    V = np.array(V)
    F = []
    for ring in range(3):
        for j in range(n):
            F.append([ring * n + j, ring * n + (j + 1) % n, (ring + 1) * n + (j + 1) % n, (ring + 1) * n + j][::-1])
    mb.add(V, F, 1, recalc=False)
    # post (galvanised tube) behind the panel, cap, two U-clamps with nuts on horizontal back rails
    ypost = 0.05
    V, F = cyl(0.0305, 2.62, 20, z0=-0.45)
    mb.add(V, F, 2, M=T(0, ypost, 0))
    V, F = lathe([(0.033, 0.0), (0.033, 0.018), (0.02, 0.028)], 16)
    mb.add(V, F, 3, M=T(0, ypost, 2.17))
    for zr in (SIGN_CZ - 0.16, SIGN_CZ + 0.14):
        half = 0.30 if zr < SIGN_CZ else 0.14
        mb.add_bm(rbox(2 * half, 0.012, 0.04, r=0.002, segs=1, center=(0, 0.008, zr)), 2)
        pts = [(-0.037, 0.015, zr), (-0.037, ypost + 0.01, zr), (-0.02, ypost + 0.034, zr), (0.0, ypost + 0.04, zr),
               (0.02, ypost + 0.034, zr), (0.037, ypost + 0.01, zr), (0.037, 0.015, zr)]
        V, F = tube(pts, 0.0045, 6, 16)
        mb.add(V, F, 2)
        for xn in (-0.037, 0.037):
            V, F = hexnut(0.008, 0.007)
            mb.add(V, F, 2, M=mm(T(xn, 0.014, zr), R('X', -90)))
    obj = mb.build(name, [face, back, galv, cap], sharp=40)
    return [finish_obj(obj, {'0': 1.8})]


# ---------------------------------------------------------------- guardrail
def w_profile(n=13, depth=0.083, height=0.31, thick=0.003):
    """W-beam (A-profile) front surface points (x = toward traffic from the back plane, z) + back offset, CCW."""
    zs = np.linspace(-height / 2, height / 2, n)
    t = zs / (height / 2)
    # two flat crests at |t|~0.5, valley at 0, edges folded back at |t|=1
    x = depth * np.clip(1.45 * np.sin(np.pi * np.abs(t)) ** 0.8, 0, 1)
    x[np.abs(t) > 0.97] = 0.012
    front = np.stack([x, zs], -1)
    # back surface: offset along the profile normal
    d = np.gradient(front, axis=0)
    nrm = np.stack([d[:, 1], -d[:, 0]], -1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    back = front - nrm * thick
    prof = np.concatenate([front, back[::-1]])
    return prof


C_POST = np.array([(0.025, -0.05), (0.025, 0.05), (-0.025, 0.05), (-0.025, 0.046), (0.021, 0.046), (0.021, -0.046),
                   (-0.025, -0.046), (-0.025, -0.05)])


def add_post(mb, mat, y=0.0, z0=-0.45, z1=0.72):
    Vp = np.array([(p[0], p[1] + y, z) for z in (z0, z1) for p in C_POST])
    m = len(C_POST)
    F = [[j, (j + 1) % m, m + (j + 1) % m, m + j] for j in range(m)]
    F += [list(range(m))[::-1], list(range(m, 2 * m))]
    mb.add(Vp, F, mat)
    # spacer block (short channel) between the post web and the beam
    mb.add_bm(rbox(RAIL_X - 0.025, 0.07, 0.24, r=0.002, segs=1, center=((RAIL_X + 0.025) / 2, y, RAIL_Z)), mat)


def build_guardrail(lod=False):
    """Segment: post at the origin, beam from y=0 to y=-4.0 (three +Z) plus a 0.16 m splice overlap past the next
    post, offset 3.5 mm outward so it laps over the next segment's beam start."""
    name = 'guardrail_lod' if lod else 'guardrail'
    galv = M_galv('m_' + name, rust=0.45, grime=0.6, seed=1, spray_h=0.5, white_rust=0.4) if not lod else None
    post_m = M_galv('m_' + name + '_post', rust=0.6, grime=0.8, seed=3, spray_h=0.35, white_rust=0.3) if not lod else None
    mb = MB()
    prof = w_profile(7 if lod else 13)
    m = len(prof)
    if lod:
        ys = [0.0, -4.0]
    else:
        ys = [0.0] + list(np.linspace(-0.25, -3.75, 8)) + [-4.0, -4.0002, -4.16]
    secs = []
    for y in ys:
        off = 0.0035 if (y < -4.0001 and not lod) else 0.0
        secs.append([(RAIL_X + p[0] + off, y, RAIL_Z + p[1]) for p in prof])
    secs = np.array(secs)
    nS = len(secs)
    V = secs.reshape(-1, 3)
    F = [[i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j] for i in range(nS - 1) for j in range(m)]
    F += [list(range(m))[::-1], list(range((nS - 1) * m, nS * m))]
    mb.add(V, F, 0, tag='beam')
    if lod:
        mb.add_bm(rbox(0.05, 0.1, 1.17, center=(0, 0, 0.72 - 0.585)), 0)
        mb.add_bm(rbox(RAIL_X - 0.025, 0.06, 0.24, center=((RAIL_X + 0.025) / 2, 0, RAIL_Z)), 0)
    else:
        add_post(mb, 1)
        bolts = [(0.0, RAIL_Z, 0.0)]
        for yb in (-4.045, -4.115):
            for zb in (-0.095, -0.06, 0.06, 0.095):
                bolts.append((yb, RAIL_Z + zb, 0.0035))
        for (yb, zb, off) in bolts:
            crest = abs(zb - RAIL_Z) > 0.01
            xb = RAIL_X + (0.083 * 0.97 if crest else 0.004) + off
            V, F = dome(0.011, 0.006, 8)
            mb.add(V, F, 0, M=mm(T(xb, yb, zb), R('Y', 90)))
    obj = mb.build(name, [galv or M_simple('m_lod', (160, 160, 158)), post_m or M_simple('m_lod2', (140, 140, 138))], sharp=40)
    obj['beam_range'] = json.dumps(mb.ranges.get('beam', []))
    if lod:
        return [obj]
    return [finish_obj(obj)]


def build_guardrail_end():
    """Terminal: beam starts at the origin (lapped by the previous segment), runs 0.55 m along -Y (three +Z) and curls
    back toward -X (away from traffic); its own post stands at y=-0.45."""
    galv = M_galv('m_guardrail_end', rust=0.55, grime=0.7, seed=6, spray_h=0.5, white_rust=0.4)
    post_m = M_galv('m_guardrail_end_post', rust=0.6, grime=0.8, seed=8, spray_h=0.35, white_rust=0.3)
    mb = MB()
    prof = w_profile(13)
    path = [(RAIL_X, 0.0, 0), (RAIL_X, -0.2, 0), (RAIL_X, -0.4, 0), (RAIL_X, -0.55, 0)]
    R0 = 0.3
    for a in np.linspace(0, math.pi * 0.62, 9)[1:]:
        path.append((RAIL_X - R0 + R0 * math.cos(a), -0.55 - R0 * math.sin(a), 0))
    path = np.array(path)
    n = len(path)
    Tn = np.gradient(path, axis=0)
    Tn /= np.linalg.norm(Tn, axis=1, keepdims=True)
    Bv = np.tile([0, 0, 1.0], (n, 1))
    Nv = -np.cross(Tn, Bv)
    V, F = sweep(path - np.array([RAIL_X, 0, 0]) * 0, np.stack([prof[:, 0], prof[:, 1]], -1), N=Nv, B=Bv)
    V[:, 2] += RAIL_Z
    mb.add(V, F, 0, tag='beam')
    add_post(mb, 1, y=-0.45)
    V, F = dome(0.011, 0.006, 8)
    mb.add(V, F, 0, M=mm(T(RAIL_X + 0.004, -0.45, RAIL_Z), R('Y', 90)))
    obj = mb.build('guardrail_end', [galv, post_m], sharp=40)
    return [finish_obj(obj)]


# ---------------------------------------------------------------- light tower
LT_TOP = 2.36


def build_light_tower():
    yellow = M_paint('m_lt_yellow', (214, 160, 24), chip=0.45, rust=0.4, dirt=0.6, mud=0.6, mud_h=0.25, gloss=0.4,
                     fade=0.3, edge_r=0.003, seed=12, primer=(90, 90, 90), dents=0.2)
    chrome = M_galv('m_lt_mast', rust=0.2, grime=0.4, seed=13, spray_h=0.0, white_rust=0.2)
    rub = M_rubber('m_lt_rubber', dust=0.5)
    mb = MB()
    # mast: outer yellow tube + inner galvanised tube + clamp knob
    V, F = cyl(0.026, 1.35, 18, z0=0.25)
    mb.add(V, F, 0)
    V, F = cyl(0.020, LT_TOP - 1.5 + 0.02, 16, z0=1.5)
    mb.add(V, F, 1)
    V, F = lathe([(0.032, 1.52), (0.032, 1.6), (0.027, 1.61)], 16)
    mb.add(V, F, 0)
    V, F = cyl(0.006, 0.07, 8, z0=0.0)
    mb.add(V, F, 2, M=mm(T(0.03, 0, 1.56), R('Y', 90)))
    V, F = lathe([(0.004, 0), (0.016, 0.004), (0.016, 0.018), (0.006, 0.022)], 10)
    mb.add(V, F, 2, M=mm(T(0.1, 0, 1.56), R('Y', 90)))
    # collars
    V, F = cyl(0.036, 0.07, 16, z0=0.72)
    mb.add(V, F, 0)
    V, F = cyl(0.033, 0.05, 16, z0=0.42)
    mb.add(V, F, 0)
    # tripod legs + braces + rubber feet
    for k in range(3):
        a = math.radians(90 + 120 * k)
        ca, sa = math.cos(a), math.sin(a)
        top = np.array([0.04 * ca, 0.04 * sa, 0.75])
        foot = np.array([0.78 * ca, 0.78 * sa, 0.03])
        V, F = sweep(np.linspace(top, foot, 6), circ(0.015, 10), caps=True)
        mb.add(V, F, 0)
        mid = top + (foot - top) * 0.42
        V, F = sweep(np.linspace(np.array([0.035 * ca, 0.035 * sa, 0.445]), mid, 4), circ(0.009, 8), caps=True)
        mb.add(V, F, 0)
        V, F = lathe([(0.03, 0.0), (0.032, 0.012), (0.02, 0.035)], 12)
        mb.add(V, F, 2, M=T(foot[0], foot[1], 0.0))
    # power cable: from the head down the mast (with ties) to the ground and away
    cab = [(0.0, 0.03, LT_TOP - 0.02), (0.0, 0.034, 2.1), (0.0, 0.033, 1.7), (0.0, 0.036, 1.3), (0.01, 0.04, 0.9),
           (0.05, 0.05, 0.55), (0.12, 0.08, 0.25), (0.2, 0.12, 0.05), (0.35, 0.2, 0.007), (0.8, 0.45, 0.007),
           (1.4, 0.35, 0.007), (1.9, 0.6, 0.007), (2.4, 1.0, 0.007)]
    V, F = tube(cab, 0.0065, 8, 70)
    mb.add(V, F, 2)
    for zt in (1.9, 1.35, 1.0):
        V, F = cyl(0.03 if zt < 1.5 else 0.024, 0.012, 14, z0=zt)
        mb.add(V, F, 2)
    obj = mb.build('light_tower', [yellow, chrome, rub], sharp=40)
    # ---- lamp head (child): T-bar + two floodlights facing -Y (three +Z), tilted down
    black = M_paint('m_lt_housing', (38, 38, 38), chip=0.3, rust=0.2, dirt=0.5, mud=0.0, gloss=0.5, fade=0.2,
                    edge_r=0.002, seed=14, primer=(120, 120, 120), dents=0.1)
    lens_bake = M_simple('m_lt_lens_bake', (230, 225, 210), 0.1)
    hb = MB()
    V, F = cyl(0.016, 0.66, 12, z0=-0.33)
    hb.add(V, F, 1, M=R('Y', 90))
    V, F = cyl(0.028, 0.06, 14, z0=-0.05)
    hb.add(V, F, 1)
    for sx in (-1, 1):
        cx = sx * 0.19
        Mh = mm(T(cx, -0.02, 0.12), R('X', 14))
        hb.add_bm(rbox(0.25, 0.085, 0.2, r=0.012, segs=2, center=(0, 0.0, 0.0)), 1, M=Mh)
        # cooling fins on the back
        for k in range(7):
            hb.add_bm(rbox(0.004, 0.03, 0.18, center=(-0.09 + k * 0.03, 0.055, 0.0)), 1, M=Mh)
        # bezel frame + lens (material 'lamp')
        fr = rrect(0.232, 0.182, 0.01, 2)
        fi = rrect(0.2, 0.15, 0.006, 2)
        n = len(fr)
        Vf = np.array([(p[0], -0.043, p[1]) for p in fr] + [(p[0], -0.05, p[1]) for p in fr] + [(p[0], -0.05, p[1]) for p in fi] + [(p[0], -0.043, p[1]) for p in fi])
        Ff = [[r * n + j, r * n + (j + 1) % n, (r + 1) * n + (j + 1) % n, (r + 1) * n + j] for r in range(3) for j in range(n)]
        hb.add(Vf, Ff, 1, M=Mh, recalc=False)
        Vl = np.array([(p[0], -0.046, p[1]) for p in fi])
        hb.add(Vl, [list(range(len(fi)))], 2, M=Mh, recalc=False)
        # U-bracket from the T-bar end to the housing sides + knob
        for bx in (-0.13, 0.13):
            hb.add_bm(rbox(0.006, 0.03, 0.15, r=0.002, segs=1, center=(bx, 0.0, 0.06)), 1, M=mm(T(cx, -0.02, 0.0)))
        V, F = lathe([(0.004, 0), (0.014, 0.004), (0.014, 0.016), (0.005, 0.02)], 10)
        hb.add(V, F, 1, M=mm(T(cx + 0.133, -0.02, 0.12), R('Y', 90)))
        # carry handle
        pts = [(-0.07, 0.0, 0.1), (-0.06, 0.0, 0.125), (0.06, 0.0, 0.125), (0.07, 0.0, 0.1)]
        V, F = tube(pts, 0.0045, 6, 12)
        hb.add(V, F, 1, M=Mh)
    head = hb.build('lamp_head', [chrome, black, lens_bake], sharp=40)
    head['lens_slot'] = 2
    head['lamp_origin'] = (0, 0, LT_TOP)
    return [finish_obj(obj), finish_obj(head, {'1': 1.3})]


# ---------------------------------------------------------------- sandbag
def build_sandbag():
    mat = M_fabric('m_sandbag', texname='hessian_230', tile=0.3, val=0.9, mud=0.6, mud_h=0.05, seed=2, sand=0.35)
    Lx, Wy, Hz = 0.56, 0.34, 0.17
    nu, nv = 36, 18
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0.02, np.pi - 0.02, nv)
    U, Vv = np.meshgrid(u, v)
    e1, e2 = 0.35, 0.55
    cu, su, cv, sv = np.cos(U), np.sin(U), np.cos(Vv), np.sin(Vv)
    sg = lambda a, e: np.sign(a) * np.abs(a) ** e
    X = Lx / 2 * sg(sv, e1) * sg(cu, e2)
    Y = Wy / 2 * sg(sv, e1) * sg(su, e2)
    Z = Hz / 2 * sg(cv, e1)
    rng = np.random.default_rng(4)
    # sag: flat bottom, slumped sides, tied end tapered and folded under at +X
    Z = np.where(Z < 0, Z * 0.55, Z * (0.92 + 0.08 * np.cos(X / Lx * np.pi)))
    tie = np.clip((X - Lx * 0.3) / (Lx * 0.2), 0, 1)
    Y *= 1 - 0.35 * tie
    Z = Z * (1 - 0.5 * tie) - 0.02 * tie * (Z > 0)
    wr = 0.006 * np.sin(X * 55 + Y * 20) * np.sin(Y * 40) + 0.004 * np.sin(X * 90 + Z * 70)
    R_ = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) + 1e-9
    X += wr * X / R_
    Y += wr * Y / R_
    Z += wr * Z / R_
    Z -= Z.min()
    Vt = np.stack([X, Y, Z], -1).reshape(-1, 3)
    F = []
    for i in range(nv - 1):
        for j in range(nu):
            a, b = i * nu + j, i * nu + (j + 1) % nu
            F.append([a, a + nu, b + nu, b])
    top = len(Vt)
    Vt = np.concatenate([Vt, [[0, 0, Z.max() + 0.002], [0, 0, 0.0]]])
    for j in range(nu):
        F.append([top, j, (j + 1) % nu])
        F.append([top + 1, (nv - 1) * nu + (j + 1) % nu, (nv - 1) * nu + j])
    mb = MB()
    mb.add(Vt, F, 0)
    obj = mb.build('sandbag', [mat], sharp=60)
    return [finish_obj(obj)]


# ---------------------------------------------------------------- tarp-covered pile
TARP_W, TARP_D, TARP_DX = 2.9, 2.3, 0.035      # a 3 x 2.4 m PE tarp (hem folded under), cloth grid spacing


def _tarp_supports():
    """Collision proxies for what lies under the tarp (Blender coords, Z up): a pallet with two and a half layers of
    cement bags, a steel box, a coil of drainage pipe, and a scaffold board laid across the top as a ridge (so the
    sheet tents between the bags and the box and sags into pockets that hold rain)."""
    objs = []
    rng = np.random.default_rng(12)

    def add(name, bm, loc, rz=0.0):
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        o = link(bpy.data.objects.new(name, me))
        o.location = loc
        o.rotation_euler = (0, 0, rz)
        objs.append(o)
        return o

    add('tp_pallet', rbox(1.0, 0.85, 0.14, r=0.01, segs=1, center=(0, 0, 0.07)), (-0.42, -0.05, 0))
    for layer in range(2):
        for k in range(4):
            ix, iy = k % 2, k // 2
            cx = -0.42 + (ix - 0.5) * 0.47 + rng.normal(0, 0.025)
            cy = -0.05 + (iy - 0.5) * 0.41 + rng.normal(0, 0.025)
            add(f'tp_bag{layer}{k}', rbox(0.44, 0.38, 0.12, r=0.05, segs=3, center=(0, 0, 0.06)),
                (cx, cy, 0.14 + layer * 0.122), rz=rng.normal(0, 0.12))
    # top layer: a ring of bags (left, right, back, front) round a gap where one bag was taken: the sheet sags into
    # the hole and the pocket holds rain (closed on all sides, so it cannot drain)
    z3 = 0.14 + 2 * 0.122
    add('tp_bag_l', rbox(0.24, 0.66, 0.12, r=0.05, segs=3, center=(0, 0, 0.06)), (-0.42 - 0.37, -0.05, z3), rz=0.05)
    add('tp_bag_r', rbox(0.24, 0.66, 0.12, r=0.05, segs=3, center=(0, 0, 0.06)), (-0.42 + 0.36, -0.04, z3), rz=-0.07)
    add('tp_bag_b', rbox(0.5, 0.22, 0.12, r=0.05, segs=3, center=(0, 0, 0.06)), (-0.42, 0.29, z3), rz=0.04)
    add('tp_bag_f', rbox(0.5, 0.22, 0.12, r=0.05, segs=3, center=(0, 0, 0.06)), (-0.44, -0.39, z3), rz=-0.03)
    # one bag lying across the far corner, higher: a second, smaller pocket forms in its lee
    add('tp_bag_x', rbox(0.44, 0.36, 0.12, r=0.05, segs=3, center=(0, 0, 0.06)), (-0.1, 0.35, z3 + 0.11), rz=0.5)
    add('tp_box', rbox(0.62, 0.72, 0.4, r=0.015, segs=2, center=(0, 0, 0.2)), (0.56, 0.2, 0), rz=0.08)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=24, radius1=0.34, radius2=0.34, depth=0.24)
    add('tp_coil', bm, (0.46, -0.55, 0.12))
    # a board leaning against the stack on the far side
    o = add('tp_lean', rbox(0.2, 0.9, 0.035, r=0.004, segs=1, center=(0, 0, 0)), (-1.0, 0.1, 0.3))
    o.rotation_euler = (0, math.radians(62), 0.1)
    return objs


def _tarp_cloth(supports):
    """Drops a PE tarp onto the supports with Blender cloth (self-collision) and returns (V, F, rest_uv)."""
    nx, ny = int(round(TARP_W / TARP_DX)) + 1, int(round(TARP_D / TARP_DX)) + 1
    xs = np.linspace(-TARP_W / 2, TARP_W / 2, nx)
    ys = np.linspace(-TARP_D / 2, TARP_D / 2, ny)
    X, Y = np.meshgrid(xs, ys)
    rng = np.random.default_rng(5)
    # start slightly crumpled and tilted so the folds are not perfectly symmetric
    # start crumpled (it was thrown over the pile, not laid flat): long soft waves + some buckles, pulled in 10%
    Z = (0.95 + 0.07 * np.sin(X * 3.1 + 0.7) * np.cos(Y * 2.3) + 0.05 * np.sin(X * 7.3 + Y * 5.1) + 0.03 * np.sin(Y * 11.0 - X * 3.0)
         + 0.08 * X / TARP_W)
    V0 = np.stack([X * 0.9 + 0.22, Y * 0.92 - 0.06, Z], -1).reshape(-1, 3)   # thrown off-centre: slack on one side
    F = [[i * nx + j, i * nx + j + 1, (i + 1) * nx + j + 1, (i + 1) * nx + j] for i in range(ny - 1) for j in range(nx - 1)]
    me = bpy.data.meshes.new('tarp_sim')
    me.from_pydata(V0.tolist(), [], F)
    me.update()
    cl = link(bpy.data.objects.new('tarp_sim', me))
    ground_me = bpy.data.meshes.new('tp_ground')
    bmg = bmesh.new()
    bmesh.ops.create_grid(bmg, x_segments=1, y_segments=1, size=4.0)
    bmg.to_mesh(ground_me)
    bmg.free()
    gnd = link(bpy.data.objects.new('tp_ground', ground_me))
    for o in supports + [gnd]:
        o.modifiers.new('Collision', 'COLLISION')
        o.collision.thickness_outer = 0.006
        o.collision.cloth_friction = 12.0
        o.collision.damping = 0.8
    mod = cl.modifiers.new('Cloth', 'CLOTH')
    st = mod.settings
    st.quality = 10
    st.mass = 0.85                 # wet woven PE (~200 g/m2 + water film): heavy enough to sag into the gaps
    st.tension_stiffness = 15
    st.compression_stiffness = 0.8 # buckles (folds, wrinkles) instead of compressing
    st.shear_stiffness = 4
    st.bending_stiffness = 3.2     # crinkly film: angular folds and many wrinkles, not a soft cotton drape
    st.air_damping = 1.0
    st.tension_damping = 8
    st.compression_damping = 8
    st.shear_damping = 8
    cs = mod.collision_settings
    cs.distance_min = 0.006
    cs.use_self_collision = True
    cs.self_distance_min = 0.005
    cs.collision_quality = 3
    sc = bpy.context.scene
    sc.frame_start = 1
    sc.frame_end = 150
    mod.point_cache.frame_start = 1
    mod.point_cache.frame_end = 150
    t0 = time.time()
    for f in range(1, 151):
        sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = cl.evaluated_get(dg)
    me2 = bpy.data.meshes.new_from_object(ev)
    V = np.zeros(len(me2.vertices) * 3)
    me2.vertices.foreach_get('co', V)
    V = V.reshape(-1, 3) @ np.array(cl.matrix_world)[:3, :3].T + np.array(cl.matrix_world)[:3, 3]
    # one gentle Laplacian pass: removes the collision jitter along the ground contact, keeps the folds
    nxy = (ny, nx)
    G = V.reshape(ny, nx, 3)
    L = G.copy()
    L[1:-1, 1:-1] = (G[:-2, 1:-1] + G[2:, 1:-1] + G[1:-1, :-2] + G[1:-1, 2:]) / 4
    V = (G + 0.35 * (L - G)).reshape(-1, 3)
    log(f'tarp cloth sim: {len(V)} verts, {time.time() - t0:.1f}s, z {V[:, 2].min():.3f}..{V[:, 2].max():.3f}')
    rest = np.stack([X, Y], -1).reshape(-1, 2)
    for o in [cl, gnd]:
        bpy.data.objects.remove(o)
    return V, F, rest


def _tarp_puddles(tarp_obj, res=0.015):
    """Priority-flood the tarp's top surface: returns (V, F) of the flat water surfaces in its pockets and a per-vertex
    0..1 'puddle' mask (depth-weighted) for the silt/wet stains in the tarp material."""
    import heapq
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(tarp_obj, dg)
    Vt = np.array([v.co[:] for v in tarp_obj.data.vertices])
    x0, x1 = Vt[:, 0].min() - res, Vt[:, 0].max() + res
    y0, y1 = Vt[:, 1].min() - res, Vt[:, 1].max() + res
    nx, ny = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    H = np.full((ny, nx), -1.0)
    up = Vector((0, 0, -1))
    for i in range(ny):
        for j in range(nx):
            hit = bvh.ray_cast(Vector((x0 + j * res, y0 + i * res, 3.0)), up, 5.0)
            if hit[0] is not None:
                H[i, j] = hit[0].z
    valid = H > 0.02            # the tarp lying flat on the ground drains through the gravel at its edge
    W = np.where(valid, np.inf, H)
    done = ~valid
    pq = []
    for i in range(ny):
        for j in range(nx):
            border = (i in (0, ny - 1)) or (j in (0, nx - 1))
            if valid[i, j] and (border or any((0 <= i + a < ny and 0 <= j + b < nx and not valid[i + a, j + b])
                                              for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))):
                W[i, j] = H[i, j]
                done[i, j] = True
                heapq.heappush(pq, (H[i, j], i, j))
    while pq:
        h, i, j = heapq.heappop(pq)
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + a, j + b
            if 0 <= ii < ny and 0 <= jj < nx and not done[ii, jj]:
                done[ii, jj] = True
                W[ii, jj] = max(H[ii, jj], h)
                heapq.heappush(pq, (W[ii, jj], ii, jj))
    D = np.where(valid, W - H, 0.0)
    wet = D > 0.004
    # drop false pools: 'pockets' under a fold that overhangs (the downward ray sees a gap below the top layer) show
    # up as height jumps; a real tarp pocket has a continuous bottom and holds at most a few cm before it stretches
    jump = np.zeros_like(wet)
    for a, b in ((1, 0), (0, 1)):
        dh = np.abs(np.diff(H, axis=0 if a else 1))
        bad = dh > 0.035
        if a:
            jump[1:, :] |= bad
            jump[:-1, :] |= bad
        else:
            jump[:, 1:] |= bad
            jump[:, :-1] |= bad
    lab = np.zeros(wet.shape, np.int32)
    comp = 0
    keep = np.zeros_like(wet)
    for i in range(ny):
        for j in range(nx):
            if wet[i, j] and not lab[i, j]:
                comp += 1
                stack, cells = [(i, j)], []
                lab[i, j] = comp
                while stack:
                    ci, cj = stack.pop()
                    cells.append((ci, cj))
                    for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ii, jj = ci + a, cj + b
                        if 0 <= ii < ny and 0 <= jj < nx and wet[ii, jj] and not lab[ii, jj]:
                            lab[ii, jj] = comp
                            stack.append((ii, jj))
                ci_, cj_ = np.array(cells).T
                ok = (not jump[ci_, cj_].any()) and D[ci_, cj_].max() < 0.07 and len(cells) >= 12
                if ok:
                    keep[ci_, cj_] = True
    wet = keep
    D = np.where(wet, D, 0.0)
    # shorelines by marching triangles: each wet triangle of the grid is clipped where the tarp rises above its
    # pool level, so the water edge follows the tarp surface instead of the grid
    V, F = [], []

    def clip(poly, f):
        out = []
        n = len(poly)
        for k in range(n):
            p0, p1, f0, f1 = poly[k], poly[(k + 1) % n], f[k], f[(k + 1) % n]
            if f0 >= 0:
                out.append(p0)
            if (f0 >= 0) != (f1 >= 0):
                t = f0 / (f0 - f1)
                out.append((p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t))
        return out
    for i in range(ny - 1):
        for j in range(nx - 1):
            corners = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            if not any(wet[c] for c in corners):
                continue
            for tri in ((0, 1, 2), (0, 2, 3)):
                cs = [corners[k] for k in tri]
                ws = [W[c] for c in cs if wet[c]]
                if not ws or any(H[c] < 0 for c in cs):
                    continue
                L = max(ws)
                f = [L - H[c] for c in cs]
                poly = clip([(x0 + c[1] * res, y0 + c[0] * res) for c in cs], f)
                if len(poly) >= 3:
                    base = len(V)
                    V.extend([(px, py, L) for px, py in poly])
                    F.append(list(range(base, base + len(poly))))
    # per-vertex stain mask: distance-blurred depth sampled at the tarp vertices
    Dm = np.clip(D / 0.02, 0, 1)
    for _ in range(4):
        p = np.pad(Dm, 1, mode='edge')
        Dm = np.maximum(Dm, 0.8 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4)
    jj = np.clip(((Vt[:, 0] - x0) / res).round().astype(int), 0, nx - 1)
    ii = np.clip(((Vt[:, 1] - y0) / res).round().astype(int), 0, ny - 1)
    stain = Dm[ii, jj] * (np.abs(Vt[:, 2] - H[ii, jj]) < 0.03)
    log(f'tarp puddles: {int(wet.sum())} cells ({wet.sum() * res * res:.2f} m2), max depth {D.max() * 100:.1f} cm')
    return np.array(V), F, stain


def build_tarp_pile():
    mat = M_tarp('m_tarp', (40, 92, 152), seed=1)
    supports = _tarp_supports()
    V, F, rest = _tarp_cloth(supports)
    for o in supports:
        bpy.data.objects.remove(o)
    # fine crinkle (the woven PE never lies smooth): a few mm along the normals, stronger on the slopes
    mb = MB()
    mb.add(V, F, 0, recalc=False)
    obj = mb.build('tarp_pile', [mat], sharp=70)
    me = obj.data
    nrm = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get('normal', nrm)
    nrm = nrm.reshape(-1, 3)
    rx, ry = rest[:, 0], rest[:, 1]
    # rest (flat-sheet) coordinates drive the weave, fold-memory creases, hem and grommets in the material
    at = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    at.data.foreach_set('vector', np.concatenate([rest, np.zeros((len(rest), 1))], 1).ravel())
    bpy.context.view_layer.update()
    # pools are found on the smooth draped sheet (before the crinkle): a periodic crinkle made every shoreline
    # saw-toothed. Under standing water the film is pressed flat against the pocket bottom, so no crinkle there.
    WV, WF, stain = _tarp_puddles(obj)
    # fine crinkle (woven PE never lies smooth): a few mm along the normals, irregular (random plane waves at
    # 14-32 cm, >= 4 grid spacings so it cannot alias), strongest on the slopes; the normal map carries finer wrinkles
    crng = np.random.default_rng(77)
    cr = np.zeros(len(rx))
    for _ in range(7):
        a, lam, ph = crng.uniform(0, np.pi), crng.uniform(0.14, 0.32), crng.uniform(0, 2 * np.pi)
        cr += np.sin((rx * np.cos(a) + ry * np.sin(a)) * (2 * np.pi / lam) + ph)
    cr *= 0.0065 / math.sqrt(7)
    cr *= np.clip(1.3 - np.abs(nrm[:, 2]), 0.45, 1.0)
    # fold memory: the sheet was folded in quarters x thirds in its packet; the fold lines stay as crisp
    # alternating mountain / valley ridges (~1 cm, 5-7 cm wide) for its whole life
    def ridge(v, period, off):
        f = (v / period + off) % 1.0
        dist = np.abs(f - 0.5) * period
        k = np.floor(v / period + off)
        sgn = np.where(k % 2 == 0, 1.0, -1.0)
        return sgn * np.clip(1 - dist / 0.035, 0, 1)
    fm = 0.009 * (ridge(rx, TARP_W / 4, 0.5) + 0.8 * ridge(ry, TARP_D / 3, 0.5))
    fm *= 0.6 + 0.4 * np.sin(rx * 3.1 + ry * 2.3)       # partly relaxed along its length
    cr = (cr + fm) * (1.0 - np.clip(stain * 1.5, 0, 1))
    V2 = V + nrm * cr[:, None]
    me.vertices.foreach_set('co', V2.ravel())
    me.update()
    ap = me.attributes.new('pud', 'FLOAT', 'POINT')
    ap.data.foreach_set('value', stain.astype(np.float64))
    obj['tarp_grid'] = (int(round(TARP_W / TARP_DX)) + 1, int(round(TARP_D / TARP_DX)) + 1)
    out = [finish_obj(obj)]
    if len(WF):
        wm = bpy.data.materials.get('tarp_water') or bpy.data.materials.new('tarp_water')
        wm.use_nodes = True
        bs = wm.node_tree.nodes['Principled BSDF']
        bs.inputs['Base Color'].default_value = (0.012, 0.015, 0.018, 1)
        bs.inputs['Roughness'].default_value = 0.03
        bs.inputs['IOR'].default_value = 1.33
        wme = bpy.data.meshes.new('tarp_water')
        wme.from_pydata(WV.tolist(), [], WF)
        wme.update()
        bmw = bmesh.new()
        bmw.from_mesh(wme)
        bmesh.ops.remove_doubles(bmw, verts=bmw.verts, dist=1e-5)
        bmesh.ops.dissolve_limit(bmw, angle_limit=math.radians(1), verts=bmw.verts, edges=bmw.edges)
        bmesh.ops.triangulate(bmw, faces=bmw.faces)
        bmw.to_mesh(wme)
        bmw.free()
        wme.materials.append(wm)
        wo = link(bpy.data.objects.new('tarp_water', wme))
        wo['no_atlas'] = True
        out.append(wo)
        log(f'tarp water: {len(wme.polygons)} tris')
    return out


# ---------------------------------------------------------------- roadside furniture (Alpine)
def stroke(pts, w, closed=False):
    """Thick polyline as quads + round joints (for rasterised lettering)."""
    P = np.asarray(pts, float)
    out = []
    n = len(P)
    for i in range(n if closed else n - 1):
        out.append(thick_line(P[i], P[(i + 1) % n], w))
    for p_ in P:
        out.append(circle(p_[0], p_[1], w / 2, 12))
    return out


def arc(cx, cy, r, a0, a1, n=24):
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], -1)


def glyph(ch, x, y, s=1.0, w=0.021):
    """Simple sans digits (0.17 m tall at s=1), centred at (x, y)."""
    T_ = lambda P: np.asarray(P) * s + np.array([x, y])
    if ch == '3':
        return stroke(T_(np.concatenate([arc(0, 0.042, 0.038, 155, -90), arc(0, -0.042, 0.043, 90, -160)])), w)
    if ch == '8':
        return stroke(T_(arc(0, 0.044, 0.034, 0, 360, 40)), w, True) + stroke(T_(arc(0, -0.041, 0.042, 0, 360, 44)), w, True)
    if ch == '1':
        return stroke(T_([(-0.025, 0.055), (0.008, 0.085), (0.008, -0.085)]), w)
    return []


KM_EXT = (-0.16, 0.16, -0.22, 0.78)


def km_image(text='38', res=(384, 1152)):
    W, H = res
    x0, x1, z0, z1 = KM_EXT
    img = np.zeros((H, W, 3))
    img[:] = np.array([0.86, 0.86, 0.83])                     # white paint
    zz = np.linspace(z1, z0, H)[:, None] * np.ones((1, W))
    img[zz > 0.56] = np.array([0.62, 0.1, 0.08])              # red top (state road)
    polys = []
    for k, ch in enumerate(text):
        polys += glyph(ch, (k - (len(text) - 1) / 2) * 0.1, 0.33)
    c = raster(W, H, polys, KM_EXT, 2)
    img = img * (1 - c[..., None]) + np.array([0.05, 0.05, 0.05]) * c[..., None]
    return img


def M_painted_concrete(name, image, extent, *, seed=0):
    """Concrete post painted from a planar image (x, z): chalky faded paint, chips at the edges showing concrete,
    grime streaks from the top, green algae and moss at the foot, mud splash."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed * 3.1, seed * 1.3, 0))
    x, y, z = B.sep(B.P)
    x0, x1, z0, z1 = extent
    uv = B.comb(B.mr(x, x0, x1, 0, 1, clamp=False), B.mr(z, z0, z1, 0, 1, clamp=False), 0)
    pc = B.img(None, uv, proj='FLAT', ext='EXTEND', image=image)
    pc = B.hsv(pc, 0.5, 0.85, 0.95)
    pv = B.noise(P, 4.0, 4)
    pc = B.mix(B.mr(pv, 0.3, 0.7), B.hsv(pc, 0.5, 0.95, 0.9), B.hsv(pc, 0.5, 1.0, 1.04))
    con = B.hsv(B.tri('concrete_wall_006', 'diffuse', P, 1.2), 0.5, 0.6, 1.0)
    e = B.edge(0.01, lo=0.05, hi=0.3, samples=8)
    fl = B.noise(P, 9.0, 7, 0.68)
    wear = B.maxf(B.mul(e, B.mr(fl, 0.3, 0.6, 0.5, 1.6)), B.mr(fl, 0.64, 0.68))
    wear = B.mr(wear, 0.5, 0.56)
    col = B.mix(wear, pc, con)
    st = B.noise(B.mapping(P, scale=(24, 24, 1.3)), 1.0, 4, 0.6)
    stm = B.mul(B.mr(st, 0.5, 0.7), B.mr(z, 0.1, 0.7))
    col = B.mix(B.mul(stm, 0.55), col, B.hsv(col, 0.5, 0.7, 0.6))
    cav = B.cavity(0.03)
    col = B.mix(B.mul(cav, 0.7), col, srgb((45, 42, 37)))
    alg = B.mul(B.mr(B.noise(P, 2.5, 5), 0.45, 0.7), B.mr(z, 0.25, 0.0))
    col = B.mix(B.mul(alg, 0.75), col, srgb((52, 64, 32)))
    mudt = B.hsv(B.tri('brown_mud_rocks_01', 'diffuse', P, 0.8), 0.5, 0.85, 0.55)
    mz = B.mr(B.add(z, B.mul(B.sub(B.noise(P, 8.0, 5, 0.7), 0.5), 0.1)), 0.1, 0.0)
    col = B.mix(mz, col, mudt)
    rough = B.mixf(wear, 0.55, 0.9)
    rough = B.mixf(B.maxf(mz, alg), rough, 0.95)
    hc = B.tri('concrete_wall_006', 'displacement', P, 1.2, noncolor=True)
    nrm = B.bevel(0.008, 8)
    nrm = B.bump(B.sep(hc)[0], 0.6, 0.003, nrm)
    nrm = B.bump(B.inv(wear), 0.5, 0.0005, nrm)
    return B.finish(col, rough, 0.0, nrm)


def build_km_post():
    """Kilometre stone ('cippo') as on Alpine state roads: painted concrete, 0.30 x 0.16 m, 0.75 m above the
    ground with a rounded red top; the number on both faces. Faces toward Blender -Y / +Y (three +Z / -Z)."""
    arr = km_image('38')
    im = np_image('km_post_face', arr, os.path.join(BAKE, '_km_post_face.png'), noncolor=False)
    mat = M_painted_concrete('m_km_post', im, KM_EXT, seed=4)
    W_, D_, Z0, Zs = 0.30, 0.16, -0.22, 0.6
    prof = [(-W_ / 2, Z0), (W_ / 2, Z0), (W_ / 2, Zs)]
    prof += [(W_ / 2 * math.cos(a), Zs + W_ / 2 * 0.95 * math.sin(a)) for a in np.linspace(0.12, math.pi - 0.12, 12)]
    prof += [(-W_ / 2, Zs)]
    prof = np.array(prof)
    m = len(prof)
    V = []
    for yy in (-D_ / 2, D_ / 2):
        for p_ in prof:
            V.append((p_[0], yy, p_[1]))
    V = np.array(V)
    F = [list(range(m))[::-1], list(range(m, 2 * m))]
    F += [[j, (j + 1) % m, m + (j + 1) % m, m + j] for j in range(m)]
    mb = MB()
    mb.add(V, F, 0)
    bm = bmesh.new()
    bm.from_mesh(mb.build('km_tmp', [mat], sharp=30).data)
    bpy.data.objects.remove(bpy.data.objects['km_tmp'])
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.008, segments=2, affect='EDGES', clamp_overlap=True)
    mb2 = MB()
    mb2.add_bm(bm, 0)
    obj = mb2.build('km_post', [mat], sharp=35)
    return [finish_obj(obj)]


def build_snow_pole():
    """Winter snow pole (Schneestange): 32 mm orange GRP rod, 2.0 m above ground, black tip, two reflective bands;
    UV-faded, road-spray grime on the lower half."""
    mat = M_plastic('m_snowpole', (222, 92, 30), rough=0.38, dirt=0.55, scuff=0.4, fade=0.5, mud=0.5, mud_h=0.25,
                    bands=[(1.5, 1.58), (1.64, 1.72)], black=(1.8, 2.05), seed=11)
    mb = MB()
    V, F = lathe([(0.016, -0.35), (0.016, 1.96), (0.013, 1.99), (0.006, 2.005)], 10)
    V[:, 0] += 0.012 * (V[:, 2] / 2.0) ** 2        # a slight bow
    mb.add(V, F, 0)
    obj = mb.build('snow_pole', [mat], sharp=50)
    return [finish_obj(obj)]


def M_concrete(name, *, seed=0, moss=0.6):
    B = NB(name)
    P = B.mapping(B.P, loc=(seed * 2.3, seed * 1.1, 0))
    x, y, z = B.sep(B.P)
    con = B.hsv(B.tri('concrete_wall_006', 'diffuse', P, 1.1), 0.5, 0.55, 0.85)
    con = B.mix(B.mr(B.noise(P, 1.3, 3), 0.3, 0.7), B.hsv(con, 0.5, 1, 0.8), B.hsv(con, 0.5, 1, 1.08))
    cav = B.cavity(0.04)
    col = B.mix(B.mul(cav, 0.8), con, srgb((40, 38, 33)))
    mst = B.mul(B.mr(B.noise(P, 3.0, 5), 0.5, 0.68), moss)
    col = B.mix(mst, col, srgb((45, 56, 26)))
    mudt = B.hsv(B.tri('brown_mud_rocks_01', 'diffuse', P, 0.7), 0.5, 0.85, 0.5)
    mm_ = B.mr(B.noise(P, 5.0, 5, 0.7), 0.45, 0.62)
    col = B.mix(B.mul(mm_, 0.7), col, mudt)
    rough = B.mixf(B.maxf(mst, mm_), 0.8, 0.95)
    hc = B.tri('concrete_wall_006', 'displacement', P, 1.1, noncolor=True)
    nrm = B.bevel(0.01, 8)
    nrm = B.bump(B.sep(hc)[0], 0.8, 0.004, nrm)
    return B.finish(col, rough, 0.0, nrm)


def M_litter(name, *, seed=0):
    """Wet spruce needles, leaf fragments and silt matted together."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed, seed * 0.5, 0))
    ndl = B.vor(B.mapping(P, rot=(0, 0, 30), scale=(1, 9, 1)), 70.0, 'F1')
    ndl2 = B.vor(B.mapping(P, rot=(0, 0, -50), scale=(1, 9, 1)), 60.0, 'F1')
    n_ = B.maxf(B.mr(ndl, 0.12, 0.0), B.mr(ndl2, 0.12, 0.0))
    base = B.mix(B.mr(B.noise(P, 20.0, 4), 0.3, 0.7), srgb((34, 26, 16)), srgb((70, 50, 28)))
    col = B.mix(n_, base, srgb((96, 64, 34)))
    leaf = B.mr(B.noise(P, 9.0, 3), 0.62, 0.66)
    col = B.mix(B.mul(leaf, 0.8), col, srgb((110, 78, 36)))
    silt = B.mr(B.noise(P, 5.0, 4), 0.55, 0.7)
    col = B.mix(B.mul(silt, 0.6), col, srgb((80, 70, 56)))
    r = B.mixf(silt, 0.45, 0.8)
    nrm = B.bump(n_, 0.6, 0.0015)
    return B.finish(col, r, 0.0, nrm)


def build_culvert_grate():
    """Ditch inlet of a road culvert: precast concrete frame (0.60 x 0.95 m, long side along Blender Y = along the
    ditch), a rusty bar grating over a dark shaft, twigs and a wad of needles/leaves caught on the bars.
    Origin = grating top centre."""
    con = M_concrete('m_culvert_con', seed=3, moss=0.7)
    grate = M_galv('m_culvert_grate', rust=0.95, grime=0.85, seed=13, spray_h=0.1, white_rust=0.2)
    dark = M_simple('m_culvert_dark', (28, 27, 25), rough=0.55, noise=0.15, seed=2)
    twig = M_simple('m_culvert_twig', (62, 48, 34), rough=0.8, noise=0.25, seed=5)
    litter = M_litter('m_culvert_litter', seed=6)
    mb = MB()
    OX, OY, IX, IY = 0.60, 0.95, 0.42, 0.77
    zt, zb = 0.03, -0.16
    fw = (OX - IX) / 2
    for cx, cy, sx, sy in ((0, (OY - fw) / 2, OX, fw), (0, -(OY - fw) / 2, OX, fw), ((OX - fw) / 2, 0, fw, IY), (-(OX - fw) / 2, 0, fw, IY)):
        mb.add_bm(rbox(sx, sy, zt - zb, r=0.012, segs=2, center=(cx, cy, (zt + zb) / 2)), 0)
    # shaft walls (inward) + floor with a little standing water
    for cx, cy, sx, sy in ((0, IY / 2 + 0.01, IX, 0.02), (0, -IY / 2 - 0.01, IX, 0.02), (IX / 2 + 0.01, 0, 0.02, IY), (-IX / 2 - 0.01, 0, 0.02, IY)):
        mb.add_bm(rbox(sx, sy, 0.62, center=(cx, cy, zb - 0.31 + 0.02)), 2)
    mb.add_bm(rbox(IX, IY, 0.02, center=(0, 0, zb - 0.6)), 2)
    # the dark of the shaft seen through the bars (terrain is always below this in the engine)
    mb.add(np.array([(-IX / 2, -IY / 2, -0.034), (IX / 2, -IY / 2, -0.034), (IX / 2, IY / 2, -0.034), (-IX / 2, IY / 2, -0.034)]), [[0, 1, 2, 3]], 2, recalc=False)
    # grating: bearing bars across X every 33 mm, angle rim, two tie rods
    nb = int(IY / 0.033)
    for k in range(nb):
        yb = -IY / 2 + 0.02 + k * (IY - 0.04) / (nb - 1)
        mb.add_bm(rbox(IX - 0.02, 0.005, 0.03, center=(0, yb, -0.012)), 1)
    for xr in (-0.12, 0.12):
        V, F = cyl(0.004, IY - 0.04, 6)
        mb.add(V, F, 1, M=mm(T(xr, -(IY - 0.04) / 2, -0.012), R('X', -90)))
    for cx, cy, sx, sy in ((0, IY / 2 - 0.012, IX, 0.024), (0, -IY / 2 + 0.012, IX, 0.024), (IX / 2 - 0.012, 0, 0.024, IY), (-IX / 2 + 0.012, 0, 0.024, IY)):
        mb.add_bm(rbox(sx, sy, 0.03, center=(cx, cy, -0.012)), 1)
    rng = np.random.default_rng(21)
    for k in range(6):
        a = rng.random() * math.pi
        L_ = 0.18 + 0.35 * rng.random()
        c0 = np.array([rng.normal(0, 0.12), rng.normal(0, 0.28), 0.008])
        d_ = np.array([math.cos(a), math.sin(a), 0.0]) * L_ / 2
        mid = c0 + np.array([0, 0, 0.01 * rng.random()])
        V, F = tube([c0 - d_, mid + np.array([0.01, -0.01, 0.006]), c0 + d_], 0.004 + 0.006 * rng.random(), 5, 8)
        mb.add(V, F, 3)
    # needle/leaf wad against the upstream end
    wad = []
    for i in range(10):
        a = i / 10 * 2 * math.pi
        wad.append((0.16 * math.cos(a) * (0.8 + 0.4 * rng.random()), 0.3 + 0.07 * math.sin(a) * (0.8 + 0.4 * rng.random())))
    bm = bmesh.new()
    vs = [bm.verts.new((wx, wy, 0.004)) for wx, wy in wad]
    fc = bm.faces.new(vs)
    bmesh.ops.poke(bm, faces=[fc])
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=3, use_grid_fill=True)
    for v in bm.verts:
        r_ = math.hypot(v.co.x, (v.co.y - 0.3) * 2.2)
        v.co.z = 0.004 + 0.035 * max(0, 1 - r_ / 0.17) ** 0.7 + 0.012 * rng.random() * max(0, 1 - r_ / 0.2)
        v.co.x += 0.008 * (rng.random() - 0.5)
        v.co.y += 0.008 * (rng.random() - 0.5)
    mb.add_bm(bm, 4)
    obj = mb.build('culvert_grate', [con, grate, dark, twig, litter], sharp=40)
    return [finish_obj(obj, {'1': 1.3, '2': 0.5, '4': 0.6})]


ROUND_R = 0.3


def old_sign_image(kind, res=1024):
    x0, x1, z0, z1 = SIGN_EXT
    ext = (x0, x1, z0 - SIGN_CZ, z1 - SIGN_CZ)
    img = np.zeros((res, res, 3))
    img[:] = np.array([0.6, 0.6, 0.6])
    white = np.array([0.9, 0.9, 0.87])
    black = np.array([0.07, 0.07, 0.075])
    if kind == 'bend':
        outer = rounded_triangle(SIGN_SIDE, 0.045, 0.0)
        c_out = raster(res, res, [outer], ext, 2)
        c_red = raster(res, res, [inset_poly(outer, 0.010)], ext, 2)
        c_fld = raster(res, res, [inset_poly(outer, 0.075)], ext, 2)
        img = img * (1 - c_out[..., None]) + white * c_out[..., None]
        img = img * (1 - c_red[..., None]) + np.array([0.74, 0.1, 0.11]) * c_red[..., None]
        img = img * (1 - c_fld[..., None]) + white * c_fld[..., None]
        path = np.concatenate([np.array([[0.06, -0.26]]), arc(-0.03, -0.06, 0.12, -40, 60, 14), arc(0.1, 0.1, 0.09, 230, 160, 8)])
        polys = stroke(path, 0.045)
        tip = path[-1]
        polys.append(np.array([tip + (-0.075, -0.01), tip + (0.03, 0.07), tip + (0.035, -0.06)]))
        c_pic = raster(res, res, polys, ext) * c_fld
        img = img * (1 - c_pic[..., None]) + black * c_pic[..., None]
        outline = outer
    else:
        outer = circle(0, 0, ROUND_R, 96)
        c_out = raster(res, res, [outer], ext, 2)
        c_blu = raster(res, res, [circle(0, 0, ROUND_R - 0.012, 96)], ext, 2)
        img = img * (1 - c_out[..., None]) + white * c_out[..., None]
        img = img * (1 - c_blu[..., None]) + np.array([0.08, 0.26, 0.58]) * c_blu[..., None]
        ring = stroke(arc(0, 0, 0.15, 0, 360, 60), 0.075, True)
        c_t = raster(res, res, ring, ext)
        img = img * (1 - c_t[..., None]) + white * c_t[..., None]
        links = []
        for k in range(10):
            a = 2 * math.pi * k / 10
            ca, sa = math.cos(a), math.sin(a)
            p0 = np.array([ca, sa]) * 0.105
            p1 = np.array([ca, sa]) * 0.195
            q = np.array([-sa, ca]) * 0.025
            links.append(thick_line(p0 - q, p1 + q, 0.016))
        c_l = raster(res, res, links, ext) * c_t
        img = img * (1 - c_l[..., None]) + np.array([0.08, 0.26, 0.58]) * c_l[..., None]
        outline = outer
    # edge mask (rust creeps in from the rim): coverage minus its blur
    cov = raster(res // 4, res // 4, [outline], ext, 2)
    b = cov.copy()
    for _ in range(6):
        p_ = np.pad(b, 1, mode='edge')
        b = (p_[:-2, 1:-1] + p_[2:, 1:-1] + p_[1:-1, :-2] + p_[1:-1, 2:] + p_[1:-1, 1:-1]) / 5
    edge = np.clip((cov - b) * 4.0, 0, 1)
    return img, edge


def M_sign_face_old(name, image, edge_img, extent, *, seed=0):
    """Old enamelled/painted sign: colours faded and chalky, grime streaks, rust bleeding in from the rim and from
    the two bolt holes, blistered paint with rust underneath."""
    B = NB(name)
    P = B.mapping(B.P, loc=(seed, 0, 0))
    x, y, z = B.sep(B.P)
    x0, x1, z0, z1 = extent
    uv = B.comb(B.mr(x, x0, x1, 0, 1, clamp=False), B.mr(z, z0, z1, 0, 1, clamp=False), 0)
    face = B.img(None, uv, proj='FLAT', ext='EXTEND', image=image)
    edge = B.sep(B.img(None, uv, proj='FLAT', ext='EXTEND', image=edge_img, noncolor=True))[0]
    col = B.hsv(face, 0.5, 0.5, 0.82)
    col = B.mix(B.mul(B.mr(B.noise(P, 3.0, 4), 0.35, 0.7), 0.5), col, B.hsv(col, 0.5, 0.5, 1.15))    # chalky patches
    st = B.noise(B.mapping(P, scale=(30, 30, 1.4)), 1.0, 4, 0.6)
    stm = B.mul(B.mr(st, 0.45, 0.7), 0.7)
    col = B.mix(stm, col, B.hsv(col, 0.5, 0.7, 0.55))
    rust_t = B.tri('rust_coarse_01', 'diffuse', P, 0.35)
    rn = B.noise(P, 11.0, 6, 0.7)
    rim = B.mul(B.add(edge, 0.25), B.mr(rn, 0.25, 0.55, 0.4, 1.6))
    bolts = 0.0
    for bz in (SIGN_CZ + 0.14, SIGN_CZ - 0.16):
        dz = B.sub(z, bz)
        r_ = B.math('SQRT', B.add(B.mul(x, x), B.mul(dz, dz)))
        run = B.mul(B.mr(B.absf(x), 0.03, 0.008), B.mr(dz, 0.0, -0.22))
        bolts = B.maxf(bolts, B.maxf(B.mr(r_, 0.05, 0.015), B.mul(run, B.mr(B.noise(B.mapping(P, scale=(50, 50, 3)), 1.0, 3), 0.4, 0.6))))
    blis = B.maxf(B.mr(B.noise(P, 22.0, 6, 0.7), 0.64, 0.68), B.mul(B.mr(B.noise(P, 6.0, 5, 0.65), 0.6, 0.66), 0.9))
    rm = B.maxf(B.maxf(B.mr(rim, 0.35, 0.6), bolts), blis)
    col = B.mix(B.mul(rm, 0.95), col, B.hsv(rust_t, 0.5, 1.0, 0.8))
    bleed = B.mul(B.mr(B.maxf(rim, bolts), 0.15, 0.4), B.mr(B.noise(B.mapping(P, scale=(40, 40, 2)), 1.0, 3), 0.45, 0.65))
    col = B.mix(B.mul(bleed, 0.55), col, B.hsv(rust_t, 0.5, 0.8, 0.95))
    r = B.mixf(rm, B.mixf(stm, 0.45, 0.65), 0.85)
    nrm = B.bump(B.noise(P, 3.0, 2), 0.3, 0.003)
    nrm = B.bump(B.add(blis, B.mul(rm, B.noise(P, 60.0, 4))), 0.6, 0.0006, nrm)
    return B.finish(col, r, 0.0, nrm)


def build_sign_old(kind):
    """Old, rusty roadside sign on a leaning galvanised post: 'bend' (dangerous bend to the left, triangle) or
    'chains' (snow chains compulsory, 0.6 m blue disc). Faces toward Blender -Y (three +Z)."""
    name = 'sign_' + kind
    img, edge = old_sign_image(kind)
    im = np_image(name + '_face', img, os.path.join(BAKE, f'_{name}_face.png'), noncolor=False)
    ie = np_image(name + '_edge', edge, os.path.join(BAKE, f'_{name}_edge.png'), noncolor=True)
    face = M_sign_face_old('m_' + name + '_face', im, ie, SIGN_EXT, seed=7 if kind == 'bend' else 8)
    back = M_galv('m_' + name + '_back', rust=0.8, grime=0.7, seed=17, spray_h=0.3, white_rust=0.3)
    galv = M_galv('m_' + name + '_post', rust=0.75, grime=0.6, seed=19, spray_h=0.6, white_rust=0.4)
    mb = MB()
    outer = rounded_triangle(SIGN_SIDE, 0.045, 0.0) if kind == 'bend' else circle(0, 0, ROUND_R, 64)
    n = len(outer)
    yF, yB, yL = -0.0012, 0.0012, 0.02
    mb.add(np.array([(p_[0], yF, p_[1] + SIGN_CZ) for p_ in outer]), [list(range(n))], 0, recalc=False)
    mb.add(np.array([(p_[0], yB, p_[1] + SIGN_CZ) for p_ in outer]), [list(range(n))[::-1]], 1, recalc=False)
    inner = inset_poly(outer, 0.012) if kind == 'bend' else circle(0, 0, ROUND_R - 0.012, 64)
    V = []
    for P_, yy in ((outer, yF), (outer, yL), (inner, yL), (inner, yB)):
        for p_ in P_:
            V.append((p_[0], yy, p_[1] + SIGN_CZ))
    F = []
    for ring in range(3):
        for j in range(n):
            F.append([ring * n + j, ring * n + (j + 1) % n, (ring + 1) * n + (j + 1) % n, (ring + 1) * n + j][::-1])
    mb.add(np.array(V), F, 1, recalc=False)
    ypost = 0.05
    V, F = cyl(0.0305, 2.62, 20, z0=-0.45)
    mb.add(V, F, 2, M=T(0, ypost, 0))
    for zr in (SIGN_CZ - 0.16, SIGN_CZ + 0.14):
        half = 0.26 if kind == 'chains' else (0.30 if zr < SIGN_CZ else 0.14)
        mb.add_bm(rbox(2 * half, 0.012, 0.04, r=0.002, segs=1, center=(0, 0.008, zr)), 2)
        pts = [(-0.037, 0.015, zr), (-0.037, ypost + 0.01, zr), (-0.02, ypost + 0.034, zr), (0.0, ypost + 0.04, zr),
               (0.02, ypost + 0.034, zr), (0.037, ypost + 0.01, zr), (0.037, 0.015, zr)]
        V, F = tube(pts, 0.0045, 6, 16)
        mb.add(V, F, 2)
    obj = mb.build(name, [face, back, galv], sharp=40)
    return [finish_obj(obj, {'0': 1.6})]


BUILDERS = {
    'jerrycan': build_jerrycan, 'hatchet': build_hatchet, 'plank': build_plank, 'bearer': build_bearer,
    'plank_wedge': build_plank_wedge, 'crate': build_crate, 'toolbox': build_toolbox, 'barrier': build_barrier,
    'cone': build_cone, 'delineator': build_delineator, 'rail_reflector': build_rail_reflector,
    'sign_rockfall': lambda: build_sign('rockfall'), 'sign_roadworks': lambda: build_sign('roadworks'),
    'guardrail': build_guardrail, 'guardrail_end': build_guardrail_end, 'light_tower': build_light_tower,
    'sandbag': build_sandbag, 'tarp_pile': build_tarp_pile,
    'km_post': build_km_post, 'snow_pole': build_snow_pole, 'culvert_grate': build_culvert_grate,
    'sign_bend': lambda: build_sign_old('bend'), 'sign_chains': lambda: build_sign_old('chains'),
}


# ================================================================================================
# UV atlas + bake
# ================================================================================================
def select_only(objs, active=None):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


def unwrap_atlas(objs, margin=0.006):
    for o in objs:
        select_only([o])
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(52), island_margin=0.0, area_weight=0.0, scale_to_bounds=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        me = o.data
        uvl = me.uv_layers.active.data
        uvw = json.loads(o.get('uvw', '{}'))
        # density normalise per material slot: sqrt(3D area / UV area) * weight
        n = len(me.polygons)
        area = np.zeros(n)
        me.polygons.foreach_get('area', area)
        mi = np.zeros(n, np.int32)
        me.polygons.foreach_get('material_index', mi)
        uv = np.zeros(len(uvl) * 2)
        uvl.foreach_get('uv', uv)
        uv = uv.reshape(-1, 2)
        ls = np.zeros(n, np.int32)
        lt = np.zeros(n, np.int32)
        me.polygons.foreach_get('loop_start', ls)
        me.polygons.foreach_get('loop_total', lt)
        uva = np.zeros(n)
        for p in range(n):
            q = uv[ls[p]:ls[p] + lt[p]]
            uva[p] = 0.5 * abs(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
        for slot in set(mi.tolist()):
            sel = mi == slot
            k = math.sqrt(area[sel].sum() / max(uva[sel].sum(), 1e-12)) * float(uvw.get(str(slot), 1.0))
            for p in np.nonzero(sel)[0]:
                uv[ls[p]:ls[p] + lt[p]] *= k
        uvl.foreach_set('uv', uv.ravel())
    select_only(objs)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(rotate=True, margin=margin, scale=True, shape_method='CONCAVE')
    bpy.ops.object.mode_set(mode='OBJECT')


def setup_cycles(samples, gpu=True):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    if gpu:
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'METAL'
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type == 'METAL'
            sc.cycles.device = 'GPU'
        except Exception as e:
            log('GPU setup failed, CPU', e)
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.view_settings.view_transform = 'AgX'


def bake_atlas(atlas, objs, size, samples):
    t0 = time.time()
    sc = bpy.context.scene
    setup_cycles(samples)
    if sc.world is None:
        sc.world = bpy.data.worlds.new('bakeworld')
    sc.world.light_settings.distance = AO_DIST[atlas]
    sc.render.bake.margin = 10
    sc.render.bake.margin_type = 'EXTEND'
    ims = {}
    for k in ('albedo', 'data', 'normal', 'ao'):
        im = bpy.data.images.new(f'bk_{atlas}_{k}', size, size, alpha=False, float_buffer=True)
        im.colorspace_settings.name = 'Non-Color'
        ims[k] = im
    nbs = []
    for o in objs:
        for m in o.data.materials:
            nb = next((b for b in _MATS if b.m == m), None)
            if nb and nb not in nbs:
                nbs.append(nb)
    tnodes = []
    for nb in nbs:
        t = nb.nt.nodes.new('ShaderNodeTexImage')
        nb.nt.nodes.active = t
        tnodes.append(t)
    # spread the objects so they don't occlude each other in the AO pass
    x = 0.0
    locs = {}
    for o in objs:
        locs[o.name] = tuple(o.location)
        dims = o.dimensions
        o.location = (x + dims.x / 2, 0, 0)
        x += dims.x + 3.0
    bpy.context.view_layer.update()
    select_only(objs)
    for k in ('albedo', 'data', 'normal', 'ao'):
        for t in tnodes:
            t.image = ims[k]
        for nb in nbs:
            set_pass(nb, k)
        sc.cycles.samples = samples * (3 if k == 'ao' else 1)
        tp = {'albedo': 'EMIT', 'data': 'EMIT', 'normal': 'NORMAL', 'ao': 'AO'}[k]
        bpy.ops.object.bake(type=tp, margin=10, use_clear=True, normal_space='TANGENT', target='IMAGE_TEXTURES')
        log(f'  {atlas}.{k} baked')
    A = read_pixels(ims['albedo'])[..., :3]
    D = read_pixels(ims['data'])[..., :3]
    N = read_pixels(ims['normal'])[..., :3]
    O = read_pixels(ims['ao'])[..., 0]
    np_image(f'{atlas}_albedo', lin2srgb(A), os.path.join(BAKE, f'{atlas}_albedo.png'))
    np_image(f'{atlas}_normal', N, os.path.join(BAKE, f'{atlas}_normal.png'))
    orm = np.stack([np.clip(0.25 + 0.75 * O, 0, 1), np.clip(D[..., 0], 0.03, 1), np.clip(D[..., 1], 0, 1)], -1)
    np_image(f'{atlas}_orm', orm, os.path.join(BAKE, f'{atlas}_orm.png'))
    for t, nb in zip(tnodes, nbs):
        nb.nt.nodes.remove(t)
        set_pass(nb, 'normal')
    for im in ims.values():
        bpy.data.images.remove(im)
    for o in objs:
        o.location = locs[o.name]
    log(f'atlas {atlas}: {len(objs)} objects baked in {time.time() - t0:.0f}s')


# ================================================================================================
# post-bake assembly, export
# ================================================================================================
def final_material(atlas):
    name = 'props_' + atlas
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes['Principled BSDF']
    paths = {k: os.path.join(BAKE, f'{atlas}_{k}.png') for k in ('albedo', 'normal', 'orm')}
    if all(os.path.exists(p) for p in paths.values()):
        ia = nt.nodes.new('ShaderNodeTexImage')
        ia.image = load_image(paths['albedo'], False)
        nt.links.new(ia.outputs['Color'], bs.inputs['Base Color'])
        io = nt.nodes.new('ShaderNodeTexImage')
        io.image = load_image(paths['orm'], True)
        sp = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(io.outputs['Color'], sp.inputs[0])
        nt.links.new(sp.outputs[1], bs.inputs['Roughness'])
        nt.links.new(sp.outputs[2], bs.inputs['Metallic'])
        inn = nt.nodes.new('ShaderNodeTexImage')
        inn.image = load_image(paths['normal'], True)
        nm = nt.nodes.new('ShaderNodeNormalMap')
        nt.links.new(inn.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], bs.inputs['Normal'])
        # AO into the preview only via base colour multiply (glTF uses the ORM red channel)
        mx = nt.nodes.new('ShaderNodeMix')
        mx.data_type = 'RGBA'
        mx.blend_type = 'MULTIPLY'
        _in(mx, 'Factor_Float').default_value = 0.6
        nt.links.new(ia.outputs['Color'], _in(mx, 'A_Color'))
        nt.links.new(sp.outputs[0], _in(mx, 'B_Color'))
        nt.links.new(_out(mx, 'Result_Color'), bs.inputs['Base Color'])
    return m


def lamp_material():
    m = bpy.data.materials.get('lamp')
    if m:
        return m
    m = bpy.data.materials.new('lamp')
    m.use_nodes = True
    bs = m.node_tree.nodes['Principled BSDF']
    bs.inputs['Base Color'].default_value = (0.8, 0.78, 0.72, 1)
    bs.inputs['Roughness'].default_value = 0.15
    bs.inputs['Emission Color'].default_value = (1.0, 0.82, 0.6, 1)
    bs.inputs['Emission Strength'].default_value = 8.0
    return m


def apply_final_materials(o):
    if o.get('final') or o.get('no_atlas'):
        return
    o['final'] = True
    atlas = o.get('atlas')
    lens = o.get('lens_slot', -1)
    me = o.data
    n = len(me.polygons)
    mi = np.zeros(n, np.int32)
    me.polygons.foreach_get('material_index', mi)
    new_mi = np.where(mi == lens, 1, 0).astype(np.int32)
    me.materials.clear()
    me.materials.append(final_material(atlas))
    if lens >= 0:
        me.materials.append(lamp_material())
    me.polygons.foreach_set('material_index', new_mi)


def copy_obj(o, name):
    c = o.copy()
    c.data = o.data.copy()
    c.name = name
    c.data.name = name
    link(c)
    return c


def join(objs, name):
    select_only(objs)
    bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    o.name = name
    o.data.name = name
    return o


def assemble(objs):
    """Build derived nodes that reuse baked UVs: plank_stack, guardrail_bent. Returns the new objects."""
    out = []
    byname = {o.name: o for o in objs}
    if 'plank' in byname and 'bearer' in byname:
        rng = np.random.default_rng(31)
        parts = []
        for bx in (-1.25, 1.25):
            b = copy_obj(byname['bearer'], 'st_bearer')
            b.location = (bx + rng.normal(0, 0.03), rng.normal(0, 0.02), 0)
            b.rotation_euler = (0, 0, math.radians(rng.normal(0, 3)))
            parts.append(b)
        for layer in range(6):
            for col in (-1, 1):
                p = copy_obj(byname['plank'], 'st_plank')
                p.location = (rng.normal(0, 0.035), col * (PL_W / 2 + 0.006) + rng.normal(0, 0.006), 0.075 + layer * (PL_T + 0.0015))
                p.rotation_euler = (0, 0, math.radians(rng.normal(0, 0.8)) + (math.pi if rng.random() < 0.5 else 0))
                parts.append(p)
        bpy.context.view_layer.update()
        for p in parts:
            p.data.transform(p.matrix_world)
            p.matrix_world = Matrix.Identity(4)
        st = join(parts, 'plank_stack')
        st['atlas'] = 'wood'
        out.append(st)
    if 'guardrail' in byname:
        g = copy_obj(byname['guardrail'], 'guardrail_bent')
        g['atlas'] = 'rail'
        rng = json.loads(byname['guardrail'].get('beam_range', '[]'))
        me = g.data
        co = np.zeros(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3)
        beam = np.zeros(len(co), bool)
        for a, b in rng:
            beam[a:b] = True
        # impact at y=-2.1: pushed back and down, crests flattened, a sharp crease
        yy = co[:, 1]
        f = np.exp(-((yy + 2.1) / 0.75) ** 2) * beam
        crease = np.exp(-((yy + 2.25) / 0.12) ** 2) * beam
        rel = co[:, 0] - RAIL_X
        co[:, 0] -= 0.13 * f + 0.035 * crease
        co[:, 0] -= rel * 0.55 * f
        co[:, 2] -= 0.06 * f + 0.02 * crease
        co[:, 2] += (co[:, 2] - RAIL_Z) * -0.18 * f
        me.vertices.foreach_set('co', co.ravel())
        me.update()
        out.append(g)
    return out


EXPORT_ORDER = ['jerrycan', 'hatchet', 'plank', 'plank_stack', 'plank_wedge', 'barrier', 'cone', 'crate', 'toolbox',
                'light_tower', 'lamp_head', 'sign_rockfall', 'sign_roadworks', 'guardrail', 'guardrail_bent', 'guardrail_lod',
                'guardrail_end', 'delineator', 'rail_reflector', 'sandbag', 'tarp_pile', 'tarp_water',
                'km_post', 'snow_pole', 'culvert_grate', 'sign_bend', 'sign_chains']


PACK_JS = r"""
// Generated by tools/blender/props.py - assigns the baked atlas textures to the props_<atlas> materials,
// MikkTSpace tangents, webp textures and meshopt compression. Node names/hierarchy are preserved (no join/flatten).
import fs from 'node:fs';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression } from '@gltf-transform/extensions';
import { dedup, prune, quantize, reorder, tangents, textureCompress, unweld, weld } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import { generateTangents } from 'mikktspace';
import sharp from 'sharp';
const [inp, out, bakeDir] = process.argv.slice(2);
await MeshoptEncoder.ready; await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read(inp);
const root = doc.getRoot();
for (const t of root.listTextures()) t.dispose();
const tex = (p, name) => doc.createTexture(name).setImage(fs.readFileSync(p)).setMimeType('image/png').setURI(name + '.png');
for (const mat of root.listMaterials()) {
  const n = mat.getName();
  if (n === 'lamp') {
    mat.setBaseColorTexture(null).setNormalTexture(null).setOcclusionTexture(null).setMetallicRoughnessTexture(null)
      .setBaseColorFactor([0.85, 0.83, 0.78, 1]).setEmissiveFactor([1.0, 0.84, 0.62]).setRoughnessFactor(0.12).setMetallicFactor(0);
    continue;
  }
  const m = /^props_(\w+)$/.exec(n);
  if (!m) continue;
  const a = m[1];
  const alb = `${bakeDir}/${a}_albedo.png`, nrm = `${bakeDir}/${a}_normal.png`, orm = `${bakeDir}/${a}_orm.png`;
  if (!fs.existsSync(alb)) { console.warn('missing atlas', a); continue; }
  const tA = tex(alb, `props_${a}_albedo`), tN = tex(nrm, `props_${a}_normal`), tO = tex(orm, `props_${a}_orm`);
  mat.setBaseColorTexture(tA).setNormalTexture(tN).setOcclusionTexture(tO).setMetallicRoughnessTexture(tO)
    .setBaseColorFactor([1, 1, 1, 1]).setMetallicFactor(1).setRoughnessFactor(1).setEmissiveFactor([0, 0, 0]);
}
await doc.transform(
  unweld(),
  tangents({ generateTangents, overwrite: true }),
  weld(),
  dedup(),
  prune({ keepLeaves: true, keepAttributes: false }),
  reorder({ encoder: MeshoptEncoder }),
  quantize({ pattern: /^(NORMAL|TANGENT|TEXCOORD_\d+)$/, quantizeNormal: 12, quantizeTexcoord: 14 }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^normalTexture$/, quality: 90 }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^baseColorTexture$/, quality: 86 }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^(occlusionTexture|metallicRoughnessTexture)$/, quality: 82 }),
);
doc.createExtension(EXTMeshoptCompression).setRequired(true)
  .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
await io.write(out, doc);
const nodes = root.listNodes().map((n) => n.getName() + (n.listChildren().length ? '>' + n.listChildren().map((c) => c.getName()).join('|') : ''));
const mats = root.listMaterials().map((m) => m.getName() + (m.getOcclusionTexture() ? '[orm]' : ''));
console.log(JSON.stringify({ out, bytes: fs.statSync(out).size, nodes, mats }));
"""


def export(objs, derived):
    lod = build_guardrail(lod=True)[0]
    lod['atlas'] = 'rail'
    # the LOD reuses the atlas via a planar projection of the full rail's UV range: simple box unwrap into a
    # small galvanised region (sample the beam's own UVs from the hi-res rail by nearest vertex)
    hi = next(o for o in objs if o.name == 'guardrail')
    transfer_uv(hi, lod)
    allobjs = objs + derived + [lod]
    for o in allobjs:
        apply_final_materials(o)
        o.location = (0, 0, 0)
        o.rotation_euler = (0, 0, 0)
    head = bpy.data.objects.get('lamp_head')
    tower = bpy.data.objects.get('light_tower')
    if head and tower:
        head.parent = tower
        head.location = (0, 0, LT_TOP)
    tw = bpy.data.objects.get('tarp_water')
    tp = bpy.data.objects.get('tarp_pile')
    if tw and tp:
        tw.parent = tp
        tw.location = (0, 0, 0)
    keep = [o for o in allobjs if o.name in EXPORT_ORDER]
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in keep:
        o.select_set(True)
    for o in allobjs:
        if o.name not in EXPORT_ORDER:
            o.hide_set(True)
    geo = os.path.join(SCR, '_props_geo.glb')
    bpy.ops.export_scene.gltf(filepath=geo, export_format='GLB', use_selection=True, export_materials='EXPORT',
                              export_image_format='NONE', export_yup=True, export_apply=True, export_texcoords=True,
                              export_normals=True, export_tangents=False, export_animations=False, export_cameras=False,
                              export_lights=False, export_extras=False)
    js = os.path.join(SCR, '_pack.mjs')
    with open(js, 'w') as f:
        f.write(PACK_JS)
    r = subprocess.run(['node', js, geo, OUT_GLB, BAKE], cwd=ROOT, capture_output=True, text=True)
    log('pack:', r.stdout.strip()[-3000:], r.stderr.strip()[-2000:])


def transfer_uv(src, dst):
    """Nearest-vertex UV transfer (for the LOD rail)."""
    from mathutils.kdtree import KDTree
    sm = src.data
    suv = sm.uv_layers.active.data
    kd = KDTree(len(sm.loops))
    loop_co = []
    for p in sm.polygons:
        for li in p.loop_indices:
            co = sm.vertices[sm.loops[li].vertex_index].co
            kd.insert(co, li)
    kd.balance()
    dm = dst.data
    if not dm.uv_layers:
        dm.uv_layers.new(name='UVMap')
    duv = dm.uv_layers.active.data
    for p in dm.polygons:
        cen = p.center
        for li in p.loop_indices:
            co = dm.vertices[dm.loops[li].vertex_index].co
            # nudge toward the face centre so the lookup stays on the same face/island
            q = co.lerp(cen, 0.05)
            _, idx, _ = kd.find(q)
            duv[li].uv = suv[idx].uv


# ================================================================================================
# preview renders
# ================================================================================================
def setup_world(strength=1.0):
    w = bpy.data.worlds.get('env') or bpy.data.worlds.new('env')
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    env = nt.nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(HDRI, check_existing=True)
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = strength
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(env.outputs['Color'], bg.inputs['Color'])
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])


def ground(size=30):
    me = bpy.data.meshes.new('ground')
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size / 2)
    bm.to_mesh(me)
    bm.free()
    g = link(bpy.data.objects.new('ground', me))
    m = bpy.data.materials.new('ground_m')
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (0.5, 0.5, 0.5)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    for kind, sock, nc in (('diffuse', 'Base Color', False), ('arm', None, True)):
        im = nt.nodes.new('ShaderNodeTexImage')
        im.image = load_image(tex('rocky_trail', kind), nc)
        nt.links.new(mp.outputs['Vector'], im.inputs['Vector'])
        if sock:
            hs = nt.nodes.new('ShaderNodeHueSaturation')
            hs.inputs['Value'].default_value = 0.6
            nt.links.new(im.outputs['Color'], hs.inputs['Color'])
            nt.links.new(hs.outputs['Color'], bs.inputs[sock])
        else:
            sp = nt.nodes.new('ShaderNodeSeparateColor')
            nt.links.new(im.outputs['Color'], sp.inputs[0])
            mr = nt.nodes.new('ShaderNodeMapRange')
            mr.inputs['To Min'].default_value = 0.25
            mr.inputs['To Max'].default_value = 0.7
            nt.links.new(sp.outputs[1], mr.inputs['Value'])
            nt.links.new(mr.outputs['Result'], bs.inputs['Roughness'])
    g.data.materials.append(m)
    return g


def camera(loc, target, lens=50):
    cam = bpy.data.cameras.new('cam')
    cam.lens = lens
    cam.clip_start = 0.01
    co = link(bpy.data.objects.new('cam', cam))
    co.location = loc
    d = Vector(target) - Vector(loc)
    co.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = co
    return co


def render(path, w=1400, h=900, samples=48):
    sc = bpy.context.scene
    setup_cycles(samples)
    sc.cycles.use_denoising = True
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.filepath = path
    try:
        sc.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass
    bpy.ops.render.render(write_still=True)
    log('rendered', path)


def sun(strength=1.5, direction=(-0.4, 0.6, -0.7)):
    s = bpy.data.lights.new('sun', 'SUN')
    s.energy = strength
    s.angle = math.radians(20)
    so = link(bpy.data.objects.new('sun', s))
    so.rotation_euler = Vector(direction).to_track_quat('-Z', 'Y').to_euler()
    return so


SHOTS = {
    # name: ([(object, x, y, rotz_deg)], camera loc, target, lens)
    'hero': ([('jerrycan', 0, 0, 20), ('hatchet_lying', 0.3, -0.35, 15), ('toolbox', -0.55, 0.25, -12)],
             (1.25, -1.55, 0.8), (-0.05, 0.0, 0.18), 45),
    'jerrycan': ([('jerrycan', 0, 0, 30)], (0.72, -0.95, 0.62), (0.0, 0.0, 0.24), 50),
    'hatchet': ([('hatchet_lying', 0, 0, 0)], (0.36, -0.42, 0.34), (0.17, -0.02, 0.02), 50),
    'wood': ([('plank_stack', 0, 0, 0), ('crate', 0.4, 1.1, 8), ('sandbag', -0.9, -0.75, 20), ('plank_wedge', 1.2, -0.7, 0)],
             (3.2, -3.0, 1.6), (0.3, 0.2, 0.25), 35),
    'plankend': ([('plank_stack', 0, 0, 0)], (2.6, -0.8, 0.6), (1.85, 0.0, 0.22), 40),
    'road': ([('barrier', 0, 0, 0), ('barrier', 2.05, 0.05, 1.5), ('cone', 1.0, -0.9, 0), ('cone', 2.0, -1.1, 30),
              ('delineator', -1.6, -0.8, 0), ('light_tower', -0.6, 1.8, 0)],
             (3.6, -4.2, 1.5), (0.5, 0.2, 0.7), 35),
    'barrier': ([('barrier', 0, 0, 0), ('cone', 0.9, -0.75, 0)], (1.7, -2.0, 1.1), (0.2, 0.0, 0.4), 40),
    'rail': ([('guardrail', 0, 0, 0), ('guardrail', 0, -4.0, 0), ('guardrail_bent', 0, -8.0, 0), ('guardrail_end', 0, -12.0, 0),
              ('rail_reflector', 0, 0, 0), ('sign_rockfall', -0.6, -5.5, 180)],
             (2.6, 1.6, 1.25), (0.0, -3.5, 0.6), 30),
    'railclose': ([('guardrail', 0, 0, 0), ('guardrail', 0, -4.0, 0), ('rail_reflector', 0, -4.0, 0)],
                  (1.2, -2.6, 0.95), (0.2, -4.1, 0.55), 40),
    'signs': ([('sign_rockfall', 0, 0, 0), ('sign_roadworks', 1.3, 0, 0)], (0.65, -3.2, 1.9), (0.65, 0, 1.95), 45),
    'tarp': ([('tarp_pile', 0, 0, 0), ('tarp_water', 0, 0, 0), ('sandbag', 1.35, 0.25, 80), ('sandbag', -0.5, -1.05, 5)],
             (2.0, -2.3, 1.55), (0.0, 0.0, 0.3), 38),
    'roadside': ([('km_post', 0, 0, 0), ('snow_pole', 0.6, 0.3, 0), ('culvert_grate', -0.9, -0.6, 0), ('sign_bend', 1.3, 0.8, 0),
                  ('sign_chains', 2.4, 0.9, 0), ('delineator', -0.2, 0.9, 0)], (1.9, -3.6, 1.5), (0.6, 0.4, 0.8), 32),
    'culvert': ([('culvert_grate', 0, 0, 0)], (0.9, -0.9, 0.9), (0, 0, -0.05), 40),
    'site': ([('tarp_pile', 0, 0, 0), ('tarp_water', 0, 0, 0), ('sandbag', 1.45, 0.4, 80), ('sandbag', 1.4, -0.3, 95), ('light_tower', -2.2, 1.2, 0),
              ('cone', 1.8, -1.6, 0)], (3.4, -4.4, 2.0), (-0.3, 0.1, 0.5), 32),
}


def previews(objs, which, clay=False):
    byname = {o.name: o for o in objs}
    setup_world(1.0)
    g = ground()
    sun(1.2)
    if clay:
        cm = bpy.data.materials.new('clay')
        cm.use_nodes = True
        cm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
        cm.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.6
    else:
        for o in objs:
            if o.get('atlas') and o.data.materials and o.data.materials[0].name.startswith('m_'):
                apply_final_materials(o)
    for shot, (items, cl, ct, lens) in SHOTS.items():
        if which != 'all' and shot not in which:
            continue
        placed = []
        ok = True
        for (nm, x, y, rz) in items:
            src = byname.get('hatchet' if nm == 'hatchet_lying' else nm)
            if src is None:
                ok = False
                continue
            c = src.copy()
            link(c)
            c.hide_render = False
            if clay:
                c.data = src.data.copy()
                c.data.materials.clear()
                c.data.materials.append(cm)
            if nm == 'hatchet_lying':
                c.rotation_euler = (0, math.radians(90), math.radians(rz))
                c.location = (x, y, 0.02)
            else:
                c.rotation_euler = (0, 0, math.radians(rz))
                c.location = (x, y, 0)
            placed.append(c)
            if nm == 'light_tower' and 'lamp_head' in byname:
                hc = byname['lamp_head'].copy()
                link(hc)
                hc.hide_render = False
                if clay:
                    hc.data = byname['lamp_head'].data.copy()
                    hc.data.materials.clear()
                    hc.data.materials.append(cm)
                hc.location = (x, y, LT_TOP)
                placed.append(hc)
        if not placed:
            continue
        for o in objs:
            o.hide_render = True
        cam = camera(cl, ct, lens)
        render(os.path.join(SCR, f'{"clay" if clay else "prev"}_{shot}.png'), 1000, 667, 16 if clay else 40)
        for c in placed:
            bpy.data.objects.remove(c)
        bpy.data.objects.remove(cam)


# ================================================================================================
def main():
    reset_scene()
    names = ARGS['only'] or list(BUILDERS.keys())
    objs = []
    for nm in names:
        if nm not in BUILDERS:
            continue
        t0 = time.time()
        res = BUILDERS[nm]()
        for o in res:
            tris = sum(len(p.vertices) - 2 for p in o.data.polygons)
            log(f'built {o.name}: {tris} tris ({time.time() - t0:.1f}s)')
        objs += res
    if ARGS['clay']:
        previews(objs, ARGS['preview'], clay=True)
        return
    # UV atlases (always, deterministic) + bakes
    for atlas, members in ATLASES.items():
        aobjs = [o for o in objs if o.name in members]
        if not aobjs:
            continue
        unwrap_atlas(aobjs)
        if ARGS['bake'] == 'all' or atlas in ARGS['bake'].split(','):
            if ARGS['only'] and len(aobjs) != len(members):
                log(f'WARNING atlas {atlas} baked with a subset of its members (look-dev only)')
            bake_atlas(atlas, aobjs, ARGS['size'], ARGS['samples'])
    derived = assemble(objs)
    for o in derived:
        o['atlas'] = o.get('atlas') or ATLAS_OF.get(o.name.split('_')[0], 'wood')
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(SCR, '_props.blend'), compress=True)
    if ARGS['preview'] != 'none':
        previews(objs + derived, ARGS['preview'])
    if ARGS['export']:
        export([o for o in objs if o.name in bpy.data.objects], derived)
    log('done')


if __name__ == '__main__':
    main()
