"""
LANDSLIDE - procedural fractured rocks (ROCKS workstream).

Builds rock_0..9 (boulders, <=3k tris), rock_hull_0..9 (convex hulls <=48 verts, collider only) and
pebble_0..5 (<=300 tris) entirely from scratch, bakes high->low maps into one shared 2048 atlas
(albedo, normal, ORM) and writes public/assets/models/rocks.glb (meshopt + webp).

Pipeline per rock (all deterministic / seeded):
  1. Joint-bounded block: a box cut by random planes (bmesh bisect + fill). Every face carries a label:
     0 = old weathered joint face, 1 = fresh fracture face, 2 = old chip (semi-weathered).
     Optional concave steps are boolean-subtracted (stepped fractures along two joint sets).
  2. Edge rounding by label: weathered-weathered edges get a wide bevel, fresh edges stay crisp.
  3. Voxel remesh -> uniform dense mesh; labels transferred back with a BVH lookup.
  4. Multi-scale displacement in numpy: face undulation, hackly granular roughness on fresh faces,
     foliation/strata steps (gneiss/schist), jagged crack grooves, pitting.
  5. Curvature (convexity / cavity) at two scales -> colour attributes that drive the albedo shader.
  6. Low poly = collapse-decimated high poly; normalised to bounding radius 1.0 around the volume centroid.
  7. Shared UV atlas (smart project -> conformal unwrap -> average scale -> pack).
  8. Cycles bakes (selected-to-active, per rock into its own image, composited + dilated in numpy):
     tangent normal (incl. shader micro bump), albedo (box-projected CC0 scans blended by masks),
     AO (local, 0.25 m) and roughness.
  9. Geometry-only GLB from Blender, then a gltf-transform pass (scratch/rocks/_pack.mjs, generated
     by this script) assigns the single 'rock' material, MikkTSpace tangents, webp textures, meshopt.

Usage:
  /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/blender/rocks.py -- [options]
    --stage shapes|all|pack shapes: high-poly clay lineup render only (fast look-dev). default all
                            pack: re-run only the gltf-transform pass (LODs, webp, meshopt) on the last full build
    --rocks 0,3,5           subset of rocks (default all); --pebbles 0,1 subset of pebbles
    --voxel 0.008           high-poly voxel size (normalised units, rock radius = 1)
    --samples 8             bake AA samples
    --atlas 2048            atlas size
    --no-export             skip GLB export
    --no-preview            skip preview renders
Conventions: rocks are authored with a natural (dry-to-damp) albedo; rain wetness is applied at runtime
(ctx.env.wetness) by whoever renders them. Blender Z-up; exported glTF is Y-up (mud faces -Y).
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
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

T0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SCRATCH = os.path.join(ROOT, 'scratch', 'rocks')
OUT_GLB = os.path.join(ROOT, 'public', 'assets', 'models', 'rocks.glb')
TEX = os.path.join(ROOT, 'raw_assets', 'tex')
HDRI = os.path.join(ROOT, 'raw_assets', 'hdri', 'overcast_soil_puresky_2k.hdr')
os.makedirs(SCRATCH, exist_ok=True)


def log(*a):
    print('[rocks %6.1fs]' % (time.time() - T0), *a, flush=True)


# ------------------------------------------------------------------------------------------------
# args
# ------------------------------------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    a = {'stage': 'all', 'rocks': None, 'pebbles': None, 'voxel': 0.008, 'samples': 8, 'atlas': 2048,
         'export': True, 'preview': True}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == '--stage': a['stage'] = argv[i + 1]; i += 1
        elif k == '--rocks': a['rocks'] = [int(x) for x in argv[i + 1].split(',') if x.isdigit()]; i += 1
        elif k == '--pebbles': a['pebbles'] = [int(x) for x in argv[i + 1].split(',') if x.isdigit()]; i += 1
        elif k == '--voxel': a['voxel'] = float(argv[i + 1]); i += 1
        elif k == '--samples': a['samples'] = int(argv[i + 1]); i += 1
        elif k == '--atlas': a['atlas'] = int(argv[i + 1]); i += 1
        elif k == '--no-export': a['export'] = False
        elif k == '--no-preview': a['preview'] = False
        elif k == '--quick': a['quick'] = True
        i += 1
    return a


ARGS = parse_args()

# ------------------------------------------------------------------------------------------------
# numpy noise (vectorised, deterministic)
# ------------------------------------------------------------------------------------------------
_M1 = np.uint32(0x7feb352d)
_M2 = np.uint32(0x846ca68b)


def _hash(x):
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x *= _M1
    x ^= x >> np.uint32(15)
    x *= _M2
    x ^= x >> np.uint32(16)
    return x


def hash3(ix, iy, iz, seed):
    h = _hash(ix.astype(np.int64).astype(np.uint32) + np.uint32((seed * 2654435761) & 0xffffffff))
    h = _hash(h ^ iy.astype(np.int64).astype(np.uint32))
    h = _hash(h ^ (iz.astype(np.int64).astype(np.uint32) * np.uint32(0x27d4eb2d)))
    return h


_GRAD = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1], [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]],
                 np.float64)


def gnoise(p, seed=0):
    """Gradient (Perlin-style) noise, ~[-1, 1]."""
    pi = np.floor(p)
    f = p - pi
    pi = pi.astype(np.int64)
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    out = np.zeros(len(p))
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1.0 - u[:, 0]
        for dy in (0, 1):
            wy = u[:, 1] if dy else 1.0 - u[:, 1]
            for dz in (0, 1):
                wz = u[:, 2] if dz else 1.0 - u[:, 2]
                h = hash3(pi[:, 0] + dx, pi[:, 1] + dy, pi[:, 2] + dz, seed) & np.uint32(15)
                g = _GRAD[h]
                d = f - np.array([dx, dy, dz], np.float64)
                out += wx * wy * wz * (g * d).sum(1)
    return out


def fbm(p, octaves=4, lac=2.03, gain=0.5, seed=0):
    s = np.zeros(len(p))
    amp, norm, q = 1.0, 0.0, p
    for i in range(octaves):
        s += amp * gnoise(q, seed + i * 131)
        norm += amp
        amp *= gain
        q = q * lac + 11.37
    return s / norm


def ridged(p, octaves=3, lac=2.1, gain=0.5, seed=0):
    s = np.zeros(len(p))
    amp, norm, q = 1.0, 0.0, p
    for i in range(octaves):
        n = 1.0 - np.abs(gnoise(q, seed + i * 71))
        s += amp * n * n
        norm += amp
        amp *= gain
        q = q * lac + 5.3
    return s / norm


def worley(p, seed=0, want_point=False):
    """Returns F1, F2 distances and the hash id of the nearest feature point.
    With want_point: also ID2 and the offsets (p - feature) for nearest and second nearest."""
    pi = np.floor(p)
    f = p - pi
    pi = pi.astype(np.int64)
    n = len(p)
    F1 = np.full(n, 9.0)
    F2 = np.full(n, 9.0)
    ID = np.zeros(n, np.uint32)
    ID2 = np.zeros(n, np.uint32)
    OFF = np.zeros((n, 3))
    OFF2 = np.zeros((n, 3))
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for oz in (-1, 0, 1):
                h = hash3(pi[:, 0] + ox, pi[:, 1] + oy, pi[:, 2] + oz, seed)
                jx = (h & np.uint32(1023)) / 1023.0
                jy = ((h >> np.uint32(10)) & np.uint32(1023)) / 1023.0
                jz = ((h >> np.uint32(20)) & np.uint32(1023)) / 1023.0
                dx = f[:, 0] - (ox + jx)
                dy = f[:, 1] - (oy + jy)
                dz = f[:, 2] - (oz + jz)
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                c1 = d < F1
                c2 = (~c1) & (d < F2)
                if want_point:
                    off = np.stack([dx, dy, dz], 1)
                    OFF2 = np.where(c1[:, None], OFF, np.where(c2[:, None], off, OFF2))
                    OFF = np.where(c1[:, None], off, OFF)
                ID2 = np.where(c1, ID, np.where(c2, h, ID2))
                ID = np.where(c1, h, ID)
                F2 = np.where(c1, F1, np.where(c2, d, F2))
                F1 = np.where(c1, d, F1)
    if want_point:
        return F1, F2, ID, ID2, OFF, OFF2
    return F1, F2, ID


def hash_float(h, k):
    """k-th pseudo random float in [-1, 1] derived from uint32 hash array h."""
    return _hash(h + np.uint32(k * 0x9E3779B9 & 0xffffffff)) / 2147483647.5 - 1.0


def lens_pre(p, sd):
    """Patchy on/off mask (lenses) for fine laminae."""
    return (fbm(p * 1.9 + 4.4, 2, seed=sd + 90) > 0.05).astype(np.float64)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------------------------
# rock / pebble specs
# ------------------------------------------------------------------------------------------------
# dims: box before cuts; joints: big joint-plane cuts; chips: small corner chips (fresh);
# steps: concave stepped fractures; fol: foliation normal (gneiss/schist) or None; cracks: through cracks;
# age: 0 fresh rockfall .. 1 long-exposed talus block (lichen, moss, rounding); bevel: max edge radius;
# fresh: probability a joint cut is a fresh fracture; lith: albedo family.
ROCKS = [
    dict(seed=11, veins=1, dims=(1.05, 0.9, 0.72), joints=3, chips=4, steps=0, fol=None, cracks=1, age=0.85, bevel=0.11, fresh=0.25, lith='granite'),
    dict(seed=23, veins=1, dims=(1.7, 0.72, 0.62), joints=4, chips=3, steps=1, fol=(0.35, 0.15, 0.92), strata=0.018, cracks=0, age=0.55, bevel=0.07, fresh=0.35, lith='gneiss'),
    dict(seed=37, veins=2, dims=(1.45, 1.15, 0.4), joints=4, chips=4, steps=1, fol=(0.05, 0.08, 1.0), strata=0.022, cracks=0, age=0.7, bevel=0.05, fresh=0.3, lith='gneiss'),
    dict(seed=41, dims=(1.75, 0.95, 0.72), joints=6, chips=5, steps=0, fol=None, cracks=0, age=0.12, bevel=0.025, fresh=0.85, lith='granite'),
    dict(seed=53, veins=1, dims=(1.15, 1.0, 0.88), joints=4, chips=3, steps=2, fol=None, cracks=1, age=0.95, bevel=0.13, fresh=0.1, lith='granite'),
    dict(seed=67, dims=(1.4, 0.8, 0.7), joints=6, chips=4, steps=0, fol=(0.55, 0.0, 0.83), strata=0.012, cracks=0, age=0.08, bevel=0.02, fresh=0.9, lith='gneiss'),
    dict(seed=71, dims=(1.0, 0.92, 0.8), joints=5, chips=2, steps=0, fol=None, cracks=0, age=1.0, bevel=0.12, fresh=0.0, facets=12, lith='granite'),
    dict(seed=83, veins=1, dims=(1.8, 0.62, 1.05), joints=5, chips=4, steps=0, fol=None, cracks=0, age=0.5, bevel=0.08, fresh=0.3, lith='granite'),
    dict(seed=97, veins=1, dims=(1.55, 1.2, 0.32), joints=5, chips=3, steps=1, fol=(0.08, 0.0, 1.0), strata=0.02, cracks=1, age=0.5, bevel=0.03, fresh=0.4, lith='schist'),
    dict(seed=101, veins=1, dims=(1.35, 1.1, 0.62), joints=8, chips=6, steps=1, fol=None, cracks=0, age=0.3, bevel=0.04, fresh=0.55, lith='granite'),
]
PEBBLES = [
    dict(seed=211, dims=(1.2, 0.9, 0.6), joints=4, chips=3, steps=0, fol=None, cracks=0, age=0.3, bevel=0.06, fresh=0.6, lith='granite'),
    dict(seed=223, dims=(1.3, 1.0, 0.45), joints=5, chips=2, steps=0, fol=(0, 0, 1), strata=0.0, cracks=0, age=0.5, bevel=0.1, fresh=0.3, lith='schist'),
    dict(seed=227, dims=(1.0, 0.85, 0.8), joints=5, chips=4, steps=0, fol=None, cracks=0, age=0.1, bevel=0.03, fresh=0.9, lith='granite'),
    dict(seed=229, dims=(1.1, 1.0, 0.75), joints=4, chips=2, steps=0, fol=None, cracks=0, age=0.9, bevel=0.2, fresh=0.1, lith='gneiss'),
    dict(seed=233, dims=(1.5, 0.8, 0.55), joints=5, chips=3, steps=0, fol=(0.4, 0, 0.9), strata=0.0, cracks=0, age=0.4, bevel=0.05, fresh=0.5, lith='gneiss'),
    dict(seed=239, dims=(1.0, 1.0, 0.7), joints=6, chips=3, steps=0, fol=None, cracks=0, age=0.6, bevel=0.12, fresh=0.4, lith='granite'),
]
for i, r in enumerate(ROCKS):
    r['name'] = 'rock_%d' % i
    r['kind'] = 'rock'
    r['index'] = i
for i, r in enumerate(PEBBLES):
    r['name'] = 'pebble_%d' % i
    r['kind'] = 'pebble'
    r['index'] = i

LAB_WEATHERED, LAB_FRESH, LAB_SEMI = 0, 1, 2


# ------------------------------------------------------------------------------------------------
# scene helpers
# ------------------------------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)


def label_materials():
    mats = []
    for nm, col in (('lab_weathered', (0.25, 0.25, 0.25, 1)), ('lab_fresh', (0.7, 0.66, 0.6, 1)), ('lab_semi', (0.45, 0.42, 0.38, 1))):
        m = bpy.data.materials.get(nm) or bpy.data.materials.new(nm)
        m.diffuse_color = col
        mats.append(m)
    return mats


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def apply_modifiers(obj):
    """Bake the modifier stack into obj.data without operators (works headless)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj


def mesh_arrays(me):
    nv = len(me.vertices)
    co = np.empty(nv * 3)
    me.vertices.foreach_get('co', co)
    ne = len(me.edges)
    ed = np.empty(ne * 2, np.int64)
    me.edges.foreach_get('vertices', ed)
    return co.reshape(-1, 3), ed.reshape(-1, 2)


def vertex_normals(me):
    nv = len(me.vertices)
    n = np.empty(nv * 3)
    me.vertex_normals.foreach_get('vector', n)
    return n.reshape(-1, 3)


def set_coords(me, co):
    me.vertices.foreach_set('co', co.astype(np.float64).ravel())
    me.update()


def neighbour_avg(vals, ed, nv):
    """Mean of 1-ring neighbours (vals: (nv,) or (nv,k))."""
    e0, e1 = ed[:, 0], ed[:, 1]
    cnt = np.bincount(e0, minlength=nv) + np.bincount(e1, minlength=nv)
    cnt = np.maximum(cnt, 1)
    if vals.ndim == 1:
        return (np.bincount(e0, vals[e1], nv) + np.bincount(e1, vals[e0], nv)) / cnt
    out = np.empty_like(vals)
    for k in range(vals.shape[1]):
        out[:, k] = (np.bincount(e0, vals[e1, k], nv) + np.bincount(e1, vals[e0, k], nv)) / cnt
    return out


def smooth(vals, ed, nv, iters, lam=0.5, weight=None):
    for _ in range(iters):
        avg = neighbour_avg(vals, ed, nv)
        w = lam if weight is None else (lam * weight if vals.ndim == 1 else (lam * weight)[:, None])
        vals = vals + w * (avg - vals)
    return vals


def taubin(co, ed, nv, iters, lam=0.5, mu=-0.53, weight=None):
    for _ in range(iters):
        co = smooth(co, ed, nv, 1, lam, weight)
        co = smooth(co, ed, nv, 1, mu, weight)
    return co


def rand_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------------------------------------
# 1. joint-bounded block (low-poly solid with face labels as material indices)
# ------------------------------------------------------------------------------------------------
def cut_plane(bm, co, no, label):
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=co, plane_no=no, clear_outer=True)
    cut_edges = [e for e in res['geom_cut'] if isinstance(e, bmesh.types.BMEdge)]
    if not cut_edges:
        return False
    before = set(bm.faces)
    bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)
    for f in bm.faces:
        if f not in before:
            f.material_index = label
    return True


def support(bm, n):
    d = [v.co.dot(Vector(n)) for v in bm.verts]
    return max(d), min(d)


def build_solid(spec, mats):
    rng = np.random.RandomState(spec['seed'])
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    sx, sy, sz = spec['dims']
    # slight random skew so the block is not a perfect orthogonal box (joint sets are rarely at 90 deg)
    shear = Matrix(((1, rng.uniform(-0.18, 0.18), rng.uniform(-0.15, 0.15)),
                    (rng.uniform(-0.12, 0.12), 1, rng.uniform(-0.15, 0.15)),
                    (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 1)))
    for v in bm.verts:
        v.co = shear @ Vector((v.co.x * sx, v.co.y * sy, v.co.z * sz))
    for f in bm.faces:
        f.material_index = LAB_WEATHERED
    # big joint-plane cuts
    for j in range(spec['joints']):
        n = rand_unit(rng)
        hi, lo = support(bm, n)
        depth = rng.uniform(0.1, 0.32) * (hi - lo)
        lab = LAB_FRESH if rng.rand() < spec['fresh'] else (LAB_WEATHERED if rng.rand() < 0.6 else LAB_SEMI)
        cut_plane(bm, Vector(n) * (hi - depth), Vector(n), lab)
    me = bpy.data.meshes.new(spec['name'] + '_solid')
    bm.to_mesh(me)
    bm.free()
    obj = link(bpy.data.objects.new(spec['name'] + '_solid', me))
    for m in mats:
        obj.data.materials.append(m)
    # concave stepped fractures: subtract the wedge between two joint planes (L-shaped step)
    for s in range(spec['steps']):
        co, _ = mesh_arrays(obj.data)
        n1 = rand_unit(rng)
        t = rand_unit(rng)
        n2 = t - n1 * np.dot(t, n1)
        n2 /= np.linalg.norm(n2)
        ang = rng.uniform(-0.25, 0.25)       # not exactly 90 degrees
        n2 = n2 * math.cos(ang) + n1 * math.sin(ang)
        n2 /= np.linalg.norm(n2)
        e1 = co @ n1
        e2 = co @ n2
        d1 = e1.max() - rng.uniform(0.18, 0.35) * (e1.max() - e1.min())
        d2 = e2.max() - rng.uniform(0.25, 0.5) * (e2.max() - e2.min())
        w = np.cross(n1, n2)
        w /= np.linalg.norm(w)
        L = 4.0
        bmc = bmesh.new()
        bmesh.ops.create_cube(bmc, size=1.0)
        # cube [-.5,.5]^3 -> box spanning [d1, d1+L] along n1, [d2, d2+L] along n2, and a partial span along w
        ew = co @ w
        wlo = ew.min() - 1.0 if rng.rand() < 0.5 else rng.uniform(ew.min(), 0.3 * (ew.min() + ew.max()))
        whi = ew.max() + 1.0
        for v in bmc.verts:
            a = (v.co.x + 0.5)
            b = (v.co.y + 0.5)
            c = (v.co.z + 0.5)
            p = n1 * (d1 + a * L) + n2 * (d2 + b * L) + w * (wlo + c * (whi - wlo))
            v.co = Vector(p)
        lab = LAB_FRESH if rng.rand() < spec['fresh'] else LAB_WEATHERED
        for f in bmc.faces:
            f.material_index = lab
        mc = bpy.data.meshes.new('cutter')
        bmc.to_mesh(mc)
        bmc.free()
        cutter = link(bpy.data.objects.new('cutter', mc))
        for m in mats:
            cutter.data.materials.append(m)
        mod = obj.modifiers.new('step', 'BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.object = cutter
        mod.solver = 'EXACT'
        try:
            mod.material_mode = 'TRANSFER'
        except Exception:
            pass
        apply_modifiers(obj)
        bpy.data.objects.remove(cutter)
        bpy.data.meshes.remove(mc)
    # normalise: centroid to origin, bounding radius 1
    co, _ = mesh_arrays(obj.data)
    c = co.mean(0)
    co = co - c
    r = np.sqrt((co ** 2).sum(1)).max()
    set_coords(obj.data, co / r)
    return obj


def bevel_solid(obj, spec):
    """Round edges according to the labels of the two adjacent faces (weathered = round, fresh = crisp)."""
    rng = np.random.RandomState(spec['seed'] + 5)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    lay = bm.edges.layers.float.get('bevel_weight_edge') or bm.edges.layers.float.new('bevel_weight_edge')
    for e in bm.edges:
        labs = sorted(f.material_index for f in e.link_faces)
        if len(labs) != 2:
            w = 0.5
        elif labs == [0, 0]:
            w = rng.uniform(0.55, 1.0)
        elif labs == [0, 2] or labs == [2, 2]:
            w = rng.uniform(0.3, 0.6)
        elif labs == [1, 1]:
            w = rng.uniform(0.04, 0.14)
        else:
            w = rng.uniform(0.08, 0.25)
        e[lay] = w
    bm.to_mesh(obj.data)
    bm.free()
    mod = obj.modifiers.new('bevel', 'BEVEL')
    mod.width = spec['bevel']
    mod.segments = 5
    mod.limit_method = 'WEIGHT'
    mod.profile = 0.55
    mod.use_clamp_overlap = True
    mod.harden_normals = False
    apply_modifiers(obj)


def trim_solid(obj, spec):
    """Facet cuts and fresh corner chips AFTER the weathering bevel: crisp new faces cut through old rounded edges."""
    rng = np.random.RandomState(spec['seed'] + 3)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # small facet cuts: trim edges/corners into several facets (breaks the box read)
    for j in range(spec.get('facets', 8)):
        n = rand_unit(rng)
        hi, lo = support(bm, n)
        depth = rng.uniform(0.015, 0.06) * (hi - lo)
        lab = LAB_SEMI if rng.rand() < 0.5 else (LAB_FRESH if rng.rand() < spec['fresh'] else LAB_WEATHERED)
        cut_plane(bm, Vector(n) * (hi - depth), Vector(n), lab)
    # corner chips (fresh impact damage)
    for c in range(spec['chips']):
        bm.verts.ensure_lookup_table()
        v = bm.verts[rng.randint(len(bm.verts))].co.copy()
        n = np.array(v) / max(np.linalg.norm(np.array(v)), 1e-6) + rng.normal(size=3) * 0.35
        n /= np.linalg.norm(n)
        hi, lo = support(bm, n)
        depth = rng.uniform(0.035, 0.11) * (hi - lo)
        lab = LAB_FRESH if rng.rand() < 0.8 else LAB_SEMI
        cut_plane(bm, Vector(n) * (hi - depth), Vector(n), lab)
    bm.to_mesh(obj.data)
    bm.free()
    co, _ = mesh_arrays(obj.data)
    co = co - co.mean(0)
    set_coords(obj.data, co / np.sqrt((co ** 2).sum(1)).max())


# ------------------------------------------------------------------------------------------------
# 2. high-poly: remesh + displacement + masks
# ------------------------------------------------------------------------------------------------
def build_highpoly(spec, voxel):
    mats = label_materials()
    solid = build_solid(spec, mats)
    bevel_solid(solid, spec)
    trim_solid(solid, spec)
    # BVH of the labelled solid (for label transfer)
    bm = bmesh.new()
    bm.from_mesh(solid.data)
    bm.faces.ensure_lookup_table()
    face_lab = np.array([f.material_index for f in bm.faces])
    bvh = BVHTree.FromBMesh(bm)
    bm.free()

    hp = link(bpy.data.objects.new(spec['name'] + '_high', solid.data.copy()))
    rm = hp.modifiers.new('remesh', 'REMESH')
    rm.mode = 'VOXEL'
    rm.voxel_size = voxel
    rm.adaptivity = 0.0
    apply_modifiers(hp)
    me = hp.data
    me.materials.clear()
    co, ed = mesh_arrays(me)
    nv = len(co)
    co = taubin(co, ed, nv, 3)

    # labels per vertex (nearest labelled face), softened a little across boundaries
    lab = np.empty(nv, np.int64)
    for i in range(nv):
        hit = bvh.find_nearest(Vector(co[i]))
        lab[i] = face_lab[hit[2]] if hit[2] is not None else 0
    fresh = (lab == LAB_FRESH).astype(np.float64)
    semi = (lab == LAB_SEMI).astype(np.float64)
    fresh = smooth(fresh, ed, nv, 3, 0.5)
    semi = smooth(semi, ed, nv, 3, 0.5)
    set_coords(me, co)
    nrm = vertex_normals(me)

    rng = np.random.RandomState(spec['seed'] + 17)
    sd = spec['seed'] * 7
    p = co
    disp = np.zeros(nv)
    # convex-edge mask of the clean solid (for edge chipping), ~3-4 cm scale
    base_edge = ((co - smooth(co, ed, nv, 6, 0.6)) * nrm).sum(1)
    base_edge = np.clip(base_edge / (np.percentile(base_edge, 98) + 1e-9), 0, 1)
    base_edge = smooth(base_edge, ed, nv, 2, 0.5)
    # (a) undulation of the joint planes (rough, slightly warped planes, never flat CG faces)
    disp += 0.03 * fbm(p * 0.55 + rng.uniform(-50, 50, 3), 2, seed=sd + 16)
    disp += 0.036 * fbm(p * 1.1 + rng.uniform(-50, 50, 3), 3, seed=sd + 1)
    disp += 0.012 * fbm(p * 2.7 + rng.uniform(-50, 50, 3), 2, seed=sd + 8)
    # (b) piecewise-planar facets: every Voronoi cell is a slightly tilted plane -> small fracture steps
    # (continuous: the two nearest cell planes are blended across the border -> creases, not a crack network)
    ff = 1.35
    F1f, F2f, IDf, IDf2, OFFf, OFFf2 = worley(p * ff + rng.uniform(-50, 50, 3), seed=sd + 9, want_point=True)

    def plane(idh, off):
        tl = np.stack([hash_float(idh, 1), hash_float(idh, 2), hash_float(idh, 3)], 1)
        return (tl * off).sum(1) / ff * 0.07 + hash_float(idh, 4) * 0.004
    pa, pb = plane(IDf, OFFf), plane(IDf2, OFFf2)
    # wide blend: the cell borders become soft undulations instead of a crisp polygonal crease network
    wb = 0.5 * (1.0 - smoothstep(0.0, 0.05 + 0.75 * (1 - fresh), F2f - F1f))
    facet = pa * (1 - wb) + pb * wb
    disp += facet * (0.2 + 0.06 * fresh)
    # (c) medium relief; weathered faces smoother
    disp += (0.004 + 0.004 * fresh) * fbm(p * 5.0 + rng.uniform(-50, 50, 3), 3, seed=sd + 2)
    # (c2) cm-scale surface roughness (granular disintegration / pitting on old faces, grain relief on fresh ones):
    # without it the faces read as smooth soap under the soft overcast light
    rough_n = fbm(p * 15.0 + rng.uniform(-50, 50, 3), 2, seed=sd + 15)
    disp += (0.0048 * (1.0 - fresh) + 0.0022 * fresh) * rough_n
    # fresh fractures are mostly near-planar; hackle / scallops only in patches (uniform relief = "orange peel")
    patch = smoothstep(-0.05, 0.35, fbm(p * 1.4 + rng.uniform(-50, 50, 3), 2, seed=sd + 14))
    # (d) hackle: anisotropic ridges radiating along the fracture propagation direction
    hd = rand_unit(rng)
    q = p + np.outer(p @ hd, hd) * 2.5                 # stretched across hd -> ridges run along hd
    rid = ridged(q * 9.0 + rng.uniform(-50, 50, 3), 3, seed=sd + 3)
    disp += fresh * patch * 0.006 * (rid - 0.5)
    # (e) shallow conchoidal scallops (sparse)
    F1, F2, _ = worley(p * 3.0 + rng.uniform(-50, 50, 3), seed=sd + 4)
    disp += fresh * patch * 0.006 * (F1 * F1 - 0.25)
    # (f) chipped, irregular edges: bites taken out of convex edges (more on fresh / semi edges)
    F1c, F2c, IDc = worley(p * 7.0 + rng.uniform(-50, 50, 3), seed=sd + 5)
    bite = (1.0 - smoothstep(0.1, 0.75, F1c)) * (hash_float(IDc, 7) > -0.2)
    chipk = 0.016 + 0.014 * np.maximum(fresh, semi)
    disp -= base_edge * chipk * (0.5 * bite + 0.5 * np.clip(fbm(p * 9.0 + 2.0, 2, seed=sd + 12) + 0.5, 0, 1))
    # (f) foliation / strata. Real gneiss / schist banding is mostly TONAL (light quartz-feldspar vs dark biotite
    # layers) with gentle, folded, lens-shaped layers; only the odd mica-rich layer is eroded into a groove.
    # (A sawtooth step on every layer reads as a stacked-plate / layer-cake CG rock.)
    layer_val = np.full(nv, 0.5)
    if spec.get('fol') is not None:
        a = np.array(spec['fol'], np.float64)
        a /= np.linalg.norm(a)
        thick = rng.uniform(0.04, 0.075) if spec['kind'] == 'rock' else 0.2
        # folding: low-frequency warp of the layer coordinate (open folds, pinch-and-swell)
        t = (p @ a) / thick + 2.2 * fbm(p * 0.9 + 3.1, 3, seed=sd + 6) + 0.45 * fbm(p * 2.6 + 1.7, 2, seed=sd + 7)
        t = t + 0.45 * np.sin(t * 0.9 + 0.8) + 0.3 * np.sin(t * 0.37 + 2.1)   # very irregular layer thickness
        fl = np.floor(t)
        fr = t - fl
        cross = 1.0 - np.abs(nrm @ a)         # faces cutting across the layers show the layering
        amp = spec.get('strata', 0.0)
        hid = _hash(fl.astype(np.int64).astype(np.uint32) + np.uint32(sd)) / 4294967295.0
        hid2 = _hash(fl.astype(np.int64).astype(np.uint32) + np.uint32(sd + 77)) / 4294967295.0
        # soft band boundaries (value 0..1 per band, blended over ~15% of a band): tonal banding for the albedo
        nb = _hash((fl + 1).astype(np.int64).astype(np.uint32) + np.uint32(sd)) / 4294967295.0
        layer_val = hid + (nb - hid) * smoothstep(0.93, 1.0, fr)
        # thin dark laminae inside some light bands (mica films)
        lamin = (np.abs(np.sin(t * math.pi * 2.0 * 2.5)) > 0.97) * (hid > 0.55) * lens_pre(p, sd)
        layer_val = np.where(lamin > 0, 0.05, layer_val)
        # lenses: layers pinch out along strike
        lens = smoothstep(-0.25, 0.3, fbm(p * 1.6 + 9.1, 2, seed=sd + 13))
        # recessed (weak) layers: only ~22% of bands, rounded groove profile, depth varies per band
        weak = (hid2 > 0.78).astype(np.float64) * (0.5 + 0.5 * hid)
        groove = np.sin(np.pi * np.clip(fr, 0, 1)) ** 2
        disp -= amp * 0.9 * weak * groove * (0.25 + 0.75 * cross) * lens
        # gentle differential relief between the other layers
        disp += amp * 0.12 * (hid - 0.5) * cross
        # fine laminae (sub-mm..mm), only in places
        disp += 0.0007 * np.sin(t * 2 * math.pi * 3.0) * cross * lens
    # (g2) quartz / aplite veins: thin pale bands along random planes, slightly proud (resistant to weathering)
    vein = np.zeros(nv)
    for k in range(spec.get('veins', 0)):
        nvn = rand_unit(rng)
        dv = rng.uniform(-0.3, 0.3)
        sv = p @ nvn - dv + 0.05 * fbm(p * 2.0 + k * 5.1, 3, seed=sd + 50 + k)
        wv = rng.uniform(0.004, 0.01) * (0.55 + 0.9 * np.clip(fbm(p * 1.7 + k * 2.2, 2, seed=sd + 60 + k) + 0.5, 0, 1))
        vfade = smoothstep(-0.45, -0.1, fbm(p * 1.1 + k * 4.4, 2, seed=sd + 70 + k))
        vm = (1.0 - smoothstep(wv * 0.7, wv, np.abs(sv))) * vfade
        disp += 0.0025 * vm * (1.0 - fresh * 0.6)
        vein = np.maximum(vein, vm)
    # (g) through-going cracks: jagged narrow grooves, fading out in places
    crack = np.zeros(nv)
    for k in range(spec['cracks']):
        nc = rand_unit(rng)
        dc = rng.uniform(-0.35, 0.35)
        s = p @ nc - dc + 0.035 * fbm(p * 3.5 + k * 7.7, 3, seed=sd + 20 + k) + 0.01 * fbm(p * 12 + k, 2, seed=sd + 30 + k)
        w = rng.uniform(0.0045, 0.008)
        fade = smoothstep(-0.35, 0.1, fbm(p * 1.3 + k * 3.3, 2, seed=sd + 40 + k))
        g = np.exp(-(s / w) ** 2) * fade
        # narrow fissure (a wide soft V-groove reads as a fabric fold); the dark core comes from the albedo
        disp -= 0.013 * g
        disp += 0.0018 * np.tanh(s / 0.008) * fade   # slight offset across the crack
        crack = np.maximum(crack, np.exp(-(s / (w * 2.2)) ** 2) * fade)
    co = co + nrm * disp[:, None]
    co = taubin(co, ed, nv, 1)
    set_coords(me, co)
    me.shade_smooth()

    # curvature: convexity (edges) and concavity (cavities) at two scales
    nrm = vertex_normals(me)
    sm2 = smooth(co, ed, nv, 3, 0.6)
    sm12 = smooth(co, ed, nv, 18, 0.6)
    c_fine = ((co - sm2) * nrm).sum(1) / voxel
    c_coarse = ((co - sm12) * nrm).sum(1) / voxel
    edge = np.clip(c_fine * 1.2 + c_coarse * 0.35, 0, None)
    edge = np.clip(edge / (np.percentile(edge, 97) + 1e-6), 0, 1)
    cav = np.clip(-(c_fine * 0.15 + c_coarse * 0.6), 0, None)
    cav = np.clip(cav / (np.percentile(cav, 96) + 1e-6), 0, 1)
    cav = np.maximum(cav, crack * 0.9)

    attrs = {
        'rk_a': np.stack([fresh, edge, cav, np.ones(nv)], 1),
        'rk_b': np.stack([crack, layer_val, semi, vein], 1),
    }
    for nm, arr in attrs.items():
        at = me.color_attributes.new(nm, 'FLOAT_COLOR', 'POINT')
        at.data.foreach_set('color', arr.astype(np.float32).ravel())
    # keep the solid for reference-free cleanup
    sd_me = solid.data
    bpy.data.objects.remove(solid)
    bpy.data.meshes.remove(sd_me)
    hp['tri_count'] = sum(len(pl.vertices) - 2 for pl in me.polygons)
    log(spec['name'], 'high-poly', nv, 'verts', hp['tri_count'], 'tris')
    return hp


# ------------------------------------------------------------------------------------------------
# preview rendering helpers
# ------------------------------------------------------------------------------------------------
def setup_world(strength=1.0):
    w = bpy.data.worlds.new('env')
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
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = 'AgX'
    try:
        sc.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass


def wet_ground(size=40):
    me = bpy.data.meshes.new('ground')
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size / 2)
    bm.to_mesh(me)
    bm.free()
    g = link(bpy.data.objects.new('ground', me))
    m = bpy.data.materials.new('wet_ground')
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexCoord')
    img = nt.nodes.new('ShaderNodeTexImage')
    img.image = bpy.data.images.load(os.path.join(TEX, 'brown_mud_rocks_01', 'diffuse.jpg'), check_existing=True)
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (0.4, 0.4, 0.4)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], img.inputs['Vector'])
    dark = nt.nodes.new('ShaderNodeMix')
    dark.data_type = 'RGBA'
    dark.blend_type = 'MULTIPLY'
    dark.inputs['Factor'].default_value = 1.0
    dark.inputs['B'].default_value = (0.55, 0.55, 0.55, 1)
    nt.links.new(img.outputs['Color'], dark.inputs['A'])
    nt.links.new(dark.outputs['Result'], bs.inputs['Base Color'])
    nz = nt.nodes.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = 0.6
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['From Min'].default_value = 0.45
    mr.inputs['From Max'].default_value = 0.6
    mr.inputs['To Min'].default_value = 0.08
    mr.inputs['To Max'].default_value = 0.45
    nt.links.new(nz.outputs['Factor'], mr.inputs['Value'])
    nt.links.new(mr.outputs['Result'], bs.inputs['Roughness'])
    g.data.materials.append(m)
    return g


def add_camera(loc, target, lens=50):
    cam = bpy.data.cameras.new('cam')
    cam.lens = lens
    co = link(bpy.data.objects.new('cam', cam))
    co.location = loc
    d = Vector(target) - Vector(loc)
    co.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = co
    return co


def add_sun(strength=1.2, direction=(-0.4, 0.5, -0.75)):
    s = bpy.data.lights.new('sun', 'SUN')
    s.energy = strength
    s.angle = math.radians(12)
    so = link(bpy.data.objects.new('sun', s))
    so.rotation_euler = Vector(direction).to_track_quat('-Z', 'Y').to_euler()
    return so


def render(path, w=1600, h=900):
    sc = bpy.context.scene
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    log('rendered', path)


def rest_on_ground(obj, x, y, rotz=0.0):
    obj.rotation_euler = (0, 0, rotz)
    bpy.context.view_layer.update()
    zs = [(obj.matrix_world @ v.co).z for v in obj.data.vertices]
    obj.location = (x, y, -min(zs) - 0.02)


def lineup(objs, spacing=2.5, cols=5):
    for i, o in enumerate(objs):
        r, c = divmod(i, cols)
        rest_on_ground(o, (c - (cols - 1) / 2) * spacing, r * spacing * 1.1, rotz=0.4 * i)


# ------------------------------------------------------------------------------------------------
# stage: shapes (clay render of high polys)
# ------------------------------------------------------------------------------------------------
def clay_material():
    m = bpy.data.materials.new('clay')
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes['Principled BSDF']
    at = nt.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'rk_a'
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    nt.links.new(at.outputs['Color'], sep.inputs['Color'])
    comb = nt.nodes.new('ShaderNodeCombineColor')
    # fresh -> warm light, cavity -> dark, others grey
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.inputs['A'].default_value = (0.22, 0.22, 0.23, 1)
    mix.inputs['B'].default_value = (0.5, 0.45, 0.38, 1)
    nt.links.new(sep.outputs['Red'], mix.inputs['Factor'])
    mul = nt.nodes.new('ShaderNodeMix')
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.inputs['B'].default_value = (0.25, 0.25, 0.25, 1)
    nt.links.new(sep.outputs['Blue'], mul.inputs['Factor'])
    nt.links.new(mix.outputs['Result'], mul.inputs['A'])
    nt.links.new(mul.outputs['Result'], bs.inputs['Base Color'])
    bs.inputs['Roughness'].default_value = 0.7
    return m


def stage_shapes(specs):
    setup_cycles(32)
    setup_world(0.6)
    wet_ground()
    add_sun(0.9)
    clay = clay_material()
    objs = []
    for sp in specs:
        hp = build_highpoly(sp, ARGS['voxel'] if sp['kind'] == 'rock' else ARGS['voxel'] * 2.5)
        hp.data.materials.append(clay)
        objs.append(hp)
    rocks = [o for o, s in zip(objs, specs) if s['kind'] == 'rock']
    pebs = [o for o, s in zip(objs, specs) if s['kind'] == 'pebble']
    lineup(rocks)
    for i, o in enumerate(pebs):
        rest_on_ground(o, (i - 2.5) * 2.5, -3.0, rotz=0.7 * i)
    add_camera((0, -12.5, 5.6), (0, 1.5, 0.3), lens=38)
    render(os.path.join(SCRATCH, 'clay_lineup.png'))


# ------------------------------------------------------------------------------------------------
# 3. low poly, rest orientation, normalisation, hull
# ------------------------------------------------------------------------------------------------
def tri_count(me):
    return sum(len(pl.vertices) - 2 for pl in me.polygons)


def triangulate(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()


def build_lowpoly(hp, spec):
    target = 2900 if spec['kind'] == 'rock' else 290
    limit = 3000 if spec['kind'] == 'rock' else 300
    me = hp.data.copy()
    for a in list(me.color_attributes):
        me.color_attributes.remove(a)
    me.name = spec['name']
    lp = link(bpy.data.objects.new(spec['name'], me))
    ratio = target / tri_count(me)
    for attempt in range(5):
        dec = lp.modifiers.new('dec', 'DECIMATE')
        dec.decimate_type = 'COLLAPSE'
        dec.ratio = min(1.0, ratio)
        dec.use_collapse_triangulate = True
        apply_modifiers(lp)
        triangulate(lp.data)
        n = tri_count(lp.data)
        if n <= limit:
            break
        ratio = (target - 30 * attempt) / n
    # smooth shading BEFORE uv/bake/export: the tangent-space bake and the runtime (MikkTSpace tangents from the
    # exported normals) must see the same interpolated normals; flat shading also triples the vertex count
    lp.data.shade_smooth()
    for nm in ('sharp_edge', 'sharp_face'):
        if nm in lp.data.attributes:
            lp.data.attributes.remove(lp.data.attributes[nm])
    log(spec['name'], 'low-poly', tri_count(lp.data), 'tris')
    return lp


def tri_arrays(me):
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    t = np.empty(nt * 3, np.int64)
    me.loop_triangles.foreach_get('vertices', t)
    co, _ = mesh_arrays(me)
    return co, t.reshape(-1, 3)


def volume_centroid(me):
    co, t = tri_arrays(me)
    a, b, c = co[t[:, 0]], co[t[:, 1]], co[t[:, 2]]
    v = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
    V = v.sum()
    C = (v[:, None] * (a + b + c) / 4.0).sum(0) / V
    return C, V


def hull_faces(points):
    bm = bmesh.new()
    for pnt in points:
        bm.verts.new(Vector(pnt))
    res = bmesh.ops.convex_hull(bm, input=bm.verts[:])
    dead = {g for g in res['geom_interior'] + res['geom_unused'] if isinstance(g, bmesh.types.BMVert)}
    if dead:
        bmesh.ops.delete(bm, geom=list(dead), context='VERTS')
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    return bm


def orient_and_normalise(hp, lp, spec):
    """Rest the block on its largest flat hull face (-Z), random spin, centroid -> origin, radius -> 1."""
    rng = np.random.RandomState(spec['seed'] + 99)
    co, _ = mesh_arrays(lp.data)
    bm = hull_faces(co)
    ns = np.array([np.array(f.normal) for f in bm.faces])
    ar = np.array([f.calc_area() for f in bm.faces])
    bm.free()
    score = ((ns @ ns.T) > 0.965).astype(np.float64) @ ar
    best = ns[np.argmax(score)]
    rot = Vector(best).rotation_difference(Vector((0, 0, -1))).to_matrix()
    spin = Matrix.Rotation(rng.uniform(0, 2 * math.pi), 3, 'Z')
    R = np.array(spin @ rot)
    C, V = volume_centroid(lp.data)
    lco = (co - C) @ R.T
    r = np.sqrt((lco ** 2).sum(1)).max()
    set_coords(lp.data, lco / r)
    hco, _ = mesh_arrays(hp.data)
    set_coords(hp.data, ((hco - C) @ R.T) / r)
    C2, V2 = volume_centroid(lp.data)
    lco = lco / r - C2
    set_coords(lp.data, lco)
    hco, _ = mesh_arrays(hp.data)
    set_coords(hp.data, hco - C2)
    r2 = np.sqrt((lco ** 2).sum(1)).max()
    hp['zmin'] = float(lco[:, 2].min())
    lp['kind'] = spec['kind']
    lp['radius'] = round(float(r2), 4)
    lp['volume'] = round(float(V2), 5)
    log(spec['name'], 'normalised: radius %.4f volume %.4f (sphere %.3f)' % (r2, V2, 4.18879))
    return {'C': C, 'R': R, 'r': r, 'C2': C2, 'zmin': hp['zmin']}


def free_object(o):
    me = o.data
    bpy.data.objects.remove(o)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def rebuild_highpoly(spec, xf):
    """Re-create the (deterministic) high poly in the low poly's normalised frame. Only one high poly is alive
    at a time during baking, which keeps the full build well under the memory budget."""
    vox = ARGS['voxel'] if spec['kind'] == 'rock' else ARGS['voxel'] * 2.5
    hp = build_highpoly(spec, vox)
    hco, _ = mesh_arrays(hp.data)
    set_coords(hp.data, ((hco - xf['C']) @ xf['R'].T) / xf['r'] - xf['C2'])
    hp['zmin'] = xf['zmin']
    return hp


def make_hull(lp, name, maxv=48):
    co, _ = mesh_arrays(lp.data)
    best = None
    for K in range(160, 12, -4):
        i = np.arange(K) + 0.5
        phi = np.arccos(1 - 2 * i / K)
        th = math.pi * (1 + 5 ** 0.5) * i
        dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
        idx = np.unique(np.argmax(co @ dirs.T, axis=0))
        if len(idx) <= maxv:
            best = co[idx]
            break
    bm = hull_faces(best)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = link(bpy.data.objects.new(name, me))
    pts = np.array([v.co[:] for v in me.vertices])
    ob['collider'] = 1
    ob['points'] = [round(float(x), 5) for x in np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], 1).ravel()]  # three.js Y-up
    log(name, len(me.vertices), 'hull verts', len(me.polygons), 'faces')
    return ob


# ------------------------------------------------------------------------------------------------
# 4. shared UV atlas
# ------------------------------------------------------------------------------------------------
def mark_chart_seams(obj, K, min_frac):
    """Segment the low poly into K normal-direction charts (spherical k-means), clean speckles, merge small
    components into their neighbours and mark seams on chart borders. Few large charts -> dense packing,
    seams fall on geometric edges where they are least visible."""
    me = obj.data
    nf = len(me.polygons)
    fn = np.empty(nf * 3)
    me.polygons.foreach_get('normal', fn)
    fn = fn.reshape(-1, 3)
    fa = np.empty(nf)
    me.polygons.foreach_get('area', fa)
    ls = np.empty(nf, np.int64)
    me.polygons.foreach_get('loop_start', ls)
    lt = np.empty(nf, np.int64)
    me.polygons.foreach_get('loop_total', lt)
    le = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get('edge_index', le)
    lf = np.repeat(np.arange(nf), lt)
    order = np.argsort(le, kind='stable')
    se, sf = le[order], lf[order]
    pair = np.nonzero(se[1:] == se[:-1])[0]
    adj_e = se[pair]
    A, Bf = sf[pair], sf[pair + 1]
    nbrs = [[] for _ in range(nf)]
    for a, b in zip(A, Bf):
        nbrs[a].append(b)
        nbrs[b].append(a)
    # spherical k-means (area weighted), farthest-point init
    C = [fn[np.argmax(fa)]]
    for k in range(1, K):
        C.append(fn[np.argmin(np.max(fn @ np.array(C).T, axis=1))])
    C = np.array(C)
    for it in range(25):
        lab = np.argmax(fn @ C.T, axis=1)
        for k in range(K):
            m = lab == k
            if m.any():
                v = (fn[m] * fa[m, None]).sum(0)
                C[k] = v / max(np.linalg.norm(v), 1e-9)
    # majority smoothing
    for it in range(4):
        new = lab.copy()
        for f in range(nf):
            nl = [lab[g] for g in nbrs[f]]
            for l in set(nl):
                if l != lab[f] and nl.count(l) >= 2:
                    new[f] = l
        lab = new
    # connected components
    comp = -np.ones(nf, np.int64)
    nc = 0
    for f in range(nf):
        if comp[f] >= 0:
            continue
        stack = [f]
        comp[f] = nc
        while stack:
            g = stack.pop()
            for h in nbrs[g]:
                if comp[h] < 0 and lab[h] == lab[f]:
                    comp[h] = nc
                    stack.append(h)
        nc += 1
    total = fa.sum()
    # merge small components into the neighbour sharing the longest border
    while True:
        areas = np.bincount(comp, fa, minlength=comp.max() + 1)
        live = [c for c in np.unique(comp)]
        small = [c for c in live if areas[c] < min_frac * total]
        if not small or len(live) <= 1:
            break
        c = min(small, key=lambda c: areas[c])
        border = {}
        for f in np.nonzero(comp == c)[0]:
            for h in nbrs[f]:
                if comp[h] != c:
                    border[comp[h]] = border.get(comp[h], 0) + 1
        if not border:
            break
        tgt = max(border, key=border.get)
        comp[comp == c] = tgt
    seam = np.zeros(len(me.edges), bool)
    seam[adj_e[comp[A] != comp[Bf]]] = True
    me.edges.foreach_set('use_seam', seam)
    me.update()
    return len(np.unique(comp))


def face_components(me, use_seams=True):
    """Connected face components through manifold, non-seam edges (UV islands)."""
    nf = len(me.polygons)
    ls = np.empty(nf, np.int64)
    me.polygons.foreach_get('loop_start', ls)
    lt = np.empty(nf, np.int64)
    me.polygons.foreach_get('loop_total', lt)
    le = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get('edge_index', le)
    seam = np.zeros(len(me.edges), bool)
    if use_seams:
        me.edges.foreach_get('use_seam', seam)
    lf = np.repeat(np.arange(nf), lt)
    order = np.argsort(le, kind='stable')
    se, sf = le[order], lf[order]
    pair = np.nonzero((se[1:] == se[:-1]))[0]
    ok = ~seam[se[pair]]
    A, Bf = sf[pair][ok], sf[pair + 1][ok]
    parent = np.arange(nf)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(A, Bf):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    comp = np.array([find(f) for f in range(nf)])
    return comp, ls, lt


def hull2d(P):
    pts = sorted(set(map(tuple, np.round(P, 9))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def rot2(P, ang):
    c, s_ = math.cos(ang), math.sin(ang)
    return P @ np.array([[c, s_], [-s_, c]])


def min_rect_angle(P):
    H = hull2d(P)
    best, ba = 1e18, 0.0
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        ang = -math.atan2(e[1], e[0])
        Q = rot2(H, ang)
        area = np.ptp(Q[:, 0]) * np.ptp(Q[:, 1])
        if area < best:
            best, ba = area, ang
    return ba


def raster_island(P, tris, h, w):
    M = np.zeros((h, w), bool)
    for t in tris:
        a, b, c = P[t[0]], P[t[1]], P[t[2]]
        lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        hi = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        xs = np.arange(max(lo[0], 0), min(hi[0], w)) + 0.5
        ys = np.arange(max(lo[1], 0), min(hi[1], h)) + 0.5
        if len(xs) and len(ys):
            X, Y = np.meshgrid(xs, ys)
            v0, v1 = b - a, c - a
            d = v0[0] * v1[1] - v0[1] * v1[0]
            if abs(d) > 1e-12:
                wx, wy = X - a[0], Y - a[1]
                b1 = (wx * v1[1] - wy * v1[0]) / d
                b2 = (v0[0] * wy - v0[1] * wx) / d
                ins = (b1 >= -0.05) & (b2 >= -0.05) & (b1 + b2 <= 1.05)
                M[Y[ins].astype(int), X[ins].astype(int)] = True
    V = np.clip(np.floor(P).astype(int), 0, [w - 1, h - 1])
    M[V[:, 1], V[:, 0]] = True
    return M


def dilate(M, r):
    for _ in range(r):
        D = M.copy()
        D[1:, :] |= M[:-1, :]
        D[:-1, :] |= M[1:, :]
        D[:, 1:] |= M[:, :-1]
        D[:, :-1] |= M[:, 1:]
        D[1:, 1:] |= M[:-1, :-1]
        D[:-1, :-1] |= M[1:, 1:]
        D[1:, :-1] |= M[:-1, 1:]
        D[:-1, 1:] |= M[1:, :-1]
        M = D
    return M


def pack_raster(lows, grid=1024, dil=2, fill0=0.7):
    """Deterministic raster packer: min-area-rect aligned islands, 4 rotations, FFT overlap test,
    bottom-left placement, shrink-and-retry on failure. Gap between islands >= 2*dil cells."""
    uvs, islands = [], []
    for oi, o in enumerate(lows):
        me = o.data
        a = np.empty(len(me.loops) * 2)
        me.uv_layers.active.data.foreach_get('uv', a)
        a = a.reshape(-1, 2)
        uvs.append(a)
        comp, ls, lt = face_components(me)
        for c in np.unique(comp):
            faces = np.nonzero(comp == c)[0]
            loops = (ls[faces][:, None] + np.arange(3)[None, :])
            P = a[loops.ravel()]
            ang = min_rect_angle(P)
            Q = rot2(P, ang)
            e = Q.reshape(-1, 3, 2)
            area = np.abs((e[:, 1, 0] - e[:, 0, 0]) * (e[:, 2, 1] - e[:, 0, 1]) - (e[:, 2, 0] - e[:, 0, 0]) * (e[:, 1, 1] - e[:, 0, 1])).sum() / 2
            islands.append({'o': oi, 'loops': loops.ravel(), 'ang': ang, 'area': area, 'dim': np.ptp(Q, 0).max()})
    total = sum(I['area'] for I in islands)
    order = sorted(islands, key=lambda I: -I['dim'])
    fill = fill0
    while True:
        k = math.sqrt(fill * grid * grid / total)
        O = np.zeros((grid, grid), np.float64)
        placed = []
        ok = True
        for I in order:
            FO = np.fft.rfft2(O)
            best = None
            base = rot2(uvs[I['o']][I['loops']], I['ang']) * k
            for r in range(4):
                P = rot2(base, r * math.pi / 2)
                mn = P.min(0) - dil - 0.5
                P = P - mn
                w = int(math.ceil(P[:, 0].max() + dil + 0.5))
                h = int(math.ceil(P[:, 1].max() + dil + 0.5))
                if w > grid or h > grid:
                    continue
                M = dilate(raster_island(P, np.arange(len(P)).reshape(-1, 3), h, w), dil)
                Mp = np.zeros((grid, grid))
                Mp[:h, :w] = M
                corr = np.fft.irfft2(FO * np.conj(np.fft.rfft2(Mp)), s=(grid, grid))
                valid = corr[:grid - h + 1, :grid - w + 1] < 0.5
                if not valid.any():
                    continue
                ys, xs = np.nonzero(valid)
                sc = (ys + h) * grid * 4 + xs + (ys * 0)
                j = np.argmin(sc)
                cand = (sc[j], r, xs[j], ys[j], h, w, M, mn)
                if best is None or cand[0] < best[0]:
                    best = cand
            if best is None:
                ok = False
                break
            _, r, x, y, h, w, M, mn = best
            O[y:y + h, x:x + w] += M
            placed.append((I, r, x, y, mn))
        if ok:
            break
        fill *= 0.96
        log('pack retry, fill -> %.3f' % fill)
    for I, r, x, y, mn in placed:
        P = rot2(rot2(uvs[I['o']][I['loops']], I['ang']) * k, r * math.pi / 2) - mn + np.array([x, y])
        uvs[I['o']][I['loops']] = P / grid
    # stretch the packed block to use the whole square (uniform scale keeps texel density equal)
    lo = np.min([a.min(0) for a in uvs], axis=0)
    hi = np.max([a.max(0) for a in uvs], axis=0)
    m = (dil + 0.5) / grid
    sc = (1 - 2 * m) / (hi - lo).max()
    for a in uvs:
        a[:] = (a - lo) * sc + m
    for o, a in zip(lows, uvs):
        o.data.uv_layers.active.data.foreach_set('uv', a.ravel())
    log('packed %d islands at fill %.3f' % (len(islands), fill))


def uv_atlas(lows, pebble_scale=0.3):
    for o in lows:
        n = mark_chart_seams(o, 9 if o.get('kind') == 'rock' else 6, 0.02 if o.get('kind') == 'rock' else 0.05)
        log(o.name, n, 'uv charts')
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in lows:
        o.select_set(True)
    bpy.context.view_layer.objects.active = lows[0]
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    bpy.ops.uv.average_islands_scale()
    bpy.ops.object.mode_set(mode='OBJECT')
    for o in lows:
        if o.get('kind') == 'pebble':
            uv = o.data.uv_layers.active.data
            a = np.empty(len(uv) * 2)
            uv.foreach_get('uv', a)
            uv.foreach_set('uv', a * pebble_scale)
    pack_raster(lows)
    # report utilisation
    tot = 0.0
    for o in lows:
        uv = o.data.uv_layers.active.data
        a = np.empty(len(uv) * 2)
        uv.foreach_get('uv', a)
        a = a.reshape(-1, 2)
        o.data.calc_loop_triangles()
        for lt in o.data.loop_triangles:
            p0, p1, p2 = a[lt.loops[0]], a[lt.loops[1]], a[lt.loops[2]]
            tot += abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2
    log('uv atlas utilisation %.1f%%' % (tot * 100))


# ------------------------------------------------------------------------------------------------
# 5. high-poly look-dev shader (albedo / roughness / AO / micro height), evaluated only for baking
# ------------------------------------------------------------------------------------------------
_IMGS = {}


def teximg(name, kind, noncolor):
    key = (name, kind)
    if key not in _IMGS:
        im = bpy.data.images.load(os.path.join(TEX, name, kind + '.jpg'), check_existing=True)
        im.colorspace_settings.name = 'Non-Color' if noncolor else 'sRGB'
        _IMGS[key] = im
    return _IMGS[key]


def _in(node, ident):
    for s in node.inputs:
        if s.identifier == ident:
            return s
    return node.inputs[ident]


class NB:
    """Tiny node-graph builder. Arguments may be sockets or constants."""

    def __init__(self, nt):
        self.nt = nt

    def new(self, t, **kw):
        n = self.nt.nodes.new(t)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def put(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        elif v is not None:
            if isinstance(v, (int, float)) and hasattr(sock, 'default_value') and hasattr(sock.default_value, '__len__'):
                n = len(sock.default_value)
                sock.default_value = [v] * n if n != 4 else (v, v, v, 1.0)
            else:
                sock.default_value = v

    def math(self, op, a, b=None, clamp=False):
        n = self.new('ShaderNodeMath', operation=op, use_clamp=clamp)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return n.outputs[0]

    def mix(self, fac, a, b, blend='MIX', clamp=True):
        n = self.new('ShaderNodeMix', data_type='RGBA', blend_type=blend, clamp_result=clamp)
        self.put(_in(n, 'Factor_Float'), fac)
        self.put(_in(n, 'A_Color'), a)
        self.put(_in(n, 'B_Color'), b)
        return n.outputs[2]

    def mixf(self, fac, a, b):
        n = self.new('ShaderNodeMix', data_type='FLOAT', clamp_factor=True)
        self.put(_in(n, 'Factor_Float'), fac)
        self.put(_in(n, 'A_Float'), a)
        self.put(_in(n, 'B_Float'), b)
        return n.outputs[0]

    def maprange(self, v, fmin, fmax, tmin=0.0, tmax=1.0, smooth=True):
        n = self.new('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR', clamp=True)
        self.put(n.inputs['Value'], v)
        n.inputs['From Min'].default_value = fmin
        n.inputs['From Max'].default_value = fmax
        n.inputs['To Min'].default_value = tmin
        n.inputs['To Max'].default_value = tmax
        return n.outputs[0]

    def hsv(self, col, h=0.5, s=1.0, v=1.0):
        n = self.new('ShaderNodeHueSaturation')
        n.inputs['Hue'].default_value = h
        n.inputs['Saturation'].default_value = s
        n.inputs['Value'].default_value = v
        self.put(n.inputs['Color'], col)
        return n.outputs[0]

    def mapping(self, vec, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0):
        n = self.new('ShaderNodeMapping')
        self.put(n.inputs['Vector'], vec)
        n.inputs['Location'].default_value = loc
        n.inputs['Rotation'].default_value = rot
        n.inputs['Scale'].default_value = (scale, scale, scale) if not hasattr(scale, '__len__') else scale
        return n.outputs[0]

    def img(self, image, vec, blend=0.3):
        n = self.new('ShaderNodeTexImage', projection='BOX', projection_blend=blend, interpolation='Cubic')
        n.image = image
        self.put(n.inputs['Vector'], vec)
        return n

    def noise(self, vec, scale, detail=4.0, rough=0.55, distortion=0.0):
        n = self.new('ShaderNodeTexNoise')
        self.put(n.inputs['Vector'], vec)
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        n.inputs['Roughness'].default_value = rough
        n.inputs['Distortion'].default_value = distortion
        return n

    def sep(self, col):
        n = self.new('ShaderNodeSeparateColor')
        self.put(n.inputs['Color'], col)
        return n.outputs

    def comb(self, r, g, b):
        n = self.new('ShaderNodeCombineColor')
        self.put(n.inputs['Red'], r)
        self.put(n.inputs['Green'], g)
        self.put(n.inputs['Blue'], b)
        return n.outputs[0]


def hp_material(spec, zmin=-0.6):
    """Returns (material, {albedo, data, bsdf}) sockets for switching the bake pass."""
    rng = np.random.RandomState(spec['seed'] + 404)
    m = bpy.data.materials.new(spec['name'] + '_hp')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    B = NB(nt)
    out = B.new('ShaderNodeOutputMaterial')
    tc = B.new('ShaderNodeTexCoord')
    rock = spec['kind'] == 'rock'
    s_real = 0.75 if rock else 0.22                      # metres per normalised unit
    age = spec['age']
    lith = spec['lith']
    rng.uniform(0, 2 * math.pi, 3)      # (kept: preserves the seeded sequence below)
    # NO rotation here: BOX projection picks its axis from the (unrotated) object-space normal, so rotating the
    # coordinates projects the scans along the wrong axis (long stretched streaks). Variety comes from the offset.
    Pm = B.mapping(tc.outputs['Object'], loc=tuple(rng.uniform(-20, 20, 3)), scale=s_real)   # metres
    Nz = B.sep(tc.outputs['Normal'])[2]
    Pz = B.sep(tc.outputs['Object'])[2]
    atA = B.new('ShaderNodeAttribute', attribute_name='rk_a')
    atB = B.new('ShaderNodeAttribute', attribute_name='rk_b')
    fresh, edge, cav = B.sep(atA.outputs['Color'])
    crack, layer, semi = B.sep(atB.outputs['Color'])
    vein = atB.outputs['Alpha']

    def P(tile):          # texture space for a scan with a real-world tile size in metres
        return B.mapping(Pm, scale=1.0 / tile)

    # ---- weathered joint faces (the old rind that was exposed on the cliff / along open joints). Three scan families,
    # blended by large-scale noise so no two faces match: lichen-crusted grey (mossy_rock), rust-brown iron-stained
    # joint coating (quarry_wall) and the dark lichen-spotted rock of the road cut the blocks fell from (lichen_rock).
    Wl = B.hsv(B.img(teximg('mossy_rock', 'diffuse', False), P(3.0)).outputs['Color'], s=0.5, v=0.62)
    Wr = B.hsv(B.img(teximg('quarry_wall', 'diffuse', False), P(1.6)).outputs['Color'], s=0.62, v=1.15)
    Wc = B.hsv(B.img(teximg('lichen_rock', 'diffuse', False), P(2.0)).outputs['Color'], s=0.55, v=1.12)
    rust_w = float(rng.uniform(0.15, 0.6) if lith != 'schist' else 0.3)
    dark_w = float(rng.uniform(0.1, 0.55))
    n1 = B.maprange(B.noise(Pm, 0.9, 4.0, 0.55).outputs['Factor'], 0.5 - 0.12, 0.5 + 0.12)
    n2 = B.maprange(B.noise(Pm, 1.3, 4.0, 0.55).outputs['Factor'], 0.44, 0.6)
    W = B.mix(B.math('MULTIPLY', n1, rust_w * 1.6, clamp=True), Wl, Wr)
    W = B.mix(B.math('MULTIPLY', n2, dark_w * 1.6, clamp=True), W, Wc)
    # less lichen on younger rocks: fade the lichen scan toward its own desaturated rind
    W = B.mix(float(np.clip(0.75 - age, 0.0, 0.6)), W, B.hsv(W, s=0.3, v=1.0))
    if lith == 'schist':
        Ds = B.img(teximg('dark_rock_02', 'diffuse', False), P(2.0)).outputs['Color']
        W = B.mix(0.65, W, B.hsv(Ds, s=0.6, v=1.35))
    if lith in ('gneiss', 'schist'):
        # compositional banding (tonal): dark biotite/hornblende-rich vs light quartz-feldspar layers, muted by the
        # weathering rind on old faces
        bd = B.maprange(layer, 0.3, 0.12)
        bl = B.maprange(layer, 0.68, 0.88)
        W = B.mix(B.math('MULTIPLY', bd, 0.7), W, B.hsv(W, s=0.8, v=0.5))
        W = B.mix(B.math('MULTIPLY', bl, 0.55), W, B.hsv(W, s=0.7, v=1.4))
    # ---- fresh fracture faces: clean mid-grey granular rock (rock_01 scan, desaturated) + crystal flecks
    Ft = B.img(teximg('rock_01', 'diffuse', False), P(1.5))
    F = B.hsv(Ft.outputs['Color'], s=0.36, v=1.45)   # fresh fracture: clearly lighter than the rind
    vor = B.new('ShaderNodeTexVoronoi')
    B.put(vor.inputs['Vector'], Pm)
    vor.inputs['Scale'].default_value = 48.0                             # ~2 cm cells: coarse feldspar / biotite clots
    # (finer grains are below the atlas texel size (~6 mm) and would only average into grey mush)
    grain = B.sep(vor.outputs['Color'])[0]
    dark = B.maprange(grain, 0.2, 0.12)
    white = B.maprange(grain, 0.78, 0.88)
    F = B.mix(B.math('MULTIPLY', dark, 0.55), F, B.hsv(F, s=0.5, v=0.35))       # biotite / hornblende
    F = B.mix(B.math('MULTIPLY', white, 0.45), F, (0.4, 0.39, 0.36, 1))          # feldspar
    vor2 = B.new('ShaderNodeTexVoronoi')
    B.put(vor2.inputs['Vector'], B.mapping(Pm, loc=(1.7, 2.9, 0.3)))
    vor2.inputs['Scale'].default_value = 230.0                              # ~4 mm: fine salt-and-pepper
    g2 = B.sep(vor2.outputs['Color'])[1]
    F = B.mix(B.math('MULTIPLY', B.maprange(g2, 0.22, 0.14), 0.6), F, B.hsv(F, s=0.5, v=0.3))
    F = B.mix(B.math('MULTIPLY', B.maprange(g2, 0.8, 0.9), 0.35), F, B.hsv(F, s=0.4, v=1.35))
    if lith in ('gneiss', 'schist'):
        # fresh faces show the banding clearly
        bdf = B.maprange(layer, 0.32, 0.1)
        blf = B.maprange(layer, 0.66, 0.86)
        F = B.mix(B.math('MULTIPLY', bdf, 0.85), F, B.hsv(F, s=0.6, v=0.38))
        F = B.mix(B.math('MULTIPLY', blf, 0.7), F, B.hsv(F, s=0.5, v=1.3))
    if lith == 'schist':
        F = B.hsv(F, s=0.8, v=0.72)
    # fresh faces pick up a thin film of rock flour / splashed silt in places
    film = B.maprange(B.noise(Pm, 2.0, 5.0, 0.6).outputs['Factor'], 0.5, 0.7, 0.0, 0.35)
    F = B.mix(film, F, (0.2, 0.17, 0.13, 1))
    base = B.mix(fresh, W, F)
    base = B.mix(B.math('MULTIPLY', semi, 0.8), base, B.hsv(B.mix(0.5, W, F), s=0.8, v=0.9))
    # ---- milky quartz / aplite veins (pale; stained on old faces), sugary texture
    vgr = B.maprange(B.noise(Pm, 120.0, 2.0, 0.5).outputs['Factor'], 0.3, 0.7, 0.85, 1.1)
    vcol = B.mix(fresh, (0.2, 0.19, 0.17, 1), (0.36, 0.35, 0.33, 1))
    vcol = B.mix(1.0, vcol, B.comb(vgr, vgr, vgr), 'MULTIPLY')
    base = B.mix(B.math('MULTIPLY', vein, 0.7), base, vcol)
    # ---- iron-oxide staining bleeding out of cracks and along old joint faces
    rn = B.noise(Pm, 2.4, 6.0, 0.62, 0.3)
    rmask = B.maprange(rn.outputs['Factor'], 0.52, 0.7)
    rmask = B.math('MULTIPLY', rmask, B.math('SUBTRACT', 1.0, fresh))
    rmask = B.math('ADD', rmask, B.math('MULTIPLY', crack, 0.6), clamp=True)
    base = B.mix(B.math('MULTIPLY', rmask, 0.35), base, B.mix(1.0, base, (0.95, 0.6, 0.36, 1), 'MULTIPLY'))
    # ---- rain/seepage streaks on the steep faces: dark, slightly glossy runs from the top, patchy
    stv = B.mapping(Pm, loc=(5.3, 1.1, 0.0), scale=(7.0, 7.0, 0.55))
    streak = B.maprange(B.noise(stv, 1.0, 4.0, 0.6, 0.2).outputs['Factor'], 0.52, 0.68)
    side = B.maprange(B.math('ABSOLUTE', Nz), 0.75, 0.35)
    streak = B.math('MULTIPLY', B.math('MULTIPLY', streak, side), B.math('SUBTRACT', 1.0, B.math('MULTIPLY', fresh, 0.6)))
    base = B.mix(B.math('MULTIPLY', streak, 0.55), base, B.mix(1.0, base, (0.62, 0.6, 0.56, 1), 'MULTIPLY'))
    # macro value variation (0.3-1 m blotches): no two faces with the same tone
    mac = B.maprange(B.noise(Pm, 0.8, 3.0, 0.5).outputs['Factor'], 0.35, 0.65, 0.68, 1.12)
    base = B.mix(1.0, base, B.comb(mac, mac, mac), 'MULTIPLY')
    # ---- edges: impact bruises and abraded arrises (lighter, whitish rock powder), patchy
    bruise = B.maprange(B.noise(Pm, 6.0, 3.0, 0.6).outputs['Factor'], 0.45, 0.6)
    base = B.mix(B.math('MULTIPLY', B.math('MULTIPLY', edge, bruise), 0.45), base, B.hsv(base, s=0.45, v=1.45))
    # ---- dirt in cavities, dark crack cores
    dirt = B.hsv(B.img(teximg('brown_mud_rocks_01', 'diffuse', False), P(1.4)).outputs['Color'], s=0.85, v=0.5)
    dpatch = B.maprange(B.noise(Pm, 3.0, 4.0, 0.6).outputs['Factor'], 0.35, 0.65, 0.35, 1.0)   # dirt collects patchily
    base = B.mix(B.math('MULTIPLY', B.math('MULTIPLY', cav, 0.7), dpatch), base, dirt)
    # crack cores: a damp, dirty shadowed slot, not an inked line (the geometric fissure does most of the work)
    ckv = B.maprange(B.noise(Pm, 4.0, 3.0, 0.6).outputs['Factor'], 0.3, 0.7, 0.2, 0.45)
    base = B.mix(B.math('MULTIPLY', B.maprange(crack, 0.5, 0.95), ckv), base, (0.035, 0.028, 0.021, 1))
    # ---- crustose lichen rosettes on old exposed faces (map lichen: yellow-green with a black rim; pale grey
    # Lecanora/Aspicilia crusts). They are what makes a face read as 'long exposed' next to a fresh fracture.
    lich_amt = float(np.clip((age - 0.2) / 0.6, 0.0, 1.0)) * (1.0 if rock else 0.5)
    lich_mask = None
    if lich_amt > 0.0:
        lv = B.new('ShaderNodeTexVoronoi')
        B.put(lv.inputs['Vector'], B.mapping(Pm, loc=(3.1, 7.7, 1.3)))
        lv.inputs['Scale'].default_value = 6.5                       # ~15 cm cells: lichen patches read at 5-10 m
        lvc = B.sep(lv.outputs['Color'])
        rad = B.math('ADD', 0.1, B.math('MULTIPLY', lvc[0], 0.38))   # rosette radius varies per cell
        # lobed, ragged outline: two warp scales (lobes ~1 cm, fringe ~2 mm)
        warp = B.math('ADD', B.math('MULTIPLY', B.math('SUBTRACT', B.noise(Pm, 45.0, 2.0, 0.5).outputs['Factor'], 0.5), 0.5),
                      B.math('MULTIPLY', B.math('SUBTRACT', B.noise(Pm, 160.0, 2.0, 0.5).outputs['Factor'], 0.5), 0.22))
        dist = B.math('ADD', lv.outputs['Distance'], warp)          # ragged outline
        disc = B.math('LESS_THAN', dist, rad)
        rim = B.math('MULTIPLY', B.math('LESS_THAN', dist, B.math('ADD', rad, 0.035)), B.math('SUBTRACT', 1.0, disc))
        present = B.maprange(lvc[1], 0.62 - 0.3 * lich_amt, 0.7 - 0.3 * lich_amt)
        where = B.math('MULTIPLY', B.math('SUBTRACT', 1.0, fresh), B.math('SUBTRACT', 1.0, B.math('MULTIPLY', cav, 0.8)))
        where = B.math('MULTIPLY', where, B.maprange(Nz, -0.45, 0.2))  # not on the shaded underside
        lm = B.math('MULTIPLY', B.math('MULTIPLY', present, where), lich_amt, clamp=True)
        # areole texture inside the thallus
        areo = B.maprange(B.noise(Pm, 140.0, 2.0, 0.5).outputs['Factor'], 0.35, 0.65, 0.8, 1.15)
        map_c = B.mix(B.noise(Pm, 25.0, 2.0).outputs['Factor'], (0.13, 0.15, 0.035, 1), (0.2, 0.22, 0.06, 1))
        grey_c = B.mix(B.noise(Pm, 25.0, 2.0).outputs['Factor'], (0.12, 0.12, 0.11, 1), (0.19, 0.185, 0.17, 1))
        lcol = B.mix(B.maprange(lvc[2], 0.72, 0.76), grey_c, map_c)
        lcol = B.mix(1.0, lcol, B.comb(areo, areo, areo), 'MULTIPLY')
        is_map = B.maprange(lvc[2], 0.72, 0.76)
        base = B.mix(B.math('MULTIPLY', B.math('MULTIPLY', rim, is_map), B.math('MULTIPLY', lm, 0.85)), base, (0.02, 0.02, 0.018, 1))
        lich_mask = B.math('MULTIPLY', disc, lm)
        base = B.mix(lich_mask, base, lcol)
    # ---- moss on old, up-facing surfaces and crevices
    moss_amt = max(0.0, (age - 0.45) / 0.55) * (1.0 if rock else 0.3)
    up = B.maprange(Nz, 0.15, 0.85)
    mn = B.noise(Pm, 3.2, 8.0, 0.65)
    mm = B.maprange(mn.outputs['Factor'], 0.56, 0.64)
    moss = B.math('MULTIPLY', B.math('ADD', B.math('MULTIPLY', up, mm), B.math('MULTIPLY', cav, B.math('MULTIPLY', up, 0.8))), moss_amt, clamp=True)
    mcol = B.mix(B.noise(Pm, 22.0, 4.0).outputs['Factor'], (0.022, 0.034, 0.009, 1), (0.07, 0.085, 0.022, 1))
    base = B.mix(moss, base, mcol)
    # ---- mud: underside, low band, tumble smears and splash specks (arbitrary rotation still reads plausibly)
    mud_amt = float(np.clip(0.55 + (1 - age) * 0.45 + rng.uniform(-0.1, 0.1), 0, 1))
    down = B.maprange(Nz, -0.05, -0.75)
    low = B.maprange(Pz, zmin + 0.5, zmin + 0.05)
    mn2 = B.noise(Pm, 2.2, 6.0, 0.6)
    brk = B.maprange(mn2.outputs['Factor'], 0.36, 0.52)
    smear_v = B.mapping(Pm, rot=tuple(rng.uniform(0, 3.14, 3)), scale=(7.0, 1.2, 1.2))
    sm = B.maprange(B.noise(smear_v, 1.0, 5.0, 0.6).outputs['Factor'], 0.52, 0.62)
    spl = B.new('ShaderNodeTexVoronoi')
    B.put(spl.inputs['Vector'], Pm)
    spl.inputs['Scale'].default_value = 28.0
    spk = B.math('MULTIPLY', B.maprange(spl.outputs['Distance'], 0.22, 0.12),
                 B.maprange(B.sep(spl.outputs['Color'])[0], 0.55, 0.6))
    spk = B.math('MULTIPLY', spk, B.maprange(Pz, zmin + 0.9, zmin + 0.2))
    mud = B.math('ADD', B.math('MULTIPLY', B.math('MAXIMUM', down, B.math('MULTIPLY', low, 0.8)), brk),
                 B.math('ADD', B.math('MULTIPLY', sm, 0.7), B.math('MULTIPLY', cav, 0.55)))
    mud = B.math('ADD', mud, spk)
    mud = B.math('MULTIPLY', mud, mud_amt, clamp=True)
    # mud: a smooth brown film (wet, dark) drying to pale silt at its thin edges, only a little grit from the scan
    grit = B.hsv(B.img(teximg('brown_mud_rocks_01', 'diffuse', False), P(0.9)).outputs['Color'], s=0.9, v=0.9)
    mv = B.maprange(B.noise(Pm, 3.5, 5.0, 0.6).outputs['Factor'], 0.35, 0.65)
    mudc = B.mix(mv, (0.052, 0.038, 0.026, 1), (0.1, 0.074, 0.05, 1))
    mudc = B.mix(0.28, mudc, grit)
    thin = B.math('SUBTRACT', 1.0, B.maprange(mud, 0.25, 0.9))          # thin film at the mud margins -> silty
    mudc = B.mix(B.math('MULTIPLY', thin, 0.7), mudc, (0.2, 0.165, 0.125, 1))
    base = B.mix(mud, base, mudc)
    # per-rock tint / value variation (rocks from different outcrops)
    tint = (1.0 + rng.uniform(-0.04, 0.05), 1.0, 1.0 + rng.uniform(-0.06, 0.02), 1.0)
    albedo = B.hsv(B.mix(1.0, base, tint, 'MULTIPLY'), s=1.08, v=float(rng.uniform(0.8, 0.97)))
    # power contrast around a linear pivot of 0.1 (BrightContrast pivots at 0.5 and crushes dark rock to black)
    gm = B.new('ShaderNodeGamma')
    B.put(gm.inputs['Color'], albedo)
    gm.inputs['Gamma'].default_value = 1.28
    k = 0.1 ** (1.0 - 1.28)
    albedo = B.mix(1.0, gm.outputs[0], (k, k, k, 1), 'MULTIPLY', clamp=False)
    # ---- roughness (natural, dry-to-damp; runtime wetness lowers it further)
    rW = B.sep(B.img(teximg('mossy_rock', 'arm', True), P(3.0)).outputs['Color'])[1]
    rF = B.sep(B.img(teximg('rock_01', 'arm', True), P(1.5)).outputs['Color'])[1]
    rough = B.mixf(fresh, rW, rF)
    rough = B.math('MULTIPLY', rough, 0.95)
    rough = B.mixf(moss, rough, 0.9)
    rough = B.mixf(B.math('MULTIPLY', vein, 0.6), rough, 0.55)
    if lich_mask is not None:
        rough = B.mixf(lich_mask, rough, 0.88)
    rough = B.mixf(mud, rough, 0.62)
    rough = B.mixf(B.math('MULTIPLY', streak, 0.5), rough, 0.5)
    rough = B.mixf(B.math('MULTIPLY', cav, 0.5), rough, 0.55)
    # ---- AO (local) x cavity
    ao = B.new('ShaderNodeAmbientOcclusion', only_local=True, samples=24)
    ao.inputs['Distance'].default_value = 0.3
    aof = B.math('MULTIPLY', ao.outputs['AO'], B.math('SUBTRACT', 1.0, B.math('MULTIPLY', cav, 0.3)), clamp=True)
    data = B.comb(aof, rough, 0.0)
    # ---- micro height for the normal bake
    hW = B.sep(B.img(teximg('mossy_rock', 'displacement', True), P(3.0)).outputs['Color'])[0]
    hQ = B.sep(B.img(teximg('quarry_wall', 'displacement', True), P(1.6)).outputs['Color'])[0]
    hF = B.sep(B.img(teximg('rock_01', 'displacement', True), P(1.5)).outputs['Color'])[0]
    grains = B.math('MULTIPLY', B.maprange(grain, 0.0, 1.0, 0.0, 1.0, smooth=False), 0.18)
    hW = B.mixf(B.math('MULTIPLY', n1, rust_w * 1.6, clamp=True), hW, hQ)
    # fresh fractures: hackly relief (quarry scan) + granular texture + crystal cleavage steps
    # (mostly granular at crystal scale; the quarry scan's lumpy relief only faintly, else it reads as hammered metal)
    grain2 = B.noise(Pm, 260.0, 2.0, 0.5).outputs['Factor']
    h = B.mixf(fresh, hW, B.math('ADD', B.math('ADD', B.math('MULTIPLY', hQ, 0.22), B.math('MULTIPLY', hF, 0.4)),
                                 B.math('ADD', grains, B.math('MULTIPLY', grain2, 0.2))))
    fine = B.noise(Pm, 90.0, 3.0, 0.6).outputs['Factor']
    h = B.math('ADD', h, B.math('MULTIPLY', fine, 0.12))
    h = B.math('ADD', h, B.math('MULTIPLY', B.noise(Pm, 40.0, 4.0).outputs['Factor'], B.math('MULTIPLY', moss, 0.6)))
    h = B.mixf(B.math('MULTIPLY', mud, 0.8), h, B.math('ADD', 0.4, B.math('MULTIPLY', B.noise(Pm, 9.0, 2.0).outputs['Factor'], 0.2)))
    bump = B.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 1.0
    bump.inputs['Distance'].default_value = 0.016 / s_real * 1.1
    B.put(bump.inputs['Height'], h)
    bsdf = B.new('ShaderNodeBsdfDiffuse')
    B.put(bsdf.inputs['Normal'], bump.outputs['Normal'])
    emi = B.new('ShaderNodeEmission')
    emi.inputs['Strength'].default_value = 1.0
    return m, {'out': out, 'albedo': albedo, 'data': data, 'bsdf': bsdf, 'emi': emi, 'nt': nt}


def set_pass(h, which):
    nt = h['nt']
    for l in list(nt.links):
        if l.to_node == h['out'] or l.to_node == h['emi']:
            nt.links.remove(l)
    if which == 'NORMAL':
        nt.links.new(h['bsdf'].outputs[0], h['out'].inputs['Surface'])
    else:
        nt.links.new(h[which], h['emi'].inputs['Color'])
        nt.links.new(h['emi'].outputs[0], h['out'].inputs['Surface'])


# ------------------------------------------------------------------------------------------------
# 6. baking + atlas compositing
# ------------------------------------------------------------------------------------------------
def float_image(name, size):
    im = bpy.data.images.new(name, size, size, alpha=True, float_buffer=True)
    im.colorspace_settings.name = 'Non-Color'
    return im


def read_image(im):
    a = np.empty(im.size[0] * im.size[1] * 4, np.float32)
    im.pixels.foreach_get(a)
    return a.reshape(im.size[1], im.size[0], 4)


def clear_image(im):
    im.pixels.foreach_set(np.zeros(im.size[0] * im.size[1] * 4, np.float32))


def bake_all(pairs, size, samples):
    sc = bpy.context.scene
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.render.bake.margin = 0
    tgt_mat = bpy.data.materials.new('bake_target')
    tgt_mat.use_nodes = True
    tnode = tgt_mat.node_tree.nodes.new('ShaderNodeTexImage')
    tgt_mat.node_tree.nodes.active = tnode
    ims = {k: float_image('bk_' + k, size) for k in ('NORMAL', 'albedo', 'data')}
    acc = {k: np.zeros((size, size, 4), np.float32) for k in ims}
    cover = np.zeros((size, size), bool)
    for sp, xf, lp in pairs:
        t0 = time.time()
        hp = rebuild_highpoly(sp, xf)
        mat, h = hp_material(sp, hp.get('zmin', -0.6))
        hp.data.materials.clear()
        hp.data.materials.append(mat)
        lp.data.materials.clear()
        lp.data.materials.append(tgt_mat)
        for o in bpy.context.scene.objects:
            o.select_set(False)
            o.hide_render = True
        hp.hide_render = False
        lp.hide_render = False
        hp.select_set(True)
        lp.select_set(True)
        bpy.context.view_layer.objects.active = lp
        for k, im in ims.items():
            clear_image(im)
            tnode.image = im
            set_pass(h, k)
            bpy.ops.object.bake(type='NORMAL' if k == 'NORMAL' else 'EMIT', use_selected_to_active=True,
                                cage_extrusion=0.045, max_ray_distance=0.12, margin=0, use_clear=False,
                                normal_space='TANGENT', target='IMAGE_TEXTURES')
            a = read_image(im)
            m = a[..., 3] > 0.5
            acc[k][m] = a[m]
            if k == 'albedo':
                cover |= m
                cov = m.mean()
        hp.data.materials.clear()
        bpy.data.materials.remove(mat)
        free_object(hp)
        log(sp['name'], 'baked in %.1fs, coverage %.2f%%' % (time.time() - t0, cov * 100))
    for im in ims.values():
        bpy.data.images.remove(im)
    return acc, cover


def pull_push_fill(img, mask):
    """Fill empty texels from the surrounding islands (edge padding + mip-safe background)."""
    out = img.copy()
    m = mask.astype(np.float32)
    # a few rings of exact neighbour dilation first
    cur, cm = out.copy(), m.copy()
    for _ in range(6):
        acc = np.zeros_like(cur)
        cnt = np.zeros_like(cm)
        pc = np.pad(cur * cm[..., None], ((1, 1), (1, 1), (0, 0)))
        pm = np.pad(cm, 1)
        H, W = cm.shape
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if dy == 1 and dx == 1:
                    continue
                acc += pc[dy:dy + H, dx:dx + W]
                cnt += pm[dy:dy + H, dx:dx + W]
        new = (cm == 0) & (cnt > 0)
        cur[new] = acc[new] / cnt[new][:, None]
        cm[new] = 1.0
    # pull-push for everything else
    levels = []
    a, w = cur * cm[..., None], cm.copy()
    while a.shape[0] > 1:
        levels.append((a, w))
        H, W = w.shape
        a = a.reshape(H // 2, 2, W // 2, 2, -1).sum((1, 3))
        w = w.reshape(H // 2, 2, W // 2, 2).sum((1, 3))
    fill = a / np.maximum(w, 1e-8)[..., None]
    for a, w in reversed(levels):
        up = np.repeat(np.repeat(fill, 2, 0), 2, 1)
        own = a / np.maximum(w, 1e-8)[..., None]
        al = np.clip(w, 0, 1)[..., None]
        fill = own * al + up * (1 - al)
    return fill


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_png(arr, path):
    h, w = arr.shape[:2]
    im = bpy.data.images.new(os.path.basename(path), w, h, alpha=False, float_buffer=False)
    im.colorspace_settings.name = 'Non-Color'
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = np.clip(arr[..., :3], 0, 1)
    im.pixels.foreach_set(rgba.ravel())
    im.filepath_raw = path
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)
    log('wrote', path)


def write_atlas(acc, cover, prefix):
    alb = pull_push_fill(acc['albedo'], cover)
    alb[..., :3] = lin2srgb(alb[..., :3])
    # invalid normal texels (rays that missed / hit the wrong sheet at island borders): tangent-space z must be
    # clearly positive; those texels are re-filled from their valid neighbours
    nz = acc['NORMAL'][..., 2] * 2 - 1
    ncover = cover & (nz > 0.3)
    log('normal texels rejected: %d' % int((cover & ~ncover).sum()))
    nrm = pull_push_fill(acc['NORMAL'], ncover)
    v = nrm[..., :3] * 2 - 1
    v /= np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-6)
    nrm[..., :3] = v * 0.5 + 0.5
    dat = pull_push_fill(acc['data'], cover)
    orm = np.zeros_like(dat)
    orm[..., 0] = dat[..., 0]
    orm[..., 1] = dat[..., 1]
    paths = {k: os.path.join(SCRATCH, '%s_%s.png' % (prefix, k)) for k in ('albedo', 'normal', 'orm')}
    save_png(alb, paths['albedo'])
    save_png(nrm, paths['normal'])
    save_png(orm, paths['orm'])
    return paths


# ------------------------------------------------------------------------------------------------
# 7. export (Blender geometry-only GLB -> gltf-transform: material, tangents, webp, meshopt)
# ------------------------------------------------------------------------------------------------
PACK_JS = r"""
// Generated by tools/blender/rocks.py - assigns the shared 'rock' material, MikkTSpace tangents,
// webp textures and meshopt compression. POSITION stays float (no node-transform quantization), so
// node.geometry can be used directly (e.g. InstancedMesh) and hull vertices are plain metres.
import fs from 'node:fs';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression } from '@gltf-transform/extensions';
import { dedup, prune, quantize, reorder, simplifyPrimitive, tangents, textureCompress, unweld, weld } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder, MeshoptSimplifier } from 'meshoptimizer';
import { generateTangents } from 'mikktspace';
import sharp from 'sharp';
const [inp, out, alb, nrm, orm] = process.argv.slice(2);
await MeshoptEncoder.ready; await MeshoptDecoder.ready; await MeshoptSimplifier.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read(inp);
const root = doc.getRoot();
const tex = (p, name) => doc.createTexture(name).setImage(fs.readFileSync(p)).setMimeType('image/png').setURI(name + '.png');
const tA = tex(alb, 'rock_albedo'), tN = tex(nrm, 'rock_normal'), tO = tex(orm, 'rock_orm');
const mat = doc.createMaterial('rock')
  .setBaseColorTexture(tA).setNormalTexture(tN).setOcclusionTexture(tO).setMetallicRoughnessTexture(tO)
  .setMetallicFactor(0).setRoughnessFactor(1);
for (const node of root.listNodes()) {
  const mesh = node.getMesh(); if (!mesh) continue;
  const name = node.getName();
  for (const prim of mesh.listPrimitives()) {
    if (name.startsWith('rock_hull_')) {
      prim.setMaterial(null);
      for (const s of ['NORMAL', 'TEXCOORD_0', 'TANGENT']) if (prim.getAttribute(s)) prim.setAttribute(s, null);
    } else prim.setMaterial(mat);
  }
}
await doc.transform(
  unweld(),
  tangents({ generateTangents, overwrite: true }),
  weld(),
);
// ---- LODs: rock_k_lod1 (~25%, ~720 tris) and rock_k_lod2 (~7%, ~200 tris), simplified from the final low poly so
// they share its UVs / atlas / tangents (same material). Siblings at the scene root, same origin and scale as rock_k:
// swap geometry per distance band without any transform change (no pop in shape, only in facet detail).
const scene = root.listScenes()[0];
const LODS = [[1, 0.25, 0.012], [2, 0.07, 0.035]];
for (const node of root.listNodes().slice()) {
  const name = node.getName();
  if (!/^rock_\d+$/.test(name) || !node.getMesh()) continue;
  const src = node.getMesh().listPrimitives()[0];
  for (const [lvl, ratio, error] of LODS) {
    const p = src.clone();
    for (const s of p.listSemantics()) p.setAttribute(s, p.getAttribute(s).clone());
    p.setIndices(src.getIndices().clone());
    simplifyPrimitive(p, { simplifier: MeshoptSimplifier, ratio, error });
    const m = doc.createMesh(name + '_lod' + lvl).addPrimitive(p);
    const n = doc.createNode(name + '_lod' + lvl).setMesh(m).setExtras({ lod: lvl, of: name });
    scene.addChild(n);
  }
}
await doc.transform(
  dedup(),
  prune({ keepLeaves: true, keepAttributes: false }),
  reorder({ encoder: MeshoptEncoder }),
  quantize({ pattern: /^(NORMAL|TANGENT|TEXCOORD_\d+)$/, quantizeNormal: 12, quantizeTexcoord: 14 }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^normalTexture$/, quality: 92, resize: [2048, 2048] }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^baseColorTexture$/, quality: 88, resize: [2048, 2048] }),
  // ORM (AO + roughness) is low-frequency next to albedo/normal: 1024 saves ~17 MB of GPU memory with no visible loss
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^(occlusionTexture|metallicRoughnessTexture)$/, quality: 90, resize: [1024, 1024] }),
);
doc.createExtension(EXTMeshoptCompression).setRequired(true)
  .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
await io.write(out, doc);
const nodes = root.listNodes().map((n) => [n.getName(), n.getMesh() ? n.getMesh().listPrimitives()[0].getIndices().getCount() / 3 : 0]);
console.log(JSON.stringify({ out, bytes: fs.statSync(out).size, nodes }));
"""


def export_glb(lows, hulls, out_path):
    bpy.context.view_layer.update()
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in lows + hulls:
        o.data.materials.clear()
        o.location = (0, 0, 0)
        o.select_set(True)
    bpy.context.view_layer.objects.active = lows[0]
    geo = os.path.join(SCRATCH, 'rocks_geo.glb' if out_path == OUT_GLB else 'rocks_geo_subset.glb')
    bpy.ops.export_scene.gltf(filepath=geo, export_format='GLB', use_selection=True, export_materials='NONE',
                              export_yup=True, export_apply=True, export_extras=True, export_texcoords=True,
                              export_normals=True, export_tangents=False, export_animations=False,
                              export_cameras=False, export_lights=False)
    log('exported geometry', geo)
    return geo


def pack_glb(geo, paths, out_path):
    js = os.path.join(SCRATCH, '_pack.mjs')
    with open(js, 'w') as f:
        f.write(PACK_JS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # write next to the scratch files first, then atomically move into place: the dev server / other agents may be
    # loading public/assets/models/rocks.glb at any moment and must never see a half-written file
    tmp = os.path.join(SCRATCH, '_packed_tmp.glb')
    r = subprocess.run(['node', js, geo, tmp, paths['albedo'], paths['normal'], paths['orm']],
                       cwd=ROOT, capture_output=True, text=True)
    log('pack:', r.stdout.strip()[:2000], r.stderr.strip()[-2000:])
    if r.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError('gltf-transform pack failed')
    os.replace(tmp, out_path)
    log('wrote', out_path, os.path.getsize(out_path), 'bytes')


# ------------------------------------------------------------------------------------------------
# 8. preview renders of the final low-poly + atlas (wetness proxy: albedo x0.72, roughness x0.6)
# ------------------------------------------------------------------------------------------------
def preview_material(paths, wet=True):
    m = bpy.data.materials.new('rock_preview')
    m.use_nodes = True
    nt = m.node_tree
    B = NB(nt)
    bs = nt.nodes['Principled BSDF']
    uv = B.new('ShaderNodeUVMap')
    a = B.new('ShaderNodeTexImage')
    a.image = bpy.data.images.load(paths['albedo'], check_existing=True)
    a.image.colorspace_settings.name = 'sRGB'
    n = B.new('ShaderNodeTexImage')
    n.image = bpy.data.images.load(paths['normal'], check_existing=True)
    n.image.colorspace_settings.name = 'Non-Color'
    o = B.new('ShaderNodeTexImage')
    o.image = bpy.data.images.load(paths['orm'], check_existing=True)
    o.image.colorspace_settings.name = 'Non-Color'
    for t in (a, n, o):
        nt.links.new(uv.outputs['UV'], t.inputs['Vector'])
    orm = B.sep(o.outputs['Color'])
    col = a.outputs['Color']
    if wet:
        col = B.mix(1.0, col, (0.72, 0.72, 0.72, 1), 'MULTIPLY')
    col = B.mix(0.5, col, B.comb(orm[0], orm[0], orm[0]), 'MULTIPLY')
    B.put(bs.inputs['Base Color'], col)
    B.put(bs.inputs['Roughness'], B.math('MULTIPLY', orm[1], 0.6 if wet else 1.0))
    nm = B.new('ShaderNodeNormalMap')
    nt.links.new(n.outputs['Color'], nm.inputs['Color'])
    nt.links.new(nm.outputs['Normal'], bs.inputs['Normal'])
    return m


def look_at(cam, target):
    d = Vector(target) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def game_camera(cam, pos, target):
    """Match the in-game camera: 70 deg vertical FOV."""
    cam.data.sensor_fit = 'VERTICAL'
    cam.data.angle_y = math.radians(70)
    cam.location = pos
    look_at(cam, target)


def preview_renders(lows, paths, tag='', scale=0.8):
    for o in list(bpy.data.objects):
        if o not in lows:
            bpy.data.objects.remove(o)
    quick = ARGS.get('quick', False)
    setup_cycles(24 if quick else 64)
    setup_world(0.9)
    wet_ground()
    add_sun(0.9)
    mat = preview_material(paths)
    rocks = [o for o in lows if o.get('kind') == 'rock']
    pebs = [o for o in lows if o.get('kind') == 'pebble']
    for o in lows:
        o.data.materials.clear()
        o.data.materials.append(mat)
        o.hide_render = False
        for pl in o.data.polygons:
            pl.use_smooth = True
    for i, o in enumerate(rocks):
        o.scale = (scale, scale, scale)
        r, c = divmod(i, 5)
        rest_on_ground(o, (c - 2) * 2.3, r * 2.6, rotz=0.4 * i)
    for i, o in enumerate(pebs):
        o.scale = (0.12, 0.12, 0.12)
        rest_on_ground(o, -1.0 + i * 0.4, -1.6, rotz=0.9 * i)
    cam = add_camera((0, -8.5, 3.4), (0, 1.4, 0.2), lens=33)
    if not quick:
        render(os.path.join(SCRATCH, 'lineup%s.png' % tag))
    # game-like views: player standing 2.5-3 m from a boulder (eye 1.65 m, 70 deg vertical FOV)
    for idx, nm, off in ((3, 'closeup', (-1.3, -2.4)), (7, 'closeup_b', (1.4, -2.3)))[:1 if quick else 2]:
        if rocks:
            t = rocks[idx % len(rocks)].location
            game_camera(cam, (t.x + off[0], t.y + off[1], 1.65), (t.x, t.y, 0.35))
            render(os.path.join(SCRATCH, '%s%s.png' % (nm, tag)), 1280, 720)
    # texture detail check: 50 mm from 1.8 m
    if rocks:
        t = rocks[8 % len(rocks)].location
        cam.data.sensor_fit = 'AUTO'
        cam.data.lens = 50
        cam.location = (t.x - 0.9, t.y - 1.7, 1.0)
        look_at(cam, (t.x, t.y, 0.4))
        render(os.path.join(SCRATCH, 'detail%s.png' % tag), 1280, 720)


# ------------------------------------------------------------------------------------------------
def stage_all(specs, full):
    setup_cycles(ARGS['samples'])
    pairs = []
    for sp in specs:
        vox = ARGS['voxel'] if sp['kind'] == 'rock' else ARGS['voxel'] * 2.5
        hp = build_highpoly(sp, vox)
        lp = build_lowpoly(hp, sp)
        xf = orient_and_normalise(hp, lp, sp)
        free_object(hp)                      # rebuilt one at a time for baking (memory)
        pairs.append((sp, xf, lp))
    lows = [p[2] for p in pairs]
    uv_atlas(lows)
    acc, cover = bake_all(pairs, ARGS['atlas'], ARGS['samples'])
    bpy.context.view_layer.update()
    prefix = 'atlas' if full else 'atlas_subset'
    paths = write_atlas(acc, cover, prefix)
    del acc
    hulls = []
    for sp, hp, lp in pairs:
        if sp['kind'] == 'rock':
            hn = 'rock_hull_%d' % sp['index']
            lp['hull'] = hn
            hulls.append(make_hull(lp, hn))
    manifest = {o.name: {'tris': tri_count(o.data), 'verts': len(o.data.vertices), 'radius': o.get('radius'), 'volume': o.get('volume')} for o in lows}
    manifest.update({h.name: {'verts': len(h.data.vertices)} for h in hulls})
    with open(os.path.join(SCRATCH, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    if ARGS['export']:
        geo = export_glb(lows, hulls, OUT_GLB if full else 'subset')
        pack_glb(geo, paths, OUT_GLB if full else os.path.join(SCRATCH, 'rocks_subset.glb'))
    for h in hulls:
        bpy.data.objects.remove(h)
    if ARGS['preview']:
        preview_renders(lows, paths, '' if full else '_subset')


def uv_islands_report(lows, path):
    size = 512
    img = np.zeros((size, size), np.float32)
    allmin, allmax = np.array([9.0, 9.0]), np.array([-9.0, -9.0])
    for o in lows:
        uv = o.data.uv_layers.active.data
        a = np.empty(len(uv) * 2)
        uv.foreach_get('uv', a)
        a = a.reshape(-1, 2)
        allmin = np.minimum(allmin, a.min(0))
        allmax = np.maximum(allmax, a.max(0))
        o.data.calc_loop_triangles()
        for lt in o.data.loop_triangles:
            t = a[list(lt.loops)] * size
            lo = np.floor(t.min(0)).astype(int)
            hi = np.ceil(t.max(0)).astype(int)
            xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
            px = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
            v0, v1 = t[1] - t[0], t[2] - t[0]
            d = v0[0] * v1[1] - v0[1] * v1[0]
            if abs(d) < 1e-12:
                continue
            w = px - t[0]
            b1 = (w[:, 0] * v1[1] - w[:, 1] * v1[0]) / d
            b2 = (v0[0] * w[:, 1] - v0[1] * w[:, 0]) / d
            ins = (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
            q = px[ins].astype(int) - 0
            q = q[(q[:, 0] >= 0) & (q[:, 0] < size) & (q[:, 1] >= 0) & (q[:, 1] < size)]
            img[q[:, 1], q[:, 0]] = 0.4 + 0.6 * ((hash(o.name) % 97) / 97.0)
    log('uv bounds', allmin, allmax, 'coverage %.1f%%' % ((img > 0).mean() * 100))
    save_png(np.repeat(img[..., None], 3, 2), path)


def stage_uv_debug(specs):
    pairs = []
    for sp in specs:
        vox = 0.016 if sp['kind'] == 'rock' else 0.04
        hp = build_highpoly(sp, vox)
        lp = build_lowpoly(hp, sp)
        orient_and_normalise(hp, lp, sp)
        pairs.append((sp, hp, lp))
    lows = [p[2] for p in pairs]
    uv_atlas(lows)
    for o in lows:
        me = o.data
        bm = bmesh.new()
        bm.from_mesh(me)
        uvl = bm.loops.layers.uv.active
        seams = sum(1 for e in bm.edges if e.seam)
        bm.free()
        log(o.name, 'seam edges', seams)
    uv_islands_report(lows, os.path.join(SCRATCH, 'uv_layout.png'))


def main():
    reset_scene()
    rock_ids = ARGS['rocks'] if ARGS['rocks'] is not None else list(range(len(ROCKS)))
    peb_ids = ARGS['pebbles'] if ARGS['pebbles'] is not None else list(range(len(PEBBLES)))
    specs = [ROCKS[i] for i in rock_ids] + [PEBBLES[i] for i in peb_ids]
    full = len(rock_ids) == len(ROCKS) and len(peb_ids) == len(PEBBLES)
    if ARGS['stage'] == 'shapes':
        stage_shapes(specs)
        return
    if ARGS['stage'] == 'pack':
        # re-run only the gltf-transform pass on the last full build (scratch/rocks/rocks_geo.glb + atlas_*.png)
        paths = {k: os.path.join(SCRATCH, 'atlas_%s.png' % k) for k in ('albedo', 'normal', 'orm')}
        pack_glb(os.path.join(SCRATCH, 'rocks_geo.glb'), paths, OUT_GLB)
        return
    if ARGS['stage'] == 'uv':
        stage_uv_debug(specs)
        return
    stage_all(specs, full)


main()
log('done')
