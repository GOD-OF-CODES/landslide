"""
LANDSLIDE - player car. An original early-90s compact mountain 4x4, modelled from scratch.

Run (headless):
  /Applications/Blender.app/Contents/MacOS/Blender -b --python-exit-code 1 -P tools/blender/car.py -- [options]

Options:
  --stage model   build geometry only and render quick previews with flat materials
  --stage full    (default) build, UV, bake textures, export GLB + car.json, render previews of the export (~5 min on an M1)
  --stage recompose  reopen scratch/car/car_final.blend, recompose the textures from the cached G-buffers, re-export
  --int-only      (with recompose) recompose only the interior textures (~45 s)
  --preview-only  re-render previews from the exported GLB
  --fast          1024 textures and low sample counts (for iteration)
  --views a,b     preview views to render (front34,rear34,side,wheel,cockpit,low,front,rear,interior,dash,key,mirror,gaiter,wheelint)
  --no-preview    skip preview renders

(v4 realism pass) The cabin light is computed analytically: compose_int stores in the interior ORM red channel the
fraction of the overcast-sky irradiance that reaches each texel through the glass (window_vis), and in ORM blue a
material class for the engine's micro-detail shader (vehicle.js forces the interior metalness to 0). New interior
details: ignition key + dangling key ring (node key_ring), prayer beads (node mirror_charm, replaces air_freshener),
stitched gaiters, screws, A-pillar mouldings, defroster grilles, a road map / receipt / visor papers, a mirror angled
toward the driver (car.json rearMirror = its face frame for the engine's render-target mirror), gaugeLens.

Frame: car coords are (l, f, h): l = car LEFT (+X in three.js, driver side), f = forward (+Z in three.js),
h = up. glTF export uses +Y up, which maps Blender (x, y, z) to three (x, z, -y); so Blender = (l, -f, h).
Origin: on the ground (tire contact plane), centered between the axles.
"""
import bpy, bmesh, math, sys, os, json, random, time
from mathutils import Vector, Matrix, Euler, Quaternion, noise as mnoise
import numpy as np

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SCR = os.path.join(ROOT, 'scratch', 'car')
TEXDIR = os.path.join(SCR, 'tex')
OUT_GLB_RAW = os.path.join(SCR, 'car_raw.glb')
OUT_GLB = os.path.join(ROOT, 'public', 'assets', 'models', 'car.glb')
OUT_JSON = os.path.join(ROOT, 'public', 'assets', 'models', 'car.json')
HDRI = os.path.join(ROOT, 'raw_assets', 'hdri', 'overcast_soil_puresky_2k.hdr')
os.makedirs(SCR, exist_ok=True); os.makedirs(TEXDIR, exist_ok=True)

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
def arg(name, default=None):
    if name in argv:
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('--') else True
    return default
STAGE = arg('--stage', 'full')
FAST = bool(arg('--fast', False))
VIEWS = (arg('--views', 'front34,rear34,side,wheel,cockpit') or '').split(',')
NO_PREVIEW = bool(arg('--no-preview', False))
PREVIEW_ONLY = bool(arg('--preview-only', False))
LAMPS_ON = bool(arg('--lamps', False))
TEX = 1024 if FAST else 2048
INT_ONLY = bool(arg('--int-only', False))   # (v4) recompose: interior textures only (exterior PNGs reused)
EXT_EMIT = None

random.seed(7); np.random.seed(7)
def log(*a): print(f'[car {time.time() - T0:6.1f}s]', *a, flush=True)

# ------------------------------------------------------------------------------------------------
# Dimensions (DESIGN.md "CAR")
# ------------------------------------------------------------------------------------------------
WB, TRACK, R_TIRE, W_TIRE = 2.30, 1.42, 0.36, 0.22
FAX, RAX, HT = WB / 2, -WB / 2, TRACK / 2
W_BODY = 0.80          # half width of lower body side (flares reach 0.853)
F_FRONT, F_REAR = 1.78, -1.765
H_BOT = 0.42           # body bottom edge
BELT = 1.06
ROOF_E = 1.635         # roof edge height
ARCH_R, ARCH_HC = 0.43, 0.40
GAP = 0.0036           # panel gap width

def V(l, f, h): return Vector((l, -f, h))
def clamp(x, a=0.0, b=1.0): return a if x < a else b if x > b else x
def smooth(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t)
def lerp(a, b, t): return a + (b - a) * t

# ------------------------------------------------------------------------------------------------
# Scene / object helpers
# ------------------------------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.scale_length = 1.0
    return sc

def coll(name):
    c = bpy.data.collections.get(name)
    if not c:
        c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
    return c

MATS = {}
def mat(name, color=(0.5, 0.5, 0.5), rough=0.5, metal=0.0, **kw):
    """Preview material (Principled). Replaced by baked materials in the full stage."""
    m = bpy.data.materials.get(name)
    if m: return m
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value = (*color, 1)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    if 'coat' in kw: b.inputs['Coat Weight'].default_value = kw['coat']; b.inputs['Coat Roughness'].default_value = kw.get('coat_rough', 0.1)
    if 'transmission' in kw: b.inputs['Transmission Weight'].default_value = kw['transmission']
    if 'emit' in kw: b.inputs['Emission Color'].default_value = (*kw['emit'], 1); b.inputs['Emission Strength'].default_value = kw.get('emit_strength', 0.0)
    if 'alpha' in kw: b.inputs['Alpha'].default_value = kw['alpha']
    m.diffuse_color = (*color, 1)
    MATS[name] = m
    return m

def init_materials():
    # surface materials (merged into the DESIGN.md material names at export, see EXPORT_MAP)
    mat('paint', (0.16, 0.025, 0.02), 0.45, coat=0.6, coat_rough=0.15)
    mat('paint_inner', (0.12, 0.02, 0.018), 0.5)
    mat('plastic', (0.025, 0.025, 0.025), 0.6)
    mat('rubber_trim', (0.015, 0.015, 0.015), 0.7)
    mat('steel_black', (0.02, 0.02, 0.02), 0.45)
    mat('chrome', (0.85, 0.85, 0.85), 0.12, 1.0)
    mat('underbody', (0.03, 0.025, 0.02), 0.8)
    mat('liner', (0.02, 0.02, 0.02), 0.8)
    mat('wheel', (0.35, 0.35, 0.33), 0.5, 0.6)
    mat('brake', (0.2, 0.15, 0.12), 0.6, 0.8)
    mat('headlight', (0.9, 0.9, 0.9), 0.05, alpha=0.25, emit=(1.0, 0.92, 0.78))
    mat('brakelight', (0.5, 0.02, 0.02), 0.15, emit=(1.0, 0.05, 0.03))
    mat('indicator', (0.8, 0.35, 0.02), 0.15, emit=(1.0, 0.45, 0.05))
    mat('plate', (0.8, 0.8, 0.78), 0.4)
    mat('cover', (0.03, 0.03, 0.03), 0.6)
    mat('glass', (0.85, 0.9, 0.88), 0.03, transmission=1.0)
    mat('rubber', (0.018, 0.018, 0.018), 0.8)
    mat('interior', (0.05, 0.048, 0.045), 0.7)
    mat('fabric', (0.06, 0.055, 0.05), 0.9)
    mat('carpet', (0.03, 0.03, 0.03), 0.95)
    mat('int_black', (0.02, 0.02, 0.02), 0.6)
    mat('gauge', (0.02, 0.02, 0.02), 0.35, emit=(0.9, 0.95, 1.0))
    mat('charm', (0.10, 0.03, 0.015), 0.35)            # (v4) wooden prayer beads + tassel hanging from the mirror
    mat('leather', (0.02, 0.019, 0.018), 0.5)          # (v4) steering-wheel rim, gaiters, key fob (high texel density)
    mat('paper', (0.6, 0.58, 0.52), 0.8)               # (v4) road map on the dash, papers under the visor strap

def obj_from_bm(name, bm, material=None, collection=None, smooth_angle=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    (collection or coll('CAR')).objects.link(ob)
    if material:
        ob.data.materials.append(MATS[material] if isinstance(material, str) else material)
    if smooth_angle is not None: shade(ob, smooth_angle)
    return ob

def shade(ob, angle=35):
    me = ob.data
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(angle))

def apply_mods(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    me.name = ob.name
    if old.users == 0: bpy.data.meshes.remove(old)
    return ob

def bevel(ob, width, segs=2, angle=35, apply=True, profile=0.5, harden=False):
    m = ob.modifiers.new('bevel', 'BEVEL')
    m.width = width; m.segments = segs; m.limit_method = 'ANGLE'; m.angle_limit = math.radians(angle)
    m.use_clamp_overlap = True; m.profile = profile; m.harden_normals = harden
    m.miter_outer = 'MITER_ARC'
    if apply: apply_mods(ob)
    return ob

def weighted(ob, apply=True):
    m = ob.modifiers.new('wn', 'WEIGHTED_NORMAL'); m.keep_sharp = True; m.mode = 'FACE_AREA'; m.weight = 50
    if apply: apply_mods(ob)
    return ob

def boolean(ob, cutter, op='DIFFERENCE', solver='MANIFOLD', transfer=True):
    m = ob.modifiers.new('bool', 'BOOLEAN')
    m.operation = op; m.solver = solver
    if isinstance(cutter, bpy.types.Collection):
        m.operand_type = 'COLLECTION'; m.collection = cutter
    else:
        m.object = cutter
    if transfer: m.material_mode = 'TRANSFER'
    apply_mods(ob)
    return ob

def delete_obj(ob):
    me = ob.data if ob.type == 'MESH' else None
    bpy.data.objects.remove(ob, do_unlink=True)
    if me and me.users == 0: bpy.data.meshes.remove(me)

def join(objs, name=None):
    objs = [o for o in objs if o]
    if len(objs) == 1:
        if name: objs[0].name = name
        return objs[0]
    ctx = bpy.context.copy()
    with bpy.context.temp_override(active_object=objs[0], selected_editable_objects=objs, selected_objects=objs):
        bpy.ops.object.join()
    ob = objs[0]
    if name: ob.name = name; ob.data.name = name
    return ob

def mirror_copy(ob, name=None, axis=0):
    """Return a copy mirrored across l=0 (Blender X) with normals fixed (data applied)."""
    me = ob.data.copy()
    bm = bmesh.new(); bm.from_mesh(me)
    s = [1, 1, 1]; s[axis] = -1
    bmesh.ops.scale(bm, vec=Vector(s), verts=bm.verts)
    bmesh.ops.reverse_faces(bm, faces=bm.faces, flip_multires=False)
    bm.to_mesh(me); bm.free()
    o2 = bpy.data.objects.new(name or ob.name + '_m', me)
    for c in ob.users_collection: c.objects.link(o2)
    for k in ob.keys(): o2[k] = ob[k]
    return o2

def sym(ob):
    """Mirror across l=0 and join (for parts built on the left side only)."""
    m = mirror_copy(ob)
    return join([ob, m])

def tag(ob, atlas='ext'):
    ob['atlas'] = atlas
    return ob

# ------------------------------------------------------------------------------------------------
# Geometry builders (car coords -> Blender)
# ------------------------------------------------------------------------------------------------
def fillet(pts, radius, segs=4, closed=True):
    """Round the corners of a 2D polyline. radius: float or list per vertex (0 = keep sharp)."""
    n = len(pts); out = []
    P = [Vector((p[0], p[1])) for p in pts]
    rr = radius if isinstance(radius, (list, tuple)) else [radius] * n
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            out.append(tuple(P[i])); continue
        a, p, b = P[i - 1], P[i], P[(i + 1) % n]
        r = rr[i]
        u = (a - p); v = (b - p)
        lu, lv = u.length, v.length
        if r <= 0 or lu < 1e-6 or lv < 1e-6:
            out.append(tuple(p)); continue
        u.normalize(); v.normalize()
        cosang = max(-0.9999, min(0.9999, u.dot(v)))
        ang = math.acos(cosang)
        t = r / math.tan(ang / 2)
        t = min(t, lu * 0.49, lv * 0.49)
        r = t * math.tan(ang / 2)
        p1 = p + u * t; p2 = p + v * t
        bis = (u + v); bis.normalize()
        c = p + bis * (r / math.sin(ang / 2))
        a1 = math.atan2(p1.y - c.y, p1.x - c.x); a2 = math.atan2(p2.y - c.y, p2.x - c.x)
        da = a2 - a1
        while da > math.pi: da -= 2 * math.pi
        while da < -math.pi: da += 2 * math.pi
        for k in range(segs + 1):
            aa = a1 + da * k / segs
            out.append((c.x + r * math.cos(aa), c.y + r * math.sin(aa)))
    return out

def rrect(x0, y0, x1, y1, r, segs=4):
    return fillet([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], r, segs)

def circle2(cx, cy, r, n=24, a0=0.0):
    return [(cx + r * math.cos(a0 + 2 * math.pi * i / n), cy + r * math.sin(a0 + 2 * math.pi * i / n)) for i in range(n)]

def to3(plane, a, b, c):
    """plane 'fh': (a=f, b=h, c=l); 'lh': (a=l, b=h, c=f); 'lf': (a=l, b=f, c=h)."""
    if plane == 'fh': return V(c, a, b)
    if plane == 'lh': return V(a, c, b)
    if plane == 'lf': return V(a, b, c)
    raise ValueError(plane)

def prism(poly, plane, c0, c1, name='prism', material=None, bm=None):
    """Closed prism from a 2D polygon in `plane`, extruded along the third axis from c0 to c1."""
    own = bm is None
    bm = bm or bmesh.new()
    A = [bm.verts.new(to3(plane, x, y, c0)) for x, y in poly]
    B = [bm.verts.new(to3(plane, x, y, c1)) for x, y in poly]
    n = len(poly)
    fs = [bm.faces.new(A[::-1]), bm.faces.new(B)]
    for i in range(n):
        j = (i + 1) % n
        fs.append(bm.faces.new((A[i], A[j], B[j], B[i])))
    bmesh.ops.recalc_face_normals(bm, faces=fs)
    if own: return obj_from_bm(name, bm, material)
    return fs

def box(name, c, s, material=None, bev=0.0, segs=2, ang=35):
    """Axis aligned box. c=(l,f,h) center, s=(sl,sf,sh) full size."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = V(c[0] + v.co.x * s[0], c[1] - v.co.y * s[1], c[2] + v.co.z * s[2])
    ob = obj_from_bm(name, bm, material)
    if bev > 0: bevel(ob, bev, segs, ang)
    return ob

def loft(sections, closed=True, cap0=True, cap1=True, name='loft', material=None):
    """sections: list of lists of Vectors (Blender coords), equal length."""
    bm = bmesh.new()
    rings = [[bm.verts.new(p) for p in sec] for sec in sections]
    n = len(sections[0])
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            try: bm.faces.new((a[i], a[j], b[j], b[i]))
            except ValueError: pass
    if cap0: bm.faces.new(rings[0][::-1])
    if cap1: bm.faces.new(rings[-1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return obj_from_bm(name, bm, material)

def frame_basis(axis):
    a = Vector(axis).normalized()
    ref = Vector((0, 0, 1)) if abs(a.z) < 0.9 else Vector((1, 0, 0))
    u = a.cross(ref).normalized(); v = a.cross(u).normalized()
    return a, u, v

def lathe(prof, segs, center, axis, name='lathe', material=None, closed_prof=False, a0=0.0, a1=2 * math.pi, u=None, bm=None):
    """Surface of revolution. prof: list of (r, x) with x along axis. center/axis in Blender coords."""
    a, uu, vv = frame_basis(axis)
    if u is not None:
        uu = Vector(u).normalized(); vv = a.cross(uu).normalized()
    full = abs((a1 - a0) - 2 * math.pi) < 1e-6
    ns = segs if full else segs + 1
    own = bm is None
    bm = bm or bmesh.new()
    rings = []
    for k in range(ns):
        t = a0 + (a1 - a0) * k / segs
        d = uu * math.cos(t) + vv * math.sin(t)
        rings.append([bm.verts.new(Vector(center) + a * x + d * r) for r, x in prof])
    np_ = len(prof)
    fs = []
    for k in range(segs):
        A = rings[k]; B = rings[(k + 1) % ns]
        for i in range(np_ if closed_prof else np_ - 1):
            j = (i + 1) % np_
            try: fs.append(bm.faces.new((A[i], B[i], B[j], A[j])))
            except ValueError: pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    if not own: return fs
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = obj_from_bm(name, bm, material)
    return ob

def sweep(path, prof, closed_path=False, up=None, name='sweep', material=None, cap=True, twist_up=None, bm=None, scale=None):
    """Sweep a closed 2D profile [(s, u)] along a 3D path (Blender coords).
    Frame: t = tangent, uvec = up projected orthogonal to t, svec = t x uvec."""
    own = bm is None
    bm = bm or bmesh.new()
    P = [Vector(p) for p in path]; n = len(P)
    up0 = Vector(up) if up is not None else Vector((0, 0, 1))
    rings = []
    for i in range(n):
        if closed_path:
            t = (P[(i + 1) % n] - P[i - 1])
        else:
            t = (P[min(i + 1, n - 1)] - P[max(i - 1, 0)])
        t.normalize()
        upv = (twist_up(i) if twist_up else up0)
        uvec = (upv - t * upv.dot(t)).normalized()
        svec = t.cross(uvec).normalized()
        sc = scale(i) if scale else 1.0
        rings.append([bm.verts.new(P[i] + svec * (s * sc) + uvec * (w * sc)) for s, w in prof])
    m = len(prof); fs = []
    for i in range(n if closed_path else n - 1):
        A = rings[i]; B = rings[(i + 1) % n]
        for k in range(m):
            j = (k + 1) % m
            try: fs.append(bm.faces.new((A[k], A[j], B[j], B[k])))
            except ValueError: pass
    if cap and not closed_path:
        try: fs.append(bm.faces.new(rings[0][::-1])); fs.append(bm.faces.new(rings[-1]))
        except ValueError: pass
    bmesh.ops.recalc_face_normals(bm, faces=fs)
    if not own: return fs
    return obj_from_bm(name, bm, material)

def tube(path, r, n=8, name='tube', material=None, closed_path=False, up=None, cap=True):
    prof = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return sweep(path, prof, closed_path, up, name, material, cap)

def polyline_resample(pts, step):
    out = [Vector(pts[0])]
    for a, b in zip(pts[:-1], pts[1:]):
        a = Vector(a); b = Vector(b); L = (b - a).length
        k = max(1, int(math.ceil(L / step)))
        for i in range(1, k + 1): out.append(a.lerp(b, i / k))
    return out

def cyl(name, c, axis, r, depth, segs=24, material=None, bev=0.0):
    """Closed cylinder centered at c (car coords), axis given in car coords (l,f,h)."""
    ax = Vector((axis[0], -axis[1], axis[2])).normalized()
    prof = [(0.0001, -depth / 2), (r, -depth / 2), (r, depth / 2), (0.0001, depth / 2)]
    ob = lathe(prof, segs, V(*c), ax, name, material)
    bm = bmesh.new(); bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0003)
    bm.to_mesh(ob.data); bm.free()
    if bev: bevel(ob, bev, 2, 35)
    return ob

# ------------------------------------------------------------------------------------------------
# BODY: lower body (solid loft), greenhouse (shell loft), unified with booleans
# ------------------------------------------------------------------------------------------------
def hood_top(f):
    hood = 1.000 + 0.045 * clamp((1.76 - f) / 0.96)
    t = smooth(0.80, 0.72, f)
    return lerp(hood, 1.066, t)

def lb_half(f, iw=0.0, it=0.0, ib=0.0):
    """Lower body half cross-section (l>=0) from bottom center to top center: list of (l, h)."""
    w = W_BODY - iw
    ht = hood_top(f) - it
    hb = H_BOT + ib
    crown = 0.010
    rb, rs, tuck = 0.05, 0.042, 0.012
    pts = []
    xb = w - tuck - rb
    for t in (0.0, 0.33, 0.66): pts.append((xb * t, hb))
    for k in range(5):
        a = -math.pi / 2 + (math.pi / 2) * k / 4
        pts.append((xb + rb * math.cos(a), hb + rb + rb * math.sin(a)))
    y0 = hb + rb; y1 = ht - crown - rs
    # side with a stamped feature line (5 mm step) at h ~ 0.935
    ha, hb_, hc = min(0.924, y1 - 0.030), min(0.931, y1 - 0.020), min(0.937, y1 - 0.010)
    for hh, dl in ((y0 + 0.10, 0.004), (y0 + 0.22, 0.004), (ha, 0.004), (hb_, 0.0012), (hc, 0.0), (y1, 0.0)):
        pts.append((w - dl - tuck * clamp(1 - (hh - y0) / 0.22), hh))
    for k in range(1, 7):
        a = (math.pi / 2) * k / 6
        pts.append((w - rs + rs * math.cos(a), y1 + rs * math.sin(a)))
    xe = w - rs
    for t in (0.66, 0.33, 0.0):
        l = xe * t
        pts.append((l, ht - crown * (l / xe) ** 2))
    return pts

def full_ring(half, f):
    """Mirror a half profile (bottom center -> top center) into a closed ring of Blender coords."""
    ring = [V(l, f, h) for l, h in half]
    ring += [V(-l, f, h) for l, h in reversed(half[1:-1])]
    return ring

def end_stations(f_end, depth, rw, rt, rb, n, front=True):
    """(f, iw, it, ib) stations for a rounded body end."""
    st = []
    for k in range(n + 1):
        th = (math.pi / 2) * k / n
        f = f_end - depth + depth * math.sin(th) if front else f_end + depth - depth * math.sin(th)
        c = 1 - math.cos(th)
        st.append((f, rw * c, rt * c, rb * c))
    return st

def build_lower_body():
    front = end_stations(F_FRONT, 0.05, 0.06, 0.035, 0.03, 6, True)
    rear = end_stations(F_REAR, 0.04, 0.05, 0.0, 0.03, 5, False)
    mids = [1.60, 1.30, 1.00, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.5, -0.5, -1.5]
    st = [s for s in reversed(rear)] + [(f, 0, 0, 0) for f in sorted(mids) if F_REAR + 0.04 < f < F_FRONT - 0.05] + front
    sections = [full_ring(lb_half(f, iw, it, ib), f) for f, iw, it, ib in st]
    return loft(sections, True, True, True, 'lower_body', 'paint')

# greenhouse silhouette (edge height of the top along f)
_GH = None
def gh_top(f):
    global _GH
    if _GH is None:
        pts = fillet([(0.90, 0.88), (0.75, 1.07), (0.30, ROOF_E), (-2.0, ROOF_E + 0.004)], [0, 0, 0.12, 0], 8, closed=False)
        _GH = sorted(pts, key=lambda p: p[0])
    xs = [p[0] for p in _GH]; ys = [p[1] for p in _GH]
    return float(np.interp(f, xs, ys))

GH_HB, GH_WB, GH_WT = 1.045, 0.776, 0.712
def gh_l(h, iw=0.0):
    """Greenhouse side l at height h (tumblehome line)."""
    return (GH_WB - iw) + (GH_WT - GH_WB) * (h - GH_HB) / (ROOF_E - GH_HB)

def gh_half(f, iw=0.0, it=0.0):
    he = gh_top(f) - it
    hb = GH_HB
    crown = 0.016 * smooth(0.30, 0.05, f)
    rg = min(0.045, (he - hb) * 0.45)
    pts = []
    y1 = he - rg
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        hh = hb + (y1 - hb) * t
        pts.append((gh_l(hh, iw), hh))
    ls = gh_l(y1, iw)
    for k in range(1, 7):
        a = (math.pi / 2) * k / 6
        pts.append((ls - rg + rg * math.cos(a), y1 + rg * math.sin(a)))
    xe = ls - rg
    for t in (0.66, 0.33, 0.0):
        l = xe * t
        pts.append((l, he + crown * (1 - (l / xe) ** 2)))
    return pts

def build_greenhouse():
    fs = [0.755, 0.72, 0.66, 0.60, 0.54, 0.48, 0.42, 0.38, 0.35, 0.32, 0.29, 0.26, 0.22, 0.15, 0.05, -0.2, -0.8, -1.4, -1.60, -1.70]
    rear = end_stations(F_REAR, 0.04, 0.05, 0.035, 0.0, 5, False)
    st = [(f, 0, 0) for f in fs] + [(f, iw, it) for f, iw, it, ib in rear]
    sections = []
    for f, iw, it in st:
        half = gh_half(f, iw, it)
        ring = [V(l, f, h) for l, h in half] + [V(-l, f, h) for l, h in reversed(half[:-1])]
        sections.append(ring)
    ob = loft(sections, False, False, True, 'greenhouse', 'paint')
    # make sure normals point outward (roof normal +Z)
    me = ob.data
    top = max(me.polygons, key=lambda p: p.center.z)
    if top.normal.z < 0:
        bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.reverse_faces(bm, faces=bm.faces); bm.to_mesh(me); bm.free()
    ob.data.materials.append(MATS['paint_inner'])
    m = ob.modifiers.new('solid', 'SOLIDIFY'); m.thickness = 0.028; m.offset = -1; m.use_rim = True; m.use_even_offset = True
    m.material_offset = 1; m.material_offset_rim = 0
    apply_mods(ob)
    return ob

# --- window / seam outlines (side plane (f, h)) ---
K_WS = (ROOF_E - 1.07) / 0.45        # windshield slope dh/df
def ws_f(h): return 0.75 - (h - 1.07) / K_WS          # windshield edge line (outer)
DOOR_F0, DOOR_F1, DOOR_H0, DOOR_TOP = 0.655, -0.45, 0.47, 1.605
def door_front_f(h):  # door seam front edge, follows A-pillar above the belt
    return DOOR_F0 if h <= BELT else DOOR_F0 - (h - BELT) / K_WS
def door_win_poly():
    h0, h1 = 1.095, 1.568
    fa = lambda h: door_front_f(h) - 0.034
    return fillet([(fa(h0), h0), (-0.41, h0), (-0.41, h1), (fa(h1), h1)], [0.015, 0.02, 0.035, 0.03], 5)
def quarter_win_poly():
    return rrect(-1.628, 1.095, -0.53, 1.568, 0.04, 5)
def door_seam_poly():
    return [(DOOR_F0, DOOR_H0), (DOOR_F1, DOOR_H0), (DOOR_F1, DOOR_TOP), (door_front_f(DOOR_TOP), DOOR_TOP), (DOOR_F0, BELT)]

def body_outer_l(f, h):
    if h < GH_HB + 0.01:
        return W_BODY - 0.012 * clamp(1 - (h - (H_BOT + 0.05)) / 0.22)
    return gh_l(h)

def offset_poly(poly, d, closed=True):
    """Offset a 2D polyline by d (left normal). Miter joins."""
    P = [Vector(p) for p in poly]; n = len(P); out = []
    for i in range(n):
        if not closed and i == 0: t = (P[1] - P[0]).normalized(); nn = Vector((-t.y, t.x)); out.append(P[i] + nn * d); continue
        if not closed and i == n - 1: t = (P[i] - P[i - 1]).normalized(); nn = Vector((-t.y, t.x)); out.append(P[i] + nn * d); continue
        t0 = (P[i] - P[i - 1]).normalized(); t1 = (P[(i + 1) % n] - P[i]).normalized()
        n0 = Vector((-t0.y, t0.x)); n1 = Vector((-t1.y, t1.x))
        m = (n0 + n1);
        if m.length < 1e-6: m = n0
        m.normalize()
        k = d / max(0.3, m.dot(n0))
        out.append(P[i] + m * k)
    return [(p.x, p.y) for p in out]

def signed_area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))

def grow(poly, d):
    """Offset a closed polygon outward by d (inward if d<0), independent of winding."""
    return offset_poly(poly, d if signed_area(poly) < 0 else -d, True)

def ribbon_cutter(poly, plane, depth_fn, outer=1.4, gap=GAP, closed=True, name='seam', side=1):
    """Thin solid along a 2D polyline in `plane` used to cut panel gaps.
    depth_fn(a, b) -> inner third-coordinate bound (cut goes from there to `outer`)."""
    A = offset_poly(poly, gap / 2, closed); B = offset_poly(poly, -gap / 2, closed)
    bm = bmesh.new()
    def mk(p, c): return bm.verts.new(to3(plane, p[0], p[1], c * side if plane != 'lf' else c))
    n = len(poly)
    Ai = [mk(p, depth_fn(*p)) for p in A]; Ao = [mk(p, outer) for p in A]
    Bi = [mk(p, depth_fn(*p)) for p in B]; Bo = [mk(p, outer) for p in B]
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        j = (i + 1) % n
        bm.faces.new((Ai[i], Ai[j], Ao[j], Ao[i]))
        bm.faces.new((Bi[j], Bi[i], Bo[i], Bo[j]))
        bm.faces.new((Ao[i], Ao[j], Bo[j], Bo[i]))
        bm.faces.new((Ai[j], Ai[i], Bi[i], Bi[j]))
    if not closed:
        bm.faces.new((Ai[0], Ao[0], Bo[0], Bi[0]))
        bm.faces.new((Ai[-1], Bi[-1], Bo[-1], Ao[-1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return obj_from_bm(name, bm, 'paint', coll('CUT'))

def build_body():
    log('body: lofts')
    lb = build_lower_body()
    gh = build_greenhouse()
    boolean(lb, gh, 'UNION'); delete_obj(gh)
    body = lb; body.name = 'body_shell'
    cut = coll('CUT')
    # cabin cavity
    cav = prism(rrect(-0.748, 0.50, 0.748, 1.075, 0.03), 'lh', -1.724, 0.735, 'cavity', 'paint_inner')
    for c in cav.users_collection: c.objects.unlink(cav)
    cut.objects.link(cav)
    boolean(body, cut); delete_obj(cav)
    log('body: wheel houses + wells')
    # wheel houses inside the cabin (rear) then the wells
    houses = []
    for fa in (FAX, RAX):
        for s in (1, -1):
            h = cyl('house', (s * 0.60, fa, ARCH_HC), (1, 0, 0), ARCH_R + 0.025, 0.30, 40, 'paint_inner')
            houses.append(h)
    hj = join(houses, 'houses')
    # clip houses to above floor
    clip = box('clip', (0, 0, 0.20), (3, 5, 0.58), 'paint')
    boolean(hj, clip, 'DIFFERENCE', transfer=False); delete_obj(clip)
    boolean(body, hj, 'UNION'); delete_obj(hj)
    for c in list(cut.objects): delete_obj(c)
    for fa in (FAX, RAX):
        for s in (1, -1):
            w = cyl('well', (s * 0.73, fa, ARCH_HC), (1, 0, 0), ARCH_R, 0.54, 48, 'liner')
            for c in w.users_collection: c.objects.unlink(w)
            cut.objects.link(w)
    # recesses: grille panel (front), tail lamps (rear corners)
    for ob in [prism(rrect(-0.708, 0.612, 0.708, 0.932, 0.02), 'lh', F_FRONT - 0.026, 2.1, 'grille_rec', 'paint')]:
        for c in ob.users_collection: c.objects.unlink(ob)
        cut.objects.link(ob)
    for s in (1, -1):
        ob = box('tl_rec', (s * 0.80, F_REAR - 0.2, 0.785), (0.27, 0.46, 0.40), 'paint')
        for c in ob.users_collection: c.objects.unlink(ob)
        cut.objects.link(ob)
    # window openings
    for poly in (door_win_poly(), quarter_win_poly()):
        ob = prism(poly, 'fh', 0.45, 1.2, 'win', 'paint');
        for c in ob.users_collection: c.objects.unlink(ob)
        cut.objects.link(ob)
        ob2 = prism(poly, 'fh', -1.2, -0.45, 'win', 'paint')
        for c in ob2.users_collection: c.objects.unlink(ob2)
        cut.objects.link(ob2)
    # windshield opening: polygon in the windshield plane
    ob = windshield_cutter()
    cut.objects.link(ob)
    # rear window
    ob = prism(rrect(-0.60, 1.14, 0.60, 1.555, 0.05, 5), 'lh', F_REAR - 0.3, F_REAR + 0.2, 'rwin', 'paint')
    for c in ob.users_collection: c.objects.unlink(ob)
    cut.objects.link(ob)
    log('body: openings boolean (%d cutters)' % len(cut.objects))
    boolean(body, cut)
    for c in list(cut.objects): delete_obj(c)
    log('body: seams')
    build_seam_cutters()
    boolean(body, cut)
    for c in list(cut.objects): delete_obj(c)
    log('body: bevel')
    bm = bmesh.new(); bm.from_mesh(body.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0002)
    bmesh.ops.dissolve_degenerate(bm, dist=0.0002, edges=bm.edges)
    bm.to_mesh(body.data); bm.free()
    bevel(body, 0.0022, 2, 40)
    shade(body, 40)
    weighted(body)
    return tag(body)

def ws_point(u, v):
    """Point on the windshield plane: u = l, v = height h on the outer surface."""
    return V(u, ws_f(v), v)

def windshield_cutter():
    h0, h1 = 1.100, 1.585
    w0, w1 = 0.625, 0.585
    poly = fillet([(-w0, h0), (w0, h0), (w1, h1), (-w1, h1)], [0.035, 0.035, 0.05, 0.05], 6)
    nrm = Vector((0, -(1.0), 0))  # placeholder
    # normal of windshield plane in blender coords (points forward-up)
    d_up = (ws_point(0, 1.5) - ws_point(0, 1.1)).normalized()
    n = d_up.cross(Vector((1, 0, 0))).normalized()
    if n.y > 0: n = -n  # forward is -Y
    bm = bmesh.new()
    A = [bm.verts.new(ws_point(u, v) - n * 0.15) for u, v in poly]
    B = [bm.verts.new(ws_point(u, v) + n * 0.15) for u, v in poly]
    bm.faces.new(A[::-1]); bm.faces.new(B)
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        bm.faces.new((A[i], A[j], B[j], B[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new('wscut'); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new('wscut', me); ob.data.materials.append(MATS['paint'])
    return ob

def build_seam_cutters():
    cut = coll('CUT')
    # doors (both sides)
    poly = door_seam_poly()
    for s in (1, -1):
        ribbon_cutter(poly, 'fh', lambda f, h: 0.735 if 0.99 < h < 1.10 else body_outer_l(f, h) - 0.02, 1.4, GAP, True, 'door_seam', s)
    # hood: sides + rear on top, front face line
    hp = [(0.694, 1.9), (0.694, 0.792), (-0.694, 0.792), (-0.694, 1.9)]
    ribbon_cutter(hp, 'lf', lambda l, f: hood_top(f) - 0.03 - 0.035 * smooth(1.70, 1.785, f), 1.4, GAP, False, 'hood_seam')
    ribbon_cutter([(-0.95, 0.962), (0.95, 0.962)], 'lh', lambda l, h: F_FRONT - 0.022, 2.2, GAP, False, 'hood_front')
    # tail door (rear face)
    tp = rrect(-0.652, 0.47, 0.652, 1.598, 0.04, 4)
    ribbon_cutter(tp, 'lh', lambda l, h: F_REAR + 0.022, -2.4, GAP, True, 'tail_seam')
    # fuel flap (left rear quarter)
    ribbon_cutter(circle2(-1.30, 0.885, 0.056, 28), 'fh', lambda f, h: W_BODY - 0.02, 1.4, GAP * 0.9, True, 'fuel_seam', 1)

# ------------------------------------------------------------------------------------------------
# Exterior details
# ------------------------------------------------------------------------------------------------
def arch_ring(fc, a, s, u):
    return V(u, fc + (ARCH_R + s) * math.cos(a), ARCH_HC + (ARCH_R + s) * math.sin(a))

def build_flares():
    parts = []
    prof = [(0.070, 0.797), (0.064, 0.818), (0.047, 0.840), (0.024, 0.851), (0.004, 0.854), (-0.009, 0.849),
            (-0.015, 0.834), (-0.015, 0.80), (-0.013, 0.776), (-0.005, 0.770), (0.004, 0.778), (0.010, 0.795)]
    for fc in (FAX, RAX):
        a0, a1 = math.radians(-8), math.radians(188)
        n = 44
        bm = bmesh.new()
        rings = []
        for k in range(n + 1):
            a = a0 + (a1 - a0) * k / n
            rings.append([bm.verts.new(arch_ring(fc, a, s, u)) for s, u in prof])
        m = len(prof)
        for k in range(n):
            for i in range(m):
                j = (i + 1) % m
                bm.faces.new((rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]))
        bm.faces.new(rings[0][::-1]); bm.faces.new(rings[-1])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        ob = obj_from_bm('flare', bm, 'plastic')
        # screws
        for k in range(7):
            a = a0 + (a1 - a0) * (k + 0.5) / 7
            p = arch_ring(fc, a, 0.034, 0.842)
            sc = lathe([(0.0001, 0.004), (0.004, 0.0035), (0.0065, 0.0015), (0.0068, -0.003)], 10, p, (1, 0, 0), 'screw', 'chrome')
            # orient screw axis along the flare surface normal (approx +X tilted)
            parts.append(sc)
        parts.append(ob)
    j = join(parts, 'flares_l')
    shade(j, 50)
    return sym(tag(j))

def build_rockers():
    ob = box('rocker', (0.772, (FAX - ARCH_R - 0.06 + RAX + ARCH_R + 0.06) / 2, 0.435), (0.066, (FAX - RAX) - 2 * (ARCH_R + 0.05), 0.075), 'plastic', 0.012, 3)
    ob2 = box('molding', (0.804, 0.105, 0.64), (0.02, 1.08, 0.058), 'plastic', 0.007, 3)
    ob3 = box('molding', (0.804, -0.585, 0.64), (0.02, 0.25, 0.058), 'plastic', 0.007, 3)
    j = join([ob, ob2, ob3], 'rockers_l')
    shade(j, 45)
    return sym(tag(j))

def bumper_path(fz, wrap, sgn):
    pts = [(-0.835, fz - sgn * wrap), (-0.80, fz - sgn * 0.02), (0.0, fz), (0.80, fz - sgn * 0.02), (0.835, fz - sgn * wrap)]
    return fillet(pts, [0, 0.07, 0, 0.07, 0], 6, closed=False)

def build_bumpers():
    parts = []
    # profile: s = outward (horizontal normal), u = height
    bprof = fillet([(-0.085, -0.095), (0.022, -0.090), (0.030, 0.0), (0.020, 0.092), (-0.085, 0.095)], [0.01, 0.02, 0.03, 0.02, 0.01], 3)
    rprof = fillet([(0.0, -0.024), (0.040, -0.022), (0.046, 0.0), (0.040, 0.022), (0.0, 0.024)], 0.008, 3)
    for sgn, fz, hc in ((1, 1.862, 0.462), (-1, -1.848, 0.452)):
        path2 = bumper_path(fz, 0.19, sgn)
        path = [V(l, f, hc) for l, f in path2]
        up = Vector((0, 0, 1))
        # profile s must point outward (away from car). sweep svec = t x up; t along +l in path order
        prof = [(s * sgn, u) for s, u in bprof]
        b = sweep(path, prof, False, up, 'bumper', 'steel_black')
        parts.append(b)
        # rubber strip in two pieces (leave room for the plate)
        for seg in ((-0.83, -0.30), (0.30, 0.83)):
            p2 = [p for p in path2 if seg[0] <= p[0] <= seg[1]]
            if len(p2) < 2: continue
            pth = [V(l, f, hc - 0.005) for l, f in p2]
            pr = [(s * sgn, u) for s, u in rprof]
            parts.append(sweep(pth, pr, False, up, 'bstrip', 'rubber_trim'))
        # brackets
        for s in (1, -1):
            parts.append(box('bbracket', (s * 0.46, fz - sgn * 0.14, hc - 0.01), (0.07, 0.20, 0.08), 'steel_black', 0.008))
        # plate holder + plate
        parts.append(box('plateholder', (0, fz + sgn * 0.034, hc), (0.545, 0.012, 0.135), 'plastic', 0.004))
        pl = box('plate', (0, fz + sgn * 0.043, hc), (0.52, 0.006, 0.11), 'plate', 0.002)
        pl['plate'] = 'front' if sgn > 0 else 'rear'
        parts.append(pl)
    # tow hooks (front)
    for s in (1, -1):
        c = V(s * 0.46, 1.86, 0.33)
        path = [c + Vector((0, -0.035 * math.sin(a), -0.035 * math.cos(a))) for a in np.linspace(-0.3, math.pi + 0.3, 12)]
        parts.append(tube(path, 0.009, 8, 'towhook', 'steel_black'))
    # rear step pad
    parts.append(box('steppad', (0, -1.86, 0.545), (0.40, 0.09, 0.012), 'rubber_trim', 0.004))
    j = join(parts, 'bumpers')
    shade(j, 40)
    return tag(j)

def build_front():
    """Black grille panel with headlight openings, grille slats, headlamps, indicators, badge."""
    parts = []; lenses = []
    face = F_FRONT - 0.004   # panel front face (slightly recessed)
    back = F_FRONT - 0.026
    panel = prism(rrect(-0.703, 0.617, 0.703, 0.927, 0.018), 'lh', back, face, 'gpanel', 'plastic')
    # holes: headlamps, grille opening
    cutters = []
    for s in (1, -1):
        cutters.append(cyl('hlcut', (s * 0.565, face, 0.785), (0, 1, 0), 0.094, 0.1, 40, 'plastic'))
    g = prism(rrect(-0.37, 0.668, 0.37, 0.905, 0.012), 'lh', back - 0.05, face + 0.05, 'gcut', 'plastic')
    cutters.append(g)
    for c in cutters:
        boolean(panel, c, transfer=False); delete_obj(c)
    bevel(panel, 0.003, 2, 40)
    parts.append(panel)
    # grille slats (horizontal) and dark radiator behind
    parts.append(box('radiator', (0, back - 0.03, 0.786), (0.76, 0.02, 0.26), 'underbody'))
    for i in range(6):
        h = 0.684 + i * 0.0405
        parts.append(box('slat', (0, face - 0.012, h), (0.74, 0.022, 0.016), 'plastic', 0.004, 2))
    for l in (-0.12, 0.12):
        parts.append(box('vbar', (l, face - 0.014, 0.786), (0.018, 0.02, 0.24), 'plastic', 0.004, 2))
    # headlamps: chrome bezel, reflector, lens (headlight), bulb shield
    for s in (1, -1):
        c = V(s * 0.565, face, 0.785)
        ax = Vector((0, -1, 0))  # forward
        parts.append(lathe([(0.087, -0.004), (0.093, 0.002), (0.099, 0.006), (0.103, 0.005), (0.104, 0.000), (0.101, -0.008), (0.092, -0.010)], 48, c, ax, 'bezel', 'chrome', closed_prof=True))
        parts.append(lathe([(0.0005, -0.075), (0.02, -0.074), (0.045, -0.062), (0.066, -0.042), (0.080, -0.022), (0.087, -0.006)], 40, c, ax, 'reflector', 'chrome'))
        lenses.append(headlamp_lens(s * 0.565, face, 0.785))
        parts.append(lathe([(0.0005, -0.030), (0.010, -0.031), (0.013, -0.040), (0.012, -0.055), (0.006, -0.062)], 16, c, ax, 'bulbcap', 'chrome'))
        # H4 bulb: glass envelope + shield cap, visible through the clear lens
        parts.append(lathe([(0.0005, -0.012), (0.004, -0.013), (0.0055, -0.018), (0.0058, -0.028), (0.005, -0.031)], 12, c, ax, 'bulb', 'chrome'))
        # indicators under the lamps
        parts.append(box('ind_f', (s * 0.61, face - 0.002, 0.648), (0.13, 0.014, 0.036), 'indicator', 0.004, 2))
        parts.append(box('ind_fb', (s * 0.61, face - 0.006, 0.648), (0.138, 0.012, 0.044), 'chrome', 0.003, 2))
    # badge (emblem) centered on the grille
    parts.append(badge_text('MARAL', V(0, F_FRONT + 0.0005, 0.9465), 0.019, (0, -1, 0), 'chrome'))
    j = join(parts, 'front')
    shade(j, 40)
    # faceted lenses joined after shading so their prism facets keep hard (flat) normals
    j = join([j] + lenses, 'front')
    return tag(j)

def headlamp_lens(cl, cf, ch, R=0.088, step=0.004):
    """Round 7" sealed-beam style lens with real optics geometry: a shallow dome carrying vertical prism flutes
    (16 mm period) over the outer area and a block-prism grid in the centre. Rectangular grid clipped to the circle
    (outer vertices clamped radially onto the rim). Flat shaded: the facets catch the sky like real pressed glass."""
    bm = bmesh.new()
    n = int(math.ceil((R + step) / step))
    tri = lambda t: abs((t % 1.0) - 0.5) * 2.0          # triangle wave 0..1
    grid = {}
    for i in range(-n, n + 1):
        for k in range(-n, n + 1):
            lx, lz = i * step, k * step
            r = math.hypot(lx, lz)
            if r > R:
                lx, lz = lx * R / r, lz * R / r; r = R
            dome = 0.0045 * (1 - (r / R) ** 2) - 0.006 * (r / R) ** 6
            centre = 1 - smooth(0.030, 0.042, r)
            flute = 0.0011 * tri(lx / 0.016 + 0.5) * (1 - centre) * (1 - smooth(0.080, R, r))
            block = 0.0007 * tri(lx / 0.008 + 0.5) * tri(lz / 0.012 + 0.5) * centre
            grid[(i, k)] = bm.verts.new(V(cl + lx, cf + dome + flute + block, ch + lz))
    lim = (R + step * 0.999) ** 2
    for i in range(-n, n):
        for k in range(-n, n):
            if min((a * step) ** 2 + (b * step) ** 2 for a, b in ((i, k), (i + 1, k), (i + 1, k + 1), (i, k + 1))) > lim: continue
            try: bm.faces.new((grid[(i, k)], grid[(i + 1, k)], grid[(i + 1, k + 1)], grid[(i, k + 1)]))
            except ValueError: pass
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0009)
    bmesh.ops.dissolve_degenerate(bm, dist=0.0002, edges=bm.edges[:])
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    for f in bm.faces:
        if f.calc_area() < 1e-9: continue
        if f.normal.y > 0: f.normal_flip()           # face forward (Blender -Y)
    ob = obj_from_bm('lens', bm, 'headlight')
    for p in ob.data.polygons: p.use_smooth = False
    return ob

def badge_text(txt, loc, size, facing, material, extrude=0.0012, name='badge'):
    """Extruded text mesh. loc and facing in Blender coords; text reads left-to-right for a viewer facing it."""
    cu = bpy.data.curves.new(name, 'FONT')
    cu.body = txt; cu.size = size; cu.extrude = extrude; cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
    cu.space_character = 1.1
    for fp in ('/System/Library/Fonts/Supplemental/Arial Black.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'):
        if os.path.exists(fp):
            cu.font = bpy.data.fonts.load(fp, check_existing=True); break
    cu.bevel_depth = 0.00025; cu.bevel_resolution = 1; cu.resolution_u = 3
    ob = bpy.data.objects.new(name, cu); coll('CAR').objects.link(ob)
    z = Vector(facing).normalized()
    upw = Vector((0, 0, 1)) if abs(z.z) < 0.95 else Vector((0, -1, 0))
    y = (upw - z * upw.dot(z)).normalized(); x = y.cross(z).normalized()
    M = Matrix((x, y, z)).transposed().to_4x4(); M.translation = Vector(loc)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    me.transform(M)
    o2 = bpy.data.objects.new(name, me); coll('CAR').objects.link(o2)
    bpy.data.objects.remove(ob); bpy.data.curves.remove(cu)
    o2.data.materials.clear(); o2.data.materials.append(MATS[material])
    return o2

def build_rear():
    parts = []
    # tail lamps at rear corners: lens outline in plan (l, f)
    for s in (1, -1):
        r = 0.045
        plan = fillet([(0.668, F_REAR + 0.037), (0.668, F_REAR - 0.003), (0.802, F_REAR - 0.003), (0.802, F_REAR + 0.037)], [0, 0, 0.022, 0], 6)
        # bezel / housing
        hs = prism(plan, 'lf', 0.585, 0.985, 'tl_house', 'plastic')
        bevel(hs, 0.003, 2, 40)
        # lens sections slightly proud
        up = prism(grow(plan, 0.003), 'lf', 0.735, 0.978, 'tl_red', 'brakelight')
        lo = prism(grow(plan, 0.003), 'lf', 0.592, 0.728, 'tl_amber', 'indicator')
        for o in (up, lo):
            bevel(o, 0.004, 2, 40)
        if s < 0:
            for o in (hs, up, lo):
                m = mirror_copy(o); delete_obj(o); parts.append(m)
        else:
            parts += [hs, up, lo]
    # spare wheel cover on the tail door
    sc = V(0.0, F_REAR - 0.02, 0.915)
    ax = Vector((0, 1, 0))  # backward
    cov = [(0.0005, 0.228), (0.12, 0.227), (0.25, 0.223), (0.325, 0.215), (0.352, 0.203), (0.366, 0.185), (0.372, 0.14), (0.372, 0.08), (0.364, 0.045), (0.335, 0.030)]
    cv = lathe(cov, 64, sc, ax, 'spare_cover', 'cover')
    # sag / wrinkles
    for v in cv.data.vertices:
        p = v.co; rel = p - sc
        rr = math.hypot(rel.x, rel.z)
        n = mnoise.noise(Vector((p.x * 9, p.z * 9, 3.3)))
        v.co.y += 0.004 * n * clamp(rr / 0.3)
        if rr > 0.3: v.co += Vector((rel.x, 0, rel.z)).normalized() * 0.003 * mnoise.noise(Vector((p.x * 20, p.z * 20, 1.1)))
    parts.append(cv)
    parts.append(lathe([(0.03, 0.0), (0.06, 0.005), (0.06, 0.035), (0.03, 0.04)], 16, sc, ax, 'spare_mount', 'steel_black'))
    # tail door hinges (right side, -l) and handle
    for h in (0.70, 1.38):
        parts.append(box('hinge', (-0.64, F_REAR - 0.012, h), (0.08, 0.025, 0.07), 'steel_black', 0.006))
    parts.append(box('tdhandle', (0.52, F_REAR - 0.010, 1.00), (0.12, 0.02, 0.035), 'chrome', 0.006))
    # rear wiper (pivot above the rear window, parked along its top edge)
    parts.append(cyl('rwpivot', (0.0, F_REAR - 0.012, 1.582), (0, 1, 0), 0.018, 0.022, 14, 'plastic'))
    parts.append(tube([V(0.0, F_REAR - 0.022, 1.582), V(-0.06, F_REAR - 0.020, 1.548), V(-0.43, F_REAR - 0.016, 1.538)], 0.0055, 6, 'rwarm', 'steel_black'))
    parts.append(tube([V(-0.05, F_REAR - 0.010, 1.532), V(-0.47, F_REAR - 0.010, 1.532)], 0.0045, 6, 'rwblade', 'steel_black'))
    parts.append(badge_text('MARAL 4x4', V(0.42, F_REAR - 0.004, 0.62), 0.028, (0, 1, 0), 'chrome'))
    # plate lamps
    parts.append(box('platelamp', (0, -1.842, 0.54), (0.14, 0.03, 0.02), 'plastic', 0.005))
    # exhaust tailpipe (right side)
    path = [V(-0.42, -1.30, 0.30), V(-0.43, -1.60, 0.29), V(-0.44, -1.80, 0.275), V(-0.45, -1.90, 0.265)]
    parts.append(tube(path, 0.024, 14, 'tailpipe', 'underbody', cap=False))
    j = join(parts, 'rear')
    shade(j, 40)
    return tag(j)

def build_greenhouse_details():
    parts = []
    # --- rain gutters along roof sides + down the A pillar (left side, mirrored)
    gp = [(-1.74, 1.608), (-1.2, 1.609), (0.0, 1.610), (0.22, 1.610)]
    pth = []
    for f, h in gp: pth.append((f, h))
    # A-pillar down to belt following windshield edge offset back
    for h in np.linspace(1.59, 1.12, 6):
        pth.append((ws_f(h) - 0.012, h))
    path = [V(gh_l(h) - 0.004, f, h) for f, h in pth]
    prof = [(0.0, -0.004), (0.010, -0.005), (0.013, 0.002), (0.011, 0.012), (0.007, 0.013), (0.006, 0.004), (0.0, 0.004)]
    g = sweep(path, [(s, u) for s, u in prof], False, (1, 0, 0), 'gutter', 'paint')
    # --- window seals + glass
    glass = []
    for poly in (door_win_poly(), quarter_win_poly()):
        seal_poly = grow(poly, 0.004)
        path = [V(gh_l(h) - 0.001, f, h) for f, h in seal_poly]
        # side seal: profile in (s=in-plane outward, u=outward normal +l)
        path = resample_closed(path, 0.042)
        parts.append(sweep(path, rrect(-0.010, -0.006, 0.006, 0.004, 0.003, 2), True, (1, 0, 0), 'seal', 'rubber_trim'))
        gpoly = grow(poly, 0.012)
        bm = bmesh.new()
        vs = [bm.verts.new(V(gh_l(h) - 0.009, f, h)) for f, h in gpoly]
        bm.faces.new(vs)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        for fce in bm.faces:
            if fce.normal.x < 0: bmesh.ops.reverse_faces(bm, faces=[fce])
        glass.append(obj_from_bm('sideglass', bm, 'glass'))
    # door glass run channel (vertical black strip at B-pillar front) - painted frame reads better with black
    # windshield
    h0, h1 = 1.100, 1.585; w0, w1 = 0.625, 0.585
    poly = fillet([(-w0, h0), (w0, h0), (w1, h1), (-w1, h1)], [0.035, 0.035, 0.05, 0.05], 6)
    d_up = (ws_point(0, 1.5) - ws_point(0, 1.1)).normalized()
    n = d_up.cross(Vector((1, 0, 0))).normalized()
    if n.y > 0: n = -n
    sp = grow(poly, 0.004)
    path = resample_closed([ws_point(u, v) - n * 0.001 for u, v in sp], 0.042)
    wseal = sweep(path, rrect(-0.010, -0.006, 0.006, 0.004, 0.003, 2), True, tuple(n), 'wseal', 'rubber_trim')
    parts.append(wseal)
    gp2 = grow(poly, 0.012)
    bm = bmesh.new(); vs = [bm.verts.new(ws_point(u, v) - n * 0.008) for u, v in gp2]; bm.faces.new(vs)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for fce in bm.faces:
        if fce.normal.dot(n) < 0: bmesh.ops.reverse_faces(bm, faces=[fce])
    ws = obj_from_bm('windshield', bm, 'glass')
    # rear window
    rp = rrect(-0.60, 1.14, 0.60, 1.555, 0.05, 5)
    path = resample_closed([V(l, F_REAR - 0.001, h) for l, h in grow(rp, 0.004)], 0.042)
    parts.append(sweep(path, rrect(-0.010, -0.006, 0.006, 0.004, 0.003, 2), True, (0, 1, 0), 'rseal', 'rubber_trim'))
    bm = bmesh.new(); vs = [bm.verts.new(V(l, F_REAR + 0.008, h)) for l, h in grow(rp, 0.012)]; bm.faces.new(vs)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for fce in bm.faces:
        if fce.normal.y < 0: bmesh.ops.reverse_faces(bm, faces=[fce])
    rw = obj_from_bm('rearglass', bm, 'glass')
    gl = join(glass, 'glass_l')
    gl = sym(gl)
    glass_all = join([gl, ws, rw], 'glass')
    glass_all.data.shade_smooth()
    tag(glass_all, 'none')
    others = [p for p in parts if not p.name.startswith('seal')]
    sealsl = join([g] + [p for p in parts if p.name.startswith('seal')], 'seals_l')
    sealsl = sym(sealsl)
    j = join([sealsl] + others, 'gh_details')
    shade(j, 50)
    return tag(j), glass_all

def resample_closed(path, step):
    P = [Vector(p) for p in path] + [Vector(path[0])]
    out = polyline_resample(P, step)[:-1]
    return out

def build_side_details():
    parts = []
    # mirrors (left side; mirrored later)
    base = V(0.79, 0.60, 1.085)
    arm = [base, V(0.86, 0.60, 1.10), V(0.93, 0.585, 1.13)]
    parts.append(tube(arm, 0.010, 8, 'mirrorarm', 'steel_black'))
    parts.append(box('mirrorfoot', (0.79, 0.60, 1.08), (0.03, 0.07, 0.045), 'plastic', 0.008))
    hc = (0.975, 0.575, 1.16)
    hsg = box('mirror_house', hc, (0.175, 0.07, 0.125), 'plastic', 0.022, 3)
    parts.append(hsg)
    parts.append(box('mirror_glass', (hc[0], hc[1] - 0.036, hc[2]), (0.155, 0.004, 0.105), 'chrome', 0.012, 3))
    # door handle + lock
    parts.append(box('dhandle_plate', (0.803, -0.33, 0.925), (0.008, 0.13, 0.042), 'plastic', 0.004))
    parts.append(box('dhandle', (0.809, -0.33, 0.925), (0.010, 0.10, 0.022), 'chrome', 0.004))
    parts.append(cyl('dlock', (0.806, -0.23, 0.925), (1, 0, 0), 0.011, 0.012, 16, 'chrome'))
    # fuel flap face (slightly proud) - painted, left side only (built separately, not mirrored)
    # side repeater
    parts.append(box('repeater', (0.803, 1.36, 0.80), (0.008, 0.06, 0.025), 'indicator', 0.004))
    # mudflaps
    for fc, drop in ((FAX - ARCH_R - 0.04, 0.17), (RAX - ARCH_R - 0.045, 0.15)):
        bm = bmesh.new()
        pts = [(0.64, 0.46), (0.845, 0.46), (0.845, drop), (0.64, drop)]
        vs = [bm.verts.new(V(l, fc - 0.02 * (0.46 - h), h)) for l, h in pts]
        bm.faces.new(vs)
        r = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
        for v in [e for e in r['geom'] if isinstance(e, bmesh.types.BMVert)]: v.co.y += 0.006
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        o = obj_from_bm('mudflap', bm, 'rubber_trim'); bevel(o, 0.002, 1, 40)
        parts.append(o)
    # wiper arms are built separately (asymmetric)
    j = join(parts, 'side_l')
    shade(j, 45)
    return sym(tag(j))

def build_asym_details():
    parts = []
    # fuel flap (left rear quarter): a slightly proud disc, flush in the seam ring
    parts.append(cyl('fuelflap', (W_BODY - 0.0005, -1.30, 0.885), (1, 0, 0), 0.052, 0.004, 32, 'paint'))
    # wipers: pivots on cowl, parked along the base of the windshield toward -l (passenger)
    d_up = (ws_point(0, 1.5) - ws_point(0, 1.1)).normalized()
    wn = d_up.cross(Vector((1, 0, 0))).normalized()
    if wn.y > 0: wn = -wn
    for pl in (0.34, -0.20):
        p0 = V(pl, 0.785, 1.072)
        arm = [p0, ws_point(pl - 0.06, 1.118) + wn * 0.028, ws_point(pl - 0.44, 1.128) + wn * 0.022]
        parts.append(tube(arm, 0.0055, 6, 'wiperarm', 'steel_black'))
        parts.append(cyl('wiperpivot', (pl, 0.785, 1.066), (0, 0, 1), 0.016, 0.02, 12, 'plastic'))
        blade = [ws_point(pl - 0.03, 1.122) + wn * 0.010, ws_point(pl - 0.47, 1.128) + wn * 0.010]
        parts.append(tube(blade, 0.0045, 6, 'wiperblade', 'steel_black'))
        parts.append(tube([ws_point(pl - 0.03, 1.122) + wn * 0.004, ws_point(pl - 0.47, 1.128) + wn * 0.004], 0.002, 4, 'wiperedge', 'rubber_trim'))
    # cowl vent panel
    cw = box('cowl', (0, 0.772, 1.061), (1.36, 0.05, 0.012), 'plastic', 0.003)
    parts.append(cw)
    for i in range(22):
        l = -0.55 + i * 0.0525
        parts.append(box('cowlslot', (l, 0.772, 1.0675), (0.034, 0.022, 0.003), 'underbody'))
    # antenna (front left, at A-pillar base)
    parts.append(cyl('antbase', (0.765, 0.70, 1.075), (0, 0, 1), 0.012, 0.02, 12, 'plastic'))
    ant = [V(0.765, 0.70, 1.08), V(0.77, 0.62, 1.50), V(0.775, 0.56, 1.80)]
    parts.append(tube(ant, 0.0022, 6, 'antenna', 'chrome'))
    # roof rack
    parts += build_roof_rack()
    j = join(parts, 'asym')
    shade(j, 45)
    return tag(j)

def build_roof_rack():
    parts = []
    hr = 1.705
    for s in (1, -1):
        rail = [V(s * 0.64, 0.20, hr - 0.02), V(s * 0.645, 0.12, hr), V(s * 0.645, -1.55, hr), V(s * 0.64, -1.63, hr - 0.02)]
        parts.append(tube(rail, 0.013, 10, 'rail', 'steel_black'))
        for f in (0.13, -0.55, -1.56):
            foot = [V(s * 0.645, f, hr), V(s * 0.66, f, 1.64), V(s * 0.70, f, 1.62)]
            parts.append(tube(foot, 0.011, 8, 'foot', 'steel_black'))
            parts.append(box('footpad', (s * 0.705, f, 1.617), (0.05, 0.06, 0.016), 'rubber_trim', 0.004))
    for f in (0.15, -0.20, -0.55, -0.90, -1.25, -1.58):
        parts.append(tube([V(-0.645, f, hr), V(0.645, f, hr)], 0.011, 10, 'xbar', 'steel_black'))
    # front wind deflector bar
    parts.append(tube([V(-0.64, 0.21, hr - 0.02), V(-0.55, 0.26, hr + 0.035), V(0.55, 0.26, hr + 0.035), V(0.64, 0.21, hr - 0.02)], 0.011, 10, 'deflector', 'steel_black'))
    return parts

def build_underbody():
    parts = []
    for s in (1, -1):
        parts.append(box('rail', (s * 0.47, -0.05, 0.36), (0.075, 3.62, 0.12), 'underbody', 0.006))
    for f in (1.55, 0.55, -0.60, -1.60):
        parts.append(box('xmember', (0, f, 0.35), (0.90, 0.08, 0.08), 'underbody', 0.005))
    # axles + diffs
    for fa, dl in ((FAX, 0.10), (RAX, 0.0)):
        parts.append(cyl('axle', (0, fa, R_TIRE), (1, 0, 0), 0.038, 1.28, 14, 'underbody'))
        d = lathe([(0.03, 0.10), (0.09, 0.07), (0.12, 0.02), (0.12, -0.03), (0.09, -0.08), (0.04, -0.10)], 20, V(dl, fa, R_TIRE), (0, 1, 0), 'diff', 'underbody')
        parts.append(d)
        for s in (1, -1):
            # leaf springs
            for k in range(3):
                pth = [V(s * 0.47, fa + 0.42, 0.42 - k * 0.012), V(s * 0.47, fa, 0.395 - k * 0.012 - 0.02 * k), V(s * 0.47, fa - 0.42, 0.42 - k * 0.012)]
                pth = [V(s * 0.47, fa + t, 0.40 + 0.25 * t * t - k * 0.011) for t in np.linspace(0.44 - k * 0.1, -0.44 + k * 0.1, 7)]
                parts.append(sweep(pth, rrect(-0.03, -0.004, 0.03, 0.004, 0.002, 1), False, (0, 0, 1), 'leaf', 'underbody'))
            # shocks
            parts.append(tube([V(s * 0.55, fa - 0.12, 0.34), V(s * 0.52, fa - 0.20, 0.62)], 0.022, 10, 'shock', 'underbody'))
    # drivetrain
    parts.append(box('sump', (0.0, 0.85, 0.40), (0.34, 0.40, 0.14), 'underbody', 0.02))
    parts.append(box('gearbox', (0.0, 0.25, 0.42), (0.26, 0.55, 0.16), 'underbody', 0.03))
    parts.append(box('tcase', (0.05, -0.12, 0.38), (0.24, 0.22, 0.14), 'underbody', 0.02))
    parts.append(tube([V(0.05, -0.2, 0.37), V(0.0, RAX + 0.12, R_TIRE + 0.02)], 0.03, 10, 'propshaft_r', 'underbody'))
    parts.append(tube([V(0.05, -0.05, 0.37), V(0.10, FAX - 0.12, R_TIRE + 0.02)], 0.028, 10, 'propshaft_f', 'underbody'))
    # exhaust along right side with muffler
    ex = [V(-0.18, 1.0, 0.36), V(-0.30, 0.4, 0.33), V(-0.34, -0.5, 0.32), V(-0.40, -1.0, 0.31), V(-0.42, -1.30, 0.30)]
    parts.append(tube(ex, 0.024, 10, 'exhaust', 'underbody'))
    mf = lathe([(0.0005, -0.30), (0.07, -0.29), (0.085, -0.25), (0.09, 0.0), (0.085, 0.25), (0.07, 0.29), (0.0005, 0.30)], 20, V(-0.33, -0.1, 0.32), (0.03, -1, 0), 'muffler', 'underbody')
    parts.append(mf)
    # fuel tank
    parts.append(box('tank', (0.22, -1.42, 0.37), (0.44, 0.42, 0.14), 'underbody', 0.02))
    # steering tie rod
    parts.append(tube([V(-0.62, FAX + 0.13, 0.33), V(0.62, FAX + 0.13, 0.33)], 0.013, 8, 'tierod', 'underbody'))
    j = join(parts, 'underbody')
    shade(j, 40)
    return tag(j)

# ------------------------------------------------------------------------------------------------
# WHEELS
# ------------------------------------------------------------------------------------------------
TIRE_TILES = 2  # texture repeats around circumference

def tire_profile():
    """(r, x) outer profile from inner bead (x<0) to outer bead (x>0), plus arclength parameter v."""
    hw = W_TIRE / 2
    pts = [(0.197, -0.080), (0.206, -0.090), (0.228, -0.102), (0.258, -0.110), (0.290, -0.112), (0.318, -0.109),
           (0.337, -0.103), (0.348, -0.094), (0.352, -0.086)]
    rhs = [(r, -x) for r, x in reversed(pts)]
    mid = [(0.3525, -0.06), (0.3530, 0.0), (0.3525, 0.06)]
    return pts + mid + rhs

def build_tire(segs=96):
    prof = tire_profile()
    # arclength along profile for UV v
    L = [0.0]
    for a, b in zip(prof[:-1], prof[1:]): L.append(L[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    tot = L[-1]
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new('UVMap')
    ax = Vector((1, 0, 0))
    rings = []
    for k in range(segs + 1):
        t = 2 * math.pi * k / segs
        rings.append([bm.verts.new(Vector((x, -r * math.sin(t), r * math.cos(t)))) for r, x in prof])
    for k in range(segs):
        for i in range(len(prof) - 1):
            f = bm.faces.new((rings[k][i], rings[k][i + 1], rings[k + 1][i + 1], rings[k + 1][i]))
            for lp, (kk, ii) in zip(f.loops, ((k, i), (k, i + 1), (k + 1, i + 1), (k + 1, i))):
                lp[uv].uv = (TIRE_TILES * kk / segs, 0.8 * L[ii] / tot)
    # inner bead closure (short cylinder hidden inside rim)
    # tread blocks
    N = 40
    rb, rt = 0.3505, 0.3615
    blocks = []
    for p in range(N):
        th0 = 2 * math.pi * p / N
        pitch = 2 * math.pi / N
        # center blocks (staggered)
        blocks.append((th0 + pitch * 0.06, th0 + pitch * 0.46, 0.004, 0.047))
        blocks.append((th0 + pitch * 0.56, th0 + pitch * 0.96, -0.047, -0.004))
        # zig: second center rows
        blocks.append((th0 + pitch * 0.52, th0 + pitch * 0.92, 0.004, 0.047))
        blocks.append((th0 + pitch * 0.02, th0 + pitch * 0.42, -0.047, -0.004))
        # shoulders
        blocks.append((th0 + pitch * 0.08, th0 + pitch * 0.88, 0.055, 0.100))
        blocks.append((th0 + pitch * 0.58, th0 + pitch * 1.38, -0.100, -0.055))
    for th0, th1, x0, x1 in blocks:
        ins = 0.0035
        def P(th, x, r): return Vector((x, -r * math.sin(th), r * math.cos(th)))
        dth = ins / rt
        # bottom slightly sunk
        rbb = rb - 0.002 if abs(x0) < 0.09 and abs(x1) < 0.09 else rb - 0.012
        bot = [P(th0, x0, rbb), P(th1, x0, rbb), P(th1, x1, rbb), P(th0, x1, rbb)]
        rtt_in = rt if abs(x1) < 0.09 and abs(x0) < 0.09 else rt - 0.002
        top = [P(th0 + dth, x0 + ins, rtt_in), P(th1 - dth, x0 + ins, rtt_in), P(th1 - dth, x1 - ins, rtt_in), P(th0 + dth, x1 - ins, rtt_in)]
        # shoulder blocks slope down at the outer edge
        for q in range(4):
            xx = top[q].x
            if abs(xx) > 0.085:
                top[q] = P(th0 + dth if q in (0, 3) else th1 - dth, xx, rt - 0.010)
                bot[q] = P(th0 if q in (0, 3) else th1, xx + (0.004 if xx > 0 else -0.004), 0.336)
        vb = [bm.verts.new(p) for p in bot]; vt = [bm.verts.new(p) for p in top]
        faces = [bm.faces.new(vt)]
        for i in range(4):
            j = (i + 1) % 4
            faces.append(bm.faces.new((vb[i], vb[j], vt[j], vt[i])))
        for f in faces:
            for lp in f.loops:
                co = lp.vert.co
                th = math.atan2(-co.y, co.z) % (2 * math.pi)
                # tread band: v in [0.5 - w, 0.5 + w] of the first half of texture (profile arclen ~ mid)
                lp[uv].uv = (TIRE_TILES * th / (2 * math.pi), 0.82 + 0.16 * clamp((co.x + 0.105) / 0.21))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = obj_from_bm('tire', bm, 'rubber')
    shade(ob, 50)
    return ob

def build_rim():
    parts = []
    ax = (1, 0, 0)
    c = Vector((0, 0, 0))
    # barrel inner surface + flanges (visible), x>0 outboard. Wheel offset: deep dish.
    barrel = [(0.1995, 0.080), (0.2065, 0.076), (0.2070, 0.070), (0.2010, 0.066), (0.1930, 0.063), (0.1885, 0.058),
              (0.1880, 0.040), (0.1700, 0.025), (0.1665, 0.010), (0.1665, -0.035), (0.1850, -0.050), (0.1880, -0.062),
              (0.2010, -0.067), (0.2070, -0.071), (0.2065, -0.077), (0.1995, -0.080),
              (0.1955, -0.078), (0.1840, -0.066), (0.1810, -0.052), (0.1620, -0.036), (0.1620, 0.008), (0.1660, 0.024),
              (0.1840, 0.038), (0.1845, 0.058), (0.1955, 0.078)]
    parts.append(lathe(barrel, 48, c, ax, 'barrel', 'wheel', closed_prof=True))   # (v4) 64 -> 48 segs (tri budget)
    # disc (center): dish with raised ring
    disc_o = [(0.1650, 0.004), (0.155, 0.006), (0.140, 0.014), (0.125, 0.018), (0.112, 0.016), (0.100, 0.006), (0.090, -0.004),
              (0.080, -0.006), (0.050, -0.006), (0.040, -0.004), (0.036, 0.004)]
    disc_i = [(r, x - 0.005) for r, x in reversed(disc_o)]
    disc = lathe(disc_o + disc_i, 48, c, ax, 'disc', 'wheel', closed_prof=True)
    # vent slots cut through the disc
    cutters = []
    for k in range(6):
        a = 2 * math.pi * (k + 0.5) / 6
        bmc = bmesh.new()
        pts = []
        for j in range(10):
            aa = a - 0.20 + 0.40 * j / 9
            pts.append((0.139 * math.cos(aa), 0.139 * math.sin(aa)))
        for j in range(10):
            aa = a + 0.17 - 0.34 * j / 9
            pts.append((0.112 * math.cos(aa), 0.112 * math.sin(aa)))
        pts = fillet(pts, [0.009 if j in (0, 9, 10, 19) else 0 for j in range(20)], 3)
        A = [bmc.verts.new(Vector((-0.05, -y, z))) for y, z in pts]
        B = [bmc.verts.new(Vector((0.05, -y, z))) for y, z in pts]
        bmc.faces.new(A[::-1]); bmc.faces.new(B)
        for i in range(len(pts)):
            j = (i + 1) % len(pts); bmc.faces.new((A[i], A[j], B[j], B[i]))
        bmesh.ops.recalc_face_normals(bmc, faces=bmc.faces)
        cutters.append(obj_from_bm('slotcut', bmc, 'wheel', coll('CUT')))
    boolean(disc, coll('CUT'), transfer=False)
    for o in cutters: delete_obj(o)
    parts.append(disc)
    # hub cap (center dome) and lug nuts
    parts.append(lathe([(0.0005, 0.048), (0.018, 0.047), (0.030, 0.042), (0.037, 0.030), (0.039, 0.012), (0.039, -0.004)], 24, c, ax, 'hubcap', 'wheel'))
    for k in range(5):
        a = 2 * math.pi * k / 5
        pc = Vector((0.0, -0.0698 * math.sin(a), 0.0698 * math.cos(a)))
        nut = lathe([(0.0005, 0.020), (0.008, 0.020), (0.0105, 0.017), (0.0105, 0.004), (0.013, 0.0), (0.013, -0.006)], 6, pc, ax, 'nut', 'wheel')
        parts.append(nut)
        parts.append(lathe([(0.0005, 0.026), (0.004, 0.026), (0.0045, 0.018)], 8, pc, ax, 'stud', 'wheel'))
    # brake disc + dust shield (inboard)
    parts.append(lathe([(0.075, -0.045), (0.133, -0.045), (0.134, -0.050), (0.134, -0.066), (0.133, -0.070), (0.075, -0.070), (0.06, -0.06), (0.06, -0.045)], 40, c, ax, 'bdisc', 'brake', closed_prof=True))
    parts.append(lathe([(0.05, -0.095), (0.150, -0.090), (0.155, -0.080), (0.15, -0.075)], 32, c, ax, 'shield', 'brake'))
    j = join(parts, 'rim')
    shade(j, 40)
    return j

def build_wheels():
    tire = build_tire(96 if not FAST else 72)
    rim = build_rim()
    tag(rim, 'ext'); tag(tire, 'tire')
    return tire, rim

# ------------------------------------------------------------------------------------------------
# INTERIOR
# ------------------------------------------------------------------------------------------------
SEAT_L = 0.36
SEAT_F = -0.36                          # backrest pivot (bottom of backrest)
EYE = (SEAT_L, -0.31, 1.385)            # driver eye (seat_cam)
SW_C = (SEAT_L, 0.285, 1.075)           # steering wheel hub center
SW_ALPHA = math.radians(28)             # column angle above horizontal
POD_T, POD_B = (0.455, 1.155), (0.485, 1.045)   # gauge face line (f,h) top/bottom
GAUGE_TILT = math.atan2(POD_B[0] - POD_T[0], POD_T[1] - POD_B[1])
FRESH_PIV = (0.0, 0.252, 1.492)          # (legacy) air freshener string top
# (v4) rear-view mirror: pivot = ball joint under the header stem; yawed toward the driver so its normal bisects
# eye->mirror and mirror->rear window (a real driver sets it so), pitched down a little
RVM_PIV = (0.0, 0.250, 1.545)
RVM_YAW, RVM_PITCH = -16.5, -6.8               # degrees (Blender Z, then local X): centre ray ~1 deg below horizontal
RVM_GLASS = (0.0, 0.2245, 1.522, 0.190, 0.042)  # face centre (l,f,h) before rotation, visible silvered width/height
CHARM_PIV = (0.0, 0.2575, 1.562)               # prayer-bead loop around the mirror stem (just above the housing)
KEY_LOCK = (0.305, 0.356, 1.022)               # ignition lock face centre on the right side of the column shroud (rear edge: visible below the horn pad)
KEY_AXIS = (-1.0, -0.28, 0.10)                 # key insertion axis, pointing out of the lock (car coords)

# Dash print areas in the gauge texture (u0, v0, u1, v1); the dials use the top half and the lower-left quadrant.
DASH_PRINTS = {
    'radio': (0.52, 0.36, 0.98, 0.491),
    'heater': (0.52, 0.225, 0.98, 0.34),
    'switches': (0.52, 0.13, 0.76, 0.202),
    'warn': (0.52, 0.08, 0.98, 0.0994),
}
NEEDLE_UV = (0.89, 0.165)                # orange swatch for the needles
NEEDLE_CAP_UV = (0.75, 0.03)             # black background (needle hub caps)

def obox(name, c, ax, size, material=None):
    """Oriented box: c = centre (car coords), ax = 3 car-space axis vectors, size = full extents along them."""
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    A = [Vector((a[0], -a[1], a[2])).normalized() for a in ax]
    cc = V(*c)
    for v in bm.verts:
        v.co = cc + A[0] * (v.co.x * size[0]) + A[1] * (v.co.y * size[1]) + A[2] * (v.co.z * size[2])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return obj_from_bm(name, bm, material)

def screw(c, nrm, r=0.0036, rot=0.0):
    """Phillips pan-head screw (zinc) at car point c facing car normal nrm, with a dark cross recess. ~70 tris."""
    n = Vector(nrm).normalized()
    t = n.cross(Vector((0, 0, 1)) if abs(n.z) < 0.9 else Vector((1, 0, 0))).normalized()
    b = n.cross(t).normalized()
    ca, sa = math.cos(rot), math.sin(rot)
    t, b = t * ca + b * sa, b * ca - t * sa
    nb = Vector((n.x, -n.y, n.z))
    head = lathe([(r, -0.0004), (r * 0.95, 0.0009), (r * 0.55, 0.0018), (0.0002, 0.0020)], 8, V(*c), tuple(nb), 'screw', 'chrome')
    parts = [head]
    top = Vector(c) + n * 0.0016
    for a in (t, b):
        parts.append(obox('screwx', tuple(top), (a, a.cross(n), n), (r * 1.25, 0.0009, 0.0012), 'int_black'))
    return parts

def gaiter(base, top, r0, r1, sq=0.35, nh=15, na=22, folds=4.5, amp=0.006, seed=1, material='leather', lean=None):
    """Stitched leatherette gear gaiter: squarish base tied to the trim ring, round neck at the lever, irregular
    horizontal folds (deeper toward the base, sagging where the leather bunches). base/top in car coords."""
    b0 = Vector(base); b1 = Vector(top); ax = (b1 - b0); Hh = ax.length; ax.normalize()
    u = ax.cross(Vector((0, 0, 1)) if abs(ax.z) < 0.9 else Vector((1, 0, 0)))
    if u.length < 1e-6: u = Vector((1, 0, 0))
    u = Vector((1, 0, 0)) - ax * ax.x; u.normalize(); v = ax.cross(u).normalized()
    bm = bmesh.new(); rings = []
    for i in range(nh + 1):
        tt = i / nh
        rr = r0 + (r1 - r0) * (tt ** 0.75)
        fold_w = (1 - tt) ** 0.6
        ring = []
        for k in range(na):
            a = 2 * math.pi * k / na
            ca, sa = math.cos(a), math.sin(a)
            # superellipse toward the base: square-ish
            e = 2.0 + 6.0 * sq * (1 - tt) ** 2
            den = (abs(ca) ** e + abs(sa) ** e) ** (1.0 / e)
            rq = rr / max(den, 1e-6)
            nz = mnoise.noise(Vector((ca * 3.1 + seed, sa * 3.1, tt * 2.0)))
            fold = amp * fold_w * max(0.0, math.sin(2 * math.pi * (folds * tt + 0.25 * nz + 0.1 * seed))) ** 1.5 * (0.75 + 0.5 * nz)
            sag = 0.004 * fold_w * (0.5 + 0.5 * mnoise.noise(Vector((ca * 2 + 7, sa * 2, seed))))
            p = b0 + ax * (Hh * tt - sag) + (u * ca + v * sa) * (rq + fold)
            ring.append(bm.verts.new(V(*p)))
        rings.append(ring)
    for i in range(nh):
        A, B = rings[i], rings[i + 1]
        for k in range(na):
            j = (k + 1) % na
            bm.faces.new((A[k], A[j], B[j], B[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = obj_from_bm('gaiter', bm, material)
    # make sure normals point away from the axis
    me = ob.data; pc = me.polygons[len(me.polygons) // 2]
    cen = V(*(b0 + ax * Hh * 0.5))
    if (pc.center - cen).dot(pc.normal) < 0:
        bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.reverse_faces(bm, faces=bm.faces); bm.to_mesh(me); bm.free()
    return ob

def trim_frame(c, w, d, bar, th, name='gtrim'):
    """Square trim ring (4 bevelled bars) lying on a horizontal surface at c (car coords, bottom height)."""
    l, f, h = c; parts = []
    for sl, sf, sw, sd in ((0, (d - bar) / 2, w, bar), (0, -(d - bar) / 2, w, bar), ((w - bar) / 2, 0, bar, d - 2 * bar), (-(w - bar) / 2, 0, bar, d - 2 * bar)):
        parts.append(box(name, (l + sl, f + sf, h + th / 2), (sw, sd, th), 'int_black', 0.0025, 2))
    return parts

def apillar_trim(sgn):
    """(v4) A-pillar inner trim: a moulded D-section cover spanning windshield edge -> door-glass front edge (7-9 cm
    face, like a real early-90s 4x4), instead of the thin round rod."""
    secs = []
    for h in np.linspace(1.065, 1.60, 7):
        wsw = 0.625 + (0.585 - 0.625) * (h - 1.10) / 0.485
        fw = ws_f(h) - 0.012
        fd = door_front_f(h) - 0.034
        lg = gh_l(h) - 0.009
        A = (wsw - 0.018, fw - 0.010)
        Bp = (lg - 0.028, fd - 0.006)
        Ao = (wsw + 0.03, fw + 0.004); Bo = (lg, fd + 0.02); K = (lg, fw + 0.004)
        dx, dy = Bp[0] - A[0], Bp[1] - A[1]; L = math.hypot(dx, dy)
        nx, ny = dy / L, -dx / L
        if nx > 0: nx, ny = -nx, -ny
        pts_in = []
        for t in np.linspace(0, 1, 7):
            bul = 4 * t * (1 - t) * 0.018
            pts_in.append((A[0] + dx * t + nx * bul, A[1] + dy * t + ny * bul))
        poly = [Ao] + pts_in + [Bo, K]
        secs.append([V(sgn * l, f, h) for l, f in poly])
    ob = loft(secs, True, True, True, 'atrim', 'interior')
    mid_h = 1.34
    wsw = 0.625 + (0.585 - 0.625) * (mid_h - 1.10) / 0.485
    A = Vector((wsw - 0.018, ws_f(mid_h) - 0.022)); B = Vector((gh_l(mid_h) - 0.037, door_front_f(mid_h) - 0.040))
    m = (A + B) / 2; d = (B - A).normalized(); nrm = Vector((d.y, -d.x))
    if nrm.x > 0: nrm = -nrm
    sp = m + nrm * 0.0185
    return ob, (sgn * sp.x, sp.y, mid_h), (sgn * nrm.x, nrm.y, 0.0)

def rvm_parts():
    """(v4) Rear-view mirror housing + silvered glass + day/night tab, rotated about the stem ball joint."""
    parts = []
    parts.append(box('rvm', (0, 0.245, 1.522), (0.22, 0.035, 0.065), 'int_black', 0.012, 3))
    parts.append(box('rvm_glass', (0, 0.2265, 1.522), (0.205, 0.004, 0.052), 'chrome', 0.008, 2))
    parts.append(box('rvm_tab', (0, 0.238, 1.485), (0.03, 0.012, 0.012), 'int_black', 0.003, 2))
    piv = V(*RVM_PIV)
    R = Matrix.Translation(piv) @ Matrix.Rotation(math.radians(RVM_YAW), 4, 'Z') @ Matrix.Rotation(math.radians(RVM_PITCH), 4, 'X') @ Matrix.Translation(-piv)
    for o in parts: o.data.transform(R)
    return parts

def rvm_frame():
    """Mirror face frame in three.js car coords (x=l, y=h, z=f): centre, normal (toward the driver), up, width, height."""
    piv = V(*RVM_PIV)
    R = Matrix.Rotation(math.radians(RVM_YAW), 4, 'Z') @ Matrix.Rotation(math.radians(RVM_PITCH), 4, 'X')
    l, f, h, w, hh = RVM_GLASS
    c = piv + (R @ (V(l, f, h) - piv))
    n = R @ Vector((0, 1, 0)); up = R @ Vector((0, 0, 1))
    th = lambda b: [round(b.x, 5), round(b.z, 5), round(-b.y, 5)]
    return {'center': th(c), 'normal': th(n), 'up': th(up), 'width': w, 'height': hh}

def map_mesh(c, w, d, yaw_deg, folds=3, curl=0.0, name='roadmap'):
    """Folded sheet of paper lying on a surface at c (car coords): w along l, d along f; zig-zag folds across l,
    optional curl (lifted far edge). Thin (two-sided via a 0.6 mm solid)."""
    nx, ny = (folds * 2 + 1 if folds else 6), 4
    bm = bmesh.new(); vs = []
    ca, sa = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    for j in range(ny + 1):
        row = []
        for i in range(nx + 1):
            u = i / nx - 0.5; v = j / ny - 0.5
            lift = 0.0
            if folds:
                panel = i * (folds + 1) / nx
                lift = 0.0035 * abs(math.sin(math.pi * panel)) + 0.0012 * (j % 2)
            if curl: lift += curl * (v + 0.5) ** 2.2 + 0.002 * math.sin(u * 7)
            lx, ly = u * w, v * d
            row.append(bm.verts.new(V(c[0] + lx * ca - ly * sa, c[1] + lx * sa + ly * ca, c[2] + lift)))
        vs.append(row)
    for j in range(ny):
        for i in range(nx):
            bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for fce in bm.faces:
        if fce.normal.z < 0: fce.normal_flip()
    ob = obj_from_bm(name, bm, 'paper')
    m = ob.modifiers.new('solid', 'SOLIDIFY'); m.thickness = 0.0006; m.offset = -1
    apply_mods(ob)
    return ob

def key_frame():
    a = Vector(KEY_AXIS).normalized()
    up = Vector((0, 0, 1)); up = (up - a * up.dot(a)).normalized()
    n = a.cross(up).normalized()
    return a, up, n

def key_static():
    """Ignition lock bezel + the inserted key (metal shoulder, black rubberised head)."""
    a, up, n = key_frame()
    c = Vector(KEY_LOCK)
    parts = [cyl('lockbezel', tuple(c - a * 0.004), tuple(a), 0.0150, 0.010, 16, 'chrome', 0.0012)]
    parts.append(cyl('lockface', tuple(c + a * 0.0012), tuple(a), 0.0105, 0.0016, 14, 'int_black'))
    parts.append(obox('keyshoulder', tuple(c + a * 0.0085), (a, up, n), (0.010, 0.012, 0.0022), 'chrome'))
    # head: rounded plate in the (a, up) plane
    hp = rrect(0.0, -0.0135, 0.034, 0.0135, 0.009, 4)
    bm = bmesh.new()
    A = [bm.verts.new(V(*(c + a * (0.012 + x) + up * y - n * 0.0032))) for x, y in hp]
    B = [bm.verts.new(V(*(c + a * (0.012 + x) + up * y + n * 0.0032))) for x, y in hp]
    bm.faces.new(A[::-1]); bm.faces.new(B)
    for i in range(len(hp)):
        j = (i + 1) % len(hp); bm.faces.new((A[i], A[j], B[j], B[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    head = obj_from_bm('keyhead', bm, 'leather')
    bevel(head, 0.0012, 2, 40)
    parts.append(head)
    return parts

def key_ring_pivot():
    a, up, n = key_frame()
    return Vector(KEY_LOCK) + a * 0.040 + up * 0.0045

def build_key_ring():
    """(v4) Split ring through the key head hole with a spare key and a worn leather fob hanging from it.
    Built around its pivot (the hole); exported as node key_ring (vehicle.js swings it)."""
    a, up, n = key_frame()
    p0 = key_ring_pivot()
    parts = []
    # ring: plane spanned by n (through the hole) and gravity; hangs below the hole
    rc = p0 - Vector((0, 0, 0.0125))
    ring = []
    for k in range(17):
        t = 2 * math.pi * k / 16
        ring.append(V(*(rc + n * (0.0125 * math.cos(t)) + Vector((0, 0, 0.0125 * math.sin(t))))))
    parts.append(sweep(ring[:-1], [(0.0011 * math.cos(q), 0.0011 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 7)[:-1]], True, (a.x, -a.y, a.z), 'kring', 'chrome'))
    bot = rc - Vector((0, 0, 0.0125))
    # spare key: hangs from the ring bottom, turned a little
    kd = (Vector((0, 0, -1)) + n * 0.25).normalized(); kn = kd.cross(a).normalized(); kw = kn.cross(kd).normalized()
    parts.append(obox('skeyhead', tuple(bot + kd * 0.013), (kd, kw, kn), (0.024, 0.022, 0.0022), 'chrome'))
    parts.append(obox('skeyblade', tuple(bot + kd * 0.042), (kd, kw, kn), (0.036, 0.0075, 0.0018), 'chrome'))
    # leather fob tag on a short strap, hanging on the other side
    fd = (Vector((0, 0, -1)) - n * 0.2 + a * 0.15).normalized(); fn = fd.cross(n).normalized(); fw = fn.cross(fd).normalized()
    parts.append(obox('fobstrap', tuple(bot + fd * 0.012), (fd, fw, fn), (0.024, 0.008, 0.0015), 'leather'))
    fob = obox('fob', tuple(bot + fd * 0.046), (fd, fw, fn), (0.050, 0.027, 0.0035), 'leather')
    bevel(fob, 0.0012, 2, 40)
    parts.append(fob)
    j = join(parts, 'key_ring')
    shade(j, 35)
    return tag(j, 'int')

def build_charm():
    """(v4) Wooden prayer beads (mala) looped around the mirror stem, both strands hanging behind the mirror down to a
    larger guru bead and a faded maroon tassel. Exported as node mirror_charm (pivot = loop top, vehicle.js swings it).
    ~45 beads of 8 mm: the kind of charm that hangs in countless Himalayan and Alpine drivers' cabs."""
    px, pf, ph = CHARM_PIV
    parts = []
    def bead(c, r, name='bead', segs=6):
        prof = [(0.0003, -r * 0.86), (r * 0.62, -r * 0.72), (r * 0.97, 0.0), (r * 0.62, r * 0.72), (0.0003, r * 0.86)]
        return lathe(prof, segs, V(*c), (0, 0, 1), name, 'charm')
    # path: loop around the stem (above the housing), then two strands down behind the mirror to the guru bead
    pts = []
    for k in range(13):
        t = 2 * math.pi * k / 13
        pts.append(Vector((px + 0.017 * math.cos(t), pf + 0.013 * math.sin(t), ph + 0.0025 * math.sin(t))))
    guru = Vector((px + 0.004, pf + 0.040, ph - 0.116))
    NB = 15                                          # beads per strand: 8.6 mm beads touching on the string
    left = [Vector((px + 0.017, pf + 0.006, ph - 0.008)).lerp(guru + Vector((0.006, 0, 0.010)), (i / float(NB))) for i in range(1, NB)]
    right = [Vector((px - 0.017, pf + 0.006, ph - 0.008)).lerp(guru + Vector((-0.006, 0, 0.010)), (i / float(NB))) for i in range(1, NB)]
    # sag the strands outward a little (catenary-ish) and push them behind the mirror housing
    for arr, sg in ((left, 1), (right, -1)):
        for i, q in enumerate(arr):
            t = (i + 1) / float(NB)
            q.x += sg * 0.012 * math.sin(math.pi * t)
            q.y += 0.022 * math.sin(math.pi * t * 0.9)
    for q in pts + left + right:
        parts.append(bead(tuple(q), 0.0043))
    parts.append(bead(tuple(guru), 0.0062, 'guru', 10))
    # tassel: bound neck + splayed threads
    tc = guru - Vector((0, 0, 0.012))
    parts.append(lathe([(0.0028, 0.0), (0.0034, -0.006), (0.0032, -0.009), (0.0048, -0.012), (0.0072, -0.030), (0.0085, -0.046), (0.0003, -0.047)],
                       12, V(*tc), (0, 0, 1), 'tassel', 'charm'))
    j = join(parts, 'mirror_charm')
    shade(j, 50)
    return tag(j, 'int')

def print_quad(name, l_left, l_right, h0, h1, f, rect):
    """Vertical quad facing the driver (-f) at depth f, textured with rect of the gauge texture. l_left is the edge on
    the driver's left (+l) = texture left."""
    bm = bmesh.new(); uv = bm.loops.layers.uv.new('UVMap')
    vs = [bm.verts.new(V(l, f, h)) for l, h in ((l_left, h0), (l_right, h0), (l_right, h1), (l_left, h1))]
    fc = bm.faces.new(vs)
    u0, v0, u1, v1 = rect
    for lp, (a, b) in zip(fc.loops, ((u0, v0), (u1, v0), (u1, v1), (u0, v1))): lp[uv].uv = (a, b)
    bm.normal_update()
    if fc.normal.y < 0: fc.normal_flip()          # toward the driver = Blender +Y
    return obj_from_bm(name, bm, 'gauge')

def pod_f(h):
    return POD_B[0] + (POD_T[0] - POD_B[0]) * (h - POD_B[1]) / (POD_T[1] - POD_B[1])

def xform_about(ob, pivot_car, rot_x_deg):
    piv = V(*pivot_car)
    R = Matrix.Translation(piv) @ Matrix.Rotation(math.radians(rot_x_deg), 4, 'X') @ Matrix.Translation(-piv)
    ob.data.transform(R)
    return ob

def build_interior():
    parts = []
    I = 'interior'
    # floor, tunnel, firewall
    parts.append(box('floor', (0, -0.50, 0.505), (1.49, 2.44, 0.03), 'carpet'))
    tun = prism(fillet([(-0.15, 0.50), (0.15, 0.50), (0.12, 0.66), (-0.12, 0.66)], [0, 0, 0.05, 0.05], 4), 'lh', -0.35, 0.74, 'tunnel', 'carpet')
    bevel(tun, 0.01, 2, 40)
    parts.append(tun)
    parts.append(box('firewall', (0, 0.735, 0.78), (1.49, 0.02, 0.56), I))
    parts.append(box('toeboard', (0, 0.66, 0.56), (1.49, 0.16, 0.02), I))
    # dashboard main body (profile in f,h, extruded across)
    dash_prof = fillet([(0.745, 0.72), (0.745, 1.035), (0.55, 1.050), (0.44, 1.04), (0.405, 1.00), (0.43, 0.83), (0.52, 0.76)], [0, 0.02, 0.03, 0.035, 0.03, 0.03, 0.02], 4)
    dash = prism(dash_prof, 'fh', -0.745, 0.745, 'dash', I)
    bevel(dash, 0.006, 2, 40)
    parts.append(dash)
    # instrument pod on top of the dash (driver side) with visor; gauge faces on its rear face
    pod_prof = [(0.60, 1.035), (0.60, 1.155), (0.54, 1.198), (0.44, 1.204), (0.416, 1.197), (0.42, 1.189), POD_T, POD_B, (0.50, 1.035)]
    pod_prof = fillet(pod_prof, [0, 0.02, 0.03, 0.01, 0.004, 0.0, 0.0, 0.0, 0], 3)
    pod = prism(pod_prof, 'fh', SEAT_L - 0.195, SEAT_L + 0.195, 'pod', I)
    bevel(pod, 0.006, 2, 40)
    parts.append(pod)
    # center stack: radio, heater controls, vents
    parts.append(box('radio', (0.0, 0.415, 0.93), (0.19, 0.03, 0.055), 'int_black', 0.003))
    parts.append(cyl('radioknob', (-0.07, 0.398, 0.93), (0, 1, 0), 0.009, 0.012, 12, 'chrome'))
    parts.append(cyl('radioknob', (0.07, 0.398, 0.93), (0, 1, 0), 0.009, 0.012, 12, 'chrome'))
    parts.append(box('heater', (0.0, 0.418, 0.86), (0.20, 0.03, 0.05), 'int_black', 0.003))
    for i in range(3):
        parts.append(box('slider', (-0.06 + 0.06 * i, 0.401, 0.86), (0.012, 0.012, 0.018), 'chrome', 0.002))
    for s in (-0.07, 0.07):
        parts.append(box('cvent', (s, 0.412, 0.99), (0.11, 0.03, 0.045), 'int_black', 0.004))
        for k in range(4):
            parts.append(box('vslat', (s, 0.40, 0.975 + k * 0.01), (0.10, 0.012, 0.003), 'int_black'))
    for s in (1, -1):
        parts.append(cyl('svent', (s * 0.66, 0.43, 0.98), (0, 1, 0), 0.035, 0.03, 16, 'int_black'))
    # printed / backlit faces (material gauge, UVs into the free lower-right quadrant of the gauge texture)
    parts.append(print_quad('radioface', 0.093, -0.093, 0.9035, 0.9565, 0.3990, DASH_PRINTS['radio']))
    parts.append(print_quad('heaterface', 0.098, -0.098, 0.8355, 0.8845, 0.4020, DASH_PRINTS['heater']))
    # switch bank + warning-lamp strip under the instrument pod (driver side of the centre stack)
    parts.append(box('switchbank', (SEAT_L - 0.17, 0.418, 0.975), (0.13, 0.036, 0.04), 'int_black', 0.003))
    parts.append(print_quad('switchface', SEAT_L - 0.105, SEAT_L - 0.235, 0.9555, 0.9945, 0.3995, DASH_PRINTS['switches']))
    parts.append(print_quad('warnface', SEAT_L + 0.095, SEAT_L - 0.095, 1.0475, 1.0555, pod_f(1.0555) - 0.002, DASH_PRINTS['warn']))
    parts.append(box('glovebox', (-0.36, 0.425, 0.90), (0.34, 0.012, 0.13), I, 0.004))
    parts.append(box('glovelock', (-0.36, 0.418, 0.95), (0.03, 0.01, 0.015), 'chrome', 0.002))
    # steering column shroud + stalks
    cdir = Vector((0, math.cos(SW_ALPHA), -math.sin(SW_ALPHA)))
    c0 = Vector(SW_C)
    parts.append(tube([V(*(c0 + cdir * 0.05)), V(*(c0 + cdir * 0.26))], 0.042, 12, 'shroud', 'int_black'))
    parts.append(box('shroudbox', tuple(c0 + cdir * 0.11 + Vector((0, 0, -0.01))), (0.10, 0.10, 0.08), 'int_black', 0.015, 2))
    for s in (1, -1):
        parts.append(tube([V(*(c0 + cdir * 0.10 + Vector((s * 0.04, 0, 0.0)))), V(*(c0 + cdir * 0.07 + Vector((s * 0.15, 0, 0.01))))], 0.006, 6, 'stalk', 'int_black'))
    # pedals
    for l, h, w in ((SEAT_L + 0.14, 0.66, 0.05), (SEAT_L + 0.02, 0.66, 0.07), (SEAT_L - 0.16, 0.60, 0.07)):
        parts.append(box('pedalpad', (l, 0.58, h), (w, 0.018, 0.07), 'int_black', 0.004))
        parts.append(tube([V(l, 0.585, h + 0.03), V(l, 0.66, 0.84)], 0.007, 6, 'pedalarm', 'int_black'))
    # (v4) gear stick, transfer lever and handbrake with stitched leatherette gaiters on screwed trim rings
    parts.append(gaiter((0.02, 0.18, 0.664), (0.0245, 0.152, 0.805), 0.066, 0.0145, sq=0.55, folds=4.2, amp=0.0065, seed=1))
    parts += trim_frame((0.02, 0.18, 0.660), 0.150, 0.150, 0.016, 0.007)
    for dl, df in ((0.063, 0.063), (-0.063, 0.063), (0.063, -0.063), (-0.063, -0.063)):
        parts += screw((0.02 + dl, 0.18 + df, 0.667), (0, 0, 1), 0.0032, 0.4 + dl * 9)
    parts.append(tube([V(0.0245, 0.152, 0.79), V(0.03, 0.12, 0.93)], 0.008, 8, 'gearstick', 'chrome'))
    parts.append(lathe([(0.0005, 0.024), (0.014, 0.020), (0.02, 0.0), (0.014, -0.02), (0.006, -0.026)], 12, V(0.03, 0.118, 0.945), (0, 0, 1), 'knob', 'int_black'))
    parts.append(gaiter((-0.06, 0.03, 0.664), (-0.06, 0.014, 0.752), 0.042, 0.0105, sq=0.5, folds=3.2, amp=0.0045, seed=2))
    parts += trim_frame((-0.06, 0.03, 0.660), 0.095, 0.095, 0.012, 0.006, 'ttrim')
    parts.append(tube([V(-0.06, 0.016, 0.74), V(-0.06, 0.0, 0.83)], 0.007, 8, 'tlever', 'chrome'))
    parts.append(lathe([(0.0005, 0.018), (0.012, 0.012), (0.012, -0.012)], 10, V(-0.06, -0.002, 0.845), (0, 0, 1), 'tknob', 'int_black'))
    hb = box('handbrake', (0.0, -0.17, 0.70), (0.032, 0.20, 0.03), 'int_black', 0.01)
    hg = box('hbgrip', (0.0, -0.315, 0.70), (0.037, 0.12, 0.036), 'leather', 0.014, 3)
    hbb = cyl('hbbutton', (0.0, -0.378, 0.70), (0, 1, 0), 0.008, 0.012, 12, 'chrome')
    for o in (hb, hg, hbb): xform_about(o, (0, -0.07, 0.68), 10)
    parts += [hb, hg, hbb]
    parts.append(gaiter((0.0, -0.075, 0.664), (0.0, -0.085, 0.705), 0.045, 0.024, sq=0.9, folds=2.2, amp=0.004, seed=3))
    # seats
    for s in (1, -1):
        parts += build_seat(s * SEAT_L)
    # rear bench between the wheel houses
    parts.append(box('rbench', (0, -0.95, 0.70), (0.90, 0.46, 0.13), 'fabric', 0.04, 3))
    back = box('rback', (0, -1.20, 0.98), (0.90, 0.10, 0.50), 'fabric', 0.035, 3)
    xform_about(back, (0, -1.20, 0.74), -12)
    parts.append(back)
    # door cards and rear side trims
    for s in (1, -1):
        L = s * 0.737
        parts.append(box('doorcard', (L, 0.10, 0.80), (0.018, 1.02, 0.50), I, 0.006))
        parts.append(box('armrest', (L - s * 0.04, -0.05, 0.84), (0.06, 0.32, 0.04), I, 0.012, 3))
        parts.append(box('dpull', (L - s * 0.015, 0.25, 0.94), (0.02, 0.08, 0.025), 'chrome', 0.004))
        parts.append(cyl('crankbase', (L - s * 0.02, 0.12, 0.72), (1, 0, 0), 0.018, 0.02, 12, 'chrome'))
        parts.append(box('crank', (L - s * 0.03, 0.12, 0.69), (0.012, 0.012, 0.07), 'chrome', 0.003))
        parts.append(cyl('speaker', (L - s * 0.012, 0.42, 0.66), (1, 0, 0), 0.06, 0.01, 20, 'int_black'))
        parts.append(cyl('lockknob', (L - s * 0.025, -0.35, 1.07), (0, 0, 1), 0.006, 0.05, 8, 'chrome'))
        parts.append(box('qtrim', (s * 0.735, -1.15, 0.80), (0.02, 1.10, 0.50), I, 0.006))
        parts.append(box('btrim', (s * 0.70, -0.49, 1.33), (0.05, 0.10, 0.55), I, 0.012, 2))
        belt = [V(s * 0.67, -0.47, 1.45), V(s * 0.52, -0.40, 1.30), V(s * 0.40, -0.36, 1.05), V(s * 0.25, -0.33, 0.78)]
        parts.append(sweep(belt, rrect(-0.024, -0.0015, 0.024, 0.0015, 0.001, 1), False, (1, 0, 0), 'belt', 'int_black'))
    # headliner, dome light
    parts.append(prism(rrect(-0.655, -1.72, 0.655, 0.25, 0.05, 4), 'lf', 1.590, 1.603, 'headliner', 'fabric'))
    parts.append(box('domelight', (0, -0.60, 1.586), (0.14, 0.07, 0.01), 'int_black', 0.004))
    for s in (1, -1):
        # (v4) padded visors on a hinge rod (bracket screwed to the header at the pillar end, clip at the inner end)
        parts.append(box('visor', (s * 0.35, 0.225, 1.573), (0.40, 0.14, 0.02), 'fabric', 0.008, 2))
        parts.append(tube([V(s * 0.155, 0.292, 1.580), V(s * 0.655, 0.292, 1.580)], 0.0042, 8, 'visorrod', 'chrome'))
        parts.append(box('visorbr', (s * 0.665, 0.285, 1.587), (0.028, 0.032, 0.014), 'int_black', 0.003, 2))
        parts.append(box('visorclip', (s * 0.135, 0.287, 1.586), (0.022, 0.024, 0.012), 'int_black', 0.003, 2))
        parts += screw((s * 0.665, 0.278, 1.5795), (0, 0, -1), 0.003, 0.7)
    # driver visor: elastic strap with a fuel receipt and a folded document tucked under it
    parts.append(box('vstrap', (0.27, 0.225, 1.5618), (0.024, 0.146, 0.0016), 'int_black', 0.0005, 1))
    pp = box('vpaper1', (0.33, 0.222, 1.5624), (0.135, 0.095, 0.0008), 'paper'); xform_about(pp, (0.33, 0.222, 1.5624), 0)
    pp.data.transform(Matrix.Translation(V(0.33, 0.222, 1.5624)) @ Matrix.Rotation(math.radians(5), 4, 'Z') @ Matrix.Translation(-V(0.33, 0.222, 1.5624)))
    parts.append(pp)
    pp = box('vpaper2', (0.25, 0.215, 1.5616), (0.085, 0.060, 0.0008), 'paper')
    pp.data.transform(Matrix.Translation(V(0.25, 0.215, 1.5616)) @ Matrix.Rotation(math.radians(-8), 4, 'Z') @ Matrix.Translation(-V(0.25, 0.215, 1.5616)))
    parts.append(pp)
    # rear view mirror (from the header), angled toward the driver
    parts.append(tube([V(0, 0.262, 1.598), V(0, 0.250, 1.545)], 0.006, 6, 'rvm_stem', 'int_black'))
    parts += rvm_parts()
    # (v4) A-pillar inner trims (D-section mouldings) with one retaining screw each
    for s in (1, -1):
        tr, sp, sn = apillar_trim(s)
        parts.append(tr)
        parts += screw(sp, sn, 0.0034, 0.3)
    # (v4) dash top: defroster grilles along the windshield base (their slats read in the glass reflection), side
    # demisters, a folded road map on the passenger side and a curled fuel receipt at the driver's windshield corner
    for s in (1, -1):
        parts.append(box('defrost', (s * 0.30, 0.683, 1.0402), (0.30, 0.036, 0.004), 'int_black', 0.0015, 1))
        for k in range(6):
            parts.append(box('dslat', (s * 0.30, 0.6695 + k * 0.0054, 1.0419), (0.292, 0.0021, 0.0026), 'int_black'))
        parts.append(box('sdemist', (s * 0.60, 0.70, 1.0385), (0.07, 0.022, 0.004), 'int_black', 0.0015, 1))
    mp = map_mesh((-0.42, 0.535, 1.0495), 0.27, 0.15, 11)
    parts.append(mp)
    rc = map_mesh((0.16, 0.668, 1.0418), 0.055, 0.105, -22, folds=0, curl=0.006, name='receipt')
    parts.append(rc)
    # (v4) ignition lock + key (the dangling ring / spare key / fob is the separate node key_ring)
    parts += key_static()
    # (v4) visible trim screws: dash lower face, radio/heater surrounds, door cards, kick panels
    DF = lambda h: 0.405 + 0.025 * (1.0 - h) / 0.17          # dash face (driver side of the prism) at height h
    for c, n in (((0.55, DF(0.862), 0.862), (0, -1, -0.15)), ((0.17, DF(0.862), 0.862), (0, -1, -0.15)), ((-0.70, DF(0.95), 0.95), (0, -1, -0.15)),
                 ((0.112, DF(0.905), 0.905), (0, -1, -0.15)), ((-0.112, DF(0.905), 0.905), (0, -1, -0.15)),
                 ((0.118, DF(0.86), 0.86), (0, -1, -0.15)), ((-0.118, DF(0.86), 0.86), (0, -1, -0.15))):
        parts += screw(c, n, 0.0033, c[0] * 17)
    for s in (1, -1):
        for f, h in ((0.52, 0.60), (-0.30, 0.60), (0.46, 0.99)):
            parts += screw((s * 0.7272, f, h), (-s, 0, 0), 0.0036, f * 11 + s)
    j = join(parts, 'interior')
    shade(j, 40)
    return tag(j, 'int')

def build_seat(l):
    parts = []
    I = 'fabric'
    F = SEAT_F
    parts.append(box('seatbase', (l, F + 0.25, 0.56), (0.40, 0.44, 0.10), 'int_black', 0.01))
    parts.append(box('cushion', (l, F + 0.31, 0.69), (0.46, 0.50, 0.13), I, 0.045, 4))
    for s in (1, -1):
        parts.append(box('bolster', (l + s * 0.225, F + 0.31, 0.725), (0.07, 0.48, 0.10), I, 0.03, 3))
    piv = (l, F, 0.74)
    back = box('backrest', (l, F - 0.06, 1.02), (0.46, 0.12, 0.58), I, 0.045, 4)
    xform_about(back, piv, -14); parts.append(back)
    for s in (1, -1):
        bb = box('bbolster', (l + s * 0.225, F - 0.04, 1.00), (0.07, 0.15, 0.50), I, 0.03, 3)
        xform_about(bb, piv, -14); parts.append(bb)
    hr = box('headrest', (l, F - 0.07, 1.47), (0.26, 0.09, 0.18), I, 0.035, 3)
    xform_about(hr, piv, -14); parts.append(hr)
    for s in (1, -1):
        p = tube([V(l + s * 0.08, F - 0.07, 1.29), V(l + s * 0.08, F - 0.07, 1.40)], 0.006, 6, 'hrpost', 'chrome')
        xform_about(p, piv, -14); parts.append(p)
    return parts

def build_steering_wheel():
    """(v4) Early-90s 3-spoke wheel: oval-section rim moulded in grained polyurethane (worn smooth where the hands sit,
    see compose_int) with finger-grip ridges on the inner back, dished spokes, a big padded horn pad and a small badge.
    Built in its local frame: wheel plane = local XZ, local +Y faces the driver (three local -Z). The object is placed
    at the hub and rotated about X by SW_ALPHA, so rotating about three's local Z turns the wheel in its plane."""
    parts = []
    R = 0.190
    NA, NS = 144, 12
    bm = bmesh.new(); rings = []
    for k in range(NA):
        th = 2 * math.pi * k / NA
        ct, st = math.cos(th), math.sin(th)
        lower = 1.0 - math.exp(-((math.degrees(math.atan2(st, ct)) - 90.0) / 38.0) ** 2)
        ring = []
        for i in range(NS):
            ph = 2 * math.pi * i / NS
            rr = 0.0152 * math.cos(ph); yy = 0.0126 * math.sin(ph)
            dph = (math.degrees(ph) - 215.0 + 180.0) % 360.0 - 180.0
            w = math.exp(-(dph / 40.0) ** 2)
            ridge = 0.0021 * (0.5 + 0.5 * math.cos(th * 46)) ** 2 * w * lower
            rr += ridge * math.cos(ph); yy += ridge * math.sin(ph)
            ring.append(bm.verts.new(Vector(((R + rr) * ct, yy, (R + rr) * st))))
        rings.append(ring)
    for k in range(NA):
        A, B = rings[k], rings[(k + 1) % NA]
        for i in range(NS):
            j = (i + 1) % NS
            bm.faces.new((A[i], B[i], B[j], A[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    rim = obj_from_bm('swrim', bm, 'leather')
    me = rim.data; pc = me.polygons[0]
    if pc.center.normalized().dot(pc.normal) < -0.5:
        bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.reverse_faces(bm, faces=bm.faces); bm.to_mesh(me); bm.free()
    parts.append(rim)
    # padded horn pad (dished: sits forward of the rim plane)
    pad = box('swpad', (0, 0, 0), (0.158, 0.046, 0.122), 'int_black', 0.019, 4)
    pad.data.transform(Matrix.Translation(Vector((0, -0.010, -0.014))))
    parts.append(pad)
    hub = box('swhub', (0, 0, 0), (0.07, 0.05, 0.07), 'int_black', 0.01, 2)
    hub.data.transform(Matrix.Translation(Vector((0, -0.040, -0.012))))
    parts.append(hub)
    for ang, w0, w1 in ((math.radians(-14), 0.046, 0.030), (math.radians(194), 0.046, 0.030), (math.radians(270), 0.050, 0.034)):
        d = Vector((math.cos(ang), 0, math.sin(ang)))
        p0 = d * 0.066 + Vector((0, -0.020, -0.012 * (1 if ang > 4 else 0)))
        p1 = d * (R - 0.010) + Vector((0, -0.004, 0))
        prof = rrect(-0.5, -0.0065, 0.5, 0.0065, 0.005, 2)
        parts.append(sweep([p0, p1], prof, False, (0, 1, 0), 'spoke', 'int_black', scale=None))
        sp = parts[-1]
        # taper the width (profile x) from w0 at the hub to w1 at the rim
        for v in sp.data.vertices:
            t = max(0.0, min(1.0, (v.co - p0).dot(p1 - p0) / (p1 - p0).length_squared))
            base = p0.lerp(p1, t)
            side = Vector((v.co.x - base.x, 0, v.co.z - base.z))
            v.co = base + side * (w0 + (w1 - w0) * t) + Vector((0, v.co.y - base.y, 0))
    # moulded-in emblem: a low raised disc with a rim, same black material (a chrome ring read as a toy)
    parts.append(lathe([(0.0003, 0.0016), (0.0082, 0.0016), (0.0094, 0.0008), (0.0100, -0.001)], 20, Vector((0, 0.0135, 0.004)), (0, 1, 0), 'badge', 'int_black'))
    j = join(parts, 'steering_wheel')
    shade(j, 45)
    return j

GAUGES = [  # (name, l offset from SEAT_L, h, radius, uv rect in gauge texture (u0,v0,u1,v1))
    ('speedo', 0.062, 1.103, 0.046, (0.0, 0.5, 0.5, 1.0)),
    ('tacho', -0.062, 1.103, 0.046, (0.5, 0.5, 1.0, 1.0)),
    ('fuel', 0.146, 1.082, 0.024, (0.0, 0.0, 0.25, 0.25)),
    ('temp', -0.146, 1.082, 0.024, (0.25, 0.0, 0.5, 0.25)),
]

NEEDLE_SIGN = 1   # +rotation.z (three.js, about the needle's local Z = into the dial) turns the needle clockwise for the driver

def build_freshener():
    """Cardboard air freshener (pine silhouette) hanging on a string below the rear-view mirror.
    Built around its pivot (string top) at FRESH_PIV; exported as node `air_freshener`."""
    parts = []
    px, pf, ph = FRESH_PIV
    # string: thin twisted strip down to the card
    parts.append(tube([V(px, pf, ph), V(px + 0.001, pf + 0.002, ph - 0.035), V(px, pf, ph - 0.062)], 0.0006, 4, 'fstring', 'int_black'))
    # card: stylised 3-tier conifer, 0.066 x 0.105 m, in the (l, h) plane facing the driver (slightly turned)
    w = 0.033
    top = ph - 0.060
    # (t = fraction from the apex down, half width); tier bottoms step inward and slightly up
    tiers = [(0.0, 0.0), (0.33, 0.017), (0.30, 0.008), (0.62, 0.026), (0.595, 0.013), (0.88, w), (0.88, 0.006), (1.0, 0.006)]
    H = 0.100
    right = [(dx, top - H * t) for t, dx in tiers]
    outline = right + [(-dx, h) for dx, h in reversed(right[1:])]   # apex only once
    th = 0.0008
    bm = bmesh.new()
    ang = math.radians(25)   # turned a little toward the driver
    def P3(dx, h, s):
        return V(px + dx * math.cos(ang), pf + s * th - dx * math.sin(ang), h)
    A = [bm.verts.new(P3(dx, h, -1)) for dx, h in outline]
    B = [bm.verts.new(P3(dx, h, 1)) for dx, h in outline]
    try:
        bm.faces.new(A[::-1]); bm.faces.new(B)
    except ValueError:
        pass
    for i in range(len(outline)):
        j = (i + 1) % len(outline)
        bm.faces.new((A[i], A[j], B[j], B[i]))
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    parts.append(obj_from_bm('fcard', bm, 'freshener'))
    j = join(parts, 'air_freshener')
    shade(j, 30)
    return tag(j, 'int')

def gauge_frame():
    n = Vector((0, -math.cos(GAUGE_TILT), math.sin(GAUGE_TILT)))  # car coords, toward driver/up
    ex = Vector((1, 0, 0)); ey = n.cross(ex).normalized()
    if ey.z < 0: ey = -ey
    return n, ex, ey

def build_gauges():
    """Gauge faces (material gauge) with UVs into the gauge texture, plus chrome bezels."""
    parts = []
    n, ex, ey = gauge_frame()
    nb = Vector((n.x, -n.y, n.z))
    for name, dl, h, r, rect in GAUGES:
        c = Vector((SEAT_L + dl, pod_f(h), h)) + n * 0.002
        bm = bmesh.new(); uv = bm.loops.layers.uv.new('UVMap')
        N = 40
        center = bm.verts.new(V(*c))
        ring = [bm.verts.new(V(*(c + (ex * math.cos(2 * math.pi * i / N) + ey * math.sin(2 * math.pi * i / N)) * r))) for i in range(N)]
        cu, cv = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        hu, hv = (rect[2] - rect[0]) / 2 * 0.98, (rect[3] - rect[1]) / 2 * 0.98
        for i in range(N):
            j = (i + 1) % N
            f = bm.faces.new((center, ring[j], ring[i]))
            for lp in f.loops:
                if lp.vert is center: lp[uv].uv = (cu, cv)
                else:
                    k = ring.index(lp.vert); a = 2 * math.pi * k / N
                    # +l (ex) is the driver's LEFT -> texture left
                    lp[uv].uv = (cu - hu * math.cos(a), cv + hv * math.sin(a))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        for fce in bm.faces:
            if fce.normal.dot(nb) < 0: fce.normal_flip()
        parts.append(obj_from_bm('gface_' + name, bm, 'gauge'))
        # (v4) black moulded bezel (chrome rings read as white discs at cockpit distance; 90s clusters had none)
        parts.append(lathe([(r + 0.0005, 0.0015), (r + 0.004, 0.003), (r + 0.006, 0.0), (r + 0.006, -0.006)], 32, V(*c), tuple(nb), 'gbezel', 'int_black'))
    return parts

def build_needles():
    """Needle meshes. Origin at the pivot; at rest (rotation.z = 0) the needle points straight UP on the dial.
    In three.js the driven node `needle_<k>` has an identity rest transform (the dial tilt lives on its parent
    `needle_<k>_tilt`), so vehicle.js can write rotation.z absolutely; +rotation.z = clockwise as seen by the driver."""
    out = []
    n, ex, ey = gauge_frame()
    nb = Vector((n.x, -n.y, n.z)); eyb = Vector((ey.x, -ey.y, ey.z))
    for name, dl, h, r, rect in GAUGES:
        c = Vector((SEAT_L + dl, pod_f(h), h)) + n * 0.005
        bm = bmesh.new()
        L = r * 0.84
        # (v4) wider needle (4.5 mm root) and a black hub cap (UV'd onto the black print background)
        vs = [bm.verts.new(Vector((x, 0, y))) for x, y in [(-0.0026, -0.012), (0.0026, -0.012), (0.0013, L), (-0.0013, L)]]
        bm.faces.new(vs)
        cap = [bm.verts.new(Vector((0.0065 * math.cos(a), 0.0012, 0.0065 * math.sin(a)))) for a in np.linspace(0, 2 * math.pi, 13)[:-1]]
        bm.faces.new(cap)
        for fce in bm.faces:
            if fce.normal.y < 0: fce.normal_flip()
        me = bpy.data.meshes.new('needle_' + name); bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new('needle_' + name, me); coll('CAR').objects.link(ob)
        ob.data.materials.append(MATS['gauge'])
        # local +Y = toward driver (face normal), local +Z = dial up, local X = Y x Z
        Y = nb.normalized(); Z = eyb.normalized(); X = Y.cross(Z).normalized()
        M = Matrix((X, Y, Z)).transposed().to_4x4(); M.translation = V(*c)
        ob.matrix_world = M
        ob['needle'] = name
        out.append(ob)
    return out

# ------------------------------------------------------------------------------------------------
# Preview rendering
# ------------------------------------------------------------------------------------------------
def setup_render(samples=64, res=(1280, 720)):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    try:
        prefs.compute_device_type = 'METAL'; prefs.get_devices()
        for d in prefs.devices: d.use = True
        sc.cycles.device = 'GPU'
    except Exception as e:
        log('GPU unavailable', e)
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    try: sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception: pass
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = 'AgX'
    try: sc.view_settings.look = 'AgX - Base Contrast'
    except Exception: pass
    sc.render.film_transparent = False
    sc.cycles.max_bounces = 8; sc.cycles.transmission_bounces = 8; sc.cycles.glossy_bounces = 4
    # world
    w = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    sc.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    env = nt.nodes.new('ShaderNodeTexEnvironment'); env.image = bpy.data.images.load(HDRI, check_existing=True)
    mp = nt.nodes.new('ShaderNodeMapping'); tc = nt.nodes.new('ShaderNodeTexCoord')
    mp.inputs['Rotation'].default_value[2] = math.radians(150)
    bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value = 1.0
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], env.inputs['Vector'])
    nt.links.new(env.outputs['Color'], bg.inputs['Color']); nt.links.new(bg.outputs['Background'], out.inputs['Surface'])

def preview_ground():
    pc = coll('PREVIEW')
    if bpy.data.objects.get('pv_ground'): return
    bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=40)
    ob = obj_from_bm('pv_ground', bm, None, pc)
    m = bpy.data.materials.new('pv_wet'); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (0.035, 0.034, 0.033, 1)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 0.6; nz.inputs['Detail'].default_value = 8
    nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeMapRange'); ramp.inputs['From Min'].default_value = 0.42; ramp.inputs['From Max'].default_value = 0.58
    ramp.inputs['To Min'].default_value = 0.05; ramp.inputs['To Max'].default_value = 0.45
    nt.links.new(nz.outputs['Fac'], ramp.inputs['Value']); nt.links.new(ramp.outputs['Result'], b.inputs['Roughness'])
    nz2 = nt.nodes.new('ShaderNodeTexNoise'); nz2.inputs['Scale'].default_value = 90; nz2.inputs['Detail'].default_value = 4
    nt.links.new(tc.outputs['Object'], nz2.inputs['Vector'])
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.15
    nt.links.new(nz2.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
    ob.data.materials.append(m)

VIEWDEF = {  # car coords: camera position, target, lens(mm) (35mm sensor), or 'fov' for cockpit
    'front34': ((3.9, 4.6, 1.15), (0.0, 0.25, 0.80), 50),
    'rear34': ((-3.7, -4.9, 1.45), (0.0, -0.3, 0.85), 50),
    'side': ((6.8, 0.0, 0.95), (0.0, 0.0, 0.85), 50),
    'front': ((0.0, 7.0, 1.0), (0.0, 0.0, 0.85), 55),
    'rear': ((0.0, -7.0, 1.2), (0.0, 0.0, 0.85), 55),
    'wheel': ((1.75, 2.10, 0.55), (0.72, 1.15, 0.38), 50),
    'low': ((2.6, 3.6, 0.28), (0.0, 0.3, 0.55), 30),
    'top': ((0.01, 0.0, 9.0), (0.0, 0.0, 0.0), 50),
    'cockpit': (EYE, (EYE[0] + 0.02, EYE[1] + 1.0, EYE[2] - 0.12), 'fov70'),
    'chase': ((0.0, -6.5, 2.4), (0.0, 1.5, 1.0), 'fov70'),
    'interior': ((0.95, -0.10, 1.25), (-0.1, 0.35, 0.95), 24),
    'lamp': ((1.05, 2.95, 0.95), (0.50, 1.78, 0.78), 85),
    'dash': ((0.12, -0.25, 1.22), (0.08, 0.40, 0.95), 26),
    'key': ((0.14, 0.12, 1.12), (0.29, 0.40, 1.00), 32),
    'mirror': ((0.26, -0.08, 1.40), (0.0, 0.26, 1.50), 30),
    'gaiter': ((0.34, -0.18, 1.08), (0.02, 0.17, 0.72), 30),
    'wheelint': ((0.36, -0.12, 1.20), (0.36, 0.30, 1.05), 30),
}

def render_views(prefix, views, samples=64, res=(1280, 720)):
    sc = bpy.context.scene
    setup_render(samples, res)
    preview_ground()
    cam = bpy.data.objects.get('pv_cam')
    if not cam:
        cam = bpy.data.objects.new('pv_cam', bpy.data.cameras.new('pv_cam')); coll('PREVIEW').objects.link(cam)
    sc.camera = cam
    outs = []
    for v in views:
        if v not in VIEWDEF: continue
        p, t, lens = VIEWDEF[v]
        cam.location = V(*p)
        d = V(*t) - V(*p)
        cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        if lens == 'fov70':
            cam.data.sensor_fit = 'VERTICAL'; cam.data.angle = math.radians(70); cam.data.clip_start = 0.05
        else:
            cam.data.sensor_fit = 'AUTO'; cam.data.lens = lens; cam.data.clip_start = 0.05
        sc.render.filepath = os.path.join(SCR, f'{prefix}_{v}.png')
        bpy.ops.render.render(write_still=True)
        outs.append(sc.render.filepath)
        log('rendered', sc.render.filepath)
    return outs

# ------------------------------------------------------------------------------------------------
# TEXTURING: materials groups, UV atlases, G-buffer bake, numpy composition
# ------------------------------------------------------------------------------------------------
EXT_MATS = ['paint', 'paint_inner', 'plastic', 'rubber_trim', 'steel_black', 'chrome', 'underbody', 'liner',
            'wheel', 'brake', 'headlight', 'brakelight', 'indicator', 'plate', 'cover', 'glass']
INT_MATS = ['interior', 'fabric', 'carpet', 'int_black', 'charm', 'leather', 'paper']
MID = {n: i + 1 for i, n in enumerate(EXT_MATS + INT_MATS)}          # material id stored in the G-buffer
EXPORT_MAP = {'paint': 'paint', 'paint_inner': 'paint', 'plastic': 'trim_black', 'rubber_trim': 'trim_black',
              'steel_black': 'trim_black', 'underbody': 'trim_black', 'liner': 'trim_black', 'cover': 'trim_black',
              'wheel': 'trim_black', 'brake': 'trim_black', 'chrome': 'chrome', 'headlight': 'headlight',
              'brakelight': 'brakelight', 'indicator': 'indicator', 'plate': 'plate', 'glass': 'glass',
              'rubber': 'rubber', 'interior': 'interior', 'fabric': 'interior', 'carpet': 'interior',
              'int_black': 'interior', 'charm': 'interior', 'leather': 'interior', 'paper': 'interior', 'gauge': 'gauge'}
UV_PRIO = {'paint_inner': 0.28, 'underbody': 0.45, 'liner': 0.5, 'brake': 0.6, 'plate': 3.2, 'headlight': 1.4,
           'brakelight': 1.5, 'indicator': 1.4, 'cover': 2.8, 'chrome': 1.3, 'wheel': 1.3,
           'carpet': 0.6, 'fabric': 1.0, 'interior': 1.1, 'int_black': 1.4, 'glass': 0.8, 'charm': 2.5,
           'leather': 3.0, 'paper': 1.6}

def set_src_uv(ob, fn, mats=None):
    """Secondary UV 'src' used only while baking (plate text, logo, lens patterns). fn(car_co, car_normal)->(u,v)."""
    me = ob.data
    if 'src' not in me.uv_layers: me.uv_layers.new(name='src')
    if 'UVMap' not in me.uv_layers:
        me.uv_layers.new(name='UVMap')
    lay = me.uv_layers['src']
    for poly in me.polygons:
        if mats is not None and me.materials[poly.material_index].name not in mats: continue
        n = poly.normal; nc = Vector((n.x, -n.y, n.z))
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            lay.data[li].uv = fn(Vector((co.x, -co.y, co.z)), nc)

def mat_names(ob): return [m.name for m in ob.data.materials]

BEAD_TILE = 0.20      # m per repeat of the rain-bead normal map (UV 'detail')
BEAD_FLAT = (0.5, 0.5)  # bead-free spot in the bead texture (windshield, interior, non-wet parts map here)

def detail_uv(ob):
    """Second UV set 'detail': per-face triplanar projection in metres / BEAD_TILE (paint + side/rear glass);
    other faces collapse onto the bead-free spot so the tiling bead map leaves them untouched."""
    me = ob.data
    lay = me.uv_layers.get('detail') or me.uv_layers.new(name='detail')
    wet = {'paint', 'glass', 'plastic', 'steel_black', 'brakelight', 'indicator', 'cover'}
    mats = [m.name if m else '' for m in me.materials]
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    for poly in me.polygons:
        mn = mats[poly.material_index] if poly.material_index < len(mats) else ''
        n = poly.normal
        ws = mn == 'glass' and (-n.y) > 0.4 and n.z > 0.25            # windshield (Blender -Y = forward)
        if mn not in wet or ws:
            for li in poly.loop_indices: lay.data[li].uv = BEAD_FLAT
            continue
        ax = max(range(3), key=lambda i: abs(n[i]))
        a, b = [(1, 2), (0, 2), (0, 1)][ax]
        for li in poly.loop_indices:
            c = co[me.loops[li].vertex_index]
            lay.data[li].uv = (c[a] / BEAD_TILE, c[b] / BEAD_TILE)

def bead_texture(size=1024, seed=11):
    """Seamless tiling normal map of rain beads (sessile drops), BEAD_TILE metres per repeat."""
    rng = np.random.RandomState(seed)
    H = np.zeros((size, size), np.float32)
    px_m = BEAD_TILE / size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    nb = 2600
    rs = np.exp(rng.uniform(np.log(0.35e-3), np.log(2.6e-3), nb)) / px_m     # radius in px (0.35..2.6 mm)
    cx = rng.uniform(0, size, nb); cy = rng.uniform(0, size, nb)
    fx, fy = BEAD_FLAT[0] * size, BEAD_FLAT[1] * size
    for r, x0, y0 in zip(rs, cx, cy):
        dxf = (x0 - fx + size / 2) % size - size / 2; dyf = (y0 - fy + size / 2) % size - size / 2
        if math.hypot(dxf, dyf) < r + 40: continue
        ex = 1.0 + 0.25 * rng.rand()                                       # slightly elongated (run-off)
        R = int(math.ceil(r * ex)) + 2
        xs = np.arange(int(x0) - R, int(x0) + R + 1); ys = np.arange(int(y0) - R, int(y0) + R + 1)
        X, Y = np.meshgrid(xs, ys)
        d2 = ((X - x0) / 1.0) ** 2 + ((Y - y0) / ex) ** 2
        q = np.clip(1 - d2 / (r * r), 0, 1)
        h = np.sqrt(q) * r * 0.55                                          # spherical cap, ~55 deg contact angle
        H[Y % size, X % size] = np.maximum(H[Y % size, X % size], h)
    gx = (np.roll(H, -1, 1) - np.roll(H, 1, 1)) * 0.5
    gy = (np.roll(H, -1, 0) - np.roll(H, 1, 0)) * 0.5
    n = np.stack([-gx, -gy, np.ones_like(H)], 2); n /= np.linalg.norm(n, axis=2, keepdims=True)
    save_png(n * 0.5 + 0.5, os.path.join(TEXDIR, 'beads_normal.png'))
    # (v4) paint clearcoat: the same beads on top of orange peel (2-8 mm waviness of sprayed enamel). Real peel slopes
    # are ~0.2 deg, below one 8-bit normal-map step, so it is exaggerated to ~0.8 deg to survive quantisation.
    # band-limited periodic noise (FFT filtered, so the 0.2 m tile has no seams): wavelengths ~2-10 mm
    wn = np.random.RandomState(seed + 70).randn(size, size).astype(np.float32)
    kx = np.fft.fftfreq(size) * size; KX, KY = np.meshgrid(kx, kx); K = np.sqrt(KX ** 2 + KY ** 2)   # cycles per tile
    filt = np.exp(-((K - 45.0) / 22.0) ** 2) + 0.5 * np.exp(-((K - 18.0) / 8.0) ** 2)
    filt[0, 0] = 0
    peel_h = np.real(np.fft.ifft2(np.fft.fft2(wn) * filt)).astype(np.float32)
    gxp = (np.roll(peel_h, -1, 1) - np.roll(peel_h, 1, 1)) * 0.5; gyp = (np.roll(peel_h, -1, 0) - np.roll(peel_h, 1, 0)) * 0.5
    peel_h *= 0.014 / max(float(np.sqrt((gxp ** 2 + gyp ** 2).mean())), 1e-9)   # RMS slope 0.014 rad (px height units)
    H2 = H + peel_h
    gx = (np.roll(H2, -1, 1) - np.roll(H2, 1, 1)) * 0.5
    gy = (np.roll(H2, -1, 0) - np.roll(H2, 1, 0)) * 0.5
    n2_ = np.stack([-gx, -gy, np.ones_like(H2)], 2); n2_ /= np.linalg.norm(n2_, axis=2, keepdims=True)
    save_png(n2_ * 0.5 + 0.5, os.path.join(TEXDIR, 'peel_normal.png'))

def assemble(P):
    """Join all car_body parts into one object, build the wheel (tire+rim) object."""
    # src UVs for plates / cover / lamps (before joining)
    for pl in [o for o in P['bumpers'].data.materials]: pass
    def plate_uv(c, n):
        front = c.y > 0
        hc = 0.462 if front else 0.452
        u = (c.x + 0.26) / 0.52 if front else (0.26 - c.x) / 0.52
        return (u, (c.z - (hc - 0.055)) / 0.11)
    set_src_uv(P['bumpers'], plate_uv, {'plate'})
    set_src_uv(P['rear'], lambda c, n: (0.5 - c.x / 0.74, (c.z - 0.915) / 0.74 + 0.5), {'cover'})
    set_src_uv(P['front'], lambda c, n: ((c.x + 0.8) / 1.6, c.z), {'headlight', 'indicator'})
    set_src_uv(P['rear'], lambda c, n: (c.x, c.z), {'brakelight', 'indicator'})
    parts = [P[k] for k in ('body', 'flares', 'rockers', 'bumpers', 'front', 'rear', 'ghd', 'side', 'asym', 'under', 'interior', 'gauges', 'glass')]
    for o in parts:
        if 'UVMap' not in o.data.uv_layers: o.data.uv_layers.new(name='UVMap')
        if 'src' not in o.data.uv_layers: o.data.uv_layers.new(name='src')
    body = join(parts, 'car_body')
    detail_uv(body)
    for o in (P['sw'], P['tire'], P['rim'], P['fresh'], P['keyring']):
        if 'UVMap' not in o.data.uv_layers: o.data.uv_layers.new(name='UVMap')
        if 'src' not in o.data.uv_layers: o.data.uv_layers.new(name='src')
    wheel = join([P['tire'], P['rim']], 'wheel_L')
    wheel.location = V(HT, FAX, R_TIRE)
    for o in (wheel, P['sw'], P['fresh'], P['keyring']): detail_uv(o)
    for o in (body, wheel, P['sw'], P['fresh'], P['keyring']):
        o.data.uv_layers.active = o.data.uv_layers['UVMap']
        o.data.uv_layers['UVMap'].active_render = True
    return body, wheel

def _face_sets(objs, mats):
    out = []
    for o in objs:
        idx = [i for i, m in enumerate(o.data.materials) if m and m.name in mats]
        out.append(set(idx))
    return out

def unwrap_atlas(objs, mats, margin):
    """Smart-project faces of `mats` group by group, normalize texel density * priority, pack into 0-1."""
    for o in bpy.context.view_layer.objects: o.select_set(False)
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    for m in mats:
        bpy.ops.object.mode_set(mode='OBJECT')
        present = False
        for o in objs:
            mi = [i for i, mm in enumerate(o.data.materials) if mm and mm.name == m]
            for p in o.data.polygons:
                p.select = p.material_index in mi
                present |= p.select
        if not present: continue
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.hide(unselected=True)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(52), island_margin=0.0, area_weight=0.0, scale_to_bounds=False)
        bpy.ops.mesh.reveal(select=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        for o in objs:
            mi = [i for i, mm in enumerate(o.data.materials) if mm and mm.name == m]
            for p in o.data.polygons: p.select = p.material_index in mi
        # density normalize: sqrt(uv area / 3d area) -> target 1 * prio
        a3 = 0.0; auv = 0.0
        for o in objs:
            me = o.data; uvl = me.uv_layers['UVMap'].data
            mw = o.matrix_world
            for p in me.polygons:
                if not p.select: continue
                a3 += p.area * (mw.to_scale()[0] ** 2)
                pts = [uvl[li].uv for li in p.loop_indices]
                s = 0.0
                for i in range(len(pts)):
                    x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % len(pts)]
                    s += x0 * y1 - x1 * y0
                auv += abs(s) * 0.5
        k = UV_PRIO.get(m, 1.0) * math.sqrt(a3 / max(auv, 1e-12))
        for o in objs:
            me = o.data; uvl = me.uv_layers['UVMap'].data
            for p in me.polygons:
                if not p.select: continue
                for li in p.loop_indices: uvl[li].uv *= k
        log(f'  uv group {m}: area {a3:.2f} m2')
    # pack all faces of the atlas
    bpy.ops.object.mode_set(mode='OBJECT')
    for o in objs:
        mi = [i for i, mm in enumerate(o.data.materials) if mm and mm.name in mats]
        for p in o.data.polygons: p.select = p.material_index in mi
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.hide(unselected=True)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(rotate=True, margin=margin, scale=True, shape_method='CONCAVE' if not FAST else 'AABB')
    bpy.ops.mesh.reveal(select=False)
    bpy.ops.object.mode_set(mode='OBJECT')

# ---------------- numpy noise ----------------
def _h3(ix, iy, iz, seed):
    h = (ix.astype(np.uint32) * np.uint32(0x8da6b343)) ^ (iy.astype(np.uint32) * np.uint32(0xd8163841)) ^ \
        (iz.astype(np.uint32) * np.uint32(0xcb1ab31f)) ^ np.uint32((seed * 0x9E3779B9) & 0xffffffff)
    h ^= h >> np.uint32(15); h *= np.uint32(0x2c1b3c6d); h ^= h >> np.uint32(12); h *= np.uint32(0x297a2d39); h ^= h >> np.uint32(15)
    return (h & np.uint32(0xffffff)).astype(np.float32) * np.float32(2.0 / 0xffffff) - np.float32(1.0)

def vnoise(P, seed=0):
    Pf = np.floor(P); f = (P - Pf).astype(np.float32); Pi = Pf.astype(np.int32)
    u = f * f * (3 - 2 * f)
    x0, y0, z0 = Pi[:, 0], Pi[:, 1], Pi[:, 2]; x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    a = _h3(x0, y0, z0, seed) * (1 - ux) + _h3(x1, y0, z0, seed) * ux
    b = _h3(x0, y1, z0, seed) * (1 - ux) + _h3(x1, y1, z0, seed) * ux
    c = _h3(x0, y0, z1, seed) * (1 - ux) + _h3(x1, y0, z1, seed) * ux
    d = _h3(x0, y1, z1, seed) * (1 - ux) + _h3(x1, y1, z1, seed) * ux
    return (a * (1 - uy) + b * uy) * (1 - uz) + (c * (1 - uy) + d * uy) * uz

def fbm(P, scale, oct=4, seed=0, gain=0.5):
    Q = (P * np.asarray(scale, np.float32)).astype(np.float32)
    s = np.zeros(len(P), np.float32); a = 1.0; tot = 0.0
    for i in range(oct):
        s += a * vnoise(Q * (2.0 ** i) + np.float32(i * 17.31), seed * 7 + i)
        tot += a; a *= gain
    return s / tot

def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

def mixc(a, b, t):
    t = np.asarray(t, np.float32)
    if t.ndim == 1: t = t[:, None]
    return a + (b - a) * t

def C(*v): return np.array(v, np.float32)

# ---------------- G-buffer baking ----------------
def _img(name, size, float_buf=True):
    im = bpy.data.images.get(name)
    if im: bpy.data.images.remove(im)
    im = bpy.data.images.new(name, size, size, alpha=True, float_buffer=float_buf)
    im.colorspace_settings.name = 'Non-Color'
    im.generated_color = (0, 0, 0, 0)
    return im

def _emit_tree(m, build):
    """Replace material m's node tree by an emission shader. build(nt) returns an output socket (color)."""
    nt = m.node_tree
    if '_orig' not in m: m['_orig'] = True
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0
    sock = build(nt)
    nt.links.new(sock, em.inputs['Color'])
    nt.links.new(em.outputs[0], out.inputs['Surface'])
    return nt

def _set_target(m, img):
    nt = m.node_tree
    tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = img
    for n in nt.nodes: n.select = False
    tn.select = True; nt.nodes.active = tn

def bake_gbuffer(objs, occluders, mats, size, tag_, hide_for_ao=None):
    """Bake position / normal / id+src / AO+cavity+edge for faces with materials in `mats`."""
    sc = bpy.context.scene
    setup_render(1, (64, 64))
    sc.render.bake.margin = 6 if size <= 1024 else 10
    sc.render.bake.margin_type = 'EXTEND'
    dummy = _img('_dummy', 8)
    allm = set()
    for o in objs: allm |= {m.name for m in o.data.materials if m}
    passes = {}
    for o in bpy.context.view_layer.objects: o.select_set(False)
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]

    def geo(nt): return nt.nodes.new('ShaderNodeNewGeometry')
    builders = {
        'P': lambda nt, mn: geo(nt).outputs['Position'],
        'N': lambda nt, mn: geo(nt).outputs['Normal'],
    }
    def b_id(nt, mn):
        cx = nt.nodes.new('ShaderNodeCombineXYZ')
        cx.inputs[0].default_value = MID.get(mn, 0) / 32.0
        uv = nt.nodes.new('ShaderNodeUVMap'); uv.uv_map = 'src'
        sp = nt.nodes.new('ShaderNodeSeparateXYZ')
        nt.links.new(uv.outputs['UV'], sp.inputs[0])
        nt.links.new(sp.outputs[0], cx.inputs[1]); nt.links.new(sp.outputs[1], cx.inputs[2])
        return cx.outputs[0]
    def b_ao(nt, mn):
        ao1 = nt.nodes.new('ShaderNodeAmbientOcclusion'); ao1.samples = 8; ao1.inputs['Distance'].default_value = 0.45
        ao2 = nt.nodes.new('ShaderNodeAmbientOcclusion'); ao2.samples = 8; ao2.inputs['Distance'].default_value = 0.05
        bv = nt.nodes.new('ShaderNodeBevel'); bv.samples = 8; bv.inputs['Radius'].default_value = 0.007
        g = geo(nt)
        dp = nt.nodes.new('ShaderNodeVectorMath'); dp.operation = 'DOT_PRODUCT'
        nt.links.new(bv.outputs[0], dp.inputs[0]); nt.links.new(g.outputs['Normal'], dp.inputs[1])
        inv = nt.nodes.new('ShaderNodeMath'); inv.operation = 'SUBTRACT'; inv.inputs[0].default_value = 1.0
        nt.links.new(dp.outputs['Value'], inv.inputs[1])
        cx = nt.nodes.new('ShaderNodeCombineXYZ')
        nt.links.new(ao1.outputs['AO'], cx.inputs[0]); nt.links.new(ao2.outputs['AO'], cx.inputs[1]); nt.links.new(inv.outputs[0], cx.inputs[2])
        return cx.outputs[0]
    builders['M'] = b_id
    builders['A'] = b_ao
    samples = {'P': 1, 'N': 1, 'M': 1, 'A': 24 if FAST else 30}   # (v4) the cabin light now comes from the analytic window visibility
    for key in ('P', 'N', 'M', 'A'):
        img = _img(f'g{key}_{tag_}', size)
        for mn in allm:
            m = bpy.data.materials[mn]
            if mn in mats:
                _emit_tree(m, lambda nt, mn=mn: builders[key](nt, mn))
                _set_target(m, img)
            else:
                _emit_tree(m, lambda nt: nt.nodes.new('ShaderNodeRGB').outputs[0])
                _set_target(m, dummy)
        sc.cycles.samples = samples[key]
        t = time.time()
        if key == 'A' and hide_for_ao: hide_for_ao(True)
        bpy.ops.object.bake(type='EMIT', use_clear=True, margin=sc.render.bake.margin)
        if key == 'A' and hide_for_ao: hide_for_ao(False)
        a = np.empty(size * size * 4, np.float32); img.pixels.foreach_get(a)
        passes[key] = a.reshape(size, size, 4)
        log(f'  baked {key}/{tag_} in {time.time() - t:.1f}s')
    return passes

# ---------------- text / 2D texture rendering ----------------
FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Black.ttf'
FONT_DIN = '/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf'
FONT_REG = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'

def render_2d(items, world_w, world_h, px_w, px_h, path, bg=(0, 0, 0)):
    """Render flat 2D shapes/text (x in [0,world_w], y in [0,world_h]) with an ortho camera (Workbench)."""
    main = bpy.context.window.scene if bpy.context.window else bpy.context.scene
    sc = bpy.data.scenes.new('tex2d')
    col = bpy.data.collections.new('tex2d_c'); sc.collection.children.link(col)
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'FLAT'; sc.display.shading.color_type = 'OBJECT'
    sc.display.render_aa = '16'
    sc.view_settings.view_transform = 'Standard'
    sc.render.resolution_x, sc.render.resolution_y = px_w, px_h; sc.render.resolution_percentage = 100
    w = bpy.data.worlds.new('tex2d_w'); w.color = bg; sc.world = w
    sc.render.film_transparent = False
    cam = bpy.data.objects.new('tex2d_cam', bpy.data.cameras.new('tex2d_cam')); col.objects.link(cam)
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = max(world_w, world_h)
    cam.location = (world_w / 2, world_h / 2, 5); sc.camera = cam
    made = []
    for it in items:
        t = it['type']; z = it.get('z', 0.0)
        if t == 'text':
            cu = bpy.data.curves.new('t', 'FONT'); cu.body = it['body']; cu.size = it['size']
            cu.align_x = it.get('align', 'CENTER'); cu.align_y = 'CENTER'
            fp = it.get('font', FONT_BOLD)
            if os.path.exists(fp): cu.font = bpy.data.fonts.load(fp, check_existing=True)
            cu.space_character = it.get('spacing', 1.0)
            ob = bpy.data.objects.new('t', cu)
            ob.location = (it['x'], it['y'], z); ob.rotation_euler = (0, 0, math.radians(it.get('rot', 0)))
            if 'sx' in it: ob.scale = (it['sx'], 1, 1)
        else:
            bm = bmesh.new()
            if t == 'rect':
                pts = [(it['x0'], it['y0']), (it['x1'], it['y0']), (it['x1'], it['y1']), (it['x0'], it['y1'])]
            elif t == 'poly':
                pts = it['pts']
            elif t == 'circle':
                pts = circle2(it['x'], it['y'], it['r'], it.get('n', 48))
            elif t == 'ring':
                n = it.get('n', 96); r0, r1 = it['r0'], it['r1']
                a0, a1 = math.radians(it.get('a0', 0)), math.radians(it.get('a1', 360))
                vs0 = []; vs1 = []
                for k in range(n + 1):
                    a = a0 + (a1 - a0) * k / n
                    vs0.append(bm.verts.new((it['x'] + r0 * math.cos(a), it['y'] + r0 * math.sin(a), z)))
                    vs1.append(bm.verts.new((it['x'] + r1 * math.cos(a), it['y'] + r1 * math.sin(a), z)))
                for k in range(n): bm.faces.new((vs0[k], vs0[k + 1], vs1[k + 1], vs1[k]))
                pts = None
            if pts is not None:
                vs = [bm.verts.new((x, y, z)) for x, y in pts]; bm.faces.new(vs)
            me = bpy.data.meshes.new('s'); bm.to_mesh(me); bm.free()
            ob = bpy.data.objects.new('s', me)
        ob.color = (*it.get('color', (1, 1, 1)), 1)
        col.objects.link(ob); made.append(ob)
    sc.render.filepath = path
    with bpy.context.temp_override(scene=sc):
        bpy.ops.render.render(write_still=True, scene=sc.name)
    for ob in made:
        d = ob.data; bpy.data.objects.remove(ob)
        if isinstance(d, bpy.types.Curve): bpy.data.curves.remove(d)
        elif d: bpy.data.meshes.remove(d)
    bpy.data.objects.remove(cam); bpy.data.scenes.remove(sc); bpy.data.collections.remove(col); bpy.data.worlds.remove(w)
    return load_png_np(path)

def load_png_np(path):
    im = bpy.data.images.load(path, check_existing=False)
    im.colorspace_settings.name = 'Non-Color'
    w, h = im.size
    a = np.empty(w * h * 4, np.float32); im.pixels.foreach_get(a)
    bpy.data.images.remove(im)
    return a.reshape(h, w, 4)

def save_png(arr, path, srgb=False, name=None):
    """arr: HxWx3 or HxWx4 float (linear if srgb=True: converted)."""
    h, w = arr.shape[:2]
    if arr.shape[2] == 3: arr = np.concatenate([arr, np.ones((h, w, 1), np.float32)], 2)
    a = np.clip(arr, 0, 1).astype(np.float32)
    if srgb:
        c = a[..., :3]
        a = a.copy(); a[..., :3] = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    im = bpy.data.images.new(name or os.path.basename(path), w, h, alpha=True)
    im.colorspace_settings.name = 'Non-Color'
    im.pixels.foreach_set(a.ravel())
    im.filepath_raw = path; im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)
    return path

def plate_texture():
    items = [{'type': 'rect', 'x0': 0, 'y0': 0, 'x1': 0.52, 'y1': 0.11, 'color': (0.92, 0.92, 0.9)},
             {'type': 'rect', 'x0': 0.004, 'y0': 0.004, 'x1': 0.516, 'y1': 0.106, 'color': (0.05, 0.05, 0.05), 'z': 0.001},
             {'type': 'rect', 'x0': 0.008, 'y0': 0.008, 'x1': 0.512, 'y1': 0.102, 'color': (0.92, 0.92, 0.9), 'z': 0.002},
             {'type': 'text', 'body': 'K 4719 RA', 'x': 0.285, 'y': 0.0535, 'size': 0.105, 'font': FONT_DIN, 'color': (0.03, 0.03, 0.03), 'z': 0.003, 'spacing': 1.05, 'sx': 1.0},
             {'type': 'circle', 'x': 0.045, 'y': 0.055, 'r': 0.022, 'color': (0.12, 0.35, 0.18), 'z': 0.003},
             {'type': 'circle', 'x': 0.045, 'y': 0.055, 'r': 0.015, 'color': (0.8, 0.75, 0.3), 'z': 0.004},
             {'type': 'text', 'body': '26', 'x': 0.045, 'y': 0.055, 'size': 0.014, 'font': FONT_REG, 'color': (0.05, 0.05, 0.05), 'z': 0.005}]
    return render_2d(items, 0.52, 0.11, 1040, 220, os.path.join(TEXDIR, 'src_plate.png'), (0.92, 0.92, 0.9))

def cover_texture():
    items = []
    # stylised mountains + brand on the spare wheel cover (white = print)
    items.append({'type': 'ring', 'x': 0.5, 'y': 0.5, 'r0': 0.375, 'r1': 0.405, 'color': (1, 1, 1)})
    items.append({'type': 'poly', 'pts': [(0.20, 0.46), (0.35, 0.67), (0.42, 0.59), (0.53, 0.76), (0.67, 0.55), (0.73, 0.62), (0.82, 0.46)], 'color': (1, 1, 1)})
    items.append({'type': 'text', 'body': 'MARAL', 'x': 0.5, 'y': 0.355, 'size': 0.16, 'font': FONT_BOLD, 'color': (1, 1, 1), 'spacing': 1.05})
    items.append({'type': 'text', 'body': '4x4', 'x': 0.5, 'y': 0.235, 'size': 0.085, 'font': FONT_BOLD, 'color': (1, 1, 1), 'spacing': 1.1})
    return render_2d(items, 1.0, 1.0, 1024, 1024, os.path.join(TEXDIR, 'src_cover.png'), (0, 0, 0))

def gauge_texture():
    items = []
    W = (0.93, 0.95, 0.92)
    def dial(cx, cy, r, labels, a_from=225, a_to=-45, major=None, minor=4, red_from=None, title=None, sub=None):
        items.append({'type': 'circle', 'x': cx, 'y': cy, 'r': r, 'color': (0.012, 0.012, 0.014), 'n': 64})
        items.append({'type': 'ring', 'x': cx, 'y': cy, 'r0': r * 0.965, 'r1': r, 'color': (0.18, 0.18, 0.18), 'z': 0.001})
        n = len(labels) - 1
        for i, lab in enumerate(labels):
            a = math.radians(a_from + (a_to - a_from) * i / n)
            for k in range(minor + 1 if i < n else 1):
                aa = math.radians(a_from + (a_to - a_from) * (i + k / (minor + 1)) / n)
                L = 0.13 if k == 0 else 0.07
                wdt = 0.012 if k == 0 else 0.006
                col = (0.85, 0.12, 0.08) if (red_from is not None and i + k / (minor + 1) >= red_from) else W
                ca, sa = math.cos(aa), math.sin(aa)
                p0 = (cx + ca * r * 0.93, cy + sa * r * 0.93); p1 = (cx + ca * r * (0.93 - L), cy + sa * r * (0.93 - L))
                nx, ny = -sa * r * wdt, ca * r * wdt
                items.append({'type': 'poly', 'pts': [(p0[0] + nx, p0[1] + ny), (p0[0] - nx, p0[1] - ny), (p1[0] - nx, p1[1] - ny), (p1[0] + nx, p1[1] + ny)], 'color': col, 'z': 0.002})
            if lab:
                items.append({'type': 'text', 'body': lab, 'x': cx + math.cos(a) * r * 0.64, 'y': cy + math.sin(a) * r * 0.64, 'size': r * 0.16, 'font': FONT_REG, 'color': W, 'z': 0.003})
        if title: items.append({'type': 'text', 'body': title, 'x': cx, 'y': cy - r * 0.38, 'size': r * 0.1, 'font': FONT_REG, 'color': (0.7, 0.72, 0.7), 'z': 0.003})
        if sub: items.append({'type': 'text', 'body': sub, 'x': cx, 'y': cy + r * 0.34, 'size': r * 0.09, 'font': FONT_REG, 'color': (0.7, 0.72, 0.7), 'z': 0.003})
    dial(0.25, 0.75, 0.24, ['0', '20', '40', '60', '80', '100', '120', '140', '160'], title='km/h', sub='0 4 7 1 2 6')
    dial(0.75, 0.75, 0.24, ['0', '1', '2', '3', '4', '5', '6', '7'], red_from=5.5, title='x1000 r/min')
    dial(0.125, 0.125, 0.12, ['E', '', '1/2', '', 'F'], a_from=150, a_to=30, minor=1)
    dial(0.375, 0.125, 0.12, ['C', '', '', '', 'H'], a_from=150, a_to=30, minor=1)
    items.append({'type': 'rect', 'x0': 0.80, 'y0': 0.13, 'x1': 0.98, 'y1': 0.20, 'color': (1.0, 0.42, 0.08)})
    dash_prints(items)
    return render_2d(items, 1.0, 1.0, 1024, 1024, os.path.join(TEXDIR, 'gauge.png'), (0.01, 0.01, 0.012))

def dash_prints(items):
    """Radio face, heater panel, switch bank and warning-lamp strip (drawn in panel millimetres into DASH_PRINTS rects).
    The gauge material uses this texture as albedo AND emission, so only what a real 90s dash backlights is bright:
    the radio tuning scale, heater symbols, switch tell-tales; the rest is dark print on black plastic."""
    PANEL_MM = {'radio': (186, 53), 'heater': (196, 49), 'switches': (130, 39), 'warn': (190, 8)}
    def panel(key):
        u0, v0, u1, v1 = DASH_PRINTS[key]; wmm, hmm = PANEL_MM[key]
        sx = (u1 - u0) / wmm; sy = (v1 - v0) / hmm
        P = lambda x, y: (u0 + x * sx, v0 + y * sy)
        def rect(x0, y0, x1, y1, col, z=0.001):
            a = P(x0, y0); b = P(x1, y1)
            items.append({'type': 'rect', 'x0': a[0], 'y0': a[1], 'x1': b[0], 'y1': b[1], 'color': col, 'z': z})
        def text(body, x, y, hmm_, col, z=0.004, font=FONT_REG, spacing=1.0):
            p = P(x, y)
            items.append({'type': 'text', 'body': body, 'x': p[0], 'y': p[1], 'size': hmm_ * sy * 1.35, 'font': font, 'color': col, 'z': z,
                          'spacing': spacing, 'sx': sx / sy})
        def circ(x, y, r, col, z=0.002):
            p = P(x, y); items.append({'type': 'circle', 'x': p[0], 'y': p[1], 'r': r * sy, 'color': col, 'z': z, 'n': 32})
        return rect, text, circ, sx, sy
    GREY = (0.30, 0.30, 0.29); DIM = (0.16, 0.16, 0.155); BG = (0.028, 0.028, 0.03)
    AMBER = (0.95, 0.55, 0.16); RED = (0.9, 0.12, 0.06)
    # ---- radio (knobs sit at x=23 and x=163 mm) ----
    rect, text, circ, sx, sy = panel('radio')
    rect(0, 0, 186, 53, BG, 0.0005)
    rect(0.8, 0.8, 185.2, 52.2, (0.045, 0.045, 0.047), 0.0007)
    rect(1.6, 1.6, 184.4, 51.4, BG, 0.0009)
    for x, lab in ((23, 'VOL'), (163, 'TUNE')):
        text(lab, x, 7.0, 2.6, GREY)
        text('PUSH', x, 45.5, 2.0, DIM)
    rect(38, 20, 148, 47, (0.10, 0.10, 0.10), 0.001)                          # tuning window bezel
    rect(39, 21, 147, 46, (0.030, 0.024, 0.018), 0.0015)                     # smoked window
    for row, (band, labs) in enumerate((('FM', ['88', '92', '96', '100', '104', '108']), ('AM', ['53', '60', '70', '90', '120', '160']))):
        y = 39.0 - row * 10.5
        text(band, 44.5, y, 2.8, AMBER)
        for i, lab in enumerate(labs):
            x = 55 + i * 17.5
            text(lab, x, y, 2.8, AMBER)
            rect(x - 0.25, y - 4.6, x + 0.25, y - 2.8, AMBER, 0.003)
            if i < len(labs) - 1:
                for q in (1, 2, 3):
                    xx = x + q * 17.5 / 4
                    rect(xx - 0.15, y - 4.0, xx + 0.15, y - 2.8, AMBER, 0.003)
    rect(97.6, 22.0, 98.4, 45.0, RED, 0.0035)                                # tuning pointer
    rect(44, 7.5, 142, 16.5, (0.09, 0.09, 0.09), 0.001)                      # cassette door
    rect(45, 8.5, 141, 15.5, (0.010, 0.010, 0.011), 0.0015)
    rect(88, 11.2, 98, 12.8, (0.20, 0.20, 0.20), 0.002)                      # door latch
    text('MARAL  STEREO  CASSETTE', 93, 49.0, 2.0, GREY, spacing=1.1)
    for i, lab in enumerate(('1', '2', '3', '4', '5')):
        x = 46 + i * 5.5
        rect(x - 2.2, 17.2, x + 2.2, 19.4, (0.12, 0.12, 0.12), 0.002)
    text('ST', 141, 42.5, 2.2, (0.25, 0.75, 0.3))
    # ---- heater (slider knobs at x=38, 98, 158 mm) ----
    rect, text, circ, sx, sy = panel('heater')
    rect(0, 0, 196, 49, BG, 0.0005)
    rect(0.8, 0.8, 195.2, 48.2, (0.05, 0.05, 0.052), 0.0007)
    rect(1.6, 1.6, 194.4, 47.4, BG, 0.0009)
    for c in (38, 98, 158):
        rect(c - 29, 22.2, c + 29, 26.8, (0.006, 0.006, 0.006), 0.001)    # slider slot
    n = 24
    for i in range(n):                                                         # temperature bar: blue -> red
        t = i / (n - 1)
        rect(10 + i * 56 / n, 33.5, 10 + (i + 1) * 56 / n + 0.05, 36.0, (0.15 + 0.8 * t, 0.25 + 0.1 * (1 - abs(2 * t - 1)), 0.95 - 0.85 * t), 0.002)
    for i, lab in enumerate(('0', '1', '2', '3')):
        text(lab, 74 + i * 16, 34.8, 3.0, AMBER)
    for i, lab in enumerate(('FACE', 'FEET', 'DEF')):
        text(lab, 134 + i * 24, 34.8, 2.6, AMBER)
    for c, lab in ((38, 'TEMP'), (98, 'FAN'), (158, 'AIR')):
        text(lab, c, 10.5, 2.6, GREY)
    text('A/C', 186, 10.5, 2.0, DIM)
    # ---- switch bank: 4 rockers ----
    rect, text, circ, sx, sy = panel('switches')
    rect(0, 0, 130, 39, BG, 0.0005)
    for i, (lab, tell) in enumerate((('LIGHTS', (0.25, 0.8, 0.3)), ('FOG', (0.25, 0.8, 0.3)), ('RR DEF', AMBER), ('HAZARD', RED))):
        x0 = 6 + i * 31
        rect(x0, 6, x0 + 25, 33, (0.06, 0.06, 0.062), 0.001)
        rect(x0 + 1, 20, x0 + 24, 32, (0.105, 0.105, 0.105), 0.0015)            # rocker upper half catches light
        rect(x0 + 1, 7, x0 + 24, 19.2, (0.035, 0.035, 0.036), 0.0015)
        rect(x0 + 10, 28, x0 + 15, 30, tell, 0.002)                              # tell-tale window
        text(lab, x0 + 12.5, 3.0, 2.2, GREY)
    # hazard triangle on the last rocker
    x0 = 6 + 3 * 31 + 12.5
    a = (x0 - 5, 22); b = (x0 + 5, 22); c = (x0, 26.5)
    u0, v0, _, _ = DASH_PRINTS['switches']
    items.append({'type': 'poly', 'pts': [(u0 + p[0] * sx, v0 + p[1] * sy) for p in (a, b, c)], 'color': RED, 'z': 0.003})
    # ---- warning-lamp strip (dim coloured windows) ----
    rect, text, circ, sx, sy = panel('warn')
    rect(0, 0, 190, 8, (0.012, 0.012, 0.013), 0.0005)
    # (v4) lamps are OFF while driving: dark tinted windows only (they glowed like LEDs through the emissive map)
    cols = [(0.06, 0.012, 0.01), (0.06, 0.012, 0.01), (0.06, 0.035, 0.01), (0.012, 0.045, 0.014), (0.012, 0.025, 0.07),
            (0.012, 0.045, 0.014), (0.06, 0.035, 0.01), (0.06, 0.012, 0.01)]
    for i, col in enumerate(cols):
        x = 12 + i * 23.7
        rect(x - 7, 1.5, x + 7, 6.5, col, 0.001)
        rect(x - 1.5, 3.4, x + 1.5, 4.6, tuple(min(1, k * 1.5) for k in col), 0.002)

def tire_letter_strip():
    """Text strip for the outer sidewall: x in [0,1] = half circumference, y = radius (m) 0.19..0.36."""
    W_ = 1.0; H_ = 0.17; y0 = 0.19
    it = []
    it.append({'type': 'text', 'body': 'MARAL', 'x': 0.25, 'y': 0.300 - y0, 'size': 0.030, 'font': FONT_BOLD, 'sx': 1.25, 'spacing': 1.2})
    it.append({'type': 'text', 'body': 'TERRA - TRAC  A/T', 'x': 0.25, 'y': 0.268 - y0, 'size': 0.012, 'font': FONT_REG, 'sx': 1.25, 'spacing': 1.2})
    it.append({'type': 'text', 'body': '215/75 R15  100S', 'x': 0.72, 'y': 0.288 - y0, 'size': 0.018, 'font': FONT_BOLD, 'sx': 1.25, 'spacing': 1.1})
    it.append({'type': 'text', 'body': 'M+S   TUBELESS   RADIAL', 'x': 0.72, 'y': 0.262 - y0, 'size': 0.009, 'font': FONT_REG, 'sx': 1.25, 'spacing': 1.2})
    it.append({'type': 'text', 'body': 'DOT X7K3 2319', 'x': 0.95, 'y': 0.228 - y0, 'size': 0.007, 'font': FONT_REG, 'sx': 1.25})
    for rr in (0.236, 0.323):
        it.append({'type': 'rect', 'x0': 0, 'y0': rr - y0 - 0.0008, 'x1': 1.0, 'y1': rr - y0 + 0.0008, 'color': (0.6, 0.6, 0.6)})
    return render_2d(it, W_, H_, 2048, 348, os.path.join(TEXDIR, 'src_tire_text.png'), (0, 0, 0))

# ---------------- composition ----------------
WHEELS_B = [(HT, -FAX, R_TIRE), (-HT, -FAX, R_TIRE), (HT, -RAX, R_TIRE), (-HT, -RAX, R_TIRE)]   # blender coords

def sample_img(img, u, v):
    """Bilinear sample HxWxC image at uv (arrays) with wrap on u, clamp v."""
    h, w = img.shape[:2]
    x = (np.asarray(u) % 1.0) * w - 0.5; y = np.clip(np.asarray(v), 0, 1) * h - 0.5
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    x0 %= w; x1 = (x0 + 1) % w; y0c = np.clip(y0, 0, h - 1); y1c = np.clip(y0 + 1, 0, h - 1)
    a = img[y0c, x0] * (1 - fx) + img[y0c, x1] * fx
    b = img[y1c, x0] * (1 - fx) + img[y1c, x1] * fx
    return a * (1 - fy) + b * fy

def height_to_normal(H, Pimg, cov, strength=1.0):
    """Tangent-space normal map from a UV-space height map (meters), using the baked position map for texel size."""
    size = H.shape[0]
    P = Pimg[..., :3]
    def grad(axis):
        Hp = np.roll(H, -1, axis); Hm = np.roll(H, 1, axis)
        Pp = np.roll(P, -1, axis); Pm = np.roll(P, 1, axis)
        cp = np.roll(cov, -1, axis); cm = np.roll(cov, 1, axis)
        dp = np.linalg.norm(Pp - P, axis=2); dm = np.linalg.norm(P - Pm, axis=2)
        med = np.median(dp[cov & cp]) if np.any(cov & cp) else 0.003
        okp = cp & (dp < med * 4) & (dp > 1e-7); okm = cm & (dm < med * 4) & (dm > 1e-7)
        g = np.zeros_like(H)
        both = okp & okm
        g[both] = (Hp[both] - Hm[both]) / (dp[both] + dm[both])
        only_p = okp & ~okm; g[only_p] = (Hp[only_p] - H[only_p]) / dp[only_p]
        only_m = okm & ~okp; g[only_m] = (H[only_m] - Hm[only_m]) / dm[only_m]
        return g
    gx = grad(1); gy = grad(0)
    n = np.stack([-gx * strength, -gy * strength, np.ones_like(H)], 2)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    n = n * 0.5 + 0.5
    n[~cov] = (0.5, 0.5, 1.0)
    return n.astype(np.float32)

def worley(P, cell, seed, stretch=(1.0, 1.0, 1.0)):
    """Cellular noise: F1 distance (cell units, anisotropic via stretch) to jittered feature points + 2 per-cell hashes."""
    Q = (P / (np.asarray(stretch, np.float32) * np.float32(cell))).astype(np.float32)
    Qf = np.floor(Q); fr = (Q - Qf).astype(np.float32); Qi = Qf.astype(np.int32)
    del Q, Qf
    n = len(P)
    best = np.full(n, 9.0, np.float32); h1 = np.zeros(n, np.float32); h2 = np.zeros(n, np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cx = Qi[:, 0] + dx; cy = Qi[:, 1] + dy; cz = Qi[:, 2] + dz
                ox = np.float32(dx + 0.5) + np.float32(0.5) * _h3(cx, cy, cz, seed) - fr[:, 0]
                oy = np.float32(dy + 0.5) + np.float32(0.5) * _h3(cx, cy, cz, seed + 1) - fr[:, 1]
                oz = np.float32(dz + 0.5) + np.float32(0.5) * _h3(cx, cy, cz, seed + 2) - fr[:, 2]
                d = ox * ox + oy * oy + oz * oz
                m = d < best
                best[m] = d[m]
                h1[m] = _h3(cx[m], cy[m], cz[m], seed + 3) * 0.5 + 0.5
                h2[m] = _h3(cx[m], cy[m], cz[m], seed + 4) * 0.5 + 0.5
    return np.sqrt(best), h1, h2

def mip_chain(img, levels=10):
    ch = [img.astype(np.float32)]
    for _ in range(levels):
        a = ch[-1]; h, w = a.shape[:2]
        if h < 8 or w < 8: break
        a = a[:h // 2 * 2, :w // 2 * 2]
        ch.append(a.reshape(h // 2, 2, w // 2, 2, a.shape[2]).mean((1, 3)))
    return ch

def src_lod(G, mid_value, img_w, img_h):
    """Mip level for sampling a 2D print through the baked 'src' UV: median source-pixel footprint of one atlas texel."""
    M = G['M']; cov = G['P'][..., 3] > 0.5
    mm = np.rint(M[..., 0] * 32).astype(np.int32) == mid_value
    fps = []
    for axis in (0, 1):
        dU = np.abs(np.roll(M[..., 1], -1, axis) - M[..., 1]) * img_w
        dV = np.abs(np.roll(M[..., 2], -1, axis) - M[..., 2]) * img_h
        ok = cov & mm & np.roll(cov & mm, -1, axis)
        d = np.maximum(dU, dV)[ok]
        if d.size: fps.append(d[d < 64])
    fp = np.concatenate(fps) if fps else np.array([1.0])
    med = float(np.median(fp)) if fp.size else 1.0
    return max(0.0, math.log2(max(med, 1.0)))

def sample_lod(chain, u, v, lod):
    l0 = int(min(math.floor(lod), len(chain) - 1)); l1 = min(l0 + 1, len(chain) - 1); t = lod - l0
    a = sample_img(chain[l0], u, v)
    if l1 == l0 or t < 1e-3: return a
    return a * (1 - t) + sample_img(chain[l1], u, v) * t

# rain / dirty-water tear streaks: emitters in car coords (l, f, h), 'side' (panel normal +-l) or 'rear'/'front' faces
STREAK_EMITTERS = [
    ('side', 0.80, -0.395, 1.085), ('side', 0.80, 0.612, 1.085), ('side', 0.80, -0.545, 1.085), ('side', 0.80, -1.615, 1.085),
    ('side', 0.80, 0.585, 1.075), ('side', 0.80, -0.26, 0.905), ('side', 0.80, -1.25, 0.84), ('side', 0.80, -1.735, 1.60),
    ('rear', 0.585, -1.77, 1.13), ('rear', 0.67, -1.77, 0.585), ('rear', 0.30, -1.77, 0.55), ('rear', 0.52, -1.77, 0.98),
    ('front', 0.565, 1.79, 0.69), ('front', 0.30, 1.79, 0.66),
]

def tear_streaks(P, N, seed=60):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]; f = -y
    out = np.zeros(len(P), np.float32)
    rng = np.random.RandomState(seed)
    wob = 0.004 * vnoise(np.stack([x * 3, y * 3, z * 22], 1).astype(np.float32), 61)
    for kind, l, fe, he in STREAK_EMITTERS:
        for sgn in (1, -1):
            for k in range(3):
                off = (rng.rand() - 0.5) * 0.045
                L = 0.12 + 0.45 * rng.rand()
                amp = 0.35 + 0.65 * rng.rand()
                dz = he - z
                below = sstep(-0.004, 0.012, dz)
                w = 0.0035 + 0.010 * np.clip(dz, 0, 1)
                if kind == 'side':
                    face = sstep(0.6, 0.85, N[:, 0] * sgn) * sstep(0.55, 0.70, x * sgn)
                    hd = f - (fe + off) + wob
                elif kind == 'rear':
                    face = sstep(0.6, 0.85, N[:, 1])
                    hd = x - sgn * (l + off) + wob
                else:
                    face = sstep(0.6, 0.85, -N[:, 1])
                    hd = x - sgn * (l + off) + wob
                if kind != 'side' and sgn < 0 and l < 0.01: continue
                s = np.exp(-(hd / w) ** 2) * np.exp(-np.maximum(dz, 0) / L) * below * face * amp
                out = np.maximum(out, s)
    return np.clip(out, 0, 1)

def compose_ext(G, size, plate_img, cover_img):
    log('compose ext')
    cov = G['P'][..., 3] > 0.5
    idx = np.nonzero(cov)
    P = G['P'][idx][:, :3].astype(np.float32)
    N = G['N'][idx][:, :3].astype(np.float32)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-6)
    mid = np.rint(G['M'][idx][:, 0] * 32).astype(np.int32)
    su = G['M'][idx][:, 1]; sv = G['M'][idx][:, 2]
    AO = np.clip(G['A'][idx][:, 0], 0, 1); CAV = np.clip(G['A'][idx][:, 1], 0, 1); EDGE = np.clip(G['A'][idx][:, 2] * 6.0, 0, 1)
    n = len(P)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    f = -y
    is_ = lambda name: mid == MID[name]
    alb = np.zeros((n, 3), np.float32); rough = np.full(n, 0.5, np.float32); metal = np.zeros(n, np.float32)
    coat = np.zeros(n, np.float32); coat_r = np.full(n, 0.08, np.float32); height = np.zeros(n, np.float32)
    # ---- shared masks ----
    up = sstep(0.35, 0.9, N[:, 2]); down = sstep(0.2, 0.7, -N[:, 2])
    side = sstep(0.55, 0.9, np.abs(N[:, 0])); vert = 1 - np.abs(N[:, 2])
    rearf = sstep(0.5, 0.9, N[:, 1]); frontf = sstep(0.5, 0.9, -N[:, 1])
    low = sstep(1.0, 0.38, z)
    n1 = fbm(P, (2.2, 2.2, 2.2), 4, 1)
    n2 = fbm(P, (9, 9, 9), 4, 2)
    n3 = fbm(P, (38, 38, 38), 3, 3)
    is_wheel = is_('wheel') | is_('brake')

    # ---- mud: road-spray band, rearward wheel fans, arch surrounds, rear/front low bands, underside ----
    band = sstep(0.82, 0.30, z + 0.10 * n1 + 0.04 * n2)
    fan = np.zeros(n, np.float32)
    for wx, wy, wz in WHEELS_B:
        sidew = sstep(0.42, 0.70, x * np.sign(wx))
        dy = y - wy                                          # > 0: behind this wheel
        behind = sstep(0.22, 0.42, dy)
        reach = np.exp(-np.maximum(dy - 0.35, 0) / 0.85)
        top = 0.42 + 0.42 * np.clip(dy - 0.30, 0, 1.4)       # fan top rises rearward
        under = sstep(top + 0.10, top - 0.24, z + 0.05 * n2)
        d = np.sqrt((y - wy) ** 2 + (z - ARCH_HC) ** 2)
        ring = np.exp(-((d - ARCH_R - 0.07) / 0.09) ** 2) * (0.45 + 0.55 * sstep(wy - 0.25, wy + 0.30, y))
        fan = np.maximum(fan, np.clip(behind * reach * under + ring * 0.6, 0, 1) * sidew)
    rear_low = rearf * sstep(1.15, 0.40, z) * 1.15      # (v4) boxy tail: the wake deposits road spray high up the tailgate
    front_low = frontf * sstep(0.62, 0.34, z) * 0.7
    dens = np.clip(band * 0.72 + fan * 0.95 + rear_low * 0.6 + front_low + down * 0.85 + 0.10 * n1, 0, 1)
    dens[is_wheel] = 0.0
    # discrete splats at the fringe (elongated along the car: spray hits at a grazing angle)
    log('  splats')
    wv, h1, h2 = worley(P, 0.040, 50, (1.0, 1.8, 1.0))
    rb = 0.10 + 0.30 * h2 ** 1.4
    e = rb - (wv + 0.09 * vnoise(P * 300, 51))
    splat_b = sstep(-0.025, 0.025, e) * (h1 < sstep(0.05, 0.52, dens) * 0.95)
    del wv
    ws_, g1, g2 = worley(P, 0.014, 52, (1.0, 1.4, 1.0))
    e2 = (0.12 + 0.30 * g2) - (ws_ + 0.12 * vnoise(P * 600, 53))
    splat_s = sstep(-0.03, 0.03, e2) * (g1 < sstep(0.02, 0.45, dens)) * sstep(1.0, 0.62, z)
    splat_w = sstep(-0.03, 0.03, e2) * (g1 < 0.28)          # density-free speckle (rims rotate: no gravity bias)
    del ws_
    solid = sstep(0.46, 0.74, dens + 0.18 * n2 + 0.06 * n3)
    wash = sstep(0.30, 0.72, fbm(P, (48, 48, 1.3), 3, 42)) * vert * 0.6          # rain-washed vertical channels
    drip = sstep(0.60, 0.82, fbm(P, (95, 95, 2.2), 2, 41)) * vert * sstep(0.08, 0.40, dens) * sstep(0.95, 0.55, dens)
    splat = np.maximum(splat_b, splat_s * 0.9)
    # (v4) radial spray streaks: droplets leave the tyre tangentially from its rear-bottom quadrant at 5-60 deg and hit
    # the sill / door bottom / rear quarter at a grazing angle -> elongated streak fans radiating from behind each wheel
    rays = np.zeros(n, np.float32)
    for wx, wy, wz in WHEELS_B:
        sidew = sstep(0.45, 0.72, x * np.sign(wx))
        ry, rz = y - (wy + 0.30), z - 0.12
        rr = np.sqrt(ry ** 2 + rz ** 2); th = np.arctan2(rz, ry)
        sec = sstep(0.02, 0.16, th) * sstep(1.15, 0.80, th) * sstep(0.05, 0.25, rr) * np.exp(-np.maximum(rr - 0.35, 0) / 0.40)
        m_ = (sec * sidew) > 0.01
        if not m_.any(): continue
        Q = np.stack([th[m_] * 55, rr[m_] * 5, np.full(m_.sum(), wy * 10, np.float32)], 1).astype(np.float32)
        st = sstep(0.30, 0.75, vnoise(Q, 97) * 0.5 + 0.5)
        Q2 = np.stack([th[m_] * 170, rr[m_] * 22, np.zeros(m_.sum(), np.float32)], 1).astype(np.float32)
        fine = sstep(0.45, 0.8, vnoise(Q2, 98) * 0.5 + 0.5)
        v_ = (sec * sidew)[m_] * np.clip(st * 0.85 + fine * 0.55, 0, 1)
        rays[m_] = np.maximum(rays[m_], v_)
    splat = np.maximum(splat, rays * 0.95)
    thick = np.clip(solid * (0.55 + 0.45 * n2) + splat_b * (0.45 + 0.4 * h2) + rays * 0.5, 0, 1)
    dry = np.clip(sstep(0.10, 0.50, fbm(P, (3.5, 3.5, 3.5), 3, 4)) * 0.45 + down * 0.35 + sstep(0.30, 0.15, z) * 0.2 + 0.25 * sstep(0.35, 0.7, z), 0, 1) * (1 - 0.5 * up)
    dry = dry * (0.55 + 0.45 * down)      # (v2) it is raining: exposed mud is wet and dark, only sheltered undersides stay dry
    mvar = sstep(-0.5, 0.5, n3) * 0.6 + h2 * 0.4
    # thin film: tailgate wake dirt, crevices, lower panels
    film = np.clip((1 - CAV) * 0.75 + (1 - AO) * 0.12 + rearf * (0.35 + 0.45 * sstep(1.6, 0.6, z)) + band * 0.25 + 0.10 * n1, 0, 1) \
        * (0.55 + 0.45 * sstep(-0.3, 0.3, n2)) * (1 - 0.5 * up)
    streak_n = fbm(P, (26, 26, 0.9), 3, 5)
    streak = sstep(0.18, 0.45, streak_n) * vert * sstep(1.62, 1.35, z) * (0.2 + 0.4 * sstep(-0.2, 0.4, n1))
    tears = tear_streaks(P, N)
    # rust (edges, arch lips, bottom edges, seams, windshield corners)
    arch = np.zeros(n, np.float32)
    for wx, wy, wz in WHEELS_B:
        d = np.sqrt((y - wy) ** 2 + (z - ARCH_HC) ** 2)
        arch = np.maximum(arch, np.exp(-((d - ARCH_R - 0.03) / 0.05) ** 2) * sstep(0.6, 0.78, np.abs(x)))
    seam = sstep(0.75, 0.35, CAV)
    ws_base = np.exp(-(((z - 1.08) / 0.05) ** 2)) * sstep(0.55, 0.78, np.abs(x)) * sstep(0.45, 0.8, f)
    door_bot = np.exp(-(((z - DOOR_H0) / 0.035) ** 2)) * sstep(0.7, 0.79, np.abs(x)) * sstep(DOOR_F1 - 0.05, DOOR_F1 + 0.05, f) * sstep(DOOR_F0 + 0.05, DOOR_F0 - 0.05, f)
    tail_bot = np.exp(-(((z - 0.49) / 0.05) ** 2)) * rearf
    rust_base = EDGE * 0.55 + arch * 0.95 + sstep(0.62, 0.43, z) * 0.5 + seam * sstep(0.8, 0.45, z) * 0.5 + ws_base * 0.6 + down * 0.3 \
        + door_bot * 0.6 + tail_bot * 0.5
    rn = 0.45 * n2 + 0.2 * n3
    rust = sstep(0.62, 0.80, rust_base * 0.85 + rn)
    rust_halo = np.clip(sstep(0.50, 0.66, rust_base * 0.85 + rn) - rust, 0, 1)
    blister = sstep(0.55, 0.62, rust_base * 0.85 + rn) * (1 - rust) * sstep(0.3, 0.7, vnoise(P * 220, 44) * 0.5 + 0.5)
    # scratches (long, mostly horizontal, on side panels) and stone chips (front, low, hood front edge)
    sc_n = fbm(P, (1.3, 1.3, 26), 3, 6)
    scratch = (1 - sstep(0.0, 0.03, np.abs(sc_n))) * sstep(0.35, 0.6, fbm(P, (1.5, 1.5, 3), 2, 7)) * side * sstep(0.45, 0.6, z) * sstep(1.45, 1.3, z)
    chips = sstep(0.72, 0.8, vnoise(P * 140, 8)) * sstep(0.2, 0.6, fbm(P, (4, 4, 4), 2, 9)) * np.clip(frontf * 0.9 + low * 0.35 + up * sstep(0.9, 1.3, f) * 0.6 + EDGE * 0.6, 0, 1)

    MUD_WET_D = C(0.058, 0.045, 0.032); MUD_WET_M = C(0.108, 0.087, 0.063); MUD_DRY = C(0.205, 0.172, 0.132)
    FILM_WET = C(0.115, 0.097, 0.077); FILM_DRY = C(0.19, 0.168, 0.14)

    def grime(base, r, m, sel, k_solid=1.0, k_splat=1.0, k_film=1.0, k_streak=1.0, cw=None):
        """Film, streaks, tear marks, drips, splats and caked mud on top of (base, r, m) for selection sel (in place)."""
        b = base[sel]; rr = r[sel]; mm = m[sel]
        # film (wet dirt haze)
        fd = film[sel] * 0.50 * k_film
        fcol = mixc(FILM_WET, FILM_DRY, dry[sel] * 0.6)
        b = mixc(b, fcol, fd); rr = rr + (0.62 - rr) * fd * 0.6; mm = mm * (1 - fd)
        # generic vertical streaks + tear marks from window corners / lamps / handles
        st = np.clip(streak[sel] * 0.5 + tears[sel] * 0.85, 0, 1) * k_streak
        b = mixc(b, b * C(0.55, 0.52, 0.50) + C(0.030, 0.025, 0.019), st); rr = rr + (0.55 - rr) * st * 0.35
        # mud
        md = np.clip(np.maximum(solid[sel] * k_solid * (1 - wash[sel]), splat[sel] * k_splat) + drip[sel] * 0.55, 0, 1)
        dr = dry[sel]
        mcol = mixc(mixc(MUD_WET_D, MUD_WET_M, mvar[sel]), MUD_DRY, dr)
        b = mixc(b, mcol, md)
        mr = 0.34 + 0.12 * n3[sel] + 0.5 * dr
        rr = rr + (mr - rr) * md; mm = mm * (1 - md)
        height[sel] += md * (thick[sel] * (0.0018 + 0.0012 * n3[sel]) + 0.0005 * vnoise(P[sel] * 120, 18)) + splat_b[sel] * 0.0005 * k_splat
        base[sel] = b; r[sel] = np.clip(rr, 0.03, 1); m[sel] = mm
        if cw is not None: cw[sel] = cw[sel] * (1 - 0.55 * md * (0.4 + 0.6 * dr))

    # ---- paint (single-stage red, faded; the clearcoat layer stands for the rain water film) ----
    s = is_('paint')
    if s.any():
        # (v2) wet single-stage red: rain darkens and saturates oxidised paint, so the chalky bloom is only a hint on
        # the horizontals (it read as pale pink plastic in-engine under the sky-only env map)
        deep = C(0.078, 0.0066, 0.0054); faded = C(0.100, 0.0135, 0.0105); chalk = C(0.118, 0.026, 0.021)
        fade_side = np.clip(0.28 + 0.35 * n1[s] + 0.20 * sstep(0.7, 1.5, z[s]), 0, 1)
        b = mixc(deep, faded, fade_side)
        chalk_k = np.clip(up[s] * (0.18 + 0.28 * n1[s]) + 0.10 * sstep(0.2, 0.6, n2[s]) * up[s], 0, 1)
        b = mixc(b, chalk, chalk_k)
        b = b * (0.94 + 0.12 * vnoise(P[s] * 7, 70))[:, None]          # panel-to-panel respray / fade variation
        r = 0.26 + 0.20 * up[s] + 0.06 * n2[s]
        mt = np.zeros(s.sum(), np.float32)
        sc = np.clip(scratch[s] * 0.9 + chips[s], 0, 1)
        b = mixc(b, C(0.14, 0.13, 0.12), sc * 0.8); r = r + (0.6 - r) * sc
        bare = chips[s] * sstep(0.2, 0.7, n3[s])
        b = mixc(b, C(0.42, 0.41, 0.40), bare); mt = np.maximum(mt, bare); r = r + (0.35 - r) * bare
        rs = rust[s]; rh = rust_halo[s]
        b = mixc(b, b * C(0.72, 0.58, 0.50), rh)
        b = mixc(b, b * C(0.8, 0.7, 0.62), blister[s] * 0.5)
        rcol = mixc(C(0.048, 0.017, 0.008), C(0.19, 0.064, 0.021), sstep(-0.4, 0.5, n3[s]))
        b = mixc(b, rcol, rs); r = r + (0.85 - r) * rs; mt = mt * (1 - rs)
        height[s] += rs * (0.0006 + 0.0008 * vnoise(P[s] * 160, 19)) + rh * 0.0002 + blister[s] * 0.0005 - sc * 0.00025 - chips[s] * 0.0003 \
            + 0.0006 * fbm(P[s], (2.2, 2.2, 2.2), 3, 20) + 0.0002 * vnoise(P[s] * 18, 27)
        # (v4) dents: a round parking dent in the driver-side rear quarter, a rock hit on the passenger front wing and a
        # bumper tap in the tailgate corner (5 / 3 / 4 mm deep, 6-10 cm: they read in the sky reflection), with a
        # crease line and cracked paint at the deepest point
        dent = np.zeros(s.sum(), np.float32); dcore = np.zeros(s.sum(), np.float32)
        for (dl_, df_, dh_, rad, dep, sidek) in ((0.80, -0.98, 0.78, 0.10, 0.011, 'L'), (-0.80, 1.36, 0.86, 0.07, 0.007, 'R'),
                                                 (0.42, -1.77, 0.64, 0.08, 0.008, 'B')):
            if sidek == 'B':
                d2 = ((x[s] - dl_) ** 2 + (z[s] - dh_) ** 2) / rad ** 2; msk = rearf[s]
            else:
                d2 = ((f[s] - df_) ** 2 * 0.7 + (z[s] - dh_) ** 2) / rad ** 2; msk = side[s] * (np.sign(x[s]) == np.sign(dl_))
            g = np.exp(-d2 * 1.6) * msk
            crease = np.exp(-(((z[s] - dh_ - 0.25 * (f[s] - df_)) / (rad * 0.08)) ** 2)) * np.exp(-d2 * 0.8) * msk * (sidek != 'B')
            dent += dep * (g + 0.25 * crease); dcore = np.maximum(dcore, np.exp(-d2 * 9) * msk)
        crk = dcore * sstep(0.35, 0.7, vnoise(P[s] * 900, 45) * 0.5 + 0.5)
        b = mixc(b, C(0.13, 0.12, 0.11), crk * 0.6); r = r + (0.6 - r) * crk
        height[s] -= dent
        alb[s] = b; rough[s] = r; metal[s] = mt
        coat[s] = np.clip(1.0 - rs * 0.95 - 0.4 * sc - 0.5 * crk, 0, 1)
        # horizontals: the rain film is broken into sheeted (mirror) and beaded/drying (satin) patches, so the sky
        # reflection on the hood/roof is mottled instead of one uniform grey sheet
        sheet = sstep(0.38, 0.62, fbm(P[s], (4.5, 4.5, 4.5), 4, 71) * 0.5 + 0.5 + 0.15 * n2[s])
        coat[s] = coat[s] * (1 - up[s] * (0.25 + 0.30 * (1 - sheet)))
        coat_r[s] = 0.03 + 0.05 * chalk_k + up[s] * (0.03 + 0.20 * (1 - sheet)) + 0.08 * film[s]
        # upper body is rinsed by the rain: the haze film belongs on the lower half and the tailgate wake
        grime(alb, rough, metal, s, k_film=0.45 + 0.55 * sstep(1.25, 0.6, z[s]) + 0.35 * rearf[s], cw=coat)
    s = is_('paint_inner')
    if s.any():
        alb[s] = C(0.09, 0.011, 0.009) * (0.75 + 0.25 * AO[s])[:, None]; rough[s] = 0.5
        grime(alb, rough, metal, s, 0.5, 0.5, 1.2, 0.3)
    # ---- black plastic (flares, rockers, grille panel, mirror housings, cowl) ----
    s = is_('plastic')
    if s.any():
        b = mixc(C(0.016, 0.016, 0.016), C(0.050, 0.050, 0.048), np.clip(up[s] * 0.8 + 0.25 * n1[s] + 0.1, 0, 1))
        r = 0.46 + 0.12 * up[s] + 0.06 * n2[s]
        sc = np.clip(scratch[s] * 0.6 + chips[s] * 0.5, 0, 1)
        b = mixc(b, C(0.09, 0.09, 0.085), sc)
        alb[s] = b; rough[s] = r; metal[s] = 0
        height[s] += 0.00015 * vnoise(P[s] * 180, 12) + 0.0001 * vnoise(P[s] * 420, 28) - sc * 0.0002
        coat[s] = 0.35; coat_r[s] = 0.12
        grime(alb, rough, metal, s, 0.8, 1.0, 1.1, 0.8, cw=coat)
    s = is_('rubber_trim')
    if s.any():
        alb[s] = mixc(C(0.011, 0.011, 0.011), C(0.03, 0.03, 0.028), np.clip(up[s] * 0.6 + 0.2 * n2[s], 0, 1)); rough[s] = 0.55 + 0.1 * n2[s]
        height[s] += 0.00004 * vnoise(P[s] * 1200, 13)
        grime(alb, rough, metal, s, 0.6, 0.8, 0.8, 0.3)
    s = is_('steel_black')
    if s.any():
        b = np.tile(C(0.014, 0.014, 0.014), (s.sum(), 1)); r = 0.36 + 0.1 * n2[s]; mt = np.zeros(s.sum(), np.float32)
        ch = np.clip(chips[s] * 1.5 + EDGE[s] * 0.3 * sstep(0.2, 0.6, n3[s]), 0, 1)
        rs = sstep(0.55, 0.75, EDGE[s] * 0.6 + 0.5 * n2[s] + 0.25 * n3[s] + ch * 0.4)
        b = mixc(b, C(0.18, 0.17, 0.16), ch * (1 - rs)); r = r + (0.5 - r) * ch
        b = mixc(b, mixc(C(0.05, 0.02, 0.009), C(0.17, 0.06, 0.02), sstep(-0.4, 0.5, n3[s])), rs); r = r + (0.85 - r) * rs
        alb[s] = b; rough[s] = r; metal[s] = mt
        height[s] += rs * 0.0008 * (0.5 + 0.5 * vnoise(P[s] * 150, 29)) - ch * 0.0003
        coat[s] = 0.5 * (1 - rs); coat_r[s] = 0.1
        # bumpers: caked underneath, splats + film on the faces so the black reads through
        grime(alb, rough, metal, s, 0.35 + 0.65 * down[s] + 0.0, 1.0, 1.0, 0.6, cw=coat)
    s = is_('chrome')
    if s.any():
        pit = sstep(0.45, 0.8, vnoise(P[s] * 260, 14) * 0.6 + n2[s] * 0.5)
        alb[s] = mixc(C(0.72, 0.72, 0.70), C(0.35, 0.30, 0.26), pit * 0.6); metal[s] = 1.0 - 0.5 * pit
        rough[s] = 0.07 + 0.35 * pit
        rs = sstep(0.7, 0.85, EDGE[s] * 0.5 + n2[s] * 0.5 + n3[s] * 0.3)
        alb[s] = mixc(alb[s], C(0.12, 0.045, 0.015), rs); metal[s] *= (1 - rs); rough[s] = rough[s] + (0.85 - rough[s]) * rs
        height[s] += pit * 0.00008
        grime(alb, rough, metal, s, 0.6, 0.8, 0.7, 0.4)
        # mirror glass (side mirrors + rear-view mirror): silvered glass behind a slightly dirty, wet front face.
        # The in-game env is sky-only, so a full-strength mirror reads as a flat white card: keep it dim.
        mir_side = s & (np.abs(x) > 0.85) & (np.abs(f - 0.539) < 0.03) & (N[:, 1] > 0.6)
        mir_rvm = s & (np.abs(x) < 0.23) & (np.abs(f - 0.2265) < 0.02) & (np.abs(z - 1.522) < 0.07) & (N[:, 1] > 0.6)
        for msel, lvl in ((mir_side, 0.20), (mir_rvm, 0.16)):
            if msel.any():
                spots = sstep(0.55, 0.8, vnoise(P[msel] * 180, 61) * 0.5 + 0.5) * 0.5
                alb[msel] = mixc(np.tile(C(lvl, lvl, lvl * 1.02), (msel.sum(), 1)), C(0.10, 0.09, 0.08), spots * 0.5)
                metal[msel] = 1.0; rough[msel] = 0.05 + 0.18 * spots
    s = is_('underbody')
    if s.any():
        b = mixc(C(0.030, 0.022, 0.016), C(0.12, 0.045, 0.016), sstep(0.1, 0.6, n2[s] + 0.3 * n1[s]))
        alb[s] = b; rough[s] = 0.75 + 0.1 * n3[s]; metal[s] = 0.1
        height[s] += 0.0008 * vnoise(P[s] * 90, 30)
        loc_mud = np.clip(0.55 + 0.35 * n1[s] + 0.25 * down[s], 0, 1)
        mcol = mixc(mixc(MUD_WET_D, MUD_WET_M, mvar[s]), MUD_DRY, dry[s] * 0.8)
        alb[s] = mixc(alb[s], mcol, loc_mud * 0.9); rough[s] = rough[s] * (1 - loc_mud) + (0.4 + 0.45 * dry[s]) * loc_mud; metal[s] *= (1 - loc_mud)
        height[s] += loc_mud * 0.0012 * (0.5 + 0.5 * n3[s])
    s = is_('liner')
    if s.any():
        alb[s] = C(0.018, 0.017, 0.016); rough[s] = 0.7
        lm = np.clip(0.65 + 0.35 * n1[s] + 0.3 * (1 - AO[s]), 0, 1)
        alb[s] = mixc(alb[s], mixc(mixc(MUD_WET_D, MUD_WET_M, mvar[s]), MUD_DRY, dry[s] * 0.5), lm); rough[s] = 0.7 - 0.3 * lm
        height[s] += lm * 0.0012 * n3[s]
    # ---- steel wheels: painted silver-grey, heavy brake dust, mud packed into the dish, rusty edges ----
    s = is_('wheel')
    if s.any():
        r_ax = np.sqrt((y[s] - (-FAX)) ** 2 + (z[s] - R_TIRE) ** 2)
        b = np.tile(C(0.30, 0.30, 0.285), (s.sum(), 1)); r = 0.40 + 0.08 * n2[s]; mt = np.full(s.sum(), 0.3, np.float32)
        bdust = np.clip(0.30 + 0.30 * n1[s] + 0.45 * (1 - CAV[s]) + 0.25 * (1 - AO[s]) + sstep(0.13, 0.19, r_ax) * 0.25, 0, 1)
        b = mixc(b, C(0.075, 0.058, 0.042), bdust * 0.70); r = r + (0.62 - r) * bdust * 0.7; mt *= (1 - 0.8 * bdust)
        rs = sstep(0.55, 0.75, EDGE[s] * 0.6 + (1 - CAV[s]) * 0.4 + 0.4 * n3[s] + 0.25 * n2[s])
        b = mixc(b, mixc(C(0.06, 0.022, 0.01), C(0.19, 0.07, 0.025), sstep(-0.4, 0.5, n3[s])), rs); r = r + (0.85 - r) * rs; mt *= (1 - rs)
        # mud: packed in the barrel/dish near the rim and in recesses, splashes on the face (rotationally symmetric)
        wm = np.clip(sstep(0.15, 0.20, r_ax) * (0.5 + 0.5 * sstep(-0.2, 0.3, n1[s])) + (1 - AO[s]) * 0.5 + 0.45 * splat_w[s], 0, 1)
        mcol = mixc(mixc(MUD_WET_D, MUD_WET_M, mvar[s]), MUD_DRY, dry[s] * 0.5)
        b = mixc(b, mcol, wm * 0.85); r = r + (0.45 - r) * wm
        alb[s] = b; rough[s] = r; metal[s] = mt
        height[s] += rs * 0.0003 * n3[s] + wm * 0.0006 * n3[s]
    s = is_('brake')
    if s.any():
        rs = np.clip(0.55 + 0.4 * n2[s], 0, 1)
        alb[s] = mixc(C(0.30, 0.29, 0.28), C(0.13, 0.055, 0.02), rs); metal[s] = 1 - rs; rough[s] = 0.4 + 0.45 * rs
    # ---- lamps ----
    s = is_('headlight')
    if s.any():
        dirt = np.clip(sstep(0.84, 0.70, z[s]) * 0.5 + 0.25 * n2[s] + splat_s[s] * 0.5, 0, 1)
        alb[s] = mixc(C(0.85, 0.86, 0.85), C(0.20, 0.17, 0.13), dirt * 0.55); rough[s] = 0.04 + 0.30 * dirt
        lx = x[s] - np.sign(x[s]) * 0.565
        height[s] += 0.0007 * np.sin(lx * 2 * math.pi / 0.011) * (1 - sstep(0.03, 0.05, np.sqrt(lx ** 2 + (z[s] - 0.785) ** 2))) \
            + 0.0005 * np.sin(lx * 2 * math.pi / 0.006) * np.sin((z[s] - 0.785) * 2 * math.pi / 0.012)
    for nm, colr, hi in (('brakelight', C(0.105, 0.0035, 0.003), C(0.30, 0.012, 0.008)), ('indicator', C(0.24, 0.062, 0.003), C(0.50, 0.17, 0.01))):
        s = is_(nm)
        if s.any():
            # lens optics: vertical flutes + cat-eye grid; strong in the normal map, subtle in albedo
            k = 2 * math.pi / 0.011
            cell = np.abs(np.sin(x[s] * k)) * np.abs(np.sin(z[s] * k)) + np.abs(np.sin(y[s] * k)) * np.abs(np.sin(z[s] * k))
            flute = 0.5 + 0.5 * np.sin((x[s] + y[s]) * 2 * math.pi / 0.006)
            b = mixc(colr, hi, np.clip(cell * 0.35 + 0.15 * flute, 0, 1) * (1 - AO[s] * 0.2))
            dirt = np.clip(low[s] * 0.35 + 0.25 * n2[s] + rearf[s] * 0.2 + splat_s[s] * 0.5, 0, 1)
            alb[s] = mixc(b, C(0.08, 0.065, 0.05), dirt * 0.45); rough[s] = 0.05 + 0.3 * dirt
            coat[s] = 1.0 - 0.6 * dirt; coat_r[s] = 0.04
            height[s] += 0.0006 * cell + 0.00035 * flute
    # ---- (v2) lamp emission map: lit lamps show a bulb hotspot, the reflector bowl and the lens optics instead of a flat
    # glowing disc/block (vehicle.js only scales emissiveIntensity, so the pattern and colour live in this map) ----
    emit = np.zeros((n, 3), np.float32)
    s = is_('headlight')
    if s.any():
        lx = x[s] - np.sign(x[s]) * 0.565; lz = z[s] - 0.785
        rr_ = np.sqrt(lx ** 2 + lz ** 2)
        flute = 0.5 + 0.5 * np.sin(lx * 2 * math.pi / 0.011)
        k = np.exp(-(rr_ / 0.020) ** 2) * 1.0 + np.exp(-(rr_ / 0.050) ** 2) * 0.35 + 0.07
        k = k * (0.82 + 0.18 * flute) * (1 - 0.75 * sstep(0.062, 0.088, rr_))
        k = k * (1 - 0.35 * np.clip(sstep(0.84, 0.70, z[s]) * 0.5 + splat_s[s] * 0.5, 0, 1))    # dirt on the lens dims it
        emit[s] = C(1.0, 0.90, 0.74)[None, :] * np.clip(k, 0, 1)[:, None]
    for nm, ecol in (('brakelight', C(1.0, 0.030, 0.012)), ('indicator', C(1.0, 0.40, 0.035))):
        s = is_(nm)
        if s.any():
            kk = 2 * math.pi / 0.011
            cell = np.abs(np.sin(x[s] * kk)) * np.abs(np.sin(z[s] * kk)) + np.abs(np.sin(y[s] * kk)) * np.abs(np.sin(z[s] * kk))
            k = (0.30 + 0.70 * np.clip(cell, 0, 1)) * (0.45 + 0.55 * AO[s]) * (0.7 + 0.3 * sstep(-0.3, 0.3, n2[s]))
            emit[s] = ecol[None, :] * np.clip(k, 0, 1)[:, None]
    # ---- printed parts (sampled with a mip level matching the atlas texel footprint) ----
    s = is_('plate')
    if s.any():
        ch = mip_chain(plate_img); lod = src_lod(G, MID['plate'], plate_img.shape[1], plate_img.shape[0])
        tex = sample_lod(ch, su[s], sv[s], lod)[:, :3]
        lin = np.power(np.clip(tex, 0, 1), 2.2)
        dirt = np.clip(sstep(0.62, 0.4, sv[s] * 0.11 + 0.40) * 0.2 + 0.35 * n2[s] + 0.25 * (1 - AO[s]) + 0.2 * rearf[s], 0, 1)
        alb[s] = mixc(lin * 0.82, C(0.14, 0.115, 0.09), dirt * 0.5); rough[s] = 0.3 + 0.35 * dirt
        ink = 1 - np.clip(tex.mean(1) * 1.4, 0, 1)
        height[s] += ink * 0.0006
        grime(alb, rough, metal, s, 0.3, 0.6, 0.5, 0.3)
    s = is_('cover')
    if s.any():
        ch = mip_chain(cover_img); lod = src_lod(G, MID['cover'], cover_img.shape[1], cover_img.shape[0])
        log(f'  cover print lod {lod:.2f}')
        tex = sample_lod(ch, su[s], sv[s], lod)[:, 0]
        b = mixc(C(0.016, 0.016, 0.017), C(0.045, 0.045, 0.043), np.clip(up[s] * 0.7 + 0.2 * n1[s] + 0.2, 0, 1))
        pr = tex * (0.80 + 0.20 * sstep(-0.3, 0.3, n3[s]))
        b = mixc(b, C(0.33, 0.33, 0.31), pr)
        alb[s] = b; rough[s] = 0.42 + 0.1 * n2[s] - 0.1 * pr
        height[s] += 0.0003 * fbm(P[s], (60, 60, 60), 2, 15) + pr * 0.0001
        coat[s] = 0.4; coat_r[s] = 0.15
        grime(alb, rough, metal, s, 0.7, 1.0, 1.0, 1.0, cw=coat)
    # ---- glass: dirt film mostly as roughness (blurred transmission), wiper-swept arcs, dirty rear/sides ----
    s = is_('glass')
    if s.any():
        gx, gz = x[s], z[s]
        # (v2) rain keeps the glass mostly clear; grime lives along the bottom edge, the corners and the rear window
        filmg = np.clip(0.04 + 0.10 * n1[s] + 0.34 * sstep(1.20, 1.07, gz) + 0.30 * rearf[s], 0, 1)
        clean = np.zeros(s.sum(), np.float32)
        ws = (N[s, 1] < -0.4) & (N[s, 2] > 0.25)          # windshield (faces forward/up)
        for pl, reach in ((0.34, 0.50), (-0.20, 0.47)):
            du = gx - pl; dv = (gz - 1.07) * 1.278
            d = np.sqrt(du ** 2 + dv ** 2); ang = np.degrees(np.arctan2(dv, du)) % 360
            arc = sstep(0.09, 0.12, d) * sstep(reach + 0.01, reach - 0.02, d) * sstep(52, 62, ang) * sstep(186, 178, ang)
            edge = np.exp(-((d - reach) / 0.012) ** 2) * sstep(52, 62, ang) * sstep(186, 178, ang)
            clean = np.maximum(clean, arc * ws)
            filmg = np.clip(filmg + edge * ws * 0.6, 0, 1)
        rw = N[s, 1] > 0.6
        du = -gx; dv = gz - 1.582
        d = np.sqrt(du ** 2 + dv ** 2); ang = np.degrees(np.arctan2(dv, du)) % 360
        rarc = sstep(0.07, 0.10, d) * sstep(0.47, 0.44, d) * sstep(178, 186, ang) * sstep(335, 325, ang)
        clean = np.maximum(clean, rarc * rw)
        filmg = np.clip(filmg + rw * 0.35, 0, 1)
        dirt = filmg * (1 - 0.88 * clean)
        ws_k = np.where(ws, 0.12, 0.45).astype(np.float32)
        dirt = np.clip(dirt + streak[s] * 0.20 * (1 - clean) + splat_s[s] * ws_k * sstep(1.35, 1.1, gz), 0, 1)
        alb[s] = mixc(C(0.90, 0.92, 0.91), C(0.50, 0.45, 0.38), dirt * 0.45)
        rough[s] = 0.02 + 0.30 * dirt ** 1.4; metal[s] = 0.0
        AO[s] = 1.0; CAV[s] = 1.0
    # cavity darkening (dirt in crevices) everywhere but glass
    cav_d = sstep(0.95, 0.35, CAV)
    alb *= (1 - 0.40 * cav_d)[:, None]
    # ---- write images ----
    def to_img(vals, ch):
        im = np.zeros((size, size, ch), np.float32)
        im[idx] = vals.reshape(n, ch)
        return im
    albedo = to_img(alb, 3)
    orm = to_img(np.stack([np.clip(AO * (0.55 + 0.45 * CAV) * 0.9 + 0.1, 0, 1), np.clip(rough, 0.03, 1), np.clip(metal, 0, 1)], 1), 3)
    coat_im = to_img(np.stack([np.clip(coat, 0, 1), np.clip(coat_r, 0.02, 1), np.zeros(n, np.float32)], 1), 3)
    H = np.zeros((size, size), np.float32); H[idx] = height
    nrm = height_to_normal(H, G['P'], cov, 1.0)
    global EXT_EMIT
    EXT_EMIT = to_img(emit, 3)
    return albedo, orm, coat_im, nrm, cov

# ---------------- (v4) cabin light: analytic window visibility ----------------
WV_ROOF, WV_FLOOR, WV_REAR = 1.596, 0.52, -1.755
WV_C = 0.75 + 1.07 / K_WS - 0.008       # windshield inner plane: f + h/K_WS = WV_C
WV_BSL = (0.776 - 0.712) / (1.635 - 1.045)

def window_vis(P, N, nsamp=64, ground=0.14, seed=5):
    """Fraction of the (overcast-sky weighted) cosine irradiance at each cabin texel that arrives through a window.
    The cabin is the convex volume floor/roof/rear wall/windshield plane/tumblehome sides; a ray escapes if its exit
    point lies inside the windshield, door-glass, quarter-glass or rear-glass outline. Sky radiance follows the CIE
    overcast law (1 + 2 cos zenith) / 3; below the horizon the (wet road / forest) ground returns ~14 %.
    P, N: (n,3) car coords (l, f, h). Local blockers (seats, wheel) come from the baked AO."""
    n = len(P)
    h0 = np.clip(P[:, 2], WV_FLOOR + 0.005, WV_ROOF - 0.005)
    f0 = np.clip(P[:, 1], WV_REAR + 0.005, None)
    f0 = np.minimum(f0, WV_C - h0 / K_WS - 0.005)
    lim = 0.776 - 0.009 - WV_BSL * (h0 - 1.045) - 0.005
    l0 = np.clip(P[:, 0], -lim, lim)
    Nn = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-6)
    a = np.where(np.abs(Nn[:, 2:3]) < 0.9, np.array([[0, 0, 1]], np.float32), np.array([[1, 0, 0]], np.float32))
    T = np.cross(a, Nn); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-6)
    B = np.cross(Nn, T)
    rng = np.random.RandomState(seed)
    k = int(math.sqrt(nsamp))
    num = np.zeros(n, np.float32); den = np.zeros(n, np.float32)
    BIG = np.float32(1e9)
    cS = 0.776 - 0.009 + WV_BSL * 1.045
    for i in range(k):
        for j in range(k):
            u1 = (i + rng.rand()) / k; u2 = (j + rng.rand()) / k
            r = math.sqrt(u1); ph = 2 * math.pi * u2
            d = T * (r * math.cos(ph)) + B * (r * math.sin(ph)) + Nn * math.sqrt(max(0.0, 1 - u1))
            dl, df, dh = d[:, 0], d[:, 1], d[:, 2]
            Lw = np.where(dh > 0, (1 + 2 * np.clip(dh, 0, 1)) / 3, ground).astype(np.float32)
            den += Lw
            ts = np.stack([
                np.where(dh > 1e-5, (WV_ROOF - h0) / np.maximum(dh, 1e-5), BIG),
                np.where(dh < -1e-5, (WV_FLOOR - h0) / np.minimum(dh, -1e-5), BIG),
                np.where(df < -1e-5, (WV_REAR - f0) / np.minimum(df, -1e-5), BIG),
                np.where(df + dh / K_WS > 1e-5, (WV_C - (f0 + h0 / K_WS)) / np.maximum(df + dh / K_WS, 1e-5), BIG),
                np.where(dl + WV_BSL * dh > 1e-5, (cS - (l0 + WV_BSL * h0)) / np.maximum(dl + WV_BSL * dh, 1e-5), BIG),
                np.where(-dl + WV_BSL * dh > 1e-5, (cS - (-l0 + WV_BSL * h0)) / np.maximum(-dl + WV_BSL * dh, 1e-5), BIG)], 0)
            face = np.argmin(ts, 0); t = np.min(ts, 0); del ts
            el, ef, eh = l0 + dl * t, f0 + df * t, h0 + dh * t
            wsw = 0.625 + (0.585 - 0.625) * (eh - 1.10) / 0.485 - 0.012
            esc = (face == 3) & (eh > 1.112) & (eh < 1.573) & (np.abs(el) < wsw)
            esc |= (face == 2) & (np.abs(el) < 0.588) & (eh > 1.152) & (eh < 1.543)
            dff = np.where(eh <= BELT, DOOR_F0, DOOR_F0 - (eh - BELT) / K_WS) - 0.046
            side = (face == 4) | (face == 5)
            esc |= side & (eh > 1.107) & (eh < 1.556) & (((ef < dff) & (ef > -0.398)) | ((ef < -0.542) & (ef > -1.616)))
            num += np.where(esc, Lw, 0)
    return (num / np.maximum(den, 1e-6)).astype(np.float32)

GAITERS = [((0.02, 0.18, 0.664), (0.0245, 0.152, 0.805), 0.075), ((-0.06, 0.03, 0.664), (-0.06, 0.014, 0.752), 0.05),
           ((0.0, -0.075, 0.664), (0.0, -0.085, 0.705), 0.05)]

def _axis_frame(p0, p1, P):
    """Height t along the axis p0->p1 and angle around it for car points P."""
    a = np.asarray(p1, np.float32) - np.asarray(p0, np.float32); L = np.linalg.norm(a); a /= L
    u = np.array([1, 0, 0], np.float32) - a * a[0]; u /= np.linalg.norm(u); v = np.cross(a, u)
    d = P - np.asarray(p0, np.float32)
    t = d @ a; x = d @ u; y = d @ v
    return t / L, np.arctan2(y, x), np.sqrt(x * x + y * y)

def compose_int(G, size):
    """(v4) Interior textures. ORM: R = occlusion (analytic window visibility x baked local AO x cavity: the cabin is
    lit only through its glass), G = roughness, B = MATERIAL CLASS for the engine's micro-detail shader (0 none,
    0.25 pebble-grain plastic, 0.5 fabric weave, 0.75 leather/rubber grain); vehicle.js forces metalness 0 for the
    interior material. Albedos follow measured car-interior values: black PP/ABS 0.02-0.04, grey vinyl 0.05-0.08,
    velour 0.03-0.06, headliner 0.25, dust film pushes horizontals to ~0.07-0.10 with roughness ~0.9."""
    log('compose int (v4)')
    cov = G['P'][..., 3] > 0.5
    idx = np.nonzero(cov)
    P = G['P'][idx][:, :3].astype(np.float32)
    N = G['N'][idx][:, :3].astype(np.float32); N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-6)
    mid = np.rint(G['M'][idx][:, 0] * 32).astype(np.int32)
    AO = np.clip(G['A'][idx][:, 0], 0, 1); CAV = np.clip(G['A'][idx][:, 1], 0, 1); EDGE = np.clip(G['A'][idx][:, 2] * 6.0, 0, 1)
    n = len(P); x, y, z = P[:, 0], P[:, 1], P[:, 2]; f = -y
    Pc = np.stack([x, f, z], 1).astype(np.float32); Nc = np.stack([N[:, 0], -N[:, 1], N[:, 2]], 1).astype(np.float32)
    is_ = lambda name: mid == MID[name]
    alb = np.zeros((n, 3), np.float32); rough = np.full(n, 0.6, np.float32); cls = np.zeros(n, np.float32); height = np.zeros(n, np.float32)
    up = sstep(0.35, 0.9, N[:, 2]); down = sstep(0.35, 0.9, -N[:, 2])
    n1 = fbm(P, (3, 3, 3), 3, 21); n2 = fbm(P, (14, 14, 14), 3, 22); n3 = fbm(P, (55, 55, 55), 2, 23)
    t0 = time.time()
    vis = window_vis(Pc, Nc, 64)
    log(f'  window visibility {time.time() - t0:.1f}s  median {np.median(vis):.3f}')
    # steering-wheel local frame (the wheel mesh was baked in place at SW_C, tilted by SW_ALPHA)
    swc = np.array([SW_C[0], SW_C[1], SW_C[2]], np.float32)
    ca, sa = math.cos(SW_ALPHA), math.sin(SW_ALPHA)
    d_ = Pc - swc
    # wheel axis (toward the driver) = (0, -cos, +sin); wheel-plane up = (0, sin, cos)
    sw_ax = d_[:, 1] * -ca + d_[:, 2] * sa
    sw_up = d_[:, 1] * sa + d_[:, 2] * ca
    sw_x = d_[:, 0]
    sw_r = np.sqrt(sw_x ** 2 + sw_up ** 2)
    sw_th = np.degrees(np.arctan2(sw_up, sw_x))            # 0 = driver's left (9 o'clock), 90 = top
    near_sw = (np.abs(sw_ax) < 0.07) & (sw_r < 0.23)

    # ------------------------------------------------ moulded plastics (dash, pod, trims, door cards)
    s = is_('interior')
    if s.any():
        xs, zs, fs = x[s], z[s], f[s]; ups = up[s]
        base = C(0.030, 0.027, 0.024)                                        # warm black PP (sRGB ~48)
        b = base * (0.93 + 0.14 * sstep(-0.4, 0.4, n1[s]))[:, None]
        doorcard = sstep(0.69, 0.72, np.abs(xs)) * sstep(1.12, 1.06, zs)
        b = mixc(b, C(0.042, 0.036, 0.030), doorcard)                         # brown-grey door vinyl
        insert = doorcard * sstep(-0.43, -0.38, fs) * sstep(0.66, 0.62, fs) * sstep(0.78, 0.80, zs) * sstep(0.93, 0.91, zs)
        weave_i = np.sin((fs + zs) * 2 * math.pi / 0.007) * np.sin((zs - fs) * 2 * math.pi / 0.007) * 0.5 + 0.5
        b = mixc(b, C(0.050, 0.048, 0.050) * (0.9 + 0.2 * weave_i)[:, None], insert)
        pillar = sstep(1.10, 1.16, zs) * sstep(0.56, 0.62, np.abs(xs))
        b = mixc(b, C(0.036, 0.035, 0.033), pillar)
        # dash top: sun-bleached, chalky; dust film thickest in the corner at the windshield base; cloth wipe arcs
        dash_top = ups * sstep(0.98, 1.02, zs) * sstep(0.40, 0.46, fs) * sstep(1.12, 1.08, zs)
        fade = dash_top * (0.35 + 0.35 * sstep(-0.3, 0.5, n1[s]))
        b = mixc(b, b * 1.35 + C(0.010, 0.010, 0.009), fade)
        wipe = sstep(0.25, 0.65, np.abs(np.sin((fs * 11 + 3.2 * np.sin(xs * 2.3) + 0.4 * n1[s]) * math.pi))) * sstep(0.45, 0.2, np.abs(xs - 0.25))
        corner = sstep(0.62, 0.72, fs) + 0.6 * sstep(0.55, 0.72, np.abs(xs))
        dust = np.clip(ups * (0.28 + 0.30 * sstep(-0.3, 0.4, n2[s]) + 0.35 * np.clip(corner, 0, 1)) * (1 - 0.55 * wipe * dash_top)
                       + (1 - CAV[s]) * 0.45 * ups + 0.10 * sstep(0.6, 0.9, 1 - AO[s]), 0, 1)
        dust *= sstep(0.25, 0.55, ups + 0.35 * dash_top) + 0.15
        DUST = C(0.105, 0.097, 0.086)
        b = mixc(b, DUST, dust * 0.72)
        # kick scuffs (lower door cards, dash knee panel), stress-whitened edges
        kick = sstep(0.56, 0.44, zs) * doorcard * sstep(-0.1, 0.5, fs) * sstep(0.1, 0.55, n2[s] + 0.3 * n3[s])
        knee = sstep(0.72, 0.80, zs) * sstep(0.90, 0.84, zs) * sstep(0.38, 0.40, fs) * sstep(0.46, 0.43, fs) * sstep(0.2, 0.6, n2[s] + 0.2)
        stress = sstep(0.55, 0.95, EDGE[s]) * sstep(0.0, 0.5, n3[s] + 0.25)
        scuff = np.clip(kick * 0.9 + knee * 0.6 + stress * 0.5, 0, 1)
        b = mixc(b, C(0.075, 0.071, 0.066), scuff * 0.7)
        # panel seams: upper/lower dash split, centre-stack sides, pod-to-dash joint shadow
        dface = sstep(0.36, 0.40, fs) * sstep(0.49, 0.45, fs) * sstep(0.5, 0.9, -Nc[s, 1])
        seam = dface * (np.exp(-((zs - 0.968) / 0.0016) ** 2) * sstep(0.215, 0.23, np.abs(xs))
                        + np.exp(-((np.abs(xs) - 0.215) / 0.0016) ** 2) * sstep(0.76, 0.78, zs) * sstep(1.0, 0.97, zs))
        seam += doorcard * np.exp(-((zs - 0.70) / 0.0018) ** 2) * sstep(-0.35, -0.30, fs)
        seam = np.clip(seam, 0, 1)
        b = b * (1 - 0.55 * seam)[:, None]
        # greasy touch zones (glovebox button/lid top, door pulls, radio surround): darker + glossier smudges
        touch = np.exp(-(((xs + 0.36) / 0.06) ** 2 + ((zs - 0.97) / 0.035) ** 2)) * sstep(0.40, 0.43, fs)
        touch += doorcard * np.exp(-(((fs - 0.25) / 0.09) ** 2 + ((zs - 0.94) / 0.05) ** 2))
        touch += doorcard * np.exp(-(((fs + 0.05) / 0.25) ** 2 + ((zs - 0.87) / 0.03) ** 2)) * 0.8      # armrest top
        touch = np.clip(touch * (0.6 + 0.4 * sstep(-0.2, 0.3, n3[s])), 0, 1)
        b = b * (1 - 0.18 * touch)[:, None]
        alb[s] = b
        r = 0.56 + 0.10 * n2[s] + 0.06 * n3[s]                          # grained PP: satin, mottled breakup
        r = r + (0.93 - r) * dust * 0.85 + fade * 0.08
        r = r + (0.75 - r) * scuff
        r = r - 0.24 * touch - 0.05 * pillar
        r = r + (0.9 - r) * insert
        rough[s] = r
        cls[s] = np.where(insert > 0.5, 0.5, 0.25)
        height[s] += -0.0009 * seam + 0.00025 * scuff * n3[s] + 0.0002 * weave_i * insert + 0.00012 * n3[s]

    # ------------------------------------------------ fabric: seats, headliner, visors
    s = is_('fabric')
    if s.any():
        xs, ys, zs = x[s], y[s], z[s]; fs = -ys
        k = 2 * math.pi / 0.006
        weave = np.sin((xs + ys) * k) * np.sin((zs - ys) * k) * 0.5 + 0.5 * vnoise(P[s] * 250, 33)
        dl = np.abs(np.abs(xs) - SEAT_L)
        seat_c = sstep(0.17, 0.14, dl) * sstep(0.62, 0.72, zs) * sstep(1.40, 1.30, zs)
        pipe = np.exp(-((dl - 0.155) / 0.006) ** 2) * sstep(0.62, 0.72, zs) * sstep(1.45, 1.35, zs)
        cush = sstep(0.80, 0.76, zs)
        pl_coord = np.where(cush > 0.5, -ys, zs)
        pleat = np.abs(np.sin(pl_coord * math.pi / 0.065))
        seam = sstep(0.12, 0.02, pleat)
        chk = sstep(-0.35, 0.35, np.sin(xs * 2 * math.pi / 0.018) * np.sin((zs - ys) * 2 * math.pi / 0.018))
        fleck = sstep(0.62, 0.8, vnoise(P[s] * 700, 34))
        centre = mixc(C(0.070, 0.068, 0.066), C(0.046, 0.045, 0.045), chk * 0.6)
        centre = mixc(centre, C(0.045, 0.052, 0.075), fleck * 0.35)
        centre = centre * (1 - 0.35 * seam)[:, None]
        b = mixc(C(0.030, 0.029, 0.029), centre, seat_c)
        b = mixc(b, C(0.016, 0.016, 0.016), pipe * 0.8)
        b = mixc(b, b * 1.3 + C(0.010, 0.009, 0.008), sstep(1.05, 1.25, zs) * seat_c * 0.5)
        slide = sstep(0.14, 0.26, xs - SEAT_L + 0.2) * sstep(0.30, 0.22, np.abs(xs - SEAT_L - 0.22)) * sstep(0.85, 0.72, zs) * (xs > 0)
        # lint / dust on the cushions, a dried coffee stain on the passenger cushion, pilled front edge
        lint = up[s] * sstep(0.2, 0.7, n2[s] + 0.3 * n3[s]) * 0.35 * (1 - sstep(1.5, 1.55, zs))
        stain = np.exp(-(((xs + 0.30) / 0.045) ** 2 + ((fs + 0.18) / 0.035) ** 2)) * cush * up[s]
        stain = np.clip(stain * (0.7 + 0.5 * n3[s]), 0, 1)
        hl = sstep(1.55, 1.59, zs)
        perf = 0.5 + 0.5 * vnoise(P[s] * 60, 62)
        hcol = C(0.23, 0.218, 0.19) * (0.93 + 0.07 * perf)[:, None]
        wstain = sstep(0.45, 0.62, np.abs(xs)) * sstep(0.1, 0.5, n1[s] + 0.3 * n2[s])
        hcol = mixc(hcol, C(0.17, 0.14, 0.095), wstain * 0.55)
        hcol = mixc(hcol, C(0.19, 0.16, 0.11), sstep(0.05, 0.25, fs) * 0.35)             # nicotine-yellowed front header
        hcol = hcol * (1 - 0.25 * sstep(0.10, 0.02, np.abs(fs + 0.60)) * sstep(0.2, 0.05, np.abs(xs)))[:, None]  # dome light soot
        b = mixc(b, hcol, hl)
        wear = sstep(0.3, 0.8, EDGE[s]) * sstep(-0.2, 0.4, n2[s])
        b = mixc(b, C(0.075, 0.07, 0.066), wear * 0.5 * (1 - hl))
        b = mixc(b, b * 1.35, slide * 0.6)
        b = mixc(b, C(0.10, 0.095, 0.088), lint * (1 - hl))
        b = mixc(b, C(0.050, 0.030, 0.016), stain * 0.6)
        b *= (0.88 + 0.12 * weave)[:, None]
        alb[s] = b
        rough[s] = 0.93 - 0.1 * wear - 0.22 * hl - 0.15 * slide - 0.1 * stain
        cls[s] = np.where(hl > 0.5, 0.0, 0.5)
        height[s] += (0.00025 * weave + 0.0006 * n2[s] * seat_c - 0.0016 * seam * seat_c + 0.0008 * pipe) * (1 - hl) + 0.00003 * perf * hl

    s = is_('carpet')
    if s.any():
        b = np.tile(C(0.026, 0.025, 0.024), (s.sum(), 1)) * (0.9 + 0.2 * vnoise(P[s] * 400, 35))[:, None]
        foot = np.exp(-(((x[s] - 0.36) / 0.25) ** 2 + ((f[s] - 0.45) / 0.3) ** 2)) + np.exp(-(((x[s] + 0.36) / 0.25) ** 2 + ((f[s] - 0.45) / 0.3) ** 2)) * 0.6
        m = np.clip(foot * (0.6 + 0.5 * n2[s]) + 0.25 * n1[s] + 0.2, 0, 1)
        b = mixc(b, mixc(C(0.045, 0.036, 0.027), C(0.11, 0.092, 0.07), sstep(-0.2, 0.3, n2[s])), m * 0.8)
        alb[s] = b; rough[s] = 0.9 - 0.25 * m * sstep(0.0, 0.4, n2[s]); height[s] += 0.0002 * vnoise(P[s] * 900, 25) + m * 0.0004 * n2[s]
        cls[s] = 0.5

    # ------------------------------------------------ hard black parts (switchgear, vents, column, spokes, pad, knobs)
    s = is_('int_black')
    if s.any():
        xs, zs, fs = x[s], z[s], f[s]
        b = mixc(C(0.017, 0.017, 0.017), C(0.030, 0.029, 0.027), np.clip(0.25 + 0.25 * n1[s] + up[s] * 0.3, 0, 1))
        r = 0.46 + 0.08 * n2[s] + 0.05 * n3[s]
        onw = near_sw[s]
        # horn pad: soft-touch, polished in the middle where it is thumped; dusty in the moulded valley at its rim
        pad = onw & (sw_r[s] < 0.10)
        polish = np.exp(-((sw_r[s] / 0.035) ** 2)) * pad
        r = np.where(pad, 0.62 - 0.25 * polish + 0.05 * n3[s], r)
        # knobs / stalk tips / switch rockers: rubbed shiny
        knob = np.exp(-(((xs - 0.03) ** 2 + (fs - 0.118) ** 2 + (zs - 0.945) ** 2) / 0.03 ** 2))
        knob += np.exp(-(((xs + 0.06) ** 2 + (fs + 0.002) ** 2 + (zs - 0.845) ** 2) / 0.02 ** 2))
        knob += sstep(0.12, 0.05, np.abs(np.abs(xs - SEAT_L) - 0.15)) * np.exp(-(((zs - 1.07) / 0.03) ** 2 + ((fs - 0.40) / 0.05) ** 2)) * 0.8
        knob = np.clip(knob, 0, 1)
        r = r - 0.28 * knob
        b = mixc(b, b * 1.25, knob * 0.5)
        dust = up[s] * sstep(0.1, 0.6, n2[s] + 0.25) * 0.45 * (1 - knob) + (1 - CAV[s]) * 0.35
        b = mixc(b, C(0.095, 0.088, 0.078), np.clip(dust, 0, 1) * 0.6)
        r = r + (0.9 - r) * np.clip(dust, 0, 1) * 0.7
        edge = sstep(0.6, 0.95, EDGE[s]) * sstep(-0.1, 0.4, n3[s])
        b = mixc(b, C(0.06, 0.058, 0.055), edge * 0.5)
        alb[s] = b; rough[s] = r
        cls[s] = np.where(knob > 0.5, 0.0, 0.25)
        height[s] += 0.00005 * vnoise(P[s] * 1500, 26) + 0.00012 * n3[s]

    # ------------------------------------------------ leather / polyurethane: wheel rim, gaiters, key head, fob, handbrake grip
    s = is_('leather')
    if s.any():
        xs, zs, fs = x[s], z[s], f[s]; Ps = Pc[s]
        b = np.tile(C(0.021, 0.0195, 0.0185), (s.sum(), 1)) * (0.94 + 0.12 * sstep(-0.4, 0.4, n2[s]))[:, None]
        r = 0.52 + 0.06 * n2[s] + 0.05 * n3[s]
        c_ = np.full(s.sum(), 0.75, np.float32)
        hgt = 0.00008 * n3[s]
        # steering-wheel rim: polished smooth + greyed where the hands sit (10-2 / 9-3 and the top), sun-cracked on top
        rim = near_sw[s] & (sw_r[s] > 0.16)
        th = sw_th[s]
        grip = np.exp(-(((th - 20) / 26) ** 2)) + np.exp(-(((np.abs(th) - 160) / 26) ** 2)) + 0.45 * np.exp(-(((th - 90) / 22) ** 2)) \
            + 0.35 * np.exp(-(((th + 45) / 30) ** 2)) + 0.35 * np.exp(-(((th + 135) / 30) ** 2))
        grip = np.clip(grip * (0.75 + 0.35 * sstep(-0.3, 0.4, n2[s])), 0, 1) * rim
        inner = sstep(0.0, -0.8, (sw_r[s] - 0.19) / 0.015) * rim                        # finger side of the rim
        grip = np.clip(grip * (0.7 + 0.5 * inner), 0, 1)
        w_, h1_, h2_ = worley(Ps[rim] if rim.any() else Ps[:1], 0.006, 91)
        crack = np.zeros(s.sum(), np.float32)
        if rim.any():
            top = sstep(0.2, 0.9, np.sin(np.radians(th[rim]))) * sstep(0.0, 0.6, (sw_ax[s][rim]) / 0.012 + 0.3)
            e = sstep(0.12, 0.03, np.abs(w_ - 0.5 * (h1_ + 0.3)))
            crack[rim] = e * top * sstep(0.1, 0.5, n2[s][rim] + 0.35)
        b = mixc(b, C(0.040, 0.036, 0.032), grip * 0.75)
        b = b * (1 - 0.45 * crack)[:, None]
        r = r - 0.30 * grip + 0.15 * crack
        hgt = hgt - 0.00035 * crack
        c_ = np.where(grip > 0.55, 0.0, c_)                                              # worn smooth: no grain
        # gaiters: fold valleys dusty + matt, crests rubbed glossy, 4 sewn seams with pale thread stitches
        for p0, p1, rad in GAITERS:
            tt, ang, rr_ = _axis_frame(p0, p1, Ps)
            g = (tt > -0.05) & (tt < 1.05) & (rr_ < rad) & ~near_sw[s]
            if not g.any(): continue
            seam_a = np.abs(((ang[g] - math.pi / 4) % (math.pi / 2)) - math.pi / 4)                   # angle to nearest seam
            seam_d = seam_a * rr_[g]
            groove = np.exp(-((seam_d / 0.0009) ** 2))
            stitch_line = np.exp(-((np.abs(seam_d) - 0.0028) / 0.0007) ** 2)
            along = tt[g] * np.linalg.norm(np.asarray(p1) - np.asarray(p0))
            dash = sstep(0.25, 0.45, np.abs(np.sin(along * math.pi / 0.0045)))
            stitch = stitch_line * dash
            cav = 1 - CAV[s][g]
            bg = b[g]
            bg = mixc(bg, C(0.085, 0.078, 0.068), np.clip(cav * 0.9 + 0.15 * up[s][g], 0, 1) * 0.55)       # dust in folds
            bg = bg * (1 - 0.5 * groove)[:, None]
            bg = mixc(bg, C(0.11, 0.105, 0.098), stitch * 0.85)
            b[g] = bg
            r[g] = np.clip(0.55 - 0.18 * sstep(0.3, 0.9, CAV[s][g]) + 0.3 * cav + 0.2 * stitch, 0.2, 0.95)
            hgt[g] += -0.0006 * groove + 0.0003 * stitch
        # key head: worn rubber (edges scuffed grey), fob: brown leather with a stitched border and darkened edges
        kc = np.array(KEY_LOCK) + np.array(KEY_AXIS) / np.linalg.norm(KEY_AXIS) * 0.029
        keyh = np.sqrt((xs - kc[0]) ** 2 + (fs - kc[1]) ** 2 + (zs - kc[2]) ** 2) < 0.03
        b = np.where(keyh[:, None], mixc(b, C(0.05, 0.048, 0.046), sstep(0.4, 0.9, EDGE[s]) * 0.6), b)
        fob = (np.sqrt((xs - kc[0] + 0.02) ** 2 + (fs - kc[1]) ** 2 + (zs - kc[2] + 0.06) ** 2) < 0.07) & ~keyh & ~(near_sw[s] & (sw_r[s] > 0.15))
        if fob.any():
            fb = mixc(C(0.070, 0.036, 0.018), C(0.040, 0.022, 0.012), sstep(0.3, 0.9, EDGE[s][fob]) + 0.3 * n3[s][fob])
            b[fob] = fb; r[fob] = 0.5 + 0.1 * n3[s][fob]
        # handbrake grip: polished on top
        hbg = (np.abs(xs) < 0.03) & (fs < -0.24) & (fs > -0.39) & (zs > 0.69)
        r = np.where(hbg, r - 0.2 * up[s], r)
        alb[s] = b; rough[s] = np.clip(r, 0.18, 0.97); cls[s] = c_; height[s] += hgt

    # ------------------------------------------------ paper: road map, fuel receipt, visor documents
    s = is_('paper')
    if s.any():
        xs, zs, fs = x[s], z[s], f[s]
        b = np.tile(C(0.60, 0.58, 0.51), (s.sum(), 1)); r = np.full(s.sum(), 0.78, np.float32)
        # road map (passenger dash): local coords
        mp = (zs < 1.2) & (xs < -0.2)
        if mp.any():
            yaw = math.radians(11); lx = (xs[mp] + 0.42) * math.cos(yaw) + (fs[mp] - 0.535) * math.sin(yaw)
            ly = -(xs[mp] + 0.42) * math.sin(yaw) + (fs[mp] - 0.535) * math.cos(yaw)
            Pm = np.stack([lx * 9, ly * 9, np.zeros_like(lx)], 1).astype(np.float32)
            land = fbm(Pm, (1, 1, 1), 4, 81)
            mb = mixc(C(0.56, 0.54, 0.44), C(0.30, 0.42, 0.24), sstep(0.05, 0.25, land))           # lowland / forest tint
            contour = np.exp(-((((land * 7) % 1.0) - 0.5) / 0.08) ** 2) * sstep(-0.1, 0.2, land)
            mb = mixc(mb, C(0.36, 0.22, 0.12), contour * 0.45)
            rd = fbm(Pm * 0.7 + 3.3, (1, 1, 1), 3, 82)
            road = np.exp(-((rd / 0.02) ** 2))
            mb = mixc(mb, C(0.62, 0.10, 0.06), road * 0.9)
            rd2 = fbm(Pm * 1.3 + 7.1, (1, 1, 1), 3, 83)
            mb = mixc(mb, C(0.70, 0.55, 0.10), np.exp(-((rd2 / 0.015) ** 2)) * 0.8)
            txt = sstep(0.55, 0.75, vnoise(np.stack([lx * 400, ly * 900, np.zeros_like(lx)], 1).astype(np.float32), 84)) * sstep(0.3, 0.6, fbm(Pm * 3, (1, 1, 1), 2, 85))
            mb = mixc(mb, C(0.08, 0.07, 0.06), txt * 0.6)
            fold = np.exp(-((np.abs(((lx / 0.27 + 0.5) * 3) % 1.0 - 0.5) - 0.5) / 0.02) ** 2) + np.exp(-((ly / 0.004) ** 2)) * 0.6
            mb = mb * (1 - 0.25 * np.clip(fold, 0, 1))[:, None]
            mb = mixc(mb, C(0.45, 0.40, 0.30), sstep(0.2, 0.7, n2[s][mp]) * 0.25)               # sun-yellowed, grubby
            b[mp] = mb; r[mp] = 0.72 + 0.08 * n3[s][mp]
        rc = (zs < 1.2) & (xs > 0.05)
        if rc.any():                                                                             # thermal receipt
            yaw = math.radians(-22); lx = (xs[rc] - 0.16) * math.cos(yaw) + (fs[rc] - 0.668) * math.sin(yaw)
            ly = -(xs[rc] - 0.16) * math.sin(yaw) + (fs[rc] - 0.668) * math.cos(yaw)
            rows = sstep(0.5, 0.8, np.abs(np.sin(ly * math.pi / 0.0035))) * sstep(0.022, 0.018, np.abs(lx)) * sstep(0.3, 0.6, vnoise(np.stack([lx * 300, ly * 40, lx * 0], 1).astype(np.float32), 86) + 0.3)
            b[rc] = mixc(C(0.64, 0.63, 0.60), C(0.25, 0.25, 0.26), rows * 0.6)
        vs = zs > 1.4
        if vs.any():                                                                             # green card + parking ticket
            big = vs & (xs > 0.285)
            b[big] = mixc(C(0.34, 0.40, 0.31), C(0.20, 0.25, 0.18), sstep(0.5, 0.8, np.abs(np.sin(fs[big] * math.pi / 0.004))) * 0.4)
            b[vs & ~big] = C(0.62, 0.60, 0.55)
        # dust + handling grime
        b = mixc(b, C(0.30, 0.28, 0.24), sstep(0.3, 0.9, 1 - CAV[s]) * 0.4)
        alb[s] = b; rough[s] = r; cls[s] = 0.0
        height[s] += 0.00006 * n3[s]

    # ------------------------------------------------ prayer beads (rosewood, hand-polished) + faded maroon tassel
    s = is_('charm')
    if s.any():
        xs, zs, fs = x[s], z[s], f[s]
        tas = zs < CHARM_PIV[2] - 0.124
        wood = fbm(np.stack([xs * 900, fs * 900, zs * 250], 1).astype(np.float32), (1, 1, 1), 3, 87)
        b = mixc(C(0.070, 0.022, 0.010), C(0.13, 0.050, 0.022), sstep(-0.4, 0.4, wood))
        b = b * (0.85 + 0.3 * vnoise(P[s] * 180, 88)[:, None])
        r = 0.30 + 0.12 * sstep(0.2, 0.8, 1 - CAV[s])
        if tas.any():
            ang = np.arctan2(fs[tas] - (CHARM_PIV[1] + 0.04), xs[tas] - (CHARM_PIV[0] + 0.004))
            strand = 0.5 + 0.5 * np.sin(ang * 60 + 3 * vnoise(np.stack([ang * 5, zs[tas] * 200, ang * 0], 1).astype(np.float32), 89))
            tb = mixc(C(0.060, 0.020, 0.020), C(0.110, 0.040, 0.034), strand)          # sun-faded maroon thread
            neck = zs[tas] > CHARM_PIV[2] - 0.134
            tb = np.where(neck[:, None], C(0.30, 0.21, 0.07), tb)
            b[tas] = tb; r[tas] = 0.85
        alb[s] = b; rough[s] = r; cls[s] = 0.0

    cav_d = sstep(0.95, 0.35, CAV)
    alb *= (1 - 0.35 * cav_d)[:, None]
    def to_img(vals, ch):
        im = np.zeros((size, size, ch), np.float32); im[idx] = vals.reshape(n, ch); return im
    albedo = to_img(alb, 3)
    # occlusion = light that can reach the texel through the glass x local AO x cavity. The cabin bounce the IBL can't
    # know about is added in the engine (vehicle.js cabin fill, not scaled by this map).
    occ = np.clip((0.03 + 0.97 * vis) * (0.55 + 0.45 * np.sqrt(AO)) * (0.72 + 0.28 * CAV), 0.02, 1.0)
    orm = to_img(np.stack([occ, np.clip(rough, 0.03, 1), cls], 1), 3)
    H = np.zeros((size, size), np.float32); H[idx] = height
    nrm = height_to_normal(H, G['P'], cov, 1.0)
    return albedo, orm, nrm

def tire_textures(size=1024):
    """Tire albedo/ORM/normal in tire UV space: u = around (TIRE_TILES tiles), v in [0,0.8] = profile, [0.82,0.98] = blocks."""
    log('tire textures')
    strip = tire_letter_strip()[..., 0]      # H x W, y = radius 0.19..0.36
    prof = tire_profile()
    L = [0.0]
    for a, b in zip(prof[:-1], prof[1:]): L.append(L[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    L = np.array(L) / L[-1] * 0.8
    R = np.array([p[0] for p in prof]); X = np.array([p[1] for p in prof])
    vv = (np.arange(size) + 0.5) / size
    uu = (np.arange(size) + 0.5) / size
    Hh = np.zeros((size, size), np.float32)
    alb = np.zeros((size, size, 3), np.float32); rough = np.zeros((size, size), np.float32)
    U, Vv = np.meshgrid(uu, vv)
    # profile rows
    rr = np.interp(vv, L, R); xx = np.interp(vv, L, X)
    is_prof = vv <= 0.8
    outer = xx > 0.07
    inner = xx < -0.07
    tread = (~outer) & (~inner)
    Pn = np.stack([U.ravel() * 7.0, Vv.ravel() * 3.0, np.zeros(size * size)], 1).astype(np.float32)
    nz = fbm(Pn, (1, 1, 1), 4, 31).reshape(size, size)
    nz2 = fbm(Pn, (18, 18, 1), 3, 32).reshape(size, size)
    base = C(0.022, 0.022, 0.022)
    alb[:] = base; rough[:] = 0.82
    # sidewalls: faded, brownish dust near bead and shoulder bottom
    for row in range(size):
        v = vv[row]
        if v > 0.8: continue
        r = rr[row]
        if outer[row] or inner[row]:
            fade = 0.25 + 0.2 * (r - 0.2) / 0.15
            alb[row] = mixc(base, C(0.040, 0.038, 0.035), fade)
            # lettering from strip (outer = reads correct; inner uses same)
            ys = (r - 0.19) / 0.17
            if 0 <= ys < 1:
                yi = min(int(ys * strip.shape[0]), strip.shape[0] - 1)
                xs = ((1 - uu) if outer[row] else uu) * strip.shape[1]
                txt = strip[yi, np.clip(xs.astype(int), 0, strip.shape[1] - 1)]
                Hh[row] += txt * 0.0007
            # molded rings texture
            Hh[row] += 0.00003 * nz2[row]
            rough[row] = 0.60
        else:
            alb[row] = C(0.042, 0.033, 0.024)   # groove bottoms (wet mud)
            rough[row] = 0.55
    # dirt: brown dust on lower sidewall (near shoulder) and bead area
    sidew = (Vv <= 0.8) & (np.abs(np.interp(Vv, L, X)) > 0.07)
    rv = np.interp(Vv, L, R)
    dirt = np.clip(sstep(0.30, 0.345, rv) * 0.7 + sstep(0.23, 0.2, rv) * 0.5 + 0.35 * nz, 0, 1) * sidew
    alb = mixc(alb.reshape(-1, 3), C(0.075, 0.060, 0.045), (dirt * 0.38).ravel()).reshape(size, size, 3)
    rough = rough * (1 - 0.25 * dirt) + 0.72 * 0.25 * dirt
    # blocks band
    bb = Vv > 0.8
    blk = np.zeros_like(Hh)
    # sipes: zig-zag lines across blocks
    sip = np.abs(np.sin(U * 2 * math.pi * 40 * 3 / TIRE_TILES + 0.6 * np.sin(Vv * 2 * math.pi * 30)))
    blk = sstep(0.12, 0.05, sip) * bb
    Hh -= blk * 0.0008
    alb[bb] = mixc(C(0.030, 0.029, 0.028), C(0.060, 0.050, 0.038), np.clip(0.25 + 0.5 * nz[bb], 0, 1))
    rough[bb] = 0.58
    alb += (0.004 * nz2)[..., None]
    # normal from height: texel size ~ (half circumference / size, profile/size)
    Ltot = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(prof[:-1], prof[1:]))
    du = math.pi * 0.28 / size; dv = Ltot / (0.8 * size)
    gx = (np.roll(Hh, -1, 1) - np.roll(Hh, 1, 1)) / (2 * du)
    gy = (np.roll(Hh, -1, 0) - np.roll(Hh, 1, 0)) / (2 * dv)
    nrm = np.stack([-gx, -gy, np.ones_like(Hh)], 2); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    nrm = nrm * 0.5 + 0.5
    orm = np.stack([np.ones_like(Hh), rough, np.zeros_like(Hh)], 2)
    save_png(alb, os.path.join(TEXDIR, 'tire_albedo.png'), srgb=True)
    save_png(orm, os.path.join(TEXDIR, 'tire_orm.png'))
    save_png(nrm, os.path.join(TEXDIR, 'tire_normal.png'))

# ---------------- export materials ----------------
def _gltf_output_group():
    g = bpy.data.node_groups.get('glTF Material Output')
    if g: return g
    g = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
    g.interface.new_socket('Occlusion', in_out='INPUT', socket_type='NodeSocketFloat')
    g.interface.new_socket('Thickness', in_out='INPUT', socket_type='NodeSocketFloat')
    g.nodes.new('NodeGroupOutput'); g.nodes.new('NodeGroupInput')
    return g

def _load(path, noncolor=True):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = 'Non-Color' if noncolor else 'sRGB'
    return im

def export_material(name, albedo=None, orm=None, normal=None, coat_img=None, base=(1, 1, 1), rough=1.0, metal=1.0,
                    coat=0.0, emit=None, emit_tex=None, alpha=1.0, transmission=0.0, ior=1.5, blend=False, nstrength=1.0,
                    coat_normal=None, detail_normal=None, detail_strength=1.0, coat_nstrength=1.0):
    m = bpy.data.materials.get(name)
    if m: m.name = name + '_old'
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*base, 1)
    b.inputs['Roughness'].default_value = rough; b.inputs['Metallic'].default_value = metal
    if albedo:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = _load(albedo, False)
        nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    if orm:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = _load(orm)
        sp = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(t.outputs['Color'], sp.inputs['Color'])
        nt.links.new(sp.outputs['Green'], b.inputs['Roughness']); nt.links.new(sp.outputs['Blue'], b.inputs['Metallic'])
        grp = nt.nodes.new('ShaderNodeGroup'); grp.node_tree = _gltf_output_group()
        nt.links.new(sp.outputs['Red'], grp.inputs['Occlusion'])
    if normal:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = _load(normal)
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value = nstrength
        nt.links.new(t.outputs['Color'], nm.inputs['Color']); nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    if coat > 0:
        b.inputs['Coat Weight'].default_value = coat
        b.inputs['Coat Roughness'].default_value = 0.1
        b.inputs['Coat IOR'].default_value = 1.5
        if coat_img:
            t = nt.nodes.new('ShaderNodeTexImage'); t.image = _load(coat_img)
            sp = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(t.outputs['Color'], sp.inputs['Color'])
            nt.links.new(sp.outputs['Red'], b.inputs['Coat Weight']); nt.links.new(sp.outputs['Green'], b.inputs['Coat Roughness'])
    def detail_nmap(path, socket, strength):
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = _load(path)
        uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = 'detail'
        nt.links.new(uvn.outputs['UV'], t.inputs['Vector'])
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.uv_map = 'detail'; nm.inputs['Strength'].default_value = strength
        nt.links.new(t.outputs['Color'], nm.inputs['Color']); nt.links.new(nm.outputs['Normal'], b.inputs[socket])
    if coat_normal and coat > 0: detail_nmap(coat_normal, 'Coat Normal', coat_nstrength)
    if detail_normal and not normal: detail_nmap(detail_normal, 'Normal', detail_strength)
    if emit is not None:
        b.inputs['Emission Color'].default_value = (*emit, 1); b.inputs['Emission Strength'].default_value = 1.0
        if emit_tex:
            t = nt.nodes.new('ShaderNodeTexImage'); t.image = _load(emit_tex, False)
            nt.links.new(t.outputs['Color'], b.inputs['Emission Color'])
    if transmission > 0:
        b.inputs['Transmission Weight'].default_value = transmission; b.inputs['IOR'].default_value = ior
    # (v2) single-sided unless see-through: every car part is a closed/solid shell (checked in-engine: no holes), and
    # the cockpit camera otherwise shades all back faces of the full-screen interior. vehicle.js patchGlass relies
    # on gl_FrontFacing, so glass (and the headlight lens) stay double-sided.
    m.use_backface_culling = not (transmission > 0 or blend)
    if alpha < 1.0 or blend:
        b.inputs['Alpha'].default_value = alpha
        try: m.surface_render_method = 'BLENDED'
        except Exception: pass
    return m

def build_export_materials():
    T = lambda n: os.path.join(TEXDIR, n)
    E = dict(albedo=T('ext_albedo.png'), orm=T('ext_orm.png'), normal=T('ext_normal.png'))
    BN = T('beads_normal.png')
    mats = {
        # old oxidised single-stage paint sheets water rather than beading it: weak bead relief in the coat
        'paint': export_material('paint', coat=1.0, coat_img=T('ext_coat.png'), coat_normal=T('peel_normal.png'), coat_nstrength=0.45, **E),
        'trim_black': export_material('trim_black', coat=1.0, coat_img=T('ext_coat.png'), coat_normal=BN, coat_nstrength=0.6, **E),
        'chrome': export_material('chrome', **E),
        'headlight': export_material('headlight', normal=T('ext_normal.png'), base=(0.90, 0.92, 0.90), rough=0.02, metal=0.0,
                                     emit=(1.0, 1.0, 1.0), emit_tex=T('ext_emit.png'), alpha=0.12, blend=True),   # clear pressed glass
        'brakelight': export_material('brakelight', emit=(1.0, 1.0, 1.0), emit_tex=T('ext_emit.png'), coat=1.0, coat_img=T('ext_coat.png'), coat_normal=BN, **E),
        'indicator': export_material('indicator', emit=(1.0, 1.0, 1.0), emit_tex=T('ext_emit.png'), coat=1.0, coat_img=T('ext_coat.png'), coat_normal=BN, **E),
        'plate': export_material('plate', **E),
        # (v3) ALPHA glass, not transmission: KHR_materials_transmission makes three.js render the whole opaque scene a
        # second time every frame the car is visible. Thin, faintly green-grey tinted, mirror-smooth where clean
        # (roughness map carries the dirt film / wiper arcs); vehicle.js keeps the Fresnel reflection un-scaled by alpha.
        'glass': export_material('glass', albedo=T('ext_albedo.png'), orm=T('ext_orm.png'), base=(0.86, 0.9, 0.88), rough=0.02, metal=0.0,
                                 alpha=0.2, blend=True, detail_normal=BN, detail_strength=0.35),
        'interior': export_material('interior', albedo=T('int_albedo.png'), orm=T('int_orm.png'), normal=T('int_normal.png')),
        'gauge': export_material('gauge', albedo=T('gauge.png'), rough=0.3, metal=0.0, emit=(1, 1, 1), emit_tex=T('gauge.png')),
        'rubber': export_material('rubber', albedo=T('tire_albedo.png'), orm=T('tire_orm.png'), normal=T('tire_normal.png')),
    }
    return mats

def remap_materials(ob, mats):
    """Replace surface material slots by export materials (merging slots)."""
    me = ob.data
    import re
    def base(nm):
        while True:
            nn = re.sub(r'\.\d+$', '', re.sub(r'_old$', '', nm))
            if nn == nm: return nn
            nm = nn
    old = [base(m.name) if m else None for m in me.materials]
    target = []
    for nm in old:
        t = EXPORT_MAP.get(nm, nm)
        if t not in target: target.append(t)
    remap = [target.index(EXPORT_MAP.get(nm, nm)) for nm in old]
    mi = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get('material_index', mi)
    mi = np.array(remap, np.int32)[mi] if len(remap) else mi
    me.materials.clear()
    for t in target: me.materials.append(mats[t])
    me.polygons.foreach_set('material_index', mi)
    me.update()

# ---------------- full pipeline ----------------
GB_DIR = os.path.join(SCR, 'gbuf')

def save_gbuf(G, tag_):
    """Cache the baked G-buffer so textures can be recomposed without rebuilding/rebaking (--stage recompose)."""
    os.makedirs(GB_DIR, exist_ok=True)
    for k, a in G.items():
        np.save(os.path.join(GB_DIR, f'{tag_}_{k}.npy'), a if k in ('P', 'M') else a.astype(np.float16))

def load_gbuf(tag_):
    return {k: np.load(os.path.join(GB_DIR, f'{tag_}_{k}.npy')).astype(np.float32) for k in 'PNMA'}

def half_cov(im, cov):
    """2x2 downsample (only when > 1024) averaging covered texels only, so island borders don't blend with the empty
    atlas background (which would make seams artificially glossy)."""
    if im.shape[0] <= 1024: return im
    h2 = im.shape[0] // 2; c = im.shape[2]
    w = cov.reshape(h2, 2, h2, 2, 1)
    num = (im.reshape(h2, 2, h2, 2, c) * w).sum((1, 3)); den = w.sum((1, 3))
    return np.where(den > 0, num / np.maximum(den, 1e-6), 0).astype(np.float32)

def compose_ext_textures(plate_img, cover_img):
    Gext = load_gbuf('ext')
    size = Gext['P'].shape[0]
    albedo, orm, coat_im, nrm, cov = compose_ext(Gext, size, plate_img, cover_img)
    save_png(albedo, os.path.join(TEXDIR, 'ext_albedo.png'), srgb=True)
    ocov = (orm[..., 0] > 0.05).astype(np.float32)        # covered texels (AO channel is >= 0.1 wherever baked)
    save_png(half_cov(orm, ocov), os.path.join(TEXDIR, 'ext_orm.png'))   # (v2) AO/roughness/metal: half resolution
    if coat_im.shape[0] > 1024:   # clearcoat mask is low frequency: half resolution
        coat_im = half_cov(coat_im, ocov)
    save_png(coat_im, os.path.join(TEXDIR, 'ext_coat.png'))
    save_png(nrm, os.path.join(TEXDIR, 'ext_normal.png'))
    em = EXT_EMIT
    if em.shape[0] > 1024:        # lamp islands are large in the atlas: half resolution is plenty
        h2 = em.shape[0] // 2
        em = em.reshape(h2, 2, h2, 2, 3).mean((1, 3))
    save_png(em, os.path.join(TEXDIR, 'ext_emit.png'), srgb=True)
    del albedo, orm, coat_im, nrm, Gext, em

def compose_textures():
    """All texture PNGs from the cached G-buffers (+ 2D prints rendered here)."""
    log('text textures')
    gauge_texture()
    if not INT_ONLY:
        plate_img = plate_texture(); cover_img = cover_texture()
        tire_textures(1024)
        bead_texture(1024)
        compose_ext_textures(plate_img, cover_img)
    Gint = load_gbuf('int')
    ialb, iorm, inrm = compose_int(Gint, Gint['P'].shape[0])
    save_png(ialb, os.path.join(TEXDIR, 'int_albedo.png'), srgb=True)
    icov = (Gint['P'][..., 3] > 0.5).astype(np.float32)          # (v4) occlusion can be ~0.02 now: use the real coverage
    save_png(half_cov(iorm, icov), os.path.join(TEXDIR, 'int_orm.png'))
    save_png(inrm, os.path.join(TEXDIR, 'int_normal.png'))

def recompose():
    """--stage recompose: reopen car_final.blend, recompose textures from the cached G-buffers, rebuild the export
    materials, re-export GLB + car.json, optimize."""
    bpy.ops.wm.open_mainfile(filepath=os.path.join(SCR, 'car_final.blend'))
    for o in bpy.data.objects:   # name fix-ups for blends saved by older versions of this script
        for nm in ('air_freshener', 'mirror_charm', 'key_ring'):
            if o.type == 'EMPTY' and o.name.startswith(nm + '.'): o.name = nm
    compose_textures()
    for im in list(bpy.data.images):
        if im.filepath and im.source == 'FILE':
            try: im.reload()
            except Exception as e: log('reload failed', im.name, e)
    mats = build_export_materials()
    exp = [o for o in bpy.data.objects if o.get('export')]
    for ob in exp:
        if ob.type == 'MESH': remap_materials(ob, mats)
    for m in list(bpy.data.materials):
        if m.name.endswith('_old') and m.users == 0: bpy.data.materials.remove(m)
    export_glb(exp)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(SCR, 'car_final.blend'))

def texture_and_export(P):
    body, wheel = assemble(P)
    sw = P['sw']
    # other 3 wheels (linked) for occlusion while baking
    occ = []
    for l, f in ((-HT, FAX), (HT, RAX), (-HT, RAX)):
        o = bpy.data.objects.new('occ_wheel', wheel.data); coll('CAR').objects.link(o)
        o.location = V(l, f, R_TIRE)
        if l < 0: o.rotation_euler = (0, 0, math.pi)
        occ.append(o)
    gbm = bmesh.new(); bmesh.ops.create_grid(gbm, x_segments=1, y_segments=1, size=6)
    ground = obj_from_bm('bake_ground', gbm, 'plastic')
    fresh = P['fresh']; keyring = P['keyring']
    log('uv unwrap ext'); unwrap_atlas([body, wheel, sw, keyring], EXT_MATS, 0.0035 if TEX >= 2048 else 0.005)
    log('uv unwrap int'); unwrap_atlas([body, sw, fresh, keyring], INT_MATS, 0.004)
    # glass does not occlude light: move it away while baking AO
    gi = [i for i, m in enumerate(body.data.materials) if m.name == 'glass']
    gverts = sorted({v for p in body.data.polygons if p.material_index in gi for v in p.vertices})
    def hide_glass(on):
        for v in gverts: body.data.vertices[v].co.z += (-50.0 if on else 50.0)
        body.data.update()
    log('bake ext gbuffer')
    G = bake_gbuffer([body, wheel, sw, keyring], occ, set(EXT_MATS), TEX, 'ext', hide_glass)
    save_gbuf(G, 'ext'); del G
    log('bake int gbuffer')
    G = bake_gbuffer([body, sw, fresh, keyring], occ, set(INT_MATS), TEX, 'int', hide_glass)
    save_gbuf(G, 'int'); del G
    compose_textures()
    for o in occ + [ground]: delete_obj(o)
    for im in list(bpy.data.images):
        if im.name.startswith('g') and '_' in im.name or im.name == '_dummy': bpy.data.images.remove(im)
    log('export materials')
    mats = build_export_materials()
    for ob in [body, wheel, sw, fresh, keyring] + P['needles']:
        remap_materials(ob, mats)
    # needles: UV onto the orange swatch in the gauge texture
    for nd in P['needles']:
        me = nd.data
        if not me.uv_layers: me.uv_layers.new(name='UVMap')
        for poly in me.polygons:        # face 0 = needle (orange swatch), the rest = hub cap (black print background)
            for li in poly.loop_indices: me.uv_layers[0].data[li].uv = NEEDLE_UV if poly.index == 0 else NEEDLE_CAP_UV
    for ob in [body, wheel, sw, fresh, keyring]:
        if 'src' in ob.data.uv_layers: ob.data.uv_layers.remove(ob.data.uv_layers['src'])
    return body, wheel, sw, fresh, keyring

def finalize_and_export(body, wheel, sw, needles, fresh=None, keyring=None):
    """Pivot hierarchy + GLB + car.json. Pivot nodes (car_body, wheel_*, steering_wheel, needle_*) are EMPTY nodes
    whose mesh lives in a child node at identity, so glTF optimizers (quantization) never alter the pivots."""
    wheel.location = (0, 0, 0)
    wl = wheel.data; wl.name = 'wheel_L'
    wr = wl.copy(); wr.name = 'wheel_R'
    wr.transform(Matrix.Rotation(math.pi, 4, 'Z'))
    def empty(name, M, parent=None):
        e = bpy.data.objects.new(name, None); e.empty_display_size = 0.1; coll('CAR').objects.link(e)
        if parent is not None:
            e.parent = parent; e.matrix_parent_inverse = Matrix.Identity(4)
        e.matrix_basis = M
        return e
    def attach(o, parent, name):
        o.name = name; o.data.name = name
        o.parent = parent; o.matrix_parent_inverse = Matrix.Identity(4); o.matrix_basis = Matrix.Identity(4)
        return o
    body.name = 'car_body_mesh'; sw.name = 'steering_wheel_mesh'
    for nd in needles: nd['_nm'] = nd.name; nd.name = nd.name + '_mesh'
    root = empty('car_body', Matrix.Identity(4))
    attach(body, root, 'car_body_mesh')
    swM = Matrix.Translation(V(*SW_C)) @ Matrix.Rotation(SW_ALPHA, 4, 'X')
    swp = empty('steering_wheel', swM, root)
    attach(sw, swp, 'steering_wheel_mesh')
    nps = []
    for nd in needles:
        M = nd.matrix_world.copy(); nm = nd['_nm']
        et = empty(nm + '_tilt', M, root)                 # carries the dial tilt
        e = empty(nm, Matrix.Identity(4), et)             # driven node: identity rest transform
        attach(nd, e, nm + '_mesh'); nps += [et, e, nd]
    # (v4) dangling nodes (pivot empty + mesh child at identity): vehicle.js swings them like pendulums
    for ob, nm, piv in ((fresh, 'mirror_charm', V(*CHARM_PIV)), (keyring, 'key_ring', V(*key_ring_pivot()))):
        if ob is None: continue
        ob.name = nm + '_mesh'; ob.data.name = nm + '_mesh'
        fp = empty(nm, Matrix.Translation(piv), root)
        fp.name = nm
        ob.data.transform(Matrix.Translation(-piv))
        attach(ob, fp, nm + '_mesh'); nps += [fp, ob]
    nodes = []
    for nm, l, f in (('wheel_fl', HT, FAX), ('wheel_fr', -HT, FAX), ('wheel_rl', HT, RAX), ('wheel_rr', -HT, RAX)):
        e = empty(nm, Matrix.Translation(V(l, f, R_TIRE)))
        o = bpy.data.objects.new(nm + '_mesh', wl if l > 0 else wr); coll('CAR').objects.link(o)
        o.parent = e; o.matrix_parent_inverse = Matrix.Identity(4); o.matrix_basis = Matrix.Identity(4)
        nodes += [e, o]
    delete_obj(wheel)
    empties = {}
    for nm, p in (('seat_cam', EYE), ('fuel_cap', (W_BODY + 0.035, -1.30, 0.885)), ('door_point', (W_BODY + 0.90, 0.10, 0.0))):
        empties[nm] = empty(nm, Matrix.Translation(V(*p)), root)
    bpy.context.view_layer.update()
    exp = [root, body, swp, sw] + nps + nodes + list(empties.values())
    for o in exp: o['export'] = 1
    return export_glb(exp)

def export_glb(exp):
    """Export the marked objects to OUT_GLB_RAW and write car.json."""
    for o in bpy.context.view_layer.objects: o.select_set(False)
    for o in exp: o.select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['car_body']
    bpy.ops.export_scene.gltf(filepath=OUT_GLB_RAW, export_format='GLB', use_selection=True, export_apply=True, export_yup=True,
                              export_texcoords=True, export_normals=True, export_tangents=False, export_materials='EXPORT',
                              export_image_format='AUTO', export_cameras=False, export_lights=False, export_extras=False,
                              export_animations=False)
    log('exported', OUT_GLB_RAW, os.path.getsize(OUT_GLB_RAW) // 1024, 'KB')
    tris = sum(len(p.vertices) - 2 for o in exp if o.type == 'MESH' for p in o.data.polygons)
    info = {
        'mass': 1250, 'wheelbase': WB, 'track': TRACK, 'wheelRadius': R_TIRE, 'wheelWidth': W_TIRE,
        'suspensionRest': 0.30, 'suspensionTravel': 0.20, 'cgHeight': 0.62, 'frontAxleZ': FAX, 'rearAxleZ': RAX,
        '_conventions': 'three.js car-local frame: +Z forward, +X left (driver side), +Y up, origin on the ground between the axles. '
                        'wheels = wheel node rest positions (wheel centres at ride height, tyre touching y=0). '
                        'suspensionRest = rest length of the suspension ray from the chassis connection point (wheel centre + suspensionRest up) '
                        'to the wheel centre; suspensionTravel = max compression; cgHeight above ground.',
        'wheels': {'fl': [HT, R_TIRE, FAX], 'fr': [-HT, R_TIRE, FAX], 'rl': [HT, R_TIRE, RAX], 'rr': [-HT, R_TIRE, RAX]},
        'chassisBoxes': [
            {'center': [0, 0.74, 0.0], 'halfExtents': [0.80, 0.32, 1.80]},
            {'center': [0, 1.34, -0.49], 'halfExtents': [0.74, 0.30, 1.25]},
        ],
        'seatCam': list(EYE[:1]) + [EYE[2], EYE[1]],
        'fuelCap': [W_BODY + 0.035, 0.885, -1.30], 'doorPoint': [W_BODY + 0.90, 0.0, 0.10],
        'headlights': [[0.565, 0.785, 1.79], [-0.565, 0.785, 1.79]],
        'brakelights': [[0.735, 0.86, -1.77], [-0.735, 0.86, -1.77]],
        'exhaust': [-0.45, 0.265, -1.90],        # tailpipe exit (rear right, under the bumper), car-local three.js coords
        'steeringWheel': {'position': [SW_C[0], SW_C[2], SW_C[1]], 'axis': 'local Z', 'maxAngle': 7.85},
        'needles': {'speedo': {'zeroDeg': NEEDLE_SIGN * -135, 'maxDeg': NEEDLE_SIGN * 135, 'maxKmh': 160},
                    'tacho': {'zeroDeg': NEEDLE_SIGN * -135, 'maxDeg': NEEDLE_SIGN * 135, 'maxRpm': 7000},
                    'fuel': {'emptyDeg': NEEDLE_SIGN * -60, 'fullDeg': NEEDLE_SIGN * 60},
                    'temp': {'coldDeg': NEEDLE_SIGN * -60, 'hotDeg': NEEDLE_SIGN * 60},
                    '_note': 'needle_<k> nodes have an identity rest transform (tilt on parent needle_<k>_tilt); at rotation.z = 0 '
                             'the needle points straight up on the dial. Set rotation.z = deg * PI/180, interpolating zeroDeg->maxDeg '
                             '(emptyDeg->fullDeg, coldDeg->hotDeg). Speedo/tacho dials sweep 270 deg, fuel/temp 120 deg over the top.'},
        'dangling': {'mirror_charm': {'pivot': [CHARM_PIV[0], CHARM_PIV[2], CHARM_PIV[1]], 'length': 0.12},
                     'key_ring': {'pivot': [round(key_ring_pivot().x, 5), round(key_ring_pivot().z, 5), round(key_ring_pivot().y, 5)], 'length': 0.035},
                     '_note': 'pivot empties with identity-rest mesh children: swing them (rotation.x / rotation.z) like pendulums.'},
        'rearMirror': rvm_frame(),
        'gaugeLens': {'center': [SEAT_L, 1.100, pod_f(1.100) - 0.012], 'tilt': round(math.degrees(GAUGE_TILT), 3), 'width': 0.345, 'height': 0.10,
                      '_note': 'clear cluster lens: a quad in the pod opening, normal toward the driver tilted by tilt deg (top leaning away)'},
        'emissive': 'headlight/brakelight/indicator/gauge materials carry an emissiveMap that holds the lit colour AND pattern (lamps: ext_emit bulb hotspot/reflector/lens optics; gauges: dial print); emissive factor is white. Drive emissiveIntensity only (0 = off).',
        'triangles': int(tris),
    }
    with open(OUT_JSON, 'w') as fh: json.dump(info, fh, indent=2)
    log('car.json written; triangles', tris)
    return exp

def optimize_glb():
    """WebP textures + meshopt geometry. Individual gltf-transform commands (not `optimize`), because
    `optimize` dedups materials that share textures/factors (chrome/plate/trim_black) and flattens nodes."""
    import subprocess
    tmp1 = os.path.join(SCR, 'car_tmp1.glb'); tmp2 = os.path.join(SCR, 'car_tmp2.glb')
    steps = [
        ['webp', OUT_GLB_RAW, tmp1, '--slots', '{baseColorTexture,emissiveTexture,metallicRoughnessTexture,occlusionTexture,clearcoatTexture,clearcoatRoughnessTexture}', '--quality', '88'],
        ['webp', tmp1, tmp2, '--slots', '*ormalTexture', '--quality', '95'],
        ['meshopt', tmp2, OUT_GLB, '--level', 'medium', '--quantize-texcoord', '14'],
    ]
    gt = os.path.join(ROOT, 'node_modules', '.bin', 'gltf-transform')   # not npx: npx re-parses args through a shell
    for st in steps:
        r = subprocess.run([gt] + st, cwd=ROOT, capture_output=True, text=True)
        log('gltf-transform', st[0], 'rc', r.returncode, (r.stdout + r.stderr).strip()[-300:])
    for t in (tmp1, tmp2):
        if os.path.exists(t): os.remove(t)
    if os.path.exists(OUT_GLB): log('optimized', os.path.getsize(OUT_GLB) // 1024, 'KB')
    check_contract(OUT_GLB)

def check_contract(path):
    """Verify node names/pivots and material names in a GLB (reads the JSON chunk)."""
    import struct
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + n])
    byname = {nd.get('name'): nd for nd in j['nodes']}
    want = {'car_body': None, 'wheel_fl': [HT, R_TIRE, FAX], 'wheel_fr': [-HT, R_TIRE, FAX], 'wheel_rl': [HT, R_TIRE, RAX],
            'wheel_rr': [-HT, R_TIRE, RAX], 'steering_wheel': [SW_C[0], SW_C[2], SW_C[1]], 'seat_cam': [EYE[0], EYE[2], EYE[1]],
            'fuel_cap': [W_BODY + 0.035, 0.885, -1.30], 'door_point': [W_BODY + 0.90, 0.0, 0.10]}
    ok = True
    for k, t in want.items():
        nd = byname.get(k)
        if nd is None: log('CONTRACT MISSING node', k); ok = False; continue
        if 'scale' in nd: log('CONTRACT node has scale', k, nd['scale']); ok = False
        if t is not None:
            tr = nd.get('translation', [0, 0, 0])
            if max(abs(a - c) for a, c in zip(tr, t)) > 1e-3: log('CONTRACT bad translation', k, tr, t); ok = False
    mats = {m.get('name') for m in j.get('materials', [])}
    for m in ('paint', 'glass', 'chrome', 'trim_black', 'rubber', 'interior', 'headlight', 'brakelight', 'indicator', 'plate', 'gauge'):
        if m not in mats: log('CONTRACT missing material', m); ok = False
    log('CONTRACT', 'OK' if ok else 'FAILED', sorted(mats))
    return ok

def preview_from_glb(path, prefix, views, samples):
    """Import the exported GLB into a clean scene and render the preview views (validates the export)."""
    reset_scene(); coll('CAR')
    bpy.ops.import_scene.gltf(filepath=path)
    for o in list(bpy.context.scene.collection.objects):
        pass
    # thin-glass fix for Cycles (glTF transmission on single-sided planes refracts): IOR 1.0
    for m in bpy.data.materials:
        if m.name.startswith('glass'):
            nt = m.node_tree
            imgs = [nd.image for nd in nt.nodes if nd.type == 'TEX_IMAGE' and nd.image]
            alb_im = next((i for i in imgs if 'albedo' in i.name), None); orm_im = next((i for i in imgs if 'orm' in i.name), None)
            for nd in list(nt.nodes): nt.nodes.remove(nd)
            out = nt.nodes.new('ShaderNodeOutputMaterial')
            tr = nt.nodes.new('ShaderNodeBsdfTransparent'); tr.inputs['Color'].default_value = (0.86, 0.90, 0.88, 1)
            gl = nt.nodes.new('ShaderNodeBsdfGlossy'); gl.inputs['Roughness'].default_value = 0.03
            fr = nt.nodes.new('ShaderNodeLayerWeight'); fr.inputs['Blend'].default_value = 0.06
            mx = nt.nodes.new('ShaderNodeMixShader')
            if alb_im is not None:
                ta = nt.nodes.new('ShaderNodeTexImage'); ta.image = alb_im
                nt.links.new(ta.outputs['Color'], tr.inputs['Color'])
                df = nt.nodes.new('ShaderNodeBsdfDiffuse'); nt.links.new(ta.outputs['Color'], df.inputs['Color'])
                mx2 = nt.nodes.new('ShaderNodeMixShader')
                inv = nt.nodes.new('ShaderNodeRGBToBW'); nt.links.new(ta.outputs['Color'], inv.inputs[0])
                mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.88; mr.inputs['From Max'].default_value = 0.40
                mr.inputs['To Max'].default_value = 0.35
                nt.links.new(inv.outputs[0], mr.inputs['Value'])
                nt.links.new(mr.outputs['Result'], mx2.inputs[0]); nt.links.new(tr.outputs[0], mx2.inputs[1]); nt.links.new(df.outputs[0], mx2.inputs[2])
                trans_out = mx2.outputs[0]
            else:
                trans_out = tr.outputs[0]
            if orm_im is not None:
                to = nt.nodes.new('ShaderNodeTexImage'); to.image = orm_im
                sp = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(to.outputs['Color'], sp.inputs['Color'])
                nt.links.new(sp.outputs['Green'], gl.inputs['Roughness'])
            nt.links.new(fr.outputs[0], mx.inputs[0]); nt.links.new(trans_out, mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2])
            nt.links.new(mx.outputs[0], out.inputs['Surface'])
    # (v4) interior ORM blue = material class for the engine's micro-detail shader, not metalness
    for m in bpy.data.materials:
        if m.name.startswith('interior') and m.use_nodes:
            b = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if b:
                for l in list(b.inputs['Metallic'].links): m.node_tree.links.remove(l)
                b.inputs['Metallic'].default_value = 0.0
    if not LAMPS_ON:
        for m in bpy.data.materials:
            if m.name in ('headlight', 'brakelight', 'indicator') and m.use_nodes:
                b = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if b: b.inputs['Emission Strength'].default_value = 0.0
    render_views(prefix, views, samples, (1280, 720))

# ------------------------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------------------------
def build_all():
    reset_scene(); init_materials()
    coll('CAR'); coll('CUT').hide_render = True
    bpy.context.view_layer.layer_collection.children['CUT'].exclude = False
    body = build_body()
    log('body tris', sum(len(p.vertices) - 2 for p in body.data.polygons))
    flares = build_flares()
    rockers = build_rockers()
    bumpers = build_bumpers()
    front = build_front()
    rear = build_rear()
    ghd, glass = build_greenhouse_details()
    side = build_side_details()
    asym = build_asym_details()
    under = build_underbody()
    interior = build_interior()
    gauges = join(build_gauges(), 'gauges'); tag(gauges, 'gauge')
    needles = build_needles()
    sw = build_steering_wheel(); tag(sw, 'int')
    fresh = build_charm()                  # (v4) prayer beads (node mirror_charm) replace the cardboard freshener
    keyring = build_key_ring()
    sw.matrix_world = Matrix.Translation(V(*SW_C)) @ Matrix.Rotation(SW_ALPHA, 4, 'X')
    tire, rim = build_wheels()
    return dict(body=body, flares=flares, rockers=rockers, bumpers=bumpers, front=front, rear=rear, ghd=ghd, glass=glass,
                side=side, asym=asym, under=under, interior=interior, gauges=gauges, needles=needles, sw=sw, tire=tire, rim=rim,
                fresh=fresh, keyring=keyring)

def place_preview_wheels(tire, rim):
    """Instances for preview renders (model stage)."""
    obs = []
    for nm, l, f in (('wheel_fl', HT, FAX), ('wheel_fr', -HT, FAX), ('wheel_rl', HT, RAX), ('wheel_rr', -HT, RAX)):
        for src in (tire, rim):
            o = bpy.data.objects.new(nm + '_' + src.name, src.data); coll('CAR').objects.link(o)
            o.location = V(l, f, R_TIRE)
            if l < 0: o.rotation_euler = (0, 0, math.pi)
            obs.append(o)
    tire.hide_render = True; rim.hide_render = True
    return obs

def count_tris():
    t = 0
    for o in coll('CAR').objects:
        if o.type == 'MESH' and not o.hide_render:
            t += sum(len(p.vertices) - 2 for p in o.data.polygons)
    return t

if __name__ == '__main__':
    if arg('--optimize-only', False):
        optimize_glb()
    elif STAGE == 'recompose':
        recompose()
        optimize_glb()
        if not NO_PREVIEW:
            preview_from_glb(OUT_GLB_RAW, 'p', VIEWS, 48 if FAST else 128)
        log('done')
    elif PREVIEW_ONLY:
        preview_from_glb(OUT_GLB_RAW, 'p', VIEWS, 48 if FAST else 128)
    elif STAGE == 'model':
        P = build_all()
        place_preview_wheels(P['tire'], P['rim'])
        log('total tris ~', count_tris())
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(SCR, 'car_model.blend'))
        if not NO_PREVIEW:
            render_views('m', VIEWS, 32 if FAST else 64, (960, 540))
        log('done')
    else:
        P = build_all()
        body, wheel, sw, fresh, keyring = texture_and_export(P)
        finalize_and_export(body, wheel, sw, P['needles'], fresh, keyring)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(SCR, 'car_final.blend'))
        optimize_glb()
        if not NO_PREVIEW:
            preview_from_glb(OUT_GLB_RAW, 'p', VIEWS, 48 if FAST else 128)
        log('done')
