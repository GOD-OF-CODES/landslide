"""
LANDSLIDE - first-person hands (HANDS workstream, GAME viewmodel).

Gloved hands with forearms in rain-jacket sleeves, modelled from scratch (no downloaded models). CC0 Poly Haven scans
(raw_assets/tex/brown_leather, stretch_poplin) are only used as texture sources inside the bake materials.

Build:
  1. Glove "sculpt": closed parts (palm slab, metacarpal capsules, knuckle row, thenar/hypothenar pads, thumb web,
     per-phalanx lofted tubes with superelliptic sections, joint bulges and finger pads, glove fingertip caps, wrist
     gauntlet) -> joined -> voxel remesh (union) -> smooth.  = glove_hi (bake source) after a numpy displacement pass
     (dorsal joint wrinkles, palmar flexion creases, seam grooves + welts, lumpy leather).
  2. glove_lo: QuadriFlow retopology of the smoothed union (~7k tris), UV unwrapped.
  3. Sleeve: parametric tube (lo 32 x 22 quads + rolled cuff hem + velcro tab); sleeve_hi = the same surface, dense,
     with compression folds at the cuff and a few long folds.
  4. Bake (Cycles, selected-to-active) into one shared atlas: albedo (sRGB), normal (OpenGL), ORM (AO, rough, metal).
  5. Armature per hand: <S>_forearm (root) -> <S>_hand (wrist) -> fingers <S>_{thumb,index,middle,ring,pinky}{1,2,3}.
     Bone Y runs along the bone, Z = dorsal (back of hand / thumbnail), so flexion = rotation about local -X.
     Glove skinned with heat weights (bones hand + fingers), sleeve rigid to the forearm; the jacket cuff overlaps
     the glove gauntlet so the wrist never needs blending.
  6. Left hand = mirrored copy (own bones L_*). Pose table (local bone rotations) is stored as JSON in the armature
     node extras (userData.poses in three.js):  {pose: {bone: [x, y, z, w]}}  (Blender pose-basis quaternions; three.js
     bone.quaternion = restQuaternion * pose). Left-hand poses are the mirrored right poses (x, -y, -z, w).

Rest frame (three.js, after the exporter's Y-up conversion): the arm points along -Z (elbow at z=+0.27, wrist at the
origin), palm down (-Y), right thumb toward -X. In Blender: arm along +Y, palm facing -Z, right thumb at -X.

Real dimensions matched (adult male ~50th percentile + 2.5 mm leather/lining per side): hand length (wrist crease to
middle fingertip) 19.5 cm with glove, palm breadth 9.0 cm, finger widths 2.3-2.4 cm (pinky 2.0), forearm 27 cm,
jacket cuff ~25 cm circumference (velcro closed over the glove wrist).

Usage:
  /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/blender/hands.py -- [options]
    --stage all|clay     clay = geometry, rig, poses, clay previews + clay GLB (no bake)
    --size 1024          atlas size          --samples 16    bake samples
    --preview all|none   preview renders (scratch/hands/*.png)
    --gpu                Cycles on the Metal GPU (faster, but ~2.6 GB peak instead of ~1.3 GB on the CPU)
    --no-export
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
from mathutils import Vector, Matrix, Quaternion, Euler

T0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SCR = os.path.join(ROOT, 'scratch', 'hands')
BAKE = os.path.join(SCR, 'bake')
OUT_GLB = os.path.join(ROOT, 'public', 'assets', 'models', 'hands.glb')
TEX = os.path.join(ROOT, 'raw_assets', 'tex')
HDRI = os.path.join(ROOT, 'raw_assets', 'hdri', 'overcast_soil_puresky_2k.hdr')
os.makedirs(BAKE, exist_ok=True)


def log(*a):
    print('[hands %6.1fs]' % (time.time() - T0), *a, flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    a = {'stage': 'all', 'size': 1024, 'samples': 16, 'preview': 'all', 'export': True, 'gpu': False}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == '--stage':
            a['stage'] = argv[i + 1]; i += 1
        elif k == '--size':
            a['size'] = int(argv[i + 1]); i += 1
        elif k == '--samples':
            a['samples'] = int(argv[i + 1]); i += 1
        elif k == '--preview':
            a['preview'] = argv[i + 1]; i += 1
        elif k == '--no-export':
            a['export'] = False
        elif k == '--gpu':
            a['gpu'] = True
        i += 1
    return a


ARGS = parse_args()


# ================================================================================================
# hand layout (Blender coords, metres; right hand: thumb at -x, fingers along +y, back of hand +z)
# ================================================================================================
def rotz(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])


# per finger: MCP joint centre, fan angle (deg, + toward the pinky side), rest flex per joint (deg, + = toward palm),
# phalanx lengths (glove adds ~4 mm to the tip), section widths / heights at MCP, PIP, DIP, tip.
FINGERS = {
    'index':  dict(mcp=(-0.0300, 0.0930, 0.0010), ang=-11.0, flex=(4, 7, 5), L=(0.0445, 0.0265, 0.0245),
                   W=(0.0252, 0.0231, 0.0210, 0.0189), H=(0.0235, 0.0210, 0.0187, 0.0159)),
    'middle': dict(mcp=(-0.0093, 0.0972, 0.0015), ang=-2.0, flex=(4, 7, 5), L=(0.0485, 0.0300, 0.0255),
                   W=(0.0257, 0.0235, 0.0212, 0.0191), H=(0.0238, 0.0212, 0.0189, 0.0161)),
    'ring':   dict(mcp=(0.0112, 0.0935, 0.0008), ang=7.0, flex=(5, 8, 6), L=(0.0455, 0.0290, 0.0245),
                   W=(0.0244, 0.0223, 0.0201, 0.0180), H=(0.0226, 0.0201, 0.0180, 0.0155)),
    'pinky':  dict(mcp=(0.0292, 0.0850, -0.0012), ang=16.0, flex=(6, 9, 7), L=(0.0365, 0.0220, 0.0220),
                   W=(0.0223, 0.0204, 0.0187, 0.0165), H=(0.0206, 0.0187, 0.0167, 0.0142)),
}
# thumb: carpometacarpal joint inside the thenar, then MCP, IP, tip. Rest: relaxed, ~40 deg out from the index,
# slightly palmar, thumbnail facing mostly sideways.
THUMB = dict(cmc=(-0.0205, 0.0170, -0.0075), dir1=(-0.58, 0.78, -0.20), L=(0.0435, 0.0335, 0.0300),
             bend=(10.0, 8.0), W=(0.0278, 0.0261, 0.0235, 0.0208), H=(0.0246, 0.0233, 0.0208, 0.0170),
             nail=(-0.70, 0.12, 0.70))
FOREARM_LEN = 0.27
VOXEL = 0.0006          # high-poly glove voxel size (m); ~0.2 M verts, keeps the build under ~2 GB


def finger_joints(f):
    """Joint centres [MCP, PIP, DIP, tip] and dorsal vectors per joint for a finger (with its rest flex)."""
    mcp = np.array(f['mcp'], float)
    d = rotz(np.array([0.0, 1.0, 0.0]), -f['ang'])          # fan: + ang toward +x (pinky side)
    lat = np.cross(d, [0, 0, 1.0]); lat /= np.linalg.norm(lat)   # points toward +x for d ~ +y ... (d x z)
    up = np.cross(lat, d); up /= np.linalg.norm(up)
    P = [mcp]
    U = [up]
    cur = d.copy()
    cu = up.copy()
    for k in range(3):
        a = math.radians(f['flex'][k])
        # flex about the lateral axis: direction tips toward -up (palm)
        cur = cur * math.cos(a) - cu * math.sin(a)
        cu = np.cross(lat, cur); cu /= np.linalg.norm(cu)
        P.append(P[-1] + cur * f['L'][k])
        U.append(cu.copy())
    return np.array(P), np.array(U), lat


def thumb_joints():
    t = THUMB
    cmc = np.array(t['cmc'], float)
    d = np.array(t['dir1'], float); d /= np.linalg.norm(d)
    nail = np.array(t['nail'], float)
    nail = nail - d * np.dot(nail, d); nail /= np.linalg.norm(nail)
    lat = np.cross(d, nail); lat /= np.linalg.norm(lat)
    P = [cmc]
    U = [nail]
    cur, cu = d.copy(), nail.copy()
    for k in range(3):
        if k > 0:
            a = math.radians(t['bend'][k - 1])
            cur = cur * math.cos(a) - cu * math.sin(a)
            cu = np.cross(lat, cur); cu /= np.linalg.norm(cu)
        P.append(P[-1] + cur * t['L'][k])
        U.append(cu.copy())
    return np.array(P), np.array(U), lat


# ================================================================================================
# mesh helpers
# ================================================================================================
def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_obj(name, V, F):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in V], [], [list(map(int, f)) for f in F])
    me.validate(clean_customdata=False)
    me.update()
    return link(bpy.data.objects.new(name, me))


def select_only(objs, active=None):
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or (objs[0] if objs else None)


def apply_mods(o):
    select_only([o])
    for m in list(o.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def superellipse(n, w, h, ex=2.4, palm_full=0.0, dorsal_flat=0.0):
    """Closed section (n,2) in (lateral, dorsal) coords. palm_full pushes the palmar half outward (finger pads)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    x = w / 2 * np.sign(ca) * np.abs(ca) ** (2 / ex)
    y = h / 2 * np.sign(sa) * np.abs(sa) ** (2 / ex)
    pal = sa < 0
    y = np.where(pal, y * (1 + palm_full), y * (1 - dorsal_flat))
    return np.stack([x, y], -1)


def loft(sections, centers, T, N, B, cap0=True, cap1=True):
    """sections: list of (n,2) arrays in (B, N) coords; centres (k,3); frames T,N,B (k,3). Closed tube with fan caps."""
    k = len(sections)
    n = len(sections[0])
    V = []
    for i in range(k):
        s = sections[i]
        V.extend(centers[i] + np.outer(s[:, 0], B[i]) + np.outer(s[:, 1], N[i]))
    F = []
    for i in range(k - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            F.append([a, b, b + n, a + n])
    V = list(V)
    if cap0:
        c = len(V); V.append(centers[0] - T[0] * 1e-4)
        for j in range(n):
            F.append([c, (j + 1) % n, j])
    if cap1:
        c = len(V); V.append(centers[-1] + T[-1] * 1e-4)
        base = (k - 1) * n
        for j in range(n):
            F.append([c, base + j, base + (j + 1) % n])
    return np.array(V), F


def ellipsoid(center, radii, rot=None, seg=24, ring=16):
    V, F = [], []
    for i in range(1, ring):
        th = math.pi * i / ring
        for j in range(seg):
            ph = 2 * math.pi * j / seg
            V.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    V = np.array(V) * np.array(radii)
    top = len(V); bot = top + 1
    V = np.vstack([V, [[0, 0, radii[2]], [0, 0, -radii[2]]]])
    for i in range(ring - 2):
        for j in range(seg):
            a, b = i * seg + j, i * seg + (j + 1) % seg
            F.append([a, a + seg, b + seg, b])
    for j in range(seg):
        F.append([top, j, (j + 1) % seg])
        F.append([bot, (ring - 2) * seg + (j + 1) % seg, (ring - 2) * seg + j])
    if rot is not None:
        V = V @ np.asarray(rot).T
    return V + np.asarray(center), F


def capsule(p0, p1, r0, r1, seg=20, ring=8):
    """Tapered capsule between p0 and p1 (radii r0, r1)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = p1 - p0
    L = np.linalg.norm(t); t /= L
    a = np.cross(t, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(t, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(t, a)
    rows = []
    for i in range(ring, 0, -1):            # cap at p0
        th = math.pi / 2 * i / ring
        rows.append((p0 - t * r0 * math.sin(th), r0 * math.cos(th)))
    for i in range(0, 9):
        u = i / 8
        rows.append((p0 + t * L * u, r0 + (r1 - r0) * u))
    for i in range(1, ring + 1):
        th = math.pi / 2 * i / ring
        rows.append((p1 + t * r1 * math.sin(th), r1 * math.cos(th)))
    V, F = [], []
    for c, r in rows:
        for j in range(seg):
            ph = 2 * math.pi * j / seg
            V.append(c + (a * math.cos(ph) + b * math.sin(ph)) * max(r, 1e-5))
    n = len(rows)
    for i in range(n - 1):
        for j in range(seg):
            q0, q1 = i * seg + j, i * seg + (j + 1) % seg
            F.append([q0, q1, q1 + seg, q0 + seg])
    return np.array(V), F


def frame_from(t, up):
    t = np.asarray(t, float); t = t / np.linalg.norm(t)
    n = np.asarray(up, float); n = n - t * np.dot(n, t); n /= np.linalg.norm(n)
    b = np.cross(t, n)
    return t, n, b


def finger_tube(P, U, W, H, n=28, tip_len=None, root_ext=0.009, thumb=False):
    """Lofted glove finger through joints P (4,3) with dorsal vectors U. Sections every ~1.4 mm, joint bulges,
    palmar pads mid-phalanx, rounded glove tip, closed root inside the palm."""
    cents, secs, TT, NN, BB = [], [], [], [], []
    # root extension back into the palm (closed dome, hidden inside)
    t0, n0, b0 = frame_from(P[1] - P[0], U[0])
    for i in range(5, 0, -1):
        u = i / 5
        c = P[0] - t0 * root_ext * u
        k = math.sqrt(max(1 - u * u, 0.05))
        cents.append(c); secs.append(superellipse(n, W[0] * k, H[0] * k)); TT.append(t0); NN.append(n0); BB.append(b0)
    for seg in range(3):
        a, b = P[seg], P[seg + 1]
        L = np.linalg.norm(b - a)
        m = max(4, int(L / 0.0014))
        for i in range(m + (1 if seg == 2 else 0)):
            u = i / m
            c = a + (b - a) * u
            # frame: blend dorsal vectors, tangent from segment (softened at joints)
            up = U[seg] * (1 - u) + U[seg + 1] * u
            t_, n_, b_ = frame_from(b - a, up)
            w = W[seg] + (W[seg + 1] - W[seg]) * u
            h = H[seg] + (H[seg + 1] - H[seg]) * u
            # joint bulge (condyles) near u=0 (except the MCP, covered by the palm) and pad fullness mid-phalanx
            jb = math.exp(-((u) / 0.16) ** 2) * (0.05 if seg > 0 else 0.0) + math.exp(-((u - 1) / 0.16) ** 2) * (0.045 if seg < 2 else 0.0)
            pad = math.sin(math.pi * u) ** 2 * (0.12 if not thumb else 0.10)
            crease = math.exp(-((u) / 0.07) ** 2) * (0.10 if seg > 0 else 0.0)
            sec = superellipse(n, w * (1 + jb), h * (1 + jb * 0.6), ex=2.35, palm_full=pad - crease, dorsal_flat=0.05)
            cents.append(c); secs.append(sec); TT.append(t_); NN.append(n_); BB.append(b_)
    # glove fingertip: rounded, the leather extends ~4 mm past the fingertip, pad-side fuller
    tl = tip_len or H[3] * 0.62
    tE, nE, bE = TT[-1], NN[-1], BB[-1]
    last = cents[-1]
    for i in range(1, 9):
        u = i / 8
        k = math.sqrt(max(1 - u ** 2.2, 0.0))
        c = last + tE * tl * u - nE * H[3] * 0.10 * u
        sec = superellipse(n, W[3] * max(k, 0.04), H[3] * max(k, 0.04), ex=2.2, palm_full=0.08 * k)
        cents.append(c); secs.append(sec); TT.append(tE); NN.append(nE); BB.append(bE)
    return loft(secs, np.array(cents), np.array(TT), np.array(NN), np.array(BB))


# ================================================================================================
# glove sculpt
# ================================================================================================
def glove_parts():
    parts = []
    joints = {}
    for name, f in FINGERS.items():
        P, U, lat = finger_joints(f)
        joints[name] = (P, U)
        parts.append(finger_tube(P, U, f['W'], f['H']))
    TP, TU, tlat = thumb_joints()
    joints['thumb'] = (TP, TU)
    # thumb: metacarpal as a thick capsule embedded in the thenar, then the phalanges as a finger tube from MCP
    parts.append(capsule(TP[0], TP[1], 0.0150, 0.0128))
    parts.append(_thumb_tube(TP, TU, THUMB['W'], THUMB['H']))
    # palm slab: lofted along y, superelliptic section, dorsal arched, from the wrist to under the knuckle row
    ys = np.linspace(-0.004, 0.080, 22)
    cents, secs, T, N, B = [], [], [], [], []
    for y in ys:
        u = (y + 0.004) / 0.084
        w = 0.0655 + 0.0185 * math.sin(min(u * 1.25, 1) * math.pi / 2)      # 6.6 cm at the wrist -> 8.4 cm
        h = 0.0300 + 0.0030 * math.sin(u * math.pi)
        cx = -0.0010 + 0.0015 * u
        sec = superellipse(40, w, h, ex=2.8, palm_full=0.05, dorsal_flat=0.0)
        # dorsal arch: raise the middle of the top a little
        sec[:, 1] += np.where(sec[:, 1] > 0, 0.0025 * np.cos(np.clip(sec[:, 0] / (w / 2), -1, 1) * math.pi / 2), 0)
        cents.append(np.array([cx, y, 0.0005]))
        secs.append(sec)
        T.append(np.array([0, 1.0, 0])); N.append(np.array([0, 0, 1.0])); B.append(np.array([1.0, 0, 0]))
    parts.append(loft(secs, np.array(cents), np.array(T), np.array(N), np.array(B)))
    # metacarpals (knuckle rays) from the wrist region to each MCP: give the back of the hand its ridges
    for name, f in FINGERS.items():
        P, U = joints[name]
        base = np.array([P[0][0] * 0.55, 0.012, 0.002])
        parts.append(capsule(base, P[0] + np.array([0, -0.002, 0.0005]), 0.0112, 0.0118))
    # knuckle row: capsules between neighbouring MCP heads (rounds the distal palm edge along the arc)
    names = list(FINGERS)
    for a, b in zip(names[:-1], names[1:]):
        pa, pb = joints[a][0][0], joints[b][0][0]
        parts.append(capsule(pa, pb, 0.0112, 0.0106))
    # palmar pads under the MCPs (the glove bunches there)
    for a, b in zip(names[:-1], names[1:]):
        pa, pb = joints[a][0][0] + np.array([0, -0.004, -0.0065]), joints[b][0][0] + np.array([0, -0.004, -0.0065])
        parts.append(capsule(pa, pb, 0.0085, 0.0080))
    # thenar (thumb ball) and hypothenar pads
    rz = lambda d: np.array(Matrix.Rotation(math.radians(d), 3, 'Z'))
    parts.append(ellipsoid((-0.0225, 0.0330, -0.0095), (0.0165, 0.0270, 0.0125), rz(-32)))
    parts.append(ellipsoid((0.0275, 0.0370, -0.0080), (0.0130, 0.0310, 0.0110), rz(6)))
    # thumb web (the leather spans between the thumb and the index metacarpal)
    ip = joints['index'][0][0]
    tm = TP[1]
    mid = (ip * 0.45 + tm * 0.55) + np.array([0.004, -0.010, -0.0015])
    dirv = (ip - tm); dirv /= np.linalg.norm(dirv)
    zz = np.array([0, 0, 1.0]); xx = np.cross(dirv, zz); xx /= np.linalg.norm(xx)
    Rw = np.stack([dirv, xx, np.cross(dirv, xx)], 1)
    parts.append(ellipsoid(mid, (0.0200, 0.0120, 0.0062), Rw))
    # wrist gauntlet (goes up inside the jacket cuff)
    ys = np.linspace(-0.075, 0.006, 14)
    cents, secs, T, N, B = [], [], [], [], []
    for y in ys:
        u = (y + 0.075) / 0.081
        w = 0.0640 + 0.0030 * u
        h = 0.0440 - 0.0100 * u
        cents.append(np.array([0.0, y, 0.0]))
        secs.append(superellipse(36, w, h, ex=2.2))
        T.append(np.array([0, 1.0, 0])); N.append(np.array([0, 0, 1.0])); B.append(np.array([1.0, 0, 0]))
    parts.append(loft(secs, np.array(cents), np.array(T), np.array(N), np.array(B)))
    return parts, joints


def _thumb_tube(TP, TU, W, H):
    # a finger tube needs 4 joints; use MCP, IP, (tip - 1 mm), tip: the last segment is tiny, the glove cap rounds it
    P = np.array([TP[1], TP[2], TP[3] - (TP[3] - TP[2]) * 0.15, TP[3]])
    U = np.array([TU[1], TU[2], TU[3], TU[3]])
    return finger_tube(P, U, (W[1], W[2], W[3] * 1.02, W[3]), (H[1], H[2], H[3] * 1.02, H[3]), root_ext=0.012, thumb=True)


def build_glove_union():
    parts, joints = glove_parts()
    V_all, F_all = [], []
    off = 0
    for V, F in parts:
        V_all.append(np.asarray(V))
        F_all.extend([[i + off for i in f] for f in F])
        off += len(V)
    o = mesh_obj('glove_union', np.vstack(V_all), F_all)
    select_only([o])
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    o.data.remesh_voxel_size = VOXEL
    o.data.use_remesh_fix_poles = True
    bpy.ops.object.voxel_remesh()
    # smoothing in physical units: ~2.5 mm fillets where the parts meet (leather bridging the joins)
    m = o.modifiers.new('sm', 'SMOOTH'); m.factor = 0.9; m.iterations = int(round(9 * (0.0007 / VOXEL) ** 2))
    apply_mods(o)
    log('glove union: %d verts' % len(o.data.vertices))
    return o, joints


# ================================================================================================
# armature
# ================================================================================================
BONE_FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']


def build_armature(side, joints):
    S = side
    data = bpy.data.armatures.new('hand_' + S)
    arm = link(bpy.data.objects.new('hand_' + S, data))
    select_only([arm])
    bpy.ops.object.mode_set(mode='EDIT')
    eb = data.edit_bones
    fo = eb.new(S + '_forearm'); fo.head = (0, -FOREARM_LEN, 0); fo.tail = (0, 0, 0); fo.align_roll(Vector((0, 0, 1)))
    hb = eb.new(S + '_hand'); hb.head = (0, 0, 0); hb.tail = (0, 0.07, 0); hb.parent = fo; hb.use_connect = True
    hb.align_roll(Vector((0, 0, 1)))
    for fname in BONE_FINGERS:
        P, U = joints[fname]
        idx0 = 1 if fname == 'thumb' else 0     # thumb bones start at the CMC (P[0]) but the thumb1 bone spans CMC->MCP
        idx0 = 0
        par = hb
        for k in range(3):
            b = eb.new(f'{S}_{fname}{k + 1}')
            b.head = Vector(P[k]); b.tail = Vector(P[k + 1])
            b.parent = par
            b.use_connect = k > 0
            b.align_roll(Vector(U[k] if k < 3 else U[-1]))
            par = b
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm


# ================================================================================================
# poses (pose-basis rotations, degrees). Fingers: (spread about local Z [+ = toward the thumb side for the right
# hand], flex MCP, PIP, DIP about local X [+ = curl toward the palm]). Thumb: per bone (flex X, twist Y, abduct Z).
# ================================================================================================
POSES = {
    'rest': dict(index=(-3, 10, 14, 8), middle=(-0.5, 12, 17, 9), ring=(1.5, 15, 20, 11), pinky=(3, 18, 22, 13),
                 thumb=((0, 0, 0), (6, 0, 0), (8, 0, 0))),
    # power grip round a ~32 x 25 mm oval handle (hatchet); thumb wraps over the index/middle
    'fist': dict(index=(-9, 62, 92, 46), middle=(-1.5, 70, 96, 46), ring=(5.5, 76, 96, 44), pinky=(12, 80, 92, 42),
                 thumb=((18, -22, -30), (34, 0, 0), (30, 0, 0))),
    # hook grip on a ~20 mm bar (jerrycan handle): fingers curled at PIP/DIP, knuckles fairly straight
    'hook': dict(index=(-8, 30, 98, 58), middle=(-1.5, 32, 100, 58), ring=(4.5, 36, 98, 56), pinky=(10, 40, 94, 52),
                 thumb=((24, -30, -18), (10, 0, 0), (12, 0, 0))),
    # reaching out to take something: open, slightly spread
    'open': dict(index=(-1, -2, 6, 4), middle=(-0.5, 0, 7, 4), ring=(-0.5, 2, 8, 5), pinky=(-2, 4, 9, 6),
                 thumb=((-8, 6, 16), (0, 0, 0), (4, 0, 0))),
    # flat support (plank edge / under the jerrycan): fingers together, gently curled
    'flat': dict(index=(-8, 18, 22, 10), middle=(-1.5, 20, 24, 10), ring=(4.5, 22, 26, 12), pinky=(10, 26, 28, 12),
                 thumb=((-4, -6, 12), (8, 0, 0), (10, 0, 0))),
    # loose grip (supporting hand on the hatchet handle, holding the plank edge)
    'grip': dict(index=(-8, 50, 74, 38), middle=(-1.5, 56, 80, 38), ring=(4.5, 60, 80, 36), pinky=(11, 64, 78, 34),
                 thumb=((14, -16, -22), (26, 0, 0), (24, 0, 0))),
}


# grip sockets (right-hand rest frame = hand-bone local frame, metres). The fingers of the named pose are closed
# onto this cylinder by solve_grips(); the engine attaches items so their handle axis lies on it.
#   p: point on the axis, d: axis direction (toward the pinky side), r: cylinder radius (+0.5 mm glove clearance)
GRIPS = {
    # hatchet: 32 x 25 mm oval ash handle, diagonal across the palm from the index base to the hypothenar
    'fist': dict(p=(0.0, 0.0815, -0.0330), d=(1.0, -0.30, 0.02), r=0.0150, max=(80, 105, 70), thumb_r=0.0185),
    # jerrycan: 17 mm steel carrying bar hooked in the fingers under the middle phalanges
    'hook': dict(p=(0.0, 0.1125, -0.0215), d=(1.0, -0.22, 0.0), r=0.0092, max=(12, 115, 85), thumb_r=0.020, wrap=False),
    # loose grip round a ~44 mm edge (plank edge, supporting hand)
    'grip': dict(p=(0.0, 0.090, -0.0420), d=(1.0, -0.22, 0.0), r=0.0220, max=(70, 95, 65), thumb_r=0.0255),
}


def _seg_line_dist(A, B, C, e, n=7):
    best = 1e9
    for i in range(n):
        p = A + (B - A) * (i / (n - 1))
        v = p - C
        best = min(best, np.linalg.norm(v - e * np.dot(v, e)))
    return best


def solve_grips(arm, S='R'):
    """Close each finger joint by joint (MCP, then PIP, then DIP, 1.5 deg steps) until any phalanx distal of that
    joint would pass into the grip cylinder, using Blender's own FK on the pose bones. Writes POSES[name]."""
    for name, G in GRIPS.items():
        base = POSES[name]
        C = np.array(G['p']); e = np.array(G['d'], float); e /= np.linalg.norm(e)
        pose = {k: (tuple(v) if k != 'thumb' else [tuple(t) for t in v]) for k, v in base.items()}
        def setp(pz):
            qs = pose_quats(pz, S)
            for pb in arm.pose.bones:
                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = qs.get(pb.name, Quaternion())
            bpy.context.view_layer.update()
        for fname in ('index', 'middle', 'ring', 'pinky', 'thumb'):
            if fname == 'thumb' and G.get('wrap', True):
                _thumb_search(arm, S, pose, G, C, e, setp)
                continue
            if fname == 'thumb':
                continue
            if fname == 'thumb':
                Hs = THUMB['H'][1:]
                bones = [f'{S}_thumb2', f'{S}_thumb3']
                R = G['thumb_r']
                ang = [pose['thumb'][1][0], pose['thumb'][2][0]]
                mx = (60, 70)
            else:
                Hs = FINGERS[fname]['H']
                bones = [f'{S}_{fname}{k + 1}' for k in range(3)]
                R = G['r']
                ang = [0.0, 0.0, 0.0]
                mx = G['max']
            def pen(kfrom):
                for j in range(kfrom, len(bones)):
                    pb = arm.pose.bones[bones[j]]
                    A = np.array(pb.head); B = np.array(pb.tail)
                    if j == len(bones) - 1:        # glove tip beyond the last joint
                        B = B + (B - A) / max(np.linalg.norm(B - A), 1e-6) * Hs[-1] * 0.55
                    rr = 0.5 * (Hs[j] + Hs[j + 1]) * 0.5 if j + 1 < len(Hs) else Hs[j] * 0.5
                    if _seg_line_dist(A, B, C, e) < R + rr:
                        return True
                return False
            def write():
                if fname == 'thumb':
                    t = pose['thumb']
                    pose['thumb'] = [t[0], (ang[0], t[1][1], t[1][2]), (ang[1], t[2][1], t[2][2])]
                else:
                    pose[fname] = (pose[fname][0], ang[0], ang[1], ang[2])
            for k in range(len(bones)):
                a = ang[k]
                while a < mx[k]:
                    ang[k] = a + 1.5
                    write(); setp(pose)
                    if pen(k):
                        ang[k] = a
                        break
                    a += 1.5
                write(); setp(pose)
        POSES[name] = pose
        log('grip %s: ' % name + ', '.join('%s %s' % (k, np.round(np.array(v, float).ravel(), 0).tolist()) for k, v in pose.items()))
    for pb in arm.pose.bones:
        pb.rotation_quaternion = Quaternion()
    bpy.context.view_layer.update()


def _thumb_search(arm, S, pose, G, C, e, setp):
    """Wrap the thumb: coarse search over the CMC rotation (flex, twist, abduction); for each, close the MCP and IP
    onto the handle (+ the fingers wrapped over it) and score the thumb pad's distance to the side of the index finger's
    middle phalanx (where a hammer-grip thumb presses), penalising penetration of the index finger."""
    pb2 = arm.pose.bones
    idx2 = pb2[f'{S}_index2']
    best = None
    Ht = THUMB['H']
    for fx in (0, 15, 30, 45, 60):
        for ty in (-95, -75, -55, -35, -15, 5):
            for az in (-60, -40, -20, 0, 20):
                t = [(fx, ty, az), (0.0, 0, 0), (0.0, 0, 0)]
                pose['thumb'] = t
                setp(pose)
                a = [0.0, 0.0]
                for k, bn in enumerate((f'{S}_thumb2', f'{S}_thumb3')):
                    while a[k] < (65, 75)[k]:
                        a[k] += 3.0
                        pose['thumb'] = [t[0], (a[0], 0, 0), (a[1], 0, 0)]
                        setp(pose)
                        hit = False
                        for j, bj in enumerate((f'{S}_thumb2', f'{S}_thumb3')):
                            if j < k:
                                continue
                            A = np.array(pb2[bj].head); B = np.array(pb2[bj].tail)
                            if _seg_line_dist(A, B, C, e) < G['thumb_r'] + Ht[j + 2] * 0.5:
                                hit = True
                        if hit:
                            a[k] -= 3.0
                            break
                pose['thumb'] = [t[0], (a[0], 0, 0), (a[1], 0, 0)]
                setp(pose)
                # thumb pad: 60 % along the distal phalanx, offset to the palmar side
                A = np.array(pb2[f'{S}_thumb3'].head); B = np.array(pb2[f'{S}_thumb3'].tail)
                pad = A + (B - A) * 0.6
                # target: the thumb-side face of the index middle phalanx
                iA = np.array(idx2.head); iB = np.array(idx2.tail)
                side = np.array(idx2.x_axis) * (-1.0 if S == 'R' else 1.0)
                tgt = (iA + iB) * 0.5 + side * 0.013
                cost = np.linalg.norm(pad - tgt)
                # penetration of the index finger (thumb distal vs index segments)
                for bn in (f'{S}_index1', f'{S}_index2', f'{S}_index3'):
                    q = pb2[bn]
                    dmin = min(np.linalg.norm(pad - (np.array(q.head) + (np.array(q.tail) - np.array(q.head)) * u)) for u in (0.0, 0.5, 1.0))
                    if dmin < 0.017:
                        cost += (0.017 - dmin) * 4
                if best is None or cost < best[0]:
                    best = (cost, [t[0], (a[0], 0, 0), (a[1], 0, 0)])
    pose['thumb'] = best[1]
    setp(pose)
    log('  thumb search %s: cost %.4f' % (str(best[1]), best[0]))


def store_sockets(arm, S):
    out = {}
    for name, G in GRIPS.items():
        p = list(G['p']); d = np.array(G['d'], float); d /= np.linalg.norm(d)
        if S == 'L':
            p[0] = -p[0]; d[0] = -d[0]
        out[name] = {'p': [round(v, 5) for v in p], 'd': [round(float(v), 5) for v in d], 'r': G['r']}
    arm['sockets'] = json.dumps(out, separators=(',', ':'))


def pose_quats(pose, side):
    """{bone: (w,x,y,z)} pose-basis quaternions for a pose table entry."""
    out = {}
    mir = side == 'L'
    for fname in BONE_FINGERS:
        v = pose[fname]
        for k in range(3):
            if fname == 'thumb':
                fx, ty, az = v[k]
            else:
                fx = v[1 + k]; ty = 0.0; az = v[0] if k == 0 else 0.0
            # flexion toward the palm is rotation about local -X
            e = Euler((math.radians(-fx), math.radians(ty), math.radians(az)), 'YZX')
            q = e.to_quaternion()
            if mir:
                q = Quaternion((q.w, q.x, -q.y, -q.z))
            out[f'{side}_{fname}{k + 1}'] = q
    return out


def apply_pose(arm, side, name):
    qs = pose_quats(POSES[name], side)
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = qs.get(pb.name, Quaternion())
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()


def store_poses(arm, side):
    table = {}
    for name, pose in POSES.items():
        qs = pose_quats(pose, side)
        table[name] = {b: [round(q.x, 5), round(q.y, 5), round(q.z, 5), round(q.w, 5)] for b, q in qs.items()}
    arm['poses'] = json.dumps(table, separators=(',', ':'))


# ================================================================================================
# main (clay stage)
# ================================================================================================
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'


def clay_material():
    m = bpy.data.materials.new('clay')
    m.use_nodes = True
    bs = m.node_tree.nodes['Principled BSDF']
    bs.inputs['Base Color'].default_value = (0.35, 0.25, 0.17, 1)
    bs.inputs['Roughness'].default_value = 0.5
    return m


def setup_world(strength=1.0):
    sc = bpy.context.scene
    w = bpy.data.worlds.new('w')
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    env = nt.nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(HDRI)
    bg = nt.nodes['Background']
    bg.inputs['Strength'].default_value = strength
    nt.links.new(env.outputs['Color'], bg.inputs['Color'])


def setup_cycles(samples, gpu=None):
    gpu = ARGS['gpu'] if gpu is None else gpu      # CPU by default: Metal adds ~1.6 GB of unified memory
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


def render_preview(path, cam_loc, target, lens=45, res=(720, 540), samples=48, objs_hide=()):
    sc = bpy.context.scene
    setup_cycles(samples)
    sc.cycles.use_denoising = True
    cam = bpy.data.objects.get('prev_cam')
    if cam is None:
        cam = link(bpy.data.objects.new('prev_cam', bpy.data.cameras.new('prev_cam')))
    cam.data.lens = lens
    cam.data.clip_start = 0.01
    cam.location = cam_loc
    d = Vector(target) - Vector(cam_loc)
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.filepath = path
    for o in objs_hide:
        o.hide_render = True
    bpy.ops.render.render(write_still=True)
    for o in objs_hide:
        o.hide_render = False
    log('preview', os.path.relpath(path, ROOT))


def skin_glove(glove, arm, S):
    """Heat (bone-glow) weights for the glove over the hand + finger bones. The heat solver fails at hand scale, so it
    runs on a x10 copy of the rig; the forearm bone is excluded (the sleeve is rigid to it)."""
    arm.data.bones[S + '_forearm'].use_deform = False

    def rescale(k):
        glove.data.transform(Matrix.Scale(k, 4))
        glove.data.update()
        select_only([arm])
        bpy.ops.object.mode_set(mode='EDIT')
        for b in arm.data.edit_bones:
            b.head = b.head * k
            b.tail = b.tail * k
        bpy.ops.object.mode_set(mode='OBJECT')
    rescale(10.0)
    select_only([glove, arm], arm)
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    rescale(0.1)
    arm.data.bones[S + '_forearm'].use_deform = True
    vg = [g.name for g in glove.vertex_groups]
    me = glove.data
    # the glove's gauntlet wraps the wrist and forearm: it follows the forearm, blending into the hand across the
    # wrist crease (y -22 mm .. +14 mm), so a bent wrist never swings the cuff out of the jacket sleeve
    fore_g = glove.vertex_groups.get(S + '_forearm') or glove.vertex_groups.new(name=S + '_forearm')
    gi = {g.index: g for g in glove.vertex_groups}
    for v in me.vertices:
        y = v.co.y
        wf = min(max((0.014 - y) / 0.036, 0.0), 1.0)
        wf = wf * wf * (3 - 2 * wf)
        if wf <= 0:
            continue
        for g in v.groups:
            if gi[g.group].name != S + '_forearm':
                gi[g.group].add([v.index], g.weight * (1 - wf), 'REPLACE')
        fore_g.add([v.index], wf, 'REPLACE')
    # vertices the solver left unweighted (rare) -> the hand bone
    hand_g = glove.vertex_groups.get(S + '_hand') or glove.vertex_groups.new(name=S + '_hand')
    lone = [v.index for v in me.vertices if sum(g.weight for g in v.groups) < 1e-4]
    if lone:
        hand_g.add(lone, 1.0, 'REPLACE')
    log('skinned %s: %d groups, %d unweighted fixed' % (glove.name, len(vg), len(lone)))


# ================================================================================================
# numpy noise
# ================================================================================================
def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xffffff).astype(np.float64) / float(0xffffff)


def vnoise(P, scale, seed=0):
    """3D value noise in [-1, 1] at feature size `scale` (m)."""
    Q = np.asarray(P, np.float64) / scale
    i0 = np.floor(Q).astype(np.int64)
    f = Q - i0
    u = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1]) * (u[:, 2] if dz else 1 - u[:, 2])
                out = out + w * _hash3(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz, seed)
    return out * 2 - 1


def fbm(P, scale, oct=3, seed=0):
    a, s, tot, amp = 0.0, scale, 0.0, 1.0
    for k in range(oct):
        a = a + vnoise(P, s, seed + k * 17) * amp
        tot += amp
        amp *= 0.5
        s *= 0.5
    return a / tot


def mesh_arrays(o):
    me = o.data
    n = len(me.vertices)
    V = np.zeros(n * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
    N = np.zeros(n * 3); me.vertices.foreach_get('normal', N); N = N.reshape(-1, 3)
    return V, N


def set_float_attr(o, name, vals):
    me = o.data
    a = me.attributes.get(name) or me.attributes.new(name, 'FLOAT', 'POINT')
    a.data.foreach_set('value', np.asarray(vals, np.float32).ravel())


# ================================================================================================
# glove detail pass (high poly): wrinkles, creases, seams, lumpy leather + attributes for the bake material
# ================================================================================================
def finger_coords(V, joints):
    """Per vertex: nearest finger (index into names or -1 = palm), arc coordinate from the MCP (m), angle about the
    finger axis (rad, 0 = dorsal, +-pi/2 lateral, pi palmar), signed distance from the finger's lateral plane (m)."""
    names = ['index', 'middle', 'ring', 'pinky', 'thumb']
    n = len(V)
    best = np.full(n, np.inf)
    fid = np.full(n, -1)
    arc = np.zeros(n); th = np.zeros(n); dpl = np.zeros(n); rad = np.zeros(n)
    for fi, name in enumerate(names):
        P, U = joints[name]
        if name == 'thumb':
            P = P[1:]; U = U[1:]              # thumb phalanges (MCP -> tip); the metacarpal counts as palm
            P = np.vstack([P, P[-1:]])
            U = np.vstack([U, U[-1:]])
        W = THUMB['W'][1:] if name == 'thumb' else FINGERS[name]['W']
        acc = 0.0
        for k in range(3):
            a, b = P[k], P[k + 1]
            d = b - a
            L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            d = d / L
            last = (k == 2) or (name == 'thumb' and k == 1)
            t = (V - a) @ d
            tc = np.clip(t, 0 if k > 0 else 0.004, L + (0.012 if last else 0.0))
            q = a + np.outer(tc, d)
            r = V - q
            dist = np.linalg.norm(r, axis=1)
            rr = 0.5 * (W[min(k, 3)] if name != 'thumb' else THUMB['W'][1 + k])
            nd = dist / rr
            # the finger owns a vertex only beyond its MCP (palm otherwise)
            ok = (t > (0.004 if k == 0 else -1)) & (nd < best) & (nd < 1.6)
            up = U[k] - d * np.dot(U[k], d); up /= np.linalg.norm(up)
            lat = np.cross(d, up)
            best = np.where(ok, nd, best)
            fid = np.where(ok, fi, fid)
            arc = np.where(ok, acc + t, arc)
            th = np.where(ok, np.arctan2(r @ lat, r @ up), th)
            dpl = np.where(ok, r @ up, dpl)
            rad = np.where(ok, dist, rad)
            acc += L
    return fid, arc, th, dpl, rad


def glove_details(o, joints):
    V, N = mesh_arrays(o)
    n = len(V)
    fid, arc, th, dpl, rad = finger_coords(V, joints)
    mm = 1000.0
    disp = np.zeros(n)
    fin = fid >= 0
    dors = np.clip(np.cos(th), 0, 1)             # 1 on the back of the finger
    palmar = np.clip(-np.cos(th), 0, 1)
    thdeg = np.degrees(th)
    outB = np.full(n, 99.0)                      # signed distance (mm) to a stitched outseam
    outS = np.zeros(n)                           # mm along that seam
    inseam = np.full(n, 99.0)                    # distance (mm) to an inseam (groove only)
    wear = np.zeros(n)
    names = ['index', 'middle', 'ring', 'pinky', 'thumb']
    for fi, name in enumerate(names):
        sel = fid == fi
        if not sel.any():
            continue
        if name == 'thumb':
            L = THUMB['L'][1:]
        else:
            L = FINGERS[name]['L']
        jarcs = [0.0, L[0], L[0] + L[1]] if name != 'thumb' else [0.0, L[0]]
        a = arc[sel] * mm
        tdeg = thdeg[sel]
        dd = dors[sel]; pp = palmar[sel]
        dz = np.zeros(sel.sum())
        for j, ja in enumerate(jarcs):
            delta = a - ja * mm
            # dorsal crescents: 3-5 wrinkles bowing toward the tip on the sides
            curve = 1.6 * (tdeg / 90.0) ** 2
            amp = (0.10 if j == 0 else 0.26) * dd ** 1.5
            span = 2.4 if j == 0 else 2.1
            for m_ in range(-2, 3):
                env = math.exp(-(m_ / 2.2) ** 2)
                dz -= amp * env * np.exp(-((delta - m_ * span - curve + (1.5 if j == 0 else 0)) / 0.42) ** 2)
            # palmar flexion creases (two lines) at the PIP / DIP / thumb IP
            if j > 0 or name == 'thumb':
                dz -= 0.30 * pp ** 2 * (np.exp(-((delta - 1.2) / 0.40) ** 2) + 0.7 * np.exp(-((delta + 1.6) / 0.38) ** 2))
        # knuckle over the MCP / PIP joints: worn, slightly raised
        for j, ja in enumerate(jarcs[:2]):
            wear[sel] = np.maximum(wear[sel], dd ** 2 * np.exp(-((a - ja * mm) / 6.0) ** 2))
        # fingertip wear + pad
        tip = (sum(L) * mm)
        wear[sel] = np.maximum(wear[sel], np.clip((a - (tip - 9)) / 9, 0, 1) * (0.5 + 0.5 * pp))
        disp[sel] += dz / mm
        # inseams on the finger sides / over the tip (lateral plane of the finger)
        if name != 'thumb':
            inseam[sel] = np.abs(dpl[sel]) * mm + np.where(np.abs(tdeg) < 40, 50, 0)
        else:
            inseam[sel] = np.where(tdeg > 0, np.abs(dpl[sel]) * mm, 99) + np.where(np.abs(tdeg) < 40, 50, 0)
            # keystone thumb: stitched loop round the thumb base, 5 mm past the MCP
            ring = a - 5.0
            outB[sel] = np.where(np.abs(ring) < np.abs(outB[sel]), ring, outB[sel])
            outS[sel] = th[sel] * rad[sel] * mm
    # finger inseam groove + welts
    g = np.clip(inseam, 0, 99)
    disp += (-0.34 * np.exp(-(g / 0.45) ** 2) + 0.16 * np.exp(-((g - 1.15) / 0.55) ** 2)) / mm * fin
    # ---- palm (vertices not owned by a finger)
    pal = ~fin
    x, y, z = V[:, 0] * mm, V[:, 1] * mm, V[:, 2] * mm
    # palm patch outline (palmar side): rounded rectangle, stitched
    cx, cy, hx, hy, cr = 5.0, 41.0, 35.0, 33.0, 12.0
    qx = np.abs(x - cx) - (hx - cr); qy = np.abs(y - cy) - (hy - cr)
    sd = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - cr
    ppal = pal & (z < -3.0) & (y > -4)
    use = ppal & (np.abs(sd) < np.abs(outB))
    outB = np.where(use, -sd, outB)
    outS = np.where(use, np.arctan2(y - cy, x - cx) * 38.0, outS)
    # knuckle strap (dorsal): band over the MCP heads following the knuckle arc
    mcx = np.array([FINGERS[k]['mcp'][0] for k in ('index', 'middle', 'ring', 'pinky')]) * mm
    mcy = np.array([FINGERS[k]['mcp'][1] for k in ('index', 'middle', 'ring', 'pinky')]) * mm
    arcy = np.interp(x, mcx, mcy)
    dpal = pal & (z > 4.0)
    for edge, sgn in ((arcy - 22.0, 1), (arcy + 5.0, -1)):
        e = (y - edge) * sgn
        use = dpal & (np.abs(e) < np.abs(outB)) & (x > -42) & (x < 42)
        outB = np.where(use, e, outB)
        outS = np.where(use, x, outS)
    # patch/strap step: the patch side is one leather layer (~0.9 mm) proud
    step = np.clip(outB / 0.6, 0, 1) * np.exp(-np.maximum(outB - 2.5, 0) / 50.0)
    grooveB = np.exp(-(outB / 0.4) ** 2)
    disp += (0.22 * step * (np.abs(outB) < 60) - 0.26 * grooveB) / mm
    # palm creases (distal transverse + thenar) and the back-of-hand wrinkles over the MCP knuckles
    distal = np.exp(-((y - (arcy - 12.0)) / 0.8) ** 2) * (z < -6)
    thenar = np.exp(-((np.hypot(x + 38, y - 12) - 34.0) / 0.8) ** 2) * (z < -6) * (y > 8) * (y < 70)
    disp -= (0.35 * distal + 0.28 * thenar) * pal / mm
    wear = np.maximum(wear, pal * np.clip((z - 8) / 6, 0, 1) * np.exp(-((y - arcy) / 7.0) ** 2))
    # wrist: gathered elastic shirring (mostly under the jacket cuff)
    shir = (y < 2) & (y > -30)
    disp += shir * 0.25 * np.sin(np.arctan2(z, x) * 22) * np.exp(-((y + 12) / 9) ** 2) / mm
    # lumpy leather everywhere
    disp += (0.10 * fbm(V, 0.009, 3, 11) + 0.05 * vnoise(V, 0.0028, 5)) / mm
    V2 = V + N * disp[:, None]
    o.data.vertices.foreach_set('co', V2.ravel())
    o.data.update()
    # attributes for the bake material
    mud = np.clip(0.5 * fbm(V, 0.02, 3, 21) + 0.35 * palmar * fin + 0.45 * (pal & (z < -4)) - 0.1, 0, 1)
    set_float_attr(o, 'outB', np.clip(outB, -50, 50))
    set_float_attr(o, 'outS', outS)
    set_float_attr(o, 'inseam', np.clip(inseam, 0, 50))
    set_float_attr(o, 'wear', np.clip(wear, 0, 1))
    set_float_attr(o, 'mud', mud)
    set_float_attr(o, 'patch', np.clip(step, 0, 1) * (np.abs(outB) < 60))
    log('glove details: %d verts, disp %.2f..%.2f mm' % (n, disp.min() * mm, disp.max() * mm))


# ================================================================================================
# sleeve (rain jacket forearm) + cuff + velcro tab
# ================================================================================================
Y_CUFF = -0.024        # cuff edge (2.4 cm behind the wrist crease)
Y_END = -0.340


def sleeve_radius(y, th):
    """Outer sleeve section (semi-axes) at arm coordinate y (m, negative toward the elbow) and angle th (0 = dorsal)."""
    u = np.clip((Y_CUFF - y) / (Y_CUFF - Y_END), 0, 1)
    ax = 0.0372 + 0.0180 * np.sin(np.minimum(u * 1.6, 1) * np.pi / 2) + 0.004 * u
    az = 0.0290 + 0.0200 * np.sin(np.minimum(u * 1.5, 1) * np.pi / 2) + 0.004 * u
    return ax, az


def sleeve_folds(y, th, fine):
    """Radial offset (m) of the fabric: gathered ruffles behind the cinched cuff, a few long compression folds."""
    u = (Y_CUFF - y)                                   # m from the cuff edge
    g = np.exp(-np.maximum(u - 0.012, 0) / 0.045) * (u > 0.010)
    ruffle = 0.0042 * g * np.sin(th * 7 + 0.8 * np.sin(u * 60)) * (0.6 + 0.4 * np.sin(th * 3 + 1.3))
    # long compression folds spiralling along the forearm
    long_ = 0.0036 * np.sin(u * 38 + th * 1.2) * np.exp(-((u - 0.16) / 0.10) ** 2)
    long_ += 0.0020 * np.sin(u * 55 - th * 2.0 + 1.0) * np.exp(-((u - 0.09) / 0.05) ** 2) * np.cos(th - 0.6) ** 2
    out = ruffle + long_
    if fine:
        P = np.stack([np.cos(th) * 0.05, y, np.sin(th) * 0.05], -1).reshape(-1, 3)
        out = out + (0.0007 * fbm(P, 0.018, 3, 31)).reshape(np.shape(th)) + (0.00025 * vnoise(P, 0.004, 33)).reshape(np.shape(th))
    return out


def build_sleeve(name, n_th, n_y, fine):
    """Tube from the cuff edge to the elbow end (closed dome there); the cuff rolls inward into a short lining so the
    opening has real thickness. Returns object; per-vertex attrs: 'hem' (mm from the cuff edge), 'su', 'sv' (mm)."""
    ths = np.linspace(0, 2 * np.pi, n_th, endpoint=False) + np.pi     # seam on the palmar side (th = pi)
    # rows: inner lining (short, inside the cuff), rolled edge, outer tube, elbow dome
    rows = []
    for k in range(4):              # lining: 14 mm inside, 2.5 mm inset
        f = k / 3
        rows.append(('in', Y_CUFF - 0.014 * (1 - f), -0.0026, -18.0 + 14.0 * f))
    for k in range(1, 6):           # rolled hem edge (half torus, r 1.3 mm)
        a = math.pi * k / 6
        rows.append(('roll', Y_CUFF + 0.0013 * math.sin(a), -0.0013 - 0.0013 * math.cos(a), -4.0 + 4.0 * k / 6))
    ys = Y_CUFF - (np.linspace(0, 1, n_y) ** 1.35) * (Y_CUFF - Y_END)
    for y in ys:
        rows.append(('out', y, 0.0, (Y_CUFF - y) * 1000.0))
    for k in range(1, 5):
        rows.append(('dome', Y_END - 0.012 * math.sin(math.pi / 2 * k / 5), k / 5, (Y_CUFF - Y_END) * 1000.0 + 12.0 * k / 5))
    V, HEM, SV = [], [], []
    for kind, y, extra, vmm in rows:
        ax, az = sleeve_radius(np.array(y), ths)
        if kind == 'dome':
            sc = math.cos(math.pi / 2 * extra)
            off = 0.0
        else:
            sc = 1.0
            off = extra if kind in ('in', 'roll') else 0.0
        fold = sleeve_folds(np.full_like(ths, y), ths, fine) if kind == 'out' else 0.0
        # the cuff band (first 16 mm) is stiffer: folds fade in behind it
        if kind == 'out':
            band = np.clip((Y_CUFF - y - 0.012) / 0.012, 0, 1)
            fold = fold * band
            # cuff band slightly proud (double layer) with a soft step
            off = 0.0011 * (1 - band)
        r_x = (ax + off + fold) * sc
        r_z = (az + off + fold) * sc
        V.append(np.stack([np.sin(ths) * r_x, np.full_like(ths, y), np.cos(ths) * r_z], -1))
        hv = vmm
        HEM.append(np.full_like(ths, hv))
    V = np.array(V)            # (rows, n_th, 3)
    R = V.shape[0]
    # x = sin(th) * rx: th measured from dorsal (+z) toward +x
    Vf = V.reshape(-1, 3)
    F = []
    for i in range(R - 1):
        for j in range(n_th):
            a, b = i * n_th + j, i * n_th + (j + 1) % n_th
            F.append([a, a + n_th, b + n_th, b])
    # close the elbow end
    c = len(Vf)
    Vf = np.vstack([Vf, [[0, Y_END - 0.012, 0]]])
    for j in range(n_th):
        F.append([c, (R - 1) * n_th + (j + 1) % n_th, (R - 1) * n_th + j])
    # close the lining start (inside the sleeve, never seen)
    c2 = len(Vf)
    Vf = np.vstack([Vf, [[0, Y_CUFF - 0.016, 0]]])
    for j in range(n_th):
        F.append([c2, j, (j + 1) % n_th])
    o = mesh_obj(name, Vf, F)
    hem = np.concatenate([np.concatenate(HEM), [400.0, -20.0]])
    set_float_attr(o, 'hem', hem)
    # parametric UV (u around, v along), stored per loop later
    o['rows'] = R
    o['nth'] = n_th
    return o


def build_tab(name, fine):
    """Velcro cuff tab: a 64 x 25 mm strap lying on the cuff, centred 40 deg toward the pinky side, 2.2 mm thick."""
    nu, nv = (48, 18) if fine else (10, 3)
    th0, th1 = math.radians(-2), math.radians(84)
    y0, y1 = Y_CUFF - 0.0035, Y_CUFF - 0.0285
    top, bot = [], []
    for i in range(nv + 1):
        y = y0 + (y1 - y0) * i / nv
        rt, rb = [], []
        for j in range(nu + 1):
            th = th0 + (th1 - th0) * j / nu
            ax, az = sleeve_radius(np.array(Y_CUFF - 0.004), np.array(th))
            ax += 0.0011; az += 0.0011
            # rounded strap end at th1 (the free end), slight lift at the free end
            lift = 0.0006 * (j / nu) ** 3
            rt.append((math.sin(th) * (ax + 0.0022 + lift), y, math.cos(th) * (az + 0.0022 + lift)))
            rb.append((math.sin(th) * (ax + 0.0002), y, math.cos(th) * (az + 0.0002)))
        top.append(rt); bot.append(rb)
    top = np.array(top).reshape(-1, 3); bot = np.array(bot).reshape(-1, 3)
    V = np.vstack([top, bot])
    W = nu + 1
    off = len(top)
    F = []
    for i in range(nv):
        for j in range(nu):
            a = i * W + j
            F.append([a, a + 1, a + W + 1, a + W])
            F.append([off + a, off + a + W, off + a + W + 1, off + a + 1])
    for j in range(nu):          # long sides
        F.append([j, off + j, off + j + 1, j + 1])
        a = nv * W + j
        F.append([a, a + 1, off + a + 1, off + a])
    for i in range(nv):          # ends
        a = i * W
        F.append([a, a + W, off + a + W, off + a])
        a = i * W + nu
        F.append([a, off + a, off + a + W, a + W])
    o = mesh_obj(name, V, F)
    return o


def sleeve_uv(o):
    """Cylindrical UVs in metres (u = arc round the arm, v = distance along it); the seam is on the palmar side."""
    me = o.data
    if not me.uv_layers:
        me.uv_layers.new(name='UVMap')
    uvl = me.uv_layers.active.data
    V, _ = mesh_arrays(o)
    th = np.arctan2(V[:, 0], V[:, 2])                      # 0 = dorsal
    uu = ((th - np.pi) % (2 * np.pi)) / (2 * np.pi)
    circ = 2 * np.pi * 0.047                               # mean circumference (m)
    hem = np.zeros(len(V)); me.attributes['hem'].data.foreach_get('value', hem)
    vv = (hem + 20.0) / 1000.0
    for poly in me.polygons:
        us = [uu[me.loops[li].vertex_index] for li in poly.loop_indices]
        if max(us) - min(us) > 0.5:
            us = [x + 1.0 if x < 0.5 else x for x in us]
        for li, uval in zip(poly.loop_indices, us):
            vi = me.loops[li].vertex_index
            uvl[li].uv = (uval * circ, float(vv[vi]))


# ================================================================================================
# bake materials (procedural Cycles networks on the high-poly meshes)
# ================================================================================================
def load_image(path, noncolor=False):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = 'Non-Color' if noncolor else 'sRGB'
    return im


def tex(name, kind):
    for ext in ('jpg', 'png'):
        p = os.path.join(TEX, name, f'{kind}.{ext}')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{name}/{kind}')


class NB:
    """Tiny node-graph builder. Every helper returns an output socket."""
    def __init__(self, mat):
        mat.use_nodes = True
        self.m = mat
        self.nt = mat.node_tree
        for n in list(self.nt.nodes):
            self.nt.nodes.remove(n)
        self.out = self.nt.nodes.new('ShaderNodeOutputMaterial')
        self.passes = {}

    def node(self, t, **kw):
        n = self.nt.nodes.new(t)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def _in(self, sock, val):
        if hasattr(val, 'is_output'):
            self.nt.links.new(val, sock)
        elif val is not None:
            sock.default_value = val

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        if c is not None:
            self._in(n.inputs[2], c)
        return n.outputs[0]

    def attr(self, name):
        n = self.node('ShaderNodeAttribute', attribute_name=name)
        return n.outputs['Fac']

    def mix(self, fac, a, b):
        n = self.node('ShaderNodeMix', data_type='RGBA', blend_type='MIX')
        self._in(n.inputs['Factor'], fac)
        self._in(n.inputs[6], a)
        self._in(n.inputs[7], b)
        return n.outputs[2]

    def mixf(self, fac, a, b):
        n = self.node('ShaderNodeMix', data_type='FLOAT')
        self._in(n.inputs['Factor'], fac)
        self._in(n.inputs[2], a)
        self._in(n.inputs[3], b)
        return n.outputs[0]

    def rgb(self, c):
        n = self.node('ShaderNodeRGB')
        n.outputs[0].default_value = (c[0], c[1], c[2], 1.0)
        return n.outputs[0]

    def mul_rgb(self, a, b):
        n = self.node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
        n.inputs['Factor'].default_value = 1.0
        self._in(n.inputs[6], a)
        self._in(n.inputs[7], b)
        return n.outputs[2]

    def img(self, path, noncolor, vec, box=True):
        n = self.node('ShaderNodeTexImage')
        n.image = load_image(path, noncolor)
        if box:
            n.projection = 'BOX'
            n.projection_blend = 0.3
        self._in(n.inputs['Vector'], vec)
        return n

    def objcoord(self, scale):
        tc = self.node('ShaderNodeTexCoord')
        mp = self.node('ShaderNodeMapping')
        self.nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
        mp.inputs['Scale'].default_value = (scale, scale, scale)
        return mp.outputs[0]

    def smooth(self, e0, e1, x):
        n = self.node('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP', clamp=True)
        self._in(n.inputs['Value'], x)
        n.inputs['From Min'].default_value = e0
        n.inputs['From Max'].default_value = e1
        return n.outputs['Result']

    def combine(self, r, g, b):
        n = self.node('ShaderNodeCombineColor')
        self._in(n.inputs[0], r); self._in(n.inputs[1], g); self._in(n.inputs[2], b)
        return n.outputs[0]

    def luminance(self, col):
        n = self.node('ShaderNodeRGBToBW')
        self._in(n.inputs[0], col)
        return n.outputs[0]

    def finish(self, albedo, rough, normal, ao_dist=0.012):
        """Store the three bake branches; set_pass() wires one of them to the output."""
        ao = self.node('ShaderNodeAmbientOcclusion', only_local=True, samples=24)
        ao.inputs['Distance'].default_value = ao_dist
        self._in(ao.inputs['Normal'], normal)
        em_a = self.node('ShaderNodeEmission'); self._in(em_a.inputs['Color'], albedo)
        data = self.combine(ao.outputs['AO'], rough, 0.0)
        em_d = self.node('ShaderNodeEmission'); self._in(em_d.inputs['Color'], data)
        bs = self.node('ShaderNodeBsdfPrincipled')
        self._in(bs.inputs['Base Color'], albedo)
        self._in(bs.inputs['Roughness'], rough)
        self._in(bs.inputs['Normal'], normal)
        self.passes = {'albedo': em_a.outputs[0], 'data': em_d.outputs[0], 'normal': bs.outputs[0], 'preview': bs.outputs[0]}
        self.set_pass('preview')

    def set_pass(self, which):
        self.nt.links.new(self.passes[which], self.out.inputs['Surface'])


MATS = []


def glove_material():
    """Wet tan grain-leather work glove. Albedo: dry tan cowhide ~0.30 linear luminance -> soaked ~0.14 (water fills
    the grain and cuts the diffuse scatter by ~half); roughness 0.62 dry -> 0.40 wet, 0.30 soaked, scuffed knuckles and
    fingertips lighter/greyer and rougher (abraded grain), grime and road mud in the creases and on the palm."""
    m = bpy.data.materials.new('m_glove_hi')
    nb = NB(m)
    vec = nb.objcoord(1.0 / 0.30)                 # brown_leather scan covers 40 cm; glove grain ~25 % finer
    dif = nb.img(tex('brown_leather', 'diffuse'), False, vec)
    rgh = nb.img(tex('brown_leather', 'rough'), True, vec)
    dsp = nb.img(tex('brown_leather', 'displacement'), True, vec)
    nrm = nb.img(tex('brown_leather', 'nor_gl'), True, vec)
    lum = nb.luminance(dif.outputs['Color'])       # scan luminance -> mottling only (the scan is dark brown upholstery)
    mott = nb.math('MULTIPLY_ADD', lum, 3.4, 0.10)   # ~0.65..1.35 around the base tone
    base = nb.rgb((0.150, 0.088, 0.042))           # soaked tan leather (linear) ~0.10 luminance
    col = nb.mul_rgb(base, nb.combine(mott, mott, mott))
    # low-frequency blotches of wetness (drier patches read lighter and matte)
    nz = nb.node('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 1 / 0.035; nz.inputs['Detail'].default_value = 3
    nb.nt.links.new(nb.node('ShaderNodeTexCoord').outputs['Object'], nz.inputs['Vector'])
    dry = nb.smooth(0.50, 0.72, nz.outputs['Fac'])
    col = nb.mix(nb.math('MULTIPLY', dry, 0.55), col, nb.rgb((0.235, 0.150, 0.080)))
    # soaked-through patches (darker, more saturated: the water fills the grain)
    soak = nb.smooth(0.38, 0.30, nz.outputs['Fac'])
    col = nb.mix(nb.math('MULTIPLY', soak, 0.6), col, nb.rgb((0.085, 0.045, 0.020)))
    # scuffed grain on knuckles / fingertips (lighter, greyer) - wear attribute x pointiness
    geo = nb.node('ShaderNodeNewGeometry')
    pt = nb.smooth(0.50, 0.58, geo.outputs['Pointiness'])
    wear = nb.math('MULTIPLY', nb.attr('wear'), nb.math('MULTIPLY_ADD', pt, 0.7, 0.3), clamp=True)
    wear = nb.math('MULTIPLY', wear, nb.smooth(0.25, 0.75, rgh.outputs['Color']), clamp=True)
    col = nb.mix(nb.math('MULTIPLY', wear, 0.8), col, nb.rgb((0.30, 0.225, 0.150)))
    # grime in creases (cavity) and mud on the palm / pads
    cav = nb.math('SUBTRACT', 1.0, nb.smooth(0.44, 0.50, geo.outputs['Pointiness']))
    col = nb.mix(nb.math('MULTIPLY', cav, 0.75), col, nb.rgb((0.040, 0.027, 0.018)))
    mud = nb.smooth(0.35, 0.8, nb.math('ADD', nb.attr('mud'), nb.math('MULTIPLY_ADD', nz.outputs['Fac'], 0.5, -0.25)))
    col = nb.mix(nb.math('MULTIPLY', mud, 0.75), col, nb.rgb((0.080, 0.065, 0.050)))
    # seams: inseam line darker; outseam stitch rows (bonded kevlar/cotton thread, 3.2 mm stitches, 1.6 mm in from the edge)
    ins = nb.math('SUBTRACT', 1.0, nb.smooth(0.2, 0.7, nb.attr('inseam')))
    col = nb.mix(nb.math('MULTIPLY', ins, 0.6), col, nb.rgb((0.035, 0.022, 0.012)))
    ob = nb.attr('outB'); os_ = nb.attr('outS')
    dash = nb.math('LESS_THAN', nb.math('FRACT', nb.math('DIVIDE', os_, 3.2)), 0.72)
    row = nb.math('SUBTRACT', 1.0, nb.smooth(0.28, 0.48, nb.math('ABSOLUTE', nb.math('SUBTRACT', ob, 1.6))))
    thread = nb.math('MULTIPLY', dash, row)
    col = nb.mix(thread, col, nb.rgb((0.22, 0.18, 0.105)))
    # roughness: soaked 0.30-0.42 with the scan's variation, drier patches / scuffs / mud rougher, thread 0.62
    r = nb.math('MULTIPLY_ADD', rgh.outputs['Color'], 0.30, 0.25)
    r = nb.mixf(dry, r, 0.56)
    r = nb.mixf(soak, r, 0.30)
    r = nb.mixf(wear, r, 0.66)
    r = nb.mixf(mud, r, 0.58)
    r = nb.mixf(thread, r, 0.62)
    # normal: scan normal + grain displacement + raised thread with needle holes
    nmap = nb.node('ShaderNodeNormalMap'); nmap.inputs['Strength'].default_value = 0.85
    nb.nt.links.new(nrm.outputs['Color'], nmap.inputs['Color'])
    h = nb.math('MULTIPLY', thread, 0.35)
    hole = nb.math('MULTIPLY', row, nb.math('LESS_THAN', nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', os_, 3.2)), 0.86)), 0.07))
    h = nb.math('SUBTRACT', h, nb.math('MULTIPLY', hole, 0.3))
    h = nb.math('ADD', h, nb.math('MULTIPLY', dsp.outputs['Color'], 0.45))
    bump = nb.node('ShaderNodeBump'); bump.inputs['Distance'].default_value = 0.0008; bump.inputs['Strength'].default_value = 0.6
    nb.nt.links.new(h, bump.inputs['Height']); nb.nt.links.new(nmap.outputs['Normal'], bump.inputs['Normal'])
    nb.finish(col, r, bump.outputs['Normal'])
    MATS.append(nb)
    return m


def sleeve_material():
    """Dark navy nylon hardshell (albedo ~0.03), DWR still beading: 1-3 mm water beads (near-mirror, roughness 0.06),
    a 5 mm ripstop grid, wetted-out darker patches, a stitched cuff hem and the velcro tab (box stitch)."""
    m = bpy.data.materials.new('m_sleeve_hi')
    nb = NB(m)
    tc = nb.node('ShaderNodeTexCoord')
    obj = tc.outputs['Object']
    base = nb.rgb((0.0155, 0.0180, 0.0225))           # dark navy-charcoal nylon (~0.018 linear)
    nz = nb.node('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 1 / 0.05; nz.inputs['Detail'].default_value = 4
    nb.nt.links.new(obj, nz.inputs['Vector'])
    wet = nb.smooth(0.52, 0.66, nz.outputs['Fac'])                 # wetted-out patches (DWR failing where it rubs)
    col = nb.mix(nb.math('MULTIPLY', wet, 0.7), base, nb.rgb((0.0095, 0.0110, 0.0140)))
    # beads: Voronoi cells ~2.2 mm, ~30 % of cells hold a drop of radius 0.35-0.9 of the cell
    vo = nb.node('ShaderNodeTexVoronoi', feature='F1'); vo.inputs['Scale'].default_value = 1 / 0.0028
    nb.nt.links.new(obj, vo.inputs['Vector'])
    rnd = nb.node('ShaderNodeSeparateColor'); nb.nt.links.new(vo.outputs['Color'], rnd.inputs[0])
    has = nb.math('LESS_THAN', rnd.outputs[0], 0.22)
    has = nb.math('MULTIPLY', has, nb.math('SUBTRACT', 1.0, wet))
    rad = nb.math('MULTIPLY_ADD', rnd.outputs[1], 0.35, 0.28)
    dd = nb.math('DIVIDE', vo.outputs['Distance'], rad)
    bead = nb.math('MULTIPLY', has, nb.math('SQRT', nb.math('MAXIMUM', nb.math('SUBTRACT', 1.0, nb.math('MULTIPLY', dd, dd)), 0.0)))
    beadm = nb.math('GREATER_THAN', bead, 0.02)
    # ripstop grid (world mm via object coords)
    sep = nb.node('ShaderNodeSeparateXYZ'); nb.nt.links.new(obj, sep.inputs[0])
    def grid(sock):
        f = nb.math('FRACT', nb.math('DIVIDE', sock, 0.005))
        return nb.math('SUBTRACT', 1.0, nb.smooth(0.0, 0.06, nb.math('ABSOLUTE', nb.math('SUBTRACT', f, 0.5)) if False else nb.math('MINIMUM', f, nb.math('SUBTRACT', 1.0, f))))
    gr = nb.math('MAXIMUM', grid(sep.outputs[1]), grid(nb.math('ADD', sep.outputs[0], sep.outputs[2])))
    # hem stitch row 7 mm from the cuff edge (hem attr = mm from the edge along the tube)
    hem = nb.attr('hem')
    hrow = nb.math('SUBTRACT', 1.0, nb.smooth(0.25, 0.45, nb.math('ABSOLUTE', nb.math('SUBTRACT', hem, 7.0))))
    ang = nb.node('ShaderNodeVectorMath', operation='LENGTH')
    hdash = nb.math('LESS_THAN', nb.math('FRACT', nb.math('DIVIDE', nb.math('ARCTAN2', sep.outputs[0], sep.outputs[2]), 0.052)), 0.7)
    hst = nb.math('MULTIPLY', hrow, hdash)
    col = nb.mix(nb.math('MULTIPLY', hst, 0.8), col, nb.rgb((0.030, 0.035, 0.044)))
    # velcro tab attribute (1 on the tab): slightly lighter webbing + a box stitch border
    tab = nb.attr('tab')
    col = nb.mix(nb.math('MULTIPLY', tab, 0.6), col, nb.rgb((0.018, 0.020, 0.024)))
    tb = nb.attr('tabedge')
    tst = nb.math('MULTIPLY', tab, nb.math('SUBTRACT', 1.0, nb.smooth(0.25, 0.45, nb.math('ABSOLUTE', nb.math('SUBTRACT', tb, 2.2)))))
    col = nb.mix(nb.math('MULTIPLY', tst, 0.8), col, nb.rgb((0.030, 0.035, 0.044)))
    # roughness: coated face 0.42, wetted-out 0.55, beads 0.06, grid lines 0.36, webbing 0.7
    # weave micro-roughness from the scan normal's variation (coated face 0.46-0.58)
    r = nb.mixf(wet, 0.50, 0.62)
    r = nb.mixf(nb.math('MULTIPLY', gr, 0.5), r, 0.42)
    r = nb.mixf(tab, r, 0.68)
    r = nb.mixf(beadm, r, 0.06)
    # normal: weave scan (fine) + grid + beads + stitches
    wv = nb.img(tex('stretch_poplin', 'nor_gl'), True, nb.objcoord(1.0 / 0.12))
    nmap = nb.node('ShaderNodeNormalMap'); nmap.inputs['Strength'].default_value = 0.6
    nb.nt.links.new(wv.outputs['Color'], nmap.inputs['Color'])
    h = nb.math('ADD', nb.math('MULTIPLY', gr, 0.10), nb.math('MULTIPLY', bead, 1.0))
    h = nb.math('ADD', h, nb.math('MULTIPLY', nb.math('ADD', hst, tst), 0.25))
    bump = nb.node('ShaderNodeBump'); bump.inputs['Distance'].default_value = 0.0007; bump.inputs['Strength'].default_value = 0.7
    nb.nt.links.new(h, bump.inputs['Height']); nb.nt.links.new(nmap.outputs['Normal'], bump.inputs['Normal'])
    nb.finish(col, r, bump.outputs['Normal'], ao_dist=0.02)
    MATS.append(nb)
    return m


# ================================================================================================
# UV, bake, final material
# ================================================================================================
def smart_uv(o, angle=60):
    select_only([o])
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=0.0, area_weight=0.0, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def normalise_uv(o, weight=1.0):
    """Scale the object's UVs so 1 UV unit = 1 m (x weight): uniform texel density across objects before packing."""
    me = o.data
    uvl = me.uv_layers.active.data
    n = len(me.polygons)
    area = np.zeros(n); me.polygons.foreach_get('area', area)
    uv = np.zeros(len(uvl) * 2); uvl.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
    ls = np.zeros(n, np.int32); lt = np.zeros(n, np.int32)
    me.polygons.foreach_get('loop_start', ls); me.polygons.foreach_get('loop_total', lt)
    uva = 0.0
    for p in range(n):
        q = uv[ls[p]:ls[p] + lt[p]]
        uva += 0.5 * abs(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    k = math.sqrt(area.sum() / max(uva, 1e-12)) * weight
    uvl.foreach_set('uv', (uv * k).ravel())


def pack_all(objs, margin=0.006):
    select_only(objs)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(rotate=True, margin=margin, scale=True, shape_method='CONCAVE')
    bpy.ops.object.mode_set(mode='OBJECT')


def read_pixels(im):
    W, H = im.size
    a = np.zeros(W * H * 4, np.float32)
    im.pixels.foreach_get(a)
    return a.reshape(H, W, 4)[::-1]


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_png(arr, path):
    arr = np.asarray(arr, np.float32)
    H, W, C = arr.shape
    rgba = np.ones((H, W, 4), np.float32)
    rgba[..., :C] = arr
    im = bpy.data.images.new('tmp_save', W, H, alpha=True, float_buffer=False)
    im.colorspace_settings.name = 'Non-Color'
    im.pixels.foreach_set(rgba[::-1].ravel())
    im.filepath_raw = path
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)


def bake_atlas(lo, his, size, samples):
    t0 = time.time()
    sc = bpy.context.scene
    setup_cycles(samples)
    sc.render.bake.margin = 8
    sc.render.bake.margin_type = 'EXTEND'
    sc.render.bake.use_selected_to_active = True
    sc.render.bake.cage_extrusion = 0.0018
    sc.render.bake.max_ray_distance = 0.0045
    ims = {}
    for k in ('albedo', 'data', 'normal'):
        im = bpy.data.images.new(f'bk_{k}', size, size, alpha=False, float_buffer=True)
        im.colorspace_settings.name = 'Non-Color'
        ims[k] = im
    # the low object needs an active image node in its (bake target) material
    lm = bpy.data.materials.new('m_lo_bake')
    lm.use_nodes = True
    tn = lm.node_tree.nodes.new('ShaderNodeTexImage')
    lm.node_tree.nodes.active = tn
    lo.data.materials.clear()
    lo.data.materials.append(lm)
    select_only(his + [lo], lo)
    for k in ('albedo', 'data', 'normal'):
        tn.image = ims[k]
        for nb in MATS:
            nb.set_pass(k)
        tp = 'NORMAL' if k == 'normal' else 'EMIT'
        bpy.ops.object.bake(type=tp, margin=8, use_clear=True, normal_space='TANGENT', target='IMAGE_TEXTURES',
                            use_selected_to_active=True, cage_extrusion=0.0018, max_ray_distance=0.0045)
        log(f'  baked {k}')
    A = read_pixels(ims['albedo'])[..., :3]
    D = read_pixels(ims['data'])[..., :3]
    N = read_pixels(ims['normal'])[..., :3]
    save_png(lin2srgb(A), os.path.join(BAKE, 'hands_albedo.png'))
    save_png(N, os.path.join(BAKE, 'hands_normal.png'))
    orm = np.stack([np.clip(0.30 + 0.70 * D[..., 0], 0, 1), np.clip(D[..., 1], 0.04, 1), np.zeros_like(D[..., 0])], -1)
    save_png(orm, os.path.join(BAKE, 'hands_orm.png'))
    for im in ims.values():
        bpy.data.images.remove(im)
    for nb in MATS:
        nb.set_pass('preview')
    log(f'bake done in {time.time() - t0:.0f}s')


def final_material():
    m = bpy.data.materials.new('hands')
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes['Principled BSDF']
    ia = nt.nodes.new('ShaderNodeTexImage'); ia.image = load_image(os.path.join(BAKE, 'hands_albedo.png'), False)
    io = nt.nodes.new('ShaderNodeTexImage'); io.image = load_image(os.path.join(BAKE, 'hands_orm.png'), True)
    inn = nt.nodes.new('ShaderNodeTexImage'); inn.image = load_image(os.path.join(BAKE, 'hands_normal.png'), True)
    sp = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(io.outputs['Color'], sp.inputs[0])
    nm = nt.nodes.new('ShaderNodeNormalMap'); nt.links.new(inn.outputs['Color'], nm.inputs['Color'])
    mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'
    mx.inputs['Factor'].default_value = 1.0
    nt.links.new(ia.outputs['Color'], mx.inputs[6]); nt.links.new(sp.outputs[0], mx.inputs[7])
    nt.links.new(mx.outputs[2], bs.inputs['Base Color'])
    nt.links.new(sp.outputs[1], bs.inputs['Roughness'])
    nt.links.new(sp.outputs[2], bs.inputs['Metallic'])
    nt.links.new(nm.outputs['Normal'], bs.inputs['Normal'])
    return m


# ================================================================================================
# assembly
# ================================================================================================
def retopo(src, faces):
    lo = src.copy(); lo.data = src.data.copy(); lo.name = 'glove_lo'
    link(lo)
    for vg in list(lo.vertex_groups):
        lo.vertex_groups.remove(vg)
    lo.modifiers.clear()
    select_only([lo])
    # (QuadriFlow cancels in background mode) collapse-decimate to the triangle budget: triangles concentrate where
    # the surface curves (knuckles, joints, fingertips), which is also where the skin deforms
    ntri = sum(len(p.vertices) - 2 for p in lo.data.polygons)
    m = lo.modifiers.new('dec', 'DECIMATE'); m.ratio = faces / max(ntri, 1)
    apply_mods(lo)
    # shrinkwrap back onto the smoothed union (quadriflow can drift a little)
    sw = lo.modifiers.new('sw', 'SHRINKWRAP'); sw.target = src; sw.wrap_method = 'NEAREST_SURFACEPOINT'
    apply_mods(lo)
    log('glove_lo: %d faces' % len(lo.data.polygons))
    return lo


def tri_count(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


def mirror_object(o, name):
    c = o.copy(); c.data = o.data.copy(); c.name = name
    link(c)
    c.data.transform(Matrix.Scale(-1, 4, (1, 0, 0)))
    c.data.flip_normals()
    c.data.update()
    for vg in c.vertex_groups:
        if vg.name.startswith('R_'):
            vg.name = 'L_' + vg.name[2:]
    return c


def mirrored_joints(joints):
    out = {}
    for k, (P, U) in joints.items():
        P2 = P.copy(); P2[:, 0] *= -1
        U2 = U.copy(); U2[:, 0] *= -1
        out[k] = (P2, U2)
    return out


def assemble_arm(glove_lo, sleeve_lo, tab_lo, arm, S):
    """Skin the glove to hand+fingers, make the sleeve + tab rigid to the forearm, join into one mesh."""
    skin_glove(glove_lo, arm, S)
    for o in (sleeve_lo, tab_lo):
        g = o.vertex_groups.new(name=S + '_forearm')
        g.add(list(range(len(o.data.vertices))), 1.0, 'REPLACE')
    select_only([glove_lo, sleeve_lo, tab_lo], glove_lo)
    bpy.ops.object.join()
    glove_lo.name = 'arm_' + S
    glove_lo.data.name = 'arm_' + S
    # make sure the armature modifier + parent are in place
    if not any(m.type == 'ARMATURE' for m in glove_lo.modifiers):
        mod = glove_lo.modifiers.new('arm', 'ARMATURE'); mod.object = arm
    glove_lo.parent = arm
    return glove_lo


PACK_JS = r"""
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
  if (mat.getName() !== 'hands') continue;
  const tA = tex(`${bakeDir}/hands_albedo.png`, 'hands_albedo'), tN = tex(`${bakeDir}/hands_normal.png`, 'hands_normal'), tO = tex(`${bakeDir}/hands_orm.png`, 'hands_orm');
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
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^normalTexture$/, quality: 92 }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^baseColorTexture$/, quality: 88 }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^(occlusionTexture|metallicRoughnessTexture)$/, quality: 85 }),
);
doc.createExtension(EXTMeshoptCompression).setRequired(true)
  .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
await io.write(out, doc);
let tris = 0;
for (const m of root.listMeshes()) for (const p of m.listPrimitives()) tris += (p.getIndices()?.getCount() ?? p.getAttribute('POSITION').getCount()) / 3;
console.log(JSON.stringify({ out, bytes: fs.statSync(out).size, tris, nodes: root.listNodes().length, skins: root.listSkins().length }));
"""


def export_glb(objs):
    select_only(objs)
    geo = os.path.join(SCR, '_hands_geo.glb')
    bpy.ops.export_scene.gltf(filepath=geo, export_format='GLB', use_selection=True, export_materials='EXPORT',
                              export_image_format='NONE', export_yup=True, export_apply=False, export_texcoords=True,
                              export_normals=True, export_tangents=False, export_animations=False, export_skins=True,
                              export_def_bones=False, export_cameras=False, export_lights=False, export_extras=True)
    js = os.path.join(SCR, '_pack.mjs')
    with open(js, 'w') as f:
        f.write(PACK_JS)
    r = subprocess.run(['node', js, geo, OUT_GLB, BAKE], cwd=ROOT, capture_output=True, text=True)
    log('pack:', r.stdout.strip()[-1500:], r.stderr.strip()[-1500:])


def prop_handle(name='fist'):
    """Grip cylinder proxy for the previews (the handle the pose was solved on)."""
    G = GRIPS[name]
    C = np.array(G['p']); e = np.array(G['d'], float); e /= np.linalg.norm(e)
    V, F = capsule(C - e * 0.07, C + e * 0.07, G['r'] - 0.0005, G['r'] - 0.0005)
    o = mesh_obj('proxy_' + name, V, F)
    return o


def main():
    reset_scene()
    fast = ARGS['stage'] == 'clay'
    union, joints = build_glove_union()
    # low poly from the smoothed union (before details)
    glove_lo = retopo(union, 7200)
    hi = union
    hi.name = 'glove_hi'
    glove_details(hi, joints)
    gm = glove_material(); hi.data.materials.append(gm)
    sleeve_hi = build_sleeve('sleeve_hi', 256, 200, True)
    sleeve_lo = build_sleeve('sleeve_lo', 30, 22, False)
    tab_hi = build_tab('tab_hi', True)
    tab_lo = build_tab('tab_lo', False)
    # tab attributes on the hi: mask + distance (mm) from the strap edge for the box stitch
    Vt, _ = mesh_arrays(tab_hi)
    th = np.arctan2(Vt[:, 0], Vt[:, 2])
    th0, th1 = math.radians(-2), math.radians(84)
    rr = np.hypot(Vt[:, 0], Vt[:, 2])
    e_th = np.minimum(th - th0, th1 - th) * rr * 1000
    e_y = np.minimum(Vt[:, 1] - (Y_CUFF - 0.0285), (Y_CUFF - 0.0035) - Vt[:, 1]) * 1000
    set_float_attr(tab_hi, 'tab', np.ones(len(Vt)))
    set_float_attr(tab_hi, 'tabedge', np.minimum(e_th, e_y))
    set_float_attr(tab_hi, 'hem', np.full(len(Vt), 99.0))
    for o in (sleeve_hi,):
        set_float_attr(o, 'tab', np.zeros(len(o.data.vertices)))
        set_float_attr(o, 'tabedge', np.full(len(o.data.vertices), 99.0))
    sm = sleeve_material()
    sleeve_hi.data.materials.append(sm); tab_hi.data.materials.append(sm)
    for o in (hi, sleeve_hi, tab_hi, glove_lo, sleeve_lo, tab_lo):
        for p in o.data.polygons:
            p.use_smooth = True
    # UVs: glove (smart), sleeve (cylindrical), tab (smart); common texel density, glove weighted up
    smart_uv(glove_lo, 62); normalise_uv(glove_lo, 1.25)
    sleeve_uv(sleeve_lo); normalise_uv(sleeve_lo, 0.72)
    smart_uv(tab_lo, 60); normalise_uv(tab_lo, 0.9)
    pack_all([glove_lo, sleeve_lo, tab_lo], margin=0.008)
    # bake: one low object carrying all three parts' UVs (joined copy), sources = hi glove + hi sleeve + hi tab
    bake_lo = glove_lo.copy(); bake_lo.data = glove_lo.data.copy(); link(bake_lo)
    s2 = sleeve_lo.copy(); s2.data = sleeve_lo.data.copy(); link(s2)
    t2 = tab_lo.copy(); t2.data = tab_lo.data.copy(); link(t2)
    select_only([bake_lo, s2, t2], bake_lo)
    bpy.ops.object.join()
    for p in bake_lo.data.polygons:
        p.use_smooth = True
    if not fast:
        setup_world(1.0)
        bake_atlas(bake_lo, [hi, sleeve_hi, tab_hi], ARGS['size'], ARGS['samples'])
    fm = final_material() if not fast else clay_material()
    bpy.data.objects.remove(bake_lo)
    for o in (glove_lo, sleeve_lo, tab_lo):
        o.data.materials.clear(); o.data.materials.append(fm)
    # rigs: right, then the mirrored left (grips solved on the bare right rig first)
    arm_R = build_armature('R', joints)
    solve_grips(arm_R, 'R')
    store_sockets(arm_R, 'R')
    right = assemble_arm(glove_lo, sleeve_lo, tab_lo, arm_R, 'R')
    store_poses(arm_R, 'R')
    arm_L = build_armature('L', mirrored_joints(joints))
    left = mirror_object(right, 'arm_L')
    for m in list(left.modifiers):
        left.modifiers.remove(m)
    mod = left.modifiers.new('arm', 'ARMATURE'); mod.object = arm_L
    left.parent = arm_L
    store_poses(arm_L, 'L')
    store_sockets(arm_L, 'L')
    log('tris: right %d, left %d' % (tri_count(right), tri_count(left)))
    # the high-poly bake sources are not needed any more: free them before the export and the preview renders
    for o in (hi, sleeve_hi, tab_hi):
        me = o.data
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(me)
    if ARGS['export']:
        export_glb([arm_R, right, arm_L, left])
    if ARGS['preview'] != 'none':
        setup_world(1.0)
        props_ = {k: prop_handle(k) for k in GRIPS}
        arm_L.location = (-0.30, 0, 0)
        for pose, cam in (('rest', 'fp'), ('fist', 'thumb'), ('fist', 'fp'), ('hook', 'thumb'), ('grip', 'thumb'), ('open', 'fp')):
            apply_pose(arm_R, 'R', pose)
            apply_pose(arm_L, 'L', pose)
            for k, o in props_.items():
                o.hide_render = k != pose
            left.hide_render = cam != 'fp'
            if cam == 'fp':
                loc, tgt = (-0.20, -0.42, 0.24), (-0.04, 0.05, 0.0)
            else:
                loc, tgt = (-0.26, 0.13, 0.05), (0.0, 0.08, -0.02)
            render_preview(os.path.join(SCR, f'prev_{pose}_{cam}.png'), loc, tgt, lens=38,
                           samples=24 if fast else 64)


if __name__ == '__main__':
    main()
    log('done')
