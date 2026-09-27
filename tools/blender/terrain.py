"""LANDSLIDE terrain builder (TERRAIN workstream).

Run headless (numpy only for the geometry; Blender for decimation + Cycles AO bake):
  /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/blender/terrain.py -- [--no-ao] [--no-decimate]

Outputs
  public/assets/world/terrain.glb   nodes: terrain_near (chunks terrain_near_XX), terrain_far (tiles terrain_far_XX),
                                    road (chunks road_XX), tunnel, tunnel_lamps
  public/assets/world/scatter.json  trees, rocks, slideSpawns, guardrails, grassMask (+ misc)
  scratch/terrain/*                 intermediates + previews

Design (see DESIGN.md "TERRAIN")
  * natural(X, Z) is one world-space height function shared by every mesh: valley profile down to a river
    ~300 m below the road, the opposite range (ridged multifractal, peaks 600-1200 m above the floor),
    the uphill slope to a ridge, down-slope ravines that follow the road's own bends (spurs/re-entrants),
    rock bands, the tunnel spur, the landslide scar, the V-gullies and the washout ravine.
  * terrain_near is a road-aligned (s, column) grid. The road bench is carved into natural() like a real
    road design: a steep fractured rock cut where the ground is above the road template, a rock-fill
    embankment where it is below. Cut faces get Voronoi-faceted fractures, strata ledges/overhangs and talus.
  * terrain_far is a graded tensor grid of natural() without near-only detail; inside the corridor it is
    lowered (skirt). The near mesh's outer columns take the far mesh's exact triangle-interpolated height,
    so the seam has no cracks.
"""
import argparse
import base64
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from common import (ROOT, GLBWriter, Noise, Road, compact, grid_faces, lerp,  # noqa: E402
                    smoothstep as ss, to_blender, from_blender, vertex_normals)

try:
    import bpy  # noqa: F401
except ImportError:  # pure-python run (no decimation / AO)
    bpy = None

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--no-ao', action='store_true')
ap.add_argument('--no-decimate', action='store_true')
ap.add_argument('--ratio-near', type=float, default=0.45)
ap.add_argument('--ratio-far', type=float, default=0.35)
ap.add_argument('--ao-samples', type=int, default=48)
ap.add_argument('--ratio-lod1', type=float, default=0.2, help='terrain_near LOD1 triangle ratio (of LOD0)')
ap.add_argument('--chunk', type=float, default=50.0, help='terrain_near chunk length along s (m)')
ap.add_argument('--no-compress', action='store_true')
ap.add_argument('--preview', action='store_true')
ap.add_argument('--out', default=None, help='terrain.glb output path (default public/assets/world/terrain.glb)')
ap.add_argument('--scatter-out', default=None, help='scatter.json output path')
ARGS = ap.parse_args(argv)

T0 = time.time()


def log(*a):
    print('[terrain %6.1fs]' % (time.time() - T0), *a, flush=True)


OUT_GLB = os.path.abspath(ARGS.out) if ARGS.out else os.path.join(ROOT, 'public', 'assets', 'world', 'terrain.glb')
OUT_SCATTER = os.path.abspath(ARGS.scatter_out) if ARGS.scatter_out else os.path.join(ROOT, 'public', 'assets', 'world', 'scatter.json')
SCR = os.path.join(ROOT, 'scratch', 'terrain')
os.makedirs(SCR, exist_ok=True)

R = Road()
MK = R.markers
NS = Noise(1337)
NS2 = Noise(4242)
NS3 = Noise(99)
NS4 = Noise(7)
RNG = np.random.default_rng(20260926)

S0, S1 = -200.0, 1300.0          # near corridor s range
DV, DU = -45.0, 60.0             # near corridor d range (valley, uphill)
GAP = float(MK['gap'])
TUN = float(MK['tunnel'])
TUN_END = 1298.0                 # tube runs on around the bend (road.json ends at 1300); dark beyond the lamps
LAMP_END = 1276.0                # last lamps ~20 m before the end cap (matches roadMaterial.js TUNNEL.lampEnd)
GULLIES = [float(g) for g in MK['gullies']]
SCAR = [float(v) for v in MK['scar']]
PULL = [float(v) for v in MK['pulloff']]
ROAD_S0 = S0 + 12.0

# ==============================================================================================
# 1D helper tables along X (the valley runs roughly along X)
# ==============================================================================================
XT = np.arange(-9000.0, 10000.0, 2.0)


def gsmooth(v, sigma, dx=2.0):
    k = int(3 * sigma / dx)
    t = np.arange(-k, k + 1) * dx
    w = np.exp(-0.5 * (t / sigma) ** 2)
    w /= w.sum()
    pad = np.concatenate([np.full(k, v[0]), v, np.full(k, v[-1])])
    return np.convolve(pad, w, mode='valid')


X_LO, X_HI = R.P[0, 0], R.P[-1, 0]
_zr = R.z_of_x(XT)
ML_T = gsmooth(_zr, 150.0)
DEV_T = (_zr - ML_T) * ss(X_LO, X_LO + 150, XT) * (1 - ss(X_HI - 150, X_HI, XT))
YB_T = np.where(XT < X_LO, R.P[0, 1] + (XT - X_LO) * 0.045,
                np.where(XT > X_HI, R.P[-1, 1] + (XT - X_HI) * 0.045, R.y_of_x(XT)))
YF_T = -275.0 + 0.035 * XT
W_T = 520 + 110 * np.sin(XT / 760 + 0.6) + 60 * np.sin(XT / 310 + 2.0)
CR_T = 480 + 110 * NS.perlin2(XT / 420, 0.37)
HS_T = 1.0 + 0.14 * NS.perlin2(XT / 350, 5.1)
P0_T = 3.5 + 1.4 * NS.perlin2(XT / 75, 9.3) + 0.6 * NS2.perlin2(XT / 23, 1.7)
SPUR_T = 150.0 * ss(1040, 1250, XT) * (1 - ss(1420, 1700, XT))

# canonical uphill profile (ridge at c=480)
CT = np.arange(0.0, 12000.0, 1.0)
# slope (rise/run) of the uphill side vs distance from the road: ~34 deg right above the cut (holds soil and
# forest), steepening to ~41 deg mid-slope, rolling over to the ridge at c=480 (~330 m above the road)
_g = np.interp(CT, [0, 30, 90, 160, 300, 420, 480, 520, 830, 870, 1300, 1400, 2600, 2700],
               [0.66, 0.74, 0.87, 0.85, 0.78, 0.45, 0.0, -0.45, -0.45, -0.1, -0.1, 0.55, 0.55, 0.05])
UPT = np.concatenate([[0.0], np.cumsum(_g[:-1])])


def fx(tab, X):
    return np.interp(X, XT, tab)


# landmark frames
def frame(s):
    p = R.point(np.array([s]))[0]
    t = R.tangent(np.array([s]))[0]
    t = np.array([t[0], 0.0, t[2]])
    t /= np.linalg.norm(t)
    return p, t


SCAR_P, SCAR_T = frame(0.5 * (SCAR[0] + SCAR[1]))
GULLY_F = [frame(g) for g in GULLIES]
GAP_P, GAP_T = frame(GAP)


# ==============================================================================================
# natural terrain
# ==============================================================================================
def natural(X, Z, D=None, S=None, hi=None, masks=False):
    """Height of the undisturbed mountain at world (X, Z) [three coords].
    D, S: exact road projection (nan where unknown). hi: 0..1 weight of near-only detail."""
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    ml = fx(ML_T, X)
    yb = fx(YB_T, X)
    p0 = fx(P0_T, X)
    c_far = ml - Z
    lam = 0.35 + 0.65 * np.exp(-np.abs(c_far) / 250.0)
    c = c_far + fx(DEV_T, X) * lam
    if D is not None:
        D = np.asarray(D, dtype=np.float64)
        ok = np.isfinite(D)
        w = ss(25, 70, np.abs(np.where(ok, D, 1e3)))
        c = np.where(ok, np.where(ok, D, 0) * (1 - w) + c * w, c)
    if S is None:
        Sx = R.s_of_x(X)
    else:
        S = np.asarray(S, dtype=np.float64)
        Sx = np.where(np.isfinite(S), S, R.s_of_x(X))
    ce = c + fx(SPUR_T, X) * (1 - ss(0, 330, c))
    a = np.abs(ce)

    # ---- macro profile
    cr = fx(CR_T, X)
    # ridge distance cr varies along the valley; the slope scales only with sqrt(480/cr) (ridge height too)
    up = p0 + fx(HS_T, X) * np.sqrt(cr / 480.0) * np.interp(np.maximum(ce, 0) * 480.0 / cr, CT, UPT)
    W = fx(W_T, X)
    yf = fx(YF_T, X)
    v = -ce / W
    val = yf + (yb + p0 - yf) * np.clip(1 - v, 0, 1) ** 1.6
    u = np.maximum(-ce - W, 0)
    opp = 950 * (1 - np.exp(-u / 900.0)) ** 1.3
    base = np.where(ce >= 0, yb + up, val + opp)
    # river channel meandering on the floor
    rv = -ce - W - 30 - 25 * NS.perlin2(X / 330, 2.2)
    base -= 4.5 * np.exp(-(rv / 16.0) ** 2)
    # big mountains
    M = NS2.ridged2(X / 1500 + 7.1, Z / 1500 - 3.3, 6)
    env_opp = ss(60, 1300, u)
    env_up = ss(180, 700, ce)
    env_val = ss(30, 260, -ce) * (1 - ss(0.75, 1.0, v))
    base += (M - 0.42) * (950 * env_opp + 320 * env_up + 130 * env_val)
    # opposite-side ravines running up the far range
    warp = NS.fbm2(X / 260, Z / 260, 3) * 0.7
    Ro = NS3.ridged2(X / 280 + warp, u / 800 - 1.1, 4)
    base -= 90 * env_opp * (Ro - 0.45)

    # ---- slope detail (suppressed inside the fresh scar)
    Rv = NS.ridged2(X / 95 + warp, ce / 430 + 2.3, 4)
    A_rv = np.where(ce > 0, 9 * ss(6, 40, ce) + 30 * ss(40, 260, ce),
                    10 * ss(5, 40, -ce) + 40 * ss(40, 220, -ce)) * (1 - ss(0.8, 1.0, v))
    det = -A_rv * (Rv - 0.45)
    Rv2 = NS3.ridged2(X / 38 + warp * 1.7, ce / 140 - 0.7, 3)
    det -= (5 * ss(3, 30, a) + 4 * ss(50, 300, a)) * (Rv2 - 0.45) * (1 - ss(0.8, 1.0, v))
    # meso relief on the slopes right above/below the road: fall-line ribs and hollows (8-20 m), knolls
    Rv3 = NS4.ridged2(X / 17 + warp * 2.3, ce / 55 + 0.4, 3)
    det -= 2.6 * ss(4, 18, a) * (1 - ss(120, 260, a)) * (Rv3 - 0.45)
    det += 1.4 * ss(4, 14, a) * (1 - ss(150, 300, a)) * NS3.fbm2(X / 9.5 + 3.1, Z / 9.5 - 7.7, 3)
    R2 = NS3.ridged2(X / 170 + 1.7, Z / 170 - 4.2, 5)
    det += (R2 - 0.5) * (4 + 14 * ss(15, 300, a))
    det += NS2.fbm2(X / 48, Z / 48, 4) * (2.4 + 4 * ss(15, 250, a))
    # damp detail right next to the road on the valley side (keeps a clean drop-off)
    det *= np.where(ce < 0, 0.3 + 0.7 * ss(3, 22, -ce), 1.0)
    if hi is not None:
        hi = np.asarray(hi, dtype=np.float64)
        det += hi * (NS.fbm2(X / 11, Z / 11, 3) * 1.3 + (NS2.ridged2(X / 26, Z / 26, 3) - 0.5) * 2.2)

    # ---- pull-off bench: gentler valley side where the pull-off sits
    bench = 7.5 * ss(335, 368, Sx) * (1 - ss(422, 455, Sx)) * ss(-60, -26, ce) * (1 - ss(-4, 2, ce))
    base += bench

    # ---- landslide scar (s 90-190): smooth concave slide surface, head scarp, debris lobes
    su = (X - SCAR_P[0]) * SCAR_T[0] + (Z - SCAR_P[2]) * SCAR_T[2]
    sv = c
    sw = (44 + 24 * ss(0, 110, sv) - 24 * ss(150, 235, sv)) * (1 + 0.07 * NS.perlin2(sv / 22, 1.1))
    r = np.abs(su + 6 * NS2.perlin2(sv / 60, 4.4)) / sw
    vh = 228 - 30 * r ** 2 + 8 * NS.perlin2(su / 30, 8.8)
    # the fresh slide reaches the road: the lower slope and the old rock cut were stripped away, so the
    # slide surface comes down at ~25-30 deg to a debris toe right at the ditch
    m_scar = ss(1.03, 0.9, r) * ss(0.0, 2.0, sv) * (1 - ss(vh - 3, vh + 1, sv))
    dprof = np.interp(sv, [-5, 0, 2.5, 6, 14, 40, 80, 150, 220], [0, 2.0, 4.2, 6.0, 7.6, 12.0, 14.5, 16.0, 15.0])
    dprof = dprof + (p0 - 3.0)                       # follows the local road-level offset
    cross = np.sqrt(np.clip(1 - r ** 6, 0, 1))        # flat floor, steep flanks
    depth = dprof * cross * (1 - ss(vh - 3.5, vh + 0.5, sv))
    lobe = (2.0 * np.exp(-((sv - 11) / 7.0) ** 2) * (0.45 + 0.55 * NS3.perlin2(su / 9, 7.7))
            + 1.3 * np.exp(-((sv - 30) / 12.0) ** 2) * ss(-0.2, 0.5, NS2.perlin2(su / 14, 2.9)))
    lobe = np.maximum(lobe, 0) * ss(1.1, 0.8, r) * ss(1.5, 5.0, sv)
    H = base + det * (1 - 0.85 * m_scar) - depth * m_scar + lobe * m_scar
    # rills and small slump steps on the fresh slide surface
    # fresh slide surface: flow-parallel ridges and rills, hummocks, transverse tension steps
    # irregular hummocks (domain-warped, roughly isotropic: no contour-parallel regularity)
    wa = 5.0 * NS.perlin2(su / 23.0, sv / 23.0 + 3.3)
    wb = 5.0 * NS3.perlin2(su / 23.0 + 7.1, sv / 23.0)
    hum = NS2.ridged2((su + wa) / 9.5, (sv + wb) / 11.0, 3) - 0.45
    # flow-parallel ridges and furrows (elongated down the fall line) typical of a debris slide surface
    flow = NS4.ridged2((su + 0.6 * wa) / 6.5, sv / 48.0, 3) - 0.45
    tstep = 0.5 * ss(0.55, 0.9, np.abs(NS3.perlin2(sv / 13.0 + 1.1 * NS.perlin2(su / 11, 1.2), su / 30.0 + 6.1)))
    tstep *= ss(-0.1, 0.4, NS2.perlin2(su / 35.0, sv / 35.0 + 1.3))
    H += m_scar * ss(4, 16, sv) * (0.5 * (np.abs(NS4.perlin2(su / 2.4, sv / 5.0)) - 0.3)
                                   + 1.3 * NS.fbm2((su + wa) / 6.0, (sv + wb) / 6.5, 3) + 1.6 * hum + 1.2 * flow
                                   - 1.5 * tstep * ss(30, 60, sv))
    # slumped blocks in the upper slide: back-tilted benches behind transverse tension scarps (1-3 m),
    # irregular spacing, broken laterally
    sl = sv / 19.0 + 0.55 * NS.perlin2(su / 21.0, 4.2) + 0.35 * NS2.perlin2(sv / 31.0, su / 40.0)
    slf = sl - np.floor(sl)
    slump = (1.0 - slf) * 2.0 * ss(0.05, 0.45, NS2.perlin2(np.floor(sl) * 1.7, su / 17.0))
    H += m_scar * ss(45, 90, sv) * (1 - ss(vh - 25, vh - 6, sv)) * slump
    # bedrock ribs: the slip surface cut down to rock in places; sharp ribs running down the fall line
    rib_n = NS3.ridged2(su / 16.0 + 0.4 * NS.perlin2(sv / 30, 1.9), sv / 70.0, 3)
    rib = ss(0.62, 0.9, rib_n) * ss(18, 45, sv) * ss(0.95, 0.55, r) * ss(-0.2, 0.3, NS4.perlin2(sv / 45, su / 60))
    H += m_scar * rib * 2.6

    # ---- rock bands (partial terracing along contours)
    # discontinuous outcrop bands: patchy (two noise scales), thickness varies, off-contour wobble
    rz = ss(0.05, 0.3, NS3.fbm2(X / 170 + 3, Z / 170, 3)) * ss(-0.25, 0.25, NS.fbm2(X / 55 - 2, Z / 55 + 9, 2))
    lt = 9.0 + 6.0 * NS4.perlin2(X / 260, Z / 260)
    k = (H + 4.0 * NS2.perlin2(X / 90, Z / 90)) / lt
    f = k - np.floor(k)
    tr = (np.floor(k) + ss(0.4, 1.0, f)) * lt - 4.0 * NS2.perlin2(X / 90, Z / 90)
    strength = 0.75 * rz * ss(12, 45, a) * (1 - m_scar)
    H = H + strength * (tr - H)

    # ---- gullies: V-shaped rock chutes crossing the road
    g_mask = np.zeros_like(H)
    for (gp, gt) in GULLY_F:
        gu = (X - gp[0]) * gt[0] + (Z - gp[2]) * gt[2]
        gum = gu - 6 * NS.perlin2(c / 70, gp[0] * 0.013) * ss(8, 90, np.abs(c))
        dep = np.where(c > 0, lerp(p0 + 0.4, 5.5 + 0.012 * c, ss(4, 30, c)), 3.5 + 0.8 * ss(0, -30, c))
        dep *= (1 - ss(520, 600, c)) * (1 - ss(-150, -220, c))
        kside = 1.15
        cut = np.maximum(0, dep - kside * np.maximum(np.abs(gum) - 0.7, 0))
        H -= cut
        g_mask = np.maximum(g_mask, np.clip(cut / np.maximum(dep, 0.1), 0, 1) * (dep > 0.5))

    # ---- washout ravine at the gap (uphill wet ravine, down-valley erosion gully)
    ru = (X - GAP_P[0]) * GAP_T[0] + (Z - GAP_P[2]) * GAP_T[2]
    rum = ru - 3.0 * NS2.perlin2(c / 40, 3.3) * ss(6, 40, np.abs(c))
    rdep = np.where(c > 0, lerp(p0 + 2.1, 3.6 + 0.01 * c, ss(2, 28, c)), 2.3 + 0.07 * np.minimum(-c, 90))
    rdep *= (1 - ss(260, 330, c)) * (1 - ss(-130, -190, c))
    rk = np.where(c > 0, 1.5, 1.0)
    rcut = np.maximum(0, rdep - rk * np.maximum(np.abs(rum) - 0.6, 0))
    H -= rcut
    r_mask = np.clip(rcut / np.maximum(rdep, 0.1), 0, 1)

    # ---- valley side must fall away from the shoulder: cap the ground below a gentle (~29 deg) plane
    # through the shoulder. Natural usually falls faster (so this only bites on anomalously high ground);
    # faded out 35-70 m from the road so the far grid (which knows D there) agrees with the corridor.
    if D is not None:
        okd = np.isfinite(D)
        Dv = np.where(okd, D, 1e3)
        yroad = R.height(Sx)
        cap = yroad - 0.12 - 0.55 * np.maximum(-Dv - 3.9, 0)
        wcap = (Dv < -3.0) * (1 - ss(35, 70, -Dv)) * (1 - ss(1060, 1085, Sx)) * okd
        H = H - wcap * np.maximum(H - cap, 0)

    if not masks:
        return H
    tl = 560 + 60 * NS.perlin2(X / 300, Z / 300)
    return H, {
        'c': c, 'ce': ce, 'scar': m_scar, 'lobe': np.clip(lobe / 3.0, 0, 1) * ss(-2, 6, sv), 'rib': m_scar * rib,
        'gully': g_mask, 'ravine': r_mask, 'rockzone': rz * ss(12, 45, a), 'treeline': ss(tl + 60, tl - 40, H),
        'river': np.exp(-(rv / 26.0) ** 2) * ss(0.85, 1.0, v),
    }


def natural_chunked(X, Z, D=None, S=None, hi=None, masks=False, chunk=200000):
    X = np.asarray(X, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64).ravel()
    D = None if D is None else np.asarray(D, dtype=np.float64).ravel()
    S = None if S is None else np.asarray(S, dtype=np.float64).ravel()
    if hi is not None:
        hi = np.asarray(hi, dtype=np.float64)
        hi = np.full(X.shape, float(hi)) if hi.ndim == 0 else hi.ravel()
    Hs, Ms = [], []
    for a in range(0, X.shape[0], chunk):
        sl = slice(a, a + chunk)
        r = natural(X[sl], Z[sl], None if D is None else D[sl], None if S is None else S[sl],
                    None if hi is None else hi[sl], masks)
        if masks:
            Hs.append(r[0])
            Ms.append(r[1])
        else:
            Hs.append(r)
    H = np.concatenate(Hs)
    if masks:
        return H, {k: np.concatenate([m[k] for m in Ms]) for k in Ms[0]}
    return H


# ==============================================================================================
# far grid (built first: the near corridor seam samples it)
# ==============================================================================================
def graded_axis(core_lo, core_hi, h, lo_end, hi_end, steps):
    """steps: list of (growth, hmax, until) applied outward on both sides."""
    core = np.arange(core_lo, core_hi + 1e-6, h)
    out = []
    for sign, start, end in ((-1, core_lo, lo_end), (1, core_hi, hi_end)):
        pos, cur, pts = start, h, []
        for (growth, hmax, until) in steps:
            lim = abs(until)
            while abs(pos) < lim and ((sign < 0 and pos > end) or (sign > 0 and pos < end)):
                cur = min(cur * growth, hmax)
                pos = pos + sign * cur
                pts.append(pos)
        out.append(np.array(pts))
    ax = np.concatenate([out[0][::-1], core, out[1]])
    return ax


def build_far_axes():
    xs = graded_axis(-280, 1380, 5.0, -7000, 8000, [(1.05, 12, 900), (1.07, 90, 9000)])
    # Z: separate growth for uphill (-Z) and valley (+Z)
    core = np.arange(-120.0, 100.0 + 1e-6, 5.0)
    up, pos, cur = [], -120.0, 5.0
    while pos > -6500:
        cur = min(cur * (1.035 if pos > -700 else 1.07), 14 if pos > -700 else 90)
        pos -= cur
        up.append(pos)
    dn, pos, cur = [], 100.0, 5.0
    while pos < 7500:
        cur = min(cur * (1.035 if pos < 1100 else 1.07), 14 if pos < 1100 else 90)
        pos += cur
        dn.append(pos)
    zs = np.concatenate([np.array(up)[::-1], core, np.array(dn)])
    # fine core over the landslide scar (the hero feature runs ~230 m up the slope, far beyond the near
    # corridor): 2.5 m so the slide relief (hummocks, slump scarps, bedrock ribs) is not aliased
    (x0, x1), (z0, z1) = scar_bounds()
    xs = np.unique(np.concatenate([xs[(xs < x0) | (xs > x1)], np.arange(x0, x1 + 1e-6, 2.5)]))
    zs = np.unique(np.concatenate([zs[(zs < z0) | (zs > z1)], np.arange(z0, z1 + 1e-6, 2.5)]))
    return xs, zs


def scar_bounds(pad=14.0):
    """World X/Z extents of the scar mask (sampled on a 4 m grid around the scar frame)."""
    xs = np.arange(SCAR_P[0] - 320, SCAR_P[0] + 320, 4.0)
    zs = np.arange(SCAR_P[2] - 360, SCAR_P[2] + 60, 4.0)
    XX, ZZ = np.meshgrid(xs, zs)
    _, M = natural_chunked(XX.ravel(), ZZ.ravel(), None, None, None, masks=True)
    on = (M['scar'] > 0.01) | (M['lobe'] > 0.01)
    X, Z = XX.ravel()[on], ZZ.ravel()[on]
    b = ((float(np.floor((X.min() - pad) / 2.5) * 2.5), float(np.ceil((X.max() + pad) / 2.5) * 2.5)),
         (float(np.floor((Z.min() - pad) / 2.5) * 2.5), float(np.ceil((Z.max() + pad) / 2.5) * 2.5)))
    log('scar bounds', b)
    return b


def corridor_inside(s, d):
    """Signed distance (m) inside the near corridor in (s,d) space; negative outside."""
    return np.minimum.reduce([s - S0, S1 - s, d - DV, DU - d])


def build_far():
    xs, zs = build_far_axes()
    nx, nz = xs.size, zs.size
    XX, ZZ = np.meshgrid(xs, zs)            # rows = Z, cols = X
    X, Z = XX.ravel(), ZZ.ravel()
    near = (np.abs(Z - R.z_of_x(X)) < 150) & (X > X_LO + 5) & (X < X_HI - 5)
    Sp = np.full(X.shape, np.nan)
    Dp = np.full(X.shape, np.nan)
    s_, d_ = R.project(X[near], Z[near])
    Sp[near], Dp[near] = s_, d_
    log('far grid', nx, 'x', nz, '=', X.size, 'verts; projected', near.sum())
    H, M = natural_chunked(X, Z, Dp, Sp, None, masks=True)
    din = np.where(np.isfinite(Sp), corridor_inside(np.nan_to_num(Sp, nan=-1e4), np.nan_to_num(Dp, nan=1e4)), -1e3)
    H_low = H - (0.25 * ss(1.0, 2.5, din) + 30.0 * ss(8.0, 16.0, din))
    return dict(xs=xs, zs=zs, X=X, Z=Z, H=H_low, Hn=H, S=Sp, D=Dp, din=din, M=M)


def far_interp(F, X, Z):
    """Exact triangle interpolation of the far grid (matches grid_faces diagonal a-cc)."""
    xs, zs, H = F['xs'], F['zs'], F['H']
    nx = xs.size
    c = np.clip(np.searchsorted(xs, X) - 1, 0, nx - 2)
    r = np.clip(np.searchsorted(zs, Z) - 1, 0, zs.size - 2)
    fxx = np.clip((X - xs[c]) / (xs[c + 1] - xs[c]), 0, 1)
    fz = np.clip((Z - zs[r]) / (zs[r + 1] - zs[r]), 0, 1)
    ha = H[r * nx + c]
    hb = H[(r + 1) * nx + c]
    hc = H[(r + 1) * nx + c + 1]
    hd = H[r * nx + c + 1]
    return np.where(fz >= fxx, ha + fz * (hb - ha) + fxx * (hc - hb), ha + fxx * (hd - ha) + fz * (hc - hd))


# ==============================================================================================
# near corridor
# ==============================================================================================
K_VO, K_VF, K_VP = 20, 12, 5
ROAD_COLS = np.array([-3.05, -2.0, -1.0, 0.0, 1.0, 2.0, 3.05])
DITCH_COLS = np.array([3.35, 3.65, 3.95, 4.3])
K_UP, K_UF, K_UO = 3, 26, 28
FINE_N, FINE_L = 150, 36.0


def graded01(k, power=1.9):
    """k values in (0, 1], dense near 0."""
    t = np.arange(1, k + 1) / k
    return t ** power


def near_rows():
    s = np.arange(S0, S1 + 1e-6, 0.5)
    extra = [np.arange(GAP - 3.0, GAP + 3.0001, 0.1),
             np.arange(TUN - 1.2, TUN - 0.05, 0.1), np.array([TUN - 0.02, TUN, TUN + 0.02]),
             np.arange(TUN + 0.1, TUN + 1.21, 0.1), np.arange(TUN + 1.45, TUN + 9.0, 0.25)]
    s = np.concatenate([s] + extra)
    s = s[(np.abs(s - TUN) > 0.035) | np.isin(np.round(s, 4), np.round([TUN - 0.02, TUN, TUN + 0.02], 4))]
    return np.unique(np.round(s, 4))


def pull_w(s):
    """Pull-off bench weight. Starts 2 m earlier / ends 2 m later than PROPS' pullW (P0-10..P0+1, P1-1..P1+10),
    so the bench (edge -3.9 - 11.0 w) always clears PROPS' rail line (-3.8 - 10.1 w) by >= 0.1 + 0.9 w."""
    return ss(PULL[0] - 12, PULL[0] - 1, s) * (1 - ss(PULL[1] + 1, PULL[1] + 12, s))


def trench_depth(s, d):
    """Depth (m, positive) of the washed-out trench below the road surface (road-bed zone)."""
    u = np.abs(s - GAP)
    jag = 0.12 * NS.perlin2(d / 0.9, 3.7) + 0.06 * NS2.perlin2(d / 0.35, 1.1)
    u = u + jag
    dep = 0.14 * (1 - ss(1.95, 2.15, u)) + 1.66 * (1 - ss(1.22, 1.6, u))
    rub = 0.18 * NS3.perlin2(d / 0.8, s / 0.8) * (1 - ss(0.9, 1.3, u))
    return (dep + rub) * (1 - ss(6.5, 9.0, np.abs(d)))


def displace_face(s, h, dsign, span):
    """Horizontal fracture displacement (m, + = into the mountain) for a rock cut face.
    s: along road; h: height above the toe; span: total face height (for top fade)."""
    wu = 0.28 * NS3.perlin2(s / 4.3, h / 4.3 + 7.0)
    wv = 0.28 * NS4.perlin2(s / 4.3 + 3.0, h / 4.3) + 0.3 * NS2.perlin2(s / 17, 0.5)
    F1, F2, cid, cx, cy = NS.voronoi2(s / 3.4 + wu, h / 1.9 + wv, 0.95)
    h1 = Noise.hash2(cx, cy, 31)
    h2 = Noise.hash2(cx, cy, 57)
    h3 = Noise.hash2(cx, cy, 91)
    lu = s / 3.4 + wu - (cx + 0.5)
    lv = h / 1.9 + wv - (cy + 0.5)
    # relief mostly from tilted joint facets, little from recessed cells (recesses read as dark boxy notches)
    vor = (h1 - 0.5) * 0.6 + (h2 - 0.5) * 1.0 * lu + (h3 - 0.5) * 1.1 * lv
    q = h / 1.65 + 0.25 * NS.perlin2(s / 35, 2.0) + 0.0025 * s
    lay = np.floor(q)
    f = q - lay
    kind = Noise.hash2(lay.astype(np.int64), np.floor(s / 23).astype(np.int64), 3)
    strata = np.where(kind > 0.5, -0.24 * (1 - f), -0.18 * f) * (0.4 + 0.6 * ss(-0.3, 0.3, NS3.perlin2(s / 19, lay * 0.7)))
    fine = 0.13 * NS2.fbm3(s / 1.3, h / 1.3, dsign * 3.1, 3)
    fr = 0.55 + 0.6 * ss(-0.25, 0.45, NS4.perlin2(s / 60, 0.9))
    tot = (vor + strata + fine) * fr
    fade = ss(0.3, 1.4, h) * (1 - ss(span - 0.9, span - 0.05, h))
    return tot * fade


def resample_rows(dfine, yfine, fe, K):
    """Per row, resample polyline (dfine[i, :fe[i]+1], yfine) into K+1 points uniform in arc length."""
    nr = dfine.shape[0]
    outd = np.empty((nr, K + 1))
    outy = np.empty((nr, K + 1))
    seg = np.hypot(np.diff(dfine, axis=1), np.diff(yfine, axis=1))
    cum = np.concatenate([np.zeros((nr, 1)), np.cumsum(seg, axis=1)], axis=1)
    t = np.linspace(0, 1, K + 1)
    for i in range(nr):
        e = fe[i]
        L = cum[i, e]
        tg = t * L
        outd[i] = np.interp(tg, cum[i, :e + 1], dfine[i, :e + 1])
        outy[i] = np.interp(tg, cum[i, :e + 1], yfine[i, :e + 1])
    return outd, outy


def hi_weight(s, d):
    w = np.where(d > 0, 1 - ss(DU - 26, DU - 4, d), 1 - ss(-DV - 22, -DV - 4, -d))
    return w * ss(S0, S0 + 25, s) * (1 - ss(S1 - 25, S1, s))


def chunk_boundary(s):
    """True on near rows that lie exactly on a chunk boundary (s = S0 + k * ARGS.chunk)."""
    q = (np.asarray(s) - S0) / ARGS.chunk
    return np.abs(q - np.round(q)) * ARGS.chunk < 0.02


def seam_weight(s, d):
    wd = np.where(d > 0, ss(DU - 10, DU, d), ss(-DV - 10, -DV, -d))
    return np.maximum.reduce([wd, ss(S0 + 10, S0, s), ss(S1 - 10, S1, s)])


def build_near(F):
    s = near_rows()
    nr = s.size
    log('near rows', nr)
    P = R.point(s)
    L = R.left(s)
    yc = P[:, 1]
    mode = np.where(s < TUN - 0.01, 0, np.where(s > TUN + 0.01, 2, 1))
    carved = mode == 0
    portal = np.clip((s - 1131.0) / (1149.5 - 1131.0), 0, 1) * carved
    u_tr = np.abs(s - GAP)
    w_tr = 1 - ss(1.8, 3.6, u_tr)

    pw = pull_w(s)
    d_e_v = -3.9 - 11.0 * pw - 11.3 * portal
    d_e_u = 4.6 + 10.6 * portal
    # shoulder falls ~1.8% outward; the pull-off bench only ~0.9% (level gravel at road height)
    y_e_v = -0.10 - (0.018 - 0.009 * pw) * (np.abs(d_e_v) - 3.9)
    y_e_u = np.full(nr, -0.10)
    y_e_v = y_e_v - trench_depth(s, d_e_v)
    y_e_u = y_e_u - trench_depth(s, d_e_u)
    tan_u = 4.2 + 6.0 * ss(-0.3, 0.5, NS.perlin2(s / 55, 0.7))
    tan_v = 3.2 + 3.0 * ss(-0.3, 0.5, NS2.perlin2(s / 47, 5.7))
    tt = 0.15 + 1.4 * ss(-0.1, 0.6, NS2.perlin2(s / 23, 1.9))
    tt = tt * (1 - portal)

    # ---- fine profiles on both sides
    FU = np.linspace(0, 1.0, FINE_N)
    sides = {}
    for side, de, ye in (('u', d_e_u, y_e_u), ('v', d_e_v, y_e_v)):
        sgn = 1.0 if side == 'u' else -1.0
        # fine profile length per row: never reaches the outer corridor band (else the columns fold)
        Lrow = np.minimum(FINE_L, (DU - de - 8.0) if side == 'u' else (-DV + de - 8.0))
        dd = FU[None, :] * Lrow[:, None]
        dfine = de[:, None] + sgn * dd
        Wp = P[:, None, :] + L[:, None, :] * dfine[..., None]
        ss_ = np.broadcast_to(s[:, None], dfine.shape)
        N = natural_chunked(Wp[..., 0], Wp[..., 2], dfine, ss_, hi_weight(ss_, dfine)).reshape(dfine.shape)
        n_rel = N - yc[:, None]
        if side == 'u':
            y_cut = ye[:, None] + np.maximum(dd - tt[:, None], 0) * tan_u[:, None]
            y_ramp = ye[:, None] + dd * 0.72
            y_ramp = np.where(dd > 1.9 * tt[:, None] + 0.3, -1e4, y_ramp)
            y_top = np.maximum(y_cut, y_ramp)
            y_fill = ye[:, None] - dd * 0.85
        else:
            y_top = ye[:, None] + np.maximum(dd - 0.4, 0) * tan_v[:, None]
            # rock-fill embankment below the shoulder (~52 deg), ends in a short drop if the ground is far below
            y_fill = ye[:, None] - dd * 1.3
        y_fill = np.where(ye[:, None] - y_fill > (14.0 if side == 'u' else 11.0), -1e4, y_fill) - 1e4 * w_tr[:, None]
        prof = np.maximum(np.minimum(n_rel, y_top), y_fill)
        prof = np.where(carved[:, None], prof, n_rel)
        diff = np.abs(prof - n_rel) > 0.01
        idx = np.arange(FINE_N)[None, :]
        last = np.where(diff, idx, -1).max(axis=1)
        fe = np.clip(last + 1, 3, FINE_N - 1)
        fe = np.where(carved, fe, 4)
        fd, fy = resample_rows(dfine, prof, fe, K_UF if side == 'u' else K_VF)
        # the fine profile must re-meet natural ground (no jump into the outer columns)
        miss = np.abs(prof[np.arange(nr), fe] - n_rel[np.arange(nr), fe])
        if (miss[carved] > 0.5).any():
            log('WARN side', side, 'fine profile misses natural at', int((miss[carved] > 0.5).sum()), 'rows; max', round(float(miss[carved].max()), 1),
                'at s', s[carved][np.argmax(miss[carved])])
        # steepness along the face (for fracture displacement)
        slope = np.abs(np.gradient(fy, axis=1)) / np.maximum(np.abs(np.gradient(fd, axis=1)), 1e-3)
        sides[side] = dict(fd=fd, fy=fy, fe_d=dfine[np.arange(nr), fe], slope=slope)

    # ---- fracture displacement on the cut faces
    for side in ('u', 'v'):
        sd = sides[side]
        fd, fy = sd['fd'], sd['fy']
        ye = y_e_u if side == 'u' else y_e_v
        h = fy - ye[:, None]
        span = np.maximum(h.max(axis=1), 0.5)
        steep = ss(1.8, 3.2, sd['slope']) * (h > 0.2)
        disp = displace_face(np.broadcast_to(s[:, None], h.shape), np.maximum(h, 0), 1.0 if side == 'u' else -1.0,
                             span[:, None]) * steep * carved[:, None]
        # round the cell steps a little (grid-aligned steps read as rectangular dark notches)
        for _ in range(2):
            dp = np.pad(disp, 1, mode='edge')
            disp = (0.5 * dp[1:-1, 1:-1] + 0.125 * (dp[:-2, 1:-1] + dp[2:, 1:-1] + dp[1:-1, :-2] + dp[1:-1, 2:]))
        if side == 'u':
            nd = fd + disp
            nd = np.maximum(nd, d_e_u[:, None] + 0.04 * np.arange(nd.shape[1])[None, :] / nd.shape[1])
        else:
            nd = fd - disp
            nd = np.minimum(nd, d_e_v[:, None] - 0.04 * np.arange(nd.shape[1])[None, :] / nd.shape[1])
        sd['fd'] = np.where(carved[:, None], nd, fd)
        sd['face'] = steep * carved[:, None]
        sd['hrel'] = h

    # ---- assemble columns (increasing d)
    cols_d, cols_y, cols_kind = [], [], []
    # kinds: 0 natural, 1 cut face/ fill face, 2 shoulder/platform, 3 under-road, 4 ditch, 5 apron
    fev = sides['v']['fe_d']
    q = graded01(K_VO)[::-1]                                   # 1 .. small
    dvo = fev[:, None] + (DV - fev[:, None]) * q[None, :]      # from DV towards the face end (exclusive)
    cols_d.append(dvo)
    cols_y.append(None)
    cols_kind.append(np.zeros_like(dvo))
    vf_d = sides['v']['fd'][:, ::-1]
    vf_y = sides['v']['fy'][:, ::-1]
    cols_d.append(vf_d)
    cols_y.append(vf_y)
    cols_kind.append(np.ones_like(vf_d))
    tp = np.linspace(0, 1, K_VP + 2)[1:-1]
    dvp = d_e_v[:, None] + (-3.05 - d_e_v[:, None]) * tp[None, :]
    ye_plain = -0.10 - (0.018 - 0.009 * pw) * (np.abs(d_e_v) - 3.9)
    yvp = -0.062 + (ye_plain[:, None] + 0.062) * tp[None, :]
    # compacted gravel: gentle ruts / low spots across the bench (never at the road edge or the lip)
    yvp = yvp + (0.035 * NS3.perlin2(s[:, None] / 2.6, dvp / 2.6) * np.sin(np.pi * tp)[None, :]) * pw[:, None]
    cols_d.append(dvp)
    cols_y.append(yvp)
    cols_kind.append(np.full(dvp.shape, 2.0))
    dro = np.broadcast_to(ROAD_COLS[None, :], (nr, ROAD_COLS.size)).copy()
    yro = -0.02 * np.abs(dro) - 0.045
    cols_d.append(dro)
    cols_y.append(yro)
    cols_kind.append(np.full(dro.shape, 3.0))
    ddi = np.broadcast_to(DITCH_COLS[None, :], (nr, DITCH_COLS.size)).copy()
    ydi = np.broadcast_to(np.array([-0.2, -0.36, -0.30, -0.10])[None, :], ddi.shape).copy()
    ydi += 0.04 * NS.perlin2(s[:, None] / 3.0, ddi)
    cols_d.append(ddi)
    cols_y.append(ydi)
    cols_kind.append(np.full(ddi.shape, 4.0))
    tp = np.linspace(0, 1, K_UP + 2)[1:-1]
    dup = 4.3 + (d_e_u[:, None] - 4.3) * tp[None, :]
    cols_d.append(dup)
    cols_y.append(np.full(dup.shape, -0.10))
    cols_kind.append(np.full(dup.shape, 5.0))
    uf_d, uf_y = sides['u']['fd'], sides['u']['fy']
    cols_d.append(uf_d)
    cols_y.append(uf_y)
    cols_kind.append(np.ones_like(uf_d))
    feu = sides['u']['fe_d']
    q = graded01(K_UO)
    duo = feu[:, None] + (DU - feu[:, None]) * q[None, :]
    cols_d.append(duo)
    cols_y.append(None)
    cols_kind.append(np.zeros_like(duo))

    Dm = np.concatenate(cols_d, axis=1)
    nc = Dm.shape[1]
    # fold guards: outer columns strictly monotonic and face ends inside the corridor
    assert np.all(np.diff(dvo, axis=1) > 0) and np.all(np.diff(duo, axis=1) > 0), 'outer corridor columns fold'
    assert np.all(fev > DV + 2) and np.all(feu < DU - 2), ('face end outside corridor', fev.min(), feu.max())
    Ym = np.zeros_like(Dm)
    kind = np.concatenate(cols_kind, axis=1)
    Sm = np.broadcast_to(s[:, None], Dm.shape).copy()
    # heights: natural for kind 0
    off = 0
    for cd, cy in zip(cols_d, cols_y):
        k = cd.shape[1]
        if cy is not None:
            Ym[:, off:off + k] = cy
        off += k
    nat = kind == 0
    Wp = P[:, None, :] + L[:, None, :] * Dm[..., None]
    Hn = natural_chunked(Wp[..., 0][nat], Wp[..., 2][nat], Dm[nat], Sm[nat], hi_weight(Sm[nat], Dm[nat]))
    Yabs = Ym + yc[:, None]
    Yabs[nat] = Hn
    # trench carve in the road-bed / ditch / platform zones
    bed = (kind >= 2)
    tdep = trench_depth(Sm, Dm)
    Yabs = np.where(bed, np.minimum(Yabs, yc[:, None] - tdep - 0.045 * (tdep > 0.05)), Yabs)
    # natural (non carved) rows: everything natural
    nat_rows = ~carved
    if nat_rows.any():
        Hr = natural_chunked(Wp[nat_rows][..., 0], Wp[nat_rows][..., 2], Dm[nat_rows], Sm[nat_rows],
                             hi_weight(Sm[nat_rows], Dm[nat_rows])).reshape(Dm[nat_rows].shape)
        Yabs[nat_rows] = Hr
        kind[nat_rows] = 0
    # portal: terrain directly above the headwall at least up to its coping
    pz = (mode[:, None] >= 1) & (np.abs(Dm) < 17.5) & (Sm < TUN + 22)
    Yabs = np.where(pz, np.maximum(Yabs, yc[:, None] + 16.7 + 0.12 * (Sm - TUN) + 0.02 * np.abs(Dm)), Yabs)
    midrow = mode == 1
    if midrow.any():
        i = np.where(midrow)[0][0]
        inside = np.abs(Dm[i]) < 15.4
        Yabs[i] = np.where(inside, yc[i] + 16.55, 0.5 * (Yabs[i - 1] + Yabs[i + 1]))
    # portal bluff: the spur face rising behind the headwall coping is steep but smooth (low-frequency natural()).
    # Push it into the mountain along the road tangent with joint-facet / strata relief so it reads as fractured rock.
    bl = (mode[:, None] == 2) & (Sm < TUN + 30) & (np.abs(Dm) < 26)
    bdisp = np.zeros_like(Dm)
    if bl.any():
        # the spur surface jumps ~17 m right behind the coping (one row): replace it by a steep (~72-78 deg) face
        # climbing from the coping to natural ground over the fine rows TUN .. TUN+9
        wl = 1 - ss(17.0, 25.0, np.abs(Dm))
        kf = 3.3 + 0.9 * ss(-0.3, 0.4, NS.perlin2(Dm / 7.0, 3.3))
        ramp = yc[:, None] + 16.75 + kf * np.maximum(Sm - TUN, 0)
        Yabs = np.where(bl, lerp(Yabs, np.minimum(Yabs, ramp), wl), Yabs)
        # rock relief on the 45-deg spur slope above (vertical displacement: rows are 0.5-1 m apart)
        hb0 = Yabs - (yc[:, None] + 17.0)
        rel = NS3.fbm2(Dm / 3.2, Sm / 3.2, 4)
        rel = 0.9 * rel + 0.5 * np.abs(NS4.perlin2(Dm / 9.0, Sm / 9.0)) - 0.25
        Yabs = Yabs + np.where(bl, rel * ss(1.5, 6.0, hb0) * (1 - ss(TUN + 20, TUN + 30, Sm)) * wl, 0)
        hb = Yabs - (yc[:, None] + 17.0)
        slope_s = np.abs(np.gradient(Yabs, s, axis=0))
        dsp = displace_face(Dm, np.maximum(hb, 0), 1.0, np.full(Dm.shape, 200.0))
        dsp = 0.45 + 1.4 * dsp + 0.4 * NS2.fbm2(Dm / 5.0, hb / 5.0, 3)
        w = ss(0.25, 2.5, hb) * (1 - ss(TUN + 16, TUN + 30, Sm)) * (1 - ss(17, 26, np.abs(Dm))) * ss(1.3, 3.0, slope_s)
        bdisp = np.clip(dsp, 0, None) * w * bl
        log('portal bluff displacement: max %.2f m on %d verts' % (bdisp.max(), int((bdisp > 0.05).sum())))
    Tn = R.tangent(s)
    Tn[:, 1] = 0
    Tn /= np.linalg.norm(Tn, axis=1, keepdims=True)
    # seam: blend outer region to the far grid's exact surface
    Wp = P[:, None, :] + L[:, None, :] * Dm[..., None] + Tn[:, None, :] * bdisp[..., None]
    ws = seam_weight(Sm, Dm)
    sel = ws > 0
    Yabs[sel] = lerp(Yabs[sel], far_interp(F, Wp[..., 0][sel], Wp[..., 2][sel]), ws[sel])

    V = np.stack([Wp[..., 0], Yabs, Wp[..., 2]], axis=-1).reshape(-1, 3)
    Fc = grid_faces(nr, nc)
    # delete the step faces between row (TUN-0.02) and the mid row inside the headwall span
    if midrow.any():
        i = np.where(midrow)[0][0]
        a = Fc // nc
        rows_f = a.min(axis=1)
        cmax = np.abs(Dm.ravel()[Fc]).max(axis=1)
        kill = (rows_f == i - 1) & (cmax < 15.45)
        Fc = Fc[~kill]
    feat = dict(s=Sm.ravel(), d=Dm.ravel(), kind=kind.ravel(), yc=np.broadcast_to(yc[:, None], Dm.shape).ravel(),
                face_u=np.concatenate([np.zeros((nr, nc - K_UO - K_UF - 1)), sides['u']['face'], np.zeros((nr, K_UO))], axis=1).ravel(),
                )
    info = dict(s=s, fe_u=feu, fe_v=fev, d_e_u=d_e_u, d_e_v=d_e_v, tt=tt, nc=nc, nr=nr, mode=mode)
    log('near grid', nr, 'x', nc, '=', V.shape[0], 'verts', Fc.shape[0], 'tris')
    return V, Fc, feat, info


# ==============================================================================================
# masks (COLOR_0): R = AO, G = gravel, B = mud/wet, A = vegetation density
# ==============================================================================================
def terrain_masks(V, N, M, extra=None):
    ny = N[:, 1]
    n = V.shape[0]
    X, Z = V[:, 0], V[:, 2]
    G = np.zeros(n)
    B = np.zeros(n)
    noise = NS.fbm2(X / 7, Z / 7, 3)
    B = np.maximum.reduce([B, M['scar'] * 0.95 * (1 - 0.85 * M['rib']), M['lobe'], M['gully'] * 0.75, M['ravine'], M['river'] * 0.7])
    veg = ss(0.54, 0.74, ny)
    clear = ss(-0.45, 0.1, NS2.fbm2(X / 90, Z / 90, 3))
    A = veg * (1 - M['scar']) * (1 - M['lobe']) * (1 - 0.9 * M['gully']) * (1 - M['ravine']) * M['treeline']
    A *= 0.6 + 0.4 * clear
    A *= 1 - 0.85 * M['rockzone'] * ss(0.8, 0.62, ny)
    if extra is not None:
        kind = extra['kind']
        d = extra['d']
        s = extra['s']
        hrel = V[:, 1] - extra['yc']
        shoulder = (kind == 2) | (kind == 5)
        G = np.where(shoulder, 0.85 + 0.15 * noise, G)
        G = np.where(kind == 4, 0.55, G)
        G = np.where(kind == 3, 0.9, G)
        fill = (kind == 1) & (d < 0) & (hrel < -0.2)
        G = np.where(fill, 0.45 + 0.2 * noise, G)
        talus = (kind == 1) & (d > 0) & (extra['face_u'] < 0.5) & (hrel < 2.2) & (hrel > -0.2)
        G = np.where(talus, 0.75, G)
        # inside the scar the toe bank and ditch are slide mud, not gravel
        G = np.where((kind >= 1) & (d > 3.0), G * (1 - M['scar']), G)
        B = np.where(kind == 4, np.maximum(B, 0.4), B)
        tr = 1 - ss(2.0, 4.5, np.abs(s - GAP))
        B = np.maximum(B, tr * (np.abs(d) < 9))
        scar_road = ss(SCAR[0] - 10, SCAR[0] + 10, s) * (1 - ss(SCAR[1] - 10, SCAR[1] + 15, s))
        B = np.where((kind >= 2) & (d > -1), np.maximum(B, scar_road * 0.6), B)
        # water seeping down the cut faces
        streak = ss(0.25, 0.6, NS3.perlin2(s / 2.3, 0.3) * 0.7 + NS.perlin2(s / 9, 4.4) * 0.5)
        B = np.where(extra['face_u'] > 0.3, np.maximum(B, streak * 0.55), B)
        A = np.where(kind >= 1, A * (kind == 1) * (d < 0) * (hrel < -1.5) * 0.6, A)  # fills can host scrub
        A = np.where(extra['face_u'] > 0.2, 0, A)
        near_road = (np.abs(d) < 7.0)
        A = np.where(near_road, 0, A)
        G = np.clip(G, 0, 1)
    # AO fallback (overwritten by the Cycles bake)
    ao = np.clip(0.55 + 0.45 * ny, 0.45, 1.0) * (1 - 0.25 * M['gully'] - 0.25 * M['ravine'])
    return np.stack([ao, np.clip(G, 0, 1), np.clip(B, 0, 1), np.clip(A, 0, 1)], axis=1)


# ==============================================================================================
# road strip
# ==============================================================================================
ROAD_D = np.array([-3.05, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.05])
FINE_D = np.linspace(-3.05, 3.05, 33)


def road_y(s, d):
    return R.height(s) - 0.02 * np.abs(d)


def trench_edge(d, side):
    j = 0.08 + 0.26 * np.abs(NS.perlin2(d / 0.75, 1.3 + side)) + 0.1 * Noise.hash2(np.round(d * 9).astype(np.int64), side + 5, 2)
    return GAP + side * (1.3 + j)


def zipper(a_idx, a_d, b_idx, b_d, a_before=True):
    """Triangulate between two polylines (sorted by d). a is the row with smaller s if a_before."""
    tris = []
    i = j = 0
    while i < len(a_idx) - 1 or j < len(b_idx) - 1:
        adv_a = j >= len(b_idx) - 1 or (i < len(a_idx) - 1 and a_d[i + 1] <= b_d[j + 1])
        if adv_a:
            t = (a_idx[i], b_idx[j], a_idx[i + 1]) if a_before else (a_idx[i], a_idx[i + 1], b_idx[j])
            i += 1
        else:
            t = (a_idx[i], b_idx[j], b_idx[j + 1]) if a_before else (a_idx[i], b_idx[j + 1], b_idx[j])
            j += 1
        tris.append(t)
    # orientation: rows along +s, cols +d -> normal up when (a(s0), b(s1), a_next) ... checked numerically below
    return np.array(tris, dtype=np.int64)


class MeshAcc:
    def __init__(self):
        self.V, self.UV, self.C, self.F = [], [], [], []
        self.n = 0

    def add(self, V, UV, C, F):
        self.V.append(V)
        self.UV.append(UV)
        self.C.append(C)
        self.F.append(F + self.n)
        self.n += V.shape[0]
        return self.n - V.shape[0]

    def done(self):
        return (np.concatenate(self.V), np.concatenate(self.UV), np.concatenate(self.C), np.concatenate(self.F))


def road_color(s, d):
    """R = AO, G = puddle / damage mask."""
    edge = ss(2.0, 2.95, np.abs(d)) * 0.55
    rut = np.exp(-((np.abs(d) - 0.75) / 0.35) ** 2) * 0.25 + np.exp(-((np.abs(d) - 2.25) / 0.35) ** 2) * 0.25
    n1 = ss(0.0, 0.5, NS2.fbm2(s / 11, d / 3.5, 3))
    dmg = (1 - ss(3, 14, np.abs(s - GAP))) + 0.6 * ss(SCAR[0] - 5, SCAR[0] + 15, s) * (1 - ss(SCAR[1] - 10, SCAR[1] + 20, s))
    g = np.clip((edge + rut) * (0.4 + 0.9 * n1) + 0.5 * dmg * n1, 0, 1)
    ao = np.ones_like(s) - 0.12 * ss(2.4, 3.05, np.abs(d))
    return np.stack([ao, g, np.zeros_like(s), np.ones_like(s)], axis=-1)


def road_grid(s_rows, dcols, acc, skirts=True):
    nr, nc = s_rows.size, dcols.size
    S, Dg = np.meshgrid(s_rows, dcols, indexing='ij')
    W = R.world(S, Dg)
    W[..., 1] = road_y(S, Dg)
    V = W.reshape(-1, 3)
    UV = np.stack([Dg.ravel(), S.ravel()], axis=1)
    base = acc.add(V, UV, road_color(S.ravel(), Dg.ravel()), grid_faces(nr, nc))
    if skirts:
        for side in (-1, 1):
            dd = side * 3.05
            top = R.world(s_rows, np.full(nr, dd))
            top[:, 1] = road_y(s_rows, np.full(nr, dd)) + 0.0
            bot = R.world(s_rows, np.full(nr, dd + side * 0.02))
            bot[:, 1] = road_y(s_rows, np.full(nr, dd)) - 0.09
            Vs = np.concatenate([top, bot])
            UVs = np.concatenate([np.stack([np.full(nr, dd), s_rows], 1), np.stack([np.full(nr, dd + side * 0.09), s_rows], 1)])
            Cs = road_color(np.concatenate([s_rows, s_rows]), np.full(2 * nr, dd))
            Cs[:, 1] = 1.0
            Cs[nr:, 0] = 0.6
            Fs = grid_faces(2, nr)  # rows: top/bot ; cols: s
            # map: grid_faces indexes r*nc+c with r in {0,1}: our layout is [top..., bot...]
            if side > 0:
                Fs = Fs[:, ::-1]
            acc.add(Vs, UVs, Cs, Fs)
    return base


def build_road():
    acc = MeshAcc()
    sA = np.arange(ROAD_S0, GAP - 5.0 + 1e-6, 0.5)
    sB = np.arange(GAP + 5.0, TUN_END + 1e-6, 0.5)
    road_grid(sA, ROAD_D, acc)
    road_grid(sB, ROAD_D, acc)
    for side in (-1, 1):
        # fine patch between the coarse strip (at GAP -/+ 5) and the jagged trench edge
        edge_s = trench_edge(FINE_D, side)
        rows = np.arange(GAP - 4.5, GAP - 1.75 + 1e-6, 0.5) if side < 0 else np.arange(GAP + 1.75, GAP + 4.5 + 1e-6, 0.5)
        nr = rows.size + 1
        S = np.zeros((nr, FINE_D.size))
        if side < 0:
            S[:-1] = rows[:, None]
            S[-1] = edge_s
        else:
            S[1:] = rows[:, None]
            S[0] = edge_s
        Dg = np.broadcast_to(FINE_D[None, :], S.shape)
        W = R.world(S, Dg)
        W[..., 1] = road_y(S, Dg)
        base = acc.add(W.reshape(-1, 3), np.stack([Dg.ravel(), S.ravel()], 1), road_color(S.ravel(), Dg.ravel()), grid_faces(nr, FINE_D.size))
        # zipper to the coarse strip
        coarse_s = GAP - 5.0 if side < 0 else GAP + 5.0
        Wc = R.world(np.full(ROAD_D.size, coarse_s), ROAD_D)
        Wc[:, 1] = road_y(np.full(ROAD_D.size, coarse_s), ROAD_D)
        cb = acc.add(Wc, np.stack([ROAD_D, np.full(ROAD_D.size, coarse_s)], 1), road_color(np.full(ROAD_D.size, coarse_s), ROAD_D), np.zeros((0, 3), dtype=np.int64))
        frow = np.arange(FINE_D.size) + (base if side < 0 else base + (nr - 1) * FINE_D.size)
        crow = np.arange(ROAD_D.size) + cb
        if side < 0:
            tris = zipper(crow, ROAD_D, frow, FINE_D, a_before=True)
        else:
            tris = zipper(frow, FINE_D, crow, ROAD_D, a_before=True)
        acc.F.append(tris)
        # broken edge thickness (vertical face into the trench)
        er = base + ((nr - 1) * FINE_D.size if side < 0 else 0) + np.arange(FINE_D.size)
        top = W[-1] if side < 0 else W[0]
        bot = top.copy()
        drop = 0.12 + 0.06 * NS3.perlin2(FINE_D / 0.5, 2.0 + side)
        bot[:, 1] -= drop
        t_h = R.tangent(edge_s)
        t_h[:, 1] = 0
        t_h /= np.linalg.norm(t_h, axis=1, keepdims=True)
        bot -= t_h * side * (0.03 + 0.03 * NS.perlin2(FINE_D / 0.3, 9.0))[:, None] * -1
        cbot = road_color(edge_s, FINE_D)
        cbot[:, 0] = 0.7
        cbot[:, 1] = 1.0
        bb = acc.add(bot, np.stack([FINE_D, edge_s - side * drop], 1), cbot, np.zeros((0, 3), dtype=np.int64))
        brow = np.arange(FINE_D.size) + bb
        nF = FINE_D.size
        tri = []
        for j in range(nF - 1):
            a, b, c, d = er[j], er[j + 1], brow[j + 1], brow[j]
            if side < 0:
                tri += [(a, d, c), (a, c, b)]
            else:
                tri += [(a, b, c), (a, c, d)]
        acc.F.append(np.array(tri, dtype=np.int64))
    V, UV, C, Fc = acc.done()
    # fix triangle winding so all faces point up / outward
    N = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]])
    ctr = V[Fc].mean(axis=1)
    sc, dc = R.project(ctr[:, 0], ctr[:, 2])
    up = N[:, 1]
    horiz = np.abs(up) < 0.3 * np.linalg.norm(N, axis=1)
    # horizontal faces: must face up. vertical faces: skirts face outward (|d| grows), trench faces face the gap
    Lc = R.left(sc)
    Tc = R.tangent(sc)
    outward = np.where(np.abs(sc - GAP) < 3.2, np.sign(GAP - sc) * np.einsum('ij,ij->i', N, Tc), np.sign(dc) * np.einsum('ij,ij->i', N, Lc))
    bad = np.where(horiz, outward < 0, up < 0)
    Fc[bad] = Fc[bad][:, ::-1]
    log('road', V.shape[0], 'verts', Fc.shape[0], 'tris; flipped', bad.sum())
    return V, UV, C, Fc


# ==============================================================================================
# tunnel: portal headwall + wing walls + tube + end cap; lamps
# ==============================================================================================
def tunnel_profile():
    pts = [(-3.05, -0.061), (-3.05, 0.17), (-3.25, 0.2), (-4.75, 0.2), (-4.9, 0.35), (-4.9, 1.9)]
    arc = [(-4.9 * math.cos(t), 1.9 + 5.3 * math.sin(t)) for t in np.linspace(0, math.pi, 29)[1:-1]]
    pts += arc
    pts += [(4.9, 1.9), (4.9, 0.35), (4.75, 0.2), (3.25, 0.2), (3.05, 0.17), (3.05, -0.061)]
    return np.array(pts)


def frame_at(s):
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    P = R.point(s)
    L = R.left(s)
    T = R.tangent(s)
    T[:, 1] = 0
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return P, L, T


def box(acc, c, hx, hy, hz, ax, ay, az, color=(1, 0.5, 0.2, 0)):
    """Oriented box. ax/ay/az: unit axes (3,)."""
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(c + sx * hx * ax + sy * hy * ay + sz * hz * az)
    Cn = np.array(corners)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    V, UV, F = [], [], []
    for q in quads:
        base = len(V)
        for k in q:
            V.append(Cn[k])
        pv = Cn[list(q)]
        e1 = pv[1] - pv[0]
        e2 = pv[3] - pv[0]
        UV += [(0, 0), (np.linalg.norm(e1), 0), (np.linalg.norm(e1), np.linalg.norm(e2)), (0, np.linalg.norm(e2))]
        F += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    V = np.array(V)
    F = np.array(F)
    # outward orientation
    cen = V.mean(axis=0)
    for i, f in enumerate(F):
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        if np.dot(n, V[f].mean(axis=0) - cen) < 0:
            F[i] = f[::-1]
    acc.add(V, np.array(UV, dtype=np.float64), np.tile(np.array(color, dtype=np.float64), (V.shape[0], 1)), F)


def build_tunnel():
    acc = MeshAcc()
    prof = tunnel_profile()
    npf = prof.shape[0]
    arcl = np.concatenate([[0], np.cumsum(np.hypot(np.diff(prof[:, 0]), np.diff(prof[:, 1])))])
    s_rows = np.concatenate([[TUN - 1.2], np.arange(TUN - 0.5, TUN_END - 0.25, 1.0), [TUN_END]])  # last row = end cap ring
    P, L, T = frame_at(s_rows)
    V = (P[:, None, :] + L[:, None, :] * prof[None, :, 0:1] + np.array([0, 1.0, 0])[None, None, :] * prof[None, :, 1:2])
    Vt = V.reshape(-1, 3)
    UV = np.stack([np.broadcast_to(arcl[None, :], V.shape[:2]).ravel(), np.broadcast_to(s_rows[:, None], V.shape[:2]).ravel()], 1)
    h = prof[None, :, 1] * np.ones((s_rows.size, 1))
    grime = np.clip(1 - h / 2.2, 0, 1).ravel()
    C = np.stack([np.ones(Vt.shape[0]) * (0.85 + 0.15 * (h.ravel() > 0.3)), grime, grime * 0.5, np.ones(Vt.shape[0])], 1)
    acc.add(Vt, UV, C, grid_faces(s_rows.size, npf, flip=True))
    # end cap (dark wall)
    Pe, Le, Te = frame_at(TUN_END)
    ring = Pe[0] + Le[0] * prof[:, 0:1] + np.array([0, 1.0, 0]) * prof[:, 1:2]
    cen = Pe[0] + np.array([0, 3.0, 0])
    Vc = np.concatenate([ring, cen[None]])
    Fcap = np.array([(i, i + 1, npf) for i in range(npf - 1)] + [(npf - 1, 0, npf)])
    n = np.cross(Vc[Fcap[0, 1]] - Vc[Fcap[0, 0]], Vc[Fcap[0, 2]] - Vc[Fcap[0, 0]])
    if np.dot(n, Te[0]) > 0:
        Fcap = Fcap[:, ::-1]
    # UV.y = s (the tunnel material keys its daylight falloff on uv.y for tube parts: the cap must read as deep inside)
    acc.add(Vc, np.stack([prof[:, 0].tolist() + [0], [TUN_END] * (npf + 1)], 1), np.tile([0.3, 0.6, 0.3, 1], (npf + 1, 1)), Fcap)
    # rock cover over the crown (the tube must stay buried all the way to the cap)
    sc_ = np.arange(TUN + 5, TUN_END + 0.1, 5.0)
    Pc, Lc_, _ = frame_at(sc_)
    cover = [float((natural(np.array([p[0] + l[0] * dd]), np.array([p[2] + l[2] * dd]))[0] - p[1] - 7.2)) for p, l in zip(Pc, Lc_) for dd in (-5.0, 0.0, 5.0)]
    log('tunnel rock cover over crown: min %.1f m' % min(cover))

    # ---- headwall (front face at TUN-0.95, back at TUN+0.6), with the opening
    P0, L0, T0v = frame_at(TUN)
    P0, L0, T0v = P0[0], L0[0], T0v[0]
    Y = np.array([0, 1.0, 0])
    HW, HH, HB = 15.2, 16.4, -0.8
    open_poly = np.concatenate([prof, [(3.05, -0.061), (-3.05, -0.061)]])
    cen2 = np.array([0.0, 3.2])

    def ray_poly(dirv, poly):
        best = 1e9
        for i in range(len(poly) - 1):
            a, b = poly[i], poly[i + 1]
            e = b - a
            M = np.array([[dirv[0], -e[0]], [dirv[1], -e[1]]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            t, u = np.linalg.solve(M, a - cen2)
            if t > 0 and 0 <= u <= 1:
                best = min(best, t)
        return cen2 + dirv * best

    def ray_rect(dirv):
        ts = []
        for bound, k in ((HW, 0), (-HW, 0), (HH, 1), (HB, 1)):
            if abs(dirv[k]) > 1e-9:
                t = (bound - cen2[k]) / dirv[k]
                if t > 0:
                    p = cen2 + dirv * t
                    if -HW - 1e-6 <= p[0] <= HW + 1e-6 and HB - 1e-6 <= p[1] <= HH + 1e-6:
                        ts.append(t)
        return cen2 + dirv * min(ts)

    Mang = 120
    angs = np.linspace(0, 2 * math.pi, Mang, endpoint=False)
    # ensure rectangle corners are hit exactly
    corners = [(HW, HH), (-HW, HH), (-HW, HB), (HW, HB)]
    angs = np.sort(np.concatenate([angs, [math.atan2(c[1] - cen2[1], c[0] - cen2[0]) % (2 * math.pi) for c in corners]]))
    inner, outer = [], []
    for a in angs:
        dv = np.array([math.cos(a), math.sin(a)])
        inner.append(ray_poly(dv, np.concatenate([open_poly, open_poly[:1]])))
        outer.append(ray_rect(dv))
    inner, outer = np.array(inner), np.array(outer)

    def to_w(p2, depth):
        return P0 + L0 * p2[:, 0:1] + Y * p2[:, 1:2] + T0v * depth

    fdepth = -0.95
    collar = inner + (inner - cen2) / np.linalg.norm(inner - cen2, axis=1, keepdims=True) * 0.85
    collar[:, 1] = np.maximum(collar[:, 1], inner[:, 1])
    Vi = to_w(inner, fdepth - 0.25)
    Vco = to_w(collar, fdepth - 0.25)
    Vco2 = to_w(collar, fdepth)
    Vo = to_w(outer, fdepth)
    m = len(angs)
    rings = [Vi, Vco, Vco2, Vo]
    uvs = [inner, collar, collar, outer]
    base = []
    for Vr, uv in zip(rings, uvs):
        base.append(acc.add(Vr, uv.copy(), np.tile([0.9, 0.4, 0.1, 0], (m, 1)), np.zeros((0, 3), dtype=np.int64)))
    tri = []
    for k in range(3):
        for j in range(m):
            j2 = (j + 1) % m
            a, b, c, d = base[k] + j, base[k] + j2, base[k + 1] + j2, base[k + 1] + j
            tri += [(a, b, c), (a, c, d)]
    tri = np.array(tri, dtype=np.int64)
    # remove faces fully below the road inside the opening (bottom sliver)
    Vall = np.concatenate(acc.V)
    ctr = Vall[tri].mean(axis=1)
    rel = ctr - P0
    hh = rel @ Y
    dd_ = rel @ L0
    keep = ~((hh < -0.05) & (np.abs(dd_) < 3.2))
    tri = tri[keep]
    n = np.cross(Vall[tri[:, 1]] - Vall[tri[:, 0]], Vall[tri[:, 2]] - Vall[tri[:, 0]])
    flip = (n @ T0v) > 0
    tri[flip] = tri[flip][:, ::-1]
    acc.F.append(tri)
    # headwall coping, top and ends
    box(acc, P0 + Y * (HH + 0.3) + T0v * (-0.25), HW + 0.35, 0.3, 0.95, L0, Y, T0v, (0.85, 0.4, 0.1, 0))
    box(acc, P0 + L0 * (HW + 0.3) + Y * (HH + HB) / 2 + T0v * (-0.2), 0.3, (HH - HB) / 2, 0.8, L0, Y, T0v, (0.85, 0.4, 0.1, 0))
    box(acc, P0 - L0 * (HW + 0.3) + Y * (HH + HB) / 2 + T0v * (-0.2), 0.3, (HH - HB) / 2, 0.8, L0, Y, T0v, (0.85, 0.4, 0.1, 0))
    # wing walls: continuous flared retaining walls from the headwall ends back to the ditch line
    ws = np.arange(1131.0, TUN - 0.94, 0.5)
    ws = np.append(ws, TUN - 0.95)
    Pw, Lw, Tw = frame_at(ws)
    prw = np.clip((ws - 1131.0) / (1149.5 - 1131.0), 0, 1)
    hgt = 1.2 + 15.2 * prw ** 1.2
    for side in (-1, 1):
        wd = (4.6 + 10.6 * prw) if side > 0 else (3.9 + 11.3 * prw)
        dF = side * (wd - 0.75)
        dB = side * (wd + 0.35)
        yb_ = Pw[:, 1]
        rows = np.stack([
            Pw + Lw * dF[:, None] + Y * (-0.4),
            Pw + Lw * dF[:, None] + Y * hgt[:, None],
            Pw + Lw * (dF + side * 0.25)[:, None] + Y * (hgt + 0.25)[:, None],
            Pw + Lw * dB[:, None] + Y * (hgt + 0.25)[:, None],
        ], axis=1)
        nr_, k_ = rows.shape[0], rows.shape[1]
        Vw = rows.reshape(-1, 3)
        rel = np.stack([np.zeros(nr_), hgt + 0.4, hgt + 0.65, hgt + 1.75], 1)
        UVw = np.stack([np.broadcast_to(ws[:, None], (nr_, k_)).ravel(), rel.ravel()], 1)
        Fw = grid_faces(nr_, k_)
        # orient: front faces toward the road, top faces up
        cen = Vw[Fw].mean(axis=1)
        nrm = np.cross(Vw[Fw[:, 1]] - Vw[Fw[:, 0]], Vw[Fw[:, 2]] - Vw[Fw[:, 0]])
        col = (Fw % k_).min(axis=1)
        Lc = R.left(R.project(cen[:, 0], cen[:, 2])[0])
        want = np.where(col[:, None] == 0, -side * Lc, np.array([0, 1.0, 0]))
        flip = np.einsum('ij,ij->i', nrm, want) < 0
        Fw[flip] = Fw[flip][:, ::-1]
        acc.add(Vw, UVw, np.tile([0.85, 0.45, 0.1, 0], (Vw.shape[0], 1)), Fw)
    V, UV, C, Fc = acc.done()
    log('tunnel', V.shape[0], 'verts', Fc.shape[0], 'tris')
    return V, UV, C, Fc


def build_lamps():
    acc = MeshAcc()
    Y = np.array([0, 1.0, 0])
    for side, off in ((1, 0.0), (-1, 4.0)):
        for s in np.arange(TUN + 4 + off, LAMP_END, 8.0):
            P, L, T = frame_at(s)
            dd = 2.45 * side
            hv = 1.9 + 5.3 * math.sqrt(1 - (dd / 4.9) ** 2) - 0.14
            c = P[0] + L[0] * dd + Y * hv
            box(acc, c, 0.17, 0.07, 0.62, L[0], Y, T[0], (1, 1, 1, 1))
    V, UV, C, Fc = acc.done()
    return V, UV, C, Fc


# ==============================================================================================
# scatter
# ==============================================================================================
def face_end_at(info, s, side):
    return np.interp(s, info['s'], info['fe_u'] if side > 0 else info['fe_v'])


NEAR_BVH = None   # mathutils BVHTree of the final (decimated) terrain_near, set in main() when run inside Blender


def surface_height(F, info, X, Z):
    """Final surface height: ray cast into the decimated terrain_near when available (exact), else the
    natural surface (with detail) inside the corridor and the far grid outside."""
    s, d = R.project(X, Z)
    inside = corridor_inside(s, d) > 0
    H = far_interp(F, X, Z)
    if inside.any():
        hn = natural_chunked(X[inside], Z[inside], d[inside], s[inside], hi_weight(s[inside], d[inside]))
        H[inside] = hn
    if NEAR_BVH is not None:
        from mathutils import Vector
        down = Vector((0.0, 0.0, -1.0))
        for i in np.where(inside)[0]:
            hit = NEAR_BVH.ray_cast(Vector((float(X[i]), float(-Z[i]), 3000.0)), down, 6000.0)
            if hit[0] is not None:
                H[i] = hit[0].z
    return H, s, d, inside


def build_scatter(F, info):
    trees = []
    # candidate points: jittered grids with density falling off away from the road
    cand = []
    for (x0, x1, z0, z1, h) in ((-420, 1500, -520, 430, 4.2), (-1500, 2600, -1600, 2400, 11.0)):
        xs = np.arange(x0, x1, h)
        zs = np.arange(z0, z1, h)
        XX, ZZ = np.meshgrid(xs, zs)
        XX = XX.ravel() + (RNG.random(XX.size) - 0.5) * h * 0.9
        ZZ = ZZ.ravel() + (RNG.random(ZZ.size) - 0.5) * h * 0.9
        cand.append((XX, ZZ, h))
    out = []
    for (XX, ZZ, h) in cand:
        near = h < 6
        dz = np.abs(ZZ - R.z_of_x(XX))
        if near:
            keep = dz < 330
        else:
            keep = (dz >= 250) | (XX < -420) | (XX > 1500)
        XX, ZZ = XX[keep], ZZ[keep]
        H, M = natural_chunked(XX, ZZ, None, None, None, masks=True)
        # slope by finite differences
        e = 1.5
        hx = natural_chunked(XX + e, ZZ, None, None, None)
        hz = natural_chunked(XX, ZZ + e, None, None, None)
        gx, gz = (hx - H) / e, (hz - H) / e
        ny = 1 / np.sqrt(1 + gx * gx + gz * gz)
        veg = ss(0.56, 0.76, ny) * (1 - M['scar']) * (1 - M['lobe']) * (1 - M['gully']) * (1 - M['ravine']) * M['treeline']
        veg *= 1 - 0.9 * M['rockzone'] * ss(0.8, 0.64, ny)
        forest = ss(-0.3, 0.3, NS2.fbm2(XX / 110, ZZ / 110, 3)) * 0.8 + 0.2
        dist = np.abs(M['c'])
        if near:
            dens = np.where(dist < 150, 1.0, 0.45)
            p = veg * forest * dens * 0.17
        else:
            p = veg * forest * 0.1 * (1 - ss(1800, 3200, np.hypot(XX - 600, ZZ - 300)))
        # scar margin: no trees within 8 m of the scar (broken zone)
        keep = RNG.random(XX.size) < p
        XX, ZZ = XX[keep], ZZ[keep]
        if XX.size == 0:
            continue
        s, d = R.project(XX, ZZ)
        inside = (corridor_inside(s, d) > -2)
        ok = np.ones(XX.size, dtype=bool)
        feu = face_end_at(info, s, 1)
        fev = face_end_at(info, s, -1)
        ok &= ~(inside & (d > -6.5) & (d < feu + 2.0))
        ok &= ~(inside & (d < 0) & (d > fev - 1.5))
        ok &= ~((s > TUN - 30) & (s < TUN_END + 10) & (np.abs(d) < 28) & inside)
        ok &= ~((np.abs(s - PULL[0] - 30) < 42) & (d < 0) & (d > -20) & inside)
        XX, ZZ = XX[ok], ZZ[ok]
        Hs, s2, d2, ins2 = surface_height(F, info, XX, ZZ)
        out.append(np.stack([XX, Hs, ZZ], 1))
    pts = np.concatenate(out)
    n = pts.shape[0]
    scale = np.clip(RNG.normal(1.0, 0.16, n), 0.65, 1.4)
    rot = RNG.random(n) * 2 * math.pi
    var = RNG.integers(0, 4, n)
    trees = np.concatenate([pts - np.array([0, 0.15, 0]), scale[:, None], rot[:, None], var[:, None]], axis=1)
    log('trees', n)

    # ---- rocks
    rocks = []
    # talus at the cut toe
    ss_ = np.arange(-150, TUN - 25, 1.6)
    ss_ = ss_ + RNG.random(ss_.size) * 1.2
    tt = np.interp(ss_, info['s'], info['tt'])
    keep = RNG.random(ss_.size) < (0.25 + 0.6 * ss(0.3, 1.2, tt))
    ss_ = ss_[keep]
    tt = tt[keep]
    dd = 4.35 + RNG.random(ss_.size) * np.maximum(tt, 0.3) * 1.4
    W = R.world(ss_, dd)
    sc = np.clip(RNG.lognormal(-1.35, 0.55, ss_.size), 0.1, 0.9)
    rocks.append((W, sc, 0.35))
    # boulders in gullies, scar lobes and the trench
    for g in GULLIES:
        k = 26
        sv = g + RNG.normal(0, 2.0, k)
        dv = RNG.uniform(6, 55, k)
        W = R.world(sv, dv)
        rocks.append((W, np.clip(RNG.lognormal(-0.4, 0.5, k), 0.25, 2.2), 0.3))
    # slide debris: boulders in clusters and trains along the flow (not a uniform sprinkle), a sparse
    # background of loners, a dense spill of cobbles/stones on the toe near the ditch
    k = 60
    sv = RNG.uniform(SCAR[0] + 2, SCAR[1] - 2, k)
    dv = RNG.uniform(8, 60, k)
    rocks.append((R.world(sv, dv), np.clip(RNG.lognormal(-0.2, 0.6, k), 0.3, 3.0), 0.4))
    ncl = 26
    cs = RNG.uniform(SCAR[0] + 6, SCAR[1] - 6, ncl)
    cd = RNG.uniform(7, 58, ncl)
    for c_s, c_d in zip(cs, cd):
        m = int(RNG.integers(5, 15))
        # elongated down the fall line (along d)
        sv = c_s + RNG.normal(0, 2.2, m)
        dv = np.clip(c_d + RNG.normal(0, 5.0, m), 5.0, 62)
        big = np.clip(RNG.lognormal(-0.25, 0.7, m), 0.2, 2.8)
        rocks.append((R.world(sv, dv), big, 0.4))
    k = 420
    sv = RNG.uniform(SCAR[0] + 4, SCAR[1] - 4, k)
    dv = 4.6 + RNG.gamma(1.6, 5.5, k)
    dv = np.clip(dv, 4.6, 45)
    rocks.append((R.world(sv, dv), np.clip(RNG.lognormal(-1.55, 0.55, k), 0.08, 0.9), 0.3))
    k = 14
    sv = GAP + RNG.uniform(-1.0, 1.0, k)
    dv = RNG.uniform(-6, 6, k)
    Wt = R.world(sv, dv)
    rocks.append((Wt, np.clip(RNG.lognormal(-1.2, 0.4, k), 0.12, 0.6), 0.2))
    # scattered boulders on the slopes
    k = 260
    sv = RNG.uniform(-150, 1140, k)
    dv = np.where(RNG.random(k) < 0.5, RNG.uniform(14, 58, k), RNG.uniform(-44, -8, k))
    rocks.append((R.world(sv, dv), np.clip(RNG.lognormal(-0.6, 0.6, k), 0.2, 2.4), 0.3))
    allW = np.concatenate([r[0] for r in rocks])
    allS = np.concatenate([r[1] for r in rocks])
    sink = np.concatenate([np.full(r[1].size, r[2]) for r in rocks])
    Hs, s2, d2, ins = surface_height(F, info, allW[:, 0], allW[:, 2])
    # talus rocks sit on the carved profile: approximate by the ditch/apron heights
    yc = R.height(s2)
    trench = np.abs(s2 - GAP) < 1.3
    if NEAR_BVH is None:   # approximate the carved profile (exact heights come from the BVH otherwise)
        fe = face_end_at(info, s2, 1)
        carved_zone = (d2 > 4.0) & (d2 < fe) & (s2 < TUN)
        Hs = np.where(carved_zone, yc - 0.1 + (d2 - 4.3) * 0.72, Hs)
        Hs = np.where(trench & (np.abs(d2) < 6.5), yc - 1.85, Hs)
    ok = ~((d2 > -4.5) & (d2 < 4.3) & ~trench)
    rk = np.stack([allW[:, 0], Hs - sink * allS, allW[:, 2], allS, RNG.random(allS.size) * 2 * math.pi,
                   RNG.uniform(-0.6, 0.6, allS.size), RNG.integers(0, 10, allS.size)], 1)[ok]
    log('rocks', rk.shape[0])

    # ---- deadwood: snapped / uprooted spruce trunks carried by the slide, a few in the gullies and the trench
    # [x, y, z, length, rotY, pitch, radius]
    dw = []
    k = 34
    sv = RNG.uniform(SCAR[0] + 3, SCAR[1] - 3, k)
    dv = np.clip(5.5 + RNG.gamma(1.5, 9.0, k), 5.5, 58)
    dw.append((sv, dv, RNG.uniform(3.5, 13.0, k), RNG.uniform(0.13, 0.28, k)))
    for g in GULLIES:
        m = 3
        dw.append((g + RNG.normal(0, 1.5, m), RNG.uniform(6, 40, m), RNG.uniform(2.5, 8.0, m), RNG.uniform(0.1, 0.22, m)))
    dw.append((GAP + RNG.uniform(-0.8, 0.8, 2), RNG.uniform(-5, 5, 2), RNG.uniform(3.0, 5.0, 2), RNG.uniform(0.1, 0.16, 2)))
    dS = np.concatenate([a[0] for a in dw])
    dD = np.concatenate([a[1] for a in dw])
    dL = np.concatenate([a[2] for a in dw])
    dR = np.concatenate([a[3] for a in dw])
    Wd = R.world(dS, dD)
    Hd, s3, d3, _ = surface_height(F, info, Wd[:, 0], Wd[:, 2])
    trd = np.abs(s3 - GAP) < 1.3
    if NEAR_BVH is None:
        Hd = np.where(trd & (np.abs(d3) < 6.5), R.height(s3) - 1.8, Hd)
    # mostly lying across or down the slope, some tilted (propped on debris)
    Tg = R.tangent(dS)
    yaw_road = np.arctan2(Tg[:, 0], Tg[:, 2])
    rotY = yaw_road + np.where(RNG.random(dS.size) < 0.5, RNG.normal(0, 0.5, dS.size), np.pi / 2 + RNG.normal(0, 0.6, dS.size))
    pitch = np.clip(RNG.normal(0.0, 0.18, dS.size), -0.45, 0.45)
    okd = ~((d3 > -4.5) & (d3 < 4.4) & ~trd)
    deadwood = np.stack([Wd[:, 0], Hd + dR * 0.4, Wd[:, 2], dL, rotY, pitch, dR], 1)[okd]
    log('deadwood', deadwood.shape[0])

    # ---- slide spawns: gullies first (index = gully index), then the scar
    spawns = []
    for i, g in enumerate(GULLIES):
        pts = []
        for dd_ in np.linspace(26, 57, 7):
            for off in (-1.2, 0.0, 1.2):
                pts.append((g + off, dd_))
        pts = np.array(pts)
        W = R.world(pts[:, 0], pts[:, 1])
        Hh, _, _, _ = surface_height(F, info, W[:, 0], W[:, 2])
        W[:, 1] = Hh + 1.2
        hr = W[:, 1] - R.height(pts[:, 0])
        top = W[pts[:, 1].argmax()]
        bot = R.world(np.array([g]), np.array([4.0]))[0]
        bot[1] = R.height(np.array([g]))[0]
        dv = bot - top
        dv /= np.linalg.norm(dv)
        spawns.append({'id': 'gully%d' % i, 'kind': 'gully', 's': g, 'points': np.round(W, 2).tolist(),
                       'heights': np.round(hr, 1).tolist(), 'dir': np.round(dv, 4).tolist()})
    pts = []
    for sv_ in np.linspace(SCAR[0] + 12, SCAR[1] - 12, 9):
        for dd_ in (32, 44, 56):
            pts.append((sv_, dd_))
    pts = np.array(pts)
    W = R.world(pts[:, 0], pts[:, 1])
    Hh, _, _, _ = surface_height(F, info, W[:, 0], W[:, 2])
    W[:, 1] = Hh + 1.5
    top = R.world(np.array([140.0]), np.array([56.0]))[0]
    top[1] = W[:, 1].max()
    bot = R.world(np.array([140.0]), np.array([4.0]))[0]
    dv = bot - top
    dv /= np.linalg.norm(dv)
    spawns.append({'id': 'scar', 'kind': 'scar', 's': 140.0, 'points': np.round(W, 2).tolist(),
                   'heights': np.round(W[:, 1] - R.height(pts[:, 0]), 1).tolist(), 'dir': np.round(dv, 4).tolist()})
    return trees, rk, spawns, deadwood


def grass_grid(near_V, near_C, feat):
    """Vegetation density grid in (s, d) road space for grass placement: 2 m x 1 m cells."""
    s0, ds, d0, dd = S0, 2.0, DV, 1.0
    ns = int((S1 - S0) / ds) + 1
    nd = int((DU - DV) / dd) + 1
    si = np.clip(np.round((feat['s'] - s0) / ds).astype(int), 0, ns - 1)
    di = np.clip(np.round((feat['d'] - d0) / dd).astype(int), 0, nd - 1)
    acc = np.zeros(ns * nd)
    cnt = np.zeros(ns * nd)
    np.add.at(acc, si * nd + di, near_C[:, 3])
    np.add.at(cnt, si * nd + di, 1)
    g = np.where(cnt > 0, acc / np.maximum(cnt, 1), -1)
    # fill holes from neighbours along d
    g = g.reshape(ns, nd)
    for _ in range(6):
        hole = g < 0
        if not hole.any():
            break
        nb = np.maximum(np.roll(g, 1, axis=1), np.roll(g, -1, axis=1))
        g = np.where(hole, nb, g)
    g = np.clip(g, 0, 1)
    b = np.round(g * 255).astype(np.uint8)
    return {'s0': s0, 'ds': ds, 'd0': d0, 'dd': dd, 'ns': ns, 'nd': nd, 'layout': 'row-major [s][d], uint8 density 0..255',
            'data': base64.b64encode(b.tobytes()).decode('ascii')}


# ==============================================================================================
# Blender stage: decimation + Cycles AO bake
# ==============================================================================================
def bl_mesh(name, V, F, colors=None, weights=None, float_attrs=None):
    import bpy
    Vb = to_blender(V).astype(np.float32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(Vb.shape[0])
    me.vertices.foreach_set('co', Vb.ravel())
    nf = F.shape[0]
    me.loops.add(nf * 3)
    me.loops.foreach_set('vertex_index', F.astype(np.int32).ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set('loop_start', np.arange(0, nf * 3, 3, dtype=np.int32))
    me.update(calc_edges=True)
    if colors is not None:
        ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
        ca.data.foreach_set('color', colors.astype(np.float32).ravel())
        me.color_attributes.active_color = ca
    if float_attrs:
        for k, v in float_attrs.items():
            at = me.attributes.new(k, 'FLOAT', 'POINT')
            at.data.foreach_set('value', v.astype(np.float32))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if weights is not None:
        vg = ob.vertex_groups.new(name='dec')
        # add vertices by weight bins (fast enough with a handful of bins)
        wq = np.round(weights * 10) / 10
        for w in np.unique(wq):
            idx = np.where(wq == w)[0].tolist()
            if w > 0 and idx:
                vg.add(idx, float(w), 'REPLACE')
    return ob


def bl_read(ob, float_names=()):
    me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get('co', co)
    V = from_blender(co.reshape(-1, 3).astype(np.float64))
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tri = np.empty(nt * 3, dtype=np.int32)
    me.loop_triangles.foreach_get('vertices', tri)
    F = tri.reshape(-1, 3).astype(np.int64)
    out = {}
    if 'Col' in me.color_attributes:
        c = np.empty(n * 4, dtype=np.float32)
        me.color_attributes['Col'].data.foreach_get('color', c)
        out['Col'] = c.reshape(-1, 4).astype(np.float64)
    for k in float_names:
        a = np.empty(n, dtype=np.float32)
        me.attributes[k].data.foreach_get('value', a)
        out[k] = a.astype(np.float64)
    return V, F, out


def bl_decimate(ob, ratio):
    import bpy
    md = ob.modifiers.new('dec', 'DECIMATE')
    md.decimate_type = 'COLLAPSE'
    md.ratio = ratio
    md.use_collapse_triangulate = True
    if ob.vertex_groups.get('dec'):
        md.vertex_group = 'dec'
        md.vertex_group_factor = 1000.0
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.modifier_apply(modifier=md.name)
    ob.select_set(False)


def bl_setup_cycles(samples):
    import bpy
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for dv in prefs.devices:
            dv.use = True
        sc.cycles.device = 'GPU'
        log('cycles devices', [(d.name, d.type, d.use) for d in prefs.devices])
    except Exception as e:  # noqa: BLE001
        log('GPU setup failed, using CPU', e)
    sc.cycles.samples = samples
    if sc.world is None:
        sc.world = bpy.data.worlds.new('W')
    sc.world.light_settings.distance = 30.0
    sc.render.bake.target = 'VERTEX_COLORS'
    sc.render.bake.margin = 0


def bl_bake_ao(obs):
    import bpy
    mat = bpy.data.materials.get('bakemat') or bpy.data.materials.new('bakemat')
    for ob in obs:
        if not ob.data.materials:
            ob.data.materials.append(mat)
        ca = ob.data.color_attributes.get('AO') or ob.data.color_attributes.new('AO', 'FLOAT_COLOR', 'POINT')
        ob.data.color_attributes.active_color = ca
    for ob in bpy.context.scene.objects:
        ob.select_set(False)
    for ob in obs:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    t = time.time()
    bpy.ops.object.bake(type='AO', target='VERTEX_COLORS')
    log('AO bake', [o.name for o in obs], '%.1fs' % (time.time() - t))
    res = []
    for ob in obs:
        n = len(ob.data.vertices)
        c = np.empty(n * 4, dtype=np.float32)
        ob.data.color_attributes['AO'].data.foreach_get('color', c)
        res.append(c.reshape(-1, 4)[:, 0].astype(np.float64))
    return res


# ==============================================================================================
# chunking + export
# ==============================================================================================
def split_by(key_face, V, F, attrs):
    """Split mesh by integer key per face. Returns list of (key, V, F, attrs)."""
    out = []
    for k in np.unique(key_face):
        f = F[key_face == k]
        a2 = dict(attrs)
        a2['__V'] = V
        sub, f2 = compact(a2, f)
        V2 = sub.pop('__V')
        out.append((int(k), V2, f2, sub))
    return out


def main():
    F = build_far()
    V_n, F_n, feat, info = build_near(F)
    # faces fully under the asphalt strip are never visible (the road has its own collider): drop them,
    # except at the trench, beyond the strip ends and inside the tunnel
    fs, fd = feat['s'][F_n], feat['d'][F_n]
    hid = (np.abs(fd) < 2.9).all(axis=1) & (np.abs(fs - GAP) > 5.5).all(axis=1) & (fs > ROAD_S0 + 1).all(axis=1) & \
          (fs < TUN - 2).all(axis=1) & (feat['kind'][F_n] == 3).all(axis=1)
    F_n = F_n[~hid]
    log('near: dropped', int(hid.sum()), 'faces under the road strip')
    N_n = vertex_normals(V_n, F_n)
    _, Mn = natural_chunked(V_n[:, 0], V_n[:, 2], feat['d'], feat['s'], None, masks=True)
    C_n = terrain_masks(V_n, N_n, Mn, feat)

    # far mesh: faces, drop hidden faces deep inside the corridor
    nx, nz = F['xs'].size, F['zs'].size
    V_f = np.stack([F['X'], F['H'], F['Z']], 1)
    F_f = grid_faces(nz, nx)
    din = F['din']
    hidden = (din[F_f] > 17).all(axis=1)
    F_f = F_f[~hidden]
    N_f = vertex_normals(V_f, F_f)
    C_f = terrain_masks(V_f, N_f, F['M'], None)
    C_f[:, 3] *= ss(-1.0, -6.0, din)       # covered by terrain_near: keep trees off the hidden far surface

    V_r, UV_r, C_r, F_r = build_road()
    V_t, UV_t, C_t, F_t = build_tunnel()
    V_l, UV_l, C_l, F_l = build_lamps()
    V_l1 = F_l1 = C_l1 = S_l1 = None

    # ---- Blender: decimate + AO
    if bpy is not None and not (ARGS.no_decimate and ARGS.no_ao):
        import bpy as B
        for ob in list(B.data.objects):
            B.data.objects.remove(ob)
        # near: protect road zone (|d|<6), the seam band and the portal/trench rows
        wn = np.ones(V_n.shape[0])
        wn[np.abs(feat['d']) < 6.5] = 0
        wn[seam_weight(feat['s'], feat['d']) > 0] = 0
        wn[np.abs(feat['s'] - GAP) < 4] = 0
        wn[np.abs(feat['s'] - TUN) < 3] = 0
        wn[(feat['kind'] == 1) & (feat['face_u'] > 0.2)] *= 0.5
        # chunk-boundary rows are kept intact in every LOD, so neighbouring chunks at different LODs share
        # exactly the same vertices along their seam (no cracks)
        wn[chunk_boundary(feat['s'])] = 0
        ob_n = bl_mesh('terrain_near', V_n, F_n, C_n, wn, {'s': feat['s'], 'd': feat['d'], 'kind': feat['kind']})
        wf = np.ones(V_f.shape[0])
        wf[(F['din'] > -14) & (F['din'] < 14)] = 0
        wf[(F['M']['scar'] > 0.02) | (F['M']['lobe'] > 0.02)] = 0
        ob_f = bl_mesh('terrain_far', V_f, F_f, C_f, wf)
        if not ARGS.no_decimate:
            n0 = len(ob_n.data.polygons)
            bl_decimate(ob_n, ARGS.ratio_near)
            log('near decimated', n0, '->', len(ob_n.data.polygons))
            n0 = len(ob_f.data.polygons)
            bl_decimate(ob_f, ARGS.ratio_far)
            log('far decimated', n0, '->', len(ob_f.data.polygons))
        ob_r = bl_mesh('road', V_r, F_r, C_r)
        ob_t = bl_mesh('tunnel', V_t, F_t, C_t)
        V_n, F_n, an = bl_read(ob_n, ('s', 'd', 'kind'))
        C_n = an['Col']
        feat = dict(s=an['s'], d=an['d'], kind=np.round(an['kind']))
        V_f, F_f, af = bl_read(ob_f)
        C_f = af['Col']
        if not ARGS.no_ao:
            bl_setup_cycles(ARGS.ao_samples)
            # bake near without the far mesh (avoid grazing hits along the seam), far with everything
            ob_f.hide_render = True
            ao_n, ao_r, ao_t = bl_bake_ao([ob_n, ob_r, ob_t])
            # short-range cavity AO (3 m): crevices in the cut, the ditch, hollows between slide hummocks
            B.context.scene.world.light_settings.distance = 3.0
            (ao_c,) = bl_bake_ao([ob_n])
            B.context.scene.world.light_settings.distance = 30.0
            ao_n = np.clip(ao_n, 0, 1) * (0.4 + 0.6 * np.clip(ao_c, 0, 1))
            ob_f.hide_render = False
            (ao_f,) = bl_bake_ao([ob_f])
            C_n[:, 0] = np.clip(ao_n, 0, 1)
            C_f[:, 0] = np.clip(ao_f, 0, 1)
            C_r[:, 0] = np.clip(0.35 + 0.65 * ao_r, 0, 1) * C_r[:, 0]
            C_t[:, 0] = np.clip(ao_t, 0, 1)
        # ---- terrain_near LOD1: decimate a copy of the AO-baked LOD0 (colours + s/d ride along)
        me = ob_n.data
        ca = me.color_attributes.get('Col')
        ca.data.foreach_set('color', C_n.astype(np.float32).ravel())
        ob_l = ob_n.copy()
        ob_l.data = me.copy()
        ob_l.name = 'terrain_near_lod1'
        B.context.scene.collection.objects.link(ob_l)
        # LOD1 is only seen from > ~120 m: the road zone may simplify (the road strip covers it) but the
        # chunk seams, the far seam band, the trench and the portal stay protected
        vg = ob_l.vertex_groups.get('dec')
        free = np.where((np.abs(feat['d']) < 6.5) & ~chunk_boundary(feat['s']) & (np.abs(feat['s'] - GAP) > 4)
                        & (np.abs(feat['s'] - TUN) > 3) & (seam_weight(feat['s'], feat['d']) <= 0))[0]
        if vg is not None and free.size:
            vg.add(free.tolist(), 1.0, 'REPLACE')
        n0 = len(ob_l.data.polygons)
        bl_decimate(ob_l, ARGS.ratio_lod1)
        log('near LOD1', n0, '->', len(ob_l.data.polygons))
        V_l1, F_l1, al1 = bl_read(ob_l, ('s',))
        C_l1 = al1['Col']
        S_l1 = al1['s']
        # seam check: LOD0 and LOD1 must hold the same vertex set on every chunk-boundary row
        def bset(V, S):
            m = chunk_boundary(S)
            return set(map(tuple, np.round(V[m], 3).tolist()))
        b0, b1 = bset(V_n, feat['s']), bset(V_l1, S_l1)
        log('LOD seam check: boundary verts LOD0', len(b0), 'LOD1', len(b1), 'missing in LOD1', len(b0 - b1))
        N_n = vertex_normals(V_n, F_n)
        N_f = vertex_normals(V_f, F_f)
        global NEAR_BVH
        from mathutils.bvhtree import BVHTree
        Vb = to_blender(V_n)
        NEAR_BVH = BVHTree.FromPolygons([tuple(v) for v in Vb.tolist()], F_n.tolist(), all_triangles=True)
        B.ops.wm.save_as_mainfile(filepath=os.path.join(SCR, 'terrain_debug.blend'), compress=True)

    # ---- scatter (needs final near)
    trees, rocks, spawns, deadwood = build_scatter(F, info)
    # guardrail runs at d=-3.7: none at the fallen-tree crown (crushed), the pull-off (PROPS edges it) and the washout
    ft = float(MK['fallenTree'])
    guard = [[0, ft - 7], [ft + 7, PULL[0] - 2], [PULL[1] + 2, GAP - 12], [GAP + 12, 1105]]
    scatter = {
        'version': 2,
        'note': 'three.js world coords. trees [x,y,z,scale,rotY,variant]; rocks [x,y,z,scale,rotY,rotX,variant]; '
                'slideSpawns: gullies 0..3 in marker order, then scar; guardrails: [s0,s1] runs at d=-3.7; '
                'grassMask: vegetation density grid in road (s,d) space; deadwood [x,y,z,length,rotY,pitch,radius] '
                '(snapped trunks, drawn by terrain.js).',
        'trees': np.round(trees, 2).tolist(),
        'rocks': np.round(rocks, 3).tolist(),
        'slideSpawns': spawns,
        'deadwood': np.round(deadwood, 3).tolist(),
        'guardrails': guard,
        'grassMask': grass_grid(V_n, C_n, feat),
        'corridor': {'s0': S0, 's1': S1, 'd0': DV, 'd1': DU},
        'tunnel': {'s0': TUN, 's1': TUN_END, 'lampEnd': LAMP_END, 'profileHalfWidth': 4.9, 'crown': 7.2},
    }
    for t in scatter['trees']:
        t[5] = int(t[5])
    for r in scatter['rocks']:
        r[6] = int(r[6])
    with open(OUT_SCATTER, 'w') as f:
        json.dump(scatter, f, separators=(',', ':'))
    log('scatter.json', os.path.getsize(OUT_SCATTER) // 1024, 'KB')

    # ---- GLB
    W = GLBWriter()
    m_ter = W.add_material('terrain', (0.35, 0.33, 0.3, 1), 0.95)
    m_road = W.add_material('road', (0.12, 0.12, 0.12, 1), 0.7)
    m_tun = W.add_material('tunnel_concrete', (0.5, 0.5, 0.48, 1), 0.85)
    m_lamp = W.add_material('tunnel_lamp', (1, 0.6, 0.25, 1), 0.4, emissive=(1.0, 0.55, 0.18))

    roots = []
    # near chunks by s (ARGS.chunk m); LOD1 chunks share the key (terrain_near_lod1_XX <-> terrain_near_XX)
    ctr_s = feat['s'][F_n].mean(axis=1)
    key = np.floor((ctr_s - S0) / ARGS.chunk).astype(int)
    kids = []
    for k, Vc, Fc, at in split_by(key, V_n, F_n, {'N': N_n, 'C': C_n}):
        mi = W.add_mesh('terrain_near_%02d' % k, Vc, Fc, at['N'], at['C'], None, m_ter)
        kids.append(W.add_node('terrain_near_%02d' % k, mi))
    roots.append(W.add_node('terrain_near', None, kids))
    if V_l1 is not None:
        N_l1 = vertex_normals(V_l1, F_l1)
        key = np.floor((S_l1[F_l1].mean(axis=1) - S0) / ARGS.chunk).astype(int)
        kids = []
        for k, Vc, Fc, at in split_by(key, V_l1, F_l1, {'N': N_l1, 'C': C_l1}):
            mi = W.add_mesh('terrain_near_lod1_%02d' % k, Vc, Fc, at['N'], at['C'], None, m_ter)
            kids.append(W.add_node('terrain_near_lod1_%02d' % k, mi))
        roots.append(W.add_node('terrain_near_lod1', None, kids))
    # far tiles
    ctr = V_f[F_f].mean(axis=1)
    bx = np.digitize(ctr[:, 0], [-900, 250, 750, 1250, 2100])
    bz = np.digitize(ctr[:, 2], [-700, -150, 350, 1300])
    key = bx * 10 + bz
    kids = []
    for k, Vc, Fc, at in split_by(key, V_f, F_f, {'N': N_f, 'C': C_f}):
        mi = W.add_mesh('terrain_far_%02d' % k, Vc, Fc, at['N'], at['C'], None, m_ter)
        kids.append(W.add_node('terrain_far_%02d' % k, mi))
    roots.append(W.add_node('terrain_far', None, kids))
    # road chunks
    N_r = vertex_normals(V_r, F_r)
    sct = UV_r[F_r][:, :, 1].mean(axis=1)
    key = np.floor((sct - ROAD_S0) / 150).astype(int)
    kids = []
    for k, Vc, Fc, at in split_by(key, V_r, F_r, {'N': N_r, 'C': C_r, 'UV': UV_r}):
        mi = W.add_mesh('road_%02d' % k, Vc, Fc, at['N'], at['C'], at['UV'], m_road)
        kids.append(W.add_node('road_%02d' % k, mi))
    roots.append(W.add_node('road', None, kids))
    N_t = vertex_normals(V_t, F_t)
    roots.append(W.add_node('tunnel', W.add_mesh('tunnel', V_t, F_t, N_t, C_t, UV_t, m_tun)))
    N_l = vertex_normals(V_l, F_l)
    roots.append(W.add_node('tunnel_lamps', W.add_mesh('tunnel_lamps', V_l, F_l, N_l, None, None, m_lamp)))
    raw = os.path.join(SCR, 'terrain_raw.glb')
    W.write(raw, roots)
    log('raw glb', os.path.getsize(raw) // 1024, 'KB', 'tris near', F_n.shape[0], 'far', F_f.shape[0], 'road', F_r.shape[0], 'tunnel', F_t.shape[0])
    if ARGS.no_compress:
        import shutil
        shutil.copy(raw, OUT_GLB)
    else:
        cmd = ['npx', 'gltf-transform', 'meshopt', raw, OUT_GLB, '--level', 'high', '--quantize-position', '16', '--quantize-normal', '10']
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            log('meshopt failed, writing raw', r.stderr[-800:])
            import shutil
            shutil.copy(raw, OUT_GLB)
    log('terrain.glb', os.path.getsize(OUT_GLB) // 1024, 'KB')
    np.savez_compressed(os.path.join(SCR, 'info.npz'), **{k: v for k, v in info.items() if isinstance(v, np.ndarray)})


if __name__ == '__main__':
    main()
    log('done')
