"""Shared helpers for LANDSLIDE Blender scripts (TERRAIN workstream).

Coordinates
-----------
Everything here works in **three.js world space** (meters, +Y up, road along +X, uphill = -Z)
unless a function says otherwise. The glTF exporter (export_yup=True) maps Blender (x, y, z) to
three (x, z, -y), so:   three (X, Y, Z)  ->  Blender (X, -Z, Y)      (see to_blender / from_blender)

Road
----
Road(...) loads public/assets/world/road.json (frozen, s in [0, 1300], step 0.5 m) and extends it
analytically on both ends with the same generator function as tools/gen_road.mjs, so geometry can
continue smoothly for s < 0 and s > 1300. All sampling functions are numpy-vectorised.

  road.point(s) (N,3)   road.tangent(s)   road.left(s) (horizontal unit, uphill)
  road.world(s, d)      centreline + d * left (flat, no crown)
  road.height(s)        centreline y
  road.project(X, Z)    -> s, d  (windowed nearest search; the road is monotonic in x)

Also: gradient/ridged/Voronoi noise (numpy), vertex-normal computation, and a tiny GLB writer.
"""
import json
import math
import os
import struct

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ROAD_JSON = os.path.join(ROOT, 'public', 'assets', 'world', 'road.json')


# ----------------------------------------------------------------------------------------------
# coordinate helpers
# ----------------------------------------------------------------------------------------------
def to_blender(v):
    """three (X,Y,Z) -> Blender (X,-Z,Y). Works on (...,3) arrays."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def from_blender(v):
    """Blender (x,y,z) -> three (x, z, -y)."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


# ----------------------------------------------------------------------------------------------
# road
# ----------------------------------------------------------------------------------------------
def _center_analytic(t):
    """Same generator as tools/gen_road.mjs (three coords)."""
    t = np.asarray(t, dtype=np.float64)
    x = t
    z = -(26 * np.sin(t / 110) + 12 * np.sin(t / 46 + 1.3) + 3 * np.sin(t / 23 + 0.4))
    y = 40 + 0.045 * t + 2.2 * np.sin(t / 140)
    return np.stack([x, y, z], axis=-1)


class Road:
    def __init__(self, path=ROAD_JSON, s_min=-400.0, s_max=1560.0):
        with open(path) as f:
            j = json.load(f)
        self.json = j
        self.step = float(j['step'])
        self.length = float(j['length'])
        self.markers = j['markers']
        P = np.array(j['points'], dtype=np.float64).reshape(-1, 3)
        n = P.shape[0]
        st = self.step
        # --- extension before s=0: walk the analytic curve backwards from t=0
        tb = np.arange(0.0, -700.0, -0.02)
        cb = _center_analytic(tb)
        lb = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cb, axis=0), axis=1))])
        n_pre = int(round(-s_min / st))
        s_pre = -np.arange(n_pre, 0, -1) * st          # s_min .. -step (ascending)
        pre = np.stack([np.interp(-s_pre, lb, cb[:, k]) for k in range(3)], axis=-1)
        # --- extension after s=length: start at the analytic t matching the last json point
        t_end = P[-1, 0]
        ta = np.arange(t_end, t_end + 700.0, 0.02)
        ca = _center_analytic(ta)
        la = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ca, axis=0), axis=1))])
        n_post = int(round((s_max - self.length) / st))
        s_post = np.arange(1, n_post + 1) * st
        post = np.stack([np.interp(s_post, la, ca[:, k]) for k in range(3)], axis=-1)
        post += P[-1] - ca[0]  # json values are rounded to mm; keep continuity
        self.P = np.concatenate([pre, P, post], axis=0)
        self.s0 = -n_pre * st
        self.n = self.P.shape[0]
        self.i_json0 = n_pre
        self.s = self.s0 + np.arange(self.n) * st
        # tangents (central differences, 3D) and horizontal lefts, as in gen_road.mjs
        a = np.concatenate([self.P[:1], self.P[:-2], self.P[-2:-1]])
        b = np.concatenate([self.P[1:2], self.P[2:], self.P[-1:]])
        T = b - a
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        L = np.stack([T[:, 2], np.zeros(self.n), -T[:, 0]], axis=-1)
        L /= np.linalg.norm(L, axis=1, keepdims=True)
        # use the exact json tangents/lefts where available
        Tj = np.array(j['tangents'], dtype=np.float64).reshape(-1, 3)
        Lj = np.array(j['lefts'], dtype=np.float64).reshape(-1, 3)
        T[n_pre:n_pre + n] = Tj
        L[n_pre:n_pre + n] = Lj
        self.T, self.L = T, L
        assert np.all(np.diff(self.P[:, 0]) > 0), 'road must be monotonic in x'

    # --- sampling -------------------------------------------------------------------------
    def _idx(self, s):
        f = (np.asarray(s, dtype=np.float64) - self.s0) / self.step
        f = np.clip(f, 0, self.n - 1.000001)
        i = np.floor(f).astype(np.int64)
        return i, (f - i)[..., None]

    def point(self, s):
        i, f = self._idx(s)
        return self.P[i] + (self.P[i + 1] - self.P[i]) * f

    def tangent(self, s):
        i, f = self._idx(s)
        t = self.T[i] + (self.T[i + 1] - self.T[i]) * f
        return t / np.linalg.norm(t, axis=-1, keepdims=True)

    def left(self, s):
        i, f = self._idx(s)
        l = self.L[i] + (self.L[i + 1] - self.L[i]) * f
        l[..., 1] = 0
        return l / np.linalg.norm(l, axis=-1, keepdims=True)

    def height(self, s):
        return self.point(s)[..., 1]

    def world(self, s, d):
        s = np.asarray(s, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        s, d = np.broadcast_arrays(s, d)
        return self.point(s) + self.left(s) * d[..., None]

    def s_of_x(self, X):
        """Arc position whose centreline x equals X (road is monotonic in x)."""
        return np.interp(X, self.P[:, 0], self.s)

    def z_of_x(self, X):
        return np.interp(X, self.P[:, 0], self.P[:, 2])

    def y_of_x(self, X):
        return np.interp(X, self.P[:, 0], self.P[:, 1])

    # --- projection -----------------------------------------------------------------------
    def project(self, X, Z, window=70.0, chunk=150000):
        """Nearest-point projection of world (X, Z) onto the centreline. Returns (s, d).
        Valid for |d| up to ~70 m (min curve radius is 77 m)."""
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        shape = X.shape
        X = X.ravel()
        Z = Z.ravel()
        xs, zs = self.P[:, 0], self.P[:, 2]
        n = self.n
        S = np.empty(X.shape[0])
        D = np.empty(X.shape[0])
        w = int(window / self.step)
        offs_c = np.arange(-w, w + 1, 4)
        offs_f = np.arange(-4, 5)
        for a in range(0, X.shape[0], chunk):
            x = X[a:a + chunk]
            z = Z[a:a + chunk]
            m = x.shape[0]
            i0 = np.clip(np.searchsorted(xs, x), 0, n - 1)
            idx = np.clip(i0[:, None] + offs_c[None, :], 0, n - 1)
            dd = (x[:, None] - xs[idx]) ** 2 + (z[:, None] - zs[idx]) ** 2
            best = idx[np.arange(m), np.argmin(dd, axis=1)]
            idx = np.clip(best[:, None] + offs_f[None, :], 0, n - 1)
            dd = (x[:, None] - xs[idx]) ** 2 + (z[:, None] - zs[idx]) ** 2
            best = idx[np.arange(m), np.argmin(dd, axis=1)]
            # segment refinement (towards next, or previous if behind)
            i1 = np.minimum(best + 1, n - 1)
            ax, az = xs[best], zs[best]
            sx, sz = xs[i1] - ax, zs[i1] - az
            f = ((x - ax) * sx + (z - az) * sz) / np.maximum(sx * sx + sz * sz, 1e-12)
            back = (f < 0) & (best > 0)
            ib = np.where(back, best - 1, best)
            ax, az = xs[ib], zs[ib]
            i1 = np.minimum(ib + 1, n - 1)
            sx, sz = xs[i1] - ax, zs[i1] - az
            f = ((x - ax) * sx + (z - az) * sz) / np.maximum(sx * sx + sz * sz, 1e-12)
            f = np.clip(f, 0, 1)
            s = self.s0 + (ib + f) * self.step
            c = self.point(s)
            l = self.left(s)
            S[a:a + chunk] = s
            D[a:a + chunk] = (x - c[:, 0]) * l[:, 0] + (z - c[:, 2]) * l[:, 2]
        return S.reshape(shape), D.reshape(shape)


# ----------------------------------------------------------------------------------------------
# noise (numpy, vectorised)
# ----------------------------------------------------------------------------------------------
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


class Noise:
    """Seeded gradient noise, fBm, ridged multifractal and Voronoi (all numpy)."""

    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256).astype(np.int64)
        self.perm = np.concatenate([p, p])
        a = rng.random(256) * 2 * np.pi
        self.g2 = np.stack([np.cos(a), np.sin(a)], axis=-1)
        v = rng.normal(size=(256, 3))
        self.g3 = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.seed = seed

    def perlin2(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xi = np.floor(x)
        yi = np.floor(y)
        xf = x - xi
        yf = y - yi
        xi = xi.astype(np.int64) & 255
        yi = yi.astype(np.int64) & 255
        p = self.perm
        g = self.g2
        a0 = p[xi]
        a1 = p[xi + 1]
        h00 = p[a0 + yi]
        h01 = p[a0 + yi + 1]
        h10 = p[a1 + yi]
        h11 = p[a1 + yi + 1]
        n00 = g[h00, 0] * xf + g[h00, 1] * yf
        n10 = g[h10, 0] * (xf - 1) + g[h10, 1] * yf
        n01 = g[h01, 0] * xf + g[h01, 1] * (yf - 1)
        n11 = g[h11, 0] * (xf - 1) + g[h11, 1] * (yf - 1)
        u = _fade(xf)
        v = _fade(yf)
        return (lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)) * 1.41

    def perlin3(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
        xf, yf, zf = x - xi, y - yi, z - zi
        xi = xi.astype(np.int64) & 255
        yi = yi.astype(np.int64) & 255
        zi = zi.astype(np.int64) & 255
        p, g = self.perm, self.g3
        res = 0
        u, v, w = _fade(xf), _fade(yf), _fade(zf)
        acc = []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    h = p[p[p[xi + dx] + yi + dy] + zi + dz]
                    acc.append(g[h, 0] * (xf - dx) + g[h, 1] * (yf - dy) + g[h, 2] * (zf - dz))
        x00 = lerp(acc[0], acc[4], u)
        x01 = lerp(acc[1], acc[5], u)
        x10 = lerp(acc[2], acc[6], u)
        x11 = lerp(acc[3], acc[7], u)
        y0 = lerp(x00, x10, v)
        y1 = lerp(x01, x11, v)
        res = lerp(y0, y1, w)
        return res * 1.6

    def fbm2(self, x, y, octaves=5, lac=2.03, gain=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        tot = np.zeros(np.broadcast(x, y).shape)
        amp, norm = 1.0, 0.0
        c, s = math.cos(0.61), math.sin(0.61)
        for o in range(octaves):
            tot += amp * self.perlin2(x + 17.3 * o, y - 9.1 * o)
            norm += amp
            x, y = (c * x - s * y) * lac, (s * x + c * y) * lac
            amp *= gain
        return tot / norm

    def fbm3(self, x, y, z, octaves=4, lac=2.03, gain=0.5):
        tot = 0.0
        amp, norm = 1.0, 0.0
        for o in range(octaves):
            tot = tot + amp * self.perlin3(x + 5.1 * o, y - 3.7 * o, z + 1.3 * o)
            norm += amp
            x, y, z = x * lac, y * lac, z * lac
            amp *= gain
        return tot / norm

    def ridged2(self, x, y, octaves=6, lac=2.05, gain=2.0, offset=1.0, H=1.0):
        """Musgrave ridged multifractal, roughly in [0, 1]."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        c, s = math.cos(0.83), math.sin(0.83)
        sig = offset - np.abs(self.perlin2(x, y))
        sig = sig * sig
        res = sig.copy()
        w = np.ones_like(sig)
        freq = 1.0
        norm = 1.0
        for o in range(1, octaves):
            x, y = (c * x - s * y) * lac + 3.3, (s * x + c * y) * lac - 1.7
            freq *= lac
            w = np.clip(sig * gain, 0, 1)
            sig = offset - np.abs(self.perlin2(x, y))
            sig = sig * sig * w
            a = freq ** (-H)
            res += sig * a
            norm += a
        return res / norm

    @staticmethod
    def hash2(ix, iy, k=0):
        """Integer hash of cell coordinates -> float in [0,1)."""
        h = (np.asarray(ix, dtype=np.int64) * 374761393 + np.asarray(iy, dtype=np.int64) * 668265263 + k * 1442695041) & 0xFFFFFFFF
        h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
        h = h ^ (h >> 16)
        return (h & 0xFFFFFF) / float(0x1000000)

    def voronoi2(self, x, y, jitter=0.9):
        """Returns (F1, F2, cell_id_hash in [0,1), cell_ix, cell_iy)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        bx = np.floor(x).astype(np.int64)
        by = np.floor(y).astype(np.int64)
        F1 = np.full(x.shape, 1e9)
        F2 = np.full(x.shape, 1e9)
        cid = np.zeros(x.shape)
        cx = np.zeros(x.shape, dtype=np.int64)
        cy = np.zeros(x.shape, dtype=np.int64)
        k = self.seed
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                ix = bx + ox
                iy = by + oy
                px = ix + 0.5 + (self.hash2(ix, iy, k) - 0.5) * jitter
                py = iy + 0.5 + (self.hash2(ix, iy, k + 7) - 0.5) * jitter
                d = np.hypot(x - px, y - py)
                closer = d < F1
                F2 = np.where(closer, F1, np.minimum(F2, d))
                F1 = np.where(closer, d, F1)
                cid = np.where(closer, self.hash2(ix, iy, k + 13), cid)
                cx = np.where(closer, ix, cx)
                cy = np.where(closer, iy, cy)
        return F1, F2, cid, cx, cy


# ----------------------------------------------------------------------------------------------
# mesh helpers (numpy)
# ----------------------------------------------------------------------------------------------
def grid_faces(nr, nc, flip=False):
    """Triangles for an nr x nc vertex grid indexed [r*nc + c]. Normal = (row dir) x (col dir)."""
    r = np.arange(nr - 1)[:, None]
    c = np.arange(nc - 1)[None, :]
    a = (r * nc + c).ravel()
    b = ((r + 1) * nc + c).ravel()
    cc = ((r + 1) * nc + c + 1).ravel()
    d = (r * nc + c + 1).ravel()
    if not flip:
        t1 = np.stack([a, b, cc], axis=1)
        t2 = np.stack([a, cc, d], axis=1)
    else:
        t1 = np.stack([a, cc, b], axis=1)
        t2 = np.stack([a, d, cc], axis=1)
    return np.concatenate([t1, t2], axis=0).astype(np.int64)


def vertex_normals(V, F):
    """Area-weighted vertex normals."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    N = np.zeros_like(V)
    for k in range(3):
        for j in range(3):
            N[:, k] += np.bincount(F[:, j], weights=fn[:, k], minlength=V.shape[0])
    l = np.linalg.norm(N, axis=1, keepdims=True)
    N /= np.maximum(l, 1e-12)
    N[l[:, 0] < 1e-12] = (0, 1, 0)
    return N


def compact(V_attrs, F):
    """Remove unused vertices. V_attrs: dict name->array (first axis = vertex). Returns (attrs, F)."""
    n = next(iter(V_attrs.values())).shape[0]
    used = np.zeros(n, dtype=bool)
    used[F.ravel()] = True
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(used.sum())
    return {k: v[used] for k, v in V_attrs.items()}, remap[F]


# ----------------------------------------------------------------------------------------------
# minimal GLB writer
# ----------------------------------------------------------------------------------------------
class GLBWriter:
    """Build a glTF 2.0 binary with meshes of indexed triangles.

    add_mesh(name, positions (N,3) f32, normals (N,3), indices (M,3), colors (N,4) 0..1 -> u8 normalized,
             uvs (N,2) f32, material index) -> mesh index
    add_node(name, mesh=None, children=[]) -> node index ; set_scene_roots([...])
    """

    def __init__(self):
        self.bin = bytearray()
        self.accessors, self.views, self.meshes, self.nodes, self.materials = [], [], [], [], []
        self.roots = []

    def _view(self, data: bytes, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin.extend(data)
        v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target:
            v['target'] = target
        self.views.append(v)
        return len(self.views) - 1

    def _acc(self, arr, ctype, typ, normalized=False, target=34962, minmax=False):
        arr = np.ascontiguousarray(arr)
        view = self._view(arr.tobytes(), target)
        a = {'bufferView': view, 'componentType': ctype, 'count': int(arr.shape[0]), 'type': typ}
        if normalized:
            a['normalized'] = True
        if minmax:
            a['min'] = [float(v) for v in arr.min(axis=0)]
            a['max'] = [float(v) for v in arr.max(axis=0)]
        self.accessors.append(a)
        return len(self.accessors) - 1

    def add_material(self, name, color=(0.5, 0.5, 0.5, 1), rough=0.9, metal=0.0, emissive=None, double=False):
        m = {'name': name, 'pbrMetallicRoughness': {'baseColorFactor': list(color), 'roughnessFactor': rough, 'metallicFactor': metal}}
        if emissive is not None:
            m['emissiveFactor'] = list(emissive)
        if double:
            m['doubleSided'] = True
        self.materials.append(m)
        return len(self.materials) - 1

    def add_mesh(self, name, positions, indices, normals=None, colors=None, uvs=None, material=None):
        attrs = {'POSITION': self._acc(positions.astype(np.float32), 5126, 'VEC3', minmax=True)}
        if normals is not None:
            attrs['NORMAL'] = self._acc(normals.astype(np.float32), 5126, 'VEC3')
        if uvs is not None:
            attrs['TEXCOORD_0'] = self._acc(uvs.astype(np.float32), 5126, 'VEC2')
        if colors is not None:
            c = np.clip(np.round(np.asarray(colors) * 255), 0, 255).astype(np.uint8)
            attrs['COLOR_0'] = self._acc(c, 5121, 'VEC4', normalized=True)
        idx = np.asarray(indices, dtype=np.uint32).ravel()
        ia = self._acc(idx, 5125, 'SCALAR', target=34963)
        prim = {'attributes': attrs, 'indices': ia, 'mode': 4}
        if material is not None:
            prim['material'] = material
        self.meshes.append({'name': name, 'primitives': [prim]})
        return len(self.meshes) - 1

    def add_node(self, name, mesh=None, children=None, translation=None):
        n = {'name': name}
        if mesh is not None:
            n['mesh'] = mesh
        if children:
            n['children'] = list(children)
        if translation is not None:
            n['translation'] = [float(v) for v in translation]
        self.nodes.append(n)
        return len(self.nodes) - 1

    def write(self, path, roots):
        gl = {
            'asset': {'version': '2.0', 'generator': 'landslide terrain.py'},
            'scene': 0,
            'scenes': [{'nodes': list(roots)}],
            'nodes': self.nodes, 'meshes': self.meshes, 'materials': self.materials,
            'accessors': self.accessors, 'bufferViews': self.views,
            'buffers': [{'byteLength': len(self.bin)}],
        }
        js = json.dumps(gl, separators=(',', ':')).encode('utf8')
        while len(js) % 4:
            js += b' '
        while len(self.bin) % 4:
            self.bin.append(0)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, total))
            f.write(struct.pack('<II', len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack('<II', len(self.bin), 0x004E4942))
            f.write(bytes(self.bin))


def self_test():
    r = Road()
    j0 = np.array(r.json['points'][:3])
    assert np.allclose(r.point(0.0), j0, atol=1e-6), (r.point(0.0), j0)
    assert np.allclose(r.point(1300.0), np.array(r.json['points'][-3:]), atol=1e-6)
    ss = np.array([-120.0, 0.0, 40.0, 305.0, 560.0, 900.0, 1150.0, 1290.0])
    dd = np.array([30.0, -20.0, 5.0, -44.0, 59.0, -10.0, 3.0, 0.0])
    W = r.world(ss, dd)
    ps, pd = r.project(W[:, 0], W[:, 2])
    assert np.allclose(ps, ss, atol=0.2), (ps, ss)
    assert np.allclose(pd, dd, atol=0.05), (pd, dd)
    # uphill (d>0) is -Z where the road runs along +x
    w1 = r.world(np.array([700.0]), np.array([10.0]))
    assert w1[0, 2] < r.point(np.array([700.0]))[0, 2]
    # continuity of the extension
    pts = r.point(np.arange(-5, 5, 0.5))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.all(np.abs(seg - 0.5) < 0.01), seg
    pts = r.point(np.arange(1295, 1305, 0.5))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.all(np.abs(seg - 0.5) < 0.01), seg
    print('common.self_test OK; s range', r.s0, r.s[-1])


if __name__ == '__main__':
    self_test()
