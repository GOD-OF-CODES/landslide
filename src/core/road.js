import * as THREE from 'three';

// RoadPath wraps public/assets/world/road.json (see DESIGN.md "Road").
// s = arc length along the road (m), d = signed lateral offset (m): d > 0 uphill/left, d < 0 valley/right.
export class RoadPath {
  constructor(json) {
    this.json = json;
    this.step = json.step;
    this.length = json.length;
    this.count = json.count;
    this.halfWidth = json.halfWidth;
    this.markers = json.markers;
    this.p = Float32Array.from(json.points);
    this.t = Float32Array.from(json.tangents);
    this.l = Float32Array.from(json.lefts);
    this._hint = 0;
  }

  _idx(s) {
    const f = THREE.MathUtils.clamp(s / this.step, 0, this.count - 1.0001);
    const i = Math.floor(f);
    return [i, f - i];
  }

  /** Centerline position at s (y = road surface height at the centerline). */
  pointAt(s, out = new THREE.Vector3()) {
    const [i, f] = this._idx(s), a = i * 3, b = a + 3, p = this.p;
    return out.set(p[a] + (p[b] - p[a]) * f, p[a + 1] + (p[b + 1] - p[a + 1]) * f, p[a + 2] + (p[b + 2] - p[a + 2]) * f);
  }
  /** Unit tangent (direction of increasing s; includes grade). */
  tangentAt(s, out = new THREE.Vector3()) {
    const [i, f] = this._idx(s), a = i * 3, b = a + 3, t = this.t;
    return out.set(t[a] + (t[b] - t[a]) * f, t[a + 1] + (t[b + 1] - t[a + 1]) * f, t[a + 2] + (t[b + 2] - t[a + 2]) * f).normalize();
  }
  /** Unit horizontal vector pointing uphill (left when driving toward +s). */
  leftAt(s, out = new THREE.Vector3()) {
    const [i, f] = this._idx(s), a = i * 3, b = a + 3, l = this.l;
    return out.set(l[a] + (l[b] - l[a]) * f, 0, l[a + 2] + (l[b + 2] - l[a + 2]) * f).normalize();
  }
  /** World position at (s, d) on the flat road plane (ignores 2% crown). */
  worldAt(s, d, out = new THREE.Vector3()) {
    const l = this.leftAt(s, _v1);
    return this.pointAt(s, out).addScaledVector(l, d);
  }
  /** Yaw (radians, rotation about +Y) that makes an object whose forward is +Z face along +s. */
  yawAt(s) {
    const t = this.tangentAt(s, _v1);
    return Math.atan2(t.x, t.z);
  }
  /** Quaternion orienting +Z along the tangent, +Y up (grade applied), +X toward uphill/left. */
  frameQuat(s, out = new THREE.Quaternion()) {
    const t = this.tangentAt(s, _v1);
    const l = this.leftAt(s, _v2);
    const up = _v3.crossVectors(t, l).normalize(); // basis (X=left, Y=up, Z=tangent) is right-handed: Z x X = Y
    _m.makeBasis(l, up, t);
    return out.setFromRotationMatrix(_m);
  }

  /**
   * Project a world position onto the road. Returns {s, d, dy, dist}
   * dy = pos.y - centerline height at s; dist = horizontal distance to centerline.
   * Uses a local search around the previous result (fast for continuous motion), with a global fallback.
   */
  project(pos, out = {}) {
    const p = this.p, n = this.count;
    let best = -1, bestD = Infinity;
    const scan = (from, to, stride) => {
      for (let i = Math.max(0, from); i <= Math.min(n - 1, to); i += stride) {
        const dx = pos.x - p[i * 3], dz = pos.z - p[i * 3 + 2];
        const dd = dx * dx + dz * dz;
        if (dd < bestD) { bestD = dd; best = i; }
      }
    };
    const h = out._hint ?? this._hint;
    const W = 80;
    scan(h - W, h + W, 1);
    // Accept the windowed result only if it is a clear interior minimum close to the road; otherwise do a
    // global coarse scan (a teleport or stale hint can leave a false local minimum tens of meters away).
    const edge = best <= Math.max(0, h - W) + 1 || best >= Math.min(n - 1, h + W) - 1;
    if (edge || bestD > 8 * 8) {
      const localBest = best, localD = bestD;
      bestD = Infinity; best = -1;
      scan(0, n - 1, 4);
      scan(best - 6, best + 6, 1);
      if (localD < bestD) { best = localBest; bestD = localD; }
    }
    // refine on the segment toward the neighbour
    let i0 = best, i1 = Math.min(n - 1, best + 1);
    const ax = p[i0 * 3], az = p[i0 * 3 + 2];
    let sx = p[i1 * 3] - ax, sz = p[i1 * 3 + 2] - az;
    let f = ((pos.x - ax) * sx + (pos.z - az) * sz) / (sx * sx + sz * sz || 1);
    if (f < 0 && best > 0) {
      i0 = best - 1; i1 = best;
      const bx = p[i0 * 3], bz = p[i0 * 3 + 2];
      sx = p[i1 * 3] - bx; sz = p[i1 * 3 + 2] - bz;
      f = ((pos.x - bx) * sx + (pos.z - bz) * sz) / (sx * sx + sz * sz || 1);
    }
    f = THREE.MathUtils.clamp(f, 0, 1);
    const s = (i0 + f) * this.step;
    const c = this.pointAt(s, _v1), l = this.leftAt(s, _v2);
    out.s = s;
    out.d = (pos.x - c.x) * l.x + (pos.z - c.z) * l.z;
    out.dy = pos.y - c.y;
    out.dist = Math.abs(out.d);
    out._hint = best;
    this._hint = best;
    return out;
  }
}

const _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3(), _m = new THREE.Matrix4();
