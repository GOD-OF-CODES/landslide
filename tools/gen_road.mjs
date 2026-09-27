// Generates the authoritative road centerline: public/assets/world/road.json
// Coordinates are three.js world space: meters, +Y up. Road runs roughly along +X.
// "left" = uphill side (roughly -Z). Signed lateral offset d > 0 means uphill/left, d < 0 valley/right.
import fs from 'node:fs';

const LENGTH = 1300;      // meters of road (s = 0 .. LENGTH)
const STEP = 0.5;         // resample spacing in meters

function center(t) {
  const x = t;
  const z = -(26 * Math.sin(t / 110) + 12 * Math.sin(t / 46 + 1.3) + 3 * Math.sin(t / 23 + 0.4));
  const y = 40 + 0.045 * t + 2.2 * Math.sin(t / 140);
  return [x, y, z];
}

// dense polyline then arc-length resample (arc length measured in the horizontal plane + vertical)
const dense = [];
for (let t = -20; t < LENGTH + 200; t += 0.05) dense.push(center(t));
const cum = [0];
for (let i = 1; i < dense.length; i++) {
  const a = dense[i - 1], b = dense[i];
  cum.push(cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
}
// s = 0 at t = 0
let i0 = dense.findIndex((p) => p[0] >= 0);
const sOff = cum[i0];
const points = [], tangents = [], lefts = [];
let j = i0;
for (let s = 0; s <= LENGTH + 1e-6; s += STEP) {
  const target = s + sOff;
  while (cum[j + 1] < target) j++;
  const f = (target - cum[j]) / (cum[j + 1] - cum[j]);
  const a = dense[j], b = dense[j + 1];
  const p = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  points.push(p);
}
for (let k = 0; k < points.length; k++) {
  const a = points[Math.max(0, k - 1)], b = points[Math.min(points.length - 1, k + 1)];
  let tx = b[0] - a[0], ty = b[1] - a[1], tz = b[2] - a[2];
  const l = Math.hypot(tx, ty, tz); tx /= l; ty /= l; tz /= l;
  tangents.push([tx, ty, tz]);
  // left = up x tangent (horizontal), up = +Y  => left = (tz, 0, -tx) normalized ... check: up(0,1,0) x t(tx,ty,tz) = (1*tz - 0*ty, 0*tx - 0*tz, 0*ty - 1*tx) = (tz, 0, -tx)
  const hl = Math.hypot(tz, tx);
  lefts.push([tz / hl, 0, -tx / hl]);
}
const r = (v) => Math.round(v * 1000) / 1000;
const json = {
  version: 1,
  note: 'three.js world space, meters, +Y up. left = uphill side. d>0 uphill(left), d<0 valley(right).',
  step: STEP,
  length: LENGTH,
  count: points.length,
  halfWidth: 3.0,          // asphalt half width (2 lanes x 3.0 m)
  shoulderValley: 0.9,     // gravel shoulder beyond asphalt on valley side (d from -3.0 to -3.9)
  ditchUphill: 1.3,        // drainage ditch beyond asphalt on uphill side (d from 3.0 to 4.3), then rock cut face
  laneCenter: 1.5,         // lane centers at d = +/-1.5 ; traffic drives on the right => player lane is d = -1.5
  points: points.flat().map(r),
  tangents: tangents.flat().map(r),
  lefts: lefts.flat().map(r),
  markers: {
    carStart: 40,          // car spawns here in lane d=-1.5, facing +s, cold-open drive begins
    scar: [90, 190],       // big landslide scar on uphill slope (origin of the main slide / debris front)
    rockfallIntro: 150,    // boulders crash onto road around s=135..155 just after the car passes s~175
    stall: 235,            // engine dies around here (car coasts to a stop ~235-250)
    fallenTree: 305,       // conifer lying across the whole road (blocks car, player can climb over)
    pulloff: [365, 425],   // valley-side widened gravel pull-off (roadworks site), flat out to d = -14
    roadworks: 395,        // center of roadworks site (items: jerrycan, hatchet, planks)
    gap: 560,              // washed-out trench across entire road: |s-560| < 1.3, depth 1.8 m
    gullies: [650, 770, 890, 1010],  // rock chutes on the uphill slope; boulders roll across the road here
    tunnel: 1150,          // tunnel portal plane (road enters mountain spur); tunnel interior 1150..1260
    tunnelEnd: 1260,
    win: 1175,             // player (in car or on foot) past this s = escaped
  },
};
fs.writeFileSync('public/assets/world/road.json', JSON.stringify(json));
// report curvature stats
let minR = Infinity, minRs = 0;
for (let k = 2; k < points.length - 2; k++) {
  const a = tangents[k - 1], b = tangents[k + 1];
  const dAng = Math.acos(Math.min(1, a[0] * b[0] + a[2] * b[2]) / (Math.hypot(a[0], a[2]) * Math.hypot(b[0], b[2])) );
  const R = (2 * STEP) / Math.max(dAng, 1e-9);
  if (R < minR) { minR = R; minRs = k * STEP; }
}
const P = (s) => points[Math.round(s / STEP)].map((v) => v.toFixed(1)).join(',');
console.log(`samples=${points.length} minRadius=${minR.toFixed(1)}m at s=${minRs}  start=(${P(0)}) tunnel=(${P(1150)}) end=(${P(1300)})`);
