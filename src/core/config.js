// Central configuration: quality presets + gameplay tuning.
// Quality is chosen on the pre-load quality screen (src/ui/gate.js), via ?quality=ultra|high|medium|low, or the
// title-screen menu; stored in localStorage. Default is 'low' so any machine can start. Ultra/High load the original
// full-quality assets; medium/low load the lighter variants from tools/make_variants.mjs (assetTier).

export const PARAMS = new URLSearchParams(location.search);

// ---------------------------------------------------------------------------------------------------------------------
// Device class (MOBILEPERF). Touch-primary = the primary pointer is coarse AND the device has touch points AND touch
// events exist (a touchscreen laptop with a mouse/trackpad is NOT touch-primary). iPadOS Safari reports a desktop Mac
// user agent, so a "Macintosh" with more than one touch point is an iPad. ?touch=1 / ?touch=0 force either mode (tests).
// DEVICE.mobile decides the quality adjustment below; it follows `touch` unless ?mobileq=1|0 overrides it (perf A/B).
// Other systems (touch controls, UI) can import DEVICE instead of rolling their own detection.
export const DEVICE = (() => {
  const nav = globalThis.navigator || {};
  const ua = nav.userAgent || '';
  const tp = nav.maxTouchPoints || 0;
  const mm = (q) => { try { return !!globalThis.matchMedia?.(q)?.matches; } catch { return false; } };
  const ios = /iPhone|iPad|iPod/i.test(ua) || (/Macintosh/i.test(ua) && tp > 1);
  const android = /Android/i.test(ua);
  const coarse = mm('(pointer: coarse)');
  const touchEvents = typeof window !== 'undefined' && 'ontouchstart' in window;
  let touch = (coarse && tp > 0 && touchEvents) || ((ios || android) && tp > 0);
  const flag = (k) => { const v = PARAMS.get(k); return v === null ? null : !(v === '0' || v === 'false' || v === 'off'); };
  const ft = flag('touch');
  if (ft !== null) touch = ft;
  const fm = flag('mobileq');
  const mobile = fm !== null ? fm : touch;
  let short = 0;
  try { short = Math.min(screen.width, screen.height); } catch {}
  return { touch, mobile, ios, android, tablet: touch && short >= 600, coarse };
})();

export const QUALITY_PRESETS = {
  ultra: {
    name: 'Ultra', pixelRatio: 1.0, maxDpr: 2, shadowMapSize: 4096, shadowRadius: 70,
    ao: 'high', bloom: true, motionBlur: true, dof: true, grassDensity: 1.0, grassRadius: 70,
    treeMeshDistance: 110, impostorDistance: 2200, rain: 1.0, maxRocks: 90, anisotropy: 16,
    assetTier: null, adaptiveRes: false, download: 90,
  },
  // High renders at CSS-pixel resolution (maxDpr 1): at DPR 1.5 on the target M1's Retina display the frame is
  // 2.25x the pixels and the game ran at 26-35 fps in a 1280x720 window. Ultra is the Retina-native preset.
  // sunShadowMapSize (optional) overrides the sun's map size only; shadowMapSize stays the tier flag that other
  // systems test (SMAA preset, PMREM size, sky texture). High: 2048 saves ~1.1 ms/frame on the M1 (storing a 4096²
  // depth map every frame is bandwidth-bound); under the overcast sky the shadows are soft anyway (PCF radius 3 ->
  // ~18 cm penumbra at 60 m).
  high: {
    name: 'High', pixelRatio: 1.0, maxDpr: 1.0, shadowMapSize: 4096, sunShadowMapSize: 2048, shadowRadius: 60,
    ao: 'medium', bloom: true, motionBlur: false, dof: true, grassDensity: 0.75, grassRadius: 55,
    treeMeshDistance: 80, impostorDistance: 1800, rain: 0.8, maxRocks: 70, anisotropy: 8,
    assetTier: null, adaptiveRes: false, download: 90,
  },
  // (LOWEND) medium/low: the sun's map is 1024² (soft overcast shadows: ~9-10 cm texels, PCF radius 3 -> ~0.3 m
  // penumbra, like a real shadow under thick cloud); shadowMapSize stays the tier flag other systems test.
  medium: {
    name: 'Medium', pixelRatio: 0.8, maxDpr: 1, shadowMapSize: 2048, sunShadowMapSize: 1024, shadowRadius: 46,
    ao: 'low', bloom: true, motionBlur: false, dof: false, grassDensity: 0.45, grassRadius: 40,
    treeMeshDistance: 55, impostorDistance: 1400, rain: 0.6, maxRocks: 50, anisotropy: 4,
    assetTier: 'mid', adaptiveRes: true, download: 39,
  },
  // (LOWPOLISH) low: mesh trees out to 26 m (treeMeshDistance 52; was 35 -> 17.5 m). The low impostor atlas is half
  // resolution (192 px frames), so a 25 m spruce 17.5 m away was a 2.4x magnified, painterly blob; LOD1 meshes to 26 m
  // cost ~+0.15 ms on the M1. treeShadowDist keeps the shadow-casting LOD1 range where it was (~12 m).
  low: {
    name: 'Low', pixelRatio: 0.65, maxDpr: 1, shadowMapSize: 1024, shadowRadius: 40,
    ao: 'off', bloom: false, motionBlur: false, dof: false, grassDensity: 0.2, grassRadius: 28,
    treeMeshDistance: 52, treeShadowDist: 12, impostorDistance: 1000, rain: 0.4, maxRocks: 30, anisotropy: 2,
    assetTier: 'lo', adaptiveRes: true, download: 26,
  },
};

// Mobile adjustment (MOBILEPERF), applied on top of whichever preset is chosen when DEVICE.mobile is true. The preset
// `key` and `name` never change (dozens of systems branch on them); only these numbers do. Phone/tablet GPUs
// (Adreno 610-660, Mali-G57, Apple A12-A14) have roughly a fifth to a tenth of an M1's fill rate and iOS kills a tab
// at ~1-1.5 GB, so every preset renders fewer pixels, keeps shadow maps and render targets small, draws less grass
// and fewer mesh trees and rain streaks, and always runs adaptive resolution (adaptiveMin: floor of the scale; it steps
// down while the average frame is slower than adaptiveDown seconds and back up when faster than adaptiveUp).
// shadowMapSize is the tier flag other systems test (>= 4096 selects the 8k sky dome, a 512 PMREM and SMAA high); on
// mobile it stays at 2048 at most, which alone saves ~75 MB of GPU memory on High/Ultra. Desktop presets are untouched.
const MOBILE_ADJUST = {
  ultra: {
    maxDpr: 1.5, shadowMapSize: 2048, sunShadowMapSize: 2048, shadowRadius: 60, ao: 'medium', grassDensity: 0.6,
    grassRadius: 50, treeMeshDistance: 80, impostorDistance: 1800, rain: 0.6, maxRocks: 60, anisotropy: 8,
    adaptiveRes: true, adaptiveMin: 0.5,
  },
  high: {
    maxDpr: 1.0, shadowMapSize: 2048, sunShadowMapSize: 2048, shadowRadius: 52, ao: 'low', grassDensity: 0.5,
    grassRadius: 45, treeMeshDistance: 65, impostorDistance: 1500, rain: 0.55, maxRocks: 55, anisotropy: 4,
    adaptiveRes: true, adaptiveMin: 0.5,
  },
  medium: {
    pixelRatio: 0.8, maxDpr: 1, sunShadowMapSize: 1024, shadowRadius: 42, ao: 'off', grassDensity: 0.3, grassRadius: 32,
    treeMeshDistance: 45, impostorDistance: 1200, rain: 0.45, maxRocks: 40, anisotropy: 2,
    adaptiveRes: true, adaptiveMin: 0.55,
  },
  // 0.75 x CSS px (an 844x390 iPhone landscape renders 633x293; the desktop Low is 0.65 x a much larger window)
  low: {
    pixelRatio: 0.75, maxDpr: 1, shadowMapSize: 1024, shadowRadius: 36, ao: 'off', grassDensity: 0.12, grassRadius: 22,
    treeMeshDistance: 40, treeShadowDist: 10, impostorDistance: 900, rain: 0.3, maxRocks: 24, anisotropy: 2,
    adaptiveRes: true, adaptiveMin: 0.6,
  },
};
// Mobile frame-time thresholds for adaptive resolution: aim for a steady >= 30 fps (a 30 fps cap, e.g. iOS Low Power
// Mode, must not drive the resolution down), step back up only with clear headroom.
const MOBILE_ADAPTIVE = { adaptiveDown: 1 / 27, adaptiveUp: 1 / 45 };

/** The presets as this device uses them (desktop: the plain presets; mobile: with MOBILE_ADJUST on top). */
export const EFFECTIVE_PRESETS = Object.fromEntries(Object.entries(QUALITY_PRESETS).map(([k, p]) => [k,
  DEVICE.mobile ? Object.freeze({ ...p, ...MOBILE_ADAPTIVE, ...MOBILE_ADJUST[k], mobile: true }) : p]));

export function storedQuality() {
  try { const q = localStorage.getItem('landslide.quality'); return QUALITY_PRESETS[q] ? q : null; } catch { return null; }
}
function pickQuality() {
  const q = PARAMS.get('quality') || storedQuality();
  return QUALITY_PRESETS[q] ? q : 'low';
}

export const config = {
  qualityKey: pickQuality(),
  get quality() { return { key: this.qualityKey, ...EFFECTIVE_PRESETS[this.qualityKey] }; },
  setQuality(key) {
    if (!QUALITY_PRESETS[key]) return;
    this.qualityKey = key;
    try { localStorage.setItem('landslide.quality', key); } catch {}
  },

  physics: { step: 1 / 60, maxSubSteps: 5, gravity: -9.81 },

  camera: { fov: 70, near: 0.08, far: 6000 },

  // Gameplay tuning (owned by game/sequence.js; other systems read what they need)
  game: {
    fuelCapacity: 45,          // liters (tank)
    fuelStart: 0.35,           // liters at cold open: runs out around markers.stall
    jerrycanLiters: 5,         // given by the jerrycan
    fuelPerMeterBase: 0.0028,  // liters per meter at light throttle (exaggerated for gameplay)
    fuelThrottleFactor: 1.8,   // extra consumption multiplier at full throttle / high rpm
    chopHitsRequired: 3,
    debrisFront: {             // the advancing mud/rock flow behind the player (s coordinate)
      startS: 160,             // where it sits when the car stalls
      speed0: 0.22,            // m/s right after the stall
      accel: 0.0045,           // m/s^2 (speed grows over time while on foot)
      chaseSpeed: 11,          // m/s once the escape drive begins (after refuel & start)
      chaseLag: 55,            // meters behind car it tries to stay during chase (never slower than chaseSpeed)
    },
    walkSpeed: 1.6, runSpeed: 5.2, stamina: 9, // seconds of sprint
  },
};
