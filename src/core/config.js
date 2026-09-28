// Central configuration: quality presets + gameplay tuning.
// Quality is chosen on the pre-load quality screen (src/ui/gate.js), via ?quality=ultra|high|medium|low, or the
// title-screen menu; stored in localStorage. Default is 'low' so any machine can start. Ultra/High load the original
// full-quality assets; medium/low load the lighter variants from tools/make_variants.mjs (assetTier).

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
  low: {
    name: 'Low', pixelRatio: 0.65, maxDpr: 1, shadowMapSize: 1024, shadowRadius: 40,
    ao: 'off', bloom: false, motionBlur: false, dof: false, grassDensity: 0.2, grassRadius: 28,
    treeMeshDistance: 35, impostorDistance: 1000, rain: 0.4, maxRocks: 30, anisotropy: 2,
    assetTier: 'lo', adaptiveRes: true, download: 26,
  },
};

export const PARAMS = new URLSearchParams(location.search);

export function storedQuality() {
  try { const q = localStorage.getItem('landslide.quality'); return QUALITY_PRESETS[q] ? q : null; } catch { return null; }
}
function pickQuality() {
  const q = PARAMS.get('quality') || storedQuality();
  return QUALITY_PRESETS[q] ? q : 'low';
}

export const config = {
  qualityKey: pickQuality(),
  get quality() { return { key: this.qualityKey, ...QUALITY_PRESETS[this.qualityKey] }; },
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
