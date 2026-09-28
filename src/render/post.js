// LANDSLIDE post-processing (RENDER workstream).
//
// Chain (HalfFloat linear HDR until tone mapping; renderer.toneMapping stays NoToneMapping; the last pass encodes sRGB):
//   RenderPass -> N8AO (config.quality.ao; half-res on the 'low' AO setting) -> [DOF pass, off unless setDOF(dist)]
//   -> mist (volumetric) -> SSR history (wet-road reflections, read next frame) -> exposure meter -> [DOF] -> [motion blur]
//   -> EffectPass A: Lens (barrel distortion, lateral CA, ISP sharpening, veiling glare; HDR) + Bloom (mipmap halation)
//      + ToneMapping AgX + Grade (phone tone curve, auto white balance incl. the tunnel's sodium light, split tone)
//   -> EffectPass B: SMAA (on display-referred colours) + Vignette + Final (exposure-matched sensor noise, flash, fade)
//
// API: render(dt), setDOF(focusDist | null, range?), setExposure(v), getExposure(), flash(color, seconds),
//      fadeBlack(0..1), impactBlur(amount 0..1), setSize(), dispose().  Also: post.composer / post.passes / post.effects.
// Exposure: renderer.toneMappingExposure (three uploads it to every program, including the AgX pass).
//
// SCENE DEPTH FOR SOFT PARTICLES (post.depthTexture)
// ---------------------------------------------------------------------------------------------------------------
//   post.depthTexture      THREE.DepthTexture (FloatType) with the OPAQUE scene depth of the CURRENT frame. It is
//                          copied (blitFramebuffer) inside the main scene render, after all opaque objects and before
//                          the first transparent object, so transparent materials can sample it without a feedback
//                          loop and without a frame of lag. Values are hardware depth 0..1 (perspective, non-linear).
//   post.cameraNear / post.cameraFar   near/far of ctx.camera (numbers; read them every frame or share the uniforms below)
//   post.depthUniforms     { tSceneDepth, sceneDepthRes (Vector2, drawing-buffer px), sceneCamNF (Vector2 near, far) }:
//                          shared uniform objects, kept up to date on resize; merge them into a ShaderMaterial.
//   The copy is lazy: it only runs once someone has read post.depthTexture or post.depthUniforms (~0.1 ms at 720p).
//   It is null with ?nopost or before post.init(); materials must handle that (e.g. #define SOFT_OFF / fade = 1).
//   Only transparent materials may sample it (opaques render before the copy). Usage in a fragment shader:
//     #include <packing>
//     uniform sampler2D tSceneDepth; uniform vec2 sceneDepthRes; uniform vec2 sceneCamNF;
//     float sceneViewZ = perspectiveDepthToViewZ(texture2D(tSceneDepth, gl_FragCoord.xy / sceneDepthRes).r, sceneCamNF.x, sceneCamNF.y);
//     float fragViewZ  = perspectiveDepthToViewZ(gl_FragCoord.z, sceneCamNF.x, sceneCamNF.y);
//     float soft = clamp((fragViewZ - sceneViewZ) / softness, 0.0, 1.0);     // view z is negative: particle in front > 0
//     gl_FragColor.a *= soft;
import * as THREE from 'three';
import {
  EffectComposer, RenderPass, EffectPass, Effect, BlendFunction, BloomEffect, SMAAEffect, SMAAPreset, EdgeDetectionMode,
  ToneMappingEffect, ToneMappingMode, VignetteEffect, VignetteTechnique, DepthOfFieldEffect, Pass, EffectAttribute,
} from 'postprocessing';
import { N8AOPostPass } from 'n8ao';
import { SSR_UNIFORMS, applyScreenReflections, floatRTSupport } from './materials.js';
import { MIST_DEFAULTS, createMistNoise3D } from './fog.js';

// Photographic grade (display-referred, after AgX). AgX alone is a deliberately flat, desaturated base; a phone or
// dashcam pipeline adds an S-curve, a white balance and colour rendering on top. Real references matched:
//  - white balance: overcast rain light is ~7000-8000 K; cameras on auto WB settle ~5500-6500 K, which leaves the
//    blue-grey cast of rain footage (gradeWB) with slightly cooler shadows than highlights (split tone)
//  - wet conifer forest reads dark blue-green and desaturated, not grass green (hue-selective green desaturation)
//  - tone curve: pivoted contrast in perceptual space: toe pushes wet trees / asphalt down, the shoulder lifts the
//    overcast sky toward near-white (a camera exposing for a dark forest scene nearly clips the sky)
const GRADE_FRAG = /* glsl */`
uniform vec4 gradeA;      // x toe power (>1 = deeper shadows), y saturation, z gain, w black point (perceptual)
uniform vec4 gradeB;      // x shoulder power (>1 = brighter highlights), y pivot, z green desaturation 0..1, w green->teal hue shift
uniform vec3 gradeWB;     // white-balance gains (display linear)
uniform vec3 gradeShadow; // multiplicative tint in shadows (perceptual)
uniform vec3 gradeHigh;   // multiplicative tint in highlights (perceptual)
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  vec3 c = clamp(inputColor.rgb * gradeWB, 0.0, 1.0);
  vec3 p = pow(c, vec3(1.0 / 2.2));                      // perceptual working space
  // pivoted contrast: power toe below the pivot, inverted power shoulder above it
  float pv = gradeB.y;
  vec3 lo = pv * pow(p / pv, vec3(gradeA.x));
  vec3 hi = 1.0 - (1.0 - pv) * pow(max((1.0 - p) / (1.0 - pv), 0.0), vec3(gradeB.x));
  p = mix(lo, hi, step(pv, p));
  float l = dot(p, vec3(0.2126, 0.7152, 0.0722));
  // hue-selective: greens / yellow-greens lose saturation and lean toward blue-green (wet spruce, moss, grass)
  float green = smoothstep(0.0, 0.08, p.g - max(p.r, p.b)) * smoothstep(0.0, 0.05, p.g - p.b * 0.9);
  p.g -= green * gradeB.w * (p.g - p.b) * 0.5;
  p.b += green * gradeB.w * (p.g - p.b) * 0.25;
  float sat = gradeA.y * (1.0 - gradeB.z * green);
  p = mix(vec3(l), p, sat);
  p *= mix(gradeShadow, gradeHigh, smoothstep(0.05, 0.75, l));
  p = max(p - gradeA.w, 0.0) / (1.0 - gradeA.w) * gradeA.z;
  outputColor = vec4(pow(max(p, 0.0), vec3(2.2)), inputColor.a);
}`;

// Lens: slight barrel distortion (phone main camera / dashcam optics, not fully corrected), lateral chromatic
// aberration growing toward the corners, and the impact smear. One CONVOLUTION effect (replaces the stock CA).
// Also the camera ISP's detail sharpening, the strongest tell of phone / dashcam video versus a CG render: an unsharp
// mask of radius ~1 px on luminance. It runs here, in linear HDR, as a multiplicative (log-luminance) boost, which
// behaves like display-referred sharpening in the mid-tones but cannot ring around lamps (bounded ratio) and fades
// in the AgX shoulder. The corners are sharpened less (real lenses are softer off-axis).
const LENS_FRAG = /* glsl */`
uniform vec4 lensA;   // x barrel k, y zoom (keeps the corners inside the frame), z lateral CA (uv at the corner), w sharpen amount
uniform vec3 lensVeil; // veiling glare (linear scene units): stray light from the bright sky spread over the frame
float lensLum(vec3 c) { return max(dot(c, vec3(0.2126, 0.7152, 0.0722)), 1e-5); }
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  vec2 d = uv - 0.5; d.x *= aspect;
  float r2 = dot(d, d);
  vec2 dd = d * (1.0 + lensA.x * r2) * lensA.y;
  vec2 ca = dd * lensA.z * (0.25 + r2);
  dd.x /= aspect; ca.x /= aspect;
  vec2 u = 0.5 + dd;
  vec3 c = vec3(texture2D(inputBuffer, u + ca).r, texture2D(inputBuffer, u).g, texture2D(inputBuffer, u - ca).b);
  if (lensA.w > 0.0) {
    vec2 o = texelSize;
    float nb = log2(lensLum(texture2D(inputBuffer, u + vec2(o.x, 0.0)).rgb)) + log2(lensLum(texture2D(inputBuffer, u - vec2(o.x, 0.0)).rgb))
             + log2(lensLum(texture2D(inputBuffer, u + vec2(0.0, o.y)).rgb)) + log2(lensLum(texture2D(inputBuffer, u - vec2(0.0, o.y)).rgb));
    float hp = log2(lensLum(c)) - 0.25 * nb;
    c *= exp2(clamp(hp * lensA.w * (1.0 - 0.6 * r2), -0.6, 0.45));
  }
  outputColor = vec4(c + lensVeil, inputColor.a);
}`;

// Sensor noise: photon shot noise (variance ~ signal) plus read noise, both scaled by the analogue gain the auto
// exposure applies, so a dim tunnel frame is visibly noisier than an overcast daylight frame; a little chroma noise
// like a small phone sensor; temporal (new pattern every frame). Also flash and fade.
const FINAL_FRAG = /* glsl */`
uniform vec4 finalA;    // x noise amount, y grain size (px), z fade to black 0..1, w gain (exposure multiplier)
uniform vec4 flashCol;  // rgb flash colour (display linear), a amount
float grainHash(vec3 p) {
  p = fract(p * vec3(443.897, 441.423, 437.195));
  p += dot(p, p.yxz + 19.19);
  return fract((p.x + p.y) * p.z);
}
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  vec3 c = inputColor.rgb + flashCol.rgb * flashCol.a;
  vec2 px = floor(uv * resolution / max(finalA.y, 1.0));
  float fr = floor(time * 60.0);
  // roughly gaussian luma noise (sum of 3 uniforms) + weaker chroma noise
  float n = (grainHash(vec3(px, fr)) + grainHash(vec3(px + 17.0, fr + 3.0)) + grainHash(vec3(px - 31.0, fr + 7.0))) / 3.0 - 0.5;
  vec2 ch = vec2(grainHash(vec3(px + 5.0, fr + 11.0)), grainHash(vec3(px - 9.0, fr + 13.0))) - 0.5;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float g = clamp(finalA.w, 0.5, 4.0);
  float sigma = finalA.x * sqrt(g) * (sqrt(max(l, 0.0)) * 0.9 + 0.035 * g);   // shot + read noise (display linear)
  c += sigma * (n * 2.2 + vec3(ch.x, -0.5 * (ch.x + ch.y), ch.y) * 0.8);
  c *= 1.0 - clamp(finalA.z, 0.0, 1.0);
  outputColor = vec4(max(c, 0.0), inputColor.a);
}`;

class GradeEffect extends Effect {
  constructor() {
    super('GradeEffect', GRADE_FRAG, {
      blendFunction: BlendFunction.SRC,
      uniforms: new Map([
        // gradeA.y (saturation) is rewritten every frame by Post.render (auto white balance / tunnel chroma loss)
        ['gradeA', new THREE.Uniform(new THREE.Vector4(1.5, 0.9, 1.0, 0.012))],
        ['gradeB', new THREE.Uniform(new THREE.Vector4(2.5, 0.42, 0.35, 0.25))],
        ['gradeWB', new THREE.Uniform(new THREE.Vector3(0.975, 1.0, 1.035))],
        ['gradeShadow', new THREE.Uniform(new THREE.Vector3(0.96, 1.0, 1.05))],
        ['gradeHigh', new THREE.Uniform(new THREE.Vector3(1.0, 1.0, 1.0))],
      ]),
    });
  }
}
class LensEffect extends Effect {
  constructor(k = 0.035, ca = 0.0016, sharpen = 0.5) {
    super('LensEffect', LENS_FRAG, {
      blendFunction: BlendFunction.SRC,
      attributes: EffectAttribute.CONVOLUTION,
      uniforms: new Map([['lensA', new THREE.Uniform(new THREE.Vector4(k, 1, ca, sharpen))], ['lensVeil', new THREE.Uniform(new THREE.Vector3())]]),
    });
    this.k = k; this.caBase = ca;
  }
  update(renderer, inputBuffer) {
    const a = this.uniforms.get('lensA').value;
    const asp = inputBuffer.width / Math.max(1, inputBuffer.height);
    a.x = this.k; a.y = 1 / (1 + this.k * (0.25 * asp * asp + 0.25));
  }
}
class FinalEffect extends Effect {
  constructor() {
    super('FinalEffect', FINAL_FRAG, {
      blendFunction: BlendFunction.SRC,
      uniforms: new Map([
        ['finalA', new THREE.Uniform(new THREE.Vector4(0.028, 1.0, 0, 1))],
        ['flashCol', new THREE.Uniform(new THREE.Vector4(1, 1, 1, 0))],
      ]),
    });
  }
}

// Camera motion blur (ultra/high, only while the camera moves fast): per-pixel reprojection of the depth buffer with
// the previous frame's view-projection, 8 taps along the screen velocity; shutter 0.35 of the frame (~1/170 s at
// 60 fps, a phone/dashcam in overcast daylight), capped at 2 % of the screen: subtle. The car (chase
// or cockpit camera moves with it) is reprojected with its own previous transform, so it stays sharp while the road
// streaks; anything within 2.2 m of the lens (dashboard, hood edge) is never blurred.
const MBLUR_FRAG = /* glsl */`
uniform sampler2D tIn; uniform sampler2D tDepth;
uniform mat4 projInv; uniform mat4 camWorld; uniform mat4 prevVP;
uniform mat4 carInv; uniform mat4 carPrev; uniform vec3 carHalf;
uniform vec4 mbA;     // x shutter fraction, y max blur (uv), z strength 0..1, w unused
varying vec2 vUv;
void main() {
  vec4 base = texture2D(tIn, vUv);
  float z = texture2D(tDepth, vUv).r;
  vec4 vp = projInv * vec4(vUv * 2.0 - 1.0, min(z, 0.99999) * 2.0 - 1.0, 1.0);
  vec3 vv = vp.xyz / vp.w;
  float dist = length(vv);
  vec3 wp = (camWorld * vec4(vv, 1.0)).xyz;
  if (z >= 0.9999999) wp = (camWorld * vec4(normalize(vv) * 5000.0, 1.0)).xyz;   // sky: rotation blur only
  vec3 lp = (carInv * vec4(wp, 1.0)).xyz;                                         // car origin: on the ground
  if (all(lessThan(abs(lp - vec3(0.0, carHalf.y, 0.0)), carHalf))) wp = (carPrev * vec4(lp, 1.0)).xyz;   // on the car: its own motion
  vec4 pc = prevVP * vec4(wp, 1.0);
  vec2 prevUv = pc.xy / max(pc.w, 1e-4) * 0.5 + 0.5;
  vec2 vel = (vUv - prevUv) * mbA.x * mbA.z;
  if (dist < 2.2 || pc.w <= 0.0) vel = vec2(0.0);
  float L = length(vel);
  if (L > mbA.y) vel *= mbA.y / L;
  if (L < 0.0008) { gl_FragColor = base; return; }
  vec3 acc = base.rgb; float wsum = 1.0;
  for (int i = 1; i <= 7; i++) {
    float t = float(i) / 7.0 - 0.5;
    vec2 u = vUv + vel * t;
    // do not smear the near car / interior into the background
    float zs = texture2D(tDepth, u).r;
    float w = zs < z - 1e-5 && dist > 2.2 ? 0.25 : 1.0;
    acc += texture2D(tIn, u).rgb * w; wsum += w;
  }
  gl_FragColor = vec4(acc / wsum, base.a);
}`;
class MotionBlurPass extends Pass {
  constructor(ctx) {
    super('MotionBlurPass');
    this.ctx = ctx;
    this.needsSwap = true;
    this.needsDepthTexture = true;
    this.u = {
      tIn: { value: null }, tDepth: { value: null }, projInv: { value: new THREE.Matrix4() }, camWorld: { value: new THREE.Matrix4() },
      prevVP: { value: new THREE.Matrix4() }, carInv: { value: new THREE.Matrix4() }, carPrev: { value: new THREE.Matrix4() },
      carHalf: { value: new THREE.Vector3(0, 0, 0) }, mbA: { value: new THREE.Vector4(0.35, 0.02, 1, 0) },
    };
    this.fullscreenMaterial = new THREE.ShaderMaterial({ name: 'MotionBlur', depthWrite: false, depthTest: false, uniforms: this.u, vertexShader: MIST_VERT, fragmentShader: MBLUR_FRAG });
    this._prevVP = new THREE.Matrix4(); this._curVP = new THREE.Matrix4();
    this._carPrevW = new THREE.Matrix4(); this._hasPrev = false; this.speed = 0;
    this._pp = new THREE.Vector3(); this._pq = new THREE.Quaternion(); this._cq = new THREE.Quaternion(); this._cp = new THREE.Vector3();
  }
  setDepthTexture(t) { this.u.tDepth.value = t; }
  /** Called every frame (even when disabled) so the previous matrices stay current. Returns the blur strength. */
  track(dt) {
    const cam = this.ctx.camera, u = this.u;
    // paused (dt = 0, e.g. pause menu, frozen screenshots): keep the last frame's blur exactly as it was
    if (!(dt > 0) && this._hasPrev) return u.mbA.value.z;
    cam.updateMatrixWorld();
    this._curVP.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    cam.getWorldPosition(this._cp); cam.getWorldQuaternion(this._cq);
    let str = 0;
    if (this._hasPrev && dt > 0) {
      const v = this._cp.distanceTo(this._pp) / dt;
      const w = 2 * Math.acos(Math.min(1, Math.abs(this._cq.dot(this._pq)))) / dt;
      this.speed += (v - this.speed) * Math.min(1, dt * 6);
      str = THREE.MathUtils.smoothstep(this.speed, 6, 14) + THREE.MathUtils.smoothstep(w, 1.5, 4) * 0.6;
      // teleports / camera cuts: no blur
      if (v > 80 || w > 30) str = 0;
    }
    u.prevVP.value.copy(this._hasPrev ? this._prevVP : this._curVP);
    u.projInv.value.copy(cam.projectionMatrixInverse);
    u.camWorld.value.copy(cam.matrixWorld);
    const car = this.ctx.car?.object;
    if (car) {
      car.updateMatrixWorld();
      u.carInv.value.copy(car.matrixWorld).invert();
      u.carPrev.value.copy(this._hasPrev ? this._carPrevW : car.matrixWorld);
      u.carHalf.value.set(1.0, 1.0, 2.15);
      this._carPrevW.copy(car.matrixWorld);
    } else u.carHalf.value.set(0, 0, 0);
    this._prevVP.copy(this._curVP);
    this._pp.copy(this._cp); this._pq.copy(this._cq);
    this._hasPrev = dt > 0 || this._hasPrev;
    u.mbA.value.z = Math.min(1, str);
    return str;
  }
  render(renderer, inputBuffer, outputBuffer) {
    this.u.tIn.value = inputBuffer.texture;
    renderer.setRenderTarget(this.renderToScreen ? null : outputBuffer);
    renderer.render(this.scene, this.camera);
  }
}

// Exposure meter: downsamples the linear HDR scene (after AO, before tone mapping) to a 32x18 log-luminance grid.
// The CPU reads it back asynchronously every few frames (no pipeline stall) and computes a centre-weighted average.
const METER_W = 32, METER_H = 18;
class MeterPass extends Pass {
  constructor() {
    super('MeterPass');
    this.needsSwap = false;
    this.rt = new THREE.WebGLRenderTarget(METER_W, METER_H, { type: THREE.FloatType, depthBuffer: false, stencilBuffer: false });
    this.rt.texture.generateMipmaps = false;
    this.fullscreenMaterial = new THREE.ShaderMaterial({
      name: 'ExposureMeter', depthWrite: false, depthTest: false,
      uniforms: { tIn: { value: null }, texel: { value: new THREE.Vector2(1 / METER_W, 1 / METER_H) } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 1.0, 1.0); }',
      fragmentShader: /* glsl */`
        uniform sampler2D tIn; uniform vec2 texel; varying vec2 vUv;
        void main() {
          float acc = 0.0;
          for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {
            vec2 o = (vec2(float(i), float(j)) - 1.5) * 0.25 * texel;
            vec3 c = texture2D(tIn, vUv + o).rgb;
            acc += log(max(dot(c, vec3(0.2126, 0.7152, 0.0722)), 1e-4));
          }
          gl_FragColor = vec4(acc / 16.0, 0.0, 0.0, 1.0);
        }`,
    });
  }
  render(renderer, inputBuffer) {
    this.fullscreenMaterial.uniforms.tIn.value = inputBuffer.texture;
    renderer.setRenderTarget(this.rt);
    renderer.render(this.scene, this.camera);
  }
  dispose() { super.dispose(); this.rt.dispose(); }
}
// SSR history: half-res copy of the linear HDR scene (after AO and mist, before exposure) with the distance to the
// camera in alpha (1e4 = sky), mipmapped for roughness blur. Written every frame; wet materials read it on the NEXT
// frame through the previous view-projection (materials.js applyScreenReflections), so there is no feedback loop.
class HistoryPass extends Pass {
  constructor(camera) {
    super('SSRHistoryPass');
    this.needsSwap = false;
    this.needsDepthTexture = true;
    this.cam = camera;
    this.rt = new THREE.WebGLRenderTarget(2, 2, {
      type: THREE.HalfFloatType, depthBuffer: false, stencilBuffer: false, generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter,
    });
    this.rt.texture.name = 'post.ssrHistory';
    this.valid = false;
    this.fullscreenMaterial = new THREE.ShaderMaterial({
      name: 'SSRHistory', depthWrite: false, depthTest: false,
      uniforms: { tIn: { value: null }, tDepth: { value: null }, projInv: { value: new THREE.Matrix4() } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 1.0, 1.0); }',
      fragmentShader: /* glsl */`
        uniform sampler2D tIn; uniform sampler2D tDepth; uniform mat4 projInv; varying vec2 vUv;
        float invDist(ivec2 p, vec2 uv) {
          float z = texelFetch(tDepth, p, 0).r;
          if (z >= 0.9999999) return 0.0;                      // sky dome / far plane
          vec4 v = projInv * vec4(uv * 2.0 - 1.0, z * 2.0 - 1.0, 1.0);
          return 1.0 / max(length(v.xyz / v.w), 0.05);
        }
        void main() {
          vec3 c = min(texture2D(tIn, vUv).rgb, vec3(6e4));
          // 2x2 box of inverse distances (the full-res footprint of this half-res texel)
          vec2 fs = vec2(textureSize(tDepth, 0));
          ivec2 p = ivec2(floor(vUv * fs - 0.5));
          vec2 tx = 1.0 / fs;
          float a = invDist(p, (vec2(p) + 0.5) * tx) + invDist(p + ivec2(1, 0), (vec2(p) + vec2(1.5, 0.5)) * tx)
                  + invDist(p + ivec2(0, 1), (vec2(p) + vec2(0.5, 1.5)) * tx) + invDist(p + ivec2(1, 1), (vec2(p) + 1.5) * tx);
          gl_FragColor = vec4(max(c, 0.0), 0.25 * a);
        }`,
    });
  }
  setDepthTexture(t) { this.fullscreenMaterial.uniforms.tDepth.value = t; }
  setSize(w, h) { this.rt.setSize(Math.max(2, Math.round(w / 2)), Math.max(2, Math.round(h / 2))); }
  render(renderer, inputBuffer) {
    const u = this.fullscreenMaterial.uniforms, cam = this.cam;
    if (!u.tDepth.value) return;
    u.tIn.value = inputBuffer.texture;
    u.projInv.value.copy(cam.projectionMatrixInverse);
    renderer.setRenderTarget(this.rt);
    renderer.render(this.scene, this.camera);
    // the view this history was captured from (read by the materials during the next frame)
    SSR_UNIFORMS.ssrPrevVP.value.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    SSR_UNIFORMS.ssrPrevCam.value.setFromMatrixPosition(cam.matrixWorld);
    SSR_UNIFORMS.tSSR.value = this.rt.texture;
    const r = SSR_UNIFORMS.ssrRes.value;
    r.set(this.rt.width, this.rt.height, 1 / this.rt.width, 1 / this.rt.height);
    SSR_UNIFORMS.ssrParams.value.z = this.rt.height / (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov) * 0.5));
    this.valid = true;
  }
  dispose() { super.dispose(); this.rt.dispose(); }
}
// Volumetric mist (see fog.js MIST_DEFAULTS): march at reduced resolution into mistRT (r = inscatter in units of the
// fog colour, g = transmittance, b = distance used), then a depth-aware (bilateral) upsample blends it over the scene
// in place: dst = dst * T + fogColour * S. Runs after AO, before the SSR history (so puddles reflect the mist).
const MIST_VERT = 'varying vec2 vUv; void main(){ vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 1.0, 1.0); }';
const MIST_MARCH = /* glsl */`
precision highp sampler3D;
uniform sampler2D tDepth; uniform sampler3D tNoise; uniform sampler2D tBlue;
uniform mat4 projInv; uniform mat4 camWorld; uniform vec3 camPos;
uniform vec4 mPlane;    // x road-plane y at the camera, y slope (dy/dm along zw), zw unit xz direction of the road
uniform vec4 mGround;   // x density, y height, z near start, w far end
uniform vec4 mGround2;  // x coverage, y noise freq multiplier, z strength (tunnel fade), w camera lateral offset (m)
uniform vec4 mValley;   // x base y, y top y, z density, w coverage
uniform vec4 mBand;     // x base y, y top y, z density, w coverage
uniform vec4 mBand2;    // x band hug distance (m), y band noise freq multiplier, z valley noise freq multiplier
uniform vec4 mNoise;    // x base freq (1/m), y time (s), zw wind (m/s)
uniform vec4 mLight;    // x top brightness, y base brightness, z sun glow, w flash
uniform vec3 mSun;
varying vec2 vUv;
float mist_n(vec3 p, float fm) {
  vec3 q = (p - vec3(mNoise.z, 0.0, mNoise.w) * mNoise.y) * (mNoise.x * fm);
  return texture(tNoise, q).r * 0.72 + texture(tNoise, q * 3.13 + vec3(0.37, 0.61, 0.13)).r * 0.28;
}
float mist_cov(float n, float cov) { float t0 = 0.5 + (0.5 - cov) * 0.3; return smoothstep(t0 - 0.03, t0 + 0.09, n); }
// integrate one horizontal slab [yb, yt] over the ray segment [t0, t1]; accumulates S (inscatter) and T
// hug > 0: the cloud only lives in the last 'hug' metres in front of the surface the ray hits (orographic cloud that
// clings to the slopes and wraps the tree tops); sky rays see none of it. fm: noise frequency multiplier.
void mist_slab(vec3 ro, vec3 rd, float t0, float t1, float yb, float yt, float dens, float cov, float hug, float fm, float jit, inout float S, inout float T) {
  if (dens <= 0.0 || T < 0.02) return;
  float ta = t0, tb = t1;
  if (hug > 0.0) { if (t1 > 5e4) return; ta = max(ta, t1 - hug); }
  if (abs(rd.y) > 1e-5) {
    float a = (yb - ro.y) / rd.y, b = (yt - ro.y) / rd.y;
    ta = max(ta, min(a, b)); tb = min(tb, max(a, b));
  } else if (ro.y < yb || ro.y > yt) return;
  tb = min(tb, ta + 4500.0);
  if (tb <= ta) return;
  const int NS = 5;
  float dt = (tb - ta) / float(NS);
  for (int i = 0; i < NS; i++) {
    float t = ta + (float(i) + jit) * dt;
    vec3 p = ro + rd * t;
    float u = clamp((p.y - yb) / (yt - yb), 0.0, 1.0);
    float n = mist_n(p, fm);
    // denser cloud stands taller (billowing tops), thin cloud only near the base
    float c = mist_cov(n + 0.22 * (0.5 - u), cov) * smoothstep(0.0, 0.18, u) * (1.0 - smoothstep(0.72, 1.0, u));
    if (hug > 0.0) c *= smoothstep(t1 - hug, t1 - hug * 0.4, t);
    float sig = dens * c * (1.0 - smoothstep(2500.0, 4500.0, t));
    if (sig <= 0.0) continue;
    float a = 1.0 - exp(-sig * dt);
    S += T * a * mix(mLight.y, mLight.x, u);
    T *= 1.0 - a;
  }
}
void main() {
  float z = texture(tDepth, vUv).r;
  vec4 vp = projInv * vec4(vUv * 2.0 - 1.0, (z >= 0.9999999 ? 0.999 : z) * 2.0 - 1.0, 1.0);
  vec3 vv = vp.xyz / vp.w;
  float D = z >= 0.9999999 ? 1e5 : length(vv);
  vec3 rd = normalize((camWorld * vec4(vv, 0.0)).xyz);
  vec3 ro = camPos;
  // (DITHER fix) per-texel march offset from a tiled void-and-cluster blue-noise mask. The former interleaved-gradient
  // noise is a rank-1 lattice: under the upsample it printed a regular diamond / screen-door grid over every misty
  // tree line (docs screenshot 01). Blue noise has no low-frequency or periodic energy, so the upsample filter removes
  // it completely; it is static (temporally stable, no crawl while the camera is still).
  float jit = texelFetch(tBlue, ivec2(gl_FragCoord.xy) & 63, 0).r;
  float S = 0.0, T = 1.0;
  // ground wisps: height above the road plane h(t) = h0 + k t
  if (mGround.x > 0.0 && mGround2.z > 0.0) {
    float h0 = ro.y - mPlane.x, k = rd.y - mPlane.y * dot(rd.xz, mPlane.zw);
    // it lies on the ground: only the last ~120 m in front of the surface a ray hits (never floating over the valley)
    float ta = max(mGround.z, D - 120.0), tb = min(mGround.w, D);
    if (abs(k) > 1e-5) {
      float a = (0.0 - h0) / k, b = (mGround.y - h0) / k;
      ta = max(ta, min(a, b)); tb = min(tb, max(a, b));
    } else if (h0 < 0.0 || h0 > mGround.y) tb = -1.0;
    if (tb > ta) {
      const int NG = 6;
      for (int i = 0; i < NG; i++) {
        float f0 = (float(i) + jit) / float(NG), f1 = (float(i) + 1.0) / float(NG), fp = float(i) / float(NG);
        float t = ta + (tb - ta) * f0 * f0, dt = (tb - ta) * (f1 * f1 - fp * fp);
        vec3 p = ro + rd * t;
        float h = h0 + k * t;
        float n = mist_n(p, mGround2.y);
        // a body of mist a few metres thick (lifted off the asphalt, fraying out on top), not a skin on the ground;
        // its top undulates (no flat edge across the trees) and it stays within ~25 m of the road
        float htop = mGround.y * (0.15 + 1.7 * smoothstep(0.3, 0.72, texture(tNoise, vec3((p.xz - mNoise.zw * mNoise.y) * mNoise.x * 1.3, 0.37)).r));
        float hn = h / htop;
        float lat = abs(mGround2.w + dot(p.xz - ro.xz, vec2(-mPlane.w, mPlane.z)));
        float sig = mGround.x * mGround2.z * mist_cov(n + 0.12 * (0.5 - hn), mGround2.x) * smoothstep(0.0, 0.2, hn) * (1.0 - smoothstep(0.25, 1.0, hn))
                  * smoothstep(mGround.z, mGround.z * 2.5, t) * (1.0 - smoothstep(mGround.w * 0.55, mGround.w, t)) * (1.0 - smoothstep(12.0, 28.0, lat));
        if (sig <= 0.0) continue;
        float a = 1.0 - exp(-sig * dt);
        S += T * a * mix(mLight.x, mLight.y, clamp(h / mGround.y, 0.0, 1.0) * 0.4);
        T *= 1.0 - a;
      }
    }
  }
  // valley cloud sea and the mid-slope cloud bank, nearest first
  float tv = abs(rd.y) > 1e-5 ? (mix(mValley.x, mValley.y, 0.5) - ro.y) / rd.y : 1e9;
  float tbd = abs(rd.y) > 1e-5 ? (mix(mBand.x, mBand.y, 0.5) - ro.y) / rd.y : 1e9;
  if (tv < 0.0) tv = 1e9; if (tbd < 0.0) tbd = 1e9;
  if (tv <= tbd) {
    mist_slab(ro, rd, 0.0, D, mValley.x, mValley.y, mValley.z, mValley.w, 0.0, mBand2.z, jit, S, T);
    mist_slab(ro, rd, 0.0, D, mBand.x, mBand.y, mBand.z, mBand.w, mBand2.x, mBand2.y, jit, S, T);
  } else {
    mist_slab(ro, rd, 0.0, D, mBand.x, mBand.y, mBand.z, mBand.w, mBand2.x, mBand2.y, jit, S, T);
    mist_slab(ro, rd, 0.0, D, mValley.x, mValley.y, mValley.z, mValley.w, 0.0, mBand2.z, jit, S, T);
  }
  float glow = 1.0 + mLight.z * pow(max(dot(rd, mSun), 0.0), 6.0);
  gl_FragColor = vec4(S * glow * (1.0 + mLight.w * 2.5), T, min(D, 6e4), 1.0);
}`;
const MIST_COMPOSITE = /* glsl */`
uniform sampler2D tMist; uniform sampler2D tDepth; uniform mat4 projInv; uniform vec3 mistCol;
varying vec2 vUv;
void main() {
  float z = texture(tDepth, vUv).r;
  float d0 = 6e4;
  if (z < 0.9999999) { vec4 v = projInv * vec4(vUv * 2.0 - 1.0, z * 2.0 - 1.0, 1.0); d0 = length(v.xyz / v.w); }
  vec2 res = vec2(textureSize(tMist, 0));
  vec2 hp = vUv * res - 0.5;
  ivec2 i0 = ivec2(floor(hp));
  vec2 f = hp - floor(hp);
  vec4 acc = vec4(0.0); float ws = 0.0;
  // (QA) 4x4 tent (radius 2 mist texels) instead of 2x2 bilinear: the march's interleaved-gradient jitter is static
  // and the mist buffer is ~0.42x res, so a bilinear upsample printed the IGN lattice over every misty tree line as a
  // fixed screen-door grain (~2.4 px cells). Mist has no detail at that scale; the depth weight keeps silhouettes.
  for (int j = -1; j < 3; j++) for (int i = -1; i < 3; i++) {
    vec4 m = texelFetch(tMist, clamp(i0 + ivec2(i, j), ivec2(0), ivec2(res) - 1), 0);
    vec2 dd = abs(vec2(float(i), float(j)) - f);
    float wb = max(0.0, 1.0 - dd.x * 0.5) * max(0.0, 1.0 - dd.y * 0.5);
    float w = (wb + 1e-3) / (0.03 + abs(m.b - d0) / max(d0, 1.0));
    acc += m * w; ws += w;
  }
  vec4 m = acc / max(ws, 1e-6);
  gl_FragColor = vec4(mistCol * m.r, clamp(m.g, 0.0, 1.0));
}`;

class MistPass extends Pass {
  constructor(ctx, scale = 0.5) {
    super('MistPass');
    this.ctx = ctx;
    this.needsSwap = false;
    this.needsDepthTexture = true;
    this.scale = scale;
    this.noise = createMistNoise3D(64);
    this.blue = createBlueNoise(64);
    this.rt = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType, depthBuffer: false, stencilBuffer: false });
    this.rt.texture.minFilter = this.rt.texture.magFilter = THREE.NearestFilter;
    this.rt.texture.generateMipmaps = false;
    this.rt.texture.name = 'post.mist';
    const V4 = () => new THREE.Vector4();
    this.u = {
      tDepth: { value: null }, tNoise: { value: this.noise }, tBlue: { value: this.blue },
      projInv: { value: new THREE.Matrix4() }, camWorld: { value: new THREE.Matrix4() }, camPos: { value: new THREE.Vector3() },
      mPlane: { value: V4() }, mGround: { value: V4() }, mGround2: { value: V4() }, mValley: { value: V4() }, mBand: { value: V4() }, mBand2: { value: V4() },
      mNoise: { value: V4() }, mLight: { value: V4() }, mSun: { value: new THREE.Vector3(0, 1, 0) },
    };
    this.march = new THREE.ShaderMaterial({ name: 'MistMarch', depthWrite: false, depthTest: false, uniforms: this.u, vertexShader: MIST_VERT, fragmentShader: MIST_MARCH });
    this.cu = { tMist: { value: this.rt.texture }, tDepth: this.u.tDepth, projInv: this.u.projInv, mistCol: { value: new THREE.Color(0.6, 0.68, 0.8) } };
    this.composite = new THREE.ShaderMaterial({
      name: 'MistComposite', depthWrite: false, depthTest: false, uniforms: this.cu, vertexShader: MIST_VERT, fragmentShader: MIST_COMPOSITE,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.SrcAlphaFactor,
      blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
    });
    this.fullscreenMaterial = this.march;
  }
  setDepthTexture(t) { this.u.tDepth.value = t; }
  setSize(w, h) { this.rt.setSize(Math.max(2, Math.round(w * this.scale)), Math.max(2, Math.round(h * this.scale))); }
  render(renderer, inputBuffer) {
    if (!this.u.tDepth.value) return;
    this.fullscreenMaterial = this.march;
    renderer.setRenderTarget(this.rt);
    renderer.render(this.scene, this.camera);
    this.fullscreenMaterial = this.composite;
    renderer.setRenderTarget(inputBuffer);
    renderer.render(this.scene, this.camera);
    this.fullscreenMaterial = this.march;
  }
  /** Per-frame uniforms from ctx.env.mist (+ the road plane at the camera, tunnel fade). */
  update(tunnel) {
    const { camera, road, env, scene } = this.ctx;
    const M = env?.mist || MIST_DEFAULTS, u = this.u;
    camera.updateMatrixWorld();
    u.projInv.value.copy(camera.projectionMatrixInverse);
    u.camWorld.value.copy(camera.matrixWorld);
    const cp = u.camPos.value.setFromMatrixPosition(camera.matrixWorld);
    if (road) {
      const pr = road.project(cp, this._proj || (this._proj = {}));
      const p = road.pointAt(pr.s), t = road.tangentAt(pr.s);
      const h = Math.hypot(t.x, t.z) || 1;
      const dx = t.x / h, dz = t.z / h;
      // plane through the centreline point, rising along the road; expressed at the camera's xz
      const slope = t.y / h;
      u.mPlane.value.set(p.y + slope * ((cp.x - p.x) * dx + (cp.z - p.z) * dz), slope, dx, dz);
      this._dCam = (cp.x - p.x) * -dz + (cp.z - p.z) * dx;   // lateral offset of the camera from the centreline
    } else u.mPlane.value.set(cp.y - 1.6, 0, 1, 0);
    const on = M.enabled ? 1 : 0, G = M.ground, Va = M.valley, B = M.band, Lt = M.light;
    u.mGround.value.set(G.density * on, G.height, G.nearStart, G.farEnd);
    u.mGround2.value.set(G.coverage, G.freqMul, 1 - tunnel, this._dCam ?? 0);
    u.mValley.value.set(Va.base, Va.top, Va.density * on, Va.coverage);
    u.mBand.value.set(B.base, B.top, B.density * on, B.coverage);
    u.mBand2.value.set(B.hug ?? 250, B.freqMul ?? 0.4, Va.freqMul ?? 0.6, 0);
    u.mNoise.value.set(M.noiseFreq, this.ctx.time?.now ?? 0, M.wind[0], M.wind[1]);
    u.mLight.value.set(Lt.top, Lt.base, Lt.sunGlow, env?.flashLevel ?? 0);
    if (env?.sunDirection) u.mSun.value.copy(env.sunDirection);
    if (scene.fog?.color) this.cu.mistCol.value.copy(scene.fog.color);
  }
  dispose() { super.dispose(); this.rt.dispose(); this.noise.dispose(); this.blue.dispose(); this.march.dispose(); this.composite.dispose(); }
}

/**
 * N x N tiling blue-noise threshold mask (void-and-cluster, Ulichney 1993), deterministic (seeded), R8 texture with
 * values rank / N^2. ~20 ms for 64 x 64 at init. Used as the per-pixel jitter of the ray-marched mist.
 */
function createBlueNoise(N = 64) {
  const n = N * N, SIG = 1.9, R = 7;
  const K = new Float32Array((2 * R + 1) * (2 * R + 1));
  for (let j = -R; j <= R; j++) for (let i = -R; i <= R; i++) K[(j + R) * (2 * R + 1) + i + R] = Math.exp(-(i * i + j * j) / (2 * SIG * SIG));
  const E = new Float32Array(n), bits = new Uint8Array(n);
  const splat = (p, s) => {
    const x = p % N, y = (p / N) | 0;
    for (let j = -R; j <= R; j++) {
      const yy = ((y + j + N) % N) * N;
      for (let i = -R; i <= R; i++) E[yy + ((x + i + N) % N)] += s * K[(j + R) * (2 * R + 1) + i + R];
    }
  };
  const tightest = () => { let b = -1, v = -Infinity; for (let p = 0; p < n; p++) if (bits[p] && E[p] > v) { v = E[p]; b = p; } return b; };
  const loosest = () => { let b = -1, v = Infinity; for (let p = 0; p < n; p++) if (!bits[p] && E[p] < v) { v = E[p]; b = p; } return b; };
  let seed = 0x9e3779b9;
  const rnd = () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  // initial pattern: 10 % random points, relaxed until the tightest cluster is also the largest void
  const ones = Math.round(n * 0.1);
  for (let k = 0; k < ones;) { const p = Math.floor(rnd() * n); if (!bits[p]) { bits[p] = 1; splat(p, 1); k++; } }
  for (let it = 0; it < n; it++) {
    const c = tightest(); bits[c] = 0; splat(c, -1);
    const v = loosest();
    bits[v] = 1; splat(v, 1);
    if (v === c) break;
  }
  const proto = bits.slice(), protoE = E.slice(), rank = new Uint32Array(n);
  // phase 1: rank the prototype's points by repeatedly removing the tightest cluster
  for (let k = ones - 1; k >= 0; k--) { const c = tightest(); bits[c] = 0; splat(c, -1); rank[c] = k; }
  // phases 2 + 3: from the prototype, fill the largest void until every texel has a rank
  bits.set(proto); E.set(protoE);
  for (let k = ones; k < n; k++) { const v = loosest(); bits[v] = 1; splat(v, 1); rank[v] = k; }
  const data = new Uint8Array(n);
  for (let p = 0; p < n; p++) data[p] = Math.min(255, Math.floor((rank[p] + 0.5) / n * 256));
  const t = new THREE.DataTexture(data, N, N, THREE.RedFormat, THREE.UnsignedByteType);
  t.minFilter = t.magFilter = THREE.NearestFilter;
  t.generateMipmaps = false;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.name = 'post.blueNoise';
  t.needsUpdate = true;
  return t;
}
// mist march resolution per quality key (null = off)
const MIST_SCALE = { ultra: 0.5, high: 0.42, medium: 0.33, low: null };

// SSR march steps per quality key (0 = screen "infinity" sample only; null = off, env-map reflections)
const SSR_STEPS = { ultra: 20, high: 10, medium: 0, low: null };

// centre-weighted metering weights (like a camera's centre-weighted average, slightly below centre: the road)
const METER_WT = (() => {
  const w = new Float32Array(METER_W * METER_H);
  for (let y = 0; y < METER_H; y++) for (let x = 0; x < METER_W; x++) {
    const u = (x + 0.5) / METER_W - 0.5, v = (y + 0.5) / METER_H - 0.45;
    w[y * METER_W + x] = 0.25 + Math.exp(-(u * u) / 0.06 - (v * v) / 0.05);
  }
  return w;
})();
// Veiling glare: fraction of the mean frame luminance scattered uniformly over the sensor. ISO 9358 veiling glare
// index of coated phone / dashcam lenses is ~0.5-2 %; 0.5 % keeps the wet forest dark while lifting CG-black shadows.
const VEIL_K = 0.005;
// Log-average scene luminance (linear HDR, before exposure) that maps to multiplier 1.0: a typical daylight road view.
const METER_REF = 0.1;

// config.quality.ao -> [N8AO quality mode, half resolution]. Measured on M1 @1280x720 (full-res 'Medium' costs ~5 ms).
// ultra ('high' AO) is not the M1 budget target, so it runs full resolution.
const AO_MODES = { high: ['Medium', false], medium: ['Low', true], low: ['Performance', true] };

export default class Post {
  constructor(ctx) {
    this.ctx = ctx;
    this.exposure = 0.86;       // base exposure (setExposure)
    this._autoExp = 1.0;        // smoothed tunnel/brightness adaptation multiplier
    this._flash = { amount: 0, decay: 0 };
    this._impact = 0;
    this._fade = 0;
    this.veilK = VEIL_K;        // veiling-glare fraction (runtime tweakable)
    this.effects = {};
    this.passes = {};
    this._depthRT = null;       // opaque-depth copy (see header: post.depthTexture)
    this._depthWanted = false;
    this.depthUniforms = null;
  }

  get depthTexture() {
    if (this.ctx.flags?.nopost || !this.composer) return null;
    this._depthWanted = true; return this._depthRT ? this._depthRT.depthTexture : null;
  }
  get cameraNear() { return this.ctx.camera.near; }
  get cameraFar() { return this.ctx.camera.far; }

  async init() {
    const { renderer, scene, camera, config } = this.ctx;
    const q = config.quality;
    // debug: ?postdbg=noao,nodof,nobloom,noca,nosmaa,nograde,novig,nofinal  (perf A/B; not for gameplay)
    const dbg = this._dbg = new Set((new URLSearchParams(location.search).get('postdbg') || '').split(',').filter(Boolean));
    renderer.toneMapping = THREE.NoToneMapping;          // tone mapping lives in the AgX pass only
    renderer.toneMappingExposure = this.exposure;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // (LOWEND) Robustness: the chain needs a renderable half-float buffer (linear HDR until AgX). Without one (no
    // EXT_color_buffer_float / _half_float) the scene renders straight to the canvas with three's built-in AgX tone
    // mapping (no AO / mist / SSR / bloom / lens / grade); render(), flash() etc. keep working.
    const fl = this.floatRT = floatRTSupport(renderer);
    if (!fl.half) { this._initDirect(); return; }

    const composer = this.composer = new EffectComposer(renderer, { frameBufferType: THREE.HalfFloatType, multisampling: 0 });
    const size = renderer.getSize(new THREE.Vector2());

    this.passes.render = new RenderPass(scene, camera);
    composer.addPass(this.passes.render);

    // ---- ambient occlusion ----------------------------------------------------------------------------------
    // (N8AO's half-resolution mode downsamples depth into an R32F target: it needs full-float colour buffers)
    if (q.ao && q.ao !== 'off' && !dbg.has('noao') && (fl.full || !(AO_MODES[q.ao] || AO_MODES.medium)[1])) {
      try {
        const ao = new N8AOPostPass(scene, camera, size.x, size.y);
        ao.autoDetectTransparency = false;
        ao.configuration.transparencyAware = false;
        ao.configuration.gammaCorrection = false;
        const [mode, half] = AO_MODES[q.ao] || AO_MODES.medium;
        ao.setQualityMode(mode);
        ao.configuration.halfRes = half;
        ao.configuration.depthAwareUpsampling = true;
        ao.configuration.aoRadius = 2.2;          // metres
        ao.configuration.distanceFalloff = 1.2;
        ao.configuration.intensity = 2.0;         // modulated per frame in render() (portal slot / tunnel)
        ao.configuration.color = new THREE.Color(0.012, 0.016, 0.022);  // cold, not pure black
        composer.addPass(ao);
        this.passes.ao = ao;
        this._aoBase = { intensity: 2.0, radius: 2.2 };
      } catch (e) { console.warn('[post] N8AO unavailable', e); }
    }

    // ---- volumetric mist (ground wisps, valley cloud sea, cloud banks on the slopes) ----------------------------
    const mistScale = MIST_SCALE[q.key] !== undefined ? MIST_SCALE[q.key] : 0.5;
    if (mistScale && !dbg.has('nomist')) {
      try {
        this.passes.mist = new MistPass(this.ctx, mistScale);
        composer.addPass(this.passes.mist);
      } catch (e) { console.warn('[post] mist pass unavailable', e); }
    }

    // ---- SSR history (wet road / puddle reflections) -------------------------------------------------------------
    const urlSteps = new URLSearchParams(location.search).get('ssrsteps');   // debug override
    const ssrSteps = urlSteps != null ? +urlSteps : SSR_STEPS[q.key] !== undefined ? SSR_STEPS[q.key] : (q.shadowMapSize >= 4096 ? 12 : 0);
    if (ssrSteps !== null && !dbg.has('nossr')) {
      this.passes.history = new HistoryPass(camera);
      this.passes.history.rt.texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
      composer.addPass(this.passes.history);
      this._ssrOn = true;
      this._applySSR(ssrSteps);
    }

    // ---- exposure meter (reads the linear scene; tone mapping comes later) -----------------------------------------
    // (needs a 32-bit float target for the readback: without one the exposure follows the road-position prior only)
    if (!dbg.has('nometer') && fl.full) {
      this.passes.meter = new MeterPass();
      composer.addPass(this.passes.meter);
      this._meterBuf = new Float32Array(METER_W * METER_H * 4);
      this._meterBusy = false; this._meterFrame = 0;
      this.meter = { avgLum: null, mult: 1 };
    }

    // ---- depth of field (own pass so enabling/disabling never recompiles) -------------------------------------
    if (q.dof && !dbg.has('nodof')) {
      const dof = this.effects.dof = new DepthOfFieldEffect(camera, { focusDistance: 8, focusRange: 5, bokehScale: 2.2, resolutionScale: 0.5 });
      this.passes.dof = new EffectPass(camera, dof);
      this.passes.dof.enabled = false;
      composer.addPass(this.passes.dof);
    }

    // ---- camera motion blur (ultra/high; zero cost while slow: the pass is disabled) ----------------------------
    if ((q.key === 'ultra' || q.key === 'high' || q.motionBlur) && !dbg.has('nomblur')) {
      this.passes.mblur = new MotionBlurPass(this.ctx);
      this.passes.mblur.enabled = false;
      composer.addPass(this.passes.mblur);
    }

    // ---- lens + tone mapping + grade ----------------------------------------------------------------------------
    const ca = this.effects.ca = new LensEffect(dbg.has('nodistort') ? 0 : 0.035, 0.0016, dbg.has('nosharp') ? 0 : (q.key === 'low' ? 0.35 : 0.5));
    this._caBase = 0.0016;
    const toneMapping = this.effects.toneMapping = new ToneMappingEffect({ mode: ToneMappingMode.AGX });
    const grade = this.effects.grade = new GradeEffect();
    const listA = dbg.has('noca') ? [] : [ca];
    if (q.bloom && !dbg.has('nobloom')) {
      const bloom = this.effects.bloom = new BloomEffect({
        // halation / veiling glare: a wide soft tail around lamps, headlights, the blown tunnel portal
        mipmapBlur: true, luminanceThreshold: 0.95, luminanceSmoothing: 0.6, intensity: 0.36, radius: 0.84, levels: 7,
      });
      listA.push(bloom);
      this._bloomTh = 0.95;
    }
    listA.push(toneMapping);
    if (!dbg.has('nograde')) listA.push(grade);
    this.passes.lens = new EffectPass(camera, ...listA);
    composer.addPass(this.passes.lens);

    // ---- display-referred: AA, vignette, grain, flash, fade ----------------------------------------------------
    const smaa = this.effects.smaa = new SMAAEffect({
      preset: q.shadowMapSize >= 4096 ? SMAAPreset.HIGH : SMAAPreset.MEDIUM,
      edgeDetectionMode: EdgeDetectionMode.COLOR,
    });
    smaa.edgeDetectionMaterial.edgeDetectionThreshold = 0.06;
    const vignette = this.effects.vignette = new VignetteEffect({ technique: VignetteTechnique.DEFAULT, offset: 0.32, darkness: 0.42 });
    const fin = this.effects.final = new FinalEffect();
    const listB = [];
    if (!dbg.has('nosmaa')) listB.push(smaa);
    if (!dbg.has('novig')) listB.push(vignette);
    if (!dbg.has('nofinal')) listB.push(fin);
    this.passes.final = new EffectPass(camera, ...listB);
    this.passes.final.dithering = true;
    composer.addPass(this.passes.final);

    this._setupDepthCopy();

    this._offResize = this.ctx.events?.on?.('resize', () => this.setSize());
    // close impacts (rockfall, crashes) give a brief lens smear; the camera shake itself belongs to cameraRig
    this._offImpact = this.ctx.events?.on?.('impact', (e) => {
      if (!e?.position || !(e.energy > 0.2)) return;
      const d = this.ctx.camera.getWorldPosition(_v).distanceTo(e.position);
      const a = Math.min(0.6, (e.energy * 0.12) / (1 + (d * d) / 150));
      if (a > 0.02) this.impactBlur(a);
    });
    this.setSize();
    // count draw calls / triangles of the whole frame (scene + shadow + passes), not just the last full-screen pass
    if (!this.ctx.flags?.nopost) renderer.info.autoReset = false;

    // Warm up the render-target program variants (the scene renders into the composer's linear buffer; main.js's
    // compileAsync only warms the screen variants), so the first frames do not hitch on shader compiles.
    if (!this.ctx.flags?.nopost) {
      try {
        renderer.setRenderTarget(composer.inputBuffer);
        const ready = renderer.compileAsync(scene, camera);   // compile() runs synchronously with this target
        // (LOWEND, load time) The render target is deliberately left on the composer's input buffer: main.js's boot
        // warm-up (renderer.compileAsync right after the last system init) then builds the render-target variants it
        // actually needs, not a second, never-used set of canvas-output (sRGB) variants of every scene program,
        // which doubled the shader compile work at boot. Its warm-up render (post.render) resets the target.
        await ready;
      } catch (e) { console.warn('[post] warm-up compile failed', e); }
    }
  }

  /** Fallback without float render targets: direct canvas render with three's AgX (see init). */
  _initDirect() {
    const { renderer } = this.ctx;
    this.direct = true;
    renderer.toneMapping = THREE.AgXToneMapping;
    renderer.toneMappingExposure = this.exposure;
    const sky = this.ctx.env?.sky?.material;
    if (sky) { sky.toneMapped = true; sky.needsUpdate = true; }
    console.warn('[post] no float render targets: direct rendering, post effects off');
  }

  setSize() {
    if (!this.composer) return;
    this.composer.setSize();
    this._resizeDepthCopy();
  }

  // ---- opaque scene depth copy (soft particles) --------------------------------------------------------------------
  _setupDepthCopy() {
    const { scene, camera } = this.ctx;
    const dt = new THREE.DepthTexture(1, 1, THREE.FloatType);
    dt.minFilter = dt.magFilter = THREE.NearestFilter;
    dt.name = 'post.sceneDepth';
    this._depthRT = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: true, stencilBuffer: false, depthTexture: dt, type: THREE.UnsignedByteType });
    this._depthRT.texture.name = 'post.sceneDepthColor';
    const uni = this._depthUni = {
      tSceneDepth: { value: dt },
      sceneDepthRes: { value: new THREE.Vector2(1, 1) },
      sceneCamNF: { value: new THREE.Vector2(camera.near, camera.far) },
    };
    Object.defineProperty(this, 'depthUniforms', { configurable: true, get: () => {
      if (this.ctx.flags?.nopost || !this.composer) return null;
      this._depthWanted = true; return uni;
    } });
    this._resizeDepthCopy();
    // Hook: an invisible transparent object sorted before every other transparent one. three calls onBeforeRender
    // right after the opaque list, so the depth buffer is complete; we blit it into our own target.
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0, 0, 0, 0], 3));
    g.setDrawRange(0, 0);
    const m = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, depthTest: false, colorWrite: false, fog: false });
    const hook = this._depthHook = new THREE.Mesh(g, m);
    hook.name = 'post_depth_copy_hook';
    hook.frustumCulled = false; hook.castShadow = hook.receiveShadow = false;
    hook.renderOrder = -1e9;
    hook.userData.shadowCull = false;
    hook.onBeforeRender = (renderer, _s, cam) => this._copyDepth(renderer, cam);
    scene.add(hook);
  }

  _resizeDepthCopy() {
    const rt = this._depthRT, ib = this.composer?.inputBuffer;
    if (!rt || !ib) return;
    if (rt.width !== ib.width || rt.height !== ib.height) rt.setSize(ib.width, ib.height);
    this._depthUni.sceneDepthRes.value.set(ib.width, ib.height);
  }

  _copyDepth(renderer, cam) {
    if (!this._depthWanted || !this.composer) return;
    const src = this.composer.inputBuffer;
    if (cam !== this.ctx.camera || renderer.getRenderTarget() !== src) return;
    const rt = this._depthRT;
    if (rt.width !== src.width || rt.height !== src.height) this._resizeDepthCopy();
    this._depthUni.sceneCamNF.value.set(cam.near, cam.far);
    const props = renderer.properties;
    let dst = props.get(rt).__webglFramebuffer;
    if (!dst) { renderer.initRenderTarget(rt); dst = props.get(rt).__webglFramebuffer; }
    const srcFbo = props.get(src).__webglFramebuffer;
    if (!srcFbo || !dst) return;
    const gl = renderer.getContext();
    // restore exactly what was bound (three's WebGLState caches bindings; raw save/restore keeps it consistent)
    const prevR = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING), prevD = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, srcFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, dst);
    gl.blitFramebuffer(0, 0, src.width, src.height, 0, 0, rt.width, rt.height, gl.DEPTH_BUFFER_BIT, gl.NEAREST);
    if (!this._blitChecked) {   // a depth-format mismatch fails silently (GL_INVALID_OPERATION): say so once
      this._blitChecked = true;
      const e = gl.getError();
      if (e) console.warn('[post] scene depth copy failed, GL error', e, 'src depth type', src.depthTexture?.type);
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, prevR);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, prevD);
  }

  /** Patch the wet flat surfaces (TERRAIN's road + near-terrain puddles) with screen-space reflections. */
  _applySSR(steps) {
    const t = this.ctx.terrain;
    const mats = new Map();
    const add = (mesh, opts) => { const m = mesh?.material; if (m && m.isMeshStandardMaterial && !mats.has(m)) mats.set(m, opts); };
    // road: its own lobe + the water film (patchy near-mirror reflections of the tree line, car, barriers, lamps)
    (t?.meshes?.roadChunks || []).forEach((m) => add(m, { maxRoughness: 0.55, strength: 1, film: 0.8 }));
    // near-terrain puddles (pull-off gravel, ditch) only on ultra: the terrain shader is the heaviest in the scene and
    // the extra loop costs register pressure on every terrain pixel even where the gate rejects it
    if (this.ctx.config?.quality?.key === 'ultra' && !this._dbg?.has('ssrroad')) (t?.meshes?.nearChunks || []).forEach((m) => add(m, { maxRoughness: 0.22, strength: 1 }));
    for (const [m, o] of mats) {
      if (m.userData.ssr) continue;
      try { applyScreenReflections(m, { steps, ...o }); } catch (e) { console.warn('[post] SSR patch failed', m.name, e); }
    }
    this._ssrMats = [...mats.keys()];
  }

  /** Screen-space reflections on/off at runtime (off = env-map reflections only). */
  setSSR(on) { this._ssrOn = !!on; }

  /** Focus distance in metres, or null to disable (only when config.quality.dof). */
  setDOF(focusDist, range = null) {
    const dof = this.effects.dof, pass = this.passes.dof;
    if (!dof || !pass) return;
    if (focusDist == null || !isFinite(focusDist)) { pass.enabled = false; return; }
    dof.cocMaterial.focusDistance = focusDist;
    dof.cocMaterial.focusRange = range ?? Math.max(1.5, focusDist * 0.6);
    pass.enabled = true;
  }

  setExposure(v) { this.exposure = Math.max(0.01, +v || 1); }
  getExposure() { return this.exposure * this._autoExp; }

  /** Screen flash: colour (THREE.Color | hex | css) fading out over `t` seconds. */
  flash(color = 0xffffff, t = 0.25) {
    const c = new THREE.Color(color);
    const u = this.effects.final?.uniforms.get('flashCol').value;
    if (!u && !this.direct) return;
    u?.set(c.r, c.g, c.b, 1);
    this._flash.amount = 1; this._flash.decay = 1 / Math.max(0.02, t);
  }

  fadeBlack(v) { this._fade = THREE.MathUtils.clamp(+v || 0, 0, 1); }

  /** Brief lens smear on impacts (chromatic fringing + vignette pulse), decays by itself. */
  impactBlur(amount = 0.5) { this._impact = Math.min(1.5, this._impact + amount); }

  /** Camera position along the road: {tunnel 0..1 (inside the tube), slot 0..1 (deep portal cut + tube)}. */
  _zones() {
    const { road, camera } = this.ctx;
    const m = road?.markers, z = this._zone || (this._zone = { tunnel: 0, slot: 0 });
    z.tunnel = z.slot = 0;
    if (!m || m.tunnel == null) return z;
    const p = road.project(camera.getWorldPosition(_v), _proj);
    const end = m.tunnelEnd ?? m.tunnel + 110;
    if (p.dist > 14 || p.dy > 12 || p.s > end + 5) return z;
    z.slot = THREE.MathUtils.smoothstep(p.s, m.tunnel - 70, m.tunnel - 40);
    if (p.s >= m.tunnel + 4 && p.dist < 9 && p.dy < 9) z.tunnel = THREE.MathUtils.smoothstep(p.s, m.tunnel + 4, m.tunnel + 30);
    return z;
  }

  /** Async exposure metering: kick a readback every 4th frame, consume the result when it lands. */
  _meterTick() {
    const mp = this.passes.meter;
    if (!mp || this._meterBusy || (++this._meterFrame & 3)) return;
    const r = this.ctx.renderer;
    if (!r.readRenderTargetPixelsAsync) return;
    this._meterBusy = true;
    r.readRenderTargetPixelsAsync(mp.rt, 0, 0, METER_W, METER_H, this._meterBuf).then(() => {
      const b = this._meterBuf;
      let sw = 0, sl = 0, sa = 0, na = 0;
      for (let i = 0; i < METER_W * METER_H; i++) { const w = METER_WT[i], l = b[i * 4]; if (isFinite(l)) { sw += w; sl += w * l; sa += Math.exp(l); na++; } }
      if (na) this.meter.meanLum = sa / na;   // ~arithmetic mean scene luminance (drives the veiling glare)
      if (sw > 0) {
        const avg = Math.exp(sl / sw);
        const first = this.meter.avgLum == null;
        this.meter.avgLum = avg;
        // partial adaptation (a real camera never fully normalises): ~70 % of the EV difference, limited range
        this.meter.mult = THREE.MathUtils.clamp(Math.pow(METER_REF / Math.max(avg, 1e-4), 0.7), 0.7, 3.2);
        if (first) this._meterSnap = true;
      }
    }).catch(() => {}).finally(() => { this._meterBusy = false; });
  }

  render(dt = 0.016) {
    const { renderer } = this.ctx;
    if (!renderer.info.autoReset) renderer.info.reset();
    if (!this.composer) {
      if (this.direct) {
        const f = this._flash;
        if (f.amount > 0) f.amount = Math.max(0, f.amount - Math.min(0.1, Math.max(0, dt)) * f.decay);
        renderer.toneMappingExposure = this.exposure * (1 - this._fade) * (1 + 2 * f.amount * f.amount);
      }
      renderer.setRenderTarget(null);
      renderer.render(this.ctx.scene, this.ctx.camera);
      return;
    }
    const d = Math.min(0.1, Math.max(0, dt));
    // eye adaptation: blend of a road-position prior (open up inside the tunnel, so the portal blows out like a
    // real camera) and the centre-weighted scene meter (looking back out toward the exit pulls exposure down a bit)
    const z = this._zones();
    // Phone-camera behaviour: approaching, the camera stays exposed for the daylight so the tunnel mouth reads as a
    // dark hole (the meter may only nudge in the portal cut); inside, auto exposure opens up ~2 EV over ~1.5 s and
    // weights the dark interior, so the daylight portal behind blows out to white with glare.
    const prior = 1.0 + 2.3 * z.tunnel + 0.05 * z.slot;
    let target = prior;
    if (this.meter?.avgLum != null) {
      // outdoors the meter may only nudge (+-0.4 EV: a dark cockpit or cliff must not wash the daylight out)
      const hi = 1.3 + 0.25 * z.slot + 2.2 * z.tunnel, lo = 0.8 - 0.1 * z.tunnel;
      const m = THREE.MathUtils.clamp(this.meter.mult, lo, hi);
      const wp = 0.5 + 0.2 * z.tunnel;   // inside, the prior (dark tube) dominates the bright-portal meter
      target = Math.exp(wp * Math.log(prior) + (1 - wp) * Math.log(m));
    }
    target = THREE.MathUtils.clamp(target, 0.7, 3.6);
    if (this.ctx.env?.flashLevel) target /= 1 + this.ctx.env.flashLevel * 0.25;
    if (this._meterSnap) { this._meterSnap = false; this._autoExp = target; }
    const k = 1 - Math.exp(-d * (target > this._autoExp ? 0.9 : 2.5));   // opening up is slow, closing down fast
    this._autoExp += (target - this._autoExp) * k;
    // auto white balance: under the tunnel's high-pressure sodium lamps (~2000 K) a phone's AWB pulls toward neutral
    // over ~1.5 s, but never fully (it clamps the correction range), and low light costs chroma (noise reduction)
    const gr = this.effects.grade;
    if (gr) {
      this._awb = this._awb ?? z.tunnel;
      this._awb += (z.tunnel - this._awb) * (1 - Math.exp(-d * 0.7));
      const a = this._awb, wb = gr.uniforms.get('gradeWB').value;
      wb.set(0.975 + (0.84 - 0.975) * a, 1.0 - 0.04 * a, 1.035 + (1.3 - 1.035) * a);
      gr.uniforms.get('gradeA').value.y = 0.9 - 0.14 * a;
    }
    // AO: calmer in the deep portal cut and the tube (big enclosing walls otherwise go muddy)
    const ao = this.passes.ao;
    if (ao && this._aoBase) {
      const e = Math.max(z.slot, z.tunnel);
      const inten = this._aoBase.intensity * (1 - 0.45 * e), rad = this._aoBase.radius * (1 - 0.35 * e);
      if (Math.abs(inten - (this._aoI ?? -1)) > 0.02) { ao.configuration.intensity = inten; this._aoI = inten; }
      if (Math.abs(rad - (this._aoR ?? -1)) > 0.02) { ao.configuration.aoRadius = rad; this._aoR = rad; }
    }
    if (this.passes.mist?.enabled) this.passes.mist.update(Math.max(z.tunnel, 0.85 * z.slot));
    // SSR: needs a valid history from the previous frame; skip one frame after camera cuts (teleports, mode switches)
    if (this.passes.history) {
      const cam = this.ctx.camera, P = SSR_UNIFORMS.ssrParams.value;
      const pc = SSR_UNIFORMS.ssrPrevCam.value;
      const cut = cam.getWorldPosition(_v).distanceTo(pc) > 6;
      P.x = this._ssrOn && this.passes.history.valid && !cut ? 1 : 0;
      P.y = this.ctx.env?.wetness ?? 0.75;
      this.passes.history.enabled = this._ssrOn;
      if (!this._ssrOn) this.passes.history.valid = false;
    }
    renderer.toneMappingExposure = this.exposure * this._autoExp;
    // bloom threshold follows exposure: it tests the linear scene before exposure, so without this a blown-out
    // tunnel exit (bright on screen, only ~0.4 in scene units) would never glow while lamps outdoors would
    if (this.effects.bloom) {
      const lm = this.effects.bloom.luminanceMaterial;
      // inside the tube the dark-adapted eye/lens glares around the daylight exit: lower knee, stronger bloom
      const th = this._bloomTh * (1 - 0.5 * z.tunnel) * 0.86 / Math.max(0.05, renderer.toneMappingExposure);
      if (lm && Math.abs(lm.threshold - th) > 0.005) lm.threshold = th;
      const bi = 0.36 + 0.9 * z.tunnel;
      if (Math.abs(this.effects.bloom.intensity - bi) > 0.005) this.effects.bloom.intensity = bi;
    }

    const fin = this.effects.final;
    if (fin) {
      const fu = fin.uniforms.get('flashCol').value;
      if (this._flash.amount > 0) { this._flash.amount = Math.max(0, this._flash.amount - d * this._flash.decay); }
      fu.w = this._flash.amount * this._flash.amount;
      fin.uniforms.get('finalA').value.z = this._fade;
    }
    if (this._impact > 0) this._impact = Math.max(0, this._impact - d * 2.2);
    if (this.effects.ca) {
      this.effects.ca.uniforms.get('lensA').value.z = this._caBase * (1 + this._impact * 5);
      // veiling glare: a phone lens scatters ~0.5-2 % of the light in the frame uniformly over the sensor, so deep
      // shadows (cockpit, wet forest, the tunnel walls against the portal) sit on a faint haze instead of CG black
      const ml = this.meter?.meanLum;
      if (ml != null) {
        this._veil = (this._veil ?? ml) + (ml - (this._veil ?? ml)) * Math.min(1, d * 3);
        const v = this._veil * this.veilK;
        this.effects.ca.uniforms.get('lensVeil').value.set(v * 0.97, v, v * 1.04);
      }
    }
    if (fin) fin.uniforms.get('finalA').value.w = renderer.toneMappingExposure / Math.max(0.05, this.exposure);   // sensor gain
    const mb = this.passes.mblur;
    if (mb) { const st = mb.track(this.ctx.paused ? 0 : d); mb.enabled = st > 0.01; }   // paused: frozen blur
    if (this.effects.vignette) this.effects.vignette.darkness = 0.42 + this._impact * 0.25;
    this.composer.render(d);
    this._meterTick();
  }

  dispose() {
    this._offResize?.();
    this._offImpact?.();
    if (this.ctx.renderer) this.ctx.renderer.info.autoReset = true;
    if (this._depthHook) { this.ctx.scene.remove(this._depthHook); this._depthHook.geometry.dispose(); this._depthHook.material.dispose(); }
    this._depthRT?.dispose(); this._depthRT = null;
    this.composer?.dispose();
    this.composer = null;
  }
}

const _v = new THREE.Vector3();
const _proj = {};
