# LANDSLIDE: design contract

This is the **single source of truth** for everyone building this game. Read it fully before writing code or assets.
If you must deviate from a contract, keep the change backward compatible and document it in the "Change log" at the bottom.

## 0. Goal and hard rules

- **Goal:** a short (5–8 min), **hyper-realistic** first-person escape game that runs in the browser (three.js r186, WebGL2).
  - A landslide hits a mountain road. Your old 4x4 runs out of fuel. You find a jerrycan, a hatchet and scaffold planks.
  - You cut through a fallen tree, bridge a washed-out trench, and outrun the collapsing mountain into a tunnel.
- **Rule 1: no downloaded 3D models.**
  - Every mesh (terrain, car, trees, rocks, props, tunnel) is built from scratch, either by Blender Python scripts (`tools/blender/*.py`, run headless) or procedurally in JS.
  - Scanned CC0 **textures** from Poly Haven and the HDRI are allowed; they are in `public/assets/tex/` and `raw_assets/`.
  - You may download more **CC0 textures** from Poly Haven through `https://api.polyhaven.com/files/<id>`. Log any you add in the change log.
- **Rule 2: realism first, then performance.**
  - Target machine: Apple M1 (7-core GPU, 8 GB RAM). Aim for ≥ 45 fps at 1280×720 on `high` and ≥ 60 fps on `medium`.
  - Budget: ≤ 1.5 M triangles/frame and ≤ 400 draw calls.
  - Instance everything repeated. Use real PBR everywhere: albedo, normal, roughness, AO. Avoid flat colors.
- **Rule 3: don't break others.**
  - Only edit files you own (section 3). Shared core files (`src/main.js`, `src/core/*`, `src/physics/world.js`) are owned by the lead. If you need a change there, describe it in your final report instead of editing.
  - Each system must degrade gracefully when another system is missing (null-check `ctx.car`, `ctx.audio`, etc.).
- **Rule 4: verify visually.**
  - Blender assets: render preview PNGs and look at them (Read tool).
  - Engine code: use `node tools/shot.mjs "/?autostart&..." out.png`, which drives headless Chrome with GPU. Look at the screenshot and read the console log it prints.
  - Iterate until it looks photographic, not "CG demo". Put all scratch files in `scratch/<your-area>/` (git-ignored); keep them out of `public/`.

## 1. Tooling

| Tool | Path / command |
|---|---|
| Blender 5.2 (headless) | `/Applications/Blender.app/Contents/MacOS/Blender -b -P tools/blender/<script>.py -- <args>` (bundles numpy) |
| Dev server (already running) | `http://localhost:5173`. **Do not start another one.** If it is down, start it with `npx vite` in the background. |
| Screenshot + console | `node tools/shot.mjs "/?autostart&debug&camS=300,-1.5,1.6,30" scratch/x.png [--wait=3000] [--w=1280 --h=720] [--eval="js"]`. It prints JSON with console errors and `window.__STATS` (fps, calls, triangles). |
| glTF optimize | `npx gltf-transform optimize in.glb out.glb --compress meshopt --texture-compress webp --texture-size 2048`. Also `npx gltf-transform inspect x.glb`. |
| Road data generator | `node tools/gen_road.mjs` (lead only; road.json is frozen) |
| Libraries | three@0.186.1, postprocessing@6.39.5, n8ao@2.0.1, @dimforge/rapier3d-compat@0.21.0 |

GLBs are loaded with meshopt + KTX2 support (`ctx.assets.gltf(path)`), so meshopt geometry compression and WebP textures inside GLBs both work.

## 2. World space and the road

- **Space:** three.js world space, meters, **+Y up**.
  - The road runs roughly along **+X** (from x≈0 to x≈1260). The **uphill** side is **−Z** ("left" when driving) and the valley is **+Z**.
- **Blender:** Blender is Z-up. The glTF exporter's default `export_yup=True` maps Blender `(x, y, z)` to three `(x, z, −y)`.
  - So in Blender, three's −Z (uphill) is Blender **+Y**.
  - Build in Blender coords with that mapping, or do the conversion yourself. Always double-check by loading in the engine.
- **`public/assets/world/road.json`** is frozen (written by `tools/gen_road.mjs`):
  - `points`, `tangents`, `lefts`: flat arrays with one xyz per sample, every `step = 0.5` m of arc length. `s = index * step`, `s ∈ [0, 1300]`.
  - `left` = unit horizontal vector toward uphill.
- **Signed lateral offset `d`:** `d > 0` is uphill/left, `d < 0` is valley/right.
- **Cross-section**, at `y = centerline y` plus a 2% crown falling to both sides:
  - Asphalt: `|d| ≤ 3.0`, 2 lanes. Lane centers are at d = ±1.5. People drive on the right, so the **player lane is d = −1.5**.
  - Valley shoulder: gravel from `d = −3.0 … −3.9`, then the drop-off (steel guardrail at d ≈ −3.7 except in the gaps listed below).
  - Uphill ditch: `d = 3.0 … 4.3`, about 0.35 m deep, then the **rock cut face**. It rises near-vertically for 6–14 m, then continues as a steep natural slope.
- **JS API:** `ctx.road` is a `RoadPath` (`src/core/road.js`):
  - `pointAt(s)`, `tangentAt(s)`, `leftAt(s)`, `worldAt(s, d)`, `yawAt(s)` (for +Z-forward objects), `frameQuat(s)`, `project(pos) → {s, d, dy, dist}`, and `markers`.
- **Python helper:** `tools/blender/common.py` loads road.json and provides the same functions for Blender scripts.

### Level markers (`road.markers`, s in meters)

| key | s | meaning |
|---|---|---|
| carStart | 40 | cold-open drive starts here, lane d=−1.5, facing +s |
| scar | 90–190 | big landslide scar on the uphill slope: bare mud, no trees; the main slide comes from here |
| rockfallIntro | 150 | scripted boulders crash onto the road at s≈135–155 right after the car passes s≈175, blocking the way back |
| stall | 235 | engine dies, car coasts to a stop around 235–250 |
| fallenTree | 305 | big conifer lying across the whole road, root plate uphill, crown hanging over the valley edge. It blocks the car; the player can climb over it (trunk top ≤ 0.65 m above the road at the crossing point d≈0) |
| pulloff | 365–425 | valley-side widened gravel pull-off, flat, out to d = −14 (guardrail follows its outer edge) |
| roadworks | 395 | roadworks site: barriers, cones, light tower, crates, toolbox, **jerrycan**, **hatchet**, **plank stack** |
| gap | 560 | washed-out trench across the whole road, `|s−560| < 1.3`, 1.8 m deep, broken asphalt edges. It continues as an erosion gully down the valley side and as a wet ravine up the uphill side. No guardrail at 548–572. |
| gullies | 650, 770, 890, 1010 | rock chutes (V-shaped ravines) on the uphill slope; rockfall crosses the road here during the escape |
| tunnel | 1150 | tunnel portal plane: a concrete headwall about 30 m wide and 16 m tall in a rock spur; the road enters the tunnel |
| tunnelEnd | 1260 | tunnel interior runs 1150–1260 (lit by sodium lamps); dead end/dark beyond |
| win | 1175 | player or car beyond this s means escaped |

## 3. Systems and ownership

Every system module default-exports a class `new X(ctx)` with optional `async init()`, `fixedUpdate(h)` (before each 1/60 s physics step), `update(dt)` (every frame; dt=0 while paused) and `dispose()`. Instances live at `ctx.<key>`. Init order and update order are in `src/main.js`.

| key | file(s) | owner (workstream) |
|---|---|---|
| hud | `src/ui/hud.js`, `src/ui/style.css`, `src/ui/*` | UI |
| env | `src/render/environment.js`, `src/render/fog.js`, `src/render/materials.js` | RENDER |
| post | `src/render/post.js` | RENDER |
| terrain | `src/world/terrain.js`, `src/render/terrainMaterial.js`, `src/render/roadMaterial.js`, `tools/blender/terrain.py`, `tools/blender/common.py` | TERRAIN |
| vegetation | `src/world/vegetation.js`, `src/render/impostor.js`, `src/render/grass.js`, `tools/blender/trees.py` | TREES |
| rocks (assets only) | `tools/blender/rocks.py` → `public/assets/models/rocks.glb` | ROCKS |
| props | `src/world/props.js`, `tools/blender/props.py` | PROPS |
| landslide | `src/world/landslide.js` | SLIDE |
| particles | `src/world/particles.js` | SLIDE |
| car | `src/physics/vehicle.js`, `tools/blender/car.py` | CAR (Blender model) + VEHICLE (physics/driving) |
| player | `src/physics/character.js` | VEHICLE |
| cameraRig | `src/game/camera.js` | GAME |
| interact | `src/game/interact.js` | GAME |
| game | `src/game/sequence.js`, `src/game/inventory.js` | GAME |
| audio | `src/audio/*.js` | AUDIO |
| core | `src/main.js`, `src/core/*`, `src/physics/world.js`, `index.html`, `vite.config.js`, `tools/shot.mjs` | LEAD |

### Shared context `ctx`

```
ctx.THREE, ctx.renderer, ctx.scene, ctx.camera (PerspectiveCamera; fov 70, near .08, far 6000)
ctx.config (src/core/config.js: config.quality preset, config.game tuning), ctx.flags (URL flags)
ctx.events (bus), ctx.input (Input), ctx.assets (Assets), ctx.road (RoadPath), ctx.physics (Physics), ctx.debug
ctx.control: 'car' | 'foot' | 'none'    — which controller consumes gameplay input (set by game)
ctx.paused: boolean, ctx.time: {now, dt, frame, scale}
ctx.<system> for each system above (may be undefined if it failed to load!)
```

### Public APIs (implement at least these; extra is fine)

**env** (`render/environment.js`)
- `sun: DirectionalLight` (casts shadows; shadow camera follows `ctx.camera` automatically in update).
- `hemi` / ambient as needed; `scene.environment` = PMREM of `assets/sky/env_2k.hdr`; a visible sky (high-res sky dome from the 8k HDR, see RENDER).
- `wetness: number 0..1` (rain wetness, default 0.75); `rain: number 0..1` (rain intensity, default 0.35); `lightning(intensity)` optional.
- `fog`: height fog installed globally via ShaderChunk override. **All built-in materials get it automatically.**
  - A custom `ShaderMaterial` must set `fog: true` and merge `THREE.UniformsLib.fog` plus `ctx.env.fogUniforms` into its uniforms.
  - In the vertex shader, declare `vec3 transformed` (object-space position) and call `#include <fog_pars_vertex>` / `#include <fog_vertex>` after `mvPosition` is computed. In the fragment shader, call `#include <fog_pars_fragment>` / `#include <fog_fragment>`.
  - The height-fog chunk computes the world position itself, using `modelMatrix` (and `instanceMatrix` when instanced).
- `materials` helpers (`render/materials.js`):
  - `await pbrMaterial(ctx, texName, {repeat, triplanar?})` → MeshStandardMaterial with the ARM texture mapped to aoMap/roughnessMap/metalnessMap.
  - `triplanarPatch(material, {scale})` for world-space triplanar mapping.

**post** (`render/post.js`)
- `render(dt)` renders the frame. It also provides `setDOF(focusDist | null)` and `setExposure(v)`. `flash(color, t)`, `fadeBlack(0..1)` are optional.
- `impactBlur(amount)` is optional.
- The chain is N8AO → Bloom → SMAA → tone mapping (AgX) → subtle chromatic aberration, vignette and film grain. Quality comes from `config.quality`.

**terrain** (`world/terrain.js`)
- It loads `assets/world/terrain.glb` and creates the static colliders: terrain near, road, tunnel.
- `heightAt(x, z) → y | null` (raycast into static geometry), `meshes: {near, far, road, tunnel}`, `surfaceAt(pos) → 'asphalt'|'gravel'|'dirt'|'rock'|'grass'|'mud'`.

**vegetation**: loads trees/impostors from `assets/models/trees.glb` + `assets/models/impostors/*`, places them from `assets/world/scatter.json`, and draws instanced grass near the camera. It exposes `fallenTree` handling helpers for GAME (see the Trees section).

**props** (`world/props.js`): builds the roadworks site, guardrails, delineator posts and signs. It exposes interactable items:
- `items: { jerrycan: Object3D, hatchet: Object3D, planks: Object3D }` (world objects that can be picked up).
- `hideItem(id)`.
- `placePlanks()` spawns the plank bridge at the gap and creates its colliders. It returns the bridge Object3D.

**landslide** (`world/landslide.js`)
- `frontS: number` is the current s of the debris front. It is 0/undefined before it exists; the front mesh advances toward +s.
- `setFront(s)`, `setFrontSpeed(v)`, `triggerIntroRockfall()` (scripted boulders at s≈135–155).
- `triggerGully(i, opts)` rolls boulders down gully i. `startEscape()` ramps rockfall ahead of the car.
- `rumble: number 0..1` (continuous ground-rumble level for audio/camera).
- `spawnBoulder(pos, vel, radius)`.
- It emits `impact` events. It kills the player/car if the front reaches them or a boulder hits hard enough (it emits `hazard:hit`).

**particles** (`world/particles.js`): `dust(pos, size, energy)`, `debris(pos, count)`, `splash(pos)`; rain (camera-following) and windshield drops (when `ctx.cameraRig.mode === 'car-cockpit'`). It listens for `impact` events and spawns dust automatically.

**car** (`physics/vehicle.js`)
- `object: Group` (visual car), `body` (Rapier RigidBody), `controller` (DynamicRayCastVehicleController).
- `speed` (m/s, signed forward), `rpm`, `throttle`, `brake`, `steer`, `gear`, `fuel` (liters), `engineOn`, `stalling`.
- `lights: {head, brake}`, `seatCam: Object3D` (driver eye), `fuelCap: Object3D`, `doorPoint: Object3D` (where the player exits/enters).
- `start()` returns false if there is no fuel. `stop()`, `refuel(liters)`, `setControlsEnabled(b)`.
- `teleport(s, d, yawOffset=0)`, `damage` 0..1.
- Emits: `car:stall`, `car:start`, `car:startFail`, `car:impact {impulse}`, `car:gear`.

**player** (`physics/character.js`)
- `object` (capsule root), `eye: Vector3` (world eye position), `yaw`, `pitch`, `lookDir(out)`.
- `setEnabled(b)`, `teleport(pos, yaw)`, `onGround`, `velocity`, `stamina 0..1`, `sprinting`.
- Emits `footstep {surface, run}`, `player:land {speed}`.
- On foot it reads WASD + mouse only when `ctx.control === 'foot'`. It autosteps ≤ 0.65 m (to climb over the fallen trunk) and refuses slopes steeper than 50°.

**cameraRig** (`game/camera.js`)
- `mode: 'foot'|'car-chase'|'car-cockpit'|'cinematic'`, `setMode(m)`.
- `shake(trauma 0..1)` (trauma model: decays; offset/rotation noise ∝ trauma²), `cinematic(pathFn, duration)` → Promise.
- It adds head-bob on foot, FOV kick with speed, and C to toggle chase/cockpit.

**interact** (`game/interact.js`)
- `register({id, object | position, radius, prompt, canUse(ctx)→bool, use(ctx)})`, `unregister(id)`.
- It shows `[E] prompt` through the hud when looking at or near an item. It handles E and uses a hold-to-use progress bar for chopping and refueling.

**game** (`game/sequence.js`)
- `state: 'title'|'intro'|'stalled'|'onfoot'|'driving'|'escape'|'win'|'dead'` and `checkpoint`.
- `inventory: Set<'jerrycan'|'hatchet'|'planks'>`.
- `fail(cause)`, `restartFromCheckpoint()`.
- Supports `?skip=stall|onfoot|refuel|escape|gap|tunnel` for testing.

**audio** (`audio/audio.js`)
- Procedural Web Audio: `unlock()` (called on the first user gesture; hud calls it on Start), `setMasterVolume`.
- `play(name, {position, volume})` for one-shots: 'chop', 'pickup', 'thud', 'crack', 'splash', 'door', 'fuelPour', 'ignitionFail', 'ignition'.
- It reads `ctx.car` (rpm, throttle, engineOn) and `ctx.landslide.rumble` and `ctx.env.rain` every frame for continuous loops.
- It listens to `impact`, `footstep`, and the car events.

**hud** (`ui/hud.js`)
- It owns all DOM UI under `#ui`: the loading screen with a progress bar (`ctx.assets.progress`) and the title screen.
  - Title: title, Start, a quality select (`config.setQuality` + reload), and controls help.
  - Also: pause (Esc), the death screen (cause and "Retry from checkpoint"), the win screen with time, and credits.
- It **emits `ui:start`** when Start is clicked; that click also calls `ctx.audio?.unlock()` and `ctx.input.requestLock()`.
- API:
  - `setPrompt(text|null, progress=null)`, `setObjective(text)`, `toast(text, seconds)`, `subtitle(text, seconds)`.
  - `setDriving(bool)`; `setFuel(0..1)` (analog gauge); `setSpeed(kmh)`; `setInventory(list)`.
  - `fade(toBlack:boolean, seconds) → Promise`.
  - `showScreen('title'|'pause'|'dead'|'win'|null, data)`.
  - `damageVignette(0..1)`.

### Events (`ctx.events`)

| name | payload | emitted by |
|---|---|---|
| `boot:progress` / `boot:done` | {key} | main |
| `ui:start` | {auto} | hud (or main on ?autostart) |
| `ui:pause` / `ui:resume` / `ui:retry` | – | hud |
| `impact` | {position: Vector3, energy 0..∞ (≈ kinetic energy / 1e5), radius} | landslide (rock hits), car (crash) |
| `hazard:hit` | {target: 'player'|'car', cause: 'boulder'|'front'|'fall', energy} | landslide / game |
| `car:stall`, `car:start`, `car:startFail`, `car:impact` | {…} | car |
| `car:enter` / `car:exit` | – | game |
| `footstep` | {surface, run, position} | player |
| `item:pickup` | {id} | game/interact |
| `objective` | {text} | game |
| `game:state` | {state, prev} | game |
| `tree:chop` | {hits, done} | game |
| `planks:placed` | – | game |
| `win`, `dead` | {cause} | game |

## 4. Asset contracts (Blender outputs)

All GLBs go to `public/assets/...`. Keep them small: meshopt-compressed, textures ≤ 2048, WebP or JPG inside the GLB. Use Principled BSDF-based materials so the glTF PBR export is correct (baseColor, normal, ORM packed as occlusion/roughness/metallic). Each script must be re-runnable, deterministic and seeded. Write a preview render to `scratch/<area>/*.png`.

### TERRAIN → `public/assets/world/terrain.glb` (+ `scatter.json`)
- Build it from `road.json`. The road strip **must** match road.json exactly: centerline height and a 2% crown.
- Meshes (node names are exact):
  - **`terrain_near`**: a corridor around the road about ±45 m laterally over s ∈ [−40, 1300]. Resolution about 0.3–0.5 m near the road and cut face, coarser farther out. It is decimated where flat.
  - **`terrain_far`**: large surroundings (the valley below, the mountains above, and the opposite side of the valley ~1–2 km away) out to ~3 km at low resolution. It meets `terrain_near` without visible cracks, using a small overlap and a slightly lowered skirt.
  - **`road`**: the asphalt strip `|d| ≤ 3.05` with the 2% crown, 3–5 cm above the terrain beneath it. **UV0 = (d, s) in meters.** It has the trench cut at the gap with broken edges and thickness.
  - **`tunnel`**: portal headwall, wing walls, and a tube 1150–1260 with a road-surface floor inside. UV0 is in meters.
  - **`tunnel_lamps`**: small emissive lamp fixtures along the tunnel ceiling.
  - **`guardrail_path`** (optional): a polyline, or put guardrail runs in scatter.json.
- **COLOR_0** on terrain meshes (RGBA, linear):
  - R = baked **ambient occlusion** (Cycles AO bake to vertex colors, ~30 m distance).
  - G = **gravel/shoulder** mask.
  - B = **mud/debris/wet** mask (scar, gullies, trench, water runoff).
  - A = **vegetation density** (0 on rock, cut faces, road, scar; 1 on gentle soil slopes).
  - The **road** mesh uses COLOR_0.R for AO and COLOR_0.G for a puddle/damage mask.
- The shape must *read* as a real mountain road:
  - A near-vertical, fractured rock cut on the uphill side with overhang-ish ledges (noise-displaced), then a steep forested slope rising 200–400 m to a ridge line.
  - On the valley side, a steep drop into a misty valley ~250–400 m below, with the far mountain range across the valley.
  - Spurs and re-entrant ravines should follow the road curvature. The landslide scar is a big fresh bare-mud slide surface. The gullies are V-ravines. The gap has an erosion channel. The tunnel spur is a rock ridge that the road tunnels through.
- **`scatter.json`**, all in three.js world coords:
  - `trees: [[x,y,z, scale, rotY, variant], ...]` for ~3–8k trees on soil slopes. Denser forest below and above the road; none on rock faces, the scar or the road.
  - `rocks: [[x,y,z, scale, rotY, rotX, variant], ...]` for static decorative rocks: on the scree, at the foot of the cut face, and in the trench.
  - `grassMask` hints are optional.
  - `slideSpawns: [{s, points:[[x,y,z]...], dir:[x,y,z]}]` per gully and for the scar: release points 20–60 m above the road with a downhill direction.
  - `guardrails: [[s0, s1], ...]` for runs where a guardrail exists at d = −3.7 (the pull-off edge is handled by PROPS using `pulloff`).
- The JS side (`world/terrain.js`, `render/terrainMaterial.js`, `render/roadMaterial.js`) builds these materials:
  - **Terrain:** a triplanar splat. aerial_rocks_02 and lichen_rock on slopes above ~42° and cut faces; aerial_grass_rock plus forest_ground_04 on gentle slopes (through the COLOR_0.A density); rocky_trail on shoulders and gravel; brown_mud_rocks_01 for the mud mask. Add macro variation noise to kill tiling, and darken and smooth everything for wetness (rain).
  - **Road:** asphalt_02 with worn white edge lines at d=±2.85 and a dashed center line, patched cracks, a wet sheen, procedural puddles (a noise-based mask that is mirror-smooth with a darker albedo), tire-track darkening in the lanes, and dirt/mud washed over the edges near the scar and the gap.

### CAR → `public/assets/models/car.glb` (+ `car.json`)
- An **old boxy mountain 4x4** (think early-90s compact SUV like a Lada Niva / Suzuki Samurai / Pajero). Boxy panels are what make it convincing from a procedural build.
  - Size: 3.9 m long, 1.70 m wide, 1.75 m tall. Wheelbase 2.30 m, track 1.42 m, wheel radius 0.36 m (tire), wheel width 0.22 m.
  - Paint: faded **dark red** (or olive), with dirt and mud on the lower third, rust spots and scratches.
  - Glass with transmission and roughness; chrome/black trim; rubber; a readable interior (dashboard, gauges, seats, steering wheel), because a cockpit camera exists.
  - Headlights: glass lens plus reflector; tail lights.
- **+Z forward, +Y up, +X = the car's LEFT side** (the driver side: left-hand drive). The origin is on the ground (y=0 at tire contact), centered between the axles.
- Nodes (exact names):
  - `car_body` (all non-wheel meshes as children).
  - `wheel_fl`, `wheel_fr`, `wheel_rl`, `wheel_rr`. Origins are at the wheel centers; the spin axis is local X; each is a separate object containing tire + rim + brake disc.
  - `steering_wheel` (origin at the hub; it rotates about its local Z).
  - `seat_cam` (empty at the driver's eye position), `fuel_cap` (empty at the filler flap on the rear LEFT (+X) side, just outside the body), `door_point` (empty 0.9 m outside the driver door at ground level).
- Materials (exact names, which the engine looks up): `paint`, `glass`, `chrome`, `trim_black`, `rubber`, `interior`, `headlight` (emissive when on), `brakelight` (emissive red when braking), `indicator`, `plate`, `gauge` (dashboard emissive).
- `car.json`: `{ mass, wheelbase, track, wheelRadius, wheelWidth, suspensionRest, suspensionTravel, cgHeight, frontAxleZ, rearAxleZ }`.
- Budget: ≤ 150k triangles and ≤ 12 MB.

### TREES → `public/assets/models/trees.glb` + `public/assets/models/impostors/`
- 4 conifer (Norway spruce / fir) variants, 14–28 m tall, built procedurally in Blender (trunk, branch whorls, needle **cards**).
  - The needle card texture is **rendered from procedural needle geometry** (RGBA albedo + normal).
  - Bark uses a procedural or CC0 texture (you may download a Poly Haven bark texture).
  - Nodes: `conifer_0..3` (LOD0, ≤ 20k tris each; used within `treeMeshDistance`). The origin is at the trunk base.
- **Impostors:** for each variant, an atlas of N views (for example 8 azimuths × 3 elevations) with RGBA albedo and a normal map, plus `impostors.json` (grid layout, elevations, per-variant height/width/pivot). The engine shows these beyond `treeMeshDistance` and handles thousands of trees.
- Also: `fallen_tree` (a conifer lying along local +X, about 18 m, with a root plate at −X, broken branches, and flattened branches on the underside; origin at the trunk center; the trunk top at the middle should be ≤ 0.65 m above the ground), plus `fallen_tree_a` and `fallen_tree_b` (the two halves after cutting at local x=0, with cut-wood faces).
- Also: `shrub_0..2` (low bushes) and `deadwood_0..1` (fallen branches). Grass is procedural in the engine.

### ROCKS → `public/assets/models/rocks.glb`
- `rock_0..9`: boulders with distinct shapes (angular fractured granite and gneiss blocks; no blobby potatoes).
  - Normalize to a bounding radius of ≈1.0 with the origin at the centroid. ≤ 3k tris each.
  - Normal/AO/cavity maps are baked from a high-poly sculpt. Use box-projected CC0 rock albedo (lichen_rock / aerial_rocks_02), baked into UV textures (a shared 2048 atlas is ideal).
- `rock_hull_0..9`: convex hulls ≤ 48 vertices, not rendered, used for Rapier colliders.
- `pebble_0..5`: small stones for scatter and debris (≤ 300 tris).

### PROPS → `public/assets/models/props.glb`
Nodes (origin at the base center unless stated):
- `jerrycan`: 20 L steel NATO-style, olive or red, embossed X pattern, ≈0.35×0.165×0.47 m.
- `hatchet`: ≈0.45 m; the origin is at the grip.
- `plank`: one scaffold board, 3.9×0.225×0.038 m, weathered wood, with steel band ends.
- `plank_stack`: 6 boards on 2 bearers.
- `barrier`: concrete jersey barrier, 2.0×0.6×0.8 m, red and white paint, worn.
- `cone`: 0.75 m traffic cone with reflective bands.
- `crate`: wooden. `toolbox`: steel.
- `light_tower`: a portable 2.5 m tripod work light. Its child `lamp_head` has the emissive material `lamp`.
- `sign_rockfall`: triangular warning sign on a post. `sign_roadworks`.
- `guardrail`: W-beam 4.0 m segment plus one post; it runs along +Z from the origin.
- `guardrail_end`: terminal piece.
- `delineator`: a white roadside post with a reflector, 1.0 m.
- `sandbag`.
- Optional: `excavator_bucket` or `wheelbarrow` for the site.

## 5. Rendering and look targets

- **Mood:** a cold, wet, overcast mountain afternoon with light rain, low clouds and mist in the valley. The sky is the `overcast_soil_puresky` HDRI.
  - There is no harsh sun: use a soft directional "sun" matched to the brightest HDRI region, low intensity, soft shadows.
  - Use aerial perspective and height fog. It thickens below the road level and toward the valley; far mountains become blue-grey silhouettes.
- **Tone mapping:** AgX in post. Target an exposure where the asphalt is mid-dark grey and the sky is bright but not clipped.
- **Wet look:** darker albedo, lower roughness on asphalt and rock, puddles reflecting the sky (env map), rain streaks and droplets.
- Everything casts and receives shadows where it matters: the car, props, near trees and rocks.
- **Reference:** real dashcam and photographic footage of Alpine and Himalayan mountain-road landslides. If it looks like a video game from 2012, it's not done.

## 6. Physics

- The Rapier world is `ctx.physics.world`, a fixed 1/60 s step.
- **Collision membership bits** (`G` in `src/physics/world.js`): STATIC 1, CAR 2, PLAYER 4, ROCK 8, DEBRIS 16, PROP 32, SENSOR 64.
  - Use `groups(member, filter)`.
- **Static geometry:** terrain_near, road, tunnel and the barriers are trimeshes or boxes with the STATIC group. The fallen tree is a compound of capsules/boxes (STATIC until cut).
- **Car:** a dynamic chassis (compound of cuboids) plus `DynamicRayCastVehicleController`. The mass is from car.json.
- **Player:** `KinematicCharacterController` with a capsule of radius 0.3 and half-height 0.6 (eye at 1.65 m).
- **Rocks:** dynamic convex hulls with restitution about 0.2 and high friction. Pool them and put them to sleep or remove them when far behind.

## 7. Debug & testing

- The URL flags are in `src/core/debug.js`: `?autostart`, `?debug`, `?only=`, `?skip=`, `?cam=x,y,z,tx,ty,tz`, `?camS=s,d,h,ahead`, `?freecam`, `?quality=`, `?nopost`, `?mute`.
- `window.__ctx` is always set, and `window.__READY` becomes true after the first frames.
- `ctx.input.simulate({KeyW:true})` holds keys, for automated driving tests.

## Change log
- (append entries: date, workstream, what changed and why)
- 2026-09-27 LEAD: config.quality now includes `key` ('ultra'|'high'|'medium'|'low'); compare keys, not display names. road.project() falls back to a global scan when the windowed result is at the window edge or > 8 m away (fixes stale-hint errors after teleports).
- 2026-09-27 PROPS: plank_stack is 12 boards; extra nodes plank_wedge, guardrail_bent, guardrail_lod, rail_reflector, tarp_pile. Extra API: showItem, removePlanks, setLampOn, bridge, lightTower, lampLight. Lamp spot shadow is Ultra-only. Extra CC0 textures: rust_coarse_01, concrete_wall_006, ash_veneer, hessian_230.
- 2026-09-27 SLIDE: the intro parks the front at s=150 (speed 0) if unset. The front wall collider is kinematic STATIC at frontS+0.3..2.7 and ignores ROCK; the front kills only while moving. Tyre spray/exhaust vapour live in vehicle.js.
- 2026-09-27 AUDIO: play() returns {stop(fade)}. Extra names: rock, crash, clunk, drip, keyClick, footstep, breath. window.__AUDIO meter. Engine runs in an AudioWorklet (src/audio/engine.worklet.js); banks render in a worker. Reads car.fuelCut, car.cranking, car.sputter.
- 2026-09-27 UI: setInventory(ids, details?), letterbox(bool|null), hideHud(bool), pause(), resume(), playTime, settings; events ui:quit, ui:settings; a `[K]` in prompt text sets the key glyph. Fonts: Barlow via Google Fonts with offline fallback.
- 2026-09-27 TERRAIN: extra CC0 texture sets dark_rock_02, concrete_wall_006, pine_bark. scatter.json adds a 'deadwood' key. terrain.insideTunnel(0..1). Rule: NO transmissive materials (transmission forces a second full render pass).
- 2026-09-27 TREES: mesh trees out to 0.6×treeMeshDistance, impostors beyond. Shadow proxies are used for foliage shadows.
- 2026-09-27 RENDER: low cloud deck in the fog (hfogCloud uniforms, HFOG_NO_CLOUD opt-out), procedural sky cloud structure (env.look.skyVar), ?postdbg perf overlay.
- 2026-09-28 LEAD: pre-load quality gate (src/boot.js entry -> src/ui/gate.js), Low default; Medium/Low load variants from tools/make_variants.mjs (public/assets/q/variants.json, rewritten in core/assets.js url()); Ultra/High always load originals. Parallel asset prefetch in main.js; adaptive resolution on Medium/Low; vercel.json cache headers.
- 2026-09-28 LOWEND: lite terrain/road/fog shader tiers for Low/Medium (?terrainLite=0|1|2), float-RT fallbacks (materials.floatRTSupport), boot compiles only render-target shader variants.
- 2026-09-28 HANDS: tools/blender/hands.py -> public/assets/models/hands.glb (hand_R/hand_L armatures, arm_R/arm_L meshes, node extras poses/sockets). cameraRig.viewmodel API: setHeld, windup, strike, pour, place, reach; cameraRig.kick(); thin-lens focusOn. CC0 textures brown_leather, stretch_poplin (bake-only).
- 2026-09-28 TREES: fallen tree v2 (fallen_debris, fallen_chips nodes; material 'fallen' unique baked atlas; collider JSON v:2 with yaw 0.80). CC0 knotted_pine_bark (bake-only).
- 2026-09-28 LEAD: CrazyGames HTML5 SDK v3 (src/core/platform.js): env-guarded calls, loading/gameplay events, midgame ads only on Retry/Quit breaks (game held, audio suspended), muteAudio setting, happytime on win, iframe key/wheel scroll guards. window.__platform exposed for tests.
