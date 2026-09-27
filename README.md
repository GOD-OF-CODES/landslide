# LANDSLIDE

A short (5–8 minute) first-person escape game in the browser, aiming for the look of dashcam footage from a real Alpine or Himalayan mountain road in the rain.

It is a cold, wet afternoon, and you are driving an old 4x4 up a narrow cliff road. Behind you the slope lets go and boulders close the way back. Then the fuel runs out. On foot, you find a roadworks site with a jerrycan, a hatchet and a stack of scaffold planks. You cut through a spruce that has fallen across the road, refuel, and coax the engine back to life. Then you drive for your life while the mountain comes down around you. Boulders pour out of the gullies, the road is washed out, and your only way through is a tunnel in the rock spur ahead.

![Intro drive](docs/screenshots/01_intro_drive_chase.jpg)

| | |
|---|---|
| ![Rockfall](docs/screenshots/02_rockfall.jpg) | ![Roadworks](docs/screenshots/03_onfoot_roadworks.jpg) |
| ![Chopping the fallen tree](docs/screenshots/04_chopping_tree.jpg) | ![Escape: rocks crossing the road](docs/screenshots/05_escape_rocks_cockpit.jpg) |
| ![The tunnel](docs/screenshots/06_tunnel_win.jpg) | |

*All screenshots are real-time captures at 1280×720 on the default "High" preset.*

Built with three.js r186 (WebGL2), Rapier physics, pmndrs postprocessing and N8AO. **There are no downloaded 3D models.** The terrain, car, trees, rocks, props and tunnel are all built from scratch, either by Blender Python scripts or procedurally in JavaScript. All audio is synthesized at runtime. The only external assets are CC0 textures and one HDRI from [Poly Haven](https://polyhaven.com).

## Running it

You need Node.js 20.19+ or 22.12+ and a recent desktop browser with WebGL2. The game was developed and tested in Chrome.

```bash
npm install
npm run dev          # http://localhost:5173
```

Open **http://localhost:5173**, pick a quality preset under *Settings* if you like, and press **Start**. Click into the page to capture the mouse. Headphones are recommended.

Production build (static files, relative paths, so it can be hosted from any sub-folder):

```bash
npm run build        # -> dist/
npm run preview      # serves dist/ at http://localhost:4173
```

## Controls

| On foot | | Driving | |
|---|---|---|---|
| W A S D | move | W / S | throttle / brake, reverse |
| Mouse | look | A / D | steer |
| Shift | sprint | Space | handbrake |
| Space | jump | C | cockpit / chase camera |
| E | interact (hold to chop, pour, lay planks) | E | get in / out, start the engine |
| Tab | show the current objective | Esc | pause (settings, controls, retry) |

## Quality settings

Choose a preset on the title screen (*Settings → Graphics quality*; the page reloads), or add `?quality=ultra|high|medium|low` to the URL. The choice is remembered in `localStorage`.

| Preset | Render resolution | Sun shadow | AO | Wet-road reflections | Extras | Vegetation |
|---|---|---|---|---|---|---|
| **Ultra** | native, up to 2× DPR (Retina) | 4096², 70 m, lamp spot shadow | N8AO medium, full-res | screen-space, 20-step march (road + puddles) | depth of field, motion blur | grass 70 m, mesh trees 110 m |
| **High** (default) | 1× CSS pixels | 2048², 60 m | N8AO low, half-res | screen-space, 10-step march (road) | depth of field, motion blur | grass 55 m, mesh trees 80 m |
| **Medium** | 0.8× | 2048², 50 m | N8AO performance | screen "infinity" sample only | | grass 40 m, mesh trees 55 m |
| **Low** | 0.65× | 1024², 40 m | off | env map only | no bloom | grass 28 m, mesh trees 35 m |

The post chain is N8AO, a volumetric mist pass, a half-resolution history copy for the wet-road reflections, then bloom, a lens model (slight barrel distortion, lateral chromatic aberration, veiling glare, off-axis softness), AgX tone mapping and a colour grade, then SMAA, vignette and film grain. Eye adaptation is driven by a centre-weighted exposure meter that behaves like a phone camera (the tunnel mouth stays a dark hole until you are inside). The presets live in `src/core/config.js`. Under the overcast sky shadows are soft anyway, so High renders the sun's shadow map at 2048² (`sunShadowMapSize`), which saves about 1 ms a frame on the M1 compared with 4096².

**Performance target:** at least 45 fps at 1280×720 on *High* on an Apple M1 (7-core GPU, 8 GB), within about 1.5 M triangles and 400 draw calls. The final QA pass measured this on a fanless MacBook Air M1:

| Measurement (High, 1280×720) | Result |
|---|---|
| Full automated playthrough, average fps per 50 m section (three full runs, the last on the final build; no deaths) | 54–61 fps in the intro, on foot and in the escape. The slowest section is the washout (laying the planks on foot, looking into the trench): 54 fps after a 10-minute cool-down, 48–52 fps on a warm machine |
| Lowest one-second sample in those runs | 47 fps |
| Fixed viewpoints every 100–150 m along the route | 56–60 fps, 0.31–1.62 M triangles, 83–321 draw calls |
| Two of those viewpoints (s = 150, 560) after ~90 min of continuous load (thermal throttling) | 50–52 fps |
| Full-screen 1440×900 window (1× DPR), four viewpoints, throttled | 38–41 fps |

The MacBook Air has no fan, so after 30–60 minutes of full GPU load it throttles and every number above drops by about 10 %. Frame rates also drop sharply if another app is using the GPU at the same time: in one session another app kept the GPU about 55 % busy, and the same build measured 36–49 fps. Choose *Ultra* on a faster GPU for Retina-native resolution, or *Medium* on a warm or busy laptop.

## How it was made

### Rendering (`src/render/`)
- **Terrain:** a biplanar splat of nine scanned CC0 texture sets, packed into texture arrays and tiled at each scan's real capture size (0.2–2 m fracture blocks on the rock cut). It is driven by baked vertex masks (AO, gravel, mud, vegetation) plus slope and macro noise. The rain-wetness model darkens porous surfaces, leaves overhangs dry, and draws water on the rock faces: black cyanobacteria "ink stripes", live seeps, iron staining and calcite crusts below joints, sheet flow, blasting half-casts on the cut, and a soaked halo around each gully fall. The washout shows the road structure in section (asphalt, pale sub-base, brown fill). Far mountains get alpine zonation (turf, scree fans, rock bands, fresh snow). The tunnel lining ends in a concrete arch ring set into the headwall, so no sky shows through the seam around the tunnel mouth.
- **Road:** saw-cut repair patches and trench reinstatements with tar overbands, sealed and open cracks, alligator cracking, faded thermoplastic lines, polished wheel paths, rutted puddles with rain ripples, sheet flow and mud tongues below the gullies, and grit spilled from the slide. Screen-space reflections add a patchy water-film lobe, so the wet road reflects the real tree line, car and barriers, stretched vertically as on real wet asphalt.
- **Water:** thin running-water ribbons in the ditch and the gullies, with falls where the gullies meet the road cut. The rock around each fall is soaked dark and glossy (only the falls within about 220 m of the camera are evaluated).
- **Atmosphere:** global height fog and aerial perspective, done with ShaderChunk overrides so every material gets it. It adds valley mist sheets, a low cloud deck and sun in-scatter. The volumetric mist is ray-marched at reduced resolution and upsampled with a depth-aware 4×4 filter, so it stays smooth and shows no dither pattern. The sky dome and the image-based lighting both come from the 8k overcast HDRI.
- **Rain:** streaks drawn from a Marshall-Palmer drop-size distribution, each falling at its terminal velocity and smeared over the camera's exposure time, plus drifting rain veils, ground mist, splashes on the road, tyre spray, and refracting drops and a wiper-swept water film on the windshield in the cockpit view.
  - **Photometry:** the far rain is calibrated to real coverage. At about 4 mm/h there are about 220 drops larger than 0.8 mm per m³, and each one covers a pixel for only about 1.5 % of a 1/55 s exposure. Rain beyond a few metres therefore shows as a faint, fine texture (mean opacity 0.5–1 %) against dark rock and trees, not as bright dashes.
  - **Tyre spray:** matched to a small car at 40–50 km/h on a film of water about 1 mm deep: a low, continuous haze behind each tyre, about 1 m high, that clears in about half a second. The heavier drops are a few centimetres across.
- **Dust:** rockfall dust has the albedo of the crushed rock itself (about 0.3 dry, 0.2 wet), so it reads about as bright as the concrete, well below the overcast sky. The rain thins it quickly.
- **Loading:** every shader is compiled during the loading screen, including the shadow-map variants of objects that only appear later (boulders, the chopped tree), so nothing stalls mid-game.
- **Forest:** thousands of instanced conifers (three Norway spruce variants, one silver fir and a dead snag). They go from LOD0 to LOD1 meshes, then to multi-view impostors rendered from the Blender trees, with dithered cross-fades, wind, and procedural grass and ferns near the camera.

### Gameplay and physics (`src/game/`, `src/physics/`, `src/world/`)
- **Car:** a Rapier `DynamicRayCastVehicleController` with a fuel model and a stalling engine.
- **Player:** a kinematic character controller. It can step over the fallen trunk.
- **Rockfall:** convex-hull boulders on a "fair" timing model, so rocks land where a sensible driver can still avoid them, plus a mud-and-debris front that chases you. From about 60 m before the tunnel, rocks only fall behind the car, so the approach to the portal stays clear. If a boulder rolls the car onto its roof, the run ends after 2.5 s with a "The car rolled over" screen, rather than leaving you stuck until the front arrives.
- **Interactions:** hold-to-use chopping (each blow widens a V notch in the trunk), refuelling and plank laying.

### Audio (`src/audio/`)
Everything is procedural Web Audio: rain, wind, rumble, rock impacts, wood chops, footsteps on seven surfaces (asphalt, gravel, mud, rock, dirt, grass, wood), the fuel glug and the door clunk. The engine is an `AudioWorklet` synthesizer driven by rpm and load. One-shot sample banks are rendered in a Web Worker at start-up. There are no recorded samples.

### Assets: Blender scripts (`tools/blender/`)
Every model is generated by a deterministic, seeded Python script that runs headless in Blender 5.2 (it bundles numpy). Each script bakes its textures, writes a meshopt/WebP-compressed GLB into `public/assets/`, and renders preview images into `scratch/`. Re-running the scripts regenerates everything. Run one Blender process at a time; each needs up to about 2 GB of RAM.

```bash
B=/Applications/Blender.app/Contents/MacOS/Blender        # adjust for your OS
node tools/fetch_assets.mjs                                 # CC0 source textures + HDRI -> raw_assets/ (textures only, no models)
node tools/gen_road.mjs                                     # road centreline -> public/assets/world/road.json (frozen)
$B -b -P tools/blender/terrain.py                           # terrain.glb + scatter.json: cut face, slide scar, gullies, washout, tunnel
$B -b -P tools/blender/trees.py                             # trees.glb (spruce/fir LODs, fallen tree, shrubs) + impostor atlases
$B -b -P tools/blender/rocks.py                             # rocks.glb: fractured granite/gneiss boulders, hulls, pebbles
$B -b --factory-startup -P tools/blender/props.py           # props.glb: jerrycan, hatchet, planks, barriers, light tower, guardrail...
$B -b --python-exit-code 1 -P tools/blender/car.py          # car.glb + car.json: early-90s compact 4x4 with a full interior
node tools/sky.mjs                                          # sky-dome textures from the 8k HDRI
```

- `tools/blender/common.py` loads `road.json` and exposes the same road frame as the engine (`s` = distance along the road, `d` = signed lateral offset).
- The **needle cards** of the trees are rendered from procedural needle geometry.
- **Rock** normal, AO and cavity maps are baked from voxel-remeshed high-poly sculpts.
- The **car** paint wear, mud and rust are baked from procedural masks.
- Each script has options (`--stage`, `--only`, `--fast`, …) documented at its top.
- `public/assets/tex/` holds the engine's resized copies of the Poly Haven texture sets. `raw_assets/` (git-ignored) holds the full-size downloads that the Blender bakes and the sky tool read.

## Debug and testing

URL flags (see `src/core/debug.js`):

| Flag | Effect |
|---|---|
| `?debug` | fps, draw-call and triangle overlay |
| `?autostart` | skip the title screen |
| `?skip=stall\|onfoot\|refuel\|escape\|gap\|gap_after\|tunnel` | start at a checkpoint |
| `?camS=s,d,h,ahead` | fixed camera relative to the road |
| `?freecam` | fly camera: WASD/QE, right-drag to look |
| `?only=env,terrain,…` | load only some systems |
| `?nopost` | no post-processing |
| `?mute` | no audio |
| `?quality=…` | pick a quality preset |

- `node tools/shot.mjs "/?autostart&debug&camS=300,-1.5,1.6,30" out.png` takes a headless-Chrome (GPU) screenshot and prints console errors plus frame stats.
- `node scratch/game/playthrough.mjs [--shots] [--perf] [--base=http://127.0.0.1:4173]` plays the whole game automatically from the title to the win, and reports deaths, console errors and per-section fps. Note that `scratch/` is git-ignored: these test scripts exist only in the local working copy, not in the repository.

## Project layout

```
index.html, src/main.js     boot + system registry (update order, fixed 60 Hz physics step)
src/core/                   config (quality presets, tuning), asset loader, road frame, input, events, debug flags
src/render/                 environment (sky, IBL, sun), fog, materials, terrain/road shaders, trees/impostors, grass, post
src/world/                  terrain, vegetation, props, landslide, particles
src/physics/                Rapier world, vehicle, character
src/game/                   game sequence and checkpoints, interactions, inventory, camera rig
src/audio/                  procedural audio (worklet engine, worker-rendered banks, DSP)
src/ui/                     HUD, title/pause/death/win screens, analog gauges
tools/                      Blender scripts, asset fetch, road generator, sky builder, screenshot tool
public/assets/              GLBs, textures, sky, road/scatter data
DESIGN.md                   the design contract (world space, APIs, asset contracts, budgets)
```

## Credits

- **Textures and HDRI:** [Poly Haven](https://polyhaven.com), [CC0](https://creativecommons.org/publicdomain/zero/1.0/).
  - HDRI: *overcast_soil_puresky*.
  - Textures: *asphalt_02, aerial_rocks_02, aerial_grass_rock, lichen_rock, dark_rock_02, marble_cliff_03, gray_rocks, mossy_rock, quarry_wall, rock_01, brown_mud_rocks_01, brown_mud_02, brown_mud_03, mud_forest, rocky_trail, forest_ground_04, precast_concrete_wall, concrete_wall_006, pine_bark, rough_wood, ash_veneer, hessian_230, rust_coarse_01*.
- **Made from scratch for this project:** all 3D models (the car, trees, rocks, props, the tunnel), the terrain, and all audio.
- **Libraries:** [three.js](https://threejs.org) (MIT), [Rapier](https://rapier.rs) (Apache-2.0), [postprocessing](https://github.com/pmndrs/postprocessing) (Zlib), [N8AO](https://github.com/N8python/n8ao) (ISC), [Vite](https://vite.dev) (MIT).
- **Font:** [Barlow](https://fonts.google.com/specimen/Barlow) (SIL OFL), loaded from Google Fonts with an offline fallback.
