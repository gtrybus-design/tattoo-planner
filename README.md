# Tattoo Planner

A 3D tattoo planner that runs in the browser: a realistic, posable human body built from real anatomy, with a
true-scale flat map of its skin so flat tattoo designs can be placed on the body at their real size.

The body is not a hand-sculpted model. A Python build pipeline takes a real skeleton and muscles (Z-Anatomy /
BodyParts3D), grows fat and skin over them, rigs everything to a skeleton so it bends like a person, and flattens
the skin into a 2D map. The web page (three.js) then shows the body with realistic skin shading and lets you pose
it live.

For a plain-language walkthrough of how everything works, including a 15-lesson Python course that runs in the
page, open **[docs/how-it-works.html](docs/how-it-works.html)**.

---

## Running it

Open `index.html` in Chrome. three.js loads from a CDN, so the first load needs an internet connection. If your
browser blocks local files, serve the folder instead:

```sh
python3 -m http.server 8000      # then open http://localhost:8000
```

**Controls**

| | |
|---|---|
| Left-drag on the body | move that body part |
| Right-drag on the body | rotate it |
| Drag off the body | orbit the camera (right-drag pans, scroll zooms) |
| Pose buttons (top left) | A-pose, Standing, Star, Reach, Hands behind head |
| Layer buttons (top right) | Skin, Adipose (fat), Muscles + cartilage, Skeleton, Skin map |
| Flex slider | how hard the muscles are working (more bulge, tighter shape) |
| Creases | draws the inside of the skin dark, so a leftover fold reads as a crease |
| Copy / Paste pose | share an exact pose as text |

---

## Rebuilding the body

You only need this after changing the build tools. The page uses the generated `.js` files directly.

```sh
python3 -m venv .venv && .venv/bin/pip install numpy scipy libigl pillow    # once

.venv/bin/python -I tools/fit_rig.py anatomy-src assets/male_base_mesh.glb .   # body, rig, weights -> body-rig.js
.venv/bin/python -I tools/uv_atlas.py .                                        # skin map -> skin-uv.js, skin-map.*
.venv/bin/python -I tools/skin_detail.py .                                     # skin pores/furrows -> skin-detail.js
```

Always rerun `uv_atlas.py` after `fit_rig.py`: the skin map belongs to one particular body, and the page checks
this with a fingerprint.

---

## How it works (short version)

1. **Skeleton and muscles.** Real anatomy meshes (Z-Anatomy) are matched onto a rig: every joint of the rig is
   moved onto the matching joint of the real skeleton. Every bone is attached to its rig bone, and every muscle is
   weighted to the joints it spans.
2. **Fat and skin.** The muscles and bones are voxelized, and a smooth surface is extracted around them. Fat
   thickness varies by body region and comes from skinfold and ultrasound studies, scaled to a **17% body fat**
   build with the Jackson–Pollock 3-site and Siri equations (skinfold sum 49.7 → 56.6 mm; average fat target
   6.6 → 7.5 mm). Hands, feet and face come from the MakeHuman base mesh, fitted onto the body. The wrists, ankles
   and neck are sized to ANSUR II men's averages.
3. **Rigging.** Skin weights come from heat diffusion, then forearm and thigh twist and the wrist get dedicated
   weight ramps. Optimized centres of rotation stop joints from collapsing ("candy-wrapper" twists).
4. **Live deformation (in the page).** CPU skinning, delta-mush smoothing, soft collision against the bones,
   muscle-layer ordering (muscles can't pass through each other), fold relaxation and smoothed normals.
5. **Skin map.** The skin is cut along seams a tattoo artist would recognise (sides of the torso, inner arms and
   legs, head front/back, hands and feet top/bottom, each finger and toe) into 56 regions. Each region is
   flattened with SLIM, which keeps lengths and areas as close to true as it can, and all regions are laid out at
   one scale: 1 cm on the map ≈ 1 cm of skin.
6. **Rendering.** three.js `MeshPhysicalMaterial` with shader injections: a subsurface-scattering approximation,
   a repeating real-scale micro-detail tile (pores and skin furrows from Worley noise), regional skin-colour
   variation (knuckles, elbows, knees, palms, soles), ACES tone mapping and soft shadows.

---

## Performance

All numbers were measured on this project's own data with the test probes described below.

### Build pipeline (`tools/fit_rig.py`)

Profiled with `python -m cProfile`. Nearly all the time went into two patterns: brute-force all-pairs
nearest-point searches, and `np.add.at` scatter-adds inside smoothing loops.

| Step | Before | After | Fix |
|---|---|---|---|
| Nearest-point searches (`nearest_index`, `knn`, `nearest`) | 162 s | ~2 s (building the trees) | KD-tree (`scipy.spatial.cKDTree`) |
| Scatter-adds (`np.add.at`) | 64 s | ~3 s (bincount + sparse products) | `np.bincount`, and a cached sparse neighbour matrix for repeated smoothing (`tools/fastgeo.py`) |
| Bone distance grids (`edt2`, whole section) | 8 s | 2 s | `scipy.ndimage.distance_transform_edt` (exact, same values) |
| Volume blur (up to 44 passes) | 7 s | ~5× faster in a benchmark | repeated 1-2-1 averages replaced by one binomial-kernel pass (identical result) |
| **Whole build** | **287 s (4 min 47 s)** | **~43 s** | **≈6.7× faster** |

The output is essentially unchanged: the new skin surface is a mean of 0.003 mm from the old one. A few spots
differ by up to ~3 mm, where the mesh simplifier breaks ties differently. One real fix came with the speed-up:
the old neighbour search returned neighbours in random order, so "the nearest point of another muscle" was really
a random one of the 8 nearest. It's now the actual nearest, and 6% more muscle-layer pairs are found. Muscle
colours are also the same on every build now: they were seeded with Python's `hash()`, which is randomised
on every run, and now use `zlib.crc32`, which isn't.

> **Status:** the faster build code is in `tools/`. The 43 s figure is from test builds; the checked-in
> `body-rig.js` was made by the previous (slower) build and hasn't been regenerated with it yet.

### Skin map (`tools/uv_atlas.py`)

| | |
|---|---|
| Full skin-map build | **31 s** |
| Regions / seam copies | 56 charts, 3,228 seam points |
| Area accuracy, limbs (5th–95th percentile) | 0.90–1.10 (arms 0.93–1.09) |
| Area accuracy, torso | 0.91–1.16 |
| Area accuracy, head (hardest to flatten) | 0.78–1.28 |
| Folded triangles (flips) | 0 |

One earlier run took ~5½ minutes. Profiling showed that was only CPU contention from other jobs running at the
same time: on its own, SLIM flattens the right leg in about 1 s.

### Live posing (in the browser)

| | Before | After |
|---|---|---|
| Pose update, skin only | 38–44 ms | **12–28 ms** |
| Accuracy vs. full recompute | — | within 0.26 mm and 0.55° (normals) |
| Page load / ready | | ~2.2 s / ~3.2 s |

The speed-up comes from processing only the skin that is changing: each update finds the region near moving
joints, collisions and muscles, and runs the smoothing passes only there plus a safety margin of rings.

### Motion smoothness

Measured as frame-to-frame "lurch" (the second difference of each pass's correction over a slow motion):

| Motion | Before | After |
|---|---|---|
| Hip | 9.1 mm | 0.6 mm |
| Shoulder | 4.9 mm | 0.9 mm |
| Reach (arm overhead) | 84 mm sudden flick | 6.7 mm, smooth |
| Spine twist | | 3.6 mm (still open, see below) |

Fixes behind these: soft (quadratic) bone contact instead of a hard push, no dead zone in the muscle-order
constraint, a continuous fold-relax amount in place of an on/off untangle step, and smoothing spread in proportion
to how far skin was pushed.

**Wrist and fingers** were checked in 16 wrist positions, each from several camera angles. Finger thickness kept
through a bend went from 0.22 → 0.91 of rest (deviated wrist) and 0.40 → 0.87 (fist), after the hand weights
were stopped from leaking 4–9 cm up the forearm.

### File sizes

| File | Size | Contents |
|---|---|---|
| `body-rig.js` | 20 MB | skin, fat, muscles, skeleton, weights, collision data (packed: byte-shuffle + gzip, deduplicated arrays, unpacked in the browser with `DecompressionStream`) |
| `skin-uv.js` | 1.46 MB | skin map triangles, seams and UVs |
| `skin-detail.js` | 1.8 MB | 25 mm skin micro-detail tile (normal + roughness) |
| `skin-map.js` / `skin-map.png` | 0.8 / 1.8 MB | skin-tone grid texture / printable 1 cm template |

Skin mesh: 80,440 points, 160,876 triangles.

---

## Testing

Visual and numeric checks run headlessly in Chrome. A small local server receives screenshots and JSON from probe
scripts injected into a test copy of the page. The probes:

- pose the wrist in 16 positions × 4 views and save photo sheets;
- play motions (elbow, shoulder, reach, wrist, hip, spine, fist, hands-behind) and measure frame-to-frame skin lurch;
- compare region-only updates against full recomputes (position and normal error, time);
- time each step of a pose update;
- take skin close-ups with and without micro-detail, and before/after body builds.

---

## Project layout

| Path | What it is |
|---|---|
| `index.html` | the whole app: rendering, posing, live skin deformation |
| `tools/fit_rig.py` | main build: anatomy → rigged skin, fat, muscles, skeleton → `body-rig.js` |
| `tools/fastgeo.py` | fast nearest-point search and scatter-add helpers used by the build |
| `tools/envelope.py` | voxel solid, closing, blur and surface extraction for the skin and fat |
| `tools/mh_parts.py` | fits MakeHuman hands, feet and head onto the body |
| `tools/uv_atlas.py` | flat skin map (UV atlas) → `skin-uv.js`, `skin-map.png`, `skin-map.js` |
| `tools/skin_detail.py` | skin micro-detail tile → `skin-detail.js` |
| `tools/rigpack.py`, `tools/pack_rig.py` | compact data format (pack / unpack / repack) |
| `tools/tissue_depth.py` | facial soft-tissue depths from the C-Table |
| `tools/convert_fbx.py`, `tools/meshes.py` | convert and read the Z-Anatomy source meshes |
| `docs/how-it-works.html` | teaching guide and Python course |
| `anatomy-src/`, `assets/` | source data (see Credits) |

---

## Known issues / next steps

- A small notch on the little-finger side of the wrist when it bends back (where the MakeHuman hand joins the arm).
- A 3.6 mm pop during spine twist, from bone contact.
- Showing the muscle layer is slow (~99 ms per update, from muscle-layer collisions).
- Weight jumps between twist knots on the thigh and forearm.
- Possible: a body-definition slider.

---

## Credits and licences

- **Anatomy:** Z-Anatomy, built on BodyParts3D © DBCLS, CC BY-SA 4.0.
- **Base body:** "Male Base Mesh" by Orange Juice Games / BoQsc, CC0.
- **Hands, feet and face:** MakeHuman base mesh, rig and weights, CC0 (see `assets/makehuman/SOURCE.md`).
- **Hands (rig reference):** WebXR Input Profiles © W3C, W3C Software License.
- **Body sizes:** ANSUR II public data (Gordon et al. 2014), used as a size guide (see `assets/ansur2/SOURCE.md`).
- **Facial tissue depths:** computed from the C-Table (craniofacialidentification.com); see `tools/tissue_depth.py`
  for the references its terms of use ask for.
- **Skin micro-detail:** generated by `tools/skin_detail.py` (no third-party texture).
- **Rendering:** three.js r160.
