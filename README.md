# Tattoo Planner

A browser-based 3D tattoo planner. It shows a realistic, posable human body built from real anatomy, with a
true-scale flat map of the skin, so flat tattoo designs can be placed on the body at their real size.

A Python pipeline takes a real skeleton and muscles (Z-Anatomy / BodyParts3D) and grows fat and skin over them. It
then rigs the body so it bends naturally and flattens the skin into a 2D map. The web app (three.js) renders the
result and lets you pose it live.

📖 **[How it works](docs/how-it-works.html)**: a plain-language guide, with a 15-lesson Python course you can run
in the page. Download it, or open it on the live site, to view it; GitHub only shows the code.

---

## Quick start

Open `index.html` in Chrome (needs internet for the three.js CDN), or serve the folder:

```sh
python3 -m http.server 8000   # then open http://localhost:8000
```

| Control | Action |
|---|---|
| Drag on the body | move that part |
| Right-drag on the body | rotate it |
| Drag off the body | orbit (right-drag pans, scroll zooms) |
| Buttons | poses, layers (skin, fat, muscle, skeleton, skin map) |
| Flex slider | muscle tension |

---

## Rebuilding the body

```sh
python3 -m venv .venv && .venv/bin/pip install numpy scipy libigl pillow
.venv/bin/python -I tools/fit_rig.py anatomy-src assets/male_base_mesh.glb .   # → body-rig.js
.venv/bin/python -I tools/uv_atlas.py .                                        # → skin-uv.js, skin-map.*
.venv/bin/python -I tools/skin_detail.py .                                     # → skin-detail.js
```

> **Note:** Run `uv_atlas.py` after every `fit_rig.py`. The `anatomy-src/` folder (Z-Anatomy, ~1.2 GB) is not in
> this repo; you'll probably have to download it online beforehand.

---

## How it works

- **Anatomy → rig:** rig joints are fitted onto a real skeleton, and each muscle is weighted to the joints it spans.
- **Fat and skin:** a smooth surface is built from voxelized muscle and bone. Fat thickness comes from skinfold
  studies and is scaled to 17% body fat (Jackson–Pollock + Siri). Hands, feet and face come from MakeHuman, and
  joints are sized to ANSUR II averages.
- **Deformation:** heat diffusion skin weights, optimized centres of rotation, delta mush, soft bone collision,
  muscle layer ordering and fold relaxation.
- **Skin map:** 56 regions flattened with SLIM at one scale (1 cm on the map ≈ 1 cm of skin).
- **Rendering:** physically based skin with a subsurface-scattering approximation, procedural pores and furrows,
  and regional color variation.

---

## Performance

| Metric | Before | After |
|---|---|---|
| Body build (`fit_rig.py`)\* | 287 s | **~43 s** (6.7× faster) |
| Skin map build (`uv_atlas.py`) | — | **31 s** |
| Live pose update (skin) | 38–44 ms | **12–28 ms** |
| Skin lurch, reach motion | 84 mm | **6.7 mm** |
| Skin lurch, hip / shoulder | 9.1 / 4.9 mm | **0.6 / 0.9 mm** |
| Page ready | — | **~3 s** |

- **Build speed-up:** found with `cProfile`. KD-trees replaced brute-force nearest-point search (162 s → ~2 s).
  `np.bincount` and cached sparse matrices replaced `np.add.at` (64 s → ~3 s). SciPy's distance transform and a
  single-pass binomial blur replaced hand-written loops. The output is unchanged: the new skin is a mean of
  0.003 mm from the old one.
- **Live posing:** recomputes only the skin that is changing, within 0.26 mm of a full recompute.
- **Skin map accuracy:** area within 0.90–1.10 on the limbs and 0.91–1.16 on the torso, with 0 folded triangles.
- **Mesh:** 80,440 points and 160,876 triangles, shipped as ~20 MB of packed data (byte-shuffle + gzip, unpacked
  in the browser).

<sub>\* Measured on test builds. The checked-in <code>body-rig.js</code> has not been regenerated with the faster
pipeline yet.</sub>

---

## Testing

Probe scripts drive the app in headless Chrome. They pose the body (e.g. the wrist in 16 positions × 4 views),
capture screenshots, and measure frame-to-frame skin jitter and update timing.

---

## Project layout

| Path | Purpose |
|---|---|
| `index.html` | the app: rendering, posing, live deformation |
| `tools/fit_rig.py` | main build: anatomy → rigged body |
| `tools/uv_atlas.py` | skin map |
| `tools/skin_detail.py` | skin micro-detail texture |
| `tools/fastgeo.py` | fast nearest-point search and scatter-add |
| `tools/envelope.py` | voxel surface building |
| `tools/mh_parts.py` | MakeHuman hand, foot and head fitting |
| `tools/rigpack.py` | data packing |
| `docs/how-it-works.html` | guide and Python course |

---

## Known issues

- Small notch at the wrist when it bends back
- 3.6 mm skin pop during spine twist
- Muscle layer view is slow (~99 ms per update)

---

## Credits

| Part | Source | Licence |
|---|---|---|
| Anatomy | Z-Anatomy / BodyParts3D © DBCLS | CC BY-SA 4.0 |
| Base body | "Male Base Mesh" by Orange Juice Games | CC0 |
| Hands, feet, face | MakeHuman | CC0 |
| Hand reference | WebXR Input Profiles © W3C | W3C Software License |
| Body sizes | ANSUR II public data | Public |
| Facial tissue depths | C-Table (craniofacialidentification.com) | See `tools/tissue_depth.py` |
| Rendering | three.js | MIT |
