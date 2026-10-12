# Hard negatives to capture with our own footage

Things that look like tufts to a model but are not. Each one should appear in
training as **real frames with an empty mask** (no tufts), so the model learns
they are background. Capture them on the same surface, lighting and camera as
the tuft runs.

How to add them:

```powershell
python -m tools.sources add-local D:\footage\bare_panel_glare.mp4 --split train --use negative --notes "door trim + handle, sun"
python -m tools.extract_frames --use negative --total 80
python -m tools.auto_label            # negative clips become empty-mask proposals
python -m tools.review_labels         # accept (or reject) each one
```

Keep negatives in the same split rules as everything else: whole clips go to
one split, and some go to val/test so we can measure false positives.

## Found so far

Source: run A (trained on DLR red tufts only) predicted on our two car photos
(`runs/car_check_A.jpg`), 2026-10-12. These are the 23 predicted regions
outside the tufted area. Coordinates are x, y, w, h in the original photo.

| # | Category | Example (photo, bbox) | Why it fools the model |
|---|---|---|---|
| 1 | Chrome door handle, specular highlight | Static 2427,568,72,15; Static 0,633,150,17; Moving 3094,364,47,11; Moving 3135,378,78,11 | long, thin, bright streak: tuft-shaped |
| 2 | Body crease / panel-line highlight | Static 1747,1723,105,11; Static 1648,1735,139,11 | thin bright line along a crease |
| 3 | Lower door edge, sill / rocker trim edge | Static 1694,2050,70,13; Static 2283,2071,73,6 | dark-light edge, long and thin |
| 4 | Chrome window trim with water droplets | Moving 183,2,100,18 (and 6 more along the top edge) | bright, irregular, string-like blobs |
| 5 | Chrome sill trim with reflections | Moving 0,489,33,6; Moving 36,491,29,6 | thin bright segments |
| 6 | Reflections of reddish objects (leaves) on paint, along dried water-drip trails | Moving 3162,1039,24,11; Moving 1150,1339,33,39; Moving 1088,1364,27,7; Moving 1127,1366,21,5; Moving 2136,1521,32,6 | colour bias from red DLR tufts + streak shape |
| 7 | Blurred orange/red reflection blobs | Moving 1653,1766,23,17; Moving 1615,1774,41,9 | colour bias from red DLR tufts |

## Still to capture (from the project brief)

Panel lines and seams, rivet rows, cables, painted stripes and stencilled text,
shadow edges (including the shadows of the tufts themselves), wing / flap
edges, tape edges without a tuft, glare and sun spots, water droplets and
drip trails, and reflections of people or objects in glossy paint.

Wanted per category: footage of the bare surface (no tufts) under the lighting
conditions we fly or test in, from the same camera position.
