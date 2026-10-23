# TuftView dashboard

A local web app around the existing detector, tracker and CSV code. Nothing in `src/`, `core/`,
`tools/` or `app/` is changed; the dashboard only calls them.

## Start it

| Where | How |
|---|---|
| Mac | double-click `Start-TuftView.command` (first run installs the packages) |
| Windows | double-click `Start-TuftView.bat` |
| Any | `python -m pip install -r Requirements.txt -r Requirements-dashboard.txt`, then `python -m dashboard` |

It opens http://127.0.0.1:8765 in the browser. `--host 0.0.0.0` lets other computers on the
network open it; `--port`, `--data DIR` and `--no-browser` are also available.

## What the customer does

1. Drag a video or pictures onto the page, or press the **+** button. (**Try a sample clip** makes a made-up wind tunnel video.)
2. Wait for the progress ring. The picture shows what is being found live.
3. **Video** tab: watch the result. Click any tuft to see its position, direction, steadiness and a
   direction-over-time chart. Type your own X, Y, Z for it if you measured it.
4. **Data** tab: one line per tuft (sort, search, filter by flow), or the full time history. Click a
   grey number to type your own. **Download** gives CSV, Excel, the annotated video, or everything as one ZIP.
5. **Setup** tab: define the coordinate system, import positions, change detection settings.

## Requirement map

| # | Requirement | Where |
|---|---|---|
| 1 | MP4 in standard resolutions | `dashboard/analysis.py` (any size; settings scale with the picture) |
| 2 | User-defined coordinate system | `coordinates.py`, Setup tab (2 points = scale+rotation, 3 = affine, 4+ = perspective) |
| 3 | Automatic tuft location | existing detector + tracker (`core.Pipeline`); position = median root over the video, in your axes |
| 4 | Optional user (X, Y, Z) per tuft | `coordinates.TuftLocations`; click a tuft, edit a cell in Data, or import `tuft_id,x,y,z` CSV |
| 5 | Any background colour | tuft colour (light/dark) picked automatically from sample frames, or set by hand in Setup |
| 6 | Video with flow vectors | `annotated.mp4` (H.264), plus a live, clickable overlay in the Video tab |
| 7 | Vector time history | `*_vector_time_history.csv`: `time_s, frame, tuft_id, x, y, z, vector_x, vector_y, vector_z, angle_deg, flow_state, length, confidence` + pixel columns |

Vectors are unit vectors in the user's axes. Z of the vector is 0 (flat on the surface) unless
**Estimate Z direction from tuft shortening** is switched on; that estimate (a tuft that lifts off the
surface looks shorter) is experimental and clearly labelled.

## Extras beyond the requirements

Flow state per tuft (steady / fluttering / stalled, from the existing stall analysis), flow-health
timeline, direction-over-time chart, world grid overlay, coordinate axes in the exported video,
area-of-interest picker, Excel workbook, ZIP of everything, saved history of analyses.

## Detector choice

Automatic. If a U-Net checkpoint exists (`models/tuft_unet.pt`, or the newest
`runs/*/checkpoints/best.pt`, or `$TUFT_WEIGHTS`) and PyTorch is installed, the U-Net is used;
otherwise the classical detector with the README's settings scaled to the picture size. Pick
manually under Setup > Analysis settings.

## Tests

```
python -m unittest tests.test_coordinates tests.test_dashboard
```
