# U-Net work review

Reviewed: `origin/main` at `f7ad46b` ("fixed unet issues, before training starts"), plus the
two commits before it (`8330ffe`, `1685cac`), all by Aryan Desai, 2–4 Oct 2026. The U-Net
work is on `main`. The only other remote branch (`feat-added-tufting-image-processing`) is
older and was already merged.

Reviewed on 2026-10-12 on the RTX 5090 workstation. No teammate code has been changed yet.

## 0. Before anything else: the pull conflicts with your local work

Your local `main` is 3 commits behind `origin/main` and has **uncommitted** changes to
`detector.py`, `main.py` and `Commands.txt`:

- light/dark `polarity` in the adaptive threshold
- `tape_root`: yellow anchor tape (HSV) picks the root end and rejects unanchored strays
- `_one_per_anchor`: one tuft kept per tape patch
- car-image commands

Commit `1685cac` replaces `detector.py` with a thin wrapper around a new
`src/mask_to_tufts.py`, so `git pull` would refuse or conflict. I did **not** pull into your
checkout. This review was done in a separate worktree (`../tvds-origin-main`) on a new local
branch, `unet-training`.

Polarity already exists in the teammate's `ClassicalBackend` (it inverts gray instead of
negating C, which is equivalent). **Tape-root anchoring does not exist there.** It fits the
new design cleanly: the tape mask can be passed as the `root_mask` that
`extract_tufts()`/`orient_path()` already accept. → Question 1 at the end.

## 1. What exists

| Piece | Where | Summary |
|---|---|---|
| Model | `src/unet.py` | Hand-written U-Net, from scratch (no pretrained encoder), base 16, depth 4, **1.94 M params**, **2 output channels** (tuft, root), raw logits |
| Dataset / loader | `src/dataset.py`, `src/labeling.py` | 3-channel label PNG (tuft, root, root-valid). Train: `epoch_size` random crops, 60 % centred on a tuft pixel, deterministic per (seed, epoch, index). Val: fixed non-overlapping tiles |
| Augmentation | `src/augment.py` | Hand-written OpenCV (not albumentations). Flips, ±180° rotation, scale 0.75–1.3; hue, channel permute, grayscale, invert, brightness/contrast, gamma, linear illumination ramp, Gaussian/motion blur, noise, JPEG |
| Loss | `tools/train_unet.py` | BCEWithLogits (pos_weight 2) + soft Dice on tuft; same on root channel, masked by root-valid, ×0.5 |
| Training loop | `tools/train_unet.py` | AdamW, cosine LR, grad-clip 5, fp32, best (by val Dice) + last checkpoint, CSV log, threshold sweep (0.3–0.7, picked on F0.5) saved to `<ckpt>.json` |
| Evaluation | `src/metrics.py`, `tools/eval_detection.py` | Pixel TP/FP/FN meter. Object P/R by mask overlap on the val split |
| Inference | `src/unet.py: UNetBackend` | Full image, or 768-px tiles with 64-px overlap (averaged), replicate-padded to a multiple of 16, one sigmoid |
| Integration | `src/backends.py`, `detector.py`, `main.py` | `TuftDetector(backend=...)`; backends return `MaskResult(tuft, root)`; `main.py --backend {classical,unet} --weights`. Geometry lives in `src/mask_to_tufts.py` |
| Labelling | `src/sam2_labeler.py`, `tools/autolabel.py` | Classical proposals → SAM 2.1 **image** predictor (hiera-small) prompted with 4 centre-line points + 2 normal negatives + box, in 512-px tiles. Strict accept/reject by IoU/cover/area |
| Review | `src/review.py`, `tools/review_labels.py` | OpenCV window: click = reject tuft, shift-click = flip root/tip, right-drag = reject box, undo, save |
| Synthetic | `src/synth.py`, `tools/make_synthetic_dataset.py` | Flat-colour backgrounds, regular grid of tapered polylines with optional hook, one bulk flow angle per image |
| Post-filters | `src/filtering.py` | `normalize_scale` (downscale to 1400 px long side), tuft-field cluster filter, contrast filter |
| Tests | `tests/` | 46 tests, **all pass** on this machine |

## 2. Correctness: the specific checks

| Check | Result | Evidence |
|---|---|---|
| Mask/image misalignment after augmentation | ✅ **OK** | `random_geometric` applies the same flips and the same affine matrix to both (`INTER_NEAREST` for the label). Photometric ops touch only the image. Minor: with scale 0.75 plus ~45° rotation, the 1.5× pre-crop is not large enough, so mirrored border content shows in corners. Image and label are still consistent there, so there is no misalignment, just a few fake mirrored tufts. |
| Wrong encoder normalisation | ✅ OK today / ❌ **will be wrong after the switch** | Inputs are RGB in [0,1] with no mean/std. That is fine for a from-scratch net, but an ImageNet-pretrained resnet34 needs ImageNet mean/std. This must change together with the model, in both training and `UNetBackend`. |
| Double sigmoid | ✅ **OK** | Model returns logits. Loss uses `binary_cross_entropy_with_logits`. Dice and inference each apply exactly one sigmoid. |
| Train/val leakage | ❌ **Leaks for video** | `split_of()` hashes the source-image name (part before `__`). Tiles of one photo stay together, but every **frame** of a video has its own name, so frames from the same clip land in both train and val. There is no grouping by video/session. |
| Test set | ❌ **None** | Only train/val exist. `eval_detection.py` scores the same val split that picked the checkpoint and the threshold, so its numbers are optimistic. |
| Downscaling so thin tufts disappear | ⚠️ **Partly** | Training does use native-resolution crops. However, `autolabel.py` and `main.py` first downscale every image to 1400 px on the long side (`normalize_scale`). 720p is untouched; **1080p is downscaled ×0.73**; 4K photos ×0.35. The reason was to keep tufts inside the hard-coded `max_length_px=110` filter. The agreed design says never downscale, which means length limits must become per-setup config instead. |
| Hard-coded paths | ✅ OK in code / ⚠️ docs | No absolute paths in `.py` files. `Commands.txt` has `cd "/Users/aryandesai/..."`, `--device mps`, `sips` and `open`, which are all macOS-only. |
| Seeds | ✅ **OK** | `torch.manual_seed`, `np.random.seed`, per-sample `default_rng([seed, epoch, i])`. cuDNN is not set deterministic; acceptable. |
| Checkpoints | ✅ **OK** | best (`out`), last (`out.last.pt`), and JSON with args, split and metrics. No resume. |
| Validation metric | ✅ **OK** | Pixel Dice/IoU/P/R, accumulated globally (not averaged per tile, which is the right choice), plus a threshold sweep. |
| AMP / early stopping / YAML / sample images | ❌ Missing | fp32 only, fixed epoch count, CLI flags only, no prediction previews. |

Other issues found:

1. **Flow vector default is the chord, not the tip tangent.** `TuftDetector(vector_mode="chord")`
   and `main.py --vector-mode chord` are the defaults. The agreed pipeline uses the tip
   tangent because tufts curl. `"tip"` mode exists (`path[-1] - path[-1-k]` on the smoothed
   48-point centre-line) but is not the default. Easy fix.
2. **SAM 2 is used as a detector-side filter.** `main.py --verify-sam2` keeps only tufts that
   SAM 2 confirms. This conflicts with "SAM 2 only builds training data". It is off by
   default, but I'd remove it from `main.py`.
3. **Tuft-field cluster filter on by default** (`clean_tufts` in `main.py`). This is a
   heuristic that sits outside the agreed pipeline. It is reasonable for cluttered web photos,
   but on a fixed camera it can hide U-Net false positives or drop isolated real tufts.
   `evaluate.py` should report with and without it.
4. **Review tool silently drops labels on save.** `ReviewSession` re-extracts tufts from the
   saved mask using `LABEL_EXTRACT_KW` (max length 260 px, max area 2500), then re-rasterises
   only those contours. Any real tuft outside those limits (likely at native 1080p) is erased
   from the label when you press *save*. It also has no brush, no frame accept/reject and no
   edit log. `Commands.txt` already notes that the tool cannot add missed tufts.
5. **Autolabel drops hard negatives.** `autolabel.py` saves only 1000-px tiles that contain at
   least ~45 tuft pixels, so background-only regions (panel lines, seams) never reach
   training except via in-photo crops. It is also image-only: no video frames, no SAM 2
   video propagation, no per-frame JSON.
6. **Synthetic data is not realistic enough to help much.** It uses flat-colour backgrounds, a
   regular grid, one flow angle per image, no stall/flutter/reversed shapes, no hard
   negatives and no videos. Its `flow_deg` truth is the bulk angle, **not** the true tip
   tangent of hooked tufts, so it cannot be used to score tip-angle error.
7. **Inference speed.** At 720p (1280 px), `UNetBackend` tiles at 768 px, runs each tile
   separately in fp32 and copies back to CPU per tile. On a 32 GB 5090 the whole padded
   frame fits in one fp16 pass. Not measured yet; FPS will come from `evaluate.py`.
8. **Requirements**: everything is `>=` (unpinned). `torch` is in the *base*
   `Requirements.txt`, so the classical path needs torch. `segmentation_models_pytorch` and
   `albumentations` are missing.
9. **Repo hygiene** (not code bugs, but they affect the "no large data in git" rule):
   - `venv/`, a macOS Python 3.14 venv with 490 files, is tracked; it was committed before the
     `.gitignore` entry existed.
   - Two `.DS_Store` files are tracked.
   - `output/` results and 7.4 MB of `Images/` are tracked.
   - `Images/tufts3.jpg` and `tufts6.jpg` are AVIF files with a `.jpg` extension; OpenCV on
     Windows cannot read them.
   - `.env` is tracked; it contains only a comment, so there is no secret.

## 3. Does it run on the RTX 5090?

**Yes, with a correct torch build.** Verified on this machine:

```
torch 2.11.0+cu128, CUDA 12.8, cuda.is_available() = True
NVIDIA GeForce RTX 5090, capability (12, 0), arch list includes sm_120
```

- Teammate test suite: 46/46 pass.
- `train_unet.py`, 3 epochs × 160 crops on 20 of their synthetic images, on `cuda`:
  loss 2.196 → 1.974 → 1.845, best and last checkpoints plus JSON and CSV written.
- `main.py --backend unet` loads that checkpoint and runs end to end.

This was a plumbing check only. The run is far too short to say anything about accuracy, and
no accuracy figures are claimed here.

Note: the existing `.venv` has no torch at all. `Requirements-train.txt` says
`torch>=2.2`, and a plain `pip install torch` on Windows gives a CPU build, while older CUDA
builds lack sm_120. The requirements must point at the cu128 index. The new venv is
`.venv-unet` (Python 3.11).

## 4. Verdict per piece

| Piece | Verdict | Reason |
|---|---|---|
| `src/unet.py` model | **Rewrite** | Replace with `smp.Unet(encoder="resnet34", encoder_weights="imagenet", classes=1)`, encoder configurable. A 1.9 M-param from-scratch net throws away the pretrained features that make the method robust to shadow and glare. Keep `save_checkpoint`/`load_model` shape but store the encoder name. |
| `UNetBackend` (inference) | **Fix** | Good structure (tiling, overlap averaging, threshold from checkpoint JSON). Changes: ImageNet normalisation, pad to a multiple of 32, fp16 autocast, whole-frame pass when it fits, batched tiles otherwise. |
| `src/backends.py` + `detector.py` backend abstraction | **Keep** | This is already the `--backend {classical,unet}` seam the plan asks for, and the geometry is shared. Change `vector_mode` default to `"tip"`, and port your tape-root anchoring in (question 1). |
| `src/mask_to_tufts.py` (geometry) | **Keep, small fix** | Skeleton → double-sweep longest path → thickness/root-prob orientation → smoothed centre-line is sound and tested. Fix: tip-tangent default; length/area limits from config rather than tuned for 1400-px images. |
| `src/dataset.py` crop sampler | **Fix** | The tuft-biased crop logic is good and deterministic; keep it. Changes: 512 crops, albumentations, 1-channel target, split from a sources/sessions manifest instead of a name hash, real-vs-synthetic mixing ratio, empty-mask hard-negative samples. |
| `src/labeling.py` split (`split_of`) | **Rewrite** | Name hashing leaks frames across splits and has no test split. Replace with explicit train/val/test by clip/session from `sources.csv`. |
| `src/augment.py` | **Rewrite** (albumentations) | Alignment is correct, but it lacks shadows, glare, elastic and video compression. Invert, hue and channel-permute are odd for white-on-backing tufts. The plan specifies albumentations. |
| Loss | **Keep + extend** | BCE + Dice is correct. Add optional soft-clDice. Drop the root term if we go to 1 channel (question 2). |
| `tools/train_unet.py` | **Fix / extend** (substantial) | Seeds, checkpoints, val metric and threshold sweep are good. Add YAML config, AMP, early stopping, sample prediction images, TensorBoard/CSV, smp model, test split never touched. |
| `src/metrics.py` `SegMeter` | **Keep** | Correct. Add clDice. |
| `tools/eval_detection.py` | **Rewrite** as `evaluate.py` | Needs Hungarian root matching, tip-angle error (circular), flip rate, IoU/clDice, FPS, per-condition/source breakdown, and the frozen test set. |
| `src/sam2_labeler.py` | **Keep core, extend** | Prompt building (centre-line points + normal negatives + box) and `accept_mask`/best-of-multimask are good ideas. Add the video predictor for propagation; work on native-resolution frames. |
| `tools/autolabel.py` | **Rewrite** for video frames | Image-only, downscales, drops background tiles, no per-frame JSON. |
| `src/review.py` + `tools/review_labels.py` | **Rewrite** | Silent label loss on save (issue 4), no brush, no frame accept/reject, no edit log. |
| `src/synth.py` + `make_synthetic_dataset.py` | **Rewrite** as `synth_tufts.py` | See issue 6. |
| `main.py --verify-sam2` | **Remove** | SAM 2 must not be part of detection. |
| `src/filtering.py` `normalize_scale` | **Remove from the pipeline** | Never downscale; replace with configurable px limits. |
| `src/filtering.py` field/contrast filter | **Keep, optional** | Off by default on the fixed-camera path; evaluate both. |
| Tests | **Keep** | Update as code changes. |
| Requirements | **Fix** | Pin everything; separate base vs train; cu128 index. |
| Repo hygiene | **Fix** | Untrack `venv/`, `.DS_Store`, `output/`; add `data/`, `models/`, `*.mp4` to `.gitignore`. Untracking `Images/` is your call. |

## 5. Small fixes made during the review

None so far. Every issue above is either a design change or touches code you asked to see
reviewed first.

## 6. Questions before Step 2

1. **Your uncommitted `detector.py` work (tape-root, one-per-anchor).** Recommended: I commit
   it to a side branch so it's safe, then port tape anchoring into the new architecture on
   `unet-training` as a root-mask source plus stray filter. Your local `main` stays as is
   until you decide.
2. **Root channel.** The plan says 1 output channel. The teammate's net also predicts the taped
   root end, which helps root/tip orientation where thickness is ambiguous. Recommended:
   1 channel by default (per plan), with `root_head: true` as an option in the YAML that we
   only enable if `evaluate.py` shows it lowers the flip rate.
3. **Tuft-field filter.** Off by default for the fixed-camera/video path. OK?
4. **`Images/` and `output/` in git.** Untrack them too (files stay on disk), or keep the
   sample images tracked as small test fixtures?

## 7. Addendum (2026-10-12): found while building on this code

Decisions taken (from the discussion): one output channel by default, root
channel optional; tuft-field filter off by default; native resolution only;
tape-root anchoring ported from `aditya-wip`; real data first, synthetic off.

Conflicts between `aditya-wip` and Aryan's rewrite, and how they were resolved:

| Topic | aditya-wip | Aryan's rewrite | Now |
|---|---|---|---|
| Light polarity threshold | negate C on the CLAHE image | invert gray, then threshold | aditya-wip method; the two differ by 5-7 % mask IoU on the car photos, and the car settings were tuned on it |
| Tape-root anchoring | yes | missing | ported as a root source for the shared geometry |
| Flow vector | tip tangent | chord by default | tip tangent by default (agreed design); Aryan's test measures against the bulk flow angle, so it now asks for chord explicitly |
| Skeleton / path | Zhang-Suen + BFS between endpoints | skimage skeleton + double-sweep longest path, smoothed | Aryan's (tested); car photos give the same tuft counts (20, 12), no flips, 2-4 deg median angle difference |

Bugs found in the existing code (left in place where Aryan's tests use them;
the new pipeline does not use them):

1. `src/sam2_labeler.py` fixed-grid tiling only refines tufts whose bounding
   box fits inside one 512 px tile with 128 px overlap. At native resolution
   half the car-photo tufts (about 215 px long) never reached SAM and were
   marked "rejected". Replaced by `src/sam2_autolabel.py` (one window per tuft).
2. `src/sam2_labeler.py` negative prompts sit a fixed 6 px off the centre-line;
   now scaled to the measured yarn width.
3. `src/dataset.py: crop_or_pad` reflect-pads the image but zero-pads the label,
   so mirrored tufts would be labelled background. Latent with Aryan's crop
   sizes; the new dataset pads image and label together.
4. `tools/review_labels.py` (old) re-rasterised filtered contours on save and
   dropped tufts outside the size limits. Rewritten; masks are saved pixel-exactly.
5. `polarity=auto` picks the polarity with more candidates; on a tie the dark
   polarity traces wide halos around light strings. Auto-labelling now prefers
   the thinner candidates on a near tie.

Legacy code kept for Aryan's 46 tests (still passing): `UNet` (from-scratch),
`src/augment.py`, `src/dataset.py`, `split_samples`, `tools/train_unet.py`,
`tools/make_synthetic_dataset.py`, `tools/eval_detection.py`,
`src/sam2_labeler.py`. Removing them is a separate decision.

Test changes (assertions that encoded the old defaults): `CliTests.test_defaults`
(no downscaling, field filter off, tip-tangent default); the large-image
end-to-end test passes size limits that match its tufts; the bulk-flow test
uses chord mode explicitly; sample paths moved to `tests/fixtures/`.
