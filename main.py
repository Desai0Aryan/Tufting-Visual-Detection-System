import argparse
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from detector import TuftDetector, draw_tufts
from src.backends import build_backend
from src.export import tufts_dataframe
from src.filtering import DEFAULT_NORM_SIZE, clean_tufts, normalize_scale, rescale_tufts, scale_roi
from src.mask_to_tufts import VECTOR_MODES
from src.image_io import (
    ImageLoadError,
    annotate_info_banner,
    get_image_info,
    load_image,
    resize_max_dim,
    save_image,
)


def parse_roi(value: str) -> tuple[int, int, int, int]:
    """Parse an ROI in x,y,w,h form and reject malformed/negative values."""
    try:
        parts = [int(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--roi must be x,y,w,h (e.g. 100,120,400,250)"
        ) from exc

    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--roi must be x,y,w,h (e.g. 100,120,400,250)"
        )

    x, y, w, h = parts
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(
            "--roi requires x,y >= 0 and width/height > 0"
        )
    return x, y, w, h


def _validate_roi(roi, image_shape):
    """Validate that an ROI is completely inside the input image."""
    if roi is None:
        return None

    x, y, w, h = roi
    height, width = image_shape[:2]
    if x + w > width or y + h > height:
        raise ValueError(
            f"ROI {roi} is outside the image bounds ({width}x{height})"
        )
    return roi


def _clamp_roi(roi, shape):
    """Rounding when an ROI is rescaled can overshoot the image by a pixel."""
    if roi is None:
        return None
    x, y, w, h = roi
    H, W = shape[:2]
    x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
    return x, y, max(1, min(w, W - x)), max(1, min(h, H - y))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect tufts and flow vectors in a single image."
    )
    # Keep the old positional form working while supporting the image-processing
    # branch's --image interface.
    parser.add_argument("image_positional", nargs="?", help="Path to input image")
    parser.add_argument("--image", dest="image_option", help="Path to input image")
    parser.add_argument(
        "--outdir", "--out-dir", dest="outdir", default="outputs",
        help="Directory for generated images and CSV data",
    )
    parser.add_argument("--roi", type=parse_roi, default=None,
                        help="Restrict detection to x,y,w,h (pixels)")
    parser.add_argument("--select-roi", action="store_true",
                        help="Interactively select the ROI before detection")
    parser.add_argument("--max-dim", type=int, default=1400,
                        help="Maximum dimension for the preprocessing preview")
    parser.add_argument("--backend", choices=["classical", "unet"], default="classical",
                        help="Mask source: classical threshold code or a trained U-Net")
    parser.add_argument("--weights", default=None,
                        help="U-Net checkpoint (.pt); required with --backend unet")
    parser.add_argument("--threshold", type=float, default=None,
                        help="U-Net tuft probability threshold (0..1). Default: the value the "
                             "checkpoint's own validation picked (weights.pt.json), else 0.5")
    parser.add_argument("--polarity", choices=["dark", "light", "auto"], default="dark",
                        help="classical backend only: dark strings on a light surface (default), "
                             "light on dark, or auto = try both and keep the one with more tufts")
    parser.add_argument("--norm-size", type=int, default=DEFAULT_NORM_SIZE,
                        help="downscale big photos so the long side is at most this before "
                             "detecting (0 = off). The size limits and the U-Net's training crops "
                             "assume ~15-110 px tufts; use the same value you labelled/trained with")
    parser.add_argument("--no-clean", action="store_true",
                        help="skip the tuft-field filter that drops detections far from the main "
                             "cluster of tufts (sky, water, insets, stray blobs)")
    parser.add_argument("--verify-sam2", action="store_true",
                        help="keep only tufts SAM 2 can independently segment as a compact "
                             "string-like object (needs the sam2 package; slower)")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--sam2-config", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--vector-mode", choices=list(VECTOR_MODES), default="chord",
                        help="chord = root->tip; base = root->midpoint (ignores hooked "
                             "tip); tip = local tangent at the free end")
    parser.add_argument("--flow-prior-deg", type=float, default=None,
                        help="Expected bulk flow direction in degrees (math convention, "
                             "0 = +x/right, 90 = up). Only used to break root/tip ties.")
    parser.add_argument("--save-mask", action="store_true",
                        help="Also save the backend's raw tuft mask as a PNG")
    parser.add_argument("--min-area", type=float, default=15)
    parser.add_argument("--max-area", type=float, default=600)
    parser.add_argument("--min-elongation", type=float, default=2.0,
                        help="Minimum perimeter^2/(4*pi*area); curl-tolerant "
                             "elongation threshold (replaces bounding-box aspect ratio)")
    parser.add_argument("--min-length-px", type=float, default=12)
    parser.add_argument("--max-length-px", type=float, default=110)
    parser.add_argument("--block-size", type=int, default=25,
                        help="Adaptive threshold block size (odd, > 1)")
    parser.add_argument("--c", type=float, default=5,
                        help="Adaptive threshold constant subtracted from the mean")
    parser.add_argument("--clahe-clip", type=float, default=3.0,
                        help="CLAHE clip limit for local contrast normalization")
    parser.add_argument("--clahe-tile", type=int, default=8,
                        help="CLAHE tile grid size (NxN) for local contrast normalization")
    parser.add_argument("--tape-root", action="store_true",
                        help="Use yellow anchor tape to pick each tuft's root end and "
                             "reject candidates not anchored to tape")
    parser.add_argument("--tape-max-dist", type=float, default=60,
                        help="Max px from a tuft end to tape for --tape-root")
    return parser


def parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.image_option and args.image_positional:
        parser.error("provide the image path either positionally or with --image, not both")
    args.image = args.image_option or args.image_positional
    if not args.image:
        parser.error("an image path is required (use --image PATH or provide PATH positionally)")

    if args.max_dim <= 0:
        parser.error("--max-dim must be greater than 0")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.norm_size < 0:
        parser.error("--norm-size must be >= 0")
    if args.block_size <= 1 or args.block_size % 2 == 0:
        parser.error("--block-size must be an odd integer greater than 1")
    if args.min_area < 0 or args.max_area <= args.min_area:
        parser.error("--max-area must be greater than --min-area >= 0")
    if args.min_length_px < 0 or args.max_length_px <= args.min_length_px:
        parser.error("--max-length-px must be greater than --min-length-px >= 0")
    if args.min_elongation <= 0:
        parser.error("--min-elongation must be greater than 0")
    if args.backend == "unet" and not args.weights:
        parser.error("--backend unet requires --weights")
    return args


def _next_run_label(outdir: Path, stem: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{stem}_{timestamp}"
    suffix = 0
    while (outdir / f"{label}_annotated.png").exists():
        suffix += 1
        label = f"{stem}_{timestamp}_{suffix}"
    return label


def _tufts_dataframe(tufts, frame=0, timestamp_s=0.0):
    """Kept for backwards compatibility; the schema now lives in src.export."""
    return tufts_dataframe(tufts, frame=frame, timestamp_s=timestamp_s)


def main(argv=None):
    args = parse_args(argv)

    try:
        image = load_image(args.image)
        roi = _validate_roi(args.roi, image.shape)
    except (ImageLoadError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    info = get_image_info(image)
    print(f"Loaded image: {args.image}")
    print(f"  Resolution: {info['width']}x{info['height']}  Channels: {info['channels']}")

    # Image-processing branch: produce a normalized preview, but do NOT feed
    # the resized image into the detector. Detection coordinates must stay in
    # the original image's coordinate system.
    processed = resize_max_dim(image, max_dim=args.max_dim)
    processed_info = get_image_info(processed)
    print(f"  Preview size: {processed_info['width']}x{processed_info['height']}")

    preview = annotate_info_banner(
        processed, f"Preprocessed | {processed_info['width']}x{processed_info['height']}"
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    run_label = _next_run_label(outdir, stem)

    preview_path = outdir / f"{run_label}_preprocessed.jpg"
    save_image(preview, str(preview_path))

    if args.select_roi:
        x, y, w, h = cv2.selectROI(
            "Select ROI - press ENTER when done", image, showCrosshair=True
        )
        cv2.destroyAllWindows()
        if w > 0 and h > 0:
            roi = _validate_roi((x, y, w, h), image.shape)
        else:
            print("No ROI selected; using the full image.")

    # Detect on a canonically-scaled copy, then map results back to original pixels.
    work, scale = normalize_scale(image, args.norm_size)
    work_roi = _clamp_roi(scale_roi(roi, scale), work.shape)
    if scale != 1.0:
        print(f"  Detecting at {work.shape[1]}x{work.shape[0]} (scale {scale:.3f}); "
              f"results are mapped back to the original image")

    flow_prior = None
    if args.flow_prior_deg is not None:
        rad = math.radians(args.flow_prior_deg)
        flow_prior = (math.cos(rad), -math.sin(rad))  # image coords: y points down

    def run(backend):
        detector = TuftDetector(
            min_area=args.min_area, max_area=args.max_area,
            min_elongation=args.min_elongation,
            min_length_px=args.min_length_px, max_length_px=args.max_length_px,
            roi=work_roi, backend=backend, vector_mode=args.vector_mode,
            flow_prior=flow_prior, tape_root=args.tape_root,
            tape_max_dist=args.tape_max_dist * scale,
        )
        found = detector.detect(work)
        raw_count = len(found)
        if not args.no_clean:
            found = clean_tufts(work, found)
        return backend, found, raw_count

    if args.backend == "classical":
        pols = ("dark", "light") if args.polarity == "auto" else (args.polarity,)
        runs = [run(build_backend(
            "classical", block_size=args.block_size, c=args.c,
            clahe_clip=args.clahe_clip, clahe_tile=args.clahe_tile, polarity=pol))
            for pol in pols]
        backend, tufts, raw_count = max(runs, key=lambda r: len(r[1]))
        if len(pols) > 1:
            print(f"  Polarity auto-selected: {backend.polarity}")
    else:
        backend, tufts, raw_count = run(
            build_backend("unet", weights=args.weights, threshold=args.threshold))
        print(f"  U-Net threshold: {backend.threshold:.2f}")

    if not args.no_clean:
        print(f"  Tuft-field filter: {raw_count} -> {len(tufts)}")

    if args.verify_sam2 and tufts:
        from src.sam2_labeler import load_sam2_predictor, verify_tufts_with_sam
        predictor = load_sam2_predictor(args.sam2_model, config=args.sam2_config,
                                        checkpoint=args.sam2_checkpoint)
        before = len(tufts)
        tufts, sam_stats = verify_tufts_with_sam(work, tufts, predictor)
        print(f"  SAM 2 verification: {before} -> {len(tufts)} "
              f"({sam_stats['rejected']} rejected)")

    tufts = rescale_tufts(tufts, 1.0 / scale)
    for i, t in enumerate(tufts):
        t.id = i
    print(f"Detected {len(tufts)} tuft(s)")

    annotated = draw_tufts(image, tufts)
    annotated_path = outdir / f"{run_label}_annotated.png"
    save_image(annotated, str(annotated_path))

    if args.save_mask:
        mask_path = outdir / f"{run_label}_mask.png"
        crop = work if work_roi is None else work[
            work_roi[1]:work_roi[1] + work_roi[3], work_roi[0]:work_roi[0] + work_roi[2]]
        m = backend.predict(crop).tuft
        full = np.zeros(work.shape[:2], np.uint8)
        m8 = (m > 127).astype(np.uint8) * 255 if m.dtype == np.uint8 else (m >= 0.5).astype(np.uint8) * 255
        if work_roi is None:
            full = m8
        else:
            full[work_roi[1]:work_roi[1] + work_roi[3], work_roi[0]:work_roi[0] + work_roi[2]] = m8
        if scale != 1.0:
            full = cv2.resize(full, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        save_image(full, str(mask_path))
        print(f"Tuft mask:            {mask_path}")

    csv_path = outdir / f"{run_label}_vectors.csv"
    _tufts_dataframe(tufts).to_csv(csv_path, index=False)

    print(f"Preprocessed preview: {preview_path}")
    print(f"Annotated image:      {annotated_path}")
    print(f"Vector data:          {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())