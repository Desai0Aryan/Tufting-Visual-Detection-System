"""CLI entry point for image loading, tuft detection, and vector export.

Examples:
    python main.py --image samples/sample_tufts1.jpg
    python main.py samples/sample_tufts1.jpg --outdir outputs
    python main.py --image samples/sample_tufts1.jpg --roi 100,120,400,250
    python main.py --image samples/sample_tufts1.jpg --select-roi

The detector runs on the original-resolution image. A resized copy is used only
for the preprocessing preview, so exported pixel coordinates always match the
input image.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from detector import TuftDetector, draw_tufts
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
    if args.block_size <= 1 or args.block_size % 2 == 0:
        parser.error("--block-size must be an odd integer greater than 1")
    if args.min_area < 0 or args.max_area <= args.min_area:
        parser.error("--max-area must be greater than --min-area >= 0")
    if args.min_length_px < 0 or args.max_length_px <= args.min_length_px:
        parser.error("--max-length-px must be greater than --min-length-px >= 0")
    if args.min_elongation <= 0:
        parser.error("--min-elongation must be greater than 0")
    return args


def _next_run_label(outdir: Path, stem: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{stem}_{timestamp}"
    suffix = 0
    while (outdir / f"{label}_annotated.png").exists():
        suffix += 1
        label = f"{stem}_{timestamp}_{suffix}"
    return label


def _tufts_dataframe(tufts):
    columns = [
        "tuft_id", "root_x", "root_y", "tip_x", "tip_y",
        "unit_vector_x", "unit_vector_y", "angle_deg", "length_px",
    ]
    return pd.DataFrame(
        [
            {
                "tuft_id": t.id,
                "root_x": t.root[0], "root_y": t.root[1],
                "tip_x": t.tip[0], "tip_y": t.tip[1],
                "unit_vector_x": t.vector[0], "unit_vector_y": t.vector[1],
                "angle_deg": t.angle_deg,
                "length_px": t.length_px,
            }
            for t in tufts
        ],
        columns=columns,
    )


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

    detector = TuftDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        min_elongation=args.min_elongation,
        min_length_px=args.min_length_px,
        max_length_px=args.max_length_px,
        block_size=args.block_size,
        c=args.c,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        roi=roi,
    )
    tufts = detector.detect(image)
    print(f"Detected {len(tufts)} tuft(s)")

    annotated = draw_tufts(image, tufts)
    annotated_path = outdir / f"{run_label}_annotated.png"
    save_image(annotated, str(annotated_path))

    csv_path = outdir / f"{run_label}_vectors.csv"
    _tufts_dataframe(tufts).to_csv(csv_path, index=False)

    print(f"Preprocessed preview: {preview_path}")
    print(f"Annotated image:      {annotated_path}")
    print(f"Vector data:          {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
