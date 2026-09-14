"""CLI entry point: detect tufts and their flow vectors in a single image.

Usage:
    python main.py path/to/image.jpg
    python main.py path/to/image.jpg --roi 100,120,400,250
    python main.py path/to/image.jpg --select-roi
"""
import argparse
import os
import sys
from datetime import datetime

import cv2
import pandas as pd

from detector import TuftDetector, draw_tufts


def parse_roi(value):
    try:
        x, y, w, h = (int(v) for v in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--roi must be x,y,w,h (e.g. 100,120,400,250)")
    return (x, y, w, h)


def main():
    parser = argparse.ArgumentParser(description="Detect tufts and flow vectors in an image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--out-dir", default="output", help="Directory for annotated image + CSV")
    parser.add_argument("--roi", type=parse_roi, default=None,
                         help="Restrict detection to x,y,w,h (pixels)")
    parser.add_argument("--select-roi", action="store_true",
                         help="Interactively drag a box to set the ROI before detecting")
    parser.add_argument("--min-area", type=float, default=15)
    parser.add_argument("--max-area", type=float, default=600)
    parser.add_argument("--min-aspect-ratio", type=float, default=5.5)
    parser.add_argument("--min-length-px", type=float, default=12)
    parser.add_argument("--max-length-px", type=float, default=110)
    parser.add_argument("--block-size", type=int, default=25,
                         help="Adaptive threshold block size (odd)")
    parser.add_argument("--c", type=float, default=5,
                         help="Adaptive threshold constant subtracted from the mean")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    roi = args.roi
    if args.select_roi:
        x, y, w, h = cv2.selectROI("Select ROI - press ENTER when done", image, showCrosshair=True)
        cv2.destroyAllWindows()
        if w > 0 and h > 0:
            roi = (x, y, w, h)

    detector = TuftDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect_ratio=args.min_aspect_ratio,
        min_length_px=args.min_length_px,
        max_length_px=args.max_length_px,
        block_size=args.block_size,
        c=args.c,
        roi=roi,
    )
    tufts = detector.detect(image)
    print(f"Detected {len(tufts)} tuft(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"{stem}_{timestamp}"
    suffix = 0
    while os.path.exists(os.path.join(args.out_dir, f"{run_label}_annotated.png")):
        suffix += 1
        run_label = f"{stem}_{timestamp}_{suffix}"

    annotated = draw_tufts(image, tufts)
    annotated_path = os.path.join(args.out_dir, f"{run_label}_annotated.png")
    cv2.imwrite(annotated_path, annotated)

    df = pd.DataFrame([{
        "tuft_id": t.id,
        "root_x": t.root[0], "root_y": t.root[1],
        "tip_x": t.tip[0], "tip_y": t.tip[1],
        "unit_vector_x": t.vector[0], "unit_vector_y": t.vector[1],
        "angle_deg": t.angle_deg,
        "length_px": t.length_px,
    } for t in tufts])
    csv_path = os.path.join(args.out_dir, f"{run_label}_vectors.csv")
    df.to_csv(csv_path, index=False)

    print(f"Annotated image: {annotated_path}")
    print(f"Vector data:     {csv_path}")


if __name__ == "__main__":
    main()
