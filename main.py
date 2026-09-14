"""
main.py
"""

import argparse
from pathlib import Path

from src.image_io import (
    load_image,
    get_image_info,
    resize_max_dim,
    annotate_info_banner,
    save_image,
    ImageLoadError,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tufting Visual Detection System")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--outdir", default="outputs", help="Directory to write outputs to"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        image = load_image(args.image)
    except ImageLoadError as e:
        print(f"[ERROR] {e}")
        return 1

    info = get_image_info(image)
    print(f"Loaded image: {args.image}")
    print(f"  Resolution: {info['width']}x{info['height']}  Channels: {info['channels']}")

    processed = resize_max_dim(image, max_dim=1400)
    processed_info = get_image_info(processed)
    print(f"  Resized to: {processed_info['width']}x{processed_info['height']}")

    banner_text = f"Step 1 OK | {processed_info['width']}x{processed_info['height']}"
    preview = annotate_info_banner(processed, banner_text)

    out_path = Path(args.outdir) / "step1_preprocessed.jpg"
    save_image(preview, str(out_path))
    print(f"Saved preprocessed preview to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
