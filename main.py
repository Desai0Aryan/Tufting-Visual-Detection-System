import argparse
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from detector import draw_tufts
from src.export import tufts_dataframe
from src.filtering import DEFAULT_NORM_SIZE
from src.mask_to_tufts import VECTOR_MODES
from src.pipeline import FramePipeline, clamp_roi as _clamp_roi  # noqa: F401 (_clamp_roi kept for old imports)
from src.image_io import (
    ImageLoadError,
    annotate_info_banner,
    get_image_info,
    load_image,
    resize_max_dim,
    save_image,
)
from src.video_io import VideoError, VideoReader, is_video_path


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
        description="Detect tufts and flow vectors in an image or an MP4 video."
    )
    # Keep the old positional form working while supporting the image-processing
    # branch's --image interface.
    parser.add_argument("image_positional", nargs="?",
                        help="Path to input image or video (video is detected by extension)")
    parser.add_argument("--image", dest="image_option", help="Path to input image")
    parser.add_argument("--video", dest="video_option", help="Path to input video (.mp4 etc.)")
    parser.add_argument(
        "--outdir", "--out-dir", dest="outdir", default="results",
        help="Directory for generated images and CSV data",
    )
    parser.add_argument("--roi", type=parse_roi, default=None,
                        help="Restrict detection to x,y,w,h (pixels)")
    parser.add_argument("--select-roi", action="store_true",
                        help="Interactively select the ROI before detection")
    video = parser.add_argument_group("video options (only used when the input is a video)")
    video.add_argument("--frame-step", type=int, default=1,
                       help="process every Nth frame (default 1 = every frame). The output "
                            "video then only contains the processed frames, at fps/N")
    video.add_argument("--max-frames", type=int, default=0,
                       help="stop after this many processed frames (0 = whole video)")
    video.add_argument("--save-frames", action="store_true",
                       help="also write every annotated frame as a PNG in <outdir>/<label>_frames/")
    video.add_argument("--no-track", action="store_true",
                       help="skip ID tracking: tuft IDs are then just per-frame numbers")
    video.add_argument("--track-dist", type=float, default=30.0,
                       help="tracking: max root movement between frames (px at the detection "
                            "scale, i.e. after --norm-size) to still count as the same tuft")
    video.add_argument("--track-max-missing", type=int, default=5,
                       help="tracking: frames a tuft may go undetected before its ID is retired")
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
    return parser


def parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    given = [v for v in (args.image_option, args.video_option, args.image_positional) if v]
    if len(given) > 1:
        parser.error("provide the input path only once (positionally, --image or --video)")
    args.image = args.video_option or args.image_option or args.image_positional
    if not args.image:
        parser.error("an input path is required (use --image/--video PATH or provide PATH positionally)")
    args.is_video = bool(args.video_option) or is_video_path(args.image)
    if args.frame_step < 1:
        parser.error("--frame-step must be >= 1")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.track_dist <= 0 or args.track_max_missing < 0:
        parser.error("--track-dist must be > 0 and --track-max-missing >= 0")

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


def _next_video_label(outdir: Path, stem: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{stem}_{timestamp}"
    suffix = 0
    while (outdir / f"{label}_annotated.mp4").exists():
        suffix += 1
        label = f"{stem}_{timestamp}_{suffix}"
    return label


def _tufts_dataframe(tufts, frame=0, timestamp_s=0.0):
    """Kept for backwards compatibility; the schema now lives in src.export."""
    return tufts_dataframe(tufts, frame=frame, timestamp_s=timestamp_s)


def _main_video(args):
    from src.video_pipeline import process_video

    try:
        reader = VideoReader(args.image)
    except VideoError as exc:
        print(f"[ERROR] {exc}")
        return 1

    with reader:
        print(f"Loaded video: {args.image}")
        total = f"{reader.frame_count} frames" if reader.frame_count else "unknown length"
        print(f"  Resolution: {reader.width}x{reader.height}  FPS: {reader.fps:.2f}  ({total})")

        try:
            first = reader.first_frame()
            roi = _validate_roi(args.roi, first.shape)
        except (VideoError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1

        if args.select_roi:
            x, y, w, h = cv2.selectROI(
                "Select ROI (first frame) - press ENTER when done", first, showCrosshair=True
            )
            cv2.destroyAllWindows()
            if w > 0 and h > 0:
                roi = _validate_roi((x, y, w, h), first.shape)
            else:
                print("No ROI selected; using the full frame.")

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        label = _next_video_label(outdir, Path(args.image).stem)

        pipeline = FramePipeline(args)
        print("Processing frames (video -> images -> detector -> annotated video)...")
        try:
            summary = process_video(args, pipeline, roi, outdir, label, reader)
        except VideoError as exc:
            print(f"[ERROR] {exc}")
            return 1

    if summary["frames"] == 0:
        print("[ERROR] no frames were processed")
        return 1
    print(f"Processed {summary['frames']} frame(s) in {summary['seconds']:.1f}s; "
          f"mean {summary['mean_tufts']:.1f} tuft(s) per frame")
    print(f"Annotated video:      {summary['video']}"
          + ("" if summary["h264"] else "  (mp4v encode; install ffmpeg for H.264 playback "
                                         "in browsers/PowerPoint)"))
    print(f"Vector time history:  {summary['csv']}")
    if summary["frames_dir"]:
        print(f"Annotated frames:     {summary['frames_dir']}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.is_video:
        return _main_video(args)

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
    # (The same FramePipeline is what the video path calls once per frame.)
    pipeline = FramePipeline(args)
    res = pipeline.process(image, roi=roi)
    work, work_roi, scale, backend = res.work, res.work_roi, res.scale, res.backend
    tufts = res.tufts
    if scale != 1.0:
        print(f"  Detecting at {work.shape[1]}x{work.shape[0]} (scale {scale:.3f}); "
              f"results are mapped back to the original image")
    if args.backend == "classical":
        if args.polarity == "auto":
            print(f"  Polarity auto-selected: {backend.polarity}")
    else:
        print(f"  U-Net threshold: {backend.threshold:.2f}")
    after_clean = len(tufts) + (res.sam_stats["rejected"] if res.sam_stats else 0)
    if not args.no_clean:
        print(f"  Tuft-field filter: {res.raw_count} -> {after_clean}")
    if res.sam_stats is not None:
        print(f"  SAM 2 verification: {after_clean} -> {len(tufts)} "
              f"({res.sam_stats['rejected']} rejected)")
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