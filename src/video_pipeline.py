"""Video in -> frames -> the normal image detector -> annotated frames -> video out.

Not live: the whole file is processed offline. Outputs (next to each other in --outdir):
  <label>_annotated.mp4   the original video with flow vectors drawn on every processed frame
  <label>_vectors.csv     time history: one row per tuft per frame (timestamp, tuft ID, vector ...)
  <label>_frames/         annotated frames as images (only with --save-frames)
"""
import time
from pathlib import Path

import pandas as pd

from src.export import COLUMNS, tufts_to_rows
from src.image_io import annotate_info_banner, save_image
from src.overlay import draw_tufts
from src.video_io import VideoReader, VideoWriter
from tracker import TuftTracker


def process_video(args, pipeline, roi, outdir: Path, label: str, reader: VideoReader):
    """Run the per-frame pipeline over `reader`. Returns a summary dict."""
    step = max(1, args.frame_step)
    out_fps = reader.fps / step
    video_path = outdir / f"{label}_annotated.mp4"
    csv_path = outdir / f"{label}_vectors.csv"
    frames_dir = outdir / f"{label}_frames" if args.save_frames else None
    tracker = None if args.no_track else TuftTracker(max_dist=args.track_dist,
                                                    max_missing=args.track_max_missing)

    rows, counts = [], []
    processed = 0
    t0 = time.time()
    writer = VideoWriter(video_path, out_fps, (reader.width, reader.height))
    try:
        for idx, ts, frame in reader.frames():
            if idx % step:
                continue
            if args.max_frames and processed >= args.max_frames:
                break

            res = pipeline.process(frame, roi=roi, lock_polarity=True)
            tufts = tracker.update(res.tufts) if tracker else res.tufts

            rows.extend(tufts_to_rows(tufts, frame=idx, timestamp_s=ts))
            counts.append(len(tufts))

            annotated = draw_tufts(frame, tufts)
            annotated = annotate_info_banner(
                annotated, f"frame {idx}  t={ts:6.2f}s  tufts={len(tufts)}")
            writer.write(annotated)
            if frames_dir is not None:
                save_image(annotated, str(frames_dir / f"frame_{idx:06d}.png"))

            processed += 1
            if processed % 10 == 0 or processed == 1:
                total = reader.frame_count
                of = f"/{(total + step - 1) // step}" if total else ""
                print(f"  processed {processed}{of} frames", flush=True)
    finally:
        writer.close()

    pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False)
    return {
        "frames": processed, "video": video_path, "csv": csv_path, "frames_dir": frames_dir,
        "mean_tufts": (sum(counts) / len(counts)) if counts else 0.0,
        "seconds": time.time() - t0, "h264": writer.h264, "out_fps": out_fps,
    }
