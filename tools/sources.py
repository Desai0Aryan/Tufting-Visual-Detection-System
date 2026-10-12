"""Register clips in sources.csv (and fetch YouTube clips).

    # our own footage (copied into data/clips/<clip_id>.mp4)
    python -m tools.sources add-local path/to/run3.mp4 --split test --use tufts --notes "wind tunnel run 3"

    # a YouTube clip: refused unless Creative Commons, or you pass --permission "who/when"
    python -m tools.sources add-url https://www.youtube.com/watch?v=XXXX --split train --use tufts

    python -m tools.sources check      # licenses, files, quality, own footage in test
    python -m tools.sources list
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sources import (USES, add_row, check, is_creative_commons, make_row,  # noqa: E402
                         quality_problems, read_sources, video_stats)
from src.splits import SPLITS  # noqa: E402


def _common(p):
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--use", required=True, choices=USES)
    p.add_argument("--clip-id", default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--force", action="store_true", help="accept despite heavy compression")


def cmd_add_local(a):
    src = Path(a.path)
    cid = a.clip_id or f"own_{src.stem}".replace(" ", "_")
    dst = Path(a.clips_dir) / f"{cid}{src.suffix.lower()}"
    stats = video_stats(src)
    bad = quality_problems(stats, a.force)
    if bad:
        raise SystemExit(f"refused {src}: " + "; ".join(bad))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    add_row(a.sources, make_row(cid, a.split, a.use, "own", license_text="own footage",
                                title=src.name, resolution=f"{stats['width']}x{stats['height']}@{stats['fps']}",
                                notes=a.notes))
    print(f"added {cid} -> {dst} ({a.split}, {a.use}) {stats}")


def cmd_add_url(a):
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(a.url, download=False)
    lic = info.get("license") or ""
    if not is_creative_commons(lic):
        if not a.permission:
            raise SystemExit(f"refused: license is {lic or 'standard (not Creative Commons)'}. "
                             f"Pass --permission \"who granted it, when\" only if you have permission.")
        lic = f"permission: {a.permission}" + (f" (listed license: {lic})" if lic else "")
    heights = [f.get("height") or 0 for f in info.get("formats", []) if f.get("vcodec") not in (None, "none")]
    if max(heights, default=0) < 720:
        raise SystemExit(f"refused: best available resolution is {max(heights, default=0)}p (< 720p)")
    cid = a.clip_id or f"yt_{info['id']}"
    out = Path(a.clips_dir) / f"{cid}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = f"[height<={a.max_height}]" if a.max_height else ""
    opts = {"quiet": True, "outtmpl": str(out), "merge_output_format": "mp4",
            "format": f"bv*{cap}[ext=mp4]/bv*{cap}/b{cap}"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([a.url])
    stats = video_stats(out)
    bad = quality_problems(stats, a.force)
    if bad:
        out.unlink()
        raise SystemExit(f"refused after download (file deleted): " + "; ".join(bad))
    add_row(a.sources, make_row(cid, a.split, a.use, "youtube", url=a.url, title=info.get("title", ""),
                                uploader=info.get("uploader", ""), license_text=lic,
                                resolution=f"{stats['width']}x{stats['height']}@{stats['fps']}",
                                notes=a.notes))
    print(f"added {cid}: {info.get('title')!r} by {info.get('uploader')} [{lic}] {stats}")


def cmd_set_split(a):
    from src.sources import write_sources
    rows = read_sources(a.sources)
    hit = [r for r in rows if r["clip_id"] == a.clip_id]
    if not hit:
        raise SystemExit(f"{a.clip_id} is not in {a.sources}")
    hit[0]["split"] = a.split
    if a.notes:
        hit[0]["notes"] = a.notes
    write_sources(a.sources, rows)
    print(f"{a.clip_id}: split -> {a.split}")
    return 0


def cmd_check(a):
    errors, summary = check(a.sources, a.clips_dir)
    for s, uses in summary.items():
        print(f"{s:5s} " + "  ".join(f"{u} {n}" for u, n in uses.items()))
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


def cmd_list(a):
    for r in read_sources(a.sources):
        print(f"{r['clip_id']:28s} {r['split']:5s} {r['use']:10s} {r['origin']:8s} {r['resolution']:16s} "
              f"{r['license'][:40]:40s} {r['title'][:40]}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", default="sources.csv")
    ap.add_argument("--clips-dir", default="data/clips")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add-local"); p.add_argument("path"); _common(p); p.set_defaults(fn=cmd_add_local)
    p = sub.add_parser("add-url"); p.add_argument("url"); _common(p)
    p.add_argument("--permission", default=None, help="who granted permission and when (non-CC clips)")
    p.add_argument("--max-height", type=int, default=0, help="cap the download height (0 = best available)")
    p.set_defaults(fn=cmd_add_url)
    p = sub.add_parser("set-split"); p.add_argument("clip_id"); p.add_argument("split", choices=SPLITS)
    p.add_argument("--notes", default=None)
    p.set_defaults(fn=cmd_set_split)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
