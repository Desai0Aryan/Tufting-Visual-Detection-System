"""Convert the DLR tuft dataset (Zenodo 10539950) into training data.

    python -m tools.import_dlr polygons      # HABLEG LabelMe polygons -> masks   (data/dlr/hableg_polygons)
    python -m tools.import_dlr boxes         # HABLEG VOC boxes -> SAM 2 masks    (data/dlr/hableg_sam2/proposed)
    python -m tools.import_dlr promote       # accept SAM 2 proposals of train sessions -> data/dlr/hableg_sam2/labeled
    python -m tools.import_dlr clean         # glare blobs / fat merged masks -> cleared + ignore region
    python -m tools.import_dlr tracks        # HABLEG boxes with persistent IDs -> tracker ground truth CSVs
    python -m tools.import_dlr sheet --data data/dlr/hableg_polygons --out data/dlr/preview_polygons.jpg

Every frame gets clip_id = dlr_hableg_<session> (GOPR / GP02 / GP03 / GP04),
so the split comes from sources.csv by session, never by frame. Masks are
rasterised at native resolution; the trainer takes 512 px crops from them.

Duplicate polygon IDs: same-ID polygons that overlap or touch (gap <= 3 px) are
merged into one tuft; separate ones are kept as different tufts and renamed
<ID>#2, #3 ... in the per-frame JSON. Every decision is written to the report.

The detection boxes never cover tufts T1, T2 and T4 (checked against the
polygon frames), so SAM 2 frames get an ignore region per session: the union
of those tufts' polygons in that session, dilated. No loss is applied there.
EC135 is skipped by default (its duct-ring tufts appear unlabelled).
"""
import argparse
import collections
import csv
import json
import random
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.label_io import load_frame, load_ignore, save_frame, tufts_from_mask  # noqa: E402
from src.labeling import list_samples  # noqa: E402
from src.splits import load_split_map  # noqa: E402
from tools.view_dlr import imread, read_labelme, read_voc  # noqa: E402

RAW = Path("data/external/dlr_tufts/tufts")
UNBOXED_IDS = ("T1", "T2", "T4")
MERGE_GAP_PX = 3.0


def session_of(stem):
    return stem.split("_")[0]


def clip_of(subset, stem):
    return f"dlr_hableg_{session_of(stem)}" if subset == "HABLEG" else "dlr_ec135"


def rasterize(polys, shape):
    m = np.zeros(shape[:2], np.uint8)
    for _, p in polys:
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 255)
    return m


def resolve_duplicates(polys, shape):
    """Merge same-ID polygons that overlap/touch, keep and rename separate ones.
    Returns (list of (id, [polygons]), decisions)."""
    by = collections.defaultdict(list)
    for lab, p in polys:
        by[lab].append(p)
    out, decisions = [], []
    for lab, ps in by.items():
        if len(ps) == 1:
            out.append((lab, ps))
            continue
        masks = [rasterize([(lab, p)], shape) > 0 for p in ps]
        groups = []  # union-find by touching
        for i, m in enumerate(masks):
            dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
            hit = [g for g in groups if any(float(dist[masks[j]].min()) <= MERGE_GAP_PX for j in g)]
            merged = [i] + [j for g in hit for j in g]
            groups = [g for g in groups if g not in hit] + [merged]
        for k, g in enumerate(sorted(groups, key=min)):
            name = lab if k == 0 else f"{lab}#{k + 1}"
            out.append((name, [ps[j] for j in g]))
            gaps = []
            decisions.append({"id": lab, "polygons": len(g), "action": "merged" if len(g) > 1 else
                              ("kept" if k == 0 else "kept_renamed"), "name": name})
    return out, decisions


def write_frame(out, subset, stem, image, mask, meta_extra, ignore=None, ids=None):
    split_map = load_split_map("sources.csv")
    cid = clip_of(subset, stem)
    tufts = tufts_from_mask(mask) if mask.any() else []
    extra = None
    if ids is not None:  # attach the dataset's tuft ID to each extracted tuft
        extra = []
        for t in tufts:
            c = tuple(int(v) for v in np.round(t.path[len(t.path) // 2]))
            extra.append({"dlr_id": ids[c[1], c[0]] if 0 <= c[1] < ids.shape[0] and 0 <= c[0] < ids.shape[1] else ""})
    meta = {"clip_id": cid, "frame_idx": int(stem.split("_")[1]), "split": split_map.get(cid, ""),
            "use": "tufts", "domain": "dlr", "dataset": "dlr_tufts", "subset": subset, **meta_extra}
    save_frame(out, stem, image, mask, tufts, meta, extra=extra, ignore=ignore)
    return tufts


# ---------------------------------------------------------------- polygons
def cmd_polygons(a):
    subsets = ["HABLEG"] + (["EC135-ACT⁄FHS"] if a.include_ec135 else [])
    report = {"frames": 0, "polygons": 0, "tufts_after_merge": 0, "duplicates": [], "warnings": []}
    for sub in subsets:
        for j in sorted((RAW / "segmentation" / sub).glob("*.json")):
            img = imread(j.with_suffix(".JPG")) if j.with_suffix(".JPG").exists() else imread(j.with_suffix(".jpg"))
            size, _, polys = read_labelme(j)
            if tuple(size) != (img.shape[1], img.shape[0]):
                report["warnings"].append(f"{j.stem}: json size {size} != image {img.shape[1::-1]}")
            groups, decisions = resolve_duplicates(polys, img.shape)
            mask = np.zeros(img.shape[:2], np.uint8)
            ids = np.full(img.shape[:2], "", object)
            for name, ps in groups:
                m = rasterize([(name, p) for p in ps], img.shape)
                mask |= m
                ids[m > 0] = name
            write_frame(a.out, sub, j.stem, img, mask, {"source": "dlr_polygon", "status": "accepted",
                                                         "reviewed": True}, ids=ids)
            report["frames"] += 1
            report["polygons"] += len(polys)
            report["tufts_after_merge"] += len(groups)
            report["duplicates"] += [dict(d, stem=j.stem) for d in decisions]
    counts = collections.Counter(d["action"] for d in report["duplicates"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "import_report.json").write_text(json.dumps(report, indent=1))
    print(f"{report['frames']} frames, {report['polygons']} polygons -> {report['tufts_after_merge']} tufts; "
          f"duplicate-ID handling: {dict(counts)}; warnings: {len(report['warnings'])}")
    return 0


# ---------------------------------------------------------------- SAM 2 from boxes
def unboxed_ignore_regions(dilate_px=20):
    """Per session: union of T1/T2/T4 polygons over that session's polygon frames, dilated."""
    regions = {}
    for j in sorted((RAW / "segmentation" / "HABLEG").glob("*.json")):
        size, _, polys = read_labelme(j)
        s = session_of(j.stem)
        reg = regions.setdefault(s, np.zeros((size[1], size[0]), np.uint8))
        reg |= rasterize([(l, p) for l, p in polys if l in UNBOXED_IDS], (size[1], size[0]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    return {s: cv2.dilate(r, k) for s, r in regions.items()}


def sam_from_boxes(image, boxes, predictor, min_side=512, multimask=False):
    """One SAM 2 mask per box (box prompt only). Returns (mask, per-box records)."""
    from src.sam2_autolabel import _drop_stray_components
    H, W = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out = np.zeros((H, W), np.uint8)
    ids = np.full((H, W), "", object)
    recs, cur = [], None
    for lab, x0, y0, x1, y1 in sorted(boxes, key=lambda b: (b[2], b[1])):
        if cur is None or not (x0 >= cur[0] and y0 >= cur[1] and x1 <= cur[2] and y1 <= cur[3]):
            side = int(min(max(min_side, 1.5 * max(x1 - x0, y1 - y0)), max(H, W)))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            wx = int(np.clip(round(cx - side / 2), 0, max(W - side, 0)))
            wy = int(np.clip(round(cy - side / 2), 0, max(H - side, 0)))
            cur = (wx, wy, min(wx + side, W), min(wy + side, H))
            predictor.set_image(rgb[cur[1]:cur[3], cur[0]:cur[2]])
        box = np.array([x0 - cur[0], y0 - cur[1], x1 - cur[0], y1 - cur[1]], np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=multimask)
        k = int(np.argmax(scores))
        m = masks[k] > 0
        inbox = np.zeros_like(m)
        bx0, by0 = max(int(box[0]) - 2, 0), max(int(box[1]) - 2, 0)
        inbox[by0:int(box[3]) + 3, bx0:int(box[2]) + 3] = True
        m = _drop_stray_components(m & inbox, m & inbox)
        area = int(m.sum())
        box_area = max((x1 - x0) * (y1 - y0), 1)
        ok = area > 0 and area / box_area < 0.9  # a filled box means SAM took the background
        if ok:
            out[cur[1]:cur[3], cur[0]:cur[2]][m] = 255
            ids[cur[1]:cur[3], cur[0]:cur[2]][m] = lab
        recs.append({"id": lab, "box": [round(v, 1) for v in (x0, y0, x1, y1)], "score": round(float(scores[k]), 3),
                     "area": area, "fill": round(area / box_area, 3), "accepted": bool(ok)})
    return out, ids, recs


def cmd_boxes(a):
    import torch
    from src.sam2_autolabel import load_image_predictor
    pred = load_image_predictor(a.checkpoint, a.sam2_config)
    det = RAW / "detection" / "HABLEG"
    seg_stems = {j.stem for j in (RAW / "segmentation" / "HABLEG").glob("*.json")}
    sessions = set(a.sessions)
    ignore = unboxed_ignore_regions()

    # 1) quality check against the human polygons on frames that have both (frame level:
    #    a binary training mask does not need instances, so this is the number that matters)
    if not a.skip_check:
        stats = []
        for j in sorted((RAW / "segmentation" / "HABLEG").glob("*.json")):
            img = imread(det / "JPEGImages" / f"{j.stem}.JPG")
            _, boxes = read_voc(det / "Annotations" / f"{j.stem}.xml")
            _, _, polys = read_labelme(j)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m, _, _ = sam_from_boxes(img, boxes, pred)
            gt = rasterize([(l, p) for l, p in polys if l not in UNBOXED_IDS], img.shape) > 0
            pm = m > 0
            tp = (pm & gt).sum()
            stats.append((tp / max((pm | gt).sum(), 1), tp / max(pm.sum(), 1), tp / max(gt.sum(), 1)))
        S = np.array(stats)
        q = {"frames": len(S), "note": "SAM 2 box prompt vs human polygons (T1/T2/T4 excluded: never boxed)",
             "iou_median": round(float(np.median(S[:, 0])), 3), "iou_p10": round(float(np.percentile(S[:, 0], 10)), 3),
             "iou_min": round(float(S[:, 0].min()), 3), "precision_median": round(float(np.median(S[:, 1])), 3),
             "recall_median": round(float(np.median(S[:, 2])), 3)}
        print("SAM 2 (box prompt) vs human polygons:", q)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "sam_vs_polygons.json").write_text(json.dumps(q, indent=1))

    # 2) proposals for every other frame of the chosen sessions
    done = set(list_samples(a.out))
    stems = sorted(x.stem for x in (det / "Annotations").glob("*.xml")
                   if session_of(x.stem) in sessions and x.stem not in seg_stems and x.stem not in done)
    if a.limit:
        stems = stems[:a.limit]
    t0, n_rej = time.time(), 0
    for i, stem in enumerate(stems):
        img = imread(det / "JPEGImages" / f"{stem}.JPG")
        _, boxes = read_voc(det / "Annotations" / f"{stem}.xml")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m, ids, recs = sam_from_boxes(img, boxes, pred)
        n_rej += sum(not r["accepted"] for r in recs)
        tufts = write_frame(a.out, "HABLEG", stem, img, m,
                            {"source": "sam2_box", "status": "proposed", "reviewed": False,
                             "sam_rejected_boxes": sum(not r["accepted"] for r in recs)},
                            ignore=ignore.get(session_of(stem)), ids=ids)
        info_p = Path(a.out) / "json" / f"{stem}.json"
        info = json.loads(info_p.read_text())
        info["guide_boxes"] = recs  # shown in the review tool
        info_p.write_text(json.dumps(info, indent=1))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(stems)} frames ({time.time() - t0:.0f}s)")
    print(f"proposed {len(stems)} frames in {time.time() - t0:.0f}s; boxes SAM could not segment: {n_rej}")

    # 3) review batch: a fixed random sample, stratified over sessions
    if a.review_n <= 0:
        return 0
    all_stems = sorted(list_samples(a.out))
    rng = random.Random(a.seed)
    by = collections.defaultdict(list)
    for s in all_stems:
        by[session_of(s)].append(s)
    batch, k = [], 0
    while len(batch) < min(a.review_n, len(all_stems)):
        sess = sorted(by)[k % len(by)]
        if by[sess]:
            batch.append(by[sess].pop(rng.randrange(len(by[sess]))))
        k += 1
    (Path(a.out) / "review_batch1.txt").write_text("\n".join(sorted(batch)) + "\n")
    print(f"review batch of {len(batch)} frames -> {Path(a.out) / 'review_batch1.txt'}")
    return 0


# ---------------------------------------------------------------- accept proposals
def cmd_promote(a):
    """Copy proposals of TRAIN sessions (per sources.csv) to the labelled set, with
    their ignore regions and JSON. Validation sessions are never promoted: their
    validation labels are the human polygons only."""
    import shutil
    split_map = load_split_map("sources.csv")
    src, dst = Path(a.src), Path(a.out)
    meta = json.loads((src / "meta.json").read_text())
    batch = set((src / a.batch_file).read_text().split()) if (src / a.batch_file).exists() else set()
    out_meta_p = dst / "meta.json"
    out_meta = json.loads(out_meta_p.read_text()) if out_meta_p.exists() else {}
    counts = collections.Counter()
    for stem, m in sorted(meta.items()):
        split = split_map.get(m["clip_id"])
        if split != "train" or m.get("status") not in ("proposed", "accepted"):
            counts[f"skipped ({split}, {m.get('status')})"] += 1
            continue
        for sub, ext in (("images", ".png"), ("labels", ".png"), ("ignore", ".png"), ("json", ".json")):
            f = src / sub / f"{stem}{ext}"
            if f.exists():
                (dst / sub).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst / sub / f.name)
        how = "approved_visually_batch1_no_edits" if stem in batch else a.approval
        out_meta[stem] = dict(m, status="accepted", split=split, reviewed=False, approval=how)
        meta[stem] = dict(m, status="accepted")
        counts[how] += 1
    out_meta_p.write_text(json.dumps(out_meta, indent=1))
    (src / "meta.json").write_text(json.dumps(meta, indent=1))
    print(dict(counts))
    return 0


# ---------------------------------------------------------------- automatic clean-up
def cmd_clean(a):
    """Per accepted box: a mask that is mostly near-white (glare) or far wider than that
    tuft ID usually is (tuft + shadow merged, filled loops) is cleared and its box
    (padded) becomes an ignore region: no loss either way. Logged in meta.json."""
    from skimage.morphology import skeletonize
    D = Path(a.data)
    meta = json.loads((D / "meta.json").read_text())
    per = []
    for f in sorted((D / "json").glob("*.json")):
        img = cv2.imread(str(D / "images" / f"{f.stem}.png"))
        lab = cv2.imread(str(D / "labels" / f"{f.stem}.png"))[..., 0] > 127
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        glare = (hsv[..., 1] < 40) & (hsv[..., 2] > 225)
        for b in json.loads(f.read_text()).get("guide_boxes", []):
            if not b["accepted"]:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in b["box"])
            sl = (slice(max(y0 - 2, 0), y1 + 3), slice(max(x0 - 2, 0), x1 + 3))
            m = lab[sl]
            if m.sum() < 20:
                continue
            per.append({"stem": f.stem, "id": b["id"], "box": (x0, y0, x1, y1),
                        "glare": float((glare[sl] & m).sum() / m.sum()),
                        "width": float(m.sum() / max(skeletonize(m).sum(), 1))})
    med = collections.defaultdict(list)
    for r in per:
        med[r["id"]].append(r["width"])
    med = {k: float(np.median(v)) for k, v in med.items()}
    flagged = [dict(r, width_ratio=round(r["width"] / med[r["id"]], 2)) for r in per
               if r["glare"] > a.max_glare or r["width"] / med[r["id"]] > a.max_width_ratio]
    by = collections.defaultdict(list)
    for r in flagged:
        by[r["stem"]].append(r)
    for stem, rs in by.items():
        lab_p, ign_p = D / "labels" / f"{stem}.png", D / "ignore" / f"{stem}.png"
        lab = cv2.imread(str(lab_p))
        ign = cv2.imread(str(ign_p), cv2.IMREAD_GRAYSCALE) if ign_p.exists() else np.zeros(lab.shape[:2], np.uint8)
        for r in rs:
            x0, y0, x1, y1 = r["box"]
            ys, xs = slice(max(y0 - a.pad, 0), y1 + a.pad + 1), slice(max(x0 - a.pad, 0), x1 + a.pad + 1)
            lab[ys, xs, 0] = 0
            lab[ys, xs, 1] = 0
            ign[ys, xs] = 255
        cv2.imwrite(str(lab_p), lab)
        info_p = D / "json" / f"{stem}.json"
        info = json.loads(info_p.read_text())
        mask = (lab[..., 0] > 127).astype(np.uint8) * 255
        tufts = tufts_from_mask(mask) if mask.any() else []
        from src.label_io import tuft_records
        info["tufts"] = tuft_records(tufts)
        info_p.write_text(json.dumps(info, indent=1))
        (D / "ignore").mkdir(exist_ok=True)
        cv2.imwrite(str(ign_p), ign)
        meta[stem]["auto_fixed_boxes"] = [{"id": r["id"], "glare": round(r["glare"], 2),
                                            "width_ratio": r["width_ratio"]} for r in rs]
    (D / "meta.json").write_text(json.dumps(meta, indent=1))
    (D / "clean_report.json").write_text(json.dumps({"max_glare": a.max_glare, "max_width_ratio": a.max_width_ratio,
                                                     "boxes_checked": len(per), "fixed": flagged}, indent=1, default=str))
    print(f"checked {len(per)} boxes; cleared + ignored {len(flagged)} boxes in {len(by)} frames: "
          f"{sorted((r['stem'], r['id']) for r in flagged)}")
    return 0


# ---------------------------------------------------------------- tracker ground truth
def cmd_tracks(a):
    det = RAW / "detection" / "HABLEG"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by = collections.defaultdict(list)
    for x in sorted((det / "Annotations").glob("*.xml")):
        by[session_of(x.stem)].append(x)
    for sess, xs in sorted(by.items()):
        rows = []
        for x in sorted(xs, key=lambda p: int(p.stem.split("_")[1])):
            _, boxes = read_voc(x)
            for lab, x0, y0, x1, y1 in boxes:
                rows.append({"session": sess, "frame": int(x.stem.split("_")[1]), "stem": x.stem, "tuft_id": lab,
                             "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
                             "cx": round((x0 + x1) / 2, 2), "cy": round((y0 + y1) / 2, 2)})
        p = out / f"hableg_{sess}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        ids = collections.Counter(r["tuft_id"] for r in rows)
        frames = sorted({r["frame"] for r in rows})
        print(f"{sess}: {len(frames)} frames ({frames[0]}-{frames[-1]}), {len(ids)} track IDs, {len(rows)} boxes -> {p}")
    return 0


# ---------------------------------------------------------------- contact sheet
def cmd_sheet(a):
    stems = list_samples(a.data)
    if a.stems_file:
        keep = set(Path(a.stems_file).read_text().split())
        stems = [s for s in stems if s in keep]
    rng = random.Random(a.seed)
    pick = sorted(rng.sample(stems, min(a.n, len(stems))))
    rows = []
    for stem in pick:
        img, mask, info, meta = load_frame(a.data, stem)
        ign = load_ignore(a.data, stem)
        m = mask > 0
        ov = img.copy()
        ov[m] = (0.35 * ov[m] + 0.65 * np.array((60, 220, 60))).astype(np.uint8)
        if ign is not None:
            ig = ign > 0
            ov[ig] = (0.6 * ov[ig] + 0.4 * np.array((255, 120, 0))).astype(np.uint8)
        for b in info.get("guide_boxes", []):
            x0, y0, x1, y1 = (int(v) for v in b["box"])
            cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 165, 255) if b["accepted"] else (0, 0, 255), 1)
        for t in info.get("tufts", []):
            lab = t.get("dlr_id") or ""
            r = tuple(int(v) for v in t["root"])
            cv2.circle(ov, r, 3, (0, 0, 255), -1)
            if lab:
                cv2.putText(ov, lab, (r[0] + 4, r[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        # crop to the labelled region so thin tufts are visible
        ys, xs = np.nonzero(m | (ign > 0 if ign is not None else False))
        if len(xs):
            x0, x1 = max(xs.min() - 60, 0), min(xs.max() + 60, img.shape[1])
            y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 40, img.shape[0])
        else:
            x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
        tiles = [img[y0:y1, x0:x1], np.dstack([mask[y0:y1, x0:x1]] * 3), ov[y0:y1, x0:x1]]
        s = a.tile_w / tiles[0].shape[1]
        tiles = [cv2.resize(t, (a.tile_w, int(t.shape[0] * s)), interpolation=cv2.INTER_AREA) for t in tiles]
        sep = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
        row = np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]])
        cap = np.zeros((20, row.shape[1], 3), np.uint8)
        cv2.putText(cap, f"{stem}  {meta.get('source')}  tufts {len(info.get('tufts', []))}  "
                         f"split {meta.get('split')}  zoom {s:.2f}x", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([cap, row, np.full((6, row.shape[1], 3), 255, np.uint8)]))
    w = max(r.shape[1] for r in rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows])
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(pick)} frames: image | mask | overlay (green mask, blue ignore, orange/red boxes) -> {a.out}")
    if a.open:
        import os
        os.startfile(str(Path(a.out).resolve()))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("polygons")
    p.add_argument("--out", default="data/dlr/hableg_polygons")
    p.add_argument("--include-ec135", action="store_true", help="not recommended: unlabelled duct-ring tufts")
    p.set_defaults(fn=cmd_polygons)
    p = sub.add_parser("boxes")
    p.add_argument("--out", default="data/dlr/hableg_sam2/proposed")
    p.add_argument("--sessions", nargs="+", default=["GOPR", "GP02", "GP03"],
                   help="GP04 is the external validation session; it uses human polygons only")
    p.add_argument("--checkpoint", default="models/sam2/sam2.1_hiera_large.pt")
    p.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--review-n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-check", action="store_true")
    p.set_defaults(fn=cmd_boxes)
    p = sub.add_parser("promote")
    p.add_argument("--src", default="data/dlr/hableg_sam2/proposed")
    p.add_argument("--out", default="data/dlr/hableg_sam2/labeled")
    p.add_argument("--batch-file", default="review_batch1.txt")
    p.add_argument("--approval", default="bulk_accepted_after_batch1")
    p.set_defaults(fn=cmd_promote)
    p = sub.add_parser("clean")
    p.add_argument("--data", default="data/dlr/hableg_sam2/labeled")
    p.add_argument("--max-glare", type=float, default=0.3, help="max near-white fraction of a box's mask")
    p.add_argument("--max-width-ratio", type=float, default=2.5, help="max width vs that tuft ID's median")
    p.add_argument("--pad", type=int, default=4)
    p.set_defaults(fn=cmd_clean)
    p = sub.add_parser("tracks")
    p.add_argument("--out", default="data/dlr/hableg_tracks")
    p.set_defaults(fn=cmd_tracks)
    p = sub.add_parser("sheet")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stems-file", default=None)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tile-w", type=int, default=520)
    p.add_argument("--open", action="store_true")
    p.set_defaults(fn=cmd_sheet)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
