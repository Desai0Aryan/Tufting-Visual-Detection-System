import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.label_io import load_ignore, load_frame, save_frame, tufts_from_mask
from src.tuft_cards import append_vote, build_frame, frame_cards, load_votes, stems_with_votes

STEM = "clipA__f000010"


def _frame():
    """Two SAM tufts, one SAM-rejected candidate, one tiny SAM fragment."""
    img = np.full((400, 600, 3), 50, np.uint8)
    mask = np.zeros((400, 600), np.uint8)
    for x in (100, 250):
        cv2.line(img, (x, 50), (x + 20, 350), (240, 240, 240), 5)
        cv2.line(mask, (x, 50), (x + 20, 350), 255, 5)
    mask[380:383, 10:13] = 255  # 9 px fragment: no card
    rejected = np.array([[430, 60], [440, 60], [460, 340], [450, 340]])
    info = {"tufts": [], "rejected_candidates": [{"status": "rejected", "contour": rejected.tolist()}]}
    return img, mask, info


class CardTests(unittest.TestCase):
    def test_cards_are_stable_and_skip_fragments(self):
        _, mask, info = _frame()
        ids = [c.id for c in frame_cards(STEM, mask, info)]
        self.assertEqual(ids, [f"{STEM}#s0", f"{STEM}#s1", f"{STEM}#r0"])
        self.assertEqual(ids, [c.id for c in frame_cards(STEM, mask, info)])
        x, y, w, h = frame_cards(STEM, mask, info)[2].bbox
        self.assertTrue(x <= 430 and x + w >= 460 and y <= 60 and y + h >= 340)

    def test_votes_become_tuft_background_ignore(self):
        _, mask, info = _frame()
        votes = {f"{STEM}#s0": "y", f"{STEM}#r0": "n"}       # s1 unvoted
        tuft, ignore, counts = build_frame(STEM, mask, info, votes, dilate=3)
        self.assertEqual(counts, {"y": 1, "n": 1, "u": 0, "unvoted": 1})
        self.assertTrue(tuft[200, 110])                       # voted tuft
        self.assertFalse(tuft[200, 260] or tuft[200, 445])
        self.assertTrue(ignore[200, 260] and ignore[200, 265])  # unvoted tuft + dilation
        self.assertFalse(ignore[200, 445])                    # voted-no = background
        self.assertTrue(ignore[381, 11])                      # fragment without a card
        self.assertFalse(ignore[200, 550] or tuft[200, 550])  # far from everything: background
        self.assertFalse((tuft & ignore).any())

    def test_rejected_voted_yes_uses_its_outline(self):
        _, mask, info = _frame()
        tuft, _, _ = build_frame(STEM, mask, info, {f"{STEM}#r0": "y"})
        self.assertTrue(tuft[200, 445])

    def test_unsure_is_ignore(self):
        _, mask, info = _frame()
        tuft, ignore, counts = build_frame(STEM, mask, info, {f"{STEM}#s1": "u"}, dilate=0)
        self.assertEqual(counts["u"], 1)
        self.assertTrue(ignore[200, 260] and not tuft[200, 260])


class VoteLogTests(unittest.TestCase):
    def test_last_record_wins_and_null_undoes(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "votes.jsonl"
            append_vote(log, "a#s0", "y", 0.8)
            append_vote(log, "a#s1", "n")
            append_vote(log, "a#s0", "n")
            append_vote(log, "b#s0", "u")
            append_vote(log, "a#s1", None)
            self.assertEqual(load_votes(log), {"a#s0": "n", "b#s0": "u"})
            self.assertEqual(stems_with_votes(load_votes(log)), ["a"])  # unsure alone adds no frame
            self.assertEqual(load_votes(Path(d) / "missing.jsonl"), {})


class BuildToolTests(unittest.TestCase):
    def test_build_writes_label_and_ignore(self):
        from tools import build_from_votes
        img, mask, info = _frame()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_frame(d / "proposed", STEM, img, mask, tufts_from_mask(mask),
                       {"clip_id": "clipA", "frame_idx": 10, "split": "train", "use": "tufts", "source": "sam2"},
                       rejected=info["rejected_candidates"])
            append_vote(d / "votes.jsonl", f"{STEM}#s0", "y")
            append_vote(d / "votes.jsonl", f"{STEM}#r0", "n")
            rc = build_from_votes.main(["--data", str(d / "proposed"), "--votes", str(d / "votes.jsonl"),
                                        "--out", str(d / "voted")])
            self.assertEqual(rc, 0)
            _, m, out_info, meta = load_frame(d / "voted", STEM)
            self.assertEqual(int(m[200, 110]), 255)
            self.assertEqual(int(m[200, 260]), 0)
            self.assertEqual(len(out_info["tufts"]), 1)
            self.assertEqual(meta["source"], "votes")
            self.assertEqual(meta["clip_id"], "clipA")
            self.assertEqual(int(load_ignore(d / "voted", STEM)[200, 260]), 255)
            self.assertTrue((d / "voted_sheet.jpg").exists())


class RenderTests(unittest.TestCase):
    def test_card_renders(self):
        from tools.vote_tufts import render_card
        img, mask, info = _frame()
        for c in frame_cards(STEM, mask, info):
            canvas = render_card(img, c, "status")
            self.assertEqual(canvas.ndim, 3)


class ProposalIgnoreTests(unittest.TestCase):
    def test_proposal_ignore_kept_except_votes(self):
        from tools import build_from_votes
        img, mask, info = _frame()
        prop_ignore = np.zeros(mask.shape, bool)
        prop_ignore[100:120, 500:520] = True      # missed tuft: stays ignore
        prop_ignore[195:205, 440:450] = True      # overlaps the voted-no candidate: background wins
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_frame(d / "proposed", STEM, img, mask, tufts_from_mask(mask),
                       {"clip_id": "clipA", "frame_idx": 10, "split": "train", "use": "tufts"},
                       rejected=info["rejected_candidates"], ignore=prop_ignore)
            append_vote(d / "votes.jsonl", f"{STEM}#s0", "y")
            append_vote(d / "votes.jsonl", f"{STEM}#r0", "n")
            build_from_votes.main(["--data", str(d / "proposed"), "--votes", str(d / "votes.jsonl"),
                                   "--out", str(d / "voted"), "--exclude-clips", "other"])
            ig = load_ignore(d / "voted", STEM)
            self.assertEqual(int(ig[110, 510]), 255)
            self.assertEqual(int(ig[200, 445]), 0)

    def test_exclude_clips_handles_double_underscore_ids(self):
        from tools import build_from_votes
        img, mask, info = _frame()
        stem = "yt__abc__f000001"
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_frame(d / "proposed", stem, img, mask, tufts_from_mask(mask), {"clip_id": "yt__abc"})
            append_vote(d / "votes.jsonl", f"{stem}#s0", "y")
            rc = build_from_votes.main(["--data", str(d / "proposed"), "--votes", str(d / "votes.jsonl"),
                                        "--out", str(d / "voted"), "--exclude-clips", "yt__abc"])
            self.assertEqual(rc, 1)   # nothing left to build


class ColorBackendTests(unittest.TestCase):
    def test_orange_line_on_olive(self):
        from src.backends import ColorBackend
        from tools.auto_label import color_ignore
        img = np.zeros((200, 300, 3), np.uint8)
        img[:] = cv2.cvtColor(np.uint8([[[24, 70, 138]]]), cv2.COLOR_HSV2BGR)[0, 0]   # olive paint
        cv2.line(img, (20, 100), (200, 110), cv2.cvtColor(np.uint8([[[6, 180, 210]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist(), 4)
        m = ColorBackend().predict(img).tuft
        self.assertEqual(int(m[105, 100]), 255)
        self.assertEqual(int(m[20, 20]), 0)
        ig = color_ignore(img, np.zeros(img.shape[:2], np.uint8), {"ignore_color": {"s_min": 80, "v_min": 90}})
        self.assertTrue(ig[105, 100] and not ig[20, 20])
        self.assertIsNone(color_ignore(img, m, {}))


if __name__ == "__main__":
    unittest.main()
