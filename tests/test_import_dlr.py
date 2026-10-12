import unittest

import numpy as np

from tools.import_dlr import clip_of, resolve_duplicates, session_of


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


class DuplicateIdTests(unittest.TestCase):
    def test_separate_same_id_polygons_are_kept_and_renamed(self):
        polys = [("T9", _rect(10, 10, 60, 14)), ("T9", _rect(200, 10, 260, 14)), ("T3", _rect(10, 50, 40, 54))]
        groups, dec = resolve_duplicates(polys, (100, 300))
        self.assertEqual(sorted(n for n, _ in groups), ["T3", "T9", "T9#2"])
        self.assertEqual(sorted(d["action"] for d in dec), ["kept", "kept", "kept_renamed"])

    def test_touching_same_id_polygons_are_merged(self):
        polys = [("T9", _rect(10, 10, 60, 14)), ("T9", _rect(61, 10, 120, 14))]  # 1 px gap: one tuft
        groups, dec = resolve_duplicates(polys, (100, 300))
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0][1]), 2)
        self.assertEqual(dec[0]["action"], "merged")

    def test_sessions_map_to_clip_ids(self):
        self.assertEqual(session_of("GP02_0037"), "GP02")
        self.assertEqual(clip_of("HABLEG", "GOPR_0090"), "dlr_hableg_GOPR")
        self.assertEqual(clip_of("EC135-ACT⁄FHS", "DSC_2409"), "dlr_ec135")


if __name__ == "__main__":
    unittest.main()
