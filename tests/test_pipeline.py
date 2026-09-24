import unittest
from pathlib import Path

import cv2

from detector import TuftDetector
from main import parse_args


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "Images" / "sample_tufts1.jpg"


class PipelineTests(unittest.TestCase):
    def test_new_and_legacy_cli_forms(self):
        args = parse_args(["--image", str(SAMPLE)])
        self.assertEqual(args.image, str(SAMPLE))

        args = parse_args([str(SAMPLE), "--out-dir", "outputs"])
        self.assertEqual(args.image, str(SAMPLE))
        self.assertEqual(args.outdir, "outputs")

    def test_detector_preserves_original_coordinate_system(self):
        image = cv2.imread(str(SAMPLE))
        detector = TuftDetector()
        tufts = detector.detect(image)

        self.assertGreater(len(tufts), 0)

        height, width = image.shape[:2]

        for tuft in tufts:
            self.assertGreaterEqual(tuft.root[0], 0)
            self.assertLess(tuft.root[0], width)
            self.assertGreaterEqual(tuft.root[1], 0)
            self.assertLess(tuft.root[1], height)

    def test_empty_csv_has_expected_columns(self):
        from main import _tufts_dataframe

        df = _tufts_dataframe([])

        self.assertEqual(
            list(df.columns),
            [
                "tuft_id",
                "root_x",
                "root_y",
                "tip_x",
                "tip_y",
                "unit_vector_x",
                "unit_vector_y",
                "angle_deg",
                "length_px",
            ],
        )

    def test_detector_rejects_invalid_roi_and_block_size(self):
        image = cv2.imread(str(SAMPLE))

        with self.assertRaises(ValueError):
            TuftDetector(block_size=10)

        with self.assertRaises(ValueError):
            TuftDetector(
                roi=(0, 0, image.shape[1] + 1, 10)
            ).detect(image)


if __name__ == "__main__":
    unittest.main()
