import importlib.util
import unittest

import cv2
import numpy as np

HAS_ALBU = importlib.util.find_spec("albumentations") is not None

GEOMETRY_ONLY = {k: 0.0 for k in ("brightness_contrast_p", "gamma_p", "hue_p", "shadow_gradient_p",
                                 "shadow_polygon_p", "glare_p", "blur_p", "noise_p", "compression_p")}


def _pair(n):
    """Gray surface with white strings; label = the same strings in channel 0."""
    rng = np.random.default_rng(0)
    img = np.full((n, n, 3), 120, np.uint8)
    lab = np.zeros((n, n, 3), np.uint8)
    for _ in range(12):
        p, q = rng.integers(0, n, 2), rng.integers(0, n, 2)
        cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), (250, 250, 250), 3)
        cv2.line(lab, tuple(map(int, p)), tuple(map(int, q)), (255, 255, 255), 3)
    return img, lab


@unittest.skipUnless(HAS_ALBU, "albumentations not installed")
class TransformTests(unittest.TestCase):
    def test_geometry_keeps_image_and_all_label_channels_aligned(self):
        from src.transforms import build_train_transform
        tf, window = build_train_transform(128, dict(GEOMETRY_ONLY, affine_p=1.0, elastic_p=1.0))
        img, lab = _pair(window)
        for seed in range(30):
            tf.set_random_seed(seed)
            out = tf(image=img, mask=lab)
            i2, l2 = out["image"], out["mask"]
            self.assertEqual(i2.shape, (128, 128, 3))
            self.assertEqual(l2.shape, (128, 128, 3))
            self.assertTrue(set(np.unique(l2)) <= {0, 255})
            from_img = i2[..., 0] > 185
            from_lab = l2[..., 0] > 127
            iou = (from_img & from_lab).sum() / max((from_img | from_lab).sum(), 1)
            self.assertGreater(iou, 0.8, f"seed {seed}")
            np.testing.assert_array_equal(l2[..., 0], l2[..., 1])  # channels move together

    def test_full_pipeline_is_seeded_and_never_changes_resolution(self):
        from src.transforms import build_train_transform
        tf, window = build_train_transform(96)
        self.assertGreater(window, 96)
        img, lab = _pair(window)
        tf.set_random_seed(7); a = tf(image=img, mask=lab)
        tf.set_random_seed(7); b = tf(image=img, mask=lab)
        np.testing.assert_array_equal(a["image"], b["image"])
        np.testing.assert_array_equal(a["mask"], b["mask"])
        self.assertEqual(a["image"].shape[:2], (96, 96))

    def test_photometric_ops_change_pixels_but_not_the_mask(self):
        from src.transforms import GlareBlobs, ShadowGradient
        import albumentations as A
        img, lab = _pair(128)
        for op in (ShadowGradient(p=1.0), GlareBlobs(p=1.0)):
            tf = A.Compose([op], seed=3)
            out = tf(image=img, mask=lab)
            self.assertGreater(np.abs(out["image"].astype(int) - img).sum(), 0)
            np.testing.assert_array_equal(out["mask"], lab)

    def test_window_covers_rotation_and_zoom_out(self):
        from src.transforms import window_factor
        self.assertGreaterEqual(window_factor({"rotate_deg": 15, "scale": [0.8, 1.25]}),
                                (np.cos(np.radians(15)) + np.sin(np.radians(15))) / 0.8)
        self.assertAlmostEqual(window_factor({"rotate_deg": 0, "scale": [1, 1]}), 1.02)


if __name__ == "__main__":
    unittest.main()
