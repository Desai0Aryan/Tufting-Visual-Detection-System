import importlib.util
import os
import tempfile
import unittest

import numpy as np

HAS = all(importlib.util.find_spec(m) for m in ("torch", "segmentation_models_pytorch"))


@unittest.skipUnless(HAS, "torch / segmentation_models_pytorch not installed")
class SmpUNetTests(unittest.TestCase):
    CFG = {"arch": "smp_unet", "encoder": "resnet18", "encoder_weights": None, "classes": 1}

    def _model(self, **over):
        import torch
        from src.unet import build_model
        torch.manual_seed(0)
        return build_model(dict(self.CFG, **over)).eval()

    def test_logits_shape_and_channel_count(self):
        import torch
        m1, m2 = self._model(), self._model(classes=2)
        x = torch.rand(2, 3, 64, 96)
        self.assertEqual(tuple(m1(x).shape), (2, 1, 64, 96))
        self.assertEqual(tuple(m2(x).shape), (2, 2, 64, 96))

    def test_imagenet_normalisation_is_inside_forward(self):
        import torch
        from src.unet import IMAGENET_MEAN, IMAGENET_STD
        m = self._model()
        x = torch.rand(1, 3, 64, 64)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        with torch.no_grad():
            torch.testing.assert_close(m(x), m.net((x - mean) / std))

    def test_encoder_is_configurable(self):
        m = self._model(encoder="resnet34")
        self.assertEqual(type(m.net.encoder).__name__, "ResNetEncoder")
        n18 = sum(p.numel() for p in self._model().parameters())
        n34 = sum(p.numel() for p in m.parameters())
        self.assertGreater(n34, n18)

    def test_checkpoint_roundtrip_and_native_resolution_output(self):
        import torch
        from src.unet import UNetBackend, load_model, save_checkpoint
        m = self._model()
        img = (np.random.default_rng(0).random((101, 157, 3)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            save_checkpoint(m, p, self.CFG)
            again = load_model(p)
            x = torch.rand(1, 3, 64, 64)
            with torch.no_grad():
                torch.testing.assert_close(m(x), again(x))
            be = UNetBackend(p, device="cpu", threshold=0.5)
            probs = be.predict_proba(img)
            self.assertEqual(probs.shape, (1, 101, 157))  # padded internally, never resized
            res = be.predict(img)
            self.assertEqual(res.tuft.shape, (101, 157))
            self.assertIsNone(res.root)

    def test_tiling_and_blending_reassemble_the_frame_exactly(self):
        """With a pointwise model (no spatial context) tiled output must equal
        the full-frame output, which checks tile placement, padding, batching
        and blend weights independently of receptive-field effects."""
        import torch
        from src.unet import UNetBackend, save_checkpoint

        class Pointwise(torch.nn.Module):
            pad_multiple = 32

            def forward(self, x):
                return (x * torch.tensor([3.0, -2.0, 1.0]).view(1, 3, 1, 1)).sum(1, keepdim=True) - 0.5

        img = (np.random.default_rng(1).random((200, 300, 3)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            save_checkpoint(self._model(), p, self.CFG)
            full_be = UNetBackend(p, device="cpu")
            tiled_be = UNetBackend(p, device="cpu", tile=100, overlap=40, batch_tiles=3)
            self.assertEqual(tiled_be.tile, 128)  # rounded up to the model's multiple
            full_be.model = tiled_be.model = Pointwise()
            full, tiled = full_be.predict_proba(img), tiled_be.predict_proba(img)
            np.testing.assert_allclose(full, tiled, atol=1e-5)

    def test_root_channel_is_returned_when_enabled(self):
        from src.unet import UNetBackend, save_checkpoint
        cfg = dict(self.CFG, classes=2)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            save_checkpoint(self._model(classes=2), p, cfg)
            res = UNetBackend(p, device="cpu").predict(np.zeros((64, 64, 3), np.uint8))
            self.assertEqual(res.root.shape, (64, 64))


if __name__ == "__main__":
    unittest.main()
