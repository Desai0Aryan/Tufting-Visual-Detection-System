import importlib.util
import unittest

import cv2
import numpy as np

from src.metrics import ClDiceMeter, SegMeter

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _line(width=5, gap=0, shave=0):
    m = np.zeros((64, 128), np.uint8)
    cv2.line(m, (10, 32), (118, 32), 1, width)
    if gap:
        m[:, 64 - gap // 2:64 + (gap - gap // 2)] = 0
    if shave:
        m[32 - width // 2:32 - width // 2 + shave, :] = 0
    return m


class HardClDiceTests(unittest.TestCase):
    def test_perfect_and_empty(self):
        m = ClDiceMeter(); m.update(_line(), _line())
        self.assertAlmostEqual(m.cldice, 1.0)
        e = ClDiceMeter(); e.update(np.zeros((8, 8)), np.zeros((8, 8)))
        self.assertEqual(e.cldice, 1.0)

    def test_gap_hurts_cldice_more_than_dice(self):
        gt = _line(width=5)
        broken, shaved = _line(width=5, gap=10), _line(width=5, shave=1)
        d_b, d_s = SegMeter(), SegMeter()
        d_b.update(broken, gt); d_s.update(shaved, gt)
        c_b, c_s = ClDiceMeter(), ClDiceMeter()
        c_b.update(broken, gt); c_s.update(shaved, gt)
        self.assertGreater(d_b.dice, d_s.dice - 0.05)   # pixel Dice barely tells them apart
        self.assertLess(c_b.cldice, c_s.cldice)          # clDice penalises the break


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class SoftLossTests(unittest.TestCase):
    def _t(self, a):
        import torch
        return torch.from_numpy(a.astype(np.float32))[None, None]

    def test_soft_skeleton_is_thin_and_on_the_line(self):
        from src.losses import soft_skeleton
        x = self._t(_line(width=7))
        sk = soft_skeleton(x, iters=10)
        self.assertGreater(float(sk.sum()), 50)
        self.assertLess(float(sk.sum()), 0.35 * float(x.sum()))
        self.assertTrue(bool(((sk > 0.5) <= (x > 0.5)).all()))

    def test_cldice_near_zero_for_perfect_and_larger_for_broken(self):
        from src.losses import soft_cldice_loss
        gt = self._t(_line(width=5))
        perfect = (gt * 2 - 1) * 20.0        # logits that sigmoid back to the mask
        broken = (self._t(_line(width=5, gap=12)) * 2 - 1) * 20.0
        self.assertLess(float(soft_cldice_loss(perfect, gt)), 0.02)
        self.assertGreater(float(soft_cldice_loss(broken, gt)), float(soft_cldice_loss(perfect, gt)) + 0.02)

    def test_segloss_terms_and_gradients(self):
        import torch
        from src.losses import SegLoss
        tgt = torch.cat([self._t(_line()), self._t(_line(width=3))], 1)
        logits = torch.zeros(1, 2, 64, 128, requires_grad=True)
        total, parts = SegLoss(w_cldice=0.5)(logits, tgt, root_valid=torch.ones(1, 1, 64, 128))
        self.assertEqual(set(parts), {"bce", "dice", "cldice", "root"})
        total.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        total1, parts1 = SegLoss()(logits[:, :1].detach(), tgt[:, :1])
        self.assertEqual(set(parts1), {"bce", "dice"})  # 1 channel, clDice off by default

    def test_empty_target_penalises_false_positives(self):
        import torch
        from src.losses import SegLoss
        tgt = torch.zeros(2, 1, 32, 32)
        quiet, loud = torch.full((2, 1, 32, 32), -8.0), torch.full((2, 1, 32, 32), 4.0)
        self.assertLess(SegLoss()(quiet, tgt)[0], SegLoss()(loud, tgt)[0])



@unittest.skipUnless(HAS_TORCH, "torch not installed")
class IgnoreRegionTests(unittest.TestCase):
    def test_predictions_inside_ignore_do_not_change_the_loss(self):
        import torch
        from src.losses import SegLoss
        tgt = torch.zeros(1, 1, 32, 32); tgt[..., 10:12, 4:28] = 1
        valid = torch.ones(1, 2, 32, 32); valid[:, 1, 20:30, 20:30] = 0      # ignore box
        base = torch.where(tgt > 0, 6.0, -6.0)
        other = base.clone(); other[..., 20:30, 20:30] = 6.0                  # "tuft" predicted in ignore box
        for w_cl in (0.0, 0.5):
            f = SegLoss(w_cldice=w_cl)
            self.assertAlmostEqual(float(f(base, tgt, valid)[0]), float(f(other, tgt, valid)[0]), places=4)
            self.assertGreater(float(f(other, tgt)[0]), float(f(base, tgt)[0]) + 0.01)  # without ignore it counts



@unittest.skipUnless(HAS_TORCH, "torch not installed")
class SourceWeightTests(unittest.TestCase):
    def test_half_weight_halves_bce_even_in_a_uniform_batch(self):
        import torch
        from src.losses import bce_loss
        logits, tgt = torch.randn(2, 1, 16, 16), (torch.rand(2, 1, 16, 16) > 0.8).float()
        full = bce_loss(logits, tgt, 2.0, torch.ones_like(tgt))
        half = bce_loss(logits, tgt, 2.0, torch.full_like(tgt, 0.5))
        self.assertAlmostEqual(float(half), 0.5 * float(full), places=5)
        self.assertAlmostEqual(float(full), float(bce_loss(logits, tgt, 2.0)), places=5)


if __name__ == "__main__":
    unittest.main()
