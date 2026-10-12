import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
class PlotRunsTests(unittest.TestCase):
    def test_curves_png_from_log_csv(self):
        from tools.plot_runs import domain_of, plot_curves, read_log
        with tempfile.TemporaryDirectory() as d:
            cols = ["epoch", "lr", "train_loss", "val_dlr_loss", "val_dlr_dice", "val_dlr_cldice@0.5"]
            for name, k in (("a", 1.0), ("b", 0.9)):
                Path(d, name).mkdir()
                with open(Path(d, name, "log.csv"), "w", newline="") as fh:
                    w = csv.writer(fh); w.writerow(cols)
                    for e in range(1, 6):
                        w.writerow([e, 1e-4, k / e, k / e, 0.5 + e / 20, 0.4 + e / 25])
            self.assertEqual(domain_of(read_log(Path(d, "a"))), "dlr")
            out = Path(d, "c.png")
            plot_curves([Path(d, "a"), Path(d, "b")], ["A", "B"], str(out))
            self.assertGreater(out.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
