import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from audit_data import audit_split


class AuditDataTests(unittest.TestCase):
    def make_triplet(self, root, split, name, mask_value=1):
        for part in ("A", "B", "label"):
            (root / split / part).mkdir(parents=True, exist_ok=True)
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:4, :4] = mask_value
        Image.fromarray(rgb).save(root / split / "A" / f"{name}.png")
        Image.fromarray(rgb).save(root / split / "B" / f"{name}.png")
        Image.fromarray(mask).save(root / split / "label" / f"{name}.png")

    def test_counts_binary_masks_with_zero_one_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_triplet(root, "train", "sample", mask_value=1)
            report, names, _ = audit_split(root, "train")
            self.assertEqual(names, {"sample"})
            self.assertEqual(report["changed_pixels"], 16)
            self.assertEqual(report["change_prevalence"], 16 / 256)

    def test_rejects_unmatched_triplets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_triplet(root, "train", "sample")
            (root / "train" / "B" / "sample.png").unlink()
            with self.assertRaises(ValueError):
                audit_split(root, "train")


if __name__ == "__main__":
    unittest.main()
