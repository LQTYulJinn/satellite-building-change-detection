import unittest

import torch

from change_detection.metrics import counts, scores
from change_detection.model import ChangeUNet


class PipelineTests(unittest.TestCase):
    def test_both_architectures_produce_one_logit_per_pixel(self):
        a = torch.rand(1, 3, 32, 32)
        b = torch.rand(1, 3, 32, 32)
        for architecture in ("early", "siamese"):
            with self.subTest(architecture=architecture):
                model = ChangeUNet(base=4, architecture=architecture)
                output = model(a, b)
                self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
                output.mean().backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_global_counts_and_f1_iou_relation(self):
        prediction = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
        target = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        tp, fp, fn = counts(prediction, target)
        result = scores(tp, fp, fn)
        self.assertEqual((tp, fp, fn), (1, 1, 1))
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertAlmostEqual(result["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
