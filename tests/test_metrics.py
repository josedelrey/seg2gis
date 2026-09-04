import unittest

import numpy as np

from src.metrics import (
    boundary_metrics,
    boundary_metrics_multi,
    confusion_from_masks,
    metrics_from_confusion,
)


class SegmentationMetricsTests(unittest.TestCase):
    def test_confusion_and_derived_metrics(self):
        prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        target = np.array([[1, 0], [1, 0]], dtype=np.uint8)

        confusion = confusion_from_masks(prediction, target)
        metrics = metrics_from_confusion(*confusion)

        self.assertEqual(confusion, (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["iou_building"], 1 / 3, places=6)
        self.assertAlmostEqual(metrics["dice_f1"], 0.5, places=6)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=6)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 0.5, places=6)

    def test_identical_masks_have_perfect_boundary_metrics(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[3:9, 4:10] = 1

        metrics = boundary_metrics(mask, mask, tolerance_px=0)

        for value in metrics.values():
            self.assertEqual(value, 1.0)

    def test_boundary_tolerance_improves_shifted_match(self):
        target = np.zeros((16, 16), dtype=np.uint8)
        prediction = np.zeros_like(target)
        target[4:10, 4:10] = 1
        prediction[5:11, 5:11] = 1

        exact = boundary_metrics(prediction, target, tolerance_px=0)
        tolerant = boundary_metrics(prediction, target, tolerance_px=2)

        self.assertGreater(tolerant["boundary_f1"], exact["boundary_f1"])

    def test_empty_masks_have_perfect_boundary_metrics(self):
        empty = np.zeros((8, 8), dtype=np.uint8)

        metrics = boundary_metrics_multi(empty, empty, tolerances=(1, 3))

        self.assertTrue(all(value == 1.0 for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
