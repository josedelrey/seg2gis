import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gis_utils import (  # noqa: E402
    pad_image_to_tile_grid,
    predict_full_image_tiled,
    threshold_probability_map,
)


class ZeroLogitModel(torch.nn.Module):
    def forward(self, inputs):
        batch_size, _, height, width = inputs.shape
        return torch.zeros(
            (batch_size, 1, height, width),
            dtype=inputs.dtype,
            device=inputs.device,
        )


class TiledInferenceTests(unittest.TestCase):
    def test_padding_aligns_image_to_tile_grid(self):
        image = np.zeros((37, 45, 3), dtype=np.uint8)

        padded, original_height, original_width = pad_image_to_tile_grid(
            image,
            tile_size=16,
            stride=8,
        )

        self.assertEqual(padded.shape, (40, 48, 3))
        self.assertEqual((original_height, original_width), (37, 45))

    def test_tiled_prediction_preserves_shape_and_averages_overlaps(self):
        image = np.full((7, 9, 3), 127, dtype=np.uint8)

        probabilities = predict_full_image_tiled(
            model=ZeroLogitModel(),
            image_rgb=image,
            tile_size=4,
            stride=2,
            device="cpu",
        )

        self.assertEqual(probabilities.shape, image.shape[:2])
        np.testing.assert_allclose(probabilities, 0.5, rtol=0, atol=1e-7)

    def test_stride_larger_than_tile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stride should be <= tile_size"):
            predict_full_image_tiled(
                model=ZeroLogitModel(),
                image_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                tile_size=4,
                stride=5,
                device="cpu",
            )

    def test_threshold_is_inclusive_and_validated(self):
        probabilities = np.array([[0.49, 0.5, 0.51]], dtype=np.float32)

        np.testing.assert_array_equal(
            threshold_probability_map(probabilities, 0.5),
            np.array([[0, 1, 1]], dtype=np.uint8),
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            threshold_probability_map(probabilities, 1.1)


if __name__ == "__main__":
    unittest.main()
