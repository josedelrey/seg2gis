import unittest

import numpy as np

from src.postprocess import (
    morphological_open,
    postprocess_mask,
    remove_small_components,
)


class PostprocessTests(unittest.TestCase):
    def test_remove_small_components_preserves_large_regions(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1, 1] = 1
        mask[5:8, 5:8] = 1

        cleaned = remove_small_components(mask, min_area=4)

        self.assertEqual(cleaned[1, 1], 0)
        self.assertEqual(int(cleaned.sum()), 9)

    def test_morphological_open_removes_isolated_pixel(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 1

        opened = morphological_open(mask, kernel_size=3)

        self.assertEqual(int(opened.sum()), 0)

    def test_disabled_operations_only_binarize_input(self):
        mask = np.array([[0, 2], [255, 0]], dtype=np.uint8)

        cleaned = postprocess_mask(
            mask,
            min_area=0,
            open_kernel_size=0,
        )

        np.testing.assert_array_equal(
            cleaned,
            np.array([[0, 1], [1, 0]], dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()
