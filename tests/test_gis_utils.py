import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from seg2gis.gis_utils import load_rgb_image


class LoadRgbImageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.image_path = Path(self.temp_dir.name) / "image.tif"

    def test_preserves_uint8_values_and_converts_to_rgb(self):
        image_bgr = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        self.assertTrue(cv2.imwrite(str(self.image_path), image_bgr))

        image_rgb = load_rgb_image(self.image_path)

        np.testing.assert_array_equal(image_rgb, image_bgr[..., ::-1])

    def test_scales_uint16_values_independently_per_band(self):
        image_bgr = np.array(
            [
                [[1000, 100, 10], [2000, 400, 20]],
                [[3000, 700, 30], [4000, 1000, 40]],
            ],
            dtype=np.uint16,
        )
        self.assertTrue(cv2.imwrite(str(self.image_path), image_bgr))

        image_rgb = load_rgb_image(self.image_path)

        expected_rgb = np.stack(
            [
                cv2.normalize(
                    image_bgr[..., channel],
                    None,
                    alpha=0,
                    beta=255,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U,
                )
                for channel in (2, 1, 0)
            ],
            axis=-1,
        )
        self.assertEqual(image_rgb.dtype, np.uint8)
        np.testing.assert_array_equal(image_rgb, expected_rgb)

    def test_rejects_single_band_images(self):
        grayscale = np.arange(4, dtype=np.uint16).reshape(2, 2)
        self.assertTrue(cv2.imwrite(str(self.image_path), grayscale))

        with self.assertRaisesRegex(ValueError, "three- or four-channel"):
            load_rgb_image(self.image_path)


if __name__ == "__main__":
    unittest.main()
