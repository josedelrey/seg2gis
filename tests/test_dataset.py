import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from seg2gis.dataset import (
    BuildingDataset,
    collect_image_mask_pairs,
    describe_image_ids,
    image_id_list,
    parse_inria_name,
)


class InriaDatasetHelpersTests(unittest.TestCase):
    def test_parse_inria_name(self):
        self.assertEqual(parse_inria_name("/data/Tyrol-w12.tif"), ("tyrol-w", 12))

    def test_parse_inria_name_rejects_missing_id(self):
        with self.assertRaisesRegex(ValueError, "Could not parse"):
            parse_inria_name("austin.tif")

    def test_image_id_conversion_and_description(self):
        self.assertEqual(image_id_list("1, 3,5"), [1, 3, 5])
        self.assertEqual(image_id_list((2, 4)), [2, 4])
        self.assertIsNone(image_id_list(None))
        self.assertEqual(describe_image_ids([1, 3, 5]), "1,3,5")

    def test_collect_image_mask_pairs_filters_and_orders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            mask_dir = root / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()

            for stem in ("austin2", "austin1", "chicago1"):
                (image_dir / f"{stem}.tif").touch()
                (mask_dir / f"{stem}.tif").touch()

            pairs = collect_image_mask_pairs(
                image_dir,
                mask_dir,
                image_ids=[1, 2],
                cities=["austin"],
            )

            self.assertEqual(
                [Path(image_path).stem for image_path, _ in pairs],
                ["austin1", "austin2"],
            )

    def test_collect_image_mask_pairs_reports_missing_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            mask_dir = root / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            (image_dir / "austin1.tif").touch()

            with self.assertRaisesRegex(RuntimeError, "Missing mask"):
                collect_image_mask_pairs(
                    image_dir,
                    mask_dir,
                    image_ids=[1],
                    cities=["austin"],
                )


class BuildingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.image_dir = root / "images"
        self.mask_dir = root / "masks"
        self.image_dir.mkdir()
        self.mask_dir.mkdir()

    def test_loads_rgb_image_and_binary_mask_without_transform(self):
        image_bgr = np.zeros((3, 4, 3), dtype=np.uint8)
        image_bgr[..., 2] = 255
        mask = np.array(
            [
                [0, 127, 128, 255],
                [0, 0, 255, 255],
                [127, 128, 0, 255],
            ],
            dtype=np.uint8,
        )
        cv2.imwrite(str(self.image_dir / "tile1.png"), image_bgr)
        cv2.imwrite(str(self.mask_dir / "tile1.png"), mask)

        image_tensor, mask_tensor = BuildingDataset(
            self.image_dir,
            self.mask_dir,
        )[0]

        self.assertEqual(tuple(image_tensor.shape), (3, 3, 4))
        self.assertEqual(tuple(mask_tensor.shape), (1, 3, 4))
        self.assertEqual(image_tensor[0, 0, 0].item(), 255.0)
        self.assertEqual(image_tensor[1, 0, 0].item(), 0.0)
        np.testing.assert_array_equal(
            mask_tensor.numpy(),
            (mask > 127).astype(np.float32)[None, ...],
        )

    def test_rejects_mismatched_tile_names(self):
        cv2.imwrite(
            str(self.image_dir / "image_tile.png"),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
        cv2.imwrite(
            str(self.mask_dir / "mask_tile.png"),
            np.zeros((2, 2), dtype=np.uint8),
        )

        with self.assertRaisesRegex(RuntimeError, "basename mismatch"):
            BuildingDataset(self.image_dir, self.mask_dir)


if __name__ == "__main__":
    unittest.main()
