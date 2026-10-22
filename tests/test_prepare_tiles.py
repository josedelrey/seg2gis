import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts import prepare_tiles


class PrepareTilesTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)
        self.output = self.root / "data" / "tiles"
        self.output.mkdir(parents=True)
        self.marker = self.output / "existing.png"
        self.marker.write_bytes(b"keep existing tiles")
        self.image = self.root / "image.tif"
        self.mask = self.root / "mask.tif"
        cv2.imwrite(str(self.image), np.zeros((4, 4, 3), dtype=np.uint8))
        cv2.imwrite(str(self.mask), np.full((4, 4), 255, dtype=np.uint8))

        # Exercise the real path guard in an isolated temporary repository.
        fake_script = self.root / "scripts" / "prepare_tiles.py"
        self.script_patch = patch.object(prepare_tiles, "__file__", str(fake_script))
        self.script_patch.start()
        self.addCleanup(self.script_patch.stop)

    def run_preparation(self, pairs=None, tile_size=2, stride=2):
        prepare_tiles.prepare_tiles(
            self.output,
            {"train": [(self.image, self.mask)] if pairs is None else pairs},
            [],
            tile_size,
            stride,
            protected_paths=(self.image, self.mask),
        )

    def test_success_replaces_old_tiles_with_complete_new_set(self):
        self.run_preparation()
        self.assertFalse(self.marker.exists())
        images = sorted((self.output / "train" / "images").glob("*.png"))
        masks = sorted((self.output / "train" / "masks").glob("*.png"))
        self.assertEqual(len(images), 4)
        self.assertEqual([p.name for p in images], [p.name for p in masks])
        np.testing.assert_array_equal(
            cv2.imread(str(masks[0]), 0), np.full((2, 2), 255)
        )
        self.assertEqual(list(self.output.parent.glob(".tiles-*")), [])

    def test_unreadable_input_keeps_existing_tiles(self):
        self.mask.write_bytes(b"corrupt image")
        with self.assertRaisesRegex(RuntimeError, "Could not read mask"):
            self.run_preparation()
        self.assertEqual(self.marker.read_bytes(), b"keep existing tiles")

    def test_dimension_mismatch_keeps_existing_tiles(self):
        cv2.imwrite(str(self.mask), np.zeros((3, 4), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.run_preparation()
        self.assertTrue(self.marker.exists())

    def test_write_failure_keeps_existing_tiles(self):
        with patch.object(prepare_tiles.cv2, "imwrite", return_value=False):
            with self.assertRaisesRegex(OSError, "Could not write tile"):
                self.run_preparation()
        self.assertTrue(self.marker.exists())

    def test_failed_replacement_restores_old_tiles(self):
        replace = os.replace

        def fail_install(source, destination):
            if Path(source).name == "tiles" and Path(source) != self.output:
                raise OSError("failed to install generated tiles")
            return replace(source, destination)

        with patch.object(prepare_tiles.os, "replace", side_effect=fail_install):
            with self.assertRaisesRegex(OSError, "failed to install"):
                self.run_preparation()
        self.assertEqual(self.marker.read_bytes(), b"keep existing tiles")

    def test_invalid_tiling_keeps_existing_tiles(self):
        for tile_size, stride in [(0, 1), (2, 0), (2, 3)]:
            with self.subTest(tile_size=tile_size, stride=stride):
                with self.assertRaisesRegex(ValueError, "stride"):
                    self.run_preparation(tile_size=tile_size, stride=stride)
                self.assertTrue(self.marker.exists())

    def test_missing_source_pairs_are_checked_before_preparation(self):
        with (
            patch.object(
                prepare_tiles,
                "parse_args",
                return_value=prepare_tiles.argparse.Namespace(
                    config="configs/default.json",
                    image_dir=None,
                    mask_dir=None,
                    test_image_dir=None,
                    out_dir=str(self.output),
                    tile_size=None,
                    stride=None,
                    train_image_ids=None,
                    val_image_ids=None,
                    test_image_ids=None,
                ),
            ),
            patch.object(
                prepare_tiles,
                "collect_image_mask_pairs",
                side_effect=RuntimeError("missing pairs"),
            ),
            patch.object(prepare_tiles, "prepare_tiles") as generate,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing pairs"):
                prepare_tiles.main()
        generate.assert_not_called()
        self.assertTrue(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
