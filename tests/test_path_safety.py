import tempfile
import unittest
from pathlib import Path

from seg2gis.path_safety import resolve_safe_tile_output_dir


class ResolveSafeTileOutputDirTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo_root = Path(self.temp_dir.name) / "seg2gis"
        self.data_root = self.repo_root / "data"
        self.raw_images = self.data_root / "AerialImageDataset" / "train" / "images"
        self.raw_masks = self.data_root / "AerialImageDataset" / "train" / "gt"

    def resolve(self, output_path):
        return resolve_safe_tile_output_dir(
            output_path,
            self.repo_root,
            protected_paths=(self.raw_images, self.raw_masks),
        )

    def test_accepts_dedicated_tile_directory_inside_repo_data(self):
        output_path = self.data_root / "tiles_256_inria155"

        self.assertEqual(self.resolve(output_path), output_path.resolve())

    def test_rejects_repository_parent_directory(self):
        with self.assertRaisesRegex(ValueError, "dedicated subdirectory"):
            self.resolve(self.repo_root.parent)

    def test_rejects_repository_root(self):
        with self.assertRaisesRegex(ValueError, "dedicated subdirectory"):
            self.resolve(self.repo_root)

    def test_rejects_data_root(self):
        with self.assertRaisesRegex(ValueError, "dedicated subdirectory"):
            self.resolve(self.data_root)

    def test_rejects_directory_outside_repo_data(self):
        with self.assertRaisesRegex(ValueError, "dedicated subdirectory"):
            self.resolve(Path(self.temp_dir.name) / "external_tiles")

    def test_rejects_raw_input_directory(self):
        with self.assertRaisesRegex(ValueError, "overlaps an input directory"):
            self.resolve(self.raw_images)

    def test_rejects_descendant_of_raw_input_directory(self):
        with self.assertRaisesRegex(ValueError, "overlaps an input directory"):
            self.resolve(self.raw_images / "tiles")

    def test_rejects_ancestor_of_raw_input_directory(self):
        with self.assertRaisesRegex(ValueError, "overlaps an input directory"):
            self.resolve(self.data_root / "AerialImageDataset")


if __name__ == "__main__":
    unittest.main()
