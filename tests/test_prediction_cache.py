import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.prediction_cache import PredictionCache


class PredictionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.checkpoint_path = self.root / "model.pth"
        self.image_path = self.root / "image.tif"
        self.mask_path = self.root / "mask.tif"
        self.cache_path = self.root / "cache" / "image.npz"
        self.checkpoint_path.write_bytes(b"checkpoint-v1")
        self.image_path.write_bytes(b"image-v1")
        self.mask_path.write_bytes(b"mask-v1")
        self.prob_map = np.array([[0.1, 0.9]], dtype=np.float32)
        self.target_mask = np.array([[0, 1]], dtype=np.uint8)

    def new_cache(self, **kwargs):
        return PredictionCache(
            checkpoint_path=self.checkpoint_path,
            architecture=kwargs.get("architecture", "unet"),
            encoder=kwargs.get("encoder", "efficientnet-b3"),
            tile_size=kwargs.get("tile_size", 256),
            stride=kwargs.get("stride", 128),
            device=kwargs.get("device", "cpu"),
            refresh=kwargs.get("refresh", False),
        )

    def save_cache(self, cache):
        metadata = cache.metadata_for(self.image_path, self.mask_path)
        cache.save(
            self.cache_path,
            metadata,
            self.prob_map,
            self.target_mask,
        )
        return metadata

    def assert_cache_miss(self, cache):
        metadata = cache.metadata_for(self.image_path, self.mask_path)
        self.assertIsNone(cache.load(self.cache_path, metadata))

    def test_round_trip_returns_matching_cached_arrays(self):
        cache = self.new_cache()
        metadata = self.save_cache(cache)

        cached = cache.load(self.cache_path, metadata)

        self.assertIsNotNone(cached)
        cached_prob_map, cached_target_mask = cached
        np.testing.assert_array_equal(cached_prob_map, self.prob_map)
        np.testing.assert_array_equal(cached_target_mask, self.target_mask.astype(bool))

    def test_checkpoint_content_change_invalidates_cache(self):
        self.save_cache(self.new_cache())
        self.checkpoint_path.write_bytes(b"checkpoint-v2")

        self.assert_cache_miss(self.new_cache())

    def test_image_content_change_invalidates_cache(self):
        self.save_cache(self.new_cache())
        self.image_path.write_bytes(b"image-v2")

        self.assert_cache_miss(self.new_cache())

    def test_mask_content_change_invalidates_cache(self):
        self.save_cache(self.new_cache())
        self.mask_path.write_bytes(b"mask-v2")

        self.assert_cache_miss(self.new_cache())

    def test_tiling_change_invalidates_cache(self):
        self.save_cache(self.new_cache())

        self.assert_cache_miss(self.new_cache(stride=64))

    def test_model_configuration_change_invalidates_cache(self):
        self.save_cache(self.new_cache())

        self.assert_cache_miss(self.new_cache(architecture="fpn"))

    def test_refresh_forces_cache_miss(self):
        self.save_cache(self.new_cache())

        self.assert_cache_miss(self.new_cache(refresh=True))

    def test_legacy_cache_without_metadata_is_ignored(self):
        self.cache_path.parent.mkdir(parents=True)
        np.savez_compressed(
            self.cache_path,
            prob_map=self.prob_map,
            target_mask=self.target_mask,
        )

        self.assert_cache_miss(self.new_cache())

    def test_corrupt_cache_is_ignored(self):
        self.cache_path.parent.mkdir(parents=True)
        self.cache_path.write_bytes(b"not-a-valid-npz")

        self.assert_cache_miss(self.new_cache())


if __name__ == "__main__":
    unittest.main()
