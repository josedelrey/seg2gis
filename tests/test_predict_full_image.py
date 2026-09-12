import unittest

from scripts.predict_full_image import build_output_paths


class OutputPathTests(unittest.TestCase):
    def test_builds_only_documented_output_paths(self):
        output_paths = build_output_paths("results", "prediction")

        self.assertEqual(
            set(output_paths),
            {
                "prob_npy",
                "prob_png",
                "raw_mask_png",
                "clean_mask_png",
                "polygon_overlay_png",
                "polygons_geojson",
            },
        )


if __name__ == "__main__":
    unittest.main()
