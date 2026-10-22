import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin
from shapely.geometry import shape

from scripts.predict_full_image import (
    build_output_paths,
    generate_polygon_overlay,
    polygon_to_geojson_feature,
    save_geojson_from_polygons,
)
from seg2gis.vectorize import mask_to_pixel_polygons


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


class PolygonExportTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((10, 10), dtype=np.uint8)
        self.mask[2:5, 3:7] = 1
        self.transform = from_origin(100, 200, 2, 2)

    def test_rectangle_preserves_pixel_edges_and_area(self):
        polygon = mask_to_pixel_polygons(self.mask)[0]
        feature = polygon_to_geojson_feature(polygon, self.transform, min_area=0)

        self.assertEqual(feature["properties"]["area_px"], 12)
        self.assertEqual(feature["properties"]["area"], 48)
        self.assertEqual(shape(feature["geometry"]).bounds, (106, 190, 114, 196))

    def test_courtyard_survives_simplification_and_geojson_round_trip(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:9, 1:9] = 1
        mask[3:7, 3:7] = 0
        _, polygons = generate_polygon_overlay(
            SimpleNamespace(polygon_min_area=0, epsilon_ratio=0.002),
            np.zeros((10, 10, 3), dtype=np.uint8),
            mask,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            raster_path = Path(temporary_dir) / "source.tif"
            output_path = Path(temporary_dir) / "footprints.geojson"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=1,
                dtype="uint8",
                crs="EPSG:3857",
                transform=self.transform,
            ) as dataset:
                dataset.write(mask, 1)
            save_geojson_from_polygons(polygons, raster_path, output_path, 0, False)
            feature = json.loads(output_path.read_text())["features"][0]

        geometry = shape(feature["geometry"])
        self.assertTrue(geometry.is_valid)
        self.assertEqual(len(geometry.interiors), 1)
        self.assertEqual(geometry.area, 192)

    def test_rotated_raster_uses_full_affine_transform(self):
        transform = Affine(2, 0.5, 100, 0.25, -3, 200)
        polygon = mask_to_pixel_polygons(self.mask)[0]
        feature = polygon_to_geojson_feature(polygon, transform, min_area=0)
        self.assertAlmostEqual(feature["properties"]["area"], 12 * 6.125)

    def test_single_pixels_and_diagonal_neighbors_remain_valid(self):
        polygons = mask_to_pixel_polygons(np.eye(2, dtype=np.uint8))
        self.assertEqual(len(polygons), 2)
        self.assertTrue(
            all(polygon.is_valid and polygon.area == 1 for polygon in polygons)
        )

    def test_area_filters_use_pixel_and_crs_units(self):
        self.assertEqual(mask_to_pixel_polygons(self.mask, min_area=13), [])
        polygon = mask_to_pixel_polygons(self.mask, min_area=12)[0]
        self.assertIsNone(polygon_to_geojson_feature(polygon, self.transform, 49))
        self.assertIsNotNone(polygon_to_geojson_feature(polygon, self.transform, 48))

    def test_empty_mask_produces_no_polygons(self):
        self.assertEqual(mask_to_pixel_polygons(np.zeros((3, 3), dtype=np.uint8)), [])


if __name__ == "__main__":
    unittest.main()
