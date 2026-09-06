import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from seg2gis.vectorize import (
    mask_to_contours,
    mask_to_geodataframe,
    save_vector_polygons,
    simplify_contours,
    validate_area_filter_crs,
)


class VectorizeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.raster_path = self.root / "source.tif"
        self.mask = np.zeros((10, 10), dtype=np.uint8)
        self.mask[2:5, 3:7] = 1

        with rasterio.open(
            self.raster_path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=3,
            dtype=np.uint8,
            crs="EPSG:3857",
            transform=from_origin(100, 200, 2, 2),
        ) as dataset:
            dataset.write(np.zeros((3, 10, 10), dtype=np.uint8))

    def test_contour_extraction_and_simplification(self):
        contours = mask_to_contours(self.mask, min_area=5)
        polygons = simplify_contours(contours, epsilon_ratio=0.01)

        self.assertEqual(len(contours), 1)
        self.assertEqual(len(polygons), 1)
        self.assertGreaterEqual(len(polygons[0]), 3)

    def test_geodataframe_uses_raster_transform_and_crs(self):
        gdf = mask_to_geodataframe(
            self.mask,
            self.raster_path,
            min_area=0,
        )

        self.assertEqual(len(gdf), 1)
        self.assertEqual(gdf.crs, CRS.from_epsg(3857))
        self.assertAlmostEqual(gdf.geometry.iloc[0].area, 48.0)
        np.testing.assert_allclose(gdf.total_bounds, [106, 190, 114, 196])

    def test_geographic_area_filter_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "geographic CRS"):
            validate_area_filter_crs(CRS.from_epsg(4326), min_area=10)

        validate_area_filter_crs(
            CRS.from_epsg(4326),
            min_area=10,
            allow_geographic_area=True,
        )

    def test_save_vector_polygons_writes_readable_geojson(self):
        output_path = self.root / "buildings.geojson"

        saved = save_vector_polygons(
            self.mask,
            self.raster_path,
            output_path,
            min_area=0,
        )
        loaded = gpd.read_file(output_path)

        self.assertTrue(output_path.is_file())
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.crs, CRS.from_epsg(3857))


if __name__ == "__main__":
    unittest.main()
