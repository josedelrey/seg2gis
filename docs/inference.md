# Inference guide

Inference uses CUDA when available, otherwise CPU.

The pretrained config is tuned for the released checkpoint.
To use a different checkpoint, set `--model_path` and choose a config that
matches its architecture and encoder. Start with
[configs/default.json](../configs/default.json).

## Outputs

Files are written to `results/full_predictions/` by default. Use `--out_dir`
to choose another directory. With `--output_name "prediction"`, inference writes:

| File | Contents |
| --- | --- |
| `prediction_prob.npy` | Probability map as a NumPy array |
| `prediction_prob.png` | Probability preview |
| `prediction_mask.png` | Thresholded binary mask |
| `prediction_clean_mask.png` | Mask after component filtering and morphological opening |
| `prediction_polygons_overlay.png` | Simplified polygon outlines over the input image |
| `prediction_buildings.geojson` | Building polygons with area and vertex-count attributes |

GeoJSON preserves the source raster's coordinate reference system (CRS). For
RFC 7946 GeoJSON, reproject the polygons to WGS84 longitude/latitude first.
PNG and NumPy outputs use image pixel coordinates. Use `--no_export_vectors`
for raster-only output.

## Settings

Command-line options override configuration values. The pretrained config uses:

| Option | Value | Meaning |
| --- | ---: | --- |
| `--tile_size` | `256` | Inference tile width and height in pixels |
| `--stride` | `128` | Tile step in pixels. Predictions from overlapping tiles are averaged. |
| `--threshold` | `0.47` | Probability cutoff for the building mask |
| `--min_area` | `100` | Minimum connected-component size in pixels |
| `--open_kernel_size` | `3` | Morphological opening kernel width and height in pixels |
| `--polygon_min_area` | `0` | Minimum contour area in square pixels before simplification |
| `--epsilon_ratio` | `0.002` | Simplification tolerance as a fraction of contour perimeter |
| `--vector_min_area` | `0` | Minimum exported polygon area in squared CRS units |

`--vector_min_area` uses the raster CRS's squared units. For rasters with
latitude/longitude coordinates, leave it at `0` or reproject to a suitable
projected CRS before filtering by area.
