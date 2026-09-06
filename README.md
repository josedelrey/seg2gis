# seg2gis

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f)

**From aerial imagery to candidate building footprints for GIS.**

seg2gis turns RGB aerial imagery into building masks and georeferenced GeoJSON
polygons. Train a segmentation model, run inference on full scenes, and tune
mask cleanup and polygon simplification for your imagery.

![Input image, probability map, cleaned mask, and polygon overlay](results/figures/building_footprint_showcase.png)

<p align="center"><em>Held-out aerial crop: input → probability map → cleaned mask → polygon overlay.</em></p>

## Contents

- [Installation](#installation)
- [Use your own imagery](#use-your-own-imagery)
- [INRIA experiments](#inria-experiments)
- [Reproduce the INRIA experiments](#reproduce-the-inria-experiments)
- [Tests](#tests)
- [Repository map](#repository-map)

## Pipeline

```text
RGB aerial scene → overlapping tile inference → probability map
                 → mask cleanup → contour extraction
                 → polygon simplification → GeoJSON
```

- Training with U-Net, FPN, and DeepLabV3+, geometric augmentation, and boundary-weighted loss.
- Configurable tiling, inference, mask cleanup, and polygon simplification.
- Raster, boundary, component, and vector-quality diagnostics.
- Georeferenced export using the source raster transform and coordinate reference system (CRS).

## Installation

Use Python 3.11. Commands below use PowerShell and run from the repository root.

Create the environment with Conda:

```powershell
conda env create -f environment.yml
conda activate seg2gis
```

For a pip-only environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The project is installed in editable mode, so source changes take effect without
reinstalling. Shared code is imported through `seg2gis`, for example
`from seg2gis.config import load_config`.

For CUDA acceleration, install a PyTorch 2.5 build supported by your NVIDIA
driver before installing the remaining dependencies.

## Use your own imagery

Start with an 8-bit RGB georeferenced raster. Download the
[pretrained U-Net EfficientNet-B3 checkpoint](https://github.com/josedelrey/seg2gis/releases/download/v1.0.0/phase2_unet_effb3_aug_boundary_bce_w2_e50.pth)
from release **v1.0.0** and save it, without renaming it, in
`models/phase2_augmentation/` (create the folder if needed).

Inference automatically uses CUDA when available, otherwise CPU.

```powershell
python scripts/predict_full_image.py `
  --config configs/pretrained_unet_effb3.json `
  --image_path "path/to/rgb-raster.tif" `
  --output_name "prediction"
```

The pretrained config uses the [validation-selected vector settings](#vector-results).
To train your own model, follow the [INRIA workflow](#reproduce-the-inria-experiments).
For another checkpoint, use `--model_path` and a config matching its architecture
and encoder; [configs/default.json](configs/default.json) is a starting point.

### Outputs

Files are written to `results/full_predictions/` by default; use `--out_dir`
to choose another directory. With `--output_name "prediction"`, the outputs are:

| File | Contents |
| --- | --- |
| `prediction_prob.npy` | Numerical probability map for further analysis |
| `prediction_prob.png` | Probability preview |
| `prediction_mask.png` | Thresholded binary mask |
| `prediction_clean_mask.png` | Mask after component filtering and morphological opening |
| `prediction_polygons_overlay.png` | Simplified polygon outlines over the input image |
| `prediction_showcase_crop.png` | Four-panel crop showing the pipeline stages |
| `prediction_buildings.geojson` | Building polygons with area and vertex-count attributes |

GeoJSON preserves the source raster's coordinate reference system (CRS).
For consumers requiring standard RFC 7946 GeoJSON, reproject the polygons to
WGS84 longitude/latitude first.
PNG and NumPy outputs use image pixel coordinates. Use `--no_export_vectors`
for raster-only output.

### Inference settings

Command-line options override the corresponding configuration values.
The main controls in `configs/default.json` are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--tile_size` | `256` | Inference tile width and height in pixels |
| `--stride` | `128` | Tile step in pixels; overlapping predictions are averaged |
| `--threshold` | `0.50` | Probability cutoff for the building mask |
| `--min_area` | `500` | Minimum connected-component size in pixels |
| `--open_kernel_size` | `5` | Morphological opening kernel width and height in pixels |
| `--polygon_min_area` | `150` | Minimum contour area in square pixels before simplification |
| `--epsilon_ratio` | `0.002` | Simplification tolerance as a fraction of contour perimeter |
| `--vector_min_area` | `150` | Minimum exported polygon area in squared CRS units |

Vector-area filtering uses squared units of a projected CRS, such as square
metres. For geographic-coordinate rasters, reproject or set `--vector_min_area 0`.

The defaults are tuned for INRIA imagery. For a new region or resolution,
evaluate a representative sample and adjust the model and post-processing settings.

## INRIA experiments

### Evaluation protocol

Experiments use the 180 labelled images from the
[INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/):
five cities, with 36 images of `5000 × 5000 px` per city. Scenes are split
before tiling, keeping training, validation, and evaluation spatially separate.

| Split | Image IDs per city | Scenes | Purpose |
| --- | --- | ---: | --- |
| Train | 11–36 | 130 | Model fitting |
| Validation | 6–10 | 25 | Model and post-processing selection |
| Held-out test | 1–5 | 25 | Final reporting |

Reported scores use a local holdout of the first five labelled scenes per city,
following the INRIA(155) convention.

### Segmentation results

The model selected on validation data is a **U-Net with an EfficientNet-B3
encoder**, geometric augmentation, and Dice plus boundary-weighted binary
cross-entropy.

The fixed baseline uses threshold `0.50`, minimum component area `500 px`, and
an opening kernel of `5 px`.

| Split | IoU | Dice/F1 | Precision | Recall | BF1 @2 px | BF1 @5 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.8016 | 0.8899 | 0.9052 | 0.8751 | 0.6246 | 0.8023 |
| Held-out test | 0.7876 | 0.8812 | 0.9064 | 0.8573 | 0.6307 | 0.7981 |

IoU and Dice measure area overlap; boundary F1 (BF1) measures outline alignment
at 2-pixel and 5-pixel tolerances.

### Vector results

The vector pipeline uses a separate configuration selected on validation data:
threshold `0.47`, minimum component area `100 px`, opening kernel `3 px`, and
Douglas–Peucker epsilon ratio `0.002`. These values are fixed before evaluating
the held-out split.

| Held-out metric | Result |
| --- | ---: |
| Cleaned-mask IoU | 0.8023 |
| Polygon-raster IoU | 0.7736 |
| Valid polygons | 99.46% |
| Predicted polygons | 25,564 |
| Reference connected components | 32,794 |
| Component AP @ IoU 0.50 | 0.6021 |
| Component AP @ IoU 0.75 | 0.3826 |

Detailed results:
[model selection](results/tables/phase2_augmentation_training_metrics.csv),
[full-image validation](results/tables/phase2_full_image_validation_metrics_by_city.csv),
[full-image test](results/tables/phase2_full_image_test_metrics_by_city.csv),
[post-processing selection](results/tables/postprocess_ablation_validation_summary.csv),
[vector quality](results/tables/vector_quality_test_best_val_config_summary.csv),
and [component diagnostics](results/tables/instance_ap_test_best_val_config_by_city.csv).

## Reproduce the INRIA experiments

### 1. Prepare the data and tiles

Download the INRIA dataset and arrange it as follows:

```text
data/AerialImageDataset/
  train/
    images/
    gt/
  test/
    images/
```

The labelled images under `train/` supply all three local splits. The optional
`test/` directory holds the benchmark's unlabelled scenes for additional inference.

```powershell
python scripts/prepare_tiles.py --config configs/default.json
```

This applies the scene-level split above, then extracts `256 × 256 px` tiles.

### 2. Train the selected model

Generate the experiment configurations with `--dry_run`, then train the selected model:

```powershell
python scripts/run_experiments.py `
  --experiments_config configs/experiments_phase2_augmentation_boundary_loss.yaml `
  --dry_run

python -m seg2gis.train `
  --config configs/generated/phase2_unet_effb3_aug_boundary_bce_w2_e50.json
```

Run configs are generated in `configs/generated/` and ignored by Git. To train
the complete final comparison, omit `--dry_run` from the experiment command.
Checkpoints are saved under `model.model_dir` using `training.run_name`;
the generated config also sets the inference checkpoint path.

### 3. Evaluate full images

```powershell
$CFG = "configs/generated/phase2_unet_effb3_aug_boundary_bce_w2_e50.json"
python -m seg2gis.evaluate --config $CFG --split val
python -m seg2gis.evaluate --config $CFG --split test
```

These commands reproduce the fixed `0.50/500/5` full-image baseline. The
validation-selected vector settings are supplied explicitly in the next step.

### 4. Export building polygons

Export a scene with the validation-selected settings. Setting both polygon
area filters to zero matches the extraction used in the vector diagnostics:

```powershell
$CFG = "configs/generated/phase2_unet_effb3_aug_boundary_bce_w2_e50.json"
python scripts/predict_full_image.py `
  --config $CFG `
  --image_path "data/AerialImageDataset/train/images/austin1.tif" `
  --threshold 0.47 `
  --min_area 100 `
  --open_kernel_size 3 `
  --epsilon_ratio 0.002 `
  --polygon_min_area 0 `
  --vector_min_area 0 `
  --output_name "austin1"
```

## Tests

With the project environment active, run:

```powershell
python -m unittest discover -s tests -v
```

The suite is self-contained, using synthetic arrays and temporary files.

## Repository map

```text
configs/          base configuration and experiment manifests
scripts/          preparation, experiment, inference, and analysis commands
seg2gis/           training, metrics, post-processing, and vectorisation code
tests/            tests for data loading, metrics, inference, and GIS export
results/tables/   experiment results in CSV format
results/figures/  example predictions and evaluation figures
```

## Citation and license

The dataset was introduced by E. Maggiori, Y. Tarabalka, G. Charpiat, and
P. Alliez in
[“Can Semantic Labeling Methods Generalize to Any City? The Inria Aerial Image Labeling Benchmark”](https://doi.org/10.1109/IGARSS.2017.8127684),
IGARSS 2017.

Code is released under the [MIT License](LICENSE). The INRIA dataset remains
subject to its own terms.
