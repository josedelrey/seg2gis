# seg2gis

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f)

**From aerial imagery to candidate building footprints for GIS.**

seg2gis is a reproducible raster-to-vector pipeline for building segmentation,
full-scene inference, mask post-processing, polygon simplification, and
georeferenced GeoJSON export. It is intended for research and assisted mapping:
the exported polygons are useful building candidates, not cadastral boundaries.

![Input image, probability map, cleaned mask, and polygon overlay](results/figures/building_footprint_showcase.png)

<p align="center"><em>Held-out aerial crop: input → probability map → cleaned mask → polygon overlay.</em></p>

## Pipeline

```text
aerial scenes → image-level split → 256 px tiles → segmentation model
              → overlapping full-image inference → mask cleanup
              → contour extraction → polygon simplification → GeoJSON
```

- Compares U-Net, FPN, and DeepLabV3+ segmentation models.
- Supports geometric augmentation and boundary-weighted training.
- Reconstructs complete scenes from overlapping tile predictions.
- Selects thresholds and post-processing settings on validation data only.
- Reports raster, boundary, component, and vector-quality diagnostics.
- Preserves the source raster transform and coordinate reference system on export.

## Evaluation protocol

Experiments use the 180 labelled images from the
[INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/):
five cities, with 36 images of `5000 × 5000 px` per city. Complete scenes are
assigned to a split before tiling, preventing spatially related tiles from
leaking across training and evaluation.

| Split | Image IDs per city | Scenes | Purpose |
| --- | --- | ---: | --- |
| Train | 11–36 | 130 | Model fitting |
| Validation | 6–10 | 25 | Model and post-processing selection |
| Held-out test | 1–5 | 25 | Final reporting |

The held-out set follows the labelled first-five-per-city convention often
called INRIA(155). It is not the benchmark's unlabelled official test set.

## Results

The selected model is a **U-Net with an EfficientNet-B3 encoder**, geometric
augmentation, and Dice plus boundary-weighted binary cross-entropy. The final
EfficientNet-B3 candidates were closely matched on validation data; the selected
U-Net obtained the best tuned Dice/F1 (`0.8890`) and was retained as a strong
baseline rather than as evidence that one architecture is universally superior.

### Full-image segmentation

The fixed baseline uses threshold `0.50`, minimum component area `500 px`, and
an opening kernel of `5 px`.

| Split | IoU | Dice/F1 | Precision | Recall | BF1 @2 px | BF1 @5 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.8016 | 0.8899 | 0.9052 | 0.8751 | 0.6246 | 0.8023 |
| Held-out test | 0.7876 | 0.8812 | 0.9064 | 0.8573 | 0.6307 | 0.7981 |

The small validation-to-test drop indicates stable full-scene segmentation.
Precision is higher than recall, so the model is conservative: predicted
building area is usually correct, but some buildings or building parts are
missed. The gap between boundary F1 at 2 and 5 pixels shows that most outlines
are approximately correct without being precisely aligned.

### Vector output

The vector pipeline uses a separate configuration selected on validation data:
threshold `0.47`, minimum component area `100 px`, opening kernel `3 px`, and
Douglas–Peucker epsilon ratio `0.002`. These values are fixed before evaluating
the held-out split.

On the held-out scenes, the cleaned mask reaches `0.8023` IoU. After contour
extraction and simplification, polygon-raster IoU is `0.7736`, and `0.54%` of
the exported polygons are invalid. The pipeline produces 25,564 polygons for
32,794 reference connected components; component AP is `0.6021` at IoU 0.50
and `0.3826` at IoU 0.75. In practical terms, the workflow preserves building
area well and usually emits valid candidates, but it still misses small
buildings, merges adjacent roofs, and lacks cadastral boundary precision.

The figures and tables above are backed by committed CSV files:
[model selection](results/tables/phase2_augmentation_training_metrics.csv),
[full-image validation](results/tables/phase2_full_image_validation_metrics_by_city.csv),
[full-image test](results/tables/phase2_full_image_test_metrics_by_city.csv),
[post-processing selection](results/tables/postprocess_ablation_validation_summary.csv),
[vector quality](results/tables/vector_quality_test_best_val_config_summary.csv),
and [component diagnostics](results/tables/instance_ap_test_best_val_config_by_city.csv).

## Installation

The recommended environment uses Python 3.11:

```powershell
conda env create -f environment.yml
conda activate seg2gis
```

For a pip-only environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build matching the local CUDA
version before installing the remaining dependencies.

## Data

Download the INRIA dataset and arrange it as follows:

```text
data/AerialImageDataset/
  train/
    images/
    gt/
  test/
    images/
```

The labelled images under `train/` supply all three local splits. The public
`test/` images have no labels and are not used in the reported metrics.

## Run the pipeline

Run all commands from the repository root.

### 1. Prepare tiles

```powershell
python scripts/prepare_tiles.py --config configs/default.json
```

### 2. Train the selected model

Generate the run configurations from the experiment manifest, without starting
training, then train the selected run:

```powershell
python scripts/run_experiments.py `
  --experiments_config configs/experiments_phase2_augmentation_boundary_loss.yaml `
  --dry_run

python src/train.py `
  --config configs/generated/phase2_unet_effb3_aug_boundary_bce_w2_e50.json
```

Run configs are generated locally in `configs/generated/` and ignored by Git.
The committed base config and experiment manifests are the source of truth.
To train the complete final comparison, run the experiment command without
`--dry_run`.

Trained checkpoints are not distributed with the repository. Evaluation uses
the checkpoint defined by `model.model_dir` and `training.run_name`; inference
also accepts an explicit `--model_path`.

### 3. Evaluate full images

```powershell
$CFG = "configs/generated/phase2_unet_effb3_aug_boundary_bce_w2_e50.json"
python src/evaluate.py --config $CFG --split val
python src/evaluate.py --config $CFG --split test
```

These commands reproduce the fixed `0.50/500/5` full-image baseline. The
validation-selected vector settings are supplied explicitly in the next step.

### 4. Export building polygons

```powershell
$CFG = "configs/generated/phase2_unet_effb3_aug_boundary_bce_w2_e50.json"
python scripts/predict_full_image.py `
  --config $CFG `
  --image_path <path-to-georeferenced-raster.tif> `
  --threshold 0.47 `
  --min_area 100 `
  --open_kernel_size 3 `
  --epsilon_ratio 0.002 `
  --output_name <output-name>
```

By default, inference writes the probability map, raw and cleaned masks,
polygon overlay, showcase crop, and GeoJSON file to `results/full_predictions/`.

GeoJSON coordinates use the source raster transform and CRS. Map-unit area
filtering requires a projected CRS; for geographic-coordinate rasters, reproject
the input or disable vector-area filtering with `--vector_min_area 0`.

## Scope and limitations

The current system is a mask-driven research baseline. It does not predict
instances, corners, topology, or polygon vertices directly. Results cover five
INRIA city domains and one seeded training run; component AP is derived from
semantic-mask connected components rather than official instance annotations.
Outputs are suitable for experimentation and assisted digitisation, but require
review for complete inventories, legal boundaries, or high-precision mapping.

## Repository map

```text
configs/          base configuration and experiment manifests
scripts/          preparation, experiment, inference, and analysis commands
src/              training, metrics, post-processing, and vectorisation code
tests/            path-safety and configuration regression tests
results/tables/   numerical evidence committed as CSV
results/figures/  curated qualitative and analytical figures
```

## Citation and license

The dataset was introduced by E. Maggiori, Y. Tarabalka, G. Charpiat, and
P. Alliez in
[“Can Semantic Labeling Methods Generalize to Any City? The Inria Aerial Image Labeling Benchmark”](https://doi.org/10.1109/IGARSS.2017.8127684),
IGARSS 2017.

Code is released under the [MIT License](LICENSE). The INRIA dataset remains
subject to its own terms.
