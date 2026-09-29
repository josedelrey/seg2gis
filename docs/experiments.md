# INRIA experiments

## Evaluation protocol

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

## Segmentation results

The reported model is a **U-Net with an EfficientNet-B3 encoder**, geometric
augmentation, and Dice plus boundary-weighted binary cross-entropy.

The fixed baseline uses threshold `0.50`, minimum component area `500 px`, and
an opening kernel of `5 px`.

| Split | IoU | Dice/F1 | Precision | Recall | BF1 @2 px | BF1 @5 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.8016 | 0.8899 | 0.9052 | 0.8751 | 0.6246 | 0.8023 |
| Held-out test | 0.7876 | 0.8812 | 0.9064 | 0.8573 | 0.6307 | 0.7981 |

IoU and Dice measure area overlap. Boundary F1 (BF1) measures outline alignment
at 2-pixel and 5-pixel tolerances.

## Vector results

Validation tuning produced the vector-export settings: threshold `0.47`, minimum
component area `100 px`, opening kernel `3 px`, and Douglas–Peucker epsilon ratio
`0.002`. These values are fixed before evaluating the held-out split.

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
[model selection](../results/tables/phase2_augmentation_training_metrics.csv),
[full-image validation](../results/tables/phase2_full_image_validation_metrics_by_city.csv),
[full-image test](../results/tables/phase2_full_image_test_metrics_by_city.csv),
[post-processing selection](../results/tables/postprocess_ablation_validation_summary.csv),
[vector quality](../results/tables/vector_quality_test_best_val_config_summary.csv),
and [component diagnostics](../results/tables/instance_ap_test_best_val_config_by_city.csv).

## Reproduce the experiments

Install the project as described in the [README](../README.md) and activate
the environment with `source .venv/bin/activate` before running these commands.

### 1. Prepare the data and tiles

Download the dataset from the
[official INRIA download page](https://project.inria.fr/aerialimagelabeling/download/)
and extract it into `data/AerialImageDataset/` with this layout:

```text
data/AerialImageDataset/
  train/
    images/
    gt/
  test/
    images/
```

```bash
python scripts/prepare_tiles.py --config configs/default.json
```

This applies the scene-level split above, then extracts `256 × 256 px` tiles.

### 2. Train the model

Generate the experiment config with `--dry_run`, then train the reported model:

```bash
python scripts/run_experiments.py \
  --experiments_config configs/experiments_phase2_augmentation_boundary_loss.yaml \
  --dry_run

python -m seg2gis.train \
  --config configs/generated/phase2_unet_effb3_aug_boundary_bce_w2_e50.json
```

Run configs are generated in `configs/generated/` and ignored by Git. Omit
`--dry_run` to run all experiments. Checkpoints are saved under
`model.model_dir` using `training.run_name`. The generated config also sets
the checkpoint path used for inference.

### 3. Evaluate full images

```bash
CFG="configs/generated/phase2_unet_effb3_aug_boundary_bce_w2_e50.json"
python -m seg2gis.evaluate --config "$CFG" --split val
python -m seg2gis.evaluate --config "$CFG" --split test
```

These commands reproduce the fixed `0.50/500/5` full-image baseline.

### 4. Export building polygons

Export a scene with the vector settings. Setting both polygon area filters to
zero matches the extraction used in the vector diagnostics:

```bash
CFG="configs/generated/phase2_unet_effb3_aug_boundary_bce_w2_e50.json"
python scripts/predict_full_image.py \
  --config "$CFG" \
  --image_path "data/AerialImageDataset/train/images/austin1.tif" \
  --threshold 0.47 \
  --min_area 100 \
  --open_kernel_size 3 \
  --epsilon_ratio 0.002 \
  --polygon_min_area 0 \
  --vector_min_area 0 \
  --output_name "austin1"
```
