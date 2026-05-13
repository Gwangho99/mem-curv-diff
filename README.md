# Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences

Official repository for **"Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences"** (ICML 2026).

If you have any questions, feel free to email Gwangho Kim (ggh1999@hanyang.ac.kr).

## Repository Structure

This repository is organized into three main steps: Dataset Preparation, Metric Map Generation, and Evaluation.

### Step 1: Ground-Truth Dataset Preparation
In this step, we match memorized samples (prompts and seeds) with original template images to create Ground-Truth (GT) masks.
* `synthall_from_parquet.py`: Generates synthetic images from known datasets.
* `gather_groundtruth_labels.py`: Compares generated images to the original images to extract edge-based GT masks.
* `parquet_to_jsonl.py`: Converts the parquet metadata format into a JSONL format for easy loading.
* **Example Usage:** (See `verify_sdv1_wb_attack.sh` for an end-to-end example)

### Step 2: Metric Map Generation (Localization)
In this step, we compute various localization maps (e.g., Diag_H_Cond, Score Difference) across our dataset.
* `generate_metric_maps.py`: The main script to iterate over the dataset and generate `.npy` metric maps.
* **Example Usage:**
  ```bash
  python generate_metric_maps.py --model_version 1 --metrics cov score_diff --output_dir metrics_outputs_v1
  ```

### Step 3: Evaluation
In this step, we evaluate the generated `.npy` localization maps against the GT masks.
* `eval_detection_auc.py`: Computes the ROC AUC for detecting memorization.
* `test_global_iou.py`: Computes Global IoU metrics to measure precise localization against the GT masks.
* `eval_diag_cond_only.py`: An all-in-one script that computes the `Diag_H_Cond` metric on-the-fly and immediately calculates IoU/Accuracy without saving intermediate `.npy` files.
* **Example Usage:**
  ```bash
  python eval_detection_auc.py --data_dir metrics_outputs_v1 --suffix score_diff.npy
  python test_global_iou.py
  ```

---
### Baseline Attacks
We also provide code to run baseline membership inference and extraction attacks based on previous research:
* `run_bb_attack.py` (Black-Box)
* `run_wb_attack.py` (White-Box)

### Requirements
See `pyproject.toml` or `uv.lock` for exact dependency versions. (e.g. `uv sync` or `pip install -r requirements.txt`)