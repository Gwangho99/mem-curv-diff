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
In this step, we compute various localization maps across our dataset. There are a total of 4 metric types you can generate:

**Standard Metrics (Cond vs Uncond):**
* `cov` ($\Delta h_\emptyset$): Coordinate-Wise Curvature Difference (Main Proposed Metric)
* `score_diff` ($\Delta s_\emptyset$): Standard Score Difference

**Baseline Metrics (Cond vs Bad_Cond):**
Using a "bad model" (SD v1-1) as a baseline comparison.
* `cov_bad` ($\Delta h_\tilde{\theta}$): Curvature difference against bad model
* `score_diff_bad` ($\Delta s_\tilde{\theta}$): Score difference against bad model

* `generate_metric_maps.py`: The main script to iterate over the dataset and generate `.npy` metric maps.
* **Example Usage (Standard):**
  ```bash
  python generate_metric_maps.py --model_version 1 --metrics cov score_diff --output_dir metrics_outputs_v1/TV_metric_maps
  ```
* **Example Usage (With Bad Model):**
  ```bash
  python generate_metric_maps.py --model_version 1 --use_bad_model --metrics cov score_diff --output_dir metrics_outputs_v1/TV_metric_maps_bad
  ```

### Step 3: Evaluation
In this step, we evaluate the generated `.npy` localization maps against the GT masks.
* `evaluate_metrics.py`: Computes the exact metrics reported in the paper: IoU, mIoU, and Accuracy.
* **Example Usage:**
  ```bash
  python evaluate_metrics.py --data_dir metrics_outputs_v1/TV_metric_maps
  ```

---
### Requirements
See `pyproject.toml` or `uv.lock` for exact dependency versions. (e.g. `uv sync` or `pip install -r requirements.txt`)