<h1 align="center">[ICML '26] Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences</h1>

<p align="center">
  <strong>Gwangho Kim, Sungyoon Lee</strong>
</p>

<p align="center">
  <a href="https://github.com/Gwangho99/mem-curv-diff">
    <img src="https://img.shields.io/badge/PROJECT-555555?style=for-the-badge&logo=github&logoColor=white" alt="Project">
  </a>
  <a href="https://gwangho99.github.io/mem-curv-diff/">
    <img src="https://img.shields.io/badge/PAGE-1688CA?style=for-the-badge" alt="Page">
  </a>
  <a href="https://arxiv.org/abs/2605.26756">
    <img src="https://img.shields.io/badge/ARXIV-555555?style=for-the-badge" alt="arXiv">
  </a>
  <a href="https://arxiv.org/abs/2605.26756">
    <img src="https://img.shields.io/badge/2605.26756-B91422?style=for-the-badge" alt="2605.26756">
  </a>
</p>

<p align="center">
  <img src="figure2.png" alt="Overview figure for Localizing Memorized Regions in Diffusion Models" width="65%">
</p>

This repository contains the official code for the ICML '26 paper **"Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences"**.

This repository is built upon [ryanwebster90/onestep-extraction](https://github.com/ryanwebster90/onestep-extraction) by Ryan Webster. We sincerely thank the authors for providing their implementation of "A Reproducible Extraction of Training Images from Diffusion Models".

If you have any questions, feel free to email Gwangho Kim (ggh1999@hanyang.ac.kr).

## Installation & Setup

This repository uses [uv](https://github.com/astral-sh/uv) for reproducible dependency management. To set up the environment, simply run:

```bash
# Synchronize the environment (installs exact versions from uv.lock)
uv sync
```

## Repository Structure


This repository is organized into three main steps: Dataset Preparation, Metric Map Generation, and Evaluation.

### Step 1: Ground-Truth Dataset Preparation
In this step, we match memorized samples (prompts and seeds) with original template images to create Ground-Truth (GT) masks.
* `synthall_from_parquet.py`: Generates synthetic images from known datasets.
* `gather_groundtruth_labels.py`: Compares generated images to the original images to extract edge-based GT masks.
* `parquet_to_jsonl.py`: Converts the parquet metadata format into a JSONL format for easy loading.
* **Example Usage:** (See `verify_sdv1_wb_attack.sh` for an end-to-end example)

*Note: the exact number of detected memorization prompts and seeds in Step 1 may vary slightly across different environments.*


### Step 2: Metric Map Generation (Localization)
In this step, we compute various localization maps across our dataset. This automatically processes BOTH the memorized (TV) prompts and the non-memorized (Nmem) baseline prompts. There are a total of 4 metric types you can generate:

**Standard Metrics (Cond vs Uncond):**
* `cov` ($\Delta h_\emptyset$): Coordinate-Wise Curvature Difference (Main Proposed Metric)
* `score_diff` ($\Delta s_\emptyset$): Standard Score Difference

**Baseline Metrics (Cond vs Bad_Cond):**
Using a "bad model" (SD v1-1) as a baseline comparison.
* `cov_bad` ($\Delta h_\tilde{\theta}$): Curvature difference against bad model
* `score_diff_bad` ($\Delta s_\tilde{\theta}$): Score difference against bad model

* `generate_metric_maps.py`: The main script to iterate over the dataset and generate `.npy` metric maps.
* **Example Usage (Standard TV dataset):**
  ```bash
  python generate_metric_maps.py --model_version 1 --metrics cov score_diff --output_dir metrics_outputs_v1/TV_metric_maps
  ```
* **Example Usage (MVRV dataset):**
  To evaluate non-template based memorization (MV/RV) where the entire image is memorized (GT mask = all-ones):
  ```bash
  python generate_metric_maps.py --model_version 1 --dataset sdv1-4_bb_attack_gt_verify_MVRV.jsonl --skip_nmem --metrics cov score_diff --output_dir metrics_outputs_v1/MVRV_metric_maps
  ```
* **Example Usage (With Bad Model):**
  ```bash
  python generate_metric_maps.py --model_version 1 --use_bad_model --metrics cov score_diff --output_dir metrics_outputs_v1/TV_metric_maps_bad
  ```

### Step 3: Evaluation
In this step, we evaluate the generated `.npy` localization maps against the GT masks.
* `evaluate_metrics.py`: Computes the exact metrics reported in the paper: IoU, mIoU, and Accuracy.
* **Example Usage (TV):**
  ```bash
  python evaluate_metrics.py --data_dir metrics_outputs_v1/TV_metric_maps
  ```
* **Example Usage (MVRV):**
  ```bash
  python evaluate_metrics.py --data_dir metrics_outputs_v1/MVRV_metric_maps --dataset sdv1-4_bb_attack_gt_verify_MVRV.jsonl
  ```

### Interactive Exploration (Jupyter)
For interactive visualization and exploratory analysis of the diffusion process, check out the provided Jupyter Notebook:
* `stable_diffusion_sampling.ipynb`: Contains step-by-step visualizations of the coordinate-wise curvature difference.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{
  kim2026localizing,
  title={Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences},
  author={Gwangho Kim and Sungyoon Lee},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
}
```
