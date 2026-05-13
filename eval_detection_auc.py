
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import argparse

def evaluate_detection_auc(data_dir, metric_suffix="score_diff_bad.npy"):
    print(f"Evaluating Detection AUC in: {data_dir}")
    print(f"Target Metric Suffix: *{metric_suffix}")
    
    # 1. Load Files
    # Note: Using glob to find files. Assuming standard naming convention.
    # Exclude files that might be deleted or missing.
    
    # TV: startswith 'prompt_' AND does NOT contain 'mv_' or 'nmem_'
    all_files = glob.glob(os.path.join(data_dir, f"*{metric_suffix}"))
    
    tv_files = []
    mv_files = []
    nmem_files = []
    
    for f in all_files:
        bn = os.path.basename(f)
        if bn.startswith("nmem_prompt_"):
            nmem_files.append(f)
        elif bn.startswith("mv_prompt_"):
            mv_files.append(f)
        elif bn.startswith("prompt_"):
            tv_files.append(f)
            
    print(f"Found files: TV={len(tv_files)}, MV={len(mv_files)}, Nmem={len(nmem_files)}")
    
    if not nmem_files or (not tv_files and not mv_files):
        print("Insufficient data for AUC calculation.")
        return

    # 2. Compute Scalar Scores (Mean)
    def load_scores(files, label):
        scores = []
        valid_count = 0
        for f in files:
            try:
                m = np.load(f)
                if np.isfinite(m).all():
                    # Scalar Score: Mean of the map
                    # You could also try Max or Top-K mean
                    scores.append(np.mean(m)) 
                    valid_count += 1
            except: pass
        return np.array(scores)

    print("Computing scalar scores (Mean)...")
    tv_scores = load_scores(tv_files, "TV")
    mv_scores = load_scores(mv_files, "MV")
    nmem_scores = load_scores(nmem_files, "Nmem")
    
    print(f"Valid Scores: TV={len(tv_scores)}, MV={len(mv_scores)}, Nmem={len(nmem_scores)}")

    # 3. Compute AUC
    results = {}
    
    # helper
    def compute_auc(pos_scores, neg_scores, label):
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5
        
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])
        
        auc = roc_auc_score(y_true, y_scores)
        return auc

    # A. TV vs Nmem
    auc_tv = compute_auc(tv_scores, nmem_scores, "TV vs Nmem")
    print(f"\n[Detection Performance]")
    print(f"  TV vs Nmem AUC: {auc_tv:.4f}")
    
    # B. MV vs Nmem
    auc_mv = compute_auc(mv_scores, nmem_scores, "MV vs Nmem")
    print(f"  MV vs Nmem AUC: {auc_mv:.4f}")
    
    # C. All Mem (TV+MV) vs Nmem
    mem_scores = np.concatenate([tv_scores, mv_scores]) if len(mv_scores) > 0 else tv_scores
    auc_all = compute_auc(mem_scores, nmem_scores, "Mem(TV+MV) vs Nmem")
    print(f"  Mem (TV+MV) vs Nmem AUC: {auc_all:.4f}")
    
    # 4. Visualization (Distributions)
    plt.figure(figsize=(10, 6))
    
    if len(nmem_scores) > 0:
        plt.hist(nmem_scores, bins=50, alpha=0.5, label=f'Nmem (n={len(nmem_scores)})', density=True, color='gray')
    if len(tv_scores) > 0:
        plt.hist(tv_scores, bins=50, alpha=0.5, label=f'TV (n={len(tv_scores)})', density=True, color='blue')
    if len(mv_scores) > 0:
        plt.hist(mv_scores, bins=50, alpha=0.5, label=f'MV (n={len(mv_scores)})', density=True, color='orange')
        
    plt.title(f"Scalar Score Distribution (Mean Metric Value)\nMetric: {metric_suffix} | AUC(All)={auc_all:.3f}")
    plt.xlabel("Mean Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale x-axis might be better if range is huge
    # plt.xscale('log') 
    
    out_png = "detection_auc_dist.png"
    plt.savefig(out_png)
    print(f"Saved distribution plot to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="ablation_results_score_diff_bad/v1/s49_e49")
    parser.add_argument("--suffix", type=str, default="score_diff_bad.npy")
    args = parser.parse_args()
    
    evaluate_detection_auc(args.data_dir, args.suffix)
