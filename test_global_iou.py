
import os
import glob
import numpy as np
import pandas as pd
import argparse
from PIL import Image

from utils.data_loaders import load_tv_data, load_nmem_data

def evaluate_global_iou(data_dir, metric_suffix="score_diff_bad"):
    print(f"Evaluating Global IoU in: {data_dir}")
    
    # Needs metadata loading
    try:
        import pandas as pd
        metadata_path = "templates/metadata.parquet"
        tv_jsonl_path = "sdv1-4_bb_attack_gt_verify_TV.jsonl"
        nmem_file = "sd1_nmem.txt"
        
        metadata = pd.read_parquet(metadata_path)
        with open(tv_jsonl_path, "r") as f:
            import json
            tv_jsonl = [json.loads(line) for line in f]
        with open(nmem_file, "r") as f:
            nmem_prompts = [line.strip() for line in f if line.strip()]
            
        print("Loading Data...")
        # Use full data or subset? Let's use full to be accurate.
        # But for speed in this test, maybe subset.
        # Let's use 100 TV and 100 Nmem for quick test.
        
        tv_maps, tv_masks = load_tv_data(data_dir, tv_jsonl, metadata)
        nmem_maps, nmem_masks = load_nmem_data(data_dir, nmem_prompts)
        
        all_maps = np.concatenate([tv_maps, nmem_maps])
        all_masks = np.concatenate([tv_masks, nmem_masks])
        
        print(f"Total Samples: {len(all_maps)} (TV={len(tv_maps)}, Nmem={len(nmem_maps)})")
        
        # --- Threshold Search for Global IoU ---
        
        # 1. Normalize Global (log1p -> 0-1)
        # Using simple Min-Max from data
        maps_log = np.log1p(all_maps)
        min_val = maps_log.min()
        max_val = maps_log.max() # Or np.percentile(maps_log, 99)
        
        norm_maps = (maps_log - min_val) / (max_val - min_val + 1e-8)
        norm_maps = np.clip(norm_maps, 0, 1)
        
        eps = 1e-6
        thresholds = np.linspace(0.0 - eps, 1.0 + eps, 101)
        
        best_miou = 0.0
        best_global_iou = 0.0
        
        print("\nScanning Thresholds...")
        
        for th in thresholds:
            pred = (norm_maps > th)
            gt = (all_masks > 0.5)
            
            # A. Mean IoU (Original Metric)
            inter_per_sample = np.logical_and(pred, gt).sum(axis=(1, 2))
            union_per_sample = np.logical_or(pred, gt).sum(axis=(1, 2))
            
            iou_per_sample = np.ones_like(inter_per_sample, dtype=float)
            valid = (union_per_sample > 0)
            iou_per_sample[valid] = inter_per_sample[valid] / union_per_sample[valid]
            
            miou = iou_per_sample.mean()
            if miou > best_miou:
                best_miou = miou
                
            # B. Global IoU (New Metric)
            total_inter = inter_per_sample.sum()
            total_union = union_per_sample.sum()
            
            if total_union > 0:
                g_iou = total_inter / total_union
            else:
                g_iou = 1.0 # Or 0.0? Usually if nothing predicted and nothing GT, it's perfect.
            
            if g_iou > best_global_iou:
                best_global_iou = g_iou
                
        print(f"\n[Results Comparison]")
        print(f"  Best Mean IoU (mIoU): {best_miou:.4f} (Avg of sample IoUs)")
        print(f"  Best Global IoU:      {best_global_iou:.4f} (Total Inter / Total Union)")
        
        print("\n[Interpretation]")
        diff = best_global_iou - best_miou
        print(f"  Difference: +{diff:.4f}")
        if diff > 0.1:
            print("  -> Global IoU is significantly higher. This confirms that low mIoU is caused by 'Empty GT' samples being penalized heavily for small noise.")
        else:
            print("  -> Difference is small.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    default_dir = "ablation_results_score_diff_bad/v1/s49_e49"
    evaluate_global_iou(default_dir)
