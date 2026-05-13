
import os
import glob
import numpy as np
import pandas as pd
import argparse
from PIL import Image

from utils.data_loaders import load_tv_data, load_nmem_data

def evaluate_metrics(data_dir, metric_suffix="score_diff"):
    print(f"Evaluating Metrics (IoU, mIoU, Acc) in: {data_dir}")
    
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
        
        tv_maps, tv_masks = load_tv_data(data_dir, tv_jsonl, metadata, metric_name=metric_suffix)
        nmem_maps, nmem_masks = load_nmem_data(data_dir, nmem_prompts, metric_name=metric_suffix)
        
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
        best_acc = 0.0
        
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
                
            # C. Accuracy
            acc = (pred == gt).mean()
            if acc > best_acc:
                best_acc = acc
                
        print(f"\n[Results]")
        print(f"  IoU (Global):  {best_global_iou:.4f}")
        print(f"  mIoU (Mean):   {best_miou:.4f}")
        print(f"  Accuracy:      {best_acc:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="metrics_outputs_v1/TV_metric_maps")
    parser.add_argument("--metric_name", type=str, default="cov", help="Name of the metric to evaluate (e.g. cov, score_diff, cov_bad)")
    args = parser.parse_args()
    evaluate_metrics(args.data_dir, metric_suffix=args.metric_name)
