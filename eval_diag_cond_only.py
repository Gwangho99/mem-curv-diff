
import os
import argparse
import numpy as np
import pandas as pd
import torch
import json
import random
from PIL import Image
from tqdm import tqdm
from diffusers import DDIMScheduler, StableDiffusionPipeline
from types import SimpleNamespace
from metric_utils import compute_diag_h_cond_only

# ============================================================================
# Configuration & Utils
# ============================================================================

def load_prompts(file_path, file_type, limit=50, stride=5):
    items = []
    collected_count = 0
    if file_type == 'jsonl':
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if collected_count >= limit: break
                if i % stride != 0: continue
                item = json.loads(line)
                if 'caption' in item and 'prompt' not in item: item['prompt'] = item['caption']
                if 'prompt' in item:
                    items.append(item)
                    collected_count += 1
    elif file_type == 'txt':
        with open(file_path, 'r') as f:
            lines = f.readlines()
            selected_lines = lines[::stride]
            if len(selected_lines) > limit: selected_lines = selected_lines[:limit]
            for line in selected_lines:
                items.append({'prompt': line.strip()})
    return items

def normalize(maps):
    # Log scale + Global MinMax
    maps_log = np.log1p(maps)
    
    # Optional: Percentile Clip (Robustness)
    # clip_val = np.percentile(maps_log, 99.9)
    # maps_log = np.clip(maps_log, None, clip_val)
    
    min_v, max_v = maps_log.min(), maps_log.max()
    if max_v > min_v:
        return (maps_log - min_v) / (max_v - min_v)
    return np.zeros_like(maps_log)

def get_iou_acc(preds, gts):
    # preds, gts: bool arrays
    inter = np.logical_and(preds, gts).sum()
    union = np.logical_or(preds, gts).sum()
    iou = inter / union if union > 0 else 1.0
    acc = (preds == gts).sum() / preds.size
    return iou, acc

# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=int, default=1, choices=[1, 2])
    parser.add_argument("--device", type=str, default="cuda")
    # limit_samples removed
    parser.add_argument("--hutchinson_k", type=int, default=16)
    args = parser.parse_args()
    
    print(f"Running Quick Eval: Diag_H(Cond) Only | SD v{args.model_version}")
    
    # 1. Model Setup
    if args.model_version == 1:
        model_id = "CompVis/stable-diffusion-v1-4"
        tv_path = "sdv1-4_bb_attack_gt_verify_TV.jsonl"
        nmem_path = "sd1_nmem.txt"
    else:
        model_id = "Manojb/stable-diffusion-2-1-base" # Using 2.1 base as per other scripts
        tv_path = "sdv2_bb_attack_gt_verify_TV.jsonl"
        nmem_path = "sd2_nmem.txt"
        
    print(f"Loading Model: {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    ).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    try: pipe.disable_xformers_memory_efficient_attention()
    except: pass
    
    # MV setup
    if args.model_version == 1:
        mv_path = "sdv1-4_bb_attack_gt_verify_MVRV.jsonl"
    else:
        mv_path = "sdv2_bb_attack_gt_verify_MVRV.jsonl"

    
    # 2. Data Loading
    print("Loading Prompts...")
    
    METADATA_PATH = "templates/metadata.parquet"
    if os.path.exists(METADATA_PATH):
        metadata = pd.read_parquet(METADATA_PATH)
    else:
        print("[Error] Metadata not found. Cannot load masks.")
        return

    # Load TV
    tv_items = []
    with open(tv_path, 'r') as f:
        for line in f:
            tv_items.append(json.loads(line))
            
    tv_data = [] 
    for i, item in enumerate(tv_items):
        gen_seeds = item.get('gen_seeds', [])
        if not gen_seeds: continue
        s = gen_seeds[0]
        if isinstance(s, list): seed_val, t_idx = s[0], s[1]
        else: seed_val, t_idx = s, -1
        if t_idx < 0: continue
        try:
            mask_rel = metadata.iloc[t_idx]['mask_file']
            mask_abs = os.path.abspath(mask_rel)
            if not os.path.exists(mask_abs): continue
        except: continue
        
        prompt = item.get('prompt') or item.get('caption')
        tv_data.append({
            "prompt": prompt,
            "mask_path": mask_abs,
            "seed": seed_val,
            "type": "TV"
        })
    print(f"Loaded {len(tv_data)} TV samples.")

    # Load MV
    mv_items = load_prompts(mv_path, 'jsonl', limit=1000, stride=1) # Load all (limit high)
    mv_data = []
    for item in mv_items:
        # MV Metric map gen usually uses multiple seeds? 
        # Standard logic: use seed from item or default.
        # generate_eval_dataset used first seed.
        gen_seeds = item.get('gen_seeds', [])
        if gen_seeds:
            s = gen_seeds[0]
            if isinstance(s, list): seed_val = s[0]
            else: seed_val = s
        else:
            seed_val = args.seed if hasattr(args, 'seed') else 42
            
        mv_data.append({
            "prompt": item['prompt'],
            "mask_path": None, # All Ones
            "seed": seed_val,
            "type": "MV"
        })
    print(f"Loaded {len(mv_data)} MV samples.")
    
    # Load Nmem
    nmem_prompts_raw = load_prompts(nmem_path, 'txt', limit=1000, stride=1)
    nmem_data = []
    for item in nmem_prompts_raw:
        nmem_data.append({
            "prompt": item['prompt'],
            "mask_path": None, # All Zeros
            "seed": 0,
            "type": "Nmem"
        })
    print(f"Loaded {len(nmem_data)} Nmem samples.")
    
    # 3. Compute Metrics (All Types)
    print("\nComputing Metrics...")
    
    gen_args = SimpleNamespace(
        num_inference_steps=50,
        guidance_scale=7.5,
        target_timestep=49,
        hutchinson_k=args.hutchinson_k,
        model_version=args.model_version
    )
    
    # Store results by type
    results_store = {"TV": [], "MV": [], "Nmem": []} # list of (map, mask)
    
    all_experiments = tv_data + mv_data + nmem_data
    
    for idx, item in enumerate(tqdm(all_experiments, desc="Processing All")):
        prompt = item['prompt']
        seed = item['seed']
        dtype = item['type']
        
        # Text Embeddings
        text_input = pipe.tokenizer([prompt], padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
        uncond_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        m_map = compute_diag_h_cond_only(prompt, pipe, gen_args, text_embeddings, seed)
        if m_map is None: continue
        
        # Resize Map
        m_img = Image.fromarray(m_map.astype(np.float32), mode='F').resize((256, 256), Image.BILINEAR)
        m_map_resized = np.array(m_img)
        
        # Mask
        if dtype == "TV":
            img = Image.open(item['mask_path']).convert('L').resize((256, 256), Image.NEAREST)
            mask = (np.array(img).astype(np.float32) / 255.0) > 0.5
            
            # Save Visualization for first 20 TV samples
            if len(results_store["TV"]) < 20:
                # Create a simple concatenated image: [Metric Map (normalized visually), Mask]
                # Normalize map for visualization (0-255)
                vis_map = m_map_resized
                if vis_map.max() > vis_map.min():
                    vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
                vis_map = (vis_map * 255).astype(np.uint8)
                vis_map_img = Image.fromarray(vis_map)
                
                vis_mask = (mask * 255).astype(np.uint8)
                vis_mask_img = Image.fromarray(vis_mask)
                
                # Canvas
                canvas = Image.new('L', (256 * 2, 256))
                canvas.paste(vis_map_img, (0, 0))
                canvas.paste(vis_mask_img, (256, 0))
                
                # Save
                safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:30]
                save_name = f"viz_tv_{len(results_store['TV'])}_{safe_prompt}.png"
                canvas.save(save_name)
                # print(f"Saved viz: {save_name}")

        elif dtype == "MV":
            mask = np.ones((256, 256), dtype=bool)
        else: # Nmem
            mask = np.zeros((256, 256), dtype=bool)
            
        results_store[dtype].append((m_map_resized, mask))

    # 4. Evaluate Scenarios
    
    def evaluate_scenario(name, data_sources):
        # data_sources: list of strings e.g. ["TV", "MV"]
        maps = []
        masks = []
        
        for ds in data_sources:
            for m, msk in results_store[ds]:
                maps.append(m)
                masks.append(msk)
                
        if not maps:
            print(f"[{name}] No data.")
            return

        maps_arr = np.array(maps)
        masks_arr = np.array(masks)
        
        norm_maps = normalize(maps_arr)
        
        best_iou = 0.0
        best_acc = 0.0
        best_th = 0.0
        
        thresholds = np.linspace(0, 1, 1001)
        for th in thresholds:
            preds = (norm_maps > th)
            curr_ious = []
            curr_accs = []
            # Batch stats
            inter = np.logical_and(preds, masks_arr).sum(axis=(1,2))
            union = np.logical_or(preds, masks_arr).sum(axis=(1,2))
            # valid union > 0
            curr_ious = np.ones_like(inter, dtype=float)
            valid = union > 0
            curr_ious[valid] = inter[valid] / union[valid]
            
            corr = (preds == masks_arr).sum(axis=(1,2))
            curr_accs = corr / (256*256)
            
            mean_iou = np.mean(curr_ious)
            mean_acc = np.mean(curr_accs)
            
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_th = th
            if mean_acc > best_acc:
                best_acc = mean_acc

        print(f"[{name}] Best IoU: {best_iou:.4f} (th={best_th:.2f}) | Best Acc: {best_acc:.4f} (Samples: {len(maps)})")


    print("\n" + "="*60)
    print(f"EVALUATION RESULTS (Diag_H_Cond Only | SD v{args.model_version})")
    print("="*60)
    
    # Scenario 1: TV only
    evaluate_scenario("TV only", ["TV"])
    
    # Scenario 2: TV + MV + Nmem
    evaluate_scenario("Combined (TV+MV+Nmem)", ["TV", "MV", "Nmem"])
    
    print("="*60)


if __name__ == "__main__":
    main()
