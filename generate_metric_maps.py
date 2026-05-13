
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import glob
import argparse
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from types import SimpleNamespace
from optim_utils import get_dataset, set_random_seed
from metric_utils import (
    compute_metrics_for_prompt,
    compute_metrics_for_prompt_with_bad_model
)
# =========================================================================================
# Main
# =========================================================================================

def main():
    # Setup similar to notebook
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=int, default=1)
    parser.add_argument("--metrics", nargs="+", default=["cov", "score_diff"], help="Metrics to compute: cov, score_diff")
    parser.add_argument("--use_bad_model", action="store_true", help="Use bad model for cov_bad and score_diff_bad")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--target_timestep", type=int, default=49)
    parser.add_argument("--score_diff_step_start", type=int, default=49)
    parser.add_argument("--score_diff_step_end", type=int, default=50)
    parser.add_argument("--hutchinson_k", type=int, default=16)
    parser.add_argument("--no_score", dest="is_score", action="store_false", help="Disable score metric calculation")
    parser.set_defaults(is_score=True)
    parser.add_argument("--max_seeds_per_prompt", type=int, default=4)
    
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.model_version == 1:
        args.model_id = "CompVis/stable-diffusion-v1-4"
        args.dataset = "sdv1-4_bb_attack_gt_verify_TV.jsonl"
        if args.use_bad_model:
            args.bad_model_id = "CompVis/stable-diffusion-v1-1"
            print(f"Loading bad model UNet: {args.bad_model_id}")
            bad_unet = UNet2DConditionModel.from_pretrained(
                args.bad_model_id, subfolder="unet", torch_dtype=torch.float16
            ).to(device)
            bad_unet.eval()
        else:
            bad_unet = None
    elif args.model_version == 2:
        args.model_id = "Manojb/stable-diffusion-2-1-base"
        args.dataset = "sdv2_bb_attack_gt_verify_TV.jsonl"
        if args.use_bad_model:
            args.bad_model_id = "Manojb/stable-diffusion-2-base"
            print(f"Loading bad model UNet: {args.bad_model_id}")
            bad_unet = UNet2DConditionModel.from_pretrained(
                args.bad_model_id, subfolder="unet", torch_dtype=torch.float16
            ).to(device)
            bad_unet.eval()
        else:
            bad_unet = None
    
    
    # Output Dir
    if args.output_dir:
        metric_save_dir = args.output_dir
    else:
        metric_save_dir = f"metrics_outputs_v{args.model_version}/TV_metric_maps"
    
        
    os.makedirs(metric_save_dir, exist_ok=True)
    print(f"Saving metric maps to: {metric_save_dir}")
    
    # Load Model
    print(f"Loading model: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.set_progress_bar_config(disable=True)
    print("Model loaded.")

    
    
    
    # Load Dataset
    dataset, prompt_key = get_dataset(args.dataset, pipe)
    
    # Process all prompts in dataset
    sorted_indices = list(range(len(dataset)))
    print(f"Processing {len(sorted_indices)} prompts from dataset.")
    
    # Process
    for prompt_idx in tqdm(sorted_indices):
        if prompt_idx >= len(dataset):
            continue
        
        item = dataset[prompt_idx]
        prompt = item[prompt_key]
        gen_seeds = item.get('gen_seeds', [])
        
        if args.max_seeds_per_prompt > 0:
            gen_seeds = gen_seeds[:args.max_seeds_per_prompt]

        seeds_to_run = []
        if len(gen_seeds) > 0:
            for s in gen_seeds:
                if isinstance(s, (list, tuple)):
                    seeds_to_run.append(s[0])
                else:
                    seeds_to_run.append(s)
        else:
            seeds_to_run = [args.seed]
        
        for seed in seeds_to_run:
            try:
                # Check if outputs already exist? (Optional)
                # pass

                if bad_unet is not None:
                    metrics = compute_metrics_for_prompt_with_bad_model(prompt, pipe, bad_unet, args, seed=seed)
                else:
                    metrics = compute_metrics_for_prompt(prompt, pipe, args, seed=seed)
                
                for name, m_map in metrics.items():
                    # Save raw values (no per-sample normalization)
                    # Include seed in filename
                    save_name = f"prompt_{prompt_idx:04d}_seed_{seed:02d}_{name}.npy"
                    save_path = os.path.join(metric_save_dir, save_name)
                    np.save(save_path, m_map)
                     
            except Exception as e:
                raise Exception(f"Error processing prompt {prompt_idx} seed {seed}: {e}")

if __name__ == "__main__":
    main()
