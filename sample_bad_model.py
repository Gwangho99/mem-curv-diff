
import os
import torch
import json
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler

def main():
    parser = argparse.ArgumentParser(description="Sample images from the 'bad' model to check for memorization.")
    parser.add_argument("--model_version", type=int, default=1, choices=[1, 2], help="Model Version (1 for SDv1.4 base / v1.1 bad, 2 for SDv2.0 base / v2-base bad)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="bad_model_samples", help="Output directory")
    args = parser.parse_args()

    # Configuration
    if args.model_version == 1:
        bad_model_id = "CompVis/stable-diffusion-v1-1"
        dataset_path = "sdv1-4_bb_attack_gt_verify_TV.jsonl"
    else:
        bad_model_id = "Manojb/stable-diffusion-2-base"
        dataset_path = "sdv2_bb_attack_gt_verify_TV.jsonl"

    print(f"Model: {bad_model_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Output Directory: {args.output_dir}")

    # Load Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pipeline from {bad_model_id} on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        bad_model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Load Prompts
    prompts = []
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                prompts.append(item)
    else:
        print(f"Error: Dataset {dataset_path} not found.")
        return

    # Generate
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_samples} samples...")
    for i in range(min(args.num_samples, len(prompts))):
        item = prompts[i]
        prompt = item.get('prompt') or item.get('caption')
        if not prompt: continue

        # Use the first seed from the dataset if available, otherwise use default
        gen_seeds = item.get('gen_seeds', [])
        if gen_seeds:
            if isinstance(gen_seeds[0], list):
                seed = gen_seeds[0][0]
            else:
                seed = gen_seeds[0]
        else:
            seed = args.seed

        generator = torch.Generator(device).manual_seed(seed)
        
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
        
        filename = f"sample_{i:04d}_seed_{seed}.png"
        save_path = os.path.join(args.output_dir, filename)
        image.save(save_path)
        print(f"Saved {save_path} (Prompt: {prompt[:50]}...)")

if __name__ == "__main__":
    main()
