import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_metric_means(base_dir, category, metric_name):
    """
    Loads all .npy files for a specifically named metric in the given category directory,
    computes the mean of each map, and returns a list of means.
    """
    # Construct directory path: e.g., metrics_outputs_v1/TV_metric_maps
    dir_path = os.path.join(base_dir, f"{category}_metric_maps")
    
    if not os.path.exists(dir_path):
        print(f"Warning: Directory not found: {dir_path}")
        return []

    # Pattern to match files: prompt_*_seed_*_{metric_name}.npy
    pattern = os.path.join(dir_path, f"*_{metric_name}.npy")
    files = glob.glob(pattern)
    
    means = []
    print(f"Loading {len(files)} {metric_name} maps for {category}...")
    
    for f in tqdm(files):
        try:
            data = np.load(f)
            # Compute mean of the map (scalar)
            mean_val = np.mean(data)
            means.append(mean_val)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return means

def plot_distributions(all_data, output_file="metric_distributions_v1.png"):
    """
    all_data: dict of structure:
      {
        'cov': {'TV': [values...], 'MV': [...], 'Nmem': [...]},
        'attn': { ... },
        'score_diff': { ... }
      }
    """
    # Define metrics and their display names
    metrics = ['cov', 'attn', 'score_diff']
    metric_titles = {
        'cov': 'Trace Approx (Cov/LID)',
        'attn': 'Attention Map Mean',
        'score_diff': 'Score Difference Norm'
    }
    
    # Categories to plot
    categories = ['Nmem', 'TV', 'MV']
    colors = {'Nmem': 'green', 'TV': 'orange', 'MV': 'red'}
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.set_style("whitegrid")
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot KDE for each category
        for cat in categories:
            data = all_data[metric].get(cat, [])
            if not data or len(data) == 0:
                print(f"No data for {cat} in {metric}")
                continue
                
            # Filter out NaNs or Infs just in case
            data = np.array(data)
            data = data[np.isfinite(data)]
            
            # Use log scale for x-axis if there are zeros or very small values dominating
            # Adding a small epsilon to avoid log(0) errors if data contains exact zeros
            epsilon = 1e-10 
            
            sns.kdeplot(data + epsilon, ax=ax, label=f"{cat} (n={len(data)})", color=colors.get(cat, 'blue'), fill=True, alpha=0.3, log_scale=True)
            # Alternatively use histogram if data is sparse
            # sns.histplot(data, ax=ax, label=cat, color=colors.get(cat), element="step", stat="density", common_norm=False)
            
        ax.set_title(metric_titles.get(metric, metric))
        ax.set_xlabel("Mean Value (Log Scale)")
        ax.set_ylabel("Density")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

def main():
    base_dir = "metrics_outputs_v2"
    
    # Define categories based on folder names
    # Folders are expected to be: Nmem_metric_maps, TV_metric_maps, MV_metric_maps
    categories = ['Nmem', 'TV', 'MV']
    metrics = ['cov', 'attn', 'score_diff']
    
    all_data = {m: {} for m in metrics}
    
    for metric in metrics:
        for cat in categories:
            data = load_metric_means(base_dir, cat, metric)
            all_data[metric][cat] = data
            
    plot_distributions(all_data)

if __name__ == "__main__":
    main()
