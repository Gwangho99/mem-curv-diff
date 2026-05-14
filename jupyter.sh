#!/bin/bash
#
#SBATCH --job-name=jupyter
#SBATCH --account=ms
#SBATCH --output=output/%A_%a_res.txt
#SBATCH --error=output/%A_%a_err.txt
#SBATCH --partition=node2
#SBATCH --gres=gpu:1

#SBATCH -t 10:00:00

hostname


uv run -- which python        # Unix/macOS
uv run -- python -V
uv run jupyter notebook --notebook-dir=/home/gpuadmin/ghkim/2026ICML --ip=0.0.0.0 --port=12345 --no-browser --NotebookApp.token='788c361213a007f5b050d7e0ebfa5226f5537a495e658800'

exit 0
