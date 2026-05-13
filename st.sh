#!/bin/bash
#
#SBATCH --job-name=gh
#SBATCH --account=ms
#SBATCH --output=output/%A_res.out
#SBATCH --error=output/%A_err.out
#SBATCH --partition=node2

#SBATCH --gres=gpu:1

#SBATCH -t 0-10:00:00

#export PYTHONFAULTHANDLER=1

export MASTER_ADDR="localhost"
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

PYTHON_MODULE="$1"

# 첫 번째 인자가 비어있는지 확인
if [[ -z "$PYTHON_MODULE" ]]; then
    echo "오류: 실행할 Python 모듈을 지정해주세요."
    echo "사용법: ./run.sh [모듈_경로] [추가_인자...]"
    exit 1
fi
shift
hostname
echo "Starting Python script: $PYTHON_MODULE with arguments: $@"
uv run python "$PYTHON_MODULE" "$@"
#accelerate launch --config_file ./default_config.yaml "$PYTHON_FILE" "$@"
#uv run generate_metric_maps.py --model_version 1 --dataset "sdv1-4_bb_attack_gt_verify_TV.jsonl"

exit 0
