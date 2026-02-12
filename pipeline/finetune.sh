#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./pipeline/finetune.sh <train_config_name> <exp_name> <gpu_ids>
# Example:
#   ./pipeline/finetune.sh pi05_mobile_17d my_mobile_exp 0,1
# Notes:
#   - <train_config_name> must match the repo_id you used in data conversion.
#     If you used repo_id=local/mobile0212 in convert_data.sh, use the config whose data.repo_id is the same.
#     If you want to continue training, use --resume and delete --overwrite.
#   - Runs JAX trainer (scripts/train.py). For DDP/PyTorch, use train_pytorch.py with torchrun instead.

# Optional cache/data roots
export HF_LEROBOT_HOME=/mnt/pfs/pg4hw0/shaolong/openpi/lerobot
export OPENPI_DATA_HOME=/mnt/pfs/pg4hw0/zanxin/RoboTwin-anything/RoboTwin/policy/weights/openpi
export XDG_CACHE_HOME=/mnt/pfs/pg4hw0/keheye/control_your_robot-Xspark/policy/openpi/cache

train_config_name=$1
model_name=$2
gpu_use=$3

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite
