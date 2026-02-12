#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./pipeline/convert_data.sh <raw_hdf5_dir> <repo_id> [--rdt-out-dir DIR] [--config-name NAME]
# Examples:
#   # Convert raw HDF5 -> Aloha episodes -> LeRobot, then compute norm stats for pi05_mobile_17d
#   ./pipeline/convert_data.sh dataset/mobile_demo local/mobile0212 --config-name pi05_mobile_17d
#   # Same as above but keep intermediate episodes elsewhere
#   ./pipeline/convert_data.sh dataset/mobile_demo local/mobile0212 --rdt-out-dir /tmp/rdt --config-name pi05_mobile_17d
#
# Arguments:
#   raw_hdf5_dir   Path to raw HDF5 data (e.g., dataset/mobile_demo)
#   repo_id        Target LeRobot repo id; must match TrainConfig.data.repo_id (e.g., local/mobile0212)
#   --rdt-out-dir  Optional. Output dir for intermediate Aloha episodes. Default: dataset/rdt
#   --config-name  Optional. If set, will run norm stats via scripts/compute_norm_stats.py after conversion.
#
# Env hints (uncomment & adjust if you want custom cache/output roots):
export HF_LEROBOT_HOME=/mnt/pfs/pg4hw0/shaolong/openpi/lerobot
export OPENPI_DATA_HOME=/mnt/pfs/pg4hw0/zanxin/RoboTwin-anything/RoboTwin/policy/weights/openpi
export XDG_CACHE_HOME=/mnt/pfs/pg4hw0/keheye/control_your_robot-Xspark/policy/openpi/cache

RAW_DIR=${1:?"raw_hdf5_dir is required"}
REPO_ID=${2:?"repo_id is required"}
shift 2

RDT_OUT_DIR="dataset/rdt"
TRAIN_CONFIG_NAME="pi05_mobile_17d"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rdt-out-dir)
      RDT_OUT_DIR=${2:?"--rdt-out-dir requires a value"}
      shift 2
      ;;
    --config-name|--train-config-name)
      TRAIN_CONFIG_NAME=${2:?"--config-name requires a value"}
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 <raw_hdf5_dir> <repo_id> [--rdt-out-dir DIR] [--config-name NAME]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# Task label that will be written into every frame; keep in sync with TrainConfig repack (prompt_from_task).
TASK="Move to the desk, pick up the clothes and put them into the small basket"

# 1) Raw HDF5 -> Aloha-format episodes (episode_*.hdf5)
printf "[1/3] Converting raw HDF5 -> Aloha episodes...\n"
uv run examples/aloha_real/convert_hdf5_to_aloha_data.py "$RAW_DIR" "$RDT_OUT_DIR"

# 2) Aloha episodes -> LeRobot dataset
#    --is-mobile enables base vx,vy,omega; --mode image avoids video encoding by default.
printf "[2/3] Packing Aloha episodes -> LeRobot repo (%s)...\n" "$REPO_ID"
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py \
  --raw-dir "$RDT_OUT_DIR" \
  --repo-id "$REPO_ID" \
  --task "$TASK" \
  --is-mobile \
  --mode image

# 3) Compute normalization stats (optional)
if [[ -n "$TRAIN_CONFIG_NAME" ]]; then
  printf "[3/3] Computing norm stats for config %s...\n" "$TRAIN_CONFIG_NAME"
  uv run scripts/compute_norm_stats.py --config-name "$TRAIN_CONFIG_NAME"
else
  printf "[3/3] Skip norm stats (train_config_name not provided).\n"
fi

printf "Done. LeRobot dataset stored under ${HF_LEROBOT_HOME:-~/.cache}/lerobot/%s\n" "$REPO_ID"
