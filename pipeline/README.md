# Pipeline Cheatsheet

Quick steps to convert data and fine-tune.

## 1) Convert raw HDF5 to LeRobot
```
./pipeline/convert_data.sh <raw_hdf5_dir> <repo_id> [rdt_out_dir] [train_config_name]
```
- `repo_id` must match the `TrainConfig.data.repo_id` you will train with.
- The script writes task label `mobile_task` into each frame (for prompt_from_task workflows).
- If `train_config_name` is set, it will also run `scripts/compute_norm_stats.py`.

## 2) Fine-tune
```
./pipeline/finetune.sh <train_config_name> <exp_name> <gpu_ids>
```
- `train_config_name` should point to the config whose `repo_id` matches the dataset above.
- Uses JAX trainer (`scripts/train.py`); for PyTorch DDP use `train_pytorch.py` via torchrun.

## Env vars (optional)
- `HF_LEROBOT_HOME`: where LeRobot datasets live.
- `OPENPI_DATA_HOME`: where checkpoints/assets cache.
- `XDG_CACHE_HOME`: extra cache root.

## Notes
- `convert_aloha_data_to_lerobot.py` now embeds `DEFAULT_TASK=mobile_task`; CLI flags stay unchanged.
- For multi-batch data, merge HDF5s under one raw dir before running the converter to avoid overwrite.
