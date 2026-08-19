#!/usr/bin/env bash
# PREPARED, NOT EXECUTED until explicitly launched. Run from the WSL repo root:
#   ~/projects/GlobalHumanoidRobotChallenge_2026_Baseline
#
# Recovery-only ACT fine-tune from the existing 200K checkpoint. Corrected
# version: preserves the original checkpoint's input/output normalization
# (state, action, AND all four visual features) by loading it verbatim via
# --normalization_stats_pretrained_path (see load_normalizer_stats_from_pretrained
# in src/lerobot/processor/normalize_processor.py, wired in
# src/lerobot/scripts/lerobot_train.py). The prior --normalization_stats_repo_id
# mechanism (configs/act_recovery_v2_finetune_5k.command.sh, superseded) sourced
# stats from a fresh LeRobotDatasetMetadata load, which bypasses the
# use_imagenet_stats substitution applied at make_dataset() time -- silently
# replacing the checkpoint's ImageNet visual normalization (std ~0.224-0.229)
# with the dataset's raw near-zero per-pixel std (~0.0002-0.005), producing
# NaN policy outputs. This flag sources the normalizer directly from the
# checkpoint's own saved policy_preprocessor instead.
#
# --config_path= loads the full architecture (dim_model=256, n_heads=4,
# chunk_size=50, n_action_steps=50, dim_feedforward=1024) from the 200K
# checkpoint; --policy.pretrained_path= loads its weights. Both required.
#
# Writes to a NEW output_dir; neither checkpoints/part_sorting_act_200k nor
# the prior (broken) outputs/part_sorting_act_recovery_finetune_5k are touched.

python -m src.lerobot.scripts.lerobot_train \
  --config_path=/mnt/d/guedr/Projects/GHRC2026/checkpoints/part_sorting_act_200k \
  --policy.pretrained_path=/mnt/d/guedr/Projects/GHRC2026/checkpoints/part_sorting_act_200k \
  --dataset.repo_id=local/part_sorting_recovery_v2_pilot50 \
  --dataset.root=/mnt/d/guedr/Projects/GHRC2026/datasets/Part_Sorting_Recovery_V2_Pilot50 \
  --normalization_stats_pretrained_path=/mnt/d/guedr/Projects/GHRC2026/checkpoints/part_sorting_act_200k \
  --output_dir=outputs/part_sorting_act_recovery_finetune_5k_normfix \
  --resume=false \
  --steps=5000 \
  --save_freq=1000 \
  --save_checkpoint=true \
  --batch_size=8 \
  --num_workers=4 \
  --optimizer.type=adamw \
  --optimizer.lr=1e-5 \
  --optimizer.weight_decay=0.0001 \
  --optimizer.grad_clip_norm=10.0 \
  --seed=1000 \
  --eval_freq=0 \
  --log_freq=50
