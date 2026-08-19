#!/usr/bin/env bash
# PREPARED, NOT EXECUTED. Run from the WSL repo root:
#   ~/projects/GlobalHumanoidRobotChallenge_2026_Baseline
#
# Recovery-only ACT fine-tune from the existing 200K checkpoint. Preserves the
# original checkpoint's input/output normalization via
# --normalization_stats_repo_id/--normalization_stats_root (see
# src/lerobot/scripts/lerobot_train.py + src/lerobot/configs/train.py). Writes
# to a NEW output_dir; checkpoints/part_sorting_act_200k is never touched.

python -m src.lerobot.scripts.lerobot_train \
  --policy.pretrained_path=checkpoints/part_sorting_act_200k \
  --policy.type=act \
  --dataset.repo_id=local/part_sorting_recovery_v2_pilot50 \
  --dataset.root=/mnt/d/guedr/Projects/GHRC2026/datasets/Part_Sorting_Recovery_V2_Pilot50 \
  --normalization_stats_repo_id=local/Part_Sorting \
  --normalization_stats_root=datasets/Part_Sorting \
  --output_dir=outputs/part_sorting_act_recovery_finetune_5k \
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
  --scheduler=null \
  --seed=1000 \
  --eval_freq=0 \
  --log_freq=50
