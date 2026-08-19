# Recovery V2 / Mixed-Data ACT Fine-Tuning — Final Summary

## Conclusion

**The original ACT checkpoint trained to 200K steps (`checkpoints/part_sorting_act_200k`) remains the selected final Part Sorting policy.** No fine-tuned variant produced in this experiment (recovery-only or mixed) is adopted as a replacement.

## What was demonstrated

- **Recovery V2** (collector, scenarios, disturbance-injection harness) successfully demonstrated that recovery behavior — displaced-during-approach, missed-first-grasp, drop-and-regrasp, difficult-position — can be learned via targeted fine-tuning from a base ACT checkpoint.
- **Recovery-only fine-tuning (`ft2k`, checkpoint 2000 of a 5K-step run on Recovery V2 Pilot50 alone)** substantially improved recovery behavior but degraded nominal Part Sorting behavior, and was judged not acceptable as a standalone replacement for 200K.
- **Mixed-data fine-tuning (80% nominal Part Sorting / 20% Recovery V2 Pilot50, weighted per-draw sampling, 10,000 steps from 200K)** was implemented and tested through five checkpoints (2K/4K/6K/8K/10K) to test whether the nominal/recovery trade-off could be improved over the recovery-only result.
  - `mixed6k` (step 6000) looked clearly promising in an initial small screening (n=2/scenario): 4/8 normal objects placed with 1 full completion, and 67% (2/3) aggregate confirmed recovery.
  - This result **did not replicate** in the full 100-episode paired evaluation (n=10/scenario) against 200K.

## Final 200K vs. mixed6k (100-episode paired evaluation, 50 episodes/checkpoint)

| Metric | 200K | mixed6k |
|---|---|---|
| Normal objects placed | 24/40 (**60%**) | 11/40 (28%) |
| Normal full episode completions | 2/10 | 1/10 |
| Aggregate confirmed recovery | 6/16 (**37.5%**) | 4/14 (28.6%) |
| Difficult-position success | 3/10 (**30%**) | 0/10 (0%) |
| Displaced-during-approach recovery | 2/7 (29%) | 1/7 (14%) |
| Drop-and-regrasp recovery | 4/8 (50%) | 1/4 (25%) |
| Missed-first-grasp recovery | 0/1 (0%) | 2/3 (67%, small n) |
| Total objects placed (all 50 eps) | **91/200** | 57/200 |
| Full task completions (all 50 eps) | 6/50 | 5/50 |
| Object escapes | 28/50 | 15/50 |
| Runtime errors / NaN / Inf | 0 | 0 |

Mixed6k regressed on nearly every axis relative to 200K, including recovery itself, and fell below even the recovery-only ft2k reference on normal placement. **Decision: 200K retained.** No further training (including a 90/10 mix) was pursued, per the accepted scope of this experiment.

## Reference evaluation runs (paths, not committed — see `.gitignore`)

- Final 200K-vs-mixed6k evaluation (100 episodes): `experiments/act_recovery_v2_eval_full_mixed6k/`
- Mixed 80/20 screening (60 episodes, 6 checkpoints): `experiments/act_recovery_v2_eval_screening_mixed_80_20/`
- Prior final 200K-vs-ft2k evaluation (100 episodes, reference numbers: 55%/35% vs 30%/58%): `experiments/act_recovery_v2_eval_full_normfix/`
- Prior ft2k screening (60 episodes, 6 checkpoints): `experiments/act_recovery_v2_eval_screening_normfix/`
- Recovery V2 Pilot50 collection (50-episode pilot, 43 valid episodes): `datasets/Part_Sorting_Recovery_V2_Pilot50/`
- Mixed 80/20 training run (checkpoints 2K-10K): `outputs/part_sorting_act_mixed_80_20_10k/`
- Recovery-only fine-tune (corrected normalizer), `ft2k` = checkpoint 2000: `outputs/part_sorting_act_recovery_finetune_5k_normfix/`

## Checkpoints (not committed — see `.gitignore`)

- **Selected final checkpoint**: `checkpoints/part_sorting_act_200k/` (unchanged)
- Recovery-only fine-tune checkpoints (preserved, not selected): `outputs/part_sorting_act_recovery_finetune_5k_normfix/checkpoints/{001000..005000}/`
- Mixed 80/20 fine-tune checkpoints (preserved, not selected): `outputs/part_sorting_act_mixed_80_20_10k/checkpoints/{002000,004000,006000,008000,010000}/`

## Source changes retained from this work

- **Recovery V2 collector**: `src/lerobot/auto_collect/recovery_v2.py`, `src/lerobot/auto_collect/task_part_sorting_recovery_v2.py`, `src/lerobot/scripts/auto_collect_recovery_v2.py`, `src/lerobot/scripts/validate_recovery_v2_dataset.py`
- **Evaluation harness**: `run_act_recovery_v2_eval_windows.py`, `recovery_v2_eval_disturbance.py` (policy-agnostic disturbance injection; the evaluator only ever creates the disturbance, never executes the recovery)
- **Normalizer-preservation fix**: `src/lerobot/processor/normalize_processor.py` (`load_normalizer_stats_from_pretrained`), `src/lerobot/configs/train.py` (`normalization_stats_pretrained_path`), wired into `src/lerobot/scripts/lerobot_train.py` — loads a checkpoint's saved normalizer verbatim (state, action, and all 4 image features) instead of reconstructing it from a dataset's own (in this project, unusable) raw stats
- **State-slicer scalar/count fix**: `src/lerobot/processor/state_slicer_processor.py` — `slice_stats_for_state` now guards against 0-dim scalar stats (e.g. `count`) instead of assuming every stat has a per-dimension axis
- **Mixed-dataset sampler**: `src/lerobot/datasets/mixed_sampler.py` (`WeightedConcatSampler`, `build_mixed_sampler`), `src/lerobot/datasets/factory.py` (`make_mixed_secondary_dataset`), config fields in `src/lerobot/configs/train.py` (`mixed_dataset_repo_id`/`_root`/`_ratio`) — controlled-ratio sampling across two `LeRobotDataset` sources without physically duplicating either
- **Tests**: `test_recovery_v2.py`, `test_recovery_v2_eval_harness.py`, `test_mixed_finetune.py`
- **Configs**: Recovery V2 collection configs (`configs/recovery_v2_part_sorting_*.yaml`), evaluation suite configs (`configs/recovery_v2_act_eval_{screening,full}_suite.yaml`), recorded fine-tune commands (`configs/act_recovery_v2_finetune_5k*.command.sh`)
- **`.gitignore` fix**: the pre-existing `datasets/` pattern (unanchored) was also matching `src/lerobot/datasets/`, silently excluding `mixed_sampler.py` from version control; anchored to `/datasets/` so it only affects the top-level dataset storage directory as originally intended
