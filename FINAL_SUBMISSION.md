# GHRC 2026 — Final Submission: Part Sorting

## Selected policy

**ACT, trained to 200,000 steps on the original 3,000-episode Part Sorting dataset.**

```
checkpoints/part_sorting_act_200k/
```

This checkpoint was selected over two fine-tuned alternatives (`ft2k`, a recovery-only fine-tune, and `mixed6k`, an 80/20 nominal+recovery mixed fine-tune) after a 100-episode paired evaluation of each against this baseline. Both alternatives improved recovery-from-disturbance behavior but degraded nominal Part Sorting performance more than the recovery gain justified. See [`RECOVERY_V2_MIXED_FINETUNE_SUMMARY.md`](RECOVERY_V2_MIXED_FINETUNE_SUMMARY.md) for that evaluation and full experimental history — that document is a research record, not part of this submission's runtime requirements.

## Input / output contract

| | |
|---|---|
| Robot | Walker S2 humanoid |
| Observation state | 20-D (14 arm joints + 4 gripper joints + 2 gripper control commands) |
| Cameras | 4 RGB, 480×640: `head_left`, `head_right`, `wrist_left`, `wrist_right` |
| Action | 20-D, matching the state layout |
| Policy | ACT — `dim_model=256`, `n_heads=4`, `dim_feedforward=1024`, `chunk_size=50`, `n_action_steps=50`, `resnet18` vision backbone, no temporal ensembling |
| Normalization | Checkpoint's own saved normalizer (`policy_preprocessor*.json/.safetensors`, `policy_postprocessor*.json/.safetensors`) — loaded verbatim, not recomputed |

## Checkpoint contents (verified present and loadable)

```
checkpoints/part_sorting_act_200k/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_7_normalizer_processor.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── train_config.json
```

## How to run inference

Prerequisites (see the main [`README.md`](README.md) for full environment setup): Isaac Sim 5.1.0 standalone, this repository's Python environment, and the `assets/` submodule downloaded per the README's Resource Download instructions.

```bash
set "ISAAC_SIM=C:\path\to\isaac-sim-standalone-5.1.0-windows-x86_64"
run_act_part_sorting_windows.bat
```

This launches one visible Walker S2 Part Sorting episode using `checkpoints/part_sorting_act_200k` by default (no flags required). To point at a different checkpoint or change episode duration:

```bash
run_act_part_sorting_windows.bat --checkpoint <path> --duration 120
```

The script is self-locating (derives its project root from its own file location) and requires only the `ISAAC_SIM` environment variable to be set to your local Isaac Sim install — it has no machine-specific hardcoded paths.

## What is NOT part of this submission

- Recovery V2 collector, evaluation harness, disturbance-injection module, and the `ft2k` / `mixed2k`–`mixed10k` checkpoints: preserved on disk as research records (see the summary doc above), not the selected policy.
- Datasets, training outputs, evaluation logs, and experiment results: excluded from version control (see `.gitignore`); available locally under `datasets/`, `outputs/`, `experiments/`, `logs/`.

## Known gap

The local repository and its linked official technical documentation (`docs.ubtrobot.com/GHRC2026_TechnicalDocuments`, pages 1–7) describe environment setup, data collection, training, and inference — they do not describe the competition's actual submission mechanism (upload process, deadline mechanics, required manifest/video/report format, or naming conventions). That information was not found locally and is not invented here; it must come from the competition organizers directly.
