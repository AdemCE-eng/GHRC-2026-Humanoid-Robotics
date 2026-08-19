# GHRC 2026 — Task 4: Packing Box (Carton Sealing and Packing)

## Objective

Task 4 (`task_number: 4` in `Ubtech_sim/config/Packing_Box.yaml`), officially named "Carton Sealing and Packing" (`datasets/README.md`). The Walker S2 humanoid manipulates a carton (with foam padding built into the box asset itself — there is no separate movable foam piece for this task) using both arms; the dataset's task string is `"Packing Box"`. Scene: `Collected_Task4/SubUSDs/2_small_warehouse2.usd`; episode time limit 100s.

There is no automated success scorer built into this task's environment (unlike Task 1's object-placement tracker) — task completion must be judged behaviorally.

## Dataset used

`datasets/Packing_Box` — an existing official dataset, used as-is (no new collection needed):

| | |
|---|---|
| Episodes | 1,367 |
| Frames | 769,621 |
| FPS | 30 |
| `observation.state` / `action` | 20-D each (14 arm joints + 4 finger joints + 2 gripper commands, both arms — already policy-ready, no truncation needed) |
| Cameras | 4: `head_left`, `head_right`, `wrist_left`, `wrist_right` |
| Task string | `"Packing Box"` |

## Model / training

ACT, trained from scratch (no pretrained checkpoint) reusing Task 1's proven architecture and optimizer settings:

- `dim_model=256`, `n_heads=4`, `dim_feedforward=1024`, `chunk_size=50`, `n_action_steps=50`, `resnet18` backbone, `use_amp=true`
- AdamW, `lr=1e-5`, `weight_decay=1e-4`, `grad_clip_norm=10`, `batch_size=8`, `seed=1000`, no scheduler
- 40,000 steps, checkpoints every 5,000 steps
- Output: `outputs/packing_box_act_40k/`

Loss decreased monotonically from 33.6 (step 50) to 0.035 (step 40,000) with no divergence — a healthy from-scratch training curve, not the small-dataset fine-tuning regime where later checkpoints can overfit.

One infrastructure issue during training: `torchcodec`'s native libraries are broken in this WSL environment (missing `libavutil.so.*`), which crashes DataLoader workers on video decode unless `--dataset.video_backend=pyav` is passed explicitly. Fixed by adding that flag; not a data or model issue.

## Checkpoint screening

8 checkpoints (5K–40K, every 5K) × 2 seeds = 16 episodes, 90s each, identical seeds across checkpoints. Result: **all 16 episodes ran cleanly for all 8 checkpoints** — zero runtime errors, zero non-finite actions, FPS uniform (~13–14) across every checkpoint. Runtime metrics do not discriminate between checkpoints (all equally stable), and no automated task-success signal exists in this environment, so the checkpoint choice rests on the training-loss curve (monotonic, converged, from-scratch regime favors the final checkpoint) plus this clean screening result.

## Confirmation evaluation

10 episodes on the selected checkpoint (040000), seeds 101–110, 100s each (matching the task's own time limit):

- **10/10 episodes completed cleanly** — zero runtime errors, zero non-finite actions, zero retries needed
- FPS: 12.9–13.7 (mean 13.5), consistent with Task 1's evaluation infrastructure
- Control steps per episode: 1,289–1,368

## Selected checkpoint

```
outputs/packing_box_act_40k/checkpoints/040000/
```

This is now the default checkpoint in `run_act_packing_box_windows.py` (used automatically when `--checkpoint` is not passed).

## Exact inference command

```bash
set "ISAAC_SIM=C:\path\to\isaac-sim-standalone-5.1.0-windows-x86_64"
run_act_packing_box_windows.bat
```

Runs one visible Walker S2 Packing Box episode with the selected checkpoint. Override checkpoint/duration/seed via `--checkpoint`, `--duration`, `--seed`.

## Environment requirements

Same as Task 1: Isaac Sim 5.1.0 standalone, this repository's Python environment, `assets/` submodule downloaded (see main `README.md`). `ISAAC_SIM` environment variable must be set before running the `.bat` launcher.

## Known limitations

- **No automated task-success metric exists for this environment.** Checkpoint selection is supported by training-loss convergence and 26/26 clean runtime episodes (screening + confirmation), not a measured completion/success rate. A ground-truth success scorer (e.g. tracking box-lid articulation state, if the scene exposes it) would meaningfully strengthen future evaluation of this task but was out of scope given the deadline.
- The from-scratch training budget (40K steps, ≈0.42 epochs of the dataset) is smaller than Task 1's original 200K-step run (≈0.84 epochs); more steps were not pursued given the deadline, since loss had already converged smoothly to 0.035 with no sign of further useful decrease being blocked.
