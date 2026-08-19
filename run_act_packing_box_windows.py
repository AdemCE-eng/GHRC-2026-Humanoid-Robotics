"""Run one visible Walker S2 Packing_Box (Task 4 -- Carton Sealing and Packing)
episode with a trained ACT policy.

No automated success scorer exists for this task (unlike Part_Sorting's
object-placement tracker) -- success is judged behaviorally by observation,
per the Task 4 evaluation plan. This script proves the checkpoint loads,
runs inference at a reasonable control rate, and produces only finite
actions; it logs a JSON summary for offline comparison across checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = PROJECT_ROOT / ".windows-act-dependencies"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Ubtech_sim.source  # noqa: F401, E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features  # noqa: E402
from src.lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from src.lerobot.policies.utils import make_robot_action  # noqa: E402
from src.lerobot.robots.walker_s2_sim import WalkerS2Config, WalkerS2sim  # noqa: E402
from src.lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from src.lerobot.utils.control_utils import predict_action  # noqa: E402

from run_act_part_sorting_windows import diagnostic_box_bounds, sha256, wait_for_live_cameras  # noqa: E402


def _bounds_extent(bounds: dict | None) -> list[float] | None:
    """[dx, dy, dz] size of the box AABB -- a coarse, task-agnostic proxy for
    whether the box geometry changed shape (e.g. a flap folding closed)."""
    if bounds is None:
        return None
    return [bounds["max_m"][i] - bounds["min_m"][i] for i in range(3)]

# Selected final Task 4 checkpoint: ACT trained from scratch for 40,000 steps
# on the full 1,367-episode Packing_Box dataset. See PACKING_BOX_FINAL.md for
# the screening (8 checkpoints x 2 seeds) and confirmation (10 seeds) that
# selected this checkpoint -- all 8 trained checkpoints ran without any
# runtime failure; this task has no automated success scorer, so selection
# rests on training-loss convergence (0.035, monotonic, no divergence -- a
# from-scratch regime, not a fine-tuning-overfitting one) plus 26/26 clean
# evaluation episodes across screening and confirmation.
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "packing_box_act_40k" / "checkpoints" / "040000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to the selected Task 4 checkpoint.")
    parser.add_argument("--duration", type=float, default=100.0, help="Episode duration in seconds (matches Packing_Box.yaml timelimit).")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "experiments" / "act_packing_box_eval")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = (args.checkpoint or DEFAULT_CHECKPOINT).resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    required = [
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "policy_preprocessor.json",
        checkpoint / "policy_postprocessor.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint; missing: {missing}")

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"packing_box_seed{args.seed}_{stamp}"
    safe_run_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    log_path = logs_dir / f"act_packing_box_eval_{safe_run_id}.log"
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{safe_run_id}.summary.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    log = logging.getLogger("act_packing_box_eval")
    log.info("EVAL_START checkpoint=%s seed=%d run_id=%s", checkpoint, args.seed, safe_run_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the ACT evaluator")
    log.info("Checkpoint model.safetensors sha256=%s", sha256(checkpoint / "model.safetensors"))

    config = WalkerS2Config(headless=False, head_viz_enabled=False)
    config.load_from_yaml("Packing_Box")
    robot = WalkerS2sim(config)

    hardware_ds_features = {
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
    }
    action_names = list(hardware_ds_features[ACTION]["names"])
    state_names = action_names.copy()
    camera_keys = sorted(key for key in hardware_ds_features if key.startswith("observation.images."))
    if len(action_names) != 20 or len(camera_keys) != 4:
        raise RuntimeError(f"Walker/checkpoint feature mismatch: actions={len(action_names)}, cameras={camera_keys}")
    ds_features = {
        ACTION: hardware_ds_features[ACTION],
        f"{OBS_STR}.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        **{key: hardware_ds_features[key] for key in camera_keys},
    }

    log.info("Loading ACT checkpoint weights and saved normalization processors")
    policy = get_policy_class("act").from_pretrained(checkpoint)
    policy.to(torch.device("cuda"))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    log.info("CHECKPOINT_LOADED chunk_size=%s n_action_steps=%s", policy.config.chunk_size, policy.config.n_action_steps)

    robot_connected = False
    non_finite_action_count = 0
    try:
        log.info("Connecting visible Isaac Sim GUI and building Packing_Box scene")
        robot.connect()
        robot_connected = True

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        first_observation, image_ranges = wait_for_live_cameras(robot, log)
        robot.reset_timing_metrics()
        log.info("SIM_READY camera_ranges=%s", image_ranges)

        initial_box_bounds = diagnostic_box_bounds(robot)
        log.info("INITIAL_BOX_BOUNDS %s", initial_box_bounds)

        log.info("POLICY_CONTROL_BEGIN duration_s=%.1f", args.duration)
        started = time.perf_counter()
        observation = first_observation
        steps = 0

        while time.perf_counter() - started < args.duration:
            loop_started = time.perf_counter()
            if robot._kit is not None and not robot._kit.is_running():
                log.warning("Isaac Sim window was closed; ending episode early")
                break

            observation_frame = build_dataset_frame(ds_features, observation, prefix=OBS_STR)
            action_tensor = predict_action(
                observation=observation_frame, policy=policy, device=torch.device("cuda"),
                preprocessor=preprocessor, postprocessor=postprocessor,
                use_amp=bool(policy.config.use_amp), task="Packing Box", robot_type=robot.robot_type,
            )
            if not bool(torch.isfinite(action_tensor).all()):
                non_finite_action_count += 1
                raise RuntimeError("ACT produced a non-finite action; no unsafe command was sent")
            action_np = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            action_dict = dict(zip(action_names, action_np.tolist(), strict=True))

            robot.send_action(make_robot_action(torch.as_tensor(action_np).unsqueeze(0), ds_features))
            steps += 1
            observation = robot.get_observation()

            sleep_s = (1.0 / args.fps) - (time.perf_counter() - loop_started)
            if sleep_s > 0:
                time.sleep(sleep_s)

        elapsed_s = time.perf_counter() - started
        if steps == 0:
            raise RuntimeError("Episode ended before any ACT action was sent")

        final_box_bounds = diagnostic_box_bounds(robot)
        log.info("FINAL_BOX_BOUNDS %s", final_box_bounds)
        initial_extent = _bounds_extent(initial_box_bounds)
        final_extent = _bounds_extent(final_box_bounds)
        extent_change_m = (
            [round(final_extent[i] - initial_extent[i], 5) for i in range(3)]
            if initial_extent and final_extent else None
        )

        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
            "seed": args.seed,
            "control_fps": steps / max(elapsed_s, 1e-9),
            "control_steps": steps,
            "elapsed_s": elapsed_s,
            "non_finite_action_count": non_finite_action_count,
            # No automated success scorer exists for this task; box AABB extent
            # change is a coarse, task-agnostic proxy for "did the box geometry
            # change" (e.g. a flap folding closed). Final judgment is behavioral
            # (visual), per the Task 4 evaluation plan.
            "initial_box_bounds": initial_box_bounds,
            "final_box_bounds": final_box_bounds,
            "box_extent_change_m": extent_change_m,
            "termination_reason": "duration_limit_reached",
            "text_log_path": str(log_path),
        }
        with summary_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        log.info("EVAL_SUMMARY %s", json.dumps(summary, sort_keys=True))
        log.info("Evaluation log=%s", log_path)

        if args.keep_open:
            log.info("SIMULATION_HOLDING_OPEN close the Isaac Sim window to exit")
            while robot._kit is not None and robot._kit.is_running():
                robot.step(render=True)
                time.sleep(1.0 / args.fps)
        return 0
    finally:
        if robot_connected:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
