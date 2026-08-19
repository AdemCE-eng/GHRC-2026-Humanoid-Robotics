"""Run one visible Walker S2 Part_Sorting ACT evaluation episode under a
controlled Recovery V2 scenario (normal / difficult_position /
displaced_during_approach / missed_first_grasp / drop_and_regrasp).

ACT controls the robot for the entire rollout. The evaluator may only ever
create the intended physical disturbance (move the target object, or force a
brief gripper release) -- it never drives the recovery trajectory. See
recovery_v2_eval_disturbance.py for the enforced boundary.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
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

from src.lerobot.auto_collect.recovery_v2 import classify_object_recoverability, SafetyConfig  # noqa: E402
from src.lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features  # noqa: E402
from src.lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from src.lerobot.policies.utils import make_robot_action, prepare_observation_for_inference_walker_s2  # noqa: E402
from src.lerobot.robots.walker_s2_sim import WalkerS2Config, WalkerS2sim  # noqa: E402
from src.lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from src.lerobot.utils.control_utils import predict_action  # noqa: E402

from act_displacement_diagnostics import ObjectOutcomeTracker, pose_payload  # noqa: E402
from run_act_part_sorting_windows import (  # noqa: E402
    append_jsonl,
    diagnostic_box_bounds,
    diagnostic_object_poses,
    sha256,
    state_vector,
    wait_for_live_cameras,
)

import recovery_v2_eval_disturbance as disturbance  # noqa: E402

SCENARIOS = (
    "normal",
    "difficult_position",
    "displaced_during_approach",
    "missed_first_grasp",
    "drop_and_regrasp",
)
DEFAULT_PILOT_CONFIG = PROJECT_ROOT / "configs" / "recovery_v2_part_sorting_pilot_v2.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--target-slot",
        type=int,
        default=0,
        choices=(0, 1, 2, 3),
        help="Which of the 4 spawned parts (by trailing _NN in its prim path) is the scenario target. "
        "Ignored for --scenario=normal.",
    )
    parser.add_argument(
        "--difficult-position-index",
        type=int,
        default=0,
        help="Index into disturbance config's difficult_positions_xyz. Only used for difficult_position.",
    )
    parser.add_argument(
        "--displaced-trigger-time-s",
        type=float,
        default=8.0,
        help="Fixed wall-clock time after control begins at which the displaced_during_approach push fires.",
    )
    parser.add_argument("--pilot-config", type=Path, default=DEFAULT_PILOT_CONFIG)
    parser.add_argument("--duration", type=float, default=240.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "act_recovery_v2_eval",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--keep-open", action="store_true")
    return parser.parse_args()


def slot_from_path(prim_path: str) -> int:
    match = re.search(r"_(\d+)$", str(prim_path))
    if match is None:
        raise ValueError(f"Cannot map Part Sorting slot from {prim_path}")
    return int(match.group(1))


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = disturbance.load_disturbance_config(args.pilot_config)
    safety = SafetyConfig(
        workspace_x=cfg.workspace_x,
        workspace_y=cfg.workspace_y,
        workspace_z=(1.00, 1.12),
        minimum_object_separation_m=0.10,
        max_recovery_attempts_per_object=3,
        max_episode_seconds=180.0,
        joint_limit_margin_rad=0.01,
        gripper_limit_tolerance_m=0.001,
    )

    required = [
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "policy_preprocessor.json",
        checkpoint / "policy_postprocessor.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint; missing: {missing}")

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{args.scenario}_seed{args.seed}_{stamp}"
    safe_run_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    log_path = logs_dir / f"recovery_v2_eval_{safe_run_id}.log"
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    events_path = results_dir / f"{safe_run_id}.jsonl"
    summary_path = results_dir / f"{safe_run_id}.summary.json"
    events_path.unlink(missing_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    log = logging.getLogger("recovery_v2_act_eval")
    log.info(
        "EVAL_START scenario=%s seed=%d target_slot=%d checkpoint=%s run_id=%s",
        args.scenario, args.seed, args.target_slot, checkpoint, safe_run_id,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the ACT evaluator")
    log.info("Checkpoint model.safetensors sha256=%s", sha256(checkpoint / "model.safetensors"))

    config = WalkerS2Config(headless=False, head_viz_enabled=False)
    config.load_from_yaml("Part_Sorting")
    duration_s = float(args.duration)
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
    if not (policy.config.chunk_size == 50 and policy.config.n_action_steps == 50 and policy.config.temporal_ensemble_coeff is None):
        raise RuntimeError(
            "Controlled-experiment settings changed: expected chunk_size=50, n_action_steps=50, "
            f"no temporal ensembling; got chunk_size={policy.config.chunk_size}, "
            f"n_action_steps={policy.config.n_action_steps}, "
            f"temporal_ensemble_coeff={policy.config.temporal_ensemble_coeff}"
        )
    policy.to(torch.device("cuda"))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    log.info("CHECKPOINT_LOADED chunk_size=%s n_action_steps=%s", policy.config.chunk_size, policy.config.n_action_steps)

    target_prim_path: str | None = None
    disturbance_triggered = False  # the injection code ran (attempted once)
    disturbance_confirmed = False  # the injection's physical effect was verified
    disturbance_event: dict | None = None
    disturbance_type = {
        "normal": None,
        "difficult_position": "initial_placement",
        "displaced_during_approach": "closed_loop_push",
        "missed_first_grasp": "closed_loop_push_small",
        "drop_and_regrasp": "forced_gripper_release",
    }[args.scenario]
    release_frames_remaining = 0
    grasp_detector: disturbance.GraspHoldDetector | None = None
    first_grasp_result = "not_attempted"
    recovery_attempted = False
    recovery_success = False
    reacquired_after_disturbance = False

    robot_connected = False
    try:
        log.info("Connecting visible Isaac Sim GUI and building Part_Sorting scene")
        robot.connect()
        robot_connected = True
        robot._robot_interface._smooth_alpha = 0.98

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        # Identify the target object's prim path (by trailing slot number) before
        # any scenario setup, from ground-truth scene state (diagnostics only).
        if args.scenario != "normal":
            parts = diagnostic_object_poses(robot)
            for part in parts:
                if slot_from_path(part["prim_path"]) == args.target_slot:
                    target_prim_path = str(part["prim_path"])
                    break
            if target_prim_path is None:
                raise RuntimeError(f"Could not find a part for target_slot={args.target_slot}")
            log.info("TARGET_OBJECT prim_path=%s", target_prim_path)

        # difficult_position: pure initial-condition change, BEFORE ACT's first
        # observation, so this is scene setup, not a mid-episode intervention.
        if args.scenario == "difficult_position":
            xyz = cfg.difficult_positions_xyz[args.difficult_position_index % len(cfg.difficult_positions_xyz)]
            disturbance_event = disturbance.teleport_to_difficult_position(robot, target_prim_path, xyz, log)
            disturbance_triggered = True
            settle_error_m = float(np.linalg.norm(np.asarray(disturbance_event["settled_xyz"]) - np.asarray(xyz)))
            disturbance_event["settle_error_m"] = settle_error_m
            disturbance_confirmed = settle_error_m < 0.05  # settled within 5cm of the requested spawn point

        first_observation, image_ranges = wait_for_live_cameras(robot, log)
        robot.reset_timing_metrics()

        initial_state = state_vector(first_observation, state_names)
        log.info("SIM_READY initial_state_range=(%.4f, %.4f) camera_ranges=%s", initial_state.min(), initial_state.max(), image_ranges)

        raw_initial_poses = diagnostic_object_poses(robot)
        initial_object_poses = [pose_payload(item) for item in raw_initial_poses]
        box_bounds = diagnostic_box_bounds(robot)
        tracker = ObjectOutcomeTracker(threshold_m=0.03, recovery_window_s=20.0, box_bounds=box_bounds)
        tracker.initialize(raw_initial_poses)

        if args.scenario in ("missed_first_grasp", "drop_and_regrasp"):
            grasp_detector = disturbance.GraspHoldDetector(
                target_prim_path, cfg.grasp_detect_distance_m, cfg.drop_confirm_steps
            )

        append_jsonl(events_path, {
            "event": "run_start",
            "wall_timestamp": datetime.now().astimezone().isoformat(),
            "run_id": safe_run_id,
            "scenario": args.scenario,
            "seed": args.seed,
            "target_slot": args.target_slot,
            "target_prim_path": target_prim_path,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
            "initial_object_poses": initial_object_poses,
            "destination_box_bounds": box_bounds,
            "disturbance_type": disturbance_type,
            "disturbance_event_at_setup": disturbance_event,
        })

        log.info("POLICY_CONTROL_BEGIN scenario=%s duration_s=%.1f", args.scenario, duration_s)
        started = time.perf_counter()
        observation = first_observation
        steps = 0
        placement_events: list[dict] = []
        escape_ever_detected = False

        while time.perf_counter() - started < duration_s:
            loop_started = time.perf_counter()
            if robot._kit is not None and not robot._kit.is_running():
                log.warning("Isaac Sim window was closed; ending episode early")
                break

            observation_frame = build_dataset_frame(ds_features, observation, prefix=OBS_STR)
            action_tensor = predict_action(
                observation=observation_frame, policy=policy, device=torch.device("cuda"),
                preprocessor=preprocessor, postprocessor=postprocessor,
                use_amp=bool(policy.config.use_amp), task="Part_Sorting", robot_type=robot.robot_type,
            )
            if not bool(torch.isfinite(action_tensor).all()):
                raise RuntimeError("ACT produced a non-finite action; no unsafe command was sent")
            action_np = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            action_dict = dict(zip(action_names, action_np.tolist(), strict=True))

            elapsed_s = time.perf_counter() - started

            # --- displaced_during_approach: fixed-time object-only push, no
            # queue flush / policy.reset() (ACT must deal with its own queue). ---
            if (
                args.scenario == "displaced_during_approach"
                and not disturbance_triggered
                and elapsed_s >= args.displaced_trigger_time_s
            ):
                # Fire exactly once, regardless of outcome. The push can be
                # physically blocked (e.g. the robot's own gripper already in
                # contact with the object) -- that is itself valid, informative
                # data to log, not a reason to keep retrying forever.
                disturbance_triggered = True
                disturbance_event = disturbance.apply_closed_loop_push(
                    robot, target_prim_path, cfg.displacement_velocity_world_mps, cfg.displacement_steps,
                    cfg.displacement_threshold_m, cfg.displacement_upper_bound_m, cfg.displacement_edge_margin_m,
                    cfg.workspace_x, cfg.workspace_y, log,
                )
                disturbance_event["reached_threshold"] = disturbance_event["final_displacement_m"] >= cfg.displacement_threshold_m
                disturbance_confirmed = disturbance_event["reached_threshold"]
                append_jsonl(events_path, {"event": "disturbance", "control_step": steps, "timestamp_s": elapsed_s, **disturbance_event})
                observation = robot.get_observation()
                continue  # re-observe post-disturbance state before sending ACT's next action

            # --- missed_first_grasp: small nudge exactly as ACT first closes on the target. ---
            if args.scenario == "missed_first_grasp" and grasp_detector is not None and not disturbance_triggered:
                first_close, _ = grasp_detector.update(robot, steps)
                if first_close:
                    disturbance_event = disturbance.apply_closed_loop_push(
                        robot, target_prim_path, cfg.missed_grasp_velocity_world_mps, cfg.missed_grasp_steps,
                        cfg.missed_grasp_lower_bound_m, cfg.missed_grasp_upper_bound_m, cfg.missed_grasp_edge_margin_m,
                        cfg.workspace_x, cfg.workspace_y, log,
                    )
                    disturbance_triggered = True
                    disturbance_confirmed = disturbance_event["final_displacement_m"] >= cfg.missed_grasp_lower_bound_m
                    first_grasp_result = "disturbed"
                    append_jsonl(events_path, {"event": "disturbance", "control_step": steps, "timestamp_s": elapsed_s, **disturbance_event})
                    observation = robot.get_observation()
                    continue

            # --- drop_and_regrasp: detect a stable ACT-achieved grasp, force a
            # brief gripper-open override (ONLY the gripper scalar), then hand
            # control straight back. ---
            if args.scenario == "drop_and_regrasp" and grasp_detector is not None:
                first_close, stable_hold = grasp_detector.update(robot, steps)
                if first_close:
                    first_grasp_result = "in_progress"
                if stable_hold and not disturbance_triggered:
                    first_grasp_result = "success"
                    release_frames_remaining = cfg.drop_release_steps
                    disturbance_triggered = True
                    # stable_hold already required a sustained (drop_confirm_steps)
                    # gripping+proximity read from the robot's own ground-truth
                    # state, so the "genuine grasp" precondition is verified here,
                    # not assumed.
                    disturbance_confirmed = True
                    disturbance_event = {"trigger_control_step": steps, "release_steps": cfg.drop_release_steps}
                    log.info("DISTURBANCE drop_and_regrasp forcing release at step=%d", steps)
                    append_jsonl(events_path, {"event": "disturbance", "control_step": steps, "timestamp_s": elapsed_s, **disturbance_event})
                if release_frames_remaining > 0:
                    action_dict = disturbance.force_gripper_release(action_dict, release_frames_remaining)
                    release_frames_remaining -= 1

            action_np = np.asarray([action_dict[name] for name in action_names], dtype=np.float32)
            robot.send_action(make_robot_action(torch.as_tensor(action_np).unsqueeze(0), ds_features))
            steps += 1
            observation = robot.get_observation()

            displacement_events, new_placements = tracker.update(diagnostic_object_poses(robot), elapsed_s=elapsed_s, control_step=steps)
            for placement in new_placements:
                placement["wall_timestamp"] = datetime.now().astimezone().isoformat()
                append_jsonl(events_path, placement)
                placement_events.append(placement)
                log.info("OBJECT_PLACED object=%s step=%d total_placed=%d", placement["object_id"], steps, len(tracker.placed_at))

            if target_prim_path is not None and disturbance_confirmed:
                distance = disturbance.end_effector_object_distance(robot, target_prim_path, "right")
                if distance is not None and distance <= cfg.grasp_detect_distance_m:
                    reacquired_after_disturbance = True

            if target_prim_path is not None:
                current_xyz = disturbance.object_xyz(robot, target_prim_path)
                current_orientation = np.asarray(disturbance.object_pose(robot, target_prim_path)["orientation"], dtype=np.float64)
                status, _reason = classify_object_recoverability(current_xyz, current_orientation, safety)
                if status == "UNRECOVERABLE_ESCAPE":
                    escape_ever_detected = True

            now = time.perf_counter()
            sleep_s = (1.0 / args.fps) - (time.perf_counter() - loop_started)
            if sleep_s > 0:
                time.sleep(sleep_s)

        elapsed_s = time.perf_counter() - started
        if steps == 0:
            raise RuntimeError("Episode ended before any ACT action was sent")

        target_placed = target_prim_path is not None and target_prim_path in tracker.placed_at
        # Per spec: recovery_success only counts if the intended disturbance
        # actually, physically occurred (disturbance_confirmed), not merely
        # attempted (disturbance_triggered).
        recovery_attempted = bool(disturbance_confirmed and (reacquired_after_disturbance or target_placed))
        recovery_success = bool(disturbance_confirmed and target_placed)
        if args.scenario == "missed_first_grasp" and first_grasp_result == "disturbed":
            first_grasp_result = "failed"  # the disturbance is defined to make the first attempt fail
        if args.scenario != "drop_and_regrasp" and target_prim_path is not None and first_grasp_result == "not_attempted":
            first_grasp_result = "success" if target_placed else "unknown"

        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
            "seed": args.seed,
            "scenario": args.scenario,
            "target_slot": args.target_slot,
            "target_prim_path": target_prim_path,
            "disturbance_triggered": bool(disturbance_triggered),
            "disturbance_confirmed": bool(disturbance_confirmed),
            "disturbance_type": disturbance_type,
            "disturbance_event": disturbance_event,
            "first_grasp_result": first_grasp_result,
            "recovery_attempted": recovery_attempted if args.scenario != "normal" else None,
            "recovery_success": recovery_success if args.scenario != "normal" else None,
            "target_placement_success": target_placed if target_prim_path is not None else None,
            "total_objects_placed": len(tracker.placed_at),
            "objects_total": len(initial_object_poses),
            "full_task_completion": len(tracker.placed_at) == len(initial_object_poses),
            "object_escape": escape_ever_detected,
            "control_fps": steps / max(elapsed_s, 1e-9),
            "control_steps": steps,
            "elapsed_s": elapsed_s,
            "termination_reason": "duration_limit_reached",
            "events_path": str(events_path),
            "text_log_path": str(log_path),
        }
        with summary_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        append_jsonl(events_path, {"event": "run_summary", "wall_timestamp": datetime.now().astimezone().isoformat(), **summary})
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
