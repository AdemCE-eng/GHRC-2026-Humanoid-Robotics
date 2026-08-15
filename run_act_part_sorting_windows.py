"""Run one visible Walker S2 Part_Sorting episode with a trained LeRobot policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = PROJECT_ROOT / ".windows-act-dependencies"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register the project-local Pinocchio binaries before importing Walker S2.
import Ubtech_sim.source  # noqa: F401, E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features  # noqa: E402
from src.lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from src.lerobot.policies.utils import make_robot_action  # noqa: E402
from src.lerobot.robots.walker_s2_sim import WalkerS2Config, WalkerS2sim  # noqa: E402
from src.lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from src.lerobot.utils.control_utils import predict_action  # noqa: E402

from act_displacement_diagnostics import (  # noqa: E402
    ObjectOutcomeTracker,
    pose_payload,
    timing_summary,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "part_sorting_act_50k" / "pretrained_model"
EXPERIMENT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "part_sorting_act_200k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory. Diagnostic modes default to the pinned 200K checkpoint.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Episode duration in seconds (defaults to Part_Sorting.yaml timelimit).",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="ACT-only actions executed before replanning (defaults to the checkpoint value).",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep Isaac Sim open at the final state until its window is closed.",
    )
    parser.add_argument(
        "--experiment",
        choices=("off", "normal", "replan-on-displacement"),
        default="off",
        help="Opt-in stale-queue diagnostic. 'off' preserves the ordinary evaluator path.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Python/NumPy/Torch scene and policy seed.")
    parser.add_argument(
        "--displacement-threshold-cm",
        type=float,
        default=3.0,
        help="Sustained horizontal tabletop motion required for a displacement event.",
    )
    parser.add_argument(
        "--recovery-window",
        type=float,
        default=20.0,
        help="Seconds after displacement in which placement counts as recovery.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "act_queue_displacement",
    )
    parser.add_argument("--run-id", default=None, help="Stable paired-run identifier used in output names.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_vector(observation: dict, names: list[str]) -> np.ndarray:
    return np.asarray([float(observation[name]) for name in names], dtype=np.float32)


def wait_for_live_cameras(robot, log: logging.Logger, max_steps: int = 60) -> tuple[dict, dict]:
    """Render until every policy camera has a non-uniform RGB frame."""
    for camera_warmup_step in range(1, max_steps + 1):
        robot.step(render=True)
        candidate = robot.get_observation()
        candidate_ranges = {
            name: (float(candidate[name].min()), float(candidate[name].max()))
            for name in robot.CAMERA_NAMES
        }
        if all(high > low for low, high in candidate_ranges.values()):
            log.info("CAMERAS_READY rendered_warmup_steps=%d ranges=%s", camera_warmup_step, candidate_ranges)
            return candidate, candidate_ranges
    raise RuntimeError(
        f"Isaac cameras did not produce non-uniform RGB frames after {max_steps} rendered warm-up steps"
    )


def diagnostic_object_poses(robot) -> list[dict]:
    """Read simulator ground truth for diagnostics only; never return policy inputs."""
    scene = robot._scene_builder
    type_by_path = getattr(scene, "part_type_by_prim_path", {})
    poses = []
    for raw_pose in scene.get_parts_world_poses():
        item = dict(raw_pose)
        item["object_type"] = type_by_path.get(item["prim_path"])
        poses.append(item)
    return poses


def diagnostic_box_bounds(robot) -> dict[str, list[float]] | None:
    """Return the locked destination box world AABB for outcome measurement."""
    import omni.usd
    from pxr import Usd, UsdGeom

    scene = robot._scene_builder
    paths = getattr(scene, "box_prim_paths", [])
    if not paths:
        return None
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(paths[0])
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    lower = aligned.GetMin()
    upper = aligned.GetMax()
    return {
        "prim_path": str(paths[0]),
        "min_m": [float(lower[i]) for i in range(3)],
        "max_m": [float(upper[i]) for i in range(3)],
    }


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def action_payload(action: np.ndarray | None) -> list[float] | None:
    if action is None:
        return None
    return [float(value) for value in np.asarray(action).reshape(-1)]


def main() -> int:
    args = parse_args()
    checkpoint = (args.checkpoint or (EXPERIMENT_CHECKPOINT if args.experiment != "off" else DEFAULT_CHECKPOINT)).resolve()
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
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint; missing: {missing}")
    with (checkpoint / "config.json").open("r", encoding="utf-8") as stream:
        checkpoint_config = json.load(stream)
    policy_type = str(checkpoint_config.get("type", "")).lower()
    if policy_type not in {"act", "diffusion"}:
        raise ValueError(
            f"Unsupported checkpoint policy type {policy_type!r}; expected 'act' or 'diffusion'"
        )
    if policy_type != "act" and args.n_action_steps is not None:
        raise ValueError("--n-action-steps is an ACT-only inference override")

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"seed{args.seed}_{args.experiment}_{stamp}"
    safe_run_id = "".join(character if character.isalnum() or character in "-_." else "_" for character in run_id)
    log_path = logs_dir / f"part_sorting_{policy_type}_eval_{safe_run_id}.log"
    results_dir = args.results_dir.resolve()
    events_path = results_dir / f"{safe_run_id}.jsonl"
    summary_path = results_dir / f"{safe_run_id}.summary.json"
    if args.experiment != "off":
        results_dir.mkdir(parents=True, exist_ok=True)
        events_path.unlink(missing_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    log = logging.getLogger("act_part_sorting_eval")

    log.info(
        "EVAL_START task=Part_Sorting policy_type=%s episodes=1 headless=false experiment=%s seed=%d run_id=%s",
        policy_type,
        args.experiment,
        args.seed,
        safe_run_id,
    )
    log.info("Runtime Python=%s NumPy=%s Torch=%s", sys.version.split()[0], np.__version__, torch.__version__)
    if np.__version__.split(".")[:2] != ["1", "26"]:
        raise RuntimeError(f"Expected NumPy 1.26.x, found {np.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the ACT evaluator")
    log.info("CUDA device=%s", torch.cuda.get_device_name(0))
    log.info("Checkpoint=%s", checkpoint)
    log.info("Checkpoint model.safetensors sha256=%s", sha256(checkpoint / "model.safetensors"))
    log.info(
        "Processors preprocessor_sha256=%s postprocessor_sha256=%s",
        sha256(checkpoint / "policy_preprocessor.json"),
        sha256(checkpoint / "policy_postprocessor.json"),
    )

    # The Isaac viewport remains visible. Disable only the optional OpenCV HighGUI
    # mirror, because the repo intentionally depends on opencv-python-headless.
    config = WalkerS2Config(headless=False, head_viz_enabled=False)
    config.load_from_yaml("Part_Sorting")
    duration_s = float(args.duration if args.duration is not None else config.task_cfg["timelimit"])
    robot = WalkerS2sim(config)

    hardware_ds_features = {
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
    }
    action_names = list(hardware_ds_features[ACTION]["names"])
    state_names = action_names.copy()
    camera_keys = sorted(key for key in hardware_ds_features if key.startswith("observation.images."))
    if len(action_names) != 20 or len(camera_keys) != 4:
        raise RuntimeError(
            f"Walker/checkpoint feature mismatch: actions={len(action_names)}, cameras={camera_keys}"
        )
    # The robot exposes object pose scalars for dataset replay.  They are
    # intentionally excluded here: ACT receives only the checkpoint's original
    # 20 joint/gripper state values and four camera images.  Diagnostic pose
    # reads happen separately through diagnostic_object_poses().
    ds_features = {
        ACTION: hardware_ds_features[ACTION],
        f"{OBS_STR}.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        **{key: hardware_ds_features[key] for key in camera_keys},
    }

    log.info("Loading %s checkpoint weights and saved normalization processors", policy_type.upper())
    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(checkpoint)
    checkpoint_n_action_steps = int(policy.config.n_action_steps)
    if policy_type == "act":
        effective_n_action_steps = (
            checkpoint_n_action_steps if args.n_action_steps is None else int(args.n_action_steps)
        )
        if not 1 <= effective_n_action_steps <= int(policy.config.chunk_size):
            raise ValueError(
                "--n-action-steps must be between 1 and chunk_size "
                f"({policy.config.chunk_size}), got {effective_n_action_steps}"
            )
        # ACT-only inference override. This changes its queue cadence without
        # changing the checkpoint or prediction chunk.
        policy.config.n_action_steps = effective_n_action_steps
    if args.experiment != "off":
        required_experiment_settings = {
            "policy_type": policy_type,
            "chunk_size": int(policy.config.chunk_size),
            "n_action_steps": int(policy.config.n_action_steps),
            "n_obs_steps": int(policy.config.n_obs_steps),
            "temporal_ensemble_coeff": policy.config.temporal_ensemble_coeff,
        }
        expected_experiment_settings = {
            "policy_type": "act",
            "chunk_size": 50,
            "n_action_steps": 50,
            "n_obs_steps": 1,
            "temporal_ensemble_coeff": None,
        }
        if required_experiment_settings != expected_experiment_settings:
            raise RuntimeError(
                "Controlled experiment settings changed: "
                f"actual={required_experiment_settings}, expected={expected_experiment_settings}"
            )
    policy.to(torch.device("cuda"))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if policy_type == "act":
        log.info(
            "CHECKPOINT_LOADED class=%s parameters=%d device=%s use_amp=%s chunk_size=%s checkpoint_n_action_steps=%s effective_n_action_steps=%s",
            type(policy).__name__,
            parameter_count,
            next(policy.parameters()).device,
            policy.config.use_amp,
            policy.config.chunk_size,
            checkpoint_n_action_steps,
            policy.config.n_action_steps,
        )
    else:
        log.info(
            "CHECKPOINT_LOADED class=%s parameters=%d device=%s use_amp=%s n_obs_steps=%s horizon=%s n_action_steps=%s",
            type(policy).__name__,
            parameter_count,
            next(policy.parameters()).device,
            policy.config.use_amp,
            policy.config.n_obs_steps,
            policy.config.horizon,
            policy.config.n_action_steps,
        )
    log.info("Features state=20 cameras=%s action=20", camera_keys)
    if args.experiment != "off":
        log.info(
            "DIAGNOSTIC_CONFIG experiment=%s threshold_cm=%.2f confirmation_samples=2 "
            "refractory_s=2.0 lift_exclusion_cm=4.0 recovery_definition=same_object_placed_within_%.1fs",
            args.experiment,
            args.displacement_threshold_cm,
            args.recovery_window,
        )

    initial_state = None
    final_state = None
    max_state_displacement = 0.0
    max_command_delta = 0.0
    action_min = float("inf")
    action_max = float("-inf")
    steps = 0
    reset_count = 0
    robot_connected = False
    reset_window = None
    reset_requested = threading.Event()
    started = time.perf_counter()
    diagnostic_enabled = args.experiment != "off"
    tracker = None
    initial_object_poses: list[dict] = []
    box_bounds = None
    loop_durations_s: list[float] = []
    chunk_inference_durations_s: list[float] = []
    chunk_boundaries: list[dict] = []
    chunk_count = 0
    replan_count = 0
    displacement_count = 0
    placement_events: list[dict] = []
    last_action_np = None
    last_chunk_generated_at = None
    next_chunk_reason = "initial"
    pending_first_action_event: dict | None = None

    try:
        log.info("Connecting visible Isaac Sim GUI and building Part_Sorting scene")
        robot.connect()
        robot_connected = True
        robot._robot_interface._smooth_alpha = 0.98

        # UI callbacks only enqueue work. The simulation loop performs the reset
        # between policy steps, avoiding nested World.step() calls from a UI callback.
        import omni.ui as ui

        def request_episode_reset() -> None:
            reset_requested.set()
            log.info("RESET_REQUESTED source=Isaac_UI")

        reset_window = ui.Window("ACT Evaluation Controls", width=300, height=120)
        with reset_window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label(f"Part_Sorting {policy_type.upper()} policy", height=24)
                ui.Button("Reset Episode", height=42, clicked_fn=request_episode_reset)

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        # Isaac camera render products need a few rendered frames after initialize().
        # Do this before the timed episode so ACT never receives placeholder black frames.
        first_observation, image_ranges = wait_for_live_cameras(robot, log)

        initial_state = state_vector(first_observation, state_names)
        log.info("SIM_READY initial_state_range=(%.4f, %.4f) camera_ranges=%s", initial_state.min(), initial_state.max(), image_ranges)
        if diagnostic_enabled:
            raw_initial_poses = diagnostic_object_poses(robot)
            initial_object_poses = [pose_payload(item) for item in raw_initial_poses]
            box_bounds = diagnostic_box_bounds(robot)
            tracker = ObjectOutcomeTracker(
                threshold_m=args.displacement_threshold_cm / 100.0,
                recovery_window_s=args.recovery_window,
                box_bounds=box_bounds,
            )
            tracker.initialize(raw_initial_poses)
            run_start_event = {
                "event": "run_start",
                "wall_timestamp": datetime.now().astimezone().isoformat(),
                "run_id": safe_run_id,
                "experiment": args.experiment,
                "seed": args.seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
                "chunk_size": int(policy.config.chunk_size),
                "n_action_steps": int(policy.config.n_action_steps),
                "n_obs_steps": int(policy.config.n_obs_steps),
                "state_names": state_names,
                "camera_keys": camera_keys,
                "diagnostic_ground_truth_in_policy": False,
                "initial_object_poses": initial_object_poses,
                "destination_box_bounds": box_bounds,
            }
            append_jsonl(events_path, run_start_event)
            log.info("DIAGNOSTIC_READY objects=%d box_bounds=%s events=%s", len(initial_object_poses), box_bounds, events_path)
        log.info("POLICY_CONTROL_BEGIN duration_s=%.1f target_fps=%.1f", duration_s, args.fps)

        started = time.perf_counter()
        next_progress = started
        observation = first_observation
        while time.perf_counter() - started < duration_s:
            loop_started = time.perf_counter()
            if robot._kit is not None and not robot._kit.is_running():
                log.warning("Isaac Sim window was closed; ending episode early")
                break

            if reset_requested.is_set():
                reset_requested.clear()
                if diagnostic_enabled:
                    raise RuntimeError("Manual episode reset invalidates a controlled diagnostic run")
                reset_count += 1
                log.info("RESET_BEGIN count=%d", reset_count)
                robot.reset()
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                observation, image_ranges = wait_for_live_cameras(robot, log)
                initial_state = state_vector(observation, state_names)
                final_state = initial_state.copy()
                max_state_displacement = 0.0
                max_command_delta = 0.0
                action_min = float("inf")
                action_max = float("-inf")
                steps = 0
                started = time.perf_counter()
                next_progress = started
                log.info("RESET_COMPLETE count=%d episode_timer_restarted=true", reset_count)
                continue

            queue_before = len(policy._action_queue) if policy_type == "act" else 0
            observation_frame = build_dataset_frame(ds_features, observation, prefix=OBS_STR)
            inference_started = time.perf_counter()
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=torch.device("cuda"),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=bool(policy.config.use_amp),
                task="Part_Sorting",
                robot_type=robot.robot_type,
            )
            inference_duration_s = time.perf_counter() - inference_started
            if not bool(torch.isfinite(action_tensor).all()):
                raise RuntimeError("ACT produced a non-finite action; no unsafe command was sent")

            action_np = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if policy_type == "act" and queue_before == 0:
                chunk_count += 1
                chunk_inference_durations_s.append(inference_duration_s)
                generated_at = time.perf_counter()
                loop_rate_hz = (1.0 / loop_durations_s[-1]) if loop_durations_s else None
                boundary = {
                    "event": "chunk_generation",
                    "wall_timestamp": datetime.now().astimezone().isoformat(),
                    "timestamp_s": generated_at - started,
                    "control_step": steps,
                    "chunk_number": chunk_count,
                    "reason": next_chunk_reason,
                    "loop_rate_hz": loop_rate_hz,
                    "queue_length_after_select": len(policy._action_queue),
                    "inference_duration_ms": inference_duration_s * 1000.0,
                    "previous_action": action_payload(last_action_np),
                    "first_action": action_payload(action_np),
                    "boundary_l2": (
                        float(np.linalg.norm(action_np - last_action_np)) if last_action_np is not None else None
                    ),
                    "boundary_max_abs": (
                        float(np.max(np.abs(action_np - last_action_np))) if last_action_np is not None else None
                    ),
                }
                chunk_boundaries.append(boundary)
                if diagnostic_enabled:
                    append_jsonl(events_path, boundary)
                log.info(
                    "CHUNK_GENERATED chunk=%d t=%.3f step=%d reason=%s loop_rate_hz=%s queue_length=%d inference_ms=%.2f boundary_l2=%s boundary_max_abs=%s",
                    chunk_count,
                    generated_at - started,
                    steps,
                    next_chunk_reason,
                    f"{loop_rate_hz:.3f}" if loop_rate_hz is not None else "null",
                    len(policy._action_queue),
                    inference_duration_s * 1000.0,
                    f"{boundary['boundary_l2']:.6f}" if boundary["boundary_l2"] is not None else "null",
                    f"{boundary['boundary_max_abs']:.6f}" if boundary["boundary_max_abs"] is not None else "null",
                )
                last_chunk_generated_at = generated_at
                next_chunk_reason = "scheduled_queue_empty"

            if pending_first_action_event is not None:
                first_action_event = {
                    "event": "replan_first_action",
                    "wall_timestamp": datetime.now().astimezone().isoformat(),
                    "timestamp_s": time.perf_counter() - started,
                    "control_step": steps,
                    "trigger_event_ids": pending_first_action_event["trigger_event_ids"],
                    "first_action_command": action_payload(action_np),
                    "pre_replan_action_command": pending_first_action_event["pre_replan_action_command"],
                    "action_delta_l2": float(
                        np.linalg.norm(action_np - np.asarray(pending_first_action_event["pre_replan_action_command"]))
                    ),
                    "action_delta_max_abs": float(
                        np.max(np.abs(action_np - np.asarray(pending_first_action_event["pre_replan_action_command"])))
                    ),
                }
                append_jsonl(events_path, first_action_event)
                log.info(
                    "REPLAN_FIRST_ACTION trigger_events=%s step=%d delta_l2=%.6f delta_max_abs=%.6f",
                    pending_first_action_event["trigger_event_ids"],
                    steps,
                    first_action_event["action_delta_l2"],
                    first_action_event["action_delta_max_abs"],
                )
                pending_first_action_event = None

            current_state = state_vector(observation, state_names)
            max_command_delta = max(max_command_delta, float(np.max(np.abs(action_np - current_state))))
            action_min = min(action_min, float(action_np.min()))
            action_max = max(action_max, float(action_np.max()))
            robot.send_action(make_robot_action(action_tensor, ds_features))
            steps += 1
            last_action_np = action_np.copy()

            observation = robot.get_observation()
            final_state = state_vector(observation, state_names)
            max_state_displacement = max(
                max_state_displacement,
                float(np.max(np.abs(final_state - initial_state))),
            )

            if diagnostic_enabled and tracker is not None:
                event_time = time.perf_counter()
                elapsed_for_event = event_time - started
                displacement_events, new_placements = tracker.update(
                    diagnostic_object_poses(robot),
                    elapsed_s=elapsed_for_event,
                    control_step=steps,
                )
                placement_events.extend(new_placements)
                for placement in new_placements:
                    placement["wall_timestamp"] = datetime.now().astimezone().isoformat()
                    append_jsonl(events_path, placement)
                    log.info(
                        "OBJECT_PLACED object=%s step=%d t=%.3f total_placed=%d",
                        placement["object_id"],
                        steps,
                        placement["timestamp_s"],
                        len(tracker.placed_at),
                    )

                trigger_event_ids = []
                for displacement in displacement_events:
                    displacement_count += 1
                    event_id = f"displacement_{displacement_count:04d}"
                    trigger_event_ids.append(event_id)
                    queue_remaining = len(policy._action_queue)
                    displacement.update(
                        {
                            "event_id": event_id,
                            "wall_timestamp": datetime.now().astimezone().isoformat(),
                            "experiment": args.experiment,
                            "queue_position": int(policy.config.n_action_steps) - queue_remaining,
                            "remaining_queued_actions": queue_remaining,
                            "time_since_chunk_generated_s": (
                                event_time - last_chunk_generated_at if last_chunk_generated_at is not None else None
                            ),
                            "action_command_before_replan": action_payload(action_np),
                            "queue_cleared": args.experiment == "replan-on-displacement",
                        }
                    )
                    append_jsonl(events_path, displacement)
                    log.info(
                        "DISPLACEMENT event_id=%s object=%s step=%d t=%.3f planar_cm=%.2f queue_position=%d remaining=%d queue_cleared=%s",
                        event_id,
                        displacement["object_id"],
                        steps,
                        displacement["timestamp_s"],
                        displacement["planar_displacement_m"] * 100.0,
                        displacement["queue_position"],
                        queue_remaining,
                        displacement["queue_cleared"],
                    )

                if trigger_event_ids and args.experiment == "replan-on-displacement":
                    replan_count += 1
                    pre_replan_action = action_payload(action_np)
                    policy.reset()
                    next_chunk_reason = "displacement_queue_flush"
                    pending_first_action_event = {
                        "trigger_event_ids": trigger_event_ids,
                        "pre_replan_action_command": pre_replan_action,
                    }
                    append_jsonl(
                        events_path,
                        {
                            "event": "queue_reset",
                            "wall_timestamp": datetime.now().astimezone().isoformat(),
                            "timestamp_s": elapsed_for_event,
                            "control_step": steps,
                            "replan_number": replan_count,
                            "trigger_event_ids": trigger_event_ids,
                            "pre_replan_action_command": pre_replan_action,
                            "remaining_queued_actions_before_reset": displacement_events[-1]["remaining_queued_actions"],
                            "remaining_queued_actions_after_reset": len(policy._action_queue),
                        },
                    )

            now = time.perf_counter()
            if now >= next_progress:
                log.info(
                    "POLICY_CONTROL_ACTIVE t=%.1fs steps=%d action_range=(%.4f, %.4f) max_command_delta=%.4f max_joint_motion=%.4f",
                    now - started,
                    steps,
                    action_min,
                    action_max,
                    max_command_delta,
                    max_state_displacement,
                )
                next_progress = now + 5.0

            sleep_s = (1.0 / args.fps) - (time.perf_counter() - loop_started)
            if sleep_s > 0:
                time.sleep(sleep_s)
            loop_durations_s.append(time.perf_counter() - loop_started)

        elapsed_s = time.perf_counter() - started
        if steps == 0:
            raise RuntimeError("Episode ended before any ACT action was sent")
        log.info(
            "EVAL_SUCCESS checkpoint_loaded=true policy_actions_sent=%d elapsed_s=%.2f average_fps=%.2f action_range=(%.4f, %.4f) max_command_delta=%.4f max_joint_motion=%.4f",
            steps,
            elapsed_s,
            steps / max(elapsed_s, 1e-9),
            action_min,
            action_max,
            max_command_delta,
            max_state_displacement,
        )
        if diagnostic_enabled and tracker is not None:
            recoveries = tracker.finalize()
            recovered_count = sum(1 for recovery in recoveries if recovery["recovered"])
            scheduled_boundaries = [
                item for item in chunk_boundaries if item["reason"] == "scheduled_queue_empty"
            ]
            boundary_max_abs = [
                item["boundary_max_abs"] for item in scheduled_boundaries if item["boundary_max_abs"] is not None
            ]
            placement_times = sorted(float(value) for value in tracker.placed_at.values())
            summary = {
                "run_id": safe_run_id,
                "experiment": args.experiment,
                "seed": args.seed,
                "duration_limit_s": duration_s,
                "elapsed_s": elapsed_s,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
                "policy_settings": {
                    "chunk_size": int(policy.config.chunk_size),
                    "n_action_steps": int(policy.config.n_action_steps),
                    "n_obs_steps": int(policy.config.n_obs_steps),
                    "temporal_ensemble_coeff": policy.config.temporal_ensemble_coeff,
                    "state_dimension": len(state_names),
                    "cameras": camera_keys,
                    "diagnostic_ground_truth_in_policy": False,
                },
                "detection": {
                    "threshold_cm": args.displacement_threshold_cm,
                    "confirmation_samples": tracker.confirmation_samples,
                    "refractory_s": tracker.refractory_s,
                    "lift_exclusion_cm": tracker.lift_exclusion_m * 100.0,
                    "placement_dwell_s": tracker.placement_dwell_s,
                    "recovery_window_s": tracker.recovery_window_s,
                    "recovery_definition": "same displaced object remains in destination box within window",
                },
                "initial_object_poses": initial_object_poses,
                "destination_box_bounds": box_bounds,
                "objects_total": len(initial_object_poses),
                "objects_placed": len(tracker.placed_at),
                "full_task_completion": len(tracker.placed_at) == len(initial_object_poses),
                "time_to_each_successful_placement_s": placement_times,
                "average_successful_placement_time_s": (
                    float(np.mean(placement_times)) if placement_times else None
                ),
                "failed_grasp_attempts": None,
                "failed_grasp_attempts_note": "Not inferred: no reliable contact/grasp-success signal exists in the current evaluator.",
                "significant_displacement_events": displacement_count,
                "replans": replan_count,
                "recoveries": recoveries,
                "recovery_success_rate": (recovered_count / len(recoveries)) if recoveries else None,
                "control_steps": steps,
                "timing": timing_summary(loop_durations_s, chunk_inference_durations_s),
                "chunks_generated": chunk_count,
                "scheduled_chunk_boundaries": len(scheduled_boundaries),
                "chunk_boundary_discontinuity": {
                    "maximum_abs_action_jump": max(boundary_max_abs) if boundary_max_abs else None,
                    "mean_abs_action_jump": float(np.mean(boundary_max_abs)) if boundary_max_abs else None,
                    "large_boundary_threshold": 0.25,
                    "large_boundary_count": sum(value >= 0.25 for value in boundary_max_abs),
                },
                "events_path": str(events_path),
                "text_log_path": str(log_path),
            }
            with summary_path.open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
            append_jsonl(
                events_path,
                {
                    "event": "run_summary",
                    "wall_timestamp": datetime.now().astimezone().isoformat(),
                    "summary_path": str(summary_path),
                    **summary,
                },
            )
            log.info(
                "DIAGNOSTIC_SUMMARY objects_placed=%d/%d full_completion=%s displacements=%d replans=%d recovery_rate=%s mean_hz=%s summary=%s",
                summary["objects_placed"],
                summary["objects_total"],
                summary["full_task_completion"],
                displacement_count,
                replan_count,
                f"{summary['recovery_success_rate']:.3f}" if summary["recovery_success_rate"] is not None else "null",
                f"{summary['timing'].get('mean_hz', 0.0):.3f}",
                summary_path,
            )
        log.info("Evaluation log=%s", log_path)
        if args.keep_open:
            log.info("SIMULATION_HOLDING_OPEN close the Isaac Sim window to exit")
            while robot._kit is not None and robot._kit.is_running():
                if reset_requested.is_set():
                    reset_requested.clear()
                    reset_count += 1
                    log.info("RESET_BEGIN count=%d policy_active=false", reset_count)
                    robot.reset()
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
                    wait_for_live_cameras(robot, log)
                    log.info("RESET_COMPLETE count=%d policy_active=false", reset_count)
                robot.step(render=True)
                time.sleep(1.0 / args.fps)
            log.info("Isaac Sim window closed by user")
        return 0
    finally:
        if reset_window is not None:
            reset_window.visible = False
        if robot_connected:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
