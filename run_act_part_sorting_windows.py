"""Run one visible Walker S2 Part_Sorting episode with a trained ACT policy."""

from __future__ import annotations

import argparse
import hashlib
import logging
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
from src.lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from src.lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from src.lerobot.policies.utils import make_robot_action  # noqa: E402
from src.lerobot.robots.walker_s2_sim import WalkerS2Config, WalkerS2sim  # noqa: E402
from src.lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from src.lerobot.utils.control_utils import predict_action  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "part_sorting_act_50k" / "pretrained_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
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
        help="ACT actions executed before replanning (defaults to the checkpoint value).",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep Isaac Sim open at the final state until its window is closed.",
    )
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


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
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
    log_path = logs_dir / f"part_sorting_act_eval_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    log = logging.getLogger("act_part_sorting_eval")

    log.info("EVAL_START task=Part_Sorting episodes=1 headless=false")
    log.info("Runtime Python=%s NumPy=%s Torch=%s", sys.version.split()[0], np.__version__, torch.__version__)
    if np.__version__.split(".")[:2] != ["1", "26"]:
        raise RuntimeError(f"Expected NumPy 1.26.x, found {np.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the ACT evaluator")
    log.info("CUDA device=%s", torch.cuda.get_device_name(0))
    log.info("Checkpoint=%s", checkpoint)
    log.info("Checkpoint model.safetensors sha256=%s", sha256(checkpoint / "model.safetensors"))

    # The Isaac viewport remains visible. Disable only the optional OpenCV HighGUI
    # mirror, because the repo intentionally depends on opencv-python-headless.
    config = WalkerS2Config(headless=False, head_viz_enabled=False)
    config.load_from_yaml("Part_Sorting")
    duration_s = float(args.duration if args.duration is not None else config.task_cfg["timelimit"])
    robot = WalkerS2sim(config)

    ds_features = {
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
    }
    action_names = list(ds_features[ACTION]["names"])
    state_names = action_names.copy()
    camera_keys = sorted(key for key in ds_features if key.startswith("observation.images."))
    if len(action_names) != 20 or len(camera_keys) != 4:
        raise RuntimeError(
            f"Walker/checkpoint feature mismatch: actions={len(action_names)}, cameras={camera_keys}"
        )

    log.info("Loading ACT checkpoint weights and saved normalization processors")
    policy = ACTPolicy.from_pretrained(checkpoint)
    checkpoint_n_action_steps = int(policy.config.n_action_steps)
    effective_n_action_steps = (
        checkpoint_n_action_steps if args.n_action_steps is None else int(args.n_action_steps)
    )
    if not 1 <= effective_n_action_steps <= int(policy.config.chunk_size):
        raise ValueError(
            "--n-action-steps must be between 1 and chunk_size "
            f"({policy.config.chunk_size}), got {effective_n_action_steps}"
        )
    # Inference-only override. This changes the action queue/replanning cadence
    # without changing the checkpoint or its 50-step prediction chunk.
    policy.config.n_action_steps = effective_n_action_steps
    policy.to(torch.device("cuda"))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
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
    log.info("Features state=20 cameras=%s action=20", camera_keys)

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
                ui.Label("Part_Sorting ACT policy", height=24)
                ui.Button("Reset Episode", height=42, clicked_fn=request_episode_reset)

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        # Isaac camera render products need a few rendered frames after initialize().
        # Do this before the timed episode so ACT never receives placeholder black frames.
        first_observation, image_ranges = wait_for_live_cameras(robot, log)

        initial_state = state_vector(first_observation, state_names)
        log.info("SIM_READY initial_state_range=(%.4f, %.4f) camera_ranges=%s", initial_state.min(), initial_state.max(), image_ranges)
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

            observation_frame = build_dataset_frame(ds_features, observation, prefix=OBS_STR)
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
            if not bool(torch.isfinite(action_tensor).all()):
                raise RuntimeError("ACT produced a non-finite action; no unsafe command was sent")

            action_np = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            current_state = state_vector(observation, state_names)
            max_command_delta = max(max_command_delta, float(np.max(np.abs(action_np - current_state))))
            action_min = min(action_min, float(action_np.min()))
            action_max = max(action_max, float(action_np.max()))
            robot.send_action(make_robot_action(action_tensor, ds_features))
            steps += 1

            observation = robot.get_observation()
            final_state = state_vector(observation, state_names)
            max_state_displacement = max(
                max_state_displacement,
                float(np.max(np.abs(final_state - initial_state))),
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
