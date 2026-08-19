"""Recovery-capable, causally aligned Part Sorting demonstration collector.

This is intentionally separate from :mod:`task_part_sorting`; the original
collector and existing Part_Sorting dataset remain reproducible and unchanged.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.lerobot.datasets.utils import build_dataset_frame
from src.lerobot.utils.constants import ACTION, OBS_STR

from .recovery_v2 import (
    RecoveryEpisodeMetadata,
    MotionBounds,
    RecoveryObjectMetadata,
    RecoveryState,
    ScenarioSpec,
    ScenarioType,
    append_jsonl,
    build_pilot_schedule,
    classify_object_recoverability,
    execute_causal_step,
    load_pilot_config,
    object_relative_recovery_allowed,
    safety_from_config,
    validate_action_vector,
    validate_episode_metadata,
    validate_motion_target_position,
    validate_workspace_position,
)
from .task_part_sorting import TaskPartSorting
from .utils import get_part_sorting_part_type


class RecoveryV2Abort(RuntimeError):
    """A bounded, recoverable collection attempt failed a safety/quality gate."""


class TaskPartSortingRecoveryV2(TaskPartSorting):
    """Targeted Recovery V2 collector with causal observation/action pairing."""

    TEMPORAL_CONVENTION = (
        "capture observation[t], choose/record action[t], apply action[t], world.step"
    )
    # Derived from the unchanged Part_Sorting trajectory and scene:
    # table/object targets x=0.575..0.925, missed-grasp x up to 0.995,
    # box placement x=1.20, y=0.22/0.36, and retreat z up to 1.42.
    # The small margins cover float/settling variation without making the
    # envelope an unrestricted robot workspace.
    ARM_MOTION_BOUNDS = {
        "right": MotionBounds(x=(0.575, 1.225), y=(0.130, 0.430), z=(0.990, 1.430)),
        "left": MotionBounds(x=(0.575, 1.225), y=(0.130, 0.430), z=(0.990, 1.430)),
    }

    def __init__(self) -> None:
        super().__init__()
        self._pilot_config: dict[str, Any] = {}
        self._schedule: list[ScenarioSpec] = []
        self._safety = None
        self._active_spec: ScenarioSpec | None = None
        self._active_metadata: RecoveryEpisodeMetadata | None = None
        self._object_metadata_by_path: dict[str, RecoveryObjectMetadata] = {}
        self._robot = None
        self._dataset_root: Path | None = None
        self._attempts_path: Path | None = None
        self._episodes_path: Path | None = None
        self._attempt_number = 0
        self._episode_frame_count = 0
        self._episode_started_at = 0.0
        self._limits_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._camera_contract_checked = False
        self._scenario_injected = False
        self._stop_on_first_failure = False
        self._stop_after_first_scenario_exhausted = False
        self._attempt_seed = 0
        self._current_part: dict[str, Any] | None = None
        self._current_recovery_state = "not_started"
        self._current_recovery_attempt = 0
        self._active_target_context: dict[str, Any] = {}
        self._latest_camera_frames: dict[str, np.ndarray] = {}
        self._last_validation_failure: dict[str, Any] | None = None

    def run(self, robot, cfg) -> None:
        if not getattr(cfg, "confirm_collection", False):
            raise RuntimeError(
                "Recovery V2 collection is approval-gated. Pass "
                "--auto_collect.confirm_collection=true only after reviewing the dry-run report."
            )
        if not cfg.record_data:
            raise ValueError("Recovery V2 requires record_data=true")
        if cfg.push_to_hub:
            raise ValueError("Recovery V2 pilot must remain local; push_to_hub must be false")
        if cfg.root is None:
            raise ValueError("Recovery V2 requires an explicit new --auto_collect.root")
        root = Path(cfg.root).resolve()
        if root.exists():
            raise FileExistsError(f"Recovery V2 refuses to overwrite an existing path: {root}")
        if root.name.lower() in {"part_sorting", "part-sorting"}:
            raise ValueError("Recovery V2 root must not be the original Part_Sorting dataset")

        self._pilot_config = load_pilot_config(cfg.pilot_config)
        self._schedule = build_pilot_schedule(self._pilot_config)
        self._safety = safety_from_config(self._pilot_config)
        if cfg.num_episodes != len(self._schedule):
            raise ValueError(
                f"CLI num_episodes={cfg.num_episodes} does not match pilot schedule {len(self._schedule)}"
            )
        if cfg.fps != int(self._pilot_config["dataset_contract"]["fps"]):
            raise ValueError("Recovery V2 FPS must match its declared dataset contract")

        self._robot = robot
        self._stop_on_first_failure = bool(getattr(cfg, "stop_on_first_failure", False))
        self._stop_after_first_scenario_exhausted = bool(
            getattr(cfg, "stop_after_first_scenario_exhausted", False)
        )
        self._dataset_root = root
        self._attempts_path = root / "meta" / "recovery_v2_attempts.jsonl"
        self._episodes_path = root / "meta" / "recovery_v2_episodes.jsonl"
        logging.info("Recovery V2 pilot schedule validated: %d episodes", len(self._schedule))
        super().run(robot, cfg)

    # Called by the additive no-op hook in AutoCollectBase immediately before reset.
    def _before_episode_reset(self, episode_index: int, attempt_index: int) -> None:
        self._active_spec = self._schedule[episode_index]
        self._attempt_number = attempt_index
        seed = self._active_spec.seed + (attempt_index - 1) * 1000
        self._attempt_seed = seed
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        logging.info(
            "Recovery V2 episode=%d attempt=%d scenario=%s seed=%d target_slot=%d",
            episode_index,
            attempt_index,
            self._active_spec.scenario.value,
            seed,
            self._active_spec.target_object_slot,
        )
        if attempt_index > 1:
            logging.info(
                "Recovery V2 retry started: episode=%d attempt=%d seed=%d fresh_scene_reset=pending",
                episode_index, attempt_index, seed,
            )

    def _on_episode_start(self, parts: list[dict] | None = None) -> None:
        super()._on_episode_start(parts)
        if self._active_spec is None or self._robot is None or parts is None:
            raise RecoveryV2Abort("Recovery V2 episode started without an active scenario")
        self._episode_frame_count = 0
        self._episode_started_at = time.perf_counter()
        self._camera_contract_checked = False
        self._scenario_injected = False
        self._current_part = None
        self._current_recovery_state = "episode_start"
        self._current_recovery_attempt = 0
        self._active_target_context = {}
        self._latest_camera_frames = {}

        if self._active_spec.scenario is ScenarioType.DIFFICULT_POSITION:
            self._initialize_difficult_position(parts)
            parts[:] = self._robot._scene_builder.get_parts_world_poses()

        ordered = sorted(parts, key=lambda part: self._slot_from_path(part["prim_path"]))
        object_metadata = []
        self._object_metadata_by_path = {}
        for part in ordered:
            position = np.asarray(part["position"], dtype=np.float64)
            orientation = np.asarray(part["orientation"], dtype=np.float64)
            metadata = RecoveryObjectMetadata(
                object_identity=self._part_name(part),
                prim_path=str(part["prim_path"]),
                part_type=get_part_sorting_part_type(part, self._robot._scene_builder),
                asset_identity=self._asset_identity(str(part["prim_path"])),
                initial_pose_xyzw=np.concatenate([position, orientation]).tolist(),
                recovery_type=(
                    self._active_spec.scenario.value
                    if self._slot_from_path(part["prim_path"]) == self._active_spec.target_object_slot
                    else "normal_control"
                ),
            )
            object_metadata.append(metadata)
            self._object_metadata_by_path[metadata.prim_path] = metadata

        self._active_metadata = RecoveryEpisodeMetadata(
            schema_version=2,
            pilot_index=self._active_spec.pilot_index,
            saved_episode_id=None,
            collection_attempt=self._attempt_number,
            random_seed=self._attempt_seed,
            scenario_type=self._active_spec.scenario.value,
            difficulty=("targeted_sparse_workspace" if self._active_spec.scenario is ScenarioType.DIFFICULT_POSITION else "targeted_recovery"),
            target_object_slot=self._active_spec.target_object_slot,
            temporal_convention=self.TEMPORAL_CONVENTION,
            object_poses_policy_input=False,
            objects=object_metadata,
        )
        for item in object_metadata:
            logging.info(
                "Recovery V2 initial object pose: attempt=%d seed=%d object=%s pose=%s",
                self._attempt_number, self._attempt_seed, item.prim_path, item.initial_pose_xyzw,
            )

    def _on_episode_saved(self, dataset, episode_index: int, episode_length: int) -> None:
        if self._active_metadata is None or self._episodes_path is None:
            raise RuntimeError("Recovery V2 saved an episode without metadata")
        self._active_metadata.saved_episode_id = int(episode_index)
        self._active_metadata.number_of_frames = int(episode_length)
        self._active_metadata.final_task_success = True
        self._active_metadata.termination_reason = "success"
        self._active_metadata.attempt_success = True
        payload = self._active_metadata.to_dict()
        validate_episode_metadata(payload)
        append_jsonl(self._episodes_path, payload)

    def _execute_sequence(
        self,
        robot,
        parts: list[dict],
        box_pos: np.ndarray,
        dt: float,
        dataset,
        single_task: str,
        objects_per_episode: int,
    ) -> bool:
        del objects_per_episode
        if self._active_metadata is None or self._active_spec is None:
            raise RecoveryV2Abort("Missing active Recovery V2 metadata")
        success = False
        reason = "unknown_failure"
        abort_to_raise = None
        try:
            ordered_parts = self._sort_parts_by_grasp_order(robot, parts)
            for part in ordered_parts:
                self._run_object_state_machine(robot, part, box_pos, dt, dataset, single_task)
                self._return_right_arm_home(robot, dt, dataset, single_task)
            success = all(item.final_success for item in self._active_metadata.objects)
            reason = "success" if success else "one_or_more_objects_failed"
        except RecoveryV2Abort as exc:
            reason = str(exc)
            if self._active_metadata.failure_diagnostics is None:
                self._capture_failure(robot, exc, traceback.format_exc())
            self._write_failed_attempt_diagnostics()
            logging.error("Recovery V2 bounded abort: %s", exc, exc_info=True)
            if self._stop_on_first_failure and self._active_spec.pilot_index == 0:
                abort_to_raise = exc
        except Exception as exc:
            wrapped = self._capture_failure(robot, exc, traceback.format_exc())
            reason = str(wrapped)
            self._write_failed_attempt_diagnostics()
            logging.error("Recovery V2 unexpected failure converted to bounded abort", exc_info=True)
            if self._stop_on_first_failure and self._active_spec.pilot_index == 0:
                abort_to_raise = wrapped
        finally:
            elapsed = max(0.0, time.perf_counter() - self._episode_started_at)
            self._active_metadata.final_task_success = success
            self._active_metadata.attempt_success = success
            self._active_metadata.number_of_frames = self._episode_frame_count
            self._active_metadata.termination_reason = reason
            self._active_metadata.elapsed_wall_seconds = elapsed
            self._active_metadata.achieved_fps = self._episode_frame_count / elapsed if elapsed else 0.0
            if self._attempts_path is not None:
                append_jsonl(self._attempts_path, self._active_metadata.to_dict())
        if abort_to_raise is not None:
            raise abort_to_raise
        return success

    def _run_object_state_machine(self, robot, part, box_pos, dt, dataset, single_task) -> None:
        self._current_part = part
        prim_path = str(part["prim_path"])
        metadata = self._object_metadata_by_path[prim_path]
        slot = self._slot_from_path(prim_path)
        targeted = slot == self._active_spec.target_object_slot
        max_attempts = self._safety.max_recovery_attempts_per_object
        initial_xy = np.asarray(metadata.initial_pose_xyzw[:2], dtype=np.float64)
        had_successful_grasp_before_drop = False

        for attempt in range(1, max_attempts + 1):
            metadata.recovery_attempts = attempt - 1
            current = self._refresh_part_pose(robot, part)
            current_xyz = np.asarray(current["position"], dtype=np.float64)
            metadata.maximum_displacement_before_grasp_m = max(
                metadata.maximum_displacement_before_grasp_m,
                float(np.linalg.norm(current_xyz[:2] - initial_xy)),
            )
            self._transition(metadata, RecoveryState.ACQUIRE, attempt)
            poses = self.compute_grasp_poses(current)
            approach = copy.deepcopy(poses["right"]["approach"])
            grasp = copy.deepcopy(poses["right"]["grasp"])
            lift = copy.deepcopy(poses["right"]["lift"])

            if (
                targeted
                and self._active_spec.scenario is ScenarioType.MISSED_FIRST_GRASP
                and attempt == 1
            ):
                offset = np.asarray(
                    self._pilot_config["recovery"]["missed_grasp_offset_world_m"],
                    dtype=np.float64,
                )
                missed_xyz = current_xyz + offset
                for axis, bounds in enumerate((self._safety.workspace_x, self._safety.workspace_y)):
                    if not bounds[0] <= missed_xyz[axis] <= bounds[1]:
                        offset[axis] *= -1.0
                grasp["position"] = np.asarray(grasp["position"]) + offset
                lift["position"] = np.asarray(lift["position"]) + offset
                self._scenario_injected = True

            self._transition(metadata, RecoveryState.APPROACH, attempt)
            self._joint_interpolate_to_pose(robot, {"right": approach}, dt, 0.8, dataset, single_task)

            if (
                targeted
                and self._active_spec.scenario is ScenarioType.DISPLACED_DURING_APPROACH
                and attempt == 1
            ):
                planned_xyz = np.asarray(current["position"], dtype=np.float64)
                self._apply_physical_displacement(robot, prim_path, dataset, single_task)
                reacquired = self._refresh_part_pose(robot, current)
                new_xyz = np.asarray(reacquired["position"], dtype=np.float64)
                displacement = float(np.linalg.norm(new_xyz[:2] - planned_xyz[:2]))
                metadata.maximum_displacement_before_grasp_m = max(
                    metadata.maximum_displacement_before_grasp_m, displacement
                )
                if displacement < float(self._pilot_config["recovery"]["displacement_threshold_m"]):
                    raise RecoveryV2Abort(
                        f"displacement scenario moved only {displacement:.3f} m; refusing mislabeled episode"
                    )
                metadata.displacement_occurred = True
                self._scenario_injected = True
                self._transition(metadata, RecoveryState.REOBSERVE, attempt, displacement_m=displacement)
                self._safe_retreat(robot, reacquired, dt, dataset, single_task)
                continue

            self.move_gripper(robot, {"right": -1.0}, dt, 0.2, dataset, single_task)
            self._transition(metadata, RecoveryState.GRASP, attempt)
            self._joint_interpolate_to_pose(robot, {"right": grasp}, dt, 0.8, dataset, single_task)
            self.move_gripper(robot, {"right": 1.0}, dt, 0.5, dataset, single_task)
            self._transition(metadata, RecoveryState.LIFT, attempt)
            self._joint_interpolate_to_pose(robot, {"right": lift}, dt, 0.8, dataset, single_task)

            grasp_succeeded = self.check_grasp_success(robot, current)
            if metadata.first_grasp_succeeded is None:
                metadata.first_grasp_succeeded = bool(grasp_succeeded)
            self._transition(metadata, RecoveryState.VERIFY_GRASP, attempt, success=bool(grasp_succeeded))
            if not grasp_succeeded:
                self.move_gripper(robot, {"right": -1.0}, dt, 0.2, dataset, single_task)
                refreshed = self._refresh_part_pose(robot, current)
                self._abort_if_object_escaped(robot, refreshed, metadata, dt, dataset, single_task)
                self._safe_retreat(robot, refreshed, dt, dataset, single_task)
                continue

            if had_successful_grasp_before_drop:
                metadata.regrasp_occurred = True

            if (
                targeted
                and self._active_spec.scenario is ScenarioType.DROP_AND_REGRASP
                and not self._scenario_injected
            ):
                had_successful_grasp_before_drop = True
                metadata.drop_occurred = True
                self._scenario_injected = True
                self._perform_controlled_drop(robot, current, dt, dataset, single_task)
                refreshed = self._refresh_part_pose(robot, current)
                self._abort_if_object_escaped(robot, refreshed, metadata, dt, dataset, single_task)
                self._safe_retreat(robot, refreshed, dt, dataset, single_task)
                continue

            place_pose = self.get_place_pose(robot, current, box_pos)["right"]
            self._transition(metadata, RecoveryState.TRANSPORT, attempt)
            self._joint_interpolate_to_pose(robot, {"right": place_pose}, dt, 1.1, dataset, single_task)
            if not self.check_grasp_success(robot, current):
                metadata.drop_occurred = True
                refreshed = self._refresh_part_pose(robot, current)
                self._abort_if_object_escaped(robot, refreshed, metadata, dt, dataset, single_task)
                self._safe_retreat(robot, refreshed, dt, dataset, single_task)
                continue

            self._transition(metadata, RecoveryState.PLACE, attempt)
            self.move_gripper(robot, {"right": -1.0}, dt, 0.3, dataset, single_task)
            self._causal_hold(robot, dataset, single_task, int(self._pilot_config["recovery"]["post_release_settle_steps"]))
            metadata.final_success = self._verify_placement(robot, current, box_pos)
            self._transition(metadata, RecoveryState.VERIFY_PLACE, attempt, success=metadata.final_success)
            if not metadata.final_success:
                raise RecoveryV2Abort(f"placement verification failed for {prim_path}")
            self._transition(metadata, RecoveryState.DONE, attempt)
            return

        self._transition(metadata, RecoveryState.ABORT, max_attempts)
        raise RecoveryV2Abort(f"maximum recovery attempts reached for {prim_path}")

    # ------------------------------------------------------------------
    # Causal low-level control: observe -> record selected action -> apply -> step
    # ------------------------------------------------------------------
    def _causal_step(
        self,
        robot,
        dataset,
        single_task: str,
        arm_positions: np.ndarray,
        finger_positions: np.ndarray,
        left_gripping: bool,
        right_gripping: bool,
    ) -> None:
        if self._episode_frame_count >= math.ceil(self._safety.max_episode_seconds * self._pilot_config["dataset_contract"]["fps"]):
            raise RecoveryV2Abort("maximum episode duration reached")
        if time.perf_counter() - self._episode_started_at > self._safety.max_episode_seconds * 3.0:
            raise RecoveryV2Abort("maximum wall-clock safety duration reached")

        lower, upper = self._joint_limits(robot)

        def select_action(_observation):
            selected = self._build_explicit_action_dict(
                arm_positions, finger_positions, left_gripping, right_gripping
            )
            candidate = np.array(list(selected.values()), dtype=np.float64)
            try:
                validate_action_vector(
                    candidate,
                    lower,
                    upper,
                    robot._robot_interface.gripper_open_width,
                    robot._robot_interface.gripper_close_width,
                    self._safety,
                )
            except ValueError as exc:
                self._log_finger_validation_failure(
                    robot, candidate, lower, upper, left_gripping, right_gripping, exc
                )
                raise
            return selected

        def record_pair(observation, selected_action):
            self._validate_observation_contract(robot, dataset, observation)
            for name in robot.CAMERA_NAMES:
                if name in observation:
                    value = observation[name]
                    if hasattr(value, "detach"):
                        value = value.detach().cpu().numpy()
                    self._latest_camera_frames[name] = np.asarray(value).copy()
            if dataset is None:
                return
            observation_frame = build_dataset_frame(dataset.features, observation, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, selected_action, prefix=ACTION)
            dataset.add_frame({**observation_frame, **action_frame, "task": single_task})

        def apply_action(_):
            robot._hold_arm_positions = np.asarray(arm_positions, dtype=np.float32).copy()
            robot._hold_finger_positions = np.asarray(finger_positions, dtype=np.float32).copy()
            robot._left_gripping = bool(left_gripping)
            robot._right_gripping = bool(right_gripping)

        execute_causal_step(
            capture_observation=robot.get_observation,
            select_action=select_action,
            record_pair=record_pair,
            apply_action=apply_action,
            step_world=lambda: robot.step(render=True),
        )
        self._episode_frame_count += 1

    def _log_finger_validation_failure(
        self,
        robot,
        candidate: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        left_gripping: bool,
        right_gripping: bool,
        exc: ValueError,
    ) -> None:
        """Record why validate_action_vector rejected a candidate action.

        Logged immediately (survives even if the run aborts before
        _capture_failure runs) and also stashed on self so _capture_failure
        can fold it into the persisted failure_diagnostics for the attempt.
        """
        try:
            measured = robot._robot_interface.get_joint_states()
            measured_fingers = self._jsonable(measured.get("finger_positions")) if measured else None
        except Exception:
            measured_fingers = None
        payload = {
            "validation_error": str(exc),
            "candidate_action_20d": candidate.tolist(),
            "candidate_finger_targets_14_18": candidate[14:18].tolist(),
            "measured_finger_positions": measured_fingers,
            "tracked_hold_finger_positions": np.asarray(
                robot._hold_finger_positions, dtype=np.float64
            ).tolist(),
            "configured_gripper_open_width": float(robot._robot_interface.gripper_open_width),
            "configured_gripper_close_width": float(robot._robot_interface.gripper_close_width),
            "configured_gripper_limit_tolerance_m": float(self._safety.gripper_limit_tolerance_m),
            "isaac_finger_lower_limits": np.asarray(lower[14:18], dtype=np.float64).tolist(),
            "isaac_finger_upper_limits": np.asarray(upper[14:18], dtype=np.float64).tolist(),
            "gripper_state": {"left_gripping": bool(left_gripping), "right_gripping": bool(right_gripping)},
        }
        logging.error("Recovery V2 validate_action_vector rejected candidate action: %s", payload)
        self._last_validation_failure = payload

    def _joint_interpolate_to_pose(self, robot, target_poses, dt, duration, dataset=None, single_task=None):
        self._active_target_context = {"world_targets": target_poses}
        try:
            right_pose = target_poses.get("right")
            left_pose = target_poses.get("left")
            # TaskPartSorting pose generation is in world coordinates; the Walker
            # S2 IK interface expects robot-base coordinates, as in the legacy collector.
            right_6d = self._validate_and_convert_ik_target(robot, "right", right_pose)
            left_6d = self._validate_and_convert_ik_target(robot, "left", left_pose)
            self._active_target_context = {
                "world_targets": target_poses,
                "robot_base_targets_xyzrpy": {"left": left_6d, "right": right_6d},
            }
            ik_result = robot._robot_interface.control_dual_arm_ik(
                step_size=0.02,
                left_target_xyzrpy=left_6d,
                right_target_xyzrpy=right_6d,
            )
            if ik_result is None:
                raise RuntimeError("control_dual_arm_ik returned None")
        except RecoveryV2Abort:
            raise
        except Exception as exc:
            raise self._capture_failure(robot, exc, traceback.format_exc()) from exc

        target_arm = np.asarray(robot._hold_arm_positions, dtype=np.float32).copy()
        if "right_joint_positions" in ik_result:
            right = np.asarray(ik_result["right_joint_positions"], dtype=np.float32)
            target_arm[:] = right[:14] if len(right) == 14 else target_arm
            if len(right) == 7:
                target_arm[7:14] = right
        if "left_joint_positions" in ik_result:
            left = np.asarray(ik_result["left_joint_positions"], dtype=np.float32)
            if len(left) == 7:
                target_arm[:7] = left

        # Arm DOFs interpolate from live measured state (IK targets are only
        # meaningful relative to where the arm actually is). Finger DOFs must
        # NOT do this: while gripping, the real controller drives the fingers
        # with PD disabled and a constant torque only (see walkers2sim.py
        # _robot_control_callback), so their measured position is contact-
        # dependent and can legitimately sit outside the narrow validated
        # open/close band. Seeding from that measured value fed spurious
        # "commanded" targets into validate_action_vector. Use the Python-
        # tracked hold label for both the finger start and target instead,
        # matching what move_gripper already does correctly.
        states = robot._robot_interface.get_joint_states()
        start_arm = np.asarray(states["all_positions"], dtype=np.float32)[robot._robot_interface.arm_joint_indices]
        start_fingers = np.asarray(robot._hold_finger_positions, dtype=np.float32)
        start = np.concatenate([start_arm, start_fingers])
        target = np.concatenate([target_arm, np.asarray(robot._hold_finger_positions, dtype=np.float32)])
        steps = max(1, int(duration * self._speed_scale / dt))
        robot._robot_interface.joint_interpolator.reset()
        robot._robot_interface.joint_interpolator.set_target(
            start_q=torch.tensor(start), target_q=torch.tensor(target), num_steps=steps
        )
        for _ in range(steps):
            value = robot._robot_interface.joint_interpolator.step()
            value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            self._causal_step(
                robot, dataset, single_task, value[:14], value[14:18],
                robot._left_gripping, robot._right_gripping,
            )

    def move_gripper(self, robot, targets, dt, duration, dataset=None, single_task=None):
        target_fingers = np.asarray(robot._hold_finger_positions, dtype=np.float32).copy()
        left_gripping = robot._left_gripping
        right_gripping = robot._right_gripping
        for side, command in targets.items():
            indices = slice(0, 2) if side == "left" else slice(2, 4)
            gripping = command > 0
            width = (
                robot._robot_interface.gripper_close_width
                if gripping else robot._robot_interface.gripper_open_width
            )
            target_fingers[indices] = width
            if side == "left":
                left_gripping = gripping
            else:
                right_gripping = gripping
        steps = max(1, int(duration * self._speed_scale / dt))
        start = np.asarray(robot._hold_finger_positions, dtype=np.float32)
        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            fingers = start + (target_fingers - start) * alpha
            self._causal_step(
                robot, dataset, single_task, robot._hold_arm_positions, fingers,
                left_gripping, right_gripping,
            )

    def _return_right_arm_home(self, robot, dt, dataset, single_task):
        target_arm = np.asarray(robot._hold_arm_positions, dtype=np.float32).copy()
        target_arm[7:14] = np.asarray(robot._robot_interface.arm_joint_initial_positions)[7:14]
        target_fingers = np.asarray(robot._hold_finger_positions, dtype=np.float32).copy()
        target_fingers[2:4] = robot._robot_interface.gripper_open_width
        steps = self._get_home_return_steps(robot)
        start = np.concatenate([robot._hold_arm_positions, robot._hold_finger_positions])
        target = np.concatenate([target_arm, target_fingers])
        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            value = start + (target - start) * alpha
            self._causal_step(robot, dataset, single_task, value[:14], value[14:18], robot._left_gripping, False)
        self._causal_hold(robot, dataset, single_task, 10)

    # ------------------------------------------------------------------
    # Recovery disturbances, validation, and metadata helpers
    # ------------------------------------------------------------------
    def _causal_hold(self, robot, dataset, single_task, steps: int) -> None:
        for _ in range(max(0, int(steps))):
            self._causal_step(
                robot, dataset, single_task, robot._hold_arm_positions,
                robot._hold_finger_positions, robot._left_gripping, robot._right_gripping,
            )

    def _safe_retreat(self, robot, part, dt, dataset, single_task) -> None:
        self._transition(self._object_metadata_by_path[str(part["prim_path"])], RecoveryState.RETREAT, 0)
        xyz = np.asarray(part["position"], dtype=np.float64).copy()
        xyz[2] += float(self._pilot_config["recovery"]["retreat_height_m"])
        pose = {"position": xyz, "rotation": np.array([-np.pi, 0.0, -1.9])}
        self._joint_interpolate_to_pose(robot, {"right": pose}, dt, 0.8, dataset, single_task)

    def _abort_if_object_escaped(self, robot, part, metadata, dt, dataset, single_task) -> None:
        status, reason = classify_object_recoverability(
            part.get("position"), part.get("orientation"), self._safety
        )
        logging.info(
            "Recovery V2 object classification: status=%s object=%s pose=%s reason=%s",
            status, part.get("prim_path"), self._jsonable(part), reason,
        )
        if object_relative_recovery_allowed(status):
            return
        metadata.object_escaped_workspace = True
        position = np.asarray(part.get("position"), dtype=np.float64).reshape(-1)
        orientation = np.asarray(part.get("orientation"), dtype=np.float64).reshape(-1)
        metadata.escaped_object_pose = (
            np.concatenate([position, orientation]).tolist()
            if len(position) == 3 and len(orientation) == 4
            else self._jsonable(part)
        )
        metadata.escape_reason = reason
        self._transition(
            metadata, RecoveryState.ABORT, self._current_recovery_attempt,
            escape_status=status, escape_reason=reason,
            escaped_object_pose=metadata.escaped_object_pose,
        )
        logging.error("Recovery V2 unrecoverable object escape: %s", reason)
        try:
            self._return_right_arm_home(robot, dt, dataset, single_task)
        except Exception:
            logging.exception("Recovery V2 safe joint-space home return failed after object escape")
        raise RecoveryV2Abort(f"UNRECOVERABLE_ESCAPE: {reason}")

    def _after_episode_attempts_exhausted(self, episode_index: int, attempts: int) -> None:
        if self._stop_after_first_scenario_exhausted and episode_index == 0:
            raise RecoveryV2Abort(
                f"normal_control exhausted {attempts} attempts; stopping remaining smoke scenarios"
            )

    def _object_xyz(self, robot, prim_path: str) -> np.ndarray:
        part = next(
            part for part in robot._scene_builder.get_parts_world_poses()
            if str(part["prim_path"]) == prim_path
        )
        return np.asarray(part["position"], dtype=np.float64)

    def _edge_distance(self, xyz: np.ndarray) -> float:
        return min(
            xyz[0] - self._safety.workspace_x[0], self._safety.workspace_x[1] - xyz[0],
            xyz[1] - self._safety.workspace_y[0], self._safety.workspace_y[1] - xyz[1],
        )

    def _apply_physical_displacement(self, robot, prim_path, dataset, single_task) -> dict[str, Any]:
        """Closed-loop disturbance push.

        A fixed-duration push (velocity * steps) was unreliable: friction
        made most trials fall well short of the 0.03 m minimum, while an
        occasional tip/roll event could send others far past a safe range
        before the single end-of-push check ever looked. Instead, re-apply
        the push velocity every physics step and inspect the object's live
        position after each one, stopping the instant it lands in the
        target 0.03-0.08 m band or comes within a conservative margin of a
        workspace edge, and aborting outright if it overshoots the band
        entirely. The pre-existing 0.03 m minimum check and everything that
        runs after this function returns (retreat, reacquire, re-approach)
        are unchanged.
        """
        rigid = self._rigid_for_path(robot, prim_path)
        velocity = np.asarray(self._pilot_config["recovery"]["displacement_velocity_world_mps"], dtype=np.float64)
        recovery_cfg = self._pilot_config["recovery"]
        lower_bound = float(recovery_cfg["displacement_threshold_m"])
        upper_bound = float(recovery_cfg.get("displacement_upper_bound_m", 0.08))
        edge_margin = float(recovery_cfg.get("displacement_edge_margin_m", 0.02))
        max_steps = int(recovery_cfg["displacement_steps"])

        initial = self._object_xyz(robot, prim_path)
        predicted = initial + velocity * (max_steps / float(self._pilot_config["dataset_contract"]["fps"]))
        for axis, bounds in enumerate((self._safety.workspace_x, self._safety.workspace_y)):
            if not bounds[0] <= predicted[axis] <= bounds[1]:
                velocity[axis] *= -1.0

        logging.info(
            "Recovery V2 displacement start: object=%s initial_xyz=%s velocity=%s "
            "target_band_m=(%.3f, %.3f) edge_margin_m=%.3f max_steps=%d",
            prim_path, initial.tolist(), velocity.tolist(), lower_bound, upper_bound, edge_margin, max_steps,
        )

        displacement = 0.0
        edge_distance = self._edge_distance(initial)
        stop_reason = "step_budget_exhausted"
        steps_taken = 0
        for step_index in range(max_steps):
            rigid.set_linear_velocity(velocity)
            self._causal_hold(robot, dataset, single_task, 1)
            steps_taken = step_index + 1
            current_xyz = self._object_xyz(robot, prim_path)
            displacement = float(np.linalg.norm(current_xyz[:2] - initial[:2]))
            edge_distance = self._edge_distance(current_xyz)
            logging.info(
                "Recovery V2 displacement step %d/%d: displacement=%.4f m edge_distance=%.4f m",
                steps_taken, max_steps, displacement, edge_distance,
            )
            if lower_bound <= displacement <= upper_bound:
                stop_reason = "target_band_reached"
                break
            if edge_distance <= edge_margin:
                stop_reason = "edge_margin_reached"
                break
            if displacement > upper_bound:
                stop_reason = "exceeded_upper_bound"
                break
        # Zero both linear and angular velocity when the push stops (same
        # settle mechanism _initialize_difficult_position already uses):
        # a tip/roll event during the push can leave the object spinning,
        # and clearing linear velocity alone doesn't stop that rotation
        # from continuing to carry it further after we've stopped pushing.
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))

        summary = {
            "prim_path": prim_path,
            "initial_xyz": initial.tolist(),
            "steps_taken": steps_taken,
            "max_steps": max_steps,
            "final_displacement_m": displacement,
            "edge_distance_m": edge_distance,
            "stop_reason": stop_reason,
        }
        logging.info("Recovery V2 displacement finished: %s", summary)

        if displacement > upper_bound:
            raise RecoveryV2Abort(
                f"displacement scenario overshot to {displacement:.3f} m; exceeds {upper_bound:.3f} m cap "
                f"(stop_reason={stop_reason})"
            )

        current_parts = robot._scene_builder.get_parts_world_poses()
        moved = next(part for part in current_parts if str(part["prim_path"]) == prim_path)
        others = np.asarray(
            [part["position"] for part in current_parts if str(part["prim_path"]) != prim_path],
            dtype=np.float64,
        )
        validate_workspace_position(np.asarray(moved["position"]), self._safety, others)
        return summary

    def _perform_controlled_drop(self, robot, part, dt, dataset, single_task) -> None:
        current = self._refresh_part_pose(robot, part)
        initial_z = self._object_metadata_by_path[str(part["prim_path"])].initial_pose_xyzw[2]
        drop_xyz = np.asarray(current["position"], dtype=np.float64).copy()
        drop_xyz[2] = initial_z + float(self._pilot_config["recovery"]["drop_release_height_above_table_m"])
        pose = {"position": drop_xyz, "rotation": np.array([-np.pi, -0.2, -1.9])}
        self._joint_interpolate_to_pose(robot, {"right": pose}, dt, 0.6, dataset, single_task)
        self.move_gripper(robot, {"right": -1.0}, dt, 0.2, dataset, single_task)
        self._causal_hold(robot, dataset, single_task, int(self._pilot_config["recovery"]["drop_settle_steps"]))

    def _initialize_difficult_position(self, parts: list[dict]) -> None:
        spec = self._active_spec
        target = next(part for part in parts if self._slot_from_path(part["prim_path"]) == spec.target_object_slot)
        others = np.asarray([part["position"] for part in parts if part is not target], dtype=np.float64)
        configured = [
            np.asarray(position, dtype=np.float64)
            for position in self._pilot_config["difficult_positions_xyz"]
        ]
        preferred = np.asarray(spec.difficult_position_xyz, dtype=np.float64)
        preferred_index = next(
            (index for index, position in enumerate(configured) if np.allclose(position, preferred)),
            0,
        )
        # A reset may randomly put another object near the preferred point. Try
        # every approved difficult point, rotating the order across bounded
        # episode retries, rather than teleporting into a collision or looping.
        start = (preferred_index + self._attempt_number - 1) % len(configured)
        xyz = None
        rejected: list[str] = []
        for offset in range(len(configured)):
            candidate = configured[(start + offset) % len(configured)]
            try:
                validate_workspace_position(candidate, self._safety, others)
            except ValueError as exc:
                rejected.append(str(exc))
                continue
            xyz = candidate
            break
        if xyz is None:
            raise RecoveryV2Abort(
                "no configured difficult position is collision-free after reset: "
                + "; ".join(rejected)
            )
        rigid = self._rigid_for_path(self._robot, str(target["prim_path"]))
        qx, qy, qz, qw = target["orientation"]
        rigid.set_world_pose(position=xyz, orientation=np.array([qw, qx, qy, qz]))
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))
        for _ in range(15):
            self._robot.step(render=True)
        refreshed = self._refresh_part_pose(self._robot, target)
        refreshed_xyz = np.asarray(refreshed["position"], dtype=np.float64)
        validate_workspace_position(refreshed_xyz, self._safety, others)

    def _verify_placement(self, robot, part, box_pos) -> bool:
        current = self._refresh_part_pose(robot, part)
        xyz = np.asarray(current["position"], dtype=np.float64)
        return bool(np.linalg.norm(xyz[:2] - np.asarray(box_pos)[:2]) <= 0.28 and 0.95 <= xyz[2] <= 1.35)

    def _joint_limits(self, robot) -> tuple[np.ndarray, np.ndarray]:
        if self._limits_cache is not None:
            return self._limits_cache
        articulation = robot._robot_interface._articulation
        raw = getattr(articulation, "dof_limits", None)
        if raw is None and hasattr(articulation, "get_dof_limits"):
            raw = articulation.get_dof_limits()
        if raw is None:
            raise RecoveryV2Abort("Isaac articulation did not expose joint limits")
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 3:
            raw = raw[0]
        indices = robot._robot_interface.arm_joint_indices + robot._robot_interface.finger_joint_indices
        selected = raw[indices]
        if selected.shape != (18, 2) or not np.isfinite(selected).all():
            raise RecoveryV2Abort(f"Unexpected articulation limits shape: {selected.shape}")
        self._limits_cache = selected[:, 0], selected[:, 1]
        return self._limits_cache

    def _validate_observation_contract(self, robot, dataset, observation) -> None:
        if self._camera_contract_checked:
            return
        camera_names = list(robot.CAMERA_NAMES)
        missing = [name for name in camera_names if name not in observation]
        if len(camera_names) != 4 or missing:
            raise RecoveryV2Abort(f"Expected four live cameras; names={camera_names}, missing={missing}")
        if dataset is not None:
            state_shape = tuple(dataset.features["observation.state"]["shape"])
            action_shape = tuple(dataset.features["action"]["shape"])
            camera_keys = [key for key in dataset.features if key.startswith("observation.images.")]
            if state_shape != (48,) or action_shape != (20,) or len(camera_keys) != 4:
                raise RecoveryV2Abort(
                    f"Dataset contract mismatch: state={state_shape}, action={action_shape}, cameras={camera_keys}"
                )
        self._camera_contract_checked = True

    def _build_explicit_action_dict(self, arm, fingers, left_gripping, right_gripping):
        values = np.concatenate([
            np.asarray(arm, dtype=np.float64),
            np.asarray(fingers, dtype=np.float64),
            np.array([-1.0 if left_gripping else 1.0, -1.0 if right_gripping else 1.0]),
        ])
        names = [f"{name}.pos" for name in self._arm_joint_names + self._finger_joint_names]
        names += ["left_gripper", "right_gripper"]
        return dict(zip(names, values.tolist(), strict=True))

    def _transition(self, metadata, state: RecoveryState, attempt: int, **details) -> None:
        self._current_recovery_state = state.value
        self._current_recovery_attempt = int(attempt)
        metadata.state_transitions.append(
            {"state": state.value, "attempt": int(attempt), "frame": int(self._episode_frame_count), **details}
        )

    def _validate_and_convert_ik_target(self, robot, arm: str, pose):
        if pose is None:
            return None
        position = np.asarray(pose.get("position"), dtype=np.float64)
        rotation = np.asarray(pose.get("rotation"), dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (3,):
            raise RecoveryV2Abort(f"{arm} IK target must contain position[3] and XYZ RPY[3]")
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise RecoveryV2Abort(f"{arm} IK target contains NaN/Inf: position={position}, rpy={rotation}")
        if np.any(np.abs(rotation) > 2 * np.pi + 1e-6):
            raise RecoveryV2Abort(f"{arm} IK target has invalid RPY radians: {rotation}")
        try:
            validate_motion_target_position(position, self.ARM_MOTION_BOUNDS[arm], arm)
        except (KeyError, ValueError) as exc:
            raise RecoveryV2Abort(str(exc)) from exc
        robot_6d = self._world_pose_to_robot_6d(robot, pose)
        if robot_6d is None or robot_6d.shape != (6,) or not np.isfinite(robot_6d).all():
            raise RecoveryV2Abort(f"{arm} world-to-robot target conversion is invalid: {robot_6d}")
        max_reach = float(self._pilot_config.get("ik_safety", {}).get("max_robot_base_target_norm_m", 1.45))
        if np.linalg.norm(robot_6d[:3]) > max_reach:
            raise RecoveryV2Abort(
                f"{arm} robot-base target norm {np.linalg.norm(robot_6d[:3]):.3f} exceeds {max_reach:.3f} m"
            )
        ee_poses = robot._robot_interface.get_ee_poses()
        if ee_poses is not None and arm in ee_poses:
            current_ee = np.asarray(ee_poses[arm], dtype=np.float64)
            if current_ee.shape[0] < 3 or not np.isfinite(current_ee[:3]).all():
                raise RecoveryV2Abort(f"{arm} current end-effector pose is invalid: {current_ee}")
            delta = float(np.linalg.norm(robot_6d[:3] - current_ee[:3]))
            max_delta = float(self._pilot_config.get("ik_safety", {}).get("max_target_delta_m", 1.25))
            if delta > max_delta:
                raise RecoveryV2Abort(f"{arm} IK target delta {delta:.3f} exceeds {max_delta:.3f} m")
        if self._current_part is not None:
            quaternion = np.asarray(self._current_part.get("orientation"), dtype=np.float64)
            if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not 0.8 <= np.linalg.norm(quaternion) <= 1.2:
                raise RecoveryV2Abort(f"Current object quaternion is invalid: {quaternion}")
        return robot_6d

    def _capture_failure(self, robot, exc: BaseException, traceback_text: str) -> RecoveryV2Abort:
        part = self._current_part or {}
        prim_path = str(part.get("prim_path", "unknown"))
        try:
            current_part = self._refresh_part_pose(robot, part) if part else None
        except Exception as pose_exc:
            current_part = {"pose_capture_error": f"{type(pose_exc).__name__}: {pose_exc}"}
        try:
            joint_state = robot._robot_interface.get_joint_states()
        except Exception as state_exc:
            joint_state = {"joint_state_capture_error": f"{type(state_exc).__name__}: {state_exc}"}
        diagnostic = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback_text,
            "scenario": self._active_spec.scenario.value if self._active_spec else "unknown",
            "pilot_index": self._active_spec.pilot_index if self._active_spec else None,
            "collection_attempt": self._attempt_number,
            "seed": self._active_spec.seed if self._active_spec else None,
            "object_name": self._part_name(part) if part else "unknown",
            "object_path": prim_path,
            "recovery_state": self._current_recovery_state,
            "recovery_attempt": self._current_recovery_attempt,
            "target": self._active_target_context,
            "current_robot_joint_state": joint_state,
            "current_object_pose": current_part,
            "frame_index": self._episode_frame_count,
            "validation_failure": self._last_validation_failure,
        }
        diagnostic = self._jsonable(diagnostic)
        self._last_validation_failure = None
        if self._active_metadata is not None:
            self._active_metadata.failure_diagnostics = diagnostic
        return RecoveryV2Abort(f"{type(exc).__name__} during {self._current_recovery_state}: {exc}")

    def _write_failed_attempt_diagnostics(self) -> None:
        if self._dataset_root is None or self._active_metadata is None:
            return
        directory = self._dataset_root / "diagnostics" / "failed_attempts" / (
            f"episode_{self._active_metadata.pilot_index:03d}_attempt_{self._attempt_number:02d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        diagnostic = self._active_metadata.failure_diagnostics or {}
        (directory / "failure.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        (directory / "traceback.txt").write_text(str(diagnostic.get("traceback", "")), encoding="utf-8")
        try:
            from PIL import Image
            for name, frame in self._latest_camera_frames.items():
                image = np.asarray(frame)
                if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                    image = np.moveaxis(image, 0, -1)
                if image.dtype != np.uint8:
                    image = np.clip(image * 255.0 if image.max(initial=0) <= 1.0 else image, 0, 255).astype(np.uint8)
                Image.fromarray(image[..., :3]).save(directory / f"latest_{name.replace('.', '_')}.png")
        except Exception:
            logging.exception("Could not preserve failed-attempt camera images")

    @staticmethod
    def _jsonable(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): TaskPartSortingRecoveryV2._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [TaskPartSortingRecoveryV2._jsonable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _rigid_for_path(self, robot, prim_path):
        scene = robot._scene_builder
        scene._ensure_rigid_prims()
        try:
            index = scene.parts_prim_paths.index(prim_path)
            rigid = scene._parts_rigid_prims[index]
        except (ValueError, IndexError):
            rigid = None
        if rigid is None:
            raise RecoveryV2Abort(f"No rigid body available for {prim_path}")
        return rigid

    def _asset_identity(self, prim_path: str) -> str | None:
        try:
            import omni.usd

            prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
            references = prim.GetMetadata("references")
            if references is None:
                return None
            items = references.GetAddedOrExplicitItems()
            if not items:
                return None
            return str(items[0].assetPath)
        except Exception:
            logging.warning("Could not resolve asset identity for %s", prim_path)
            return None

    @staticmethod
    def _pose_to_6d(pose):
        if pose is None:
            return None
        return np.asarray([*pose["position"], *pose["rotation"]], dtype=np.float32)

    def _world_pose_to_robot_6d(self, robot, pose):
        if pose is None:
            return None
        robot_pose = {
            "position": robot._scene_builder.world_to_robot_coords(pose["position"]),
            "rotation": pose["rotation"],
        }
        return self._pose_to_6d(robot_pose)

    @staticmethod
    def _slot_from_path(prim_path: str) -> int:
        match = re.search(r"_(\d+)$", str(prim_path))
        if match is None or int(match.group(1)) not in range(4):
            raise RecoveryV2Abort(f"Cannot map Part Sorting slot from {prim_path}")
        return int(match.group(1))

    @staticmethod
    def _part_name(part: dict) -> str:
        return str(part.get("name") or str(part["prim_path"]).rstrip("/").split("/")[-1])
