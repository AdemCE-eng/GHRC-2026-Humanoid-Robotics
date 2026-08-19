"""Pure helpers for the Part Sorting Recovery Dataset V2 collector.

This module deliberately has no Isaac Sim imports so schedule, metadata, safety,
and temporal ordering can be tested in WSL before launching the simulator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


class ScenarioType(StrEnum):
    NORMAL = "normal_control"
    DISPLACED_DURING_APPROACH = "object_displaced_during_approach"
    MISSED_FIRST_GRASP = "missed_first_grasp"
    DROP_AND_REGRASP = "drop_and_regrasp"
    DIFFICULT_POSITION = "difficult_position"


class RecoveryState(StrEnum):
    ACQUIRE = "acquire"
    APPROACH = "approach"
    REOBSERVE = "reobserve"
    RETREAT = "retreat"
    GRASP = "grasp"
    VERIFY_GRASP = "verify_grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    VERIFY_PLACE = "verify_place"
    DONE = "done"
    ABORT = "abort"


@dataclass(frozen=True)
class ScenarioSpec:
    pilot_index: int
    seed: int
    scenario: ScenarioType
    target_object_slot: int
    difficult_position_xyz: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SafetyConfig:
    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]
    workspace_z: tuple[float, float]
    minimum_object_separation_m: float
    max_recovery_attempts_per_object: int
    max_episode_seconds: float
    joint_limit_margin_rad: float
    gripper_limit_tolerance_m: float


@dataclass(frozen=True)
class MotionBounds:
    """World-coordinate IK envelope, separate from object spawn bounds."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


@dataclass
class RecoveryObjectMetadata:
    object_identity: str
    prim_path: str
    part_type: str
    asset_identity: str | None
    initial_pose_xyzw: list[float]
    recovery_type: str
    recovery_attempts: int = 0
    first_grasp_succeeded: bool | None = None
    displacement_occurred: bool = False
    maximum_displacement_before_grasp_m: float = 0.0
    drop_occurred: bool = False
    regrasp_occurred: bool = False
    object_escaped_workspace: bool = False
    escaped_object_pose: Any = None
    escape_reason: str | None = None
    final_success: bool = False
    state_transitions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RecoveryEpisodeMetadata:
    schema_version: int
    pilot_index: int
    saved_episode_id: int | None
    collection_attempt: int
    random_seed: int
    scenario_type: str
    difficulty: str
    target_object_slot: int
    temporal_convention: str
    object_poses_policy_input: bool
    objects: list[RecoveryObjectMetadata]
    final_task_success: bool = False
    number_of_frames: int = 0
    termination_reason: str = "not_finished"
    elapsed_wall_seconds: float = 0.0
    achieved_fps: float = 0.0
    failure_diagnostics: dict[str, Any] | None = None
    attempt_success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pilot_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 2:
        raise ValueError("Recovery V2 pilot config must be a mapping with schema_version: 2")
    return config


def build_pilot_schedule(config: dict[str, Any]) -> list[ScenarioSpec]:
    mix = config.get("scenario_mix")
    if not isinstance(mix, dict):
        raise ValueError("scenario_mix must be a mapping")
    expected = {scenario.value for scenario in ScenarioType}
    if set(mix) != expected:
        raise ValueError(f"scenario_mix keys must be exactly {sorted(expected)}")
    target_slots = [int(value) for value in config.get("target_object_slot_cycle", [0, 1, 2, 3])]
    if not target_slots or any(slot not in range(4) for slot in target_slots):
        raise ValueError("target_object_slot_cycle must contain only slots 0..3")
    difficult_positions = config.get("difficult_positions_xyz", [])
    seed_base = int(config["seed_base"])
    schedule: list[ScenarioSpec] = []
    for scenario in ScenarioType:
        count = int(mix[scenario.value])
        if count < 0:
            raise ValueError(f"Negative episode count for {scenario.value}")
        for _ in range(count):
            index = len(schedule)
            difficult = None
            if scenario is ScenarioType.DIFFICULT_POSITION:
                if not difficult_positions:
                    raise ValueError("difficult_position episodes require difficult_positions_xyz")
                difficult = tuple(float(v) for v in difficult_positions[index % len(difficult_positions)])
                if len(difficult) != 3:
                    raise ValueError("Each difficult position must be XYZ")
            schedule.append(
                ScenarioSpec(
                    pilot_index=index,
                    seed=seed_base + index,
                    scenario=scenario,
                    target_object_slot=target_slots[index % len(target_slots)],
                    difficult_position_xyz=difficult,
                )
            )
    declared_total = int(config.get("total_episodes", len(schedule)))
    if len(schedule) != declared_total:
        raise ValueError(f"Scenario mix totals {len(schedule)}, expected {declared_total}")
    return schedule


def safety_from_config(config: dict[str, Any]) -> SafetyConfig:
    raw = config["safety"]
    return SafetyConfig(
        workspace_x=tuple(float(v) for v in raw["workspace_x"]),
        workspace_y=tuple(float(v) for v in raw["workspace_y"]),
        workspace_z=tuple(float(v) for v in raw["workspace_z"]),
        minimum_object_separation_m=float(raw["minimum_object_separation_m"]),
        max_recovery_attempts_per_object=int(raw["max_recovery_attempts_per_object"]),
        max_episode_seconds=float(raw["max_episode_seconds"]),
        joint_limit_margin_rad=float(raw["joint_limit_margin_rad"]),
        gripper_limit_tolerance_m=float(raw["gripper_limit_tolerance_m"]),
    )


def validate_workspace_position(
    xyz: np.ndarray,
    safety: SafetyConfig,
    other_xyz: np.ndarray | None = None,
) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid object XYZ: {xyz}")
    for value, bounds, axis in zip(xyz, (safety.workspace_x, safety.workspace_y, safety.workspace_z), "xyz"):
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"Object {axis}={value:.4f} outside safe bounds {bounds}")
    if other_xyz is not None and len(other_xyz):
        distances = np.linalg.norm(np.asarray(other_xyz, dtype=np.float64)[:, :2] - xyz[:2], axis=1)
        if np.min(distances) < safety.minimum_object_separation_m:
            raise ValueError(
                f"Object position violates {safety.minimum_object_separation_m:.3f} m separation; "
                f"nearest={np.min(distances):.3f} m"
            )


def validate_motion_target_position(xyz: np.ndarray, bounds: MotionBounds, arm: str) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"{arm} IK target position must be finite XYZ, got {xyz}")
    for value, limits, axis in zip(xyz, (bounds.x, bounds.y, bounds.z), "xyz"):
        if not limits[0] <= value <= limits[1]:
            raise ValueError(f"{arm} IK target {axis}={value:.4f} outside motion bounds {limits}")


def classify_object_recoverability(
    position: np.ndarray,
    orientation_xyzw: np.ndarray,
    safety: SafetyConfig,
    *,
    robot_xy: tuple[float, float] = (0.7, -0.2),
    minimum_recovery_z: float = 0.95,
    maximum_recovery_z: float = 1.25,
    maximum_robot_xy_distance: float = 0.80,
) -> tuple[str, str]:
    """Classify an object independently of the arm-motion IK envelope."""
    xyz = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(orientation_xyzw, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return "UNRECOVERABLE_ESCAPE", f"invalid object XYZ: {xyz}"
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        return "UNRECOVERABLE_ESCAPE", f"invalid object quaternion: {quaternion}"
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not 0.8 <= quaternion_norm <= 1.2:
        return "UNRECOVERABLE_ESCAPE", f"invalid object quaternion norm: {quaternion_norm:.4f}"
    if not safety.workspace_x[0] <= xyz[0] <= safety.workspace_x[1]:
        return "UNRECOVERABLE_ESCAPE", f"object x={xyz[0]:.4f} outside recovery bounds {safety.workspace_x}"
    if not safety.workspace_y[0] <= xyz[1] <= safety.workspace_y[1]:
        return "UNRECOVERABLE_ESCAPE", f"object y={xyz[1]:.4f} outside recovery bounds {safety.workspace_y}"
    if not minimum_recovery_z <= xyz[2] <= maximum_recovery_z:
        return "UNRECOVERABLE_ESCAPE", (
            f"object z={xyz[2]:.4f} outside recovery height "
            f"({minimum_recovery_z}, {maximum_recovery_z})"
        )
    robot_distance = float(np.linalg.norm(xyz[:2] - np.asarray(robot_xy, dtype=np.float64)))
    if robot_distance > maximum_robot_xy_distance:
        return "UNRECOVERABLE_ESCAPE", (
            f"object robot XY distance {robot_distance:.4f} exceeds {maximum_robot_xy_distance:.4f} m"
        )
    return "RECOVERABLE", "object remains inside conservative recovery workspace"


def object_relative_recovery_allowed(status: str) -> bool:
    """Only recoverable objects may produce object-relative retreat/reacquire IK."""
    return status == "RECOVERABLE"


def validate_action_vector(
    action: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    gripper_open_width: float,
    gripper_close_width: float,
    safety: SafetyConfig,
) -> None:
    action = np.asarray(action, dtype=np.float64)
    lower_limits = np.asarray(lower_limits, dtype=np.float64)
    upper_limits = np.asarray(upper_limits, dtype=np.float64)
    if action.shape != (20,) or not np.isfinite(action).all():
        raise ValueError(f"Recovery V2 action must be finite 20D, got {action.shape}")
    if lower_limits.shape != (18,) or upper_limits.shape != (18,):
        raise ValueError("Expected 18 arm/finger joint limits")
    # The radian margin is meaningful for the 14 revolute arm joints only.
    # Finger joints are linear widths and are checked against both their Isaac
    # articulation limits and the configured open/close widths below.
    margin = safety.joint_limit_margin_rad
    if np.any(action[:14] < lower_limits[:14] + margin) or np.any(
        action[:14] > upper_limits[:14] - margin
    ):
        bad = np.flatnonzero(
            (action[:14] < lower_limits[:14] + margin)
            | (action[:14] > upper_limits[:14] - margin)
        ).tolist()
        raise ValueError(f"Action exceeds safe joint limits at indices {bad}")
    if np.any(action[14:18] < lower_limits[14:18]) or np.any(
        action[14:18] > upper_limits[14:18]
    ):
        raise ValueError("Finger target exceeds Isaac articulation limits")
    grip_low = min(gripper_open_width, gripper_close_width) - safety.gripper_limit_tolerance_m
    grip_high = max(gripper_open_width, gripper_close_width) + safety.gripper_limit_tolerance_m
    if np.any(action[14:18] < grip_low) or np.any(action[14:18] > grip_high):
        raise ValueError("Finger target exceeds configured gripper limits")
    if not all(np.isclose(value, -1.0) or np.isclose(value, 1.0) for value in action[18:20]):
        raise ValueError("Gripper commands must be -1.0 (closed) or 1.0 (open)")


def execute_causal_step(
    *,
    capture_observation: Callable[[], Any],
    select_action: Callable[[Any], Any],
    record_pair: Callable[[Any, Any], None],
    apply_action: Callable[[Any], None],
    step_world: Callable[[], None],
) -> Any:
    """Execute V2 order: observe, select, record pair, apply, then step."""
    observation = capture_observation()
    action = select_action(observation)
    record_pair(observation, action)
    apply_action(action)
    step_world()
    return observation


REQUIRED_EPISODE_METADATA_KEYS = {
    "schema_version", "pilot_index", "saved_episode_id", "collection_attempt", "random_seed",
    "scenario_type", "difficulty", "target_object_slot", "temporal_convention",
    "object_poses_policy_input", "objects", "final_task_success", "number_of_frames",
    "termination_reason",
}


def validate_episode_metadata(metadata: dict[str, Any]) -> None:
    missing = REQUIRED_EPISODE_METADATA_KEYS - set(metadata)
    if missing:
        raise ValueError(f"Recovery metadata missing keys: {sorted(missing)}")
    if metadata["object_poses_policy_input"] is not False:
        raise ValueError("Recovery metadata must state that object poses are not policy inputs")
    if len(metadata["objects"]) != 4:
        raise ValueError("Recovery V2 Part Sorting episodes must describe all four objects")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    validate_episode_metadata(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
