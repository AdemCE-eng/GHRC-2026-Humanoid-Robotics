"""Policy-agnostic disturbance injection for ACT evaluation of Recovery V2 scenarios.

These functions are ports of the validated Recovery V2 collector's own disturbance
primitives (src/lerobot/auto_collect/task_part_sorting_recovery_v2.py), decoupled
from that scripted collector's state machine and dataset-recording pipeline so they
can be layered on top of an ACT-driven rollout instead.

CRITICAL INVARIANT (enforced by construction, see test_recovery_v2_eval_injection.py):
every function in this module touches ONLY the target object's rigid body (its
world pose / linear / angular velocity) or reads robot state read-only. Nothing here
ever calls robot.send_action(), sets robot._hold_arm_positions, or otherwise commands
the arm/gripper. Recovery from a disturbance is entirely the evaluated policy's
responsibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("recovery_v2_eval_disturbance")


@dataclass(frozen=True)
class DisturbanceConfig:
    """Parameters sourced from the same validated Recovery V2 pilot config used
    for Smoke5/Pilot50 collection, so eval disturbances match collection-time
    tuning exactly rather than being re-guessed."""

    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]

    displacement_velocity_world_mps: tuple[float, float, float]
    displacement_steps: int
    displacement_threshold_m: float
    displacement_upper_bound_m: float
    displacement_edge_margin_m: float

    # Missed-first-grasp nudge: same closed-loop mechanism, smaller target band,
    # since the goal is "make the object slip from between the fingers", not a
    # full recovery-scale relocation.
    missed_grasp_lower_bound_m: float = 0.015
    missed_grasp_upper_bound_m: float = 0.05
    missed_grasp_edge_margin_m: float = 0.02
    missed_grasp_steps: int = 30
    missed_grasp_velocity_world_mps: tuple[float, float, float] = (0.06, 0.0, 0.0)

    grasp_detect_distance_m: float = 0.08
    drop_confirm_steps: int = 15
    drop_release_steps: int = 3

    difficult_positions_xyz: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)


def load_disturbance_config(pilot_config_path: str | Path) -> DisturbanceConfig:
    with Path(pilot_config_path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    safety = raw["safety"]
    recovery = raw["recovery"]
    return DisturbanceConfig(
        workspace_x=tuple(float(v) for v in safety["workspace_x"]),
        workspace_y=tuple(float(v) for v in safety["workspace_y"]),
        displacement_velocity_world_mps=tuple(float(v) for v in recovery["displacement_velocity_world_mps"]),
        displacement_steps=int(recovery["displacement_steps"]),
        displacement_threshold_m=float(recovery["displacement_threshold_m"]),
        displacement_upper_bound_m=float(recovery.get("displacement_upper_bound_m", 0.08)),
        displacement_edge_margin_m=float(recovery.get("displacement_edge_margin_m", 0.02)),
        difficult_positions_xyz=tuple(
            tuple(float(v) for v in position) for position in raw.get("difficult_positions_xyz", [])
        ),
    )


def object_xyz(robot, prim_path: str) -> np.ndarray:
    part = next(
        part for part in robot._scene_builder.get_parts_world_poses() if str(part["prim_path"]) == prim_path
    )
    return np.asarray(part["position"], dtype=np.float64)


def object_pose(robot, prim_path: str) -> dict[str, Any]:
    return next(
        part for part in robot._scene_builder.get_parts_world_poses() if str(part["prim_path"]) == prim_path
    )


def edge_distance(xyz: np.ndarray, workspace_x: tuple[float, float], workspace_y: tuple[float, float]) -> float:
    return min(
        xyz[0] - workspace_x[0], workspace_x[1] - xyz[0],
        xyz[1] - workspace_y[0], workspace_y[1] - xyz[1],
    )


def rigid_for_path(robot, prim_path: str):
    """Verbatim port of TaskPartSortingRecoveryV2._rigid_for_path."""
    scene = robot._scene_builder
    scene._ensure_rigid_prims()
    try:
        index = scene.parts_prim_paths.index(prim_path)
        rigid = scene._parts_rigid_prims[index]
    except (ValueError, IndexError):
        rigid = None
    if rigid is None:
        raise RuntimeError(f"No rigid body available for {prim_path}")
    return rigid


def end_effector_object_distance(robot, prim_path: str, arm_side: str = "right") -> float | None:
    """Read-only port of AutoCollectBase._check_grasp_success_for_arm's distance
    metric. Never touches control; used only to detect what the policy has
    already achieved on its own."""
    ee_poses = robot._robot_interface.get_ee_poses()
    if ee_poses is None:
        return None
    ee = ee_poses.get(arm_side)
    if ee is None:
        return None
    gripper_pos = np.asarray(ee[:3], dtype=np.float64)
    part_pos = object_xyz(robot, prim_path)
    part_pos_robot = robot._scene_builder.world_to_robot_coords(part_pos)
    return float(np.linalg.norm(gripper_pos - np.asarray(part_pos_robot, dtype=np.float64)))


# ---------------------------------------------------------------------------
# difficult_position: pure initial-condition placement, before ACT ever runs.
# ---------------------------------------------------------------------------


def teleport_to_difficult_position(
    robot,
    prim_path: str,
    xyz: tuple[float, float, float],
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Place the target object at a fixed difficult-position spawn point.

    Must be called BEFORE the ACT control loop's first inference step (i.e.
    before the policy has ever observed the scene), matching Recovery V2's own
    _initialize_difficult_position: this is scene setup, not mid-episode
    intervention, so it does not touch the "never perform recovery for the
    policy" rule at all -- ACT never sees any other position.
    """
    log = log or logger
    current = object_pose(robot, prim_path)
    rigid = rigid_for_path(robot, prim_path)
    orientation = np.asarray(current["orientation"], dtype=np.float64)  # xyzw
    qx, qy, qz, qw = orientation
    rigid.set_world_pose(position=np.asarray(xyz, dtype=np.float64), orientation=np.array([qw, qx, qy, qz]))
    rigid.set_linear_velocity(np.zeros(3))
    rigid.set_angular_velocity(np.zeros(3))
    for _ in range(15):
        robot.step(render=True)
    settled = object_xyz(robot, prim_path)
    event = {
        "event": "difficult_position_placed",
        "prim_path": prim_path,
        "requested_xyz": list(xyz),
        "settled_xyz": settled.tolist(),
    }
    log.info("DISTURBANCE difficult_position object=%s requested=%s settled=%s", prim_path, xyz, settled.tolist())
    return event


# ---------------------------------------------------------------------------
# displaced_during_approach / missed_first_grasp: closed-loop object-only push.
# Identical mechanism (validated in Recovery V2 Pilot50), parameterized
# differently for the two scenarios' intended magnitudes.
# ---------------------------------------------------------------------------


def apply_closed_loop_push(
    robot,
    prim_path: str,
    velocity_world_mps: tuple[float, float, float],
    max_steps: int,
    lower_bound_m: float,
    upper_bound_m: float,
    edge_margin_m: float,
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Closed-loop rigid-body push. Verbatim port of the validated
    TaskPartSortingRecoveryV2._apply_physical_displacement logic, decoupled from
    the scripted collector's dataset-recording pipeline. Only ever touches the
    target object's rigid body; calls robot.step() directly instead of the
    scripted collector's _causal_hold/_causal_step recording wrapper.
    """
    log = log or logger
    rigid = rigid_for_path(robot, prim_path)
    velocity = np.asarray(velocity_world_mps, dtype=np.float64).copy()
    initial = object_xyz(robot, prim_path)

    displacement = 0.0
    edge_dist = edge_distance(initial, workspace_x, workspace_y)
    stop_reason = "step_budget_exhausted"
    steps_taken = 0
    for step_index in range(max_steps):
        rigid.set_linear_velocity(velocity)
        robot.step(render=True)
        steps_taken = step_index + 1
        current = object_xyz(robot, prim_path)
        displacement = float(np.linalg.norm(current[:2] - initial[:2]))
        edge_dist = edge_distance(current, workspace_x, workspace_y)
        if lower_bound_m <= displacement <= upper_bound_m:
            stop_reason = "target_band_reached"
            break
        if edge_dist <= edge_margin_m:
            stop_reason = "edge_margin_reached"
            break
        if displacement > upper_bound_m:
            stop_reason = "exceeded_upper_bound"
            break
    rigid.set_linear_velocity(np.zeros(3))
    rigid.set_angular_velocity(np.zeros(3))

    event = {
        "prim_path": prim_path,
        "initial_xyz": initial.tolist(),
        "steps_taken": steps_taken,
        "max_steps": max_steps,
        "final_displacement_m": displacement,
        "edge_distance_m": edge_dist,
        "stop_reason": stop_reason,
    }
    log.info("DISTURBANCE closed_loop_push %s", event)
    return event


# ---------------------------------------------------------------------------
# drop_and_regrasp: force-release trigger detection + brief gripper override.
# ---------------------------------------------------------------------------


class GraspHoldDetector:
    """Tracks whether the policy's OWN gripper-closed state, combined with EE
    proximity to the target object, indicates a stable grasp -- purely by
    observing robot._right_gripping (set from the policy's own action, see
    walkers2sim.py._robot_control_callback) and live poses. Read-only."""

    def __init__(self, target_prim_path: str, grasp_detect_distance_m: float, confirm_steps: int):
        self.target_prim_path = target_prim_path
        self.grasp_detect_distance_m = grasp_detect_distance_m
        self.confirm_steps = confirm_steps
        self._consecutive_holding_steps = 0
        self.first_grasp_attempt_seen = False
        self.first_grasp_close_step: int | None = None

    def update(self, robot, control_step: int) -> tuple[bool, bool]:
        """Returns (first_grasp_close_transition, stable_hold_confirmed)."""
        distance = end_effector_object_distance(robot, self.target_prim_path, "right")
        gripping = bool(getattr(robot, "_right_gripping", False))
        near_target = distance is not None and distance <= self.grasp_detect_distance_m

        first_close_transition = False
        if gripping and near_target and not self.first_grasp_attempt_seen:
            self.first_grasp_attempt_seen = True
            self.first_grasp_close_step = control_step
            first_close_transition = True

        if gripping and near_target:
            self._consecutive_holding_steps += 1
        else:
            self._consecutive_holding_steps = 0
        stable_hold_confirmed = self._consecutive_holding_steps >= self.confirm_steps
        return first_close_transition, stable_hold_confirmed


def force_gripper_release(action_dict: dict[str, float], steps_remaining: int) -> dict[str, float]:
    """The ONLY action-dict-level intervention this harness performs: override
    the gripper scalar to 'open' for a few frames so the object is dropped from
    wherever ACT currently holds it. Every other key (arm/finger positions) is
    left exactly as ACT predicted -- the arm trajectory is never touched."""
    overridden = dict(action_dict)
    overridden["right_gripper"] = 1.0  # 1.0 = open, matches the existing action contract
    return overridden
