from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from src.lerobot.auto_collect.recovery_v2 import (
    MotionBounds,
    RecoveryEpisodeMetadata,
    RecoveryObjectMetadata,
    SafetyConfig,
    ScenarioType,
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
from src.lerobot.auto_collect.task_part_sorting_recovery_v2 import RecoveryV2Abort, TaskPartSortingRecoveryV2
from src.lerobot.robots.walker_s2_sim.isaac_sim_robot_interface import JointInterpolator


class _FakeRigid:
    def __init__(self):
        self.velocity: np.ndarray | None = None
        self.angular_velocity: np.ndarray | None = None

    def set_linear_velocity(self, velocity) -> None:
        self.velocity = np.asarray(velocity, dtype=np.float64)

    def set_angular_velocity(self, velocity) -> None:
        self.angular_velocity = np.asarray(velocity, dtype=np.float64)


class _ScriptedSceneBuilder:
    """Replays a pre-scripted xy trajectory for one object, one entry per physics step."""

    def __init__(self, prim_path: str, trajectory_xy, z: float = 1.02, others=()):
        self.prim_path = prim_path
        self.trajectory = [np.array([xy[0], xy[1], z], dtype=np.float64) for xy in trajectory_xy]
        self.index = 0
        self.others = others
        self.parts_prim_paths = [prim_path]
        self._parts_rigid_prims = [_FakeRigid()]

    def _ensure_rigid_prims(self) -> None:
        pass

    def get_parts_world_poses(self):
        position = self.trajectory[min(self.index, len(self.trajectory) - 1)]
        parts = [{"prim_path": self.prim_path, "position": position.copy(), "orientation": np.array([0.0, 0.0, 0.0, 1.0])}]
        for i, other_xyz in enumerate(self.others):
            parts.append({
                "prim_path": f"/Root/other_{i}",
                "position": np.array(other_xyz, dtype=np.float64),
                "orientation": np.array([0.0, 0.0, 0.0, 1.0]),
            })
        return parts


class _ScriptedRobot:
    def __init__(self, scene_builder: _ScriptedSceneBuilder):
        self._scene_builder = scene_builder
        self.step_calls = 0
        # _causal_hold builds its _causal_step call args from these before
        # invoking the (stubbed) _causal_step, so they must exist even
        # though the stub itself ignores them.
        self._hold_arm_positions = np.zeros(14, dtype=np.float32)
        self._hold_finger_positions = np.zeros(4, dtype=np.float32)
        self._left_gripping = False
        self._right_gripping = False

    def step(self, render: bool = True) -> None:
        self.step_calls += 1
        if self._scene_builder.index < len(self._scene_builder.trajectory) - 1:
            self._scene_builder.index += 1


def _make_displacement_task(max_steps: int) -> TaskPartSortingRecoveryV2:
    task = TaskPartSortingRecoveryV2()
    task._pilot_config = {
        "recovery": {
            "displacement_threshold_m": 0.03,
            "displacement_upper_bound_m": 0.08,
            "displacement_edge_margin_m": 0.02,
            "displacement_velocity_world_mps": [0.10, 0.0, 0.0],
            "displacement_steps": max_steps,
        },
        "dataset_contract": {"fps": 30},
    }
    task._safety = SafetyConfig(
        workspace_x=(0.575, 0.925), workspace_y=(0.130, 0.430), workspace_z=(1.00, 1.12),
        minimum_object_separation_m=0.10, max_recovery_attempts_per_object=3,
        max_episode_seconds=180.0, joint_limit_margin_rad=0.01, gripper_limit_tolerance_m=0.001,
    )

    def stub_causal_step(robot, _dataset, _single_task, _arm, _fingers, _left, _right):
        robot.step(render=True)

    task._causal_step = stub_causal_step
    return task


class RecoveryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pilot_config(Path("configs/recovery_v2_part_sorting_pilot.yaml"))
        cls.safety = safety_from_config(cls.config)

    def test_pilot_schedule_has_exact_targeted_mix(self):
        schedule = build_pilot_schedule(self.config)
        self.assertEqual(len(schedule), 50)
        self.assertEqual(Counter(item.scenario for item in schedule), Counter({scenario: 10 for scenario in ScenarioType}))
        self.assertEqual(len({item.seed for item in schedule}), 50)
        self.assertTrue(all(item.target_object_slot in range(4) for item in schedule))

    def test_causal_step_order_is_observe_record_apply_step(self):
        events = []
        execute_causal_step(
            capture_observation=lambda: events.append("observe") or {"state": 1},
            select_action=lambda observation: events.append(("select", observation)) or {"action": 2},
            record_pair=lambda observation, action: events.append(("record", observation, action)),
            apply_action=lambda action: events.append(("apply", action)),
            step_world=lambda: events.append("step"),
        )
        self.assertEqual(events[0], "observe")
        self.assertEqual(events[1][0], "select")
        self.assertEqual(events[2][0], "record")
        self.assertEqual(events[3][0], "apply")
        self.assertEqual(events[4], "step")

    def test_workspace_and_separation_checks(self):
        validate_workspace_position(np.array([0.75, 0.135, 1.04]), self.safety, np.array([[0.60, 0.30, 1.04]]))
        with self.assertRaises(ValueError):
            validate_workspace_position(np.array([0.75, 0.10, 1.04]), self.safety)
        with self.assertRaises(ValueError):
            validate_workspace_position(np.array([0.75, 0.20, 1.04]), self.safety, np.array([[0.76, 0.20, 1.04]]))

    def test_motion_bounds_are_separate_from_object_spawn_bounds(self):
        bounds = MotionBounds(x=(0.575, 1.225), y=(0.130, 0.430), z=(0.990, 1.430))
        validate_motion_target_position(np.array([0.75, 0.28, 1.29]), bounds, "right")
        validate_motion_target_position(np.array([1.20, 0.36, 1.23]), bounds, "right")
        with self.assertRaises(ValueError):
            validate_motion_target_position(np.array([3.0, 0.36, 1.23]), bounds, "right")
        # The unchanged spawn validator must still reject the bin region.
        with self.assertRaises(ValueError):
            validate_workspace_position(np.array([1.20, 0.36, 1.05]), self.safety)

    def test_object_escape_classification_and_ik_gate(self):
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        status, _ = classify_object_recoverability(
            np.array([0.90, 0.30, 1.04]), quaternion, self.safety
        )
        self.assertEqual(status, "RECOVERABLE")
        self.assertTrue(object_relative_recovery_allowed(status))

        escaped_x, _ = classify_object_recoverability(
            np.array([4.7, 1.8, 0.01]), quaternion, self.safety
        )
        self.assertEqual(escaped_x, "UNRECOVERABLE_ESCAPE")
        self.assertFalse(object_relative_recovery_allowed(escaped_x))

        escaped_z, _ = classify_object_recoverability(
            np.array([0.75, 0.28, 0.50]), quaternion, self.safety
        )
        self.assertEqual(escaped_z, "UNRECOVERABLE_ESCAPE")
        self.assertFalse(object_relative_recovery_allowed(escaped_z))

    def test_episode_retry_loop_resets_scene_and_is_bounded(self):
        source = Path("src/lerobot/auto_collect/auto_collect_base.py").read_text(encoding="utf-8")
        self.assertIn("while not episode_success and retry_count < cfg.max_retries", source)
        self.assertIn("robot.reset()", source)
        self.assertIn("dataset.clear_episode_buffer()", source)
        self.assertIn("_after_episode_attempts_exhausted(episode_idx, retry_count)", source)

    def test_finger_interpolation_uses_hold_positions_not_measured_state(self):
        """_joint_interpolate_to_pose must seed finger DOFs from the
        Python-tracked hold label, never from live measured joint state
        (which is contact/torque-driven and not a valid position command
        while gripping). Arm DOFs must continue using measured state."""

        class FakeInterface:
            def __init__(self):
                self.arm_joint_indices = list(range(14))
                self.finger_joint_indices = list(range(14, 18))
                self.joint_interpolator = JointInterpolator()

            def control_dual_arm_ik(self, **_kwargs):
                return {}

            def get_joint_states(self):
                measured = np.zeros(18, dtype=np.float64)
                measured[:14] = 0.555  # arbitrary measured arm state, distinct from the hold label
                measured[14:18] = 0.777  # torque-drifted measured finger state, distinct from the hold label
                return {"all_positions": measured}

        class FakeRobot:
            def __init__(self):
                self._robot_interface = FakeInterface()
                self._hold_arm_positions = np.full(14, 0.111, dtype=np.float32)
                self._hold_finger_positions = np.full(4, 0.1, dtype=np.float32)
                self._left_gripping = True
                self._right_gripping = True

        task = TaskPartSortingRecoveryV2()
        task._active_target_context = {}
        recorded: list[tuple[np.ndarray, np.ndarray]] = []

        def spy_causal_step(_robot, _dataset, _single_task, arm_positions, finger_positions, _l, _r):
            recorded.append((np.array(arm_positions, dtype=np.float64), np.array(finger_positions, dtype=np.float64)))

        task._causal_step = spy_causal_step
        robot = FakeRobot()
        held_fingers = robot._hold_finger_positions.copy()

        task._joint_interpolate_to_pose(robot, {}, dt=1.0 / 30.0, duration=0.3, dataset=None, single_task="test")

        self.assertGreater(len(recorded), 1)
        for arm_positions, finger_positions in recorded:
            np.testing.assert_allclose(finger_positions, held_fingers, atol=1e-6)
            self.assertFalse(np.allclose(finger_positions, 0.777, atol=0.05))
        first_arm_positions, _ = recorded[0]
        np.testing.assert_allclose(first_arm_positions, np.full(14, 0.555), atol=1e-6)

    def test_displacement_stops_once_target_band_reached(self):
        trajectory = [(0.75 + 0.006 * i, 0.28) for i in range(8)]
        scene = _ScriptedSceneBuilder("/Root/target", trajectory)
        robot = _ScriptedRobot(scene)
        task = _make_displacement_task(max_steps=20)

        summary = task._apply_physical_displacement(robot, "/Root/target", None, "test")

        self.assertEqual(summary["stop_reason"], "target_band_reached")
        self.assertEqual(summary["steps_taken"], 5)
        self.assertEqual(robot.step_calls, 5)
        self.assertGreaterEqual(summary["final_displacement_m"], 0.03)
        self.assertLessEqual(summary["final_displacement_m"], 0.08)

    def test_displacement_stops_before_workspace_edge(self):
        trajectory = [(0.60 - 0.003 * i, 0.28) for i in range(8)]
        scene = _ScriptedSceneBuilder("/Root/target", trajectory)
        robot = _ScriptedRobot(scene)
        task = _make_displacement_task(max_steps=20)

        summary = task._apply_physical_displacement(robot, "/Root/target", None, "test")

        self.assertEqual(summary["stop_reason"], "edge_margin_reached")
        self.assertLess(summary["final_displacement_m"], 0.03)
        self.assertLessEqual(summary["edge_distance_m"], 0.02)

    def test_displacement_zeroes_linear_and_angular_velocity_on_stop(self):
        trajectory = [(0.75 + 0.006 * i, 0.28) for i in range(8)]
        scene = _ScriptedSceneBuilder("/Root/target", trajectory)
        robot = _ScriptedRobot(scene)
        task = _make_displacement_task(max_steps=20)

        summary = task._apply_physical_displacement(robot, "/Root/target", None, "test")

        self.assertEqual(summary["stop_reason"], "target_band_reached")
        rigid = scene._parts_rigid_prims[0]
        np.testing.assert_array_equal(rigid.velocity, np.zeros(3))
        self.assertIsNotNone(rigid.angular_velocity)
        np.testing.assert_array_equal(rigid.angular_velocity, np.zeros(3))

    def test_displacement_fails_cleanly_if_threshold_never_reached(self):
        trajectory = [(0.75 + 0.0005 * i, 0.28) for i in range(11)]
        scene = _ScriptedSceneBuilder("/Root/target", trajectory)
        robot = _ScriptedRobot(scene)
        task = _make_displacement_task(max_steps=10)

        summary = task._apply_physical_displacement(robot, "/Root/target", None, "test")

        self.assertEqual(summary["stop_reason"], "step_budget_exhausted")
        self.assertEqual(summary["steps_taken"], 10)
        self.assertLess(summary["final_displacement_m"], 0.03)

    def test_displacement_aborts_if_it_overshoots_upper_bound(self):
        trajectory = [(0.75, 0.28), (0.90, 0.28)]
        scene = _ScriptedSceneBuilder("/Root/target", trajectory)
        robot = _ScriptedRobot(scene)
        task = _make_displacement_task(max_steps=5)

        with self.assertRaises(RecoveryV2Abort):
            task._apply_physical_displacement(robot, "/Root/target", None, "test")

    def test_protected_scenarios_are_unchanged(self):
        source = Path("src/lerobot/auto_collect/task_part_sorting_recovery_v2.py").read_text(encoding="utf-8")
        # drop_and_regrasp
        self.assertIn('drop_xyz[2] = initial_z + float(self._pilot_config["recovery"]["drop_release_height_above_table_m"])', source)
        self.assertIn('self.move_gripper(robot, {"right": -1.0}, dt, 0.2, dataset, single_task)', source)
        # missed_first_grasp / normal grasp-lift-place sequence
        self.assertIn('self._transition(metadata, RecoveryState.GRASP, attempt)', source)
        self.assertIn('self._transition(metadata, RecoveryState.LIFT, attempt)', source)
        self.assertIn('self._transition(metadata, RecoveryState.PLACE, attempt)', source)
        # difficult_position
        self.assertIn('def _initialize_difficult_position(self, parts: list[dict]) -> None:', source)
        # the 0.03 m minimum check itself is untouched, still owned by the caller
        self.assertIn(
            'if displacement < float(self._pilot_config["recovery"]["displacement_threshold_m"]):',
            source,
        )
        self.assertIn('f"displacement scenario moved only {displacement:.3f} m; refusing mislabeled episode"', source)

    def test_action_contract_and_limits(self):
        action = np.zeros(20)
        action[18:] = [1.0, -1.0]
        lower = np.full(18, -2.0)
        upper = np.full(18, 2.0)
        validate_action_vector(action, lower, upper, -0.0215, 0.01, self.safety)
        action[0] = 2.1
        with self.assertRaises(ValueError):
            validate_action_vector(action, lower, upper, -0.0215, 0.01, self.safety)

    def test_finger_limits_do_not_use_revolute_joint_margin(self):
        action = np.zeros(20)
        action[14:18] = -0.0215
        action[18:] = [1.0, 1.0]
        lower = np.full(18, -2.0)
        upper = np.full(18, 2.0)
        lower[14:18] = -0.0215
        upper[14:18] = 0.01
        validate_action_vector(action, lower, upper, -0.0215, 0.01, self.safety)

    def test_metadata_requires_four_objects_and_diagnostic_only_poses(self):
        objects = [
            RecoveryObjectMetadata(
                object_identity=f"object_{index}",
                prim_path=f"/Root/Task1_part_a_{index:02d}",
                part_type="part_a" if index < 2 else "part_b",
                asset_identity=None,
                initial_pose_xyzw=[0.75, 0.28, 1.04, 0.0, 0.0, 0.0, 1.0],
                recovery_type="normal_control",
            )
            for index in range(4)
        ]
        metadata = RecoveryEpisodeMetadata(
            schema_version=2,
            pilot_index=0,
            saved_episode_id=0,
            collection_attempt=1,
            random_seed=1,
            scenario_type="normal_control",
            difficulty="control",
            target_object_slot=0,
            temporal_convention="observe-record-apply-step",
            object_poses_policy_input=False,
            objects=objects,
        ).to_dict()
        validate_episode_metadata(metadata)
        metadata["object_poses_policy_input"] = True
        with self.assertRaises(ValueError):
            validate_episode_metadata(metadata)


if __name__ == "__main__":
    unittest.main()
