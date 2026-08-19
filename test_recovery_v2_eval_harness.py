from __future__ import annotations

import ast
import glob
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import recovery_v2_eval_disturbance as disturbance
from src.lerobot.configs.policies import PreTrainedConfig
from src.lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from src.lerobot.policies.factory import make_pre_post_processors
from src.lerobot.processor import (
    StateSlicerProcessorStep,
    load_normalizer_stats_from_pretrained,
    slice_stats_for_state,
)

CHECKPOINT = Path("/mnt/d/guedr/Projects/GHRC2026/checkpoints/part_sorting_act_200k")
ORIGINAL_REPO = "local/Part_Sorting"
ORIGINAL_ROOT = Path("datasets/Part_Sorting")
RECOVERY_REPO = "local/part_sorting_recovery_v2_pilot50"
RECOVERY_ROOT = Path("/mnt/d/guedr/Projects/GHRC2026/datasets/Part_Sorting_Recovery_V2_Pilot50")


class _FakeRigid:
    def __init__(self):
        self.velocity = None
        self.angular_velocity = None
        self.world_pose = None

    def set_linear_velocity(self, v):
        self.velocity = np.asarray(v, dtype=np.float64)

    def set_angular_velocity(self, v):
        self.angular_velocity = np.asarray(v, dtype=np.float64)

    def set_world_pose(self, position, orientation):
        self.world_pose = (np.asarray(position, dtype=np.float64), np.asarray(orientation, dtype=np.float64))


class _FakeSceneBuilder:
    def __init__(self, prim_path, trajectory_xy, z=1.02):
        self.prim_path = prim_path
        self.trajectory = [np.array([xy[0], xy[1], z], dtype=np.float64) for xy in trajectory_xy]
        self.index = 0
        self.parts_prim_paths = [prim_path]
        self._parts_rigid_prims = [_FakeRigid()]

    def _ensure_rigid_prims(self):
        pass

    def get_parts_world_poses(self):
        position = self.trajectory[min(self.index, len(self.trajectory) - 1)]
        return [{"prim_path": self.prim_path, "position": position.copy(), "orientation": np.array([0.0, 0.0, 0.0, 1.0])}]

    def world_to_robot_coords(self, xyz):
        return np.asarray(xyz, dtype=np.float64)  # identity for these tests


class _FakeRobot:
    def __init__(self, scene_builder, ee_position=None, right_gripping=False):
        self._scene_builder = scene_builder
        self.step_calls = 0
        self._right_gripping = right_gripping
        self._ee_position = ee_position if ee_position is not None else np.array([0.0, 0.0, 0.0])

    class _Interface:
        def __init__(self, outer):
            self._outer = outer

        def get_ee_poses(self):
            return {"right": np.concatenate([self._outer._ee_position, [0, 0, 0, 1]])}

    @property
    def _robot_interface(self):
        return _FakeRobot._Interface(self)

    def step(self, render=True):
        self.step_calls += 1
        if self._scene_builder.index < len(self._scene_builder.trajectory) - 1:
            self._scene_builder.index += 1


class DisturbanceModuleStaticTests(unittest.TestCase):
    """The CRITICAL RULE, enforced structurally: the disturbance module must never
    reference anything that could drive the arm/gripper trajectory."""

    FORBIDDEN_NAMES = {
        "send_action", "make_robot_action", "_joint_interpolate_to_pose", "move_gripper",
        "set_arm_joint_positions", "control_dual_arm_ik", "_hold_arm_positions",
        "_hold_finger_positions", "apply_finger_efforts",
    }

    def test_module_source_never_references_arm_control(self):
        source = Path("recovery_v2_eval_disturbance.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        referenced_names = {
            node.attr if isinstance(node, ast.Attribute) else node.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Attribute, ast.Name))
        }
        offending = referenced_names & self.FORBIDDEN_NAMES
        self.assertFalse(offending, f"recovery_v2_eval_disturbance.py references forbidden arm-control names: {offending}")

    def test_only_allowed_rigid_body_calls_present(self):
        source = Path("recovery_v2_eval_disturbance.py").read_text(encoding="utf-8")
        allowed = {"set_linear_velocity", "set_angular_velocity", "set_world_pose"}
        calls = {
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in
            {"set_linear_velocity", "set_angular_velocity", "set_world_pose", "send_action"}
        }
        self.assertTrue(calls <= allowed, f"unexpected rigid-body-adjacent calls: {calls - allowed}")


class DisturbanceConfigTests(unittest.TestCase):
    def test_load_disturbance_config_matches_pilot_yaml(self):
        cfg = disturbance.load_disturbance_config("configs/recovery_v2_part_sorting_pilot_v2.yaml")
        self.assertEqual(cfg.workspace_x, (0.575, 0.925))
        self.assertEqual(cfg.workspace_y, (0.13, 0.43))
        self.assertEqual(cfg.displacement_steps, 60)
        self.assertEqual(cfg.displacement_threshold_m, 0.03)
        self.assertEqual(len(cfg.difficult_positions_xyz), 5)

    def test_edge_distance(self):
        xy = np.array([0.6, 0.28, 1.02])
        d = disturbance.edge_distance(xy, (0.575, 0.925), (0.13, 0.43))
        self.assertAlmostEqual(d, 0.025, places=6)  # closest edge is x-lower: 0.6-0.575


class ClosedLoopPushTests(unittest.TestCase):
    def test_stops_in_target_band_and_touches_only_object(self):
        trajectory = [(0.75 + 0.006 * i, 0.28) for i in range(8)]
        scene = _FakeSceneBuilder("/Root/target", trajectory)
        robot = _FakeRobot(scene)
        event = disturbance.apply_closed_loop_push(
            robot, "/Root/target", (0.1, 0.0, 0.0), 20, 0.03, 0.08, 0.02, (0.575, 0.925), (0.13, 0.43)
        )
        self.assertEqual(event["stop_reason"], "target_band_reached")
        self.assertEqual(robot.step_calls, event["steps_taken"])
        rigid = scene._parts_rigid_prims[0]
        np.testing.assert_array_equal(rigid.velocity, np.zeros(3))
        np.testing.assert_array_equal(rigid.angular_velocity, np.zeros(3))

    def test_aborts_cleanly_short_of_threshold(self):
        trajectory = [(0.75 + 0.0005 * i, 0.28) for i in range(11)]
        scene = _FakeSceneBuilder("/Root/target", trajectory)
        robot = _FakeRobot(scene)
        event = disturbance.apply_closed_loop_push(
            robot, "/Root/target", (0.1, 0.0, 0.0), 10, 0.03, 0.08, 0.02, (0.575, 0.925), (0.13, 0.43)
        )
        self.assertEqual(event["stop_reason"], "step_budget_exhausted")
        self.assertLess(event["final_displacement_m"], 0.03)


class TeleportTests(unittest.TestCase):
    def test_teleport_calls_set_world_pose_and_zeros_velocity(self):
        scene = _FakeSceneBuilder("/Root/target", [(0.7, 0.2)])
        robot = _FakeRobot(scene)
        disturbance.teleport_to_difficult_position(robot, "/Root/target", (0.62, 0.135, 1.04))
        rigid = scene._parts_rigid_prims[0]
        self.assertIsNotNone(rigid.world_pose)
        np.testing.assert_allclose(rigid.world_pose[0], [0.62, 0.135, 1.04])
        np.testing.assert_array_equal(rigid.velocity, np.zeros(3))
        np.testing.assert_array_equal(rigid.angular_velocity, np.zeros(3))
        self.assertEqual(robot.step_calls, 15)


class GraspHoldDetectorTests(unittest.TestCase):
    def test_detects_first_close_then_stable_hold(self):
        scene = _FakeSceneBuilder("/Root/target", [(0.7, 0.2)])
        robot = _FakeRobot(scene, ee_position=np.array([0.7, 0.2, 1.02]), right_gripping=False)
        detector = disturbance.GraspHoldDetector("/Root/target", grasp_detect_distance_m=0.08, confirm_steps=3)

        first_close, stable = detector.update(robot, 0)
        self.assertFalse(first_close)  # not gripping yet
        self.assertFalse(stable)

        robot._right_gripping = True
        first_close, stable = detector.update(robot, 1)
        self.assertTrue(first_close)
        self.assertFalse(stable)

        # A second call with gripping=True should NOT re-fire first_close.
        first_close, stable = detector.update(robot, 2)
        self.assertFalse(first_close)
        self.assertFalse(stable)  # only 2 consecutive steps so far, need 3

        first_close, stable = detector.update(robot, 3)
        self.assertFalse(first_close)
        self.assertTrue(stable)

    def test_releasing_gripper_resets_consecutive_count(self):
        scene = _FakeSceneBuilder("/Root/target", [(0.7, 0.2)])
        robot = _FakeRobot(scene, ee_position=np.array([0.7, 0.2, 1.02]), right_gripping=True)
        detector = disturbance.GraspHoldDetector("/Root/target", grasp_detect_distance_m=0.08, confirm_steps=2)
        detector.update(robot, 0)
        robot._right_gripping = False
        _, stable = detector.update(robot, 1)
        self.assertFalse(stable)
        robot._right_gripping = True
        detector.update(robot, 2)
        _, stable = detector.update(robot, 3)
        self.assertTrue(stable)


class ForceGripperReleaseTests(unittest.TestCase):
    def test_only_overrides_right_gripper_key(self):
        action = {name: 0.123 for name in [
            "L_shoulder_pitch_joint.pos", "R_elbow_roll_joint.pos", "L_finger1_joint.pos",
            "R_finger1_joint.pos", "left_gripper", "right_gripper",
        ]}
        overridden = disturbance.force_gripper_release(action, steps_remaining=2)
        for key in action:
            if key == "right_gripper":
                self.assertEqual(overridden[key], 1.0)
            else:
                self.assertEqual(overridden[key], action[key], f"{key} was modified but should not have been")


ALL_NORMALIZED_FEATURES = (
    "observation.state",
    "action",
    "observation.images.head_left",
    "observation.images.head_right",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


def _load_checkpoint_saved_normalizer(checkpoint: Path) -> dict[str, torch.Tensor]:
    matches = glob.glob(str(checkpoint / "policy_preprocessor*normalizer*.safetensors"))
    if not matches:
        raise FileNotFoundError(f"No normalizer safetensors found under {checkpoint}")
    return load_file(matches[0])


class NormalizerPreservationTests(unittest.TestCase):
    """Proves load_normalizer_stats_from_pretrained (wired into lerobot_train.py
    via --normalization_stats_pretrained_path) reproduces the original 200K
    checkpoint's own saved normalizer EXACTLY, for every normalized feature --
    state, action, AND all four visual features. This supersedes the original
    normalization_stats_repo_id mechanism, which only ever verified the state
    dimension: sourcing stats from a fresh LeRobotDatasetMetadata load (even of
    the checkpoint's own original training dataset) bypasses the
    use_imagenet_stats substitution applied at make_dataset() time, so visual
    stats end up as the dataset's raw near-zero per-pixel std instead of the
    checkpoint's actual ImageNet-based normalizer -- silently corrupting the
    image inputs and producing NaN policy outputs. See
    diagnose_finetune_inference_v2.py for the live repro of that failure."""

    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.is_dir():
            raise unittest.SkipTest(f"checkpoint not available at {CHECKPOINT}")

    def test_loaded_stats_match_checkpoint_saved_normalizer_for_every_feature(self):
        saved = _load_checkpoint_saved_normalizer(CHECKPOINT)
        loaded = load_normalizer_stats_from_pretrained(str(CHECKPOINT))
        truncated = slice_stats_for_state(loaded, n_dims=20)

        for key in ALL_NORMALIZED_FEATURES:
            for stat_name in ("mean", "std"):
                saved_val = saved[f"{key}.{stat_name}"]
                loaded_val = torch.as_tensor(truncated[key][stat_name], dtype=torch.float32)
                torch.testing.assert_close(
                    saved_val,
                    loaded_val,
                    atol=1e-6,
                    rtol=0,
                    msg=f"{key}.{stat_name} did not match the checkpoint's saved normalizer",
                )

    def test_image_stats_are_imagenet_scale_not_raw_dataset_scale(self):
        """Regression test for the exact bug found: raw dataset-computed image
        std is ~0.0002-0.005 (near-zero); the checkpoint's real normalizer uses
        ImageNet-scale std ~0.224-0.229. Assert the loaded stats are the latter."""
        loaded = load_normalizer_stats_from_pretrained(str(CHECKPOINT))
        for key in (
            "observation.images.head_left",
            "observation.images.head_right",
            "observation.images.wrist_left",
            "observation.images.wrist_right",
        ):
            std = torch.as_tensor(loaded[key]["std"], dtype=torch.float32)
            self.assertTrue(
                bool((std > 0.1).all()),
                f"{key} std={std.tolist()} looks like raw near-zero dataset stats, not ImageNet scale",
            )

    def test_raw_dataset_stats_would_have_corrupted_images(self):
        """Negative control: proves the OLD (buggy) mechanism's data source --
        a fresh LeRobotDatasetMetadata load of the checkpoint's own original
        training dataset -- genuinely differs from the checkpoint's real
        normalizer for images, even though it happens to match for state."""
        saved = _load_checkpoint_saved_normalizer(CHECKPOINT)
        orig_meta = LeRobotDatasetMetadata(ORIGINAL_REPO, root=ORIGINAL_ROOT)
        raw_std = torch.as_tensor(orig_meta.stats["observation.images.head_left"]["std"], dtype=torch.float32)
        self.assertFalse(
            torch.allclose(saved["observation.images.head_left.std"], raw_std, atol=1e-3),
            "raw dataset stats unexpectedly matched the checkpoint's saved normalizer -- "
            "the bug this test guards against may no longer reproduce",
        )

    def test_end_to_end_constructed_preprocessor_matches_checkpoint_for_every_feature(self):
        """Exercises the exact code path lerobot_train.py uses: load stats from
        the pretrained checkpoint, slice for state, feed into
        make_pre_post_processors via preprocessor_overrides, and confirm the
        resulting live NormalizerProcessorStep matches the checkpoint's saved
        normalizer for every feature -- not just state."""
        saved = _load_checkpoint_saved_normalizer(CHECKPOINT)
        loaded = load_normalizer_stats_from_pretrained(str(CHECKPOINT))
        truncated = slice_stats_for_state(loaded, n_dims=20)
        policy_cfg = PreTrainedConfig.from_pretrained(str(CHECKPOINT))
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(CHECKPOINT),
            dataset_stats=truncated,
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                "normalizer_processor": {
                    "stats": truncated,
                    "features": {**policy_cfg.input_features, **policy_cfg.output_features},
                    "norm_map": policy_cfg.normalization_mapping,
                },
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": truncated,
                    "features": policy_cfg.output_features,
                    "norm_map": policy_cfg.normalization_mapping,
                },
            },
        )
        preprocessor.steps.insert(0, StateSlicerProcessorStep(n_dims=20))
        normalizer_step = next(step for step in preprocessor.steps if hasattr(step, "stats"))
        unnormalizer_step = next(step for step in postprocessor.steps if hasattr(step, "stats"))

        for key in ALL_NORMALIZED_FEATURES:
            for stat_name in ("mean", "std"):
                saved_val = saved[f"{key}.{stat_name}"]
                constructed_val = torch.as_tensor(
                    normalizer_step.stats[key][stat_name], dtype=torch.float32
                )
                torch.testing.assert_close(saved_val, constructed_val, atol=1e-6, rtol=0)

        for stat_name in ("mean", "std"):
            saved_val = saved[f"action.{stat_name}"]
            constructed_val = torch.as_tensor(unnormalizer_step.stats["action"][stat_name], dtype=torch.float32)
            torch.testing.assert_close(saved_val, constructed_val, atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
