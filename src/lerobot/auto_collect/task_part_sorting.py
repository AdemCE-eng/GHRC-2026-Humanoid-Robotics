"""Task 1 —— Part Sorting: 从桌面抓取零件并按类型分类放入箱子。"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .auto_collect_base import AutoCollectBase
from .utils import get_part_sorting_part_type


class TaskPartSorting(AutoCollectBase):
    """从桌面抓取零件并按类型 (Part A / Part B) 放入箱子的不同区域。

    Part A → 箱子右侧 (+Y 偏移)
    Part B → 箱子左侧 (-Y 偏移)

    仅使用 right 臂。
    """

    _DEFAULT_HOME_RETURN_STEPS = 200
    _FIXED_GRASP_ORDER = ("up", "left", "right", "down")

    def __init__(self):
        super().__init__()
        self._current_episode_part_frames: list[dict] = []

    def compute_grasp_poses(self, part: dict) -> dict:
        """计算接近 / 下降 / 抓取 / 抬起位姿（世界坐标系）。

        位置从 part 的世界坐标派生，旋转使用预定义角度确保
        夹爪从上方接近并正确抓取。
        """
        target_pos = np.array(part["position"])

        return {
            "right": {
                "approach": {
                    "position": np.array([target_pos[0], target_pos[1], target_pos[2] + 0.25]),
                    "rotation": np.array([-np.pi, 0.0, -1.9]),
                },
                "grasp": {
                    "position": np.array([target_pos[0], target_pos[1], target_pos[2] + 0.01]),
                    "rotation": np.array([-np.pi, -0.6, -1.9]),
                },
                "lift": {
                    "position": np.array([target_pos[0], target_pos[1], target_pos[2] + 0.20]),
                    "rotation": np.array([-np.pi, -0.2, -1.9]),
                },
            },
        }

    def check_grasp_success(self, robot, part: dict, threshold: float = 0.08) -> bool:
        """通过与夹爪的距离偏差判断抓取是否成功。"""
        return self._check_grasp_success_for_arm(robot, part, arm_side="right", threshold=threshold)

    def get_place_pose(self, robot, part: dict, box_pos: np.ndarray) -> dict:
        """根据零件类型 (A/B) 计算带偏移的放置位姿。"""
        part_type = get_part_sorting_part_type(part, robot._scene_builder)
        if part_type == "part_a":
            place_offset = np.array([0.0, -0.08, 0.18])
            logging.info("  → 检测到 Part A，放置偏移 +Y")
        else:
            place_offset = np.array([0.0, 0.06, 0.18])
            logging.info("  → 检测到 Part B，放置偏移 -Y")

        return {
            "right": {
                "position": box_pos + place_offset,
                "rotation": np.array([-np.pi, -0.2, -2.8]),
            },
        }

    # =========================================================================
    # Episode metadata hooks
    # =========================================================================

    def _on_episode_start(self, parts: list[dict] | None = None) -> None:
        del parts
        self._current_episode_part_frames = []

    def _on_episode_saved(self, dataset, episode_index: int, episode_length: int) -> None:
        if dataset is None or not self._current_episode_part_frames:
            return

        part_info_path = Path(dataset.root) / "meta" / "part_info.json"
        part_info_path.parent.mkdir(parents=True, exist_ok=True)

        part_info = self._load_part_info(part_info_path)
        episode_info = {
            "episode_index": int(episode_index),
            "episode_frame_length": int(episode_length),
            "parts": list(self._current_episode_part_frames),
        }

        episodes = [
            episode
            for episode in part_info.get("episodes", [])
            if int(episode.get("episode_index", -1)) != int(episode_index)
        ]
        episodes.append(episode_info)
        episodes.sort(key=lambda episode: int(episode.get("episode_index", -1)))
        part_info["episodes"] = episodes

        part_info_path.write_text(
            json.dumps(part_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_part_info(self, part_info_path: Path) -> dict:
        if not part_info_path.exists():
            return {"episodes": []}

        try:
            loaded = json.loads(part_info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("Part_Sorting part_info.json 无法解析，将重建文件: %s", part_info_path)
            return {"episodes": []}

        if isinstance(loaded, dict) and isinstance(loaded.get("episodes"), list):
            return loaded
        if isinstance(loaded, list):
            return {"episodes": loaded}
        return {"episodes": []}

    def _record_part_frame_range(
        self,
        robot,
        part: dict,
        frame_start_index: int | None,
        frame_end_index: int,
    ) -> None:
        if frame_start_index is None or frame_end_index < frame_start_index:
            return

        scene_builder = getattr(robot, "_scene_builder", None) if robot is not None else None
        self._current_episode_part_frames.append(
            {
                "part_name": self._get_part_name(part),
                "part_type": get_part_sorting_part_type(part, scene_builder),
                "prim_path": str(part.get("prim_path", "")),
                "frame_start_index": int(frame_start_index),
                "frame_end_index": int(frame_end_index),
            }
        )

    def _get_dataset_episode_size(self, dataset) -> int | None:
        if dataset is None:
            return None
        episode_buffer = getattr(dataset, "episode_buffer", None)
        if not episode_buffer:
            return None
        return int(episode_buffer.get("size", 0))

    def _get_part_name(self, part: dict) -> str:
        for key in ("name", "part_name"):
            value = part.get(key)
            if value:
                return str(value)

        prim_path = str(part.get("prim_path", ""))
        if prim_path:
            return prim_path.rstrip("/").split("/")[-1]
        return str(part.get("index", "unknown"))

    # =========================================================================
    # Episode 流水线骨架
    # =========================================================================

    def _run_grasp_stages(
        self,
        robot,
        grasp_poses: dict,
        place_poses: dict,
        dt: float,
        dataset,
        single_task: str,
        check_success_fn: Callable[[], bool],
    ) -> bool:
        """执行 7 阶段抓取-放置流水线（共享核心）。"""
        active_arms = [arm for arm in ("left", "right") if grasp_poses.get(arm)]
        if not active_arms:
            return True

        arms_label = "/".join(active_arms)

        t_robot_to_world = robot._scene_builder.get_robot_world_transform()
        logging.info(f"[坐标变换调试] get_robot_world_transform (4x4):\n{t_robot_to_world}")

        # 阶段1: 接近
        logging.info(f"阶段1 [{arms_label}]: 接近目标...")
        target_poses = {}
        for arm in active_arms:
            p = grasp_poses[arm]["approach"]
            target_poses[arm] = {
                "position": robot._scene_builder.world_to_robot_coords(p["position"]),
                "rotation": p["rotation"],
            }
        self._joint_interpolate_to_pose(robot, target_poses, dt, 0.8, dataset, single_task)

        # 阶段2: 松开夹爪
        if (
            not robot._right_gripping
            and abs(robot._hold_finger_positions[2:4].mean() - robot._robot_interface.gripper_open_width)
            < 0.005
        ):
            logging.info(f"阶段2 [{arms_label}]: 夹爪已打开，跳过...")
        else:
            logging.info(f"阶段2 [{arms_label}]: 松开夹爪...")
            self.move_gripper(
                robot,
                {"right": -1.0},
                dt,
                0.3,
                dataset=dataset,
                single_task=single_task,
            )

        # 阶段3: 抓取
        logging.info(f"阶段3 [{arms_label}]: 抓取...")
        target_poses = {}
        for arm in active_arms:
            p = grasp_poses[arm]["grasp"]
            target_poses[arm] = {
                "position": robot._scene_builder.world_to_robot_coords(p["position"]),
                "rotation": p["rotation"],
            }
        self._joint_interpolate_to_pose(robot, target_poses, dt, 0.8, dataset, single_task)

        logging.info(f"阶段3b [{arms_label}]: 闭合夹爪...")
        self.move_gripper(
            robot,
            {"right": 1.0},
            dt,
            0.5,
            dataset=dataset,
            single_task=single_task,
        )

        # 阶段4: 抬起
        logging.info(f"阶段4 [{arms_label}]: 抬起手臂...")
        target_poses = {}
        for arm in active_arms:
            p = grasp_poses[arm]["lift"]
            target_poses[arm] = {
                "position": robot._scene_builder.world_to_robot_coords(p["position"]),
                "rotation": p["rotation"],
            }
        self._joint_interpolate_to_pose(robot, target_poses, dt, 0.8, dataset, single_task)

        # 阶段5: 移动到放置位置
        logging.info(f"阶段5 [{arms_label}]: 移动到放置位置...")
        target_poses = {}
        for arm in active_arms:
            pp = place_poses.get(arm)
            if pp is not None:
                target_poses[arm] = {
                    "position": robot._scene_builder.world_to_robot_coords(pp["position"]),
                    "rotation": pp["rotation"],
                }
        self._joint_interpolate_to_pose(robot, target_poses, dt, 1.1, dataset, single_task)

        # 阶段6: 检测抓取是否成功
        logging.info("阶段6: 检测抓取是否成功...")
        success = check_success_fn()
        if not success:
            logging.error("抓取失败！")
            return False

        # 阶段7: 松开夹爪
        logging.info(f"阶段7 [{arms_label}]: 松开夹爪...")
        self.move_gripper(
            robot,
            {"right": -1.0},
            dt,
            0.3,
            dataset=dataset,
            single_task=single_task,
        )

        return True

    # =========================================================================
    # Episode 流水线
    # =========================================================================

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
        """单臂 episode：逐个零件执行抓取-放置流水线。"""
        if not parts:
            logging.error("单臂模式下无可处理零件")
            return False
        completed = 0
        ordered_parts = self._sort_parts_by_grasp_order(robot, parts)
        for part_idx, part in enumerate(ordered_parts):
            part = self._refresh_part_pose(robot, part)
            logging.info(f"\n处理零件 {part_idx + 1}/{len(ordered_parts)}")
            grasp_poses = self.compute_grasp_poses(part)
            if not grasp_poses:
                continue
            place_poses = self.get_place_pose(robot, part, box_pos)
            part_frame_start = self._get_dataset_episode_size(dataset)
            success = self._run_grasp_stages(
                robot=robot,
                grasp_poses=grasp_poses,
                place_poses=place_poses,
                dt=dt,
                dataset=dataset,
                single_task=single_task,
                check_success_fn=lambda current_part=part: self.check_grasp_success(robot, current_part),
            )
            if not success:
                return False
            completed += 1
            logging.info(f"零件 {part_idx + 1} 处理完成")
            self._return_right_arm_home(
                robot=robot,
                dt=dt,
                dataset=dataset,
                single_task=single_task,
            )
            part_frame_end = self._get_dataset_episode_size(dataset)
            if part_frame_start is not None and part_frame_end is not None:
                self._record_part_frame_range(
                    robot=robot,
                    part=part,
                    frame_start_index=part_frame_start,
                    frame_end_index=part_frame_end - 1,
                )
            if objects_per_episode > 0 and completed >= objects_per_episode:
                logging.info(f"已达到本 episode 目标物体数 ({objects_per_episode})")
                break
        return True

    def _sort_parts_by_grasp_order(self, robot, parts: list[dict]) -> list[dict]:
        scene_builder = getattr(robot, "_scene_builder", None)
        slot_by_path = getattr(scene_builder, "part_slot_by_prim_path", {}) if scene_builder is not None else {}

        if slot_by_path:
            order_index = {slot: idx for idx, slot in enumerate(self._FIXED_GRASP_ORDER)}
            return sorted(
                parts,
                key=lambda part: order_index.get(slot_by_path.get(part.get("prim_path", "")), len(order_index)),
            )

        # 非固定槽位模式：从上到下（y 降序），y 相同时从左到右（x 升序）
        return sorted(
            parts,
            key=lambda p: (
                -float(np.asarray(p.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)[1]),
                float(np.asarray(p.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)[0]),
            ),
        )

    def _refresh_part_pose(self, robot, part: dict) -> dict:
        prim_path = part.get("prim_path", "")
        scene_builder = getattr(robot, "_scene_builder", None)
        if not prim_path or scene_builder is None:
            return part

        for current_part in scene_builder.get_parts_world_poses():
            if current_part.get("prim_path") != prim_path:
                continue
            refreshed = dict(part)
            refreshed["position"] = current_part.get("position", part.get("position"))
            refreshed["orientation"] = current_part.get("orientation", part.get("orientation"))
            return refreshed
        return part

    def _return_right_arm_home(
        self,
        robot,
        dt: float,
        dataset,
        single_task: str,
    ) -> None:
        """Move the right arm/fingers back to initial joint positions and record the transition."""
        del dt
        steps = self._get_home_return_steps(robot)
        logging.info(f"Part_Sorting: right arm returning home with {steps} recorded steps...")

        if robot._hold_arm_positions is None:
            robot._hold_arm_positions = np.array(
                robot._robot_interface.arm_joint_initial_positions,
                dtype=np.float32,
            )
        if robot._hold_finger_positions is None:
            robot._hold_finger_positions = np.array(
                robot._robot_interface.finger_joint_initial_positions,
                dtype=np.float32,
            )

        target_arm_positions = np.array(robot._hold_arm_positions, dtype=np.float32).copy()
        target_finger_positions = np.array(robot._hold_finger_positions, dtype=np.float32).copy()
        initial_arm_positions = np.array(
            robot._robot_interface.arm_joint_initial_positions,
            dtype=np.float32,
        )
        initial_finger_positions = np.array(
            robot._robot_interface.finger_joint_initial_positions,
            dtype=np.float32,
        )
        target_arm_positions[7:14] = initial_arm_positions[7:14]
        target_finger_positions[2:4] = initial_finger_positions[2:4]
        robot._right_gripping = False

        joint_states = robot._robot_interface.get_joint_states()
        if not joint_states or "all_positions" not in joint_states:
            logging.warning("Part_Sorting: cannot return right arm home, missing joint states")
            return

        target_positions = np.concatenate([target_arm_positions, target_finger_positions])
        arm_finger_indices = (
            robot._robot_interface.arm_joint_indices + robot._robot_interface.finger_joint_indices
        )
        import torch

        robot._robot_interface.joint_interpolator.reset()
        robot._robot_interface.joint_interpolator.set_target(
            start_q=torch.tensor(joint_states["all_positions"])[arm_finger_indices],
            target_q=torch.tensor(target_positions),
            num_steps=steps,
        )

        for _ in range(steps):
            if hasattr(self, "_raise_if_keyboard_requested"):
                self._raise_if_keyboard_requested()
            arm_finger_positions = robot._robot_interface.joint_interpolator.step()
            if hasattr(arm_finger_positions, "detach"):
                arm_finger_positions = arm_finger_positions.detach().cpu().numpy()
            else:
                arm_finger_positions = np.asarray(arm_finger_positions, dtype=np.float32)

            robot._hold_arm_positions = arm_finger_positions[:14]
            robot._hold_finger_positions = arm_finger_positions[14:18]
            robot.step(render=True)
            if dataset is not None and hasattr(dataset, "features"):
                self._record_frame(robot, dataset, single_task)

        for _ in range(10):
            if hasattr(self, "_raise_if_keyboard_requested"):
                self._raise_if_keyboard_requested()
            robot.step(render=True)
            if dataset is not None and hasattr(dataset, "features"):
                self._record_frame(robot, dataset, single_task)
        logging.info("Part_Sorting: right arm returned home.")

    def _get_home_return_steps(self, robot) -> int:
        grasp_cfg = self._get_grasp_cfg(robot)
        return max(
            1,
            int(grasp_cfg.get("part_sorting_home_return_steps", self._DEFAULT_HOME_RETURN_STEPS)),
        )

    def _get_grasp_cfg(self, robot) -> dict:
        return (getattr(robot.config, "task_cfg", {}) or {}).get("grasp", {})
