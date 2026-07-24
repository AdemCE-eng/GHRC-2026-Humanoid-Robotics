"""Task 3 —— Foam Inlaying: 双臂协作抓取泡沫块并嵌入对应槽位。

任务描述:
    在桌面料箱内有 6 个工件（2 类各 3 个），位置姿态随机。
    - 工件 A（小工件）→ left 臂抓取 → 放入小嵌入孔
    - 工件 B（大工件）→ right 臂抓取 → 放入大嵌入孔

小嵌入孔（3 个，几何中心世界坐标）:
    - (0.57025, 0.20973, 1.06959)  → /Replicator/Ref_Xform_02
    - (0.75985, 0.20973, 1.06959)  → /Replicator/Ref_Xform_03
    - (0.94964, 0.20973, 1.06959)  → /Replicator/Ref_Xform_04

大嵌入孔（3 个，几何中心世界坐标）:
    - (0.57025, 0.36936, 1.06959)  → /Replicator/Ref_Xform_05
    - (0.75985, 0.36936, 1.06959)  → /Replicator/Ref_Xform_06
    - (0.94964, 0.36936, 1.06959)  → /Replicator/Ref_Xform_07
"""

import json
import logging
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .auto_collect_base import AutoCollectBase
from .utils import get_foam_inlaying_part_type


class TaskFoamInlaying(AutoCollectBase):
    """双臂协作泡沫嵌入任务。

    left 臂抓取小工件放入小孔，right 臂抓取大工件放入大孔。
    每个 episode 内随机打乱孔位顺序，各零件分配唯一孔位。
    左右臂零件配对后同时控制双臂完成抓取-放置流水线。
    """

    is_dual_arm: bool = True
    _SUPPORTED_ARM_EXECUTION_MODES = {"dual", "left_then_right", "right_then_left"}
    _DEFAULT_HOME_RETURN_STEPS = 200

    # 两种盒子的中心 X 坐标
    _LEFT_BOX_CENTER_X = 0.28
    _RIGHT_BOX_CENTER_X = 1.25
    _EDGE_DISTANCE_THRESHOLD = 0.08  # box_x - part_x > 0.08 => left edge, < -0.08 => right edge
    _PLACED_PART_MAX_SUCCESS_Z = 1.088

    # 三种grasp_rpy预设
    _GRASP_RPY_OPTIONS = {
        "left": {
            # "near_left": np.array([3.1359, -0.7598, -2.6459]),
            # "near_right": np.array([-2.9971, -0.6638, 2.2078]),
            # "middle": np.array([3.1282, -0.5499, 3.0268]),

            "near_left": np.array([3.1282, -0.6999, 3.0268]),
            "near_right": np.array([3.1282, -0.6999, 3.0268]),
            "middle": np.array([3.1282, -0.6999, 3.0268]),
        },
        "right": {
            "near_left": np.array([3.1282, -0.6999, -3.0268]),
            "near_right": np.array([3.1282, -0.6999, -3.0268]),
            "middle": np.array([3.1282, -0.6999, -3.0268]),
        },
    }

    # 小嵌入孔世界坐标（3 个）
    _SMALL_HOLE_POSITIONS = [
        np.array([0.57025, 0.20973, 1.06959]),
        np.array([0.75985, 0.20973, 1.06959]),
        np.array([0.94964, 0.20973, 1.06959]),
    ]
    # 大嵌入孔世界坐标（3 个）
    _LARGE_HOLE_POSITIONS = [
        np.array([0.94964, 0.36936, 1.06959]),
        np.array([0.75985, 0.36936, 1.06959]),
        np.array([0.57025, 0.36936, 1.06959]),
    ]

    # 固定 RPY —— left 臂（part_a / 小工件）
    _LEFT_APPROACH_RPY = np.array([3.0885, 0.2659, 2.8305])
    _LEFT_GRASP_RPY = np.array([3.1282, -0.5499, 4.0268])
    _LEFT_PLACE_RPY = np.array([3.1282, -0.5499, 2.2010])

    # 固定 RPY —— right 臂（part_b / 大工件）
    _RIGHT_APPROACH_RPY = np.array([-3.0885, 0.2651, -2.8296])
    _RIGHT_GRASP_RPY = np.array([3.1282, -0.5499, -3.0268])
    _RIGHT_PLACE_RPY = np.array([3.1282, -0.5499, -2.2010])

    # Grasp height offsets relative to the current part origin.
    # Tune these per box/arm when small and large parts need different depths.
    _GRASP_Z_OFFSETS = {
        "left": {
            "descend": 0.08,
            "grasp": 0.026,
            "lift": 0.30,
        },
        "right": {
            "descend": 0.12,
            "grasp": 0.02,
            "lift": 0.30,
        },
    }

    def __init__(self):
        super().__init__()
        self._shuffled_small_holes: list[np.ndarray] = []
        self._shuffled_large_holes: list[np.ndarray] = []
        self._hole_assignment: dict[str, np.ndarray] = {}
        self._current_episode_part_frames: list[dict] = []
        self._box_center_x_by_arm = {
            "left": self._LEFT_BOX_CENTER_X,
            "right": self._RIGHT_BOX_CENTER_X,
        }

    # =========================================================================
    # 钩子方法
    # =========================================================================

    def _on_episode_start(self, parts: list[dict] | None = None) -> None:
        """随机打乱孔位顺序，清空孔位分配缓存。"""
        del parts
        self._hole_assignment.clear()
        self._current_episode_part_frames = []
        self._shuffled_small_holes = random.sample(
            self._SMALL_HOLE_POSITIONS, len(self._SMALL_HOLE_POSITIONS)
        )
        self._shuffled_large_holes = random.sample(
            self._LARGE_HOLE_POSITIONS, len(self._LARGE_HOLE_POSITIONS)
        )
        logging.info(
            f"孔位已随机打乱\n"
            f"小孔: {[np.round(p, 4).tolist() for p in self._shuffled_small_holes]},\n"
            f"大孔: {[np.round(p, 4).tolist() for p in self._shuffled_large_holes]}"
        )

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

    def get_arm_side(self, part: dict) -> str:
        """小工件（part_a）→ left，大工件（part_b）→ right。"""
        part_type = get_foam_inlaying_part_type(part, None)
        return "left" if part_type == "part_a" else "right"

    def _update_box_centers_from_scene(self, robot) -> None:
        """Sync source-box center X from the scene, falling back to class defaults."""
        scene_builder = getattr(robot, "_scene_builder", None)
        if scene_builder is None:
            return

        box_positions = scene_builder.get_box_positions()
        if len(box_positions) < 2:
            return

        boxes_by_x = sorted((np.array(pos, dtype=np.float64) for pos in box_positions), key=lambda p: p[0])
        self._box_center_x_by_arm["left"] = float(boxes_by_x[0][0])
        self._box_center_x_by_arm["right"] = float(boxes_by_x[-1][0])
        logging.info(
            "Task3 box center x synced: "
            f"left={self._box_center_x_by_arm['left']:.4f}, "
            f"right={self._box_center_x_by_arm['right']:.4f}"
        )

    # =========================================================================
    # 任务差异接口
    # =========================================================================

    def compute_grasp_poses(self, part: dict) -> dict:
        """计算单臂接近 / 下降 / 抓取 / 抬起位姿（世界坐标系）。"""
        arm_side = self.get_arm_side(part)
        return {arm_side: self._build_arm_poses(part, arm_side)}

    def check_grasp_success(self, robot, part: dict, threshold: float = 0.08) -> bool:
        """检查单臂夹爪与零件的距离。"""
        arm_side = self.get_arm_side(part)
        logging.info(f"Foam 抓取检测 [{arm_side}]: 零件 {part.get('prim_path', '')}")
        return self._check_grasp_success_for_arm(robot, part, arm_side=arm_side, threshold=threshold)

    def get_place_pose(self, robot, part: dict, box_pos: np.ndarray) -> dict:
        """返回单臂放置目标位姿（世界坐标系）。"""
        prim_path = part.get("prim_path", "")
        arm_side = self.get_arm_side(part)

        hole_pos = self._hole_assignment.get(prim_path)
        if hole_pos is None:
            logging.error(f"零件 {prim_path} 未分配孔位，回退到箱体位置")
            hole_pos = box_pos
        rotation = self._LEFT_PLACE_RPY if arm_side == "left" else self._RIGHT_PLACE_RPY
        place_pos = hole_pos.copy()
        place_pos[2] += 0.2
        return {arm_side: {"position": place_pos, "rotation": rotation.copy()}}

    def _load_part_info(self, part_info_path: Path) -> dict:
        if not part_info_path.exists():
            return {"episodes": []}

        try:
            loaded = json.loads(part_info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("Foam_Inlaying part_info.json 无法解析，将重建文件: %s", part_info_path)
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
                "part_type": get_foam_inlaying_part_type(part, scene_builder),
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

    def _check_placed_part_z_height(self, robot, part: dict) -> bool:
        """Return True only if the released part height is below the success threshold."""
        refreshed = self._refresh_part_pose(robot, part)
        prim_path = refreshed.get("prim_path", part.get("prim_path", ""))
        part_pos = refreshed.get("position")
        if part_pos is None:
            print(f"[Task3 placed height check] FAIL missing position: {prim_path}")
            return False

        part_z = float(np.asarray(part_pos, dtype=np.float64)[2])
        height_ok = part_z < self._PLACED_PART_MAX_SUCCESS_Z
        status = "PASS" if height_ok else "FAIL"
        print(
            "[Task3 placed height check] "
            f"{status} part={prim_path}, "
            f"part_z={part_z:.4f}, required_z<{self._PLACED_PART_MAX_SUCCESS_Z:.4f}"
        )
        return height_ok

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
        placed_parts: list[dict] | None = None,
    ) -> bool:
        """执行 8 阶段抓取-放置流水线（共享核心）。"""
        active_arms = [arm for arm in ("left", "right") if grasp_poses.get(arm)]
        if not active_arms:
            return True

        arms_label = "/".join(active_arms)

        T_robot_to_world = robot._scene_builder.get_robot_world_transform()
        logging.info(
            f"[坐标变换调试] get_robot_world_transform (4x4):\n{T_robot_to_world}"
        )

        # 阶段1: 接近
        logging.info(f"阶段1 [{arms_label}]: 接近目标...")
        target_poses = {}
        for arm in active_arms:
            p = grasp_poses[arm]["approach"]
            target_poses[arm] = {
                "position": robot._scene_builder.world_to_robot_coords(p["position"]),
                "rotation": p["rotation"],
            }
        self._joint_interpolate_to_pose(robot, target_poses, dt, 1.0, dataset, single_task)


        # 阶段2: 下降
        logging.info(f"阶段2 [{arms_label}]: 手臂下降...")
        target_poses = {}
        for arm in active_arms:
            p = grasp_poses[arm]["descend"]
            target_poses[arm] = {
                "position": robot._scene_builder.world_to_robot_coords(p["position"]),
                "rotation": p["rotation"],
            }
        self._joint_interpolate_to_pose(robot, target_poses, dt, 1.0, dataset, single_task)

        # 阶段3: 松开夹爪
        logging.info(f"阶段3 [{arms_label}]: 松开夹爪...")
        self.gradually_move_gripper(
            robot, {arm: -1.0 for arm in active_arms},
            steps=2, dataset=dataset, single_task=single_task,
        )

        # 阶段4: 抓取
        logging.info(f"阶段4 [{arms_label}]: 抓取...")
        target_poses = {}
        for arm in active_arms:
            p = grasp_poses[arm]["grasp"]
            target_poses[arm] = {
                "position": robot._scene_builder.world_to_robot_coords(p["position"]),
                "rotation": p["rotation"],
            }
        self._joint_interpolate_to_pose(robot, target_poses, dt, 1.0, dataset, single_task)

        logging.info(f"阶段4b [{arms_label}]: 闭合夹爪...")
        self.gradually_move_gripper(
            robot, {arm: 1.0 for arm in active_arms},
            steps=5, dataset=dataset, single_task=single_task,
        )

        # 阶段5: 抬起
        logging.info(f"阶段5 [{arms_label}]: 抬起手臂...")
        target_poses = {}
        for arm in active_arms:
            p = grasp_poses[arm]["lift"]
            target_poses[arm] = {
                "position": robot._scene_builder.world_to_robot_coords(p["position"]),
                "rotation": p["rotation"],
            }
        self._joint_interpolate_to_pose(robot, target_poses, dt, 1.5, dataset, single_task)

        logging.info(f"阶段5b [{arms_label}]: lift 后验证工件是否仍在夹爪上...")
        success = check_success_fn()
        if not success:
            logging.error("lift 验证失败：工件未跟随夹爪抬起，重试当前 episode")
            return False

        # 阶段6: 笛卡尔移动到放置位置上方（before_place）
        logging.info(f"阶段6 [{arms_label}]: 笛卡尔移动到 before_place...")
        before_place_poses = {}
        for arm in active_arms:
            bp = grasp_poses[arm].get("before_place")
            if bp is not None:
                before_place_poses[arm] = {
                    "position": robot._scene_builder.world_to_robot_coords(bp["position"]),
                    "rotation": bp["rotation"],
                }
        self._cartesian_interpolate_to_pose(robot, before_place_poses, dt, 2, dataset, single_task)

        # 阶段6b: 关节插值下降到实际放置位置
        logging.info(f"阶段6b [{arms_label}]: 关节插值下降到放置位置...")
        target_poses = {}
        for arm in active_arms:
            pp = place_poses.get(arm)
            if pp is not None:
                target_poses[arm] = {
                    "position": robot._scene_builder.world_to_robot_coords(pp["position"]),
                    "rotation": pp["rotation"],
                }
        self._joint_interpolate_to_pose(robot, target_poses, dt, 1.0, dataset, single_task)

        # 阶段7: 检测抓取是否成功
        logging.info("阶段7: 检测抓取是否成功...")
        success = check_success_fn()
        if not success:
            logging.error("抓取失败！")
            return False

        # 阶段8: 松开夹爪
        logging.info(f"阶段8 [{arms_label}]: 松开夹爪...")
        self.gradually_move_gripper(
            robot, {arm: -1.0 for arm in active_arms},
            steps=5, dataset=dataset, single_task=single_task,
        )

        if placed_parts:
            logging.info(f"Stage 9 [{arms_label}]: check placed part height...")
            for placed_part in placed_parts:
                if not self._check_placed_part_z_height(robot, placed_part):
                    logging.error(
                        f"Placed part height check failed, reset scene on retry: "
                        f"{placed_part.get('prim_path', '')}"
                    )
                    return False

        return True

    # =========================================================================
    # Episode 流水线
    # =========================================================================

    def _execute_sequence(
        self, robot, parts: list[dict], box_pos: np.ndarray, dt: float,
        dataset, single_task: str, objects_per_episode: int,
    ) -> bool:
        """Execute Task 3 with configurable arm order."""
        mode = self._normalize_arm_execution_mode()
        self._update_box_centers_from_scene(robot)
        logging.info(f"Foam_Inlaying arm execution mode: {mode}")

        if mode == "dual":
            return self._execute_dual_sequence(
                robot=robot, parts=parts, box_pos=box_pos, dt=dt,
                dataset=dataset, single_task=single_task,
                objects_per_episode=objects_per_episode,
            )

        return self._execute_sequential_sequence(
            robot=robot, parts=parts, box_pos=box_pos, dt=dt,
            dataset=dataset, single_task=single_task,
            objects_per_episode=objects_per_episode, mode=mode,
        )

    def _execute_dual_sequence(
        self, robot, parts: list[dict], box_pos: np.ndarray, dt: float,
        dataset, single_task: str, objects_per_episode: int,
    ) -> bool:
        """双臂 episode：将零件配对后同时执行双臂抓取-放置流水线。"""
        left_parts, right_parts = self._group_parts_by_arm(parts)
        num_pairs = min(len(left_parts), len(right_parts))
        if num_pairs == 0:
            logging.error(
                f"双臂模式下无可配对零件 — left: {len(left_parts)}, right: {len(right_parts)}"
            )
            return False
        for pair_idx in range(num_pairs):
            left_part = left_parts[pair_idx]
            right_part = right_parts[pair_idx]
            logging.info(
                f"\n处理第 {pair_idx + 1}/{num_pairs} 对双臂零件 — "
                f"Left: {left_part.get('prim_path', '?')}, "
                f"Right: {right_part.get('prim_path', '?')}"
            )
            left_part = self._refresh_part_pose(robot, left_part)
            right_part = self._refresh_part_pose(robot, right_part)
            left_grasp = self.compute_grasp_poses(left_part)
            right_grasp = self.compute_grasp_poses(right_part)
            grasp_poses = {}
            if left_grasp:
                grasp_poses.update(left_grasp)
            if right_grasp:
                grasp_poses.update(right_grasp)
            if not grasp_poses:
                continue
            left_place = self.get_place_pose(robot, left_part, box_pos)
            right_place = self.get_place_pose(robot, right_part, box_pos)
            place_poses = {}
            if left_place:
                place_poses.update(left_place)
            if right_place:
                place_poses.update(right_place)
            part_frame_start = self._get_dataset_episode_size(dataset)
            success = self._run_grasp_stages(
                robot=robot, grasp_poses=grasp_poses, place_poses=place_poses,
                dt=dt, dataset=dataset, single_task=single_task,
                placed_parts=[left_part, right_part],
                check_success_fn=lambda: (
                    self.check_grasp_success(robot, left_part)
                    and self.check_grasp_success(robot, right_part)
                ),
            )
            if not success:
                return False
            for arm_side in ("left", "right"):
                if grasp_poses.get(arm_side):
                    self._return_arm_home(
                        robot=robot,
                        arm_side=arm_side,
                        dt=dt,
                        dataset=dataset,
                        single_task=single_task,
                    )
            part_frame_end = self._get_dataset_episode_size(dataset)
            if part_frame_start is not None and part_frame_end is not None:
                for completed_part in (left_part, right_part):
                    self._record_part_frame_range(
                        robot=robot,
                        part=completed_part,
                        frame_start_index=part_frame_start,
                        frame_end_index=part_frame_end - 1,
                    )
        return True

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _execute_sequential_sequence(
        self, robot, parts: list[dict], box_pos: np.ndarray, dt: float,
        dataset, single_task: str, objects_per_episode: int, mode: str,
    ) -> bool:
        """Run one arm at a time while keeping each part assigned to its normal arm."""
        left_parts, right_parts = self._group_parts_by_arm(parts)
        ordered_parts = (
            left_parts + right_parts if mode == "left_then_right" else right_parts + left_parts
        )
        if not ordered_parts:
            logging.error("Sequential arm mode has no parts to process")
            return False

        completed = 0
        total = len(ordered_parts)
        for part_idx, part in enumerate(ordered_parts):
            arm_side = self.get_arm_side(part)
            logging.info(
                f"\nSequential Foam part {part_idx + 1}/{total} [{arm_side}]: "
                f"{part.get('prim_path', '?')}"
            )
            part_frame_start = self._get_dataset_episode_size(dataset)
            success = self._execute_single_part(
                robot=robot, part=part, box_pos=box_pos, dt=dt,
                dataset=dataset, single_task=single_task,
            )
            if not success:
                return False
            completed += 1
            self._return_arm_home(
                robot=robot, arm_side=arm_side, dt=dt,
                dataset=dataset, single_task=single_task,
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
                logging.info(f"Reached objects_per_episode={objects_per_episode}")
                break
        return True

    def _execute_single_part(
        self, robot, part: dict, box_pos: np.ndarray, dt: float,
        dataset, single_task: str,
    ) -> bool:
        """Execute the shared grasp-place stages for exactly one Foam part."""
        part = self._refresh_part_pose(robot, part)
        grasp_poses = self.compute_grasp_poses(part)
        if not grasp_poses:
            return True
        place_poses = self.get_place_pose(robot, part, box_pos)
        return self._run_grasp_stages(
            robot=robot, grasp_poses=grasp_poses, place_poses=place_poses,
            dt=dt, dataset=dataset, single_task=single_task,
            placed_parts=[part],
            check_success_fn=lambda: self.check_grasp_success(robot, part),
        )

    def _refresh_part_pose(self, robot, part: dict) -> dict:
        """Return a copy of part updated with its latest scene pose."""
        prim_path = part.get("prim_path", "")
        if not prim_path:
            return part

        scene_builder = getattr(robot, "_scene_builder", None)
        if scene_builder is None:
            return part

        for current_part in scene_builder.get_parts_world_poses():
            if current_part.get("prim_path") != prim_path:
                continue
            refreshed = dict(part)
            refreshed["position"] = current_part.get("position", part.get("position"))
            refreshed["orientation"] = current_part.get("orientation", part.get("orientation"))
            old_pos = np.array(part.get("position", refreshed["position"]), dtype=np.float64)
            new_pos = np.array(refreshed["position"], dtype=np.float64)
            delta = np.linalg.norm(new_pos - old_pos)
            logging.info(
                f"  -> refreshed pose: {prim_path}, xyz={np.round(new_pos, 4).tolist()}, "
                f"delta={delta:.4f}m"
            )
            return refreshed

        logging.warning(f"Could not refresh pose for part: {prim_path}")
        return part

    def _return_arm_home(
        self,
        robot,
        arm_side: str,
        dt: float,
        dataset,
        single_task: str,
        duration: float = 1.5,
    ) -> None:
        """Move one finished arm back home and record the transition frames."""
        if arm_side not in ("left", "right"):
            return

        logging.info(f"Returning {arm_side} arm home after completing part...")
        if robot._hold_arm_positions is None:
            robot._hold_arm_positions = np.array(
                robot._robot_interface.arm_joint_initial_positions, dtype=np.float32
            )
        if robot._hold_finger_positions is None:
            robot._hold_finger_positions = np.array(
                robot._robot_interface.finger_joint_initial_positions, dtype=np.float32
            )

        target_arm_positions = np.array(robot._hold_arm_positions, dtype=np.float32).copy()
        target_finger_positions = np.array(robot._hold_finger_positions, dtype=np.float32).copy()
        initial_arm_positions = np.array(
            robot._robot_interface.arm_joint_initial_positions, dtype=np.float32
        )
        initial_finger_positions = np.array(
            robot._robot_interface.finger_joint_initial_positions, dtype=np.float32
        )

        if arm_side == "left":
            target_arm_positions[:7] = initial_arm_positions[:7]
            target_finger_positions[:2] = initial_finger_positions[:2]
            robot._left_gripping = False
        else:
            target_arm_positions[7:14] = initial_arm_positions[7:14]
            target_finger_positions[2:4] = initial_finger_positions[2:4]
            robot._right_gripping = False

        target_positions = np.concatenate([target_arm_positions, target_finger_positions])
        arm_finger_indices = (
            robot._robot_interface.arm_joint_indices
            + robot._robot_interface.finger_joint_indices
        )
        joint_states = robot._robot_interface.get_joint_states()
        if not joint_states or "all_positions" not in joint_states:
            logging.warning(f"Cannot return {arm_side} arm home: missing joint states")
            return

        steps = self._get_home_return_steps(robot, dt=dt, duration=duration)
        if hasattr(robot._robot_interface.joint_interpolator, "reset"):
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
            if isinstance(arm_finger_positions, torch.Tensor):
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

        logging.info(f"{arm_side} arm returned home.")

    def _get_home_return_steps(self, robot, dt: float, duration: float) -> int:
        task_cfg = getattr(getattr(robot, "config", None), "task_cfg", {}) or {}
        grasp_cfg = task_cfg.get("grasp", {}) or {}
        if "foam_home_return_steps" in grasp_cfg:
            return max(1, int(grasp_cfg["foam_home_return_steps"]))
        if task_cfg:
            return self._DEFAULT_HOME_RETURN_STEPS
        return max(1, int(duration / dt))

    def _normalize_arm_execution_mode(self) -> str:
        """Normalize CLI-friendly aliases for Task 3 arm execution."""
        raw_mode = getattr(self, "arm_execution_mode", "dual")
        mode = str(raw_mode).strip().lower().replace("-", "_")
        aliases = {
            "both": "dual",
            "bimanual": "dual",
            "dual_arm": "dual",
            "simultaneous": "dual",
            "single": "left_then_right",
            "sequential": "left_then_right",
            "left_right": "left_then_right",
            "right_left": "right_then_left",
        }
        mode = aliases.get(mode, mode)
        if mode not in self._SUPPORTED_ARM_EXECUTION_MODES:
            raise ValueError(
                "Unsupported arm_execution_mode "
                f"'{raw_mode}'. Expected one of {sorted(self._SUPPORTED_ARM_EXECUTION_MODES)}."
            )
        return mode

    def _classify_part_position(self, part_x: float, arm_side: str) -> str:
        """根据零件 X 坐标与所属盒子中心的差值判断靠边位置。

        Args:
            part_x: 零件世界坐标 X 分量。
            arm_side: ``"left"`` 或 ``"right"``，决定使用哪个盒子。

        Returns:
            ``"near_left"``、``"near_right"`` 或 ``"middle"``。
        """
        box_x = self._box_center_x_by_arm.get(
            arm_side,
            self._LEFT_BOX_CENTER_X if arm_side == "left" else self._RIGHT_BOX_CENTER_X,
        )
        box_minus_part_x = box_x - part_x
        if box_minus_part_x > self._EDGE_DISTANCE_THRESHOLD:
            position_type = "near_left"
        elif box_minus_part_x < -self._EDGE_DISTANCE_THRESHOLD:
            position_type = "near_right"
        else:
            position_type = "middle"

        print(
            "[Task3 edge check] "
            f"arm={arm_side}, box_x={box_x:.4f}, part_x={part_x:.4f}, "
            f"box_x-part_x={box_minus_part_x:.4f}, "
            f"threshold=+/-{self._EDGE_DISTANCE_THRESHOLD:.4f}, result={position_type}"
        )
        return position_type

    def _build_arm_poses(self, part: dict, arm_side: str) -> dict:
        """为单臂构建四个阶段的抓取位姿（世界坐标系）。"""
        target_pos = np.array(part["position"])
        prim_path = part.get("prim_path", "")

        # ---- 从随机池中分配唯一孔位 ----
        if prim_path not in self._hole_assignment:
            part_type = get_foam_inlaying_part_type(part, None)
            if part_type == "part_a":
                hole_pos = self._shuffled_small_holes.pop(0) if self._shuffled_small_holes else np.zeros(3)
            else:
                hole_pos = self._shuffled_large_holes.pop(0) if self._shuffled_large_holes else np.zeros(3)
            self._hole_assignment[prim_path] = hole_pos
            logging.info(f"  → [{arm_side}] {part_type} 分配孔位: {np.round(hole_pos, 4)}")
        else:
            hole_pos = self._hole_assignment[prim_path]

        if arm_side == "left":
            approach_rpy = self._LEFT_APPROACH_RPY
        else:
            approach_rpy = self._RIGHT_APPROACH_RPY

        position_type = self._classify_part_position(target_pos[0], arm_side)
        grasp_rpy = self._GRASP_RPY_OPTIONS[arm_side][position_type]
        z_offsets = self._GRASP_Z_OFFSETS[arm_side]
        box_x = self._box_center_x_by_arm.get(
            arm_side,
            self._LEFT_BOX_CENTER_X if arm_side == "left" else self._RIGHT_BOX_CENTER_X,
        )
        logging.info(
            f"  → [{arm_side}] 位置分类: {position_type} "
            f"(box_x - part_x={box_x - target_pos[0]:.4f}, part_x={target_pos[0]:.4f})"
        )

        return {
            "approach": {
                "position": np.array([target_pos[0], target_pos[1], target_pos[2] + 0.45]),
                "rotation": approach_rpy.copy(),
            },
            "descend": {
                "position": np.array([target_pos[0], target_pos[1], target_pos[2] + z_offsets["descend"]]),
                "rotation": grasp_rpy.copy(),
            },
            "grasp": {
                "position": np.array([target_pos[0], target_pos[1], target_pos[2] + z_offsets["grasp"]]),
                "rotation": grasp_rpy.copy(),
            },
            "lift": {
                "position": np.array([target_pos[0], target_pos[1], target_pos[2] + z_offsets["lift"]]),
                "rotation": approach_rpy.copy(),
            },
            "before_place": {
                "position": np.array([hole_pos[0], hole_pos[1], hole_pos[2] + 0.30]),
                "rotation": approach_rpy.copy(),
            },
            }
