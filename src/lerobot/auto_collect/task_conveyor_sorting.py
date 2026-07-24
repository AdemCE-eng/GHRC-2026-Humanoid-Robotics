"""Task 2 —— Conveyor Sorting: 单臂抓取传送带零件并分类入箱。"""

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .auto_collect_base import AutoCollectBase
from .utils import get_conveyor_sorting_part_type


@dataclass(frozen=True)
class _StageMotionConfig:
    target_x: float | None
    target_y: float
    target_z: float
    duration_s: float | None = None


class TaskConveyorSorting(AutoCollectBase):
    """传送带分拣自动数采任务。

    简化设计：
    - 仅使用 right 臂；
    - xyz 从当前零件实时世界坐标生成；
    - rpy 使用每个 episode 内固定抓取姿态；
    - 每个 episode 处理 8 个零件，直到完成或任务 YAML 的 timelimit 超时。
    """

    _TARGET_PARTS_PER_EPISODE = 8
    _DEFAULT_TIMEOUT_S = 10000000000.0
    _DEFAULT_GRAB_ZONE_X_MIN = 0.65
    _DEFAULT_GRAB_ZONE_X_MAX = 1.20
    _WAIT_STEP_S = 0.10
    _DEFAULT_APPROACH_X_LEAD = 0.30
    _DEFAULT_TARGET_Y = -0.1492
    _DEFAULT_TARGET_Z = 0.10
    _DEFAULT_CLOSE_WORLD_X_MIN = 0.81
    _DEFAULT_CLOSE_WORLD_X_MAX = 0.83
    _DEFAULT_POSE_LOG_INTERVAL_S = 10.0
    _DEFAULT_HOME_RETURN_STEPS = 2
    _DEFAULT_HOME_LIFT_Z_OFFSET = 0.30
    _DEFAULT_HOME_LIFT_DURATION_S = 0.4
    _DEFAULT_STAGE2_DURATION_S = 0.5
    _DEFAULT_LEFT_BIN_X_RANGE = (0.39616, 0.57616)
    _DEFAULT_LEFT_BIN_Y_RANGE = (-0.202, 0.078)
    _DEFAULT_RIGHT_BIN_X_RANGE = (0.79616, 0.97616)
    _DEFAULT_RIGHT_BIN_Y_RANGE = (-0.202, 0.078)
    _LOG_COLORS = {
        "reset": "\033[0m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "magenta": "\033[95m",
        "blue": "\033[94m",
        "gray": "\033[90m",
    }

    def __init__(self):
        super().__init__()
        self._fixed_grasp_rpy = np.array([-3.0671, -0.4860, -2.1871], dtype=np.float32)
        self._fixed_place_rpy_part_a = self._fixed_grasp_rpy.copy()
        self._fixed_place_rpy_part_b = np.array([3.0965, -0.1994, -1.7034], dtype=np.float32)
        self._processed_paths: set[str] = set()
        self._last_pose_log_time = 0.0
        self._current_episode_part_frames: list[dict] = []

    # ------------------------------------------------------------------
    # Task-specific pose APIs
    # ------------------------------------------------------------------

    def compute_grasp_poses(self, part: dict) -> dict:
        """根据零件实时世界坐标生成接近 / 抓取 / 抬起位姿。"""
        target_pos = np.array(part["position"], dtype=np.float32)
        rpy = self._fixed_grasp_rpy.copy()

        return {
            "right": {
                "approach": {
                    "position": np.array(
                        [target_pos[0], target_pos[1], target_pos[2] + 0.22],
                        dtype=np.float32,
                    ),
                    "rotation": rpy.copy(),
                },
                "grasp": {
                    "position": np.array(
                        [target_pos[0], target_pos[1], target_pos[2] - 0.02],
                        dtype=np.float32,
                    ),
                    "rotation": rpy.copy(),
                },
                "lift": {
                    "position": np.array(
                        [target_pos[0], target_pos[1], target_pos[2] + 0.16],
                        dtype=np.float32,
                    ),
                    "rotation": rpy.copy(),
                },
            },
        }

    def check_grasp_success(self, robot, part: dict, threshold: float = 0.10) -> bool:
        """通过 right 夹爪与本次目标零件位置的距离判断抓取是否成功。"""
        ee_poses = robot._robot_interface.get_ee_poses()
        if ee_poses is None or ee_poses.get("right") is None:
            # logging.warning("Conveyor 抓取检测: 无法获取 right 末端位姿")
            return False

        gripper_pos = np.array(ee_poses["right"][:3], dtype=float)
        current_part = self._find_current_part(robot, part)
        part_pos_world = np.array(current_part["position"], dtype=float)
        part_pos_base = np.array(
            robot._scene_builder.world_to_robot_coords(part_pos_world),
            dtype=float,
        )
        distance = np.linalg.norm(gripper_pos - part_pos_base)

        # logging.info(
        #     f"Conveyor 抓取检测 [right]: 零件={part_pos_base}, 夹爪={gripper_pos}, "
        #     f"距离={distance:.4f}m"
        # )
        return distance <= threshold

    def get_place_pose(self, robot, part: dict, box_pos: np.ndarray) -> dict:
        """根据零件类型返回对应箱子的固定投放位姿。"""
        boxes = self._get_box_positions(robot)
        part_type = get_conveyor_sorting_part_type(part, robot._scene_builder)

        # Conveyor_Sorting.yaml 中 box_position[0] 在左侧，box_position[1] 在右侧。
        left_box = boxes[0]
        right_box = boxes[1] if len(boxes) > 1 else box_pos
        target_box = left_box if part_type == "part_a" else right_box

        # logging.info(
        #     f"  → Conveyor {part_type}: "
        #     f"{'左侧' if part_type == 'part_a' else '右侧'}箱 {np.round(target_box, 4)}"
        # )

        return {
            "right": {
                "position": np.array(
                    [target_box[0], target_box[1], target_box[2] + 0.30],
                    dtype=np.float32,
                ),
                "rotation": self._get_place_rpy_for_part(robot, part),
            },
        }

    # ------------------------------------------------------------------
    # Episode pipeline
    # ------------------------------------------------------------------

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
        """单臂实时分拣：处理 8 个零件，直到完成或超时。"""
        del parts

        self._processed_paths.clear()
        self._last_pose_log_time = 0.0
        timeout_s = self._get_timeout_s(robot)
        deadline = time.perf_counter() + timeout_s
        requested_count = int(objects_per_episode)
        target_count = self._TARGET_PARTS_PER_EPISODE
        if requested_count > 0:
            target_count = min(requested_count, self._TARGET_PARTS_PER_EPISODE)
        completed = 0
        recording_started = dataset is None

        # logging.info(
        #     f"Conveyor_Sorting episode 开始: 目标 {target_count} 个零件, "
        #     f"超时 {timeout_s:.1f}s, "
        #     f"抓取 RPY={np.round(self._fixed_grasp_rpy, 4)}, "
        #     f"part_a 投放 RPY={np.round(self._fixed_place_rpy_part_a, 4)}, "
        #     f"part_b 投放 RPY={np.round(self._fixed_place_rpy_part_b, 4)}"
        # )

        while completed < target_count:
            if time.perf_counter() >= deadline:
                # logging.error(
                #     f"Conveyor_Sorting 超时: 已完成 {completed}/{target_count}"
                # )
                return False

            live_parts = self._get_live_parts(robot)
            # self._log_runtime_poses(robot, live_parts=live_parts)
            if not recording_started:
                start_part = self._get_record_start_part(robot, live_parts)
                if start_part is None:
                    self._wait_for_parts(robot, dt, dataset=None, single_task=single_task)
                    continue
                recording_started = True
                # self._log_conveyor_event(
                #     logging.INFO,
                #     "recording_start",
                #     "green",
                #     **self._part_log_fields(start_part, robot._scene_builder),
                #     start_x=self._get_record_start_x(robot),
                # )

            part = self._select_next_part(robot, live_parts)
            if part is None:
                self._wait_for_parts(robot, dt, dataset=None, single_task=single_task)
                continue

            part_key = self._part_key(part)
            # self._log_conveyor_event(
            #     logging.INFO,
            #     "process_part",
            #     "cyan",
            #     **self._part_log_fields(part, robot._scene_builder),
            #     completed=completed,
            #     target_count=target_count,
            #     live_count=len(live_parts),
            # )

            grasp_poses = self.compute_grasp_poses(part)
            place_poses = self.get_place_pose(robot, part, box_pos)
            part_frame_start = self._get_dataset_episode_size(dataset)
            success = self._run_grasp_stages(
                robot=robot,
                part=part,
                grasp_poses=grasp_poses,
                place_poses=place_poses,
                dt=dt,
                dataset=dataset,
                single_task=single_task,
                check_success_fn=lambda p=part: self.check_grasp_success(robot, p),
            )
            if not success:
                # self._log_conveyor_event(
                #     logging.ERROR,
                #     "part_failed",
                #     "red",
                #     **self._part_log_fields(part, robot._scene_builder),
                #     completed=completed,
                #     target_count=target_count,
                # )
                return False

            part_frame_end = self._get_dataset_episode_size(dataset)
            if part_frame_start is not None and part_frame_end is not None:
                self._record_part_frame_range(
                    robot=robot,
                    part=part,
                    frame_start_index=part_frame_start,
                    frame_end_index=part_frame_end - 1,
                )
            self._processed_paths.add(part_key)
            completed += 1
            # self._log_conveyor_event(
            #     logging.INFO,
            #     "part_completed",
            #     "green",
            #     **self._part_log_fields(part, robot._scene_builder),
            #     completed=completed,
            #     target_count=target_count,
            #     processed_count=len(self._processed_paths),
            # )
            conveyor_restore_speed = self._pause_conveyor_for_home_return(robot)
            try:
                self._return_right_arm_home(
                    robot=robot,
                    dt=dt,
                    dataset=dataset,
                    single_task=single_task,
                )
            finally:
                self._restore_conveyor_after_home_return(robot, conveyor_restore_speed)

        return self._check_processed_parts_in_correct_bins(robot, target_count)

    def _load_part_info(self, part_info_path: Path) -> dict:
        if not part_info_path.exists():
            return {"episodes": []}

        try:
            loaded = json.loads(part_info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("Conveyor part_info.json 无法解析，将重建文件: %s", part_info_path)
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
                "part_type": get_conveyor_sorting_part_type(part, scene_builder),
                "prim_path": str(part.get("prim_path", "")),
                "frame_start_index": int(frame_start_index),
                "frame_end_index": int(frame_end_index),
            }
        )

    def _check_processed_parts_in_correct_bins(self, robot, target_count: int) -> bool:
        live_parts = self._get_live_parts(robot)
        checked_count = 0

        for part in live_parts:
            part_key = self._part_key(part)
            if part_key not in self._processed_paths:
                continue

            checked_count += 1
            position = np.asarray(part.get("position", []), dtype=float)
            if position.shape[0] < 2:
                logging.warning("Conveyor 最终入箱检查失败: %s 缺少有效位置", part_key)
                return False

            part_type = get_conveyor_sorting_part_type(part, robot._scene_builder)
            target_side = "left" if part_type == "part_a" else "right"
            x_range, y_range = self._get_bin_xy_ranges(robot, target_side)
            if not self._is_position_inside_xy_ranges(position[:2], x_range, y_range):
                logging.warning(
                    "Conveyor 最终入箱检查失败: %s type=%s pos_xy=%s target_side=%s "
                    "x_range=%s y_range=%s",
                    part_key,
                    part_type,
                    np.round(position[:2], 4).tolist(),
                    target_side,
                    np.round(x_range, 4).tolist(),
                    np.round(y_range, 4).tolist(),
                )
                return False

        if checked_count < target_count:
            logging.warning(
                "Conveyor 最终入箱检查失败: 已检查 %s/%s 个已处理零件",
                checked_count,
                target_count,
            )
            return False

        return True

    def _is_position_inside_xy_ranges(
        self,
        position: np.ndarray,
        x_range: np.ndarray,
        y_range: np.ndarray,
    ) -> bool:
        pos = np.asarray(position, dtype=float)
        return bool(
            float(x_range[0]) <= float(pos[0]) <= float(x_range[1])
            and float(y_range[0]) <= float(pos[1]) <= float(y_range[1])
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
        return self._part_key(part)

    def _run_grasp_stages(
        self,
        robot,
        part: dict,
        grasp_poses: dict,
        place_poses: dict,
        dt: float,
        dataset,
        single_task: str,
        check_success_fn,
    ) -> bool:
        """执行 Conveyor 的单臂抓取-投放流水线。"""
        active_arms = [arm for arm in ("right",) if grasp_poses.get(arm)]
        if not active_arms:
            return False

        logging.info("阶段1 [right]: 松开夹爪...")
        # self._log_runtime_poses(robot)
        self.move_gripper(
            robot,
            {"right": -1.0},
            dt,
            0.2,
            dataset=dataset,
            single_task=single_task,
        )

        logging.info("阶段2 [right]: 移动到零件 x 方向提前位置上方...")
        # self._log_runtime_poses(robot)
        stage2_cfg = self._get_stage_motion_cfg(
            robot,
            "stage2",
            default_duration_s=self._DEFAULT_STAGE2_DURATION_S,
        )
        approach_pose = self._build_conveyor_stage_target(
            robot,
            part,
            stage="stage2",
            x_lead=self._get_approach_x_lead(robot),
        )
        approach_x, approach_y, approach_z = approach_pose["position"]
        self._move_to_world_pose(
            robot,
            x=approach_x,
            y=approach_y,
            z=approach_z,
            rotation=approach_pose["rotation"],
            dt=dt,
            duration=stage2_cfg.duration_s,
            dataset=dataset,
            single_task=single_task,
        )

        logging.info("阶段3 [right]: 移动到零件 x 方向提前位置...")
        # self._log_runtime_poses(robot)
        grasp_pose = self._build_conveyor_stage_target(
            robot,
            part,
            stage="stage3",
            fixed_x=approach_x,
        )
        grasp_x, grasp_y, grasp_z = grasp_pose["position"]
        self._move_to_world_pose(
            robot,
            x=grasp_x,
            y=grasp_y,
            z=grasp_z,
            rotation=grasp_pose["rotation"],
            dt=dt,
            duration=0.1,
            dataset=dataset,
            single_task=single_task,
        )

        logging.info("阶段4b [right]: 保持姿态，等待零件进入闭爪阈值...")
        # self._log_runtime_poses(robot)
        # self._log_conveyor_event(
        #     logging.INFO,
        #     "close_wait_start",
        #     "yellow",
        #     **self._part_log_fields(part, robot._scene_builder),
        # )
        closed = self._wait_and_close_gripper_when_ready(
            robot,
            part,
            dt=dt,
            duration=5.0,
            dataset=dataset,
            single_task=single_task,
        )
        if not closed:
            current_part = self._find_current_part(robot, part)
            current_x = float(current_part.get("position", [float("inf")])[0])
            _, x_max = self._get_grab_zone(robot)
            if current_x > x_max:
                # logging.error(
                #     f"Conveyor 零件已离开抓取区，放弃闭爪: x={current_x:.4f}, x_max={x_max:.4f}"
                # )
                return False

            # logging.warning("Conveyor 未在阈值内闭爪，仍在抓取区内，执行兜底闭爪")
            self.move_gripper(
                robot,
                {"right": 0.3},
                dt,
                0.2,
                dataset=dataset,
                single_task=single_task,
            )

        logging.info("阶段4 [right]: 抬起...")
        # self._log_runtime_poses(robot)
        live_part = self._find_current_part(robot, part)
        lift_pose = self._build_conveyor_stage_target(
            robot,
            live_part,
            stage="stage4",
            fixed_x=approach_x,
        )

        lift_x, lift_y, lift_z = lift_pose["position"]
        self._move_to_world_pose(
            robot,
            x=lift_x,
            y=lift_y,
            z=lift_z,
            rotation=lift_pose["rotation"],
            dt=dt,
            duration=0.2,
            dataset=dataset,
            single_task=single_task,
        )



        logging.info("阶段5 [right]: 检测抓取是否成功...")
        if not check_success_fn():
            self._release_right_gripper_after_failure(
                robot,
                dt=dt,
                dataset=dataset,
                single_task=single_task,
            )
            return False

        place_rpy = self._get_place_rpy_for_part(robot, part)
        # logging.info(
        #     f"阶段6 [right]: 根据零件类型选择投放 RPY={np.round(place_rpy, 4)}"
        # )
        place_target = {
            "right": {
                "position": robot._scene_builder.world_to_robot_coords(
                    place_poses["right"]["position"]
                ),
                "rotation": place_rpy,
            }
        }
        stage6_prepare_z = self._get_stage6_prepare_target_z(robot)
        logging.info("阶段6 [right]: 移动到投放位置...")
        # self._log_runtime_poses(robot)
        self._joint_interpolate_to_pose(
            robot,
            place_target,
            dt,
            0.8,
            dataset=dataset,
            single_task=single_task,
        )
        if stage6_prepare_z is not None:
            logging.info("阶段6预定位 [right]: 移动到投放位置 XY / 配置 Z...")
            prepare_target = {
                "right": {
                    "position": np.asarray(
                        place_target["right"]["position"],
                        dtype=np.float32,
                    ).copy(),
                    "rotation": place_rpy,
                }
            }
            prepare_target["right"]["position"][2] = stage6_prepare_z
            self._joint_interpolate_to_pose(
                robot,
                prepare_target,
                dt,
                0.05,
                dataset=dataset,
                single_task=single_task,
            )

        logging.info("阶段7 [right]: 松开夹爪...")
        self.move_gripper(
            robot,
            {"right": -1.0},
            dt,
            0.2,
            dataset=dataset,
            single_task=single_task,
        )
        return True

    def _release_right_gripper_after_failure(
        self,
        robot,
        dt: float,
        dataset,
        single_task: str,
    ) -> None:
        if not getattr(robot, "_right_gripping", False):
            return

        # logging.info("Conveyor 抓取失败，重置前松开 right 夹爪...")
        self.move_gripper(
            robot,
            {"right": -1.0},
            dt,
            0.3,
            dataset=dataset,
            single_task=single_task,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_episode_parts(self, robot) -> list[dict]:
        """让基类 episode 判空也使用 Conveyor 的实时零件查询兜底。"""
        return self._get_live_parts(robot)

    def _find_current_part(self, robot, target_part: dict) -> dict:
        target_key = self._part_key(target_part)
        for part in self._get_live_parts(robot):
            if self._part_key(part) == target_key:
                return part
        return target_part

    def _build_conveyor_stage_target(
        self,
        robot,
        part: dict,
        stage: str,
        x_lead: float = 0.0,
        fixed_x: float | None = None,
    ) -> dict:
        """从零件位姿生成基座坐标目标，并按阶段配置覆盖 xyz。"""
        target_pos = np.asarray(part["position"], dtype=np.float32).copy()
        target_pos[0] += float(x_lead)
        position = np.asarray(
            robot._scene_builder.world_to_robot_coords(target_pos),
            dtype=np.float32,
        )
        if fixed_x is not None:
            position[0] = float(fixed_x)
        stage_cfg = self._get_stage_motion_cfg(robot, stage)
        if stage_cfg.target_x is not None:
            position[0] = stage_cfg.target_x
        position[1] = stage_cfg.target_y
        position[2] = stage_cfg.target_z
        return {
            "position": position,
            "rotation": self._fixed_grasp_rpy.copy(),
        }

    def _move_to_world_pose(
        self,
        robot,
        x: float,
        y: float,
        z: float,
        rotation: np.ndarray,
        dt: float,
        duration: float,
        dataset,
        single_task: str,
    ) -> None:
        target_pose = {
            "right": {
                "position": np.array([x, y, z], dtype=np.float32),
                "rotation": np.asarray(rotation, dtype=np.float32),
            }
        }
        self._joint_interpolate_to_pose(
            robot,
            target_pose,
            dt,
            duration,
            dataset=dataset,
            single_task=single_task,
        )

    def _wait_and_close_gripper_when_ready(
        self,
        robot,
        part: dict,
        dt: float,
        duration: float,
        dataset,
        single_task: str,
    ) -> bool:
        """保持当前姿态，只根据实时零件 x 误差判断是否闭爪。"""
        steps = max(1, int(duration / max(dt, 1e-6)))
        for _ in range(steps):
            self._raise_if_keyboard_requested()
            # self._log_runtime_poses(robot)
            robot.step(render=True)
            # self._log_runtime_poses(robot)
            if dataset is not None:
                self._record_frame(robot, dataset, single_task)

            if self._should_close_gripper(robot, part):
                live_part = self._find_current_part(robot, part)
                # self._log_conveyor_event(
                #     logging.INFO,
                #     "close_ready",
                #     "green",
                #     **self._part_log_fields(live_part, robot._scene_builder),
                # )
                # logging.info("Conveyor 闭爪阈值达标，闭合 right 夹爪")
                self.move_gripper(
                    robot,
                    {"right": 1.0},
                    dt,
                    0.1,
                    dataset=dataset,
                    single_task=single_task,
                )
                return True

        return False

    def _should_close_gripper(self, robot, part: dict) -> bool:
        """实时零件世界坐标 x 进入配置区间后闭爪。"""
        live_part = self._find_current_part(robot, part)
        position = np.asarray(live_part.get("position", []), dtype=float)
        if position.shape[0] < 1:
            return False

        x_min, x_max = self._get_close_world_x_range(robot)
        part_world_x = float(position[0])
        return x_min <= part_world_x <= x_max

    def _select_next_part(self, robot, parts: list[dict]) -> dict | None:
        x_min, x_max = self._get_grab_zone(robot)
        candidates = []
        for part in parts:
            if self._part_key(part) in self._processed_paths:
                # self._log_conveyor_event(
                #     logging.INFO,
                #     "skip_processed_part",
                #     "gray",
                #     **self._part_log_fields(part, robot._scene_builder),
                #     processed_count=len(self._processed_paths),
                # )
                continue
            pos = np.array(part.get("position", []), dtype=float)
            if pos.shape[0] < 3:
                continue
            part_type = get_conveyor_sorting_part_type(part, robot._scene_builder)
            adjusted_x_min = x_min - 0.05 if part_type == "part_a" else x_min
            if adjusted_x_min <= pos[0] <= x_max:
                candidates.append(part)
                # self._log_conveyor_event(
                #     logging.DEBUG,
                #     "candidate_part",
                #     "blue",
                #     **self._part_log_fields(part, robot._scene_builder),
                #     x_min=adjusted_x_min,
                #     x_max=x_max,
                # )

        if not candidates:
            return None

        selected = max(candidates, key=lambda p: float(p["position"][0]))
        # self._log_conveyor_event(
        #     logging.INFO,
        #     "selected_part",
        #     "cyan",
        #     **self._part_log_fields(selected, robot._scene_builder),
        #     candidate_count=len(candidates),
        # )
        return selected

    def _get_live_parts(self, robot) -> list[dict]:
        try:
            parts = robot._scene_builder.get_parts_world_poses()
            if parts:
                return [
                    {**part, "index": idx}
                    for idx, part in enumerate(parts)
                ]
        except Exception as exc:
            # logging.warning(f"Conveyor 查询 get_parts_world_poses 失败: {exc}")
            pass

        return self._get_rigid_prim_parts(robot._scene_builder)

    def _get_rigid_prim_parts(self, scene_builder) -> list[dict]:
        rigid_prim = getattr(scene_builder, "rigid_prim", None)
        if rigid_prim is None or not hasattr(rigid_prim, "get_world_poses"):
            return []

        try:
            positions, orientations = rigid_prim.get_world_poses()
        except Exception as exc:
            logging.warning("Conveyor 查询 rigid_prim 位姿失败: %s", exc)
            return []

        paths = self._get_rigid_prim_paths(scene_builder, len(positions))
        parts = []
        for idx, pos in enumerate(np.asarray(positions)):
            orientation = [0.0, 0.0, 0.0, 1.0]
            if orientations is not None and idx < len(orientations):
                w, x, y, z = np.asarray(orientations[idx], dtype=float).tolist()
                orientation = [x, y, z, w]
            parts.append(
                {
                    "prim_path": paths[idx],
                    "position": np.asarray(pos, dtype=float).tolist(),
                    "orientation": orientation,
                    "index": idx,
                }
            )
        return parts

    def _get_rigid_prim_paths(self, scene_builder, count: int) -> list[str]:
        paths = list(getattr(scene_builder, "parts_prim_paths", []) or [])
        if len(paths) >= count:
            return paths[:count]

        rigid_prim = getattr(scene_builder, "rigid_prim", None)
        for attr in ("prim_paths", "_prim_paths"):
            candidate = getattr(rigid_prim, attr, None)
            if candidate and len(candidate) >= count:
                return [str(p) for p in candidate[:count]]

        return [
            f"/Root/Part_{'A' if idx % 2 == 0 else 'B'}{idx // 2}"
            for idx in range(count)
        ]

    def _wait_for_parts(self, robot, dt: float, dataset, single_task: str) -> None:
        steps = max(1, int(self._WAIT_STEP_S / dt))
        for _ in range(steps):
            self._raise_if_keyboard_requested()
            robot.step(render=True)
            # self._log_runtime_poses(robot)
            if dataset is not None:
                self._record_frame(robot, dataset, single_task)

    def _get_record_start_part(self, robot, parts: list[dict]) -> dict | None:
        """Return the leading unprocessed part once world-x reaches the record threshold."""
        start_x = self._get_record_start_x(robot)
        candidates = []
        for part in parts:
            if self._part_key(part) in self._processed_paths:
                continue
            pos = np.asarray(part.get("position", []), dtype=float)
            if pos.shape[0] < 3:
                continue
            candidates.append(part)

        if not candidates:
            return None

        leading_part = max(candidates, key=lambda p: float(p["position"][0]))
        if float(leading_part["position"][0]) >= start_x:
            return leading_part
        return None

    def _return_right_arm_home(
        self,
        robot,
        dt: float,
        dataset,
        single_task: str,
    ) -> None:
        """Move the right arm/fingers back to initial joint positions without recording."""
        del dataset, single_task
        steps = self._get_home_return_steps(robot)
        # logging.info(f"Conveyor: returning right arm home with {steps} unrecorded steps...")

        self._lift_right_arm_before_home(robot, dt)

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
            # logging.warning("Conveyor: cannot return right arm home, missing joint states")
            return

        target_positions = np.concatenate([target_arm_positions, target_finger_positions])
        arm_finger_indices = (
            robot._robot_interface.arm_joint_indices
            + robot._robot_interface.finger_joint_indices
        )
        import torch

        robot._robot_interface.joint_interpolator.set_target(
            start_q=torch.tensor(joint_states["all_positions"])[arm_finger_indices],
            target_q=torch.tensor(target_positions),
            num_steps=steps,
        )

        for _ in range(steps):
            self._raise_if_keyboard_requested()
            arm_finger_positions = robot._robot_interface.joint_interpolator.step()
            if hasattr(arm_finger_positions, "detach"):
                arm_finger_positions = arm_finger_positions.detach().cpu().numpy()
            else:
                arm_finger_positions = np.asarray(arm_finger_positions, dtype=np.float32)

            robot._hold_arm_positions = arm_finger_positions[:14]
            robot._hold_finger_positions = arm_finger_positions[14:18]
            robot.step(render=True)
        # 为了确保完全回到初始位姿，额外执行几步渲染（不记录）来稳定末端状态。
        for _ in range(10):
            self._raise_if_keyboard_requested()
            robot.step(render=True)
        # logging.info("Conveyor: right arm returned home.")

    def _lift_right_arm_before_home(self, robot, dt: float) -> None:
        ee_poses = robot._robot_interface.get_ee_poses()
        if ee_poses is None or ee_poses.get("right") is None:
            return
        if not hasattr(self, "_joint_interpolate_to_pose"):
            return

        right_pose = np.asarray(ee_poses["right"], dtype=np.float32)
        if right_pose.shape[0] < 6:
            return

        lifted_position = right_pose[:3].copy()
        lifted_position[2] += self._DEFAULT_HOME_LIFT_Z_OFFSET
        target_pose = {
            "right": {
                "position": lifted_position,
                "rotation": right_pose[3:6].copy(),
            }
        }
        self._joint_interpolate_to_pose(
            robot,
            target_pose,
            dt,
            self._DEFAULT_HOME_LIFT_DURATION_S,
            dataset=None,
            single_task=None,
        )

    def _pause_conveyor_for_home_return(self, robot) -> list[float] | None:
        restore_speed = self._get_conveyor_configured_speed(robot)
        self._set_conveyor_speed(robot, [0.0, 0.0, 0.0])
        return restore_speed

    def _restore_conveyor_after_home_return(
        self,
        robot,
        restore_speed: list[float] | None,
    ) -> None:
        if restore_speed is None:
            return
        self._set_conveyor_speed(robot, restore_speed)

    def _get_conveyor_configured_speed(self, robot) -> list[float] | None:
        scene_builder = getattr(robot, "_scene_builder", None)
        if scene_builder is None:
            return None

        conveyor_cfg = getattr(scene_builder, "ConveyorBelt_cfg", None)
        if conveyor_cfg is None:
            cfg = getattr(scene_builder, "cfg", {}) or {}
            conveyor_cfg = cfg.get("ConveyorBelt", {})

        speed = (conveyor_cfg or {}).get("ConveyorBelt_speed")
        if speed is None:
            return None
        return list(speed)

    def _set_conveyor_speed(self, robot, speed: list[float]) -> None:
        scene_builder = getattr(robot, "_scene_builder", None)
        set_conveyor_speed = getattr(scene_builder, "set_conveyor_speed", None)
        if callable(set_conveyor_speed):
            set_conveyor_speed(speed)

    def _log_runtime_poses(
        self,
        robot,
        live_parts: list[dict] | None = None,
        force: bool = False,
    ) -> None:
        """每隔固定时间向终端格式化打印 right 末端和抓取区零件位置。"""
        now = time.perf_counter()
        interval_s = self._get_pose_log_interval_s(robot)
        if not force and self._last_pose_log_time > 0.0:
            if now - self._last_pose_log_time < interval_s:
                return

        self._last_pose_log_time = now

        try:
            ee_poses = robot._robot_interface.get_ee_poses()
        except Exception as exc:
            ee_poses = {"right": f"<get_ee_poses failed: {exc}>"}

        if live_parts is None:
            live_parts = self._get_live_parts(robot)

        x_min, x_max = self._get_grab_zone(robot)
        parts_in_grab_zone = []
        for part in live_parts or []:
            position = np.asarray(part.get("position", []), dtype=float)
            if position.shape[0] < 3:
                continue
            if x_min <= float(position[0]) <= x_max:
                parts_in_grab_zone.append(position[:3])

        right_pose = None
        if isinstance(ee_poses, dict):
            right_pose = ee_poses.get("right")

        if isinstance(right_pose, (list, tuple, np.ndarray)) and len(right_pose) >= 6:
            right_pose = np.asarray(right_pose, dtype=float)
            ee_lines = [
                "  当前控制位姿 right:",
                f"    position: x={right_pose[0]:.5f}, y={right_pose[1]:.5f}, z={right_pose[2]:.5f}",
                f"    rpy:      roll={right_pose[3]:.5f}, pitch={right_pose[4]:.5f}, yaw={right_pose[5]:.5f}",
            ]
        else:
            ee_lines = [
                "  当前控制位姿 right:",
                f"    unavailable: {right_pose}",
            ]

        part_lines = [
            f"  抓取区零件位置 x_range=[{x_min:.5f}, {x_max:.5f}], count={len(parts_in_grab_zone)}:"
        ]
        if parts_in_grab_zone:
            for idx, position in enumerate(parts_in_grab_zone, start=1):
                part_lines.append(
                    f"    part_{idx}: x={position[0]:.5f}, y={position[1]:.5f}, z={position[2]:.5f}"
                )
        else:
            part_lines.append("    none")

        message = "\n".join(["[Conveyor runtime pose log]", *ee_lines, *part_lines])
        logging.info(message)
        print(message, flush=True)

    def _part_log_fields(self, part: dict, scene_builder=None) -> dict:
        position = np.asarray(part.get("position", []), dtype=float)
        if position.shape[0] >= 3:
            world_pos = position[:3]
        else:
            world_pos = None
        part_type = part.get("type")
        if part_type is None and scene_builder is not None:
            part_type = get_conveyor_sorting_part_type(part, scene_builder)

        return {
            "part_key": self._part_key(part),
            "prim_path": part.get("prim_path", "<missing>"),
            "part_type": part_type or "<unknown>",
            "index": part.get("index", "<missing>"),
            "world_pos": world_pos,
        }

    def _log_conveyor_event(
        self,
        level: int,
        event: str,
        color: str = "reset",
        **fields,
    ) -> None:
        color_code = self._LOG_COLORS.get(color, "")
        reset_code = self._LOG_COLORS["reset"] if color_code else ""
        field_text = " ".join(
            f"{key}={self._format_log_value(value)}"
            for key, value in fields.items()
        )
        message = f"{color_code}[Conveyor][{event}] {field_text}{reset_code}"
        # logging.log(level, message)
        # print(message, flush=True)

    @staticmethod
    def _format_log_value(value) -> str:
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                return "[" + ", ".join(f"{float(item):.4f}" for item in value.tolist()) + "]"
            return np.array2string(value, precision=4, suppress_small=False)
        if isinstance(value, (list, tuple)):
            try:
                array_value = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                return str(value)
            if array_value.ndim == 1:
                return "[" + ", ".join(f"{float(item):.4f}" for item in array_value.tolist()) + "]"
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, np.floating):
            return f"{float(value):.4f}"
        if value is None:
            return "None"
        return str(value)

    def _get_box_positions(self, robot) -> list[np.ndarray]:
        if robot._scene_builder is not None:
            boxes = robot._scene_builder.get_box_positions()
            if boxes:
                return [np.asarray(box, dtype=float) for box in boxes]
        return [
            np.array([0.48616, -0.062, 0.92], dtype=float),
            np.array([0.88616, -0.062, 0.92], dtype=float),
        ]

    def _get_grab_zone(self, robot) -> tuple[float, float]:
        grasp_cfg = self._get_grasp_cfg(robot)
        return (
            float(grasp_cfg.get("conveyor_grab_zone_x_min", self._DEFAULT_GRAB_ZONE_X_MIN)),
            float(grasp_cfg.get("conveyor_grab_zone_x_max", self._DEFAULT_GRAB_ZONE_X_MAX)),
        )

    def _get_record_start_x(self, robot) -> float:
        grasp_cfg = self._get_grasp_cfg(robot)
        return float(
            grasp_cfg.get(
                "conveyor_record_start_x",
                grasp_cfg.get("conveyor_start_x", self._DEFAULT_GRAB_ZONE_X_MIN),
            )
        )

    def _get_resolved_world_x(self, robot) -> float:
        grasp_cfg = self._get_grasp_cfg(robot)
        return float(
            grasp_cfg.get(
                "conveyor_resolved_world_x",
                self._DEFAULT_CLOSE_WORLD_X_MAX,
            )
        )

    def _get_home_return_steps(self, robot) -> int:
        grasp_cfg = self._get_grasp_cfg(robot)
        return max(
            1,
            int(grasp_cfg.get("conveyor_home_return_steps", self._DEFAULT_HOME_RETURN_STEPS)),
        )

    def _get_bin_xy_ranges(self, robot, side: str) -> tuple[np.ndarray, np.ndarray]:
        default_x = (
            self._DEFAULT_LEFT_BIN_X_RANGE
            if side == "left"
            else self._DEFAULT_RIGHT_BIN_X_RANGE
        )
        default_y = (
            self._DEFAULT_LEFT_BIN_Y_RANGE
            if side == "left"
            else self._DEFAULT_RIGHT_BIN_Y_RANGE
        )
        return (
            self._get_float_range(robot, f"conveyor_{side}_bin_x_range", default_x),
            self._get_float_range(robot, f"conveyor_{side}_bin_y_range", default_y),
        )

    def _get_float_range(
        self,
        robot,
        cfg_key: str,
        default_value: tuple[float, float],
    ) -> np.ndarray:
        grasp_cfg = self._get_grasp_cfg(robot)
        value = grasp_cfg.get(cfg_key, default_value)
        float_range = np.asarray(value, dtype=float)
        if float_range.shape != (2,):
            logging.warning(
                "Conveyor 忽略无效 %s=%s，使用默认值 %s",
                cfg_key,
                value,
                default_value,
            )
            return np.asarray(default_value, dtype=float)
        return float_range

    def _get_grasp_cfg(self, robot) -> dict:
        return (getattr(robot.config, "task_cfg", {}) or {}).get("grasp", {})

    def _get_approach_x_lead(self, robot) -> float:
        grasp_cfg = self._get_grasp_cfg(robot)
        return float(grasp_cfg.get("conveyor_x_lead", self._DEFAULT_APPROACH_X_LEAD))

    def _get_stage_motion_cfg(
        self,
        robot,
        stage: str,
        default_duration_s: float | None = None,
    ) -> _StageMotionConfig:
        grasp_cfg = self._get_grasp_cfg(robot)
        target_x = grasp_cfg.get(f"conveyor_{stage}_target_x")
        target_y = grasp_cfg.get(f"conveyor_{stage}_target_y")
        target_z = grasp_cfg.get(f"conveyor_{stage}_target_z")
        duration_s = grasp_cfg.get(f"conveyor_{stage}_duration", default_duration_s)
        return _StageMotionConfig(
            target_x=self._parse_optional_float(target_x),
            target_y=float(self._DEFAULT_TARGET_Y if target_y is None else target_y),
            target_z=float(self._DEFAULT_TARGET_Z if target_z is None else target_z),
            duration_s=None if duration_s is None else float(duration_s),
        )

    def _parse_optional_float(self, value) -> float | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return float(value)

    def _get_stage6_prepare_target_z(self, robot) -> float | None:
        grasp_cfg = self._get_grasp_cfg(robot)
        return self._parse_optional_float(grasp_cfg.get("conveyor_stage6_prepare_target_z"))

    def _get_timeout_s(self, robot) -> float:
        task_cfg = getattr(robot.config, "task_cfg", {}) or {}
        timeout_s = task_cfg.get("timelimit")
        return float(self._DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s)

    def _get_close_world_x_range(self, robot) -> tuple[float, float]:
        grasp_cfg = self._get_grasp_cfg(robot)
        x_min = float(
            grasp_cfg.get(
                "conveyor_close_world_x_min",
                self._DEFAULT_CLOSE_WORLD_X_MIN,
            )
        )
        x_max = float(
            grasp_cfg.get(
                "conveyor_close_world_x_max",
                self._DEFAULT_CLOSE_WORLD_X_MAX,
            )
        )
        return x_min, x_max

    def _get_place_rpy_for_part(self, robot, part: dict) -> np.ndarray:
        scene_builder = getattr(robot, "_scene_builder", None) if robot is not None else None
        part_type = get_conveyor_sorting_part_type(part, scene_builder)
        attr_name = (
            "_fixed_place_rpy_part_a"
            if part_type == "part_a"
            else "_fixed_place_rpy_part_b"
        )
        default_rpy = getattr(self, attr_name)

        if robot is None:
            return default_rpy.copy()

        grasp_cfg = self._get_grasp_cfg(robot)
        cfg_key = f"conveyor_{part_type}_place_rpy"
        legacy_cfg_key = f"conveyor_{part_type}_fixed_rpy"
        cfg_rpy = grasp_cfg.get(cfg_key, grasp_cfg.get(legacy_cfg_key))
        if cfg_rpy is None:
            return default_rpy.copy()

        cfg_rpy = np.asarray(cfg_rpy, dtype=np.float32)
        if cfg_rpy.shape != (3,):
            # logging.warning(
            #     f"Conveyor 忽略无效 {cfg_key}: 期望长度为 3, 实际 shape={cfg_rpy.shape}"
            # )
            return default_rpy.copy()
        return cfg_rpy.copy()

    def _get_pose_log_interval_s(self, robot) -> float:
        grasp_cfg = self._get_grasp_cfg(robot)
        return float(
            grasp_cfg.get(
                "conveyor_pose_log_interval_s",
                self._DEFAULT_POSE_LOG_INTERVAL_S,
            )
        )

    def _part_key(self, part: dict) -> str:
        prim_path = part.get("prim_path")
        if prim_path:
            return str(prim_path)
        return f"part_{part.get('index', len(self._processed_paths))}"

    def _compute_fixed_grasp_rpy(self, robot) -> np.ndarray:
        task_cfg = getattr(robot.config, "task_cfg", {}) or {}
        robot_rotation = task_cfg.get("robot", {}).get("robot_rotation", [0.0, 0.0, 90.0])
        forearm_tilt_deg = float(task_cfg.get("fsm_forearm_tilt_deg", 25.0))

        rot_deg = np.asarray(robot_rotation, dtype=float)
        r_base_to_world = self._euler_zyx_to_rotation_matrix(*rot_deg)
        r_world_to_base = r_base_to_world.T

        tilt_rad = math.radians(forearm_tilt_deg)
        world_down_base = r_world_to_base @ np.array([0.0, 0.0, -1.0])
        tilt_axis_base = np.array([-1.0, 0.0, 0.0])
        r_tilt = self._axis_angle_to_rotation(tilt_axis_base, tilt_rad)
        z_grasp = r_tilt @ world_down_base
        z_grasp /= np.linalg.norm(z_grasp)

        base_forward = r_world_to_base @ np.array([1.0, 0.0, 0.0])
        x_grasp = base_forward - np.dot(base_forward, z_grasp) * z_grasp
        if np.linalg.norm(x_grasp) < 1e-6:
            x_grasp = r_world_to_base @ np.array([0.0, 1.0, 0.0])
            x_grasp = x_grasp - np.dot(x_grasp, z_grasp) * z_grasp
        x_grasp /= np.linalg.norm(x_grasp)

        y_grasp = np.cross(z_grasp, x_grasp)
        y_grasp /= np.linalg.norm(y_grasp)

        r_grasp = np.column_stack([x_grasp, y_grasp, z_grasp])
        if np.linalg.det(r_grasp) < 0:
            r_grasp[:, 1] = -r_grasp[:, 1]

        return self._rotation_matrix_to_rpy(r_grasp).astype(np.float32)

    @staticmethod
    def _euler_zyx_to_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
        cz, sz = np.cos(yaw), np.sin(yaw)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cx, sx = np.cos(roll), np.sin(roll)

        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        return rz @ ry @ rx

    @staticmethod
    def _axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = math.cos(angle)
        s = math.sin(angle)
        c1 = 1.0 - c
        return np.array([
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ])

    @staticmethod
    def _rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
        sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
            pitch = math.atan2(-rotation[2, 0], sy)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        else:
            roll = math.atan2(-rotation[1, 2], rotation[1, 1])
            pitch = math.atan2(-rotation[2, 0], sy)
            yaw = 0.0
        return np.array([roll, pitch, yaw], dtype=float)
