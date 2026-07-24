"""自动数采基类 —— 模板方法模式。

基类实现通用机器人控制接口（``gradually_move_gripper``、``move_gripper``、
``_joint_interpolate_to_pose``、``_cartesian_interpolate_to_pose``）和完整的 episode
流水线骨架（``run``）。任务子类必须实现 ``compute_grasp_poses``、
``check_grasp_success``、``get_place_pose`` 和 ``_execute_sequence``。

支持单臂和双臂控制：
    - ``gradually_move_gripper`` 通过 ``{"left": val, "right": val}`` 字典同时控制左右夹爪
    - ``_joint_interpolate_to_pose`` 通过关节空间插值同时控制左右手臂
    - ``_cartesian_interpolate_to_pose`` 通过笛卡尔空间插值 + 逐帧 IK 同时控制左右手臂
    - ``_execute_sequence`` 由子类实现单臂/双臂 episode 流水线
"""

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from src.lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from src.lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from src.lerobot.datasets.video_utils import VideoEncodingManager
from src.lerobot.processor import make_default_processors
from src.lerobot.utils.constants import ACTION, OBS_STR
from src.lerobot.utils.control_utils import (
    consume_episode_rerecord_request,
    consume_recording_stop_request,
    init_keyboard_listener,
)

from .auto_collect_config import AutoCollectConfig
from src.lerobot.robots.walker_s2_sim.isaac_sim_robot_interface import CartesianTrajectoryPlanner


class AutoCollectEpisodeRerecord(RuntimeError):
    """Raised when keyboard/terminal input requests rerecording the current episode."""


class AutoCollectStopRecording(RuntimeError):
    """Raised when keyboard/terminal input requests stopping the recording run."""


class AutoCollectBase(ABC):
    """自动数采基类 —— 模板方法模式。

    子类必须实现:
        - ``compute_grasp_poses(part_pos) -> dict``
        - ``check_grasp_success(robot, part) -> bool``
        - ``get_place_pose(robot, part, box_pos) -> dict``

    可选覆盖:
        - ``_on_episode_start(parts)``  每个 episode 开始前的回调
    """

    # 14 个手臂关节名称
    _arm_joint_names = [
        "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
        "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint",
        "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
        "R_elbow_roll_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint",
    ]
    # 4 个手指关节名称
    _finger_joint_names = [
        "L_finger1_joint", "L_finger2_joint", "R_finger1_joint", "R_finger2_joint",
    ]

    # 双臂任务标志，子类覆盖为 True 即可启用双臂 episode 执行模式
    is_dual_arm: bool = False
    arm_execution_mode: str = "dual"

    def __init__(self) -> None:
        self._keyboard_events: dict | None = None
        self._speed_scale: float = 1.0

    def get_arm_side(self, part: dict) -> str:
        """返回零件对应的手臂侧。基类默认返回 ``"right"``，子类可覆盖。"""
        return "right"

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _build_action_dict(self, robot) -> dict:
        """从机器人内部状态构建动作字典，匹配 ``robot.action_features`` 的键。"""
        arm_pos = robot._hold_arm_positions
        finger_pos = robot._hold_finger_positions

        action = {}
        for i, name in enumerate(self._arm_joint_names):
            action[f"{name}.pos"] = float(arm_pos[i])
        for i, name in enumerate(self._finger_joint_names):
            action[f"{name}.pos"] = float(finger_pos[i])
        action["left_gripper"] = -1.0 if robot._left_gripping else 1.0
        action["right_gripper"] = -1.0 if robot._right_gripping else 1.0
        return action

    def _record_frame(self, robot, dataset, single_task: str) -> None:
        """录制一帧：观测 + 动作 + 任务描述，写入数据集。"""
        self._raise_if_keyboard_requested()
        obs = robot.get_observation()
        action_dict = self._build_action_dict(robot)

        obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
        frame = {**obs_frame, **action_frame, "task": single_task}
        dataset.add_frame(frame)

    def _raise_if_keyboard_requested(self) -> None:
        if consume_recording_stop_request(self._keyboard_events):
            raise AutoCollectStopRecording("键盘事件请求停止录制")
        if consume_episode_rerecord_request(self._keyboard_events):
            raise AutoCollectEpisodeRerecord("当前 episode 被键盘事件中断，需要重录")

    # =========================================================================
    # 通用机器人控制接口（支持单臂 / 双臂）
    # =========================================================================

    def gradually_move_gripper(
        self,
        robot,
        targets: dict[str, float],
        steps: int = 30,
        dataset=None,
        single_task: str | None = None,
    ) -> None:
        """逐步移动夹爪，支持单臂或双臂同时控制。

        Args:
            robot: ``WalkerS2sim`` 实例。
            targets: ``{"left": 1.0, "right": -1.0}`` 或 ``{"right": 1.0}``。
                     1.0=闭合, -1.0=张开。
            steps: 物理步数。
            dataset: 可选的 ``LeRobotDataset``。
            single_task: 任务描述字符串。
        """
        gripper_step = 0.01
        g_open = robot._robot_interface.gripper_open_width
        g_close = robot._robot_interface.gripper_close_width
        g_lo, g_hi = min(g_open, g_close), max(g_open, g_close)

        arm_configs = []
        for side, t in targets.items():
            if side == "left":
                arm_configs.append((slice(0, 2), "_left_gripping", t))
            else:
                arm_configs.append((slice(2, 4), "_right_gripping", t))

        for _ in range(steps):
            for indices, gripping_attr, t in arm_configs:
                current = robot._hold_finger_positions[indices]
                delta = t * gripper_step
                robot._hold_finger_positions[indices] = np.clip(current + delta, g_lo, g_hi)
                setattr(robot, gripping_attr, t > 0)
            robot.step(render=True)

            if dataset is not None:
                self._record_frame(robot, dataset, single_task)

    def move_gripper(
        self,
        robot,
        targets: dict[str, float],
        dt: float,
        duration: float,
        dataset=None,
        single_task=None,
    ) -> None:  
        joint_states = robot._robot_interface.get_joint_states()
        if joint_states is None:
            logging.warning("无法获取关节状态")
            return
        arm_finger_indices = (
            robot._robot_interface.arm_joint_indices
            + robot._robot_interface.finger_joint_indices
        )
        
        start_q=joint_states["all_positions"]
        
        target_q = np.zeros(18, dtype=np.float32)
        target_q[:14]=np.array(start_q)[robot._robot_interface.arm_joint_indices]
        target_q[14:18] = np.array(start_q)[robot._robot_interface.finger_joint_indices]
        for side, t in targets.items():
            if side == "left":
                if t < 0:
                    target_q[14:16] = robot._robot_interface.gripper_open_width
                    robot._left_gripping = False
                else:
                    target_q[14:16] = robot._robot_interface.gripper_close_width
                    robot._left_gripping = True
            elif side == "right":
                if t < 0:
                    target_q[16:18] = robot._robot_interface.gripper_open_width
                    robot._right_gripping = False
                else:
                    target_q[16:18] = robot._robot_interface.gripper_close_width
                    robot._right_gripping = True

        steps = max(1, int(duration * self._speed_scale / dt))

        robot._robot_interface.joint_interpolator.set_target(
            start_q=torch.tensor(start_q)[arm_finger_indices],
            target_q=torch.tensor(target_q),
            num_steps=steps
        )

        for _ in range(steps):
            arm_finger_positions = robot._robot_interface.joint_interpolator.step()
            if isinstance(arm_finger_positions, torch.Tensor):
                arm_finger_positions = arm_finger_positions.detach().cpu().numpy()
            else:
                arm_finger_positions = np.asarray(arm_finger_positions, dtype=np.float32)
            robot._hold_arm_positions = arm_finger_positions[:14]
            robot._hold_finger_positions = arm_finger_positions[14:18]
            robot.step(render=True)
            
            if dataset is not None: 
                self._record_frame(robot, dataset, single_task)

    def _joint_interpolate_to_pose(
        self,
        robot,
        target_poses: dict[str, dict[str, np.ndarray]],
        dt: float,
        duration: float,
        dataset=None,
        single_task: str | None = None,
    ) -> None:
        """通过 IK + 关节插值平滑移动手臂，支持单臂或双臂同时控制。

        Args:
            robot: ``WalkerS2sim`` 实例。
            target_poses: ``{"left": {"position": [x,y,z], "rotation": [r,p,y]},
                             "right": {"position": [x,y,z], "rotation": [r,p,y]}}``。
                         左臂或右臂键可以缺失（对应手臂不移动）。
            dt: 控制周期 (1/fps)。
            duration: 移动持续时间（秒）。
            dataset: 可选的 ``LeRobotDataset``。
            single_task: 任务描述字符串。
        """
        right_pose = target_poses.get("right")
        right_6d = None
        if right_pose is not None:
            right_pos = right_pose["position"]
            right_rpy = right_pose["rotation"]
            right_6d = np.array([
                right_pos[0], right_pos[1], right_pos[2],
                right_rpy[0], right_rpy[1], right_rpy[2],
            ], dtype=np.float32)

        left_pose = target_poses.get("left")
        left_6d = None
        if left_pose is not None:
            left_pos = left_pose["position"]
            left_rpy = left_pose["rotation"]
            left_6d = np.array([
                left_pos[0], left_pos[1], left_pos[2],
                left_rpy[0], left_rpy[1], left_rpy[2],
            ], dtype=np.float32)

        logging.info(
            f"目标位姿 — 右: {right_6d}" + (f", 左: {left_6d}" if left_6d is not None else "")
        )

        ik_result = robot._robot_interface.control_dual_arm_ik(
            step_size=0.02,
            left_target_xyzrpy=left_6d,
            right_target_xyzrpy=right_6d,
        )

        if ik_result is None:
            logging.warning("IK 求解失败")
            return

        # 构建目标关节位置（14 臂关节）
        target_joints_14d = np.zeros(14, dtype=np.float32)
        if robot._hold_arm_positions is not None:
            target_joints_14d[:14] = robot._hold_arm_positions.copy()
        else:
            target_joints_14d[:14] = robot._robot_interface.initial_joint_positions[:14]

        if "right_joint_positions" in ik_result:
            rj = ik_result["right_joint_positions"]
            if len(rj) == 14:
                target_joints_14d = np.array(rj[:14], dtype=np.float32)
            elif len(rj) == 7:
                target_joints_14d[7:14] = np.array(rj, dtype=np.float32)

        if "left_joint_positions" in ik_result:
            lj = ik_result["left_joint_positions"]
            if len(lj) == 7:
                target_joints_14d[:7] = np.array(lj, dtype=np.float32)

        # 构建完整目标（手臂 + 手指）
        target_positions = np.concatenate([target_joints_14d, np.zeros(4)])
        finger_states = robot._robot_interface.get_joint_states()
        if finger_states:
            target_positions[14:18] = np.array(finger_states["finger_positions"])

        steps = int(duration * self._speed_scale / dt)
        logging.info(f"插值步数: {steps}")

        arm_finger_indices = (
            robot._robot_interface.arm_joint_indices
            + robot._robot_interface.finger_joint_indices
        )
        robot._robot_interface.joint_interpolator.set_target(
            start_q=torch.tensor(
                robot._robot_interface.get_joint_states()["all_positions"]
            )[arm_finger_indices],
            target_q=torch.tensor(target_positions),
            num_steps=steps,
        )

        for _ in range(steps):
            arm_finger_positions = robot._robot_interface.joint_interpolator.step()
            if isinstance(arm_finger_positions, torch.Tensor):
                arm_finger_positions = arm_finger_positions.detach().cpu().numpy()
            else:
                arm_finger_positions = np.asarray(arm_finger_positions, dtype=np.float32)
            robot._hold_arm_positions = arm_finger_positions[:14]
            robot._hold_finger_positions = arm_finger_positions[14:18]
            robot.step(render=True)

            if dataset is not None:
                self._record_frame(robot, dataset, single_task)

        logging.info("关节插值移动完成")

    def _cartesian_interpolate_to_pose(
        self,
        robot,
        target_poses: dict[str, dict[str, np.ndarray]],
        dt: float,
        duration: float,
        dataset=None,
        single_task: str | None = None,
    ) -> None:
        """通过笛卡尔空间插值（位置 LERP + 姿态 SLERP）+ 逐帧 IK 平滑移动手臂。

        与 ``_joint_interpolate_to_pose`` 不同，该方法在笛卡尔空间规划直线轨迹，
        每一步求解 IK，适用于需要末端执行器沿直线运动的场景。
        支持单臂或双臂同时控制。

        Args:
            robot: ``WalkerS2sim`` 实例。
            target_poses: 同 ``_joint_interpolate_to_pose``。
            dt: 控制周期 (1/fps)。
            duration: 移动持续时间（秒）。
            dataset: 可选的 ``LeRobotDataset``。
            single_task: 任务描述字符串。
        """
        right_pose = target_poses.get("right")
        right_6d = None
        if right_pose is not None:
            right_6d = np.array([
                right_pose["position"][0], right_pose["position"][1], right_pose["position"][2],
                right_pose["rotation"][0], right_pose["rotation"][1], right_pose["rotation"][2],
            ], dtype=np.float32)

        left_pose = target_poses.get("left")
        left_6d = None
        if left_pose is not None:
            left_6d = np.array([
                left_pose["position"][0], left_pose["position"][1], left_pose["position"][2],
                left_pose["rotation"][0], left_pose["rotation"][1], left_pose["rotation"][2],
            ], dtype=np.float32)

        logging.info(
            f"笛卡尔目标位姿 — 右: {right_6d}" + (f", 左: {left_6d}" if left_6d is not None else "")
        )

        # 获取当前末端执行器位姿作为插值起点
        current_ee = robot._robot_interface.get_ee_poses()
        if current_ee is None:
            logging.warning("笛卡尔插值: 无法获取当前末端执行器位姿")
            return

        # 为每只手臂创建独立的笛卡尔轨迹规划器
        planners: dict[str, CartesianTrajectoryPlanner] = {}
        for side, target_6d in (("right", right_6d), ("left", left_6d)):
            if target_6d is None:
                continue
            ee = current_ee.get(side)
            if ee is None:
                logging.warning(f"笛卡尔插值: 无法获取 {side} 手当前位姿")
                continue
            start_6d = np.array(ee[:6], dtype=np.float32)
            planner = CartesianTrajectoryPlanner()
            steps = int(duration * self._speed_scale / dt)
            planner.set_target(start_6d, target_6d, steps)
            planners[side] = planner

        if not planners:
            return

        steps = int(duration * self._speed_scale / dt)
        logging.info(f"笛卡尔插值步数: {steps}")

        for _ in range(steps):
            ik_left = None
            ik_right = None

            for side, planner in planners.items():
                interp_pose = planner.step()
                if side == "left":
                    ik_left = interp_pose
                else:
                    ik_right = interp_pose

            ik_result = robot._robot_interface.control_dual_arm_ik(
                step_size=0.02,
                left_target_xyzrpy=ik_left,
                right_target_xyzrpy=ik_right,
            )

            if ik_result is None:
                continue

            # 从 IK 结果提取关节位置并更新 hold 状态
            if robot._hold_arm_positions is None:
                robot._hold_arm_positions = np.array(
                    robot._robot_interface.initial_joint_positions[:14], dtype=np.float32
                )

            if "right_joint_positions" in ik_result:
                rj = ik_result["right_joint_positions"]
                if len(rj) == 14:
                    robot._hold_arm_positions[:14] = np.array(rj[:14], dtype=np.float32)
                elif len(rj) == 7:
                    robot._hold_arm_positions[7:14] = np.array(rj, dtype=np.float32)

            if "left_joint_positions" in ik_result:
                lj = ik_result["left_joint_positions"]
                if len(lj) == 7:
                    robot._hold_arm_positions[:7] = np.array(lj, dtype=np.float32)

            robot.step(render=True)

            if dataset is not None:
                self._record_frame(robot, dataset, single_task)

        logging.info("笛卡尔插值移动完成")

    # =========================================================================
    # 抓取成功检测 —— 通用辅助方法
    # =========================================================================

    def _check_grasp_success_for_arm(
        self, robot, part: dict, arm_side: str = "right", threshold: float = 0.08
    ) -> bool:
        """检查指定手臂的夹爪与零件之间的距离是否在阈值内。

        Args:
            robot: ``WalkerS2sim`` 实例。
            part: 零件信息字典。
            arm_side: ``"left"`` 或 ``"right"``。
            threshold: 距离阈值（米）。

        Returns:
            True 表示抓取成功。
        """
        ee_poses = robot._robot_interface.get_ee_poses()
        if ee_poses is None:
            logging.warning("无法获取末端执行器位姿")
            return False

        ee = ee_poses.get(arm_side)
        if ee is None:
            ee = ee_poses.get(list(ee_poses.keys())[0])
        if ee is None:
            logging.warning(f"无法获取 {arm_side} 手末端执行器位姿")
            return False

        gripper_pos = np.array(ee[:3])

        parts = robot._scene_builder.get_parts_world_poses()
        part_pos = None
        prim_path = part.get("prim_path", "")
        for p in parts:
            if p.get("prim_path") == prim_path:
                part_pos = np.array(p.get("position", [0, 0, 0]))
                break

        if part_pos is None:
            logging.warning(f"找不到零件: {prim_path}")
            return False

        part_pos = robot._scene_builder.world_to_robot_coords(part_pos)
        distance = np.linalg.norm(gripper_pos - part_pos)

        logging.info(
            f"抓取检测 [{arm_side}]: 零件位置={part_pos}, 夹爪位置={gripper_pos}, "
            f"距离={distance:.4f}m"
        )

        if distance > threshold:
            logging.warning(
                f"抓取失败 [{arm_side}]！距离 {distance:.4f}m 超过阈值 {threshold}m"
            )
            return False

        logging.info(f"抓取成功 [{arm_side}]！距离 {distance:.4f}m < {threshold}m")
        return True

    # =========================================================================
    # 双臂零件分组
    # =========================================================================

    def _group_parts_by_arm(self, parts: list[dict]) -> tuple[list[dict], list[dict]]:
        """将零件按手臂侧分组。

        Returns:
            ``(left_parts, right_parts)``。
        """
        left, right = [], []
        for p in parts:
            (left if self.get_arm_side(p) == "left" else right).append(p)
        return left, right

    # =========================================================================
    # 任务差异接口（子类必须实现）
    # =========================================================================

    @abstractmethod
    def _execute_sequence(
        self, robot, parts: list[dict], box_pos: np.ndarray, dt: float,
        dataset, single_task: str, objects_per_episode: int,
    ) -> bool:
        """执行完整的抓取-放置 episode 流水线。

        子类根据自身任务模式（单臂/双臂）实现:
            - 单臂任务: 逐个零件执行抓取-放置
            - 双臂任务: 将零件配对后同时执行双臂抓取-放置

        应调用 ``_run_grasp_stages`` 执行各阶段的运动控制。

        Returns:
            True 表示成功，False 表示失败。
        """
        ...

    @abstractmethod
    def compute_grasp_poses(self, part: dict) -> dict:
        """计算接近/下降/抓取/抬起位姿。

        Returns:
            ``{"left": {stages}, "right": {stages}}`` 或
            ``{"right": {stages}}``（单臂任务）。
            每个 arm 值为 ``{"approach": ..., "descend": ..., "grasp": ..., "lift": ...}``，
            各阶段为 ``{"position": np.ndarray, "rotation": np.ndarray}``（世界坐标系）。
            返回空字典 ``{}`` 表示该零件已被处理（配对任务中可跳过）。
        """
        ...

    @abstractmethod
    def check_grasp_success(self, robot, part: dict) -> bool:
        """检查零件是否成功抓取。

        对于单臂任务，委托给 ``_check_grasp_success_for_arm`` 即可。
        对于双臂任务，可同时检查左右臂。
        """
        ...

    @abstractmethod
    def get_place_pose(self, robot, part: dict, box_pos: np.ndarray) -> dict:
        """返回放置目标位姿。

        Returns:
            ``{"left": {"position": ..., "rotation": ...},
               "right": {"position": ..., "rotation": ...}}``
            或仅含单臂键（单臂任务），世界坐标系下。
        """
        ...

    def _on_episode_start(self, parts: list[dict] | None = None) -> None:
        """每个 episode 开始前的回调，子类可覆盖以重置任务状态。

        Args:
            parts: 本 episode 的零件列表（已打乱），可用于预配对等操作。
        """
        pass

    def _on_episode_saved(self, dataset, episode_index: int, episode_length: int) -> None:
        """每个 episode 成功保存后的回调，子类可覆盖以写任务侧元数据。"""
        pass

    def _get_episode_parts(self, robot) -> list[dict]:
        """返回当前 episode 可处理零件列表，子类可覆盖特殊场景查询方式。"""
        return robot._scene_builder.get_parts_world_poses()

    # =========================================================================
    # 共享辅助方法（子类可复用）
    # =========================================================================

    @classmethod
    def _build_dataset(cls, robot, cfg: AutoCollectConfig) -> tuple:
        """Create and return ``(dataset, single_task)`` from config.

        Returns ``(None, single_task)`` when ``cfg.record_data`` is False.
        """
        if not cfg.record_data:
            return None, cfg.single_task or cfg.task

        repo_id = cfg.repo_id
        root = cfg.root
        if root is None:
            root = Path("./outputs/auto_collect")
        elif isinstance(root, str):
            root = Path(root)

        single_task = cfg.single_task or cfg.task

        teleop_action_processor, _, robot_observation_processor = make_default_processors()

        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=cfg.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=cfg.video,
            ),
        )

        dataset = LeRobotDataset.create(
            repo_id,
            cfg.fps,
            features=dataset_features,
            root=root,
            robot_type=robot.name,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera
            * len(robot.cameras),
            batch_encoding_size=cfg.video_encoding_batch_size,
            vcodec=cfg.vcodec,
            streaming_encoding=cfg.streaming_encoding,
            encoder_queue_maxsize=cfg.encoder_queue_maxsize,
            encoder_threads=cfg.encoder_threads,
        )
        logging.info(f"数据集: {repo_id}, 路径: {root}, 任务描述: {single_task}")
        return dataset, single_task

    @staticmethod
    def _connect_and_settle(robot, cfg: AutoCollectConfig) -> float:
        """Connect robot, wait for physics to settle, return ``dt`` (1/fps)."""
        if not robot.is_connected:
            robot.connect()
            logging.info("机器人连接成功")

        logging.info("等待物理稳定...")
        settle_steps = int(2.0 / (1.0 / cfg.fps))
        for _ in range(settle_steps):
            robot.step(render=True)

        return 1.0 / cfg.fps

    # =========================================================================
    # 主入口
    # =========================================================================

    def run(self, robot, cfg: AutoCollectConfig) -> None:
        """连接机器人、采集数据、保存数据集、断开连接。

        Args:
            robot: 预创建的 ``WalkerS2sim`` 实例（未连接）。
            cfg: ``AutoCollectConfig`` 采集参数。
        """
        task_cfg = getattr(getattr(robot, "config", None), "task_cfg", {}) or {}
        task_auto_collect_cfg = task_cfg.get("auto_collect", {}) or {}
        default_arm_mode = AutoCollectConfig.__dataclass_fields__["arm_execution_mode"].default
        if cfg.arm_execution_mode == default_arm_mode and task_auto_collect_cfg.get("arm_execution_mode"):
            self.arm_execution_mode = task_auto_collect_cfg["arm_execution_mode"]
        else:
            self.arm_execution_mode = cfg.arm_execution_mode
        logging.info(f"Arm execution mode: {self.arm_execution_mode}")

        logging.info("=" * 60)
        logging.info(f"自动数采模式 - 任务: {cfg.task}")
        logging.info("=" * 60)

        # ===== 数据集初始化 + 连接机器人 =====
        dataset, single_task = self._build_dataset(robot, cfg)
        dt = self._connect_and_settle(robot, cfg)

        # ===== 获取场景信息 =====
        box_pos = None
        if robot._scene_builder is not None:
            box_positions = robot._scene_builder.get_box_positions()
            box_pos = box_positions[0] if box_positions else None
        if box_pos is None:
            box_pos = np.array([1.2, 0.3, 1.05])
        logging.info(f"箱子/目标位置: {box_pos}")

        # ===== Episode 循环 =====
        total_episodes = cfg.num_episodes if cfg.record_data else 1
        listener, events = init_keyboard_listener()
        self._keyboard_events = events
        logging.info("键盘监听已启动，按 ESC/Ctrl-C 结束录制，按左方向键重录当前 episode")

        def _run_episodes():
            for episode_idx in range(total_episodes):
                if cfg.record_data:
                    logging.info("=" * 60)
                    logging.info(
                        f"Episode {episode_idx + 1}/{total_episodes} "
                        f"(已录制: {dataset.num_episodes})"
                    )
                    logging.info("=" * 60)

                episode_success = False
                retry_count = 0

                while not episode_success and retry_count < cfg.max_retries:
                    retry_count += 1
                    if retry_count > 1:
                        logging.info(f"重试 {retry_count}/{cfg.max_retries}")

                    logging.info("重置场景...")
                    robot.reset()
                    # 等待物理稳定
                    for i in range(10):
                        robot.step(render=True)

                    if dataset is not None:
                        dataset.clear_episode_buffer()

                    parts = self._get_episode_parts(robot)
                    if not parts:
                        logging.error("未找到零件！")
                        break

                    random.shuffle(parts)
                    if cfg.random_speed:
                        self._speed_scale = random.uniform(cfg.speed_scale_min, cfg.speed_scale_max)
                        logging.info(f"Speed scale: {self._speed_scale:.3f}")
                    else:
                        self._speed_scale = 1.0
                    # 通知子类新 episode 开始（用于重置任务状态如孔位索引、零件配对）
                    self._on_episode_start(parts)
                    try:
                        self._raise_if_keyboard_requested()
                        episode_success = self._execute_sequence(
                            robot=robot, parts=parts, box_pos=box_pos, dt=dt,
                            dataset=dataset, single_task=single_task,
                            objects_per_episode=cfg.objects_per_episode,
                        )
                        self._raise_if_keyboard_requested()
                    except AutoCollectStopRecording:
                        logging.info("录制被 ESC 终止，保存当前 episode 并退出")
                        if dataset is not None and dataset.episode_buffer.get("size", 0) > 0:
                            dataset.save_episode()
                        return
                    except KeyboardInterrupt:
                        logging.info("录制被 Ctrl-C 终止，保存当前 episode 并退出")
                        if dataset is not None and dataset.episode_buffer.get("size", 0) > 0:
                            dataset.save_episode()
                        return
                    except AutoCollectEpisodeRerecord:
                        logging.info("左方向键触发重录当前 episode，清空缓冲区")
                        if dataset is not None:
                            dataset.clear_episode_buffer()
                        if retry_count > 0:
                            retry_count -= 1
                        episode_success = False
                        continue

                if retry_count >= cfg.max_retries and not episode_success:
                    logging.error(f"重试次数超过上限 ({cfg.max_retries})，跳过当前 episode")

                if episode_success:
                    logging.info(f"\n所有 {len(parts)} 个零件处理完成！")
                    if cfg.record_data and dataset is not None:
                        if dataset.episode_buffer.get("size", 0) == 0:
                            logging.error("Episode 缓冲区为空（未录制任何帧），跳过保存")
                        else:
                            saved_episode_index = int(dataset.episode_buffer["episode_index"])
                            saved_episode_length = int(dataset.episode_buffer["size"])
                            dataset.save_episode()
                            dataset.meta._flush_metadata_buffer()
                            self._on_episode_saved(
                                dataset,
                                episode_index=saved_episode_index,
                                episode_length=saved_episode_length,
                            )
                            logging.info(
                                f"Episode {episode_idx + 1} 已保存！"
                                f"当前总 episode 数: {dataset.num_episodes}"
                            )

        try:
            if dataset is not None:
                with VideoEncodingManager(dataset):
                    _run_episodes()
                    # ===== 推送到 Hub =====
                    if cfg.push_to_hub:
                        logging.info(f"推送数据集到 Hugging Face Hub: {cfg.repo_id}")
                        dataset.push_to_hub(
                            tags=cfg.tags if cfg.tags else None,
                            private=cfg.private,
                            path_in_repo=cfg.path_in_repo,
                        )
                        logging.info("数据集推送完成！")
            else:
                _run_episodes()
        finally:
            if listener is not None:
                listener.stop()
                logging.info("键盘监听已停止")
            self._keyboard_events = None
            if dataset is not None:
                dataset.finalize()
            if robot.is_connected:
                robot.disconnect()
                logging.info("机器人已断开连接")
