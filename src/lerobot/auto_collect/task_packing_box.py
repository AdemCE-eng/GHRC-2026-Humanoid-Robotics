"""Task 4 - Packing Box: replay-based auto data collection for the box-folding task.

This task replays actions from an existing Packing_Box dataset on the robot
while recording new observations. Unlike pipeline-based tasks (PartSorting,
FoamInlaying), this uses dataset replay instead of IK-based trajectory planning.

Key detail: the source dataset has 18-D actions (14 arm joints + 4 finger joints)
but send_action expects 20 keys (same 18 joints with ".pos" suffix + left_gripper +
right_gripper). Missing gripper values default to 0.

CLI parameters (set via --auto_collect.*):
    --auto_collect.source_repo_id    源数据集标识 (默认 "packing_box_episode_50")
    --auto_collect.source_root       源数据集根目录
    --auto_collect.source_episode        要回放的源 episode 索引 (默认 0)
    --auto_collect.source_step_interval  帧采样间隔，每隔 N 帧取一帧 (默认 1)
    --auto_collect.num_episodes          采集/replay 次数 (每条 replay 同一源 episode)
"""

import logging
import time
from pathlib import Path

import numpy as np

from src.lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.lerobot.datasets.utils import build_dataset_frame
from src.lerobot.datasets.video_utils import VideoEncodingManager
from src.lerobot.utils.constants import ACTION, OBS_STR
from src.lerobot.utils.control_utils import (
    consume_episode_rerecord_request,
    consume_recording_stop_request,
    init_keyboard_listener,
)

from .auto_collect_base import (
    AutoCollectBase,
    AutoCollectEpisodeRerecord,
    AutoCollectStopRecording,
)
from .auto_collect_config import AutoCollectConfig


class TaskPackingBox(AutoCollectBase):
    """Replay-based auto data collection for the Packing_Box task.

    Loads actions from an existing source dataset and replays them on the robot
    while recording new frames to a separate output dataset.

    Source dataset path and episode index are configured via AutoCollectConfig:
        cfg.source_repo_id  - source dataset identifier
        cfg.source_root     - source dataset root directory
        cfg.source_episode  - episode index to replay (-1 = all episodes)
    """

    # Extra gripper control keys required by send_action but absent from source dataset
    _GRIPPER_DEFAULTS = {"left_gripper": 0.0, "right_gripper": 0.0}

    # ------------------------------------------------------------------
    # Abstract method stubs (not used by this replay-based task)
    # ------------------------------------------------------------------

    def compute_grasp_poses(self, part: dict) -> dict:
        return {}

    def check_grasp_success(self, robot, part: dict) -> bool:
        return False

    def get_place_pose(self, robot, part: dict, box_pos: np.ndarray) -> dict:
        return {}

    def _execute_sequence(
        self, robot, parts: list[dict], box_pos: np.ndarray, dt: float,
        dataset, single_task: str, objects_per_episode: int,
    ) -> bool:
        return False

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _map_action_to_20d(self, action_input: dict | np.ndarray) -> dict:
        """Map an 18-D dataset action to a 20-key send_action dict.

        Handles two input forms:
          - dict with joint names as keys (e.g. {"L_shoulder_pitch_joint": 0.1, ...})
          - numpy array in the order of _arm_joint_names + _finger_joint_names

        Returns a dict keyed by "joint_name.pos" + "left_gripper" / "right_gripper".
        """
        joint_keys = self._arm_joint_names + self._finger_joint_names
        mapped: dict = {}

        if isinstance(action_input, np.ndarray):
            arr = action_input.reshape(-1)
            for i, key in enumerate(joint_keys):
                if i < len(arr):
                    mapped[f"{key}.pos"] = float(arr[i])
        else:
            for key in joint_keys:
                val = action_input.get(key)
                if val is None:
                    val = action_input.get(f"{key}.pos", 0.0)
                mapped[f"{key}.pos"] = float(val)

        mapped.update(self._GRIPPER_DEFAULTS)
        return mapped

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _record_replay_frame(self, robot, action_20d: dict, dataset, single_task: str) -> None:
        """Record one frame: observation + action + task string."""
        obs = robot.get_observation()
        obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        action_frame = build_dataset_frame(dataset.features, action_20d, prefix=ACTION)
        frame = {**obs_frame, **action_frame, "task": single_task}
        dataset.add_frame(frame)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, robot, cfg: AutoCollectConfig) -> None:
        """Load source dataset, replay a specific episode, record to output dataset.

        Replays the single source episode (cfg.source_episode) cfg.num_episodes
        times, each time producing one recorded episode in the output dataset.
        This mirrors ``lerobot_replay.py --dataset.episode=N`` but run in a loop.

        Args:
            robot: Pre-created walker_s2_sim instance (not yet connected).
            cfg: AutoCollectConfig with recording and replay source parameters.
        """
        # ---- Resolve source dataset path ----
        source_repo_id = cfg.source_repo_id
        source_root = cfg.source_root
        if source_root is not None and isinstance(source_root, str):
            source_root = Path(source_root)
        elif source_root is None:
            source_root = Path("/data/SJJ/UBT/lerobot_0.5.1/datasets/Packing_Box")

        num_episodes = cfg.num_episodes
        step_interval = cfg.source_step_interval

        logging.info("=" * 60)
        logging.info("Packing_Box Replay 模式")
        logging.info(f"  源数据集:      {source_root / source_repo_id}")
        logging.info(f"  源 episode:    {cfg.source_episode}")
        logging.info(f"  帧采样间隔:    每 {step_interval} 帧取 1 帧")
        logging.info(f"  采集 episode 数: {num_episodes}")
        logging.info("=" * 60)

        # ---- Load source dataset (single episode) ----
        source_dataset = LeRobotDataset(
            source_repo_id,
            root=source_root,
            episodes=[cfg.source_episode],
        )
        logging.info(
            f"源数据集加载完成: {source_dataset.num_episodes} episodes, "
            f"{source_dataset.num_frames} frames, fps={source_dataset.fps}"
        )

        # ---- Create output dataset + connect robot (shared helpers) ----
        dataset, single_task = self._build_dataset(robot, cfg)
        dt = self._connect_and_settle(robot, cfg)

        # ---- Keyboard monitoring (lerobot official init_keyboard_listener) ----
        listener, events = init_keyboard_listener()
        logging.info("键盘监听已启动，按 ESC/Ctrl-C 结束录制，按左方向键重录当前 episode")

        # ---- Extract frames from the single source episode ----
        source_ep = cfg.source_episode
        ep_frames = source_dataset.hf_dataset.filter(
            lambda x, ep=source_ep: x["episode_index"] == ep
        )
        if len(ep_frames) == 0:
            raise RuntimeError(f"源 episode {source_ep} 无帧数据")

        actions_col = ep_frames.select_columns(ACTION)
        action_names = source_dataset.features[ACTION].get("names", [])
        sampled_frame_count = len(range(0, len(ep_frames), step_interval))
        logging.info(
            f"源 episode {source_ep}: {len(ep_frames)} 帧, "
            f"采样间隔 {step_interval} → {sampled_frame_count} 帧, "
            f"将 replay {num_episodes} 次"
        )

        def _run_episodes(events: dict):
            recorded_episodes = 0
            retry_count = 0
            while recorded_episodes < num_episodes:
                if cfg.record_data:
                    logging.info("=" * 60)
                    logging.info(
                        f"Replay {recorded_episodes + 1}/{num_episodes} "
                        f"(已录制: {dataset.num_episodes})"
                    )
                    logging.info("=" * 60)

                try:
                    if consume_recording_stop_request(events):
                        raise AutoCollectStopRecording
                    robot.reset()
                    if dataset is not None:
                        dataset.clear_episode_buffer()

                    logging.info(
                        f"第 {recorded_episodes + 1} 次 replay 源 episode {source_ep} "
                        f"({sampled_frame_count} 采样帧)"
                    )

                    for frame_idx in range(0, len(ep_frames), step_interval):
                        if consume_recording_stop_request(events):
                            raise AutoCollectStopRecording
                        if consume_episode_rerecord_request(events):
                            raise AutoCollectEpisodeRerecord

                        start_t = time.perf_counter()

                        # Extract action from dataset
                        action_array = actions_col[frame_idx][ACTION]
                        if isinstance(action_array, np.ndarray):
                            action_array = action_array.reshape(-1)

                        # Build action dict from dataset names
                        action_dict = {}
                        for i, name in enumerate(action_names):
                            val = action_array[i] if i < len(action_array) else 0.0
                            action_dict[name] = float(val)

                        # Map to 20-key format for send_action
                        mapped_action = self._map_action_to_20d(action_dict)

                        # Send to robot
                        robot.send_action(mapped_action)

                        # Record frame
                        if dataset is not None:
                            self._record_replay_frame(robot, mapped_action, dataset, single_task)

                        dt_s = time.perf_counter() - start_t
                        sleep_s = max(dt - dt_s, 0.0)
                        if sleep_s > 0:
                            time.sleep(sleep_s)

                    if consume_recording_stop_request(events):
                        raise AutoCollectStopRecording
                    if consume_episode_rerecord_request(events):
                        raise AutoCollectEpisodeRerecord
                except AutoCollectStopRecording:
                    logging.info("录制被 ESC 终止，保存当前 replay 并退出")
                    if dataset is not None and dataset.episode_buffer.get("size", 0) > 0:
                        dataset.save_episode()
                    return
                except KeyboardInterrupt:
                    logging.info("录制被 Ctrl-C 终止，保存当前 replay 并退出")
                    if dataset is not None and dataset.episode_buffer.get("size", 0) > 0:
                        dataset.save_episode()
                    return
                except AutoCollectEpisodeRerecord:
                    logging.info("左方向键触发重录当前 replay，清空缓冲区")
                    if dataset is not None:
                        dataset.clear_episode_buffer()
                    retry_count += 1
                    if retry_count >= cfg.max_retries:
                        logging.error(f"重试次数超过上限 ({cfg.max_retries})，跳过当前 replay")
                        recorded_episodes += 1
                        retry_count = 0
                    continue

                # Save episode
                if cfg.record_data and dataset is not None:
                    if dataset.episode_buffer.get("size", 0) == 0:
                        logging.error("Episode 缓冲区为空（未录制任何帧），跳过保存")
                    else:
                        dataset.save_episode()
                        logging.info(
                            f"第 {recorded_episodes + 1} 次 replay 已保存！"
                            f"当前总 episode 数: {dataset.num_episodes}"
                        )
                recorded_episodes += 1
                retry_count = 0

        try:
            if dataset is not None:
                with VideoEncodingManager(dataset):
                    _run_episodes(events)
                    if cfg.push_to_hub:
                        logging.info(f"推送数据集到 Hugging Face Hub: {cfg.repo_id}")
                        dataset.push_to_hub(
                            tags=cfg.tags if cfg.tags else None,
                            private=cfg.private,
                            path_in_repo=cfg.path_in_repo,
                        )
                        logging.info("数据集推送完成！")
            else:
                _run_episodes(events)
        finally:
            if listener is not None:
                listener.stop()
                logging.info("键盘监听已停止")
            if dataset is not None:
                dataset.finalize()
            if robot.is_connected:
                robot.disconnect()
                logging.info("机器人已断开连接")
