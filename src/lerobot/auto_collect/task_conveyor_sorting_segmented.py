"""Conveyor Sorting segmented collector.

This collector keeps the original grasp/place logic from ``TaskConveyorSorting``
but reorganizes recording into:

1. short episodes: one pick-and-place per episode;
2. long episodes: eight pick-and-place segments concatenated into one episode.

Existing source files stay untouched. Use the companion script
``src/lerobot/scripts/auto_collect_conveyor_segmented.py`` to run it.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.lerobot.datasets.video_utils import VideoEncodingManager
from src.lerobot.utils.control_utils import init_keyboard_listener
from src.lerobot.utils.utils import SuppressProgressBars

from .auto_collect_base import (
    AutoCollectBase,
    AutoCollectEpisodeRerecord,
    AutoCollectStopRecording,
)
from .auto_collect_config import AutoCollectConfig
from .task_conveyor_sorting import TaskConveyorSorting
from .utils import get_conveyor_sorting_part_type


class _DatasetFanout:
    """Forward each recorded frame to multiple datasets."""

    def __init__(self, *datasets):
        self._datasets = [dataset for dataset in datasets if dataset is not None]
        self.features = self._datasets[0].features if self._datasets else {}

    def add_frame(self, frame: dict) -> None:
        for dataset in self._datasets:
            dataset.add_frame(copy.deepcopy(frame))


class _FrameBufferDataset:
    """In-memory episode buffer with the minimal dataset API used by _record_frame."""

    def __init__(self, features):
        self.features = features
        self.frames: list[dict] = []

    def add_frame(self, frame: dict) -> None:
        self.frames.append(copy.deepcopy(frame))


class TaskConveyorSortingSegmented(TaskConveyorSorting):
    """Collect short per-part episodes and merged long episodes together."""

    _SHORT_DIRNAME = "short_episodes"
    _LONG_DIRNAME = "long_episodes"

    def __init__(self) -> None:
        super().__init__()
        self._current_long_episode_parts: list[dict] = []
        self._current_short_episode_part: dict | None = None
        self._pending_short_episodes: list[dict] = []

    def run(self, robot, cfg: AutoCollectConfig) -> None:
        self.arm_execution_mode = cfg.arm_execution_mode
        raw_failure_mode = str(cfg.segmented_failure_mode or "strict").strip().lower()
        if raw_failure_mode in {"loosen", "skip_failed_parts"}:
            self.segmented_failure_mode = "loosen"
        else:
            self.segmented_failure_mode = "strict"
        logging.info("Arm execution mode: %s", self.arm_execution_mode)
        logging.info("Segmented failure mode: %s", self.segmented_failure_mode)
        logging.info("=" * 60)
        logging.info("分段自动数采模式 - 任务: %s", cfg.task)
        logging.info("=" * 60)

        short_dataset, long_dataset, single_task = self._build_segmented_datasets(robot, cfg)
        dt = self._connect_and_settle(robot, cfg)

        box_pos = None
        if robot._scene_builder is not None:
            box_positions = robot._scene_builder.get_box_positions()
            box_pos = box_positions[0] if box_positions else None
        if box_pos is None:
            box_pos = np.array([1.2, 0.3, 1.05], dtype=np.float32)
        logging.info("箱子/目标位置: %s", box_pos)

        total_long_episodes = cfg.num_episodes if cfg.record_data else 1
        listener, events = init_keyboard_listener()
        self._keyboard_events = events
        logging.info("键盘监听已启动，按 ESC/Ctrl-C 结束录制，按左方向键重录当前长 episode")

        def _run_long_episodes() -> None:
            for long_episode_idx in range(total_long_episodes):
                if cfg.record_data:
                    logging.info("=" * 60)
                    logging.info(
                        "Long Episode %s/%s (已录制长轨迹: %s, 短轨迹: %s)",
                        long_episode_idx + 1,
                        total_long_episodes,
                        0 if long_dataset is None else long_dataset.num_episodes,
                        0 if short_dataset is None else short_dataset.num_episodes,
                    )
                    logging.info("=" * 60)

                episode_success = False
                retry_count = 0
                while not episode_success and retry_count < cfg.max_retries:
                    retry_count += 1
                    if retry_count > 1:
                        logging.info("重试 %s/%s", retry_count, cfg.max_retries)

                    logging.info("重置场景...")
                    robot.reset()
                    for _ in range(10):
                        robot.step(render=True)

                    if short_dataset is not None:
                        short_dataset.clear_episode_buffer()
                    if long_dataset is not None:
                        long_dataset.clear_episode_buffer()

                    self._processed_paths.clear()
                    self._last_pose_log_time = 0.0
                    self._current_long_episode_parts = []
                    self._current_short_episode_part = None
                    self._pending_short_episodes = []

                    parts = self._wait_for_episode_parts_after_reset(robot, dt)
                    if not parts:
                        logging.error("未找到零件！")
                        episode_success = False
                        continue

                    random.shuffle(parts)
                    self._on_episode_start(parts)
                    try:
                        self._raise_if_keyboard_requested()
                        logging.info(
                            "Conveyor segmented: 开始执行 long episode，parts=%s objects_per_episode=%s",
                            len(parts),
                            cfg.objects_per_episode,
                        )
                        episode_success = self._execute_segmented_sequence(
                            robot=robot,
                            box_pos=box_pos,
                            dt=dt,
                            short_dataset=short_dataset,
                            long_dataset=long_dataset,
                            single_task=single_task,
                            objects_per_episode=cfg.objects_per_episode,
                            episode_parts=parts,
                        )
                        logging.info(
                            "Conveyor segmented: long episode 执行结束，success=%s",
                            episode_success,
                        )
                        self._raise_if_keyboard_requested()
                    except AutoCollectStopRecording:
                        logging.info("录制被 ESC 终止，保存当前缓冲区并退出")
                        self._save_partial_buffers(
                            short_dataset=short_dataset,
                            long_dataset=long_dataset,
                        )
                        return
                    except KeyboardInterrupt:
                        logging.info("录制被 Ctrl-C 终止，保存当前缓冲区并退出")
                        self._save_partial_buffers(
                            short_dataset=short_dataset,
                            long_dataset=long_dataset,
                        )
                        return
                    except AutoCollectEpisodeRerecord:
                        logging.info("左方向键触发重录当前长 episode，清空长短缓冲区")
                        if short_dataset is not None:
                            short_dataset.clear_episode_buffer()
                        if long_dataset is not None:
                            long_dataset.clear_episode_buffer()
                        self._current_long_episode_parts = []
                        self._current_short_episode_part = None
                        self._pending_short_episodes = []
                        if retry_count > 0:
                            retry_count -= 1
                        episode_success = False
                        continue
                    except Exception:
                        logging.exception("Conveyor segmented: long episode 执行异常")
                        episode_success = False
                        continue

                if retry_count >= cfg.max_retries and not episode_success:
                    logging.error("重试次数超过上限 (%s)，跳过当前长 episode", cfg.max_retries)

                if episode_success and cfg.record_data:
                    self._save_completed_buffers(
                        short_dataset=short_dataset,
                        long_dataset=long_dataset,
                    )

        try:
            if short_dataset is not None and long_dataset is not None:
                with VideoEncodingManager(short_dataset):
                    with VideoEncodingManager(long_dataset):
                        _run_long_episodes()
            else:
                _run_long_episodes()
        finally:
            self._keyboard_events = None
            if listener is not None:
                listener.stop()
            if short_dataset is not None:
                short_dataset.finalize()
            if long_dataset is not None:
                long_dataset.finalize()
            if robot.is_connected:
                robot.disconnect()

    def _execute_segmented_sequence(
        self,
        robot,
        box_pos: np.ndarray,
        dt: float,
        short_dataset,
        long_dataset,
        single_task: str,
        objects_per_episode: int,
        episode_parts: list[dict] | None = None,
    ) -> bool:
        timeout_s = self._get_timeout_s(robot)
        deadline = time.perf_counter() + timeout_s
        requested_count = int(objects_per_episode)
        target_count = self._TARGET_PARTS_PER_EPISODE
        if requested_count > 0:
            target_count = min(requested_count, self._TARGET_PARTS_PER_EPISODE)
        failure_mode = getattr(self, "segmented_failure_mode", "strict")
        loosen_mode = failure_mode == "loosen"
        episode_part_keys = [
            self._part_key(part)
            for part in (episode_parts or [])
        ][:target_count]
        logging.info(
            "Conveyor segmented: target_count=%s failure_mode=%s initial_parts=%s",
            target_count,
            failure_mode,
            len(episode_parts or []),
        )

        completed = 0
        attempted_parts: set[str] = set()
        while True:
            if not loosen_mode and completed >= target_count:
                break

            if time.perf_counter() >= deadline:
                logging.error(
                    "Conveyor segmented 超时: 已成功 %s/%s, 已尝试 %s/%s",
                    completed,
                    target_count,
                    len(attempted_parts),
                    target_count,
                )
                return False

            live_parts = self._get_live_parts(robot)
            if loosen_mode:
                resolved_part_keys = self._get_resolved_episode_part_keys(
                    robot=robot,
                    live_parts=live_parts,
                    episode_part_keys=episode_part_keys,
                    attempted_parts=attempted_parts,
                )
                if len(resolved_part_keys) >= target_count:
                    break

            candidate_parts = [
                part
                for part in live_parts
                if self._part_key(part) not in attempted_parts
                and (not episode_part_keys or self._part_key(part) in episode_part_keys)
            ]
            part = self._select_next_part(robot, candidate_parts)
            if part is None:
                self._log_waiting_for_candidate(
                    robot=robot,
                    live_parts=live_parts,
                    candidate_parts=candidate_parts,
                    completed=completed,
                    target_count=target_count,
                    attempted_count=len(attempted_parts),
                )
                if loosen_mode and self._all_remaining_episode_parts_past_grab_zone(
                    robot=robot,
                    live_parts=live_parts,
                    episode_part_keys=episode_part_keys,
                    attempted_parts=attempted_parts,
                ):
                    logging.warning(
                        "Conveyor segmented: 本轮剩余目标零件都已离开抓取区或不存在，"
                        "结束当前 long 轮次 (当前成功 %s/%s, 已尝试 %s/%s)",
                        completed,
                        target_count,
                        len(attempted_parts),
                        target_count,
                    )
                    break
                self._wait_for_parts(robot, dt, dataset=None, single_task=single_task)
                continue

            part_key = self._part_key(part)
            attempted_parts.add(part_key)
            short_buffer = (
                _FrameBufferDataset(short_dataset.features)
                if short_dataset is not None
                else None
            )
            grasp_record_dataset = _DatasetFanout(short_buffer, long_dataset)
            long_frame_start = self._get_dataset_episode_size(long_dataset)

            grasp_poses = self.compute_grasp_poses(part)
            place_poses = self.get_place_pose(robot, part, box_pos)
            success = self._run_grasp_stages(
                robot=robot,
                part=part,
                grasp_poses=grasp_poses,
                place_poses=place_poses,
                dt=dt,
                dataset=grasp_record_dataset,
                single_task=single_task,
                check_success_fn=lambda p=part: self.check_grasp_success(robot, p),
            )
            if not success:
                if not loosen_mode:
                    return False

                # loosen 模式下不立即重置场景，但当前失败尝试应先收尾：
                # 先显式张开右夹爪，再回到 home，避免夹爪状态和内部标志不同步，
                # 也避免下一轮看起来像“夹爪一直闭着”。
                self.move_gripper(
                    robot,
                    {"right": -1.0},
                    dt,
                    0.2,
                    dataset=None,
                    single_task=None,
                )
                self._return_right_arm_home(
                    robot=robot,
                    dt=dt,
                    dataset=None,
                    single_task=None,
                )
                logging.warning(
                    "Conveyor segmented: 零件 %s 抓取/放置失败，继续处理后续零件 "
                    "(当前成功 %s/%s, 已尝试 %s/%s)",
                    part_key,
                    completed,
                    target_count,
                    len(attempted_parts),
                    target_count,
                )
                continue

            self._return_right_arm_home(
                robot=robot,
                dt=dt,
                dataset=grasp_record_dataset,
                single_task=single_task,
            )

            self._processed_paths.add(part_key)
            completed += 1

            long_frame_end = self._get_dataset_episode_size(long_dataset)
            if long_frame_start is not None and long_frame_end is not None:
                self._current_long_episode_parts.append(
                    self._build_part_frame_record(
                        robot=robot,
                        part=part,
                        frame_start_index=long_frame_start,
                        frame_end_index=long_frame_end - 1,
                    )
                )

            if short_buffer is not None and short_buffer.frames:
                short_part_record = self._build_part_frame_record(
                    robot=robot,
                    part=part,
                    frame_start_index=0,
                    frame_end_index=len(short_buffer.frames) - 1,
                )
                short_dataset.clear_episode_buffer()
                for frame in short_buffer.frames:
                    short_dataset.add_frame(copy.deepcopy(frame))
                self._current_short_episode_part = dict(short_part_record)
                self._save_short_episode(short_dataset)

        if completed < target_count:
            logging.warning(
                "Conveyor segmented: 本轮已走完 %s 个零件，但仅成功 %s/%s，"
                "因此不保存 long episode",
                len(attempted_parts),
                completed,
                target_count,
            )
            return False

        return self._check_processed_parts_in_correct_bins(robot, target_count)

    def _log_waiting_for_candidate(
        self,
        robot,
        live_parts: list[dict],
        candidate_parts: list[dict],
        completed: int,
        target_count: int,
        attempted_count: int,
    ) -> None:
        now = time.perf_counter()
        if now - getattr(self, "_last_wait_candidate_log_time", 0.0) < 2.0:
            return
        self._last_wait_candidate_log_time = now

        x_values = []
        for part in live_parts or []:
            pos = np.asarray(part.get("position", []), dtype=float)
            if pos.shape[0] >= 1:
                x_values.append(float(pos[0]))
        grab_min, grab_max = self._get_grab_zone(robot)
        record_start_x = self._get_record_start_x(robot)
        logging.info(
            "Conveyor segmented: 等待候选零件 completed=%s/%s attempted=%s "
            "live=%s candidate=%s x_range=%s grab=[%.3f, %.3f] record_start_x=%.3f",
            completed,
            target_count,
            attempted_count,
            len(live_parts or []),
            len(candidate_parts or []),
            [round(min(x_values), 4), round(max(x_values), 4)] if x_values else [],
            grab_min,
            grab_max,
            record_start_x,
        )

    def _wait_for_episode_parts_after_reset(self, robot, dt: float) -> list[dict]:
        """Wait briefly for Task2 rigid prim poses to become readable after reset."""
        max_steps = max(1, int(2.0 / max(dt, 1e-6)))
        for step_idx in range(max_steps):
            parts = self._get_episode_parts(robot)
            if parts:
                logging.info(
                    "Conveyor segmented: reset 后读取到 %s 个零件 (等待 %s 步)",
                    len(parts),
                    step_idx,
                )
                return parts
            robot.step(render=True)
        return []

    def _get_resolved_episode_part_keys(
        self,
        robot,
        live_parts: list[dict],
        episode_part_keys: list[str],
        attempted_parts: set[str],
    ) -> set[str]:
        if not episode_part_keys:
            return set(attempted_parts)

        resolved_world_x = self._get_resolved_world_x(robot)
        live_parts_by_key = {
            self._part_key(part): part
            for part in live_parts
        }
        resolved_part_keys = set()

        for part_key in episode_part_keys:
            if part_key in attempted_parts:
                resolved_part_keys.add(part_key)
                continue

            live_part = live_parts_by_key.get(part_key)
            if live_part is None:
                resolved_part_keys.add(part_key)
                continue

            position = np.asarray(live_part.get("position", []), dtype=float)
            if position.shape[0] >= 1 and float(position[0]) > resolved_world_x:
                resolved_part_keys.add(part_key)

        return resolved_part_keys

    def _all_remaining_episode_parts_past_grab_zone(
        self,
        robot,
        live_parts: list[dict],
        episode_part_keys: list[str],
        attempted_parts: set[str],
    ) -> bool:
        if not episode_part_keys:
            return False

        _, grab_x_max = self._get_grab_zone(robot)
        live_parts_by_key = {
            self._part_key(part): part
            for part in live_parts
        }

        has_remaining = False
        for part_key in episode_part_keys:
            if part_key in attempted_parts:
                continue

            has_remaining = True
            live_part = live_parts_by_key.get(part_key)
            if live_part is None:
                continue

            position = np.asarray(live_part.get("position", []), dtype=float)
            if position.shape[0] < 1:
                continue

            if float(position[0]) <= grab_x_max:
                return False

        return has_remaining

    def _build_part_frame_record(
        self,
        robot,
        part: dict,
        frame_start_index: int,
        frame_end_index: int,
    ) -> dict:
        scene_builder = getattr(robot, "_scene_builder", None) if robot is not None else None
        return {
            "part_name": self._get_part_name(part),
            "part_type": get_conveyor_sorting_part_type(part, scene_builder),
            "prim_path": str(part.get("prim_path", "")),
            "frame_start_index": int(frame_start_index),
            "frame_end_index": int(frame_end_index),
        }

    def _return_right_arm_home(
        self,
        robot,
        dt: float,
        dataset,
        single_task: str | None,
    ) -> None:
        """Return right arm home while optionally recording the transition."""
        steps = self._get_home_return_steps(robot)

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
            return

        target_positions = np.concatenate([target_arm_positions, target_finger_positions])
        arm_finger_indices = (
            robot._robot_interface.arm_joint_indices
            + robot._robot_interface.finger_joint_indices
        )

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
            if dataset is not None and single_task is not None:
                self._record_frame(robot, dataset, single_task)

        for _ in range(10):
            self._raise_if_keyboard_requested()
            robot.step(render=True)
            if dataset is not None and single_task is not None:
                self._record_frame(robot, dataset, single_task)

    def _save_short_episode(self, short_dataset) -> None:
        if short_dataset is None or short_dataset.episode_buffer.get("size", 0) == 0:
            return

        saved_episode_index = int(short_dataset.episode_buffer["episode_index"])
        saved_episode_length = int(short_dataset.episode_buffer["size"])
        with SuppressProgressBars():
            short_dataset.save_episode()
        self._write_short_part_info(
            short_dataset=short_dataset,
            episode_index=saved_episode_index,
            episode_length=saved_episode_length,
        )
        short_dataset.clear_episode_buffer()
        self._current_short_episode_part = None

    def _save_completed_buffers(self, short_dataset, long_dataset) -> None:
        if long_dataset is not None and long_dataset.episode_buffer.get("size", 0) > 0:
            saved_episode_index = int(long_dataset.episode_buffer["episode_index"])
            saved_episode_length = int(long_dataset.episode_buffer["size"])
            with SuppressProgressBars():
                long_dataset.save_episode()
            self._write_long_part_info(
                long_dataset=long_dataset,
                episode_index=saved_episode_index,
                episode_length=saved_episode_length,
            )
            self._current_long_episode_parts = []
        self._pending_short_episodes = []

    def _save_partial_buffers(self, short_dataset, long_dataset) -> None:
        if long_dataset is not None and long_dataset.episode_buffer.get("size", 0) > 0:
            saved_episode_index = int(long_dataset.episode_buffer["episode_index"])
            saved_episode_length = int(long_dataset.episode_buffer["size"])
            with SuppressProgressBars():
                long_dataset.save_episode()
            self._write_long_part_info(
                long_dataset=long_dataset,
                episode_index=saved_episode_index,
                episode_length=saved_episode_length,
            )

    def _save_pending_short_episodes(self, short_dataset) -> None:
        if short_dataset is None:
            return

        for staged_episode in self._pending_short_episodes:
            frames = staged_episode.get("frames", [])
            if not frames:
                continue

            short_dataset.clear_episode_buffer()
            for frame in frames:
                short_dataset.add_frame(copy.deepcopy(frame))

            self._current_short_episode_part = dict(staged_episode["part"])
            self._save_short_episode(short_dataset)

    def _write_short_part_info(
        self,
        short_dataset,
        episode_index: int,
        episode_length: int,
    ) -> None:
        if self._current_short_episode_part is None:
            return

        part_info_path = Path(short_dataset.root) / "meta" / "part_info.json"
        part_info_path.parent.mkdir(parents=True, exist_ok=True)
        part_info = self._load_part_info(part_info_path)
        episode_info = {
            "episode_index": int(episode_index),
            "episode_frame_length": int(episode_length),
            "parts": [dict(self._current_short_episode_part)],
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

    def _write_long_part_info(
        self,
        long_dataset,
        episode_index: int,
        episode_length: int,
    ) -> None:
        if not self._current_long_episode_parts:
            return

        part_info_path = Path(long_dataset.root) / "meta" / "part_info.json"
        part_info_path.parent.mkdir(parents=True, exist_ok=True)
        part_info = self._load_part_info(part_info_path)
        episode_info = {
            "episode_index": int(episode_index),
            "episode_frame_length": int(episode_length),
            "parts": list(self._current_long_episode_parts),
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

    def _build_segmented_datasets(self, robot, cfg: AutoCollectConfig):
        if not cfg.record_data:
            return None, None, cfg.single_task or cfg.task

        base_root = Path(cfg.root) if cfg.root is not None else Path("./outputs/auto_collect")
        base_repo_id = cfg.repo_id or "conveyor_sorting_segmented"
        single_task = cfg.single_task or cfg.task

        short_cfg = replace(
            cfg,
            repo_id=f"{base_repo_id}_short",
            root=base_root / self._SHORT_DIRNAME,
            single_task=single_task,
        )
        long_cfg = replace(
            cfg,
            repo_id=f"{base_repo_id}_long",
            root=base_root / self._LONG_DIRNAME,
            single_task=single_task,
        )
        short_dataset, _ = AutoCollectBase._build_dataset(robot, short_cfg)
        long_dataset, _ = AutoCollectBase._build_dataset(robot, long_cfg)
        return short_dataset, long_dataset, single_task
