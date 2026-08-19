#!/usr/bin/env python
"""Validate Recovery V2 locally, including incomplete diagnostic-only roots."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REQUIRED_METADATA_KEYS = {
    "schema_version", "pilot_index", "saved_episode_id", "collection_attempt", "random_seed",
    "scenario_type", "difficulty", "target_object_slot", "temporal_convention",
    "object_poses_policy_input", "objects", "final_task_success", "number_of_frames",
    "termination_reason",
}


def validate_episode_metadata(payload: dict) -> None:
    missing = REQUIRED_METADATA_KEYS - set(payload)
    if missing:
        raise ValueError(f"Recovery metadata missing keys: {sorted(missing)}")
    if payload["object_poses_policy_input"] is not False or len(payload["objects"]) != 4:
        raise ValueError("Recovery metadata object contract is invalid")


def fixed_list_to_numpy(column) -> np.ndarray:
    array = column.combine_chunks()
    return array.values.to_numpy(zero_copy_only=False).reshape(len(array), array.type.list_size)


def read_jsonl(path: Path, validate: bool = False) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                payload = json.loads(line)
                if validate:
                    validate_episode_metadata(payload)
                rows.append(payload)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def video_change(path: Path) -> dict:
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24")
            if len(frames) < 12:
                frames.append(image)
            else:
                frames[-1] = image
    if len(frames) < 2:
        return {"frames_sampled": len(frames), "changing": False, "mean_abs_delta": 0.0}
    deltas = [float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) for a, b in zip(frames, frames[1:])]
    return {"frames_sampled": len(frames), "changing": bool(max(deltas) > 0.25), "mean_abs_delta": float(np.mean(deltas))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/part_sorting_recovery_v2_pilot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    features = info.get("features", {})
    camera_keys = sorted(key for key in features if key.startswith("observation.images."))
    state_shape = tuple(features.get("observation.state", {}).get("shape", ()))
    action_shape = tuple(features.get("action", {}).get("shape", ()))
    attempts = read_jsonl(root / "meta/recovery_v2_attempts.jsonl", validate=True)
    completed_metadata = read_jsonl(root / "meta/recovery_v2_episodes.jsonl", validate=True)
    diagnostics = sorted((root / "diagnostics/failed_attempts").glob("*/failure.json"))
    parquet_paths = sorted((root / "data").rglob("*.parquet")) if (root / "data").exists() else []

    local_summary = {
        "root": str(root),
        "declared_episodes": int(info.get("total_episodes", 0)),
        "attempt_metadata_rows": len(attempts),
        "completed_metadata_rows": len(completed_metadata),
        "failed_attempt_diagnostics": [str(path) for path in diagnostics],
        "metadata_jsonl_valid": True,
        "camera_keys": camera_keys,
        "observation_state_shape": state_shape,
        "action_shape": action_shape,
    }
    if not parquet_paths or int(info.get("total_episodes", 0)) <= 0:
        result = {
            **local_summary,
            "dataset_load": "not_attempted_zero_completed_episodes",
            "validation_error": "No completed local episodes; refusing Hugging Face fallback.",
        }
        encoded = json.dumps(result, indent=2)
        print(encoded)
        if args.output:
            args.output.write_text(encoded + "\n", encoding="utf-8")
        return 2

    if len(camera_keys) != 4 or state_shape != (48,) or action_shape != (20,):
        raise RuntimeError(f"Dataset contract mismatch: cameras={camera_keys}, state={state_shape}, action={action_shape}")

    from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded = LeRobotDataset(repo_id=args.repo_id, root=root, download_videos=False)
    if loaded.num_episodes <= 0:
        raise RuntimeError("Recovery V2 local dataset loaded but contains no episodes")

    sse, counts = Counter(), Counter()
    episodes_seen, frames_seen = set(), 0
    frame_counts = Counter()
    for path in parquet_paths:
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
        state = fixed_list_to_numpy(table["observation.state"]).astype(np.float64)
        action = fixed_list_to_numpy(table["action"]).astype(np.float64)
        episode_index = table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False)
        episodes_seen.update(int(value) for value in np.unique(episode_index))
        frames_seen += len(state)
        frame_counts.update(int(value) for value in episode_index)
        sse["action_t_vs_state_t"] += float(np.square(action[:, :18] - state[:, :18]).sum())
        counts["action_t_vs_state_t"] += state[:, :18].size
        if len(state) > 1:
            adjacent = episode_index[:-1] == episode_index[1:]
            for key, lhs, rhs in (
                ("action_t_vs_state_t_plus_1", action[:-1, :18], state[1:, :18]),
                ("action_t_plus_1_vs_state_t", action[1:, :18], state[:-1, :18]),
            ):
                lhs, rhs = lhs[adjacent], rhs[adjacent]
                sse[key] += float(np.square(lhs - rhs).sum())
                counts[key] += lhs.size
    rmse = {key: math.sqrt(sse[key] / counts[key]) for key in sse if counts[key]}
    if len(completed_metadata) != len(episodes_seen):
        raise RuntimeError(f"Metadata/dataset mismatch: metadata={len(completed_metadata)}, parquet={len(episodes_seen)}")

    videos = sorted((root / "videos").rglob("*.mp4")) if (root / "videos").exists() else []
    video_validation = {str(path.relative_to(root)): video_change(path) for path in videos}
    result = {
        **local_summary,
        "dataset_load": "success",
        "episodes": len(episodes_seen),
        "frames": frames_seen,
        "frame_counts": dict(sorted(frame_counts.items())),
        "fps": info["fps"],
        "video_validation": video_validation,
        "all_camera_videos_changing": bool(video_validation) and all(item["changing"] for item in video_validation.values()),
        "joint_position_rmse": rmse,
        "achieved_fps_by_attempt": [row.get("achieved_fps") for row in attempts],
        "causal_ordering": "observe -> select/record action[t] -> apply -> world.step",
        "alignment_interpretation": "Code ordering is primary evidence; adjacent RMSE values only quantify tracking lag.",
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
