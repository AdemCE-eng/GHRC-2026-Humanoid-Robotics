"""Pure diagnostic helpers for the ACT Part Sorting A/B experiment.

Ground-truth poses consumed here are deliberately kept outside the policy
observation/preprocessing/action path.  They are used only for event detection
and outcome measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def pose_payload(raw_pose: dict[str, Any]) -> dict[str, Any]:
    """Return a stable, JSON-ready object pose record."""
    return {
        "object_id": str(raw_pose["prim_path"]),
        "object_type": raw_pose.get("object_type"),
        "position_m": [float(value) for value in raw_pose["position"]],
        "orientation_xyzw": [float(value) for value in raw_pose["orientation"]],
    }


def inside_box(position: list[float], bounds: dict[str, list[float]] | None, margin_m: float = 0.01) -> bool:
    """Conservatively test whether an object center is inside the box world AABB."""
    if not bounds:
        return False
    lower = np.asarray(bounds["min_m"], dtype=np.float64)
    upper = np.asarray(bounds["max_m"], dtype=np.float64)
    point = np.asarray(position, dtype=np.float64)
    return bool(
        lower[0] + margin_m <= point[0] <= upper[0] - margin_m
        and lower[1] + margin_m <= point[1] <= upper[1] - margin_m
        and lower[2] - 0.02 <= point[2] <= upper[2] + 0.08
    )


@dataclass
class ObjectOutcomeTracker:
    """Detect sustained tabletop displacements and conservative placements.

    A displacement is >= ``threshold_m`` horizontal motion from the object's
    last accepted anchor, sustained for ``confirmation_samples`` observations.
    Objects more than ``lift_exclusion_m`` above their initial table height and
    objects already in the destination box are excluded.  The accepted pose
    becomes the next anchor, and a refractory window prevents repeated triggers.

    Recovery is intentionally strict and objective: the same displaced object
    must be placed in the destination box within ``recovery_window_s``.
    """

    threshold_m: float = 0.03
    confirmation_samples: int = 2
    refractory_s: float = 2.0
    settle_s: float = 1.0
    lift_exclusion_m: float = 0.04
    placement_dwell_s: float = 0.5
    recovery_window_s: float = 20.0
    box_bounds: dict[str, list[float]] | None = None
    anchors: dict[str, dict[str, Any]] = field(default_factory=dict)
    initial_z: dict[str, float] = field(default_factory=dict)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    last_event_time: dict[str, float] = field(default_factory=dict)
    placement_candidate_since: dict[str, float] = field(default_factory=dict)
    placed_at: dict[str, float] = field(default_factory=dict)
    pending_recoveries: list[dict[str, Any]] = field(default_factory=list)

    def initialize(self, poses: list[dict[str, Any]]) -> None:
        for raw_pose in poses:
            pose = pose_payload(raw_pose)
            object_id = pose["object_id"]
            self.anchors[object_id] = pose
            self.initial_z[object_id] = pose["position_m"][2]
            self.candidate_counts[object_id] = 0
            self.last_event_time[object_id] = float("-inf")

    def update(
        self,
        poses: list[dict[str, Any]],
        elapsed_s: float,
        control_step: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        displacements: list[dict[str, Any]] = []
        placements: list[dict[str, Any]] = []
        current = {pose_payload(item)["object_id"]: pose_payload(item) for item in poses}

        for object_id, pose in current.items():
            if object_id not in self.anchors:
                self.anchors[object_id] = pose
                self.initial_z[object_id] = pose["position_m"][2]
                self.candidate_counts[object_id] = 0
                self.last_event_time[object_id] = float("-inf")

            is_in_box = inside_box(pose["position_m"], self.box_bounds)
            if is_in_box and object_id not in self.placed_at:
                since = self.placement_candidate_since.setdefault(object_id, elapsed_s)
                if elapsed_s - since >= self.placement_dwell_s:
                    self.placed_at[object_id] = elapsed_s
                    placement = {
                        "event": "placement",
                        "timestamp_s": elapsed_s,
                        "control_step": control_step,
                        "object_id": object_id,
                        "object_type": pose.get("object_type"),
                        "pose": pose,
                    }
                    placements.append(placement)
                    for recovery in self.pending_recoveries:
                        if recovery["object_id"] != object_id or recovery["recovered"] is not None:
                            continue
                        delay = elapsed_s - recovery["timestamp_s"]
                        if delay <= self.recovery_window_s:
                            recovery["recovered"] = True
                            recovery["recovery_time_s"] = delay
                            recovery["recovery_control_step"] = control_step
            elif not is_in_box:
                self.placement_candidate_since.pop(object_id, None)

            if object_id in self.placed_at or elapsed_s < self.settle_s:
                continue

            anchor = self.anchors[object_id]
            old_position = np.asarray(anchor["position_m"], dtype=np.float64)
            new_position = np.asarray(pose["position_m"], dtype=np.float64)
            planar_m = float(np.linalg.norm(new_position[:2] - old_position[:2]))
            displacement_m = float(np.linalg.norm(new_position - old_position))
            on_table = new_position[2] <= self.initial_z[object_id] + self.lift_exclusion_m
            outside_box = not is_in_box
            outside_refractory = elapsed_s - self.last_event_time[object_id] >= self.refractory_s

            if planar_m >= self.threshold_m and on_table and outside_box and outside_refractory:
                self.candidate_counts[object_id] += 1
            else:
                self.candidate_counts[object_id] = 0

            if self.candidate_counts[object_id] < self.confirmation_samples:
                continue

            event = {
                "event": "displacement",
                "timestamp_s": elapsed_s,
                "control_step": control_step,
                "object_id": object_id,
                "object_type": pose.get("object_type"),
                "old_pose": anchor,
                "new_pose": pose,
                "planar_displacement_m": planar_m,
                "displacement_m": displacement_m,
            }
            displacements.append(event)
            self.pending_recoveries.append(
                {
                    "object_id": object_id,
                    "object_type": pose.get("object_type"),
                    "timestamp_s": elapsed_s,
                    "control_step": control_step,
                    "deadline_s": elapsed_s + self.recovery_window_s,
                    "recovered": None,
                    "recovery_time_s": None,
                    "recovery_control_step": None,
                }
            )
            self.anchors[object_id] = pose
            self.last_event_time[object_id] = elapsed_s
            self.candidate_counts[object_id] = 0

        for recovery in self.pending_recoveries:
            if recovery["recovered"] is None and elapsed_s > recovery["deadline_s"]:
                recovery["recovered"] = False

        return displacements, placements

    def finalize(self) -> list[dict[str, Any]]:
        for recovery in self.pending_recoveries:
            if recovery["recovered"] is None:
                recovery["recovered"] = False
        return self.pending_recoveries


def timing_summary(loop_durations_s: list[float], inference_durations_s: list[float]) -> dict[str, Any]:
    """Summarize measured loop frequency and chunk-generation inference time."""
    durations = np.asarray(loop_durations_s, dtype=np.float64)
    inferences = np.asarray(inference_durations_s, dtype=np.float64)
    if durations.size == 0:
        return {}
    hz = 1.0 / np.maximum(durations, 1e-9)
    return {
        "mean_hz": float(hz.mean()),
        "median_hz": float(np.median(hz)),
        "minimum_hz": float(hz.min()),
        "maximum_hz": float(hz.max()),
        "mean_loop_duration_ms": float(durations.mean() * 1000.0),
        "median_loop_duration_ms": float(np.median(durations) * 1000.0),
        "chunk_inference_count": int(inferences.size),
        "mean_chunk_inference_ms": float(inferences.mean() * 1000.0) if inferences.size else None,
        "median_chunk_inference_ms": float(np.median(inferences) * 1000.0) if inferences.size else None,
        "minimum_chunk_inference_ms": float(inferences.min() * 1000.0) if inferences.size else None,
        "maximum_chunk_inference_ms": float(inferences.max() * 1000.0) if inferences.size else None,
    }
