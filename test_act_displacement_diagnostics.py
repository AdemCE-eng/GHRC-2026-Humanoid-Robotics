from __future__ import annotations

import unittest

from act_displacement_diagnostics import ObjectOutcomeTracker, timing_summary


def pose(x: float, y: float, z: float = 1.04) -> dict:
    return {
        "prim_path": "/Root/Part_0",
        "object_type": "part_a",
        "position": [x, y, z],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }


class ObjectOutcomeTrackerTests(unittest.TestCase):
    def tracker(self) -> ObjectOutcomeTracker:
        return ObjectOutcomeTracker(
            threshold_m=0.03,
            settle_s=0.0,
            box_bounds={"min_m": [1.0, 0.0, 0.9], "max_m": [1.4, 0.6, 1.3]},
        )

    def test_jitter_does_not_trigger(self) -> None:
        tracker = self.tracker()
        tracker.initialize([pose(0.75, 0.28)])
        for step, offset in enumerate((0.002, -0.003, 0.004, -0.002), start=1):
            events, _ = tracker.update([pose(0.75 + offset, 0.28)], step * 0.1, step)
            self.assertEqual(events, [])

    def test_sustained_three_centimeter_tabletop_move_triggers_once(self) -> None:
        tracker = self.tracker()
        tracker.initialize([pose(0.75, 0.28)])
        first, _ = tracker.update([pose(0.781, 0.28)], 0.1, 1)
        second, _ = tracker.update([pose(0.782, 0.28)], 0.2, 2)
        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        self.assertGreaterEqual(second[0]["planar_displacement_m"], 0.03)
        third, _ = tracker.update([pose(0.783, 0.28)], 0.3, 3)
        self.assertEqual(third, [])

    def test_lifted_object_is_not_classified_as_tabletop_push(self) -> None:
        tracker = self.tracker()
        tracker.initialize([pose(0.75, 0.28)])
        for step in (1, 2, 3):
            events, _ = tracker.update([pose(0.80, 0.28, 1.10)], step * 0.1, step)
            self.assertEqual(events, [])

    def test_placement_within_window_marks_recovery(self) -> None:
        tracker = self.tracker()
        tracker.placement_dwell_s = 0.2
        tracker.initialize([pose(0.75, 0.28)])
        tracker.update([pose(0.79, 0.28)], 0.1, 1)
        events, _ = tracker.update([pose(0.79, 0.28)], 0.2, 2)
        self.assertEqual(len(events), 1)
        tracker.update([pose(1.2, 0.3)], 1.0, 3)
        _, placements = tracker.update([pose(1.2, 0.3)], 1.3, 4)
        self.assertEqual(len(placements), 1)
        self.assertTrue(tracker.finalize()[0]["recovered"])

    def test_timing_summary(self) -> None:
        summary = timing_summary([0.1, 0.05], [0.02, 0.04])
        self.assertAlmostEqual(summary["mean_hz"], 15.0)
        self.assertAlmostEqual(summary["mean_loop_duration_ms"], 75.0)
        self.assertAlmostEqual(summary["mean_chunk_inference_ms"], 30.0)


if __name__ == "__main__":
    unittest.main()
