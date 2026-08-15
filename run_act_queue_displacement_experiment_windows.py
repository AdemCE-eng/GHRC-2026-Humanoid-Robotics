"""Run paired normal/replan ACT Part Sorting trials in separate Isaac Sim processes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ISAAC_PYTHON = Path(r"D:\guedr\Downloads\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat")
EVALUATOR = PROJECT_ROOT / "run_act_part_sorting_windows.py"
CHECKPOINT_200K = PROJECT_ROOT / "checkpoints" / "part_sorting_act_200k"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments" / "act_queue_displacement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=int,
        choices=(2, 3, 4, 5),
        default=3,
        help="Paired trials to run; 2 is reserved for extending an ambiguous 3-pair batch.",
    )
    parser.add_argument("--duration", type=float, default=240.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_200K)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--displacement-threshold-cm", type=float, default=3.0)
    parser.add_argument("--recovery-window", type=float, default=20.0)
    return parser.parse_args()


def pooled_recovery_rate(rows: list[dict]) -> float | None:
    recoveries = [item for row in rows for item in row.get("recoveries", [])]
    if not recoveries:
        return None
    return sum(bool(item.get("recovered")) for item in recoveries) / len(recoveries)


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def build_aggregate(rows: list[dict], results_dir: Path, timestamp: str) -> tuple[Path, Path]:
    by_experiment = {
        experiment: [row for row in rows if row["experiment"] == experiment]
        for experiment in ("normal", "replan-on-displacement")
    }
    aggregate = {}
    for experiment, group in by_experiment.items():
        aggregate[experiment] = {
            "runs": len(group),
            "total_objects_placed": sum(row["objects_placed"] for row in group),
            "possible_objects": sum(row["objects_total"] for row in group),
            "full_completions": sum(bool(row["full_task_completion"]) for row in group),
            "full_completion_rate": (
                sum(bool(row["full_task_completion"]) for row in group) / len(group) if group else None
            ),
            "significant_displacement_events": sum(row["significant_displacement_events"] for row in group),
            "replans": sum(row["replans"] for row in group),
            "recovery_success_rate": pooled_recovery_rate(group),
            "average_successful_placement_time_s": mean_or_none(
                [row.get("average_successful_placement_time_s") for row in group]
            ),
            "mean_control_hz": mean_or_none([row.get("timing", {}).get("mean_hz") for row in group]),
            "mean_chunk_inference_ms": mean_or_none(
                [row.get("timing", {}).get("mean_chunk_inference_ms") for row in group]
            ),
            "large_scheduled_chunk_boundaries": sum(
                row.get("chunk_boundary_discontinuity", {}).get("large_boundary_count", 0) for row in group
            ),
        }

    normal = aggregate["normal"]
    replan = aggregate["replan-on-displacement"]
    normal_recovery = normal["recovery_success_rate"]
    replan_recovery = replan["recovery_success_rate"]
    object_gain = replan["total_objects_placed"] - normal["total_objects_placed"]
    comparable_events = min(
        normal["significant_displacement_events"],
        replan["significant_displacement_events"],
    )
    recovery_gain = (
        replan_recovery - normal_recovery
        if replan_recovery is not None and normal_recovery is not None
        else None
    )
    if comparable_events >= 3 and object_gain >= 2 and recovery_gain is not None and recovery_gain >= 0.30:
        classification = "1. Strong evidence stale queued actions are a major failure cause"
    elif object_gain > 0 or (recovery_gain is not None and recovery_gain > 0.0):
        classification = "2. Some evidence but not enough"
    else:
        classification = (
            "3. No meaningful improvement so visual retargeting or recovery data is more likely the main limitation"
        )

    paired = []
    seeds = sorted({row["seed"] for row in rows})
    for seed in seeds:
        normal_row = next(row for row in by_experiment["normal"] if row["seed"] == seed)
        replan_row = next(row for row in by_experiment["replan-on-displacement"] if row["seed"] == seed)
        paired.append(
            {
                "seed": seed,
                "normal_objects": normal_row["objects_placed"],
                "replan_objects": replan_row["objects_placed"],
                "object_delta": replan_row["objects_placed"] - normal_row["objects_placed"],
                "normal_full": normal_row["full_task_completion"],
                "replan_full": replan_row["full_task_completion"],
                "normal_displacements": normal_row["significant_displacement_events"],
                "replan_displacements": replan_row["significant_displacement_events"],
                "normal_recovery_rate": pooled_recovery_rate([normal_row]),
                "replan_recovery_rate": pooled_recovery_rate([replan_row]),
                "normal_mean_hz": normal_row["timing"].get("mean_hz"),
                "replan_mean_hz": replan_row["timing"].get("mean_hz"),
                "replans": replan_row["replans"],
            }
        )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "aggregate": aggregate,
        "paired": paired,
        "object_gain": object_gain,
        "recovery_rate_gain": recovery_gain,
        "classification": classification,
        "run_summaries": [row["run_id"] for row in rows],
    }
    json_path = results_dir / f"paired_report_{timestamp}.json"
    csv_path = results_dir / f"paired_report_{timestamp}.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    return json_path, csv_path


def main() -> int:
    args = parse_args()
    if not ISAAC_PYTHON.is_file():
        raise FileNotFoundError(ISAAC_PYTHON)
    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError(args.checkpoint)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = args.results_dir / f"paired_manifest_{timestamp}.json"
    manifest = {
        "started_at": datetime.now().astimezone().isoformat(),
        "pairs": args.pairs,
        "duration_s": args.duration,
        "checkpoint": str(args.checkpoint.resolve()),
        "seeds": list(range(args.first_seed, args.first_seed + args.pairs)),
        "runs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summaries = []
    for seed in manifest["seeds"]:
        for experiment in ("normal", "replan-on-displacement"):
            run_id = f"paired_{timestamp}_seed{seed}_{experiment}"
            command = [
                "cmd.exe",
                "/d",
                "/c",
                str(ISAAC_PYTHON),
                str(EVALUATOR),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--experiment",
                experiment,
                "--seed",
                str(seed),
                "--duration",
                str(args.duration),
                "--fps",
                str(args.fps),
                "--displacement-threshold-cm",
                str(args.displacement_threshold_cm),
                "--recovery-window",
                str(args.recovery_window),
                "--results-dir",
                str(args.results_dir.resolve()),
                "--run-id",
                run_id,
            ]
            print(f"\n=== seed={seed} experiment={experiment} ===", flush=True)
            child_env = dict(os.environ)
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
            console_path = args.results_dir / f"{run_id}.console.log"
            with console_path.open("w", encoding="utf-8") as console_stream:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    check=False,
                    env=child_env,
                    stdout=console_stream,
                    stderr=subprocess.STDOUT,
                )
            summary_path = args.results_dir / f"{run_id}.summary.json"
            run_record = {
                "seed": seed,
                "experiment": experiment,
                "run_id": run_id,
                "return_code": completed.returncode,
                "summary_path": str(summary_path),
                "console_path": str(console_path),
            }
            manifest["runs"].append(run_record)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if completed.returncode != 0 or not summary_path.is_file():
                print(f"Run failed or summary missing: {run_record}", file=sys.stderr)
                return completed.returncode or 1
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries.append(summary)
            print(
                f"completed seed={seed} experiment={experiment} "
                f"objects={summary['objects_placed']}/{summary['objects_total']} "
                f"displacements={summary['significant_displacement_events']} "
                f"replans={summary['replans']} mean_hz={summary['timing'].get('mean_hz'):.2f}",
                flush=True,
            )

    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    json_report, csv_report = build_aggregate(summaries, args.results_dir, timestamp)
    manifest["paired_json_report"] = str(json_report)
    manifest["paired_csv_report"] = str(csv_report)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Paired report: {json_report}")
    print(f"Paired table: {csv_report}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
