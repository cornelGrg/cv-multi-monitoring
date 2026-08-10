"""Run the controlled Phase 3 benchmark pair and report three-run medians."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = {
    "detection_compute": PROJECT_ROOT / "configs/benchmark_realtime_detection_compute.yaml",
    "tracking_compute": PROJECT_ROOT / "configs/benchmark_realtime_tracking_compute.yaml",
}
MEDIAN_KEYS = (
    "fps_input_global",
    "fps_global",
    "global_drop_rate_pct",
    "inference_time_avg_ms",
    "inference_time_p95_ms",
    "latency_e2e_p50_ms",
    "latency_e2e_p95_ms",
    "queue_wait_avg_ms",
    "tracking_time_avg_ms",
    "tracking_time_p95_ms",
    "vram_used_mb",
)


def _metrics_path(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return PROJECT_ROOT / config["output"]["metrics_file"]


def _median_metrics(runs: list[dict]) -> dict:
    medians = {}
    for key in MEDIAN_KEYS:
        values = [run["global"].get(key) for run in runs]
        numeric = [value for value in values if isinstance(value, (int, float))]
        medians[key] = statistics.median(numeric) if numeric else None
    return medians


def run_suite(runs_per_benchmark: int = 3) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = PROJECT_ROOT / "outputs" / "controlled_benchmarks" / timestamp
    suite_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    summary: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "runs_per_benchmark": runs_per_benchmark,
        "benchmarks": {},
    }

    for benchmark_name, config_path in BENCHMARKS.items():
        benchmark_runs = []
        benchmark_dir = suite_dir / benchmark_name
        benchmark_dir.mkdir()
        for run_number in range(1, runs_per_benchmark + 1):
            print(f"\n[{benchmark_name}] run {run_number}/{runs_per_benchmark}", flush=True)
            subprocess.run(
                [sys.executable, "-m", "src.main", "--config", str(config_path)],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )
            result = json.loads(_metrics_path(config_path).read_text(encoding="utf-8"))
            benchmark_runs.append(result)
            run_path = benchmark_dir / f"run_{run_number}.json"
            run_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        summary["benchmarks"][benchmark_name] = {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "median": _median_metrics(benchmark_runs),
        }

    detection = summary["benchmarks"]["detection_compute"]["median"]
    tracking = summary["benchmarks"]["tracking_compute"]["median"]
    detection_fps = detection["fps_global"]
    tracking_fps = tracking["fps_global"]
    summary["tracking_overhead"] = {
        "fps_delta": round(tracking_fps - detection_fps, 2),
        "fps_change_pct": round((tracking_fps - detection_fps) / detection_fps * 100.0, 2),
        "drop_rate_delta_points": round(
            tracking["global_drop_rate_pct"] - detection["global_drop_rate_pct"], 2
        ),
        "e2e_p95_delta_ms": round(
            tracking["latency_e2e_p95_ms"] - detection["latency_e2e_p95_ms"], 2
        ),
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nControlled benchmark summary: {summary_path}")
    print(json.dumps(summary["tracking_overhead"], indent=2))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration (default: 3)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    run_suite(args.runs)


if __name__ == "__main__":
    main()
