"""
Metrics collector for the multi-camera pipeline.

Accumulates per-camera and global statistics during a pipeline run, then
serialises everything to JSON and/or prints a human-readable summary.

Definitions
-----------
- **FPS (global)**: total frames processed / elapsed wall-clock seconds.
- **FPS (per-camera)**: global FPS / number of active cameras.
- **Latency end-to-end**: time from frame capture (``perf_counter_ns`` in
  the producer) to the moment the inference result is available.
- **Inference time**: wall-clock duration of ``session.run()``.
- **Queue wait**: time a frame packet spends sitting in the queue.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerCameraMetrics:
    """Accumulated metrics for a single camera."""

    camera_id: str
    frames_captured: int = 0
    frames_inferred: int = 0
    frames_dropped: int = 0
    read_errors: int = 0


@dataclass
class _LatencyBucket:
    """Internal helper that accumulates timing samples for percentile calc."""

    _samples: list[float] = field(default_factory=list)

    def add(self, value_s: float) -> None:
        self._samples.append(value_s)

    @property
    def count(self) -> int:
        return len(self._samples)

    def percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        return float(np.percentile(self._samples, p))

    @property
    def mean(self) -> float:
        return float(np.mean(self._samples)) if self._samples else 0.0


class MetricsCollector:
    """Collects per-camera and global metrics during a pipeline run.

    Parameters
    ----------
    camera_ids:
        List of camera identifiers that will be tracked.
    """

    def __init__(self, camera_ids: list[str]) -> None:
        self.camera_ids = list(camera_ids)
        self.per_camera: dict[str, PerCameraMetrics] = {
            cid: PerCameraMetrics(camera_id=cid) for cid in camera_ids
        }

        # Global counters
        self.total_frames_inferred: int = 0
        self.total_batches: int = 0
        self.total_detections: int = 0

        # Timing accumulators
        self._inference_times = _LatencyBucket()
        self._e2e_latencies = _LatencyBucket()
        self._queue_waits = _LatencyBucket()
        self._queue_sizes: dict[str, list[int]] = {cid: [] for cid in camera_ids}

        # Wall-clock bookkeeping
        self._start_ns: int = 0
        self._end_ns: int = 0

    # ── Recording API ───────────────────────────────────────────────────

    def start_clock(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def stop_clock(self) -> None:
        self._end_ns = time.perf_counter_ns()

    def record_batch(
        self,
        camera_ids: list[str],
        inference_time_s: float,
        capture_times_ns: list[int],
        dequeue_times_ns: list[int],
        detections: int,
    ) -> None:
        """Record metrics for one completed batch.

        Parameters
        ----------
        camera_ids:
            Camera IDs of frames in this batch.
        inference_time_s:
            Wall-clock seconds spent in ``session.run()``.
        capture_times_ns:
            ``perf_counter_ns`` at capture for each frame.
        dequeue_times_ns:
            ``perf_counter_ns`` when each frame was dequeued.
        detections:
            Number of valid detections in this batch.
        """
        now_ns = time.perf_counter_ns()
        self.total_batches += 1
        self.total_detections += detections

        self._inference_times.add(inference_time_s)

        for cid, cap_ns, deq_ns in zip(camera_ids, capture_times_ns, dequeue_times_ns):
            cam = self.per_camera.get(cid)
            if cam:
                cam.frames_inferred += 1

            # End-to-end latency: capture → inference done
            e2e_s = (now_ns - cap_ns) / 1e9
            self._e2e_latencies.add(e2e_s)

            # Queue wait: capture → dequeue
            qw_s = (deq_ns - cap_ns) / 1e9
            self._queue_waits.add(qw_s)

            self.total_frames_inferred += 1

    def record_queue_size(self, camera_id: str, size: int) -> None:
        """Snapshot current queue size for a camera."""
        if camera_id in self._queue_sizes:
            self._queue_sizes[camera_id].append(size)

    def sync_producer_stats(self, camera_id: str, captured: int, dropped: int, errors: int) -> None:
        """Copy final counters from a VideoProducer into per-camera metrics."""
        cam = self.per_camera.get(camera_id)
        if cam:
            cam.frames_captured = captured
            cam.frames_dropped = dropped
            cam.read_errors = errors

    # ── Computed metrics ────────────────────────────────────────────────

    @property
    def elapsed_s(self) -> float:
        return (self._end_ns - self._start_ns) / 1e9

    @property
    def global_fps(self) -> float:
        e = self.elapsed_s
        return self.total_frames_inferred / e if e > 0 else 0.0

    @property
    def per_camera_fps(self) -> float:
        n = len(self.camera_ids)
        return self.global_fps / n if n > 0 else 0.0

    # ── VRAM ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_vram_mb() -> float | None:
        """Query VRAM usage via nvidia-smi (best effort)."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            )
            return float(out.strip().split("\n")[0])
        except Exception:
            return None

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return all metrics as a JSON-serialisable dict."""
        vram = self._get_vram_mb()

        per_cam = {}
        for cid in self.camera_ids:
            cam = self.per_camera[cid]
            qs = self._queue_sizes.get(cid, [])
            per_cam[cid] = {
                "frames_captured": cam.frames_captured,
                "frames_inferred": cam.frames_inferred,
                "frames_dropped": cam.frames_dropped,
                "read_errors": cam.read_errors,
                "queue_size_avg": float(np.mean(qs)) if qs else 0.0,
                "queue_size_max": int(np.max(qs)) if qs else 0,
            }

        return {
            "global": {
                "total_frames_inferred": self.total_frames_inferred,
                "total_batches": self.total_batches,
                "total_detections": self.total_detections,
                "elapsed_s": round(self.elapsed_s, 4),
                "fps_global": round(self.global_fps, 2),
                "fps_per_camera": round(self.per_camera_fps, 2),
                "inference_time_avg_ms": round(self._inference_times.mean * 1000, 2),
                "inference_time_p95_ms": round(self._inference_times.percentile(95) * 1000, 2),
                "latency_e2e_p50_ms": round(self._e2e_latencies.percentile(50) * 1000, 2),
                "latency_e2e_p95_ms": round(self._e2e_latencies.percentile(95) * 1000, 2),
                "queue_wait_avg_ms": round(self._queue_waits.mean * 1000, 2),
                "vram_used_mb": vram,
                "num_cameras": len(self.camera_ids),
            },
            "per_camera": per_cam,
        }

    def save(self, path: str | Path) -> None:
        """Write metrics to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        p.write_text(json.dumps(data, indent=2))
        logger.info("Metrics saved to %s", p)

    # ── Human-readable summary ──────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a formatted Markdown summary to stdout."""
        d = self.to_dict()
        g = d["global"]

        print()
        print("=" * 72)
        print("  PIPELINE METRICS SUMMARY")
        print("=" * 72)
        print()
        print("| Global Metric                  | Value               |")
        print("|:-------------------------------|:--------------------|")
        print(f"| Cameras                        | {g['num_cameras']}                   |")
        print(f"| Total frames inferred          | {g['total_frames_inferred']:<20}|")
        print(f"| Total batches                  | {g['total_batches']:<20}|")
        print(f"| Total detections               | {g['total_detections']:<20}|")
        print(f"| Elapsed time                   | {g['elapsed_s']:.2f} s{'':<14}|")
        print(f"| **FPS global**                 | **{g['fps_global']:.2f}**{'':<14}|")
        print(f"| **FPS per camera**             | **{g['fps_per_camera']:.2f}**{'':<14}|")
        print(f"| Inference time (avg)           | {g['inference_time_avg_ms']:.2f} ms{'':<12}|")
        print(f"| Inference time (p95)           | {g['inference_time_p95_ms']:.2f} ms{'':<12}|")
        print(f"| Latency e2e (p50)              | {g['latency_e2e_p50_ms']:.2f} ms{'':<12}|")
        print(f"| Latency e2e (p95)              | {g['latency_e2e_p95_ms']:.2f} ms{'':<12}|")
        print(f"| Queue wait (avg)               | {g['queue_wait_avg_ms']:.2f} ms{'':<12}|")
        vram_str = f"{g['vram_used_mb']:.0f} MB" if g['vram_used_mb'] is not None else "N/A"
        print(f"| VRAM used                      | {vram_str:<20}|")
        print()

        # Per-camera table
        print("| Camera   | Captured | Inferred | Dropped | Errors | Q avg | Q max |")
        print("|:---------|:---------|:---------|:--------|:-------|:------|:------|")
        for cid, cm in d["per_camera"].items():
            print(
                f"| {cid:<8s} "
                f"| {cm['frames_captured']:<8} "
                f"| {cm['frames_inferred']:<8} "
                f"| {cm['frames_dropped']:<7} "
                f"| {cm['read_errors']:<6} "
                f"| {cm['queue_size_avg']:.1f}{'':>3} "
                f"| {cm['queue_size_max']:<5} |"
            )

        print()
        print("=" * 72)
