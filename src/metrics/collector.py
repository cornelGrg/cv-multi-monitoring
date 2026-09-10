"""
Metrics collector for the multi-camera pipeline.

Accumulates per-camera and global statistics during a pipeline run, then
serialises everything to JSON and/or prints a human-readable summary.

Definitions
-----------
- **FPS (global)**: total frames processed / elapsed wall-clock seconds.
- **FPS (per-camera)**: global FPS / number of active cameras.
- **Latency end-to-end**: time from frame capture through consumer processing
  and output submission; excludes asynchronous encoding completion.
- **Inference time**: wall-clock duration of ``session.run()``.
- **Queue wait**: time a frame packet spends sitting in the queue.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
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
    active_duration_s: float = 0.0
    video_frames_submitted: int = 0
    video_frames_written: int = 0
    video_frames_dropped: int = 0
    video_queue_size_avg: float = 0.0
    video_queue_size_max: int = 0
    video_worker_time_avg_ms: float = 0.0
    video_worker_time_p95_ms: float = 0.0
    line_crossing_events: int = 0


@dataclass
class _LatencyBucket:
    """Internal helper that accumulates timing samples for percentile calc."""

    _samples: list[float] | deque[float] = field(default_factory=list)

    def add(self, value_s: float) -> None:
        self._samples.append(value_s)

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
    sample_limit:
        Optional cap for distribution samples during live sessions. Totals remain
        cumulative; finite benchmark runs retain all samples by default.
    """

    def __init__(
        self, camera_ids: list[str], run_metadata: dict | None = None,
        *, sample_limit: int | None = None,
    ) -> None:
        self.camera_ids = list(camera_ids)
        self.run_metadata = dict(run_metadata or {})
        self.per_camera: dict[str, PerCameraMetrics] = {
            cid: PerCameraMetrics(camera_id=cid) for cid in camera_ids
        }

        # Global counters
        self.total_frames_inferred: int = 0
        self.total_batches: int = 0
        self.total_detections: int = 0
        self.total_track_observations: int = 0
        self.total_line_crossing_events: int = 0

        # Live sessions retain recent samples; benchmark runs retain every sample.
        if sample_limit is not None and sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        if sample_limit is not None:
            self.run_metadata["metrics_sample_limit"] = sample_limit

        def samples():
            return [] if sample_limit is None else deque(maxlen=sample_limit)

        self._inference_times = _LatencyBucket(samples())
        self._e2e_latencies = _LatencyBucket(samples())
        self._queue_waits = _LatencyBucket(samples())
        self._tracking_times = _LatencyBucket(samples())
        self._tracking_batch_times = _LatencyBucket(samples())
        self._analytics_times = _LatencyBucket(samples())
        self._output_times = _LatencyBucket(samples())
        self._video_worker_times = _LatencyBucket(samples())
        self._batch_sizes: Counter[int] = Counter()
        self._queue_sizes = {cid: samples() for cid in camera_ids}
        self._detection_counts = samples()
        self._detection_frames_seen = 0
        self._detection_confidences = samples()
        self._detection_class_histogram: Counter[str] = Counter()
        self._analytics_counts: Counter[tuple[str, str, str, str]] = Counter()
        self._latest_flow: dict[tuple[str, str], dict] = {}

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
        true_batch_size: int = 4,
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

            # End-to-end latency: capture through consumer output submission
            e2e_s = (now_ns - cap_ns) / 1e9
            self._e2e_latencies.add(e2e_s)

            # Queue wait: capture → dequeue
            qw_s = (deq_ns - cap_ns) / 1e9
            self._queue_waits.add(qw_s)

            self.total_frames_inferred += 1

        self._batch_sizes[true_batch_size] += 1

    def record_queue_size(self, camera_id: str, size: int) -> None:
        """Snapshot current queue size for a camera."""
        if camera_id in self._queue_sizes:
            self._queue_sizes[camera_id].append(size)

    def record_detection_frames(self, detections_by_frame: list[np.ndarray]) -> None:
        """Record post-filter detection counts, confidences and class IDs."""
        for detections in detections_by_frame:
            self._detection_counts.append(len(detections))
            self._detection_frames_seen += 1
            if len(detections) == 0:
                continue
            self._detection_confidences.extend(
                float(value) for value in detections[:, 4]
            )
            self._detection_class_histogram.update(
                str(int(value)) for value in detections[:, 5]
            )

    def record_tracking_time(self, duration_s: float, track_observations: int = 0) -> None:
        """Record tracker update time for one inferred source frame."""
        self._tracking_times.add(duration_s)
        self.total_track_observations += track_observations

    def record_tracking_batch_time(self, duration_s: float) -> None:
        """Record wall-clock time for a batch of sequential tracker/analytics updates."""
        self._tracking_batch_times.add(duration_s)

    def record_analytics_time(self, duration_s: float, events: list | tuple = ()) -> None:
        """Record analytics cost and structured line-crossing counters."""
        self._analytics_times.add(duration_s)
        for event in events:
            self.total_line_crossing_events += 1
            camera = self.per_camera.get(event.camera_id)
            if camera is not None:
                camera.line_crossing_events += 1
            self._analytics_counts[
                (
                    event.camera_id,
                    event.line_id,
                    event.direction,
                    event.class_name,
                )
            ] += 1

    def record_traffic_flow(self, snapshots: tuple) -> None:
        """Keep only the latest rate per camera/gate, not an unbounded history."""
        for snapshot in snapshots:
            self._latest_flow[(snapshot.camera_id, snapshot.line_id)] = asdict(snapshot)

    def record_output_time(self, duration_s: float) -> None:
        """Record CSV plus annotated-video output time for one source frame."""
        self._output_times.add(duration_s)

    def sync_producer_stats(
        self,
        camera_id: str,
        captured: int,
        dropped: int,
        errors: int,
        active_duration_s: float = 0.0,
    ) -> None:
        """Copy final counters from a VideoProducer into per-camera metrics."""
        cam = self.per_camera.get(camera_id)
        if cam:
            cam.frames_captured = captured
            cam.frames_dropped = dropped
            cam.read_errors = errors
            cam.active_duration_s = active_duration_s

    def sync_video_output_stats(self, camera_id: str, stats: object) -> None:
        """Copy final asynchronous video counters after its worker has stopped."""
        cam = self.per_camera.get(camera_id)
        if cam is None:
            return
        cam.video_frames_submitted = stats.frames_submitted
        cam.video_frames_written = stats.frames_written
        cam.video_frames_dropped = stats.frames_dropped
        cam.video_queue_size_avg = stats.queue_size_avg
        cam.video_queue_size_max = stats.queue_size_max
        cam.video_worker_time_avg_ms = stats.worker_time_avg_ms
        cam.video_worker_time_p95_ms = stats.worker_time_p95_ms
        for duration_s in stats.worker_times_s:
            self._video_worker_times.add(duration_s)

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
        total_captured = 0
        total_dropped = 0
        total_video_submitted = 0
        total_video_written = 0
        total_video_dropped = 0

        for cid in self.camera_ids:
            cam = self.per_camera[cid]
            qs = self._queue_sizes.get(cid, [])

            captured = cam.frames_captured
            dropped = cam.frames_dropped
            total_captured += captured
            total_dropped += dropped
            total_video_submitted += cam.video_frames_submitted
            total_video_written += cam.video_frames_written
            total_video_dropped += cam.video_frames_dropped

            drop_rate_pct = (dropped / captured * 100.0) if captured > 0 else 0.0
            active_duration_s = cam.active_duration_s
            input_fps_active = (
                captured / active_duration_s if active_duration_s > 0 else 0.0
            )
            output_fps_active = (
                cam.frames_inferred / active_duration_s
                if active_duration_s > 0
                else 0.0
            )

            per_cam[cid] = {
                "frames_captured": captured,
                "frames_inferred": cam.frames_inferred,
                "frames_dropped": dropped,
                "drop_rate_pct": round(drop_rate_pct, 2),
                "read_errors": cam.read_errors,
                "active_duration_s": round(active_duration_s, 4),
                "fps_input_active": round(input_fps_active, 2),
                "fps_output_active": round(output_fps_active, 2),
                "queue_size_avg": float(np.mean(qs)) if qs else 0.0,
                "queue_size_max": int(np.max(qs)) if qs else 0,
                "video_frames_submitted": cam.video_frames_submitted,
                "video_frames_written": cam.video_frames_written,
                "video_frames_dropped": cam.video_frames_dropped,
                "video_drop_rate_pct": round(
                    cam.video_frames_dropped / cam.video_frames_submitted * 100.0, 2
                ) if cam.video_frames_submitted else 0.0,
                "video_queue_size_avg": cam.video_queue_size_avg,
                "video_queue_size_max": cam.video_queue_size_max,
                "video_worker_time_avg_ms": cam.video_worker_time_avg_ms,
                "video_worker_time_p95_ms": cam.video_worker_time_p95_ms,
                "line_crossing_events": cam.line_crossing_events,
            }

        global_drop_rate = (total_dropped / total_captured * 100.0) if total_captured > 0 else 0.0
        input_fps = total_captured / self.elapsed_s if self.elapsed_s > 0 else 0.0
        video_drop_rate = (
            total_video_dropped / total_video_submitted * 100.0
            if total_video_submitted
            else 0.0
        )
        detection_counts = self._detection_counts
        detection_confidences = self._detection_confidences
        histogram_upper = max(4, max(self._batch_sizes, default=0))
        batch_size_hist = {
            str(size): self._batch_sizes[size]
            for size in range(1, histogram_upper + 1)
        }

        return {
            "run_metadata": self.run_metadata,
            "global": {
                "total_frames_inferred": self.total_frames_inferred,
                "total_batches": self.total_batches,
                "total_detections": self.total_detections,
                "total_track_observations": self.total_track_observations,
                "total_line_crossing_events": self.total_line_crossing_events,
                "batch_size_hist": batch_size_hist,
                "elapsed_s": round(self.elapsed_s, 4),
                "fps_input_global": round(input_fps, 2),
                "fps_global": round(self.global_fps, 2),
                "fps_per_camera": round(self.per_camera_fps, 2),
                "global_drop_rate_pct": round(global_drop_rate, 2),
                "inference_time_avg_ms": round(self._inference_times.mean * 1000, 2),
                "inference_time_p95_ms": round(self._inference_times.percentile(95) * 1000, 2),
                "latency_e2e_avg_ms": round(self._e2e_latencies.mean * 1000, 2),
                "latency_e2e_p50_ms": round(self._e2e_latencies.percentile(50) * 1000, 2),
                "latency_e2e_p95_ms": round(self._e2e_latencies.percentile(95) * 1000, 2),
                "queue_wait_avg_ms": round(self._queue_waits.mean * 1000, 2),
                "queue_wait_p95_ms": round(self._queue_waits.percentile(95) * 1000, 2),
                "tracking_time_avg_ms": round(self._tracking_times.mean * 1000, 2),
                "tracking_time_p95_ms": round(self._tracking_times.percentile(95) * 1000, 2),
                "tracking_batch_wall_avg_ms": round(self._tracking_batch_times.mean * 1000, 2),
                "tracking_batch_wall_p95_ms": round(
                    self._tracking_batch_times.percentile(95) * 1000, 2
                ),
                "analytics_time_avg_ms": round(self._analytics_times.mean * 1000, 2),
                "analytics_time_p95_ms": round(
                    self._analytics_times.percentile(95) * 1000, 2
                ),
                "output_time_avg_ms": round(self._output_times.mean * 1000, 2),
                "output_time_p95_ms": round(self._output_times.percentile(95) * 1000, 2),
                "video_frames_submitted": total_video_submitted,
                "video_frames_written": total_video_written,
                "video_frames_dropped": total_video_dropped,
                "video_drop_rate_pct": round(video_drop_rate, 2),
                "video_worker_time_avg_ms": round(self._video_worker_times.mean * 1000, 2),
                "video_worker_time_p95_ms": round(
                    self._video_worker_times.percentile(95) * 1000, 2
                ),
                "vram_used_mb": vram,
                "num_cameras": len(self.camera_ids),
            },
            "detection_audit": {
                "frames_observed": self._detection_frames_seen,
                "detections_per_frame_avg": round(
                    float(np.mean(detection_counts)) if detection_counts else 0.0,
                    2,
                ),
                "detections_per_frame_p50": round(
                    float(np.percentile(detection_counts, 50))
                    if detection_counts
                    else 0.0,
                    2,
                ),
                "detections_per_frame_p95": round(
                    float(np.percentile(detection_counts, 95))
                    if detection_counts
                    else 0.0,
                    2,
                ),
                "confidence_p50": round(
                    float(np.percentile(detection_confidences, 50))
                    if detection_confidences
                    else 0.0,
                    4,
                ),
                "confidence_p95": round(
                    float(np.percentile(detection_confidences, 95))
                    if detection_confidences
                    else 0.0,
                    4,
                ),
                "class_histogram_by_coco_id": dict(
                    sorted(
                        self._detection_class_histogram.items(),
                        key=lambda item: int(item[0]),
                    )
                ),
            },
            "analytics": {
                "flow": [self._latest_flow[key] for key in sorted(self._latest_flow)],
                "total_line_crossing_events": self.total_line_crossing_events,
                "counts": [
                    {
                        "camera_id": camera_id,
                        "line_id": line_id,
                        "direction": direction,
                        "class_name": class_name,
                        "count": count,
                    }
                    for (
                        camera_id,
                        line_id,
                        direction,
                        class_name,
                    ), count in sorted(self._analytics_counts.items())
                ],
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
        print(f"| Cameras                        | {g['num_cameras']:<20} |")
        print(f"| Total frames inferred          | {g['total_frames_inferred']:<20} |")
        print(f"| Total batches                  | {g['total_batches']:<20} |")
        print(f"| Total detections               | {g['total_detections']:<20} |")
        print(f"| Total track observations       | {g['total_track_observations']:<20} |")
        print(f"| Total line-crossing events     | {g['total_line_crossing_events']:<20} |")
        print(f"| Elapsed time (active)          | {g['elapsed_s']:.2f} s{'':<14} |")
        hist_str = ", ".join(f"sz{k}:{v}" for k, v in g.get('batch_size_hist', {}).items())
        print(f"| Batch size histogram           | {hist_str:<20} |")
        print(f"| **FPS input (global)**         | **{g['fps_input_global']:.2f}**{'':<13} |")
        print(f"| **FPS output (global)**        | **{g['fps_global']:.2f}**{'':<13} |")
        print(f"| **FPS output per camera**      | **{g['fps_per_camera']:.2f}**{'':<13} |")
        print(f"| **Global Drop Rate**           | **{g['global_drop_rate_pct']:.2f}%**{'':<12} |")
        print(f"| Inference time (avg)           | {g['inference_time_avg_ms']:.2f} ms{'':<11} |")
        print(f"| Inference time (p95)           | {g['inference_time_p95_ms']:.2f} ms{'':<11} |")
        print(f"| Latency e2e (avg)              | {g['latency_e2e_avg_ms']:.2f} ms{'':<11} |")
        print(f"| Latency e2e (p50)              | {g['latency_e2e_p50_ms']:.2f} ms{'':<11} |")
        print(f"| Latency e2e (p95)              | {g['latency_e2e_p95_ms']:.2f} ms{'':<11} |")
        print(f"| Queue wait (avg)               | {g['queue_wait_avg_ms']:.2f} ms{'':<11} |")
        print(f"| Queue wait (p95)               | {g['queue_wait_p95_ms']:.2f} ms{'':<11} |")
        print(f"| Tracking time (avg)            | {g['tracking_time_avg_ms']:.2f} ms{'':<11} |")
        print(f"| Tracking time (p95)            | {g['tracking_time_p95_ms']:.2f} ms{'':<11} |")
        print(f"| Tracking batch wall (avg)      | {g['tracking_batch_wall_avg_ms']:.2f} ms{'':<11} |")
        print(f"| Tracking batch wall (p95)      | {g['tracking_batch_wall_p95_ms']:.2f} ms{'':<11} |")
        print(f"| Analytics time (avg)           | {g['analytics_time_avg_ms']:.2f} ms{'':<11} |")
        print(f"| Analytics time (p95)           | {g['analytics_time_p95_ms']:.2f} ms{'':<11} |")
        print(f"| Output time (avg)              | {g['output_time_avg_ms']:.2f} ms{'':<11} |")
        print(f"| Output time (p95)              | {g['output_time_p95_ms']:.2f} ms{'':<11} |")
        print(f"| Video frames written           | {g['video_frames_written']:<20} |")
        print(f"| Video-only drop rate           | {g['video_drop_rate_pct']:.2f}%{'':<15} |")
        print(f"| Video worker time (avg)        | {g['video_worker_time_avg_ms']:.2f} ms{'':<11} |")
        print(f"| Video worker time (p95)        | {g['video_worker_time_p95_ms']:.2f} ms{'':<11} |")
        v = f"{g['vram_used_mb']:.0f} MB" if g["vram_used_mb"] is not None else "N/A"
        print(f"| VRAM used                      | {v:<20} |")
        print()
        print("| Camera   | Captured | Inferred | In FPS | Out FPS | Drop % | Active s |")
        print("|:---------|:---------|:---------|:-------|:--------|:-------|:---------|")
        for cid, cam in d["per_camera"].items():
            print(
                f"| {cid:<8} | "
                f"{cam['frames_captured']:<8} | "
                f"{cam['frames_inferred']:<8} | "
                f"{cam['fps_input_active']:<6.2f} | "
                f"{cam['fps_output_active']:<7.2f} | "
                f"{cam['drop_rate_pct']:>5.1f}% | "
                f"{cam['active_duration_s']:<8.2f} |"
            )
        print()
        print("=" * 72)
