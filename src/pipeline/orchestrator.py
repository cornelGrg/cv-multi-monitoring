"""
Pipeline orchestrator — consumer loop and lifecycle management.

Coordinates VideoProducer threads, the ONNX inference engine, and the
MetricsCollector.  Implements the batched consumer loop with clean shutdown.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from src.capture.producer import FramePacket, VideoProducer
from src.inference.engine import OnnxGpuEngine
from src.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)

CONF_THRESHOLD = 0.25  # default, overridden by config


def count_detections(output: np.ndarray, conf_threshold: float = CONF_THRESHOLD) -> int:
    """Count valid detections from YOLO end2end output.

    Output shape: ``(batch, max_det, 6)`` where ``6 = [x1, y1, x2, y2, conf, class_id]``.
    Padding entries have ``conf ≈ 0``.
    """
    if output.ndim == 3:
        return int(np.sum(output[:, :, 4] > conf_threshold))
    if output.ndim == 2:
        return int(np.sum(output[:, 4] > conf_threshold))
    return 0


class PipelineOrchestrator:
    """Manages the full lifecycle of the multi-camera batched pipeline.

    Parameters
    ----------
    config:
        Parsed configuration dict (from YAML).
    engine:
        Initialised ``OnnxGpuEngine`` ready for inference.
    collector:
        ``MetricsCollector`` to accumulate statistics.
    """

    def __init__(
        self,
        config: dict,
        engine: OnnxGpuEngine,
        collector: MetricsCollector,
    ) -> None:
        self.config = config
        self.engine = engine
        self.collector = collector

        self._producers: list[VideoProducer] = []
        self._queues: list[queue.Queue] = []
        self._camera_ids: list[str] = []

        pipeline_cfg = config.get("pipeline", {})
        self._queue_maxsize: int = pipeline_cfg.get("queue_maxsize", 10)
        self._max_frames: int = pipeline_cfg.get("max_frames_per_stream", 0)
        self._drop_policy: str = pipeline_cfg.get("frame_drop_policy", "latest")
        self._source_fps: float | None = pipeline_cfg.get("source_fps", None)
        self._conf_threshold: float = config["model"]["confidence_threshold"]
        self._input_size: int = config["model"]["input_size"]
        self._batch_size: int = config["inference"]["batch_size"]

    # ── Setup ───────────────────────────────────────────────────────────

    def _create_producers(self) -> None:
        """Instantiate one VideoProducer + Queue per configured stream."""
        project_root = Path(__file__).resolve().parent.parent.parent

        for stream_cfg in self.config["streams"]:
            camera_id = stream_cfg["camera_id"]
            video_path = project_root / stream_cfg["path"]

            q: queue.Queue = queue.Queue(maxsize=self._queue_maxsize)
            p = VideoProducer(
                camera_id=camera_id,
                video_path=video_path,
                frame_queue=q,
                input_size=self._input_size,
                max_frames=self._max_frames,
                drop_policy=self._drop_policy,
                source_fps=self._source_fps,
            )

            self._queues.append(q)
            self._producers.append(p)
            self._camera_ids.append(camera_id)

            logger.info(
                "Configured stream %s → %s",
                camera_id,
                video_path,
            )

    # ── Consumer loop ───────────────────────────────────────────────────

    def _consumer_loop(self) -> None:
        """Batched consumer: dequeue one frame per camera, stack, infer."""
        batch_count = 0
        max_batches = self._max_frames if self._max_frames > 0 else float("inf")
        num_streams = len(self._queues)
        active_streams = set(range(num_streams))

        logger.info(
            "Consumer started — batch_size=%d, max_batches=%s, streams=%d",
            self._batch_size,
            max_batches if max_batches != float("inf") else "unlimited",
            num_streams,
        )

        while batch_count < max_batches and active_streams:
            packets: list[FramePacket] = []
            dequeue_times: list[int] = []

            # Collect one frame from each active queue
            for idx in list(active_streams):
                try:
                    item = self._queues[idx].get(timeout=3.0)
                except queue.Empty:
                    logger.warning("Queue %d timed out, removing stream", idx)
                    active_streams.discard(idx)
                    continue

                deq_ns = time.perf_counter_ns()

                if item is None:
                    # Producer finished
                    active_streams.discard(idx)
                    continue

                packets.append(item)
                dequeue_times.append(deq_ns)

                # Record queue size snapshot
                self.collector.record_queue_size(
                    item.camera_id, self._queues[idx].qsize()
                )

            if len(packets) < num_streams and len(packets) == 0:
                break  # All streams exhausted

            if len(packets) == 0:
                continue

            # Pad batch if fewer packets than batch_size
            # (only happens when some streams end early)
            tensors = [p.tensor for p in packets]

            # If we have exactly batch_size tensors, stack directly
            if len(tensors) == self._batch_size:
                batch = np.stack(tensors, axis=0)
            else:
                # Pad to batch_size for models with fixed batch dim
                while len(tensors) < self._batch_size:
                    tensors.append(tensors[-1])  # duplicate last
                batch = np.stack(tensors[:self._batch_size], axis=0)

            # Inference
            detections_output, inference_time = self.engine.infer(batch)
            n_det = count_detections(detections_output, self._conf_threshold)

            # Record metrics
            self.collector.record_batch(
                camera_ids=[p.camera_id for p in packets],
                inference_time_s=inference_time,
                capture_times_ns=[p.capture_time_ns for p in packets],
                dequeue_times_ns=dequeue_times,
                detections=n_det,
            )

            batch_count += 1

            # Progress log every 30 batches
            if batch_count % 30 == 0:
                elapsed = (time.perf_counter_ns() - self.collector._start_ns) / 1e9
                fps = self.collector.total_frames_inferred / elapsed if elapsed > 0 else 0
                logger.info(
                    "[batch %3d] %d frames — %.1f FPS global",
                    batch_count,
                    self.collector.total_frames_inferred,
                    fps,
                )

    # ── Shutdown ────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """Signal all producers to stop, join threads, drain queues."""
        logger.info("Shutting down producers...")

        # Signal stop
        for p in self._producers:
            p.stop()

        # Drain queues to unblock any producer stuck on put()
        for q in self._queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Join threads
        for p in self._producers:
            p.join(timeout=3.0)
            if p.is_alive():
                logger.warning("Producer %s did not terminate in time", p.name)

        # Sync producer stats into collector
        for p in self._producers:
            self.collector.sync_producer_stats(
                p.camera_id,
                captured=p.stats.frames_captured,
                dropped=p.stats.frames_dropped,
                errors=p.stats.read_errors,
            )

        # Verify no leftover threads
        alive = [t for t in threading.enumerate() if t.name.startswith("Producer-")]
        if alive:
            logger.warning("Leftover producer threads: %s", [t.name for t in alive])
        else:
            logger.info("All producer threads terminated cleanly")

    # ── Public API ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full pipeline: setup → warmup → run → shutdown."""
        self._create_producers()

        # Warm up GPU
        self.engine.warmup()

        # Start producers
        for p in self._producers:
            p.start()
        time.sleep(0.2)  # Let queues fill slightly

        # Timed consumer loop
        self.collector.start_clock()
        self._consumer_loop()
        self.collector.stop_clock()

        # Clean shutdown
        self._shutdown()
