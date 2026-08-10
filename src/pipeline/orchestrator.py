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
from src.output.annotator import VideoAnnotator
from src.output.tracks_csv import TracksCsvWriter
from src.tracking.bytetrack import ByteTrackConfig, PerCameraByteTracker

logger = logging.getLogger(__name__)

CONF_THRESHOLD = 0.25  # default, overridden by config

def filter_detections(output: np.ndarray, conf_threshold: float, allowed_classes: list[int]) -> list[np.ndarray]:
    """Filter detections by confidence and allowed classes.

    Returns a list of arrays (one per image in batch) of shape (N, 6).
    """
    filtered = []
    # If output is 3D (batch, max_det, 6), iterate over batch
    if output.ndim == 3:
        for i in range(output.shape[0]):
            img_dets = output[i]
            mask = (img_dets[:, 4] > conf_threshold) & np.isin(img_dets[:, 5], allowed_classes)
            filtered.append(img_dets[mask])
    elif output.ndim == 2:
        mask = (output[:, 4] > conf_threshold) & np.isin(output[:, 5], allowed_classes)
        filtered.append(output[mask])
    return filtered


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
        self._annotators: dict[str, VideoAnnotator] = {}
        self._trackers: dict[str, PerCameraByteTracker] = {}
        self._tracks_writer: TracksCsvWriter | None = None

        pipeline_cfg = config.get("pipeline", {})
        self._queue_maxsize: int = pipeline_cfg.get("queue_maxsize", 10)
        self._max_frames: int = pipeline_cfg.get("max_frames_per_stream", 0)
        self._drop_policy: str = pipeline_cfg.get("frame_drop_policy", "latest")
        self._source_fps: float | None = pipeline_cfg.get("source_fps", None)

        # New Phase 2.1 properties
        self._allowed_classes = pipeline_cfg.get("allowed_classes", [2, 3, 5, 7])
        self._max_duration_s = pipeline_cfg.get("max_duration_s", 0.0)
        self._loop_video = pipeline_cfg.get("loop_video", False)
        self._audit_frames = pipeline_cfg.get("audit_frames", 0)

        tracking_cfg = config.get("tracking", {})
        self._tracking_enabled = bool(tracking_cfg.get("enabled", False))
        if self._tracking_enabled and tracking_cfg.get("tracker_type", "bytetrack") != "bytetrack":
            raise ValueError("Phase 3 supports tracker_type='bytetrack' only")
        self._write_tracked_video = bool(tracking_cfg.get("write_annotated_video", True))
        self._write_tracks_csv = bool(tracking_cfg.get("write_tracks_csv", True))
        self._tracking_config = ByteTrackConfig.from_dict(tracking_cfg)
        if (
            self._tracking_enabled
            and self._source_fps is not None
            and self._tracking_config.frame_rate != self._source_fps
        ):
            raise ValueError("tracking.frame_rate must match pipeline.source_fps")
        self._tracks_file = tracking_cfg.get("tracks_file", "outputs/tracking/tracks.csv")
        self._tracked_video_dir = tracking_cfg.get("annotated_video_dir", "outputs/tracking")

        self._barrier = threading.Barrier(len(config["streams"]) + 1) if (self._max_duration_s > 0 or self._loop_video) else None

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
                loop_video=self._loop_video,
                barrier=self._barrier,
            )

            self._queues.append(q)
            self._producers.append(p)
            self._camera_ids.append(camera_id)

            if self._audit_frames > 0:
                out_path = project_root / f"outputs/audit_{camera_id}.mp4"
                fps = self._source_fps if self._source_fps else 25.0
                self._annotators[camera_id] = VideoAnnotator(out_path, fps, self._input_size, self._input_size)

            if self._tracking_enabled:
                self._trackers[camera_id] = PerCameraByteTracker(
                    camera_id,
                    self._tracking_config,
                )
                if self._write_tracked_video:
                    out_path = project_root / self._tracked_video_dir / f"{camera_id}.mp4"
                    fps = self._source_fps or self._tracking_config.frame_rate
                    self._annotators[camera_id] = VideoAnnotator(
                        out_path, fps, self._input_size, self._input_size
                    )

            logger.info("Configured stream %s → %s", camera_id, video_path)

        if self._tracking_enabled and self._write_tracks_csv:
            self._tracks_writer = TracksCsvWriter(project_root / self._tracks_file)
        if self._tracking_enabled:
            logger.info(
                "ByteTrack enabled — %d independent trackers, %.1f FPS, %d-frame lost buffer, video=%s, csv=%s",
                len(self._trackers),
                self._tracking_config.frame_rate,
                self._tracking_config.track_buffer,
                self._write_tracked_video,
                self._write_tracks_csv,
            )

    # ── Consumer loop ───────────────────────────────────────────────────

    def _consumer_loop(self) -> None:
        """Batched consumer: dequeue one frame per camera, stack, infer."""
        batch_count = 0
        max_batches = self._max_frames if self._max_frames > 0 else float("inf")
        num_streams = len(self._queues)
        active_streams = set(range(num_streams))

        logger.info(
            "Consumer started — batch_size=%d, max_batches=%s, streams=%d, max_duration_s=%.1f",
            self._batch_size,
            max_batches if max_batches != float("inf") else "unlimited",
            num_streams,
            self._max_duration_s,
        )

        # If we have a barrier, all producers are waiting for us too
        if self._barrier:
            self._barrier.wait()

        start_time = time.perf_counter()

        while batch_count < max_batches and active_streams:
            if self._max_duration_s > 0:
                if (time.perf_counter() - start_time) >= self._max_duration_s:
                    logger.info("Reached max_duration_s (%.1f s). Stopping consumer.", self._max_duration_s)
                    break

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

            true_batch_size = len(packets)
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
            inference_time_ns = time.perf_counter_ns()
            filtered = filter_detections(detections_output, self._conf_threshold, self._allowed_classes)

            # Count only detections for the valid (unpadded) packets
            n_det = sum(len(filtered[i]) for i in range(true_batch_size))

            # Tracking is routed back to the packet's camera before any output.
            if self._tracking_enabled:
                for i in range(true_batch_size):
                    packet = packets[i]
                    tracking_started = time.perf_counter()
                    tracks = self._trackers[packet.camera_id].update(
                        filtered[i],
                        source_frame_id=packet.source_frame_id,
                        source_timestamp_ms=packet.source_timestamp_ms,
                        capture_time_ns=packet.capture_time_ns,
                        inference_time_ns=inference_time_ns,
                    )
                    self.collector.record_tracking_time(
                        time.perf_counter() - tracking_started,
                        track_observations=len(tracks),
                    )

                    output_started = time.perf_counter()
                    if self._tracks_writer is not None:
                        self._tracks_writer.write(tracks)
                    annotator = self._annotators.get(packet.camera_id)
                    if annotator is not None:
                        disp = (packet.tensor.transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
                        annotator.write_tracks(disp, tracks)
                    self.collector.record_output_time(time.perf_counter() - output_started)

            # Phase 2.1 audit path remains unchanged when tracking is disabled.
            elif self._audit_frames > 0:
                for i in range(true_batch_size):
                    cam_id = packets[i].camera_id
                    annotator = self._annotators.get(cam_id)
                    if annotator and annotator.frames_written < self._audit_frames:
                        # Reconstruct (640,640,3) BGR uint8 from (3,640,640) RGB float32
                        disp = (packets[i].tensor.transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
                        annotator.write_frame(disp, filtered[i])

            # Record metrics
            self.collector.record_batch(
                camera_ids=[p.camera_id for p in packets],
                inference_time_s=inference_time,
                capture_times_ns=[p.capture_time_ns for p in packets],
                dequeue_times_ns=dequeue_times,
                detections=n_det,
                true_batch_size=true_batch_size,
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

        # Release annotators
        for ann in self._annotators.values():
            ann.release()
        if self._tracks_writer is not None:
            self._tracks_writer.close()

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
