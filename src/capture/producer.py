"""
Video capture producer with per-camera metrics and frame-drop policy.

Each VideoProducer runs in its own thread, reads frames from a video source,
preprocesses them (letterbox + normalize), and pushes the result into a
bounded, thread-safe queue.  When the queue is full the producer either blocks
(policy ``"none"``) or drops the oldest item to make room (policy ``"latest"``),
keeping end-to-end latency low.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CameraStats:
    """Mutable counters updated by a single VideoProducer thread."""

    camera_id: str = ""
    frames_captured: int = 0
    frames_dropped: int = 0
    read_errors: int = 0
    video_width: int = 0
    video_height: int = 0
    video_total_frames: int = 0
    video_fps: float = 0.0
    first_capture_time_ns: int | None = None
    last_capture_time_ns: int | None = None

    @property
    def active_duration_s(self) -> float:
        """Wall-clock span between the first and last captured source frames."""
        if self.first_capture_time_ns is None or self.last_capture_time_ns is None:
            return 0.0
        return max(0.0, (self.last_capture_time_ns - self.first_capture_time_ns) / 1e9)


@dataclass
class FramePacket:
    """Payload travelling through the per-camera queue."""

    camera_id: str
    source_frame_id: int         # original capture sequence index; gaps survive queue drops
    source_timestamp_ms: float   # timestamp on the simulated source timeline
    capture_time_ns: int        # time.perf_counter_ns() at capture
    tensor: np.ndarray          # (3, H, W) float32, preprocessed

def letterbox(frame: np.ndarray, input_size: int = 640) -> np.ndarray:
    """Letterbox resize + normalise: BGR uint8 → CHW float32 [0, 1]."""
    h, w = frame.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    top = (input_size - nh) // 2
    left = (input_size - nw) // 2
    canvas[top: top + nh, left: left + nw] = resized

    # HWC → CHW, BGR → RGB, normalise
    return canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0


class VideoProducer(threading.Thread):
    """Thread that captures and preprocesses frames from a single video source.

    Parameters
    ----------
    camera_id:
        Unique identifier for this camera/stream.
    video_path:
        Path to the video file.
    frame_queue:
        Bounded ``queue.Queue`` shared with the consumer.
    input_size:
        Square size for YOLO letterbox preprocessing.
    max_frames:
        Maximum frames to read (0 = read entire video).
    drop_policy:
        ``"latest"`` — when the queue is full, drop the oldest item and push
        the new one (low-latency mode).  ``"none"`` — block until space is
        available (no frame loss).
    """

    MAX_CONSECUTIVE_ERRORS = 10

    def __init__(
        self,
        camera_id: str,
        video_path: str | Path,
        frame_queue: queue.Queue,
        *,
        input_size: int = 640,
        max_frames: int = 0,
        drop_policy: str = "latest",
        source_fps: float | None = None,
        loop_video: bool = False,
        barrier: threading.Barrier | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"Producer-{camera_id}")
        self.camera_id = camera_id
        self.video_path = str(video_path)
        self.frame_queue = frame_queue
        self.input_size = input_size
        self.max_frames = max_frames
        self.drop_policy = drop_policy
        self.source_fps = source_fps
        self.loop_video = loop_video
        self.barrier = barrier

        self._stop_event = threading.Event()
        self.stats = CameraStats(camera_id=camera_id)

    # ── Thread entry point ──────────────────────────────────────────────

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[{self.camera_id}] Cannot open {self.video_path}")

        # Populate video metadata
        self.stats.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stats.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stats.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stats.video_fps = cap.get(cv2.CAP_PROP_FPS)

        consecutive_errors = 0
        frame_idx = 0
        limit = self.max_frames if self.max_frames > 0 else float("inf")

        target_frame_time = 1.0 / self.source_fps if self.source_fps and self.source_fps > 0 else 0

        if self.barrier:
            logger.info("[%s] Waiting at barrier...", self.camera_id)
            self.barrier.wait()

        next_frame_time = time.perf_counter()

        try:
            while frame_idx < limit and not self._stop_event.is_set():
                if target_frame_time > 0:
                    now = time.perf_counter()
                    if now < next_frame_time:
                        time.sleep(next_frame_time - now)
                    next_frame_time = time.perf_counter() + target_frame_time

                ret, frame = cap.read()
                if not ret:
                    # End of video
                    if self.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            consecutive_errors += 1
                            self.stats.read_errors += 1
                            if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                                logger.error("[%s] Aborting after %d consecutive read errors", self.camera_id, consecutive_errors)
                                break
                            continue
                    else:
                        break

                consecutive_errors = 0
                capture_ts = time.perf_counter_ns()
                if self.stats.first_capture_time_ns is None:
                    self.stats.first_capture_time_ns = capture_ts
                self.stats.last_capture_time_ns = capture_ts
                source_frame_id = frame_idx
                timeline_fps = self.source_fps or self.stats.video_fps
                source_timestamp_ms = (
                    source_frame_id * 1000.0 / timeline_fps
                    if timeline_fps and timeline_fps > 0
                    else float(cap.get(cv2.CAP_PROP_POS_MSEC))
                )
                # Advance on every successful source read, before any queue drop.
                frame_idx += 1

                # Preprocess in producer thread (parallelised across cameras)
                tensor = letterbox(frame, self.input_size)

                packet = FramePacket(
                    camera_id=self.camera_id,
                    source_frame_id=source_frame_id,
                    source_timestamp_ms=source_timestamp_ms,
                    capture_time_ns=capture_ts,
                    tensor=tensor,
                )

                # Enqueue with drop policy
                if self.drop_policy == "latest" and self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # discard oldest
                        self.stats.frames_dropped += 1
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put(packet, timeout=5.0)
                except queue.Full:
                    self.stats.frames_dropped += 1
                    continue

                self.stats.frames_captured += 1

        finally:
            # Sentinel to signal completion
            try:
                self.frame_queue.put(None, timeout=2.0)
            except queue.Full:
                pass
            cap.release()

    # ── Control ─────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the thread to stop at the next iteration."""
        self._stop_event.set()
