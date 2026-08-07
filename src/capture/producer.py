"""
Video capture producer with per-camera metrics and frame-drop policy.

Each VideoProducer runs in its own thread, reads frames from a video source,
preprocesses them (letterbox + normalize), and pushes the result into a
bounded, thread-safe queue.  When the queue is full the producer either blocks
(policy ``"none"``) or drops the oldest item to make room (policy ``"latest"``),
keeping end-to-end latency low.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


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


@dataclass
class FramePacket:
    """Payload travelling through the per-camera queue."""

    camera_id: str
    frame_idx: int
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
    ) -> None:
        super().__init__(daemon=True, name=f"Producer-{camera_id}")
        self.camera_id = camera_id
        self.video_path = str(video_path)
        self.frame_queue = frame_queue
        self.input_size = input_size
        self.max_frames = max_frames
        self.drop_policy = drop_policy

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

        try:
            while frame_idx < limit and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # End of video
                    if self.max_frames == 0:
                        # max_frames=0 means "process all" — EOF is normal
                        break
                    # Loop video if it's shorter than max_frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_errors += 1
                        self.stats.read_errors += 1
                        if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                            break
                        continue

                consecutive_errors = 0
                capture_ts = time.perf_counter_ns()

                # Preprocess in producer thread (parallelised across cameras)
                tensor = letterbox(frame, self.input_size)

                packet = FramePacket(
                    camera_id=self.camera_id,
                    frame_idx=frame_idx,
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
                frame_idx += 1

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
