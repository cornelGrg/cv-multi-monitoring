"""Non-blocking per-camera annotated-video output."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from src.tracking.bytetrack import Track

from .annotator import VideoAnnotator

if TYPE_CHECKING:
    from src.analytics.line_crossing import AnalyticsOverlay


@dataclass(frozen=True)
class AnnotationJob:
    """Immutable references needed to render one tracked source frame."""

    source_frame_id: int
    tensor: np.ndarray
    tracks: tuple[Track, ...]
    analytics_overlay: AnalyticsOverlay | None
    detections: np.ndarray | None = None


@dataclass(frozen=True)
class AsyncVideoStats:
    """Final output statistics copied after the writer thread stops."""

    frames_submitted: int
    frames_written: int
    frames_dropped: int
    queue_size_avg: float
    queue_size_max: int
    worker_time_avg_ms: float
    worker_time_p95_ms: float
    worker_times_s: tuple[float, ...]


class AsyncVideoAnnotator:
    """Render and encode video on a dedicated bounded-queue worker."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        *,
        queue_maxsize: int = 8,
        drop_policy: str = "latest",
        encoder: str = "auto",
        annotator_factory: Callable[..., VideoAnnotator] | None = None,
    ) -> None:
        if queue_maxsize < 1:
            raise ValueError("video queue_maxsize must be at least 1")
        if drop_policy not in {"latest", "drop_newest"}:
            raise ValueError("video drop_policy must be 'latest' or 'drop_newest'")

        self.output_path = Path(output_path)
        self.drop_policy = drop_policy
        self._queue: queue.Queue[AnnotationJob | None] = queue.Queue(queue_maxsize)
        self._annotator = (
            annotator_factory(output_path, fps, width, height)
            if annotator_factory is not None
            else VideoAnnotator(output_path, fps, width, height, encoder=encoder)
        )
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"VideoWriter-{self.output_path.stem}",
        )
        self._accepting = True
        self._error: BaseException | None = None
        self._frames_submitted = 0
        self._frames_written = 0
        self._frames_dropped = 0
        self._queue_sizes: list[int] = []
        self._worker_times: list[float] = []
        self._thread.start()

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(
        self,
        tensor: np.ndarray,
        tracks: Sequence[Track],
        *,
        source_frame_id: int,
        analytics_overlay: AnalyticsOverlay | None = None,
        detections: np.ndarray | None = None,
    ) -> bool:
        """Queue a frame without blocking the analytics path.

        Returns ``True`` when the new frame was queued. With ``latest``, an
        older queued video frame may be discarded to make room.
        """
        if not self._accepting:
            raise RuntimeError("AsyncVideoAnnotator is closed")
        if self._error is not None:
            raise RuntimeError(f"Video writer failed: {self._error}") from self._error

        self._frames_submitted += 1
        job = AnnotationJob(
            source_frame_id,
            tensor,
            tuple(tracks),
            analytics_overlay,
            None if detections is None else detections.copy(),
        )

        if self._queue.full():
            if self.drop_policy == "drop_newest":
                self._frames_dropped += 1
                return False
            try:
                discarded = self._queue.get_nowait()
                if discarded is not None:
                    self._frames_dropped += 1
            except queue.Empty:
                pass

        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._frames_dropped += 1
            return False

        self._queue_sizes.append(self._queue.qsize())
        return True

    def _run(self) -> None:
        try:
            while True:
                job = self._queue.get()
                if job is None:
                    break
                started = time.perf_counter()
                frame = (job.tensor.transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
                overlays = {}
                if job.analytics_overlay is not None:
                    overlays["analytics_overlay"] = job.analytics_overlay
                if job.detections is not None:
                    overlays["detections"] = job.detections
                self._annotator.write_tracks(frame, job.tracks, **overlays)
                self._worker_times.append(time.perf_counter() - started)
                self._frames_written += 1
        except BaseException as exc:  # propagated on the next submit and logged at shutdown
            self._error = exc
        finally:
            try:
                self._annotator.release()
            except BaseException as exc:
                if self._error is None:
                    self._error = exc

    def close(self) -> None:
        """Stop accepting jobs, flush queued frames, and join the writer."""
        if not self._accepting:
            return
        self._accepting = False

        if self._thread.is_alive():
            while self._thread.is_alive():
                try:
                    self._queue.put(None, timeout=0.25)
                    break
                except queue.Full:
                    continue
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError(f"Video writer {self.output_path} did not stop")

        # A failed worker may leave jobs behind. Account for them as video-only drops.
        while True:
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                break
            if pending is not None:
                self._frames_dropped += 1

        if self._error is not None:
            raise RuntimeError(f"Video writer failed: {self._error}") from self._error

    def stats(self) -> AsyncVideoStats:
        """Return a stable snapshot; call after ``close`` for final values."""
        queue_avg = float(np.mean(self._queue_sizes)) if self._queue_sizes else 0.0
        queue_max = int(np.max(self._queue_sizes)) if self._queue_sizes else 0
        worker_avg = float(np.mean(self._worker_times)) * 1000 if self._worker_times else 0.0
        worker_p95 = (
            float(np.percentile(self._worker_times, 95)) * 1000
            if self._worker_times
            else 0.0
        )
        return AsyncVideoStats(
            frames_submitted=self._frames_submitted,
            frames_written=self._frames_written,
            frames_dropped=self._frames_dropped,
            queue_size_avg=queue_avg,
            queue_size_max=queue_max,
            worker_time_avg_ms=worker_avg,
            worker_time_p95_ms=worker_p95,
            worker_times_s=tuple(self._worker_times),
        )

    def __enter__(self) -> "AsyncVideoAnnotator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
