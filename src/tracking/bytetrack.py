"""ByteTrack adapter with strictly camera-local state and identities."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True)
class ByteTrackConfig:
    """Runtime settings matching Ultralytics' ByteTrack implementation."""

    frame_rate: float = 25.0
    track_buffer: int = 30
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    match_thresh: float = 0.8
    fuse_score: bool = True

    @classmethod
    def from_dict(cls, values: dict | None) -> "ByteTrackConfig":
        values = values or {}
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def as_namespace(self) -> Namespace:
        return Namespace(
            tracker_type="bytetrack",
            frame_rate=self.frame_rate,
            track_buffer=self.track_buffer,
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            match_thresh=self.match_thresh,
            fuse_score=self.fuse_score,
        )


@dataclass(frozen=True)
class Track:
    """One camera-local tracked vehicle at one source frame."""

    camera_id: str
    source_frame_id: int
    source_timestamp_ms: float
    capture_time_ns: int
    inference_time_ns: int
    local_track_id: int
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def track_id(self) -> str:
        return f"{self.camera_id}:{self.local_track_id}"

    @property
    def class_name(self) -> str:
        return VEHICLE_CLASSES.get(self.class_id, str(self.class_id))

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


class _Detections:
    """Small Results-like NumPy wrapper consumed by Ultralytics ByteTrack."""

    def __init__(self, detections: np.ndarray) -> None:
        data = np.asarray(detections, dtype=np.float32)
        self.data = data.reshape((-1, 6)) if data.size else np.empty((0, 6), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: Any) -> "_Detections":
        selected = self.data[item]
        if selected.ndim == 1:
            selected = selected.reshape(1, -1)
        return _Detections(selected)

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4]

    @property
    def xywh(self) -> np.ndarray:
        boxes = self.xyxy.copy()
        boxes[:, 2:4] -= boxes[:, 0:2]
        boxes[:, 0:2] += boxes[:, 2:4] / 2.0
        return boxes

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5]


def _default_tracker_factory(config: ByteTrackConfig) -> Any:
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments
        raise RuntimeError(
            "Tracking requires the packages in requirements-tracking.txt "
            "(ultralytics and lap)"
        ) from exc
    return BYTETracker(config.as_namespace())


class PerCameraByteTracker:
    """Owns exactly one ByteTrack state machine for a single camera."""

    def __init__(
        self,
        camera_id: str,
        config: ByteTrackConfig,
        *,
        tracker_factory: Callable[[ByteTrackConfig], Any] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.config = config
        self._tracker = (tracker_factory or _default_tracker_factory)(config)
        backend_track_class = getattr(self._tracker, "track_class", None)
        if isinstance(backend_track_class, type):
            # Ultralytics' default STrack counter is class-global. A private
            # subclass gives each camera its own native ID allocator as well as
            # the externally visible camera-local mapping below.
            self._tracker.track_class = type(
                f"{camera_id.replace('-', '_')}LocalSTrack",
                (backend_track_class,),
                {"_count": 0},
            )
        self._last_source_frame_id: int | None = None
        self._native_to_local: dict[int, int] = {}
        self._next_local_id = 1

    @property
    def backend(self) -> Any:
        """Expose the backend for diagnostics without sharing it across cameras."""
        return self._tracker

    def _local_id(self, native_id: int) -> int:
        if native_id not in self._native_to_local:
            self._native_to_local[native_id] = self._next_local_id
            self._next_local_id += 1
        return self._native_to_local[native_id]

    def update(
        self,
        detections: np.ndarray,
        *,
        source_frame_id: int,
        source_timestamp_ms: float,
        capture_time_ns: int,
        inference_time_ns: int,
    ) -> list[Track]:
        """Advance ByteTrack and return visible tracks for this source frame.

        Empty updates are inserted for source-frame gaps so the lost-track buffer
        remains measured in 25 FPS source time even when the input queue drops frames.
        """
        if self._last_source_frame_id is not None:
            if source_frame_id <= self._last_source_frame_id:
                raise ValueError(
                    f"{self.camera_id}: source_frame_id must increase "
                    f"({source_frame_id} <= {self._last_source_frame_id})"
                )
            skipped_frames = source_frame_id - self._last_source_frame_id - 1
        else:
            if source_frame_id < 0:
                raise ValueError("source_frame_id cannot be negative")
            skipped_frames = source_frame_id

        for _ in range(skipped_frames):
            self._tracker.update(_Detections(np.empty((0, 6), dtype=np.float32)))

        dets = np.asarray(detections, dtype=np.float32)
        dets = dets.reshape((-1, 6)) if dets.size else np.empty((0, 6), dtype=np.float32)
        if len(dets):
            dets = dets[np.isin(dets[:, 5].astype(int), tuple(VEHICLE_CLASSES))]
        raw_tracks = self._tracker.update(_Detections(dets))
        self._last_source_frame_id = source_frame_id

        tracks: list[Track] = []
        for row in np.asarray(raw_tracks, dtype=np.float32).reshape((-1, 8)) if len(raw_tracks) else []:
            x1, y1, x2, y2, native_id, score, cls_id, _ = row
            tracks.append(
                Track(
                    camera_id=self.camera_id,
                    source_frame_id=source_frame_id,
                    source_timestamp_ms=source_timestamp_ms,
                    capture_time_ns=capture_time_ns,
                    inference_time_ns=inference_time_ns,
                    local_track_id=self._local_id(int(native_id)),
                    class_id=int(cls_id),
                    confidence=float(score),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return tracks
