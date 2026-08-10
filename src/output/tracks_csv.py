"""Streaming CSV output for per-frame vehicle tracks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, TextIO

from src.tracking.bytetrack import Track


class TracksCsvWriter:
    """Write track observations incrementally without retaining a run in memory."""

    FIELDNAMES = [
        "timestamp_ms",
        "capture_time_ns",
        "inference_time_ns",
        "camera_id",
        "source_frame_id",
        "track_id",
        "local_track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def write(self, tracks: Iterable[Track]) -> None:
        for track in tracks:
            self._writer.writerow(
                {
                    "timestamp_ms": round(track.source_timestamp_ms, 3),
                    "capture_time_ns": track.capture_time_ns,
                    "inference_time_ns": track.inference_time_ns,
                    "camera_id": track.camera_id,
                    "source_frame_id": track.source_frame_id,
                    "track_id": track.track_id,
                    "local_track_id": track.local_track_id,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "confidence": round(track.confidence, 6),
                    "x1": round(track.x1, 3),
                    "y1": round(track.y1, 3),
                    "x2": round(track.x2, 3),
                    "y2": round(track.y2, 3),
                    "center_x": round(track.center_x, 3),
                    "center_y": round(track.center_y, 3),
                }
            )

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "TracksCsvWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
