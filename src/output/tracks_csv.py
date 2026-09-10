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

    def __init__(self, path: str | Path, frames_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._frames_file = None
        if frames_path is not None:
            frames_path = Path(frames_path)
            frames_path.parent.mkdir(parents=True, exist_ok=True)
            self._frames_file = frames_path.open("w", newline="", encoding="utf-8")
            self._frames_writer = csv.writer(self._frames_file)
            self._frames_writer.writerow(["camera_id", "source_frame_id", "timestamp_ms"])

    def write_frame(self, packet, tracks: Iterable[Track]) -> None:
        """Record processed frames even when no vehicles are tracked."""
        if self._frames_file is not None:
            self._frames_writer.writerow([
                packet.camera_id, packet.source_frame_id, packet.source_timestamp_ms,
            ])
        self.write(tracks)

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
        if self._frames_file is not None and not self._frames_file.closed:
            self._frames_file.close()
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "TracksCsvWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
