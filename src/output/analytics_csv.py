"""Incremental structured output for line-crossing events and counters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, TextIO

from src.analytics.line_crossing import LineCrossingEvent


class AnalyticsCsvWriter:
    """Write event rows and cumulative count snapshots as crossings occur."""

    EVENT_FIELDS = [
        "timestamp_ms",
        "source_frame_id",
        "camera_id",
        "track_id",
        "local_track_id",
        "event_type",
        "line_id",
        "direction",
        "lane_label",
        "class_id",
        "class_name",
        "crossing_x_normalized",
        "crossing_y_normalized",
    ]
    COUNT_FIELDS = [
        "timestamp_ms",
        "source_frame_id",
        "camera_id",
        "line_id",
        "direction",
        "lane_label",
        "class_id",
        "class_name",
        "count",
        "trigger_track_id",
    ]

    def __init__(self, events_path: str | Path, counts_path: str | Path) -> None:
        self.events_path = Path(events_path)
        self.counts_path = Path(counts_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.counts_path.parent.mkdir(parents=True, exist_ok=True)
        self._events_file: TextIO = self.events_path.open(
            "w", newline="", encoding="utf-8"
        )
        self._counts_file: TextIO = self.counts_path.open(
            "w", newline="", encoding="utf-8"
        )
        self._events_writer = csv.DictWriter(
            self._events_file, fieldnames=self.EVENT_FIELDS
        )
        self._counts_writer = csv.DictWriter(
            self._counts_file, fieldnames=self.COUNT_FIELDS
        )
        self._events_writer.writeheader()
        self._counts_writer.writeheader()

    def write(self, events: Iterable[LineCrossingEvent]) -> int:
        """Append new events, flush them for long-running service visibility."""
        rows_written = 0
        for event in events:
            common = {
                "timestamp_ms": round(event.timestamp_ms, 3),
                "source_frame_id": event.source_frame_id,
                "camera_id": event.camera_id,
                "line_id": event.line_id,
                "direction": event.direction,
                "lane_label": event.lane_label,
                "class_id": event.class_id,
                "class_name": event.class_name,
            }
            self._events_writer.writerow(
                {
                    **common,
                    "track_id": event.track_id,
                    "local_track_id": event.local_track_id,
                    "event_type": event.event_type,
                    "crossing_x_normalized": round(event.crossing_x, 6),
                    "crossing_y_normalized": round(event.crossing_y, 6),
                }
            )
            self._counts_writer.writerow(
                {
                    **common,
                    "count": event.cumulative_count,
                    "trigger_track_id": event.track_id,
                }
            )
            rows_written += 1
        if rows_written:
            self._events_file.flush()
            self._counts_file.flush()
        return rows_written

    def close(self) -> None:
        if not self._events_file.closed:
            self._events_file.flush()
            self._events_file.close()
        if not self._counts_file.closed:
            self._counts_file.flush()
            self._counts_file.close()

    def __enter__(self) -> "AnalyticsCsvWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
