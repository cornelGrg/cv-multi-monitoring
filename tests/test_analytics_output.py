from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.analytics.line_crossing import LineCrossingEvent
from src.output.analytics_csv import AnalyticsCsvWriter


class AnalyticsCsvWriterTests(unittest.TestCase):
    def test_events_and_cumulative_counts_preserve_event_identity(self) -> None:
        event = LineCrossingEvent(
            timestamp_ms=12540.25,
            source_frame_id=314,
            camera_id="cam_02",
            track_id="cam_02:17",
            local_track_id=17,
            event_type="line_crossing",
            line_id="northbound_lane_1",
            direction="northbound",
            lane_label="lane_1",
            class_id=2,
            class_name="car",
            crossing_x=0.4,
            crossing_y=0.6,
            cumulative_count=7,
        )

        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.csv"
            counts_path = Path(tmp) / "counts.csv"
            with AnalyticsCsvWriter(events_path, counts_path) as writer:
                self.assertEqual(writer.write([event]), 1)
            with events_path.open(newline="", encoding="utf-8") as stream:
                event_row = next(csv.DictReader(stream))
            with counts_path.open(newline="", encoding="utf-8") as stream:
                count_row = next(csv.DictReader(stream))

        self.assertEqual(event_row["timestamp_ms"], "12540.25")
        self.assertEqual(event_row["track_id"], "cam_02:17")
        self.assertEqual(event_row["event_type"], "line_crossing")
        self.assertEqual(count_row["count"], "7")
        self.assertEqual(count_row["trigger_track_id"], "cam_02:17")


if __name__ == "__main__":
    unittest.main()
