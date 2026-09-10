import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.render_demo import flow_rate, select_frame
from src.output.tracks_csv import TracksCsvWriter


class ShowcaseTests(unittest.TestCase):
    def test_timeline_holds_previous_observation_without_showing_future(self):
        timeline = [40, 120, 240]
        self.assertEqual(select_frame(timeline, 0), -1)
        self.assertEqual(select_frame(timeline, 40), 0)
        self.assertEqual(select_frame(timeline, 119), 0)
        self.assertEqual(select_frame(timeline, 120), 1)
        self.assertEqual(select_frame(timeline, 1000), 2)

    def test_flow_excludes_late_reported_events_until_their_observation(self):
        events = [dict(line_id='toward', source_frame_id='30', timestamp_ms='500')]
        self.assertEqual(flow_rate(events, 'toward', 25, 1000, 0, 15), (0, 0, True))
        self.assertEqual(flow_rate(events, 'toward', 30, 1200, 0, 15), (1, 50, True))
        self.assertEqual(flow_rate(events, 'away', 30, 1200, 0, 15), (0, 0, True))

    def test_rate_expires_but_total_count_remains(self):
        events = [dict(line_id='toward', source_frame_id='30', timestamp_ms='500')]
        self.assertEqual(flow_rate(events, 'toward', 400, 15000, 0, 15), (1, 4, False))
        self.assertEqual(flow_rate(events, 'toward', 400, 15500, 0, 15), (1, 0, False))

    def test_empty_processed_frames_are_preserved_in_recording(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with TracksCsvWriter(root / 'tracks.csv', root / 'frames.csv') as writer:
                writer.write_frame(SimpleNamespace(camera_id='cam_01', source_frame_id=8,
                                                   source_timestamp_ms=320), [])
            with (root / 'frames.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [dict(camera_id='cam_01', source_frame_id='8', timestamp_ms='320')])
            with (root / 'tracks.csv').open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
