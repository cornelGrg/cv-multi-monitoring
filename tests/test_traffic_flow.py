from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.analytics.line_crossing import LineCrossingEvent, PerCameraTrafficAnalytics
from src.analytics.traffic_flow import RollingTrafficFlow
from src.output.analytics_csv import AnalyticsCsvWriter
from test_line_crossing import _camera_config, _track


def event(timestamp_ms, line_id='toward', camera_id='cam_01'):
    return LineCrossingEvent(
        timestamp_ms=timestamp_ms, source_frame_id=1, camera_id=camera_id,
        track_id=f'{camera_id}:1', local_track_id=1, event_type='line_crossing',
        line_id=line_id, direction=line_id, lane_label='', class_id=2,
        class_name='car', crossing_x=0.5, crossing_y=0.5, cumulative_count=1,
    )


class RollingTrafficFlowTests(unittest.TestCase):
    def setUp(self):
        self.flow = RollingTrafficFlow('cam_01', {'toward': 'toward', 'away': 'away'}, window_s=30)
        self.update(0)

    def update(self, timestamp_ms, events=()):
        return self.flow.update(events, source_frame_id=int(timestamp_ms / 40), timestamp_ms=timestamp_ms)

    def test_startup_uses_observed_time_then_full_window(self):
        snapshots, _ = self.update(10000, [event(5000)])
        self.assertEqual(snapshots[0].vehicles_per_minute, 6)
        self.assertTrue(snapshots[0].warming_up)
        snapshots, _ = self.update(30000)
        self.assertEqual(snapshots[0].vehicles_per_minute, 2)
        self.assertFalse(snapshots[0].warming_up)
        self.assertEqual(snapshots[1].vehicles_per_minute, 0)

    def test_interpolated_events_out_of_order_expire_at_window_boundary(self):
        self.update(10000, [event(9000)])
        self.update(12000, [event(5000)])
        snapshots, _ = self.update(35000)
        self.assertEqual(snapshots[0].crossings, 1)
        snapshots, _ = self.update(39000)
        self.assertEqual(snapshots[0].crossings, 0)

    def test_frame_gaps_and_late_events_use_source_time(self):
        self.update(1000, [event(500)])
        snapshots, due = self.update(65000, [event(10000), event(64000, 'away')])
        self.assertTrue(due)
        self.assertEqual([s.crossings for s in snapshots], [0, 1])
        self.assertEqual(snapshots[1].vehicles_per_minute, 2)

    def test_sampling_includes_empty_windows(self):
        initial = RollingTrafficFlow('cam_01', {'toward': 'toward'})
        snapshots, due = initial.update([], source_frame_id=0, timestamp_ms=0)
        self.assertTrue(due)
        self.assertEqual(snapshots[0].vehicles_per_minute, 0)
        self.assertEqual(snapshots[0].observed_window_s, 0)
        self.assertFalse(self.update(960)[1])
        self.assertTrue(self.update(1000)[1])
        self.assertFalse(self.update(1040)[1])

    def test_invalid_windows_and_backwards_time_rejected(self):
        for key in ('window_s', 'sample_interval_s'):
            for value in (0, -1, float('nan'), float('inf')):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    RollingTrafficFlow('cam_01', {}, **{key: value})
        self.update(1000)
        with self.assertRaises(ValueError):
            self.update(500)

    def test_camera_identity_cannot_mix(self):
        with self.assertRaises(ValueError):
            self.update(1000, [event(500, camera_id='cam_02')])

    def test_csv_preserves_rate_zero_windows_and_warmup(self):
        warmup, _ = self.update(10000, [event(5000)])
        expired, _ = self.update(40000)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with AnalyticsCsvWriter(root/'events.csv', root/'counts.csv', root/'flow.csv') as writer:
                writer.write_flow(warmup)
                writer.write_flow(expired)
                with (root/'flow.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]['vehicles_per_minute']), 6)
            self.assertEqual(rows[0]['warming_up'], 'True')
            self.assertEqual(rows[2]['crossings'], '0')
            self.assertEqual(rows[2]['warming_up'], 'False')

    def test_flow_uses_deduplicated_crossings_without_changing_events(self):
        plain = PerCameraTrafficAnalytics(_camera_config(), canvas_width=100, canvas_height=100)
        enabled = PerCameraTrafficAnalytics(
            _camera_config(), canvas_width=100, canvas_height=100,
            flow_config={'enabled': True, 'window_s': 30},
        )
        for frame, bottom in ((0, .4), (10, .6), (20, .4), (30, .6)):
            track = _track(frame, .5, bottom)
            args = dict(source_frame_id=frame, source_timestamp_ms=track.source_timestamp_ms)
            old = plain.update([track], **args)
            new = enabled.update([track], **args)
            self.assertEqual(old.events, new.events)
            self.assertEqual(old.flow, ())
        self.assertEqual(new.flow[0].crossings, 1)
        self.assertEqual(new.overlay.lines[0].flow, new.flow[0])
        expired = enabled.update([], source_frame_id=400, source_timestamp_ms=40000)
        self.assertEqual(expired.flow[0].crossings, 0)
        self.assertEqual(expired.overlay.lines[0].count_total, 1)


if __name__ == '__main__':
    unittest.main()
