import unittest
from scripts.plot_results import manual_comparison


class PlotResultsTests(unittest.TestCase):
    def test_manual_intervals_are_half_open_and_exclude_other_gates(self):
        events = [dict(camera_id='cam_01', direction='away_from_camera',
                       line_id='away_from_camera_main', timestamp_ms=value) for value in ['1999', '2000', '24999', '25000']]
        events.append(dict(camera_id='cam_01', direction='away_from_camera', line_id='other', timestamp_ms='5000'))
        result = manual_comparison(events)
        self.assertEqual(result[1]['manual'], 19)
        self.assertEqual(result[1]['automated'], 2)
        self.assertEqual(result[0]['automated'], 0)
