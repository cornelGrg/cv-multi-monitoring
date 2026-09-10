import copy
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np

from src.analytics.line_crossing import AnalyticsOverlay, OverlayLine
from src.analytics.traffic_flow import FlowSnapshot
from src.capture.producer import FramePacket
from src.main import load_config
from src.output.dashboard import DashboardServer, ROOT, dashboard_config


class DashboardTests(unittest.TestCase):
    def config(self):
        return load_config(ROOT / 'configs/default.yaml')

    def server(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda key: 640 if key == cv2.CAP_PROP_FRAME_WIDTH else 360
        with patch('src.output.dashboard.cv2.VideoCapture', return_value=capture):
            return DashboardServer(self.config(), port=0)

    def publish(self, server, frame_id=25, count=3):
        flow = FlowSnapshot(frame_id * 40, frame_id, 'cam_01', 'toward',
                            'toward_camera', 15, 1, count, count * 60, True)
        overlay = AnalyticsOverlay(((0, 0), (1, 0), (1, 1)), (
            OverlayLine('toward', (.1, .5), (.9, .5), 'negative_to_positive',
                        'toward_camera', '', count, (), flow),))
        packet = FramePacket('cam_01', frame_id, frame_id * 40, 1,
                             np.full((3, 640, 640), .5, np.float32))
        server.publish(packet, [], overlay)
        return packet

    def test_live_config_preserves_geometry_and_does_not_overwrite_recordings(self):
        source = self.config()
        original = copy.deepcopy(source)
        live = dashboard_config(source)
        self.assertEqual(source, original)
        self.assertEqual(live['analytics']['cameras'], source['analytics']['cameras'])
        self.assertTrue(live['pipeline']['loop_video'])
        self.assertEqual(live['pipeline']['max_duration_s'], 0)
        self.assertEqual(live['pipeline']['max_frames_per_stream'], 0)
        self.assertFalse(live['tracking']['write_annotated_video'])
        self.assertFalse(live['tracking']['write_tracks_csv'])
        self.assertFalse(live['analytics']['write_csv'])
        self.assertEqual(live['output']['metrics_file'], 'outputs/dashboard/metrics.json')
        source['analytics']['flow']['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'requires tracking'):
            dashboard_config(source)

    def test_snapshot_keeps_matching_image_counters_and_newest_frame(self):
        server = self.server()
        self.assertFalse(any(c['ready'] for c in server.snapshot()['cameras']))
        for frame_id in range(10):
            self.publish(server, frame_id, frame_id)
        snapshot = server.snapshot()['cameras'][0]
        self.assertEqual(len(server._latest), 1)
        self.assertEqual(snapshot['frame_id'], 9)
        self.assertEqual(snapshot['lines'][0]['count'], 9)
        self.assertEqual(snapshot['lines'][0]['rate'], 540)
        self.assertTrue(snapshot['lines'][0]['warming_up'])
        self.assertEqual(snapshot['lines'][0]['p1'], [.1, .5])
        import base64
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(snapshot['image']), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (360, 640, 3))
        with patch.object(server, '_encode', side_effect=AssertionError('encoded unchanged frame')):
            self.assertEqual(server.snapshot()['cameras'][0]['image'], snapshot['image'])

    def test_publish_never_encodes_and_stalled_camera_has_zero_fps(self):
        server = self.server()
        with patch('src.output.dashboard.time.monotonic', return_value=100):
            with patch('src.output.dashboard.cv2.imencode', side_effect=AssertionError('encoding blocks inference')):
                self.publish(server)
        with patch('src.output.dashboard.time.monotonic', return_value=110):
            snapshot = server.snapshot()['cameras'][0]
        self.assertEqual(snapshot['age_s'], 10)
        self.assertEqual(snapshot['fps'], 0)

    def test_http_serves_only_dashboard_state_and_shuts_down(self):
        with self.server() as server:
            self.publish(server)
            url = f'http://127.0.0.1:{server.port}'
            with urlopen(url, timeout=3) as response:
                self.assertIn(b'Live monitor', response.read())
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
            with urlopen(url + '/api/state', timeout=3) as response:
                self.assertEqual(json.load(response)['cameras'][0]['frame_id'], 25)
            with self.assertRaises(HTTPError) as error:
                urlopen(url + '/README.md', timeout=3)
            self.assertEqual(error.exception.code, 404)
            error.exception.close()
        self.assertFalse(server._thread.is_alive())
