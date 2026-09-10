"""Local browser view of the pipeline's latest processed frames and analytics."""

from __future__ import annotations

import base64
import copy
import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def dashboard_config(config: dict) -> dict:
    """Run an indefinite replay without replacing recorded-demo outputs."""
    config = copy.deepcopy(config)
    if not (config.get('tracking', {}).get('enabled')
            and config.get('analytics', {}).get('enabled')
            and config['analytics'].get('flow', {}).get('enabled')):
        raise ValueError('The dashboard requires tracking, analytics, and flow; use configs/default.yaml')
    if len(config['streams']) != 4 or config['inference']['batch_size'] != 4:
        raise ValueError('The dashboard requires four camera streams and batch size four')
    if not config['pipeline'].get('source_fps', 0) > 0:
        raise ValueError('The dashboard requires a positive pipeline.source_fps')
    config['pipeline'].update(loop_video=True, max_duration_s=0, max_frames_per_stream=0)
    config['tracking'].update(write_annotated_video=False, write_tracks_csv=False)
    config['analytics']['write_csv'] = False
    config['output']['metrics_file'] = 'outputs/dashboard/metrics.json'
    return config


class DashboardServer:
    """Keep one snapshot per camera; encode JPEGs only when a browser requests them.

    Publishing never waits for a browser or performs image encoding. A snapshot
    holds the frame, tracks, and counter/flow overlay from the same observation.
    """

    def __init__(self, config: dict, port: int = 8000) -> None:
        self.port = port
        self.window_s = config['analytics']['flow']['window_s']
        self.size = config['model']['input_size']
        self.cameras = {}
        for camera in config['streams']:
            capture = cv2.VideoCapture(str(ROOT / camera['path']))
            try:
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if not capture.isOpened() or width <= 0 or height <= 0:
                    raise ValueError(f"Cannot open dashboard source: {camera['path']}")
            finally:
                capture.release()
            scale = min(self.size / width, self.size / height)
            nw, nh = int(width * scale), int(height * scale)
            self.cameras[camera['camera_id']] = {
                'camera_id': camera['camera_id'], 'sequence': camera.get('sequence_id', ''),
                'crop': ((self.size - nw) // 2, (self.size - nh) // 2, nw, nh),
            }
        self._lock = threading.Lock()
        self._latest = {}
        self._cache = {}
        self._times = {cam: deque(maxlen=120) for cam in self.cameras}
        self._started = time.monotonic()
        self._server = None
        self._thread = None

    def publish(self, packet, tracks, overlay) -> None:
        if overlay is None:
            return
        now = time.monotonic()
        with self._lock:
            self._latest[packet.camera_id] = (packet, tuple(tracks), overlay, now)
            self._times[packet.camera_id].append(now)

    def _encode(self, camera, job):
        packet, tracks, overlay, _ = job
        left, top, width, height = camera['crop']
        frame = packet.tensor[:, top:top + height, left:left + width]
        bgr = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
        ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            raise RuntimeError('Unable to encode dashboard frame')

        def point(x, y):
            return [(x - left) / width, (y - top) / height]

        return {
            'frame_id': packet.source_frame_id,
            'source_seconds': packet.source_timestamp_ms / 1000,
            'image': base64.b64encode(jpeg).decode('ascii'),
            'tracks': [{'id': t.local_track_id, 'class_name': t.class_name,
                        'p1': point(t.x1, t.y1), 'p2': point(t.x2, t.y2)} for t in tracks],
            'roi': [point(x * self.size, y * self.size) for x, y in overlay.roi],
            'lines': [{'id': line.line_id, 'direction': line.direction_label,
                       'p1': point(line.p1[0] * self.size, line.p1[1] * self.size),
                       'p2': point(line.p2[0] * self.size, line.p2[1] * self.size),
                       'crossing_direction': line.crossing_direction,
                       'count': line.count_total,
                       'rate': line.flow.vehicles_per_minute if line.flow else 0,
                       'warming_up': line.flow.warming_up if line.flow else True}
                      for line in overlay.lines],
        }

    def snapshot(self) -> dict:
        with self._lock:
            jobs = dict(self._latest)
            fps = {cam: (len(times) - 1) / (times[-1] - times[0])
                   if len(times) > 1 and times[-1] > times[0] else 0
                   for cam, times in self._times.items()}
        now = time.monotonic()
        cameras = []
        for cam, metadata in self.cameras.items():
            result = {'camera_id': cam, 'sequence': metadata['sequence'], 'ready': cam in jobs}
            if cam in jobs:
                job = jobs[cam]
                if self._cache.get(cam, {}).get('frame_id') != job[0].source_frame_id:
                    self._cache[cam] = self._encode(metadata, job)
                age = max(0, now - job[3])
                result.update(self._cache[cam], age_s=round(age, 2),
                              fps=round(fps[cam], 1) if age < 3 else 0)
            cameras.append(result)
        return {'uptime_s': now - self._started, 'window_s': self.window_s, 'cameras': cameras}

    def __enter__(self):
        dashboard = self
        page = (ROOT / 'assets/dashboard.html').read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = urlsplit(self.path).path
                if path == '/':
                    body, content_type = page, 'text/html; charset=utf-8'
                elif path == '/api/state':
                    body = json.dumps(dashboard.snapshot(), allow_nan=False).encode()
                    content_type = 'application/json'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # A tab was closed while receiving the latest snapshot.

            def setup(self):
                super().setup()
                self.connection.settimeout(2)

            def log_message(self, *_):
                pass

        self._server = HTTPServer(('127.0.0.1', self.port), Handler)
        self.port = self._server.server_port
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={'poll_interval': .1}, daemon=True,
                                        name='DashboardServer')
        self._thread.start()
        logging.getLogger(__name__).info('Dashboard: http://localhost:%d — Ctrl+C to stop', self.port)
        return self

    def __exit__(self, *_):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=3)
