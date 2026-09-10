"""Render a four-camera showcase from a recorded run, without GPU inference.

Run as: python -m scripts.render_demo
Frames and annotations are held together until the next processed source frame;
this preserves source speed without inventing intermediate tracking results.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.output.annotator import VideoAnnotator

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = (22, 18, 14)
PANEL = (36, 30, 24)
TEXT = (245, 241, 235)
MUTED = (167, 153, 136)
TOWARD = (169, 223, 74)
AWAY = (92, 183, 255)


def label(image, value, x, y, scale=.5, color=TEXT, thickness=1):
    cv2.putText(image, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def select_frame(timestamps, timestamp_ms):
    """Never display a future observation; hold the latest processed frame."""
    return bisect.bisect_right(timestamps, timestamp_ms) - 1


def flow_rate(events, line_id, source_frame_id, timestamp_ms, start_ms, window_s):
    observed_s = min(window_s, max(0, (timestamp_ms - start_ms) / 1000))
    accepted = [event for event in events
                if event['line_id'] == line_id
                and int(event['source_frame_id']) <= source_frame_id
                and float(event['timestamp_ms']) <= timestamp_ms]
    recent = sum(float(event['timestamp_ms']) > timestamp_ms - window_s * 1000
                 for event in accepted)
    return len(accepted), recent * 60 / observed_s if observed_s else 0, observed_s < window_s


class CameraReplay:
    def __init__(self, camera, geometry, tracks, frames, events, size, source_fps, loop):
        self.camera = camera
        self.geometry = geometry
        self.tracks = tracks
        self.frames = sorted(frames, key=lambda row: float(row['timestamp_ms']))
        self.timestamps = [float(row['timestamp_ms']) for row in self.frames]
        self.events = events
        self.size = size
        self.loop = loop
        self.capture = cv2.VideoCapture(str(ROOT / camera['path']))
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open {camera['path']}")
        self.length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.length <= 0:
            self.capture.release()
            raise ValueError('The showcase requires seekable source videos')
        self.index = -1
        self.next_source_index = 0
        self.image = np.zeros((360, 640, 3), np.uint8)
        self.frame_id = -1

    def update(self, timestamp_ms):
        index = select_frame(self.timestamps, timestamp_ms)
        if index == self.index or index < 0:
            return
        self.index = index
        self.frame_id = int(self.frames[index]['source_frame_id'])
        source_index = self.frame_id % self.length if self.loop else self.frame_id
        if source_index < self.next_source_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            self.next_source_index = source_index
        frame = None
        while self.next_source_index <= source_index:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {self.camera['camera_id']} frame {source_index}")
            self.next_source_index += 1
        h, w = frame.shape[:2]
        scale = min(self.size / w, self.size / h)
        nw, nh = int(w * scale), int(h * scale)
        left, top = (self.size - nw) // 2, (self.size - nh) // 2
        image = cv2.resize(frame, (640, 360))

        def point(x, y):
            return round((x - left) * 640 / nw), round((y - top) * 360 / nh)

        roi = np.array([point(x * self.size, y * self.size)
                        for x, y in self.geometry['roi']], dtype=np.int32)
        cv2.polylines(image, [roi], True, (125, 125, 125), 1, cv2.LINE_AA)
        for track in self.tracks.get(self.frame_id, []):
            p1 = point(float(track['x1']), float(track['y1']))
            p2 = point(float(track['x2']), float(track['y2']))
            cv2.rectangle(image, p1, p2, (225, 219, 172), 1, cv2.LINE_AA)
            # Small IDs keep the road readable while showing track persistence.
            x, y = max(1, min(p1[0], 593)), max(13, min(p1[1] - 3, 356))
            value = '#' + track['local_track_id']
            label(image, value, x, y, .35, (15, 15, 15), 3)
            label(image, value, x, y, .35)
        for line in self.geometry['lines']:
            color = TOWARD if line['direction_label'] == 'toward_camera' else AWAY
            p1, p2 = [point(x * self.size, y * self.size) for x, y in (line['p1'], line['p2'])]
            cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
            delta = np.array(p2, float) - p1
            normal = np.array([-delta[1], delta[0]]) / max(np.linalg.norm(delta), 1)
            if line['crossing_direction'] == 'positive_to_negative':
                normal *= -1
            middle = ((np.array(p1) + p2) / 2).astype(int)
            cv2.arrowedLine(image, tuple(middle), tuple((middle + normal * 24).astype(int)),
                            color, 2, cv2.LINE_AA, tipLength=.35)
        self.image = image

    def draw(self, canvas, x, y, timestamp_ms, window_s):
        self.update(timestamp_ms)
        cv2.rectangle(canvas, (x, y), (x + 924, y + 438), PANEL, -1)
        label(canvas, self.camera['camera_id'].replace('_', ' ').upper(), x + 16, y + 28, .62, TEXT, 2)
        label(canvas, self.camera.get('sequence_id', ''), x + 640, y + 28, .45, MUTED)
        canvas[y + 48:y + 408, x + 12:x + 652] = self.image
        for number, line in enumerate(self.geometry['lines']):
            color = TOWARD if line['direction_label'] == 'toward_camera' else AWAY
            title = line['direction_label'].replace('_', ' ').upper()
            count, rate, warming = flow_rate(self.events, line['line_id'], self.frame_id,
                                             timestamp_ms, self.timestamps[0], window_s)
            sx, sy = x + 674, y + 70 + number * 178
            cv2.rectangle(canvas, (sx, sy - 14), (sx + 3, sy + 112), color, -1)
            label(canvas, title, sx + 14, sy, .46, color)
            label(canvas, f'{rate:.1f}' if timestamp_ms > self.timestamps[0] else '--',
                  sx + 14, sy + 49, 1.15, TEXT, 2)
            label(canvas, 'vehicles / min', sx + 14, sy + 72, .43, MUTED)
            label(canvas, f'{count} crossings total', sx + 14, sy + 104, .48)
            if warming:
                label(canvas, 'Window warming up', sx + 14, sy + 126, .37, MUTED)
        label(canvas, 'TRACK IDS  /  DIRECTIONAL GATES', x + 16, y + 427, .35, MUTED)
        label(canvas, f'{window_s:g}s rolling window', x + 674, y + 427, .4, MUTED)


def render(run_dir, config_path, output, encoder='libx264', duration_s=None):
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    snapshot = run_dir / 'config_snapshot.yaml'
    config = yaml.safe_load((snapshot if snapshot.exists() else config_path).read_text())
    if 'streams' not in config:
        manifest = yaml.safe_load((ROOT / config['manifest_file']).read_text())
        config['streams'] = manifest['cameras']
    cameras = config['streams']
    if len(cameras) != 4 or any(len(config['analytics']['cameras'][c['camera_id']]['lines']) != 2 for c in cameras):
        raise ValueError('The showcase layout requires four cameras with two directional gates each')
    source_fps = float(metrics['run_metadata']['source_fps'])
    size = int(metrics['run_metadata']['input_size'])
    window_s = float(metrics['run_metadata']['traffic_flow_window_s'])
    if not metrics['run_metadata']['traffic_flow_enabled']:
        raise ValueError('Run the demo with analytics.flow.enabled=true first')
    tracks = defaultdict(lambda: defaultdict(list))
    frame_index = defaultdict(dict)
    for row in read_csv(run_dir / 'tracks.csv'):
        cam, frame_id = row['camera_id'], int(row['source_frame_id'])
        tracks[cam][frame_id].append(row)
        frame_index[cam][frame_id] = row
    if (run_dir / 'frames.csv').exists():
        frame_index = defaultdict(dict)
        for row in read_csv(run_dir / 'frames.csv'):
            frame_index[row['camera_id']][int(row['source_frame_id'])] = row
    # Older runs are usable only if tracks cover every processed frame.
    for camera in cameras:
        cam = camera['camera_id']
        if len(frame_index[cam]) != metrics['per_camera'][cam]['frames_inferred']:
            raise ValueError('Incomplete frame timeline. Rerun the pipeline to generate frames.csv.')
    events = defaultdict(list)
    for row in read_csv(run_dir / 'events.csv'):
        events[row['camera_id']].append(row)
    replays = []
    writer = None
    try:
        for camera in cameras:
            camera = {**camera, 'path': camera.get('path', camera.get('video_path'))}
            cam = camera['camera_id']
            replays.append(CameraReplay(camera, config['analytics']['cameras'][cam],
                                        tracks[cam], list(frame_index[cam].values()), events[cam],
                                        size, source_fps, config['pipeline'].get('loop_video', False)))
        available_s = min(replay.timestamps[-1] / 1000 + 1 / source_fps for replay in replays)
        duration = min(duration_s, available_s) if duration_s is not None else available_s
        if duration <= 0:
            raise ValueError('Duration must be positive')
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = VideoAnnotator(output, source_fps, 1920, 1080, encoder=encoder)
        total = int(duration * source_fps)
        for index in range(total):
            timestamp_ms = index * 1000 / source_fps
            canvas = np.full((1080, 1920, 3), BACKGROUND, np.uint8)
            label(canvas, 'MULTI-CAMERA TRAFFIC ANALYTICS', 24, 48, 1.0, TEXT, 2)
            label(canvas, 'Vehicle tracking and directional traffic flow', 26, 80, .56, MUTED)
            minutes, seconds = divmod(index / source_fps, 60)
            label(canvas, f'RECORDED DEMO    {int(minutes):02d}:{seconds:04.1f}', 1500, 48, .62)
            label(canvas, 'UA-DETRAC / Independent scenes', 1500, 78, .46, MUTED)
            for number, replay in enumerate(replays):
                replay.draw(canvas, 24 + (number % 2) * 948, 112 + (number // 2) * 462,
                            timestamp_ms, window_s)
            label(canvas, 'TOWARD CAMERA', 28, 1051, .47, TOWARD)
            label(canvas, 'AWAY FROM CAMERA', 236, 1051, .47, AWAY)
            label(canvas, 'File replay / Source clips loop', 520, 1051, .46, MUTED)
            label(canvas, f"Run average: {metrics['global']['fps_global']:.1f} processed FPS across 4 cameras",
                  1260, 1051, .46, MUTED)
            cv2.rectangle(canvas, (24, 1028), (24 + round(1872 * (index + 1) / total), 1030), TOWARD, -1)
            writer.write_rendered_frame(canvas)
            if index % 250 == 0:
                logging.info('Rendering %.1f / %.1f source seconds', index / source_fps, duration)
        logging.info('Showcase saved to %s', output)
    finally:
        for replay in replays:
            replay.capture.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'outputs/phase4_demo')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/phase4_analytics_demo.yaml')
    parser.add_argument('--output', type=Path, help='Defaults to showcase.mp4 inside the run folder')
    parser.add_argument('--encoder', choices=['libx264', 'h264_nvenc', 'auto'], default='libx264')
    parser.add_argument('--duration', type=float, help='Optional source duration in seconds')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    render(args.run_dir, args.config, args.output or args.run_dir / 'showcase.mp4', args.encoder, args.duration)


if __name__ == '__main__':
    main()
