from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.output.tracks_csv import TracksCsvWriter
from src.tracking.bytetrack import ByteTrackConfig, PerCameraByteTracker


class _FakeByteTracker:
    def __init__(self) -> None:
        self.updates: list[np.ndarray] = []

    def update(self, results: object) -> np.ndarray:
        data = results.data.copy()
        self.updates.append(data)
        if not len(data):
            return np.empty((0, 8), dtype=np.float32)
        det = data[0]
        return np.array([[*det[:4], 77, det[4], det[5], 0]], dtype=np.float32)


def _factory(_: ByteTrackConfig) -> _FakeByteTracker:
    return _FakeByteTracker()


class PerCameraByteTrackerTests(unittest.TestCase):
    def _update(self, tracker: PerCameraByteTracker, frame_id: int, detections: np.ndarray):
        return tracker.update(
            detections,
            source_frame_id=frame_id,
            source_timestamp_ms=frame_id * 40.0,
            capture_time_ns=1000 + frame_id,
            inference_time_ns=2000 + frame_id,
        )

    def test_trackers_have_independent_state_and_camera_scoped_ids(self) -> None:
        cam_a = PerCameraByteTracker("cam_a", ByteTrackConfig(), tracker_factory=_factory)
        cam_b = PerCameraByteTracker("cam_b", ByteTrackConfig(), tracker_factory=_factory)
        detection = np.array([[10, 20, 30, 40, 0.9, 2]], dtype=np.float32)

        track_a = self._update(cam_a, 0, detection)[0]
        track_b = self._update(cam_b, 0, detection)[0]

        self.assertIsNot(cam_a.backend, cam_b.backend)
        self.assertEqual(track_a.track_id, "cam_a:1")
        self.assertEqual(track_b.track_id, "cam_b:1")

    def test_source_frame_gaps_advance_tracker_without_renumbering(self) -> None:
        tracker = PerCameraByteTracker("cam_01", ByteTrackConfig(), tracker_factory=_factory)
        detection = np.array([[10, 20, 30, 40, 0.9, 2]], dtype=np.float32)

        self._update(tracker, 3, detection)
        track = self._update(tracker, 6, detection)[0]

        self.assertEqual([len(x) for x in tracker.backend.updates], [0, 0, 0, 1, 0, 0, 1])
        self.assertEqual(track.source_frame_id, 6)
        self.assertEqual(track.source_timestamp_ms, 240.0)

    def test_non_vehicle_detections_never_reach_bytetrack(self) -> None:
        tracker = PerCameraByteTracker("cam_01", ByteTrackConfig(), tracker_factory=_factory)
        detections = np.array(
            [[0, 0, 10, 10, 0.9, 0], [10, 10, 20, 20, 0.8, 7]], dtype=np.float32
        )

        tracks = self._update(tracker, 0, detections)

        self.assertEqual(len(tracker.backend.updates[0]), 1)
        self.assertEqual(int(tracker.backend.updates[0][0, 5]), 7)
        self.assertEqual(tracks[0].class_name, "truck")

    def test_tracks_csv_preserves_timestamps_source_id_and_geometry(self) -> None:
        tracker = PerCameraByteTracker("cam_01", ByteTrackConfig(), tracker_factory=_factory)
        track = self._update(
            tracker, 5, np.array([[10, 20, 30, 40, 0.75, 2]], dtype=np.float32)
        )[0]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.csv"
            with TracksCsvWriter(path) as writer:
                writer.write([track])
            with path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))

        self.assertEqual(row["camera_id"], "cam_01")
        self.assertEqual(row["source_frame_id"], "5")
        self.assertEqual(row["track_id"], "cam_01:1")
        self.assertEqual(row["timestamp_ms"], "200.0")
        self.assertEqual(row["center_x"], "20.0")
        self.assertEqual(row["center_y"], "30.0")

    @unittest.skipUnless(
        importlib.util.find_spec("ultralytics") and importlib.util.find_spec("lap"),
        "real ByteTrack dependencies are not installed",
    )
    def test_real_bytetrack_keeps_identity_through_short_occlusion(self) -> None:
        tracker = PerCameraByteTracker("cam_real", ByteTrackConfig(track_buffer=30))
        detection = np.array([[100, 100, 180, 180, 0.9, 2]], dtype=np.float32)

        first = self._update(tracker, 0, detection)
        self._update(tracker, 1, np.empty((0, 6), dtype=np.float32))
        recovered = self._update(tracker, 2, detection)

        self.assertEqual(first[0].track_id, "cam_real:1")
        self.assertEqual(recovered[0].track_id, "cam_real:1")

    @unittest.skipUnless(
        importlib.util.find_spec("ultralytics") and importlib.util.find_spec("lap"),
        "real ByteTrack dependencies are not installed",
    )
    def test_real_bytetrack_uses_camera_local_native_track_classes(self) -> None:
        cam_a = PerCameraByteTracker("cam_a", ByteTrackConfig())
        cam_b = PerCameraByteTracker("cam_b", ByteTrackConfig())
        self.assertIsNot(cam_a.backend.track_class, cam_b.backend.track_class)


if __name__ == "__main__":
    unittest.main()
