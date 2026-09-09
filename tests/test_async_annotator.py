from __future__ import annotations

import json
import subprocess
import threading
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.output.async_annotator import AsyncVideoAnnotator
from src.analytics.line_crossing import AnalyticsOverlay


class _BlockingAnnotator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release_first = threading.Event()
        self.written_markers: list[int] = []
        self.released = False

    def write_tracks(self, frame: np.ndarray, tracks: tuple[int, ...]) -> None:
        if not self.written_markers:
            self.started.set()
            if not self.release_first.wait(timeout=2.0):
                raise TimeoutError("test writer was not released")
        self.written_markers.append(tracks[0])

    def release(self) -> None:
        self.released = True


class _FailingReleaseAnnotator:
    def write_tracks(self, frame: np.ndarray, tracks: tuple[int, ...]) -> None:
        pass

    def release(self) -> None:
        raise RuntimeError("encoder failed")


class _OverlayAnnotator:
    def __init__(self) -> None:
        self.received_overlay: AnalyticsOverlay | None = None
        self.received_detections: np.ndarray | None = None

    def write_tracks(
        self,
        frame: np.ndarray,
        tracks: tuple[int, ...],
        *,
        analytics_overlay: AnalyticsOverlay | None = None,
        detections: np.ndarray | None = None,
    ) -> None:
        self.received_overlay = analytics_overlay
        self.received_detections = detections

    def release(self) -> None:
        pass


class AsyncVideoAnnotatorTests(unittest.TestCase):
    def test_detection_snapshot_reaches_worker_without_shared_mutation(self) -> None:
        sink = _OverlayAnnotator()
        detections = np.array([[0, 0, 1, 1, 0.9, 2]], dtype=np.float32)
        expected = detections.copy()
        annotator = AsyncVideoAnnotator(
            "ignored.mp4", 25.0, 2, 2, annotator_factory=lambda *_: sink,
        )
        annotator.submit(
            np.zeros((3, 2, 2), dtype=np.float32), [],
            source_frame_id=0, detections=detections,
        )
        detections[:] = 0
        annotator.close()
        np.testing.assert_array_equal(sink.received_detections, expected)

    def test_latest_policy_drops_only_queued_video_frame(self) -> None:
        sink = _BlockingAnnotator()
        annotator = AsyncVideoAnnotator(
            "ignored.mp4",
            25.0,
            2,
            2,
            queue_maxsize=1,
            drop_policy="latest",
            annotator_factory=lambda *_: sink,
        )
        tensor = np.zeros((3, 2, 2), dtype=np.float32)

        annotator.submit(tensor, [1], source_frame_id=1)
        self.assertTrue(sink.started.wait(timeout=1.0))
        annotator.submit(tensor, [2], source_frame_id=2)
        annotator.submit(tensor, [3], source_frame_id=3)
        sink.release_first.set()
        annotator.close()

        stats = annotator.stats()
        self.assertEqual(sink.written_markers, [1, 3])
        self.assertEqual(stats.frames_submitted, 3)
        self.assertEqual(stats.frames_written, 2)
        self.assertEqual(stats.frames_dropped, 1)
        self.assertTrue(sink.released)
        self.assertFalse(annotator.is_alive)

    def test_drop_newest_preserves_already_queued_frame(self) -> None:
        sink = _BlockingAnnotator()
        annotator = AsyncVideoAnnotator(
            "ignored.mp4",
            25.0,
            2,
            2,
            queue_maxsize=1,
            drop_policy="drop_newest",
            annotator_factory=lambda *_: sink,
        )
        tensor = np.zeros((3, 2, 2), dtype=np.float32)

        annotator.submit(tensor, [1], source_frame_id=1)
        self.assertTrue(sink.started.wait(timeout=1.0))
        self.assertTrue(annotator.submit(tensor, [2], source_frame_id=2))
        self.assertFalse(annotator.submit(tensor, [3], source_frame_id=3))
        sink.release_first.set()
        annotator.close()

        self.assertEqual(sink.written_markers, [1, 2])
        self.assertEqual(annotator.stats().frames_dropped, 1)

    def test_invalid_queue_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            AsyncVideoAnnotator("ignored.mp4", 25.0, 2, 2, queue_maxsize=0)

    def test_encoder_failure_is_reported_at_shutdown(self) -> None:
        annotator = AsyncVideoAnnotator(
            "ignored.mp4",
            25.0,
            2,
            2,
            annotator_factory=lambda *_: _FailingReleaseAnnotator(),
        )

        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            annotator.close()

    def test_analytics_overlay_snapshot_reaches_worker(self) -> None:
        sink = _OverlayAnnotator()
        overlay = AnalyticsOverlay(roi=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)), lines=())
        annotator = AsyncVideoAnnotator(
            "ignored.mp4",
            25.0,
            2,
            2,
            annotator_factory=lambda *_: sink,
        )
        tensor = np.zeros((3, 2, 2), dtype=np.float32)

        annotator.submit(
            tensor,
            [],
            source_frame_id=1,
            analytics_overlay=overlay,
        )
        annotator.close()

        self.assertIs(sink.received_overlay, overlay)

    def test_real_writer_flushes_all_queued_frames_on_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "async.mp4"
            annotator = AsyncVideoAnnotator(
                output_path,
                25.0,
                640,
                640,
                queue_maxsize=4,
                encoder="libx264",
            )
            tensor = np.zeros((3, 640, 640), dtype=np.float32)
            for source_frame_id in range(3):
                annotator.submit(tensor, [], source_frame_id=source_frame_id)
            annotator.close()

            capture = cv2.VideoCapture(str(output_path))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,codec_tag_string",
                    "-of",
                    "json",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]

        self.assertEqual(frame_count, 3)
        self.assertEqual(stream["codec_name"], "h264")
        self.assertEqual(stream["codec_tag_string"], "avc1")
        self.assertEqual(annotator.stats().frames_written, 3)
        self.assertEqual(annotator.stats().frames_dropped, 0)


if __name__ == "__main__":
    unittest.main()
