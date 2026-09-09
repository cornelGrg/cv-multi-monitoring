from __future__ import annotations

import queue
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from src.capture.producer import FramePacket
from src.main import load_config
from src.output.annotator import VideoAnnotator
from src.pipeline.orchestrator import PipelineOrchestrator


class DetectionOverlayTests(unittest.TestCase):
    def test_untracked_detection_is_visible_only_when_supplied(self) -> None:
        # Capture the rendered image before encoding, without needing a GPU.
        writer = VideoAnnotator.__new__(VideoAnnotator)
        writer.width = writer.height = 640
        writer._write = Mock()
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        writer.write_tracks(frame, [])
        np.testing.assert_array_equal(writer._write.call_args.args[0], frame)

        detections = np.array([[40, 50, 100, 120, 0.9, 2]], dtype=np.float32)
        writer.write_tracks(frame, [], detections=detections)
        rendered = writer._write.call_args.args[0]
        np.testing.assert_array_equal(rendered[80, 40], [255, 255, 0])
        self.assertTrue(np.any(rendered != frame))
        self.assertFalse(np.any(frame))

        # An empty detection set still enables the diagnostic legend.
        writer.write_tracks(frame, [], detections=np.empty((0, 6)))
        self.assertTrue(np.any(writer._write.call_args.args[0][-22:]))

    def test_toggle_routes_only_filtered_detections_to_their_camera(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs/phase4_analytics_demo.yaml"
        for setting in (None, False, True):
            with self.subTest(show_detections=setting):
                config = load_config(config_path)
                config["output"].pop("show_detections", None)
                if setting is not None:
                    config["output"]["show_detections"] = setting
                config["pipeline"]["max_frames_per_stream"] = 1
                config["pipeline"]["max_duration_s"] = 0
                config["pipeline"]["loop_video"] = False
                config["analytics"]["enabled"] = False
                raw = np.array([
                    [[10 + i, 20, 30, 40, 0.9, 2],
                     [0, 0, 5, 5, 0.1, 2],
                     [0, 0, 5, 5, 0.9, 0]]
                    for i in range(4)
                ], dtype=np.float32)
                engine = Mock()
                engine.infer.return_value = (raw, 0.01)
                pipeline = PipelineOrchestrator(config, engine, Mock())
                for stream in config["streams"]:
                    cam = stream["camera_id"]
                    q = queue.Queue()
                    q.put(FramePacket(cam, 0, 0.0, 1, np.zeros((3, 640, 640), dtype=np.float32)))
                    pipeline._queues.append(q)
                    tracker = Mock()
                    tracker.update.return_value = []
                    pipeline._trackers[cam] = tracker
                    pipeline._async_annotators[cam] = Mock()
                pipeline._consumer_loop()

                for i, stream in enumerate(config["streams"]):
                    cam = stream["camera_id"]
                    supplied = pipeline._async_annotators[cam].submit.call_args.kwargs["detections"]
                    if setting:
                        np.testing.assert_array_equal(supplied, raw[i, :1])
                    else:
                        self.assertIsNone(supplied)
                    # The diagnostic toggle never changes tracker input.
                    np.testing.assert_array_equal(
                        pipeline._trackers[cam].update.call_args.args[0], raw[i, :1],
                    )


if __name__ == "__main__":
    unittest.main()
