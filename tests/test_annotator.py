from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.output.annotator import VideoAnnotator


class VideoAnnotatorTests(unittest.TestCase):
    def test_detection_audit_is_h264_avc1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "audit.mp4"
            annotator = VideoAnnotator(
                output_path,
                25.0,
                640,
                640,
                encoder="libx264",
            )
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            detections = np.array(
                [[10, 20, 100, 120, 0.9, 2]],
                dtype=np.float32,
            )
            annotator.write_frame(frame, detections)
            annotator.release()

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

        self.assertEqual(frame_count, 1)
        self.assertEqual(stream["codec_name"], "h264")
        self.assertEqual(stream["codec_tag_string"], "avc1")


if __name__ == "__main__":
    unittest.main()
