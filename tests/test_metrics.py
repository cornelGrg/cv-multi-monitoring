from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.capture.producer import CameraStats
from src.metrics.collector import MetricsCollector


class MetricsCollectorTests(unittest.TestCase):
    @patch.object(MetricsCollector, "_get_vram_mb", return_value=123.0)
    def test_phase3_report_contains_audit_and_active_camera_metrics(
        self,
        _mock_vram: object,
    ) -> None:
        collector = MetricsCollector(
            ["cam_01", "cam_02"],
            run_metadata={"mode": "realtime_simulation"},
        )
        now_ns = time.perf_counter_ns()
        collector._start_ns = now_ns - 2_000_000_000
        collector.record_detection_frames(
            [
                np.array(
                    [
                        [0, 0, 10, 10, 0.9, 2],
                        [0, 0, 10, 10, 0.5, 7],
                    ],
                    dtype=np.float32,
                ),
                np.empty((0, 6), dtype=np.float32),
            ]
        )
        collector.record_batch(
            camera_ids=["cam_01", "cam_02"],
            inference_time_s=0.04,
            capture_times_ns=[now_ns - 20_000_000, now_ns - 20_000_000],
            dequeue_times_ns=[now_ns - 10_000_000, now_ns - 10_000_000],
            detections=2,
            true_batch_size=2,
        )
        collector._end_ns = collector._start_ns + 2_000_000_000
        collector.sync_producer_stats("cam_01", 50, 4, 0, active_duration_s=2.0)
        collector.sync_producer_stats("cam_02", 50, 4, 0, active_duration_s=2.0)
        collector.sync_synchronous_video_output("cam_01", 1)
        collector.sync_synchronous_video_output("cam_02", 1)
        collector.record_analytics_time(
            0.0002,
            (
                SimpleNamespace(
                    camera_id="cam_01",
                    line_id="main",
                    direction="toward_camera",
                    class_name="car",
                ),
            ),
        )

        report = collector.to_dict()

        self.assertEqual(report["run_metadata"]["mode"], "realtime_simulation")
        self.assertEqual(
            report["global"]["batch_size_hist"],
            {"1": 0, "2": 1, "3": 0, "4": 0},
        )
        self.assertIn("latency_e2e_avg_ms", report["global"])
        self.assertIn("queue_wait_p95_ms", report["global"])
        self.assertEqual(report["global"]["video_frames_written"], 2)
        self.assertEqual(report["global"]["total_line_crossing_events"], 1)
        self.assertEqual(report["analytics"]["counts"][0]["count"], 1)
        self.assertEqual(report["per_camera"]["cam_01"]["line_crossing_events"], 1)
        self.assertEqual(report["per_camera"]["cam_01"]["active_duration_s"], 2.0)
        self.assertEqual(report["per_camera"]["cam_01"]["fps_input_active"], 25.0)
        self.assertEqual(report["per_camera"]["cam_01"]["fps_output_active"], 0.5)
        self.assertEqual(report["detection_audit"]["frames_observed"], 2)
        self.assertEqual(report["detection_audit"]["detections_per_frame_p50"], 1.0)
        self.assertEqual(report["detection_audit"]["confidence_p50"], 0.7)
        self.assertEqual(
            report["detection_audit"]["class_histogram_by_coco_id"],
            {"2": 1, "7": 1},
        )

    def test_camera_active_duration_preserves_capture_span(self) -> None:
        stats = CameraStats(
            first_capture_time_ns=1_000_000_000,
            last_capture_time_ns=3_500_000_000,
        )
        self.assertEqual(stats.active_duration_s, 2.5)


if __name__ == "__main__":
    unittest.main()
