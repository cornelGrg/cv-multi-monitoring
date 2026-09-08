from __future__ import annotations

import unittest

from src.analytics.line_crossing import (
    CameraAnalyticsConfig,
    MultiCameraTrafficAnalytics,
    PerCameraTrafficAnalytics,
)
from src.tracking.bytetrack import Track


def _camera_config(camera_id: str = "cam_01") -> CameraAnalyticsConfig:
    return CameraAnalyticsConfig.from_dict(
        camera_id,
        {
            "roi": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]],
            "lines": [
                {
                    "line_id": "main",
                    "p1": [0.20, 0.50],
                    "p2": [0.80, 0.50],
                    "crossing_direction": "negative_to_positive",
                    "direction_label": "toward_camera",
                    "lane_label": "lane_1",
                    "allowed_classes": ["car", "bus"],
                }
            ],
        },
    )


def _track(
    frame_id: int,
    x: float,
    bottom_y: float,
    *,
    camera_id: str = "cam_01",
    local_track_id: int = 1,
    class_id: int = 2,
    confidence: float = 0.9,
) -> Track:
    return Track(
        camera_id=camera_id,
        source_frame_id=frame_id,
        source_timestamp_ms=frame_id * 100.0,
        capture_time_ns=1_000 + frame_id,
        inference_time_ns=2_000 + frame_id,
        local_track_id=local_track_id,
        class_id=class_id,
        confidence=confidence,
        x1=x * 100.0 - 5.0,
        y1=bottom_y * 100.0 - 10.0,
        x2=x * 100.0 + 5.0,
        y2=bottom_y * 100.0,
    )


class PerCameraTrafficAnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PerCameraTrafficAnalytics(
            _camera_config(),
            canvas_width=100,
            canvas_height=100,
        )

    def _update(self, track: Track):
        return self.engine.update(
            [track],
            source_frame_id=track.source_frame_id,
            source_timestamp_ms=track.source_timestamp_ms,
        )

    def test_counts_configured_direction_once_and_interpolates_timestamp(self) -> None:
        first = self._update(_track(0, 0.5, 0.4))
        crossing = self._update(_track(10, 0.5, 0.6))
        reverse = self._update(_track(20, 0.5, 0.4))
        repeated = self._update(_track(30, 0.5, 0.6))

        self.assertEqual(first.events, ())
        self.assertEqual(len(crossing.events), 1)
        event = crossing.events[0]
        self.assertAlmostEqual(event.timestamp_ms, 500.0)
        self.assertEqual(event.track_id, "cam_01:1")
        self.assertEqual(event.direction, "toward_camera")
        self.assertEqual(event.cumulative_count, 1)
        self.assertEqual(reverse.events, ())
        self.assertEqual(repeated.events, ())
        self.assertEqual(crossing.overlay.lines[0].count_total, 1)

    def test_reverse_direction_is_not_counted(self) -> None:
        self._update(_track(0, 0.5, 0.6))
        result = self._update(_track(1, 0.5, 0.4))
        self.assertEqual(result.events, ())

    def test_crossing_infinite_extension_outside_finite_line_is_not_counted(self) -> None:
        self._update(_track(0, 0.9, 0.4))
        result = self._update(_track(1, 0.9, 0.6))
        self.assertEqual(result.events, ())

    def test_first_observation_after_line_does_not_create_an_event(self) -> None:
        result = self._update(_track(5, 0.5, 0.7))
        self.assertEqual(result.events, ())

    def test_disallowed_vehicle_class_is_not_counted(self) -> None:
        self._update(_track(0, 0.5, 0.4, class_id=7))
        result = self._update(_track(1, 0.5, 0.6, class_id=7))
        self.assertEqual(result.events, ())

    def test_confidence_weighted_class_state_suppresses_single_frame_jitter(self) -> None:
        self._update(_track(0, 0.5, 0.4, class_id=2, confidence=0.9))
        result = self._update(_track(1, 0.5, 0.6, class_id=5, confidence=0.2))
        self.assertEqual(result.events[0].class_name, "car")

    def test_track_timestamp_and_frame_must_match_analytics_update(self) -> None:
        track = _track(1, 0.5, 0.4)
        with self.assertRaisesRegex(ValueError, "source frame"):
            self.engine.update(
                [track], source_frame_id=2, source_timestamp_ms=200.0
            )
        with self.assertRaisesRegex(ValueError, "source timestamp"):
            self.engine.update(
                [track], source_frame_id=1, source_timestamp_ms=200.0
            )


class MultiCameraTrafficAnalyticsTests(unittest.TestCase):
    def test_camera_state_is_independent(self) -> None:
        geometry = {
            "roi": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]],
            "lines": [
                {
                    "line_id": "main",
                    "p1": [0.2, 0.5],
                    "p2": [0.8, 0.5],
                    "crossing_direction": "negative_to_positive",
                    "direction_label": "toward_camera",
                }
            ],
        }
        analytics = MultiCameraTrafficAnalytics(
            {
                "normalized_coordinates": True,
                "cameras": {"cam_a": geometry, "cam_b": geometry},
            },
            ["cam_a", "cam_b"],
            canvas_width=100,
            canvas_height=100,
        )
        for camera_id in ("cam_a", "cam_b"):
            analytics.update(
                camera_id,
                [_track(0, 0.5, 0.4, camera_id=camera_id)],
                source_frame_id=0,
                source_timestamp_ms=0.0,
            )
            result = analytics.update(
                camera_id,
                [_track(1, 0.5, 0.6, camera_id=camera_id)],
                source_frame_id=1,
                source_timestamp_ms=100.0,
            )
            self.assertEqual(result.events[0].track_id, f"{camera_id}:1")
            self.assertEqual(result.events[0].cumulative_count, 1)

    def test_configuration_requires_exact_camera_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            MultiCameraTrafficAnalytics(
                {"cameras": {}},
                ["cam_01"],
                canvas_width=100,
                canvas_height=100,
            )

    def test_line_must_be_inside_normalized_roi(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside the ROI"):
            CameraAnalyticsConfig.from_dict(
                "cam_01",
                {
                    "roi": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                    "lines": [
                        {
                            "line_id": "bad",
                            "p1": [0.1, 0.5],
                            "p2": [0.7, 0.5],
                            "direction_label": "in",
                        }
                    ],
                },
            )


if __name__ == "__main__":
    unittest.main()
