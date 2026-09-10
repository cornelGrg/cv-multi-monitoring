from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.analytics.line_crossing import MultiCameraTrafficAnalytics
from src.main import load_config, require_active_provider


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _FakeEngine:
    def __init__(self, providers: list[str]) -> None:
        self.providers = providers


class ControlledBenchmarkConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detection = yaml.safe_load(
            (PROJECT_ROOT / "configs/benchmark_realtime_detection_compute.yaml").read_text()
        )
        self.tracking = yaml.safe_load(
            (PROJECT_ROOT / "configs/benchmark_realtime_tracking_compute.yaml").read_text()
        )
        self.demo = yaml.safe_load(
            (PROJECT_ROOT / "configs/default.yaml").read_text()
        )

    def test_controlled_pair_differs_only_in_tracking_and_metrics_path(self) -> None:
        for section in ("model", "inference", "pipeline", "manifest_file"):
            self.assertEqual(self.detection[section], self.tracking[section])

        self.assertFalse(self.detection["tracking"]["enabled"])
        self.assertTrue(self.tracking["tracking"]["enabled"])
        self.assertFalse(self.tracking["tracking"]["write_annotated_video"])
        self.assertFalse(self.tracking["tracking"]["write_tracks_csv"])

    def test_controlled_pair_requires_cuda_provider(self) -> None:
        self.assertTrue(self.detection["inference"]["require_provider"])
        self.assertEqual(self.detection["inference"]["provider"], "CUDAExecutionProvider")

    def test_default_demo_has_complete_per_camera_analytics_and_gpu_requirements(self) -> None:
        loaded = load_config(PROJECT_ROOT / "configs/default.yaml")
        camera_ids = [stream["camera_id"] for stream in loaded["streams"]]
        analytics = MultiCameraTrafficAnalytics(
            loaded["analytics"],
            camera_ids,
            canvas_width=loaded["model"]["input_size"],
            canvas_height=loaded["model"]["input_size"],
        )

        self.assertTrue(self.demo["inference"]["require_provider"])
        self.assertEqual(self.demo["output"]["video_encoder"], "h264_nvenc")
        self.assertEqual(analytics.camera_ids, tuple(camera_ids))
        self.assertTrue(self.demo["analytics"]["write_csv"])

    def test_missing_manifest_reports_the_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-manifest.yaml"
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(yaml.safe_dump({"manifest_file": str(missing)}))
            with self.assertRaises(FileNotFoundError) as error:
                load_config(config_path)
            self.assertEqual(Path(error.exception.filename), missing)

    def test_provider_check_rejects_cpu_fallback(self) -> None:
        config = {"provider": "CUDAExecutionProvider", "require_provider": True}
        with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
            require_active_provider(_FakeEngine(["CPUExecutionProvider"]), config)

    def test_provider_check_accepts_cuda_as_primary(self) -> None:
        config = {"provider": "CUDAExecutionProvider", "require_provider": True}
        require_active_provider(
            _FakeEngine(["CUDAExecutionProvider", "CPUExecutionProvider"]), config
        )


if __name__ == "__main__":
    unittest.main()
