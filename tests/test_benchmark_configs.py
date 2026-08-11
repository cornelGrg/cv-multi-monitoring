from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.main import require_active_provider


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
        self.tracking_with_output = yaml.safe_load(
            (PROJECT_ROOT / "configs/benchmark_realtime_tracking.yaml").read_text()
        )
        self.detection_with_audit = yaml.safe_load(
            (PROJECT_ROOT / "configs/benchmark_realtime_constant_load.yaml").read_text()
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

    def test_tracking_output_benchmark_requires_gpu_inference_and_encoding(self) -> None:
        self.assertTrue(self.tracking_with_output["inference"]["require_provider"])
        self.assertEqual(
            self.tracking_with_output["output"]["video_encoder"], "h264_nvenc"
        )

    def test_detection_audit_requires_gpu_inference_and_encoding(self) -> None:
        self.assertTrue(self.detection_with_audit["inference"]["require_provider"])
        self.assertEqual(
            self.detection_with_audit["output"]["video_encoder"], "h264_nvenc"
        )

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
