import copy
import unittest
from pathlib import Path

import yaml

from scripts.run_controlled_benchmarks import BENCHMARKS, isolated_config, summarize, validate_result


class FinalBenchmarkTests(unittest.TestCase):
    def test_nested_suite_redirects_every_output_without_mutating_source(self):
        source = yaml.safe_load(BENCHMARKS['full_pipeline'].read_text())
        original = copy.deepcopy(source)
        directory = Path('/tmp/final-suite/full/run_1')
        result = isolated_config(source, directory)
        self.assertEqual(source, original)
        self.assertTrue(Path(result['manifest_file']).is_absolute())
        for value in [result['output']['metrics_file'], result['tracking']['tracks_file'],
                      result['analytics']['events_file'], result['analytics']['counts_file'],
                      result['analytics']['flow']['file']]:
            self.assertEqual(Path(value).parent, directory)
        self.assertEqual(Path(result['tracking']['annotated_video_dir']), directory)

    def test_statistics_preserve_spread_and_do_not_pool_percentiles(self):
        runs = [{'global': {'fps_global': v, 'latency_e2e_p95_ms': 100 + v}} for v in [10, 40, 20]]
        stats = summarize(runs)
        self.assertEqual(stats['fps_global'], dict(median=20, min=10, max=40, runs=[10, 40, 20]))
        self.assertEqual(stats['latency_e2e_p95_ms']['median'], 120)

    def test_cpu_fallback_and_incomplete_runs_are_rejected(self):
        config = {'pipeline': {'max_duration_s': 60}}
        result = {'global': {'elapsed_s': 60}, 'run_metadata': {'active_providers': ['CPUExecutionProvider']}}
        with self.assertRaisesRegex(ValueError, 'CUDA fallback'):
            validate_result(result, config, Path('/tmp'))
        result['global']['elapsed_s'] = 10
        with self.assertRaisesRegex(ValueError, 'duration'):
            validate_result(result, config, Path('/tmp'))

    def test_full_output_initialization_writes_resolved_configuration(self):
        import tempfile
        from unittest.mock import patch
        from src.main import load_config
        from src.metrics.collector import MetricsCollector
        from src.pipeline.orchestrator import PipelineOrchestrator
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = isolated_config(load_config(BENCHMARKS['full_pipeline']), root)
            config['tracking']['write_annotated_video'] = False
            collector = MetricsCollector([s['camera_id'] for s in config['streams']])
            with patch('src.pipeline.orchestrator.PerCameraByteTracker'):
                pipeline = PipelineOrchestrator(config, None, collector)
                try:
                    pipeline._create_producers()
                    self.assertEqual(yaml.safe_load((root/'config_snapshot.yaml').read_text()), config)
                    self.assertTrue((root/'frames.csv').exists())
                finally:
                    if pipeline._tracks_writer:
                        pipeline._tracks_writer.close()
                    if pipeline._analytics_writer:
                        pipeline._analytics_writer.close()
