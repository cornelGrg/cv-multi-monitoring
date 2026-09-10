"""Measure detection, tracking, and the final full pipeline in isolated runs."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Preserve the existing `python scripts/run_controlled_benchmarks.py` entry point.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.provenance import environment, sha256

BENCHMARKS = {
    'detection_compute': PROJECT_ROOT / 'configs/benchmark_realtime_detection_compute.yaml',
    'tracking_compute': PROJECT_ROOT / 'configs/benchmark_realtime_tracking_compute.yaml',
    'full_pipeline': PROJECT_ROOT / 'configs/default.yaml',
}


def isolated_config(config, run_dir):
    config = copy.deepcopy(config)
    config['manifest_file'] = str((PROJECT_ROOT / config['manifest_file']).resolve())
    config['output']['metrics_file'] = str(run_dir / 'metrics.json')
    tracking = config.setdefault('tracking', {})
    tracking.update(tracks_file=str(run_dir / 'tracks.csv'), annotated_video_dir=str(run_dir))
    if config.get('analytics', {}).get('enabled'):
        config['analytics'].update(events_file=str(run_dir / 'events.csv'), counts_file=str(run_dir / 'counts.csv'))
        config['analytics']['flow']['file'] = str(run_dir / 'flow.csv')
    return config


def summarize(runs):
    result = {}
    for key in runs[0]['global']:
        values = [r['global'].get(key) for r in runs]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            result[key] = {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'runs': values}
    return result


def validate_flow(run_dir, config):
    def read(name):
        with (run_dir / name).open() as stream:
            return list(csv.DictReader(stream))
    frames, events, samples = read('frames.csv'), read('events.csv'), read('flow.csv')
    timeline = {(r['camera_id'], int(r['source_frame_id'])): float(r['timestamp_ms']) for r in frames}
    starts = {}
    for row in frames:
        cam = row['camera_id']
        starts[cam] = min(starts.get(cam, float('inf')), float(row['timestamp_ms']))
    expected_pairs = {(cam, line['line_id'], line['direction_label'])
                      for cam, geometry in config['analytics']['cameras'].items() for line in geometry['lines']}
    pairs = set()
    for row in samples:
        cam, line, direction = row['camera_id'], row['line_id'], row['direction']
        frame, now = int(row['source_frame_id']), float(row['timestamp_ms'])
        pairs.add((cam, line, direction))
        if timeline.get((cam, frame)) != now:
            raise ValueError('Flow sample has no matching processed frame')
        window = config['analytics']['flow']['window_s']
        if float(row['window_s']) != window:
            raise ValueError('Flow window differs from configuration')
        observed = min(window, (now - starts[cam]) / 1000)
        count = sum(e['camera_id'] == cam and e['line_id'] == line and e['direction'] == direction
                    and int(e['source_frame_id']) <= frame
                    and now - window * 1000 < float(e['timestamp_ms']) <= now for e in events)
        rate = count * 60 / observed if observed else 0
        if count != int(row['crossings']) or abs(rate - float(row['vehicles_per_minute'])) > .001:
            raise ValueError('Flow sample disagrees with accepted crossing events')
        if abs(observed - float(row['observed_window_s'])) > .001 or (row['warming_up'] == 'True') != (observed < window):
            raise ValueError('Invalid flow startup normalization')
    if pairs != expected_pairs:
        raise ValueError('Missing or unexpected camera/direction flow samples')
    return len(samples)


def validate_result(result, config, run_dir):
    g, meta = result['global'], result['run_metadata']
    expected = config['pipeline']['max_duration_s']
    if not expected - 1 <= g['elapsed_s'] <= expected + 5:
        raise ValueError('Run duration outside expected range')
    if meta['active_providers'][0] != 'CUDAExecutionProvider':
        raise ValueError('CUDA fallback invalidates the benchmark')
    if g['num_cameras'] != 4 or meta['batch_size'] != 4 or meta['input_size'] != 640:
        raise ValueError('Unexpected camera/batch/input configuration')
    if g['total_frames_inferred'] <= 0 or not math.isfinite(g['fps_global']):
        raise ValueError('No valid processed frames')
    if sum(c['frames_inferred'] for c in result['per_camera'].values()) != g['total_frames_inferred']:
        raise ValueError('Per-camera frame totals do not match')
    if config.get('analytics', {}).get('enabled'):
        if not meta['traffic_flow_enabled'] or meta['traffic_flow_window_s'] != 15 or meta['show_detections']:
            raise ValueError('Unexpected final flow/overlay settings')
        if meta['video_encoder'] != 'h264_nvenc':
            raise ValueError('Final run requires NVENC')
        for filename in ('frames.csv', 'config_snapshot.yaml', 'events.csv', 'counts.csv', 'flow.csv', 'tracks.csv'):
            if not (run_dir / filename).is_file():
                raise ValueError(f'Missing final output: {filename}')
        with (run_dir / 'frames.csv').open() as stream:
            if sum(1 for _ in csv.DictReader(stream)) != g['total_frames_inferred']:
                raise ValueError('Incomplete frame index')
        with (run_dir / 'events.csv').open() as stream:
            if sum(1 for _ in csv.DictReader(stream)) != g['total_line_crossing_events']:
                raise ValueError('Event accounting mismatch')
        if g['video_frames_written'] + g['video_frames_dropped'] != g['video_frames_submitted']:
            raise ValueError('Video accounting mismatch')
        checked_samples = validate_flow(run_dir, config)
        (run_dir / 'validation.json').write_text(json.dumps({'status': 'passed', 'flow_samples_checked': checked_samples}, indent=2))
        for camera in result['per_camera']:
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_entries', 'stream=codec_name,nb_frames', '-of', 'json',
                str(run_dir / f'{camera}.mp4')], text=True))['streams'][0]
            if probe['codec_name'] != 'h264' or int(probe['nb_frames']) != result['per_camera'][camera]['video_frames_written']:
                raise ValueError('Encoded video does not match metrics')


def run_suite(runs_per_benchmark=3, resume=None, resume_note=None):
    configs = {name: yaml.safe_load(path.read_text()) for name, path in BENCHMARKS.items()}
    for section in ('model', 'pipeline', 'inference', 'manifest_file'):
        if any(config[section] != configs['detection_compute'][section] for config in configs.values()):
            raise ValueError(f'Benchmark inputs differ in {section}; resolve before measuring')
    suite_dir = Path(resume).resolve() if resume else PROJECT_ROOT / 'outputs/final_benchmarks' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not resume:
        suite_dir.mkdir(parents=True)
    provenance = environment()
    model = PROJECT_ROOT / configs['full_pipeline']['model']['path']
    manifest_path = PROJECT_ROOT / configs['full_pipeline']['manifest_file']
    manifest = yaml.safe_load(manifest_path.read_text())
    provenance['inputs_sha256'] = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in
        [model, manifest_path, *[PROJECT_ROOT / c['video_path'] for c in manifest['cameras']]]}
    summary_path = suite_dir / 'summary.json'
    provenance_name = 'provenance.json'
    if resume:
        previous = json.loads((suite_dir / 'provenance.json').read_text())
        summary = json.loads(summary_path.read_text())
        if summary['runs_per_benchmark'] != runs_per_benchmark:
            raise ValueError('Resume run count differs from the original suite')
        if previous['inputs_sha256'] != provenance['inputs_sha256'] or previous['packages'] != provenance['packages']:
            raise ValueError('Resume requires unchanged model, sources, manifest and package versions')
        if previous['source_sha256'] != provenance['source_sha256'] and not resume_note:
            raise ValueError('Code changed; provide --resume-note explaining why completed measurements remain valid')
        provenance_name = 'resume_provenance_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json'
        summary.setdefault('resumptions', []).append({'provenance': provenance_name, 'note': resume_note})
        summary['status'] = 'running'
    else:
        (suite_dir / 'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
        summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'runs_per_benchmark': runs_per_benchmark,
                   'status': 'running', 'benchmarks': {}, 'video_retention': 'All measured videos retained; showcase rendering is outside measurement.'}
    (suite_dir / provenance_name).write_text(json.dumps(provenance, indent=2))
    for name, config in configs.items():
        runs, paths = [], []
        for number in range(1, runs_per_benchmark + 1):
            run_dir = suite_dir / name / f'run_{number}'
            isolated = isolated_config(config, run_dir)
            config_path = run_dir / 'run_config.yaml'
            if (run_dir / 'metrics.json').exists() and json.loads((run_dir / 'execution.json').read_text())['exit_code'] == 0:
                if yaml.safe_load(config_path.read_text()) != isolated:
                    raise ValueError(f'Completed configuration changed: {run_dir}')
                result = json.loads((run_dir / 'metrics.json').read_text())
                validate_result(result, isolated, run_dir)
                print(f'Reusing validated {name} run {number}', flush=True)
            else:
                if run_dir.exists():
                    failed = suite_dir / 'failed' / name / (run_dir.name + '_' + datetime.now().strftime('%H%M%S_%f'))
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    run_dir.rename(failed)
                run_dir.mkdir(parents=True)
                config_path.write_text(yaml.safe_dump(isolated, sort_keys=False))
                record = {'started_at': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha256(config_path),
                          'provenance_file': provenance_name}
                print(f'{name}: {number}/{runs_per_benchmark} -> {run_dir}', flush=True)
                with (run_dir / 'run.log').open('w') as log:
                    process = subprocess.run([sys.executable, '-m', 'src.main', '--config', str(config_path)],
                                             cwd=PROJECT_ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                             stdout=log, stderr=subprocess.STDOUT)
                record.update(finished_at=datetime.now(timezone.utc).isoformat(), exit_code=process.returncode)
                (run_dir / 'execution.json').write_text(json.dumps(record, indent=2))
                process.check_returncode()
                result = json.loads((run_dir / 'metrics.json').read_text())
                validate_result(result, isolated, run_dir)
            runs.append(result)
            paths.append(str(run_dir.relative_to(suite_dir)))
            summary['benchmarks'][name] = {'run_dirs': paths.copy(), 'statistics': summarize(runs)}
            summary_path.write_text(json.dumps(summary, indent=2))
    summary['status'] = 'complete'
    full = summary['benchmarks']['full_pipeline']
    fps = full['statistics']['fps_global']
    representative = min(range(len(fps['runs'])), key=lambda i: abs(fps['runs'][i] - fps['median']))
    summary['representative_run'] = full['run_dirs'][representative]
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'Completed suite: {summary_path}', flush=True)
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--resume', type=Path, help='Reuse validated runs from an interrupted suite')
    parser.add_argument('--resume-note', help='Explain why a code change does not invalidate completed runs')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    run_suite(args.runs, args.resume, args.resume_note)


if __name__ == '__main__':
    main()
