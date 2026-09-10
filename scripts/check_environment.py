"""Check demo inputs, CUDA execution, NVENC, and optional export prediction parity."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.export_model import inspect_model
from scripts.provenance import environment, sha256
from src.capture.producer import letterbox
from src.inference.engine import OnnxGpuEngine
from src.main import load_config, require_active_provider
from src.output.annotator import select_h264_encoder

ROOT = Path(__file__).resolve().parents[1]


def compare_detections(reference, candidate):
    """Match accepted vehicle detections independent of output row ordering."""
    max_box_error, max_score_error, matches = 0., 0., 0
    for old, new in zip(reference, candidate, strict=True):
        old = old[(old[:, 4] > .25) & np.isin(old[:, 5], [2, 3, 5, 7])]
        new = new[(new[:, 4] > .25) & np.isin(new[:, 5], [2, 3, 5, 7])]
        if len(old) != len(new):
            raise ValueError('Export changed the number of accepted detections')
        remaining = list(new)
        for row in old:
            choices = [(np.max(np.abs(row[:4] - other[:4])), i) for i, other in enumerate(remaining)
                       if row[5] == other[5]]
            if not choices:
                raise ValueError('Export changed detection classes')
            error, index = min(choices)
            other = remaining.pop(index)
            score_error = float(abs(row[4] - other[4]))
            if error > 1.0 or score_error > .01:
                raise ValueError(f'Export parity failed: box delta={error}, confidence delta={score_error}')
            max_box_error = max(max_box_error, float(error))
            max_score_error = max(max_score_error, score_error)
            matches += 1
    return {'matches': matches, 'max_box_error_pixels': max_box_error, 'max_confidence_error': max_score_error}


def check(config_path, compare=None):
    config = load_config(config_path)
    model_path = ROOT / config['model']['path']
    model = inspect_model(model_path)
    engine = OnnxGpuEngine(model_path, warmup_batches=1)
    require_active_provider(engine, {'require_provider': True, 'provider': 'CUDAExecutionProvider'})
    encoder = select_h264_encoder(config['output']['video_encoder'], 640, 640)
    candidate = None
    if compare:
        inspect_model(compare)
        candidate = OnnxGpuEngine(compare, warmup_batches=1)
        require_active_provider(candidate, {'require_provider': True, 'provider': 'CUDAExecutionProvider'})
    cameras = config['streams']
    if len(cameras) != 4:
        raise ValueError('Expected four source videos')
    results, sources = [], {}
    for second in (0, 5, 10):
        frames = []
        for camera in cameras:
            path = ROOT / camera['path']
            if not path.is_file():
                raise FileNotFoundError(path)
            cap = cv2.VideoCapture(str(path))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(second * config['pipeline']['source_fps']))
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode {camera["camera_id"]} at {second}s')
                frames.append(letterbox(image))
            finally:
                cap.release()
            if second == 0:
                sources[camera['camera_id']] = sha256(path)
        output, _ = engine.infer(np.stack(frames))
        if output.shape != (4, 300, 6) or not np.isfinite(output).all():
            raise ValueError('Invalid inference output')
        result = {'source_second': second, 'shape': list(output.shape)}
        if candidate:
            other, _ = candidate.infer(np.stack(frames))
            result['parity'] = compare_detections(output, other)
        results.append(result)
    return {'status': 'passed', 'model': model, 'active_providers': engine.providers,
            'encoder': encoder, 'sources_sha256': sources, 'checks': results,
            'candidate_sha256': sha256(compare) if compare else None, 'environment': environment()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/phase4_analytics_demo.yaml')
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--report', type=Path, default=ROOT / 'artifacts/reproducibility/environment.json')
    args = parser.parse_args()
    report = check(args.config, args.compare)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f'Environment checks passed: {args.report}')


if __name__ == '__main__':
    main()
