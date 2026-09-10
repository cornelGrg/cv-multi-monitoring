"""Generate portfolio figures from a completed benchmark suite."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/traffic-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scripts.provenance import sha256

COLORS = {'toward_camera': '#248c74', 'away_from_camera': '#bd751b'}
LABELS = ['Detection\ncompute only', 'Detection + tracking\ncompute only', 'Full pipeline\nCSV + video']
ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / 'data/manifests/manual_count_reference.yaml'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def manual_comparison(events):
    references = yaml.safe_load(REFERENCE_PATH.read_text())['comparisons']
    return [{**r, 'automated': sum(e['camera_id'] == r['camera_id'] and e['line_id'] == r['line_id']
             and e['direction'] == r['direction']
             and r['start_s'] * 1000 <= float(e['timestamp_ms']) < r['end_s'] * 1000 for e in events)}
            for r in references]



def plot(suite, destination):
    summary = json.loads((suite / 'summary.json').read_text())
    if summary['status'] != 'complete' or summary['runs_per_benchmark'] < 3:
        raise ValueError('Figures require a complete suite with at least three runs per configuration')
    run = suite / summary['representative_run']
    # The manual totals apply only to the originally reviewed source geometry.
    config = yaml.safe_load((run / 'config_snapshot.yaml').read_text())
    reference = yaml.safe_load(REFERENCE_PATH.read_text())
    execution = json.loads((run / 'execution.json').read_text())
    provenance_path = suite / execution.get('provenance_file', 'provenance.json')
    provenance = json.loads(provenance_path.read_text())
    for cam in ('cam_01', 'cam_02'):
        if config['analytics']['cameras'][cam] != reference['cameras'][cam]['geometry']:
            raise ValueError('Manual reference geometry differs from the measured configuration')
        source = next(c.get('video_path', c.get('path')) for c in config['streams'] if c['camera_id'] == cam)
        if provenance['inputs_sha256'].get(source) != reference['cameras'][cam]['source_sha256']:
            raise ValueError('Manual reference source video differs from the measured input')
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})
    def save(figure, name):
        figure.savefig(destination / f'{name}.png', dpi=170, bbox_inches='tight')
        figure.savefig(destination / f'{name}.svg', bbox_inches='tight')
        plt.close(figure)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    names = ['detection_compute', 'tracking_compute', 'full_pipeline']
    for ax, key, title, unit in zip(axes, ['fps_global', 'latency_e2e_p95_ms'],
                                    ['Processed throughput', 'Consumer latency'], ['Frames / second', 'Run p95 (ms)']):
        stats = [summary['benchmarks'][name]['statistics'][key] for name in names]
        ax.bar(range(3), [s['median'] for s in stats], color=['#92b1bb', '#508c9d', '#248c74'], width=.6)
        for i, stat in enumerate(stats):
            ax.scatter(np.linspace(i - .10, i + .10, len(stat['runs'])), stat['runs'], color='#202d35', zorder=4, s=22)
            ax.text(i, max(stat['runs']) * 1.025, f"{stat['median']:.1f}", ha='center')
        ax.set(xticks=range(3), xticklabels=LABELS, ylabel=unit, title=title)
        ax.set_ylim(0, max(max(s['runs']) for s in stats) * 1.18)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Three 60-second runs per configuration · bars: median · dots: individual runs')
    save(fig, 'performance')

    data = rows(run / 'flow.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True, layout='constrained')
    for ax, cam in zip(axes.flat, sorted(config['analytics']['cameras'])):
        for direction, color in COLORS.items():
            samples = [r for r in data if r['camera_id'] == cam and r['direction'] == direction]
            x = np.array([float(r['timestamp_ms']) / 1000 for r in samples])
            y = np.array([float(r['vehicles_per_minute']) for r in samples])
            # Break gaps exceeding twice the requested sampling cadence.
            gap = np.r_[False, np.diff(x) > config['analytics']['flow']['sample_interval_s'] * 2]
            y[gap] = np.nan
            ax.step(x, y, where='post', label=direction.replace('_', ' ').capitalize(), color=color)
        ax.axvspan(0, 15, color='#ccd1d5', alpha=.3)
        ax.set(title=cam.replace('_', ' ').upper(), xlabel='Source time (s)', ylabel='Vehicles / minute')
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Directional flow · 15s window · shaded: startup\nIndependent file sources, clips loop')
    save(fig, 'traffic_flow')

    comparison = manual_comparison(rows(run / 'events.csv'))
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    x = np.arange(4)
    ax.bar(x - .18, [r['manual'] for r in comparison], width=.36, color='#a3adb5', label='Manual')
    ax.bar(x + .18, [r['automated'] for r in comparison], width=.36, color='#248c74', label='Automated')
    for i, r in enumerate(comparison):
        ax.text(i, max(r['manual'], r['automated']) + .6, f"|difference| = {abs(r['automated'] - r['manual'])}", ha='center', fontsize=9)
    ax.set(xticks=x, xticklabels=['Cam 01\nToward', 'Cam 01\nAway', 'Cam 02\nToward', 'Cam 02\nAway'],
           ylabel='Crossing count', title='Manual count comparison · 23s for Cam 01, 30s for Cam 02')
    ax.set_ylim(0, max(max(r['manual'], r['automated']) for r in comparison) + 5)
    ax.legend(); ax.grid(axis='y', alpha=.15)
    save(fig, 'count_comparison')
    source_files = [suite/'summary.json', provenance_path, run/'flow.csv', run/'events.csv', run/'config_snapshot.yaml']
    # Publish names/hashes and small numeric summaries, not private paths or raw media.
    report = {'suite': suite.name, 'representative_run': summary['representative_run'],
              'environment': {key: provenance[key] for key in ['python', 'platform', 'gpu', 'ffmpeg']},
              'inputs_sha256': provenance['inputs_sha256'],
              'source_code_sha256': provenance['source_sha256'],
              'resumptions': summary.get('resumptions', []),
              'plot_script_sha256': sha256(Path(__file__)),
              'source_sha256': {str(p.relative_to(suite)): sha256(p) for p in source_files},
              'benchmark_statistics': {name: summary['benchmarks'][name]['statistics'] for name in names},
              'manual_comparison': comparison, 'manual_reference_sha256': sha256(REFERENCE_PATH), 'window_s': 15,
              'regenerate': f'python -m scripts.plot_results --suite outputs/final_benchmarks/{suite.name}',
              'limitations': 'Aggregate count differences are not event precision/recall. Latency excludes asynchronous encoding completion.'}
    (destination / 'results.json').write_text(json.dumps(report, indent=2))
    print(f'Figures and provenance: {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/results')
    args = parser.parse_args()
    plot(args.suite, args.output)


if __name__ == '__main__':
    main()
