"""Small provenance helpers shared by benchmark and export tools."""
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f'unavailable: {type(error).__name__}'


def environment():
    # Hash actual source bytes, including untracked implementation files.
    source = {}
    for directory in ('src', 'scripts', 'configs', 'tests'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.suffix in ('.py', '.yaml', '.sh'):
                source[str(path.relative_to(ROOT))] = sha256(path)
    for path in sorted(ROOT.glob('requirements*.txt')):
        source[path.name] = sha256(path)
    return {
        'revision': command_output(['git', 'rev-parse', 'HEAD']),
        'git_status': command_output(['git', 'status', '--short']),
        'source_sha256': source,
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())),
        'gpu': command_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
        'ffmpeg': command_output(['ffmpeg', '-version']).splitlines()[0],
        'seeds': 'No deterministic scheduling guarantee; warm-up uses random inputs, measured inputs are fixed video files.',
    }
