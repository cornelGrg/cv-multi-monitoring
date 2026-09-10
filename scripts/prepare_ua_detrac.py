"""Prepare four UA-DETRAC sequences, reusing validated local inputs."""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCES = ['MVI_20011', 'MVI_20012', 'MVI_20032', 'MVI_20052']


def image_sequence(directory):
    images = sorted(directory.glob('*.jpg'))
    if not images:
        raise ValueError(f'No JPEG frames in {directory}')
    match = re.fullmatch(r'(.*?)(\d+)\.jpg', images[0].name)
    if not match:
        raise ValueError('Expected numbered JPEG frames')
    prefix, first = match.groups()
    start, digits = int(first), len(first)
    expected = [directory / f'{prefix}{number:0{digits}d}.jpg' for number in range(start, start + len(images))]
    if images != expected:
        raise ValueError(f'Non-contiguous or mixed image sequence: {directory}')
    return str(directory / f'{prefix}%0{digits}d.jpg'), start, len(images)


def extract_sequence(archive, sequence, image_dir, annotation_dir):
    with zipfile.ZipFile(archive) as stream:
        frames = [entry for entry in stream.infolist() if sequence in Path(entry.filename).parts
                  and entry.filename.lower().endswith('.jpg')]
        annotations = [entry for entry in stream.infolist() if Path(entry.filename).name == f'{sequence}.xml']
        if not frames or len(annotations) != 1:
            raise ValueError(f'Archive must contain frames and one annotation for {sequence}')
        names = [Path(entry.filename).name for entry in frames]
        if len(names) != len(set(names)):
            raise ValueError(f'Ambiguous duplicate frame names for {sequence}')
        image_dir.parent.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=image_dir.parent) as temporary:
            staging = Path(temporary)
            for entry in frames:
                (staging / Path(entry.filename).name).write_bytes(stream.read(entry))
            image_sequence(staging)
            if image_dir.exists():
                raise FileExistsError(f'Refusing to replace incomplete image directory {image_dir}; inspect it first')
            staging.rename(image_dir)
        (annotation_dir / f'{sequence}.xml').write_bytes(stream.read(annotations[0]))


def video_valid(path, frames):
    if not path.is_file():
        return False
    try:
        result = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,r_frame_rate', '-of', 'json', str(path)], text=True)
        info = json.loads(result)['streams'][0]
        return int(info['nb_frames']) == frames and info['r_frame_rate'] == '25/1'
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError):
        return False


def prepare(slug, sequences, archive=None, root=ROOT):
    if len(sequences) != 4 or len(set(sequences)) != 4 or any(not re.fullmatch(r'MVI_\d+', s) for s in sequences):
        raise ValueError('Specify exactly four distinct MVI_<number> sequences')
    if not re.fullmatch(r'[\w-]+/[\w-]+', slug):
        raise ValueError('Dataset slug must be owner/dataset')
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            raise RuntimeError(f'Missing {executable}')
    raw = root / 'data/raw/ua_detrac'
    cameras = []
    for number, sequence in enumerate(sequences, 1):
        images = raw / 'images' / sequence
        annotations = raw / 'annotations'
        if not images.exists():
            if archive is None:
                archive = raw / '_archives' / f'{slug.split("/")[1]}.zip'
                if not archive.exists():
                    if not shutil.which('kaggle'):
                        raise RuntimeError('Install/configure Kaggle CLI, or pass --archive with the exact ZIP path')
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(['kaggle', 'datasets', 'download', '-d', slug, '-p', str(archive.parent)], check=True)
            extract_sequence(archive, sequence, images, annotations)
        if not (annotations / f'{sequence}.xml').is_file():
            raise FileNotFoundError(f'Missing annotation for {sequence}')
        pattern, start, count = image_sequence(images)
        camera = f'cam_{number:02d}'
        video = root / 'data/processed/videos' / f'{camera}.mp4'
        # A matching frame count alone cannot establish source identity. Record
        # sequence identity in a sidecar for newly generated videos.
        identity = video.with_suffix('.source.txt')
        reuse = identity.exists() and identity.read_text().strip() == sequence and video_valid(video, count)
        if not reuse:
            video.parent.mkdir(parents=True, exist_ok=True)
            temporary = video.with_name(f'{video.stem}.partial.mp4')
            try:
                subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '25', '-start_number', str(start),
                    '-i', pattern, '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(temporary)], check=True)
                if not video_valid(temporary, count):
                    raise RuntimeError(f'Incomplete encoded video for {sequence}')
                temporary.replace(video)
                identity.write_text(sequence + '\n')
            finally:
                temporary.unlink(missing_ok=True)
        cameras.append(dict(camera_id=camera, sequence_id=sequence,
                            video_path=str(video.relative_to(root)),
                            annotation_path=str((annotations / f'{sequence}.xml').relative_to(root)), scenario='unclassified'))
        print(f'{camera}: {sequence}, {count} frames ({"reused" if reuse else "encoded"})', flush=True)
    manifest = root / 'data/manifests/ua_detrac_selected.yaml'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    data = {'dataset': {'name': 'UA-DETRAC', 'source_fps': 25, 'stream_relationship': 'independent_sequences',
                        'notes': 'Independent traffic-camera sequences, not synchronized views of one road network.'},
            'cameras': cameras}
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(yaml.safe_dump(data, sort_keys=False))
    temporary.replace(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='Kaggle owner/dataset slug')
    parser.add_argument('sequences', nargs='*', default=DEFAULT_SEQUENCES)
    parser.add_argument('--archive', type=Path, help='Use this exact local ZIP; do not download')
    args = parser.parse_args()
    print(prepare(args.dataset, args.sequences or DEFAULT_SEQUENCES, args.archive))


if __name__ == '__main__':
    main()
