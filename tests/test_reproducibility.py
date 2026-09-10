import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.check_environment import compare_detections
from scripts.prepare_ua_detrac import image_sequence, extract_sequence


class ReproducibilityTests(unittest.TestCase):
    def test_detection_parity_matches_order_and_rejects_missing_or_shifted_boxes(self):
        first = np.array([[[0, 0, 10, 10, .9, 2], [20, 20, 30, 30, .8, 5]]])
        self.assertEqual(compare_detections(first, first[:, ::-1])['matches'], 2)
        changed = first.copy(); changed[0, 0, 0] += 3
        with self.assertRaisesRegex(ValueError, 'parity'):
            compare_detections(first, changed)
        with self.assertRaisesRegex(ValueError, 'number'):
            compare_detections(first, first[:, :1])

    def test_image_numbering_supports_non_one_start_and_rejects_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['img00005.jpg', 'img00006.jpg']:
                (root / name).touch()
            _, start, count = image_sequence(root)
            self.assertEqual((start, count), (5, 2))
            (root / 'img00008.jpg').touch()
            with self.assertRaisesRegex(ValueError, 'Non-contiguous'):
                image_sequence(root)

    def test_zip_extraction_matches_sequence_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'data.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('images/MVI_20011/img00001.jpg', b'frame')
                stream.writestr('images/MVI_200110/img00001.jpg', b'wrong')
                stream.writestr('annotations/MVI_20011.xml', b'<sequence/>')
            extract_sequence(archive, 'MVI_20011', root/'images/MVI_20011', root/'annotations')
            self.assertEqual((root/'images/MVI_20011/img00001.jpg').read_bytes(), b'frame')
            with self.assertRaisesRegex(ValueError, 'Archive must contain'):
                extract_sequence(archive, 'MVI_20012', root/'images/MVI_20012', root/'annotations')

    def test_prepare_four_sources_and_repeat_without_reencoding(self):
        import cv2
        from scripts.prepare_ua_detrac import prepare, DEFAULT_SEQUENCES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sequence in DEFAULT_SEQUENCES:
                images = root / 'data/raw/ua_detrac/images' / sequence
                annotations = root / 'data/raw/ua_detrac/annotations'
                images.mkdir(parents=True); annotations.mkdir(parents=True, exist_ok=True)
                (annotations / f'{sequence}.xml').write_text('<sequence/>')
                for frame in (5, 6):
                    cv2.imwrite(str(images / f'img{frame:05d}.jpg'), np.full((48, 64, 3), frame * 20, np.uint8))
            manifest = prepare('owner/dataset', DEFAULT_SEQUENCES, root=root)
            self.assertTrue(manifest.exists())
            video = root / 'data/processed/videos/cam_01.mp4'
            modified = video.stat().st_mtime_ns
            prepare('owner/dataset', DEFAULT_SEQUENCES, root=root)
            self.assertEqual(video.stat().st_mtime_ns, modified)
            old_manifest = manifest.read_bytes()
            (root / 'data/raw/ua_detrac/annotations/MVI_20052.xml').unlink()
            with self.assertRaises(FileNotFoundError):
                prepare('owner/dataset', DEFAULT_SEQUENCES, root=root)
            self.assertEqual(manifest.read_bytes(), old_manifest)
