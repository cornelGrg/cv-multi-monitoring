import unittest

import numpy as np

from src.pipeline.orchestrator import filter_detections


class DetectionFilterTests(unittest.TestCase):
    def test_only_confident_vehicle_classes_are_returned(self) -> None:
        output = np.array(
            [[[0, 0, 10, 10, 0.9, 2], [0, 0, 10, 10, 0.8, 0], [0, 0, 10, 10, 0.1, 7]]],
            dtype=np.float32,
        )
        filtered = filter_detections(output, 0.25, [2, 3, 5, 7])
        self.assertEqual(filtered[0].shape, (1, 6))
        self.assertEqual(int(filtered[0][0, 5]), 2)


if __name__ == "__main__":
    unittest.main()
