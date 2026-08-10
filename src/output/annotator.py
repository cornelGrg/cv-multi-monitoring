import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# COCO labels for traffic classes
COCO_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

class VideoAnnotator:
    """Renders annotated video frames to an MP4 file."""

    def __init__(self, output_path: str | Path, fps: float, width: int, height: int) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        self.frames_written = 0
        logger.info("Started VideoAnnotator at %s (%dx%d @ %.1ffps)", self.output_path, self.width, self.height, self.fps)

    def write_frame(self, frame: np.ndarray, detections: np.ndarray) -> None:
        """Annotate and write a single frame.
        
        detections: shape (N, 6) where each row is [x1, y1, x2, y2, conf, class_id]
        """
        # We need a writable copy if frame is read-only
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)
            
            label = f"{COCO_CLASSES.get(cls_id, str(cls_id))} {conf:.2f}"
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        self.writer.write(annotated)
        self.frames_written += 1

    def write_tracks(self, frame: np.ndarray, tracks: Iterable[object]) -> None:
        """Write a frame labelled with vehicle class and camera-local track ID."""
        annotated = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = (int(track.x1), int(track.y1), int(track.x2), int(track.y2))
            label = f"{track.class_name} {track.track_id} {track.confidence:.2f}"
            color = (0, 200, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_top = max(0, y1 - text_h - 4)
            cv2.rectangle(annotated, (x1, label_top), (x1 + text_w, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1, max(text_h, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        self.writer.write(annotated)
        self.frames_written += 1

    def release(self) -> None:
        if self.writer:
            self.writer.release()
            logger.info("Finished writing %d frames to %s", self.frames_written, self.output_path)
