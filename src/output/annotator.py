import logging
import subprocess
import threading
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.analytics.line_crossing import AnalyticsOverlay

logger = logging.getLogger(__name__)

_auto_encoders: dict[tuple[int, int], str] = {}
_encoder_probe_results: dict[tuple[str, int, int], bool] = {}
_encoder_lock = threading.RLock()

# COCO labels for traffic classes
COCO_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _probe_encoder(encoder: str, width: int, height: int) -> bool:
    """Return whether FFmpeg can initialise an encoder in this environment."""
    cache_key = (encoder, width, height)
    with _encoder_lock:
        cached = _encoder_probe_results.get(cache_key)
    if cached is not None:
        return cached

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:d=0.04",
        "-frames:v",
        "1",
        "-c:v",
        encoder,
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        usable = False
    else:
        usable = result.returncode == 0
    with _encoder_lock:
        _encoder_probe_results[cache_key] = usable
    return usable


def select_h264_encoder(requested: str, width: int, height: int) -> str:
    """Select NVENC when usable, otherwise portable libx264."""
    if requested not in {"auto", "h264_nvenc", "libx264"}:
        raise ValueError("video_encoder must be 'auto', 'h264_nvenc', or 'libx264'")
    if requested != "auto":
        if not _probe_encoder(requested, width, height):
            raise RuntimeError(
                f"FFmpeg encoder {requested!r} is not usable at {width}x{height}"
            )
        return requested

    dimensions = (width, height)
    with _encoder_lock:
        if dimensions not in _auto_encoders:
            selected = (
                "h264_nvenc"
                if _probe_encoder("h264_nvenc", width, height)
                else "libx264"
            )
            if not _probe_encoder(selected, width, height):
                raise RuntimeError("Neither h264_nvenc nor libx264 is usable through FFmpeg")
            _auto_encoders[dimensions] = selected
        return _auto_encoders[dimensions]


class VideoAnnotator:
    """Render annotations and stream browser-compatible H.264 to FFmpeg."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        *,
        encoder: str = "auto",
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.encoder = select_h264_encoder(encoder, width, height)
        self._released = False

        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.encoder,
        ]
        if self.encoder == "h264_nvenc":
            command.extend(
                ["-preset", "p4", "-tune", "ll", "-cq", "23", "-b:v", "0"]
            )
        else:
            command.extend(["-preset", "veryfast", "-crf", "23", "-threads", "1"])
        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-tag:v",
                "avc1",
                "-movflags",
                "+faststart",
                str(self.output_path),
            ]
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frames_written = 0
        logger.info(
            "Started VideoAnnotator at %s (%dx%d @ %.1ffps, encoder=%s)",
            self.output_path,
            self.width,
            self.height,
            self.fps,
            self.encoder,
        )

    def _read_stderr(self) -> str:
        if self._process.stderr is None or self._process.stderr.closed:
            return "unknown FFmpeg error"
        output = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        self._process.stderr.close()
        return output

    def _write(self, frame: np.ndarray) -> None:
        """Write one BGR frame to the FFmpeg raw-video pipe."""
        if self._released or self._process.stdin is None:
            raise RuntimeError("VideoAnnotator is closed")
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Expected frame shape {(self.height, self.width, 3)}, got {frame.shape}"
            )
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            stderr = self._read_stderr()
            raise RuntimeError(f"FFmpeg video writer failed: {stderr}") from exc
        self.frames_written += 1

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
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - 4),
                (x1 + text_w, y1),
                (0, 255, 0),
                -1,
            )

            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        self._write(annotated)

    def _draw_analytics(
        self,
        frame: np.ndarray,
        overlay: AnalyticsOverlay,
    ) -> None:
        """Draw the ROI, directed lines and immutable counter snapshot in place."""
        def pixel(point: tuple[float, float]) -> tuple[int, int]:
            return (
                min(self.width - 1, max(0, round(point[0] * self.width))),
                min(self.height - 1, max(0, round(point[1] * self.height))),
            )

        roi = np.asarray([pixel(point) for point in overlay.roi], dtype=np.int32)
        tint = frame.copy()
        cv2.fillPoly(tint, [roi], (180, 0, 180))
        cv2.addWeighted(tint, 0.10, frame, 0.90, 0.0, frame)
        cv2.polylines(frame, [roi], True, (255, 0, 255), 2, cv2.LINE_AA)

        panel_lines: list[str] = []
        for line in overlay.lines:
            start = pixel(line.p1)
            end = pixel(line.p2)
            color = (70, 255, 70)
            cv2.line(frame, start, end, color, 3, cv2.LINE_AA)

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = max(1.0, float(np.hypot(dx, dy)))
            normal_x = -dy / length
            normal_y = dx / length
            if line.crossing_direction == "positive_to_negative":
                normal_x = -normal_x
                normal_y = -normal_y
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            arrow_end = (
                round(midpoint[0] + normal_x * 32),
                round(midpoint[1] + normal_y * 32),
            )
            cv2.arrowedLine(
                frame,
                midpoint,
                arrow_end,
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.35,
            )
            cv2.putText(
                frame,
                line.line_id,
                (start[0] + 4, max(16, start[1] - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            class_counts = " ".join(
                f"{class_name}:{count}" for class_name, count in line.counts_by_class
            )
            detail = f" ({class_counts})" if class_counts else ""
            lane = f" [{line.lane_label}]" if line.lane_label else ""
            panel_lines.append(
                f"{line.direction_label}{lane}: {line.count_total}{detail}"
            )

        if panel_lines:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            line_height = 21
            panel_width = min(
                self.width - 16,
                max(cv2.getTextSize(text, font, font_scale, 1)[0][0] for text in panel_lines)
                + 18,
            )
            panel_height = len(panel_lines) * line_height + 10
            cv2.rectangle(frame, (8, 8), (8 + panel_width, 8 + panel_height), (0, 0, 0), -1)
            for index, text in enumerate(panel_lines):
                cv2.putText(
                    frame,
                    text,
                    (16, 27 + index * line_height),
                    font,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    def write_tracks(
        self,
        frame: np.ndarray,
        tracks: Iterable[object],
        *,
        analytics_overlay: AnalyticsOverlay | None = None,
    ) -> None:
        """Write tracked vehicles and an optional traffic-analytics overlay."""
        annotated = frame.copy()
        if analytics_overlay is not None:
            self._draw_analytics(annotated, analytics_overlay)
        for track in tracks:
            x1, y1, x2, y2 = (int(track.x1), int(track.y1), int(track.x2), int(track.y2))
            # The file itself is camera-scoped; keep the overlay compact and
            # retain the globally unambiguous camera:id form in tracks.csv.
            label = f"{track.class_name} #{track.local_track_id}"
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
        self._write(annotated)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait(timeout=30)
        stderr = self._read_stderr()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg exited with code {return_code} for {self.output_path}: {stderr}"
            )
        logger.info(
            "Finished writing %d H.264 frames to %s",
            self.frames_written,
            self.output_path,
        )
