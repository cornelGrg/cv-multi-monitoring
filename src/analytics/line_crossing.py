"""Polygonal ROI filtering and directional line-crossing analytics.

All configured coordinates are normalized to the model's letterboxed image
canvas. The bottom-center of each tracked bounding box is used as the vehicle's
road-contact point.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from src.tracking.bytetrack import Track, VEHICLE_CLASSES
from .traffic_flow import FlowSnapshot, RollingTrafficFlow

Point = tuple[float, float]

_CLASS_IDS_BY_NAME = {name: class_id for class_id, name in VEHICLE_CLASSES.items()}
_DIRECTIONS = {"negative_to_positive", "positive_to_negative"}


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _subtract(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _parse_point(value: object, field_name: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two coordinates")
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} coordinates must be numeric") from exc
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"{field_name} coordinates must be finite")
    if not all(0.0 <= coordinate <= 1.0 for coordinate in point):
        raise ValueError(f"{field_name} coordinates must be normalized to [0, 1]")
    return point


def _parse_allowed_classes(values: object) -> frozenset[int]:
    if values is None:
        return frozenset(VEHICLE_CLASSES)
    if not isinstance(values, (list, tuple, set)) or not values:
        raise ValueError("allowed_classes must be a non-empty list")

    class_ids: set[int] = set()
    for value in values:
        if isinstance(value, str) and not value.isdigit():
            try:
                class_id = _CLASS_IDS_BY_NAME[value.lower()]
            except KeyError as exc:
                raise ValueError(f"Unsupported vehicle class {value!r}") from exc
        else:
            try:
                class_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid vehicle class {value!r}") from exc
        if class_id not in VEHICLE_CLASSES:
            raise ValueError(f"Class {class_id} is not a configured vehicle class")
        class_ids.add(class_id)
    return frozenset(class_ids)


def _point_on_segment(point: Point, start: Point, end: Point, epsilon: float = 1e-9) -> bool:
    segment = _subtract(end, start)
    relative = _subtract(point, start)
    if abs(_cross(segment, relative)) > epsilon:
        return False
    return (
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return True for points inside or on the boundary of a polygon."""
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        if (current[1] > point[1]) != (previous[1] > point[1]):
            intersection_x = (
                (previous[0] - current[0])
                * (point[1] - current[1])
                / (previous[1] - current[1])
                + current[0]
            )
            if point[0] < intersection_x:
                inside = not inside
        previous = current
    return inside


def _polygon_area(polygon: Sequence[Point]) -> float:
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) / 2.0


def _segment_intersection(
    movement_start: Point,
    movement_end: Point,
    line_start: Point,
    line_end: Point,
    epsilon: float = 1e-9,
) -> tuple[float, Point] | None:
    """Return movement fraction and point when two finite segments intersect."""
    movement = _subtract(movement_end, movement_start)
    line = _subtract(line_end, line_start)
    denominator = _cross(movement, line)
    if abs(denominator) <= epsilon:
        return None

    offset = _subtract(line_start, movement_start)
    movement_fraction = _cross(offset, line) / denominator
    line_fraction = _cross(offset, movement) / denominator
    if not (-epsilon <= movement_fraction <= 1.0 + epsilon):
        return None
    if not (-epsilon <= line_fraction <= 1.0 + epsilon):
        return None

    movement_fraction = min(1.0, max(0.0, movement_fraction))
    point = (
        movement_start[0] + movement_fraction * movement[0],
        movement_start[1] + movement_fraction * movement[1],
    )
    return movement_fraction, point


@dataclass(frozen=True)
class DirectedLineConfig:
    """One finite virtual line and the only movement direction it accepts."""

    line_id: str
    p1: Point
    p2: Point
    crossing_direction: str
    direction_label: str
    lane_label: str
    allowed_class_ids: frozenset[int]

    @classmethod
    def from_dict(cls, values: dict, *, field_prefix: str) -> "DirectedLineConfig":
        if not isinstance(values, dict):
            raise ValueError(f"{field_prefix} must be a mapping")
        line_id = str(values.get("line_id", "")).strip()
        direction_label = str(values.get("direction_label", "")).strip()
        if not line_id:
            raise ValueError(f"{field_prefix}.line_id cannot be empty")
        if not direction_label:
            raise ValueError(f"{field_prefix}.direction_label cannot be empty")

        p1 = _parse_point(values.get("p1"), f"{field_prefix}.p1")
        p2 = _parse_point(values.get("p2"), f"{field_prefix}.p2")
        if p1 == p2:
            raise ValueError(f"{field_prefix} endpoints must be different")
        crossing_direction = str(
            values.get("crossing_direction", "negative_to_positive")
        )
        if crossing_direction not in _DIRECTIONS:
            raise ValueError(
                f"{field_prefix}.crossing_direction must be one of {sorted(_DIRECTIONS)}"
            )
        return cls(
            line_id=line_id,
            p1=p1,
            p2=p2,
            crossing_direction=crossing_direction,
            direction_label=direction_label,
            lane_label=str(values.get("lane_label", "")).strip(),
            allowed_class_ids=_parse_allowed_classes(values.get("allowed_classes")),
        )

    def side(self, point: Point, epsilon: float) -> int:
        signed_distance = _cross(_subtract(self.p2, self.p1), _subtract(point, self.p1))
        if abs(signed_distance) <= epsilon:
            return 0
        return 1 if signed_distance > 0 else -1

    def accepts_transition(self, previous_side: int, current_side: int) -> bool:
        if self.crossing_direction == "negative_to_positive":
            return previous_side < 0 and current_side > 0
        return previous_side > 0 and current_side < 0


@dataclass(frozen=True)
class CameraAnalyticsConfig:
    camera_id: str
    roi: tuple[Point, ...]
    lines: tuple[DirectedLineConfig, ...]

    @classmethod
    def from_dict(cls, camera_id: str, values: dict) -> "CameraAnalyticsConfig":
        if not isinstance(values, dict):
            raise ValueError(f"analytics.cameras.{camera_id} must be a mapping")
        raw_roi = values.get("roi")
        if not isinstance(raw_roi, (list, tuple)) or len(raw_roi) < 3:
            raise ValueError(f"analytics.cameras.{camera_id}.roi needs at least 3 points")
        roi = tuple(
            _parse_point(point, f"analytics.cameras.{camera_id}.roi[{index}]")
            for index, point in enumerate(raw_roi)
        )
        if _polygon_area(roi) <= 1e-9:
            raise ValueError(f"analytics.cameras.{camera_id}.roi has zero area")

        raw_lines = values.get("lines")
        if not isinstance(raw_lines, (list, tuple)) or not raw_lines:
            raise ValueError(f"analytics.cameras.{camera_id}.lines cannot be empty")
        lines = tuple(
            DirectedLineConfig.from_dict(
                line,
                field_prefix=f"analytics.cameras.{camera_id}.lines[{index}]",
            )
            for index, line in enumerate(raw_lines)
        )
        line_ids = [line.line_id for line in lines]
        if len(line_ids) != len(set(line_ids)):
            raise ValueError(f"analytics.cameras.{camera_id} contains duplicate line IDs")
        for line in lines:
            if not _point_in_polygon(line.p1, roi) or not _point_in_polygon(line.p2, roi):
                raise ValueError(
                    f"analytics line {camera_id}/{line.line_id} must lie inside the ROI"
                )
        return cls(camera_id=camera_id, roi=roi, lines=lines)


@dataclass(frozen=True)
class LineCrossingEvent:
    timestamp_ms: float
    source_frame_id: int
    camera_id: str
    track_id: str
    local_track_id: int
    event_type: str
    line_id: str
    direction: str
    lane_label: str
    class_id: int
    class_name: str
    crossing_x: float
    crossing_y: float
    cumulative_count: int


@dataclass(frozen=True)
class OverlayLine:
    line_id: str
    p1: Point
    p2: Point
    crossing_direction: str
    direction_label: str
    lane_label: str
    count_total: int
    counts_by_class: tuple[tuple[str, int], ...]
    flow: FlowSnapshot | None = None


@dataclass(frozen=True)
class AnalyticsOverlay:
    roi: tuple[Point, ...]
    lines: tuple[OverlayLine, ...]


@dataclass(frozen=True)
class AnalyticsFrameResult:
    events: tuple[LineCrossingEvent, ...]
    overlay: AnalyticsOverlay
    flow: tuple[FlowSnapshot, ...] = ()
    flow_sample_due: bool = False


@dataclass
class _TrackState:
    point: Point
    source_timestamp_ms: float
    source_frame_id: int
    last_seen_timestamp_ms: float
    sides: dict[str, int] = field(default_factory=dict)
    counted_lines: set[str] = field(default_factory=set)
    class_scores: Counter[int] = field(default_factory=Counter)

    @property
    def stable_class_id(self) -> int:
        return max(self.class_scores, key=self.class_scores.get)


class PerCameraTrafficAnalytics:
    """Stateful ROI and line-crossing engine for exactly one camera."""

    def __init__(
        self,
        config: CameraAnalyticsConfig,
        *,
        canvas_width: int,
        canvas_height: int,
        state_ttl_ms: float = 10_000.0,
        side_epsilon: float = 0.0005,
        flow_config: dict | None = None,
    ) -> None:
        if canvas_width <= 0 or canvas_height <= 0:
            raise ValueError("analytics canvas dimensions must be positive")
        if state_ttl_ms <= 0:
            raise ValueError("analytics.state_ttl_ms must be positive")
        if side_epsilon < 0:
            raise ValueError("analytics.side_epsilon cannot be negative")
        self.config = config
        self.canvas_width = int(canvas_width)
        self.canvas_height = int(canvas_height)
        self.state_ttl_ms = float(state_ttl_ms)
        self.side_epsilon = float(side_epsilon)
        self._states: dict[int, _TrackState] = {}
        self._counts: Counter[tuple[str, str, int]] = Counter()
        flow_config = flow_config or {}
        self._flow = (
            RollingTrafficFlow(
                config.camera_id,
                {line.line_id: line.direction_label for line in config.lines},
                window_s=float(flow_config.get("window_s", 60.0)),
                sample_interval_s=float(flow_config.get("sample_interval_s", 1.0)),
            )
            if flow_config.get("enabled", False) else None
        )

    def _track_point(self, track: Track) -> Point:
        return (
            track.center_x / self.canvas_width,
            track.y2 / self.canvas_height,
        )

    def _prune(self, source_timestamp_ms: float) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if source_timestamp_ms - state.last_seen_timestamp_ms > self.state_ttl_ms
        ]
        for track_id in stale_ids:
            del self._states[track_id]

    def _overlay(self, flow: tuple[FlowSnapshot, ...] = ()) -> AnalyticsOverlay:
        flow_by_line = {snapshot.line_id: snapshot for snapshot in flow}
        overlay_lines = []
        for line in self.config.lines:
            class_counts = tuple(
                (VEHICLE_CLASSES[class_id], self._counts[(line.line_id, line.direction_label, class_id)])
                for class_id in sorted(line.allowed_class_ids)
                if self._counts[(line.line_id, line.direction_label, class_id)] > 0
            )
            overlay_lines.append(
                OverlayLine(
                    line_id=line.line_id,
                    p1=line.p1,
                    p2=line.p2,
                    crossing_direction=line.crossing_direction,
                    direction_label=line.direction_label,
                    lane_label=line.lane_label,
                    count_total=sum(count for _, count in class_counts),
                    counts_by_class=class_counts,
                    flow=flow_by_line.get(line.line_id),
                )
            )
        return AnalyticsOverlay(self.config.roi, tuple(overlay_lines))

    def update(
        self,
        tracks: Iterable[Track],
        *,
        source_frame_id: int,
        source_timestamp_ms: float,
    ) -> AnalyticsFrameResult:
        """Process one tracked source frame and return new crossing events."""
        self._prune(source_timestamp_ms)
        events: list[LineCrossingEvent] = []
        seen_local_ids: set[int] = set()

        for track in tracks:
            if track.camera_id != self.config.camera_id:
                raise ValueError(
                    f"Tracker camera {track.camera_id!r} does not match analytics camera "
                    f"{self.config.camera_id!r}"
                )
            if track.source_frame_id != source_frame_id:
                raise ValueError("All tracks must belong to the source frame being analysed")
            if not math.isclose(
                track.source_timestamp_ms,
                source_timestamp_ms,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("All tracks must preserve the analysed source timestamp")
            if track.local_track_id in seen_local_ids:
                raise ValueError(f"Duplicate local track ID {track.local_track_id} in one frame")
            seen_local_ids.add(track.local_track_id)

            point = self._track_point(track)
            state = self._states.get(track.local_track_id)
            if state is None:
                state = _TrackState(
                    point=point,
                    source_timestamp_ms=source_timestamp_ms,
                    source_frame_id=source_frame_id,
                    last_seen_timestamp_ms=source_timestamp_ms,
                )
                state.class_scores[track.class_id] += max(track.confidence, 0.01)
                for line in self.config.lines:
                    side = line.side(point, self.side_epsilon)
                    if side:
                        state.sides[line.line_id] = side
                self._states[track.local_track_id] = state
                continue

            state.class_scores[track.class_id] += max(track.confidence, 0.01)
            stable_class_id = state.stable_class_id
            for line in self.config.lines:
                current_side = line.side(point, self.side_epsilon)
                previous_side = state.sides.get(line.line_id)
                if (
                    line.line_id not in state.counted_lines
                    and stable_class_id in line.allowed_class_ids
                    and previous_side is not None
                    and current_side != 0
                    and line.accepts_transition(previous_side, current_side)
                ):
                    intersection = _segment_intersection(
                        state.point,
                        point,
                        line.p1,
                        line.p2,
                    )
                    if intersection is not None:
                        fraction, crossing_point = intersection
                        if _point_in_polygon(crossing_point, self.config.roi):
                            crossing_timestamp_ms = state.source_timestamp_ms + fraction * (
                                source_timestamp_ms - state.source_timestamp_ms
                            )
                            count_key = (
                                line.line_id,
                                line.direction_label,
                                stable_class_id,
                            )
                            self._counts[count_key] += 1
                            state.counted_lines.add(line.line_id)
                            events.append(
                                LineCrossingEvent(
                                    timestamp_ms=crossing_timestamp_ms,
                                    source_frame_id=source_frame_id,
                                    camera_id=self.config.camera_id,
                                    track_id=track.track_id,
                                    local_track_id=track.local_track_id,
                                    event_type="line_crossing",
                                    line_id=line.line_id,
                                    direction=line.direction_label,
                                    lane_label=line.lane_label,
                                    class_id=stable_class_id,
                                    class_name=VEHICLE_CLASSES[stable_class_id],
                                    crossing_x=crossing_point[0],
                                    crossing_y=crossing_point[1],
                                    cumulative_count=self._counts[count_key],
                                )
                            )
                if current_side:
                    state.sides[line.line_id] = current_side

            state.point = point
            state.source_timestamp_ms = source_timestamp_ms
            state.source_frame_id = source_frame_id
            state.last_seen_timestamp_ms = source_timestamp_ms

        flow, sample_due = (), False
        if self._flow is not None:
            flow, sample_due = self._flow.update(
                events, source_frame_id=source_frame_id, timestamp_ms=source_timestamp_ms,
            )
        return AnalyticsFrameResult(tuple(events), self._overlay(flow), flow, sample_due)


class MultiCameraTrafficAnalytics:
    """Own one independent analytics state machine per configured camera."""

    def __init__(
        self,
        values: dict,
        camera_ids: Sequence[str],
        *,
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        if not isinstance(values, dict):
            raise ValueError("analytics configuration must be a mapping")
        if values.get("normalized_coordinates", True) is not True:
            raise ValueError("Phase 4 supports normalized analytics coordinates only")
        raw_cameras = values.get("cameras")
        if not isinstance(raw_cameras, dict):
            raise ValueError("analytics.cameras must be a mapping keyed by camera_id")

        configured = set(raw_cameras)
        expected = set(camera_ids)
        if configured != expected:
            missing = sorted(expected - configured)
            extra = sorted(configured - expected)
            raise ValueError(
                f"Analytics camera configuration mismatch; missing={missing}, extra={extra}"
            )

        state_ttl_ms = float(values.get("state_ttl_ms", 10_000.0))
        side_epsilon = float(values.get("side_epsilon", 0.0005))
        self._engines = {
            camera_id: PerCameraTrafficAnalytics(
                CameraAnalyticsConfig.from_dict(camera_id, raw_cameras[camera_id]),
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                state_ttl_ms=state_ttl_ms,
                side_epsilon=side_epsilon,
                flow_config=values.get("flow"),
            )
            for camera_id in camera_ids
        }

    def update(
        self,
        camera_id: str,
        tracks: Iterable[Track],
        *,
        source_frame_id: int,
        source_timestamp_ms: float,
    ) -> AnalyticsFrameResult:
        try:
            engine = self._engines[camera_id]
        except KeyError as exc:
            raise ValueError(f"No analytics engine configured for {camera_id!r}") from exc
        return engine.update(
            tracks,
            source_frame_id=source_frame_id,
            source_timestamp_ms=source_timestamp_ms,
        )

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self._engines)
