"""Rolling directional flow derived from accepted line-crossing events."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .line_crossing import LineCrossingEvent


@dataclass(frozen=True)
class FlowSnapshot:
    timestamp_ms: float
    source_frame_id: int
    camera_id: str
    line_id: str
    direction: str
    window_s: float
    observed_window_s: float
    crossings: int
    vehicles_per_minute: float
    warming_up: bool


class RollingTrafficFlow:
    """Count events in (now - window, now], using source time, per gate.

    Events are already deduplicated by the crossing engine. A heap handles
    interpolated timestamps that arrive out of chronological order. During
    startup, rates use the elapsed observation time and are marked warming up.
    """

    def __init__(
        self,
        camera_id: str,
        lines: dict[str, str],
        *,
        window_s: float = 60.0,
        sample_interval_s: float = 1.0,
    ) -> None:
        for name, value in (("window_s", window_s), ("sample_interval_s", sample_interval_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"analytics.flow.{name} must be finite and positive")
        self.camera_id = camera_id
        self.lines = dict(lines)
        self.window_s = float(window_s)
        self.sample_interval_ms = sample_interval_s * 1000.0
        self._timestamps: dict[str, list[float]] = {line: [] for line in lines}
        self._started_ms: float | None = None
        self._last_update_ms: float | None = None
        self._last_sample_ms: float | None = None

    def update(
        self,
        events: Iterable[LineCrossingEvent],
        *,
        source_frame_id: int,
        timestamp_ms: float,
    ) -> tuple[tuple[FlowSnapshot, ...], bool]:
        if not math.isfinite(timestamp_ms) or timestamp_ms < 0:
            raise ValueError("Flow source timestamp must be finite and nonnegative")
        if self._last_update_ms is not None and timestamp_ms < self._last_update_ms:
            raise ValueError("Flow source timestamps must not go backwards")
        if self._started_ms is None:
            self._started_ms = timestamp_ms
        self._last_update_ms = timestamp_ms
        cutoff = timestamp_ms - self.window_s * 1000.0
        for event in events:
            if event.camera_id != self.camera_id or self.lines.get(event.line_id) != event.direction:
                raise ValueError("Flow event must match its camera, line, and direction")
            if not math.isfinite(event.timestamp_ms) or not self._started_ms <= event.timestamp_ms <= timestamp_ms:
                raise ValueError("Flow event timestamp must lie in the observed source timeline")
            if event.timestamp_ms > cutoff:
                heapq.heappush(self._timestamps[event.line_id], event.timestamp_ms)

        elapsed_s = (timestamp_ms - self._started_ms) / 1000.0
        observed_s = min(self.window_s, elapsed_s)
        snapshots = []
        for line_id, direction in self.lines.items():
            timestamps = self._timestamps[line_id]
            while timestamps and timestamps[0] <= cutoff:
                heapq.heappop(timestamps)
            count = len(timestamps)
            snapshots.append(FlowSnapshot(
                timestamp_ms=timestamp_ms,
                source_frame_id=source_frame_id,
                camera_id=self.camera_id,
                line_id=line_id,
                direction=direction,
                window_s=self.window_s,
                observed_window_s=observed_s,
                crossings=count,
                vehicles_per_minute=count * 60.0 / observed_s if observed_s > 0 else 0.0,
                warming_up=elapsed_s < self.window_s,
            ))
        sample_due = (
            self._last_sample_ms is None
            or timestamp_ms - self._last_sample_ms >= self.sample_interval_ms
        )
        if sample_due:
            self._last_sample_ms = timestamp_ms
        return tuple(snapshots), sample_due
