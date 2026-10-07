"""Line-crossing logic, kept free of any model code so it is easy to unit test."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple

Point = Tuple[float, float]


def _side(p: Point, a: Point, b: Point) -> float:
    """Signed area: >0 if p is left of a->b, <0 if right, 0 on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _within_segment(p: Point, a: Point, b: Point) -> bool:
    """True if the projection of p falls between a and b."""
    ax, ay = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / max(ax * ax + ay * ay, 1e-9)
    return 0.0 <= t <= 1.0


@dataclass
class LineCounter:
    """Counts tracked objects whose centre crosses a line segment.

    "in" = crossed to the positive side of a->b (for a left-to-right horizontal line in
    image coordinates, that means moving downward); "out" = the reverse.
    Each track ID is counted at most once per direction.
    """

    start: Point
    end: Point
    last_side: Dict[int, float] = field(default_factory=dict)
    counted: Dict[int, set] = field(default_factory=lambda: defaultdict(set))
    in_count: int = 0
    out_count: int = 0
    per_class: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"in": 0, "out": 0})
    )

    def update(self, track_id: int, centre: Point, label: str = "object") -> str | None:
        """Feed one tracked centre. Returns "in"/"out" when a crossing happens."""
        side = _side(centre, self.start, self.end)
        prev = self.last_side.get(track_id)
        self.last_side[track_id] = side if side != 0 else (prev or 0)
        if prev is None or prev == 0 or side == 0 or (prev > 0) == (side > 0):
            return None
        if not _within_segment(centre, self.start, self.end):
            return None
        direction = "in" if prev < 0 < side else "out"
        if direction in self.counted[track_id]:
            return None
        self.counted[track_id].add(direction)
        if direction == "in":
            self.in_count += 1
        else:
            self.out_count += 1
        self.per_class[label][direction] += 1
        return direction

    @property
    def total(self) -> int:
        return self.in_count + self.out_count
