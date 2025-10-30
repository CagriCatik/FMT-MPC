"""Geometry utilities shared between the planner, tracker, and simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Pose:
    """Compact representation of a planar pose with helper methods."""

    x: float
    y: float
    yaw: float

    def as_vector(self) -> np.ndarray:
        """Return the pose as a numeric vector ``[x, y, yaw]``."""

        return np.array([self.x, self.y, self.yaw], dtype=float)

    def distance(self, other: "Pose") -> float:
        """Compute Euclidean distance to ``other`` ignoring heading."""

        return hypot(self.x - other.x, self.y - other.y)


def heading_between(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Return the heading angle from ``p1`` to ``p2`` in radians."""

    return atan2(p2[1] - p1[1], p2[0] - p1[0])


def interpolate_path(points: Sequence[Tuple[float, float]], resolution: float) -> List[Tuple[float, float]]:
    """Densify ``points`` so consecutive samples are spaced by ``resolution``."""

    if not points:
        return []
    samples: List[Tuple[float, float]] = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        seg_len = hypot(end[0] - start[0], end[1] - start[1])
        if seg_len < resolution:
            samples.append(end)
            continue
        steps = max(1, int(seg_len / resolution))
        for step in range(1, steps + 1):
            ratio = step / steps
            samples.append(
                (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                )
            )
    return samples


def arc_length(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Compute cumulative arc length along a polyline."""

    if not points:
        return np.array([0.0])
    s = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        s.append(s[-1] + hypot(b[0] - a[0], b[1] - a[1]))
    return np.array(s)


def cubic_spline(points: Sequence[Tuple[float, float]], num_samples: int = 200) -> List[Tuple[float, float]]:
    """Return a centripetal Catmull–Rom spline through ``points``."""

    if not points:
        return []

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be a sequence of (x, y) pairs")

    # Remove consecutive duplicates which break the parametrisation. Keep the
    # original endpoints if interior points collapse.
    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mask = np.concatenate([[True], deltas > 1e-9])
    pts = pts[mask]
    if len(pts) == 1:
        return [(float(pts[0, 0]), float(pts[0, 1]))]
    if len(pts) == 2:
        dense: List[Tuple[float, float]] = [
            (float(pts[0, 0]), float(pts[0, 1])),
            (float(pts[1, 0]), float(pts[1, 1])),
        ]
        arc = arc_length(dense)
        total = float(arc[-1])
        if total <= 1e-9:
            return dense
        targets = np.linspace(0.0, total, num_samples)
        resampled: List[Tuple[float, float]] = []
        for s_val in targets:
            ratio = s_val / total
            x = dense[0][0] + ratio * (dense[1][0] - dense[0][0])
            y = dense[0][1] + ratio * (dense[1][1] - dense[0][1])
            resampled.append((float(x), float(y)))
        return resampled

    alpha = 0.5  # centripetal exponent avoids cusps when spacing is irregular

    padded = np.vstack([pts[0], pts, pts[-1]])

    def tj(ti: float, pi: np.ndarray, pj: np.ndarray) -> float:
        return ti + float(np.linalg.norm(pj - pi) ** alpha)

    t: List[float] = [0.0]
    for i in range(1, len(padded)):
        t.append(tj(t[-1], padded[i - 1], padded[i]))

    def catmull_rom_point(
        tt: float,
        t0: float,
        t1: float,
        t2: float,
        t3: float,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
    ) -> np.ndarray:
        if abs(t1 - t0) < 1e-9 or abs(t2 - t1) < 1e-9 or abs(t3 - t2) < 1e-9:
            return p1
        a1 = ((t1 - tt) / (t1 - t0)) * p0 + ((tt - t0) / (t1 - t0)) * p1
        a2 = ((t2 - tt) / (t2 - t1)) * p1 + ((tt - t1) / (t2 - t1)) * p2
        a3 = ((t3 - tt) / (t3 - t2)) * p2 + ((tt - t2) / (t3 - t2)) * p3
        b1 = ((t2 - tt) / (t2 - t0)) * a1 + ((tt - t0) / (t2 - t0)) * a2
        b2 = ((t3 - tt) / (t3 - t1)) * a2 + ((tt - t1) / (t3 - t1)) * a3
        return ((t2 - tt) / (t2 - t1)) * b1 + ((tt - t1) / (t2 - t1)) * b2

    total_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    spacing = max(total_length / max(num_samples - 1, 1), 1e-3)

    dense: List[Tuple[float, float]] = [(float(pts[0, 0]), float(pts[0, 1]))]
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        t0, t1, t2, t3 = t[i - 1], t[i], t[i + 1], t[i + 2]
        seg_len = float(np.linalg.norm(p2 - p1))
        seg_samples = max(3, int(np.ceil(seg_len / spacing)))
        ts = np.linspace(t1, t2, seg_samples, endpoint=False)
        for tt in ts[1:]:
            point = catmull_rom_point(tt, t0, t1, t2, t3, p0, p1, p2, p3)
            dense.append((float(point[0]), float(point[1])))
    dense.append((float(pts[-1, 0]), float(pts[-1, 1])))

    cleaned: List[Tuple[float, float]] = [dense[0]]
    for px, py in dense[1:]:
        lx, ly = cleaned[-1]
        if hypot(px - lx, py - ly) > 1e-6:
            cleaned.append((px, py))

    arc = arc_length(cleaned)
    total = float(arc[-1])
    if total <= 1e-9:
        return cleaned

    targets = np.linspace(0.0, total, num_samples)
    xs = np.array([p[0] for p in cleaned])
    ys = np.array([p[1] for p in cleaned])
    resampled: List[Tuple[float, float]] = []
    for s_val in targets:
        idx = int(np.searchsorted(arc, s_val, side="right"))
        if idx == 0:
            resampled.append((float(xs[0]), float(ys[0])))
            continue
        if idx >= len(arc):
            resampled.append((float(xs[-1]), float(ys[-1])))
            continue
        s0, s1 = arc[idx - 1], arc[idx]
        ratio = (s_val - s0) / max(s1 - s0, 1e-9)
        x = xs[idx - 1] + ratio * (xs[idx] - xs[idx - 1])
        y = ys[idx - 1] + ratio * (ys[idx] - ys[idx - 1])
        resampled.append((float(x), float(y)))

    return resampled


def annotate_waypoints(
    points: Sequence[Tuple[float, float]], nominal_speed: float
) -> List[Tuple[float, float, float, float]]:
    """Attach heading and nominal speed metadata to each waypoint."""

    if not points:
        return []
    annotated: List[Tuple[float, float, float, float]] = []
    padded: List[Tuple[float, float]] = [points[0], *points, points[-1]]
    for prev, curr, nxt in zip(padded[:-2], padded[1:-1], padded[2:]):
        yaw = heading_between(prev, nxt)
        annotated.append((curr[0], curr[1], yaw, nominal_speed))
    return annotated


def wrap_angle(angle: float) -> float:
    """Wrap ``angle`` to the range ``[-pi, pi]``."""

    return atan2(sin(angle), cos(angle))
