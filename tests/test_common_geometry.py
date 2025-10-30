from __future__ import annotations

import math

from src.common.geometry import annotate_waypoints, arc_length, cubic_spline, heading_between, wrap_angle


def test_heading_between():
    assert math.isclose(heading_between((0.0, 0.0), (1.0, 0.0)), 0.0, abs_tol=1e-6)
    assert math.isclose(heading_between((0.0, 0.0), (0.0, 1.0)), math.pi / 2, abs_tol=1e-6)


def test_arc_length_monotonic():
    waypoints = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    arc = arc_length(waypoints)
    assert list(arc) == [0.0, 1.0, 2.0]


def test_annotate_waypoints_nominal_speed():
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]
    annotated = annotate_waypoints(points, nominal_speed=3.5)
    assert len(annotated) == len(points)
    _, _, yaw, speed = annotated[1]
    assert math.isclose(speed, 3.5)
    assert -math.pi <= yaw <= math.pi


def test_wrap_angle_bounds():
    assert wrap_angle(0.0) == 0.0
    wrapped = wrap_angle(3 * math.pi)
    assert -math.pi <= wrapped <= math.pi


def test_cubic_spline_softens_corner():
    base = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    smoothed = cubic_spline(base, num_samples=20)
    assert len(smoothed) == 20
    assert smoothed[0] == (0.0, 0.0)
    assert smoothed[-1] == (2.0, 2.0)
    # The smoothed path should stay within the bounding box of the inputs.
    for x, y in smoothed:
        assert 0.0 <= x <= 2.0
        assert 0.0 <= y <= 2.0
    # Ensure the curve bends before the final waypoint, not staying as a right angle.
    assert any(y > 0.0 for (_, y) in smoothed[1:-1])
