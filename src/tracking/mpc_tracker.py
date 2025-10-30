"""Model predictive controller and supporting corridor utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import logging
import numpy as np

try:  # pragma: no cover - executed when cvxpy is installed
    import cvxpy as cp
except ModuleNotFoundError:  # pragma: no cover - fallback path for minimal environments
    cp = None  # type: ignore[assignment]

from src.common.geometry import arc_length, wrap_angle

logger = logging.getLogger(__name__)


@dataclass
class TrackerState:
    """Pose and velocity fed into the MPC tracker."""

    x: float
    y: float
    yaw: float
    v: float


@dataclass
class ControlCommand:
    """Structured representation of the MPC output command."""

    steer: float
    accel: float
    feedforward: Dict[str, float]
    predicted_positions: List[Tuple[float, float]] = field(default_factory=list)


class Corridor:
    """Arc-length parameterised representation of the smoothed reference path."""

    def __init__(self, waypoints: Sequence[Tuple[float, float, float, float]]) -> None:
        """Store waypoints and pre-compute arc-length breakpoints."""

        self.waypoints = list(waypoints)
        self.positions = [(x, y) for x, y, _, _ in self.waypoints]
        self.arc_lengths = arc_length(self.positions)

    def total_length(self) -> float:
        """Return the full reference length in metres."""

        return float(self.arc_lengths[-1])

    def sample(self, s: float) -> Tuple[float, float, float, float]:
        """Return position, heading, and nominal speed at the requested arc length."""

        if s <= 0:
            return self.waypoints[0]
        if s >= self.arc_lengths[-1]:
            return self.waypoints[-1]
        idx = np.searchsorted(self.arc_lengths, s)
        prev_s = self.arc_lengths[idx - 1]
        next_s = self.arc_lengths[idx]
        ratio = (s - prev_s) / (next_s - prev_s)
        px, py, pyaw, ps = self.waypoints[idx - 1]
        nx, ny, nyaw, ns = self.waypoints[idx]
        yaw = wrap_angle(pyaw + ratio * wrap_angle(nyaw - pyaw))
        speed = ps + ratio * (ns - ps)
        return (
            px + ratio * (nx - px),
            py + ratio * (ny - py),
            yaw,
            speed,
        )

    def project(self, point: Tuple[float, float]) -> float:
        """Project a 2D point onto the corridor, returning the closest arc length."""

        px, py = point
        best_s = 0.0
        best_dist = float("inf")
        for idx in range(1, len(self.positions)):
            ax, ay = self.positions[idx - 1]
            bx, by = self.positions[idx]
            vx = bx - ax
            vy = by - ay
            seg_len = self.arc_lengths[idx] - self.arc_lengths[idx - 1]
            if seg_len <= 1e-6:
                continue
            t = ((px - ax) * vx + (py - ay) * vy) / (seg_len * seg_len)
            t = float(np.clip(t, 0.0, 1.0))
            proj_x = ax + t * vx
            proj_y = ay + t * vy
            dist = float(np.hypot(px - proj_x, py - proj_y))
            if dist < best_dist:
                best_dist = dist
                best_s = float(self.arc_lengths[idx - 1] + t * seg_len)
        return best_s


class MPCTracker:
    """Solve a convex MPC problem to steer the vehicle along the corridor."""

    def __init__(self, config, vehicle_limits) -> None:
        """Store configuration and vehicle limits for use in optimisation."""

        self.cfg = config
        self.vehicle_limits = vehicle_limits

    def compute_control(self, state: TrackerState, corridor: Corridor, dt: float) -> ControlCommand:
        """Solve the MPC problem (or fallback controller) and return the command."""

        horizon = self.cfg.horizon
        progress = corridor.project((state.x, state.y))
        total_length = corridor.total_length()
        refs: List[Tuple[float, float, float, float]] = []
        s = progress
        for _ in range(horizon + 1):
            target = corridor.sample(s)
            goal_distance = max(0.0, total_length - s)
            desired_speed = min(
                self.cfg.cruise_speed,
                target[3],
                self.cfg.cruise_speed * (goal_distance / (goal_distance + self.cfg.slowdown_radius)),
            )
            refs.append((target[0], target[1], target[2], desired_speed))
            s = min(total_length, s + max(desired_speed, 1.0) * dt)

        logger.debug(
            "MPC setup: progress %.2f/%.2f, horizon %d, dt %.2f",
            progress,
            total_length,
            horizon,
            dt,
        )

        wheelbase = getattr(self.cfg, "wheelbase", self.vehicle_limits.wheelbase)
        steer_limit = getattr(self.vehicle_limits, "max_steer", 0.6)
        accel_max = getattr(self.vehicle_limits, "max_accel", 3.0)
        accel_min = getattr(self.vehicle_limits, "max_decel", -5.0)

        if cp is None:
            logger.warning("cvxpy unavailable; falling back to heuristic controller")
            return self._fallback_control(
                state,
                refs,
                steer_limit,
                accel_min,
                accel_max,
            )

        X = cp.Variable((4, horizon + 1))
        U = cp.Variable((2, horizon))
        constraints = [
            X[:, 0] == np.array([state.x, state.y, state.yaw, state.v], dtype=float)
        ]

        cost = 0
        qx, qy, qyaw, qv = self.cfg.q_weights
        ru, ra = self.cfg.r_weights
        for k in range(horizon):
            ref_next = refs[k + 1]
            cos_yaw = float(np.cos(refs[k][2]))
            sin_yaw = float(np.sin(refs[k][2]))
            constraints.append(
                X[0, k + 1]
                == X[0, k] + dt * X[3, k] * cos_yaw
            )
            constraints.append(
                X[1, k + 1]
                == X[1, k] + dt * X[3, k] * sin_yaw
            )
            constraints.append(
                X[2, k + 1]
                == X[2, k] + dt * refs[k][3] * U[0, k] / max(wheelbase, 1e-3)
            )
            constraints.append(X[3, k + 1] == X[3, k] + dt * U[1, k])
            constraints.append(cp.abs(U[0, k]) <= steer_limit)
            constraints.append(U[1, k] <= accel_max)
            constraints.append(U[1, k] >= accel_min)
            constraints.append(X[3, k + 1] >= 0.0)
            pos_error = X[0:2, k + 1] - np.array(ref_next[:2])
            yaw_error = X[2, k + 1] - ref_next[2]
            speed_error = X[3, k + 1] - ref_next[3]
            cost += qx * cp.square(pos_error[0]) + qy * cp.square(pos_error[1])
            cost += qyaw * cp.square(yaw_error)
            cost += qv * cp.square(speed_error)
            cost += ru * cp.square(U[0, k]) + ra * cp.square(U[1, k])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True)
        logger.debug("MPC solver status: %s", problem.status)

        predicted_positions: List[Tuple[float, float]] = []
        if U.value is None or X.value is None:
            steer_cmd = 0.0
            accel_cmd = 0.0
            predicted_positions = [(float(px), float(py)) for px, py, _, _ in refs[1:]]
        else:
            steer_cmd = float(np.clip(U.value[0, 0], -steer_limit, steer_limit))
            accel_cmd = float(np.clip(U.value[1, 0], accel_min, accel_max))
            for k in range(1, horizon + 1):
                predicted_positions.append(
                    (
                        float(X.value[0, k]),
                        float(X.value[1, k]),
                    )
                )
        logger.info(
            "MPC command: steer=%.3f accel=%.3f progress=%.2f",
            steer_cmd,
            accel_cmd,
            progress,
        )

        target = refs[1] if len(refs) > 1 else refs[0]
        dx = target[0] - state.x
        dy = target[1] - state.y
        heading_error = wrap_angle(target[2] - state.yaw)
        cross_track = -dx * np.sin(state.yaw) + dy * np.cos(state.yaw)
        feedforward = {
            "target_x": target[0],
            "target_y": target[1],
            "target_yaw": target[2],
            "target_speed": target[3],
            "heading_error": heading_error,
            "cross_track": cross_track,
            "solver_status": problem.status,
        }
        return ControlCommand(
            steer=steer_cmd,
            accel=accel_cmd,
            feedforward=feedforward,
            predicted_positions=predicted_positions,
        )

    def _fallback_control(
        self,
        state: TrackerState,
        refs: Sequence[Tuple[float, float, float, float]],
        steer_limit: float,
        accel_min: float,
        accel_max: float,
    ) -> ControlCommand:
        """Fallback PD-like controller used when cvxpy is unavailable."""

        target = refs[1] if len(refs) > 1 else refs[0]
        dx = target[0] - state.x
        dy = target[1] - state.y
        heading_error = wrap_angle(target[2] - state.yaw)
        cross_track = -dx * np.sin(state.yaw) + dy * np.cos(state.yaw)
        steer_cmd = float(np.clip(heading_error + 0.5 * cross_track, -steer_limit, steer_limit))
        speed_error = target[3] - state.v
        accel_cmd = float(np.clip(1.0 * speed_error, accel_min, accel_max))
        feedforward = {
            "target_x": target[0],
            "target_y": target[1],
            "target_yaw": target[2],
            "target_speed": target[3],
            "heading_error": heading_error,
            "cross_track": cross_track,
            "solver_status": "fallback",
        }
        predicted_positions = [(float(px), float(py)) for px, py, _, _ in refs[1:]]
        logger.info(
            "Fallback command: steer=%.3f accel=%.3f target_speed=%.2f",
            steer_cmd,
            accel_cmd,
            target[3],
        )
        return ControlCommand(
            steer=steer_cmd,
            accel=accel_cmd,
            feedforward=feedforward,
            predicted_positions=predicted_positions,
        )
