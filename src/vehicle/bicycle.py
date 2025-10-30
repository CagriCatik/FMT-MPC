"""Kinematic bicycle model used by the MPC-driven simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.common.geometry import wrap_angle


@dataclass
class BicycleState:
    """Pose and scalar speed describing the bicycle state."""

    x: float
    y: float
    yaw: float
    v: float


class BicycleModel:
    """Kinematic bicycle dynamics with configurable limits."""

    def __init__(self, config) -> None:
        """Store the configuration dataclass for use during integration."""

        self.cfg = config

    def step(self, state: BicycleState, accel: float, steer: float, dt: float) -> BicycleState:
        """Advance the vehicle dynamics by ``dt`` seconds using clipped inputs."""

        steer_clipped = float(np.clip(steer, -self.cfg.max_steer, self.cfg.max_steer))
        accel_clipped = float(np.clip(accel, self.cfg.max_decel, self.cfg.max_accel))
        v = max(0.0, state.v + accel_clipped * dt)
        beta = np.arctan(0.5 * np.tan(steer_clipped))
        x = state.x + v * np.cos(state.yaw + beta) * dt
        y = state.y + v * np.sin(state.yaw + beta) * dt
        yaw = wrap_angle(
            state.yaw + v * np.sin(beta) / max(self.cfg.wheelbase, 1e-3) * dt
        )
        return BicycleState(x=x, y=y, yaw=yaw, v=v)

    def compute_forces(self, state: BicycleState, accel: float, steer: float) -> Dict[str, float]:
        """Estimate lateral tyre force and slip for logging/visualisation."""

        steer_clipped = float(np.clip(steer, -self.cfg.max_steer, self.cfg.max_steer))
        v_eff = max(0.0, state.v)
        lateral_force = v_eff ** 2 * np.tan(steer_clipped) / max(self.cfg.wheelbase, 1e-3)
        slip = np.arctan2(v_eff * np.sin(steer_clipped), max(v_eff, 1e-3))
        return {
            "lateral_force": lateral_force,
            "slip": slip,
        }
