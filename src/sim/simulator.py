"""Deterministic time-stepped simulator orchestrating MPC-controlled roll-outs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.common.geometry import Pose
from src.tracking.mpc_tracker import ControlCommand, Corridor, MPCTracker, TrackerState
from src.vehicle.bicycle import BicycleModel, BicycleState

logger = logging.getLogger(__name__)


@dataclass
class SimulationLog:
    """Structured time-series data captured during a simulator run."""

    time: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    steer: List[float] = field(default_factory=list)
    accel: List[float] = field(default_factory=list)
    cross_track: List[float] = field(default_factory=list)
    heading_error: List[float] = field(default_factory=list)
    corridor_progress: List[float] = field(default_factory=list)
    slip: List[float] = field(default_factory=list)
    lateral_force: List[float] = field(default_factory=list)
    predicted_positions: List[List[Tuple[float, float]]] = field(default_factory=list)

    def append(
        self,
        t: float,
        state: BicycleState,
        command: ControlCommand,
        forces: Dict[str, float],
        progress: float,
    ) -> None:
        """Store the provided snapshot in the log."""
        self.time.append(t)
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.steer.append(command.steer)
        self.accel.append(command.accel)
        self.cross_track.append(command.feedforward.get("cross_track", 0.0))
        self.heading_error.append(command.feedforward.get("heading_error", 0.0))
        self.corridor_progress.append(progress)
        self.slip.append(forces.get("slip", 0.0))
        self.lateral_force.append(forces.get("lateral_force", 0.0))
        self.predicted_positions.append(list(command.predicted_positions))


class Simulator:
    """Time-stepped vehicle simulator driven by MPC control commands."""

    def __init__(
        self,
        tracker: MPCTracker,
        vehicle: BicycleModel,
        dt: float,
        duration: float,
        log_states: bool = True,
    ) -> None:
        self.tracker = tracker
        self.vehicle = vehicle
        if dt <= 0.0:
            raise ValueError("Simulation timestep must be positive")
        if duration <= 0.0:
            raise ValueError("Simulation duration must be positive")
        self.dt = dt
        self.duration = duration
        self.log_states = log_states

    def run(self, start: Pose, corridor: Corridor) -> SimulationLog:
        """Run the simulation for the configured duration and return a log."""

        state = BicycleState(start.x, start.y, start.yaw, v=0.0)
        log = SimulationLog()
        steps = max(1, int(np.ceil(self.duration / max(self.dt, 1e-9))))
        total_length = corridor.total_length()
        logger.info(
            "Simulator started: dt=%.3f, duration=%.2f, steps=%d, target_length=%.2f",
            self.dt,
            self.duration,
            steps,
            total_length,
        )
        for step in range(steps):
            time = min(step * self.dt, self.duration)
            tracker_state = TrackerState(state.x, state.y, state.yaw, state.v)
            command = self.tracker.compute_control(tracker_state, corridor, self.dt)
            forces = self.vehicle.compute_forces(state, command.accel, command.steer)
            state = self.vehicle.step(state, command.accel, command.steer, self.dt)
            progress = corridor.project((state.x, state.y))
            log.append(time, state, command, forces, progress)
            if self.log_states:
                logger.info(
                    (
                        "step=%04d t=%.2f x=%.2f y=%.2f yaw=%.2f v=%.2f steer=%.2f "
                        "accel=%.2f cross=%.3f heading=%.3f progress=%.2f slip=%.3f"
                    ),
                    step,
                    time,
                    state.x,
                    state.y,
                    state.yaw,
                    state.v,
                    command.steer,
                    command.accel,
                    command.feedforward.get("cross_track", 0.0),
                    command.feedforward.get("heading_error", 0.0),
                    progress,
                    forces.get("slip", 0.0),
                )
            if progress >= total_length - 1e-2 and abs(state.v) <= 0.05:
                logger.info(
                    "Stopping simulation early at step %d (progress %.2f, speed %.2f)",
                    step,
                    progress,
                    state.v,
                )
                break
        logger.info("Simulator finished after %d recorded steps", len(log.time))
        return log
