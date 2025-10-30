from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from src.sim.simulator import SimulationLog
from src.mapping.occupancy import OccupancyGrid
from src.planning.fmt_planner import PlannedPath
from src.vis.base import (
    draw_horizon,
    draw_occupancy,
    draw_vehicle_samples,
    evenly_spaced_indices,
    map_extent,
    resolve_vehicle_params,
)
from src.vis.vehicle import VehicleDrawParams


def plot_simulation(
    grid: OccupancyGrid,
    path: PlannedPath,
    log: SimulationLog,
    output: Optional[Path] = None,
    vehicle_params: Optional[VehicleDrawParams] = None,
) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3)

    ax_map = fig.add_subplot(gs[0, :])
    draw_occupancy(ax_map, grid)
    xmin, xmax, ymin, ymax = map_extent(grid)
    ax_map.plot(
        [p[0] for p in path.raw_waypoints],
        [p[1] for p in path.raw_waypoints],
        "--",
        label="raw",
    )
    ax_map.plot(
        [p[0] for p in path.smoothed_waypoints],
        [p[1] for p in path.smoothed_waypoints],
        label="smoothed",
    )
    ax_map.scatter(log.x, log.y, s=8, c=log.time, cmap="viridis", label="trajectory")

    indices = evenly_spaced_indices(len(log.x))
    label_drawn = False
    predicted = getattr(log, "predicted_positions", None)
    if indices.size and predicted:
        for idx in indices:
            if idx < len(predicted):
                label = "MPC horizon" if not label_drawn else None
                artist = draw_horizon(ax_map, predicted[idx], label=label)
                if artist and not label_drawn:
                    label_drawn = True

    params = resolve_vehicle_params(vehicle_params)
    if indices.size:
        draw_vehicle_samples(ax_map, log.x, log.y, log.yaw, log.steer, indices, params)

    ax_map.set_title("Map and Trajectory")
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.set_aspect("equal", adjustable="box")
    occ_handles = list(getattr(ax_map, "_occupancy_handles", []))
    occ_labels = [handle.get_label() for handle in occ_handles]
    handles, labels = ax_map.get_legend_handles_labels()
    handles = list(handles)
    labels = list(labels)
    if occ_handles:
        handles = occ_handles + handles
        labels = occ_labels + labels
    if handles:
        ax_map.legend(handles, labels, loc="upper right")

    ax_velocity = fig.add_subplot(gs[1, 0])
    ax_velocity.plot(log.time, log.v, label="speed", color="tab:blue")
    ax_velocity.set_title("Velocity")
    ax_velocity.set_ylabel("m/s")
    ax_velocity.set_xlabel("time [s]")
    ax_velocity.grid(True, linestyle="--", alpha=0.5)

    ax_accel = fig.add_subplot(gs[1, 1])
    ax_accel.plot(log.time, log.accel, label="accel", color="tab:orange")
    ax_accel.set_title("Acceleration Command")
    ax_accel.set_ylabel("m/s²")
    ax_accel.set_xlabel("time [s]")
    ax_accel.grid(True, linestyle="--", alpha=0.5)

    ax_steer = fig.add_subplot(gs[1, 2])
    ax_steer.plot(log.time, log.steer, label="steer", color="tab:green")
    ax_steer.set_title("Steering Command")
    ax_steer.set_ylabel("rad")
    ax_steer.set_xlabel("time [s]")
    ax_steer.grid(True, linestyle="--", alpha=0.5)

    ax_inputs = fig.add_subplot(gs[2, 0])
    ax_inputs.plot(log.time, log.accel, label="accel", color="tab:orange")
    ax_inputs.plot(log.time, log.steer, label="steer", color="tab:green")
    ax_inputs.set_title("Control Inputs")
    ax_inputs.set_xlabel("time [s]")
    ax_inputs.legend()
    ax_inputs.grid(True, linestyle="--", alpha=0.5)

    ax_errors = fig.add_subplot(gs[2, 1])
    ax_errors.plot(log.time, log.cross_track, label="cross track", color="tab:red")
    ax_errors.plot(log.time, log.heading_error, label="heading error", color="tab:purple")
    ax_errors.set_title("Tracking Errors")
    ax_errors.set_xlabel("time [s]")
    ax_errors.legend()
    ax_errors.grid(True, linestyle="--", alpha=0.5)

    ax_dynamics = fig.add_subplot(gs[2, 2])
    ax_dynamics.plot(log.time, log.corridor_progress, label="corridor progress", color="tab:brown")
    ax_dynamics.plot(log.time, log.lateral_force, label="lateral force", color="tab:cyan")
    ax_dynamics.plot(log.time, log.slip, label="slip", color="tab:pink")
    ax_dynamics.set_title("Dynamic Indicators")
    ax_dynamics.set_xlabel("time [s]")
    ax_dynamics.legend()
    ax_dynamics.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if output:
        fig.savefig(output)
        plt.close(fig)
    else:
        plt.show()
