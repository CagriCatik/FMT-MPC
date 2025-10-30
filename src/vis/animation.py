from __future__ import annotations

from pathlib import Path
from typing import Optional

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np

from src.mapping.occupancy import OccupancyGrid
from src.planning.fmt_planner import PlannedPath
from src.sim.simulator import SimulationLog
from src.vis.base import (
    draw_horizon,
    draw_occupancy,
    map_extent,
    resolve_vehicle_params,
)
from src.vis.vehicle import VehicleDrawParams, draw_vehicle


def animate_simulation(
    grid: OccupancyGrid,
    path: PlannedPath,
    log: SimulationLog,
    output: Path,
    fps: int = 12,
    vehicle_params: Optional[VehicleDrawParams] = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_occupancy(ax, grid)
    ax.plot(
        [p[0] for p in path.smoothed_waypoints],
        [p[1] for p in path.smoothed_waypoints],
        "--",
        lw=1.5,
        c="tab:orange",
    )
    xmin, xmax, ymin, ymax = map_extent(grid)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Vehicle trajectory")
    ax.set_aspect("equal", adjustable="box")

    params = resolve_vehicle_params(vehicle_params)
    predicted = getattr(log, "predicted_positions", None) or []

    frames: list[np.ndarray] = []
    for step, (x, y, yaw, steer) in enumerate(zip(log.x, log.y, log.yaw, log.steer)):
        vehicle_artists = draw_vehicle(ax, x, y, yaw, steer, params)
        horizon_artist = None
        if step < len(predicted):
            horizon_artist = draw_horizon(ax, predicted[step])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame.copy())
        for artist in vehicle_artists:
            artist.remove()
        if horizon_artist is not None:
            horizon_artist.remove()

    plt.close(fig)
    if frames:
        iio.mimsave(output, frames, duration=1.0 / max(fps, 1))
