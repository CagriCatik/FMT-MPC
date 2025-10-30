from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import imageio.v2 as iio
import numpy as np

from src.mapping.occupancy import OccupancyGrid
from src.planning.fmt_planner import FMTDebugData, PlannedPath
from src.sim.simulator import SimulationLog
from src.vis.vehicle import VehicleDrawParams, draw_vehicle


def plot_simulation(
    grid: OccupancyGrid,
    path: PlannedPath,
    log: SimulationLog,
    output: Optional[Path] = None,
    vehicle_params: Optional[VehicleDrawParams] = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_map = axes[0, 0]
    occupancy = np.flipud(grid.data.astype(float))
    ax_map.imshow(occupancy, cmap="Greys", origin="lower", extent=[
        grid.origin[0],
        grid.origin[0] + grid.size_x * grid.resolution,
        grid.origin[1],
        grid.origin[1] + grid.size_y * grid.resolution,
    ])
    ax_map.plot([p[0] for p in path.raw_waypoints], [p[1] for p in path.raw_waypoints], "--", label="raw")
    ax_map.plot([p[0] for p in path.smoothed_waypoints], [p[1] for p in path.smoothed_waypoints], label="smoothed")
    ax_map.scatter(log.x, log.y, s=8, c=log.time, cmap="viridis", label="trajectory")
    sample_count = min(6, len(log.x)) if log.x else 0
    indices = np.unique(
        np.linspace(0, len(log.x) - 1, sample_count, dtype=int)
    ) if sample_count > 0 else np.array([], dtype=int)
    horizon_label_drawn = False
    if indices.size and getattr(log, "predicted_positions", None):
        for idx in indices:
            preds = log.predicted_positions[idx] if idx < len(log.predicted_positions) else []
            if preds:
                hx, hy = zip(*preds)
                label = "MPC horizon" if not horizon_label_drawn else None
                ax_map.plot(
                    hx,
                    hy,
                    linestyle="--",
                    marker="o",
                    markersize=3,
                    color="tab:green",
                    alpha=0.85,
                    label=label,
                )
                horizon_label_drawn = True
    if vehicle_params and indices.size:
        for idx in indices:
            draw_vehicle(
                ax_map,
                log.x[idx],
                log.y[idx],
                log.yaw[idx],
                log.steer[idx],
                vehicle_params,
            )
    ax_map.set_title("Map and Trajectory")
    ax_map.legend()
    ax_map.set_aspect("equal", adjustable="box")

    axes[0, 1].plot(log.time, log.v)
    axes[0, 1].set_title("Velocity")
    axes[0, 1].set_ylabel("m/s")

    axes[1, 0].plot(log.time, log.steer, label="steer")
    axes[1, 0].plot(log.time, log.accel, label="accel")
    axes[1, 0].legend()
    axes[1, 0].set_title("Inputs")

    axes[1, 1].plot(log.time, log.cross_track, label="cross track")
    axes[1, 1].plot(log.time, log.heading_error, label="heading error")
    axes[1, 1].legend()
    axes[1, 1].set_title("Errors")

    plt.tight_layout()
    if output:
        fig.savefig(output)
    else:
        plt.show()


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
    occupancy = np.flipud(grid.data.astype(float))
    ax.imshow(occupancy, cmap="Greys", origin="lower", extent=[
        grid.origin[0],
        grid.origin[0] + grid.size_x * grid.resolution,
        grid.origin[1],
        grid.origin[1] + grid.size_y * grid.resolution,
    ])
    ax.plot([p[0] for p in path.smoothed_waypoints], [p[1] for p in path.smoothed_waypoints], "--", lw=1.5, c="tab:orange")
    ax.set_title("Vehicle trajectory")
    ax.set_xlim(grid.origin[0], grid.origin[0] + grid.size_x * grid.resolution)
    ax.set_ylim(grid.origin[1], grid.origin[1] + grid.size_y * grid.resolution)
    ax.set_aspect("equal", adjustable="box")
    frames = []
    for step, (x, y, yaw, steer) in enumerate(zip(log.x, log.y, log.yaw, log.steer)):
        artists = []
        if vehicle_params:
            artists = draw_vehicle(ax, x, y, yaw, steer, vehicle_params)
        else:
            point = ax.scatter([x], [y], c="tab:blue", s=40)
            heading = ax.arrow(
                x,
                y,
                1.5 * np.cos(yaw),
                1.5 * np.sin(yaw),
                width=0.05,
                color="tab:blue",
            )
            artists.extend([point, heading])
        preds = []
        if getattr(log, "predicted_positions", None) and step < len(log.predicted_positions):
            preds = log.predicted_positions[step]
        if preds:
            hx, hy = zip(*preds)
            horizon_artist = ax.plot(
                hx,
                hy,
                linestyle="--",
                marker="o",
                markersize=3,
                color="tab:green",
                alpha=0.85,
            )[0]
            artists.append(horizon_artist)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame.copy())
        for artist in artists:
            artist.remove()
    plt.close(fig)
    if frames:
        iio.mimsave(output, frames, duration=1.0 / fps)


def animate_fmt_debug(
    grid: OccupancyGrid,
    path: PlannedPath,
    output: Path,
    fps: int = 10,
) -> None:
    debug: Optional[FMTDebugData] = path.debug
    if debug is None or not debug.edge_history:
        raise ValueError("Planner debug history unavailable; re-run planning with debug enabled.")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    occupancy = np.flipud(grid.data.astype(float))
    ax.imshow(
        occupancy,
        cmap="Greys",
        origin="lower",
        extent=[
            grid.origin[0],
            grid.origin[0] + grid.size_x * grid.resolution,
            grid.origin[1],
            grid.origin[1] + grid.size_y * grid.resolution,
        ],
    )
    ax.set_xlim(grid.origin[0], grid.origin[0] + grid.size_x * grid.resolution)
    ax.set_ylim(grid.origin[1], grid.origin[1] + grid.size_y * grid.resolution)
    ax.set_title("FMT tree growth")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")

    if debug.samples:
        xs = [pt[0] for pt in debug.samples]
        ys = [pt[1] for pt in debug.samples]
        ax.scatter(xs, ys, s=10, c="tab:blue", alpha=0.35, label="Samples")

    if path.raw_waypoints:
        start = path.raw_waypoints[0]
        ax.scatter([start[0]], [start[1]], c="tab:green", s=80, marker="*", label="Start")
        goal = path.raw_waypoints[-1]
        ax.scatter([goal[0]], [goal[1]], c="tab:red", s=80, marker="X", label="Goal")

    plt.tight_layout()

    frames = []
    for parent_idx, child_idx in debug.edge_history:
        parent = debug.samples[parent_idx]
        child = debug.samples[child_idx]
        ax.plot(
            [parent[0], child[0]],
            [parent[1], child[1]],
            color="tab:green",
            alpha=0.55,
            linewidth=1.0,
        )
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame.copy())

    if path.smoothed_waypoints:
        ax.plot(
            [p[0] for p in path.smoothed_waypoints],
            [p[1] for p in path.smoothed_waypoints],
            color="tab:orange",
            lw=2.0,
        )
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame.copy())

    plt.close(fig)
    if frames:
        iio.mimsave(output, frames, duration=1.0 / max(fps, 1))


def plot_fmt_debug(
    grid: OccupancyGrid,
    path: PlannedPath,
    output: Optional[Path] = None,
    show: bool = True,
) -> None:
    debug: Optional[FMTDebugData] = path.debug
    if debug is None:
        raise ValueError("Planner debug data unavailable; re-run planning with debug enabled.")

    fig, ax = plt.subplots(figsize=(8, 6))
    occupancy = np.flipud(grid.data.astype(float))
    ax.imshow(
        occupancy,
        cmap="Greys",
        origin="lower",
        extent=[
            grid.origin[0],
            grid.origin[0] + grid.size_x * grid.resolution,
            grid.origin[1],
            grid.origin[1] + grid.size_y * grid.resolution,
        ],
    )

    tree_plotted = False
    for parent_idx, child_idx in debug.tree_edges:
        parent = debug.samples[parent_idx]
        child = debug.samples[child_idx]
        label = "FMT tree" if not tree_plotted else None
        ax.plot(
            [parent[0], child[0]],
            [parent[1], child[1]],
            color="tab:green",
            alpha=0.4,
            linewidth=0.8,
            label=label,
        )
        tree_plotted = True

    if debug.samples:
        xs = [pt[0] for pt in debug.samples]
        ys = [pt[1] for pt in debug.samples]
        ax.scatter(xs, ys, s=12, c="tab:blue", alpha=0.7, label="Samples")

    if path.raw_waypoints:
        ax.plot(
            [p[0] for p in path.raw_waypoints],
            [p[1] for p in path.raw_waypoints],
            "--",
            color="tab:orange",
            lw=1.2,
            label="Raw path",
        )

    if path.smoothed_waypoints:
        ax.plot(
            [p[0] for p in path.smoothed_waypoints],
            [p[1] for p in path.smoothed_waypoints],
            color="tab:red",
            lw=2.0,
            label="Smoothed path",
        )

    if path.raw_waypoints:
        start = path.raw_waypoints[0]
        ax.scatter([start[0]], [start[1]], c="tab:green", s=90, marker="*", label="Start")
    if path.raw_waypoints:
        goal = path.raw_waypoints[-1]
        ax.scatter([goal[0]], [goal[1]], c="tab:red", s=90, marker="X", label="Goal")

    ax.set_title("Goal-biased FMT exploration")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(grid.origin[0], grid.origin[0] + grid.size_x * grid.resolution)
    ax.set_ylim(grid.origin[1], grid.origin[1] + grid.size_y * grid.resolution)
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)

    if show:
        plt.show()
    else:
        plt.close(fig)
