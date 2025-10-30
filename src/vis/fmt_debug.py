from __future__ import annotations

from pathlib import Path
from typing import Optional

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np

from src.mapping.occupancy import OccupancyGrid
from src.planning.fmt_planner import FMTDebugData, PlannedPath
from src.vis.base import draw_occupancy, map_extent


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
    draw_occupancy(ax, grid)
    xmin, xmax, ymin, ymax = map_extent(grid)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
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

    frames: list[np.ndarray] = []
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
    draw_occupancy(ax, grid)
    xmin, xmax, ymin, ymax = map_extent(grid)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

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
        start = path.raw_waypoints[0]
        goal = path.raw_waypoints[-1]
        ax.scatter([start[0]], [start[1]], c="tab:green", s=90, marker="*", label="Start")
        ax.scatter([goal[0]], [goal[1]], c="tab:red", s=90, marker="X", label="Goal")

    if path.smoothed_waypoints:
        ax.plot(
            [p[0] for p in path.smoothed_waypoints],
            [p[1] for p in path.smoothed_waypoints],
            color="tab:red",
            lw=2.0,
            label="Smoothed path",
        )

    ax.set_title("Goal-biased FMT exploration")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)

    if show:
        plt.show()
    else:
        plt.close(fig)
