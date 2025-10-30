from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from src.mapping.occupancy import OccupancyGrid
from src.vis.vehicle import VehicleDrawParams, draw_vehicle

_DEFAULT_DRAW = VehicleDrawParams(
    wheelbase=2.8,
    width=1.6,
    length=4.5,
    front_overhang=0.9,
    rear_overhang=0.9,
    wheel_track=1.5,
    tire_radius=0.3,
    tire_width=0.2,
)


def map_extent(grid: OccupancyGrid) -> Tuple[float, float, float, float]:
    return (
        grid.origin[0],
        grid.origin[0] + grid.size_x * grid.resolution,
        grid.origin[1],
        grid.origin[1] + grid.size_y * grid.resolution,
    )


_OCCUPANCY_COLORS = ListedColormap(["#ffffff", "#fdebd0", "#000000"], name="occupancy")
_OCCUPANCY_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _OCCUPANCY_COLORS.N)


def draw_occupancy(ax: Axes, grid: OccupancyGrid) -> None:
    """Render free space, inflated margins, and hard obstacles with distinct colours."""

    margin = grid.inflated_margin_mask().astype(int)
    hard = grid.hard_obstacles.astype(int) * 2
    occupancy = margin + hard
    image = ax.imshow(
        occupancy,
        cmap=_OCCUPANCY_COLORS,
        norm=_OCCUPANCY_NORM,
        origin="lower",
        extent=map_extent(grid),
        interpolation="nearest",
    )
    legend_handles = [
        Patch(color="#ffffff", label="Free space"),
        Patch(color="#fdebd0", label="Inflated margin"),
        Patch(color="#000000", label="Obstacle"),
    ]
    ax._occupancy_handles = legend_handles  # type: ignore[attr-defined]
    return image


def evenly_spaced_indices(length: int, target: int = 6) -> np.ndarray:
    if length <= 0 or target <= 0:
        return np.array([], dtype=int)
    count = min(length, target)
    return np.unique(np.linspace(0, length - 1, count, dtype=int))


def draw_vehicle_samples(
    ax: Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    yaws: Sequence[float],
    steers: Sequence[float],
    indices: Iterable[int],
    params: VehicleDrawParams,
) -> None:
    for idx in indices:
        draw_vehicle(ax, xs[idx], ys[idx], yaws[idx], steers[idx], params)


def resolve_vehicle_params(params: VehicleDrawParams | None) -> VehicleDrawParams:
    if params is None:
        return replace(_DEFAULT_DRAW)
    # Ensure we return a copy so callers modifying fields do not affect defaults.
    return replace(params)


def draw_horizon(
    ax: Axes,
    horizon: Sequence[Tuple[float, float]],
    *,
    label: str | None = None,
    color: str = "tab:green",
) -> Artist | None:
    if not horizon:
        return None
    hx, hy = zip(*horizon)
    (artist,) = ax.plot(
        hx,
        hy,
        linestyle="--",
        marker="o",
        markersize=3,
        color=color,
        alpha=0.85,
        label=label,
    )
    return artist
