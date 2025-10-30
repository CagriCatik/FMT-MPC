"""Builders for occupancy grids sourced from specs, bitmaps, or primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import logging

from src.common.config import MapConfig
from src.io_utils.map_generator import generate_map, load_spec
from src.io_utils.maps import export_occupancy_png, load_image_map
from src.mapping.occupancy import OccupancyGrid

logger = logging.getLogger(__name__)


def build_grid(
    config: MapConfig,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: Sequence[Tuple[float, float, float]],
) -> OccupancyGrid:
    """Instantiate an :class:`OccupancyGrid` based on ``config`` sources."""

    export_path = Path(config.export_bitmap) if config.export_bitmap else None

    if config.bitmap_path:
        bitmap_path = Path(config.bitmap_path)
        if not bitmap_path.exists():
            raise FileNotFoundError(f"Bitmap map '{bitmap_path}' not found")
        grid = load_image_map(
            bitmap_path,
            resolution=config.resolution,
            origin=config.origin,
            inflation=config.obstacle_inflation,
        )
        if export_path and export_path.resolve() == bitmap_path.resolve():
            export_path = None
    elif getattr(config, "spec_path", None):
        spec_path = Path(config.spec_path)  # type: ignore[arg-type]
        if not spec_path.exists():
            raise FileNotFoundError(f"Map specification '{spec_path}' not found")
        spec = load_spec(spec_path)
        grid = generate_map(spec)
        grid.inflate(config.obstacle_inflation)
    else:
        grid = OccupancyGrid.from_obstacles(
            obstacles,
            start,
            goal,
            resolution=config.resolution,
            inflation=config.obstacle_inflation,
        )
    logger.info(
        "Built occupancy grid: origin=%s size=(%d,%d) resolution=%.3f inflation=%.2f",
        grid.origin,
        grid.size_x,
        grid.size_y,
        grid.resolution,
        getattr(grid, "_inflation_radius", 0.0),
    )
    if export_path:
        export_occupancy_png(grid, export_path)
        logger.info("Exported occupancy PNG to %s", export_path)
    return grid
