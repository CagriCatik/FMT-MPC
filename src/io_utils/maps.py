"""Helper utilities for exporting and importing occupancy grid bitmaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from src.mapping.occupancy import OccupancyGrid


def _metadata_path(png_path: Path) -> Path:
    """Return the path where optional legacy metadata would reside."""

    return png_path.with_suffix(".json")


def export_occupancy_png(grid: OccupancyGrid, path: Path | str) -> Path:
    """Export an occupancy grid to a PNG using the white-free/black-occupied convention."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Convert back to the conventional top-left origin expected by image
    # processing tools before writing.
    arr = ((~grid.data) * 255).astype(np.uint8)
    image = Image.fromarray(np.flipud(arr))
    image.save(output)
    return output


def load_image_map(
    path: Path | str,
    resolution: Optional[float] = None,
    origin: Optional[Tuple[float, float]] = None,
    inflation: float = 0.0,
) -> OccupancyGrid:
    """Load an :class:`OccupancyGrid` from a bitmap plus explicit metadata."""

    png_path = Path(path)
    metadata = None
    meta_path = _metadata_path(png_path)
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    image = Image.open(png_path).convert("L")
    # Flip the bitmap vertically so the numpy array matches the planner's
    # bottom-left origin convention.
    data = np.flipud(np.array(image) <= 127)

    res = resolution if resolution is not None else (metadata or {}).get("resolution")
    if res is None:
        raise ValueError(
            "Map resolution must be provided when loading bitmaps without metadata."
        )

    if origin is not None:
        org_tuple = origin
    else:
        org_tuple = tuple((metadata or {}).get("origin", (0.0, 0.0)))

    grid = OccupancyGrid.from_bitmap(data, org_tuple, float(res), inflation)
    return grid
