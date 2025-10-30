"""Deterministic YAML-driven occupancy map generation utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised when OpenCV is installed
    import cv2
except ModuleNotFoundError:  # pragma: no cover - fallback path for minimal environments
    cv2 = None

from src.common.config import safe_load_yaml
from src.io_utils.maps import export_occupancy_png
from src.mapping.occupancy import OccupancyGrid


@dataclass
class RectangleSpec:
    """Axis-aligned rectangle described by centre coordinates and size."""

    center: Tuple[float, float]
    size: Tuple[float, float]


@dataclass
class CircleSpec:
    """Circle obstacle represented by centre coordinates and radius."""

    center: Tuple[float, float]
    radius: float


@dataclass
class PolygonSpec:
    """Polygon obstacle using world-space vertex coordinates."""

    vertices: Sequence[Tuple[float, float]]


@dataclass
class MapSpec:
    """Fully resolved map specification used to rasterise occupancy grids."""

    resolution: float
    size: Tuple[float, float]
    origin: Tuple[float, float]
    border: bool
    rectangles: Sequence[RectangleSpec]
    circles: Sequence[CircleSpec]
    polygons: Sequence[PolygonSpec]


def _world_to_index(point: Tuple[float, float], origin: Tuple[float, float], resolution: float) -> Tuple[int, int]:
    """Convert a world coordinate to integer pixel indices."""

    x, y = point
    ox, oy = origin
    i = int(round((x - ox) / resolution))
    j = int(round((y - oy) / resolution))
    return i, j


def _draw_rectangles(canvas: np.ndarray, spec: MapSpec) -> None:
    """Rasterise rectangular obstacles onto ``canvas``."""

    for rectangle in spec.rectangles:
        cx, cy = rectangle.center
        width, height = rectangle.size
        half_w = width / 2.0
        half_h = height / 2.0
        top_left = (cx - half_w, cy - half_h)
        bottom_right = (cx + half_w, cy + half_h)
        x0, y0 = _world_to_index(top_left, spec.origin, spec.resolution)
        x1, y1 = _world_to_index(bottom_right, spec.origin, spec.resolution)
        x0 = int(np.clip(x0, 0, canvas.shape[1] - 1))
        x1 = int(np.clip(x1, 0, canvas.shape[1] - 1))
        y0 = int(np.clip(y0, 0, canvas.shape[0] - 1))
        y1 = int(np.clip(y1, 0, canvas.shape[0] - 1))
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if cv2 is not None:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color=1, thickness=-1)
        else:
            canvas[y0 : y1 + 1, x0 : x1 + 1] = 1


def _draw_circles(canvas: np.ndarray, spec: MapSpec) -> None:
    """Rasterise circular obstacles onto ``canvas``."""

    for circle in spec.circles:
        cx, cy = _world_to_index(circle.center, spec.origin, spec.resolution)
        radius = int(np.ceil(circle.radius / spec.resolution))
        cx = int(np.clip(cx, 0, canvas.shape[1] - 1))
        cy = int(np.clip(cy, 0, canvas.shape[0] - 1))
        if cv2 is not None:
            cv2.circle(canvas, (cx, cy), radius, color=1, thickness=-1)
        else:
            _fill_disc(canvas, cx, cy, radius)


def _draw_polygons(canvas: np.ndarray, spec: MapSpec) -> None:
    """Rasterise polygonal obstacles onto ``canvas``."""

    for polygon in spec.polygons:
        pts: List[Tuple[int, int]] = []
        for vertex in polygon.vertices:
            i, j = _world_to_index(vertex, spec.origin, spec.resolution)
            ii = int(np.clip(i, 0, canvas.shape[1] - 1))
            jj = int(np.clip(j, 0, canvas.shape[0] - 1))
            pts.append((ii, jj))
        if pts:
            contour = np.array([pts], dtype=np.int32)
            if cv2 is not None:
                cv2.fillPoly(canvas, contour, color=1)
            else:
                _fill_polygon(canvas, pts)


def generate_map(spec: MapSpec) -> OccupancyGrid:
    """Generate an occupancy grid from the supplied :class:`MapSpec`."""

    width_px = int(np.ceil(spec.size[0] / spec.resolution))
    height_px = int(np.ceil(spec.size[1] / spec.resolution))
    canvas = np.zeros((height_px, width_px), dtype=np.uint8)

    if spec.border:
        canvas[0, :] = 1
        canvas[-1, :] = 1
        canvas[:, 0] = 1
        canvas[:, -1] = 1

    _draw_rectangles(canvas, spec)
    _draw_circles(canvas, spec)
    _draw_polygons(canvas, spec)

    grid = OccupancyGrid.from_bitmap(
        np.flipud(canvas.astype(bool)), spec.origin, spec.resolution, inflation=0.0
    )
    return grid


def load_spec(path: Path) -> MapSpec:
    """Load a :class:`MapSpec` definition from ``path``."""

    data = safe_load_yaml(path.read_text())
    if data is None:
        raise ValueError(f"Map specification '{path}' is empty")

    resolution = float(data.get("resolution", 0.5))
    size = tuple(data.get("size", (40.0, 40.0)))  # type: ignore[arg-type]
    if len(size) != 2:
        raise ValueError("Map size must be a length-2 sequence")
    origin = tuple(data.get("origin", (0.0, 0.0)))  # type: ignore[arg-type]
    border = bool(data.get("border", True))

    rectangles = [
        RectangleSpec(center=tuple(item.get("center", (0.0, 0.0))), size=tuple(item.get("size", (1.0, 1.0))))
        for item in data.get("rectangles", [])
    ]
    circles = [
        CircleSpec(center=tuple(item.get("center", (0.0, 0.0))), radius=float(item.get("radius", 1.0)))
        for item in data.get("circles", [])
    ]
    polygons = [
        PolygonSpec(vertices=[tuple(vertex) for vertex in item.get("vertices", [])])
        for item in data.get("polygons", [])
    ]

    return MapSpec(
        resolution=resolution,
        size=(float(size[0]), float(size[1])),
        origin=(float(origin[0]), float(origin[1])),
        border=border,
        rectangles=rectangles,
        circles=circles,
        polygons=polygons,
    )


def main() -> None:
    """Entry point for the deterministic occupancy map CLI."""

    parser = argparse.ArgumentParser(
        description="Generate an occupancy map PNG from a YAML specification"
    )
    parser.add_argument("spec", type=Path, help="YAML file describing the map geometry")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Destination PNG path (defaults to the spec path with .png)",
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)
    grid = generate_map(spec)

    output = args.output or args.spec.with_suffix(".png")
    if not output.is_absolute():
        output = Path.cwd() / output
    export_occupancy_png(grid, output)


if __name__ == "__main__":
    main()


def _fill_disc(canvas: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Fallback rasteriser for filled circles when OpenCV is unavailable."""

    height, width = canvas.shape
    if radius <= 0:
        if 0 <= cy < height and 0 <= cx < width:
            canvas[cy, cx] = 1
        return
    radius_sq = radius * radius
    for dy in range(-radius, radius + 1):
        ny = cy + dy
        if ny < 0 or ny >= height:
            continue
        dx_limit = int(np.floor(np.sqrt(max(radius_sq - dy * dy, 0))))
        x_min = max(0, cx - dx_limit)
        x_max = min(width - 1, cx + dx_limit)
        canvas[ny, x_min : x_max + 1] = 1


def _fill_polygon(canvas: np.ndarray, pts: Sequence[Tuple[int, int]]) -> None:
    """Fallback polygon rasteriser that avoids heavy dependencies."""

    if len(pts) < 3:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min = max(0, min(xs))
    x_max = min(canvas.shape[1] - 1, max(xs))
    y_min = max(0, min(ys))
    y_max = min(canvas.shape[0] - 1, max(ys))
    if x_min > x_max or y_min > y_max:
        return
    try:
        from matplotlib.path import Path as MplPath  # type: ignore
    except Exception:  # pragma: no cover - executed when matplotlib unavailable
        # Fallback: simple scanline fill using even-odd rule
        polygon = np.array(pts, dtype=float)
        edges = list(zip(polygon, np.roll(polygon, -1, axis=0)))
        for y in range(y_min, y_max + 1):
            intersections: List[float] = []
            for (x0, y0), (x1, y1) in edges:
                if (y0 <= y < y1) or (y1 <= y < y0):
                    if y1 == y0:
                        continue
                    t = (y - y0) / (y1 - y0)
                    intersections.append(x0 + t * (x1 - x0))
            if len(intersections) < 2:
                continue
            intersections.sort()
            for x_start, x_end in zip(intersections[0::2], intersections[1::2]):
                xs_fill_start = int(np.ceil(max(x_start, x_min)))
                xs_fill_end = int(np.floor(min(x_end, x_max)))
                if xs_fill_start <= xs_fill_end:
                    canvas[y, xs_fill_start : xs_fill_end + 1] = 1
        return

    path = MplPath(pts)
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + 1),
        np.arange(y_min, y_max + 1),
        indexing="xy",
    )
    coords = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
    mask = path.contains_points(coords, radius=-0.5)
    mask = mask.reshape((y_max - y_min + 1, x_max - x_min + 1))
    canvas[y_min : y_max + 1, x_min : x_max + 1][mask] = 1
