"""Occupancy grid construction, inflation, and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import ceil, floor, sqrt
from typing import List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised in environments with OpenCV available
    import cv2
except ModuleNotFoundError:  # pragma: no cover - fallback path exercised in minimal envs
    cv2 = None

from src.common.geometry import interpolate_path


@dataclass
class OccupancyGrid:
    """Binary occupancy grid augmented with an Euclidean distance field."""

    origin: Tuple[float, float]
    resolution: float
    size_x: int
    size_y: int
    data: np.ndarray
    distance_field: np.ndarray = field(init=False, repr=False)
    hard_obstacles: np.ndarray = field(init=False, repr=False)
    _inflation_radius: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        """Snapshot the original (non-inflated) obstacle mask for visualisation."""

        self.hard_obstacles = self.data.copy()

    @classmethod
    def from_obstacles(
        cls,
        obstacles: Sequence[Tuple[float, float, float]],
        start: Tuple[float, float],
        goal: Tuple[float, float],
        resolution: float,
        inflation: float,
    ) -> "OccupancyGrid":
        """Construct a grid that bounds the provided start, goal, and obstacles."""

        if obstacles:
            xs = [start[0], goal[0]] + [x for x, _, _ in obstacles]
            ys = [start[1], goal[1]] + [y for _, y, _ in obstacles]
            padding = inflation + 5.0
        else:
            xs = [start[0], goal[0]]
            ys = [start[1], goal[1]]
            padding = 10.0

        min_x = min(xs) - padding
        min_y = min(ys) - padding
        max_x = max(xs) + padding
        max_y = max(ys) + padding

        size_x = max(10, ceil((max_x - min_x) / resolution))
        size_y = max(10, ceil((max_y - min_y) / resolution))
        data = np.zeros((size_y, size_x), dtype=bool)
        grid = cls((min_x, min_y), resolution, size_x, size_y, data)
        grid._rasterize_obstacles(obstacles)
        grid.inflate(inflation)
        grid.clear_disc(start, max(inflation, grid.resolution))
        grid.clear_disc(goal, max(inflation, grid.resolution))
        return grid

    def world_to_grid(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert a world coordinate ``(x, y)`` into integer grid indices."""

        return (
            int(floor((point[0] - self.origin[0]) / self.resolution)),
            int(floor((point[1] - self.origin[1]) / self.resolution)),
        )

    def in_bounds(self, ij: Tuple[int, int]) -> bool:
        """Return ``True`` when the grid index ``(i, j)`` lies inside the bitmap."""

        i, j = ij
        return 0 <= i < self.size_x and 0 <= j < self.size_y

    def set_occupied(self, ij: Tuple[int, int]):
        """Mark the provided grid index as occupied and refresh the distance field."""

        i, j = ij
        if self.in_bounds((i, j)):
            self.hard_obstacles[j, i] = True
            self._apply_inflation()

    def is_occupied(self, point: Tuple[float, float]) -> bool:
        """Return ``True`` if ``point`` intersects the inflated occupancy grid."""

        i, j = self.world_to_grid(point)
        if not self.in_bounds((i, j)):
            return True
        return bool(self.data[j, i])

    def inflate(self, inflation: float) -> None:
        """Dilate the occupancy grid by ``inflation`` metres."""

        self._inflation_radius = max(0.0, float(inflation))
        self._apply_inflation()

    def _rasterize_obstacles(self, obstacles: Sequence[Tuple[float, float, float]]) -> None:
        """Fill circular obstacles centred at ``(x, y)`` with ``radius`` metres."""

        if not obstacles:
            self._update_distance_field()
            return
        canvas = np.zeros_like(self.data, dtype=np.uint8)
        for (x, y, radius) in obstacles:
            cx, cy = self.world_to_grid((x, y))
            r = int(np.ceil(radius / self.resolution))
            if cv2 is not None:
                cv2.circle(canvas, (cx, cy), max(0, r), 1, thickness=-1)
            else:
                _draw_disc(canvas, cx, cy, max(0, r))
        self.data = canvas.astype(bool)
        self.hard_obstacles = self.data.copy()
        self._update_distance_field()

    def _update_distance_field(self) -> None:
        """Compute the Euclidean distance-to-obstacle lookup array."""

        if cv2 is not None:
            free_space = (~self.data).astype(np.uint8)
            distance = cv2.distanceTransform(free_space, cv2.DIST_L2, 3)
            self.distance_field = distance * self.resolution
        else:
            distance = _distance_transform_without_cv(self.data)
            self.distance_field = distance * self.resolution

    def clear_disc(self, point: Tuple[float, float], radius: float) -> None:
        """Remove an inflated disc around ``point`` with radius ``radius``."""

        cx, cy = self.world_to_grid(point)
        r = max(0, int(np.ceil(radius / self.resolution)))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r and self.in_bounds((cx + dx, cy + dy)):
                    self.hard_obstacles[cy + dy, cx + dx] = False
        self._apply_inflation()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r and self.in_bounds((cx + dx, cy + dy)):
                    self.data[cy + dy, cx + dx] = False
        self._update_distance_field()

    def inflated_margin_mask(self) -> np.ndarray:
        """Return a boolean mask highlighting inflated-but-not-hard obstacle cells."""

        return self.data & ~self.hard_obstacles

    def validate_corridor(self, path: Sequence[Tuple[float, float]], safety_margin: float) -> bool:
        """Return ``True`` when ``path`` maintains ``safety_margin`` clearance."""

        if not path:
            return False
        samples = interpolate_path(path, max(self.resolution / 2, safety_margin / 2))
        for pt in samples:
            if self.distance_to_nearest(pt) < safety_margin:
                return False
        return True

    def distance_to_nearest(self, point: Tuple[float, float]) -> float:
        """Return the distance from ``point`` to the closest occupied cell."""

        i, j = self.world_to_grid(point)
        if not self.in_bounds((i, j)):
            return 0.0
        return float(self.distance_field[j, i])

    def to_metadata(self) -> dict:
        """Serialise minimal metadata describing the grid layout."""

        return {
            "origin": list(self.origin),
            "resolution": float(self.resolution),
            "size_x": int(self.size_x),
            "size_y": int(self.size_y),
        }

    @classmethod
    def from_bitmap(
        cls,
        bitmap: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
        inflation: float,
    ) -> "OccupancyGrid":
        """Construct a grid from a bitmap where ``True`` marks occupied cells."""

        size_y, size_x = bitmap.shape
        grid = cls(origin, resolution, size_x, size_y, bitmap.astype(bool))
        grid.inflate(inflation)
        return grid

    def _apply_inflation(self) -> None:
        """Recompute the inflated occupancy mask from the stored hard obstacles."""

        if self._inflation_radius <= 0.0:
            self.data = self.hard_obstacles.copy()
            self._update_distance_field()
            return

        kernel_radius = max(0, int(np.ceil(self._inflation_radius / self.resolution)))
        if kernel_radius == 0:
            self.data = self.hard_obstacles.copy()
            self._update_distance_field()
            return

        if cv2 is not None:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * kernel_radius + 1, 2 * kernel_radius + 1)
            )
            inflated = cv2.dilate(self.hard_obstacles.astype(np.uint8), kernel, iterations=1)
            self.data = inflated.astype(bool)
        else:
            self.data = _dilate_without_cv(self.hard_obstacles, kernel_radius)
        self._update_distance_field()


def _draw_disc(canvas: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Fill a disc in ``canvas`` using a simple raster algorithm."""

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


def _dilate_without_cv(data: np.ndarray, radius: int) -> np.ndarray:
    """Dilate ``data`` by ``radius`` pixels when OpenCV is unavailable."""

    if radius <= 0:
        return data.astype(bool)
    height, width = data.shape
    inflated = data.astype(bool).copy()
    offsets: List[Tuple[int, int]] = []
    radius_sq = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius_sq:
                offsets.append((dy, dx))
    for y, x in np.argwhere(data):
        for dy, dx in offsets:
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width:
                inflated[ny, nx] = True
    return inflated


def _distance_transform_without_cv(occupancy: np.ndarray) -> np.ndarray:
    """Compute a Euclidean distance transform without OpenCV dependencies."""

    height, width = occupancy.shape
    if not occupancy.any():
        max_range = sqrt(height * height + width * width)
        return np.full((height, width), max_range, dtype=float)

    distances = np.full((height, width), np.inf, dtype=float)
    heap: List[Tuple[float, int, int]] = []
    for y, x in np.argwhere(occupancy):
        distances[y, x] = 0.0
        heappush(heap, (0.0, y, x))

    sqrt2 = sqrt(2.0)
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, sqrt2),
        (-1, 1, sqrt2),
        (1, -1, sqrt2),
        (1, 1, sqrt2),
    ]

    while heap:
        dist, y, x = heappop(heap)
        if dist > distances[y, x]:
            continue
        for dy, dx, cost in neighbors:
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width:
                nd = dist + cost
                if nd < distances[ny, nx]:
                    distances[ny, nx] = nd
                    heappush(heap, (nd, ny, nx))

    max_range = sqrt(height * height + width * width)
    distances[np.isinf(distances)] = max_range
    return distances
