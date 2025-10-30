"""Goal-biased Fast Marching Tree planner with deterministic smoothing hooks."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from random import Random
from typing import Dict, List, Optional, Tuple

import logging

import numpy as np

from src.common.geometry import annotate_waypoints, cubic_spline
from src.common.config import MapConfig, PlannerConfig
from src.mapping.occupancy import OccupancyGrid

logger = logging.getLogger(__name__)


@dataclass
class PlannedPath:
    """Bundle containing the raw, smoothed, and annotated outputs from FMT."""

    raw_waypoints: List[Tuple[float, float]]
    smoothed_waypoints: List[Tuple[float, float]]
    annotated_waypoints: List[Tuple[float, float, float, float]]
    cost: float
    debug: Optional["FMTDebugData"] = None


@dataclass
class FMTDebugData:
    """Structured artefacts describing the growth of the FMT exploration tree."""

    samples: List[Tuple[float, float]]
    tree_edges: List[Tuple[int, int]]
    edge_history: List[Tuple[int, int]]


class GoalBiasedFMT:
    """Fast Marching Tree variant with deterministic sampling and safety checks."""

    def __init__(
        self,
        grid: OccupancyGrid,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        planner_config: PlannerConfig,
        safety_margin: float,
        seed: int = 1,
    ) -> None:
        """Create the planner and pre-validate the start/goal points."""

        self.grid = grid
        self.start = start
        self.goal = goal
        self.config = planner_config
        self.safety_margin = safety_margin
        self.random = Random(seed)
        self.samples: List[Tuple[float, float]] = []
        self.neighbors: Dict[int, List[int]] = {}
        self.tree_edges: List[Tuple[int, int]] = []
        self.edge_history: List[Tuple[int, int]] = []

        self._ensure_endpoint_is_safe(self.start, "start")
        self._ensure_endpoint_is_safe(self.goal, "goal")

    def plan(self) -> PlannedPath:
        """Execute the FMT algorithm and return a smoothed, annotated path."""

        logger.info(
            "Starting FMT planning with %d samples, radius %.2f, goal bias %.2f",
            self.config.sample_count,
            self.config.connection_radius,
            self.config.goal_bias,
        )
        self._sample_points()
        self._build_neighbor_sets()
        tree, cost = self._search()
        self.tree_edges = [(tree[idx], idx) for idx in tree if idx != tree[idx]]
        path = self._extract_path(tree)
        if self.config.pruning:
            path = self._prune(path)
        smoothed = path
        if self.config.smoothing and len(path) >= 3:
            candidate = cubic_spline(path)
            if self.grid.validate_corridor(candidate, self.safety_margin):
                smoothed = candidate
            else:
                # fall back to the raw, collision-free corridor when smoothing cuts corners
                smoothed = path
        annotated = annotate_waypoints(smoothed, self.config.nominal_speed)
        debug = FMTDebugData(
            samples=list(self.samples),
            tree_edges=list(self.tree_edges),
            edge_history=list(self.edge_history),
        )
        logger.info(
            "FMT plan complete: %d raw waypoints, cost %.2f", len(path), cost
        )
        return PlannedPath(path, smoothed, annotated, cost, debug)

    def _sample_points(self) -> None:
        """Draw random samples with optional goal bias while rejecting collisions."""

        width = self.grid.size_x * self.grid.resolution
        height = self.grid.size_y * self.grid.resolution
        ox, oy = self.grid.origin
        samples = [self.start, self.goal]
        while len(samples) < self.config.sample_count:
            if self.random.random() < self.config.goal_bias:
                pt = self.goal
            else:
                pt = (
                    ox + self.random.random() * width,
                    oy + self.random.random() * height,
                )
            if self._point_is_safe(pt):
                samples.append(pt)
        self.samples = samples
        logger.debug("Sampled %d collision-free states", len(self.samples))

    def _build_neighbor_sets(self) -> None:
        """Construct adjacency lists for all samples within the connection radius."""

        radius = self.config.connection_radius
        self.neighbors = {i: [] for i in range(len(self.samples))}
        for i, a in enumerate(self.samples):
            for j, b in enumerate(self.samples):
                if i == j:
                    continue
                if hypot(a[0] - b[0], a[1] - b[1]) <= radius:
                    # Only accept neighbours when both occupancy and safety checks pass.
                    if not self._edge_blocked(a, b) and self._edge_safe(a, b):
                        self.neighbors[i].append(j)
        logger.debug("Constructed neighbour sets for %d samples", len(self.samples))

    def _point_is_safe(self, point: Tuple[float, float]) -> bool:
        """Return ``True`` when the given sample lies strictly outside the safety margin."""

        if self.grid.is_occupied(point):
            return False
        return self.grid.distance_to_nearest(point) > self.safety_margin

    def _ensure_endpoint_is_safe(self, point: Tuple[float, float], label: str) -> None:
        """Raise if the named endpoint is inside an obstacle or safety corridor."""

        if not self._point_is_safe(point):
            raise ValueError(f"{label} lies inside an inflated obstacle or outside the valid corridor")

    def _edge_blocked(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        """Return ``True`` if any point along the edge intersects an occupied cell."""

        max_step = max(self.grid.resolution / 4.0, 1e-2)
        steps = max(2, int(hypot(a[0] - b[0], a[1] - b[1]) / max_step))
        for t in np.linspace(0.0, 1.0, steps):
            point = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if self.grid.is_occupied(point):
                return True
        return False

    def _edge_safe(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        """Return ``True`` when the full edge maintains the configured safety margin."""

        max_step = max(self.grid.resolution / 4.0, 1e-2)
        steps = max(2, int(hypot(a[0] - b[0], a[1] - b[1]) / max_step))
        for t in np.linspace(0.0, 1.0, steps):
            point = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if self.grid.distance_to_nearest(point) <= self.safety_margin:
                return False
        return True

    def _search(self) -> Tuple[Dict[int, int], float]:
        """Run the FMT dynamic programming loop returning parent links and cost."""

        start_idx = 0
        goal_idx = 1
        cost = {start_idx: 0.0}
        parent = {start_idx: start_idx}
        frontier = {start_idx}
        unexplored = set(range(2, len(self.samples))) | {goal_idx}

        while frontier:
            new_frontier = set()
            for v in list(unexplored):
                best_parent = None
                best_cost = float("inf")
                for y in frontier:
                    if v not in self.neighbors[y]:
                        continue
                    tentative = cost[y] + self._dist(y, v)
                    if tentative < best_cost:
                        best_cost = tentative
                        best_parent = y
                if best_parent is not None:
                    cost[v] = best_cost
                    parent[v] = best_parent
                    new_frontier.add(v)
                    unexplored.remove(v)
                    self.edge_history.append((best_parent, v))
            logger.debug(
                "Frontier size %d after expansion; goal in frontier=%s",
                len(new_frontier),
                goal_idx in new_frontier,
            )
            if goal_idx in new_frontier:
                break
            frontier = new_frontier
            if self.config.rewiring:
                for v in frontier:
                    for n in self.neighbors[v]:
                        if n in parent and cost[v] + self._dist(v, n) < cost[n]:
                            parent[n] = v
                            cost[n] = cost[v] + self._dist(v, n)
                            self.edge_history.append((v, n))
        if goal_idx not in parent:
            logger.error("FMT search terminated without connecting to goal")
        else:
            logger.info("FMT search connected goal with cost %.2f", cost.get(goal_idx, float("inf")))
        return parent, cost.get(goal_idx, float("inf"))

    def _dist(self, a_idx: int, b_idx: int) -> float:
        """Compute Euclidean distance between two sample indices."""

        a, b = self.samples[a_idx], self.samples[b_idx]
        return hypot(a[0] - b[0], a[1] - b[1])

    def _extract_path(self, parent: Dict[int, int]) -> List[Tuple[float, float]]:
        """Backtrack the parent dictionary to recover the goal-reaching path."""

        goal_idx = 1
        if goal_idx not in parent:
            raise RuntimeError("goal unreachable")
        idx = goal_idx
        path = [self.samples[idx]]
        while idx != parent[idx]:
            idx = parent[idx]
            path.append(self.samples[idx])
        path.reverse()
        return path

    def _prune(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Greedily remove waypoints that are unnecessary for a safe corridor."""

        if len(path) <= 2:
            return path

        pruned = [path[0]]
        i = 0
        last_index = len(path) - 1

        while i < last_index:
            j = last_index
            while j > i + 1:
                if self._edge_blocked(path[i], path[j]) or not self._edge_safe(path[i], path[j]):
                    j -= 1
                    continue
                break
            pruned.append(path[j])
            i = j

        if pruned[-1] != path[-1]:
            pruned.append(path[-1])

        return pruned
def plan_path(
    grid: OccupancyGrid,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    planner_config: PlannerConfig,
    map_config: MapConfig,
) -> PlannedPath:
    """Convenience wrapper around :class:`GoalBiasedFMT` with corridor checks."""

    planner = GoalBiasedFMT(grid, start, goal, planner_config, map_config.safety_margin)
    path = planner.plan()
    if map_config.validate_corridor:
        if not grid.validate_corridor(path.raw_waypoints, map_config.safety_margin):
            raise RuntimeError("Planned corridor violates safety margin")
        if path.smoothed_waypoints and not grid.validate_corridor(
            path.smoothed_waypoints, map_config.safety_margin
        ):
            # Fall back to the raw waypoints when downstream consumers expect a safe reference.
            path.smoothed_waypoints = list(path.raw_waypoints)
            path.annotated_waypoints = annotate_waypoints(path.smoothed_waypoints, planner_config.nominal_speed)
    return path
