from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.config import (
    ExperimentConfig,
    MapConfig,
    PlannerConfig,
    TrackerConfig,
    VehicleConfig,
    SimulationConfig,
    VisualizationConfig,
)
from src.mapping.builder import build_grid
from src.planning.fmt_planner import GoalBiasedFMT, plan_path
from src.tracking.mpc_tracker import Corridor, MPCTracker, TrackerState
from src.vehicle.bicycle import BicycleModel
from src.sim.simulator import Simulator
from src.common.geometry import Pose
from src.io_utils.maps import load_image_map, export_occupancy_png
from src.io_utils.map_generator import load_spec, generate_map
from src.mapping.occupancy import OccupancyGrid
from PIL import Image, ImageDraw

def build_default_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        map_config=MapConfig(resolution=0.5, obstacle_inflation=1.0, safety_margin=0.2, validate_corridor=True),
        planner_config=PlannerConfig(sample_count=80, connection_radius=5.0, goal_bias=0.3, rewiring=True, pruning=True, smoothing=True, nominal_speed=4.0),
        tracker_config=TrackerConfig(
            horizon=8,
            timestep=0.2,
            q_weights=(1.0, 1.0, 0.5, 0.2),
            r_weights=(0.1, 0.05),
            cruise_speed=6.0,
            slowdown_radius=5.0,
        ),
        vehicle_config=VehicleConfig(wheelbase=2.7, max_steer=0.5, max_accel=3.0, max_decel=-4.0),
        simulation_config=SimulationConfig(dt=0.1, duration=2.0, log_states=True),
        visualization_config=VisualizationConfig(enable_plots=True, animate=True),
        start=(0.0, 0.0, 0.0),
        goal=(8.0, 6.0, 0.0),
        obstacles=[(3.0, 1.5, 1.0)],
    )


def test_plan_path_generates_corridor():
    exp = build_default_experiment()
    grid = build_grid(exp.map_config, exp.start[:2], exp.goal[:2], exp.obstacles)
    path = plan_path(grid, exp.start[:2], exp.goal[:2], exp.planner_config, exp.map_config)
    assert len(path.raw_waypoints) >= 2
    assert len(path.smoothed_waypoints) >= len(path.raw_waypoints)
    assert all(len(wp) == 4 for wp in path.annotated_waypoints)
    assert path.debug is not None
    assert len(path.debug.samples) >= len(path.raw_waypoints)
    assert path.debug.edge_history  # ensure we capture growth history for animations


def test_tracker_produces_control():
    exp = build_default_experiment()
    grid = build_grid(exp.map_config, exp.start[:2], exp.goal[:2], exp.obstacles)
    path = plan_path(grid, exp.start[:2], exp.goal[:2], exp.planner_config, exp.map_config)
    corridor = Corridor(path.annotated_waypoints)
    tracker = MPCTracker(exp.tracker_config, exp.vehicle_config)
    state = TrackerState(x=0.0, y=-0.5, yaw=0.0, v=2.0)
    command = tracker.compute_control(state, corridor, exp.tracker_config.timestep)
    assert -exp.vehicle_config.max_steer <= command.steer <= exp.vehicle_config.max_steer
    assert exp.vehicle_config.max_decel <= command.accel <= exp.vehicle_config.max_accel
    assert "heading_error" in command.feedforward
    assert command.predicted_positions
    assert all(len(pt) == 2 for pt in command.predicted_positions)


def test_simulator_runs_and_logs():
    exp = build_default_experiment()
    grid = build_grid(exp.map_config, exp.start[:2], exp.goal[:2], exp.obstacles)
    path = plan_path(grid, exp.start[:2], exp.goal[:2], exp.planner_config, exp.map_config)
    corridor = Corridor(path.annotated_waypoints)
    tracker = MPCTracker(exp.tracker_config, exp.vehicle_config)
    vehicle = BicycleModel(exp.vehicle_config)
    simulator = Simulator(
        tracker,
        vehicle,
        exp.simulation_config.dt,
        exp.simulation_config.duration,
        log_states=False,
    )
    log = simulator.run(Pose(*exp.start), corridor)
    assert len(log.time) > 0
    assert len(log.v) == len(log.time)
    assert len(log.slip) == len(log.time)
    assert len(log.predicted_positions) == len(log.time)


def test_simulator_exits_when_goal_reached():
    exp = build_default_experiment()
    exp.simulation_config = SimulationConfig(dt=0.1, duration=10.0, log_states=False)
    grid = build_grid(exp.map_config, exp.start[:2], exp.goal[:2], exp.obstacles)
    path = plan_path(grid, exp.start[:2], exp.goal[:2], exp.planner_config, exp.map_config)
    corridor = Corridor(path.annotated_waypoints)
    tracker = MPCTracker(exp.tracker_config, exp.vehicle_config)
    vehicle = BicycleModel(exp.vehicle_config)
    simulator = Simulator(
        tracker,
        vehicle,
        exp.simulation_config.dt,
        exp.simulation_config.duration,
        log_states=False,
    )
    log = simulator.run(Pose(*exp.start), corridor)
    assert log.time[-1] < exp.simulation_config.duration


def test_map_export_roundtrip(tmp_path: Path):
    exp = build_default_experiment()
    export_path = tmp_path / "map.png"
    exp.map_config.export_bitmap = str(export_path)
    grid = build_grid(exp.map_config, exp.start[:2], exp.goal[:2], exp.obstacles)
    assert export_path.exists()
    loaded = load_image_map(
        export_path,
        resolution=exp.map_config.resolution,
        origin=exp.map_config.origin,
        inflation=exp.map_config.obstacle_inflation,
    )
    assert loaded.size_x == grid.size_x
    assert loaded.size_y == grid.size_y
    # Loading directly from the bitmap path should yield the same grid dimensions
    exp.map_config.bitmap_path = str(export_path)
    exp.map_config.export_bitmap = None
    from_bitmap = build_grid(exp.map_config, exp.start[:2], exp.goal[:2], [])
    assert from_bitmap.size_x == grid.size_x
    assert from_bitmap.size_y == grid.size_y


def test_yaml_map_generator(tmp_path: Path):
    spec = load_spec(Path("maps/conservative_map.yaml"))
    grid = generate_map(spec)
    out = tmp_path / "generated.png"
    export_occupancy_png(grid, out)
    assert out.exists()


def test_plan_path_rejects_unsafe_goal():
    bitmap = np.zeros((40, 40), dtype=bool)
    bitmap[15:25, 15:25] = True
    grid = OccupancyGrid.from_bitmap(bitmap, origin=(0.0, 0.0), resolution=1.0, inflation=1.0)

    map_config = MapConfig(
        resolution=1.0,
        obstacle_inflation=1.0,
        safety_margin=0.5,
        validate_corridor=True,
    )
    planner_config = PlannerConfig(
        sample_count=80,
        connection_radius=6.0,
        goal_bias=0.3,
        rewiring=False,
        pruning=False,
        smoothing=False,
        nominal_speed=2.0,
    )

    start = (2.0, 2.0)
    goal = (18.0, 18.0)

    with pytest.raises(ValueError, match="goal.*inflated obstacle"):
        plan_path(grid, start, goal, planner_config, map_config)


def test_loading_bitmap_without_resolution_errors(tmp_path: Path):
    canvas = Image.new("1", (10, 10), color=1)
    path = tmp_path / "map.png"
    canvas.save(path)

    with pytest.raises(ValueError):
        load_image_map(path)


def test_fmt_sampling_respects_bitmap_obstacles(tmp_path: Path):
    width, height = 400, 200
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([80, 60, 320, 140], fill=0)
    bitmap_path = tmp_path / "obstacle.png"
    image.save(bitmap_path)

    grid = load_image_map(bitmap_path, resolution=0.2, origin=(0.0, 0.0), inflation=1.0)
    planner_cfg = PlannerConfig(
        sample_count=200,
        connection_radius=6.0,
        goal_bias=0.2,
        rewiring=False,
        pruning=False,
        smoothing=False,
        nominal_speed=4.0,
    )
    planner = GoalBiasedFMT(grid, (5.0, 5.0), (70.0, 30.0), planner_cfg, safety_margin=0.8)
    planner._sample_points()

    assert all(not grid.is_occupied(pt) for pt in planner.samples)
    assert all(grid.distance_to_nearest(pt) > 0.8 for pt in planner.samples)


def test_inflated_margin_mask_distinguishes_obstacles():
    bitmap = np.zeros((20, 20), dtype=bool)
    bitmap[8:12, 8:12] = True
    grid = OccupancyGrid.from_bitmap(bitmap, origin=(0.0, 0.0), resolution=1.0, inflation=2.0)
    hard = grid.hard_obstacles
    margin = grid.inflated_margin_mask()
    assert hard.sum() == 16
    assert margin.any()
    # Hard obstacles and margin should not overlap
    assert not np.any(margin & hard)
