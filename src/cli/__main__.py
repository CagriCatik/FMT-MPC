"""Command-line interface for orchestrating planning, tracking, and simulation.

The CLI provides deterministic entry points for invoking the planner-only
workflow, computing a single MPC control step, or running the full end-to-end
simulation stack.  Logging is configured explicitly at start-up so that both
terminal output and `.log` files capture identical information for debugging
and regression tracking.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console

from src.common.config import load_config
from src.common.geometry import Pose
from src.common.logging_utils import configure_logging
from src.io_utils.maps import export_occupancy_png
from src.mapping.builder import build_grid
from src.planning.fmt_planner import plan_path
from src.sim.simulator import Simulator
from src.tracking.mpc_tracker import Corridor, MPCTracker
from src.vehicle.bicycle import BicycleModel
from src.vis.animation import animate_simulation
from src.vis.dashboard import plot_simulation
from src.vis.fmt_debug import animate_fmt_debug, plot_fmt_debug
from src.vis.vehicle import VehicleDrawParams

console = Console()
logger = logging.getLogger(__name__)


def _run_plan_only(cfg_path: Path) -> None:
    """Plan a path from the loaded configuration and report summary metrics."""
    config = load_config(cfg_path)
    grid = build_grid(config.map_config, config.start[:2], config.goal[:2], config.obstacles)
    path = plan_path(grid, config.start[:2], config.goal[:2], config.planner_config, config.map_config)
    logger.info("Completed planning for %s -> %s", config.start[:2], config.goal[:2])
    console.print(f"Planned cost: {path.cost:.2f}")
    console.print(f"Waypoints: {path.annotated_waypoints[:5]} ...")
    _maybe_show_fmt_debug(grid, path, config)
    _maybe_render_fmt_animation(grid, path, config)


def _run_track_only(cfg_path: Path) -> None:
    """Construct the tracker and return the first MPC command for inspection."""
    config = load_config(cfg_path)
    grid = build_grid(config.map_config, config.start[:2], config.goal[:2], config.obstacles)
    path = plan_path(grid, config.start[:2], config.goal[:2], config.planner_config, config.map_config)
    vehicle = BicycleModel(config.vehicle_config)
    tracker = MPCTracker(config.tracker_config, config.vehicle_config)
    corridor = Corridor(path.annotated_waypoints)
    from src.tracking.mpc_tracker import TrackerState

    tracker_state = TrackerState(*config.start[:3], v=0.0)
    command = tracker.compute_control(tracker_state, corridor, config.tracker_config.timestep)
    logger.info("Computed initial MPC command: steer=%.3f accel=%.3f", command.steer, command.accel)
    console.print(f"Initial control: steer={command.steer:.3f} accel={command.accel:.3f}")


def _run_e2e(cfg_path: Path) -> None:
    """Execute the full pipeline including planning, tracking, and simulation."""
    config = load_config(cfg_path)
    grid = build_grid(config.map_config, config.start[:2], config.goal[:2], config.obstacles)
    path = plan_path(grid, config.start[:2], config.goal[:2], config.planner_config, config.map_config)
    logger.info("Planned end-to-end scenario; path length %d waypoints", len(path.annotated_waypoints))
    vehicle = BicycleModel(config.vehicle_config)
    tracker = MPCTracker(config.tracker_config, config.vehicle_config)
    corridor = Corridor(path.annotated_waypoints)
    simulator = Simulator(
        tracker,
        vehicle,
        config.simulation_config.dt,
        config.simulation_config.duration,
        log_states=config.simulation_config.log_states,
    )
    log = simulator.run(Pose(*config.start), corridor)
    viz_cfg = config.visualization_config
    base_dir = Path(viz_cfg.output_dir)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    vehicle_params = _vehicle_draw_params(viz_cfg)
    if viz_cfg.enable_plots:
        plot_path = base_dir / viz_cfg.plot_filename
        plot_simulation(grid, path, log, plot_path, vehicle_params=vehicle_params)
        console.print(f"Saved summary plot to {plot_path}")
        logger.info("Summary plot written to %s", plot_path)
    if viz_cfg.animate:
        gif_path = base_dir / viz_cfg.animation_filename
        animate_simulation(grid, path, log, gif_path, vehicle_params=vehicle_params)
        console.print(f"Saved animation to {gif_path}")
        logger.info("Animation written to %s", gif_path)
    _maybe_render_fmt_animation(grid, path, config)
    if viz_cfg.show_planner_debug or viz_cfg.planner_debug_filename:
        debug_output = base_dir / viz_cfg.planner_debug_filename if viz_cfg.planner_debug_filename else None
        plot_fmt_debug(grid, path, debug_output, show=viz_cfg.show_planner_debug)
        if debug_output:
            console.print(f"Saved FMT debug plot to {debug_output}")
            logger.info("FMT debug plot written to %s", debug_output)
        if viz_cfg.show_planner_debug:
            console.print("Displayed FMT exploration window")
    console.print(f"Simulated {len(log.time)} steps with final speed {log.v[-1]:.2f} m/s")
    logger.info("Simulation finished after %d steps", len(log.time))


def _export_map(cfg_path: Path) -> None:
    """Generate a deterministic occupancy map PNG from the provided config."""
    config = load_config(cfg_path)
    grid = build_grid(config.map_config, config.start[:2], config.goal[:2], config.obstacles)
    if config.visualization_config.map_png:
        output = Path(config.visualization_config.map_png)
    elif config.map_config.export_bitmap:
        output = Path(config.map_config.export_bitmap)
    else:
        output = cfg_path.with_suffix(".png")
    if not output.is_absolute():
        output = Path.cwd() / output
    export_occupancy_png(grid, output)
    console.print(f"Occupancy map exported to {output}")
    logger.info("Occupancy PNG exported to %s", output)


def main() -> None:
    """Parse CLI arguments, configure logging, and dispatch to sub-commands."""

    parser = argparse.ArgumentParser(description="FMT based planner and MPC tracker")
    parser.add_argument("config", type=Path, help="Path to experiment YAML")
    parser.add_argument(
        "mode",
        choices=["plan_only", "track_only", "run_e2e", "export_map"],
        help="Execution mode",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    logger.info("Log file initialised at %s", log_path)
    console.print(f"Logging to {log_path}")

    if args.mode == "plan_only":
        _run_plan_only(args.config)
    elif args.mode == "track_only":
        _run_track_only(args.config)
    elif args.mode == "export_map":
        _export_map(args.config)
    else:
        _run_e2e(args.config)


def _maybe_show_fmt_debug(grid, path, config) -> None:
    viz_cfg = config.visualization_config
    if not viz_cfg.show_planner_debug and not viz_cfg.planner_debug_filename:
        return
    base_dir = Path(viz_cfg.output_dir)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    debug_output = None
    if viz_cfg.planner_debug_filename:
        base_dir.mkdir(parents=True, exist_ok=True)
        debug_output = base_dir / viz_cfg.planner_debug_filename
    plot_fmt_debug(grid, path, debug_output, show=viz_cfg.show_planner_debug)
    if debug_output:
        console.print(f"Saved FMT debug plot to {debug_output}")
        logger.info("FMT debug plot written to %s", debug_output)
    if viz_cfg.show_planner_debug:
        console.print("Displayed FMT exploration window")
        logger.info("Displayed planner debug window")


def _maybe_render_fmt_animation(grid, path, config) -> None:
    viz_cfg = config.visualization_config
    if not viz_cfg.animate_planner and not viz_cfg.planner_animation_filename:
        return
    base_dir = Path(viz_cfg.output_dir)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = viz_cfg.planner_animation_filename or "fmt_exploration.gif"
    gif_path = base_dir / filename
    fps = getattr(viz_cfg, "planner_animation_fps", 10) or 10
    animate_fmt_debug(grid, path, gif_path, fps=fps)
    console.print(f"Saved FMT animation to {gif_path}")
    logger.info("FMT animation written to %s", gif_path)


def _vehicle_draw_params(viz_cfg) -> VehicleDrawParams | None:
    draw_cfg = getattr(viz_cfg, "vehicle_draw", None)
    if draw_cfg is None:
        return None
    return VehicleDrawParams(
        wheelbase=draw_cfg.wheelbase,
        width=draw_cfg.width,
        length=draw_cfg.length,
        front_overhang=draw_cfg.front_overhang,
        rear_overhang=draw_cfg.rear_overhang,
        wheel_track=draw_cfg.wheel_track,
        tire_radius=draw_cfg.tire_radius,
        tire_width=draw_cfg.tire_width,
    )


if __name__ == "__main__":
    main()
