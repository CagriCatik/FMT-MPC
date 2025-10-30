"""Configuration dataclasses and YAML loaders for deterministic experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type, TypeVar

T = TypeVar("T")

try:  # pragma: no cover - exercised in integration tests
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    yaml = None
    import ast

    def _parse_scalar(token: str):
        lower = token.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        try:
            return ast.literal_eval(token)
        except Exception:
            return token

    def _parse_list(lines, indent):
        items = []
        while lines:
            raw = lines[0]
            if not raw.strip():
                lines.pop(0)
                continue
            current = len(raw) - len(raw.lstrip(" "))
            if current < indent:
                break
            stripped = raw.strip()
            if not stripped.startswith("- "):
                break
            lines.pop(0)
            value = stripped[2:].strip()
            if not value:
                if lines and lines[0].strip().startswith("- "):
                    items.append(_parse_list(lines, current + 2))
                else:
                    items.append(_parse_block(lines, current + 2))
                continue
            if ":" in value:
                key, _, remainder = value.partition(":")
                key = key.strip()
                remainder = remainder.strip()
                item = {}
                if remainder:
                    item[key] = _parse_scalar(remainder)
                    extra = _parse_block(lines, current + 2)
                    if isinstance(extra, dict):
                        item.update(extra)
                else:
                    if lines and lines[0].strip().startswith("- "):
                        item[key] = _parse_list(lines, current + 2)
                    else:
                        item[key] = _parse_block(lines, current + 2)
                items.append(item)
            else:
                items.append(_parse_scalar(value))
        return items

    def _parse_block(lines, indent):
        mapping = {}
        while lines:
            raw = lines[0]
            if not raw.strip():
                lines.pop(0)
                continue
            current = len(raw) - len(raw.lstrip(" "))
            if current < indent:
                break
            stripped = lines.pop(0).strip()
            if stripped.startswith("- "):
                raise ValueError("Unexpected list item without key")
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if not value:
                if lines and lines[0].strip().startswith("- "):
                    mapping[key] = _parse_list(lines, current + 2)
                else:
                    mapping[key] = _parse_block(lines, current + 2)
            else:
                mapping[key] = _parse_scalar(value)
        return mapping

    def safe_load_yaml(text: str) -> Dict[str, object]:
        """Parse YAML using a small deterministic subset of the language."""

        lines = text.splitlines()
        return _parse_block(lines, 0)
else:  # pragma: no cover - executed when PyYAML is available
    def safe_load_yaml(text: str):
        """Parse YAML using :mod:`yaml` when available."""

        return yaml.safe_load(text)


@dataclass
class MapConfig:
    """Parameters controlling how the occupancy grid is constructed or loaded."""

    resolution: float = 0.5
    obstacle_inflation: float = 1.0
    safety_margin: float = 0.5
    validate_corridor: bool = True
    bitmap_path: Optional[str] = None
    spec_path: Optional[str] = None
    export_bitmap: Optional[str] = None
    origin: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.origin is not None:
            ox, oy = self.origin
            object.__setattr__(self, "origin", (float(ox), float(oy)))


@dataclass
class PlannerConfig:
    """Configuration parameters for the FMT global planner."""

    sample_count: int = 400
    connection_radius: float = 8.0
    goal_bias: float = 0.2
    rewiring: bool = True
    pruning: bool = False
    smoothing: bool = True
    nominal_speed: float = 5.0


@dataclass
class TrackerConfig:
    """Weights and bounds used by the MPC tracker."""

    horizon: int = 12
    timestep: float = 0.2
    q_weights: Tuple[float, float, float, float] = (1.0, 1.0, 0.3, 0.2)
    r_weights: Tuple[float, float] = (0.05, 0.05)
    cruise_speed: float = 8.0
    slowdown_radius: float = 6.0
    wheelbase: float = 2.8

    def __post_init__(self) -> None:
        if len(self.q_weights) == 3:
            qx, qy, qyaw = self.q_weights
            object.__setattr__(self, "q_weights", (qx, qy, qyaw, qyaw))
        if len(self.r_weights) == 3:
            steer, accel, _ = self.r_weights
            object.__setattr__(self, "r_weights", (steer, accel))


@dataclass
class VehicleConfig:
    """Physical properties and actuator limits of the bicycle model."""

    wheelbase: float = 2.8
    max_steer: float = 0.6
    max_accel: float = 4.0
    max_decel: float = -6.0


@dataclass
class SimulationConfig:
    """Integrator timing controls for the deterministic simulator."""

    dt: float = 0.05
    duration: float = 60.0
    log_states: bool = True


@dataclass
class VehicleDrawConfig:
    """Vehicle footprint geometry used purely for rendering overlays."""

    wheelbase: float = 2.8
    width: float = 1.6
    length: float = 4.5
    front_overhang: float = 0.9
    rear_overhang: float = 0.9
    wheel_track: float = 1.5
    tire_radius: float = 0.3
    tire_width: float = 0.2


@dataclass
class VisualizationConfig:
    """Output controls for static plots, animations, and planner diagnostics."""

    enable_plots: bool = True
    animate: bool = False
    dashboard: bool = True
    output_dir: str = "artifacts"
    plot_filename: str = "trajectory.png"
    animation_filename: str = "trajectory.gif"
    map_png: Optional[str] = None
    show_planner_debug: bool = False
    planner_debug_filename: Optional[str] = None
    animate_planner: bool = False
    planner_animation_filename: Optional[str] = None
    planner_animation_fps: int = 10
    vehicle_draw: VehicleDrawConfig | None = field(default_factory=VehicleDrawConfig)

    def __post_init__(self) -> None:
        if isinstance(self.vehicle_draw, dict):
            object.__setattr__(self, "vehicle_draw", VehicleDrawConfig(**self.vehicle_draw))


@dataclass
class ExperimentConfig:
    """Top-level container bundling every subsystem configuration block."""

    map_config: MapConfig = field(default_factory=MapConfig)
    planner_config: PlannerConfig = field(default_factory=PlannerConfig)
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)
    vehicle_config: VehicleConfig = field(default_factory=VehicleConfig)
    simulation_config: SimulationConfig = field(default_factory=SimulationConfig)
    visualization_config: VisualizationConfig = field(default_factory=VisualizationConfig)
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal: Tuple[float, float, float] = (20.0, 20.0, 0.0)
    obstacles: Sequence[Tuple[float, float, float]] = field(default_factory=list)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load an experiment configuration from ``path`` with defaults applied."""

    raw = Path(path).read_text()
    data = safe_load_yaml(raw)
    if data is None:
        return ExperimentConfig()

    if not isinstance(data, dict):  # defensive: malformed YAML
        raise TypeError(f"Experiment configuration must be a mapping, got {type(data)!r}")

    def build(cls: Type[T], key: str) -> T:
        values = data.get(key, {})
        if not isinstance(values, dict):
            raise TypeError(f"Section '{key}' must be a mapping, received {type(values)!r}")
        return cls(**values)

    experiment = ExperimentConfig(
        map_config=build(MapConfig, "map"),
        planner_config=build(PlannerConfig, "planner"),
        tracker_config=build(TrackerConfig, "tracker"),
        vehicle_config=build(VehicleConfig, "vehicle"),
        simulation_config=build(SimulationConfig, "simulation"),
        visualization_config=build(VisualizationConfig, "visualization"),
        start=tuple(data.get("start", (0.0, 0.0, 0.0))),
        goal=tuple(data.get("goal", (20.0, 20.0, 0.0))),
        obstacles=[tuple(obs) for obs in data.get("obstacles", [])],
    )

    viz = experiment.visualization_config
    map_cfg = experiment.map_config
    if viz.map_png and map_cfg.export_bitmap is None:
        object.__setattr__(map_cfg, "export_bitmap", viz.map_png)
    if viz.map_png and map_cfg.bitmap_path is None and Path(viz.map_png).exists():
        object.__setattr__(map_cfg, "bitmap_path", viz.map_png)
    return experiment


def load_multiple(paths: Sequence[Path | str]) -> List[ExperimentConfig]:
    """Load experiment configurations in the order they are supplied."""

    return [load_config(path) for path in paths]
