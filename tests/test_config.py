from __future__ import annotations

from pathlib import Path

from src.common.config import VehicleDrawConfig, load_config


def test_load_config(tmp_path: Path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text(
        """
start: [1.0, 2.0, 0.1]
goal: [5.0, 6.0, 0.0]
map:
  resolution: 0.25
  obstacle_inflation: 1.2
tracker:
  cruise_speed: 7.5
visualization:
  map_png: artifacts/test_map.png
  vehicle_draw:
    wheelbase: 3.0
    width: 1.8
    length: 4.8
    front_overhang: 1.0
    rear_overhang: 0.8
    wheel_track: 1.6
    tire_radius: 0.35
    tire_width: 0.24
"""
    )
    cfg = load_config(yaml)
    assert cfg.start == (1.0, 2.0, 0.1)
    assert cfg.map_config.resolution == 0.25
    assert cfg.tracker_config.cruise_speed == 7.5
    assert cfg.map_config.export_bitmap == "artifacts/test_map.png"
    assert isinstance(cfg.visualization_config.vehicle_draw, VehicleDrawConfig)
    assert cfg.visualization_config.vehicle_draw.width == 1.8
