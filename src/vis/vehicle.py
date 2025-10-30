"""Vehicle drawing helpers for planar visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib.artist import Artist
import numpy as np


@dataclass
class VehicleDrawParams:
    """Geometric properties of the vehicle footprint in meters."""

    wheelbase: float
    width: float
    # ``length`` is optional/redundant once you have wheelbase + overhangs.
    # Keep for compatibility; it is not used to place geometry.
    length: float
    front_overhang: float
    rear_overhang: float
    wheel_track: float
    tire_radius: float
    tire_width: float


def draw_vehicle(
    ax: Axes,
    x: float,
    y: float,
    yaw: float,
    steer: float,
    params: VehicleDrawParams,
) -> List[Artist]:
    """
    Draw a planar vehicle footprint using a consistent frame:
    origin at the mid-point between front and rear axles,
    +x forward, +y left. Wheels are centered on axle lines.
    """

    artists: List[Artist] = []

    # Vehicle rotation (world-from-vehicle)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    # Body rectangle in vehicle frame
    # Front and rear x extents measured from mid-axle origin
    x_front = params.wheelbase / 2.0 + params.front_overhang
    x_rear = -(params.wheelbase / 2.0 + params.rear_overhang)
    body_corners_v = np.array(
        [
            [x_front, params.width / 2.0],
            [x_rear, params.width / 2.0],
            [x_rear, -params.width / 2.0],
            [x_front, -params.width / 2.0],
        ]
    )
    body_corners_w = body_corners_v @ R.T + np.array([x, y])
    body = patches.Polygon(
        body_corners_w, closed=True, fill=False, edgecolor="tab:blue", linewidth=1.5
    )
    ax.add_patch(body)
    artists.append(body)

    # Wheel centers in vehicle frame (front steered, rear straight)
    wheel_centers_v = [
        (params.wheelbase / 2.0, params.wheel_track / 2.0, steer),  # FL
        (params.wheelbase / 2.0, -params.wheel_track / 2.0, steer),  # FR
        (-params.wheelbase / 2.0, params.wheel_track / 2.0, 0.0),  # RL
        (-params.wheelbase / 2.0, -params.wheel_track / 2.0, 0.0),  # RR
    ]

    # Tire rectangle template in each wheel's local frame
    tire_box = np.array(
        [
            [params.tire_radius, params.tire_width / 2.0],
            [-params.tire_radius, params.tire_width / 2.0],
            [-params.tire_radius, -params.tire_width / 2.0],
            [params.tire_radius, -params.tire_width / 2.0],
        ]
    )

    for dx, dy, steer_i in wheel_centers_v:
        # Wheel center in world
        center_w = R @ np.array([dx, dy]) + np.array([x, y])

        # Wheel orientation in world = vehicle yaw then local steer
        Rw = R @ np.array([[np.cos(steer_i), -np.sin(steer_i)], [np.sin(steer_i), np.cos(steer_i)]])

        tire_pts_w = tire_box @ Rw.T + center_w
        wheel = patches.Polygon(
            tire_pts_w, closed=True, fill=False, edgecolor="0.2", linewidth=1.0
        )
        ax.add_patch(wheel)
        artists.append(wheel)

    # Heading arrow from vehicle origin
    arrow_len = 0.6 * params.wheelbase
    tip = R @ np.array([arrow_len, 0.0]) + np.array([x, y])
    arrow = ax.annotate(
        "",
        xy=(tip[0], tip[1]),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color="tab:red", linewidth=1.2),
    )
    artists.append(arrow)

    return artists
