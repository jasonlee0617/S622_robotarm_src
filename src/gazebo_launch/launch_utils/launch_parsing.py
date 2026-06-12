"""Shared launch argument parsing helpers for gazebo_launch entry files."""

from __future__ import annotations

from launch.substitutions import LaunchConfiguration


def as_bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def spawn_pose_from_context(context):
    """Read spawn xyz/rpy launch args and return two float lists."""
    xyz = [
        float(LaunchConfiguration("spawn_x").perform(context)),
        float(LaunchConfiguration("spawn_y").perform(context)),
        float(LaunchConfiguration("spawn_z").perform(context)),
    ]
    rpy = [
        float(LaunchConfiguration("spawn_roll").perform(context)),
        float(LaunchConfiguration("spawn_pitch").perform(context)),
        float(LaunchConfiguration("spawn_yaw").perform(context)),
    ]
    return xyz, rpy

