"""Load the single source of truth for position-servo launch parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CONTROLLER_TYPES = {"PID", "PD", "PI_FF", "ADAPTIVE_PID", "LADRC", "NLADRC", "MPC"}


def config_path() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("visual_servo_bringup")) / "config" / "visual_position_servo.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or config_path()).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise RuntimeError("visual_position_servo.yaml must contain a mapping")
    return config


def _flat_params(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            result.update(_flat_params(value, key))
        else:
            result[key] = value
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def gazebo_camera_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    return _flat_params(_mapping(_mapping(config.get("gazebo"), "gazebo").get("camera"), "gazebo.camera"))


def visual_servo_parameters(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    node = _mapping(_mapping(config.get("nodes"), "nodes").get("visual_servo_grasping"), "nodes.visual_servo_grasping")
    result = gazebo_camera_defaults(config)
    for name in ("task", "moveit", "runtime"):
        result.update(_flat_params(_mapping(node.get(name), f"nodes.visual_servo_grasping.{name}")))

    controllers = _mapping(node.get("controllers"), "nodes.visual_servo_grasping.controllers")
    active = str(controllers.get("active", "")).strip().upper()
    profiles = _mapping(controllers.get("profiles"), "nodes.visual_servo_grasping.controllers.profiles")
    if active not in _CONTROLLER_TYPES or active not in profiles:
        raise RuntimeError(f"Unsupported position-servo controller: {active}")
    result["servo_controller_type"] = active
    for section in _mapping(profiles[active], f"controllers.profiles.{active}").values():
        result.update(_flat_params(_mapping(section, f"controllers.profiles.{active}")))
    return result


def yolo_kalman_parameters(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    node = _mapping(_mapping(config.get("nodes"), "nodes").get("yolo_kalman"), "nodes.yolo_kalman")
    return _flat_params(node)
