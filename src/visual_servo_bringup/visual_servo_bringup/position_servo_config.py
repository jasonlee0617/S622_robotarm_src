"""Load the single source of truth for position-servo launch parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from manipulation_common.launch_utils.yaml_loader import flatten_moveit_parameters


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


def sim_camera_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    environments = _mapping(config.get("environments"), "environments")
    sim = _mapping(environments.get("sim"), "environments.sim")
    launch = _mapping(sim.get("launch"), "environments.sim.launch")
    return _flat_params(_mapping(launch.get("camera"), "environments.sim.launch.camera"))


def sim_aruco_motion_parameters(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    environments = _mapping(config.get("environments"), "environments")
    sim = _mapping(environments.get("sim"), "environments.sim")
    launch = _mapping(sim.get("launch"), "environments.sim.launch")
    return _flat_params(_mapping(launch.get("aruco_motion"), "environments.sim.launch.aruco_motion"))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for name, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name] = _deep_merge(result[name], value)
        else:
            result[name] = value
    return result


def visual_servo_parameters(environment: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    if environment not in {"sim", "real"}:
        raise ValueError("environment must be 'sim' or 'real'")
    config = config or load_config()
    common = _mapping(
        _mapping(_mapping(config.get("common"), "common").get("nodes"), "common.nodes").get("visual_servo_grasping"),
        "common.nodes.visual_servo_grasping",
    )
    environments = _mapping(config.get("environments"), "environments")
    selected = _mapping(
        _mapping(_mapping(environments.get(environment), f"environments.{environment}").get("nodes"), f"environments.{environment}.nodes").get("visual_servo_grasping"),
        f"environments.{environment}.nodes.visual_servo_grasping",
    )
    result = sim_camera_defaults(config) if environment == "sim" else {}
    merged = _deep_merge(common, selected)
    for name in ("task", "planning", "runtime"):
        result.update(_flat_params(_mapping(merged.get(name), f"nodes.visual_servo_grasping.{name}")))
    perception = _mapping(merged.get("perception"), "common.nodes.visual_servo_grasping.perception")
    active_source = str(perception.get("active_source", "")).strip().lower()
    if active_source not in {"yolo_kalman", "aruco"}:
        raise RuntimeError(f"Unsupported position-servo perception source: {active_source}")
    result["perception_source"] = active_source
    for name, value in _mapping(perception.get("aruco"), "nodes.visual_servo_grasping.perception.aruco").items():
        result[f"aruco_{name}"] = value
    result.update(flatten_moveit_parameters(_mapping(merged.get("moveit"), "common.nodes.visual_servo_grasping.moveit")))

    controllers = _mapping(merged.get("controllers"), "common.nodes.visual_servo_grasping.controllers")
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
    node = _mapping(
        _mapping(_mapping(config.get("common"), "common").get("nodes"), "common.nodes").get("yolo_kalman"),
        "common.nodes.yolo_kalman",
    )
    return _flat_params(node)
