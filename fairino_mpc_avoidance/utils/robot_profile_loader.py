"""Robot profile loader for fairino_mpc_avoidance launch/scripts.

The loader reuses `gazebo_launch/config/robots/*.yaml` so model switching is
driven by one profile source across packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import yaml
from ament_index_python.packages import get_package_share_directory


@dataclass(frozen=True)
class DemoRobotProfile:
    name: str
    moveit_config_name: str
    moveit_config_package: str
    group_name: str
    ee_frame_name: str
    planning_frame: str
    arm_controller: str
    arm_joints: List[str]


def _normalize_arm_controller(value: str) -> str:
    controller = (value or "").strip()
    if not controller:
        raise ValueError("profile field `arm_controller` is empty.")
    return controller if controller.startswith("/") else f"/{controller}"


def _required_str(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"profile required field `{key}` is missing or empty.")
    return value.strip()


def _required_list(data: Dict[str, Any], key: str) -> List[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"profile required field `{key}` is missing or empty.")
    values = [str(v).strip() for v in value if str(v).strip()]
    if not values:
        raise ValueError(f"profile required field `{key}` has no valid entries.")
    return values


def load_demo_robot_profile(profile_name: str) -> DemoRobotProfile:
    gz_share = get_package_share_directory("gazebo_launch")
    profile_path = f"{gz_share}/config/robots/{profile_name}.yaml"
    with open(profile_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return DemoRobotProfile(
        name=profile_name,
        moveit_config_name=_required_str(data, "moveit_config_name"),
        moveit_config_package=_required_str(data, "moveit_config_package"),
        group_name=_required_str(data, "group_name"),
        ee_frame_name=_required_str(data, "ee_frame_name"),
        planning_frame=_required_str(data, "planning_frame"),
        arm_controller=_normalize_arm_controller(_required_str(data, "arm_controller")),
        arm_joints=_required_list(data, "arm_joints"),
    )
