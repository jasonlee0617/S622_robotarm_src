from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from visual_servo.task.task_types import TargetType


@dataclass(frozen=True)
class GraspTaskConfig:
    servo_entry_mode: str
    safe_height: float
    place_offset: float
    home_joints: list[float]
    action_delay: float
    detection_timeout: float
    num_candidate_plans: int
    wrist_weight: float
    wrist_joint_indices: tuple[int, int, int]
    grasp_profile: Dict[TargetType, dict]


def _declare_get(node, name: str, default):
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


def _profile(node, prefix: str, defaults: dict) -> dict:
    return {
        "roll": float(_declare_get(node, f"{prefix}.roll", defaults["roll"])),
        "pitch": float(_declare_get(node, f"{prefix}.pitch", defaults["pitch"])),
        "yaw_offset": float(_declare_get(node, f"{prefix}.yaw_offset", defaults["yaw_offset"])),
        "above_z": float(_declare_get(node, f"{prefix}.above_z", defaults["above_z"])),
        "grasp_z": float(_declare_get(node, f"{prefix}.grasp_z", defaults["grasp_z"])),
    }


def load_grasp_task_config(node) -> GraspTaskConfig:
    home_default = [-2.6698, -1.4838, 2.1455, -2.6344, -0.80, -0.70]
    wrist_indices = _declare_get(node, "wrist_joint_indices", [2, 3, 4])
    if len(wrist_indices) != 3:
        raise RuntimeError("wrist_joint_indices must contain exactly three joint indices")

    profiles = {
        TargetType.PEN: _profile(
            node,
            "grasp.pen",
            {"roll": 0.0, "pitch": -180.0, "yaw_offset": -180.0, "above_z": 0.05, "grasp_z": 0.00},
        ),
        TargetType.CUBE: _profile(
            node,
            "grasp.cube",
            {"roll": 0.0, "pitch": -180.0, "yaw_offset": -165.0, "above_z": 0.08, "grasp_z": 0.02},
        ),
    }

    entry_mode = str(_declare_get(node, "servo_entry_mode", "direct_from_home")).strip()
    if entry_mode not in ("direct_from_home", "target_above_first"):
        raise RuntimeError("servo_entry_mode must be 'direct_from_home' or 'target_above_first'")

    return GraspTaskConfig(
        servo_entry_mode=entry_mode,
        safe_height=float(_declare_get(node, "safe_height", 0.04)),
        place_offset=float(_declare_get(node, "place_offset", 0.10)),
        home_joints=[float(x) for x in _declare_get(node, "home_joints", home_default)],
        action_delay=float(_declare_get(node, "action_delay", 0.5)),
        detection_timeout=float(_declare_get(node, "detection_timeout", 3.0)),
        num_candidate_plans=int(_declare_get(node, "num_candidate_plans", 5)),
        wrist_weight=float(_declare_get(node, "wrist_weight", 50.0)),
        wrist_joint_indices=tuple(int(x) for x in wrist_indices),
        grasp_profile=profiles,
    )

