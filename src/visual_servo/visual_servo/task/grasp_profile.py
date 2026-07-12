from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from manipulation_common.utils.params import param
from visual_servo.task.task_types import TargetType


@dataclass(frozen=True)
class GraspTaskConfig:
    """Task-level grasp/place parameters loaded from grasp_task.yaml."""
    safe_height: float
    place_offset: float
    home_joints: list[float]
    action_delay: float
    detection_timeout: float
    num_candidate_plans: int
    wrist_weight: float
    wrist_joint_indices: tuple[int, int, int]
    grasp_profile: Dict[TargetType, dict]


def _profile(node, prefix: str, defaults: dict) -> dict:
    return {
        "roll": float(param(node, f"{prefix}.roll", defaults["roll"])),
        "pitch": float(param(node, f"{prefix}.pitch", defaults["pitch"])),
        "yaw_offset": float(param(node, f"{prefix}.yaw_offset", defaults["yaw_offset"])),
        "above_z": float(param(node, f"{prefix}.above_z", defaults["above_z"])),
        "grasp_z": float(param(node, f"{prefix}.grasp_z", defaults["grasp_z"])),
    }


def load_grasp_task_config(node) -> GraspTaskConfig:
    """Read grasp task parameters without touching visual-servo control tuning."""
    home_default = [-2.6698, -1.4838, 2.1455, -2.6344, -0.80, -0.70]
    wrist_indices = param(node, "wrist_joint_indices", [2, 3, 4])
    if len(wrist_indices) != 3:
        raise RuntimeError("wrist_joint_indices must contain exactly three joint indices")

    profiles = {
        TargetType.ELONGATED_OBJECT: _profile(
            node,
            "grasp.elongated_object",
            {"roll": 0.0, "pitch": -180.0, "yaw_offset": -180.0, "above_z": 0.05, "grasp_z": 0.00},
        ),
        TargetType.CUBE: _profile(
            node,
            "grasp.cube",
            {"roll": 0.0, "pitch": -180.0, "yaw_offset": -165.0, "above_z": 0.08, "grasp_z": 0.02},
        ),
    }

    return GraspTaskConfig(
        safe_height=float(param(node, "safe_height", 0.04)),
        place_offset=float(param(node, "place_offset", 0.10)),
        home_joints=[float(x) for x in param(node, "home_joints", home_default)],
        action_delay=float(param(node, "action_delay", 0.5)),
        detection_timeout=float(param(node, "detection_timeout", 3.0)),
        num_candidate_plans=int(param(node, "num_candidate_plans", 5)),
        wrist_weight=float(param(node, "wrist_weight", 50.0)),
        wrist_joint_indices=tuple(int(x) for x in wrist_indices),
        grasp_profile=profiles,
    )
