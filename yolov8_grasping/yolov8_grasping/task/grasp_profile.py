from dataclasses import dataclass
from typing import Dict

import numpy as np

from yolov8_grasping.task.task_types import TargetType


@dataclass(frozen=True)
class GraspProfile:
    roll: float
    pitch: float
    yaw_offset: float
    above_z: float
    grasp_z: float


def _param(node, name: str, default):
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


def load_grasp_profiles(node) -> Dict[TargetType, GraspProfile]:
    profiles = {}
    for target in TargetType:
        prefix = f"grasp.{target.value}"
        profiles[target] = GraspProfile(
            roll=float(_param(node, f"{prefix}.roll", 0.0)),
            pitch=float(_param(node, f"{prefix}.pitch", -180.0)),
            yaw_offset=float(_param(node, f"{prefix}.yaw_offset", 0.0)),
            above_z=float(_param(node, f"{prefix}.above_z", 0.05)),
            grasp_z=float(_param(node, f"{prefix}.grasp_z", 0.01)),
        )
    return profiles


def build_task_poses(node, target: TargetType, obj_pos_base, box_pos_base, obj_rpy: dict):
    obj_y_deg = float(np.degrees(obj_rpy["yaw"]))
    obj_r_deg = float(np.degrees(obj_rpy["roll"]))
    obj_p_deg = float(np.degrees(obj_rpy["pitch"]))
    profile = node.grasp_profiles[target]
    target_name = target.value

    node.get_logger().info(
        f"✓ Using {target_name} RPY(deg): R={obj_r_deg:.1f}, P={obj_p_deg:.1f}, Y={obj_y_deg:.1f}"
    )

    obj_roll = profile.roll
    obj_pitch = profile.pitch
    obj_yaw = profile.yaw_offset + obj_y_deg

    poses = {
        "target_above": node.pose_tools.make_pose(
            obj_pos_base.x,
            obj_pos_base.y,
            profile.above_z,
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
        "target_grasp": node.pose_tools.make_pose(
            obj_pos_base.x,
            obj_pos_base.y,
            profile.grasp_z,
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
        "target_lift": node.pose_tools.make_pose(
            obj_pos_base.x,
            obj_pos_base.y,
            obj_pos_base.z + node.place_offset,
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
        "box_above": node.pose_tools.make_pose(
            box_pos_base.x,
            box_pos_base.y,
            box_pos_base.z + node.place_offset,
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
        "descend_to_box": node.pose_tools.make_pose(
            box_pos_base.x,
            box_pos_base.y,
            box_pos_base.z + 0.07,
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
    }

    node.get_logger().info(f"✓ Poses ready for {target_name}")
    for key, pose in poses.items():
        node.get_logger().info(
            f"  - {key}: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
        )
    return poses
