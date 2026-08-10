"""Shared launch argument parsing helpers for myrobot_simulation entry files."""

from __future__ import annotations

from launch.substitutions import LaunchConfiguration


def as_bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def resolve_launch_args(context, defaults: dict) -> dict:
    """一次性从 context 解析所有 launch 参数，defaults 的 key 决定解析范围。

    遍历 defaults 的每个 key，调用 LaunchConfiguration(key).perform(context)
    获取运行时解析值。当父级 IncludeLaunchDescription 传入 launch_arguments 时，
    ROS 2 launch 系统会自动用父级值覆盖子级 DeclareLaunchArgument 默认值，
    因此父级参数优先级天然高于 defaults 中的值。
    """
    return {name: LaunchConfiguration(name).perform(context) for name in defaults}


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

