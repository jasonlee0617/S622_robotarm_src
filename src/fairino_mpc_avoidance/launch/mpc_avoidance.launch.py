"""MPC 避障运行入口."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# 配置和 RViz 文件是本包固定资源；时钟是可在启动命令中覆盖的运行时参数。
_NODE_DEFAULTS = {"use_sim_time": "true"}
_USE_SIM_TIME = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
_LAUNCH_ARGUMENTS = [
    DeclareLaunchArgument(
        "use_sim_time", default_value=_NODE_DEFAULTS["use_sim_time"], description="是否使用仿真时间。"
    )
]


def generate_launch_description():
    pkg_dir = get_package_share_directory("fairino_mpc_avoidance")
    return LaunchDescription(
        [
            *_LAUNCH_ARGUMENTS,
            Node(
                package="fairino_mpc_avoidance",
                executable="mpc_avoidance_node",
                name="mpc_avoidance_node",
                output="screen",
                # YAML 位于 launch 覆盖之后，因此节点最终优先级为 CLI > YAML > launch > 默认值。
                parameters=[
                    {"use_sim_time": _USE_SIM_TIME},
                    os.path.join(pkg_dir, "config", "mpc_params.yaml"),
                ],
                remappings=[
                    ("/joint_states", "/joint_states"),
                    ("/planned_trajectory", "/fairino_planner/trajectory"),
                    ("/detected_obstacles", "/yolo/obstacles"),
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", os.path.join(pkg_dir, "rviz", "mpc_avoidance.rviz")],
                parameters=[{"use_sim_time": _USE_SIM_TIME}],
            ),
        ]
    )
