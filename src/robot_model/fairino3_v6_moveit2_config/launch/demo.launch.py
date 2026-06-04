# from moveit_configs_utils import MoveItConfigsBuilder
# from moveit_configs_utils.launches import generate_demo_launch


# def generate_launch_description():
#     moveit_config = MoveItConfigsBuilder("fairino3_v6_robot", package_name="fairino3_v6_moveit2_config").to_moveit_configs()
#     return generate_demo_launch(moveit_config)
# fairino3_v6_moveit2_config/launch/demo.launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ═══════════════════════════════════════
    #  1. 构建 MoveIt 配置
    # ═══════════════════════════════════════
    moveit_config = (
        MoveItConfigsBuilder("fairino3_v6_robot",
                             package_name="fairino3_v6_moveit2_config")
        # ★ 指定自定义规划管线配置
        .planning_pipelines(
            pipelines=["fairino", "ompl"],      # 注册两个管线: 自定义 + OMPL(备用)
            default_planning_pipeline="fairino"  # 默认使用自定义
        )
        .to_moveit_configs()
    )

    # ═══════════════════════════════════════
    #  2. 自定义规划器参数
    # ═══════════════════════════════════════
    fairino_planning_config = os.path.join(
        get_package_share_directory("fairino3_v6_moveit2_config"),
        "config",
        "fairino_planning.yaml"
    )

    # ═══════════════════════════════════════
    #  3. move_group 节点
    # ═══════════════════════════════════════
    move_group_params = [
        moveit_config.to_dict(),
        {"use_sim_time": False},
    ]

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=move_group_params,
    )

    # ═══════════════════════════════════════
    #  4. robot_state_publisher
    # ═══════════════════════════════════════
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    # ═══════════════════════════════════════
    #  5. RViz
    # ═══════════════════════════════════════
    rviz_config_file = os.path.join(
        get_package_share_directory("fairino3_v6_moveit2_config"),
        "config",
        "moveit.rviz"
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # ═══════════════════════════════════════
    #  6. joint_state_publisher (用于无真实硬件时)
    # ═══════════════════════════════════════
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    # ═══════════════════════════════════════
    #  7. ros2_control (fake controller)
    # ═══════════════════════════════════════
    ros2_controllers_path = os.path.join(
        get_package_share_directory("fairino3_v6_moveit2_config"),
        "config",
        "ros2_controllers.yaml",
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )

    # ═══════════════════════════════════════
    #  8. 启动所有节点
    # ═══════════════════════════════════════
    return LaunchDescription([
        robot_state_publisher_node,
        move_group_node,
        rviz_node,
        joint_state_publisher_node,
    ])
