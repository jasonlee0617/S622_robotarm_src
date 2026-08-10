import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# 场景与节点标量默认值集中维护；固定 YAML 与模型资源仍由包路径解析。
_LAUNCH_DEFAULTS = {
    "world": "arm_on_the_table",
    "use_sim_time": "true",
    "publish_frequency": "100.0",
    "enable_camera_model": "true",
    "enable_camera_bridge": "true",
    "enable_servo": "true",
    "camera_profile": "d435_color_640x480x30_depth_640x480x30",
    "camera_profile_file": "",
    "camera_noise_mode": "off",
    "camera_depth_far_m": "3.0",
    "camera_fps": "60",
    "camera_image_width": "640",
    "camera_image_height": "480",
    "spawn_z": "1.02",
    "controller_spawn_delay": "5.0",
    "command_burst_count": "1",
}
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in _LAUNCH_DEFAULTS
}


def _declare_launch_arguments():
    """声明可由 CLI 覆盖的场景和节点运行参数."""
    return [
        DeclareLaunchArgument("world", default_value=_LAUNCH_DEFAULTS["world"], description="Gazebo 世界资源。"),
        DeclareLaunchArgument("use_sim_time", default_value=_LAUNCH_DEFAULTS["use_sim_time"], description="是否使用 Gazebo /clock。"),
        DeclareLaunchArgument("camera_profile", default_value=_LAUNCH_DEFAULTS["camera_profile"], description="D435 命名相机配置。"),
        DeclareLaunchArgument("camera_profile_file", default_value=_LAUNCH_DEFAULTS["camera_profile_file"], description="外部 D435 配置 YAML。"),
        DeclareLaunchArgument("camera_noise_mode", default_value=_LAUNCH_DEFAULTS["camera_noise_mode"], choices=["off", "d435_empirical"], description="相机噪声模型。"),
        DeclareLaunchArgument("camera_depth_far_m", default_value=_LAUNCH_DEFAULTS["camera_depth_far_m"], description="D435 深度远裁剪距离，单位米。"),
        DeclareLaunchArgument("publish_frequency", default_value=_LAUNCH_DEFAULTS["publish_frequency"], description="机器人状态发布频率（Hz）。"),
        DeclareLaunchArgument("enable_camera_model", default_value=_LAUNCH_DEFAULTS["enable_camera_model"], description="是否生成仿真相机模型。"),
        DeclareLaunchArgument("enable_camera_bridge", default_value=_LAUNCH_DEFAULTS["enable_camera_bridge"], description="是否桥接相机话题到 ROS 2。"),
        DeclareLaunchArgument("enable_servo", default_value=_LAUNCH_DEFAULTS["enable_servo"], description="是否启动视觉伺服节点。"),
        DeclareLaunchArgument("camera_fps", default_value=_LAUNCH_DEFAULTS["camera_fps"], description="相机帧率。"),
        DeclareLaunchArgument("camera_image_width", default_value=_LAUNCH_DEFAULTS["camera_image_width"], description="彩色图像宽度。"),
        DeclareLaunchArgument("camera_image_height", default_value=_LAUNCH_DEFAULTS["camera_image_height"], description="彩色图像高度。"),
        DeclareLaunchArgument("spawn_z", default_value=_LAUNCH_DEFAULTS["spawn_z"], description="机器人初始高度。"),
        DeclareLaunchArgument("controller_spawn_delay", default_value=_LAUNCH_DEFAULTS["controller_spawn_delay"], description="控制器启动等待时间。"),
        DeclareLaunchArgument("command_burst_count", default_value=_LAUNCH_DEFAULTS["command_burst_count"], description="单次控制命令发送次数。"),
    ]


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")

    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            "world": _LAUNCH_CONFIGURATIONS["world"],
            "rviz_config": os.path.join(gz_share, "rviz", "myrobot_simulation.rviz"),
            "publish_frequency": _LAUNCH_CONFIGURATIONS["publish_frequency"],
            "enable_camera_model": _LAUNCH_CONFIGURATIONS["enable_camera_model"],
            "enable_camera_bridge": _LAUNCH_CONFIGURATIONS["enable_camera_bridge"],
            "enable_servo": _LAUNCH_CONFIGURATIONS["enable_servo"],
            "camera_fps": _LAUNCH_CONFIGURATIONS["camera_fps"],
            "camera_image_width": _LAUNCH_CONFIGURATIONS["camera_image_width"],
            "camera_image_height": _LAUNCH_CONFIGURATIONS["camera_image_height"],
            "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
            "camera_profile": _LAUNCH_CONFIGURATIONS["camera_profile"],
            "camera_profile_file": _LAUNCH_CONFIGURATIONS["camera_profile_file"],
            "camera_noise_mode": _LAUNCH_CONFIGURATIONS["camera_noise_mode"],
            "camera_depth_far_m": _LAUNCH_CONFIGURATIONS["camera_depth_far_m"],
            "spawn_z": _LAUNCH_CONFIGURATIONS["spawn_z"],
            "controller_spawn_delay": _LAUNCH_CONFIGURATIONS["controller_spawn_delay"],
        }.items(),
    )

    retime_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        )
    )

    yolo_obb = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("yolo_perception"),
                        "launch",
                        "yolov8_obb.launch.py",
                    )
                )
            )
        ],
    )
    pose_monitor_node = Node(
        package="llm_arm_control",
        executable="fairino_pose_monitor",
        name="llm_yolo_pose_monitor",
        output="screen",
        parameters=[{"use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"]}],
    )
    task_server_node = Node(
        package="llm_arm_control",
        executable="llm_yolo_task_server",
        name="llm_yolo_task_server",
        output="screen",
        parameters=[
            {"use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"]},
            os.path.join(
                get_package_share_directory("llm_arm_control"),
                "config",
                "llm_yolo_task_sim.yaml",
            )
        ],
    )
    motion_control_node = Node(
        package="manipulation_common",
        executable="motion_control",
        name="motion_control",
        output="screen",
        parameters=[
            {
                "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
                "command_burst_count": _LAUNCH_CONFIGURATIONS["command_burst_count"],
            }
        ],
    )

    return LaunchDescription(
        [
            *_declare_launch_arguments(),
            myrobot_simulation,
            retime_server_launch,
            yolo_obb,
            pose_monitor_node,
            task_server_node,
            motion_control_node,
        ]
    )
