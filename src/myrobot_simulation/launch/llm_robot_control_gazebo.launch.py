import os
import shlex
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from manipulation_common.launch_utils.yaml_loader import load_node_parameters_yaml


# 场景与节点标量默认值集中维护；固定 YAML 与模型资源仍由包路径解析。
_LAUNCH_DEFAULTS = {
    "robot_profile": "fairino_arm_gripper_inhand",
    "world": "visual_grasping",
    "use_sim_time": "true",
    "publish_frequency": "100.0",
    "enable_camera_model": "true",
    "enable_camera_bridge": "true",
    "enable_servo": "true",
    "camera_profile": "d435_color_1280x720x30_depth_848x480x30",
    "camera_profile_file": "",
    "camera_noise_mode": "off",
    "camera_depth_far_m": "3.0",
    "camera_fps": "30",
    "camera_image_width": "1280",
    "camera_image_height": "720",
    "spawn_z": "1.02",
    "controller_spawn_delay": "5.0",
    "command_burst_count": "1",
    "graspnet_model_profile": "rs",
    "use_continuous_yolo": "true",
}
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in _LAUNCH_DEFAULTS
}


def _declare_launch_arguments():
    """声明可由 CLI 覆盖的场景和节点运行参数."""
    return [
        DeclareLaunchArgument("robot_profile", default_value=_LAUNCH_DEFAULTS["robot_profile"], description="Gazebo 机器人配置。"),
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
        DeclareLaunchArgument("graspnet_model_profile", default_value=_LAUNCH_DEFAULTS["graspnet_model_profile"], choices=["rs", "kn"], description="GraspNet 模型权重。"),
        DeclareLaunchArgument("use_continuous_yolo", default_value=_LAUNCH_DEFAULTS["use_continuous_yolo"], description="是否持续执行 YOLO 推理。"),
    ]


def _graspnet_inference_process(context):
    model_profile = _LAUNCH_CONFIGURATIONS["graspnet_model_profile"].perform(context)
    use_sim_time = _LAUNCH_CONFIGURATIONS["use_sim_time"].perform(context)
    install_setup = str(Path(get_package_prefix("graspnet_bringup")).parent / "setup.bash")
    source_share = get_package_share_directory("graspnet_source")
    config_path = os.path.join(
        get_package_share_directory("llm_arm_control"), "config", "llm_robot_control.yaml"
    )
    conda_setup = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    command = (
        "set -e; "
        f"source {shlex.quote(conda_setup)}; conda activate graspnet; "
        "source /opt/ros/humble/setup.bash; "
        f"source {shlex.quote(install_setup)}; "
        "export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/graspnet_mpl_config XDG_CACHE_HOME=/tmp/graspnet_xdg_cache; "
        "mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME; "
        "exec python -m graspnet_bringup.graspnet_inference_node --ros-args "
        f"--params-file {shlex.quote(config_path)} -r __node:=graspnet_inference "
        f"-p use_sim_time:={use_sim_time} "
        f"-p baseline_dir:={shlex.quote(os.path.join(source_share, 'graspnet_baseline'))} "
        f"-p checkpoint_path:={shlex.quote(os.path.join(source_share, 'models', f'checkpoint-{model_profile}.tar'))}"
    )
    return [ExecuteProcess(cmd=["bash", "-lc", command], output="screen")]


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")
    llm_arm_share = get_package_share_directory("llm_arm_control")

    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            # 眼在手外：fairino_arm_gripper_calibration_onbase；眼在手上：fairino_arm_gripper_inhand。
            "robot_profile": _LAUNCH_CONFIGURATIONS["robot_profile"],
            "world": _LAUNCH_CONFIGURATIONS["world"],
            "rviz_config": os.path.join(llm_arm_share, "rviz", "llm_robot_control.rviz"),
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
                        get_package_share_directory("visual_perception"),
                        "launch",
                        "llm_visual_perception.launch.py",
                    )
                ),
                launch_arguments={
                    "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
                    "use_continuous_yolo": _LAUNCH_CONFIGURATIONS["use_continuous_yolo"],
                }.items(),
            )
        ],
    )
    pose_monitor_node = Node(
        package="llm_arm_control",
        executable="robot_pose_monitor_node",
        name="robot_pose_monitor_node",
        output="screen",
        parameters=[{"use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"]}],
    )
    task_server_node = Node(
        package="llm_arm_control",
        executable="llm_control_task_server",
        name="llm_control_task_server",
        output="screen",
        parameters=[
            load_node_parameters_yaml(
                "llm_arm_control", "config/llm_robot_control.yaml", "llm_control_task_server"
            ),
            {
                "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
                "use_continuous_yolo": _LAUNCH_CONFIGURATIONS["use_continuous_yolo"],
            },
        ],
    )
    # The CLI owns a real terminal. A launch Node only receives launch-managed
    # stdin, so its input() prompt would be hidden by the shared node output.
    cli_terminal = ExecuteProcess(
        cmd=[
            "gnome-terminal",
            "--title=LLM Robot CLI",
            "--wait",
            "--",
            "ros2",
            "run",
            "llm_arm_control",
            "llm_control_cli",
            "--ros-args",
            "-p",
            ["use_sim_time:=", _LAUNCH_CONFIGURATIONS["use_sim_time"]],
            "-p",
            ["command_burst_count:=", _LAUNCH_CONFIGURATIONS["command_burst_count"]],
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            *_declare_launch_arguments(),
            myrobot_simulation,
            retime_server_launch,
            yolo_obb,
            OpaqueFunction(function=_graspnet_inference_process),
            pose_monitor_node,
            task_server_node,
            cli_terminal,
        ]
    )
