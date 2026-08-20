import os
import shlex
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# 场景与节点标量参数集中在此处。YAML、RViz 和权重等固定资源在使用位置
# 直接通过包共享目录定位，不作为可变的 launch 参数。
_LAUNCH_ARGUMENT_SPECS = (
    ("robot_profile", "fairino_arm_gripper_handeye", "Gazebo 机器人配置。", None),
    ("world", "visual_grasping", "Gazebo 世界资源。", None),
    ("enable_rviz", "true", "是否启动 RViz。", None),
    ("use_sim_time", "true", "是否使用 Gazebo 的 /clock。", None),
    ("model_profile", "rs", "GraspNet 模型权重。", ("rs", "kn")),
    ("publish_frequency", "30.0", "机器人状态发布频率（Hz）。", None),
    ("enable_camera_model", "true", "是否生成仿真相机模型。", None),
    ("enable_camera_bridge", "true", "是否桥接相机话题到 ROS 2。", None),
    ("enable_servo", "false", "是否启动 Gazebo 内置伺服节点。", None),
    ("camera_fps", "30", "仿真相机帧率。", None),
    ("camera_image_width", "1280", "仿真彩色图像宽度。", None),
    ("camera_image_height", "720", "仿真彩色图像高度。", None),
    ("camera_profile", "d435_color_1280x720x30_depth_848x480x30", "命名 D435 配置。", None),
    ("camera_profile_file", "", "外部 D435 配置文件；使用时清空 camera_profile。", None),
    ("camera_noise_mode", "off", "相机噪声模型。", ("off", "d435_empirical")),
    ("camera_depth_far_m", "3.0", "深度远裁剪距离（米）。", None),
    ("spawn_x", "0.0", "机器人初始 X 坐标（米）。", None),
    ("spawn_y", "0.0", "机器人初始 Y 坐标（米）。", None),
    ("spawn_z", "1.02", "机器人初始 Z 坐标（米）。", None),
    ("controller_spawn_delay", "5.0", "控制器启动前等待时间（秒）。", None),
    ("calibration_name", "robot_calibration", "手眼标定名称。", None),
)
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name, *_ in _LAUNCH_ARGUMENT_SPECS
}
_SCENE_ARGUMENT_NAMES = tuple(
    name
    for name, *_ in _LAUNCH_ARGUMENT_SPECS
    if name not in {"model_profile", "calibration_name"}
)


def _declare_launch_arguments():
    declarations = []
    for name, default_value, description, choices in _LAUNCH_ARGUMENT_SPECS:
        kwargs = {"default_value": default_value, "description": description}
        if choices:
            kwargs["choices"] = list(choices)
        declarations.append(DeclareLaunchArgument(name, **kwargs))
    return declarations


def _graspnet_inference_process(context):
    model_profile = _LAUNCH_CONFIGURATIONS["model_profile"].perform(context)
    use_sim_time = _LAUNCH_CONFIGURATIONS["use_sim_time"].perform(context)
    install_setup = str(Path(get_package_prefix("graspnet_bringup")).parent / "setup.bash")
    config_path = os.path.join(
        get_package_share_directory("graspnet_bringup"),
        "config",
        "graspnet_grasping.yaml",
    )
    source_share = get_package_share_directory("graspnet_source")
    baseline_dir = os.path.join(source_share, "graspnet_baseline")
    checkpoint_path = os.path.join(source_share, "models", f"checkpoint-{model_profile}.tar")
    conda_setup = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    command_prefix = (
        "set -e; "
        f"source {shlex.quote(conda_setup)}; "
        "conda activate graspnet; "
        "source /opt/ros/humble/setup.bash; "
        f"source {shlex.quote(install_setup)}; "
        "export PYTHONUNBUFFERED=1; "
        "export MPLCONFIGDIR=/tmp/graspnet_mpl_config; "
        "export XDG_CACHE_HOME=/tmp/graspnet_xdg_cache; "
        "mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME; "
        "exec python -m graspnet_bringup.graspnet_inference_node "
        "--ros-args "
        f"--params-file {shlex.quote(config_path)} "
        "-r __node:=graspnet_inference "
        "-p use_sim_time:="
    )
    command_suffix = (
        " "
        "-p rgb_topic:=/camera/camera/color/image_raw "
        "-p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw "
        "-p camera_info_topic:=/camera/camera/aligned_depth_to_color/camera_info "
        f"-p baseline_dir:={shlex.quote(baseline_dir)} "
        f"-p checkpoint_path:={shlex.quote(checkpoint_path)} "
        "-p num_point:=20000 "
        "-p top_k_publish:=50 "
        "-p min_valid_points:=2000 "
        "-p support_plane_filter:=true "
        "-p confirm_before_publish:=true "
        "-p confirm_visual_top_k:=50"
    )
    return [
        ExecuteProcess(
            cmd=["bash", "-lc", command_prefix + use_sim_time + command_suffix],
            output="screen",
        )
    ]


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")
    graspnet_share = get_package_share_directory("graspnet_bringup")
    launch_config = _LAUNCH_CONFIGURATIONS

    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            **{name: launch_config[name] for name in _SCENE_ARGUMENT_NAMES},
            "rviz_config": os.path.join(gz_share, "rviz", "graspnet_visual_grasping.rviz"),
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
    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[{
            "use_sim_time": launch_config["use_sim_time"],
            "calibration_name": launch_config["calibration_name"],
            "storage_directory": str(Path.home() / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim"),
        }],
        output="screen",
    )
    graspnet_visual_grasping = Node(
        package="graspnet_bringup",
        executable="graspnet_visual_grasping",
        name="graspnet_visual_grasping",
        output="screen",
        parameters=[
            {
                "use_sim_time": launch_config["use_sim_time"],
            },
            os.path.join(graspnet_share, "config", "graspnet_grasping.yaml"),
            {
                # The fixed Gazebo table is 5 mm below base_link. Keep a 10 mm
                # clearance while allowing valid grasps on the 3 cm cube.
                "max_grasp_candidates": 50,
                "min_grasp_z": 0.005,
            },
        ],
    )

    return LaunchDescription(
        [
            *_declare_launch_arguments(),
            myrobot_simulation,
            retime_server_launch,
            hand_eye_tf_publisher,
            OpaqueFunction(function=_graspnet_inference_process),
            graspnet_visual_grasping,
        ]
    )
