import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_srdf_group_state(package_name, relative_path, state_name, group_name):
    srdf_path = os.path.join(get_package_share_directory(package_name), relative_path)
    root = ET.parse(srdf_path).getroot()
    for group_state in root.findall("group_state"):
        if group_state.get("name") != state_name or group_state.get("group") != group_name:
            continue
        names = []
        positions = []
        for joint in group_state.findall("joint"):
            names.append(joint.get("name"))
            positions.append(float(joint.get("value")))
        if names and positions:
            return names, positions
    raise RuntimeError(f"Missing SRDF group_state '{state_name}' for group '{group_name}' in {srdf_path}")


def _graspnet_inference_process():
    install_setup = "/home/robot/S622_robotarm/install/setup.bash"
    conda_setup = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    cmd = (
        "set -e; "
        f"source {conda_setup}; "
        "conda activate graspnet; "
        "source /opt/ros/humble/setup.bash; "
        f"source {install_setup}; "
        "export PYTHONUNBUFFERED=1; "
        "export MPLCONFIGDIR=/tmp/graspnet_mpl_config; "
        "export XDG_CACHE_HOME=/tmp/graspnet_xdg_cache; "
        "mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME; "
        "exec python -m graspnet_grasping.graspnet_inference_node "
        "--ros-args "
        "-r __node:=graspnet_inference "
        "-p use_sim_time:=true "
        "-p rgb_topic:=/camera/camera/color/image_raw "
        "-p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw "
        "-p camera_info_topic:=/camera/camera/aligned_depth_to_color/camera_info "
        "-p camera_frame:=camera_color_optical_frame "
        "-p baseline_dir:=/home/robot/manipulator_grasp/graspnet-baseline "
        "-p checkpoint_path:=/home/robot/manipulator_grasp/logs/log_rs/checkpoint-rs.tar "
        "-p num_point:=20000 "
        "-p top_k_publish:=5 "
        "-p min_valid_points:=2000 "
        "-p roi_norm:='[0.20, 0.20, 0.90, 0.85]' "
        "-p auto_once:=false "
        "-p auto_visualize:=false "
        "-p confirm_before_publish:=true "
        "-p confirm_visual_top_k:=50"
    )
    return ExecuteProcess(cmd=["bash", "-lc", cmd], output="screen")


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    graspnet_share = get_package_share_directory("graspnet_grasping")
    graspnet_visual_grasping_config = os.path.join(
        graspnet_share,
        "config",
        "graspnet_visual_grasping.yaml",
    )

    pos1_joint_names, pos1_joint_positions = _load_srdf_group_state(
        "s622_moveit_config",
        "config/s622_moveit_descriptions.srdf",
        "pos1",
        "robot_arm",
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            "robot_profile": "s622_gripper_handeye",
            "world": "visual_grasping_table",
            "rviz_config": os.path.join(gz_share, "rviz", "graspnet_visual_grasping.rviz"),
            "enable_rviz": "true",
            "use_sim_time": "true",
            "publish_frequency": "30.0",
            "enable_camera_model": "true",
            "enable_camera_bridge": "true",
            "enable_servo": "false",
            "camera_fps": "60",
            "camera_image_width": "1024",
            "camera_image_height": "728",
            "spawn_x": "0.0",
            "spawn_y": "0.0",
            "spawn_z": "1.02",
            "controller_spawn_delay": "5.0",
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
        parameters=[{"calibration_name": "robot_calibration"}],
        output="screen",
    )
    graspnet_visual_grasping = Node(
        package="graspnet_grasping",
        executable="graspnet_visual_grasping",
        name="graspnet_visual_grasping",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "startup_joint_state_name": "pos1",
                "startup_joint_names": pos1_joint_names,
                "startup_joint_positions": pos1_joint_positions,
            },
            LaunchConfiguration("graspnet_visual_grasping_config"),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "graspnet_visual_grasping_config",
                default_value=graspnet_visual_grasping_config,
                description="YAML file for the graspnet_visual_grasping executor node.",
            ),
            gazebo_launch,
            retime_server_launch,
            hand_eye_tf_publisher,
            _graspnet_inference_process(),
            graspnet_visual_grasping,
        ]
    )
