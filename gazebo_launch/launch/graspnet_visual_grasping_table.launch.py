import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


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
    grasping_share = get_package_share_directory("yolov8_grasping")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo_yolo.launch.py")),
        launch_arguments={
            "robot_profile": "s622_gripper_handeye",
            "world": "visual_grasping_table",
            "enable_rviz": "true",
            "use_sim_time": "true",
            "enable_servo": "false",
            "camera_fps": "30",
            "camera_image_width": "640",
            "camera_image_height": "480",
            "spawn_x": "0.0",
            "spawn_y": "0.0",
            "spawn_z": "1.02",
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
            os.path.join(grasping_share, "config", "pen_box_moveit.yaml"),
            {
                "use_sim_time": True,
                "approach_distance": 0.10,
                "lift_distance": 0.08,
                "use_pregrasp": False,
                "use_fixed_grasp_z": True,
                "fixed_grasp_z_m": 0.03,
                "max_grasp_candidates": 5,
                "compute_timeout_sec": 600.0,
                "manual_grasp_confirmation": True,
                "graspnet_to_ee_rpy_deg": [0.0, 0.0, 0.0],
                "debug_compare_target_pose": True,
                "debug_target_world_xyz": [0.2, 0.35, 1.05],
                "debug_robot_spawn_xyz": [0.0, 0.0, 1.02],
                "enable_target_gate": True,
                "max_target_xy_error_m": 0.12,
                "max_target_z_error_m": 0.15,
                "home_pose.x": 0.140,
                "home_pose.y": 0.32,
                "home_pose.z": 0.35,
                "home_pose.roll": 0.0,
                "home_pose.pitch": -180.0,
                "home_pose.yaw": 0.0,
            },
        ],
    )

    return LaunchDescription(
        [
            gazebo_launch,
            retime_server_launch,
            hand_eye_tf_publisher,
            _graspnet_inference_process(),
            graspnet_visual_grasping,
        ]
    )
