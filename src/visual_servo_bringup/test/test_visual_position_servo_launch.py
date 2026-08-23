from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
HARDWARE_LAUNCH = PACKAGE / "launch" / "visual_position_servo.launch.py"
GAZEBO_LAUNCH = PACKAGE.parent / "myrobot_simulation" / "launch" / "visual_position_servo_gazebo.launch.py"
SHARED_RVIZ = PACKAGE / "rviz" / "visual_position_servo.rviz"
GAZEBO_RVIZ = PACKAGE.parent / "myrobot_simulation" / "rviz" / "visual_position_servo_gazebo.rviz"


def test_hardware_position_servo_uses_the_visual_grasping_hardware_contract():
    source = HARDWARE_LAUNCH.read_text(encoding="utf-8")

    for text in (
        'camera_launch(',
        '"realsense",',
        '"color_profile": "640x480x60"',
        '"depth_profile": "640x480x60"',
        '"rgb_camera.color_profile": value(context, "color_profile")',
        '"depth_module.depth_profile": value(context, "depth_profile")',
        '"align_depth.enable": "true"',
        '"moveit_hardware.launch.py"',
        '"handeye_publisher.py"',
        '"calibration_name": "robot_calibration"',
        '"yolo_kalman_detector_obb.py"',
        'executable="visual_servo_grasping"',
    ):
        assert text in source
    assert '"demo.launch.py"' not in source
    assert "dual_move_group_nodes" not in source


def test_hardware_position_servo_centralizes_runtime_launch_arguments():
    source = HARDWARE_LAUNCH.read_text(encoding="utf-8")
    defaults = source.split("DEFAULTS = {", 1)[1].split("DESCRIPTIONS = {", 1)[0]

    for name in (
        "use_sim_time",
        "camera_serial_no",
        "color_profile",
        "depth_profile",
        "pointcloud_enable",
        "use_rviz",
        "rviz_config",
        "active_executor",
        "debug",
        "allow_trajectory_execution",
        "publish_monitored_planning_scene",
        "monitor_dynamics",
        "capabilities",
        "disable_capabilities",
        "publish_frequency",
    ):
        assert f'"{name}":' in defaults

    assert "camera_type" not in defaults
    assert "*(_argument(name, default) for name, default in DEFAULTS.items())" in source
    assert "OpaqueFunction(function=_launch_setup)" in source
    assert "name: value(context, name)" in source


def test_position_servo_entries_share_one_rviz_file():
    source = GAZEBO_LAUNCH.read_text(encoding="utf-8")

    assert SHARED_RVIZ.is_file()
    assert not GAZEBO_RVIZ.exists()
    assert 'get_package_share_directory("visual_servo_bringup")' in source
    assert '"visual_position_servo.rviz"' in source
    assert "vision_velocity" + "_evaluator" not in source
