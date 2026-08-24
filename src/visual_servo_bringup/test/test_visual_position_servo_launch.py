from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
HARDWARE_LAUNCH = PACKAGE / "launch" / "visual_position_servo.launch.py"
IMAGE_HARDWARE_LAUNCH = PACKAGE / "launch" / "visual_image_servo.launch.py"
GAZEBO_LAUNCH = PACKAGE.parent / "myrobot_simulation" / "launch" / "visual_position_servo_gazebo.launch.py"
IMAGE_GAZEBO_LAUNCH = PACKAGE.parent / "myrobot_simulation" / "launch" / "visual_image_servo_gazebo.launch.py"
SERVO_PARAMETERS = (
    PACKAGE.parent / "myrobot_support" / "fairino_arm_moveit_config" / "config" / "servo_parameters.yaml"
)
HARDWARE_MOVEIT_LAUNCH = (
    PACKAGE.parent / "myrobot_support" / "fairino_arm_moveit_config" / "launch"
    / "moveit_hardware.launch.py"
)
SHARED_RVIZ = PACKAGE / "rviz" / "visual_position_servo.rviz"
IMAGE_RVIZ = PACKAGE / "rviz" / "visual_image_servo.rviz"
GAZEBO_RVIZ = PACKAGE.parent / "myrobot_simulation" / "rviz" / "visual_position_servo_gazebo.rviz"


def test_business_yaml_moveit_layers_share_the_same_client_contract():
    configs = (
        (PACKAGE.parent / "llm_arm_control" / "config" / "llm_robot_control.yaml", "llm_control_task_server", (), True),
        (PACKAGE.parent / "graspnet_ws" / "graspnet_bringup" / "config" / "graspnet_grasping.yaml", "graspnet_visual_grasping", (), True),
        (PACKAGE.parent / "visual_grasping_bringup" / "config" / "visual_grasping.yaml", "visual_grasping", (), True),
        (PACKAGE / "config" / "visual_position_servo.yaml", "visual_servo_grasping", ("nodes",), True),
        (PACKAGE / "config" / "visual_image_servo.yaml", "visual_image_servo", ("nodes",), False),
    )
    for path, node, parents, needs_discrete_planner in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for parent in parents:
            config = config[parent]
        parameters = config[node].get("ros__parameters", config[node])
        moveit = parameters["moveit"]
        assert set(moveit["move_groups"]) == {"fairino", "kdl"}
        assert moveit["ik"]["default"] in {"fairino", "kdl"}
        if needs_discrete_planner:
            assert {"pipeline", "planner", "readiness"} <= set(moveit)
        else:
            assert "pipeline" not in moveit and "planner" not in moveit


def test_hardware_moveit_executes_only_the_yaml_selected_ik_client():
    source = HARDWARE_MOVEIT_LAUNCH.read_text(encoding="utf-8")

    assert 'LaunchConfiguration("execution_ik")' in source
    assert 'LaunchConfiguration("execution_pipeline")' in source
    assert 'DeclareLaunchArgument(\n            "execution_ik"' in source
    assert 'DeclareLaunchArgument(\n            "execution_pipeline"' in source
    assert "_planning_only_parameters" in source


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
    assert "active_executor" not in defaults
    assert 'moveit_args["execution_ik"] = visual_servo_params["ik_plugin"]' in source
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


def test_image_servo_gazebo_launch_uses_marker_scene_without_follower():
    source = IMAGE_GAZEBO_LAUNCH.read_text(encoding="utf-8")
    calibration_source = (
        PACKAGE.parent / "myrobot_simulation" / "launch" / "calibration_gazebo.launch.py"
    ).read_text(encoding="utf-8")
    aruco_params = (
        PACKAGE.parent
        / "calibration_ws"
        / "hand_eye_calibration"
        / "config"
        / "aruco_parameters.yaml"
    ).read_text(encoding="utf-8")

    assert '"calibration_gazebo.launch.py"' in source
    assert 'executable="handeye_publisher.py"' in source
    assert 'executable="visual_image_servo"' in source
    assert '"enable_servo": "true"' in source
    assert '"rviz_config"' in source
    assert '"visual_image_servo.rviz"' in source
    assert '"rviz_config",' in calibration_source
    assert '"rviz_config": LaunchConfiguration("rviz_config")' in calibration_source
    assert "follow_aruco_marker" not in source
    assert "enable_validation_follower" not in source
    assert "aruco_5x5_250_id1" in calibration_source
    assert "DICT_5X5_250" in aruco_params


def test_image_servo_hardware_launch_uses_realsense_handeye_and_moveit_servo():
    source = IMAGE_HARDWARE_LAUNCH.read_text(encoding="utf-8")

    for text in (
        'camera_launch(',
        '"color_profile": "640x480x60"',
        '"depth_profile": "640x480x60"',
        '"align_depth.enable": "true"',
        '"moveit_hardware.launch.py"',
        'executable="servo_node_main"',
        'executable="handeye_publisher.py"',
        'executable="visual_image_servo"',
        '"use_gazebo": False',
        '"visual_image_servo.rviz"',
        'def _node_parameter(context, name):',
        'image_params = {name: _node_parameter(context, name) for name in _IMAGE_SERVO_DEFAULTS}',
        '*(_argument(name, default) for name, default in DEFAULTS.items())',
    ):
        assert text in source
    assert "follow_aruco_marker" not in source
    assert "enable_validation_follower" not in source
    assert "use_gazebo: true" in SERVO_PARAMETERS.read_text(encoding="utf-8")
    config = (PACKAGE / "config" / "visual_image_servo.yaml").read_text(encoding="utf-8")
    assert "      auto_start: true" in config
    assert "profiles:" not in config
    gazebo_source = IMAGE_GAZEBO_LAUNCH.read_text(encoding="utf-8")
    assert 'def _node_parameter(context, name):' in gazebo_source
    assert 'image_params = {name: _node_parameter(context, name) for name in _IMAGE_SERVO_DEFAULTS}' in gazebo_source


def test_image_servo_rviz_is_a_copy_with_visible_interactive_markers():
    position_config = SHARED_RVIZ.read_text(encoding="utf-8")
    image_config = IMAGE_RVIZ.read_text(encoding="utf-8")

    assert IMAGE_RVIZ.is_file()
    assert "Interactive Marker Size: 0.25" in position_config
    assert "Interactive Marker Size: 0.25" in image_config
    assert "Move Group Namespace: /move_group_fairino" in position_config
    assert "Move Group Namespace: /move_group_fairino" in image_config
    assert "Planning Scene Topic: /move_group_fairino/monitored_planning_scene" in position_config
    assert "Planning Scene Topic: /move_group_fairino/monitored_planning_scene" in image_config
