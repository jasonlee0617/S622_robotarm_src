from pathlib import Path
import re
from xml.etree import ElementTree

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
POSITION_SERVO_NODE = (
    PACKAGE / "visual_servo_bringup" / "nodes" / "visual_position_servo_node.py"
)
HARDWARE_LAUNCH = PACKAGE / "launch" / "visual_position_servo.launch.py"
IMAGE_HARDWARE_LAUNCH = PACKAGE / "launch" / "visual_image_servo.launch.py"
GAZEBO_LAUNCH = PACKAGE.parent / "myrobot_simulation" / "launch" / "visual_position_servo_gazebo.launch.py"
ROBOTARM_WORLD = PACKAGE.parent / "myrobot_simulation" / "worlds" / "robotarm_world.sdf"
ARUCO_MODEL = (
    PACKAGE.parent / "myrobot_simulation" / "worlds" / "models" / "aruco_5x5_250_id1" / "model.sdf"
)
ARUCO_NODE = (
    PACKAGE.parent / "calibration_ws" / "ros2_aruco" / "ros2_aruco" / "ros2_aruco" / "aruco_node.py"
)
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
    assert '"open_gripper_after_home":' in defaults
    assert '"open_gripper_after_home": _bool_launch_value(' in source


def test_position_servo_open_gripper_is_shared_and_launch_overridable():
    source = GAZEBO_LAUNCH.read_text(encoding="utf-8")
    config = yaml.safe_load((PACKAGE / "config" / "visual_position_servo.yaml").read_text(encoding="utf-8"))

    assert "open_gripper_after_home" not in config["gazebo"]
    assert config["nodes"]["visual_servo_grasping"]["task"]["open_gripper_after_home"] is False
    assert '"open_gripper_after_home"' in source
    assert "gazebo_open_gripper_after_home" not in source
    assert "_GAZEBO_OPEN_GRIPPER_AFTER_HOME" not in source
    assert '"open_gripper_after_home": open_gripper_after_home' in source
    assert '("robot_profile", "fairino_arm_gripper_onbase"' in source
    assert '"fairino_arm_gripper_inhand"' in source
    assert '"fairino3_v6"' in source
    assert 'target_motion_controller_node.py' in source
    assert 'aruco_marker_pose_publisher.py' in source


def test_gazebo_aruco_target_is_predefined_in_the_world_not_spawned_by_launch():
    launch_source = GAZEBO_LAUNCH.read_text(encoding="utf-8")
    world_source = ROBOTARM_WORLD.read_text(encoding="utf-8")
    model_source = ARUCO_MODEL.read_text(encoding="utf-8")

    assert 'aruco_5x5_250_id1_dynamic' not in launch_source
    assert "ros_gz_sim', executable='create'" not in launch_source
    assert '<uri>model://aruco_5x5_250_id1</uri>' in world_source
    assert '<name>aruco_marker_model</name>' in world_source
    assert re.search(
        r"<pose>0\.2 0\.45 [-+]?\d+(?:\.\d+)? 0\.0 0\.0 0\.0</pose>",
        world_source,
    )
    assert '<static>false</static>' in model_source
    assert '<self_collide>false</self_collide>' in model_source
    assert '<pose>0 0 0 1.5708 0 0</pose>' in model_source
    assert '<gravity>false</gravity>' in model_source
    assert '<allow_auto_disable>false</allow_auto_disable>' in model_source
    assert '<mu>0.08</mu>' in model_source
    assert 'ignition::gazebo::systems::VelocityControl' in model_source
    assert '<topic>/model/aruco_marker_model/cmd_vel</topic>' in model_source


def test_position_servo_yaml_has_exclusive_aruco_source_and_target_rpy():
    config = yaml.safe_load((PACKAGE / "config" / "visual_position_servo.yaml").read_text(encoding="utf-8"))
    node = config["nodes"]["visual_servo_grasping"]

    assert node["perception"]["active_source"] in {"yolo_kalman", "aruco"}
    assert node["perception"]["aruco"]["marker_id"] == 1
    assert node["perception"]["aruco"]["visualization_image_topic"] == "/aruco_marker/visualization"
    assert node["perception"]["aruco"]["visualization_marker_id"] == 1
    assert node["perception"]["aruco"]["prediction_hold_sec"] == 0.25
    assert node["planning"]["target_above_rpy_deg"] == [0.0, -180.0, 0.0]


def test_aruco_overlay_reuses_the_detection_and_rviz_displays_it():
    source = ARUCO_NODE.read_text(encoding="utf-8")
    rviz = SHARED_RVIZ.read_text(encoding="utf-8")

    assert source.count("cv2.aruco.detectMarkers(") == 1
    assert 'value="/aruco_marker/visualization"' in source
    assert 'name="visualization_marker_id"' in source
    assert "self.visualization_pub.get_subscription_count() == 0" in source
    assert "cv2.circle(image, center, 4, (0, 255, 0), -1)" in source
    assert "output.header = img_msg.header" in source
    assert "Name: Aruco Center Overlay" in rviz
    assert "Value: /aruco_marker/visualization" in rviz
    assert "Reliability Policy: Best Effort" in rviz


def test_position_servo_uses_the_visual_position_node_instance_name():
    for path in (POSITION_SERVO_NODE, HARDWARE_LAUNCH, GAZEBO_LAUNCH):
        source = path.read_text(encoding="utf-8")
        assert "visual_position_servo_node" in source
        assert "visual_servo_grasping" + "_node" not in source


def test_position_servo_entries_share_one_rviz_file():
    source = GAZEBO_LAUNCH.read_text(encoding="utf-8")

    assert SHARED_RVIZ.is_file()
    assert not GAZEBO_RVIZ.exists()
    assert 'get_package_share_directory("visual_servo_bringup")' in source
    assert '"visual_position_servo.rviz"' in source
    assert "vision_velocity" + "_evaluator" not in source


def test_position_servo_uses_a_shared_6d_home_pose():
    source = POSITION_SERVO_NODE.read_text(encoding="utf-8")
    config = yaml.safe_load((PACKAGE / "config" / "visual_position_servo.yaml").read_text(encoding="utf-8"))
    task = config["nodes"]["visual_servo_grasping"]["task"]

    assert task["home_pose"] == {
        "xyz": [-0.230756564, 0.273052088, 0.231534975],
        "rpy_deg": [-174.728833, -3.765975, 25.653215],
    }
    assert "home_joints" not in task
    assert 'pose_values("home_pose.xyz", [0.0, 0.0, 0.0])' in source
    assert 'pose_values("home_pose.rpy_deg", [0.0, 0.0, 0.0])' in source
    assert "self.home_pose = self.pose_tools.make_pose(*home_xyz, *home_rpy_deg)" in source
    assert "self.motion.move_to_pose(\n                self.home_pose," in source
    assert "self.motion.move_to_joints(\n                self.home_joints," not in source
    assert 'param_b(self, "open_gripper_after_home", False)' in source


def test_robotarm_world_is_valid_xml_with_cube_as_the_active_target():
    ElementTree.parse(ROBOTARM_WORLD)
    world = ROBOTARM_WORLD.read_text(encoding="utf-8")
    assert '<name>cube_model</name>' in world


def test_image_servo_gazebo_entry_is_removed():
    assert not (
        PACKAGE.parent
        / "myrobot_simulation"
        / "launch"
        / ("visual_image_servo" + "_gazebo.launch.py")
    ).exists()


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
