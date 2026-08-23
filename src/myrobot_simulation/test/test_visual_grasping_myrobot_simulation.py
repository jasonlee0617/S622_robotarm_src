"""视觉抓取 Gazebo 启动参数边界检查."""

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "launch" / "visual_grasping_gazebo.launch.py"
GRASPNET_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "graspnet_grasping_gazebo.launch.py"
)
GRASPNET_SYSTEM_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "graspnet_ws"
    / "graspnet_bringup"
    / "launch"
    / "graspnet_grasping.launch.py"
)
YOLO_SYSTEM_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "yolo_bringup"
    / "launch"
    / "visual_grasping.launch.py"
)
PROFILE_DIR = SOURCE.parents[1] / "config" / "robots"


def _node_calls():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Node"
    ]


def _keyword(call, name):
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _literal_dicts(parameters_node):
    return [
        element
        for element in parameters_node.elts
        if isinstance(element, ast.Dict)
    ]


def _dict_keys(dictionary):
    return {
        key.value
        for key in dictionary.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _has_launch_config_reference(dictionary, name):
    return any(
        isinstance(key, ast.Constant)
        and key.value == name
        and isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and value.value.id == "launch_config"
        for key, value in zip(dictionary.keys, dictionary.values)
    )


def test_nodes_use_central_launch_configurations():
    calls = _node_calls()
    packages = {
        _keyword(call, "package").value: call
        for call in calls
        if isinstance(_keyword(call, "package"), ast.Constant)
    }

    assert {
        "hand_eye_calibration",
        "yolo_perception",
        "yolo_bringup",
    } <= packages.keys()

    for package in ("hand_eye_calibration", "yolo_perception", "yolo_bringup"):
        dictionaries = _literal_dicts(_keyword(packages[package], "parameters"))
        assert dictionaries
        assert "use_sim_time" in _dict_keys(dictionaries[0])
        assert _has_launch_config_reference(dictionaries[0], "use_sim_time")


def test_each_grasp_entry_embeds_one_motion_control_node():
    for source_path in (
        SOURCE,
        GRASPNET_SOURCE,
        GRASPNET_SYSTEM_SOURCE,
        YOLO_SYSTEM_SOURCE,
    ):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        motion_control_nodes = [
            call
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "Node"
            and isinstance(_keyword(call, "package"), ast.Constant)
            and _keyword(call, "package").value == "manipulation_common"
        ]
        assert len(motion_control_nodes) == 1


def test_visual_grasping_runtime_parameters_do_not_include_startup_joint_state():
    visual = next(
        call
        for call in _node_calls()
        if _keyword(call, "package").value == "yolo_bringup"
    )
    dictionaries = _literal_dicts(_keyword(visual, "parameters"))
    keys = _dict_keys(dictionaries[0])
    assert "camera_mode" in keys
    assert not {
        "startup_joint_state_name",
        "startup_joint_names",
        "startup_joint_positions",
    } & keys

    assert _has_launch_config_reference(dictionaries[0], "camera_mode")

    parameters = _keyword(visual, "parameters")
    yaml_source = next(element for element in parameters.elts if isinstance(element, ast.Call))
    assert isinstance(yaml_source, ast.Call)
    assert isinstance(yaml_source.func, ast.Attribute)
    assert yaml_source.func.attr == "join"
    assert any(
        isinstance(argument, ast.Constant)
        and argument.value == "yolo_visual_grasping.yaml"
        for argument in yaml_source.args
    )


def test_yolo_model_path_is_fixed_and_numeric_parameters_are_launch_configurations():
    yolo = next(
        call
        for call in _node_calls()
        if _keyword(call, "package").value == "yolo_perception"
    )
    dictionary = _literal_dicts(_keyword(yolo, "parameters"))[0]
    assert any(
        isinstance(key, ast.Constant)
        and key.value == "model_path"
        and isinstance(value, ast.Constant)
        and value.value == "yolo-obb-1280.pt"
        for key, value in zip(dictionary.keys, dictionary.values)
    )
    assert _has_launch_config_reference(dictionary, "imgsz")
    assert _has_launch_config_reference(dictionary, "conf")


def test_launch_arguments_are_centralized_before_generate_function():
    source = SOURCE.read_text(encoding="utf-8")
    generate_offset = source.index("def generate_launch_description")
    declarations = source[:generate_offset]
    for name in (
        "robot_profile",
        "world",
        "use_sim_time",
        "camera_profile",
        "calibration_name",
        "camera_mode",
        "imgsz",
        "conf",
    ):
        assert f'"{name}"' in declarations
    assert "_LAUNCH_CONFIGURATIONS" in declarations
    assert "_declare_launch_arguments" in declarations


def test_nested_gazebo_values_remain_launch_arguments_and_cli_defaults_exist():
    source = SOURCE.read_text(encoding="utf-8")
    assert 'name: launch_config[name]' in source
    assert '"rviz_config": os.path.join(' in source
    assert 'gz_share, "rviz", "visual_grasping_gazebo.rviz"' in source
    for name in (
        "robot_profile",
        "world",
        "camera_profile",
        "camera_profile_file",
        "camera_fps",
        "camera_image_width",
        "camera_image_height",
        "spawn_z",
        "controller_spawn_delay",
    ):
        assert f'"{name}"' in source


def test_fixed_rviz_and_storage_paths_are_not_launch_configurations():
    source = SOURCE.read_text(encoding="utf-8")
    assert (
        '"storage_directory": '
        '"/home/robot/fairino_robotarm/src/calibration_ws/'
        'hand_eye_calibration/calib/sim"'
    ) in source
    assert '"storage_directory": launch_config["storage_directory"]' not in source
    assert '"rviz_config": LaunchConfiguration("rviz_config")' not in source


def test_graspnet_entry_uses_the_same_parameter_boundary():
    source = GRASPNET_SOURCE.read_text(encoding="utf-8")
    generate_offset = source.index("def generate_launch_description")
    assert "_LAUNCH_ARGUMENT_SPECS" in source[:generate_offset]
    assert "_LAUNCH_CONFIGURATIONS" in source[:generate_offset]
    assert 'os.path.join(graspnet_share, "config", "graspnet_grasping.yaml")' in source
    assert 'LaunchConfiguration("graspnet_visual_grasping_config")' not in source
    assert '("robot_profile", "fairino_arm_gripper_handeye"' in source
    assert "**{name: launch_config[name] for name in _SCENE_ARGUMENT_NAMES}" in source
    assert '"-p top_k_publish:=50 "' not in source
    assert "min_grasp_z" not in source
    assert "support_plane_filter" not in source
    assert "startup_joint" not in source
    assert "_load_srdf_group_state" not in source


def test_robot_profile_names_match_their_camera_layouts():
    expected = {
        "fairino_arm_gripper": "fairino_arm_gazebo.urdf.xacro",
        "fairino_arm_gripper_handeye": "fairino_arm_handeye_gazebo.urdf.xacro",
        "fairino_arm_gripper_eye_on_base": "fairino_arm_eye_on_base_gazebo.urdf.xacro",
    }
    for profile_name, xacro_name in expected.items():
        profile = PROFILE_DIR / f"{profile_name}.yaml"
        assert profile.is_file()
        assert xacro_name in profile.read_text(encoding="utf-8")



def test_real_graspnet_entry_matches_the_hardware_rgbd_and_moveit_contract():
    source = GRASPNET_SYSTEM_SOURCE.read_text(encoding="utf-8")

    for text in (
        "camera_launch(",
        '"camera_type": "realsense"',
        '"camera_serial_no": ""',
        '"rgb_camera.color_profile"',
        '"depth_module.depth_profile"',
        '"align_depth.enable": "true"',
        '"moveit_hardware.launch.py"',
        '"handeye_publisher.py"',
        '"retime_server.launch.py"',
        '"graspnet_grasping.rviz"',
        '"storage_directory": str(Path.home() / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real")',
        "TimerAction(period=3.0",
        "TimerAction(period=8.0",
    ):
        assert text in source


def test_graspnet_real_and_gazebo_entries_share_the_one_yaml_and_rviz_is_packaged():
    real_source = GRASPNET_SYSTEM_SOURCE.read_text(encoding="utf-8")
    gazebo_source = GRASPNET_SOURCE.read_text(encoding="utf-8")
    package_root = GRASPNET_SYSTEM_SOURCE.parents[1]

    assert '"config", "graspnet_grasping.yaml"' in real_source
    assert '"config", "graspnet_grasping.yaml"' in gazebo_source
    assert (package_root / "rviz" / "graspnet_grasping.rviz").is_file()
    assert "glob('rviz/*.rviz')" in (package_root / "setup.py").read_text(encoding="utf-8")


def test_real_graspnet_launch_dependencies_are_declared():
    package_xml = GRASPNET_SYSTEM_SOURCE.parents[1] / "package.xml"
    source = package_xml.read_text(encoding="utf-8")

    for dependency in (
        "depthai_ros_driver",
        "fairino_arm_moveit_config",
        "hand_eye_calibration",
        "trajectory_retime_server",
    ):
        assert f"<exec_depend>{dependency}</exec_depend>" in source


def test_graspnet_inference_launch_callbacks_return_action_lists():
    for source_path in (GRASPNET_SOURCE, GRASPNET_SYSTEM_SOURCE):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        callback = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_graspnet_inference_process"
        )
        return_node = next(
            node for node in ast.walk(callback) if isinstance(node, ast.Return)
        )
        assert isinstance(return_node.value, ast.List)
        assert len(return_node.value.elts) == 1
        action = return_node.value.elts[0]
        assert isinstance(action, ast.Call)
        assert isinstance(action.func, ast.Name)
        assert action.func.id == "ExecuteProcess"
