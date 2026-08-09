"""视觉抓取 Gazebo 启动参数边界检查."""

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "launch" / "visual_grasping_gazebo.launch.py"
GRASPNET_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "graspnet_visual_grasping_gazebo.launch.py"
)


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
        "yolov8_grasping",
    } <= packages.keys()

    for package in packages:
        dictionaries = _literal_dicts(_keyword(packages[package], "parameters"))
        assert dictionaries
        assert "use_sim_time" in _dict_keys(dictionaries[0])
        assert _has_launch_config_reference(dictionaries[0], "use_sim_time")


def test_visual_grasping_runtime_defaults_are_not_launch_configurations():
    visual = next(
        call
        for call in _node_calls()
        if _keyword(call, "package").value == "yolov8_grasping"
    )
    dictionaries = _literal_dicts(_keyword(visual, "parameters"))
    keys = _dict_keys(dictionaries[0])
    assert {
        "camera_mode",
        "startup_joint_state_name",
        "startup_joint_names",
        "startup_joint_positions",
    } <= keys

    assert _has_launch_config_reference(dictionaries[0], "camera_mode")
    assert _has_launch_config_reference(dictionaries[0], "startup_joint_state_name")
    assert any(
        isinstance(key, ast.Constant)
        and key.value == "startup_joint_names"
        and isinstance(value, ast.Name)
        and value.id == "pos1_joint_names"
        for key, value in zip(dictionaries[0].keys, dictionaries[0].values)
    )
    assert any(
        isinstance(key, ast.Constant)
        and key.value == "startup_joint_positions"
        and isinstance(value, ast.Name)
        and value.id == "pos1_joint_positions"
        for key, value in zip(dictionaries[0].keys, dictionaries[0].values)
    )

    parameters = _keyword(visual, "parameters")
    yaml_source = parameters.elts[-1]
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
        and value.value == "yolo-obb-gazebo-1024.pt"
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
        "startup_joint_state_name",
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
    assert 'gz_share, "rviz", "visual_grasping_table.rviz"' in source
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
    assert 'os.path.join(graspnet_share, "config", "graspnet_visual_grasping.yaml")' in source
    assert 'LaunchConfiguration("graspnet_visual_grasping_config")' not in source
    assert '"robot_profile": "fairino_arm_gripper_handeye"' not in source
    assert "**{name: launch_config[name] for name in _SCENE_ARGUMENT_NAMES}" in source
