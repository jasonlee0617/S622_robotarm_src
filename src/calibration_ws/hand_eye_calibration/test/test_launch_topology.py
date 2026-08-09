import ast
import importlib.util
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace

from launch import LaunchContext
import yaml

LAUNCH_ROOT = Path(__file__).resolve().parents[1] / "launch"
if str(LAUNCH_ROOT) not in sys.path:
    sys.path.insert(0, str(LAUNCH_ROOT))

from handeye_launch_utils import default_from_settings, load_handeye_profile
MOVEIT_DEMO = (
    Path(__file__).resolve().parents[3]
    / "fairino_robot_support" / "fairino_arm_moveit_config" / "launch" / "demo.launch.py"
)
MOVEIT_HARDWARE = MOVEIT_DEMO.with_name("moveit_hardware.launch.py")


def _source(name):
    return (LAUNCH_ROOT / name).read_text(encoding="utf-8")


def _load_moveit_hardware():
    spec = importlib.util.spec_from_file_location("fairino_hardware_launch", MOVEIT_HARDWARE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_calibrate_only_starts_environment_and_vision():
    source = _source("calibrate.launch.py")

    assert "ros2_aruco" in source
    assert "visualize_aruco_marker.py" in source
    assert '"moveit_hardware.launch.py"' in source
    assert "start_demo_moveit" not in source
    assert '"rgb_camera.color_profile"' in source
    assert '"depth_module.depth_profile"' in source
    assert "calibrate.rviz" in source
    assert '"active_executor"' in source
    for argument in (
        "debug",
        "allow_trajectory_execution",
        "publish_monitored_planning_scene",
        "monitor_dynamics",
        "capabilities",
        "disable_capabilities",
        "publish_frequency",
    ):
        assert f'"{argument}"' in source
    assert "easy_handeye2" not in source
    assert "manual_calibration_assistant.py" not in source
    assert "auto_calibration_collector.py" not in source


def test_real_hardware_launch_has_no_warehouse_database_path():
    hardware_source = MOVEIT_HARDWARE.read_text(encoding="utf-8")
    calibrate_source = _source("calibrate.launch.py")

    for name in (
        "warehouse_db.launch.py",
        '"db"',
        "moveit_warehouse_database_path",
        "moveit_warehouse_port",
        "moveit_warehouse_host",
        '"reset"',
    ):
        assert name not in hardware_source
        assert name not in calibrate_source


def test_assisted_launch_only_starts_easy_and_manual_assistant():
    source = _source("assisted_calibration.launch.py")
    parameters = yaml.safe_load(
        (LAUNCH_ROOT.parent / "config" / "manual_calibration_assistant.yaml").read_text(
            encoding="utf-8"
        )
    )["manual_calibration_assistant"]["ros__parameters"]

    assert "easy_handeye2" in source
    assert "manual_calibration_assistant.py" in source
    assert "auto_calibration.launch.py" not in source
    assert 'package="ros2_aruco"' not in source
    assert 'package="realsense2_camera"' not in source
    assert "manual_calibration_assistant.yaml" in source
    assert "auto_calibration_collector.yaml" not in source
    assert "_LAUNCH_DEFAULTS" in source
    assert "_LAUNCH_CONFIGURATIONS" in source
    assert "_TOPOLOGY_PARAMETER_NAMES" in source
    assert "if name not in _TOPOLOGY_PARAMETER_NAMES" in source
    assert "_ASSISTANT_NODE_DEFAULTS," in source
    assert "_ASSISTANT_YAML_PARAMETERS," in source
    assert parameters["calibration_type"] == "eye_on_base"
    assert parameters["use_sim_time"] is True
    assert "calibration_output_directory" in parameters
    assert "ground_truth_check_enabled" not in source


def test_handeye_launches_centralize_defaults_without_copying_profiles():
    for name in (
        "calibrate.launch.py",
        "assisted_calibration.launch.py",
        "evaluate.launch.py",
        "validate.launch.py",
    ):
        source = _source(name)
        assert "_LAUNCH_DEFAULTS" in source
        assert "_LAUNCH_CONFIGURATIONS" in source
        assert "LaunchConfiguration" in source
        assert "handeye_profiles.yaml" not in source

    assisted_source = _source("assisted_calibration.launch.py")
    assert '"calibration_type": calibration_type' in assisted_source
    assert '"storage_directory": storage_directory' in assisted_source
    assert '"calibration_output_directory": storage_directory' in assisted_source


def test_real_profiles_are_builtin_and_eye_on_base_uses_grasp_frame():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "config" / "handeye_profiles.yaml").exists()
    assert default_from_settings("calibration_type", "invalid") == "eye_in_hand"
    assert default_from_settings("camera_type", "invalid") == "realsense"
    for calibration_type in ("eye_in_hand", "eye_on_base"):
        profile = load_handeye_profile(calibration_type)
        assert profile["calibration_type"] == calibration_type
        assert profile["robot_effector_frame"] == "grasp_frame"


def test_native_demo_is_not_the_real_hardware_entrypoint():
    source = MOVEIT_DEMO.read_text(encoding="utf-8")

    assert "generate_demo_launch" in source
    assert "move_group_fairino" not in source
    assert "ros2_control_node" not in source


def test_real_moveit_hardware_has_two_isolated_servers_and_one_controller_stack():
    source = MOVEIT_HARDWARE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    move_group_namespaces = [
        ast.literal_eval(call.args[0])
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_move_group"
        and call.args
    ]
    executables = [
        ast.literal_eval(keyword.value)
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "Node"
        for keyword in call.keywords
        if keyword.arg == "executable" and isinstance(keyword.value, ast.Constant)
    ]

    assert move_group_namespaces.count("move_group_fairino") == 2
    assert move_group_namespaces.count("move_group_kdl") == 2
    assert set(move_group_namespaces) == {"move_group_fairino", "move_group_kdl"}
    assert '"active_executor"' in source
    assert executables.count("ros2_control_node") == 1
    assert executables.count("spawner") == 3
    assert '"allow_trajectory_execution": False' in source
    assert "moveit_controllers_hardware.yaml" in source
    assert "MoveGroupExecuteTrajectoryAction" in source
    assert "GroupAction" in source
    assert "OnProcessExit" in source
    assert 'remappings=[("joint_states", "/joint_states")]' in source
    assert "rviz_remappings" not in source


def test_inactive_moveit_parameters_bind_no_real_controller():
    module = _load_moveit_hardware()
    common = {
        "moveit_controller_manager": "manager",
        "moveit_simple_controller_manager": {"controller_names": ["robot_arm_controller"]},
        "moveit_manage_controllers": True,
        "trajectory_execution": {"allowed_goal_duration_margin": 1.0},
        "robot_description": "robot",
    }
    result = module._planning_only_parameters(common)
    controller = result["moveit_simple_controller_manager"]
    assert result["robot_description"] == "robot"
    assert result["allow_trajectory_execution"] is False
    assert "moveit_manage_controllers" not in result
    assert "trajectory_execution" not in result
    assert result["disable_capabilities"] == (
        "move_group/MoveGroupExecuteTrajectoryAction "
        "move_group/MoveGroupExecuteService"
    )
    assert result["moveit_controller_manager"] == (
        "moveit_simple_controller_manager/MoveItSimpleControllerManager"
    )
    assert controller["controller_names"] == ["__planning_only_controller__"]
    assert "/robot_arm_controller" not in controller
    assert "/hand_controller" not in controller
    assert controller["__planning_only_controller__"] == {
        "type": "FollowJointTrajectory",
        "action_ns": "follow_joint_trajectory",
        "joints": ["__planning_only_joint__"],
        "default": False,
    }

    # launch_ros converts non-empty arrays to typed tuples.  Empty arrays become
    # an untyped (), which rclpy rejects while constructing the node.
    assert controller["controller_names"]
    assert controller["__planning_only_controller__"]["joints"]
    assert common["moveit_simple_controller_manager"]["controller_names"] == [
        "robot_arm_controller"
    ]


def test_rviz_config_uses_selected_absolute_move_group_namespace(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_logs"))
    module = _load_moveit_hardware()
    module.Node = lambda **kwargs: kwargs
    config = SimpleNamespace(
        robot_description={},
        robot_description_semantic={},
        joint_limits={},
    )
    parameters = {
        "robot_description_kinematics": {},
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": {},
    }
    rviz_configs = (
        MOVEIT_HARDWARE.parents[1] / "config" / "moveit.rviz",
        LAUNCH_ROOT.parent / "rviz" / "calibrate.rviz",
    )
    for rviz_config in rviz_configs:
        rviz_source = StringIO(rviz_config.read_text(encoding="utf-8"))
        for namespace in ("move_group_fairino", "move_group_kdl"):
            rviz = module._rviz(config, namespace, parameters)
            configured_rviz = rviz["arguments"][1]
            rendered = StringIO()
            rviz_source.seek(0)
            configured_rviz.replace(
                rviz_source,
                rendered,
                configured_rviz.resolve_replacements(LaunchContext()),
            )
            output = rendered.getvalue()
            assert f"Move Group Namespace: /{namespace}" in output
            assert (
                f"Planning Scene Topic: /{namespace}/monitored_planning_scene"
                in output
            )


def test_real_controller_configuration_uses_root_action_endpoints():
    source = (
        MOVEIT_HARDWARE.parents[1] / "config" / "moveit_controllers_hardware.yaml"
    ).read_text(encoding="utf-8")

    assert "- /robot_arm_controller" in source
    assert "- /hand_controller" in source
    assert "/robot_arm_controller:" in source
    assert "/hand_controller:" in source
