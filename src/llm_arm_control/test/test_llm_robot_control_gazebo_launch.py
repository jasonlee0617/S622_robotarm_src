from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "myrobot_simulation" / "launch" / "llm_robot_control_gazebo.launch.py"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "llm_robot_control.yaml"
GRASPNET_CONFIG = ROOT / "graspnet_ws" / "graspnet_bringup" / "config" / "graspnet_grasping.yaml"
YOLO_CONFIG = ROOT / "visual_grasping_bringup" / "config" / "visual_grasping.yaml"
HARDWARE_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "llm_robot_control.launch.py"


def _effective_params(config, node_name):
    shared = config.get("/**", {}).get("ros__parameters", {})
    node = config[node_name]["ros__parameters"]
    return {**shared, **node}


def test_llm_robot_launch_declares_and_forwards_only_its_robot_profile():
    source = LAUNCH.read_text(encoding="utf-8")

    assert '"robot_profile": "fairino_arm_gripper_inhand"' in source
    assert 'DeclareLaunchArgument("robot_profile"' in source
    assert '"robot_profile": _LAUNCH_CONFIGURATIONS["robot_profile"]' in source
    assert "fairino_arm_gripper_calibration_onbase" in source
    assert "fairino_arm_gripper_inhand" in source


def test_shared_llm_robot_config_has_pregrasp_and_no_fixed_sim_time():
    source = CONFIG.read_text(encoding="utf-8")

    assert "use_sim_time:" not in source
    for line in (
        "pregrasp_pose.x: 0.1",
        "pregrasp_pose.y: 0.35",
        "pregrasp_pose.z: 0.30",
        "pregrasp_pose.roll: 0.0",
        "pregrasp_pose.pitch: -180.0",
        "pregrasp_pose.yaw: 100.0",
    ):
        assert line in source
    assert not (CONFIG.parent / "llm_yolo_task_sim.yaml").exists()
    assert not (CONFIG.parent / "llm_yolo_task_hardware.yaml").exists()


def test_llm_launch_starts_cli_and_new_perception_without_motion_control_node():
    source = LAUNCH.read_text(encoding="utf-8")

    assert '"llm_visual_perception.launch.py"' in source
    assert "ExecuteProcess" in source
    assert '"gnome-terminal"' in source
    assert '"llm_control_cli"' in source
    assert '["use_sim_time:=", _LAUNCH_CONFIGURATIONS["use_sim_time"]]' in source
    assert '["command_burst_count:=", _LAUNCH_CONFIGURATIONS["command_burst_count"]]' in source
    assert 'emulate_tty=True' not in source
    assert 'executable="llm_control_cli"' not in source
    assert 'executable="robot_pose_monitor_node"' in source
    assert 'executable="llm_control_task_server"' in source
    old_monitor = "fairino" + "_pose_monitor"
    old_server = "llm" + "_yolo_task_server"
    assert f'executable="{old_monitor}"' not in source
    assert f'executable="{old_server}"' not in source
    assert 'package="manipulation_common"' not in source
    assert 'executable="motion_control"' not in source


def test_hardware_llm_launch_delays_yolo_task_server_and_cli():
    source = HARDWARE_LAUNCH.read_text(encoding="utf-8")

    assert source.count("period=8.0") == 3
    assert "yolo_obb = TimerAction(" in source
    assert "task = TimerAction(" in source
    assert "cli = TimerAction(" in source
    assert "task_server = TimerAction(" not in source
    assert "cli_terminal = TimerAction(" not in source
    assert 'llm_visual_perception.launch.py' in source
    assert 'executable="llm_control_task_server"' in source
    assert '"llm_control_cli"' in source


def test_llm_launch_uses_the_rviz_file_installed_by_llm_arm_control():
    source = LAUNCH.read_text(encoding="utf-8")

    assert 'llm_arm_share = get_package_share_directory("llm_arm_control")' in source
    assert 'os.path.join(llm_arm_share, "rviz", "llm_robot_control.rviz")' in source


def test_shared_config_uses_new_yolo_topics_and_grasp_profile():
    source = CONFIG.read_text(encoding="utf-8")

    assert "yolo_topic: /yolo/detected_result" in source
    assert "depth_topic: /yolo/detected_result/depth" in source
    assert "descend_to_box: 0.04" in source
    assert "grasp.stone.yaw_offset: -45.0" in source
    assert "arm_max_velocity: 0.20" in source
    assert "arm_max_acceleration: 0.20" in source
    assert "llm_control_task_server:\n" in source
    assert "llm_visual_perception:" in source
    assert "graspnet_visual_grasping:" not in source
    assert "graspnet_inference:" in source


def test_llm_graspnet_inference_filters_match_standalone_config():
    llm = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["graspnet_inference"]["ros__parameters"]
    standalone = _effective_params(
        yaml.safe_load(GRASPNET_CONFIG.read_text(encoding="utf-8")),
        "graspnet_inference",
    )
    common = (
        "rgb_topic",
        "depth_topic",
        "camera_info_topic",
        "poses_topic",
        "scores_topic",
        "metadata_topic",
        "preview_best_pose_topic",
        "preview_best_score_topic",
        "num_point",
        "top_k_publish",
        "min_valid_points",
        "base_frame",
        "depth_min_m",
        "depth_max_m",
        "sync_queue_size",
        "sync_slop_s",
        "random_seed",
        "min_grasp_width_m",
        "max_grasp_width_m",
        "support_plane_distance_m",
        "support_plane_min_inlier_ratio",
        "support_plane_max_tilt_deg",
        "workspace_x_min_m",
        "workspace_x_max_m",
        "workspace_y_min_m",
        "workspace_y_max_m",
        "object_min_height_m",
        "object_max_height_m",
        "collision_detection_enabled",
        "collision_voxel_size_m",
        "collision_approach_distance_m",
        "collision_threshold",
        "confirm_before_publish",
        "confirm_visual_top_k",
        "confirm_window_name",
    )

    assert {key: llm[key] for key in common} == {key: standalone[key] for key in common}
    assert llm["object_min_height_m"] == 0.0
    assert llm["min_grasp_width_m"] == 0.0


def test_llm_task_keeps_graspnet_and_yolo_parameters_separate():
    task = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["llm_control_task_server"]["ros__parameters"]
    standalone = _effective_params(
        yaml.safe_load(GRASPNET_CONFIG.read_text(encoding="utf-8")),
        "graspnet_visual_grasping",
    )

    assert all(key.startswith("graspnet_") for key in task if "graspnet" in key)
    assert task["graspnet_max_candidates"] == standalone["max_grasp_candidates"]
    assert task["graspnet_approach_distance_m"] == standalone["approach_distance_m"]
    assert task["graspnet_grasp_offset_m"] == standalone["grasp_offset_m"]
    assert task["graspnet_lift_distance"] == standalone["lift_distance"]
    assert task["graspnet_to_ee_rpy_deg"] == standalone["graspnet_to_ee_rpy_deg"]
    assert task["graspnet_use_width"] == standalone["use_graspnet_width"]
    assert task["grasp_above"] == 0.04
    assert task["grasp_offset"] == 0.013
    assert task["place_offset"] == 0.08
    assert task["descend_to_box"] == 0.04
    assert "min_grasp_width_m" not in task
    assert "object_min_height_m" not in task


def test_layered_visual_configs_and_node_boundaries():
    graspnet = yaml.safe_load(GRASPNET_CONFIG.read_text(encoding="utf-8"))
    yolo = yaml.safe_load(YOLO_CONFIG.read_text(encoding="utf-8"))

    assert {"/**", "graspnet_visual_grasping", "graspnet_inference"} <= set(graspnet)
    assert {"visual_grasping", "yolo_detector_obb"} <= set(yolo)

    visual = yolo["visual_grasping"]["ros__parameters"]
    detector = yolo["yolo_detector_obb"]["ros__parameters"]
    assert "grasp_above" in visual
    assert "depth_inlier_m" in detector
    assert "grasp_above" not in detector
    assert "depth_inlier_m" not in visual


def test_graspnet_launches_load_static_inference_from_yaml_only():
    launches = (
        ROOT / "graspnet_ws" / "graspnet_bringup" / "launch" / "graspnet_grasping.launch.py",
        ROOT / "myrobot_simulation" / "launch" / "graspnet_grasping_gazebo.launch.py",
    )
    forbidden = (
        "-p rgb_topic:=",
        "-p depth_topic:=",
        "-p camera_info_topic:=",
        "-p num_point:=",
        "-p top_k_publish:=",
        "-p min_valid_points:=",
        "-p confirm_before_publish:=",
        "-p confirm_visual_top_k:=",
    )
    for launch in launches:
        source = launch.read_text(encoding="utf-8")
        assert "graspnet_grasping.yaml" in source
        assert "baseline_dir" in source
        assert "checkpoint_path" in source
        assert all(item not in source for item in forbidden)


def test_yolo_launches_load_layered_config_for_visual_and_detector():
    launches = (
        ROOT / "visual_grasping_bringup" / "launch" / "visual_grasping.launch.py",
        ROOT / "myrobot_simulation" / "launch" / "visual_grasping_gazebo.launch.py",
    )
    for launch in launches:
        source = launch.read_text(encoding="utf-8")
        assert source.count("visual_grasping.yaml") >= 2
        assert 'name="visual_grasping"' in source
        assert 'name="yolo_detector_obb"' in source


def test_dual_mode_launches_share_the_llm_config_without_motion_control_node():
    for launch in (LAUNCH, HARDWARE_LAUNCH):
        source = launch.read_text(encoding="utf-8")
        assert "graspnet_visual_grasping" not in source
        assert "graspnet_inference" in source
        assert "llm_control_cli" in source
        assert "motion_control" not in source


def test_control_nodes_use_the_new_ros_interface_prefix():
    node_root = ROOT / "llm_arm_control" / "llm_arm_control_nodes"
    for name in ("robot_pose_control_server.py", "robot_pose_monitor_node.py", "llm_control_cli.py", "llm_control_task_server.py"):
        assert (node_root / name).exists()
    assert not (node_root / "entry_point.py").exists()
    assert "/llm_control/" in (node_root / "robot_pose_control_server.py").read_text(encoding="utf-8")
    assert "/llm_control/" in (node_root / "robot_pose_monitor_node.py").read_text(encoding="utf-8")
    assert "/llm_control/" in (node_root / "llm_control_cli.py").read_text(encoding="utf-8")
    assert "/llm_control/" in (node_root / "llm_control_task_server.py").read_text(encoding="utf-8")


def test_old_llm_launch_and_resources_are_removed():
    root = ROOT
    old_control = "llm" + "_control_gazebo.launch.py"
    old_yolo_launch = "llm_yolo" + "_control_gazebo.launch.py"
    old_config = "llm_yolo" + "_task.yaml"
    old_rviz = "llm_yolo" + ".rviz"
    assert not (root / "myrobot_simulation" / "launch" / old_control).exists()
    assert not (root / "myrobot_simulation" / "launch" / old_yolo_launch).exists()
    assert not (CONFIG.parent / old_config).exists()
    assert not (CONFIG.parent.parent / "rviz" / old_rviz).exists()
