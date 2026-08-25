from pathlib import Path


_PACKAGE = Path(__file__).resolve().parents[1]
_LAUNCH = _PACKAGE / "launch"
_MOVEIT_HARDWARE_LAUNCH = (
    _PACKAGE.parent / "myrobot_support_ws" / "fairino_arm_moveit_config" / "launch"
    / "moveit_hardware.launch.py"
)


def test_real_hardware_visual_launches_replace_legacy_names():
    assert (_LAUNCH / "visual_grasping.launch.py").is_file()
    assert (_LAUNCH / "visual_octmap.launch.py").is_file()
    assert not list(_LAUNCH.glob("elongated_object_box_*.launch.py"))


def test_each_entry_directly_declares_the_hardware_rgbd_contract():
    assert not (_LAUNCH / "visual_grasping_common.py").exists()
    for name in ("visual_grasping.launch.py", "visual_octmap.launch.py"):
        source = (_LAUNCH / name).read_text(encoding="utf-8")
        assert "camera_launch(" in source
        assert '"rgb_camera.color_profile"' in source
        assert '"depth_module.depth_profile"' in source
        assert '"align_depth.enable": "true"' in source
        assert '"moveit_hardware.launch.py"' in source
        assert '"yolo-obb-1280.pt"' in source
        if name == "visual_grasping.launch.py":
            assert '"config/visual_grasping.yaml"' in source
        else:
            assert '"camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info"' in source
        assert "visual_grasping_common" not in source


def test_octmap_entry_uses_four_class_base_chain_and_optional_scene_nodes():
    source = (_LAUNCH / "visual_octmap.launch.py").read_text(encoding="utf-8")
    assert "camera_launch(" in source
    assert "semantic_octomap_cloud_filter.py" in source
    assert "dynamic_collision_objects" in source
    assert '"enable_semantic_cloud_filter": "false"' in source
    assert '"enable_dynamic_collision_objects": "false"' in source


def test_visual_grasping_rviz_is_static_fairino_config_without_temp_rewrite():
    rviz = (_PACKAGE / "rviz" / "visual_grasping.rviz").read_text(encoding="utf-8")
    moveit_launch = _MOVEIT_HARDWARE_LAUNCH.read_text(encoding="utf-8")

    assert "Move Group Namespace: /move_group_fairino" in rviz
    assert "Planning Scene Topic: /move_group_fairino/monitored_planning_scene" in rviz
    assert "ReplaceString" not in moveit_launch
    assert 'arguments=["-d", LaunchConfiguration("rviz_config")]' in moveit_launch


def test_real_moveit_launch_uses_explicit_controller_config_and_unique_servers():
    moveit_launch = _MOVEIT_HARDWARE_LAUNCH.read_text(encoding="utf-8")
    rsp = (_MOVEIT_HARDWARE_LAUNCH.parent / "rsp.launch.py").read_text(encoding="utf-8")
    static_tf = (
        _MOVEIT_HARDWARE_LAUNCH.parent / "static_virtual_joint_tfs.launch.py"
    ).read_text(encoding="utf-8")

    for source in (moveit_launch, rsp, static_tf):
        assert 'file_path="config/moveit_controllers_real.yaml"' in source
    assert '"name": f"{namespace}_server"' in moveit_launch
    assert 'remappings=[("~/robot_description", "/robot_description")]' in moveit_launch
