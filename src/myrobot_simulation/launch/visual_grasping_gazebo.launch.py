import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# 所有可配置的标量参数都在这里声明，使 launch 接口一目了然。
# YAML 文件和模型路径则有意作为固定的文件/模型引用，放在下方代码中。
_LAUNCH_ARGUMENT_SPECS = (
    ("robot_profile", "fairino_arm_gripper_handeye", "Gazebo 机器人配置；选择机器人模型与传感器。", None),
    ("world", "visual_grasping_table", "Gazebo 世界资源。", None),
    ("enable_rviz", "true", "是否启用 Gazebo 的 RViz 实例。", None),
    ("use_sim_time", "true", "是否让所有运行时节点使用 Gazebo 的 /clock 仿真时间。", None),
    ("publish_frequency", "30.0", "机器人状态发布频率，单位 Hz。", None),
    ("enable_camera_model", "true", "是否生成仿真相机模型。", None),
    ("enable_camera_bridge", "true", "是否将相机话题桥接到 ROS 2。", None),
    ("enable_servo", "true", "是否在 Gazebo 中启动视觉伺服节点。", None),
    ("camera_fps", "30", "仿真相机帧率。", None),
    ("camera_image_width", "1280", "仿真彩色图像宽度。", None),
    ("camera_image_height", "720", "仿真彩色图像高度。", None),
    (
        "camera_profile",
        "d435_color_1280x720x30_depth_848x480x30",
        "用于视觉抓取仿真相机的命名 D435 配置文件。",
        None,
    ),
    (
        "camera_profile_file",
        "",
        "外部 D435 配置文件 YAML；使用时需要将 camera_profile 设为空字符串。",
        None,
    ),
    ("camera_noise_mode", "off", "相机噪声模型。", ("off", "d435_empirical")),
    ("camera_depth_far_m", "3.0", "D435 深度远裁剪距离（米），最大可设为 10.0。", None),
    ("spawn_z", "1.02", "机器人初始高度，单位米。", None),
    ("controller_spawn_delay", "5.0", "控制器启动前的等待时间，单位秒。", None),
    ("calibration_name", "robot_calibration", "手眼标定名称。", None),
    ("camera_mode", "eye_in_hand", "视觉抓取相机模式。", None),
    ("startup_joint_state_name", "pos1", "SRDF 中定义的启动关节状态名称。", None),
    ("imgsz", "1024", "YOLO 推理图像尺寸。", None),
    ("conf", "0.5", "YOLO 置信度阈值。", None),
)
# 前 17 个参数属于场景/仿真相关参数，会传递给被包含的 Gazebo launch 文件
_SCENE_ARGUMENT_NAMES = tuple(name for name, *_ in _LAUNCH_ARGUMENT_SPECS[:17])
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name, *_ in _LAUNCH_ARGUMENT_SPECS
}


def _declare_launch_arguments():
    """根据规格表创建所有参数的声明."""
    declarations = []
    for name, default_value, description, choices in _LAUNCH_ARGUMENT_SPECS:
        kwargs = {"default_value": default_value, "description": description}
        if choices is not None:
            kwargs["choices"] = list(choices)
        declarations.append(DeclareLaunchArgument(name, **kwargs))
    return declarations


def _load_srdf_group_state(package_name, relative_path, state_name, group_name):
    """从 SRDF 文件中加载指定关节组的状态（关节名和位置列表）."""
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
    raise RuntimeError(
        f"在 {srdf_path} 中未找到 group='{group_name}' 的 group_state '{state_name}'"
    )


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")
    grasping_share = get_package_share_directory("yolov8_grasping")
    pos1_joint_names, pos1_joint_positions = _load_srdf_group_state(
        "fairino_arm_moveit_config",
        "config/fairino_arm_moveit_descriptions.srdf",
        "pos1",
        "robot_arm",
    )

    launch_config = _LAUNCH_CONFIGURATIONS

    # 包含的子 launch 文件参数均使用 LaunchConfiguration 传递，
    # 从而保证命令行覆盖（例如 --ros-args -p use_sim_time:=false）仍然有效。
    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            **{
                name: launch_config[name]
                for name in _SCENE_ARGUMENT_NAMES
            },
            "rviz_config": os.path.join(
                gz_share, "rviz", "visual_grasping_table.rviz"
            ),
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
        parameters=[
            {
                "use_sim_time": launch_config["use_sim_time"],
                "calibration_name": launch_config["calibration_name"],
                "storage_directory": "/home/robot/fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim",
            },
        ],
        output="screen",
    )
    yolo_obb = Node(
        package="yolo_perception",
        executable="yolo_detector_obb.py",
        name="yolo_obb_detector",
        output="screen",
        parameters=[
            {
                "use_sim_time": launch_config["use_sim_time"],
                "model_path": "yolo-obb-gazebo-1024.pt",
                "imgsz": launch_config["imgsz"],
                "conf": launch_config["conf"],
            },
        ],
    )
    visual_grasping = Node(
        package="yolov8_grasping",
        executable="visual_grasping",
        name="visual_grasping",
        output="screen",
        parameters=[
            {
                "use_sim_time": launch_config["use_sim_time"],
                "camera_mode": launch_config["camera_mode"],
                "startup_joint_state_name": launch_config["startup_joint_state_name"],
                "startup_joint_names": pos1_joint_names,
                "startup_joint_positions": pos1_joint_positions,
            },
            os.path.join(grasping_share, "config", "yolo_visual_grasping.yaml"),
        ],
    )
    return LaunchDescription(
        [
            *_declare_launch_arguments(),
            myrobot_simulation,
            retime_server_launch,
            hand_eye_tf_publisher,
            yolo_obb,
            visual_grasping,
        ]
    )
