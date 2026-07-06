import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# 获取当前启动文件所在目录，并添加到 sys.path，以便导入同级目录下的辅助模块
_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)

# 从同级 handeye_launch_utils 模块中导入辅助函数
from handeye_launch_utils import camera_launch as _camera_launch
from handeye_launch_utils import default_from_settings as _default_from_settings
from handeye_launch_utils import load_handeye_profile as _load_profile
from handeye_launch_utils import profile_value as _profile_value
from handeye_launch_utils import value as _value


def _launch_setup(context, *args, **kwargs):
    """
    实际组装所有启动项的 OpaqueFunction 回调。
    在此可以访问 LaunchConfiguration 的运行时值，并根据参数动态生成启动描述列表。
    """
    # 获取启动参数值
    calibration_type = _value(context, "calibration_type")
    camera_type = _value(context, "camera_type")
    use_rviz = _value(context, "use_rviz").lower()

    # 加载对应类型的标定配置文件
    profile = _load_profile(calibration_type)

    # 从启动参数或配置文件中读取各个坐标系参数
    calibration_name = _profile_value(context, profile, "calibration_name")
    robot_base_frame = _profile_value(context, profile, "robot_base_frame")
    robot_effector_frame = _profile_value(context, profile, "robot_effector_frame")
    tracking_base_frame = _profile_value(context, profile, "tracking_base_frame")
    tracking_marker_frame = _profile_value(context, profile, "tracking_marker_frame")
    marker_id = int(_profile_value(context, profile, "marker_id"))

    # RViz 配置文件路径，可从 profile 中指定，否则使用默认值
    rviz_config_file = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "rviz",
        str(profile.get("rviz_config", "moveit_with_camera.rviz")),
    )

    # 启动 MoveIt 演示启动文件，同时传入是否使用 RViz 以及自定义 RViz 配置
    ar_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("s622_moveit_config"),
                    "launch",
                    "demo.launch.py",
                )
            ]
        ),
        launch_arguments={
            "use_rviz": use_rviz,
            "rviz_config": rviz_config_file,
        }.items(),
    )

    # ArUco 检测节点参数文件路径
    aruco_params = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "aruco_parameters.yaml",
    )
    # 启动 ArUco 标记识别节点（ros2_aruco 包）
    aruco_recognition_node = Node(
        package="ros2_aruco",
        executable="aruco_node",
        parameters=[aruco_params],
        output="screen",
    )

    # 启动标定 ArUco 发布节点，用于根据 TF 发布指定标记的位姿（方便可视化或调试）
    calibration_aruco_publisher = Node(
        package="hand_eye_calibration",
        executable="calibration_aruco_publisher.py",
        name="calibration_aruco_publisher",
        output="screen",
        parameters=[
            {
                "tracking_base_frame": tracking_base_frame,
                "tracking_marker_frame": tracking_marker_frame,
                "marker_id": marker_id,
            }
        ],
    )

    # 启动 easy_handeye2 标定主流程（采样、计算、保存等服务）
    easy_handeye2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("easy_handeye2"),
                    "launch",
                    "calibrate.launch.py",
                )
            ]
        ),
        launch_arguments={
            "name": calibration_name,
            "calibration_type": calibration_type,
            "robot_base_frame": robot_base_frame,
            "robot_effector_frame": robot_effector_frame,
            "tracking_base_frame": tracking_base_frame,
            "tracking_marker_frame": tracking_marker_frame,
        }.items(),
    )

    # 启动 ArUco 可视化节点，用于在 RViz 中显示标记的三维位姿
    aruco_visualize = Node(
        package="hand_eye_calibration",
        executable="visualize_aruco_marker.py",
        name="aruco_pose_estimator",
        output="screen",
    )

    # 返回所有需要启动的 actions/nodes 列表
    return [
        _camera_launch(camera_type),          # 相机驱动
        ar_moveit,                            # MoveIt 演示 + RViz
        aruco_recognition_node,               # ArUco 标记检测
        calibration_aruco_publisher,          # 标定标记发布器
        easy_handeye2,                        # easy_handeye2 标定服务与界面
        aruco_visualize,                      # ArUco 标记可视化
    ]


def generate_launch_description():
    """
    启动描述生成函数（ROS 2 launch 系统入口）。
    声明所有可通过命令行配置的参数，然后通过 OpaqueFunction 延迟执行实际启动逻辑。
    """
    # 从 YAML 配置文件中读取默认标定类型和相机类型
    default_calib_type = _default_from_settings("calibration_type", "eye_on_base")
    default_camera_type = _default_from_settings("camera_type", "realsense")

    return LaunchDescription(
        [
            # 声明标定类型参数（眼在手外 / 眼在手上）
            DeclareLaunchArgument(
                "calibration_type",
                default_value=default_calib_type,
                choices=["eye_on_base", "eye_in_hand"],
                description="Hand-eye calibration mode (default from handeye_profiles.yaml settings).",
            ),
            # 声明相机类型参数
            DeclareLaunchArgument(
                "camera_type",
                default_value=default_camera_type,
                choices=["realsense", "oak"],
                description="Camera type (default from handeye_profiles.yaml settings).",
            ),
            # 以下参数允许在命令行覆盖配置文件中的默认值
            DeclareLaunchArgument("calibration_name", default_value="", description="Override profile calibration name."),
            DeclareLaunchArgument("robot_base_frame", default_value="", description="Override profile robot base frame."),
            DeclareLaunchArgument("robot_effector_frame", default_value="", description="Override profile robot effector frame."),
            DeclareLaunchArgument("tracking_base_frame", default_value="", description="Override profile tracking base frame."),
            DeclareLaunchArgument("tracking_marker_frame", default_value="", description="Override profile tracking marker frame."),
            DeclareLaunchArgument("marker_id", default_value="", description="Override profile ArUco marker id."),
            DeclareLaunchArgument("use_rviz", default_value="true", description="Forwarded to MoveIt demo launch."),
            # OpaqueFunction 在 launch 运行时调用 _launch_setup 并传入上下文
            OpaqueFunction(function=_launch_setup),
        ]
    )