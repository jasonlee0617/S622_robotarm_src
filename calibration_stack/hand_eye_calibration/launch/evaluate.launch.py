import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# 将当前启动文件所在目录加入 sys.path，以便导入同级辅助模块
_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)

from handeye_launch_utils import camera_launch as _camera_launch
from handeye_launch_utils import default_from_settings as _default_from_settings
from handeye_launch_utils import load_handeye_profile as _load_profile
from handeye_launch_utils import profile_value as _profile_value
from handeye_launch_utils import value as _value


def _launch_setup(context, *args, **kwargs):
    """
    验证模式的主启动逻辑。
    加载已保存的标定结果，并通过手眼 TF 发布节点、ArUco 检测、
    easy_handeye2 评估工具及 MoveIt+Rviz 来直观验证标定精度。
    """
    calibration_type = _value(context, "calibration_type")
    camera_type = _value(context, "camera_type")
    use_rviz = _value(context, "use_rviz").lower()
    # 加载对应标定类型的配置文件（包含坐标系、标记等参数）
    profile = _load_profile(calibration_type)

    calibration_name = _profile_value(context, profile, "calibration_name")
    camera_link_frame = _profile_value(context, profile, "camera_link_frame")
    publish_child_frame = _profile_value(context, profile, "publish_camera_link_frame")
    tracking_base_frame = _profile_value(context, profile, "tracking_base_frame")
    tracking_marker_frame = _profile_value(context, profile, "tracking_marker_frame")
    marker_id = int(_profile_value(context, profile, "marker_id"))
    # 是否使用 tracking -> camera_link 补偿（默认开启）
    use_compensation = str(profile.get("use_tracking_to_camera_link_compensation", True)).lower()

    # ArUco 识别参数文件
    aruco_params = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "aruco_parameters.yaml",
    )

    # ArUco 标记检测节点（ros2_aruco）
    aruco_recognition_node = Node(
        package="ros2_aruco",
        executable="aruco_node",
        parameters=[aruco_params],
        output="screen",
    )

    # 标定用 ArUco 发布节点：根据 TF 发布指定标记位姿，供可视化或调试
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

    # 手眼标定 TF 发布节点：读取已保存的标定结果，动态发布
    # 末端执行器 -> 相机 的 TF 变换，实现虚拟相机位姿可视化
    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[
            {
                "calibration_name": calibration_name,
                "camera_link_frame": camera_link_frame,
                "publish_child_frame": publish_child_frame,
                "publish_rate_hz": 10.0,
                "use_tracking_to_camera_link_compensation": use_compensation in ("1", "true", "yes"),
            }
        ],
        output="screen",
    )

    # 启动 easy_handeye2 官方评估工具（rqt 界面，可查看标定残差等）
    easy_handeye2_evaluate = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("easy_handeye2"),
                "launch/evaluate.launch.py",
            ])
        ]),
        launch_arguments={
            "name": calibration_name,
        }.items(),
    )

    # 验证专用 RViz 配置文件
    rviz_config_file = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "rviz",
        "validate.rviz",
    )
    # 启动 MoveIt 演示（不含夹爪），使用上述 RViz 配置
    ar_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("fairino_arm_moveit_config"),
                    "launch",
                    "demo.launch.py",
                )
            ]
        ),
        launch_arguments={
            "include_gripper": "False",
            "use_rviz": use_rviz,
            "rviz_config": rviz_config_file,
        }.items(),
    )

    # 返回所有启动项的列表
    return [
        _camera_launch(camera_type),          # 相机驱动
        aruco_recognition_node,               # ArUco 检测
        calibration_aruco_publisher,          # 标定标记发布
        hand_eye_tf_publisher,                # 手眼 TF 发布（加载保存的标定结果）
        easy_handeye2_evaluate,               # easy_handeye2 评估工具
        ar_moveit,                            # MoveIt 演示 + RViz 验证界面
    ]


def generate_launch_description():
    """
    生成验证用启动描述。
    声明可配置参数，并通过 OpaqueFunction 延迟执行启动逻辑。
    """
    # 从 YAML 配置获取默认标定类型和相机类型
    default_calib_type = _default_from_settings("calibration_type", "eye_on_base")
    default_camera_type = _default_from_settings("camera_type", "realsense")

    return LaunchDescription(
        [
            # 标定类型（眼在手外 / 眼在手上）
            DeclareLaunchArgument(
                "calibration_type",
                default_value=default_calib_type,
                choices=["eye_on_base", "eye_in_hand"],
            ),
            # 相机类型（RealSense 或 OAK-D）
            DeclareLaunchArgument(
                "camera_type",
                default_value=default_camera_type,
                choices=["realsense", "oak"],
            ),
            # 标定名称，与保存时一致
            DeclareLaunchArgument(
                "calibration_name",
                default_value="",
            ),
            # 相机 link 坐标系名称（覆盖配置文件）
            DeclareLaunchArgument(
                "camera_link_frame",
                default_value="",
            ),
            # 发布相机 link 时使用的子坐标系名称（覆盖配置文件）
            DeclareLaunchArgument(
                "publish_camera_link_frame",
                default_value="",
            ),
            # 跟踪基准坐标系（相机光心）
            DeclareLaunchArgument(
                "tracking_base_frame",
                default_value="",
            ),
            # 跟踪标记坐标系（ArUco 标记）
            DeclareLaunchArgument(
                "tracking_marker_frame",
                default_value="",
            ),
            # ArUco 标记 ID
            DeclareLaunchArgument(
                "marker_id",
                default_value="",
            ),
            # 是否启动 RViz
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
            ),
            # 将实际启动逻辑包装为 OpaqueFunction，在上下文就绪后执行
            OpaqueFunction(function=_launch_setup),
        ]
    )