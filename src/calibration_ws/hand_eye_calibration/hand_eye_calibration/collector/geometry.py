"""Geometry helpers: TransformMatrix, CandidatePose, CollectorGeometry.

本模块提供手眼标定采集过程中所需的几何工具：
- TransformMatrix：统一封装旋转+平移的 4x4 齐次矩阵
- CandidatePose：描述一个候选采集位姿（包含位姿、描述、族类等）
- CollectorGeometry：坐标系配置、位姿组合、候选列表生成
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from scipy.spatial.transform import Rotation as R

from .sample_types import CandidateSpec


@dataclass
class TransformMatrix:
    """
    统一表示刚性变换的数据类。
    - rotation: scipy.spatial.transform.Rotation 对象
    - translation: (x, y, z) 元组，单位米
    提供 matrix() 方法返回 4x4 齐次变换矩阵。
    """
    rotation: R
    translation: Tuple[float, float, float]

    def matrix(self):
        """构建并返回 4x4 齐次变换矩阵（numpy 数组）。"""
        m = np.eye(4)
        m[:3, :3] = self.rotation.as_matrix()
        m[:3, 3] = self.translation
        return m


@dataclass
class CandidatePose:
    """
    候选采集位姿的完整描述。
    - idx: 候选序号
    - description: 人类可读的描述字符串（用于日志）
    - pose: ROS PoseStamped 消息，可直接发送给 MoveIt2
    - base_T_ee: 基座到末端执行器的变换（TransformMatrix）
    - family: 候选族类（如 "translation", "orientation"）
    - removable: 该候选是否可以后续被移除
    - spec: 生成该候选的原始规范（可选）
    """
    idx: int
    description: str
    pose: PoseStamped
    base_T_ee: TransformMatrix
    family: str
    removable: bool
    spec: Optional[CandidateSpec] = None


class CollectorGeometry:
    """
    采集器几何管理器。
    负责：
    - 记录基础坐标系名称（基座、末端、跟踪基准、跟踪标记）
    - 提供坐标系变换与 ROS 消息的互转工具
    - 根据参考位姿和候选规范列表生成候选采集位姿
    """

    def __init__(
        self,
        *,
        base_frame: str,
        ee_frame: str,
        tracking_base_frame: str,
        tracking_marker_frame: str,
        max_candidate_attempts: int,
    ):
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.tracking_base_frame = tracking_base_frame
        self.tracking_marker_frame = tracking_marker_frame
        self.max_candidate_attempts = int(max_candidate_attempts)

    @staticmethod
    def tf_to_matrix(transform) -> TransformMatrix:
        """
        将 ROS tf2 变换消息（geometry_msgs/TransformStamped 中的 transform 字段）
        转换为 TransformMatrix。
        """
        q = transform.transform.rotation
        p = transform.transform.translation
        return TransformMatrix(
            rotation=R.from_quat([q.x, q.y, q.z, q.w]),
            translation=(float(p.x), float(p.y), float(p.z)),
        )

    @staticmethod
    def transform_to_matrix(transform) -> TransformMatrix:
        """
        将 ROS geometry_msgs/Transform 消息转换为 TransformMatrix。
        """
        q = transform.rotation
        p = transform.translation
        return TransformMatrix(
            rotation=R.from_quat([q.x, q.y, q.z, q.w]),
            translation=(float(p.x), float(p.y), float(p.z)),
        )

    @staticmethod
    def transform_from_xyz_rpy(xyz, rpy_deg) -> TransformMatrix:
        """
        根据 XYZ 平移（米）和 RPY 欧拉角（度）构造 TransformMatrix。
        旋转顺序为 'xyz'（外旋）。
        """
        return TransformMatrix(
            rotation=R.from_euler("xyz", [float(v) for v in rpy_deg], degrees=True),
            translation=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        )

    @staticmethod
    def matrix_to_pose_stamped(transform: TransformMatrix, frame_id: str, stamp) -> PoseStamped:
        """
        将 TransformMatrix 转换为 ROS PoseStamped 消息，
        指定坐标系 ID 和时间戳。
        """
        q = transform.rotation.as_quat()  # 返回 [x, y, z, w]
        pose = Pose()
        pose.position = Point(
            x=float(transform.translation[0]),
            y=float(transform.translation[1]),
            z=float(transform.translation[2]),
        )
        pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = stamp
        ps.pose = pose
        return ps

    @staticmethod
    def compose(a: TransformMatrix, b: TransformMatrix) -> TransformMatrix:
        """
        组合两个变换：先应用 a，再应用 b（矩阵乘法 a.matrix @ b.matrix）。
        返回新的 TransformMatrix。
        """
        return CollectorGeometry.from_matrix(a.matrix() @ b.matrix())

    @staticmethod
    def from_matrix(m) -> TransformMatrix:
        """
        从 4x4 齐次变换矩阵（numpy 数组）构造 TransformMatrix。
        """
        return TransformMatrix(
            rotation=R.from_matrix(m[:3, :3]),
            translation=(float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
        )

    @staticmethod
    def rotation_delta_deg(a: R, b: R) -> float:
        """
        计算两个旋转之间的角距离（度）。
        使用 a.inv() * b 得到相对旋转，取其旋转角度。
        """
        return math.degrees(float((a.inv() * b).magnitude()))

    def build_visibility_candidates(
        self,
        *,
        reference_base_T_ee: TransformMatrix,
        candidate_specs: List[CandidateSpec],
        workspace_status: Callable[[Tuple[float, float, float]], Tuple[bool, str]],
        now_msg: Callable[[], object],
    ) -> List[CandidatePose]:
        """
        根据参考位姿和候选规范列表，生成一系列候选采集位姿。

        参数：
        - reference_base_T_ee: 参考的基座到末端变换
        - candidate_specs: 候选规范列表，每个规范定义相对于参考的偏移和姿态旋转
        - workspace_status: 工作空间检查回调，输入 (x,y,z) 返回 (是否可用, 原因字符串)
        - now_msg: 返回当前 ROS 时间戳的可调用对象（用于 PoseStamped 的时间戳）

        返回：
        - 通过工作空间检查的 CandidatePose 列表，数量不超过 max_candidate_attempts
        """
        candidates = []
        for spec in candidate_specs:
            # 计算目标平移：参考平移 + 规范中的基座偏移
            desired_translation = (
                float(reference_base_T_ee.translation[0] + spec.base_x),
                float(reference_base_T_ee.translation[1] + spec.base_y),
                float(reference_base_T_ee.translation[2] + spec.base_z),
            )
            # 计算目标旋转：参考旋转 * 规范定义的局部姿态（pitch, yaw, roll，度数）
            desired_rotation = reference_base_T_ee.rotation * R.from_euler(
                "xyz",
                [spec.pitch, spec.yaw, spec.roll],
                degrees=True,
            )
            desired_base_T_ee = TransformMatrix(
                rotation=desired_rotation,
                translation=desired_translation,
            )
            # 检查工作空间有效性
            workspace_ok, workspace_note = workspace_status(desired_base_T_ee.translation)
            if not workspace_ok:
                continue

            idx = len(candidates) + 1
            candidates.append(
                CandidatePose(
                    idx=idx,
                    description=(
                        f"{spec.family} {spec.source} base_offset x={spec.base_x:+.3f}m y={spec.base_y:+.3f}m "
                        f"z={spec.base_z:+.3f}m pitch={spec.pitch:+.1f}deg "
                        f"yaw={spec.yaw:+.1f}deg roll={spec.roll:+.1f}deg"
                    ),
                    pose=self.matrix_to_pose_stamped(desired_base_T_ee, self.base_frame, now_msg()),
                    base_T_ee=desired_base_T_ee,
                    family=spec.family,
                    removable=spec.removable,
                    spec=spec,
                )
            )
            # 达到最大尝试数量则提前终止
            if len(candidates) >= self.max_candidate_attempts:
                break
        return candidates