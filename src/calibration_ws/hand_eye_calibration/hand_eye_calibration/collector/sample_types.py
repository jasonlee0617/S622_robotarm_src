"""Shared types for sample management: families, specs, records, quality.

本模块定义手眼标定采集流程中使用的核心数据结构：
- 候选家族常量（CandidateFamily）
- 候选位姿的基础偏移配置（BaseOffsetPose）
- 解析后的候选规范（CandidateSpec）
- 已接受样本的质量度量（AcceptedSampleQuality）
- 完整的已接受样本记录（AcceptedSampleRecord）
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 候选家族常量 —— 定义四种基本的采样模式
# ---------------------------------------------------------------------------

class CandidateFamily:
    """
    候选位姿的家族分类。

    - SPHERE_ANCHOR：球体锚点，通常位于球体表面固定方向，保证多视角覆盖。
    - SPHERE_HEIGHT：球体高度变化，通过改变末端高度提供 Z 方向多样性。
    - SPHERE_SHELL：球体外壳，在球体表面不同位置采样，涵盖多种平移和姿态组合。
    - SPHERE_ROLL_COVERAGE：纯滚转覆盖，在保持位置不变的情况下增加末端滚转角变化。
    """
    SPHERE_ANCHOR = "sphere_anchor"
    SPHERE_HEIGHT = "sphere_height"
    SPHERE_SHELL = "sphere_shell"
    SPHERE_ROLL_COVERAGE = "sphere_roll_coverage"


# 候选家族的生成顺序 —— 采集器会按此列表顺序依次尝试各家族的候选位姿
FAMILY_EXECUTION_ORDER = [
    "sphere_anchor",
    "sphere_height",
    "sphere_shell",
    "sphere_roll_coverage",
]


# ---------------------------------------------------------------------------
# 内部数据类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseOffsetPose:
    """
    单个基于家族的基础偏移候选记录。

    这是从 YAML 配置文件解析出的原始定义，包含相对于参考位姿的平移和旋转偏移量，
    以及用于指导采样策略的元数据（是否可移除、意图、可观测轴、去重保护等）。
    """
    label: str = ""                # 该偏移位姿的人类可读标签
    family: str = ""               # 所属候选家族（如 "sphere_anchor"）
    base_x: float = 0.0            # 相对于参考位姿的 X 方向平移（米）
    base_y: float = 0.0            # 相对于参考位姿的 Y 方向平移（米）
    base_z: float = 0.0            # 相对于参考位姿的 Z 方向平移（米）
    pitch: float = 0.0             # 绕局部 X 轴的俯仰角（度）
    yaw: float = 0.0               # 绕局部 Y 轴的偏航角（度）
    roll: float = 0.0              # 绕局部 Z 轴的滚转角（度）
    removable: bool = False        # 该样本是否可在后续子集优化中被移除
    intent: str = ""               # 该位姿的意图描述（如 "increase_z_coverage"）
    observability_axis: str = "none"  # 该候选主要贡献的可观测轴：pitch / yaw / roll / none
    dedup_protected: bool = False     # 若为 True，则在去重时跳过常规的平移/旋转接近度检查，仅依赖精确键值去重


@dataclass(frozen=True)
class CandidateSpec:
    """
    解析后的候选规范，携带唯一键值用于去重。

    与 BaseOffsetPose 类似，但此结构用于内部流程，可能补充额外的来源信息，
    且保证所有字段明确、不可变，便于散列和集合操作。
    """
    source: str                    # 候选来源标识（通常为 offset.label）
    base_x: float                  # 基座偏移 X（米）
    base_y: float                  # 基座偏移 Y（米）
    base_z: float                  # 基座偏移 Z（米）
    pitch: float                   # 俯仰角（度）
    yaw: float                     # 偏航角（度）
    roll: float                    # 滚转角（度）
    family: str                    # 候选家族
    removable: bool                # 是否可移除
    intent: str = ""               # 意图描述
    observability_axis: str = "none"  # 可观测轴
    dedup_protected: bool = False     # 是否受去重保护

    def exact_key(self):
        """
        计算精确去重键值。
        将基座偏移四舍五入到 4 位小数、姿态角度到 2 位小数，
        组合成一个元组，用于快速判断两个候选是否完全相同（避免浮点比较误差）。
        """
        return (
            round(self.base_x, 4), round(self.base_y, 4), round(self.base_z, 4),
            round(self.pitch, 2), round(self.yaw, 2), round(self.roll, 2),
        )


@dataclass(frozen=True)
class AcceptedSampleQuality:
    """
    描述一个已通过视觉质量门控的样本的多维度质量指标。

    所有数值来自图像处理、标记检测和稳定性统计，用于后续的子集优选和质量排序。
    """
    center_error_px: float         # 标记中心与图像中心的像素误差
    margin_px: float               # 标记与图像边界的最小像素距离
    marker_side_px: float          # 标记边长像素值
    distance_m: float              # 相机到标记的距离（米）
    camera_model_error_px: float   # 相机模型误差（像素），通常为重投影误差
    center_std_px: float           # 连续帧中标记中心位置的标准差（像素）
    depth_std_m: float             # 连续帧中深度值的标准差（米）
    angle_std_deg: float           # 连续帧中姿态角度的标准差（度）
    marker_note: str               # 关于标记检测状态的说明文本
    model_note: str                # 关于相机模型或位姿估计的说明文本
    stable_note: str               # 关于图像稳定性检查的说明文本


@dataclass(frozen=True)
class AcceptedSampleRecord:
    """
    一个完整的被接受样本的记录。

    将机器人的末端姿态、跟踪标记的观测姿态、候选规范和视觉质量
    打包在一起，并记录是否进行了重新居中尝试及其收敛状态。
    """
    robot_pose: object             # 末端执行器在基座坐标系下的变换（TransformMatrix 对象）
    tracking_pose: object          # 跟踪标记在相机坐标系下的变换（TransformMatrix 对象，可能为 None）
    family: str                    # 该样本的候选家族
    spec: CandidateSpec            # 生成该样本的候选规范
    quality: AcceptedSampleQuality # 样本的视觉质量度量
    candidate_idx: int             # 候选位姿在当次生成列表中的编号
    candidate_description: str     # 候选位姿的人类可读描述
    recenter_attempted: bool       # 是否尝试了重新居中（微调末端使标记更接近图像中心）
    recenter_strict_converged: bool # 重新居中是否严格收敛（达到预设的中心误差阈值）
    removable: bool                # 该样本是否允许在子集优化中被移除