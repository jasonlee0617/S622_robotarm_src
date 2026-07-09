/**
 * file robot_kinematics.cpp
 * brief 基于正运动学(FK)的几何采样工具
 *
 * 本文件为 MPC 碰撞距离评估提供几何采样功能：
 * 1. 将 Eigen 关节向量转换为 fairino_planning 正运动学库所需的输入格式。
 * 2. 提供获取关节和工具坐标系笛卡尔位置的功能。
 * 3. 提供在相邻关节/工具点之间的线段上密集采样的功能，用于距离/APF/CBF 计算。
 *
 * 典型用法：
 * - getJointPositions(q): 仅需关节/工具坐标系位置时调用。
 * - samplePoints(q, points_per_link): 需密集身体采样点时调用。
 * - points_per_link 应为非负数，负值会被钳位为 0。
 */

#include "fairino_mpc_avoidance/robot_kinematics.hpp"
#include "fairino_planning_core/dh_kinematics.h"   // DH 参数正运动学库
#include "fairino_planning_core/types.h"            // 关节配置、变换类型
#include <algorithm>

namespace fairino_mpc {

namespace {

/**
 * brief 将本项目的关节向量 (VecN) 转换为 fairino_planning 的 JointConfig 类型
 * param q 项目使用的 N_JOINTS 维关节向量 (Eigen::Matrix<double, N_JOINTS, 1>)
 * return fairino_planning::JointConfig 对象（std::array<double, 6>）
 */
fairino_planning::JointConfig toJointConfig(const VecN& q) {
    fairino_planning::JointConfig q_fp;
    for (int i = 0; i < N_JOINTS; ++i) q_fp[i] = q(i);
    return q_fp;
}

/**
 * brief 在两点之间的线段上追加采样点
 * param pA 线段起点
 * param pB 线段终点
 * param points_per_link 两点之间插入的采样点数量（不包括起点）
 * param out 输出容器，起点和内部点都会被追加
 *
 * 追加逻辑：先添加起点 pA，然后在 pA 和 pB 之间均匀插入 points_per_link 个点。
 * 终点 pB 由调用方在其他地方（下一段的起点或末尾）添加，避免重复。
 */
void appendSegmentSamples(const Vec3& pA, const Vec3& pB, int points_per_link,
                          std::vector<Vec3>& out) {
    out.push_back(pA);   // 线段起点
    for (int s = 1; s <= points_per_link; ++s) {
        double alpha = static_cast<double>(s) / (points_per_link + 1);
        // 线性插值
        out.push_back((1.0 - alpha) * pA + alpha * pB);
    }
}

}  // namespace

/**
 * brief 获取 DH 运动学单例对象，避免重复构造/析构开销
 * return fairino_planning::DHKinematics 实例的引用
 */
fairino_planning::DHKinematics& RobotKinematics::getDHKinematics() {
    static fairino_planning::DHKinematics fk;  // 局部静态变量，全局唯一
    return fk;
}

/**
 * brief 获取关节和工具坐标系的笛卡尔位置
 * param q          当前关节位置 (N_JOINTS 维)
 * param tool_model 工具模型枚举（如 GRIPPER 表示带夹爪，需要额外输出夹爪末端位置）
 * return 包含所有关键点位置的向量，顺序为 T00, T01, ..., T06（法兰），若为 GRIPPER 则附加夹爪末端位置
 *
 * 流程：
 * 1. 调用 DH 正运动学得到所有关节的变换矩阵 T00..T06（7个，对应基座和6个关节）。
 * 2. 从每个变换矩阵中提取平移向量作为位置。
 * 3. 若工具模型为 GRIPPER，则额外计算夹爪末端变换并提取位置。
 */
std::vector<Vec3> RobotKinematics::getJointPositions(const VecN& q,
                                                      ToolModel tool_model) {
    // 1) 转换关节向量格式
    const fairino_planning::JointConfig q_fp = toJointConfig(q);

    // 2) 调用正运动学，获取全部变换 (T00..T06)
    auto& fk = getDHKinematics();
    auto all_T = fk.fkineAll(q_fp);  // 返回 std::array<Transform4d, 7>

    // 提取各关节位置 (基座 + 6个关节 -> 7个点)
    std::vector<Vec3> positions(N_JOINTS + 1);
    for (int i = 0; i <= N_JOINTS; ++i) {
        positions[i] = all_T[i].block<3, 1>(0, 3); // 取齐次矩阵的平移部分
    }

    // 3) 如果指定了 GRIPPER 工具模型，追加夹爪末端位置
    if (tool_model == ToolModel::GRIPPER) {
        fairino_planning::Transform4d T_tool =
            all_T[6] * fk.toolTransform(tool_model);
        positions.push_back(T_tool.block<3, 1>(0, 3));  // 索引 positions[7]
    }

    return positions;
}

/**
 * brief 对机器人结构进行密集采样，返回所有采样点在笛卡尔空间中的位置
 * param q                当前关节位置 (N_JOINTS 维)
 * param points_per_link  每段之间插入的采样点数量（非负）
 * param tool_model       工具模型（影响是否包含夹爪段）
 * return 采样点位置向量
 *
 * 采样策略：
 * - 首先获取所有关键点（关节和工具）的位置。
 * - 对于每两个相邻关键点构成的线段，等间距插入 points_per_link 个点。
 * - 最后单独追加最后一个关键点，保证端点完整性。
 */
std::vector<Vec3> RobotKinematics::samplePoints(const VecN& q, int points_per_link,
                                                 ToolModel tool_model) {
    // 规范化输入：负数 -> 0（表示不在内部插点，仅输出关键点）
    points_per_link = std::max(0, points_per_link);

    // 获取所有关键点位置（7个或8个）
    auto joint_pos = getJointPositions(q, tool_model);
    const int n_joints = static_cast<int>(joint_pos.size());  // 7 或 8（含夹爪）
    const int n_segs   = n_joints - 1;                        // 6 或 7 段

    std::vector<Vec3> pts;
    // 预估容量，减少重复分配
    pts.reserve(n_segs * (points_per_link + 1) + 1);

    // 逐段添加采样点
    for (int seg = 0; seg < n_segs; ++seg) {
        Vec3 pA = joint_pos[seg];      // 段起点
        Vec3 pB = joint_pos[seg + 1];  // 段终点
        appendSegmentSamples(pA, pB, points_per_link, pts);
    }
    // 添加最后一个关键点（上一段的终点未在 appendSegmentSamples 中添加）
    pts.push_back(joint_pos.back());

    return pts;
}

}  // namespace fairino_mpc
