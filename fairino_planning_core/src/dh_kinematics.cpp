// src/dh_kinematics.cpp
// DH 运动学求解器实现
// 提供正运动学、雅可比矩阵计算，支持法兰和带夹爪的工具模型

#include "fairino_planning_core/dh_kinematics.h"
#include <cmath>
#include <Eigen/Geometry>

namespace fairino_planning {

// ========================= 单关节 DH 变换矩阵 =========================
/// @brief 根据 DH 参数构造单关节的齐次变换矩阵
/// @param theta 关节角（弧度）
/// @param d     连杆偏距（米）
/// @param a     连杆长度（米）
/// @param alpha 连杆扭角（弧度）
/// @return 4×4 变换矩阵，表示从当前关节坐标系到下一关节坐标系的变换
Transform4d DHKinematics::dhTransform(double theta, double d, double a, double alpha) {
    const double ct = std::cos(theta), st = std::sin(theta);
    const double ca = std::cos(alpha), sa = std::sin(alpha);

    // 标准 DH 变换矩阵
    Transform4d T;
    T << ct, -st*ca,  st*sa, a*ct,
         st,  ct*ca, -ct*sa, a*st,
          0,     sa,     ca,    d,
          0,      0,      0,    1;
    return T;
}

// ========================= 工具变换矩阵 =========================
/// @brief 返回工具坐标系相对于法兰坐标系（坐标系6）的固定变换
/// @param model 工具模型：FLANGE（无偏移）或 GRIPPER（带夹爪）
/// @return 4×4 齐次变换矩阵 T_6_tool
/// 
/// 对于 GRIPPER 模型，工具点相对法兰的偏移为 (0, 0, 0.1168) 米，
/// 这是因为原始 DH 模型中 d6 = 0.100 米，而实际夹爪中心总距离为 0.2168 米，
/// 因此需要额外补偿 0.1168 米。
Transform4d DHKinematics::toolTransform(ToolModel model) const {
    Transform4d T = Transform4d::Identity();

    if (model == ToolModel::GRIPPER) {
        const auto& rpy = gripper_tool_.rpy;
        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix() *
            Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()).toRotationMatrix() *
            Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()).toRotationMatrix();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = gripper_tool_.offset;
    }
    return T;
}

// ========================= 正向运动学：法兰位姿 =========================
/// @brief 给定关节角，计算法兰坐标系（坐标系6）的位姿
/// @param q 6 维关节角向量（弧度）
/// @return 法兰在世界坐标系中的 4×4 齐次变换矩阵
Transform4d DHKinematics::fkineFlange(const JointConfig& q) const {
    Transform4d T = Transform4d::Identity();
    for (int i = 0; i < NUM_JOINTS; ++i) {
        // 依次左乘每个关节的 DH 变换
        T = T * dhTransform(q[i], params_.d[i], params_.a[i], params_.alpha[i]);
    }
    return T;
}

/// @brief 法兰位姿的 Pose 表示（位置 + 欧拉角）
Pose DHKinematics::fkineFlangePose(const JointConfig& q) const {
    return Pose::fromTransform(fkineFlange(q));
}

// ========================= 正向运动学：工具位姿 =========================
/// @brief 给定关节角和工具模型，计算工具坐标系（如夹爪中心）的位姿
/// @param q     关节角
/// @param model 工具模型（FLANGE 时返回法兰位姿，GRIPPER 时返回工具点位姿）
/// @return 工具在世界坐标系中的 4×4 齐次变换矩阵
Transform4d DHKinematics::fkine(const JointConfig& q, ToolModel model) const {
    Transform4d T06 = fkineFlange(q);                // 法兰位姿
    return T06 * toolTransform(model);               // 右乘工具偏移得到工具位姿
}

/// @brief 工具位姿的 Pose 表示
Pose DHKinematics::fkinePose(const JointConfig& q, ToolModel model) const {
    return Pose::fromTransform(fkine(q, model));
}

// ========================= 所有中间变换矩阵 =========================
/// @brief 计算从基座到每个关节坐标系的变换矩阵（共7个）
/// @param q 关节角
/// @return 数组 Ts[0]~Ts[6]，其中 Ts[0] 为单位矩阵（基座自身），
///         Ts[1] 为基座到关节1的变换，...，Ts[6] 为基座到法兰的变换
std::array<Transform4d, 7> DHKinematics::fkineAll(const JointConfig& q) const {
    std::array<Transform4d, 7> Ts;
    Ts[0] = Transform4d::Identity();                 // 基座坐标系
    for (int i = 0; i < NUM_JOINTS; ++i) {
        // 累积变换
        Ts[i+1] = Ts[i] * dhTransform(q[i], params_.d[i], params_.a[i], params_.alpha[i]);
    }
    return Ts;
}

// ========================= 法兰雅可比矩阵 =========================
/// @brief 计算法兰坐标系（坐标系6）的几何雅可比矩阵（在基坐标系下表示）
/// @param q 关节角
/// @return 6×6 雅可比矩阵，前3行为线速度部分，后3行为角速度部分
/// 
/// 几何雅可比第 i 列的构造：
///   线速度分量：z_i × (p_n - p_i)
///   角速度分量：z_i
/// 其中 z_i 为第 i 个关节坐标系 Z 轴在基坐标系中的方向向量，
/// p_n 为法兰位置，p_i 为第 i 个坐标系原点。
Jacobian6d DHKinematics::jacobianFlange(const JointConfig& q) const {
    auto Ts = fkineAll(q);
    Vector3d o_n = Ts[6].block<3,1>(0,3);           // 法兰位置

    Jacobian6d J = Jacobian6d::Zero();
    for (int i = 0; i < NUM_JOINTS; ++i) {
        Vector3d z = Ts[i].block<3,1>(0,2);         // 第 i 关节 Z 轴方向
        Vector3d o = Ts[i].block<3,1>(0,3);         // 第 i 坐标系原点
        J.block<3,1>(0,i) = z.cross(o_n - o);       // 线速度部分
        J.block<3,1>(3,i) = z;                      // 角速度部分
    }
    return J;
}

// ========================= 工具点雅可比矩阵 =========================
/// @brief 计算工具坐标系（考虑工具偏移）的几何雅可比矩阵
/// @param q     关节角
/// @param model 工具模型
/// @return 6×6 雅可比矩阵，表示关节速度到工具点线速度和角速度的映射
/// 
/// 与法兰雅可比的不同在于线速度部分：使用工具点位置 o_tool 代替法兰位置 o_n，
/// 而角速度部分不变（工具与法兰固连，角速度相同）。
Jacobian6d DHKinematics::jacobian(const JointConfig& q, ToolModel model) const {
    auto Ts = fkineAll(q);

    // 计算工具点在基坐标系中的位置
    Transform4d T_tool = Ts[6] * toolTransform(model);
    Vector3d o_tool = T_tool.block<3,1>(0,3);       // 工具点位置

    Jacobian6d J = Jacobian6d::Zero();
    for (int i = 0; i < NUM_JOINTS; ++i) {
        Vector3d z = Ts[i].block<3,1>(0,2);         // 第 i 关节 Z 轴
        Vector3d o = Ts[i].block<3,1>(0,3);         // 第 i 坐标系原点
        // 线速度部分：z × (p_tool - p_i)
        J.block<3,1>(0,i) = z.cross(o_tool - o);
        // 角速度部分：z（与法兰雅可比相同）
        J.block<3,1>(3,i) = z;
    }
    return J;
}

}  // namespace fairino_planning
