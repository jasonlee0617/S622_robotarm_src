// src/ik/fairino_ik.cpp
// Fairino 机器人逆运动学求解器实现（解析法）
// 支持法兰（无夹爪）和带夹爪（有工具偏移）两种末端执行器模型

#include "fairino_planning_core/ik/fairino_ik.h"
#include <algorithm>
#include <cmath>
#include <Eigen/Geometry>
#include <sstream>

namespace fairino_planning {

const char* toString(IKFailureCategory category) {
    switch (category) {
        case IKFailureCategory::kNone: return "None";
        case IKFailureCategory::kGeometryUnreachable: return "GeometryUnreachable";
        case IKFailureCategory::kModelInconsistency: return "ModelInconsistency";
        case IKFailureCategory::kCandidateFiltered: return "CandidateFiltered";
        case IKFailureCategory::kInternal: return "Internal";
    }
    return "Unknown";
}

namespace {
void setFailure(IKResult& result, IKFailureCategory category, const std::string& code, const std::string& detail = {}) {
    result.failure_category = category;
    result.failure_code = code;
    result.failure_stage = code;
    result.failure_detail = detail;
}

std::string numericDetail(const char* key, double value) {
    std::ostringstream oss;
    oss << key << "=" << value;
    return oss.str();
}
}  // namespace

// ========================= 构造函数 =========================
FairinoIK::FairinoIK() : fk_(DHParams{}), limits_(), params_() {}
FairinoIK::FairinoIK(const AnalyticalIKParams& params)
    : fk_(DHParams{}, params.gripper_tool), limits_(), params_(params) {}

// ========================= 计算 R03 矩阵 =========================
/// @brief 计算从基座到关节3的旋转矩阵（仅与 q1, q2, q3 有关）
/// 用于解析法求解腕部姿态
RotMatrix3d FairinoIK::R03(double q1, double q2, double q3) {
    const double c1 = std::cos(q1), s1 = std::sin(q1);
    const double c23 = std::cos(q2 + q3), s23 = std::sin(q2 + q3);

    RotMatrix3d R;
    R << c1*c23, -c1*s23,  s1,
         s1*c23, -s1*s23, -c1,
            s23,     c23,   0;
    return R;
}

// ========================= 工具目标 → 法兰目标转换 =========================
/// @brief 将工具坐标系的目标位姿转换为法兰坐标系（坐标系6）的目标位姿
/// 
/// 原理：
///   设 T_tool_target   : 用户期望的工具坐标系（如 grasp_frame）在世界坐标系中的位姿
///       T_flange_target: 我们需要求解的法兰坐标系（坐标系6）在世界坐标系中的位姿
///       T_tool_flange  : 工具坐标系相对于法兰坐标系的固定变换（由 ToolParams 定义）
/// 
///   由于 T_tool_target = T_flange_target * T_tool_flange
///   因此 T_flange_target = T_tool_target * (T_tool_flange)^{-1}
/// 
/// 参数：
///   T_target_tool : 工具坐标系的目标位姿
///   model         : 工具模型（FLANGE 时无偏移，GRIPPER 时有固定平移）
/// 返回：
///   对应的法兰坐标系目标位姿
Transform4d FairinoIK::toolTargetToFlangeTarget(const Transform4d& T_target_tool,ToolModel model) const {
    // 获取工具相对于法兰的变换（从法兰到工具）
    Transform4d T_6_tool = fk_.toolTransform(model);
    // 计算逆变换：从工具到法兰
    Transform4d T_tool_6 = T_6_tool.inverse();
    // 右乘得到法兰目标
    return T_target_tool * T_tool_6;
}

// ========================= 便捷接口（位置+RPY） =========================
IKResult FairinoIK::solve(const Vector3d& pos,
                          const Vector3d& rpy_zyx,
                          ToolModel model) const {
    // 将 RPY 角（ZYX 顺序）转换为旋转矩阵
    const double cy = std::cos(rpy_zyx[0]), sy = std::sin(rpy_zyx[0]);
    const double cp = std::cos(rpy_zyx[1]), sp = std::sin(rpy_zyx[1]);
    const double cr = std::cos(rpy_zyx[2]), sr = std::sin(rpy_zyx[2]);

    RotMatrix3d R;
    R << cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr,
         sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr,
           -sp,           cp*sr,           cp*cr;

    Transform4d T = Transform4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = pos;
    return solve(T, model);
}

// ========================= 便捷接口：法兰直接求解 =========================
IKResult FairinoIK::solveFlange(const Transform4d& T_target) const {
    return solve(T_target, ToolModel::FLANGE);
}

// ========================= 便捷接口：夹爪求解 =========================
IKResult FairinoIK::solveGripper(const Transform4d& T_target) const {
    return solve(T_target, ToolModel::GRIPPER);
}

// ========================= 核心逆解函数（支持工具模型） =========================
/// @brief 给定工具目标位姿，求解关节角
/// 步骤：
///   1. 将工具目标位姿转换为法兰目标位姿
///   2. 使用解析法（基于 DH 参数）求解法兰目标对应的关节角
///   3. 用正运动学验证每个候选解，确保工具位姿误差小于阈值
///   4. 去重 + 限位筛选
IKResult FairinoIK::solve(const Transform4d& T_target, ToolModel model) const {
    IKResult result;
    result.total_branches = 8;  // q1:2 * q5:2 * q3:2
    result.target_pose = T_target;

    // ---------- 1. 工具目标 → 法兰目标 ----------
    const Transform4d T_flange = toolTargetToFlangeTarget(T_target, model);
    result.flange_pose = T_flange;

    // ---------- 2. 解析法逆解（以下代码与注释代码基本相同，但使用 d6_flange_）----------
    const Vector3d p = T_flange.block<3,1>(0,3);
    const RotMatrix3d R = T_flange.block<3,3>(0,0);
    const Vector3d a = R.col(2);  // 法兰的 z 轴方向

    const double ax = a[0], ay = a[1], az = a[2];

    // 腕点（关节5中心）位置
    const Vector3d pw = p - d6_flange_ * a;
    const double xw = pw[0], yw = pw[1], zw = pw[2];
    result.wrist_x = xw;
    result.wrist_y = yw;
    result.wrist_z = zw;

    // 求解 q1
    const double rho_sq = xw*xw + yw*yw - d4_*d4_;
    result.rho_sq = rho_sq;
    if (rho_sq < params_.rho_sq_neg_eps) {
        setFailure(
            result,
            IKFailureCategory::kGeometryUnreachable,
            "rho_sq",
            numericDetail("rho_sq", rho_sq));
        return result;
    }
    const double rho_abs = std::sqrt(std::max(rho_sq, 0.0));

    const double q1_cands[2] = {
        std::atan2(yw, xw) - std::atan2(-d4_,  rho_abs),
        std::atan2(yw, xw) - std::atan2(-d4_, -rho_abs)
    };
    result.survive_q1 = 2;

    std::vector<JointConfig> candidates;

    for (int i = 0; i < 2; ++i) {
        const double q1 = wrapToPi(q1_cands[i]);
        const double c1 = std::cos(q1), s1 = std::sin(q1);

        // 求解 q5（腕关节）
        double c5 = s1*ax - c1*ay;
        c5 = std::clamp(c5, -1.0, 1.0);
        const double s5_abs = std::sqrt(std::max(0.0, 1.0 - c5*c5));
        if (s5_abs < params_.wrist_singularity_s5_min) {
            IKResult::WristRejectInfo reject;
            reject.q1_branch = i;
            reject.q1 = q1;
            reject.c5 = c5;
            reject.s5_abs = s5_abs;
            result.wrist_rejects.push_back(reject);
            continue;  // 腕奇异，跳过
        }
        result.survive_q5 += 2;

        const double C = c1*ax + s1*ay;

        for (double s5 : {s5_abs, -s5_abs}) {
            const double q5 = std::atan2(s5, c5);

            // 求解 q234（关节2、3、4的和）— 归一化保护
            double c234 = -C / s5;
            double s234 = -az / s5;
            const double n234 = std::hypot(c234, s234);
            if (n234 < 1e-9) continue;
            c234 /= n234;
            s234 /= n234;
            const double q234 = std::atan2(s234, c234);

            // 平面 2R 子问题求解 q2, q3
            const double Xp = c1*xw + s1*yw - d5_*s234;
            const double Zp = zw - d1_ + d5_*c234;
            const double Xg = -Xp;
            const double Zg = -Zp;

            double D = (Xg*Xg + Zg*Zg - L2_*L2_ - L3_*L3_) / (2.0*L2_*L3_);
            if (D < -1.0 - params_.D_domain_eps || D > 1.0 + params_.D_domain_eps) {
                IKResult::DDomainRejectInfo reject;
                reject.q1_branch = i;
                reject.s5_sign = (s5 > 0.0) ? 1 : -1;
                reject.q1 = q1;
                reject.q5 = q5;
                reject.q234 = q234;
                reject.D = D;
                reject.Xg = Xg;
                reject.Zg = Zg;
                result.d_domain_rejects.push_back(reject);
                continue;
            }
            D = std::clamp(D, -1.0, 1.0);
            const double s3_abs = std::sqrt(std::max(0.0, 1.0 - D*D));
            result.survive_q23 += 2;

            for (double s3 : {s3_abs, -s3_abs}) {
                const double q3 = std::atan2(s3, D);
                const double q2 = std::atan2(Zg, Xg) - std::atan2(L3_*s3, L2_ + L3_*D);
                const double q4 = q234 - q2 - q3;

                // 求解 q6 — R36(2,0)=s5*c6, R36(2,1)=-s5*s6, 必须除以 s5
                RotMatrix3d R03_ = R03(q1, q2, q3);
                RotMatrix3d R36 = R03_.transpose() * R;
                if (std::abs(s5) < params_.wrist_singularity_s5_min) continue;
                const double q6 = std::atan2(
                    -R36(2,1) / s5,
                     R36(2,0) / s5);

                JointConfig q;
                q << q1, q2, q3, q4, q5, q6;
                q = wrapToPi(q);

                // ---------- 3. 正运动学验证（使用正确的工具模型）----------
                Transform4d T_check = fk_.fkine(q, model);
                double pos_err = (T_check.block<3,1>(0,3) - T_target.block<3,1>(0,3)).norm();
                double rot_err = (T_check.block<3,3>(0,0) - T_target.block<3,3>(0,0)).norm();

                if (pos_err < params_.fk_verify_pos_tol && rot_err < params_.fk_verify_rot_tol) {
                    candidates.push_back(q);
                    ++result.survive_fk_verify;
                } else {
                    IKResult::FkRejectInfo r;
                    r.q = q;
                    r.pos_err = pos_err;
                    r.rot_err = rot_err;
                    r.q1_branch = i;
                    r.s5_sign = (s5 > 0.0) ? 1 : -1;
                    r.s3_sign = (s3 > 0.0) ? 1 : -1;
                    result.fk_rejects.push_back(r);
                }
            }
        }
    }

    if (candidates.empty()) {
        if (result.survive_q5 == 0 && !result.wrist_rejects.empty()) {
            setFailure(result, IKFailureCategory::kGeometryUnreachable, "wrist_singularity");
        } else if (result.survive_q23 == 0 && !result.d_domain_rejects.empty()) {
            double max_abs_violation = 0.0;
            for (const auto& r : result.d_domain_rejects) {
                max_abs_violation = std::max(max_abs_violation, std::max(0.0, std::abs(r.D) - 1.0));
            }
            setFailure(
                result,
                IKFailureCategory::kGeometryUnreachable,
                "D_domain",
                numericDetail("max_abs_D_minus_1", max_abs_violation));
        } else if (!result.fk_rejects.empty()) {
            double max_pos_err = 0.0;
            double max_rot_err = 0.0;
            for (const auto& r : result.fk_rejects) {
                max_pos_err = std::max(max_pos_err, r.pos_err);
                max_rot_err = std::max(max_rot_err, r.rot_err);
            }
            std::ostringstream detail;
            detail << "max_pos_err=" << max_pos_err << ",max_rot_err=" << max_rot_err;
            setFailure(result, IKFailureCategory::kModelInconsistency, "fk_verify", detail.str());
        } else {
            setFailure(result, IKFailureCategory::kGeometryUnreachable, "no_raw_candidates");
        }
        return result;
    }

    // ---------- 4. 去重 ----------
    candidates = uniqueSolutions(candidates, params_.solution_unique_tol);
    result.survive_unique = static_cast<int>(candidates.size());

    // ---------- 5. 限位筛选 ----------
    for (const auto& q : candidates) {
        if (limits_.isWithin(q)) {
            result.solutions.push_back(q);
            ++result.survive_joint_limits;
        } else {
            IKResult::LimitRejectInfo reject;
            reject.q = q;
            for (int i = 0; i < NUM_JOINTS; ++i) {
                if (q[i] < limits_.lower[i]) {
                    reject.lower_violation[i] = limits_.lower[i] - q[i];
                }
                if (q[i] > limits_.upper[i]) {
                    reject.upper_violation[i] = q[i] - limits_.upper[i];
                }
            }
            result.limit_rejects.push_back(reject);
        }
    }

    result.success = !result.solutions.empty();
    if (!result.success) {
        setFailure(
            result,
            IKFailureCategory::kCandidateFiltered,
            result.limit_rejects.empty() ? "no_joint_limit_candidates" : "joint_limits");
    }
    return result;
}

// ========================= 去重函数 =========================
/// @brief 基于关节角差值的欧氏距离去除近似相同的解
/// @param sols 原始解列表
/// @param tol  容差（弧度）
/// @return 去重后的解列表
std::vector<JointConfig> FairinoIK::uniqueSolutions(
    const std::vector<JointConfig>& sols, double tol) {
    std::vector<JointConfig> unique;
    for (const auto& q : sols) {
        bool dup = false;
        for (const auto& u : unique) {
            if (wrapToPi(q - u).norm() < tol) {
                dup = true;
                break;
            }
        }
        if (!dup) unique.push_back(q);
    }
    return unique;
}

}  // namespace fairino_planning
