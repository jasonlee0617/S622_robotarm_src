// // include/fairino_planning_core/constraints/orientation_checker.h
// #pragma once

// #include "fairino_planning_core/types.h"        // OrientationPolicy 已在这里定义
// #include "fairino_planning_core/dh_kinematics.h"
// #include <cmath>

// namespace fairino_planning {

// // ★ 不再重复定义 OrientationPolicy，直接使用 types.h 中的

// class OrientationChecker {
// public:
//     OrientationChecker() = default;
//     explicit OrientationChecker(const OrientationPolicy& policy)
//         : policy_(policy) {}

//     void setPolicy(const OrientationPolicy& policy) { policy_ = policy; }
//     void setTargetOrientation(const RotMatrix3d& R_target) { R_target_ = R_target; }
//     void setTargetPosition(const Vector3d& p_target) { p_target_ = p_target; }

//     bool check(const JointConfig& q, const DHKinematics& fk) const {
//         Transform4d T = fk.fkine(q);
//         Vector3d p_ee = T.block<3,1>(0,3);
//         RotMatrix3d R_ee = T.block<3,3>(0,0);
//         double d = (p_ee - p_target_).norm();
//         if (d > policy_.ori_gate_dist) return true;

//         RotMatrix3d R_err = R_target_.transpose() * R_ee;
//         double cos_angle = std::clamp((R_err.trace() - 1.0) / 2.0, -1.0, 1.0);
//         double err_deg = std::acos(cos_angle) * 180.0 / M_PI;

//         double t = 0.0;
//         double range = policy_.ori_gate_dist - policy_.near_dist;
//         if (range > 1e-6)
//             t = std::clamp((policy_.ori_gate_dist - d) / range, 0.0, 1.0);
//         else
//             t = (d <= policy_.near_dist) ? 1.0 : 0.0;

//         double tol_deg = policy_.ori_far_tol_deg * (1.0 - t)
//                        + policy_.ori_near_tol_deg * t;
//         return err_deg <= tol_deg;
//     }

//     double getOrientationWeight(const Vector3d& p_ee) const {
//         double d = (p_ee - p_target_).norm();
//         if (d > policy_.ori_gate_dist) return policy_.ori_weight_far;
//         double range = policy_.ori_gate_dist - policy_.near_dist;
//         double t = (range > 1e-6)
//                    ? std::clamp((policy_.ori_gate_dist - d) / range, 0.0, 1.0)
//                    : ((d <= policy_.near_dist) ? 1.0 : 0.0);
//         return policy_.ori_weight_far * (1.0 - t) + policy_.ori_weight_near * t;
//     }

//     double orientationError(const JointConfig& q, const DHKinematics& fk) const {
//         Transform4d T = fk.fkine(q);
//         RotMatrix3d R_ee = T.block<3,3>(0,0);
//         RotMatrix3d R_err = R_target_.transpose() * R_ee;
//         double cos_angle = std::clamp((R_err.trace() - 1.0) / 2.0, -1.0, 1.0);
//         return std::acos(cos_angle);
//     }

// private:
//     OrientationPolicy policy_;
//     RotMatrix3d R_target_ = RotMatrix3d::Identity();
//     Vector3d p_target_ = Vector3d::Zero();
// };

// }  // namespace fairino_planning
#pragma once

#include "fairino_planning_core/types.h"
#include "fairino_planning_core/dh_kinematics.h"
#include <Eigen/Dense>
#include <cmath>

namespace fairino_planning {

class OrientationChecker {
public:
    explicit OrientationChecker(const OrientationPolicy& policy = OrientationPolicy())
        : policy_(policy) {}

    void setTargetOrientation(const RotMatrix3d& R_target) {
        R_target_ = R_target;
        has_target_orientation_ = true;
    }

    void setTargetPosition(const Vector3d& p_target) {
        p_target_ = p_target;
        has_target_position_ = true;
    }

    void setToolModel(ToolModel model) {
        tool_model_ = model;
    }

    ToolModel getToolModel() const {
        return tool_model_;
    }

    void setPolicy(const OrientationPolicy& policy) {
        policy_ = policy;
    }

    const OrientationPolicy& getPolicy() const {
        return policy_;
    }

    // 主接口：内部自己做 FK，并按当前 tool_model_ 检查
    bool check(const JointConfig& q, const DHKinematics& fk) const;

    // 若外部已经有末端位姿，也可以直接传
    bool checkPose(const Vector3d& position, const RotMatrix3d& orientation) const;

    // 返回当前姿态误差角（弧度）
    double orientationErrorRad(const RotMatrix3d& R) const;

    // 返回当前姿态误差角（角度）
    double orientationErrorDeg(const RotMatrix3d& R) const;

    // 返回当前目标距离
    double distanceToTarget(const Vector3d& p) const;

private:
    OrientationPolicy policy_;

    RotMatrix3d R_target_ = RotMatrix3d::Identity();
    Vector3d p_target_ = Vector3d::Zero();

    bool has_target_orientation_ = false;
    bool has_target_position_ = false;

    ToolModel tool_model_ = ToolModel::FLANGE;

    // 根据距离决定允许的姿态误差（角度）
    double allowedOrientationToleranceDeg(double dist) const;
};

}  // namespace fairino_planning
