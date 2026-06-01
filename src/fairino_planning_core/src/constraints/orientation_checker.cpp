#include "fairino_planning_core/constraints/orientation_checker.h"
#include <algorithm>

namespace fairino_planning {

double OrientationChecker::distanceToTarget(const Vector3d& p) const
{
    if (!has_target_position_) {
        return std::numeric_limits<double>::infinity();
    }
    return (p - p_target_).norm();
}

double OrientationChecker::orientationErrorRad(const RotMatrix3d& R) const
{
    if (!has_target_orientation_) {
        return 0.0;
    }

    RotMatrix3d R_err = R_target_.transpose() * R;
    double trace_val = (R_err.trace() - 1.0) * 0.5;
    trace_val = std::clamp(trace_val, -1.0, 1.0);
    return std::acos(trace_val);
}

double OrientationChecker::orientationErrorDeg(const RotMatrix3d& R) const
{
    return orientationErrorRad(R) * 180.0 / M_PI;
}

double OrientationChecker::allowedOrientationToleranceDeg(double dist) const
{
    // 如果没目标位置，就退化成近区容差
    if (!has_target_position_) {
        return policy_.ori_near_tol_deg;
    }

    // 大于 gate 区域：远区容差
    if (dist >= policy_.ori_gate_dist) {
        return policy_.ori_far_tol_deg;
    }

    // 在 gate 区域内：从 far_tol 线性收紧到 near_tol
    double gate = std::max(policy_.ori_gate_dist, 1e-6);
    double alpha = std::clamp(dist / gate, 0.0, 1.0);

    return policy_.ori_near_tol_deg +
           alpha * (policy_.ori_far_tol_deg - policy_.ori_near_tol_deg);
}

bool OrientationChecker::checkPose(
    const Vector3d& position,
    const RotMatrix3d& orientation) const
{
    if (!has_target_orientation_) {
        return true;
    }

    double dist = distanceToTarget(position);
    double err_deg = orientationErrorDeg(orientation);
    double tol_deg = allowedOrientationToleranceDeg(dist);

    return err_deg <= tol_deg;
}

bool OrientationChecker::check(
    const JointConfig& q,
    const DHKinematics& fk) const
{
    Transform4d T = fk.fkine(q, tool_model_);
    Vector3d p = T.block<3,1>(0,3);
    RotMatrix3d R = T.block<3,3>(0,0);
    return checkPose(p, R);
}

}  // namespace fairino_planning
