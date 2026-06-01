#pragma once

#include <Eigen/Dense>

namespace fairino_planning {

constexpr int NUM_JOINTS = 6;

using JointConfig = Eigen::Matrix<double, NUM_JOINTS, 1>;
using JointMatrix = Eigen::Matrix<double, Eigen::Dynamic, NUM_JOINTS>;
using Transform4d = Eigen::Matrix4d;
using RotMatrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Jacobian6d = Eigen::Matrix<double, 6, NUM_JOINTS>;

}  // namespace fairino_planning

