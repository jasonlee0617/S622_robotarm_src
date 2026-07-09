#pragma once

#include <array>
#include <cmath>

#include "fairino_planning_core/types/aliases.hpp"

namespace fairino_planning {

struct DHParams {
    std::array<double, NUM_JOINTS> d{0.140, 0, 0, 0.102, 0.102, 0.100};
    std::array<double, NUM_JOINTS> a{0, -0.280, -0.240, 0, 0, 0};
    std::array<double, NUM_JOINTS> alpha{M_PI / 2, 0, 0, M_PI / 2, -M_PI / 2, 0};
};

enum class ToolModel {
    FLANGE = 0,
    GRIPPER = 1
};

struct ToolParams {
    Vector3d offset = Vector3d(0.0, 0.0, 0.0);
    Vector3d rpy = Vector3d(0.0, 0.0, 0.0);

    static ToolParams flange() {
        ToolParams t;
        t.offset = Vector3d(0.0, 0.0, 0.0);
        t.rpy = Vector3d(0.0, 0.0, 0.0);
        return t;
    }

    static ToolParams gripper() {
        ToolParams t;
        t.offset = Vector3d(0.0, 0.0, 0.1168);
        t.rpy = Vector3d(0.0, 0.0, 0.0);
        return t;
    }
};

struct Pose {
    Vector3d position;
    double rx{};
    double ry{};
    double rz{};

    static Pose fromTransform(const Transform4d& T) {
        Pose p;
        p.position = T.block<3, 1>(0, 3);
        const RotMatrix3d R = T.block<3, 3>(0, 0);
        const auto euler = R.eulerAngles(0, 1, 2);
        p.rx = euler[0];
        p.ry = euler[1];
        p.rz = euler[2];
        return p;
    }
};

}  // namespace fairino_planning
