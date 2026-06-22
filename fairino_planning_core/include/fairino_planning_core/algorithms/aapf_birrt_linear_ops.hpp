#pragma once

#include "fairino_planning_core/config/planning_params.hpp"
#include "fairino_planning_core/tree/rrt_tree.h"

#include <limits>

namespace fairino_planning::aapf_birrt_detail {

inline constexpr double kEpsDistZero = 1e-12;

inline JointConfig jointDeltaBounded(const JointConfig& from, const JointConfig& to) {
    return to - from;
}

inline double jointDistance(const JointConfig& a, const JointConfig& b) {
    return jointDeltaBounded(a, b).norm();
}

inline double jointDistanceSq(const JointConfig& a, const JointConfig& b) {
    return jointDeltaBounded(a, b).squaredNorm();
}

inline JointConfig steerBoundedLinear(
    const JointConfig& from,
    const JointConfig& to,
    double max_step,
    const JointLimits& limits) {
    const JointConfig delta = jointDeltaBounded(from, to);
    const double distance = delta.norm();
    if (distance < kEpsDistZero || distance <= max_step) {
        return limits.clamp(to);
    }
    return limits.clamp(from + delta * (max_step / distance));
}

inline int nearestBoundedLinear(const RRTTree& tree, const JointConfig& q) {
    int nearest = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < tree.size(); ++i) {
        const double distance = jointDistanceSq(tree.node(i).state, q);
        if (distance < best_distance) {
            nearest = i;
            best_distance = distance;
        }
    }
    return nearest;
}

}  // namespace fairino_planning::aapf_birrt_detail
