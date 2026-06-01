#pragma once

#include <algorithm>

#include "fairino_planning_core/types.h"

namespace fairino_planning::algorithm_utils {

inline JointConfig steer(const JointConfig& from, const JointConfig& to, double max_step) {
    JointConfig v = wrapToPi(to - from);
    const double nv = v.norm();
    if (nv < 1e-12) {
        return from;
    }
    const double step = std::min(max_step, nv);
    return wrapToPi(from + (step / nv) * v);
}

}  // namespace fairino_planning::algorithm_utils
