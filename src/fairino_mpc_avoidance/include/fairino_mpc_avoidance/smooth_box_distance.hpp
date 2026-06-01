#pragma once
#include "fairino_mpc_avoidance/types.hpp"

namespace fairino_mpc {

class SmoothBoxDistance {
public:
    static double compute(const Vec3& point, const Vec3& center,
                          const Vec3& box_size, double kappa);
};

}  // namespace fairino_mpc
