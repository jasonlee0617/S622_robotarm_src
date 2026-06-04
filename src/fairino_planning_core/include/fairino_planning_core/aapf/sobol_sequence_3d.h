#pragma once

#include "fairino_planning_core/types.h"
#include <array>
#include <cstdint>

namespace fairino_planning {

class SobolSequence3D {
public:
    SobolSequence3D();

    Vector3d next();
    void reset(uint32_t index = 1);

private:
    std::array<std::array<uint32_t, 32>, 3> directions_{};
    std::array<uint32_t, 3> x_{{0U, 0U, 0U}};
    uint32_t index_{1U};
};

}  // namespace fairino_planning
