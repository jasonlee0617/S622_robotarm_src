#include "fairino_planning_core/aapf/sobol_sequence_3d.h"

#include <algorithm>
#include <cmath>

namespace fairino_planning {

namespace {

void initDimension(
    std::array<uint32_t, 32>& v,
    int s,
    uint32_t a,
    const std::array<uint32_t, 2>& m) {
    for (int j = 0; j < s; ++j) {
        v[j] = m[j] << (31 - j);
    }
    for (int j = s; j < 32; ++j) {
        uint32_t value = v[j - s] ^ (v[j - s] >> s);
        for (int k = 1; k < s; ++k) {
            if ((a >> (s - 1 - k)) & 1U) {
                value ^= v[j - k];
            }
        }
        v[j] = value;
    }
}

}  // namespace

SobolSequence3D::SobolSequence3D() {
    for (int j = 0; j < 32; ++j) {
        directions_[0][j] = 1U << (31 - j);
    }
    initDimension(directions_[1], 1, 0U, {1U, 0U});
    initDimension(directions_[2], 2, 1U, {1U, 3U});
}

void SobolSequence3D::reset(uint32_t index) {
    index_ = std::max<uint32_t>(1U, index);
    x_ = {{0U, 0U, 0U}};
    for (uint32_t i = 1; i < index_; ++i) {
        const int bit = __builtin_ctz(i);
        for (int d = 0; d < 3; ++d) {
            x_[d] ^= directions_[d][bit];
        }
    }
}

Vector3d SobolSequence3D::next() {
    const int bit = __builtin_ctz(index_);
    for (int d = 0; d < 3; ++d) {
        x_[d] ^= directions_[d][bit];
    }
    ++index_;
    constexpr double scale = 1.0 / 4294967296.0;
    return Vector3d(
        static_cast<double>(x_[0]) * scale,
        static_cast<double>(x_[1]) * scale,
        static_cast<double>(x_[2]) * scale);
}

}  // namespace fairino_planning
