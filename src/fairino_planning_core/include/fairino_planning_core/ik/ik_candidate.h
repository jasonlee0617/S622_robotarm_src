#pragma once
#include "fairino_planning_core/types.h"
#include <limits>
#include <string>
#include <vector>

namespace fairino_planning {

// ==========================================================================
// IKQualityMetrics
// ==========================================================================
struct IKQualityMetrics {
    double sigma_min{0.0};
    double cond{std::numeric_limits<double>::infinity()};
    double min_joint_margin{0.0};
    double wrist_sin_abs{0.0};
    double elbow_sin_abs{0.0};
    double base_radius{0.0};
};

// ==========================================================================
// BranchKey
// ==========================================================================
struct BranchKey {
    int shoulder{-2};
    int elbow{-2};
    int wrist{-2};
    int base_sector{-1};

    bool operator==(const BranchKey& o) const {
        return shoulder == o.shoulder && elbow == o.elbow
            && wrist == o.wrist && base_sector == o.base_sector;
    }
    bool operator!=(const BranchKey& o) const { return !(*this == o); }
    bool valid() const { return shoulder != -2 && elbow != -2 && wrist != -2 && base_sector >= 0; }
};

inline BranchKey inferBranchKey(const JointConfig& q) {
    BranchKey b;
    b.shoulder = (std::cos(q[0]) >= 0.0) ? 1 : -1;
    b.elbow    = (std::sin(q[2]) >= 0.0) ? 1 : -1;
    b.wrist    = (std::sin(q[4]) >= 0.0) ? 1 : -1;
    double wq = wrapToPi(q[0]);
    b.base_sector = static_cast<int>(std::floor((wq + M_PI) / (M_PI / 4.0)));
    if (b.base_sector >= 8) b.base_sector = 7;
    if (b.base_sector < 0) b.base_sector = 0;
    return b;
}

// ==========================================================================
// IKBranchHint — kept for backward compat with FairinoIKPlugin
// ==========================================================================
struct IKBranchHint {
    bool valid{false};
    JointConfig q_last{JointConfig::Zero()};
};

// ==========================================================================
// LinkPostureFeatures
// ==========================================================================
struct LinkPostureFeatures {
    double shoulder_z{0.0};
    double elbow_z{0.0};
    double wrist_z{0.0};
    double upper_arm_min_z{0.0};
    double forearm_min_z{0.0};
    double wrist_chain_min_z{0.0};
    double min_arm_z{0.0};
};

// ==========================================================================
// WristPostureFeatures
// ==========================================================================
struct WristPostureFeatures {
    double q4_inner_amount{0.0};
    double q4_positive_amount{0.0};
    double q5_singularity_amount{0.0};
    double forearm_tool_angle{0.0};
    double wrist_fold_amount{0.0};
    bool wrist_flip_sign{false};
};

// ==========================================================================
// MotionFeatures
// ==========================================================================
struct MotionFeatures {
    JointConfig dq = JointConfig::Zero();
    double dq_norm{0.0};
    double max_abs_dq{0.0};
};

// ==========================================================================
// ManipulabilityFeatures — Yoshikawa measure + condition
// ==========================================================================
struct ManipulabilityFeatures {
    double mu{0.0};            // sigma_min (dominant singular value)
    double yoshikawa{0.0};     // sqrt(det(J·J^T)) ≈ ∏ σ_i
    double inv_condition{0.0}; // σ_min / σ_max ∈ [0,1]
};

// ==========================================================================
// IKCandidate
// ==========================================================================
struct IKCandidate {
    JointConfig q_raw = JointConfig::Zero();
    JointConfig q = JointConfig::Zero();
    BranchKey branch;
    IKQualityMetrics metrics;
    ManipulabilityFeatures manipulability;
    LinkPostureFeatures link;
    WristPostureFeatures wrist;
    MotionFeatures motion;
};

}  // namespace fairino_planning
