// include/fairino_planning_core/trajectory/path_shortcut.h
#pragma once

#include "fairino_planning_core/types.h"
#include "fairino_planning_core/collision/collision_interface.h"
#include "fairino_planning_core/constraints/orientation_checker.h"
#include "fairino_planning_core/dh_kinematics.h"
#include <vector>
#include <random>

namespace fairino_planning {

class PathOptimizer {
public:
    PathOptimizer() = default;

    void setCollisionChecker(CollisionInterface* checker) { collision_ = checker; }
    void setOrientationChecker(const OrientationChecker& ori) { ori_checker_ = ori; }
    void setJointLimits(const JointLimits& limits) { limits_ = limits; }
    void setValidationDistance(double d) { validation_dist_ = d; }
    void setFailOpenReturnOriginal(bool enabled) { fail_open_return_original_ = enabled; }
    void setDensifyMaxSpacing(double spacing) { densify_max_spacing_ = spacing; }
    void setPullAlphaRange(double min_alpha, double max_alpha) {
        pull_alpha_min_ = min_alpha;
        pull_alpha_max_ = max_alpha;
    }
    void setOrientationCheckCount(int count) { orientation_check_count_ = count; }
    void setMaxSegmentJointJump(double jump) { max_segment_joint_jump_rad_ = jump; }

    std::vector<JointConfig> shortcutEnhanced(
        const std::vector<JointConfig>& path,
        int max_trials = 200) const;

    std::vector<JointConfig> pathPull(
        const std::vector<JointConfig>& path,
        int max_trials = 100) const;

    /// ★ 路径加密：确保相邻点间距不超过 max_spacing
    std::vector<JointConfig> densify(
        const std::vector<JointConfig>& path,
        double max_spacing = 0.05) const;

    /// 综合优化: shortcut + pull + densify
    std::vector<JointConfig> optimize(
        const std::vector<JointConfig>& path,
        int shortcut_trials = 200,
        int pull_trials = 100) const;

private:
    CollisionInterface* collision_ = nullptr;
    OrientationChecker ori_checker_;
    JointLimits limits_;
    DHKinematics fk_;
    double validation_dist_ = 0.05;
    double densify_max_spacing_ = 0.05;
    double pull_alpha_min_ = 0.1;
    double pull_alpha_max_ = 0.9;
    double max_segment_joint_jump_rad_ = 1.35;
    int orientation_check_count_ = 5;
    bool fail_open_return_original_ = true;
    mutable std::mt19937 rng_{42};

    bool isSegmentValid(const JointConfig& from, const JointConfig& to) const;
    double pathLength(const std::vector<JointConfig>& path) const;
    bool checkIntermediateOrientation(
        const JointConfig& from, const JointConfig& to, int num_checks = 5) const;
};

}  // namespace fairino_planning
