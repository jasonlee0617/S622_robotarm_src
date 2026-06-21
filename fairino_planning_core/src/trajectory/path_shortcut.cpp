// src/trajectory/path_shortcut.cpp
#include "fairino_planning_core/trajectory/path_shortcut.h"
#include <algorithm>
#include <iostream>
#include <cmath>

namespace fairino_planning {

namespace {

JointConfig jointDelta(const JointConfig& from, const JointConfig& to) {
    // These waypoints are exported directly to the bounded MoveIt joints.
    // Do not turn a large bounded-joint motion into a wrapped shortcut.
    return to - from;
}

JointConfig interpolateJointLinear(
    const JointConfig& from, const JointConfig& to, double t) {
    return from + t * jointDelta(from, to);
}

double maxAbsJointDelta(const JointConfig& from, const JointConfig& to) {
    return jointDelta(from, to).cwiseAbs().maxCoeff();
}

bool isFiniteConfig(const JointConfig& q) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (!std::isfinite(q[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool PathOptimizer::isSegmentValid(
    const JointConfig& from, const JointConfig& to) const {

    if (!isFiniteConfig(from) || !isFiniteConfig(to)) {
        return false;
    }
    if (max_segment_joint_jump_rad_ > 0.0 &&
        maxAbsJointDelta(from, to) > max_segment_joint_jump_rad_) {
        return false;
    }
    if (collision_ && !collision_->isMotionValid(from, to, validation_dist_)) {
        return false;
    }
    if (!checkIntermediateOrientation(from, to, orientation_check_count_)) {
        return false;
    }
    return true;
}

bool PathOptimizer::checkIntermediateOrientation(
    const JointConfig& from, const JointConfig& to, int num_checks) const {

    for (int i = 0; i <= num_checks; ++i) {
        double t = static_cast<double>(i) / num_checks;
        JointConfig q_mid = interpolateJointLinear(from, to, t);
        if (!ori_checker_.check(q_mid, fk_)) {
            return false;
        }
    }
    return true;
}

double PathOptimizer::pathLength(const std::vector<JointConfig>& path) const {
    double len = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        len += jointDelta(path[i - 1], path[i]).norm();
    }
    return len;
}

std::vector<JointConfig> PathOptimizer::shortcutEnhanced(
    const std::vector<JointConfig>& path,
    int max_trials) const {

    if (path.size() <= 2) return path;

    std::vector<JointConfig> result = path;
    int n = static_cast<int>(result.size());

    std::uniform_int_distribution<int> dist;

    for (int trial = 0; trial < max_trials && n > 2; ++trial) {
        dist = std::uniform_int_distribution<int>(0, n - 1);
        int i = dist(rng_);
        int j = dist(rng_);
        if (i > j) std::swap(i, j);
        if (j - i <= 1) continue;

        if (isSegmentValid(result[i], result[j])) {
            result.erase(result.begin() + i + 1, result.begin() + j);
            n = static_cast<int>(result.size());
        }
    }

    return result;
}

std::vector<JointConfig> PathOptimizer::pathPull(
    const std::vector<JointConfig>& path,
    int max_trials) const {

    if (path.size() <= 2) return path;

    std::vector<JointConfig> result = path;
    int n = static_cast<int>(result.size());

    std::uniform_int_distribution<int> mid_dist;
    std::uniform_real_distribution<double> alpha_dist(pull_alpha_min_, pull_alpha_max_);

    for (int trial = 0; trial < max_trials && n > 2; ++trial) {
        mid_dist = std::uniform_int_distribution<int>(1, n - 2);
        int k = mid_dist(rng_);

        double alpha = alpha_dist(rng_);
        JointConfig q_new = interpolateJointLinear(result[k - 1], result[k + 1], alpha);
        q_new = limits_.clamp(q_new);

        if (!isSegmentValid(result[k-1], q_new)) continue;
        if (!isSegmentValid(q_new, result[k+1])) continue;

        double old_len = jointDelta(result[k - 1], result[k]).norm()
                       + jointDelta(result[k], result[k + 1]).norm();
        double new_len = jointDelta(result[k - 1], q_new).norm()
                       + jointDelta(q_new, result[k + 1]).norm();

        if (new_len < old_len - 1e-8) {
            result[k] = q_new;
        }
    }

    return result;
}

// ═══════════════════════════════════════════════════
// ★★★ 路径加密: 防止 MoveIt2 样条插值穿越障碍物 ★★★
// ═══════════════════════════════════════════════════
std::vector<JointConfig> PathOptimizer::densify(
    const std::vector<JointConfig>& path,
    double max_spacing) const {

    if (path.size() < 2) return path;

    std::vector<JointConfig> dense;
    dense.push_back(path[0]);

    for (size_t i = 1; i < path.size(); ++i) {
        double seg_dist = jointDelta(path[i - 1], path[i]).norm();

        if (seg_dist <= max_spacing) {
            // 间距足够小，直接添加
            dense.push_back(path[i]);
        } else {
            // 需要插入中间点
            int num_sub = static_cast<int>(std::ceil(seg_dist / max_spacing));
            for (int k = 1; k <= num_sub; ++k) {
                double t = static_cast<double>(k) / num_sub;
                JointConfig q_mid = interpolateJointLinear(path[i - 1], path[i], t);
                dense.push_back(q_mid);
            }
        }
    }

    return dense;
}

std::vector<JointConfig> PathOptimizer::optimize(
    const std::vector<JointConfig>& path,
    int shortcut_trials,
    int pull_trials) const {

    if (path.size() < 2) {
        return path;
    }

    double original_len = pathLength(path);
    // 第一轮: 短路优化
    auto result = shortcutEnhanced(path, shortcut_trials);
    double after_shortcut = pathLength(result);

    // 第二轮: 拉直优化
    result = pathPull(result, pull_trials);
    double after_pull = pathLength(result);
    const std::vector<JointConfig> fallback_path = result;

    // ★★★ 第三轮: 加密路径点 ★★★
    result = densify(result, densify_max_spacing_);

    // 加密后采用 all-or-nothing 验证。不能跳过碰撞点再拼接后续点，
    // 否则会生成没有被验证过的长段，严重时会把异常状态交给 FCL。
    if (collision_) {
        bool all_valid = isFiniteConfig(result.front()) && collision_->isStateValid(result.front());
        for (size_t i = 1; all_valid && i < result.size(); ++i) {
            all_valid = isFiniteConfig(result[i]) &&
                        collision_->isStateValid(result[i]) &&
                        collision_->isMotionValid(result[i - 1], result[i], validation_dist_);
            if (!all_valid) {
                std::cout << "  WARNING: densified point " << i
                          << " invalid, returning original planner path" << std::endl;
            }
        }
        if (!all_valid) {
            if (fail_open_return_original_) {
                // Densify is only a waypoint-export convenience step.  Fall
                // back to the last validated shortcut/pull path rather than
                // the original planner output.
                result = fallback_path;
            }
        }
    }

    std::cout << "  PathOptimizer: "
              << path.size() << " → " << result.size() << " waypoints, "
              << "length: " << original_len << " → "
              << after_shortcut << " (shortcut) → "
              << after_pull << " (pull) → "
              << pathLength(result) << " (densified)" << std::endl;

    return result;
}

}  // namespace fairino_planning
