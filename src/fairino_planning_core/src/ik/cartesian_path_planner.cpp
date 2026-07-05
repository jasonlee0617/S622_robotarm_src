#include "fairino_planning_core/ik/cartesian_path_planner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace fairino_planning {

CartesianPathPlanner::CartesianPathPlanner(const IKSelectParams& selector_params,
                                           const AnalyticalIKParams& analytical_params,
                                           const CartesianPathPlannerParams& planner_params)
    : selector_(selector_params), ik_(analytical_params), params_(planner_params) {}

bool CartesianPathPlanner::sameJointConfig(const JointConfig& a, const JointConfig& b, double tol) {
    return (a - b).norm() <= tol;
}

namespace {
std::string dominantRejectCode(const std::vector<IKCandidateDiagnostic>& diagnostics) {
    if (diagnostics.empty()) return "no_candidate_after_selector";
    std::map<std::string, int> counts;
    for (const auto& d : diagnostics) {
        if (d.passed_hard_filter) continue;
        ++counts[toString(d.reject_reason)];
    }
    if (counts.empty()) return "no_candidate_after_selector";
    return std::max_element(
        counts.begin(),
        counts.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}
}  // namespace

CartesianIKPathResult CartesianPathPlanner::plan(const CartesianIKPathRequest& request) const {
    CartesianIKPathResult result;
    if (request.waypoints.empty()) {
        result.success = true;
        result.fraction = 1.0;
        return result;
    }

    std::vector<std::vector<Node>> layers;
    layers.reserve(request.waypoints.size());

    for (size_t i = 0; i < request.waypoints.size(); ++i) {
        const auto ik_result = ik_.solve(request.waypoints[i], request.tool_model);
        if (!ik_result.success || ik_result.solutions.empty()) {
            result.failed_index = static_cast<int>(i);
            result.fraction = static_cast<double>(i) / static_cast<double>(request.waypoints.size());
            result.failed_category = ik_result.failure_category;
            result.failed_code = ik_result.failure_code;
            result.message = "IK failed: category=" + std::string(toString(result.failed_category))
                           + " code=" + result.failed_code;
            result.has_failed_ik_result = true;
            result.failed_waypoint = request.waypoints[i];
            result.failed_ik_result = ik_result;
            return result;
        }

        std::vector<Node> next_layer;
        const bool first_layer = layers.empty();

        if (first_layer) {
            IKBranchHint hint{};
            hint.valid = true;
            hint.q_last = request.q_start;
            IKSelectionRequest selection_request;
            selection_request.solutions = &ik_result.solutions;
            selection_request.seed = request.q_start;
            selection_request.target_pose = request.waypoints[i];
            selection_request.tool_model = request.tool_model;
            selection_request.task_profile = request.task_profile;
            selection_request.hint = &hint;
            auto selection = selector_.select(selection_request);
            const auto& diagnostics = selection.diagnostics;

            for (const auto& d : diagnostics) {
                if (!d.passed_hard_filter) continue;
                Node n;
                n.q = d.q;
                n.cost = d.total_cost + d.dq_norm;
                n.prev = -1;
                next_layer.push_back(n);
            }
            if (next_layer.empty()) {
                result.failure_diagnostics = diagnostics;
            }
        } else {
            const auto& prev_layer = layers.back();
            std::vector<IKCandidateDiagnostic> layer_failure_diagnostics;
            for (size_t p = 0; p < prev_layer.size(); ++p) {
                IKBranchHint hint{};
                hint.valid = true;
                hint.q_last = prev_layer[p].q;

                IKSelectionRequest selection_request;
                selection_request.solutions = &ik_result.solutions;
                selection_request.seed = prev_layer[p].q;
                selection_request.target_pose = request.waypoints[i];
                selection_request.tool_model = request.tool_model;
                selection_request.task_profile = request.task_profile;
                selection_request.hint = &hint;
                auto selection = selector_.select(selection_request);
                const auto& diagnostics = selection.diagnostics;

                layer_failure_diagnostics.insert(
                    layer_failure_diagnostics.end(), diagnostics.begin(), diagnostics.end());

                for (const auto& d : diagnostics) {
                    if (!d.passed_hard_filter) continue;
                    const double candidate_cost =
                        prev_layer[p].cost + d.total_cost + d.dq_norm + d.max_abs_dq;

                    bool merged = false;
                    for (auto& existing : next_layer) {
                        if (sameJointConfig(existing.q, d.q, 1e-7)) {
                            if (candidate_cost < existing.cost) {
                                existing.cost = candidate_cost;
                                existing.prev = static_cast<int>(p);
                            }
                            merged = true;
                            break;
                        }
                    }
                    if (!merged) {
                        Node n;
                        n.q = d.q;
                        n.cost = candidate_cost;
                        n.prev = static_cast<int>(p);
                        next_layer.push_back(n);
                    }
                }
            }
            if (next_layer.empty()) {
                result.failure_diagnostics = std::move(layer_failure_diagnostics);
            }
        }

        if (next_layer.empty()) {
            result.failed_index = static_cast<int>(i);
            result.fraction = static_cast<double>(i) / static_cast<double>(request.waypoints.size());
            result.failed_category = IKFailureCategory::kCandidateFiltered;
            result.failed_code = dominantRejectCode(result.failure_diagnostics);
            result.message = "IK failed: category=" + std::string(toString(result.failed_category))
                           + " code=" + result.failed_code;
            return result;
        }

        std::sort(next_layer.begin(), next_layer.end(),
                  [](const Node& a, const Node& b) { return a.cost < b.cost; });
        const size_t max_nodes = static_cast<size_t>(std::max(1, params_.max_graph_nodes_per_layer));
        if (next_layer.size() > max_nodes) next_layer.resize(max_nodes);
        layers.push_back(std::move(next_layer));
    }

    int best_idx = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    const auto& last = layers.back();
    for (size_t i = 0; i < last.size(); ++i) {
        if (last[i].cost < best_cost) {
            best_cost = last[i].cost;
            best_idx = static_cast<int>(i);
        }
    }

    result.path.resize(layers.size());
    for (int layer = static_cast<int>(layers.size()) - 1; layer >= 0; --layer) {
        result.path[static_cast<size_t>(layer)] = layers[static_cast<size_t>(layer)][best_idx].q;
        best_idx = layers[static_cast<size_t>(layer)][best_idx].prev;
        if (layer > 0 && best_idx < 0) {
                result.failed_index = layer;
                result.fraction = static_cast<double>(layer) / static_cast<double>(request.waypoints.size());
                result.failed_category = IKFailureCategory::kInternal;
                result.failed_code = "predecessor_chain_broke";
                result.message = "internal predecessor chain broke";
                result.path.clear();
                return result;
        }
    }

    result.success = true;
    result.fraction = 1.0;
    return result;
}

}  // namespace fairino_planning
