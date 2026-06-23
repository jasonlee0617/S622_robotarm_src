// src/tree/rrt_tree.cpp
#include "fairino_planning_core/tree/rrt_tree.h"
#include <queue>
#include <algorithm>
#include <cmath>

namespace fairino_planning {

RRTTree::RRTTree(int reserve_size) {
    nodes_.resize(reserve_size);
    count_ = 0;
}

int RRTTree::addNode(const JointConfig& state, int parent, double cost) {
    if (count_ >= static_cast<int>(nodes_.size())) {
        nodes_.resize(nodes_.size() * 2);
    }
    int idx = count_++;
    nodes_[idx].state    = state;
    nodes_[idx].parent   = parent;
    nodes_[idx].cost     = cost;
    nodes_[idx].children.clear();

    if (parent >= 0) {
        nodes_[parent].children.push_back(idx);
    }
    return idx;
}

int RRTTree::nearest(const JointConfig& q) const {
    int best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (int i = 0; i < count_; ++i) {
        double d = (nodes_[i].state - q).squaredNorm();
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

std::vector<int> RRTTree::nearRadius(const JointConfig& q, double radius) const {
    std::vector<int> result;
    double r2 = radius * radius;
    for (int i = 0; i < count_; ++i) {
        if ((nodes_[i].state - q).squaredNorm() <= r2)
            result.push_back(i);
    }
    return result;
}

void RRTTree::propagateCost(int changed_idx) {
    if (changed_idx < 0 || changed_idx >= count_) return;
    std::queue<int> queue;
    queue.push(changed_idx);
    while (!queue.empty()) {
        int curr = queue.front(); queue.pop();
        for (int kid : nodes_[curr].children) {
            double nc = nodes_[curr].cost +
                        (nodes_[curr].state - nodes_[kid].state).norm();
            nodes_[kid].cost = nc;
            queue.push(kid);
        }
    }
}

bool RRTTree::reparent(int child_idx, int new_parent_idx, double new_cost) {
    if (child_idx <= 0 || child_idx >= count_ || new_parent_idx < 0 ||
        new_parent_idx >= count_ || !std::isfinite(new_cost) || new_cost < 0.0) {
        return false;
    }
    const int old_parent_idx = nodes_[child_idx].parent;
    if (old_parent_idx < 0 || old_parent_idx == new_parent_idx) return false;
    for (int ancestor = new_parent_idx; ancestor >= 0; ancestor = nodes_[ancestor].parent) {
        if (ancestor == child_idx) return false;
    }

    auto& old_children = nodes_[old_parent_idx].children;
    const auto old_child = std::find(old_children.begin(), old_children.end(), child_idx);
    auto& new_children = nodes_[new_parent_idx].children;
    if (old_child == old_children.end() ||
        std::find(new_children.begin(), new_children.end(), child_idx) != new_children.end()) {
        return false;
    }

    old_children.erase(old_child);
    new_children.push_back(child_idx);
    nodes_[child_idx].parent = new_parent_idx;
    nodes_[child_idx].cost = new_cost;
    propagateCost(child_idx);
    return true;
}

std::vector<JointConfig> RRTTree::backtrack(int leaf_idx) const {
    std::vector<JointConfig> path;
    int idx = leaf_idx;
    while (idx >= 0) {
        path.push_back(nodes_[idx].state);
        idx = nodes_[idx].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

}  // namespace fairino_planning
