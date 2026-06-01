// src/tree/rrt_tree.cpp
#include "fairino_planning_core/tree/rrt_tree.h"
#include <queue>
#include <algorithm>

namespace fairino_planning {
namespace {

double wrappedDistanceSq(const JointConfig& a, const JointConfig& b) {
    return wrapToPi(a - b).squaredNorm();
}

double wrappedDistance(const JointConfig& a, const JointConfig& b) {
    return wrapToPi(a - b).norm();
}

}  // namespace

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
    index_dirty_ = true;
    return idx;
}

void RRTTree::rebuildIndex() {
    adaptor_ = std::make_unique<TreeAdaptor>(nodes_, count_);
    kdtree_ = std::make_unique<KDTree>(
        NUM_JOINTS, *adaptor_,
        nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree_->buildIndex();
    index_dirty_ = false;
}

int RRTTree::nearest(const JointConfig& q) const {
    // KDTree uses ordinary Euclidean distance and cannot represent angular
    // wrap-around. Use linear search so nearest-neighbor semantics match
    // wrapToPi joint deltas.
    int best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (int i = 0; i < count_; ++i) {
        double d = wrappedDistanceSq(nodes_[i].state, q);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

std::vector<int> RRTTree::nearRadius(const JointConfig& q, double radius) const {
    std::vector<int> result;
    double r2 = radius * radius;
    for (int i = 0; i < count_; ++i) {
        if (wrappedDistanceSq(nodes_[i].state, q) <= r2)
            result.push_back(i);
    }
    return result;
}

void RRTTree::propagateCost(int changed_idx) {
    std::queue<int> queue;
    queue.push(changed_idx);
    while (!queue.empty()) {
        int curr = queue.front(); queue.pop();
        for (int kid : nodes_[curr].children) {
            double nc = nodes_[curr].cost +
                        wrappedDistance(nodes_[curr].state, nodes_[kid].state);
            if (nc < nodes_[kid].cost - 1e-12) {
                nodes_[kid].cost = nc;
                queue.push(kid);
            }
        }
    }
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
