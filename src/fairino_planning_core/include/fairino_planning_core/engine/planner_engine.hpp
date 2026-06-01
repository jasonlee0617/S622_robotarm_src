#pragma once

#include <memory>

#include "fairino_planning_core/algorithms/planning_algorithm.h"
#include "fairino_planning_core/collision/collision_interface.h"

namespace fairino_planning {

class PlannerEngine {
public:
    explicit PlannerEngine(std::shared_ptr<PlanningAlgorithm> algorithm = nullptr)
        : algorithm_(std::move(algorithm)) {}

    void setAlgorithm(std::shared_ptr<PlanningAlgorithm> algorithm) {
        algorithm_ = std::move(algorithm);
    }

    void setCollisionChecker(std::shared_ptr<CollisionInterface> checker) {
        collision_checker_ = std::move(checker);
        if (algorithm_) {
            algorithm_->setCollisionChecker(collision_checker_);
        }
    }

    void configure(const PlannerConfig& config) {
        config_ = config;
        if (algorithm_) {
            algorithm_->configure(config_);
        }
    }

    PlanResultCore plan(const PlanRequestCore& request) const {
        if (!algorithm_) {
            PlanResultCore result;
            result.success = false;
            result.failure_code = PlanningFailureCode::kInternalError;
            result.message = "PlannerEngine has no algorithm configured";
            return result;
        }
        return algorithm_->plan(request);
    }

private:
    std::shared_ptr<PlanningAlgorithm> algorithm_;
    std::shared_ptr<CollisionInterface> collision_checker_;
    PlannerConfig config_;
};

}  // namespace fairino_planning

