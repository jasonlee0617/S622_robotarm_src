#include "fairino_planning_ros/pipeline/fairino_planning_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include <geometric_shapes/shapes.h>
#include <moveit/robot_state/conversions.h>
#include <pluginlib/class_list_macros.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include "fairino_planning_core/constraints/orientation_checker.h"
#include "fairino_planning_core/dh_kinematics.h"
#include "fairino_planning_core/engine/planner_engine.hpp"
#include "fairino_planning_core/trajectory/path_shortcut.h"
#include "fairino_planning_core/trajectory/trajectory_smoother.h"
#include "fairino_planning_ros/config/moveit_error_mapping.h"
#include "fairino_planning_ros/moveit_collision_checker.h"

namespace fairino_planning::v2 {

namespace {

ToolModel resolveToolModelFromGroupName(const std::string& group_name) {
    if (group_name == "robot_arm" || group_name == "gripper_arm" ||
        group_name.find("gripper") != std::string::npos) {
        return ToolModel::GRIPPER;
    }
    return ToolModel::FLANGE;
}

double jointPathLength(const std::vector<JointConfig>& path) {
    if (path.size() < 2U) {
        return 0.0;
    }
    double length = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        length += wrapToPi(path[i] - path[i - 1U]).norm();
    }
    return length;
}

bool validateJointPath(
    const std::vector<JointConfig>& path,
    const std::shared_ptr<MoveItCollisionChecker>& collision,
    double validation_distance,
    int* invalid_segment) {
    if (invalid_segment) {
        *invalid_segment = -1;
    }
    if (path.empty() || !collision) {
        return false;
    }
    for (size_t i = 0; i < path.size(); ++i) {
        if (!collision->isStateValid(path[i])) {
            if (invalid_segment) {
                *invalid_segment = static_cast<int>(i);
            }
            return false;
        }
        if (i > 0U &&
            !collision->isMotionValid(path[i - 1U], path[i], validation_distance)) {
            if (invalid_segment) {
                *invalid_segment = static_cast<int>(i - 1U);
            }
            return false;
        }
    }
    return true;
}

std::vector<JointConfig> decimatePathForExport(
    const std::vector<JointConfig>& path,
    const std::shared_ptr<MoveItCollisionChecker>& collision,
    double validation_distance) {
    if (path.size() <= 2U || !collision) {
        return path;
    }

    const double max_export_segment = std::max(0.02, validation_distance * 2.0);
    std::vector<JointConfig> out;
    out.reserve(path.size());
    size_t i = 0;
    out.push_back(path.front());
    while (i + 1U < path.size()) {
        size_t best = i + 1U;
        for (size_t j = path.size() - 1U; j > i + 1U; --j) {
            const double segment_len = wrapToPi(path[j] - path[i]).norm();
            if (segment_len <= max_export_segment &&
                collision->isStateValid(path[j]) &&
                collision->isMotionValid(path[i], path[j], validation_distance)) {
                best = j;
                break;
            }
        }
        out.push_back(path[best]);
        i = best;
    }
    return out;
}

}  // namespace

bool FairinoPlanningPipeline::solve(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit_msgs::msg::MotionPlanRequest& req,
    const std::string& group_name,
    const std::shared_ptr<PlanningAlgorithm>& algorithm,
    const PipelineOptions& options,
    planning_interface::MotionPlanResponse& res) const {
    if (!scene) {
        RCLCPP_ERROR(logger_, "PlanningScene is null in FairinoPlanningPipeline::solve()");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
        return false;
    }

    const auto* jmg = scene->getRobotModel()->getJointModelGroup(group_name);
    if (!jmg) {
        RCLCPP_ERROR(logger_, "Joint model group '%s' not found", group_name.c_str());
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
        return false;
    }

    if (!algorithm) {
        RCLCPP_ERROR(logger_, "Algorithm instance is null");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
        return false;
    }

    const ToolModel tool_model = resolveToolModelFromGroupName(group_name);

    moveit::core::RobotState start_state(scene->getRobotModel());
    start_state = scene->getCurrentState();
    if (!req.start_state.joint_state.name.empty()) {
        moveit::core::robotStateMsgToRobotState(req.start_state, start_state);
    }

    std::vector<double> start_vals;
    start_state.copyJointGroupPositions(jmg, start_vals);
    JointConfig q_start = JointConfig::Zero();
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(start_vals.size()); ++i) {
        q_start[i] = start_vals[i];
    }

    JointConfig q_goal = JointConfig::Zero();
    bool goal_found = false;

    if (!req.goal_constraints.empty()) {
        const auto& gc = req.goal_constraints[0];

        if (!gc.joint_constraints.empty()) {
            const auto& joint_names = jmg->getActiveJointModelNames();
            q_goal = q_start;
            for (const auto& jc : gc.joint_constraints) {
                for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(joint_names.size()); ++i) {
                    if (jc.joint_name == joint_names[i]) {
                        q_goal[i] = jc.position;
                        break;
                    }
                }
            }
            goal_found = true;
        }

        if (!goal_found &&
            (!gc.position_constraints.empty() || !gc.orientation_constraints.empty())) {
            geometry_msgs::msg::Pose target_pose;
            bool has_pos = false, has_ori = false;
            if (!gc.position_constraints.empty()) {
                const auto& pc = gc.position_constraints[0];
                if (!pc.constraint_region.primitive_poses.empty()) {
                    target_pose.position.x = pc.constraint_region.primitive_poses[0].position.x;
                    target_pose.position.y = pc.constraint_region.primitive_poses[0].position.y;
                    target_pose.position.z = pc.constraint_region.primitive_poses[0].position.z;
                    has_pos = true;
                }
            }
            if (!gc.orientation_constraints.empty()) {
                target_pose.orientation = gc.orientation_constraints[0].orientation;
                has_ori = true;
            }

            if (has_pos && has_ori) {
                Eigen::Isometry3d iso;
                tf2::fromMsg(target_pose, iso);
                Transform4d T_target = iso.matrix();

                FairinoIK ik_solver;
                IKSelector ik_selector(options.ik_selector_params);
                auto ik_result = ik_solver.solve(T_target, tool_model);
                if (ik_result.success && !ik_result.solutions.empty()) {
                    IKBranchHint hint{};
                    // This is a global pose-goal selection, not a Cartesian waypoint
                    // stream. Keep seed-distance ranking, but do not enable the hard
                    // continuity guard that is meant for consecutive IK calls.
                    hint.valid = false;
                    bool branch_switched = false;

                    std::vector<JointConfig> remaining = ik_result.solutions;
                    while (!remaining.empty()) {
                        IKQualityMetrics metrics;
                        auto best = ik_selector.select(
                            remaining, q_start, tool_model, &hint, &metrics);
                        if (!best) {
                            break;
                        }

                        branch_switched =
                            (std::sin((*best)[4]) * std::sin(q_start[4]) < 0.0);
                        q_goal = *best;
                        goal_found = true;
                        RCLCPP_INFO(
                            logger_,
                            "IK candidate accepted: dh_sigma=%.4f dh_cond=%.2f margin=%.4f branch_switched=%s",
                            metrics.sigma_min,
                            metrics.cond,
                            metrics.min_joint_margin,
                            branch_switched ? "true" : "false");
                        break;
                    }
                }

            }
        }
    }

    if (!goal_found) {
        RCLCPP_ERROR(logger_, "No valid goal constraints found.");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
        return false;
    }

    auto collision = std::make_shared<MoveItCollisionChecker>(scene, group_name);
    PlannerEngine engine(algorithm);
    engine.setCollisionChecker(collision);
    engine.configure(options.planner_config);

    DHKinematics fk;
    const auto T_start = fk.fkine(q_start, tool_model);
    const auto T_goal = fk.fkine(q_goal, tool_model);
    const Vector3d p_start = T_start.block<3, 1>(0, 3);
    const Vector3d p_goal = T_goal.block<3, 1>(0, 3);
    const RotMatrix3d R_target = T_goal.block<3, 3>(0, 0);

    Vector3d obs_origin = options.default_obstacle_origin;
    Vector3d obs_size = options.default_obstacle_size;
    std::vector<ObstacleInfo> obstacles;
    size_t filtered_obstacles = 0;

    if (options.use_multi_obstacle_input) {
        const auto& collision_objects = scene->getWorld()->getObjectIds();
        obstacles.reserve(collision_objects.size());
        for (const auto& obj_id : collision_objects) {
            const auto obj = scene->getWorld()->getObject(obj_id);
            if (!obj || obj->shapes_.empty() || obj->shape_poses_.empty()) {
                ++filtered_obstacles;
                continue;
            }

            bool object_added = false;
            const size_t shape_count = std::min(obj->shapes_.size(), obj->shape_poses_.size());
            for (size_t i = 0; i < shape_count; ++i) {
                const auto* shape_raw = obj->shapes_[i].get();
                Eigen::Vector3d size(Eigen::Vector3d::Zero());
                const char* shape_type = "unknown";

                if (const auto* box = dynamic_cast<const shapes::Box*>(shape_raw)) {
                    size = Eigen::Vector3d(box->size[0], box->size[1], box->size[2]);
                    shape_type = "box";
                } else if (const auto* sphere = dynamic_cast<const shapes::Sphere*>(shape_raw)) {
                    const double d = 2.0 * sphere->radius;
                    size = Eigen::Vector3d(d, d, d);
                    shape_type = "sphere";
                } else if (const auto* cylinder = dynamic_cast<const shapes::Cylinder*>(shape_raw)) {
                    const double d = 2.0 * cylinder->radius;
                    size = Eigen::Vector3d(d, d, cylinder->length);
                    shape_type = "cylinder";
                } else {
                    ++filtered_obstacles;
                    continue;
                }

                if (size.minCoeff() < options.min_obstacle_size_threshold) {
                    RCLCPP_DEBUG(logger_,
                        "  obstacle '%s' shape[%zu] type=%s filtered (size too small: %.4f)",
                        obj_id.c_str(), i, shape_type, size.minCoeff());
                    ++filtered_obstacles;
                    continue;
                }

                const auto& obj_pose = obj->shape_poses_[i];
                ObstacleInfo info;
                info.center = obj_pose.translation();
                info.size = size;
                obstacles.push_back(info);
                object_added = true;

                RCLCPP_DEBUG(logger_,
                    "  obstacle '%s' shape[%zu] type=%s center=[%.4f,%.4f,%.4f] size=[%.4f,%.4f,%.4f]",
                    obj_id.c_str(), i, shape_type,
                    info.center.x(), info.center.y(), info.center.z(),
                    info.size.x(), info.size.y(), info.size.z());
            }
            if (!object_added && shape_count == 0U) {
                ++filtered_obstacles;
            }
        }
    }

    PlanRequestCore plan_req;
    plan_req.q_start = q_start;
    plan_req.q_goal = q_goal;
    plan_req.p_start = p_start;
    plan_req.p_goal = p_goal;
    plan_req.R_target = R_target;
    plan_req.obs_origin = obs_origin;
    plan_req.obs_size = obs_size;
    plan_req.obstacles = obstacles;
    plan_req.use_multi_obstacle = !obstacles.empty();
    plan_req.tool_model = tool_model;

    if (plan_req.use_multi_obstacle) {
        plan_req.obs_origin = obstacles.front().center;
        plan_req.obs_size = obstacles.front().size;
    }

    RCLCPP_INFO(
        logger_,
        "Planning obstacles aggregated: obs_count=%zu filtered=%zu multi_obs_enabled=%s",
        plan_req.obstacles.size(),
        filtered_obstacles,
        plan_req.use_multi_obstacle ? "true" : "false");

    const std::string planner_name = algorithm->name();
    RCLCPP_INFO(
        logger_,
        "Planner branch selected: %s %s",
        planner_name.c_str(),
        plan_req.use_multi_obstacle ? "multi" : "single");

    const auto result = engine.plan(plan_req);

    if (!result.success) {
        RCLCPP_WARN(
            logger_,
            "Fairino plan failure: planner=%s planning_time=%.6f path_points=%zu num_nodes=%d iterations=%d message=%s",
            planner_name.c_str(),
            result.planning_time,
            result.path.size(),
            result.num_nodes,
            result.iterations,
            result.message.c_str());
        RCLCPP_INFO(logger_, "PathOptimizer: skipped (planning failed)");
        RCLCPP_INFO(logger_, "TrajectorySmoother: skipped (planning failed before trajectory export)");
        res.error_code_.val = toMoveItError(result.failure_code);
        return false;
    }

    auto path = result.path;
    const size_t raw_path_points = path.size();
    const double raw_path_cost = std::isfinite(result.path_cost)
        ? result.path_cost
        : jointPathLength(path);
    RCLCPP_INFO(
        logger_,
        "Fairino plan result: planner=%s planning_time=%.6f path_points=%zu path_cost=%.6f num_nodes=%d iterations=%d",
        planner_name.c_str(),
        result.planning_time,
        path.size(),
        result.path_cost,
        result.num_nodes,
        result.iterations);
    if (path.size() > 2 && options.enable_path_optimizer) {
        OrientationPolicy ori_policy;
        OrientationChecker ori_checker(ori_policy);
        ori_checker.setTargetOrientation(R_target);
        ori_checker.setTargetPosition(p_goal);
        ori_checker.setToolModel(tool_model);
        PathOptimizer optimizer;
        optimizer.setCollisionChecker(collision.get());
        optimizer.setOrientationChecker(ori_checker);
        optimizer.setJointLimits(JointLimits{});
        optimizer.setValidationDistance(options.optimizer_validation_distance);
        optimizer.setFailOpenReturnOriginal(options.optimizer_fail_open_return_original);
        optimizer.setDensifyMaxSpacing(options.optimizer_densify_max_spacing);
        optimizer.setPullAlphaRange(
            options.optimizer_pull_alpha_min,
            options.optimizer_pull_alpha_max);
        optimizer.setOrientationCheckCount(options.optimizer_orientation_check_count);
        const size_t optimizer_input_points = path.size();
        const auto optimizer_start = std::chrono::steady_clock::now();
        path = optimizer.optimize(
            path,
            options.optimizer_shortcut_trials,
            options.optimizer_pull_trials);
        const double optimizer_time_s =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - optimizer_start).count();
        RCLCPP_INFO(
            logger_,
            "PathOptimizer: input_points=%zu output_points=%zu time_s=%.6f",
            optimizer_input_points,
            path.size(),
            optimizer_time_s);
    } else if (path.size() > 2) {
        RCLCPP_INFO(logger_, "PathOptimizer disabled by planner.enable_path_optimizer=false");
    }

    int invalid_segment = -1;
    const double final_validation_distance = std::max(1e-4, options.final_validation_distance);
    bool final_path_valid = validateJointPath(
        path, collision, final_validation_distance, &invalid_segment);
    RCLCPP_INFO(
        logger_,
        "FinalPathValidator: points=%zu valid=%s invalid_segment=%d validation_distance=%.4f fail_open=%s",
        path.size(),
        final_path_valid ? "true" : "false",
        invalid_segment,
        final_validation_distance,
        options.final_validation_fail_open ? "true" : "false");
    RCLCPP_INFO(
        logger_,
        "PathQuality: planner=%s raw_points=%zu raw_cost=%.6f optimized_points=%zu optimized_length=%.6f final_valid=%s",
        planner_name.c_str(),
        raw_path_points,
        raw_path_cost,
        path.size(),
        jointPathLength(path),
        final_path_valid ? "true" : "false");
    if (!final_path_valid && !options.final_validation_fail_open) {
        RCLCPP_ERROR(
            logger_,
            "FinalPathValidator rejected path; refusing trajectory export and execution");
        RCLCPP_INFO(logger_, "TrajectorySmoother: skipped (final path validation failed)");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
        return false;
    }

    const auto decimator_start = std::chrono::steady_clock::now();
    const size_t decimator_input_points = path.size();
    auto export_path = decimatePathForExport(path, collision, final_validation_distance);
    int decimator_invalid_segment = -1;
    const bool decimator_valid = validateJointPath(
        export_path, collision, final_validation_distance, &decimator_invalid_segment);
    const double decimator_time_s =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decimator_start).count();
    const double export_max_segment = std::max(0.02, final_validation_distance * 2.0);
    RCLCPP_INFO(
        logger_,
        "TrajectoryExportDecimator: input_points=%zu output_points=%zu validated=%s length=%.6f invalid_segment=%d max_segment=%.4f time_s=%.6f",
        decimator_input_points,
        export_path.size(),
        decimator_valid ? "true" : "false",
        jointPathLength(export_path),
        decimator_invalid_segment,
        export_max_segment,
        decimator_time_s);
    if (!decimator_valid && !options.final_validation_fail_open) {
        RCLCPP_ERROR(
            logger_,
            "TrajectoryExportDecimator produced invalid export path; refusing trajectory export");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
        return false;
    }
    path = std::move(export_path);

    const auto export_start = std::chrono::steady_clock::now();
    auto traj = std::make_shared<robot_trajectory::RobotTrajectory>(
        scene->getRobotModel(), group_name);
    for (size_t i = 0; i < path.size(); ++i) {
        moveit::core::RobotState state(scene->getRobotModel());
        state = scene->getCurrentState();
        std::vector<double> vals(path[i].data(), path[i].data() + NUM_JOINTS);
        state.setJointGroupPositions(jmg, vals);
        state.update();
        const double dt = (i == 0) ? 0.0 : options.trajectory_waypoint_dt;
        traj->addSuffixWayPoint(state, dt);
    }
    const double export_time_s =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - export_start).count();
    RCLCPP_INFO(
        logger_,
        "TrajectoryExport: points=%zu time_s=%.6f",
        path.size(),
        export_time_s);

    res.trajectory_ = traj;
    res.planning_time_ = result.planning_time;
    RCLCPP_INFO(
        logger_,
        "TrajectorySmoother: skipped (current pipeline exports waypoint_dt=%.3f without smoother)",
        options.trajectory_waypoint_dt);
    res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return true;
}

}  // namespace fairino_planning::v2
