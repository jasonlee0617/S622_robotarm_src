#include "fairino_planning_core/dh_kinematics.h"
#include "fairino_planning_core/ik/cartesian_path_planner.h"
#include "fairino_planning_ros/config/parameter_loader.hpp"

#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/srv/get_cartesian_path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fairino_planning {
namespace {

Transform4d poseToEigen(const geometry_msgs::msg::Pose& pose) {
    Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    q.normalize();
    Transform4d T = Transform4d::Identity();
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T(0, 3) = pose.position.x;
    T(1, 3) = pose.position.y;
    T(2, 3) = pose.position.z;
    return T;
}

ToolModel toolModelFromLink(const std::string& link) {
    if (link.find("grasp") != std::string::npos ||
        link.find("gripper") != std::string::npos ||
        link.find("tool") != std::string::npos) {
        return ToolModel::GRIPPER;
    }
    return ToolModel::FLANGE;
}

bool extractStartJointConfig(const sensor_msgs::msg::JointState& js,
                             const std::vector<std::string>& joint_names,
                             JointConfig& q) {
    q.setZero();
    for (size_t i = 0; i < joint_names.size(); ++i) {
        auto it = std::find(js.name.begin(), js.name.end(), joint_names[i]);
        if (it == js.name.end()) return false;
        const size_t idx = static_cast<size_t>(std::distance(js.name.begin(), it));
        if (idx >= js.position.size()) return false;
        q[static_cast<int>(i)] = js.position[idx];
    }
    return true;
}

double maxPointDeltaFromStart(const trajectory_msgs::msg::JointTrajectoryPoint& point,
                              const JointConfig& q_start) {
    if (point.positions.size() < static_cast<size_t>(NUM_JOINTS)) {
        return std::numeric_limits<double>::infinity();
    }
    double out = 0.0;
    for (int i = 0; i < NUM_JOINTS; ++i) {
        out = std::max(out, std::abs(point.positions[static_cast<size_t>(i)] - q_start[i]));
    }
    return out;
}

std::string jointConfigToDegreesString(const JointConfig& q) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << "[";
    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (i > 0) ss << ",";
        ss << q[i] * 180.0 / M_PI;
    }
    ss << "]";
    return ss.str();
}

std::string transformToPoseString(const Transform4d& T) {
    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    q.normalize();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4)
       << "pos=[" << T(0, 3) << "," << T(1, 3) << "," << T(2, 3) << "] "
       << "quat=[" << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "]";
    return ss.str();
}

std::string limitViolationString(const IKResult::LimitRejectInfo& reject) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    bool any = false;
    for (int i = 0; i < NUM_JOINTS; ++i) {
        const double low = reject.lower_violation[i];
        const double up = reject.upper_violation[i];
        if (low <= 0.0 && up <= 0.0) continue;
        if (any) ss << ",";
        ss << "j" << (i + 1);
        if (low > 0.0) {
            ss << ":low+" << low << "rad/" << (low * 180.0 / M_PI) << "deg";
        }
        if (up > 0.0) {
            ss << ":up+" << up << "rad/" << (up * 180.0 / M_PI) << "deg";
        }
        any = true;
    }
    return any ? ss.str() : "none";
}

std::vector<Transform4d> interpolateWaypoints(
    const Transform4d& start,
    const std::vector<geometry_msgs::msg::Pose>& request_waypoints,
    double max_step,
    double min_step) {
    std::vector<Transform4d> out;
    Transform4d from = start;
    const double step = std::max(max_step, std::max(min_step, 1e-6));

    for (const auto& target_pose : request_waypoints) {
        const Transform4d to = poseToEigen(target_pose);
        const Eigen::Vector3d p0 = from.block<3, 1>(0, 3);
        const Eigen::Vector3d p1 = to.block<3, 1>(0, 3);
        Eigen::Quaterniond q0(from.block<3, 3>(0, 0));
        Eigen::Quaterniond q1(to.block<3, 3>(0, 0));
        q0.normalize();
        q1.normalize();

        const int steps = std::max(1, static_cast<int>(std::ceil((p1 - p0).norm() / step)));
        for (int i = 1; i <= steps; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(steps);
            Transform4d Ti = Transform4d::Identity();
            Ti.block<3, 1>(0, 3) = (1.0 - t) * p0 + t * p1;
            Ti.block<3, 3>(0, 0) = q0.slerp(t, q1).toRotationMatrix();
            out.push_back(Ti);
        }
        from = to;
    }
    return out;
}

struct CartesianServerParams {
    CartesianPathPlannerParams planner;
    double trajectory_waypoint_dt{0.10};
    double min_cartesian_step_m{1.0e-4};
    bool include_start_state_point{true};
    double start_state_guard_tolerance_rad{0.005};
};

int getIntParam(const rclcpp::Node::SharedPtr& node, const std::string& key, int fallback) {
    if (!node->has_parameter(key)) node->declare_parameter<int>(key, fallback);
    return node->get_parameter(key).as_int();
}

double getDoubleParam(const rclcpp::Node::SharedPtr& node, const std::string& key, double fallback) {
    if (!node->has_parameter(key)) node->declare_parameter<double>(key, fallback);
    return node->get_parameter(key).as_double();
}

bool getBoolParam(const rclcpp::Node::SharedPtr& node, const std::string& key, bool fallback) {
    if (!node->has_parameter(key)) node->declare_parameter<bool>(key, fallback);
    return node->get_parameter(key).as_bool();
}

CartesianServerParams loadCartesianServerParams(const rclcpp::Node::SharedPtr& node) {
    CartesianServerParams params;
    const std::string ns = "fairino.cartesian_path_planner.";
    params.planner.max_graph_nodes_per_layer = std::max(
        1, getIntParam(node, ns + "max_graph_nodes_per_layer", params.planner.max_graph_nodes_per_layer));
    params.trajectory_waypoint_dt = std::max(
        1e-4, getDoubleParam(node, ns + "trajectory_waypoint_dt", params.trajectory_waypoint_dt));
    params.min_cartesian_step_m = std::max(
        1e-6, getDoubleParam(node, ns + "min_cartesian_step_m", params.min_cartesian_step_m));
    params.include_start_state_point = getBoolParam(
        node, ns + "include_start_state_point", params.include_start_state_point);
    params.start_state_guard_tolerance_rad = std::max(
        0.0,
        getDoubleParam(node, ns + "start_state_guard_tolerance_rad", params.start_state_guard_tolerance_rad));
    return params;
}

trajectory_msgs::msg::JointTrajectoryPoint makeTrajectoryPoint(
    const JointConfig& q,
    double time_from_start_sec) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.resize(NUM_JOINTS);
    for (int j = 0; j < NUM_JOINTS; ++j) point.positions[static_cast<size_t>(j)] = q[j];
    const int64_t ns = static_cast<int64_t>(std::llround(std::max(0.0, time_from_start_sec) * 1.0e9));
    point.time_from_start.sec = static_cast<int32_t>(ns / 1000000000LL);
    point.time_from_start.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
    return point;
}

}  // namespace

class FairinoCartesianPathNode : public rclcpp::Node {
public:
    FairinoCartesianPathNode()
        : Node("fairino_cartesian_path_server"),
          ik_params_(config::loadIKSelectParams(shared_from_this_safe(), "")),
          analytical_params_(config::loadAnalyticalIKParams(shared_from_this_safe(), "")),
          cartesian_params_(loadCartesianServerParams(shared_from_this_safe())),
          planner_(ik_params_, analytical_params_, cartesian_params_.planner),
          fk_(DHParams{}) {
        joint_names_ = {"j1", "j2", "j3", "j4", "j5", "j6"};
        service_ = create_service<moveit_msgs::srv::GetCartesianPath>(
            "/fairino_cartesian_path",
            std::bind(&FairinoCartesianPathNode::handleRequest, this, std::placeholders::_1, std::placeholders::_2));
        RCLCPP_INFO(get_logger(), "Fairino Cartesian path server ready: %s", service_->get_service_name());
    }

private:
    rclcpp::Node::SharedPtr shared_from_this_safe() {
        return std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node*) {});
    }

    void handleRequest(
        const std::shared_ptr<moveit_msgs::srv::GetCartesianPath::Request> req,
        std::shared_ptr<moveit_msgs::srv::GetCartesianPath::Response> res) {
        JointConfig q_start;
        if (!extractStartJointConfig(req->start_state.joint_state, joint_names_, q_start)) {
            res->fraction = 0.0;
            res->error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_ROBOT_STATE;
            RCLCPP_ERROR(get_logger(), "Fairino Cartesian path rejected: invalid start_state joint names.");
            return;
        }
        if (req->waypoints.empty()) {
            res->fraction = 1.0;
            res->error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
            return;
        }

        const ToolModel tool_model = toolModelFromLink(req->link_name);
        const Transform4d start_pose = fk_.fkine(q_start, tool_model);
        const auto waypoints = interpolateWaypoints(
            start_pose, req->waypoints, req->max_step, cartesian_params_.min_cartesian_step_m);

        CartesianIKPathRequest plan_req;
        plan_req.q_start = q_start;
        plan_req.waypoints = waypoints;
        plan_req.tool_model = tool_model;
        const auto plan_res = planner_.plan(plan_req);

        res->start_state = req->start_state;
        res->fraction = plan_res.fraction;
        res->solution.joint_trajectory.header = req->header;
        res->solution.joint_trajectory.joint_names = joint_names_;

        if (!plan_res.success) {
            res->error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
            RCLCPP_WARN(
                get_logger(),
                "Fairino Cartesian path failed: fraction=%.3f failed_index=%d total=%zu "
                "category=%s code=%s reason=%s",
                plan_res.fraction,
                plan_res.failed_index,
                waypoints.size(),
                toString(plan_res.failed_category),
                plan_res.failed_code.c_str(),
                plan_res.message.c_str());
            logRejectReasonSummary(plan_res.failure_diagnostics);
            for (size_t i = 0; i < plan_res.failure_diagnostics.size(); ++i) {
                const auto& d = plan_res.failure_diagnostics[i];
                RCLCPP_WARN(
                    get_logger(),
                    "Fairino Cartesian reject[%zu]: pass=%d reason=%s q_deg=%s dq_deg=%.2f dq_norm=%.4f "
                    "branch_changed=%d metrics={sigma=%.6f,cond=%.3f,margin=%.4f}",
                    i,
                    d.passed_hard_filter ? 1 : 0,
                    toString(d.reject_reason),
                    jointConfigToDegreesString(d.q).c_str(),
                    d.max_abs_dq * 180.0 / M_PI,
                    d.dq_norm,
                    d.branch_changed ? 1 : 0,
                    d.metrics.sigma_min,
                    d.metrics.cond,
                    d.metrics.min_joint_margin);
            }
            if (plan_res.has_failed_ik_result) {
                logAnalyticalFailure(plan_res);
            }
            return;
        }

        const size_t start_point_count = cartesian_params_.include_start_state_point ? 1u : 0u;
        res->solution.joint_trajectory.points.reserve(plan_res.path.size() + start_point_count);
        if (cartesian_params_.include_start_state_point) {
            res->solution.joint_trajectory.points.push_back(makeTrajectoryPoint(q_start, 0.0));
        }
        for (size_t i = 0; i < plan_res.path.size(); ++i) {
            res->solution.joint_trajectory.points.push_back(
                makeTrajectoryPoint(plan_res.path[i], static_cast<double>(i + 1) * cartesian_params_.trajectory_waypoint_dt));
        }

        const double start_delta = res->solution.joint_trajectory.points.empty()
            ? std::numeric_limits<double>::infinity()
            : maxPointDeltaFromStart(res->solution.joint_trajectory.points.front(), q_start);
        if (start_delta > cartesian_params_.start_state_guard_tolerance_rad) {
            res->solution.joint_trajectory.points.clear();
            res->fraction = 0.0;
            res->error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN;
            RCLCPP_ERROR(
                get_logger(),
                "Fairino Cartesian path rejected: trajectory start_delta=%.6f exceeds tolerance %.6f",
                start_delta,
                cartesian_params_.start_state_guard_tolerance_rad);
            return;
        }
        res->error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
        RCLCPP_INFO(
            get_logger(),
            "Fairino Cartesian path success: ik_points=%zu trajectory_points=%zu fraction=%.3f start_delta=%.6f",
            plan_res.path.size(),
            res->solution.joint_trajectory.points.size(),
            plan_res.fraction,
            start_delta);
    }

    void logAnalyticalFailure(const CartesianIKPathResult& plan_res) const {
        const auto& ik = plan_res.failed_ik_result;
        RCLCPP_WARN(
            get_logger(),
            "Fairino Cartesian analytical failure: waypoint_index=%d %s category=%s code=%s stage=%s detail=%s",
            plan_res.failed_index,
            transformToPoseString(plan_res.failed_waypoint).c_str(),
            toString(ik.failure_category),
            ik.failure_code.c_str(),
            ik.failure_stage.c_str(),
            ik.failure_detail.c_str());
        RCLCPP_WARN(
            get_logger(),
            "Fairino Cartesian analytical survival: q1=%d q5=%d q23=%d fk=%d unique=%d limits=%d "
            "rho_sq=%.9f wrist=[%.4f,%.4f,%.4f]",
            ik.survive_q1,
            ik.survive_q5,
            ik.survive_q23,
            ik.survive_fk_verify,
            ik.survive_unique,
            ik.survive_joint_limits,
            ik.rho_sq,
            ik.wrist_x,
            ik.wrist_y,
            ik.wrist_z);

        for (size_t i = 0; i < ik.wrist_rejects.size(); ++i) {
            const auto& r = ik.wrist_rejects[i];
            RCLCPP_WARN(
                get_logger(),
                "Fairino Cartesian wrist_reject[%zu]: q1_branch=%d q1_deg=%.2f c5=%.9f s5_abs=%.9f",
                i,
                r.q1_branch,
                r.q1 * 180.0 / M_PI,
                r.c5,
                r.s5_abs);
        }

        for (size_t i = 0; i < ik.d_domain_rejects.size(); ++i) {
            const auto& r = ik.d_domain_rejects[i];
            RCLCPP_WARN(
                get_logger(),
                "Fairino Cartesian d_domain_reject[%zu]: q1_branch=%d s5_sign=%d "
                "q1_deg=%.2f q5_deg=%.2f q234_deg=%.2f D=%.9f D_minus_limit=%.9f Xg=%.6f Zg=%.6f",
                i,
                r.q1_branch,
                r.s5_sign,
                r.q1 * 180.0 / M_PI,
                r.q5 * 180.0 / M_PI,
                r.q234 * 180.0 / M_PI,
                r.D,
                dDomainViolation(r.D),
                r.Xg,
                r.Zg);
        }

        for (size_t i = 0; i < ik.fk_rejects.size(); ++i) {
            const auto& r = ik.fk_rejects[i];
            RCLCPP_WARN(
                get_logger(),
                "Fairino Cartesian fk_reject[%zu]: q_deg=%s pos_err=%.9f rot_err=%.9f "
                "q1_branch=%d s5_sign=%d s3_sign=%d",
                i,
                jointConfigToDegreesString(r.q).c_str(),
                r.pos_err,
                r.rot_err,
                r.q1_branch,
                r.s5_sign,
                r.s3_sign);
        }

        for (size_t i = 0; i < ik.limit_rejects.size(); ++i) {
            const auto& r = ik.limit_rejects[i];
            RCLCPP_WARN(
                get_logger(),
                "Fairino Cartesian limit_reject[%zu]: q_deg=%s violation={%s}",
                i,
                jointConfigToDegreesString(r.q).c_str(),
                limitViolationString(r).c_str());
        }
    }

    void logRejectReasonSummary(const std::vector<IKCandidateDiagnostic>& diagnostics) const {
        if (diagnostics.empty()) return;
        std::map<std::string, int> counts;
        for (const auto& d : diagnostics) {
            if (d.passed_hard_filter) continue;
            ++counts[toString(d.reject_reason)];
        }
        if (counts.empty()) return;
        std::ostringstream oss;
        bool first = true;
        for (const auto& item : counts) {
            if (!first) oss << ",";
            first = false;
            oss << item.first << "=" << item.second;
        }
        RCLCPP_WARN(get_logger(), "Fairino Cartesian candidate filter summary: {%s}", oss.str().c_str());
    }

    static double dDomainViolation(double D) {
        if (D > 1.0) return D - 1.0;
        if (D < -1.0) return D + 1.0;
        return 0.0;
    }

    IKSelectParams ik_params_;
    AnalyticalIKParams analytical_params_;
    CartesianServerParams cartesian_params_;
    CartesianPathPlanner planner_;
    DHKinematics fk_;
    std::vector<std::string> joint_names_;
    rclcpp::Service<moveit_msgs::srv::GetCartesianPath>::SharedPtr service_;
};

}  // namespace fairino_planning

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<fairino_planning::FairinoCartesianPathNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
