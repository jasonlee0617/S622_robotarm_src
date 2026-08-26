// src/fairino_ik_plugin.cpp
// MoveIt2 逆运动学插件实现：为 Fairino 机器人提供 IK/FK 服务
// 支持根据末端执行器类型（法兰/夹爪）自动选择正确的工具模型

#include "myrobot_planning_ros/fairino_ik_plugin.h"

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model/joint_model_group.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include "myrobot_planning_ros/config/parameter_loader.hpp"
// /*
// 一、命名空间的作用
// 代码中所有自定义类都位于 namespace fairino_planning { ... } 中。
// 命名空间的主要作用是：
// 1.避免符号冲突：将代码封装在独立的命名空间，防止与其他库（如 MoveIt2 的 kinematics 命名空间）中的同名类或函数冲突。
// 2.逻辑组织：清晰地标识代码归属。

// 二、类与继承关系
// 1. kinematics::KinematicsBase 基类
// 这是 MoveIt2 定义的抽象基类（位于 moveit/kinematics_base/kinematics_base.h），所有 IK 插件都必须继承它。它声明了若干纯虚函数，例如：
// virtual bool getPositionIK(...) = 0;
// virtual bool searchPositionIK(...) = 0;
// virtual bool getPositionFK(...) = 0;
// 纯虚函数意味着基类只提供接口，没有实现，派生类必须实现这些函数才能被实例化。MoveIt2 在运行时通过基类指针调用这些函数，实现多态
// 2. FairinoIKPlugin 派生类
// class FairinoIKPlugin : public kinematics::KinematicsBase {
//     // 实现了所有纯虚函数
//     bool getPositionIK(...) override;
//     bool searchPositionIK(...) override;
//     bool getPositionFK(...) override;
//     // ...
// };
// override 关键字明确表示重写基类的虚函数。
// 正因为 FairinoIKPlugin 实现了所有纯虚函数，它才能被 pluginlib 实例化并注册为插件。
// 3.FairinoIK 类
// 这是一个独立类，不继承自 MoveIt2 的任何基类。它只包含纯运动学算法，与插件机制解耦。它有自己的构造函数（默认）和成员函数 solve
// 三、虚函数的作用
// 虚函数是 C++ 多态 的核心。在 MoveIt2 中，kinematics::KinematicsBase 是一个抽象基类，其中声明了纯虚函数。当 MoveIt2 通过 pluginlib 加载你的插件时，它会得到一个指向 kinematics::KinematicsBase 的基类指针（或引用），但实际上该指针指向的是你的派生类对象 FairinoIKPlugin。
// 虚函数的关键作用：
// 接口标准化：所有 IK 插件必须实现相同的接口，MoveIt2 只需知道基类中的函数签名，就能调用任何插件。
// 运行时多态：当 MoveIt2 通过基类指针调用 searchPositionIK 时，由于该函数是虚函数，实际执行的是派生类（即你的插件）中重写的版本。这样，MoveIt2 无需知道具体是哪个插件，就能动态切换 IK 求解器。
// 插件机制的基础：pluginlib 利用虚函数表实现动态加载，使得不同的 IK 算法可以编译成独立的共享库，在运行时根据配置加载。
// 如果没有虚函数，MoveIt2 就必须在编译时确定 IK 求解器的具体类型，无法实现插件化
// */
namespace fairino_planning {
namespace {
std::atomic<uint64_t> g_ik_call_id{0};

std::string poseSummary(const geometry_msgs::msg::Pose& pose) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "pos=[" << pose.position.x << "," << pose.position.y << "," << pose.position.z << "] "
        << "quat=[" << pose.orientation.x << "," << pose.orientation.y << ","
        << pose.orientation.z << "," << pose.orientation.w << "]";
    return oss.str();
}

std::string vectorSummary(const std::vector<double>& values, int precision = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ",";
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

std::vector<double> toDegrees(const std::vector<double>& rad) {
    std::vector<double> deg(rad.size(), 0.0);
    for (size_t i = 0; i < rad.size(); ++i) {
        deg[i] = rad[i] * 180.0 / M_PI;
    }
    return deg;
}

double posePositionDistance(const geometry_msgs::msg::Pose& a, const geometry_msgs::msg::Pose& b) {
    const double dx = a.position.x - b.position.x;
    const double dy = a.position.y - b.position.y;
    const double dz = a.position.z - b.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double poseOrientationDistance(const geometry_msgs::msg::Pose& a, const geometry_msgs::msg::Pose& b) {
    Eigen::Quaterniond qa(a.orientation.w, a.orientation.x, a.orientation.y, a.orientation.z);
    Eigen::Quaterniond qb(b.orientation.w, b.orientation.x, b.orientation.y, b.orientation.z);
    qa.normalize();
    qb.normalize();
    const double dot = std::min(1.0, std::abs(qa.dot(qb)));
    return 2.0 * std::acos(dot);
}

std::vector<double> toStdVector(const JointConfig& q) {
    std::vector<double> out(NUM_JOINTS, 0.0);
    for (int i = 0; i < NUM_JOINTS; ++i) {
        out[i] = q[i];
    }
    return out;
}

const char* toolModelName(const ToolModel model) {
    return model == ToolModel::GRIPPER ? "GRIPPER" : "FLANGE";
}

std::string limitViolationSummary(
    const IKResult::LimitRejectInfo& reject_info,
    bool degrees) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << "{";
    bool first = true;
    for (int i = 0; i < NUM_JOINTS; ++i) {
        const double low = reject_info.lower_violation[i];
        const double up = reject_info.upper_violation[i];
        if (low <= 0.0 && up <= 0.0) {
            continue;
        }
        if (!first) {
            oss << ",";
        }
        first = false;
        const double scale = degrees ? (180.0 / M_PI) : 1.0;
        oss << "j" << (i + 1);
        if (low > 0.0) {
            oss << ":low+" << (low * scale);
        }
        if (up > 0.0) {
            if (low > 0.0) {
                oss << "/";
            }
            oss << "up+" << (up * scale);
        }
    }
    oss << "}";
    return oss.str();
}

std::string analyticalSummary(const AnalyticalIKParams& params) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(2)
        << "rho_eps=" << params.rho_sq_neg_eps
        << ",s5_min=" << params.wrist_singularity_s5_min
        << ",D_eps=" << params.D_domain_eps
        << ",fk_pos=" << params.fk_verify_pos_tol
        << ",fk_rot=" << params.fk_verify_rot_tol
        << ",uniq=" << params.solution_unique_tol
        << ",dup=" << params.candidate_dup_norm_tol;
    return oss.str();
}

std::string transformPoseSummary(const Transform4d& T) {
    Eigen::Quaterniond q(T.block<3,3>(0,0));
    q.normalize();
    const auto p = T.block<3,1>(0,3);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "pos=[" << p.x() << "," << p.y() << "," << p.z() << "] "
        << "quat=[" << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "]";
    return oss.str();
}

Transform4d isometryToTransform(const Eigen::Isometry3d& in) {
    Transform4d out = Transform4d::Identity();
    out.block<3, 3>(0, 0) = in.linear();
    out.block<3, 1>(0, 3) = in.translation();
    return out;
}

size_t logLimit(size_t total, int max_candidates, bool log_all) {
    if (log_all || max_candidates < 0) {
        return total;
    }
    return std::min(total, static_cast<size_t>(std::max(0, max_candidates)));
}

std::string rejectSummary(const std::vector<IKCandidateDiagnostic>& diagnostics) {
    std::map<std::string, size_t> counts;
    for (const auto& diagnostic : diagnostics) {
        if (!diagnostic.passed_hard_filter) {
            ++counts[toString(diagnostic.reject_reason)];
        }
    }
    if (counts.empty()) {
        return "none";
    }

    std::ostringstream oss;
    bool first = true;
    for (const auto& [reason, count] : counts) {
        if (!first) {
            oss << ",";
        }
        first = false;
        oss << reason << ":" << count;
    }
    return oss.str();
}
}  // namespace
// //声明插件类，公有继承自 MoveIt2 的 KinematicsBase 基类
// /*在 fairino_ik_plugin.h 中，FairinoIKPlugin 类声明了私有成员：
// private:
//     FairinoIK    ik_solver_;
//     IKSelector   ik_selector_;
//     DHKinematics fk_;
// */
// ========================= 构造函数 =========================
FairinoIKPlugin::FairinoIKPlugin()
    : ik_solver_(), ik_selector_(), fk_() {}

// ========================= 插件初始化 =========================
/// @brief 由 MoveIt 调用，传入机器人模型、规划组、末端连杆等信息
bool FairinoIKPlugin::initialize(
    const rclcpp::Node::SharedPtr& node,
    const moveit::core::RobotModel& robot_model,
    const std::string& group_name,
    const std::string& base_frame,
    const std::vector<std::string>& tip_frames,
    double search_discretization)
{
    // 调用基类的存储函数（保存基本配置）
    storeValues(robot_model, group_name, base_frame, tip_frames, search_discretization);

    // 获取规划组模型
    const auto* jmg = robot_model.getJointModelGroup(group_name);
    if (!jmg) {
        RCLCPP_ERROR(node->get_logger(),
            "FairinoIKPlugin: Joint group '%s' not found", group_name.c_str());
        return false;
    }

    // 保存关节名称、连杆名称、末端连杆名称
    joint_names_ = jmg->getActiveJointModelNames();
    link_names_  = jmg->getLinkModelNames();
    tip_frames_  = tip_frames;
    group_name_  = group_name;
    base_frame_  = base_frame;

    tool_model_override_ = config::loadToolModelOverride(node);
    ik_select_params_ = config::loadIKSelectParams(node);
    analytical_ik_params_ = config::loadAnalyticalIKParams(node);

    const auto* tool_link = robot_model.getLinkModel("tool0");
    if (!tool_link || !tool_link->getParentLinkModel() ||
        tool_link->getParentLinkModel()->getName() != "wrist3_link") {
        RCLCPP_ERROR(node->get_logger(),
            "Fairino IK requires a fixed wrist3_link -> tool0 chain in robot_description.");
        return false;
    }
    const Transform4d wrist3_to_tool =
        isometryToTransform(tool_link->getJointOriginTransform());
    const Transform4d flange_to_tool = DHKinematics::flangeToToolTransform(
        DHParams{}, wrist3_to_tool);

    ik_select_params_.gripper_tool = analytical_ik_params_.gripper_tool;
    ik_solver_ = FairinoIK(analytical_ik_params_);
    ik_selector_ = IKSelector(ik_select_params_);
    ik_solver_.setToolTransform(flange_to_tool);
    ik_selector_.setToolTransform(flange_to_tool);
    fk_ = DHKinematics(DHParams{}, flange_to_tool);

    RCLCPP_INFO(node->get_logger(),
        "FairinoIKPlugin initialized: group='%s', joints=%zu, tips=%zu, override='%s'",
        group_name_.c_str(),
        joint_names_.size(),
        tip_frames_.size(),
        tool_model_override_.c_str());
    RCLCPP_INFO(
        node->get_logger(),
        "Fairino IK TCP loaded from robot_description: wrist3_link->tool0 %s, flange->tool0 %s",
        transformPoseSummary(wrist3_to_tool).c_str(),
        transformPoseSummary(flange_to_tool).c_str());
    if (analytical_ik_params_.log_threshold_summary) {
        RCLCPP_INFO(
            node->get_logger(),
            "Fairino analytical IK params: %s",
            analyticalSummary(analytical_ik_params_).c_str());
    }

    // 打印所有末端连杆名称（调试用）
    for (size_t i = 0; i < tip_frames_.size(); ++i) {
        RCLCPP_INFO(node->get_logger(), "  tip_frames_[%zu] = %s", i, tip_frames_[i].c_str());
    }

    return true;
}

// ========================= 坐标变换辅助函数 =========================
Eigen::Matrix4d FairinoIKPlugin::poseToEigen(const geometry_msgs::msg::Pose& pose) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x,
                         pose.orientation.y, pose.orientation.z);
    q.normalize();
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T(0,3) = pose.position.x;
    T(1,3) = pose.position.y;
    T(2,3) = pose.position.z;
    return T;
}

geometry_msgs::msg::Pose FairinoIKPlugin::eigenToPose(const Eigen::Matrix4d& T) {
    geometry_msgs::msg::Pose pose;
    Eigen::Quaterniond q(T.block<3,3>(0,0));
    q.normalize();
    pose.position.x = T(0,3);
    pose.position.y = T(1,3);
    pose.position.z = T(2,3);
    pose.orientation.w = q.w();
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    return pose;
}

// ========================= 工具模型解析 =========================
/// @brief 根据末端连杆名称判断应该使用哪种工具模型
ToolModel FairinoIKPlugin::resolveToolModel(const std::string& tip_frame) const {
    if (tip_frame == "tool0") return ToolModel::GRIPPER;

    // 优先使用用户强制指定的模型
    if (tool_model_override_ == "flange") return ToolModel::FLANGE;
    if (tool_model_override_ == "gripper") return ToolModel::GRIPPER;

    // 自动判断：根据常见的末端连杆名称
    // 夹爪相关的名称
    if (tip_frame == "gripper_tool_link" ||
        tip_frame == "tool0" ||
        tip_frame == "tool_link" ||
        tip_frame == "ee_tool_link") {
        return ToolModel::GRIPPER;
    }

    // 法兰相关的名称
    if (tip_frame == "link6" ||
        tip_frame == "flange_link" ||
        tip_frame == "tool0" ||
        tip_frame == "wrist3_link") {
        return ToolModel::FLANGE;
    }

    // 默认保守处理：当作法兰
    return ToolModel::FLANGE;
}

/// @brief 为 IK 求解确定工具模型（基于第一个 tip_frame）
ToolModel FairinoIKPlugin::resolveToolModelForIK() const {
    if (!tip_frames_.empty()) {
        return resolveToolModel(tip_frames_.front());
    }
    return ToolModel::FLANGE;  // 无 tip 信息时默认为法兰
}

/// @brief 为 FK 求解确定工具模型（根据请求的连杆名称）
ToolModel FairinoIKPlugin::resolveToolModelForFK(const std::string& link_name) const {
    return resolveToolModel(link_name);
}

// ========================= 核心 IK 求解（所有重载最终调用此函数） =========================
bool FairinoIKPlugin::solveIK(
    const geometry_msgs::msg::Pose& ik_pose,
    const std::vector<double>& ik_seed_state,
    double timeout,
    const std::vector<double>& consistency_limits,
    std::vector<double>& solution,
    moveit_msgs::msg::MoveItErrorCodes& error_code,
    const IKCallbackFn& solution_callback,
    bool update_continuity_state) const
{
    const auto logger = rclcpp::get_logger("FairinoIKPlugin");
    const uint64_t call_id = ++g_ik_call_id;
    const auto start_time = std::chrono::steady_clock::now();
    const auto timed_out = [&]() {
        if (timeout <= 0.0) return false;
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        return elapsed >= timeout;
    };

    // 1. 将 ROS Pose 转换为 Eigen 矩阵
    Eigen::Matrix4d T_target = poseToEigen(ik_pose);

    // 2. 确定本次 IK 使用的工具模型（法兰或夹爪）
    const ToolModel tool_model = resolveToolModelForIK();

    // 3. 调用核心逆解器（带工具模型）
    auto ik_result = ik_solver_.solve(T_target, tool_model);
    const bool verbose_log = ik_select_params_.debug_log_all_candidates;
    const int log_every = std::max(1, ik_select_params_.debug_log_every_n_calls);
    const bool sampled_log = ((call_id - 1U) % static_cast<uint64_t>(log_every) == 0U);
    const bool should_log = verbose_log && sampled_log;
    const int max_log = ik_select_params_.debug_max_candidates_to_log;
    const bool log_all_candidates = ik_select_params_.debug_log_all_candidates;
    const bool log_deg = ik_select_params_.debug_print_degrees;
    if (should_log) {
        RCLCPP_INFO(
            logger,
            "[IK][call=%lu] target=%s tool=%s seed_rad=%s seed_deg=%s raw_candidates=%zu",
            static_cast<unsigned long>(call_id),
            poseSummary(ik_pose).c_str(),
            toolModelName(tool_model),
            vectorSummary(ik_seed_state).c_str(),
            vectorSummary(toDegrees(ik_seed_state), 2).c_str(),
            ik_result.solutions.size());
        if (analytical_ik_params_.log_threshold_summary) {
            RCLCPP_INFO(
                logger,
                "[IK][call=%lu] analytical={%s}",
                static_cast<unsigned long>(call_id),
                analyticalSummary(analytical_ik_params_).c_str());
        }
        RCLCPP_INFO(
            logger,
            "[IK][call=%lu] analytical_target tool_pose={%s} flange_pose={%s} wrist=[%.5f,%.5f,%.5f] rho_sq=%.8f",
            static_cast<unsigned long>(call_id),
            transformPoseSummary(ik_result.target_pose).c_str(),
            transformPoseSummary(ik_result.flange_pose).c_str(),
            ik_result.wrist_x,
            ik_result.wrist_y,
            ik_result.wrist_z,
            ik_result.rho_sq);
        if (analytical_ik_params_.log_stage_survival) {
            RCLCPP_INFO(
                logger,
                "[IK][call=%lu] stage_survival total=%d q1=%d q5=%d q23=%d fk=%d unique=%d limits=%d failure={category=%s,stage=%s,detail=%s}",
                static_cast<unsigned long>(call_id),
                ik_result.total_branches,
                ik_result.survive_q1,
                ik_result.survive_q5,
                ik_result.survive_q23,
                ik_result.survive_fk_verify,
                ik_result.survive_unique,
                ik_result.survive_joint_limits,
                toString(ik_result.failure_category),
                ik_result.failure_stage.c_str(),
                ik_result.failure_detail.c_str());
        }
        const size_t wrist_log_count = logLimit(
            ik_result.wrist_rejects.size(), max_log, log_all_candidates);
        for (size_t i = 0; i < wrist_log_count; ++i) {
            const auto& r = ik_result.wrist_rejects[i];
            RCLCPP_INFO(
                logger,
                "[IK][call=%lu][wrist_reject=%zu/%zu] q1_branch=%d q1=%.6f c5=%.6f s5_abs=%.9f threshold=%.9f",
                static_cast<unsigned long>(call_id),
                i,
                ik_result.wrist_rejects.size(),
                r.q1_branch,
                r.q1,
                r.c5,
                r.s5_abs,
                analytical_ik_params_.wrist_singularity_s5_min);
        }
        const size_t d_domain_log_count = logLimit(
            ik_result.d_domain_rejects.size(), max_log, log_all_candidates);
        for (size_t i = 0; i < d_domain_log_count; ++i) {
            const auto& r = ik_result.d_domain_rejects[i];
            RCLCPP_INFO(
                logger,
                "[IK][call=%lu][D_reject=%zu/%zu] q1_branch=%d s5_sign=%d q1=%.6f q5=%.6f q234=%.6f D=%.9f Xg=%.6f Zg=%.6f eps=%.2e",
                static_cast<unsigned long>(call_id),
                i,
                ik_result.d_domain_rejects.size(),
                r.q1_branch,
                r.s5_sign,
                r.q1,
                r.q5,
                r.q234,
                r.D,
                r.Xg,
                r.Zg,
                analytical_ik_params_.D_domain_eps);
        }
        const size_t fk_log_count = logLimit(
            ik_result.fk_rejects.size(), max_log, log_all_candidates);
        for (size_t i = 0; i < fk_log_count; ++i) {
            const auto& r = ik_result.fk_rejects[i];
            const auto q_rad = toStdVector(r.q);
            RCLCPP_INFO(logger,
                "[IK][call=%lu][fk_reject=%zu/%zu] q1_branch=%d s5_sign=%d s3_sign=%d "
                "pos_err=%.6f rot_err=%.6f q_rad=%s q_deg=%s",
                static_cast<unsigned long>(call_id), i, ik_result.fk_rejects.size(),
                r.q1_branch, r.s5_sign, r.s3_sign,
                r.pos_err, r.rot_err,
                vectorSummary(q_rad).c_str(),
                vectorSummary(toDegrees(q_rad), 2).c_str());
        }
        const size_t limit_log_count = logLimit(
            ik_result.limit_rejects.size(), max_log, log_all_candidates);
        for (size_t i = 0; i < limit_log_count; ++i) {
            const auto& r = ik_result.limit_rejects[i];
            const auto q_rad = toStdVector(r.q);
            if (log_deg) {
                RCLCPP_INFO(
                    logger,
                    "[IK][call=%lu][limit_reject=%zu/%zu] q_rad=%s q_deg=%s violation_rad=%s violation_deg=%s",
                    static_cast<unsigned long>(call_id),
                    i,
                    ik_result.limit_rejects.size(),
                    vectorSummary(q_rad).c_str(),
                    vectorSummary(toDegrees(q_rad), 2).c_str(),
                    limitViolationSummary(r, false).c_str(),
                    limitViolationSummary(r, true).c_str());
            } else {
                RCLCPP_INFO(
                    logger,
                    "[IK][call=%lu][limit_reject=%zu/%zu] q_rad=%s violation_rad=%s",
                    static_cast<unsigned long>(call_id),
                    i,
                    ik_result.limit_rejects.size(),
                    vectorSummary(q_rad).c_str(),
                    limitViolationSummary(r, false).c_str());
            }
        }
        const size_t raw_log_count = logLimit(
            ik_result.solutions.size(), max_log, log_all_candidates);
        for (size_t i = 0; i < raw_log_count; ++i) {
            const auto q_rad = toStdVector(ik_result.solutions[i]);
            if (log_deg) {
                RCLCPP_INFO(
                    logger,
                    "[IK][call=%lu][raw=%zu/%zu] q_rad=%s q_deg=%s",
                    static_cast<unsigned long>(call_id),
                    i,
                    ik_result.solutions.size(),
                    vectorSummary(q_rad).c_str(),
                    vectorSummary(toDegrees(q_rad), 2).c_str());
            } else {
                RCLCPP_INFO(
                    logger,
                    "[IK][call=%lu][raw=%zu/%zu] q_rad=%s",
                    static_cast<unsigned long>(call_id),
                    i,
                    ik_result.solutions.size(),
                    vectorSummary(q_rad).c_str());
            }
        }
    }
    if (!ik_result.success || ik_result.solutions.empty()) {
        if (should_log || ik_select_params_.debug_always_log_failures) {
            RCLCPP_WARN(
                logger,
                "[IK][call=%lu] failed: solver category=%s stage=%s detail=%s survival=%d/%d/%d/%d/%d/%d/%d",
                static_cast<unsigned long>(call_id),
                toString(ik_result.failure_category),
                ik_result.failure_stage.c_str(),
                ik_result.failure_detail.empty() ? "-" : ik_result.failure_detail.c_str(),
                ik_result.total_branches,
                ik_result.survive_q1,
                ik_result.survive_q5,
                ik_result.survive_q23,
                ik_result.survive_fk_verify,
                ik_result.survive_unique,
                ik_result.survive_joint_limits);
        }
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // 4. 构造种子关节配置（用于选择最接近的解）
    JointConfig q_seed = JointConfig::Zero();
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(ik_seed_state.size()); ++i) {
        q_seed[i] = ik_seed_state[i];
    }

    // 5. 选择离 seed 最近的解
    // auto best = ik_selector_.select(ik_result.solutions, q_seed);
    if (timed_out()) {
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT;
        if (should_log || ik_select_params_.debug_always_log_failures) {
            RCLCPP_WARN(
                logger,
                "[IK][call=%lu] failed: timeout stage=selector code=%d",
                static_cast<unsigned long>(call_id),
                error_code.val);
        }
        return false;
    }

    bool has_last_solution_snapshot = false;
    bool has_last_ik_pose_snapshot = false;
    JointConfig last_solution_snapshot = JointConfig::Zero();
    geometry_msgs::msg::Pose last_ik_pose_snapshot;
    if (update_continuity_state) {
        std::lock_guard<std::mutex> lock(last_solution_mutex_);
        has_last_solution_snapshot = has_last_solution_;
        has_last_ik_pose_snapshot = has_last_ik_pose_;
        last_solution_snapshot = last_solution_;
        last_ik_pose_snapshot = last_ik_pose_;
    }

    const bool has_callback = static_cast<bool>(solution_callback);
    const bool seed_synced_to_last =
        has_last_solution_snapshot &&
        (wrapToPi(q_seed - last_solution_snapshot).norm() <= ik_select_params_.hint_seed_sync_max_rad);
    const double pose_delta_m =
        has_last_ik_pose_snapshot
            ? posePositionDistance(ik_pose, last_ik_pose_snapshot)
            : std::numeric_limits<double>::infinity();
    const double rot_delta_rad =
        has_last_ik_pose_snapshot
            ? poseOrientationDistance(ik_pose, last_ik_pose_snapshot)
            : std::numeric_limits<double>::infinity();
    const bool pose_synced_to_last =
        has_last_ik_pose_snapshot &&
        pose_delta_m <= ik_select_params_.cartesian_stream_max_pos_step_m &&
        rot_delta_rad <= ik_select_params_.cartesian_stream_max_rot_step_rad;
    const bool use_continuity_hint =
        update_continuity_state && seed_synced_to_last && pose_synced_to_last;
    IKBranchHint hint{};
    hint.valid = use_continuity_hint && has_last_solution_snapshot;
    hint.q_last = has_last_solution_snapshot ? last_solution_snapshot : q_seed;

    IKSelectionRequest selection_request;
    selection_request.solutions = &ik_result.solutions;
    selection_request.seed = q_seed;
    selection_request.target_pose = T_target;
    selection_request.tool_model = tool_model;
    selection_request.task_profile = ik_select_params_.task_profile;
    selection_request.hint = &hint;
    selection_request.consistency_limits = consistency_limits;
    IKSelectionResult selection = ik_selector_.select(selection_request);
    const auto& diagnostics = selection.diagnostics;
    const auto& metrics = selection.metrics;
    const auto& best = selection.selected;
    if (should_log) {
        RCLCPP_INFO(
            logger,
            "[IK][call=%lu] continuity_hint=%d callback=%d seed_synced=%d pose_delta=%.5f rot_delta=%.5f",
            static_cast<unsigned long>(call_id),
            use_continuity_hint ? 1 : 0,
            has_callback ? 1 : 0,
            seed_synced_to_last ? 1 : 0,
            pose_delta_m,
            rot_delta_rad);
    }
    if (should_log) {
        const size_t diagnostic_log_count = logLimit(
            diagnostics.size(), max_log, log_all_candidates);
        RCLCPP_INFO(
            logger,
            "[IK][call=%lu] selector_candidates=%zu logging=%zu selected_present=%d",
            static_cast<unsigned long>(call_id),
            diagnostics.size(),
            diagnostic_log_count,
            best ? 1 : 0);
        for (size_t i = 0; i < diagnostic_log_count; ++i) {
            const auto& d = diagnostics[i];
            const auto q_rad = toStdVector(d.q);
            const char* reject = toString(d.reject_reason);
            if (log_deg) {
                RCLCPP_INFO(
                    logger,
                    "[IK][call=%lu][cand=%zu/%zu] pass=%d reason=%s selected=%d flip=%d "
                    "q_rad=%s q_deg=%s "
                    "dq_deg=%.2f dq_norm=%.4f branch_changed=%d "
                    "score={S1=%.4f,S2=%.4f,S3=%.4f,S4=%.4f,total=%.4f} "
                    "metrics={sigma=%.6f,cond=%.3f,margin=%.4f}",
                    static_cast<unsigned long>(call_id), i, diagnostics.size(),
                    d.passed_hard_filter ? 1 : 0, reject, d.selected ? 1 : 0, d.wrist_flip ? 1 : 0,
                    vectorSummary(q_rad).c_str(), vectorSummary(toDegrees(q_rad), 2).c_str(),
                    d.max_abs_dq * 180.0 / M_PI, d.dq_norm, d.branch_changed ? 1 : 0,
                    d.S1, d.S2, d.S3, d.S4, d.total_cost,
                    d.metrics.sigma_min, d.metrics.cond, d.metrics.min_joint_margin);
            } else {
                RCLCPP_INFO(
                    logger,
                    "[IK][call=%lu][cand=%zu/%zu] pass=%d reason=%s selected=%d flip=%d "
                    "q_rad=%s dq=%.4f dq_norm=%.4f branch_changed=%d "
                    "score={S1=%.4f,S2=%.4f,S3=%.4f,S4=%.4f,total=%.4f} "
                    "metrics={sigma=%.6f,cond=%.3f,margin=%.4f}",
                    static_cast<unsigned long>(call_id), i, diagnostics.size(),
                    d.passed_hard_filter ? 1 : 0, reject, d.selected ? 1 : 0, d.wrist_flip ? 1 : 0,
                    vectorSummary(q_rad).c_str(),
                    d.max_abs_dq, d.dq_norm, d.branch_changed ? 1 : 0,
                    d.S1, d.S2, d.S3, d.S4, d.total_cost,
                    d.metrics.sigma_min, d.metrics.cond, d.metrics.min_joint_margin);
            }
        }
    }
    if (!best) {
        if (should_log || ik_select_params_.debug_always_log_failures) {
            RCLCPP_WARN(
                logger,
                "[IK][call=%lu] failed: selector candidates=%zu rejects=%s code=%d",
                static_cast<unsigned long>(call_id),
                diagnostics.size(),
                rejectSummary(diagnostics).c_str(),
                moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION);
        }
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // 6. 转换为 vector<double>
    solution.resize(NUM_JOINTS);
    for (int i = 0; i < NUM_JOINTS; ++i) {
        solution[i] = (*best)[i];
    }
    if (should_log) {
        const auto best_deg = toDegrees(solution);
        RCLCPP_INFO(
            logger,
            "[IK][call=%lu] selected q_rad=%s q_deg=%s metrics={sigma=%.6f,cond=%.3f,margin=%.4f}",
            static_cast<unsigned long>(call_id),
            vectorSummary(solution).c_str(),
            vectorSummary(best_deg, 2).c_str(),
            metrics.sigma_min, metrics.cond, metrics.min_joint_margin);
    }

    // 7. ★ 如果 MoveIt 提供了回调函数（通常用于碰撞检测或自定义验证）
    //    先验证当前最佳解；若失败，只在已经通过连续性/分支护栏的候选中重试。
    if (solution_callback) {
        geometry_msgs::msg::Pose p = ik_pose;  // 注意：回调需要传入目标位姿

        std::vector<size_t> order;
        order.reserve(diagnostics.size());
        for (size_t i = 0; i < diagnostics.size(); ++i) {
            if (diagnostics[i].passed_hard_filter) order.push_back(i);
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const auto& da = diagnostics[a];
            const auto& db = diagnostics[b];
            if (da.selected != db.selected) return da.selected;
            if (da.branch_changed != db.branch_changed) return !da.branch_changed;
            if (std::abs(da.max_abs_dq - db.max_abs_dq) > 1e-9) return da.max_abs_dq < db.max_abs_dq;
            if (std::abs(da.dq_norm - db.dq_norm) > 1e-9) return da.dq_norm < db.dq_norm;
            return da.total_cost < db.total_cost;
        });
        if (order.empty()) {
            order.push_back(0);
        }

        std::vector<JointConfig> tried;
        moveit_msgs::msg::MoveItErrorCodes cb_error;
        cb_error.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        for (size_t idx : order) {
            if (timed_out()) {
                error_code.val = moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT;
                if (should_log || ik_select_params_.debug_always_log_failures) {
                    RCLCPP_WARN(
                        logger,
                        "[IK][call=%lu] failed: timeout stage=callback code=%d",
                        static_cast<unsigned long>(call_id),
                        error_code.val);
                }
                return false;
            }
            const JointConfig q_try = idx < diagnostics.size() ? diagnostics[idx].q : *best;
            bool duplicate = false;
            for (const auto& q_prev : tried) {
                if ((q_try - q_prev).norm() < analytical_ik_params_.candidate_dup_norm_tol) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            tried.push_back(q_try);

            std::vector<double> try_solution = toStdVector(q_try);
            solution_callback(p, try_solution, cb_error);
            if (cb_error.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                solution = try_solution;
                if (update_continuity_state) {
                    std::lock_guard<std::mutex> lock(last_solution_mutex_);
                    last_solution_ = q_try;
                    has_last_solution_ = true;
                    last_ik_pose_ = ik_pose;
                    has_last_ik_pose_ = true;
                }
                if (!diagnostics.empty() && idx < diagnostics.size() && !diagnostics[idx].selected &&
                    (should_log || ik_select_params_.debug_always_log_failures)) {
                    RCLCPP_WARN(
                        logger,
                        "[IK][call=%lu] callback rejected selected candidate; guarded fallback accepted.",
                        static_cast<unsigned long>(call_id));
                }
                error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
                return true;
            }
        }

        // 所有已通过连续性护栏的解都未通过回调验证
        error_code.val = cb_error.val;
        if (should_log || ik_select_params_.debug_always_log_failures) {
            RCLCPP_WARN(
                logger,
                "[IK][call=%lu] failed: callback tried=%zu code=%d",
                static_cast<unsigned long>(call_id),
                tried.size(),
                error_code.val);
        }
        return false;
    }

    if (update_continuity_state) {
        std::lock_guard<std::mutex> lock(last_solution_mutex_);
        last_solution_ = *best;
        has_last_solution_ = true;
        last_ik_pose_ = ik_pose;
        has_last_ik_pose_ = true;
    }
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return true;
}

// ========================= MoveIt 接口实现（所有重载都委托给 solveIK） =========================

bool FairinoIKPlugin::getPositionIK(
    const geometry_msgs::msg::Pose& ik_pose,
    const std::vector<double>& ik_seed_state,
    std::vector<double>& solution,
    moveit_msgs::msg::MoveItErrorCodes& error_code,
    const kinematics::KinematicsQueryOptions& /*options*/) const
{
    return solveIK(ik_pose, ik_seed_state, 0.0, {}, solution, error_code, IKCallbackFn(), false);
}

bool FairinoIKPlugin::searchPositionIK(
    const geometry_msgs::msg::Pose& ik_pose,
    const std::vector<double>& ik_seed_state,
    double timeout,
    std::vector<double>& solution,
    moveit_msgs::msg::MoveItErrorCodes& error_code,
    const kinematics::KinematicsQueryOptions& /*options*/) const
{
    return solveIK(ik_pose, ik_seed_state, timeout, {}, solution, error_code, IKCallbackFn(), true);
}

bool FairinoIKPlugin::searchPositionIK(
    const geometry_msgs::msg::Pose& ik_pose,
    const std::vector<double>& ik_seed_state,
    double timeout,
    const std::vector<double>& consistency_limits,
    std::vector<double>& solution,
    moveit_msgs::msg::MoveItErrorCodes& error_code,
    const kinematics::KinematicsQueryOptions& /*options*/) const
{
    return solveIK(
        ik_pose, ik_seed_state, timeout, consistency_limits,
        solution, error_code, IKCallbackFn(), true);
}

bool FairinoIKPlugin::searchPositionIK(
    const geometry_msgs::msg::Pose& ik_pose,
    const std::vector<double>& ik_seed_state,
    double timeout,
    std::vector<double>& solution,
    const IKCallbackFn& solution_callback,
    moveit_msgs::msg::MoveItErrorCodes& error_code,
    const kinematics::KinematicsQueryOptions& /*options*/) const
{
    return solveIK(
        ik_pose, ik_seed_state, timeout, {},
        solution, error_code, solution_callback, true);
}

bool FairinoIKPlugin::searchPositionIK(
    const geometry_msgs::msg::Pose& ik_pose,
    const std::vector<double>& ik_seed_state,
    double timeout,
    const std::vector<double>& consistency_limits,
    std::vector<double>& solution,
    const IKCallbackFn& solution_callback,
    moveit_msgs::msg::MoveItErrorCodes& error_code,
    const kinematics::KinematicsQueryOptions& /*options*/) const
{
    return solveIK(
        ik_pose, ik_seed_state, timeout, consistency_limits,
        solution, error_code, solution_callback, true);
}

// ========================= 正向运动学 =========================
/// @brief 给定关节角，计算指定连杆的位姿（支持工具模型自动识别）
bool FairinoIKPlugin::getPositionFK(
    const std::vector<std::string>& link_names,
    const std::vector<double>& joint_angles,
    std::vector<geometry_msgs::msg::Pose>& poses) const
{
    // 构建关节配置向量
    JointConfig q = JointConfig::Zero();
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(joint_angles.size()); ++i) {
        q[i] = joint_angles[i];
    }

    poses.clear();
    poses.reserve(link_names.size());

    // 对每个请求的连杆，根据其名称确定工具模型，计算位姿
    for (const auto& link_name : link_names) {
        const ToolModel model = resolveToolModelForFK(link_name);
        Eigen::Matrix4d T = fk_.fkine(q, model);
        poses.push_back(eigenToPose(T));
    }

    return true;
}

}  // namespace fairino_planning

// 注册插件（pluginlib 宏）
PLUGINLIB_EXPORT_CLASS(fairino_planning::FairinoIKPlugin, kinematics::KinematicsBase)
// //这是 pluginlib 宏，它声明了一个函数，用于在运行时将 FairinoIKPlugin 类注册为 kinematics::KinematicsBase 类型的插件。当 MoveIt2 加载插件库时，pluginlib 会查找这个宏，并根据类名和基类信息实例化该插件
// PLUGINLIB_EXPORT_CLASS(fairino_planning::FairinoIKPlugin,kinematics::KinematicsBase)
