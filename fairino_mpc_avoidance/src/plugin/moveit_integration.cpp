// moveit_integration.cpp
// MoveIt bridge utilities:
// - robot model/state access
// - planning scene collision objects conversion
// - kinematics and geometric distance helpers

#include "fairino_mpc_avoidance/plugin/moveit_integration.hpp"
#include "fairino_mpc_avoidance/smooth_box_distance.hpp"
#include <rclcpp/rclcpp.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <moveit/robot_model/joint_model_group.h>

namespace fairino_mpc {

MoveItIntegration::MoveItIntegration(rclcpp::Node::SharedPtr node)
    : node_(node) {
}

bool MoveItIntegration::initialize() {
    try {
        // 初始化机器人模型加载器（对应Matlab的机器人模型加载）
        robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(
            node_, "robot_description");

        robot_model_ = robot_model_loader_->getModel();
        if (!robot_model_) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to load robot model");
            return false;
        }

        // 初始化机器人状态
        robot_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
        robot_state_->setToDefaultValues();

        // 初始化PlanningScene监控器（对应Matlab的碰撞环境构建）
        planning_scene_monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
            node_, robot_model_loader_);

        planning_scene_monitor_->startSceneMonitor();
        planning_scene_monitor_->startWorldGeometryMonitor();
        planning_scene_monitor_->startStateMonitor();

        RCLCPP_INFO(node_->get_logger(), "MoveItIntegration initialized successfully");
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "MoveItIntegration initialization failed: %s", e.what());
        return false;
    }
}

std::vector<Obstacle> MoveItIntegration::getCollisionEnvironment() const {
    std::vector<Obstacle> obstacles;

    if (!planning_scene_monitor_) {
        return obstacles;
    }

    auto scene = planning_scene_monitor_->getPlanningScene();
    if (!scene) {
        return obstacles;
    }

    // 获取世界中的所有碰撞物体（对应Matlab的buildCollisionEnv）
    auto world = scene->getWorld();
    for (const auto& [id, obj] : *world) {
        for (size_t i = 0; i < obj->shapes_.size(); ++i) {
            Obstacle obstacle;

            // 获取位置
            auto pose = obj->shape_poses_[i];
            obstacle.center = Vec3(pose.translation().x(),
                                  pose.translation().y(),
                                  pose.translation().z());

            // 获取尺寸（根据形状类型）
            auto shape = obj->shapes_[i];
            if (auto box = std::dynamic_pointer_cast<const shapes::Box>(shape)) {
                obstacle.size = Vec3(box->size[0], box->size[1], box->size[2]);
            } else if (auto sphere = std::dynamic_pointer_cast<const shapes::Sphere>(shape)) {
                double radius = sphere->radius;
                obstacle.size = Vec3(radius * 2, radius * 2, radius * 2);
            } else if (auto cylinder = std::dynamic_pointer_cast<const shapes::Cylinder>(shape)) {
                double radius = cylinder->radius;
                double height = cylinder->length;
                obstacle.size = Vec3(radius * 2, radius * 2, height);
            }

            // 计算边界框
            obstacle.bounds_min = obstacle.center - obstacle.size / 2.0;
            obstacle.bounds_max = obstacle.center + obstacle.size / 2.0;

            obstacles.push_back(obstacle);
        }
    }

    return obstacles;
}

std::shared_ptr<moveit::core::RobotModel> MoveItIntegration::getRobotModel() const {
    return std::const_pointer_cast<moveit::core::RobotModel>(robot_model_);
}

Vec3 MoveItIntegration::computeEndEffectorPosition(const VecN& q) const {
    if (!robot_state_ || !robot_model_) {
        return Vec3::Zero();
    }

    for (int i = 0; i < N_JOINTS; ++i) {
        const auto* jm = robot_model_->getJointModel("joint" + std::to_string(i + 1));
        if (jm) robot_state_->setJointPositions(jm, &q(i));
    }
    robot_state_->update();

    const auto* link_model = robot_model_->getLinkModel("wrist3_link");
    if (!link_model) {
        return Vec3::Zero();
    }

    Eigen::Isometry3d tf = robot_state_->getGlobalLinkTransform(link_model);
    return tf.translation();
}

Eigen::Matrix<double, 6, N_JOINTS> MoveItIntegration::computeJacobian(const VecN& q) const {
    Eigen::Matrix<double, 6, N_JOINTS> jacobian = Eigen::Matrix<double, 6, N_JOINTS>::Zero();

    if (!robot_state_ || !robot_model_) {
        return jacobian;
    }

    // 设置关节状态
    for (int i = 0; i < N_JOINTS; ++i) {
        const auto* jm = robot_model_->getJointModel("joint" + std::to_string(i + 1));
        if (jm) robot_state_->setJointPositions(jm, &q(i));
    }
    robot_state_->update();

    // MoveIt2 Humble: getJacobian 需要 JointModelGroup* 和 LinkModel*
    const auto* jmg = robot_model_->getJointModelGroup("arm");
    const auto* link_model = robot_model_->getLinkModel("wrist3_link");
    if (!jmg || !link_model) {
        return jacobian;
    }

    Eigen::MatrixXd full_jacobian;
    robot_state_->getJacobian(jmg, link_model, Eigen::Vector3d::Zero(), full_jacobian);

    if (full_jacobian.cols() >= N_JOINTS) {
        jacobian = full_jacobian.block(0, 0, 6, N_JOINTS);
    }

    return jacobian;
}

bool MoveItIntegration::checkCollision(const VecN& q, const std::vector<Obstacle>& obstacles) const {
    if (!planning_scene_monitor_ || !robot_state_ || !robot_model_) {
        return false;
    }

    for (int i = 0; i < N_JOINTS; ++i) {
        const auto* jm = robot_model_->getJointModel("joint" + std::to_string(i + 1));
        if (jm) robot_state_->setJointPositions(jm, &q(i));
    }
    robot_state_->update();

    auto scene = planning_scene_monitor_->getPlanningScene();
    if (!scene) {
        return false;
    }

    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;
    req.contacts = true;
    req.max_contacts = 100;

    scene->checkCollision(req, res, *robot_state_);
    return res.collision;
}

double MoveItIntegration::computeMinDistance(const VecN& q, const std::vector<Obstacle>& obstacles) const {
    if (obstacles.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    // 计算机器人连杆位置
    auto link_positions = computeLinkPositions(q);

    double min_distance = std::numeric_limits<double>::infinity();

    // 计算每个连杆到每个障碍物的最小距离
    for (const auto& link_pos : link_positions) {
        for (const auto& obstacle : obstacles) {
            double distance = pointToObstacleDistance(link_pos, obstacle);
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
    }

    return min_distance;
}

void MoveItIntegration::addDynamicObstacle(const Obstacle& obstacle, const std::string& id) {
    if (!planning_scene_monitor_) {
        return;
    }

    auto scene = planning_scene_monitor_->getPlanningScene();
    if (!scene) {
        return;
    }

    // 创建碰撞物体
    moveit_msgs::msg::CollisionObject collision_obj;
    collision_obj.header.frame_id = "world";
    collision_obj.id = id;
    collision_obj.operation = moveit_msgs::msg::CollisionObject::ADD;

    // 创建盒子形状
    shape_msgs::msg::SolidPrimitive box;
    box.type = shape_msgs::msg::SolidPrimitive::BOX;
    box.dimensions = {obstacle.size.x(), obstacle.size.y(), obstacle.size.z()};
    collision_obj.primitives.push_back(box);

    // 设置位置
    geometry_msgs::msg::Pose pose;
    pose.position.x = obstacle.center.x();
    pose.position.y = obstacle.center.y();
    pose.position.z = obstacle.center.z();
    pose.orientation.w = 1.0;
    collision_obj.primitive_poses.push_back(pose);

    // 添加到场景
    scene->processCollisionObjectMsg(collision_obj);
}

void MoveItIntegration::removeDynamicObstacle(const std::string& id) {
    if (!planning_scene_monitor_) {
        return;
    }

    auto scene = planning_scene_monitor_->getPlanningScene();
    if (!scene) {
        return;
    }

    // MoveIt2 Humble: 使用 processCollisionObjectMsg 发送 REMOVE 操作
    moveit_msgs::msg::CollisionObject obj;
    obj.id = id;
    obj.header.frame_id = "world";
    obj.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    scene->processCollisionObjectMsg(obj);
}

void MoveItIntegration::updateDynamicObstacle(const std::string& id, const Vec3& new_position) {
    if (!planning_scene_monitor_) {
        return;
    }

    auto scene = planning_scene_monitor_->getPlanningScene();
    if (!scene) {
        return;
    }

    // 获取现有物体
    auto obj = scene->getWorld()->getObject(id);
    if (!obj) {
        return;
    }

    // 更新位置
    Eigen::Isometry3d new_pose = Eigen::Isometry3d::Identity();
    new_pose.translation() = new_position;

    scene->getWorldNonConst()->moveObject(id, new_pose);
}

planning_scene::PlanningScenePtr MoveItIntegration::getPlanningScene() const {
    if (!planning_scene_monitor_) {
        return nullptr;
    }
    return planning_scene_monitor_->getPlanningScene();
}

// 私有方法实现
std::vector<Vec3> MoveItIntegration::computeLinkPositions(const VecN& q) const {
    std::vector<Vec3> positions;

    if (!robot_state_ || !robot_model_) {
        return positions;
    }

    for (int i = 0; i < N_JOINTS; ++i) {
        const auto* jm = robot_model_->getJointModel("joint" + std::to_string(i + 1));
        if (jm) robot_state_->setJointPositions(jm, &q(i));
    }
    robot_state_->update();

    std::vector<std::string> link_names = {
        "shoulder_link", "upperarm_link", "forearm_link",
        "wrist1_link", "wrist2_link", "wrist3_link"
    };

    for (const auto& name : link_names) {
        const auto* link_model = robot_model_->getLinkModel(name);
        if (link_model) {
            Eigen::Isometry3d tf = robot_state_->getGlobalLinkTransform(link_model);
            positions.push_back(tf.translation());
        }
    }

    return positions;
}

double MoveItIntegration::pointToObstacleDistance(const Vec3& point, const Obstacle& obstacle) const {
    // 使用 SmoothBoxDistance::compute（对应Matlab的距离计算）
    return SmoothBoxDistance::compute(point, obstacle.center, obstacle.size, 10.0);
}

} // namespace fairino_mpc
