/**
 * file obstacle_simulator.cpp
 * brief 动态障碍物仿真节点
 *
 * 本文件实现了 Gazebo 中动态障碍物的运动驱动与感知信息发布：
 * 1. 通过 Ignition Transport 向 Gazebo 模型发送 cmd_vel 指令，实现反弹运动。
 * 2. 订阅 Gazebo 世界位姿广播，获取障碍物真实位姿。
 * 3. 将障碍物位姿发布为 /detected_obstacles (PoseArray)，供 MPC 避障使用。
 * 4. 发布 RViz 可视化 Marker (动态/静态障碍物)。
 * 5. 周期性发布静态障碍物到 /planning_scene，供 MoveIt 场景监视器使用。
 *
 * 运动模型仅保留反弹模式，障碍物尺寸、速度、边界等参数由 ScenarioLoader 从 YAML 加载。
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

// Ignition Transport 用于与 Gazebo 通信 (cmd_vel 发送 + pose 订阅)
#include <ignition/transport/Node.hh>
#include <ignition/msgs/twist.pb.h>
#include <ignition/msgs/pose_v.pb.h>

#include "myrobot_mpc_avoidance/scenario_loader.hpp"
#include <Eigen/Geometry>
#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <fstream>

/**
 * class ObstacleSimulator
 * brief 障碍物仿真节点主类
 *
 * 负责管理所有动态障碍物的生命周期：加载配置、接收真实位姿、发送速度指令、
 * 发布感知话题和可视化数据。
 */
class ObstacleSimulator : public rclcpp::Node {
public:
    /**
     * brief 构造函数：声明参数、加载场景、初始化通信和定时器
     */
    ObstacleSimulator() : Node("obstacle_simulator") {
        // 声明 ROS 参数，可从 launch 文件或命令行覆盖
        declare_parameter("simulation_rate", 30.0);          // 主循环频率 (Hz)
        declare_parameter("world_name", std::string("empty")); // Gazebo 世界名称
        declare_parameter("scenario_config", std::string("")); // 障碍物场景配置文件路径

        simulation_rate_ = get_parameter("simulation_rate").as_double();
        world_name_ = get_parameter("world_name").as_string();

        // 加载场景配置（动态/静态障碍物）
        loadObstacleScenario();

        // 订阅 Gazebo 世界位姿广播（获取真实位姿）
        ign_node_.Subscribe(
            "/world/" + world_name_ + "/pose/info",
            &ObstacleSimulator::onWorldPoseInfo, this);

        // 发布 /detected_obstacles (供 MPC 节点使用)
        obstacles_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
            "/detected_obstacles", 10);

        // 发布 RViz 可视化标记
        markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "/obstacle_markers", 10);

        // 发布静态障碍物到 PlanningScene (MoveIt 场景监视器)，每秒一次
        scene_pub_ = create_publisher<moveit_msgs::msg::PlanningScene>(
            "/planning_scene", 10);
        static_pub_timer_ = create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&ObstacleSimulator::publishStaticObstacles, this));

        // 主循环定时器：发送 cmd_vel 并发布 /detected_obstacles
        simulation_timer_ = create_wall_timer(
            std::chrono::duration<double>(1.0 / simulation_rate_),
            std::bind(&ObstacleSimulator::simulationLoop, this));

        RCLCPP_INFO(get_logger(),
            "ObstacleSimulator (cmd_vel) started: %zu dynamic, %zu static, world=%s, mode=bouncing",
            dyn_configs_.size(), static_obstacles_.size(),
            world_name_.c_str());
    }

private:
    /**
     * struct DynConfig
     * brief 动态障碍物的配置参数（从 YAML 加载）
     */
    struct DynConfig {
        std::string name;           // Gazebo 模型名称
        Eigen::Vector3d size;       // 障碍物半尺寸 (盒子)

        // world-frame 速度 (m/s)，实际运动方向由反弹逻辑动态调整
        double vx = 0.0;
        double vy = 0.0;
        double vz = 0.0;

        double wz = 0.0;           // 绕 z 轴角速度 (rad/s)

        double bounds_lo[3];        // 运动下界 [x,y,z]
        double bounds_hi[3];        // 运动上界 [x,y,z]
    };

    /**
     * struct DynamicPose
     * brief 动态障碍物当前位姿（从 Gazebo 获取）
     */
    struct DynamicPose {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();      // 位置
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity(); // 姿态
        bool received = false;    // 是否已接收到有效位姿
    };

    /**
     * struct StaticObstacle
     * brief 静态障碍物定义
     */
    struct StaticObstacle {
        std::string id;            // 障碍物 ID
        Eigen::Vector3d center;    // 中心位置
        Eigen::Vector3d size;      // 半尺寸
    };

    /**
     * brief 使用 ScenarioLoader 统一加载障碍物场景配置
     *
     * 解析 YAML 文件，填充动态障碍物配置列表、静态障碍物列表，
     * 并为每个动态障碍物初始化位姿接收槽。
     */
    void loadObstacleScenario() {
        std::string config_path = fairino_mpc::ScenarioLoader::resolvePath(*this);

        RCLCPP_INFO(get_logger(), "Loading obstacle scenario from: %s", config_path.c_str());

        try {
            auto scenario = fairino_mpc::ScenarioLoader::loadFull(config_path);

            // 填充静态障碍物列表
            for (const auto& s : scenario.static_obstacles) {
                StaticObstacle so;
                so.id     = s.id;
                so.center = s.center;
                so.size   = s.size;
                static_obstacles_.push_back(so);
                RCLCPP_INFO(get_logger(), "  static: %s @ (%.2f,%.2f,%.2f)",
                    s.id.c_str(), s.center.x(), s.center.y(), s.center.z());
            }

            // 填充动态障碍物配置，并初始化对应位姿记录
            for (const auto& d : scenario.dynamic_obstacles) {
                DynConfig cfg;
                cfg.name = d.name;
                cfg.size = d.size;
                cfg.vx = d.velocity.x(); cfg.vy = d.velocity.y(); cfg.vz = d.velocity.z();
                cfg.bounds_lo[0] = d.bounds_lo.x();
                cfg.bounds_lo[1] = d.bounds_lo.y();
                cfg.bounds_lo[2] = d.bounds_lo.z();
                cfg.bounds_hi[0] = d.bounds_hi.x();
                cfg.bounds_hi[1] = d.bounds_hi.y();
                cfg.bounds_hi[2] = d.bounds_hi.z();

                dyn_configs_.push_back(cfg);
                dyn_poses_[cfg.name] = DynamicPose{};  // 预留位姿槽
                RCLCPP_INFO(get_logger(),
                    "  dynamic: %s v=(%.2f,%.2f,%.2f) bounds_lo=(%.2f,%.2f,%.2f) bounds_hi=(%.2f,%.2f,%.2f)",
                    cfg.name.c_str(), cfg.vx, cfg.vy, cfg.vz,
                    cfg.bounds_lo[0], cfg.bounds_lo[1], cfg.bounds_lo[2],
                    cfg.bounds_hi[0], cfg.bounds_hi[1], cfg.bounds_hi[2]);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to load scenario: %s", e.what());
        }
    }

    /**
     * brief Ignition 回调：接收 Gazebo 世界位姿广播
     * param msg 包含所有模型位姿的消息
     *
     * 遍历消息中的每个模型，若其名称与动态障碍物名称匹配（支持 "name"、"name::link" 等形式），
     * 则更新该障碍物的真实位姿，并标记 received = true。
     */
    void onWorldPoseInfo(const ignition::msgs::Pose_V& msg) {
        std::lock_guard<std::mutex> lock(pose_mutex_);  // 线程安全

        for (int i = 0; i < msg.pose_size(); ++i) {
            const auto& p = msg.pose(i);
            const std::string gz_name = p.name();  // Gazebo 中的模型/连杆名称

            for (auto& kv : dyn_poses_) {
                const std::string& obs_name = kv.first;

                // 匹配规则：完全匹配，或 "name::link"，或以 "name::" 开头
                bool matched =
                    (gz_name == obs_name) ||
                    (gz_name == obs_name + "::link") ||
                    (gz_name.find(obs_name + "::") == 0);

                if (!matched) {
                    continue;
                }

                // 更新位置和姿态
                kv.second.pos = Eigen::Vector3d(
                    p.position().x(),
                    p.position().y(),
                    p.position().z());

                kv.second.quat = Eigen::Quaterniond(
                    p.orientation().w(),
                    p.orientation().x(),
                    p.orientation().y(),
                    p.orientation().z());

                kv.second.received = true;
                break;  // 一个模型只匹配一个动态障碍物
            }
        }
    }

    /**
     * brief 主循环：依次执行速度指令发送、感知数据发布和可视化更新
     */
    void simulationLoop() {
        sendCmdVel();              // 向 Gazebo 发送 cmd_vel
        publishDetectedObstacles(); // 发布 /detected_obstacles
        publishMarkers();           // 发布 RViz 标记
    }

    /**
     * brief 为每个动态障碍物计算速度指令并发送到 Gazebo
     *
     * 采用闭环反弹控制：
     * - 对于有明确运动范围的轴（lo < hi），在边界处自动反向速度。
     * - 对于固定轴（lo == hi），使用 P 控制将障碍物拉回目标位置。
     * - 未接收到真实位姿时，使用初始配置速度开环运动。
     */
    void sendCmdVel() {
        static int loop_cnt = 0;
        loop_cnt++;

        // 辅助函数：值钳制
        auto clampValue = [](double v, double lo, double hi) {
            return std::max(lo, std::min(v, hi));
        };

        /**
         * @brief 单轴速度指令生成（反弹 + 固定位置保持）
         * @param pos 当前位置
         * @param lo  下界
         * @param hi  上界
         * @param state_v 当前轴向速度（传入传出，在边界处反转）
         * @param max_hold_v 固定轴保持时最大 P 控制速度
         * @param boundary_margin 边界触发反向的裕度
         * @return 指令速度
         */
        auto axisCommand = [&](double pos,
                            double lo,
                            double hi,
                            double& state_v,
                            double max_hold_v,
                            double boundary_margin) -> double {
            // 固定轴：hi ≈ lo，使用 P 控制保持目标位置
            if (hi <= lo + 1e-6) {
                double target = lo;
                double err = target - pos;
                double v_cmd = 2.0 * err;  // 简单 P 控制，增益 2.0
                return clampValue(v_cmd, -max_hold_v, max_hold_v);
            }

            // 往返轴：到边界附近时反转速度方向
            double base = std::abs(state_v);
            if (base < 1e-6) {
                return 0.0;  // 速度为 0 则不输出
            }

            if (pos <= lo + boundary_margin) {
                state_v = base;          // 正向
            } else if (pos >= hi - boundary_margin) {
                state_v = -base;         // 反向
            }

            return state_v;
        };

        // 遍历所有动态障碍物
        for (auto& cfg : dyn_configs_) {
            DynamicPose pose;
            bool has_pose = false;

            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                auto it = dyn_poses_.find(cfg.name);
                if (it != dyn_poses_.end() && it->second.received) {
                    pose = it->second;
                    has_pose = true;
                }
            }

            double vx = 0.0;
            double vy = 0.0;
            double vz = 0.0;
            double wx = 0.0;
            double wy = 0.0;
            double wz = cfg.wz;  // 角速度暂不使用闭环

            if (has_pose) {
                // 有真实位姿时使用闭环反弹控制
                vx = axisCommand(
                    pose.pos.x(),
                    cfg.bounds_lo[0],
                    cfg.bounds_hi[0],
                    cfg.vx,
                    0.20,   // 固定轴保持速度上限
                    0.02);  // 边界裕度

                vy = axisCommand(
                    pose.pos.y(),
                    cfg.bounds_lo[1],
                    cfg.bounds_hi[1],
                    cfg.vy,
                    0.20,
                    0.02);

                vz = axisCommand(
                    pose.pos.z(),
                    cfg.bounds_lo[2],
                    cfg.bounds_hi[2],
                    cfg.vz,
                    0.20,
                    0.02);
            } else {
                // 未收到位姿时使用初始配置速度开环运动
                vx = cfg.vx;
                vy = cfg.vy;
                vz = cfg.vz;
            }

            // 构造 ignition Twist 消息
            ignition::msgs::Twist twist;
            twist.mutable_linear()->set_x(vx);
            twist.mutable_linear()->set_y(vy);
            twist.mutable_linear()->set_z(vz);
            twist.mutable_angular()->set_x(wx);
            twist.mutable_angular()->set_y(wy);
            twist.mutable_angular()->set_z(wz);

            // 获取或创建 cmd_vel 发布器，并发布指令
            auto& pub = cmd_vel_pubs_[cfg.name];
            if (!pub) {
                pub = std::make_unique<ignition::transport::Node::Publisher>(
                    ign_node_.Advertise<ignition::msgs::Twist>(
                        "/model/" + cfg.name + "/cmd_vel"));
                RCLCPP_INFO(get_logger(),
                    "Advertised cmd_vel for %s", cfg.name.c_str());
            }
            pub->Publish(twist);
        }
    }

    /**
     * brief 读取 Gazebo 真实位姿，发布到 /detected_obstacles
     *
     * 只有当所有动态障碍物都收到有效位姿时才发布，保证数据完整性。
     */
    void publishDetectedObstacles() {
        auto msg = geometry_msgs::msg::PoseArray();
        msg.header.stamp = now();
        msg.header.frame_id = "world";

        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            for (const auto& cfg : dyn_configs_) {
                auto it = dyn_poses_.find(cfg.name);
                if (it != dyn_poses_.end() && it->second.received) {
                    geometry_msgs::msg::Pose pose;
                    pose.position.x = it->second.pos.x();
                    pose.position.y = it->second.pos.y();
                    pose.position.z = it->second.pos.z();
                    pose.orientation.w = it->second.quat.w();
                    pose.orientation.x = it->second.quat.x();
                    pose.orientation.y = it->second.quat.y();
                    pose.orientation.z = it->second.quat.z();
                    msg.poses.push_back(pose);
                }
            }
        }

        // 仅当所有动态障碍物位姿均可用时发布
        if (msg.poses.size() == dyn_configs_.size()) {
            obstacles_pub_->publish(msg);
        } else {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Waiting for all dynamic obstacle poses before publishing: %zu/%zu",
                msg.poses.size(), dyn_configs_.size());
        }
    }

    /**
     * brief 发布 RViz 可视化标记（动态障碍物橙色，静态障碍物蓝色）
     */
    void publishMarkers() {
        auto markers = visualization_msgs::msg::MarkerArray();
        int id = 0;

        // 动态障碍物标记
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            for (const auto& cfg : dyn_configs_) {
                auto it = dyn_poses_.find(cfg.name);
                visualization_msgs::msg::Marker m;
                m.header.stamp = now();
                m.header.frame_id = "world";
                m.ns = "dynamic_obstacles";
                m.id = id++;
                m.type = visualization_msgs::msg::Marker::CUBE;
                m.action = visualization_msgs::msg::Marker::ADD;

                // 使用真实位姿（若可用），否则使用下界位置
                if (it != dyn_poses_.end() && it->second.received) {
                    m.pose.position.x = it->second.pos.x();
                    m.pose.position.y = it->second.pos.y();
                    m.pose.position.z = it->second.pos.z();
                } else {
                    m.pose.position.x = cfg.bounds_lo[0];
                    m.pose.position.y = cfg.bounds_lo[1];
                    m.pose.position.z = cfg.bounds_lo[2];
                }
                m.pose.orientation.w = 1.0;
                m.scale.x = cfg.size.x();
                m.scale.y = cfg.size.y();
                m.scale.z = cfg.size.z();
                m.color.r = 1.0; m.color.g = 0.3; m.color.b = 0.0; m.color.a = 0.8; // 橙色
                m.lifetime = rclcpp::Duration::from_seconds(0.2);  // 短暂生命周期，避免残留
                markers.markers.push_back(m);
            }
        }

        // 静态障碍物标记
        for (const auto& obs : static_obstacles_) {
            visualization_msgs::msg::Marker m;
            m.header.stamp = now();
            m.header.frame_id = "world";
            m.ns = "static_obstacles";
            m.id = id++;
            m.type = visualization_msgs::msg::Marker::CUBE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = obs.center.x();
            m.pose.position.y = obs.center.y();
            m.pose.position.z = obs.center.z();
            m.pose.orientation.w = 1.0;
            m.scale.x = obs.size.x();
            m.scale.y = obs.size.y();
            m.scale.z = obs.size.z();
            m.color.r = 0.2; m.color.g = 0.4; m.color.b = 1.0; m.color.a = 0.7; // 蓝色
            m.lifetime = rclcpp::Duration::from_seconds(1.0);
            markers.markers.push_back(m);
        }

        markers_pub_->publish(markers);
    }

    /**
     * brief 周期性发布静态障碍物到 PlanningScene，供 MoveIt 场景监视器使用
     *
     * 频率：1 Hz，由 static_pub_timer_ 触发。
     */
    void publishStaticObstacles() {
        auto scene_msg = moveit_msgs::msg::PlanningScene();
        scene_msg.is_diff = true;  // 增量更新

        for (const auto& obs : static_obstacles_) {
            moveit_msgs::msg::CollisionObject obj;
            obj.header.stamp = now();
            obj.header.frame_id = "base_link";
            obj.id = obs.id;
            obj.operation = moveit_msgs::msg::CollisionObject::ADD;

            // 创建盒子形状
            shape_msgs::msg::SolidPrimitive box;
            box.type = shape_msgs::msg::SolidPrimitive::BOX;
            box.dimensions = {obs.size.x(), obs.size.y(), obs.size.z()};
            obj.primitives.push_back(box);

            geometry_msgs::msg::Pose pose;
            pose.position.x = obs.center.x();
            pose.position.y = obs.center.y();
            pose.position.z = obs.center.z();
            pose.orientation.w = 1.0;
            obj.primitive_poses.push_back(pose);

            scene_msg.world.collision_objects.push_back(obj);
        }

        scene_pub_->publish(scene_msg);
    }

    // ─── 成员变量 ──────────────────────────────────────────────────
    std::vector<DynConfig> dyn_configs_;               // 动态障碍物配置列表
    std::vector<StaticObstacle> static_obstacles_;     // 静态障碍物列表
    std::unordered_map<std::string, DynamicPose> dyn_poses_; // 动态障碍物当前位姿 (以名称为键)
    std::mutex pose_mutex_;                            // 位姿读写互斥锁

    double simulation_rate_;                           // 仿真频率
    std::string world_name_;                           // Gazebo 世界名称
    int static_publish_count_ = 0;                     // 静态障碍物发布计数 (未使用)

    ignition::transport::Node ign_node_;               // Ignition 通信节点
    std::unordered_map<std::string, std::unique_ptr<ignition::transport::Node::Publisher>> cmd_vel_pubs_; // cmd_vel 发布器缓存

    // ROS 2 发布者和定时器
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr obstacles_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
    rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr scene_pub_;
    rclcpp::TimerBase::SharedPtr simulation_timer_;
    rclcpp::TimerBase::SharedPtr static_pub_timer_;
};

/**
 * brief 程序入口：初始化 ROS 2，创建节点并进入事件循环
 */
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObstacleSimulator>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
