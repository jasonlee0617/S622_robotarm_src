#pragma once
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "fairino_mpc_avoidance/types.hpp"
#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <filesystem>
#include <Eigen/Core>

namespace fairino_mpc {

/// 完整的障碍物场景配置（供 obstacle_simulator 使用）
struct FullScenarioConfig {
    struct StaticEntry {
        std::string id;
        Eigen::Vector3d center;
        Eigen::Vector3d size;
    };

    struct DynamicEntry {
        std::string name;
        Eigen::Vector3d size;
        Eigen::Vector3d velocity;
        Eigen::Vector3d bounds_lo;
        Eigen::Vector3d bounds_hi;
    };

    std::vector<StaticEntry> static_obstacles;
    std::vector<DynamicEntry> dynamic_obstacles;
};

class ScenarioLoader {
public:
    static void validateSpawnScenarioConsistency(const YAML::Node& root,
                                                 const std::string& yaml_path) {
        if (!root["spawn"] || !root["spawn"]["dynamic"] || !root["scenario"] || !root["scenario"]["dynamic_obstacles"]) {
            return;
        }

        std::set<std::string> spawn_names;
        std::set<std::string> scenario_names;

        for (const auto& node : root["spawn"]["dynamic"]) {
            if (node["enabled"] && !node["enabled"].as<bool>()) continue;
            if (!node["name"]) continue;
            spawn_names.insert(node["name"].as<std::string>());
        }
        for (const auto& node : root["scenario"]["dynamic_obstacles"]) {
            if (!node["name"]) continue;
            scenario_names.insert(node["name"].as<std::string>());
        }

        for (const auto& name : spawn_names) {
            if (scenario_names.find(name) == scenario_names.end()) {
                std::cerr << "[ScenarioLoader][WARN] dynamic spawn '" << name
                          << "' exists in spawn.dynamic but not in scenario.dynamic_obstacles: "
                          << yaml_path << std::endl;
            }
        }
        for (const auto& name : scenario_names) {
            if (spawn_names.find(name) == spawn_names.end()) {
                std::cerr << "[ScenarioLoader][WARN] dynamic scenario '" << name
                          << "' exists in scenario.dynamic_obstacles but not in spawn.dynamic: "
                          << yaml_path << std::endl;
            }
        }
    }

    /// @brief 从 YAML 文件加载障碍物场景（完整配置，供 simulator）
    static FullScenarioConfig loadFull(const std::string& yaml_path) {
        FullScenarioConfig config;
        YAML::Node root = YAML::LoadFile(yaml_path);
        YAML::Node scenario = root["scenario"] ? root["scenario"] : root;
        validateSpawnScenarioConsistency(root, yaml_path);

        if (scenario["static_obstacles"]) {
            for (const auto& node : scenario["static_obstacles"]) {
                FullScenarioConfig::StaticEntry s;
                s.id = node["id"].as<std::string>();
                auto c = node["center"].as<std::vector<double>>();
                auto z = node["size"].as<std::vector<double>>();
                s.center = Eigen::Vector3d(c[0], c[1], c[2]);
                s.size   = Eigen::Vector3d(z[0], z[1], z[2]);
                config.static_obstacles.push_back(s);
            }
        }

        if (scenario["dynamic_obstacles"]) {
            for (const auto& node : scenario["dynamic_obstacles"]) {
                FullScenarioConfig::DynamicEntry d;
                d.name = node["name"].as<std::string>();
                auto sz  = node["size"].as<std::vector<double>>();
                auto vel = node["velocity"].as<std::vector<double>>();
                auto lo  = node["bounds"]["lo"].as<std::vector<double>>();
                auto hi  = node["bounds"]["hi"].as<std::vector<double>>();
                d.size      = Eigen::Vector3d(sz[0], sz[1], sz[2]);
                d.velocity  = Eigen::Vector3d(vel[0], vel[1], vel[2]);
                d.bounds_lo = Eigen::Vector3d(lo[0], lo[1], lo[2]);
                d.bounds_hi = Eigen::Vector3d(hi[0], hi[1], hi[2]);
                config.dynamic_obstacles.push_back(d);
            }
        }

        return config;
    }

    /// @brief 仅提取动态障碍物边界配置（供 MPC 节点 obstacle tracker）
    static std::vector<DynamicObstacleConfig> loadDynamicConfigs(const std::string& yaml_path) {
        std::vector<DynamicObstacleConfig> configs;
        YAML::Node root = YAML::LoadFile(yaml_path);
        YAML::Node scenario = root["scenario"] ? root["scenario"] : root;
        validateSpawnScenarioConsistency(root, yaml_path);

        if (scenario["dynamic_obstacles"]) {
            for (const auto& node : scenario["dynamic_obstacles"]) {
                DynamicObstacleConfig cfg;
                cfg.name = node["name"].as<std::string>();
                auto sz = node["size"].as<std::vector<double>>();
                auto lo = node["bounds"]["lo"].as<std::vector<double>>();
                auto hi = node["bounds"]["hi"].as<std::vector<double>>();
                cfg.size       = Vec3(sz[0], sz[1], sz[2]);
                cfg.bounds_min = Vec3(lo[0], lo[1], lo[2]);
                cfg.bounds_max = Vec3(hi[0], hi[1], hi[2]);
                configs.push_back(cfg);
            }
        }

        return configs;
    }

    /// @brief 从 ROS 参数获取 YAML 路径并加载
    static std::string resolvePath(rclcpp::Node& node) {
        std::string path = node.get_parameter("scenario_config").as_string();
        if (path.empty()) {
            try {
                const auto share_dir =
                    ament_index_cpp::get_package_share_directory("fairino_mpc_avoidance");
                path = (std::filesystem::path(share_dir) / "config" / "obstacle_stack.yaml")
                           .string();
            } catch (const std::exception&) {
                // 兜底：兼容源码目录直接运行（相对工作区）
                const auto cwd = std::filesystem::current_path();
                const std::vector<std::filesystem::path> candidates{
                    cwd / "src" / "fairino_mpc_avoidance" / "config" / "obstacle_stack.yaml",
                    cwd / "fairino_mpc_avoidance" / "config" / "obstacle_stack.yaml",
                    std::filesystem::path("config") / "obstacle_stack.yaml"
                };
                for (const auto& c : candidates) {
                    if (std::filesystem::exists(c)) {
                        path = c.string();
                        break;
                    }
                }
            }
        }
        return path;
    }
};

}  // namespace fairino_mpc
