/**
 * @file scene_obstacle_provider.hpp
 * @brief 场景静态障碍物提供者
 *
 * 从 MoveIt PlanningScene 中提取静态碰撞对象，并转换为 MPC 内部统一使用的 Obstacle 格式。
 * 将格式转换逻辑从节点主循环中抽离，降低耦合度。
 *
 * 典型用法：
 * - 在每个控制周期（或按需）调用 collectStaticObstacles()，
 *   传入 PlanningSceneMonitor 和 ObstacleTracker，获取当前场景中的静态障碍物列表。
 */

#pragma once

#include <memory>
#include <vector>

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

#include "fairino_mpc_avoidance/obstacle_tracker.hpp"
#include "fairino_mpc_avoidance/types.hpp"

namespace fairino_mpc {

/**
 * @class SceneObstacleProvider
 * @brief 提供从 MoveIt 规划场景收集静态障碍物的功能
 */
class SceneObstacleProvider {
public:
    /**
     * @brief 收集并转换规划场景中的静态障碍物
     *
     * @param planning_scene_monitor 指向 PlanningSceneMonitor 的共享指针。
     *                               若为空或内部场景为空，则返回空列表。
     * @param obstacle_tracker       障碍物跟踪器，用于查询已知障碍物的速度信息
     *                              （静态障碍物速度一般为零，但仍通过此接口获取）。
     * @return std::vector<Obstacle> 转换后的 MPC 内部 Obstacle 列表。
     *                               仅包含成功解析的形状（盒子、球体）。
     *                               所有障碍物的 is_dynamic 字段被标记为 false。
     */
    std::vector<Obstacle> collectStaticObstacles(
        const std::shared_ptr<planning_scene_monitor::PlanningSceneMonitor>& planning_scene_monitor,
        const ObstacleTracker& obstacle_tracker) const;
};

}  // namespace fairino_mpc