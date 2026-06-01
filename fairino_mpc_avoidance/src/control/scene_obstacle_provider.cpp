/**
 * file scene_obstacle_provider.cpp
 * brief 从 MoveIt PlanningScene 提取静态障碍物并转换为 MPC 内部格式
 *
 * 将 MoveIt 场景监视器（PlanningSceneMonitor）中的碰撞对象（CollisionObject）
 * 解析为 MPC 模块使用的 Obstacle 结构体。
 * 目的是将格式转换逻辑与节点主循环解耦，保持节点外壳的简洁性。
 *
 * 典型用法：
 * - 每个控制周期（或按需）调用 collectStaticObstacles() 获取当前场景中的静态障碍物列表。
 */

#include "fairino_mpc_avoidance/control/scene_obstacle_provider.hpp"

#include <memory>

#include <geometric_shapes/shapes.h>   // 用于识别和访问基本形状（盒子、球体等）

namespace fairino_mpc {

/**
 * brief 收集并转换规划场景中的静态障碍物
 *
 * param planning_scene_monitor  指向 MoveIt PlanningSceneMonitor 的共享指针，
 *                                若为空或内部场景为空，则返回空列表。
 * param obstacle_tracker        障碍物跟踪器，用于查询已知障碍物的速度信息
 *                                （虽然静态障碍物速度通常为零，但仍通过此接口获取，
 *                                  并最终标记 is_dynamic = false）
 * return std::vector<Obstacle>  转换后的 MPC 障碍物列表，只包含成功解析的形状。
 */
std::vector<Obstacle> SceneObstacleProvider::collectStaticObstacles(
    const std::shared_ptr<planning_scene_monitor::PlanningSceneMonitor>& planning_scene_monitor,
    const ObstacleTracker& obstacle_tracker) const {
    std::vector<Obstacle> obs;

    // 空指针保护：若场景监视器不可用，直接返回空列表，避免崩溃
    if (!planning_scene_monitor) {
        return obs;
    }
    // 获取当前规划场景
    auto scene = planning_scene_monitor->getPlanningScene();
    if (!scene) {
        return obs;
    }
    // 获取世界对象（包含所有碰撞对象）
    auto world = scene->getWorld();
    if (!world) {
        return obs;
    }

    // 遍历世界中的所有碰撞对象（以 ID 为键的映射）
    for (const auto& [id, obj] : *world) {
        // 一个碰撞对象可能包含多个形状（primitives）
        // 每个形状对应一个独立的障碍物条目
        for (size_t i = 0; i < obj->shapes_.size(); ++i) {
            Obstacle o;
            // 提取形状位姿（通常相对于世界坐标系）
            const auto& pose = obj->shape_poses_[i];
            o.center = Vec3(pose.translation().x(), pose.translation().y(), pose.translation().z());

            // 根据形状类型提取尺寸信息，仅支持盒子和球体，其他形状跳过
            if (auto box = std::dynamic_pointer_cast<const shapes::Box>(obj->shapes_[i])) {
                // 盒子：尺寸直接使用 [长, 宽, 高]
                o.size = Vec3(box->size[0], box->size[1], box->size[2]);
            } else if (auto sphere = std::dynamic_pointer_cast<const shapes::Sphere>(obj->shapes_[i])) {
                // 球体：用包围盒的边长（直径）作为尺寸
                const double r = sphere->radius;
                o.size = Vec3(r * 2.0, r * 2.0, r * 2.0);
            } else {
                // 不支持的类型（如圆柱、网格等），忽略
                continue;
            }

            // 尝试从障碍物跟踪器获取速度（静态障碍物速度通常为 0）
            o.velocity = obstacle_tracker.getVelocity(id);
            // 明确标记为静态（非动态），MPC 会将其视作环境的一部分，不做运动预测
            o.is_dynamic = false;
            obs.push_back(o);
        }
    }
    return obs;
}

}  // namespace fairino_mpc