/**
 * file obstacle_tracker.cpp
 * brief 动态障碍物跟踪与预测
 *
 * 本文件实现障碍物跟踪器（ObstacleTracker），主要功能：
 * 1. 从 ROS PoseArray 消息更新障碍物位置，并基于历史窗口估计速度。
 * 2. 维护障碍物的尺寸、运动边界等属性，支持从规划场景更新静态障碍物。
 * 3. 提供当前障碍物快照获取接口。
 * 4. 基于恒速模型进行未来 N 步预测，包含速度膨胀尺寸和边界反弹逻辑，
 *    用于 MPC 障碍物前向预测。
 *
 * 内部使用有序 map 以障碍物 ID 为键存储追踪状态，每个障碍物保存
 * 最近一段时间内的位置和时间戳历史，通过首尾差分估算速度。
 */

#include "myrobot_mpc_avoidance/obstacle_tracker.hpp"
#include <algorithm>
#include <cmath>

namespace fairino_mpc {

void ObstacleTracker::setTrackingParams(double velocity_window_sec, double dynamic_speed_threshold) {
    velocity_window_ = std::max(0.05, velocity_window_sec);
    dynamic_speed_threshold_ = std::max(0.0, dynamic_speed_threshold);
}

/**
 * brief 更新障碍物状态（来自动态障碍物检测话题）
 *
 * param msg PoseArray 消息，每个 pose 表示一个障碍物的当前位置，
 *            时间戳取自消息头。
 *
 * 为每个检测到的障碍物分配 ID "obs_0", "obs_1" ...，
 * 记录当前位置并维护最近 velocity_window_ 秒内的历史，
 * 然后重新估算其速度和动态标志。
 */
void ObstacleTracker::update(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
    // 提取消息时间戳（秒）
    double t_now = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    for (size_t i = 0; i < msg->poses.size(); ++i) {
        // 为每个障碍物分配唯一 ID
        std::string id = "obs_" + std::to_string(i);
        // 提取位置
        Vec3 pos(msg->poses[i].position.x,
                 msg->poses[i].position.y,
                 msg->poses[i].position.z);

        // 获取或创建该障碍物的追踪记录
        auto& obs = tracked_[id];
        obs.id = id;
        obs.center = pos;          // 更新当前位置

        // 将当前状态加入历史记录
        obs.position_history.push_back(pos);
        obs.timestamp_history.push_back(t_now);

        // 清理过期历史：只保留最近 velocity_window_ 秒内的数据
        while (!obs.timestamp_history.empty() &&
               (t_now - obs.timestamp_history.front()) > velocity_window_) {
            obs.position_history.pop_front();
            obs.timestamp_history.pop_front();
        }

        // 基于保留的历史数据估算速度
        obs.velocity = estimateVelocity(obs);
        // 动态性判断：速度范数大于阈值则视为动态
        obs.is_dynamic = (obs.velocity.norm() > dynamic_speed_threshold_);
    }
}

/**
 * brief 从规划场景监视器更新障碍物（通常为静态障碍物）
 *
 * param scene_objects 场景对象列表，包含 ID 和障碍物属性
 *
 * 直接覆写 tracked_ 中对应 ID 的属性，但不会修改其历史数据。
 */
void ObstacleTracker::updateFromPlanningScene(
    const std::vector<std::pair<std::string, Obstacle>>& scene_objects) {
    for (const auto& [id, obj] : scene_objects) {
        auto& obs = tracked_[id];
        obs.id = id;
        obs.center = obj.center;
        obs.size = obj.size;
        obs.bounds_min = obj.bounds_min;
        obs.bounds_max = obj.bounds_max;
    }
}

/**
 * brief 获取当前所有追踪障碍物的快照
 *
 * return 障碍物列表，每个元素包含位置、尺寸、速度、边界及动态标志
 */
std::vector<Obstacle> ObstacleTracker::getCurrentObstacles() const {
    std::vector<Obstacle> result;
    for (const auto& [id, tracked] : tracked_) {
        Obstacle obs;
        obs.center = tracked.center;
        obs.size = tracked.size;
        obs.velocity = tracked.velocity;
        obs.bounds_min = tracked.bounds_min;
        obs.bounds_max = tracked.bounds_max;
        obs.is_dynamic = tracked.is_dynamic;
        result.push_back(obs);
    }
    return result;
}

/**
 * brief 设置障碍物的尺寸和运动边界
 *
 * param id         障碍物 ID
 * param size       障碍物半尺寸（盒子）
 * param bounds_min 运动范围下界
 * param bounds_max 运动范围上界
 *
 * 通常在配置加载后调用，用于为动态障碍物赋予参数。
 */
void ObstacleTracker::setObstacleInfo(const std::string& id, const Vec3& size,
                                       const Vec3& bounds_min, const Vec3& bounds_max) {
    auto it = tracked_.find(id);
    if (it != tracked_.end()) {
        it->second.size = size;
        it->second.bounds_min = bounds_min;
        it->second.bounds_max = bounds_max;
    }
}

/**
 * brief 生成障碍物未来 N 步预测序列
 *
 * param N               预测步数（不含当前帧，总帧数为 N+1）
 * param dt              单步步长（秒）
 * param vel_expand_gain 速度膨胀增益，用于根据速度扩展尺寸以补偿预测不确定性
 * return 预测序列，索引 0 为当前帧（含膨胀尺寸），1..N 为预测帧
 */
std::vector<std::vector<Obstacle>> ObstacleTracker::predictFuture(
    int N, double dt, double vel_expand_gain) const {
    std::vector<std::vector<Obstacle>> predictions(N + 1);

    // 帧 0：当前障碍物，尺寸根据速度膨胀
    for (const auto& [id, tracked] : tracked_) {
        Obstacle obs;
        obs.center = tracked.center;
        obs.size = tracked.size;
        obs.velocity = tracked.velocity;
        obs.is_dynamic = tracked.is_dynamic;
        obs.bounds_min = tracked.bounds_min;
        obs.bounds_max = tracked.bounds_max;
        // 速度膨胀：尺寸增加 |velocity| * vel_expand_gain
        Eigen::Array3d expansion =
            tracked.velocity.array().abs() * vel_expand_gain;
        obs.size.array() += expansion;
        predictions[0].push_back(obs);
    }

    // 帧 1..N：按恒速模型传播，包括边界反弹
    for (int k = 1; k <= N; ++k) {
        double t_future = k * dt;
        for (const auto& [id, tracked] : tracked_) {
            Obstacle obs;
            obs.size = tracked.size;
            obs.velocity = tracked.velocity;
            obs.is_dynamic = tracked.is_dynamic;
            obs.bounds_min = tracked.bounds_min;
            obs.bounds_max = tracked.bounds_max;
            // 预测位置（包含边界约束）
            obs.center = predictPosition(tracked, t_future);
            predictions[k].push_back(obs);
        }
    }
    return predictions;
}

/**
 * brief 查询指定障碍物的当前速度
 *
 * param id 障碍物 ID
 * return 速度向量（未追踪到则返回零向量）
 */
Vec3 ObstacleTracker::getVelocity(const std::string& id) const {
    auto it = tracked_.find(id);
    if (it != tracked_.end()) {
        return it->second.velocity;
    }
    return Vec3::Zero();
}

/**
 * brief 从历史数据估算障碍物速度
 *
 * param obs 追踪障碍物内部记录（含历史位置与时间戳）
 * return 估算速度（历史窗口首尾差分 / 时间差）
 *
 * 若历史不足两点或时间差过小，返回零向量。
 */
Vec3 ObstacleTracker::estimateVelocity(const TrackedObstacle& obs) const {
    if (obs.position_history.size() < 2) return Vec3::Zero();

    // 首尾位置差
    Vec3 dp = obs.position_history.back() - obs.position_history.front();
    double dt = obs.timestamp_history.back() - obs.timestamp_history.front();

    if (dt < 1e-6) return Vec3::Zero();
    return dp / dt;
}

/**
 * brief 基于恒速模型预测障碍物未来位置
 *
 * param obs 追踪障碍物记录
 * param dt  预测时间跨度（秒）
 * return 预测位置，经过边界反弹处理
 */
Vec3 ObstacleTracker::predictPosition(const TrackedObstacle& obs, double dt) const {
    // 恒速外推
    Vec3 predicted = obs.center + obs.velocity * dt;

    // 若设置了运动边界，则应用反弹模型
    if (obs.bounds_min.norm() > 0 || obs.bounds_max.norm() > 0) {
        predicted = clampToBounds(predicted, obs.bounds_min, obs.bounds_max);
    }

    return predicted;
}

/**
 * brief 将位置钳制到指定边界内，模拟弹性反射
 *
 * param pos  待处理位置
 * param bmin 下界
 * param bmax 上界
 * return 钳制后的位置
 *
 * 算法：在超出边界的坐标上使用“折叠”实现反弹，
 * 类似 MATLAB 中的 propagateDynamicObs_local 行为。
 * 若 bmin >= bmax，则不做处理，直接返回原坐标。
 */
Vec3 ObstacleTracker::clampToBounds(const Vec3& pos, const Vec3& bmin, const Vec3& bmax) const {
    Vec3 result;
    for (int i = 0; i < 3; ++i) {
        if (bmin(i) < bmax(i)) {
            double range = bmax(i) - bmin(i);
            double offset = pos(i) - bmin(i);
            double cycles = std::floor(offset / range);
            double remainder = offset - cycles * range;
            // 偶数周期：正向折叠；奇数周期：反向折叠
            if ((int)cycles % 2 == 0) {
                result(i) = bmin(i) + remainder;
            } else {
                result(i) = bmax(i) - remainder;
            }
        } else {
            result(i) = pos(i);
        }
    }
    return result;
}

}  // namespace fairino_mpc
