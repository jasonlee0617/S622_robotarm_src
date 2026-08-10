/**
 * file smooth_box_distance.cpp
 * brief 点到盒子的光滑符号距离近似计算
 *
 * 本文件实现了点到盒子表面的光滑距离函数，用于：
 * - MPC 代价中的避障项（APF/CBF）
 * - 弹性带(elastic-band)路径平滑与重规划
 * - 任何需要平滑、可微的几何距离的场景
 *
 * 数学原理：
 * 传统的点到盒子的有符号距离是六个半空间距离的最大值：
 *   d_raw = max( dp.x - half.x, -dp.x - half.x,
 *                dp.y - half.y, -dp.y - half.y,
 *                dp.z - half.z, -dp.z - half.z )
 * 其中 dp = point - center, half = box_size / 2。
 * 为了可微性，使用 log-sum-exp (LSE) 函数平滑 max 操作：
 *   d_smooth = (1/k) * log( Σ exp(k * h_i) )
 * 参数 kappa (k) 控制平滑度：k 越大，越接近硬最大值；k 越小，越平滑。
 *
 * 数值稳定性：
 * - kappa 过小会导致 LSE 近似不佳，故设定下限 1e-3。
 * - 为防止指数溢出，在计算 LSE 时先减去最大值 (log-sum-exp 技巧)。
 * - box_size 各分量会被钳制到最小 1e-6，避免零半尺寸导致平面退化。
 *
 * 典型用法：
 *   double dist = SmoothBoxDistance::compute(point, center, size, kappa);
 *   margin = dist - safe_dist;  // 用于 CBF 或 APF 触发
 */

#include "myrobot_mpc_avoidance/smooth_box_distance.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace fairino_mpc {

/**
 * brief 计算点到盒子的光滑有符号距离
 *
 * param point    查询点坐标 (Vec3)
 * param center   盒子中心坐标 (Vec3)
 * param box_size 盒子的全长尺寸 (Vec3)，即 x/y/z 方向的总长度，半尺寸 = box_size / 2
 * param kappa    平滑系数 (大于 0)，越大越接近真实 max 函数，越小越平滑
 * return 光滑距离代理值，正值表示点在盒子外部，负值表示内部
 */
double SmoothBoxDistance::compute(const Vec3& point, const Vec3& center,
                                  const Vec3& box_size, double kappa) {
    // 保障数值路径：
    // - kappa 必须为正数，用于稳定的 log-sum-exp 缩放，设定下限 1e-3
    const double k = (kappa > 1e-9) ? kappa : 1e-3;

    // - 盒子半尺寸必须为正，避免退化平面，设定下限 1e-6
    Vec3 half = box_size.cwiseMax(1e-6) * 0.5;

    // 从查询点指向盒子中心的向量
    Vec3 dp = point - center;

    // 六个半空间距离：分别对应盒子的六个面
    //   h0: x > +half.x   (右侧)
    //   h1: -x > -half.x  → x < -half.x (左侧)
    //   h2: y > +half.y   (前侧)
    //   h3: -y > -half.y  → y < -half.y (后侧)
    //   h4: z > +half.z   (上侧)
    //   h5: -z > -half.z  → z < -half.z (下侧)
    // 当点在盒子外部时，对应的 h_i 为正；点在内部时所有 h_i 均为负。
    double h[6] = {
        dp.x() - half.x(),   // 正 x 面：点必须在 half 之外
       -dp.x() - half.x(),   // 负 x 面
        dp.y() - half.y(),   // 正 y 面
       -dp.y() - half.y(),   // 负 y 面
        dp.z() - half.z(),   // 正 z 面
       -dp.z() - half.z()    // 负 z 面
    };

    // 计算缩放后的值 k * h_i，并找出最大值用于数值稳定
    double hs[6];
    double h_max = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < 6; ++i) {
        hs[i] = k * h[i];
        h_max = std::max(h_max, hs[i]);
    }

    // 计算 Σ exp(k*h_i - h_max)，避免指数上溢
    double sum_exp = 0.0;
    for (int i = 0; i < 6; ++i) {
        sum_exp += std::exp(hs[i] - h_max);
    }

    // 光滑距离 = (1/k) * (h_max + log(sum_exp))
    // 当 k → ∞ 时，该值趋近于 max(h_i)
    return (h_max + std::log(sum_exp)) / k;
}

}  // namespace fairino_mpc