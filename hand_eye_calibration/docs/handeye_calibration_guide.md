# S622 机械臂手眼标定指南

## ⚠️ 重要概念

| Launch 文件 | 启动的 easy_handeye2 组件 | 功能 | 有 Compute/Save？ |
|---|---|---|---|
| `calibrate.launch.py` | `handeye_server` + **`rqt_calibrator.py`** | 采集样本、求解变换、保存结果 | ✅ **是，在这里** |
| `evaluate.launch.py` | **`rqt_evaluator.py`** | 加载已保存的结果，验证精度 | ❌ 否，只能评估 |
| `validate.launch.py` | 无 easy_handeye2 GUI | 发布标定 TF 到 RViz，目视检查 | ❌ 否 |

**evaluate.launch.py 必须在标定完成（calibrate.launch.py 中已 Save 并关闭）之后单独启动。**

## 完整工作流程

```
Step 1: calibrate.launch.py  (标定数据采集 + 求解 + 保存)
  │
  │  easy_handeye2 rqt_calibrator GUI:
  │    · Take Sample → 重复 20-25 次（每次移动到新姿势）
  │    · Compute → 求解手眼变换
  │    · Save → 保存到 ~/.ros2/easy_handeye2/
  │
  │  关闭 calibrate.launch.py
  ▼
Step 2: evaluate.launch.py  (精度评估 — 标定完成后启动)
  │
  │  easy_handeye2 rqt_evaluator GUI:
  │    · 加载 Step 1 保存的标定结果
  │    · 移动到新的验证姿势
  │    · 查看 RMSE 等误差指标
  │
  │  (可选)
  ▼
Step 3: validate.launch.py  (目视验证)
    在 RViz 中检查坐标系一致性
```

## 目录
- [完整工作流程](#完整工作流程)
- [硬件准备](#硬件准备)
- [ArUco Marker 制作](#aruco-marker-制作)
- [一、眼在手外 (eye_on_base)](#一眼在手外-eye_on_base)
- [二、眼在手内 (eye_in_hand)](#二眼在手内-eye_in_hand)
- [自动采样](#自动采样)
- [标定精度评估](#标定精度评估)
- [常见问题排查](#常见问题排查)

---

## 硬件准备

| 项目 | 要求 |
|---|---|
| 机械臂 | Fairino S622，已标定运动学 |
| 相机 | Intel RealSense D435/D415 或 OAK-D |
| ArUco Marker | 推荐 5×5_250 字典，边长 70mm (可缩放) |
| 安装底座 | 相机固定在稳定支架(眼在手外) 或 刚性安装在末端(眼在手内) |
| 标定板固定件 | 眼在手外：marker 固定在末端法兰；眼在手内：marker 固定在工作台 |

## ArUco Marker 制作

1. 在线生成 ArUco marker：https://chev.me/arucogen/
   - 字典：`5x5_250`
   - ID：`1` (默认，可自定义)
   - 尺寸：70mm
2. 打印后贴在平整硬板上（避免弯曲变形）
3. 测量实际打印尺寸，修改 `config/aruco_parameters.yaml` 中的 `marker_size`

### 注意事项
- marker 必须**完全平整**，边角不能翘起
- 避免反光材料（覆哑光膜）
- marker 安装位置在标定过程中**不可移动**
- 标定前用 `/aruco_image` topic 检查 marker 检测是否稳定

---

## 一、眼在手外 (eye_on_base)

### 1.1 原理

```
相机固定在工作环境中 → 观测安装在末端的 ArUco marker
easy_handeye2 求解: base_link → camera_link (即相机在机器人基座坐标系中的位姿)
```

### 1.2 硬件安装

```
  [相机固定支架]
       |
       v
  ┌──────────┐
  │ 相机     │  ← 不随机械臂运动
  │ (RealSense│
  │  OAK-D)  │
  └──────────┘
       |
       | 观测方向指向机器人工作空间
       v
  ┌──────────────┐
  │  机械臂末端   │ ← 贴有 ArUco marker
  │  [marker]    │    随末端运动
  └──────────────┘
```

### 1.3 启动标定

```bash
ros2 launch hand_eye_calibration calibrate.launch.py \
    calibration_type:=eye_on_base \
    camera_type:=realsense
```

### 1.4 标定步骤

1. **启动后检查**：
   - RViz 中能看到机器人模型和相机 TF
   - `/aruco_image` topic 中 marker 被正确检测（绿色边框+坐标轴）
   - `ros2 topic echo /aruco_markers` 确认 marker ID 为 1

2. **数据采集**（easy_handeye2 GUI 操作）：
   - 通过 easy_handeye2 的 rqt 插件点击 "Take Sample"
   - **采样要求**：
     - ≥ 15 个采样点（建议 20-25 个）
     - 覆盖机器人的工作空间：前后、左右、上下都要有
     - **避免**纯平移运动（至少包含旋转变化）
     - **避免**所有采样点在同一平面内（容易导致退化）
     - Marker 必须始终在相机视野内且被正确检测
   - 每次移动后等待机械臂**完全静止**再采样
   - 采样时确保 marker 检测稳定（不闪烁、无跳变）

3. **求解**：
   - 点击 "Compute" 计算标定结果
   - 检查残差：平移 <5mm，旋转 <1° 为合格
   - 点击 "Save" 保存

4. **采样姿势建议**：
   ```
   位姿类型        | 描述
   ───────────────┼─────────────────────
   正前方正视      | 末端正对相机
   左偏 30°       | 末端向左转，marker 仍可见
   右偏 30°       | 末端向右转
   上仰 20°       | 末端上仰
   下俯 20°       | 末端下俯
   近距 (0.3m)    | 靠近相机
   远距 (0.5m)    | 远离相机
   左上/右下角     | 覆盖视野边缘
   ```

### 1.5 标定后的 TF 树

```
base_link
  └── camera_link          ← 标定发布 (handeye_publisher)
        └── camera_color_optical_frame
              └── calibration_aruco  ← ArUco marker 位姿
```

### 1.6 关闭并进入评估

标定完成后，**先关闭 calibrate.launch.py (Ctrl+C)**，然后：
```bash
ros2 launch hand_eye_calibration evaluate.launch.py \
    calibration_type:=eye_on_base \
    camera_type:=realsense
```
在 easy_handeye2 rqt_evaluator GUI 中查看精度。详见 [标定精度评估](#标定精度评估)。

---

## 二、眼在手内 (eye_in_hand)

### 2.1 原理

```
相机固定在机械臂末端 → 观测固定在工作台上的 ArUco marker
easy_handeye2 求解: grasp_frame → camera_link (即相机在末端坐标系中的位姿)
```

### 2.2 硬件安装

```
  ┌──────────────────┐
  │  机械臂末端       │
  │  ┌────────────┐  │
  │  │ 相机       │  │ ← 随末端运动
  │  │ (固定在    │  │
  │  │  法兰上)   │  │
  │  └────────────┘  │
  └──────────────────┘
           |
           | 相机随末端移动，从不同角度观测
           v
      ┌─────────┐
      │ ArUco   │ ← 固定在工作台/基座附近
      │ marker  │    **不可移动**
      └─────────┘
```

### 2.3 启动标定

```bash
ros2 launch hand_eye_calibration calibrate.launch.py \
    calibration_type:=eye_in_hand \
    camera_type:=realsense
```

### 2.4 标定步骤

1. **启动前确认**：
   - Marker 固定在工作台上，**标定过程中不可移动**
   - 相机刚性安装在末端（推荐用打印件或铝型材固定）
   - 相机线缆留有足够余量，避免运动时拉扯

2. **数据采集**（同眼在手外，通过 easy_handeye2 GUI）：
   - ≥ 15 个采样点（建议 20-25 个）
   - **关键区别**：相机随末端运动，每次改变机械臂姿态从不同视角观察 marker
   - 确保每个位姿下 marker 在相机视野中且被正确检测
   - 采样姿势要求：
     - 从不同角度观察 marker（正面、侧面、斜上方等）
     - 覆盖至少 3 个不同距离
     - 包含明显的旋转变化

3. **眼在手内采样特别注意**：
   - 相机光轴与 marker 平面夹角不要太小（避免 >70° 的极端倾斜）
   - 避免相机距离 marker 太远（>1m）导致检测不稳定
   - 每次采样时等待相机图像稳定（避免运动模糊）

### 2.5 标定后的 TF 树

```
base_link
  └── grasp_frame           ← 机器人 FK
        └── camera_link     ← 标定发布 (handeye_publisher)
              └── camera_color_optical_frame
```

验证 TF：
```bash
# 相机 frame 应随末端运动，但 grasp_frame -> camera_link 应保持静态
ros2 run tf2_ros tf2_echo grasp_frame camera_link
```

### 2.6 关闭并进入评估

标定完成后，**先关闭 calibrate.launch.py (Ctrl+C)**，然后启动 evaluate.launch.py 验证精度（详见 [标定精度评估](#标定精度评估)）。

---

## 自动采样

自动采样由 `auto_calibration_collector.py` 完成，适合 Gazebo 或固定工装下的重复标定流程。详细设计、参数和 1cm 精度验收标准见：

```text
hand_eye_calibration/docs/auto_calibration_collector_design.md
```

Gazebo 自动采样启动：

```bash
ros2 launch gazebo_launch calibration_gazebo.launch.py auto_collect:=true
```

注意：

- RQT 不会自动刷新外部节点采集的 sample list，collector 日志中的 `samples=N` 才是 easy_handeye2 服务端样本数。
- collector 以图像角点质量作为采样硬门槛，marker 贴边、过小、抖动或丢失时不会采样。
- 如果第一个零偏移候选失败，应优先检查相机 optical frame、CameraInfo topic、marker 尺寸和 Gazebo 相机模型，而不是放宽采样阈值。

---

## 标定精度评估

**evaluate.launch.py 在标定完成后单独启动。** 它会加载 calibrate.launch.py 中 Save 的结果，发布标定 TF，然后通过 easy_handeye2 的 rqt_evaluator GUI 验证精度。

### 启动评估

```bash
# 先关闭 calibrate.launch.py (Ctrl+C)，然后：
ros2 launch hand_eye_calibration evaluate.launch.py \
    calibration_type:=eye_on_base \
    camera_type:=realsense
```

### 评估步骤

1. **启动后检查**：
   - `handeye_publisher` 节点已加载标定结果并发布 TF
   - easy_handeye2 的 **rqt_evaluator** 面板出现（不是 rqt_calibrator）

2. **精度验证**（在 rqt_evaluator GUI 中操作）：
   - 移动机械臂到 N 个**新**验证位姿（与标定位姿不同）
   - 等待机械臂完全静止
   - 在 evaluator 中采集验证样本
   - GUI 自动显示每个样本的误差和总体 RMSE
   精度判断标准

   指标	优秀	合格	需重新标定
   平移 RMSE	< 3mm	< 8mm	> 8mm
   旋转 RMSE	< 0.5°	< 1.5°	> 1.5°
### 评估的 TF 树

```
眼在手外:
  base_link
    └── camera_link          ← handeye_publisher 发布（标定结果）
          └── camera_color_optical_frame
                └── calibration_aruco

眼在手内:
  base_link
    └── grasp_frame
          └── camera_link    ← handeye_publisher 发布（标定结果）
                └── camera_color_optical_frame
                      └── calibration_aruco
```

### 手动 TF 验证

```bash
# 眼在手外: camera_link 应在 base_link 中是静态的
ros2 run tf2_ros tf2_echo base_link camera_link

# 眼在手内: camera_link 在 grasp_frame 中应是静态的
ros2 run tf2_ros tf2_echo grasp_frame camera_link
```

### 目视验证 (RViz) — validate.launch.py

```bash
ros2 launch hand_eye_calibration validate.launch.py \
    calibration_type:=eye_on_base \
    camera_type:=realsense
```
- RViz 中 ArUco 坐标系、相机坐标系和机器人模型应几何一致
- eye_on_base 下 camera 固定不动；eye_in_hand 下 camera 跟随末端运动

### 精度标准

| 指标 | 优秀 | 合格 | 不合格 |
|---|---|---|---|
| 平移 RMSE | < 3mm | < 8mm | > 8mm |
| 旋转 RMSE | < 0.5° | < 1.5° | > 1.5° |

---

## 常见问题排查

### Marker 检测不稳定

| 症状 | 原因 | 解决 |
|---|---|---|
| marker 闪烁/丢失 | 光照不足或反光 | 调整光源，避免直射光 |
| | marker 尺寸过小 | 增大 marker 或缩短距离 |
| | 运动模糊 | 采样前等待机械臂完全静止 |
| marker ID 错误 | 字典不匹配 | 确认 `aruco_dictionary_id: DICT_5X5_250` |

### 标定结果精度差

| 症状 | 原因 | 解决 |
|---|---|---|
| 平移误差 >10mm | 采样点姿态不够多样 | 增加旋转变化，覆盖更大工作空间 |
| | 采样点共面 | 确保采样点不在同一平面内 |
| | marker 实际尺寸与参数不符 | 精确测量 marker 尺寸并修改 YAML |
| 旋转误差 >2° | 相机内参不准确 | 先做相机内参标定 (Kalibr/MATLAB) |
| | 末端/marker 安装松动 | 检查机械固定，重新紧固 |

### easy_handeye2 不启动

```bash
# 确认 easy_handeye2 已安装
ros2 pkg list | grep easy_handeye2

# 手动启动 easy_handeye2 调试
ros2 launch easy_handeye2 calibrate.launch.py name:=robot_calibration calibration_type:=eye_on_base
```

### TF 树断裂

```bash
# 查看完整 TF 树
ros2 run tf2_tools view_frames

# 检查具体 frame
ros2 run tf2_ros tf2_echo <parent> <child>
```

### 标定保存失败

ROS 2 easy_handeye2 默认将结果保存到 `~/.ros2/easy_handeye2/` 目录。
确认目录存在且有写入权限：
```bash
ls -la ~/.ros2/easy_handeye2/
```
