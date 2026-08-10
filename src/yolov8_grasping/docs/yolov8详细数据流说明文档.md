# yolov8_grasping 重构与数据流说明

## 1. 文档目的

`yolov8_grasping` 当前同时包含 YOLO 感知节点、旧版 elongated_object-box 抓取 demo、旧 Gazebo/MoveIt 启动文件、旧 pick/drop demo 和若干工具模块。随着仿真、MoveIt、场景管理逐步迁移到 `myrobot_simulation`，这个包的职责应该收敛为：

- 提供 YOLOv8 / YOLOv8-OBB 感知节点。
- 保留旧版 `elongated_object_box_grasping` 抓取 demo，便于与新 `visual_servo` 流程对照。
- 保留最小的感知启动 launch。
- 不再维护 Gazebo/MoveIt 场景资产和旧 pick/drop demo。

本文档记录本轮已实施的清理范围、保留入口、两个主要 launch 的数据流，以及后续可维护性重构建议。

## 2. 本轮已实施清理范围

本轮已删除或停止安装的旧内容：

- `yolov8_grasping/config/*`
  - 已删除旧场景 URDF、`movelt_cpp.yaml`、`yolov8_grasping_gazebo.*.xacro` 等资产。
  - Gazebo/MoveIt/场景配置应迁移到 `myrobot_simulation` 和 MoveIt 配置包。
- `yolov8_grasping/launch/gazebo.launch.py`
  - 已删除。该文件包含旧 MoveItConfigsBuilder、旧 xacro、旧 Gazebo 入口，职责与 `myrobot_simulation/launch/gazebo.launch.py` 重叠。
- `yolov8_grasping/launch/pick_block.launch.py`
  - 已删除。它只服务旧 pick/drop demo。
- `yolov8_grasping/yolov8_grasping/pick_drop_node.py`
- `yolov8_grasping/yolov8_grasping/pick_drop_ik_node.py`
  - 已删除。它们是旧版硬编码 pick/drop 流程，已被更完整的任务流程替代。
- `setup.py`
  - 已移除 `pick_drop`、`pick_drop_ik` console scripts。
  - 已移除 config 安装规则，避免安装已经删除的旧配置资产。
  - 已新增 docs 安装规则，`docs/*.md` 会安装到 `share/yolov8_grasping/docs`。

本轮已清理的残留引用：

- `elongated_object_box_system.launch.py`
  - 已删除未加入 `LaunchDescription` 的旧 `gazebo_node` 变量。
  - 不再引用已删除的 `yolov8_grasping/launch/gazebo.launch.py`。
- `perception.demo.launch.py`
  - 已删除未加入 `LaunchDescription` 的旧 `gazebo_node` 变量。
  - 不再引用已删除的 `yolov8_grasping/launch/gazebo.launch.py`。

当前保留的入口：

- `elongated_object_box_grasping = yolov8_grasping.elongated_object_box_grasping_node:main`

当前保留的 launch：

- `launch/elongated_object_box_system.launch.py`

已迁出的入口（yolo_perception / manipulation_common）：

- `yolo_detector`, `yolo_detector_obb` → `yolo_perception`
- `motion_control` → `manipulation_common`
- 共享模块（trajectory_scoring, motion_executor, keepout_manager, detection_cache, target_selector, abort_manager, pose_tools, tf_tools, params, yaml_loader）→ `manipulation_common`

如需一键启动 Gazebo/MoveIt/相机/抓取，应优先在 `myrobot_simulation` 中编排，或者让上层 launch include `myrobot_simulation/launch/gazebo_yolo.launch.py`。不要恢复本包旧 `gazebo.launch.py`。

## 2.1 第二阶段结构化重构

本轮继续参考 `visual_servo` 的结构，对旧版 `elongated_object_box_grasping` 进行了模块化：

- 新增 `config/yolo_visual_grasping.yaml`
  - 合并视觉抓取入口所需的 MoveIt 与任务参数；`visual_grasping_gazebo.launch.py` 和旧 `elongated_object_box_system.launch.py` 均使用该文件作为主配置。
- `elongated_object_box_grasping_node.py`
  - 现在只负责 ROS 节点装配：参数、订阅器、MoveIt2 client、TF、状态机 timer、planner command topic。
- `task/`
  - `task_types.py` 定义 `TargetType` 和 `TaskState`。
  - `grasp_profile.py` 解析 YAML 中的抓取姿态和高度。
  - `elongated_object_box_state_machine.py` 承载原抓取状态机。
- `manipulation_common/perception/`
  - `detection_cache.py` 缓存目标点和 RPY。
  - `target_selector.py` 封装 preferred target、目标优先级和检测超时校验。
- `manipulation_common/planning/`
  - `motion_executor.py` 封装 MoveIt2 规划执行、候选轨迹评分、gripper 控制、IK/planner 切换。

`setup.py` 重新安装 `config/*.yaml`，但不会安装 URDF/SDF/Xacro，避免恢复旧 Gazebo 资产。

## 3. `yolo_perception/launch/yolo_detector.launch.py` 数据流

### 3.1 启动内容

`yolo_perception/launch/yolo_detector.launch.py` 是纯感知验证入口，主要动作如下：

1. 声明 YOLO 参数：
   - `model_path`
   - `device`
   - `conf`
   - `imgsz`
2. 启动 RealSense：
   - `enable_color=true`
   - `enable_depth=true`
   - `align_depth.enable=true`
   - `enable_sync=true`
   - 默认 RGB 和 Depth 均为 `640x480x30`
3. 延迟 3 秒启动 `yolo_detector_obb`。

### 3.2 输入 Topic

`yolo_detector_obb` 默认订阅：

- `/camera/camera/color/image_raw`
  - RGB 图像。
- `/camera/camera/aligned_depth_to_color/image_raw`
  - 已对齐到彩色图的深度图。
- `/camera/camera/aligned_depth_to_color/camera_info`
  - 相机内参。

这些 topic 与 RealSense launch 的命名保持一致。

### 3.3 输出 Topic

`yolo_detector_obb` 默认发布：

- `/camera/detected_image`
  - 带检测框/OBB 可视化的图像。
- `/elongated_object_position_3d`
- `/box_position_3d`
- `/cube_position_3d`
- `/stone_position_3d`
  - 检测目标在 `camera_color_optical_frame` 下的 3D 点。
- `/elongated_object_rpy`
- `/box_rpy`
- `/cube_rpy`
- `/stone_rpy`
  - OBB 推导出的目标姿态角，使用 `Float32MultiArray`。

### 3.4 数据处理链路

```text
RealSense RGB/Depth/CameraInfo
    -> yolo_detector_obb
    -> YOLOv8-OBB 推理
    -> 深度反投影生成 3D 目标点
    -> OBB yaw / RPY 估计
    -> 发布目标 3D 点、RPY 和检测图像
```

仿真检测入口已单独收口为 `ros2 launch yolo_perception yolo_detector_gazebo.launch.py`。

### 3.5 维护建议

- 将 `model_path` 默认值保留在 launch 中，节点内部只保留通用默认值。
- 将 RGB、Depth、CameraInfo topic 全部通过 launch 参数显式传入。
- 把普通 YOLO 和 OBB YOLO 的公共代码抽到 `detector_common.py`。

## 4. `elongated_object_box_system.launch.py` 数据流

### 4.1 启动内容

`elongated_object_box_system.launch.py` 是旧版完整抓取 demo 入口，当前启动链路如下：

1. 启动 RealSense。
2. 启动 `fairino_arm_moveit_config/launch/demo.launch.py`。
3. 延迟 3 秒启动 `yolo_detector_obb`。
4. 启动 `hand_eye_calibration/handeye_publisher.py`。
5. 启动 `trajectory_retime_server/launch/retime_server.launch.py`。
6. 延迟 8 秒启动 `elongated_object_box_grasping`。
   - 加载 `config/yolo_visual_grasping.yaml`。

Gazebo 视觉抓取入口 `myrobot_simulation/launch/visual_grasping_gazebo.launch.py` 同样加载 `config/yolo_visual_grasping.yaml`。

当前文件中还存在 `oak_camera` 变量，但它没有加入最终 `LaunchDescription`，属于可继续清理的历史残留。旧 `gazebo_node` 变量已经删除。

### 4.2 上游输入

RealSense 提供：

- `/camera/camera/color/image_raw`
- `/camera/camera/aligned_depth_to_color/image_raw`
- `/camera/camera/aligned_depth_to_color/camera_info`

手眼标定节点提供：

- `base_link -> camera_color_optical_frame` 相关 TF。

MoveIt demo launch 提供：

- `robot_description`
- `robot_description_semantic`
- `move_group`
- RViz
- controller / planning 相关接口

### 4.3 感知输出

`yolo_detector_obb` 发布：

- `/elongated_object_position_3d`
- `/cube_position_3d`
- `/box_position_3d`
- `/stone_position_3d`
- `/elongated_object_rpy`
- `/cube_rpy`
- `/box_rpy`
- `/stone_rpy`
- `/camera/detected_image`

### 4.4 抓取节点输入

`elongated_object_box_grasping` 订阅：

- `/elongated_object_position_3d`
- `/cube_position_3d`
- `/box_position_3d`
- `/stone_position_3d`
- `/elongated_object_rpy`
- `/cube_rpy`
- `/stone_rpy`
- `/manual_abort`
- `/elongated_object_box_grasping/planner_command`

`elongated_object_box_grasping` 通过 `TfTools` 将相机坐标系下的目标点转换到 `base_link`。

`/elongated_object_box_grasping/planner_command` 使用 `std_msgs/String`，支持：

- `ik fairino`
- `ik kdl`
- `planner fairino birrt*`
- `planner fairino rrt*`
- `planner fairino aapf_birrt*`
- `planner ompl RRTConnect`

Fairino planner 会做别名规范化，例如 `aapf`、`aapf_birrt`、`aapf-birrt` 都会转换为 `aapf_birrt*`，避免命令解析把下划线截断。

### 4.5 抓取节点输出与动作

`elongated_object_box_grasping` 发布：

- `/task_state`
- `/collision_object`
- `/planning_scene`

`elongated_object_box_grasping` 调用：

- MoveIt2 `robot_arm` 规划组执行机械臂运动。
- MoveIt2 `hand` 规划组执行夹爪动作。
- trajectory retime server 对轨迹进行重定时。

### 4.6 总体数据流

```text
RealSense
    -> RGB/Depth/CameraInfo
    -> yolo_detector_obb
    -> /elongated_object_position_3d /cube_position_3d /box_position_3d /stone_position_3d
    -> /elongated_object_rpy /cube_rpy /box_rpy /stone_rpy

handeye_publisher
    -> TF: base_link <-> camera_color_optical_frame

elongated_object_box_grasping
    -> 订阅目标点和姿态
    -> TF 转换到 base_link
    -> 选择目标
    -> 按当前 IK client / planner 选择 MoveIt2 规划和执行
    -> 发布 /task_state、PlanningScene 障碍物
```

### 4.7 当前风险点

- `elongated_object_box_system.launch.py` 仍包含未使用的 `oak_camera` 变量。
- Gazebo/MoveIt 不再由本包维护；旧 `gazebo.launch.py` 已删除。
- 相机 topic、模型路径、手眼标定名、MoveIt 配置包名仍存在硬编码。
- `elongated_object_box_grasping_node.py` 已拆成装配层，但感知节点仍有大量历史代码可继续清理。
- `elongated_object_box_grasping_node.py` 直接访问 pymoveit2 私有接口进行 retime，后续维护风险较高。

## 5. 节点 Topic 合约汇总

| 节点 | 输入 | 输出 |
| --- | --- | --- |
| `yolo_detector` | RGB、Depth、CameraInfo | `/camera/detected_image`、`/yolo_detections`、`/elongated_object_position_3d`、`/box_position_3d` |
| `yolo_detector_obb` | RGB、Depth、CameraInfo | `/camera/detected_image`、目标 3D 点、目标 RPY |
| `elongated_object_box_grasping` | 目标 3D 点、目标 RPY、`/manual_abort` | `/task_state`、`/collision_object`、`/planning_scene` |
| `motion_control` | SPACE/h/r 或 `stop/reset/resume` | `/motion_control/command`、`/manual_abort` |

## 6. 后续重构优化建议

### 6.1 感知节点重构

建议新增：

- `yolov8_grasping/detector_common.py`
  - 模型路径解析。
  - YOLO 模型加载。
  - 相机内参缓存。
  - 深度图反投影。
  - 检测可视化绘制。
- `yolov8_grasping/object_pose_estimator.py`
  - OBB yaw / RPY 估计。
  - 等价角度连续化。
  - 点云采样与鲁棒中心估计。

这样 `yolo_detector_node.py` 和 `yolo_detector_obb_node.py` 只保留 ROS 参数、订阅发布和调度逻辑。

### 6.2 抓取节点重构

已完成第一轮拆分：

- `task/elongated_object_box_state_machine.py`
  - 状态切换和错误恢复。
- `manipulation_common/perception/detection_cache.py`
  - 缓存 elongated_object/cube/box/stone 的位置和 RPY。
- `manipulation_common/perception/target_selector.py`
  - preferred target、目标优先级、检测超时判断。
- `manipulation_common/planning/motion_executor.py`
  - MoveIt2 arm/hand 封装、规划、执行、重定时、planner command。

后续建议继续拆分：

- `scene_manager.py`
  - 如后续需要重新引入桌面/环境碰撞体，应独立管理 collision object 和 planning scene。
- `grasp_policy.py`
  - 目标姿态、放置策略、失败重试策略。
- `manipulation_common/utils/`
  - 共享 `pose_tools.py`、`tf_tools.py`、`trajectory_scoring.py` 等工具，旧 `scripts/` 包装层已移除。

### 6.3 Launch 重构

`elongated_object_box_system.launch.py` 已不再引用本包旧 Gazebo launch。后续建议继续收敛为只编排感知和旧抓取 demo：

- Gazebo/MoveIt 推荐由 `myrobot_simulation` 启动。
- 如果仍需要一键全启动，应在上层 launch 明确 include `myrobot_simulation/launch/gazebo_yolo.launch.py`，不要恢复本包旧 `gazebo.launch.py`。
- 模型路径、相机 topic、handeye calibration name、MoveIt config package 都声明为 launch 参数。

### 6.4 配置治理

Gazebo 视觉抓取入口已合并到 `config/yolo_visual_grasping.yaml` 的抓取参数：

- MoveIt group/link/frame。
- Fairino/KDL move_group namespace。
- `ik_plugin`、`planning_pipeline_id`、`planner_id`。
- pre-grasp pose、抓取高度、放置高度。
- 目标优先级和 grasp profile。
- 候选轨迹评分参数。
- elongated_object/cube/stone grasp profile。

旧版 z 方向禁入盒及相关参数已移除；elongated_object-box demo 不再维护该禁入盒设计。

仍建议继续迁移到 YAML 或 launch 参数：

- YOLO 参数：
  - `model_path`
  - `device`
  - `conf`
  - `imgsz`
  - `sync_slop`
  - `inference_period`
  - `pose_publish_rate`
- 相机参数：
  - RGB topic
  - Depth topic
  - CameraInfo topic
  - camera frame
- MoveIt 配置包名和 RViz 配置路径。
- handeye calibration name。
- 相机选择 RealSense/OAK 的开关。

### 6.5 删除旧代码块

以下文件存在大量注释旧版本，建议在确认行为稳定后清理：

- `elongated_object_box_grasping_node.py`
- `yolo_detector_obb_node.py`

清理原则：

- 删除整段注释掉的历史版本。
- 保留必要的算法说明，但不要保留已废弃代码。
- 复杂逻辑使用短注释解释原因，不逐行解释实现。

## 7. 验证命令

文档存在性：

```bash
test -f yolov8_grasping/docs/refactor_and_dataflow_guide.md
```

删除方案实施后的引用检查：

```bash
rg "pick_drop|pick_block|yolov8_grasping_gazebo|movelt_cpp" yolov8_grasping
```

预期结果：只允许文档中出现这些关键字；不应再有 Python 代码、launch 或 `setup.py` 引用。

Python 静态检查：

```bash
python3 -m py_compile yolov8_grasping/yolov8_grasping/*.py yolov8_grasping/yolov8_grasping/*/*.py yolov8_grasping/launch/*.launch.py
```

YAML 静态检查：

```bash
python3 -c "import yaml; yaml.safe_load(open('yolov8_grasping/config/yolo_visual_grasping.yaml'))"
```

构建：

```bash
colcon build --packages-select yolo_perception yolov8_grasping --symlink-install
```

运行：

```bash
ros2 launch yolo_perception yolo_detector.launch.py
ros2 launch yolov8_grasping elongated_object_box_system.launch.py
ros2 run yolo_perception yolo_detector_obb
ros2 run yolov8_grasping elongated_object_box_grasping
```

运行时 IK/planner 切换：

```bash
ros2 topic pub --once /elongated_object_box_grasping/planner_command std_msgs/msg/String "{data: 'ik fairino'}"
ros2 topic pub --once /elongated_object_box_grasping/planner_command std_msgs/msg/String "{data: 'ik kdl'}"
ros2 topic pub --once /elongated_object_box_grasping/planner_command std_msgs/msg/String "{data: 'planner fairino birrt*'}"
ros2 topic pub --once /elongated_object_box_grasping/planner_command std_msgs/msg/String "{data: 'planner fairino aapf_birrt*'}"
ros2 topic pub --once /elongated_object_box_grasping/planner_command std_msgs/msg/String "{data: 'planner ompl RRTConnect'}"
```

## 8. 维护边界

- `yolov8_grasping`：感知节点、旧版 elongated_object-box 抓取 demo、手动 abort 工具。
- `myrobot_simulation`：Gazebo、MoveIt、机器人 profile、路径规划 demo、论文场景。
- `visual_servo`：当前主线视觉伺服抓取状态机。
- `myrobot_planning_core` / `myrobot_planning_ros`：IK、全局规划、Cartesian path planner。

后续开发应尽量保持这个边界，避免 `yolov8_grasping` 再次承担仿真、规划器、场景资产等跨包职责。
