#!/usr/bin/env bash
# 手动参考采集流程（本脚本只启动图像伺服，保持 auto_start:=true）：
# 1. 先将机械臂末端/相机置于期望的标定码相对距离与姿态，并确认运动区域安全。
# 2. 执行 ./run_visual_image_servo.sh，启动 RealSense、MoveIt Servo、手眼 TF、RViz 与 IBVS 节点。
# 3. 确认 RViz 图像中 aruco_5x5_250_id1 已被绿色边框稳定检测。
# 4. 在另一个已 source /opt/ros/humble/setup.bash 和 install/setup.bash 的终端执行：
#    ros2 service call /visual_image_servo/capture_reference std_srvs/srv/Trigger "{}"
# 5. 确认返回 "reference saved to ...image_servo_aruco_id1.yaml"；移动标定码后，IBVS 自动开始跟踪。
# 参考文件缺失时节点会安全停止；若旧参考已存在，auto_start:=true 可能立即允许伺服，启动前必须确认安全。
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch visual_servo_bringup visual_image_servo.launch.py 
