#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch myrobot_simulation visual_position_servo_sim.launch.py
