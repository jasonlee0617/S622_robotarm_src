#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
ros2 bag record -o ~/bags/LADRC_sample_data   /servo_ladrc_debug_$(date +%Y%m%d_%H%M)   /servo_error_xyz   /servo_cmd_stages   /servo_ff_vel_filt_xyz  /ee_pose_base
