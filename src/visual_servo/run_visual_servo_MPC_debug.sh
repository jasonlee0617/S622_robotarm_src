#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
ros2 bag record -o ~/bags/MPC_sample_data_$(date +%Y%m%d_%H%M) \
  /servo_mpc_debug \
  /servo_error_xyyaw \
  /servo_cmd_stages \
  /servo_ff_vel_filt \
  /ee_pose_base
