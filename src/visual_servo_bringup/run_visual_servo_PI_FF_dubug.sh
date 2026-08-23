#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
ros2 bag record -o ~/bags/PI_FF_sample_data_$(date +%Y%m%d_%H%M) \
  /servo_pi_ff_debug \
  /servo_error_xyz \
  /servo_cmd_stages \
  /servo_ff_vel_filt_xyz \
  /ee_pose_base
