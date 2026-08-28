#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
ros2 bag record -o ~/bags/NLADRC_sample_data /servo_nladrc_debug /servo_error_xyz /servo_cmd_stages /servo_ff_vel_filt_xyz /ee_pose_base
