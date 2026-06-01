#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                "gnome-terminal", "--",
                "bash", "-lc",
                "ros2 run yolov8_grasping stopmotion"
            ],
            output="screen",
        ),
    ])
