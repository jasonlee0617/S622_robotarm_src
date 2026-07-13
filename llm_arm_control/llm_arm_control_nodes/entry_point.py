#!/usr/bin/env python3
"""Dispatch installed ROS executables to their Python modules."""

import importlib
from pathlib import Path
import sys


_MODULES = {
    "fairino_pose_monitor": "fairino_pose_monitor_node",
    "fairino_pose_control_server": "fairino_pose_control_server",
    "llm_yolo_task_server": "llm_yolo_task_server",
    "llm_yolo_cli": "llm_yolo_cli",
}


def main():
    executable = Path(sys.argv[0]).name
    importlib.import_module(f"llm_arm_control_nodes.{_MODULES[executable]}").main()


if __name__ == "__main__":
    main()
