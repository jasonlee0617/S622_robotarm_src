from glob import glob
import os

from setuptools import find_packages, setup

package_name = "llm_arm_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robot",
    maintainer_email="robot@todo.todo",
    description="Fairino LLM-YOLO task control and reusable pose adapter nodes.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fairino_pose_monitor = llm_arm_control.fairino_pose_monitor_node:main",
            "fairino_pose_control_server = llm_arm_control.fairino_pose_control_server:main",
            "llm_yolo_task_server = llm_arm_control.llm_yolo_task_server:main",
            "llm_yolo_cli = llm_arm_control.llm_yolo_cli:main",
        ],
    },
)
