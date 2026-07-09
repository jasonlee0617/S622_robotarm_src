from setuptools import find_packages, setup

package_name = "llm_arm_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robot",
    maintainer_email="robot@todo.todo",
    description="Fairino adapter nodes for GraphExecuter and LLM-driven arm control.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fairino_pose_monitor = llm_arm_control.fairino_pose_monitor_node:main",
            "fairino_pose_control_server = llm_arm_control.fairino_pose_control_server:main",
        ],
    },
)
