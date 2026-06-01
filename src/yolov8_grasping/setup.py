from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'yolov8_grasping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')), 
        (os.path.join('share', package_name, 'config'), glob('config/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.urdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.xacro')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py'))

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector = yolov8_grasping.yolo_detector_node:main',
            'yolo_detector1 = yolov8_grasping.yolo_detector_node1:main',
            'pen_box_grasping = yolov8_grasping.pen_box_grasping_node:main',
            'pick_drop = yolov8_grasping.pick_drop_node:main',
            'pick_drop_ik = yolov8_grasping.pick_drop_ik_node:main',
            'yolo_detector_obb = yolov8_grasping.yolo_detector_obb_node:main',
            'stopmotion = yolov8_grasping.stopmotion_node:main',
        ],
    },
)
