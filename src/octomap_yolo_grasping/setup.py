from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'octomap_yolo_grasping'

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
            'octomap_yolo_grasping = octomap_yolo_grasping.octomap_yolo_grasping_node:main',
            'yolo_detector_obb = octomap_yolo_grasping.yolo_detector_obb_node:main',
            'stopmotion = octomap_yolo_grasping.stopmotion_node:main',
            'semantic_octomap_cloud_filter = octomap_yolo_grasping.semantic_octomap_cloud_filter_node:main',
            'dynamic_collision_objects = octomap_yolo_grasping.dynamic_collision_objects_node:main',
        ],
    },
)
