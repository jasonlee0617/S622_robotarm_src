from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'graspnet_bringup'

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
        (os.path.join('share', package_name, 'config'), glob('config/*.xacro'))

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='ROS 2 bringup and execution nodes for GraspNet grasping.',
    license='LicenseRef-GraspNet-NonCommercial',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graspnet_inference = graspnet_bringup.graspnet_inference_node:main',
            'graspnet_visual_grasping = graspnet_bringup.graspnetl_grasping_node:main',
        ],
    },
)
