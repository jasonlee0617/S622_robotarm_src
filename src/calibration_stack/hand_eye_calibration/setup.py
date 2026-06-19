"""Minimal setup.py — real executables come from CMakeLists.txt install(PROGRAMS ...)."""

from setuptools import find_packages, setup

package_name = 'hand_eye_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
)
