from setuptools import setup

package_name = 'hand_eye_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    py_modules=[
        'scripts.calibration_aruco_publisher',
        'scripts.follow_aruco_marker',
        'scripts.handeye_publisher',
        'scripts.visualize_aruco_marker',  # Ensure your script is added here
    ],
    install_requires=['setuptools', 'cv_bridge', 'opencv-python'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'visualize_aruco_marker = hand_eye_calibration.visualize_aruco_marker:main',  # Make sure this matches your function name
        ],
    },
)

