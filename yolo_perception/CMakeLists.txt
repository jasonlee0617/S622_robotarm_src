cmake_minimum_required(VERSION 3.8)
project(yolo_perception)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/InferenceResult.msg"
    "msg/Yolov8Inference.msg"
    "msg/ObbDebug.msg"
    "msg/TrackDebug.msg"
    "msg/VelocityEval.msg"
    DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Install yolo_perception_utils Python package (obb_geometry, model_utils)
# ── kept in a separate namespace to avoid clashing with the rosidl-generated
#     yolo_perception.msg module.
ament_python_install_package(yolo_perception_utils)

# Install Python executable scripts
install(
  PROGRAMS
    yolo_perception/nodes/yolo_detector.py
    yolo_perception/nodes/yolo_detector_obb.py
    yolo_perception/nodes/yolo_detector_obb_gazebo.py
    yolo_perception/nodes/yolo_kalman_detector_obb.py
    yolo_perception/nodes/vision_velocity_evaluator.py
    yolo_perception/nodes/yolov8_obb_publisher.py
    yolo_perception/nodes/yolov8_obb_subscriber.py
    yolo_perception/export_yolo_obb_trt.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(
  DIRECTORY
    launch
  DESTINATION
    share/${PROJECT_NAME}/
)

# Install model files
install(
  DIRECTORY
    models
  DESTINATION
    share/${PROJECT_NAME}/
  FILES_MATCHING
    PATTERN "*.pt"
    PATTERN "*.onnx"
    PATTERN "*.engine"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
