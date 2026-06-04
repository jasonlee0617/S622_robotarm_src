cmake_minimum_required(VERSION 3.8)
project(yolo_model)

find_package(ament_cmake REQUIRED)

file(GLOB YOLO_MODEL_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.pt"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.onnx"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.engine")

install(FILES ${YOLO_MODEL_FILES}
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS export_yolo_obb_trt.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
