cmake_minimum_required(VERSION 3.8)
project(graspnet_source)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY graspnet_baseline models
  DESTINATION share/${PROJECT_NAME}
  PATTERN ".git" EXCLUDE
  PATTERN "build" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.egg-info" EXCLUDE
  PATTERN "*.so" EXCLUDE
)

install(FILES runtime_requirements.txt THIRD_PARTY_NOTICES.md
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS scripts/setup_graspnet_runtime.sh
  DESTINATION lib/${PROJECT_NAME})

ament_package()
