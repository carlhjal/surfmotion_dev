cmake_minimum_required(VERSION 3.8)
project(trajectory_seed_generator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_library(seed_generator src/seed_gen.cpp)
add_library(trajectory_seed_generator::seed_generator ALIAS seed_generator)

target_include_directories(seed_generator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(seed_generator PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(seed_generator
  rclcpp
  moveit_core
  moveit_ros_planning_interface
)

ament_export_dependencies(
  rclcpp
  moveit_core
  moveit_ros_planning_interface
)

install(DIRECTORY include/
  DESTINATION include/
)

install(TARGETS seed_generator
  EXPORT export_seed_generator
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(EXPORT export_seed_generator
  NAMESPACE trajectory_seed_generator::
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_export_targets(export_seed_generator)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
