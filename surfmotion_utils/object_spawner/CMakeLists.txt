cmake_minimum_required(VERSION 3.8)
project(object_spawner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(moveit_common REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2_eigen REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set(INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)

add_library(collision_object_spawner src/collision_object_spawner.cpp)
target_include_directories(collision_object_spawner PUBLIC ${INCLUDE_DIRECTORIES})
ament_target_dependencies(
  collision_object_spawner
  "moveit_ros_planning_interface"
  "rclcpp"
)

add_executable(cylinder_spawner src/cylinder_spawner.cpp)
target_include_directories(cylinder_spawner PUBLIC ${INCLUDE_DIRECTORIES})
target_compile_features(cylinder_spawner PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(cylinder_spawner collision_object_spawner)
ament_target_dependencies(
  cylinder_spawner
  shape_msgs
  geometric_shapes
  "moveit_ros_planning_interface"
  "rclcpp"
)

add_executable(mesh_spawner src/mesh_spawner.cpp)
target_include_directories(mesh_spawner PUBLIC ${INCLUDE_DIRECTORIES})
target_compile_features(mesh_spawner PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(mesh_spawner collision_object_spawner)
ament_target_dependencies(
  mesh_spawner
  shape_msgs
  geometric_shapes
  "moveit_ros_planning_interface"
  "rclcpp"
)


install(TARGETS cylinder_spawner mesh_spawner 
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
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
