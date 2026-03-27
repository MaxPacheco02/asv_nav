cmake_minimum_required(VERSION 3.8)
project(asv_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(asv_interfaces REQUIRED)

set(DEPENDENCIES
    ament_cmake
    geometry_msgs
    visualization_msgs
    std_msgs
    rclcpp
    sensor_msgs
    nav_msgs
    tf2_ros
    std_srvs
    Eigen3
    tf2_geometry_msgs
    asv_interfaces)

add_executable(obstacle_publisher src/obstacle_publisher.cpp)
ament_target_dependencies(obstacle_publisher ${DEPENDENCIES})

install(TARGETS obstacle_publisher DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
