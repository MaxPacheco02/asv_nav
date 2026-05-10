cmake_minimum_required(VERSION 3.8)
project(robomaster_nav)

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
find_package(asv_control REQUIRED)

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
    asv_control
    asv_interfaces)

add_executable(robomaster_vicon_handler src/robomaster_vicon_handler.cpp)
ament_target_dependencies(robomaster_vicon_handler ${DEPENDENCIES})

# ================
# ||| MPC Node |||
# ================

# Find ACADOS
set(ACADOS_INSTALL_DIR "$ENV{HOME}/acados")
include_directories(
  ${ACADOS_INSTALL_DIR}/include ${ACADOS_INSTALL_DIR}/include/acados
  ${ACADOS_INSTALL_DIR}/include/blasfeo/include
  ${ACADOS_INSTALL_DIR}/include/hpipm/include)
link_directories(${ACADOS_INSTALL_DIR}/lib)

# Include generated code directory
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mpc/c_generated_code_robomaster_ocp)
# Add all generated source files
file(
  GLOB ACADOS_GENERATED_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mpc/c_generated_code_robomaster_ocp/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mpc/c_generated_code_robomaster_ocp/*/*.c)
list(FILTER ACADOS_GENERATED_SOURCES EXCLUDE REGEX "main_.*\\.c$")

add_executable(mpc_node src/mpc_node.cpp ${ACADOS_GENERATED_SOURCES})
ament_target_dependencies(mpc_node ${DEPENDENCIES})
target_link_libraries(mpc_node asv_control::catmul_rom acados hpipm blasfeo m)

install(TARGETS robomaster_vicon_handler mpc_node
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch rviz scripts DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
