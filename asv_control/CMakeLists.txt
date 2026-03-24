cmake_minimum_required(VERSION 3.8)
project(asv_control)

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
find_package(usv_interfaces REQUIRED)
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
  usv_interfaces
  asv_interfaces
)

add_executable(aitsmc_node 
  src/aitsmc_node.cpp
  src/dynamic_model.cpp
  src/AITSMC.cpp
  src/AITSMC_XYH.cpp
  src/AITSMC_SSY.cpp
)
target_include_directories(aitsmc_node PRIVATE include)
ament_target_dependencies(aitsmc_node ${DEPENDENCIES})

add_executable(dynamic_model_node 
  src/dynamic_model_node.cpp
  src/dynamic_model.cpp
)
target_include_directories(dynamic_model_node PRIVATE include)
ament_target_dependencies(dynamic_model_node ${DEPENDENCIES})

add_executable(sine_refs_node src/debug/aitsmc/sine_refs_node.cpp)
ament_target_dependencies(sine_refs_node ${DEPENDENCIES})

add_executable(rviz_refs_node src/debug/aitsmc/rviz_refs_node.cpp)
ament_target_dependencies(rviz_refs_node ${DEPENDENCIES})

add_executable(spline_publisher_node src/spline_publisher_node.cpp)
ament_target_dependencies(spline_publisher_node ${DEPENDENCIES})

# ================
# ||| MPC Node |||
# ================

# Find ACADOS
set(ACADOS_INSTALL_DIR "$ENV{HOME}/acados")
include_directories(
  ${ACADOS_INSTALL_DIR}/include
  ${ACADOS_INSTALL_DIR}/include/acados
  ${ACADOS_INSTALL_DIR}/include/blasfeo/include
  ${ACADOS_INSTALL_DIR}/include/hpipm/include
)
link_directories(${ACADOS_INSTALL_DIR}/lib)

# Include generated code directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/scripts/mpc/c_generated_code_asv_ocp)
# Add all generated source files
file(GLOB ACADOS_GENERATED_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mpc/c_generated_code_asv_ocp/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mpc/c_generated_code_asv_ocp/*/*.c
)
list(FILTER ACADOS_GENERATED_SOURCES EXCLUDE REGEX "main_.*\\.c$")

add_executable(mpc_node src/mpc_node.cpp ${ACADOS_GENERATED_SOURCES})
ament_target_dependencies(mpc_node ${DEPENDENCIES})
target_link_libraries(mpc_node acados hpipm blasfeo m)


install(TARGETS
  aitsmc_node
  dynamic_model_node
  sine_refs_node 
  rviz_refs_node 
  spline_publisher_node
  mpc_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  scripts
  DESTINATION share/${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/teleop_xbox_node.py
  scripts/mpc/mpc_gui.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
