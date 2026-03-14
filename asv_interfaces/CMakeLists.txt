cmake_minimum_required(VERSION 3.8)
project(asv_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# =================
# || CUSTOM MSGS ||
# =================
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AitsmcDebug.msg"
  "msg/Ref.msg"
  "msg/State.msg"
  "msg/Thrust.msg"
  DEPENDENCIES std_msgs 
)

# ================
# || RVIZ TOOLS ||
# ================

# Find Qt5
find_package(Qt5 REQUIRED COMPONENTS Widgets)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Find RViz dependencies
find_package(rviz_common REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(pluginlib REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Include directories
include_directories(include)

# Set the source files
set(RVIZ_TOOL_SOURCES
  src/rviz_tools/goal_tool.cpp
  src/rviz_tools/goal_from_tool.cpp
  src/rviz_tools/goal_to_tool.cpp
  src/rviz_tools/center_on_robot_tool.cpp
)

# Set the header files (important for MOC)
set(RVIZ_TOOL_HEADERS
  include/asv_interfaces/rviz_tools/goal_tool.hpp
  include/asv_interfaces/rviz_tools/goal_from_tool.hpp
  include/asv_interfaces/rviz_tools/goal_to_tool.hpp
  include/asv_interfaces/rviz_tools/center_on_robot_tool.hpp
)

# Qt5 wrap (this is important!)
qt5_wrap_cpp(RVIZ_TOOL_MOC_FILES ${RVIZ_TOOL_HEADERS})

# Create the RViz tools library
add_library(asv_rviz_tools SHARED
  ${RVIZ_TOOL_SOURCES}
  ${RVIZ_TOOL_MOC_FILES}
)

target_include_directories(asv_rviz_tools PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(asv_rviz_tools
  rviz_common
  rviz_default_plugins
  rviz_rendering
  pluginlib
  Qt5
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Install the library
install(TARGETS asv_rviz_tools
  EXPORT asv_rviz_tools
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

# Install plugin description
install(FILES plugins/plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

# Export plugin
pluginlib_export_plugin_description_file(rviz_common plugins/plugin_description.xml)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # a copyright and license is added to all source files
  
  # comment the line when this package is in a git repo and when
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

