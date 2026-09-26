"""
This script is used to launch a gz sim environment for control. This 1/2 file deals with all the gz sim's side and interface to ros2
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from launch.actions import IncludeLaunchDescription, LogInfo, DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.launch_description_sources import AnyLaunchDescriptionSource

from launch_ros.substitutions import FindPackageShare

from launch.conditions import IfCondition, UnlessCondition

from launch.substitutions import FindExecutable
from launch.actions import ExecuteProcess


def include(pack, launch_file):
    return IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(pack), "launch", launch_file])
        )
    )


def generate_launch_description():
    ks = Node(
        package="asv_utils",
        executable="killswitch_node",
        output="screen",
    )

    odom = Node(
        package="asv_utils",
        executable="odom_converter_node",
        output="screen",
        emulate_tty=True,
        arguments=[("__log_level:=debug")],
    )

    return LaunchDescription(
        [
            include("asv_description", "gazebo_launch.py"),
            ks,
            odom,
        ]
    )
