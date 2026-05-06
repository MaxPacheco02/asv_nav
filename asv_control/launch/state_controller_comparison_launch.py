# This launch file only compares the pid vs aitsmc low-level controllers when receiving sinusoidal {surge,sway,yaw} references.

import os
from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    asv1 = Node(
        package="asv_control",
        executable="dynamic_model_node",
        namespace="asv1",
    )

    asv2 = Node(
        package="asv_control",
        executable="dynamic_model_node",
        namespace="asv2",
    )

    pid_launch = GroupAction(
        [
            PushRosNamespace("asv1"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [FindPackageShare("asv_control"), "launch", "pid_launch.py"]
                        )
                    ]
                ),
            ),
        ]
    )

    aitsmc_launch = GroupAction(
        [
            PushRosNamespace("asv2"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [
                                FindPackageShare("asv_control"),
                                "launch",
                                "aitsmc_launch.py",
                            ]
                        )
                    ]
                ),
            ),
        ]
    )

    refs_node = Node(
        package="asv_control",
        executable="sine_refs_node",
        parameters=[
            {
                # "noise_u": 0.015,
                # "noise_v": 0.003,
                # "noise_r": 0.00001,
                "noise_u": 0.0,
                "noise_v": 0.0,
                "noise_r": 0.0,
            },
        ],
    )

    return LaunchDescription(
        [
            asv1,
            asv2,
            pid_launch,
            aitsmc_launch,
            refs_node,
        ]
    )
