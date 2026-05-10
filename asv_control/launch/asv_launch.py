import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    spline_publisher_node = Node(
        package="asv_control",
        executable="spline_publisher_node",
        parameters=[
            {"closed": False},
            {
                "waypoints": [
                    -10.0, 0.0,
                    -5.0, 0.0,
                    500.0, 200.0,
                    1300.0, -200.0,
                    1900.0, 200.0,
                    2500.0, -200.0,
                ]
            },
            {"marker_scale": 1.0},
        ],
    )

    obstacle_publisher = Node(
        package="asv_utils",
        executable="obstacle_publisher",
    )

    dynamic_model_node = Node(
        package="asv_control",
        executable="dynamic_model_node",
    )

    state_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("asv_control"),
                        "launch",
                        # 'aitsmc_launch.py'
                        "pid_launch.py",
                    ]
                )
            ]
        ),
    )

    mpc_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [FindPackageShare("asv_control"), "launch", "mpc_launch.py"]
                )
            ]
        ),
    )

    return LaunchDescription(
        [
            spline_publisher_node,
            obstacle_publisher,
            dynamic_model_node,
            state_controller_launch,
            mpc_launch,
        ]
    )
