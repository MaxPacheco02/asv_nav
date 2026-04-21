from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    csv_name = LaunchConfiguration("csv")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "csv",
                default_value="turning20_initHeading0_NoWind.csv",
                description="CSV filename inside asv_utils/config/",
            ),
            Node(
                package="asv_utils",
                executable="csv_path_publisher_node",
                name="csv_path_publisher",
                output="screen",
                parameters=[
                    {
                        "csv_file": PathJoinSubstitution(
                            [
                                FindPackageShare("asv_utils"),
                                "config",
                                csv_name,
                            ]
                        ),
                        "frame_id": "world",
                        "publish_hz": 1.0,
                    }
                ],
            ),
        ]
    )
