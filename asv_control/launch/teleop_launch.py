from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    joy_node = Node(
        package="joy",
        executable="joy_node",
    )

    teleop_node = Node(
        package="asv_control",
        executable="teleop_xbox_node.py",
    )

    return LaunchDescription(
        [
            joy_node,
            teleop_node,
        ]
    )
