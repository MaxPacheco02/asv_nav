from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mpc_node = Node(
        package="asv_control",
        executable="mpc_node",
        ros_arguments=["--log-level", "WARN"],
    )

    mpc_gui = Node(
        package="asv_control",
        executable="mpc_gui.py",
    )

    return LaunchDescription(
        [
            mpc_node,
            mpc_gui,
        ]
    )
