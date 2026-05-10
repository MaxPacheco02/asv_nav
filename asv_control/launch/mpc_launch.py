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
        parameters=[
            {"w_along":     0.01},
            {"w_cross":     10.0},
            {"w_heading":   100.0},
            {"w_input":     0.01},
            {"w_surge":     0.1},
            {"w_sway":      100.0},
            {"w_yaw":       0.001},
            {"terminal_w":  10.0},
            {"avoidance_w": 75000.0},
            {"mpc_tf_init": 100.0},
        ],
    )

    return LaunchDescription(
        [
            mpc_node,
            mpc_gui,
        ]
    )
