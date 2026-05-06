from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pid_node = Node(
        package="asv_control",
        executable="pid_node",
        # Only for debugging:
        # output="screen",
        # emulate_tty=True,
        # arguments=[("__log_level:=debug")],
        parameters=[
            # For velocity control:
            {"p_u": 0.5},
            {"i_u": 0.0001},
            {"d_u": 0.2},
            {"i_max_u": 50.0},
            {"d_alpha_u": 0.2},
            {"p_v": 1.5},
            {"i_v": 0.0003},
            {"d_v": 0.1},
            {"i_max_v": 50.0},
            {"d_alpha_v": 0.15},
            {"p_r": 0.5},
            {"i_r": 0.0001},
            {"d_r": 0.2},
            {"i_max_r": 50.0},
            {"d_alpha_r": 0.2},
        ],
    )

    return LaunchDescription(
        [
            pid_node,
        ]
    )
