from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    aitsmc_node = Node(
        package="usv_nav",
        executable="aitsmc_node",
        # Only for debugging:
        # output="screen",
        # emulate_tty=True,
        # arguments=[("__log_level:=debug")],
        # First-pass gains: u and r settle in ~3-6 s in sim. Tune on the boat.
        parameters=[
            {"epsilon_u": 0.1},
            {"k_alpha_u": 0.5},
            {"k_beta_u": 0.9},
            {"tc_u": 1.0},
            {"q_u": 3.0},
            {"p_u": 5.0},
            # Sway is unactuated (not tracked), but q/p must be valid
            {"epsilon_v": 0.1},
            {"k_alpha_v": 0.005},
            {"k_beta_v": 0.9},
            {"tc_v": 3.0},
            {"q_v": 3.0},
            {"p_v": 5.0},
            {"epsilon_r": 0.1},
            {"k_alpha_r": 0.5},
            {"k_beta_r": 0.9},
            {"tc_r": 0.50},
            {"q_r": 3.0},
            {"p_r": 5.0},
        ],
    )

    return LaunchDescription(
        [
            aitsmc_node,
        ]
    )
