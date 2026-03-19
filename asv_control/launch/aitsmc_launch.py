from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    aitsmc_node = Node(
        package="asv_control",
        executable="aitsmc_node",
        # Only for debugging:
        output="screen",
        emulate_tty=True,
        arguments=[("__log_level:=debug")],
        parameters=[
            {"epsilon_u": 0.1},
            {"k_alpha_u": 0.1},
            {"k_beta_u": 0.9},
            {"tc_u": 200.0},
            {"q_u": 3.0},
            {"p_u": 5.0},
            {"epsilon_v": 0.1},
            {"k_alpha_v": 0.5},
            {"k_beta_v": 0.9},
            {"tc_v": 30.0},
            {"q_v": 3.0},
            {"p_v": 5.0},
            {"beta_psi": 1.0},
            {"epsilon_psi": 0.1},
            {"k_alpha_psi": 0.1},
            {"k_beta_psi": 0.9},
            {"tc_psi": 250.0},
            {"q_psi": 3.0},
            {"p_psi": 5.0},
        ],
    )

    return LaunchDescription(
        [
            aitsmc_node,
        ]
    )
