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
            {"beta_x": 0.3},
            {"epsilon_x": 0.1},
            {"k_alpha_x": 0.01},
            {"k_beta_x": 0.9},
            {"tc_x": 400.0},
            {"q_x": 3.0},
            {"p_x": 5.0},
            {"beta_y": 0.3},
            {"epsilon_y": 0.1},
            {"k_alpha_y": 0.01},
            {"k_beta_y": 0.9},
            {"tc_y": 400.0},
            {"q_y": 3.0},
            {"p_y": 5.0},
            {"beta_psi": 0.5},
            {"epsilon_psi": 0.15},
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
