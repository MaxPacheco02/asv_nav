from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    aitsmc_node = Node(
        package="asv_control",
        executable="aitsmc_node",
        # Only for debugging:
        output='screen',
        emulate_tty=True,
        arguments=[('__log_level:=debug')],

        parameters=[
            {"epsilon_u": 1.0},
            {"k_alpha_u": 0.1},
            {"k_beta_u": 2.0},
            {"epsilon_psi": 0.5},
            {"k_alpha_psi": 1.},
            {"k_beta_psi": 0.75},
            {"tc_u": 1.0},
            {"tc_psi": 2.0},
            {"q_u": 3.0},
            {"q_psi": 3.0},
            {"p_u": 5.0},
            {"p_psi": 5.0},
        ],
    )
     
    return LaunchDescription([
        aitsmc_node,
    ])
