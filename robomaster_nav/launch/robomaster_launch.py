from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


# fmt: off
# Lemniscate of Bernoulli.
# python3 gen_lemniscate.py 3.0 3.0 -1.0 0.5 16
_WAYPOINTS = [
    +2.000, +0.500,
    +1.418, +1.425,
    +0.414, +1.500,
    -0.381, +1.072,
    -1.000, +0.500,
    -1.619, -0.072,
    -2.414, -0.500,
    -3.418, -0.425,
    -4.000, +0.500,
    -3.418, +1.425,
    -2.414, +1.500,
    -1.619, +1.072,
    -1.000, +0.500,
    -0.381, -0.072,
    +0.414, -0.500,
    +1.418, -0.425,
]
# fmt: on


def generate_launch_description():
    kinematics_node = Node(
        package="asv_control",
        executable="kinematics_node",
    )

    robomaster_vicon_handler = Node(
        package="robomaster_nav",
        executable="robomaster_vicon_handler",
    )

    robomaster_main_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("robomaster_ros"),
                    "launch",
                    "main.launch",
                ]
            )
        ),
        launch_arguments={
            "model": "s1",
            "with_model_description": "false",
            "camera": "false",
            "led": "true",
            "speaker": "false",
            "battery": "true",
            "chassis_rate": "50",
            "chassis_status_rate": "0",
            "chassis_force_level": "true",
            "armor": "false",
            "tof_0": "False",
            "tof_1": "False",
            "tof_2": "False",
            "tof_3": "False",
            "vision_targets": "[]",
            "sensor_adapter": "false",
        }.items(),
    )

    robomaster_description = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("robomaster_description"),
                    "launch",
                    "main.launch",
                ]
            )
        ),
        launch_arguments={
            "model": "s1",
            "tof_0": "False",
            "tof_1": "False",
            "tof_2": "False",
            "tof_3": "False",
        }.items(),
    )

    robomaster_rviz = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("robomaster_nav"),
                    "launch",
                    "robomaster_rviz_launch.py",
                ]
            )
        )
    )

    mpc_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("robomaster_nav"),
                    "launch",
                    "mpc_launch.py",
                ]
            )
        )
    )

    spline_publisher_node = Node(
        package="asv_control",
        executable="spline_publisher_node",
        parameters=[
            {"closed": True},
            {"waypoints": _WAYPOINTS},
            {"marker_scale": 0.02},
            {"lookahead": 1.0},
        ],
    )

    obstacle_publisher = Node(
        package="asv_utils",
        executable="obstacle_publisher",
        parameters=[
            {
                # x min, x max, y min, y max
                "bouncing_area": [-10.0, 10.0, -4.0, 4.0]
                # "bouncing_area": [-20.0, 20.0, -20.0, 20.0]
            },
            {"marker_scale": 0.01},
            {"max_vel": 2.0},
        ],
    )

    return LaunchDescription(
        [
            # If simulating:
            kinematics_node,
            #
            # If real life:
            # robomaster_vicon_handler,
            # robomaster_main_launch,
            #
            robomaster_description,
            robomaster_rviz,
            spline_publisher_node,
            obstacle_publisher,
            #
            # It's better to launch mpc separately because when it's not enabled, it fights with teleop twist node for control.
            # mpc_launch,
        ]
    )
