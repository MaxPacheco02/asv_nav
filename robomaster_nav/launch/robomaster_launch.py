from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


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
            {
                "waypoints": [
                    -3.708519220352173,
                    -0.8183400630950928,
                    -4.5222063064575195,
                    0.127112478017807,
                    -3.8781960010528564,
                    1.213521957397461,
                    -0.6969701647758484,
                    1.3372472524642944,
                    2.855658769607544,
                    1.0855484008789062,
                    3.3021929264068604,
                    0.09046795964241028,
                    2.014171838760376,
                    -0.9334477186203003,
                    -0.4770919680595398,
                    -1.1192071437835693,
                ]
            },
            {"marker_scale": 0.02},
            {"lookahead": 1.5},
        ],
    )

    obstacle_publisher = Node(
        package="asv_utils",
        executable="obstacle_publisher",
        parameters=[
            {
                # x min, x max, y min, y max
                "bouncing_area": [-5.0, 5.0, -5.0, 5.0]
            },
            {"marker_scale": 0.01},
            {"max_vel": 1.0},
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
