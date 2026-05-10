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
                    -4.684593200683594,
                    0.6731178760528564,
                    -2.134843349456787,
                    0.8934917449951172,
                    -2.105011463165283,
                    -1.0963650941848755,
                    -3.708611011505127,
                    -0.5749569535255432,
                ]
            },
            {"marker_scale": 0.01},
            {"lookahead": 1.0},
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
            robomaster_description,
            robomaster_rviz,
            spline_publisher_node,
            mpc_launch,
        ]
    )
