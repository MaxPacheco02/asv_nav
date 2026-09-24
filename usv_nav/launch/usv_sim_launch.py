from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# asv_control / asv_utils helpers still use the ASV topic names
USV_REMAPS = [
    ("/asv/state/odom", "/usv/state/odom"),
    ("/asv/state/pose", "/usv/state/pose"),
    ("/asv/path_ref", "/usv/path_ref"),
]


def include(pack, launch_file):
    return IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(pack), "launch", launch_file])
        )
    )


def generate_launch_description():
    dynamic_model_node = Node(
        package="usv_nav",
        executable="dynamic_model_node",
    )

    spline_publisher_node = Node(
        package="asv_control",
        executable="spline_publisher_node",
        remappings=USV_REMAPS,
        parameters=[
            # Same control points as the usv_mpc.py scenario
            {"waypoints": [-10.0, 0.0, -5.0, 0.0, 10.0, 20.0, 10.0, 30.0]},
            {"marker_scale": 0.05},
            {"lookahead": 3.0},
        ],
    )

    obstacle_publisher = Node(
        package="asv_utils",
        executable="obstacle_publisher",
        remappings=USV_REMAPS,
        parameters=[
            # x min, x max, y min, y max
            {"bouncing_area": [-15.0, 20.0, -10.0, 35.0]},
            {"marker_scale": 0.05},
            {"max_vel": 1.0},
        ],
    )

    return LaunchDescription(
        [
            dynamic_model_node,
            include("usv_nav", "aitsmc_launch.py"),
            spline_publisher_node,
            obstacle_publisher,
            include("asv_description", "usv_rviz_launch.py"),
            # Launch MPC separately: ros2 launch usv_nav mpc_launch.py
        ]
    )
