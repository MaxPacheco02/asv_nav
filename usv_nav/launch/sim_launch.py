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
            {
                "waypoints": [
                    0.0,
                    0.0,
                    0.3,
                    0.0,
                    3.0,
                    5.0,
                    7.0,
                    0.0,
                    6.0,
                    -5.0,
                    7.0,
                    -8.0,
                    1.0,
                    -5.0,
                    0.5,
                    -1.0,
                    0.0,
                    -1.0,
                ]
            },
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
            {"bouncing_area": [-15.0, 15.0, -15.0, 15.0]},
            {"marker_scale": 0.05},
            {"max_vel": 1.5},
            # gz sim real time factor (1.0 with the dynamic model)
            {"time_scale": 0.65},
            {"n_dyn_obs": 6},
            {"dummy_obs_pos": [1000.0, 1000.0]},
            {"static_obs": [7.0, 0.01, 0.0, -4.50]},
        ],
    )

    # Teleports gz models to the obstacles on /mpc/obs (only with Gazebo)
    gz_obstacle_node = Node(
        package="usv_nav",
        executable="gz_obstacle_node",
        output="screen",
    )

    return LaunchDescription(
        [
            # Either dynamic_model
            # dynamic_model_node,
            # or Gazebo
            include("usv_nav", "gazebo_launch.py"),
            gz_obstacle_node,
            #
            #
            include("usv_nav", "aitsmc_launch.py"),
            spline_publisher_node,
            obstacle_publisher,
            include("asv_description", "usv_rviz_launch.py"),
            # Launch MPC separately: ros2 launch usv_nav mpc_launch.py
        ]
    )
