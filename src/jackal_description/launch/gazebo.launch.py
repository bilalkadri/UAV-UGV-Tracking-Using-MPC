import os
from ament_index_python.packages import get_package_share_path, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution, FindExecutable, EnvironmentVariable, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Paths
    pkg_share = FindPackageShare('jackal_description')
    urdf_path = PathJoinSubstitution([pkg_share, 'urdf', 'jackal.urdf.xacro'])
    world_path = PathJoinSubstitution([pkg_share, 'worlds', 'empty_minimal.sdf'])  # prefer an inline world that doesn't fetch models

    # Let Gazebo resolve package:// and model:// URIs from our package
    set_res_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[pkg_share, TextSubstitution(text=':'), EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value='')]
    )
    set_model_path = SetEnvironmentVariable(
        name='GZ_SIM_MODEL_PATH',
        value=[pkg_share, TextSubstitution(text=':'), EnvironmentVariable('GZ_SIM_MODEL_PATH', default_value='')]
    )

    # Start Gazebo (server + GUI) using ros_gz_sim's launcher
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ),
        # -r (run), -v4 (verbose), and pass full path to world
        launch_arguments={'gz_args': TextSubstitution(text='-v4 -r ')}.items()
    )

    # Publish /robot_description (topic spawner listens to) — pass is_sim:=true
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': ParameterValue(
                Command([FindExecutable(name='xacro'), ' ', urdf_path, ' ', 'is_sim:=true']),
                value_type=str
            )
        }]
    )

    # Spawn the robot from /robot_description (topic mode)
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "jackal",
            "-topic", "/robot_description",
            "-z", "0.5",
            # It helps to explicitly target the world name; adjust if yours differs:
            "-world", "empty_minimal",
        ],
        output="screen",
    )

    # Ensure gz server is really up before spawn (2s)
    delayed_spawn = TimerAction(period=2.0, actions=[spawn_entity])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        set_res_path,
        set_model_path,
        # Use the ros_gz_sim world launcher that accepts a path via GZ args
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
            ),
            launch_arguments={'gz_args': PathJoinSubstitution([TextSubstitution(text='-v4 -r '), world_path])}.items()
        ),
        node_robot_state_publisher,
        delayed_spawn,
    ])
