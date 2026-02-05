# jackal_description/launch/gazebo_include.launch.py
import os, subprocess, tempfile, re
from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import FindExecutable, EnvironmentVariable, TextSubstitution, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def _run(context, *args, **kwargs):
    pkg_share = FindPackageShare('jackal_description').perform(context)
    xacro = FindExecutable(name='xacro').perform(context)
    xacro_file = os.path.join(pkg_share, 'urdf', 'jackal.urdf.xacro')

    # 1) Build URDF from xacro (simulation mode)
    urdf_xml = subprocess.check_output([xacro, xacro_file, 'is_sim:=true'])
    urdf_text = urdf_xml.decode('utf-8')
    urdf_file = os.path.join(tempfile.gettempdir(), 'jackal.urdf')
    with open(urdf_file, 'w') as f:
        f.write(urdf_text)

    # 2) Robot State Publisher needs the XML string, not a path
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_text, 'use_sim_time': True}],
    )

    # 3) Spawn the robot FROM THE URDF FILE
    #    (No /robot_description topic required; this is simpler & robust)
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-file', urdf_file,
            '-name', 'jackal',
            '-allow_renaming', 'true'
        ],
    )

    # 4) Controller params (use your package's YAML, not the demo's)
    diff_drive_yaml = PathJoinSubstitution(
        [FindPackageShare('jackal_description'), 'config', 'control_drive.yaml']
    )

    # 5) Spawn controllers after the entity exists
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '120',
            '--activate'
        ],
        output='screen',
    )

    diff_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'jackal_velocity_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '120',
            '--activate',
            '--param-file', diff_drive_yaml,
        ],
        output='screen',
    )

    # 6) Start Gazebo (empty world, running)
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                   'launch', 'gz_sim.launch.py'])]
        ),
        launch_arguments=[('gz_args', [' -r -v 1 empty.sdf'])],
    )

    # Ensure ordering: Gazebo -> spawn entity -> spawners
    return [
        gz_launch,
        node_robot_state_publisher,
        gz_spawn_entity,
        RegisterEventHandler(
            OnProcessExit(target_action=gz_spawn_entity,
                          on_exit=[joint_state_broadcaster_spawner])
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=joint_state_broadcaster_spawner,
                          on_exit=[diff_drive_controller_spawner])
        ),
    ]

def generate_launch_description():
    pkg = FindPackageShare('jackal_description')
    user_ws_lib = os.path.expanduser('~/gz_ros2_control_ws/install/gz_ros2_control/lib')

    set_res = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        [pkg, TextSubstitution(text=':'), EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value='')]
    )
    set_model = SetEnvironmentVariable(
        'GZ_SIM_MODEL_PATH',
        [pkg, TextSubstitution(text=':'), EnvironmentVariable('GZ_SIM_MODEL_PATH', default_value='')]
    )
    set_sys = SetEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH',
        [TextSubstitution(text=user_ws_lib + ':'), '/opt/ros/humble/lib:',
         EnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', default_value='')]
    )

    return LaunchDescription([set_res, set_model, set_sys, OpaqueFunction(function=_run)])
