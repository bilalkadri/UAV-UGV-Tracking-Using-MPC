# jackal_description/launch/gazebo_include.launch.py
import os, subprocess, tempfile, re
from textwrap import dedent

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.substitutions import FindExecutable, EnvironmentVariable, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_path

def _run(context, *args, **kwargs):
    pkg_share = FindPackageShare('jackal_description').perform(context)
    xacro = FindExecutable(name='xacro').perform(context)
    xacro_file = os.path.join(pkg_share, 'urdf', 'jackal.urdf.xacro')

    # 1) Build URDF from xacro (simulation mode)
    urdf_xml = subprocess.check_output([xacro, xacro_file, 'is_sim:=true'])
    urdf_file = os.path.join(tempfile.gettempdir(), 'jackal.urdf')
    with open(urdf_file, 'wb') as f:
        f.write(urdf_xml)

    # 2) URDF -> SDF (Harmonic CLI needs a file)
    #    NOTE: No plugin injection here; plugins assumed present in xacro.
    sdf_xml = subprocess.check_output(['gz', 'sdf', '-p', urdf_file])
    sdf_file = os.path.join(tempfile.gettempdir(), 'jackal.sdf')
    with open(sdf_file, 'wb') as f:
        f.write(sdf_xml)

    # 3) Use your packaged world directly
    world_file = os.path.join(pkg_share, 'worlds', 'empty.sdf')

    # 4) Start robot_state_publisher so gz_ros2_control can read robot_description
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_xml.decode('utf-8'), 'use_sim_time': True}],
    )

    # 5) Start Gazebo AFTER RSP (small delay)
    gz = TimerAction(period=1.0, actions=[
        ExecuteProcess(cmd=['gz', 'sim', world_file], output='screen')
    ])

    # 6) Spawn the robot into the running sim (since world doesn’t include it)
    spawn_entity = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                output='screen',
                arguments=['-file', sdf_file, '-name', 'jackal', '-z', '0.5']
            )
        ]
    )

    # 7) Spawn controllers after the manager is up
    spawn_jsb = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )
    spawn_diff = Node(
        package='controller_manager', executable='spawner',
        arguments=['diff_drive_base', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    return [
        gz, rsp,
        spawn_entity,
        TimerAction(period=3.0, actions=[spawn_jsb]),
        TimerAction(period=5.0, actions=[spawn_diff]),
    ]

def generate_launch_description():
    pkg = FindPackageShare('jackal_description')
    set_res = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        [pkg, TextSubstitution(text=':'), EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value='')]
    )
    set_model = SetEnvironmentVariable(
        'GZ_SIM_MODEL_PATH',
        [pkg, TextSubstitution(text=':'), EnvironmentVariable('GZ_SIM_MODEL_PATH', default_value='')]
    )
    return LaunchDescription([set_res, set_model, OpaqueFunction(function=_run)])
