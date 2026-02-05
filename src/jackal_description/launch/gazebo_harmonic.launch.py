# SPDX-License-Identifier: MIT
# ROS 2 Humble + Gazebo Harmonic robot spawn (URDF from xacro)
import os
import subprocess

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.substitutions import PathJoinSubstitution, FindExecutable, TextSubstitution, EnvironmentVariable
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def _gen_urdf_and_spawn(context, *args, **kwargs):
    """Generate /tmp/jackal.urdf from xacro and spawn with ros_gz_sim create -file."""
    pkg_share = FindPackageShare('jackal_description').perform(context)
    xacro_exe = FindExecutable(name='xacro').perform(context)
    xacro_path = os.path.join(pkg_share, 'urdf', 'jackal.urdf.xacro')
    tmp_urdf = '/tmp/jackal.urdf'

    # Generate URDF from xacro (pass is_sim:=true if your xacro uses it)
    xml = subprocess.check_output([xacro_exe, xacro_path, 'is_sim:=true'])
    with open(tmp_urdf, 'wb') as f:
        f.write(xml)

    # World name must match <world name="..."> inside your SDF
    world_name = 'empty_minimal'

    # Spawn the robot in Gazebo
    return [ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-world', world_name,
            '-name', 'jackal',
            '-file', tmp_urdf,
            '-z', '0.5'
        ],
        output='screen'
    )]

def generate_launch_description():
    pkg_share = FindPackageShare('jackal_description')
    world_path = PathJoinSubstitution([pkg_share, 'worlds', 'empty_minimal.sdf'])

    # Make sure Gazebo can resolve package://jackal_description/... and model:// URIs
    set_res_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[pkg_share, TextSubstitution(text=':'), EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value='')]
    )
    set_model_path = SetEnvironmentVariable(
        name='GZ_SIM_MODEL_PATH',
        value=[pkg_share, TextSubstitution(text=':'), EnvironmentVariable('GZ_SIM_MODEL_PATH', default_value='')]
    )

    # Start Gazebo Harmonic with your world (server+GUI in one)
    gz = ExecuteProcess(
        cmd=['gz', 'sim', world_path],
        output='screen'
    )

    # Optional: publish TFs in parallel for RViz (not required for spawn)
    # Keep it simple; disable if not needed.
    # rsp = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     parameters=[{
    #         'robot_description': ('unused here'),
    #     }],
    #     output='screen'
    # )

    # Give Gazebo a moment to boot before spawning
    spawn = TimerAction(period=2.0, actions=[OpaqueFunction(function=_gen_urdf_and_spawn)])

    return LaunchDescription([
        set_res_path,
        set_model_path,
        gz,
        spawn,
        # rsp,
    ])
