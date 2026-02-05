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

    # 2) URDF -> SDF (Harmonic CLI needs a file, not "-")
    sdf_xml = subprocess.check_output(['gz', 'sdf', '-p', urdf_file])
    sdf_text = sdf_xml.decode('utf-8')

    # 3) NO injection of ros2_control or joint_state_publisher here.
    #    You already include the ros2_control plugin inside the xacro/URDF.

    # 4) Write SDF to a temp file
    sdf_file = os.path.join(tempfile.gettempdir(), 'jackal.sdf')
    with open(sdf_file, 'w') as f:
        f.write(sdf_text)

    # 5) Minimal world that includes our robot + explicit default systems
    world_xml = dedent(f"""\
    <?xml version="1.0"?>
    <sdf version="1.9">
      <world name="empty_minimal">
        <!-- Explicit default systems -->
        <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
        <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
        <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>

        <gravity>0 0 -9.81</gravity>

        <!-- Inline ground -->
        <model name="ground">
          <static>true</static>
          <link name="link">
            <collision name="c"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision>
            <visual name="v"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></visual>
          </link>
        </model>

        <!-- Sun -->
        <light type="directional" name="sun">
          <cast_shadows>true</cast_shadows>
          <pose>0 0 10 0 0 0</pose>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <direction>-0.5 0.2 -1</direction>
        </light>              

        <!-- Include robot SDF -->
        <include>
          <uri>file://{sdf_file}</uri>
          <name>jackal</name>
          <pose>0 0 0.5 0 0 0</pose>
        </include>
      </world>

      <model name="jackal">
        <!-- your links/joints (converted) -->
        <plugin filename="gz_ros2_control-system" name="gz_ros2_control-system::GazeboSimROS2ControlPlugin">
          <robot_param>robot_description</robot_param>
          <robot_param_node>robot_state_publisher</robot_param_node>
          <controller_manager_name>controller_manager</controller_manager_name>
          <parameters>$(find jackal_description)/config/control.yaml</parameters>
          <ros><namespace>jackal</namespace></ros>
        </plugin>
      </model>

    </sdf>
    """)

    world_file = os.path.join(tempfile.gettempdir(), 'world_with_jackal.sdf')
    with open(world_file, 'w') as wf:
        wf.write(world_xml)

    # 6) Start robot_state_publisher so gz_ros2_control can read robot_description
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_xml.decode('utf-8'), 'use_sim_time': True}],
    )

    # 7) Start Gazebo AFTER RSP (small delay)
    gz = TimerAction(period=1.0, actions=[
        ExecuteProcess(cmd=['gz', 'sim', '-r', world_file], output='screen')
    ])

    # 8) Spawn controllers (your controller manager should come from the plugin inside xacro)
    spawn_jsb = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/jackal/controller_manager'],
        output='screen'
    )
    spawn_diff = Node(
        package='controller_manager', executable='spawner',
        arguments=['jackal_velocity_controller', '--controller-manager', '/jackal/controller_manager'],
        output='screen'
    )

    return [
        gz, rsp,
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
