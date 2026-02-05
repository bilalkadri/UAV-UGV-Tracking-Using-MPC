# jackal_description/launch/gazebo_include.launch.py
import os, subprocess, tempfile, re
from textwrap import dedent

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.substitutions import FindExecutable, EnvironmentVariable, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
def _run(context, *args, **kwargs):

    pkg_share = FindPackageShare('jackal_description').perform(context)
    xacro = FindExecutable(name='xacro').perform(context)
    xacro_file = os.path.join(pkg_share, 'urdf', 'jackal.urdf.xacro')

    # 1) Build URDF from xacro (simulation mode)
    urdf_xml = subprocess.check_output([xacro, xacro_file, 'is_sim:=true'])
    urdf_file = os.path.join(tempfile.gettempdir(), 'jackal.urdf')
    with open(urdf_file, 'wb') as f:
        f.write(urdf_xml)

    # 2) URDF -> SDF
    sdf_xml = subprocess.check_output(['gz', 'sdf', '-p', urdf_file])
    sdf_text = sdf_xml.decode('utf-8')

    # 3) Conditionally inject gz_ros2_control plugin if missing
    ctrl_yaml_abs = os.path.join(
        FindPackageShare('jackal_description').perform(context),
        'config',
        'control_drive.yaml'
    )  # absolute path
    
    
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare('jackal_description'),
            'config',
            'control_drive.yaml',
        ]
    )
    ros2_control_plugin_block = f"""

      <plugin filename="libgz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
        <robot_param>robot_description</robot_param>
        <robot_param_node>robot_state_publisher</robot_param_node>
        <controller_manager_name>controller_manager</controller_manager_name>
        <parameters>{ctrl_yaml_abs}</parameters>
      </plugin>
    """
    if "gz_ros2_control::GazeboSimROS2ControlPlugin" not in sdf_text:
        sdf_text = re.sub(r'(</model>)', ros2_control_plugin_block + r'\1', sdf_text, count=1)

    # 4) Write SDF to temp
    sdf_file = os.path.join(tempfile.gettempdir(), 'jackal.sdf')
    with open(sdf_file, 'w') as f:
        f.write(sdf_text)

    # 5) Minimal world
    world_xml = dedent(f"""\
    <?xml version="1.0"?>
    <sdf version="1.9">
      <world name="empty_minimal">
        <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
        <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
        <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
                       


        <gravity>0 0 -9.81</gravity>

        <model name="ground">
          <static>true</static>
          <link name="link">
            <collision name="c"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision>
            <visual name="v"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></visual>
          </link>
        </model>
                       
        <light type="directional" name="sun">
        <cast_shadows>true</cast_shadows>
        <!-- Position the sun high above the world -->
        <pose>0 0 50 0 0 50</pose>

        <!-- Warm sunlight tone -->
        <diffuse>1.0 0.95 0.85 1</diffuse>
        <specular>0.2 0.2 0.2 1</specular>

        <!-- Direction pointing slightly downward and angled -->
        <direction>-0.5 0.4 -1.0</direction>

        <!-- Make sure it's bright enough -->
        <intensity>1.0</intensity>
        </light>

        <include>
          <uri>file://{sdf_file}</uri>
        #   <name>jackal</name>
          <pose>0 0 0.5 0 0 0</pose>
        </include>
      </world>
    </sdf>
    """)
    world_file = os.path.join(tempfile.gettempdir(), 'world_with_jackal.sdf')
    with open(world_file, 'w') as wf:
        wf.write(world_xml)

    # 6) RSP before Gazebo
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        # namespace='jackal',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_xml.decode('utf-8'), 'use_sim_time': True}],
    )

    # 7) Gazebo after RSP
    gz = TimerAction(period=3.0, actions=[
        ExecuteProcess(cmd=['gz', 'sim', '-r', world_file], output='screen')
    ])

    # Spawners (JSB first, then diff-drive), with bigger timeout
    spawn_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    spawn_diff = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'jackal_velocity_controller',
            '--param-file',
            robot_controllers,
        ],
        output='screen'
    )

    return [
        rsp, gz,
        TimerAction(period=10.0, actions=[spawn_jsb]),
        TimerAction(period=13.0, actions=[spawn_diff]),
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
        [TextSubstitution(text=user_ws_lib + ':'), '/opt/ros/humble/lib:', EnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', default_value='')]
    )
    return LaunchDescription([set_res, set_model, set_sys, OpaqueFunction(function=_run)])
