# jackal_description/launch/gazebo_include.launch.py
import os, subprocess, tempfile
from textwrap import dedent
from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import FindExecutable, EnvironmentVariable, TextSubstitution
from launch_ros.substitutions import FindPackageShare

def _run(context, *args, **kwargs):
    pkg = FindPackageShare('jackal_description').perform(context)
    xacro = FindExecutable(name='xacro').perform(context)
    xacro_file = os.path.join(pkg, 'urdf', 'jackal.urdf.xacro')

    # 1) URDF from xacro (simulation mode)
    urdf = subprocess.check_output([xacro, xacro_file, 'is_sim:=true'])
    urdf_file = os.path.join(tempfile.gettempdir(), 'jackal.urdf')
    open(urdf_file, 'wb').write(urdf)

    # 2) URDF -> SDF using file path (Harmonic CLI doesn't read from "-")
    sdf = subprocess.check_output(['gz', 'sdf', '-p', urdf_file])
    sdf_file = os.path.join(tempfile.gettempdir(), 'jackal.sdf')
    open(sdf_file, 'wb').write(sdf)

    # 3) Minimal world that includes our robot + explicit default systems
    world_xml = dedent(f"""\
    <?xml version="1.0"?>
    <sdf version="1.9">
      <world name="empty_minimal">
        <!-- Explicit default systems so /create etc. behave -->
        <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
        <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
        <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>

        <gravity>0 0 -9.81</gravity>

        <!-- Inline ground -->
        <model name="ground">
          <static>true</static>
          <link name="link">
            <collision name="c"><geometry><plane><normal>0 0 0</normal><size>100 100</size></plane></geometry></collision>
            <visual name="v"><geometry><plane><normal>0 0 0</normal><size>100 100</size></plane></geometry></visual>
          </link>
        </model>

        <!-- Sun -->
        <light type="directional" name="sun">
          <cast_shadows>true</cast_shadows>
          <pose>0 0 -100 0 0 0</pose>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <direction>-1.5 0.2 -1</direction>
        </light>

        <!-- Include robot SDF -->
        <include>
          <uri>file://{sdf_file}</uri>
          <name>jackal</name>
          <pose>0 0 0.1 0 0 0</pose>
        </include>
      </world>
    </sdf>
    """)
    world_file = os.path.join(tempfile.gettempdir(), 'world_with_jackal.sdf')
    open(world_file, 'w').write(world_xml)

    return [ExecuteProcess(cmd=['gz', 'sim', world_file], output='screen')]

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
