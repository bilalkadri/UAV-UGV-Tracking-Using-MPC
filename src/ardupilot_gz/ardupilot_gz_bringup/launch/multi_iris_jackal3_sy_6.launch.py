# ROS 2 Humble + Gazebo Harmonic
# Launch Iris (ArduPilot SITL) + Jackal UGV in the same Harmonic world.

import os
import subprocess
import tempfile
import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
    SetEnvironmentVariable, TimerAction, ExecuteProcess
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition

def _prepare_and_spawn_jackal(context, *args, **kwargs):
    """
    Convert jackal xacro -> urdf -> sdf, start robot_state_publisher (namespaced),
    spawn model into gz server, and spawn controllers with a delay.
    """
    # ----- Build Jackal SDF from Xacro -----
    pkg_share = get_package_share_directory('jackal_description')
    xacro_file = os.path.join(pkg_share, 'urdf', 'jackal2.urdf.xacro')
    if not os.path.exists(xacro_file):
        raise FileNotFoundError(f"Jackal xacro not found: {xacro_file}")

    # 1) Xacro -> URDF (sim mode)
    urdf_xml = subprocess.check_output(['xacro', xacro_file, 'is_sim:=true'])
    urdf_text = urdf_xml.decode('utf-8')

    # write a temporary URDF file
    urdf_file = os.path.join(tempfile.gettempdir(), 'jackal.urdf')
    with open(urdf_file, 'wb') as f:
        f.write(urdf_xml)

    # 2) URDF -> SDF
    sdf_xml = subprocess.check_output(['gz', 'sdf', '-p', urdf_file])
    sdf_text = sdf_xml.decode('utf-8')

    # Optional: if you want to inject gz_ros2_control plugin at runtime, uncomment & adapt
    # ctrl_yaml_abs = os.path.join(pkg_share, 'config', 'control_drive.yaml')
    # plugin_block = f'''
    #   <plugin filename="libgz_ros2_control-system.so"
    #           name="gz_ros2_control::GazeboSimROS2ControlPlugin">
    #     <parameters>{ctrl_yaml_abs}</parameters>
    #   </plugin>'''
    # if "gz_ros2_control::GazeboSimROS2ControlPlugin" not in sdf_text:
    #     sdf_text = re.sub(r'(</model>)', plugin_block + r'\1', sdf_text, count=1)

    sdf_file = os.path.join(tempfile.gettempdir(), 'jackal.sdf')
    with open(sdf_file, 'w') as f:
        f.write(sdf_text)

    # 3) Robot State Publisher (TF + /robot_description) for Jackal (namespaced)
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='jackal',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_text, 'use_sim_time': True}],
    )

    # 4) Spawn Jackal into the already-running world (use LaunchConfiguration values)
    x = LaunchConfiguration('jackal_x').perform(context)
    y = LaunchConfiguration('jackal_y').perform(context)
    z = LaunchConfiguration('jackal_z').perform(context)
    yaw = LaunchConfiguration('jackal_yaw').perform(context)

    spawn = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_sim', 'create',
             '-file', sdf_file,
             '-name', 'jackal',
             '-x', x, '-y', y, '-z', z, '-Y', yaw],
        output='screen'
    )

    # 5) Start controllers after the model exists (delay spawners slightly)
    # Use TimerAction wrappers so they run after spawn completes (these timers run relative to launch time,
    # but since _prepare_and_spawn_jackal is already called inside a TimerAction, these add extra safety)
    spawn_jsb = TimerAction(
        period=2.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/jackal/controller_manager'],
            output='screen'
        )]
    )

    spawn_diff = TimerAction(
        period=3.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['jackal_velocity_controller', '--controller-manager', '/jackal/controller_manager'],
            output='screen'
        )]
    )

    # Return nodes/actions to be added to the launch system
    return [rsp, spawn, spawn_jsb, spawn_diff]


def generate_launch_description():
    # ----- Paths -----
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_project_bringup = get_package_share_directory("ardupilot_gz_bringup")
    pkg_ap_gazebo = get_package_share_directory("ardupilot_gz_gazebo")
    pkg_jackal = get_package_share_directory("jackal_description")

    # Your custom gz_ros2_control build (adjust if different)
    user_ws_lib = os.path.expanduser('~/gz_ros2_control_ws/install/gz_ros2_control/lib')

    # ----- Launch args -----
    world_default = str(Path(pkg_ap_gazebo) / "worlds" / "iris_runway.sdf")
    args = [
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("world", default_value=world_default),
        DeclareLaunchArgument("jackal_x", default_value="0.0"),
        DeclareLaunchArgument("jackal_y", default_value="2.0"),
        DeclareLaunchArgument("jackal_z", default_value="0.5"),
        DeclareLaunchArgument("jackal_yaw", default_value="0.0"),
    ]

    # ----- Environment (plugins, models, resources) -----
    # Provide explicit model/resource paths so Gazebo Harmonic finds files
    gz_sim_resource_path = (
        str(Path(pkg_jackal) / 'models') + ':' +
        str(Path(pkg_ap_gazebo) / 'models') + ':' +
        os.getenv('GZ_SIM_RESOURCE_PATH', '')
    )

    gz_sim_model_path = (
        str(Path(pkg_jackal) / 'models') + ':' +
        str(Path(pkg_ap_gazebo) / 'models') + ':' +
        os.getenv('GZ_SIM_MODEL_PATH', '')
    )

    env = [
        SetEnvironmentVariable('GZ_VERSION', 'harmonic'),

        # Models/resources (Jackal + ArduPilot worlds/models)
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', TextSubstitution(text=gz_sim_resource_path)),
        SetEnvironmentVariable('GZ_SIM_MODEL_PATH', TextSubstitution(text=gz_sim_model_path)),

        # System plugins — prepend your custom gz_ros2_control build (if present)
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', TextSubstitution(text=
            (user_ws_lib + ':' if os.path.isdir(user_ws_lib) else '') +
            '/opt/ros/humble/lib:' +
            os.getenv('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
        )),
    ]

    # ----- Start one Gazebo server with the Iris runway world -----
    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(pkg_ros_gz_sim) / "launch" / "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ['-v4 -s -r ', LaunchConfiguration('world')]
        }.items(),
    )

    # GUI (separate client)
    gz_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(pkg_ros_gz_sim) / "launch" / "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "-v4 -g"}.items(),
    )

    # ----- Optionally include SITL if a sitl.launch.py exists in the bringup package -----
    sitl_launch_path = Path(pkg_project_bringup) / "launch" / "sitl.launch.py"
    sitl_include = None
    if sitl_launch_path.exists():
        sitl_include = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(sitl_launch_path))
        )
    else:
        # If there's no sitl.launch.py, some robot/bringup launch files may start SITL internally.
        # Keep this None and continue — the iris launch will be included below.
        sitl_include = None

    # ----- Insert Iris robot (no second server) -----
    iris_launch_path = Path(pkg_project_bringup) / "launch" / "robots" / "iris.launch.py"
    if not iris_launch_path.exists():
        raise FileNotFoundError(f"Expected iris.launch.py at {iris_launch_path}")

    iris = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(iris_launch_path))
    )

    # RViz.
    rviz_cfg = Path(pkg_project_bringup) / "rviz" / "iris.rviz"
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", str(rviz_cfg)] if rviz_cfg.exists() else [],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # Add an iris robot_state_publisher only if the iris launch doesn't provide it.
    iris_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='iris_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Build the final list of launch actions
    launch_actions = []
    launch_actions += args
    launch_actions += env
    launch_actions += [gz_server, gz_gui]

    # add SITL include if available (before iris)
    if sitl_include is not None:
        launch_actions.append(sitl_include)

    # include iris and rviz
    launch_actions += [iris, rviz, iris_rsp]

    # Delay Jackal spawn a little so the world is ready (use OpaqueFunction inside TimerAction)
    launch_actions.append(TimerAction(period=10.0, actions=[OpaqueFunction(function=_prepare_and_spawn_jackal)]))

    return LaunchDescription(launch_actions)

