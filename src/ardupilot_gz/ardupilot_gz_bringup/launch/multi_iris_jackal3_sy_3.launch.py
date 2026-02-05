# ROS 2 Humble + Gazebo Harmonic
# Launch Iris (ArduPilot SITL) + your Jackal UGV in the same Harmonic world.

import os, subprocess, tempfile, re
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
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

def _prepare_and_spawn_jackal(context, *args, **kwargs):
    # ----- Build Jackal SDF from Xacro -----
    pkg_share = FindPackageShare('jackal_description').perform(context)
    xacro_file = os.path.join(pkg_share, 'urdf', 'jackal2.urdf.xacro')

    # 1) Xacro -> URDF (sim mode)
    urdf_xml = subprocess.check_output(['xacro', xacro_file, 'is_sim:=true'])
    urdf_file = os.path.join(tempfile.gettempdir(), 'jackal.urdf')
    with open(urdf_file, 'wb') as f:
        f.write(urdf_xml)

    # 2) URDF -> SDF
    sdf_xml = subprocess.check_output(['gz', 'sdf', '-p', urdf_file])
    sdf_text = sdf_xml.decode('utf-8')

    # Optional: auto‑inject gz_ros2_control (only if you need it injected at runtime)
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

    # 3) RSP (TF + /robot_description)
    # rsp = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     output='screen',
    #     parameters=[{
    #         'robot_description': urdf_xml.decode('utf-8'),
    #         'use_sim_time': True
    #     }]
    # )

    # 6) RSP before Gazebo
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='jackal',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_xml.decode('utf-8'), 'use_sim_time': True}],
    )

    # 4) Spawn Jackal into the already-running world
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

    # 5) Start controllers after the model exists
    spawn_jsb = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/jackal/controller_manager'],
        output='screen'
    )



    spawn_diff = Node(
        package='controller_manager', executable='spawner',
        arguments=['jackal_velocity_controller', '--controller-manager', '/jackal/controller_manager'],
        output='screen')
    

    return [rsp, spawn, spawn_jsb, spawn_diff]


def generate_launch_description():
    # ----- Paths -----
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_project_bringup = get_package_share_directory("ardupilot_gz_bringup")
    pkg_ap_gazebo = get_package_share_directory("ardupilot_gz_gazebo")
    jackal_share = FindPackageShare('jackal_description')
    ap_share = FindPackageShare('ardupilot_gz_gazebo')

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
    env = [
        SetEnvironmentVariable('GZ_VERSION', 'harmonic'),

        # Models/resources (Jackal + ArduPilot worlds/models)
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', [
            jackal_share, TextSubstitution(text=':'),
            ap_share,     TextSubstitution(text=':'),
            TextSubstitution(text=os.getenv('GZ_SIM_RESOURCE_PATH', ''))
        ]),
        SetEnvironmentVariable('GZ_SIM_MODEL_PATH', [
            jackal_share, TextSubstitution(text=':'),
            ap_share,     TextSubstitution(text=':'),
            TextSubstitution(text=os.getenv('GZ_SIM_MODEL_PATH', ''))
        ]),

        # System plugins — prepend your custom gz_ros2_control build
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', [
            TextSubstitution(text=user_ws_lib + ':'),               # your custom build
            TextSubstitution(text='/opt/ros/humble/lib:'),          # ROS 2 system libs
            TextSubstitution(text=os.getenv('GZ_SIM_SYSTEM_PLUGIN_PATH', ''))
        ]),
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

    # ----- Insert Iris robot (no second server) -----
    iris = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("ardupilot_gz_bringup"))
                / "launch" / "robots" / "iris.launch.py")
        )
    )
    # RViz.
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", f'{Path(pkg_project_bringup) / "rviz" / "iris.rviz"}'],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    return LaunchDescription(
        args + env + [
            gz_server,
            gz_gui,
            iris,
            rviz,
            # Delay Jackal spawn a little so the world is ready
            TimerAction(period=4.0, actions=[OpaqueFunction(function=_prepare_and_spawn_jackal)])
        ]
    )
