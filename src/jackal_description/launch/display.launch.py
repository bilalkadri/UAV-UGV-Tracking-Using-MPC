from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_jsp_gui = DeclareLaunchArgument('use_jsp_gui', default_value='false')
    rviz = DeclareLaunchArgument('rviz', default_value='true')

    xacro_path = PathJoinSubstitution([
        FindPackageShare('jackal_description'), 'urdf', 'jackal.urdf.xacro'
    ])

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', xacro_path]),
        value_type=str
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
    )

    jsp = Node(
        condition=UnlessCondition(LaunchConfiguration('use_jsp_gui')),
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
    )

    jsp_gui = Node(
        condition=IfCondition(LaunchConfiguration('use_jsp_gui')),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
    )

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('rviz')),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([FindPackageShare('jackal_description'), 'rviz', 'view.rviz'])],
        output='screen'
    )

    return LaunchDescription([
        use_jsp_gui, rviz, rsp, jsp, jsp_gui, rviz_node
    ])
