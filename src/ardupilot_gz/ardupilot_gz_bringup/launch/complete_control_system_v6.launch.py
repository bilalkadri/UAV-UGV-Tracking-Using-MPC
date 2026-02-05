from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os
from datetime import datetime

def generate_launch_description():

# Create bag filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_dir = os.path.expanduser('~/uav_ugv_bags')
    bag_file = os.path.join(bag_dir, f'tracking_session_{timestamp}')
    
    # Ensure directory exists
    os.makedirs(bag_dir, exist_ok=True)
    
    # The error says minimum is 86016 which is about 86KB
    # Let's use 100MB (100 * 1024 * 1024 = 104857600 bytes)
    # Or just use a large value like 500MB
    rosbag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', '-a',
            '-o', bag_file,
            '--storage', 'sqlite3'
        ],
        output='screen',
        shell=False
    )

# Create bag filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_dir = os.path.expanduser('~/uav_ugv_bags')
    bag_file = os.path.join(bag_dir, f'tracking_session_{timestamp}')
    
    # Ensure directory exists
    os.makedirs(bag_dir, exist_ok=True)
    
    # The error says minimum is 86016 which is about 86KB
    # Let's use 100MB (100 * 1024 * 1024 = 104857600 bytes)
    # Or just use a large value like 500MB
    # rosbag_record = ExecuteProcess(
    #     cmd=[
    #         'ros2', 'bag', 'record', '-a',
    #         '-o', bag_file,
    #         '--storage', 'sqlite3'
    #     ],
    #     output='screen',
    #     shell=False
    # )

    twist_republisher = ExecuteProcess(
        cmd=[
            "python3",
            os.path.expanduser("~/ardu_ws/src/ardupilot_gz/ardupilot_gz_bringup/scripts/twist_stamped_republisher.py")
        ],
        output="screen"
    )

    uav_vel_estimator_launch = Node(
        package='ardupilot_gz_bringup',
        executable='UAV_Velocity_Estimator.py',
        output='screen'
    )

    ekf_trajectory_estimator_launch = Node(
        package='ardupilot_gz_bringup',
        executable='Traj_Pred_EKF_Pub_v10.py',
        output='screen'
    )
    
    #Diagnostic_Plot= Node(
        #package='ardupilot_gz_bringup',
        #executable='Diagnostic_Plot_EKF_Node.py',
        #output='screen'
    #)
    
    #Data_Plot= Node(
        #package='ardupilot_gz_bringup',
        #executable='Plot_Data_v7.py',
        #output='screen'
    #)
    
    aruco_detection_launch = Node(
        package='ardupilot_gz_bringup',
        executable='Aruco_Detection_Node_v6.py',
        output='screen'
    )
    
    Rect_Path= Node(
        package='ardupilot_gz_bringup',
        executable='ugv_rect_path_node_v2.py',
        output='screen'
    )
    
    iris_mission = ExecuteProcess(
        cmd=[
            "python3 ~/ardu_ws/src/ardupilot_gz/ardupilot_gz_bringup/scripts/iris_mission_57_mbk.py"
        ],
        shell=True,
        output="screen"
    )
    # ==========================
    #   DELAYED NMPC CONTROLLER
    # ==========================
    nmpc_controller_launch = TimerAction(
        period=1.0,  # delay in seconds
        actions=[
            Node(
                package='ardupilot_gz_bringup',
                executable='NMPC_Controller_v5.py',
                output='screen'
            )
        ]
    )



    return LaunchDescription([
        # Launch arguments
      
        
        # Create directory first
 

        twist_republisher,
        iris_mission,
        uav_vel_estimator_launch,
        ekf_trajectory_estimator_launch,
        nmpc_controller_launch,   # delayed node
        #Data_Plot,
        Rect_Path,
        aruco_detection_launch,
        rosbag_record
    ])



