from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
import os

def generate_launch_description():

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
        executable='Traj_Pred_EKF_Pub_v7.py',
        parameters=[{'use_sim_time': True}],
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
        period=5.0,  # delay in seconds
        actions=[
            Node(
                package='ardupilot_gz_bringup',
                executable='NMPC_Controller_v5.py',
                output='screen'
            )
        ]
    )





    return LaunchDescription([
        twist_republisher, # this is required to map the /jackal topics (cmd_vel) so that it can move 
        iris_mission,# this is launching the UAV 
        #uav_vel_estimator_launch, # MBK is not using this node
        ekf_trajectory_estimator_launch, # Estimating the position of the UGV using Odometry and then EKF (Aruco Marker)
        nmpc_controller_launch,   # THis is the main NMPC ROS-2 node
        #Data_Plot,
        Rect_Path, # this node moves the UGV in rectangular path
        aruco_detection_launch  # THis node is detetcting the ARUCO marker and then generating it's pose
    ])




# import os
# from launch import LaunchDescription
# from launch.actions import ExecuteProcess

# from launch_ros.actions import Node


# def generate_launch_description():

#     # --- Script 1: twist_stamped_republisher.py ---
#     twist_republisher = ExecuteProcess(
#         cmd=[
#             "python3",
#             os.path.expanduser("~/ardu_ws/src/ardupilot_gz/ardupilot_gz_bringup/scripts/twist_stamped_republisher.py")
#         ],
#         output="screen"
#     )

#     uav_vel_estimator_launch = Node(
#     package='ardupilot_gz_bringup',
#     executable='UAV_Velocity_Estimator.py',
#     output='screen',
#     parameters=[],
#     arguments=[],
#     namespace=''
#     )


#     ekf_trajectory_estimator_launch = Node(
#     package='ardupilot_gz_bringup',
#     executable='Traj_Pred_EKF_Pub_v3.py',
#     output='screen',
#     parameters=[],
#     arguments=[],
#     namespace=''
#     )
    
#     nmpc_controller_launch = Node(
#     package='ardupilot_gz_bringup',
#     executable='NMPC_Controller_v4.py',
#     output='screen',
#     parameters=[],
#     arguments=[],
#     namespace=''
#     )


#     aruco_detection_launch = Node(
#     package='ardupilot_gz_bringup',
#     executable='Aruco_Detection_Node.py',
#     output='screen',
#     parameters=[],
#     arguments=[],
#     namespace=''
#     )
    

#     # --- Script 2: iris_mission_53.py inside virtualenv ---
#     iris_mission = ExecuteProcess(
#         cmd=[
            
#             "python3 ~/ardu_ws/src/ardupilot_gz/ardupilot_gz_bringup/scripts/iris_mission_57_mbk.py"
#         ],
#         shell=True,
#         output="screen"
#     )




#     return LaunchDescription([
#         twist_republisher,
#         iris_mission,
#         uav_vel_estimator_launch, #This name will not appear as ROS-2 Node , the name in Class onstructor will appear 
#         ekf_trajectory_estimator_launch,  #This name will not appear as ROS-2 Node , the name in constructor will appear 
#         nmpc_controller_launch,
#         aruco_detection_launch
#     ])





