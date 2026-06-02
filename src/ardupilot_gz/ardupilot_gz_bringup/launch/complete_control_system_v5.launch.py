from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os
from datetime import datetime
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node, SetParameter # Added SetParameter

def generate_launch_description():

    # ----- Launch arguments -----
    # Declare use_sim_time once here
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true', 
        description='Use simulation clock'
    )
    
    # Create a configuration variable to pass to nodes
    use_sim_time = LaunchConfiguration('use_sim_time')
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
            '--storage', 'sqlite3',
            '--use-sim-time' ,# Added this flag so the bag records simulation time
            '--polling-interval', '1000'  # 🌟 CRITICAL: Checks for new/renamed topics every 1000ms
        ],
        output='screen',
        shell=False
    )

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
        parameters=[{'use_sim_time': use_sim_time}], # Added param
        output='screen'
    )

    ekf_trajectory_estimator_launch = Node(
        package='ardupilot_gz_bringup',
        executable='Traj_Pred_Using_Sensor_Data.py',
        parameters=[{'use_sim_time': use_sim_time}], # Linked to LaunchConfiguration
        output='screen'
    )
    
    

    ekf_trajectory_estimator_launch_test = Node(
        package='ardupilot_gz_bringup',
        executable='UGV_Location_Prediction_EKF_v11.py',
        parameters=[{'use_sim_time': use_sim_time}], # Linked to LaunchConfiguration
        output='screen'
    )
    
    #Data_Plot= Node(
        #package='ardupilot_gz_bringup',
        #executable='Plot_Data_v7.py',
        #output='screen'
    #)
    
    aruco_detection_launch = Node(
        package='ardupilot_gz_bringup',
        executable='Aruco_Detection_Node_v6.py',
        parameters=[{'use_sim_time': use_sim_time}], # Added param
        output='screen'
    )
    
    Rect_Path= Node(
        package='ardupilot_gz_bringup',
        executable='Moving_the_UGV.py',
        parameters=[{'use_sim_time': use_sim_time}], # Added param
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
    controller_launch = TimerAction(
        period=1.0,# delay in seconds
        actions=[
            Node(
                package='ardupilot_gz_bringup',
                executable='Controller_v5.py',
                parameters=[{'use_sim_time': use_sim_time}], # Added param
                output='screen'
            )
        ]
    )

    plotting_for_jp_launch = TimerAction(
        period=1.0,  # delay in seconds
        actions=[
            Node(
                package='ardupilot_gz_bringup', 
                executable='plotting_for_JP_MBK.py',
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            )
        ]
    )

    # Why this is necessary
    # In your SDF, the camera is defined in the pitch_link. However, OpenCV (and your ArUco code) expects coordinates in an Optical Frame (Z pointing out of the lens).
    # By adding this node:
    # You create the camera_optical_frame required by your script.
    # You link it to the pitch_link which is already part of your gimbal's physical TF tree.
    # The rotation -1.5708 0 -1.5708 (which is −90∘ around Z then −90∘ around X) correctly aligns the ROS standard "Forward-Left-Up" orientation 
    # of the pitch_link with the "Forward-Right-Down" orientation of the camera sensor.
    # Define the static transform from pitch_link to camera_optical_frame
    # Args: x y z yaw pitch roll parent_frame child_frame
    camera_optical_tf = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='camera_base_to_optical',
    # Args: x y z yaw pitch roll parent child
    # This rotation maps the pitch_link to the OpenCV Optical Frame (Z-forward)
    arguments=['0', '0', '0', '-1.5708', '-0.7854', '0.7854', 'pitch_link', 'camera_optical_frame'],
    parameters=[{'use_sim_time': use_sim_time}]
    )




   

    return LaunchDescription([
        use_sim_time_arg,
        # SetParameter forces 'use_sim_time' for ALL nodes in this file automatically
        SetParameter(name='use_sim_time', value=use_sim_time),
        twist_republisher, # this is required to map the /jackal topics (cmd_vel) so that it can move 
        iris_mission,# this is launching the UAV 
        #uav_vel_estimator_launch, # MBK is not using this node
        ekf_trajectory_estimator_launch, # Estimating the position of the UGV using Odometry and then EKF (Aruco Marker)
        ekf_trajectory_estimator_launch_test,
        controller_launch,   # THis is the main controller ROS-2 node
        #Data_Plot,
        Rect_Path, # this node moves the UGV in rectangular path
        camera_optical_tf,
        aruco_detection_launch,  # THis node is detetcting the ARUCO marker and then generating it's pose
        rosbag_record, # This is recording the rosbag with simulation time (use_sim_time) and all topics (-a)
        plotting_for_jp_launch
    ])



