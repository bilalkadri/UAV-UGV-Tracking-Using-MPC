#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import traceback
from math import radians
import math
from std_msgs.msg import String
import tf2_ros
from geometry_msgs.msg import TransformStamped

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# -----------------------------------------------------------------------------
# -------------------Motivation for this modified approach---------------------
# -----------------------------------------------------------------------------
# A)When ArUco-based visual measurements are unavailable, the UAV can still access the UGV’s position 
# and velocity information through a wireless communication link, implemented in this work via a 
# ROS odometry subscriber.

# B) Using the received UGV odometry data, the UAV can navigate toward the UGV without relying on the 
# EKF-based relative state estimation.

# C) During this phase, the EKF prediction and update steps are intentionally disabled, as no reliable 
# visual measurements are available for correction.

# D)Since the control strategy is based on Nonlinear Model Predictive Control (NMPC), a prediction 
# horizon must be constructed using the UGV odometry data published on the 
# jackal/jackal_velocity_controller/odom topic.

# E)This predicted UGV trajectory is provided as an input to the NMPC, enabling the controller 
# to generate control actions that drive the UAV toward the UGV.

# F)Once the ArUco marker becomes visible, the EKF is activated to perform prediction and measurement 
# updates, providing accurate relative state estimates for closed-loop visual servoing.
    


# ------------ Node ------------
class AbsolutePoseEKF(Node):
    def __init__(self):

    # Constructor (__init__) does:
    # Initializes EKF: Creates state vector self.x (6×1) and covariance matrices P, Q, R
    # Sets up subscribers: For UAV/UGV poses, velocities, and ArUco detection
    # Sets up publishers: For relative pose, predicted trajectory, and EKF status
    # Initializes buffers: For storing UAV/UGV velocities and positions
    # Starts timer: Calls ekf_predict_publish() every dt (0.01s) seconds

        super().__init__('ugv_pose_ekf_and_trajectory_prediction_node')
        # self.declare_parameter('use_sim_time', True)

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        #----------------------------------------------------------------------------------------
        #                         Class Variables 
        #----------------------------------------------------------------------------------------
        # Mode control
        self.ekf_active = False  # True: using EKF, False: using odometry
        self.aruco_lost_count = 0
        self.aruco_lost_threshold = 10  # 1 second at 10Hz
        
        # Variables to store UGV information 
        # self.ugv_pos_where_is_this_used = np.zeros(3)
        self.ugv_predictor = None
        self.ugv_position_in_odom_frame =  [0.0, 2.0, 0.0]   # Store UGV position by transforming data from /jacakal/base_link to /odom frame 
        self.ugv_yaw_in_odom_frame=0.0

        self.ugv_lin_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link = np.zeros(3)      # UGV velocity (jacakl/odom frame),
        self.ugv_ang_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link = np.zeros(3)  # UGV angular velocity (jackal/odom frame), 
        self.roll_ugv_in_jackal_odom_frame = 0.0
        self.pitch_ugv_in_jackal_odom_frame = 0.0
        self.ugv_yaw_in_jackal_odom_frame = 0.0
        
        
        # Variables to store UAV state
        # Conclusion: self.uav_position_in_odom_frame stores world coordinates despite confusing frame_id label
        # Despite the frame_id saying base_link, the /ap/pose/filtered topic 
        # is publishing the UAV's position in the World/Local frame i.e. /odom in this case.
        self.uav_position_in_odom_frame = np.zeros(3)
        # self.uav_orientation_in_odom_frame=np.zeros(3)
        self.uav_yaw_in_odom_frame = 0.0  # ⭐⭐ NEW: Store UAV yaw ⭐⭐
        self.dt = 0.01  
        self.pred_N = 10 #prediction horizon
        self.aruco_detected = False
        self.aruco_pose_measurement = None
        self.mahalanobis_threshold = 15.0
        self.uav_lin_vel_wrt_odom_frame_expressed_in_base_link = np.zeros(3)      # UAV velocity (body frame) ⭐⭐ CHANGED ⭐⭐
        self.uav_ang_vel_wrt_odom_frame_expressed_in_base_link = np.zeros(3)  # UAV angular velocity (body frame)

        self.trajectory = []  # Stores the predicted trajectory
        

        # ============= ⭐⭐ CHANGE: State Interpretation ⭐⭐ =============
        # State is: UGV position RELATIVE TO UAV in UAV body frame
        # x[0:3] = UGV position in UAV body frame (x-forward, y-left, z-up)
        # x[3:6] = UGV orientation RELATIVE to UAV (Euler angles)
        self.ugv_pos_and_orient_in_UAV_frame = np.zeros((6,1))
        self.ugv_pos_and_orient_in_UAV_frame[0, 0] = 2.0    # x position (Forward)
        self.ugv_pos_and_orient_in_UAV_frame[1, 0] = 0.0    # y position (Left)
        self.ugv_pos_and_orient_in_UAV_frame[2, 0] = -2.0   # z position (Up) -> Drone at Z=2, UGV at Z=0 means UGV is -2m relative to drone
        self.ugv_pos_and_orient_in_UAV_frame[3, 0] = 0.0   # roll
        self.ugv_pos_and_orient_in_UAV_frame[4, 0] = 0.0   # pitch
        self.ugv_pos_and_orient_in_UAV_frame[5, 0] = -1.55   # yaw  -1.55 radians (approximately -89°)this is from Gazebo world
        #self.P = np.eye(6) * 0.1   #combination 1
        self.P = np.eye(6) * 1  # Increased initial uncertainty for more responsiveness
        #self.Q = np.eye(6) * 0.2  #combination 1
        self.Q = np.eye(6)*0.001


        # ====================== Bumpless Mode Switching Variables ======================

        # Stores the most recent EKF pose estimate (PoseStamped)
        # This is updated every time EKF runs successfully
        self.last_ekf_pose = None

        # Stores the most recent odometry-based relative pose (PoseStamped)
        # This is updated every time odometry mode runs
        self.last_odom_pose = None

        # Blending factor between odometry and EKF
        # 0.0 → 100% odometry
        # 1.0 → 100% EKF
        self.blend_alpha = 0.0

        # Rate at which blending happens per timer cycle
        # Smaller value → smoother but slower transition
        # Typical stable range: 0.02 – 0.1
        self.blend_rate = 0.05

        # Flag indicating that a mode transition is currently happening
        # (used only for logic clarity and debugging)
        self.mode_transition_active = False

        # Stores the EKF active flag from the previous timer cycle
        # Used to detect mode transitions (EKF ↔ Odometry)
        self.prev_mode_ekf = False




        # MPC command velocities computed by NMPC Controller node
        self.vx_mpc=0
        self.vy_mpc=0
        self.vz_mpc=0
        self.yaw_dot_mpc=0

        pos_sigma = 0.04 #combination 1
        pos_sigma = 0.1 #combination 2
        ang_sigma = radians(2.0)
        self.R = np.diag([
            pos_sigma**2, pos_sigma**2, pos_sigma**2,
            ang_sigma**2, ang_sigma**2, ang_sigma**2
        ])
         
        #----------------------------------------------------------------------------------------
        #                                       tf transformation
        #----------------------------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        #----------------------------------------------------------------------------------------
        #                                       Subscribers
                                                                                                                                                                            
      
        # self.create_subscription(Imu, '/imu', self.imu_cb, 10)
        #UAV odometry is in world frame i.e. /odom
        #This is the UAV position in odom frame
        # This is not required as we are getting the complete pose of UAV from /ap/pose/filtered topic
        # self.create_subscription(Odometry, '/odometry', self.odom_cb, 10) 
        
        #/jackal/odom is a FIXED frame (does not move with the UGV)
        #/jackal/odom is fixed at the initial position of the UGV
        self.create_subscription(Odometry, '/jackal/jackal_velocity_controller/odom', self.ugv_pose_cb, 10) # in jackal/odom frame
        self.create_subscription(Bool, '/aruco/detected', self.aruco_detected_cb, 10)
        self.create_subscription(PoseStamped, '/aruco/pose', self.aruco_pose_cb, 10)
        self.create_subscription(TwistStamped,'/controller/cmd_vel',self.control_signal_cmd_vel_cb,10)
        
        # UAV Position subscriber
        qos_profile = QoSProfile(
                                    reliability=ReliabilityPolicy.BEST_EFFORT,
                                    history=HistoryPolicy.KEEP_LAST,
                                    depth=10
                                )
        self.create_subscription(PoseStamped, '/ap/pose/filtered', self.uav_pose_cb, qos_profile) # publishing in base_link frame
        # ============= ⭐⭐ NEW: UAV Velocity subscriber ⭐⭐ =============
        self.create_subscription(TwistStamped, '/ap/twist/filtered', self.uav_twist_cb, qos_profile)

        #----------------------------------------------------------------------------------------
        #                                              Publishers
        #----------------------------------------------------------------------------------------
    #    /absolute_pose_odometry_OR_ekf

        self.pub_absolute_pose_blended_odometry_OR_ekf = self.create_publisher(PoseStamped, '/absolute_pose_odometry_OR_ekf', 10) # publishing in  odom frame
        
        self.pub_absolute_only_odometry = self.create_publisher(PoseStamped, '/absolute_pose_odometry', 10) # publishing in  odom frame

        self.pub_absolute_only_ekf = self.create_publisher(PoseStamped, '/absolute_pose_ekf', 10) # publishing in  odom frame

        self.pred_pub = self.create_publisher(Path, '/predicted_trajectory', 10)

        self.pub_update_flag = self.create_publisher(Bool, '/ekf/update_applied', 10)

        self.pub_maha = self.create_publisher(Float32, '/ekf/mahalanobis_distance', 10)

        self.pub_debug_kinematics = self.create_publisher(PoseStamped, '/debug/transformed_aruco_pose', 10)

        # Mode publisher for visualization
        self.mode_pub = self.create_publisher(String, '/tracking_mode', 10)

        

        #----------------------------------------------------------------------------------------
        #-------------------------------------Timer Functions-------------------------------------
        #----------------------------------------------------------------------------------------
        # Less frequent but more detailed

        self.create_timer(self.dt, self._timer_wrapper)
        self.get_logger().info("AbsolutePoseEKF node started.")

    def publish_mode_status(self):
        """Publish current tracking mode"""
       
        msg = String()
        if self.ekf_active:
            msg.data = "EKF_ACTIVE"
        else:
            msg.data = "ODOMETRY_ACTIVE"
        self.mode_pub.publish(msg)

    def _timer_wrapper(self):
        try:
            self.mode_switching_logic()
        except Exception:
            self.get_logger().error(
                "Exception in ekf_predict_publish:\n" + traceback.format_exc()
            )

    def get_ugv_world_position(self):
        """Get UGV position in world frame , remember world frame is odom, using TF"""
        try:
            
            # This code is getting the transform between two different odometry frames-specifically, 
            # it's finding where jackal/base_link is located in the global odom frame
            # REmember: jackal/odom is the initial positon of the jackal in the simulation world
            # jackal/base_link is fixed on top of the UGV. SO when UGV moves jackal/base_link moves as well

            # print("I am in get_ugv_world_position")
            # 1. Get UGV position relative to its own start (jackal/base_link -> jackal/odom)
            t1 = self.tf_buffer.lookup_transform('jackal/odom', 'jackal/base_link', rclpy.time.Time())
            # print("t1",t1)
            # 2. Get UGV starting offset relative to world (jackal/odom -> odom)
            t2 = self.tf_buffer.lookup_transform('odom', 'jackal/odom', rclpy.time.Time())
            # print("t2",t2)
            # 3. Combine them: World_Pos = Offset + (Rotation_of_Offset * Local_Pos)
            # We use geometry_msgs math or simple addition if rotations are aligned

            self.ugv_position_in_odom_frame[0] = t2.transform.translation.x + t1.transform.translation.x
            self.ugv_position_in_odom_frame[1]  = t2.transform.translation.y + t1.transform.translation.y
            self.ugv_position_in_odom_frame[2]  = t2.transform.translation.z + t1.transform.translation.z
            # pos = np.array([ugv_world_x, ugv_world_y, ugv_world_z])
            # print("t1.transform.translation.x ", t1.transform.translation.x )
            # print("t2.transform.translation.x ", t2.transform.translation.x )
            # --- ORIENTATION (Combined Yaw) ---
            # Get yaw from the local movement (t1) using your function
            yaw_local = self.get_yaw_from_quat(t1.transform.rotation)
            
            # Get yaw from the starting offset (t2) using your function
            yaw_offset = self.get_yaw_from_quat(t2.transform.rotation)

            # Total Yaw = Offset Yaw + Local Yaw
            combined_yaw = yaw_offset + yaw_local
            
            # Normalize the result to stay within [-pi, pi]
            self.ugv_yaw_in_odom_frame = np.arctan2(np.sin(combined_yaw), np.cos(combined_yaw))
              

                     
            # return np.array([ugv_world_x, ugv_world_y, ugv_world_z])
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            print(f"❌❌❌❌❌[TF] Error getting transform❌❌❌❌❌: {str(e)}")
            
            # Fallback: try other frame combinations
            try:
                transform = self.tf_buffer.lookup_transform(
                    'odom',              # target frame (common odom frame)
                    'jackal/base_link',  # source frame (UGV body)
                    rclpy.time.Time()
                )
                ugv_odom_pos_local_variable = transform.transform.translation
                return np.array([ugv_odom_pos_local_variable.x, ugv_odom_pos_local_variable.y, ugv_odom_pos_local_variable.z])
            except:
                return None
   
    def uav_pose_cb(self, msg):
        # When this callback is executed:
        # Every time a message arrives on /ap/pose/filtered topic
        # Updates self.uav_position_in_odom_frame with UAV's current position (world frame)
        # Updates self.uav_yaw with UAV's heading from quaternion (world frame)
        # Used later by ekf_predict_publish() for transformations and predictions
        # Header says: frame_id: base_link (UAV body frame)
        # But coordinates are large: y: -27.095 suggests world frame
        # Ardupilot often reports pose in local frame but labels it as base_link
        # In practice: These are WORLD/NED coordinates (not body frame)
        # Conclusion: self.uav_position_in_odom_frame stores world coordinates despite confusing frame_id label
        # Despite the frame_id saying base_link, the /ap/pose/filtered topic 
        # is publishing the UAV's position in the World/Local frame i.e. /odom in this case.
        try:
            self.uav_position_in_odom_frame[0] = msg.pose.position.x
            self.uav_position_in_odom_frame[1] = msg.pose.position.y
            self.uav_position_in_odom_frame[2] = msg.pose.position.z
            # ⭐⭐ NEW: Extract UAV yaw ⭐⭐
            self.uav_yaw_in_odom_frame = self.get_yaw_from_quat(msg.pose.orientation)
        except Exception:
            self.get_logger().error("❌❌❌❌❌Exception in uav_pose_cb:❌❌❌❌❌\n" + traceback.format_exc())
   
    def control_signal_cmd_vel_cb(self,msg):
        try:
            self.vx_mpc=msg.twist.linear.x
            self.vy_mpc=msg.twist.linear.y
            self.vz_mpc=msg.twist.linear.z
            self.yaw_dot_mpc= msg.twist.angular.z
           
        except Exception:
            self.get_logger().error("Exception in mpc_cmd_vel_cb:\n" + traceback.format_exc())

    def uav_twist_cb(self, msg):
        # When this callback is executed:
        # Every time a message arrives on /ap/twist/filtered topic
        # Updates self.v_u with UAV linear velocity (body frame)
        # Updates self.omega_u with UAV angular velocity (body frame)
        # Used in EKF prediction to calculate relative motion between UAV and UGV
        # Header says: frame_id: base_link
        # Standard ROS convention: Twist is ALWAYS in child_frame/body frame
        # Linear velocity: In UAV body frame (x-forward, y-left, z-up)
        # Angular velocity: In UAV body frame (roll-x, pitch-y, yaw-z)
        # Both self.v_u and self.omega_u are in UAV BODY FRAME 
        #❌❌❌❌❌ NOW THERE IS A CONFUSION, need to verify whether this velocity is in odom 
        # frame or base_link frame, since the position obtained from /ap/pose/filtered is in base_link frame
        # Despite the frame_id saying base_link, the /ap/pose/filtered topic 
        # is publishing the UAV's position in the World/Local frame i.e. /odom in this case.
        try:
            self.uav_lin_vel_wrt_odom_frame_expressed_in_base_link = np.array([
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.linear.z
            ])
            self.uav_ang_vel_wrt_odom_frame_expressed_in_base_link = np.array([
                msg.twist.angular.x,
                msg.twist.angular.y,
                msg.twist.angular.z
            ])
        except Exception:
            self.get_logger().error("Exception in uav_twist_cb:\n" + traceback.format_exc())


    def ugv_pose_cb(self, msg):
        # """Handle UGV odometry messages"""
        try:
            # self.create_subscription(Odometry, '/jackal/jackal_velocity_controller/odom', self.ugv_pose_cb, 10) 
            # in jackal/odom frame
            # Extract UGV orientation from /jackal/odom  (existing)
            #/jackal/odom is a FIXED frame (does not move with the UGV)
            #/jackal/odom is fixed at the initial position of the UGV
            #/jackal/base_link moves with the UGV(jackal)

            self.ugv_lin_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ])
            self.ugv_ang_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ])
            q = msg.pose.pose.orientation
            self.roll_ugv_in_jackal_odom_frame, self.pitch_ugv_in_jackal_odom_frame, self.ugv_yaw_in_jackal_odom_frame = self.quat_to_rpy(q)
            
            # ============= ⭐⭐ GET WORLD POSITION VIA TF ⭐⭐ =============
            # ugv_world_pos is in the /odom frame
            # ugv_world_pos = self.get_ugv_world_position()

            # self.ugv_position_in_odom_frame=self.get_ugv_world_position()
            #  When the above function is called all the data will be automatically stored in class variables:

            self.get_ugv_world_position()

           
            if not self.ekf_active:
                self.run_odometry_mode()
                
        except Exception as e:
            print(f"[ERROR] ugv_pose_cb: {str(e)}")
            traceback.print_exc()
       
    def aruco_detected_cb(self, msg):
    # Get the new detection status
        new_detected = bool(msg.data)
        
        # ============= ⭐⭐ NEW: Mode switching logic ⭐⭐ =============
        if new_detected:
            self.aruco_lost_count = 0
            
            if not self.ekf_active:
                # ArUco detected - switch to EKF mode
                # print("[MODE] ArUco detected - switching to EKF mode")
                self.ekf_active = True
                self.initialize_ekf_from_odometry()
        else:
            # No detection
            self.aruco_lost_count += 1
            
            if self.aruco_lost_count > self.aruco_lost_threshold:
                if self.ekf_active:
                    # Lost ArUco for too long - switch to odometry mode
                    # print("[MODE] ArUco lost - switching to odometry mode")
                    self.ekf_active = False
        
        # Existing code
        if self.aruco_detected and not new_detected:
            print("[EKF] Detection lost, clearing stale measurement")
            self.aruco_pose_measurement = None
            
        self.aruco_detected = new_detected
        print(f"[EKF] Detection status: {self.aruco_detected}")
        
        # Publish current mode
        self.publish_mode_status()

    def initialize_ekf_from_odometry(self):
        # """Initialize EKF state with current odometry estimate"""
        try:
            print(f"[MODE] Initializing EKF from odometry...")
            
            # Check if we have the required data
            if not hasattr(self, 'ugv_position_in_odom_frame') or self.ugv_position_in_odom_frame is None:
                print("[MODE] ERROR: No UGV odometry position available")
                return
                
            if not hasattr(self, 'uav_position_in_odom_frame') or self.uav_position_in_odom_frame is None:
                print("[MODE] ERROR: No UAV position available")
                return
            
            # self.uav_position_in_odom_frame is in world (odom) frame, Despite the frame_id saying base_link, the /ap/pose/filtered topic 
            # is publishing the UAV's position in the World/Local frame i.e. /odom in this case.
            print(f"[MODE] UAV pos: {self.uav_position_in_odom_frame}, UGV pos: {self.ugv_position_in_odom_frame}")
            
                      
            rel_pos_world = self.ugv_position_in_odom_frame - self.uav_position_in_odom_frame
            # Yes, rel_pos_world is the relative position vector of the UGV with respect to the UAV, expressed in the World (Odom) frame.
            print(f"[MODE] Relative world: {rel_pos_world}")
            
            # Transform to UAV body frame
            #according to the detailed analysis i have done 
            # self.uav_position_in_odom_frame is in world (odom) frame, Despite the frame_id saying base_link, the /ap/pose/filtered topic 
            # is publishing the UAV's position in the World/Local frame i.e. /odom in this case. 
            #SImilarly the uav_yaw is in the /odom frame
            # If self.uav_yaw is in the /odom frame, this matrix converts coordinates from the UAV's 
            # local frame to the Global Odom frame (or vice versa, depending on how you multiply it).
            R_uav = np.array([
                [np.cos(self.uav_yaw_in_odom_frame), -np.sin(self.uav_yaw_in_odom_frame), 0],
                [np.sin(self.uav_yaw_in_odom_frame), np.cos(self.uav_yaw_in_odom_frame), 0],
                [0, 0, 1]
            ])
            
            # Yes, rel_pos_body is exactly the position of the UGV expressed in the UAV's body frame (Front, Left, Up).
            rel_pos_body = R_uav.T @ rel_pos_world.reshape(3, 1)
            print(f"[MODE] Relative body: {rel_pos_body.flatten()}")
            
            # Initialize EKF state with odometry estimate
            self.ugv_pos_and_orient_in_UAV_frame[0:3, 0] = rel_pos_body.flatten()
            self.ugv_pos_and_orient_in_UAV_frame[3:6, 0] = np.array([0.0, 0.0, 0.0])  # Zero relative orientation
            
            # Reset covariance to higher values so it snaps to truth instantly
            self.P = np.eye(6) * 0.5  # Changed from 0.5 to 5.0
            
            print(f"[MODE] EKF initialized: state={self.ugv_pos_and_orient_in_UAV_frame.flatten()}")
            print(f"[MODE] EKF covariance diag: {np.diag(self.P)}")
            
        except Exception as e:
            print(f"[MODE] Error initializing EKF: {str(e)}")
            traceback.print_exc()

    def aruco_pose_cb(self, msg):
                
        # Check if measurement is NaN (marker lost)
        if (math.isnan(msg.pose.position.x) or 
            math.isnan(msg.pose.position.y) or 
            math.isnan(msg.pose.position.z)):
            # print("[EKF] Received NaN measurement - marker lost")
            self.aruco_pose_measurement = None  # Clear measurement
        else:
            # Valid measurement
            self.aruco_pose_measurement = msg
            # print(f"[EKF] Received valid ArUco measurement: x={msg.pose.position.x:.3f}")

    # Main EKF procedure
    def mode_switching_logic(self):
        """Main timer function - decides which mode to run"""
        try:
            # Always publish mode status
            self.publish_mode_status()
            
            # MODE SWITCHING
            if self.ekf_active:
                self.run_ekf_mode()
            else:
                self.run_odometry_mode()
                
        except Exception as e:
            print(f"[MAIN] Error: {str(e)}")
            traceback.print_exc()

    def run_odometry_mode(self):
        # """Run odometry-based tracking"""
        # print("[ODOM] Running odometry mode")
        # I will just update the /absolute_pose_odometry topic with the UGV's world position (not relative to UAV) for visualization
        # This will show the UGV's position in the world (odom) frame
        #  self.pub_absolute_only_odometry = self.create_publisher(PoseStamped, '/absolute_pose_odometry', 10) # publishing in  odom frame



        # Check if we have data
        if self.ugv_position_in_odom_frame is None or self.uav_position_in_odom_frame is None:
            self.get_ugv_world_position()
            print("[ODOM] Waiting for data...")
            # self.publish_safe_pose()
            # return
               
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'  # World frame
        
        # Position: arrow starts at UAV origin
        msg.pose.position.x = self.ugv_position_in_odom_frame[0]  # UGV world X
        msg.pose.position.y = self.ugv_position_in_odom_frame[1]  # UGV world Y
        msg.pose.position.z = self.ugv_position_in_odom_frame[2]  # UGV world Z
        
        qx, qy, qz, qw = self.get_quat_from_rpy(0.0,0.0,self.ugv_yaw_in_odom_frame) # UGV yaw in world frame, this will make the arrow point in the direction of UGV's heading in the world frame
        # Orientation: points toward UGV
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        
        # Publish to the same topic NMPC uses
       
        # publishing on the topic  /absolute_pose_odometry in /odom frame, 
        # this will be used in Fuzzy Logic Controller to compute the tracking error
        # between the UAV (position in odom frame) and that of UGV position in odom i.e. world frame
        self.pub_absolute_only_odometry.publish(msg) 

        #  self.pub_absolute = self.create_publisher(PoseStamped, '/absolute_pose_odometry_OR_ekf', 10) 
        #  publishing in  odom frame
        self.pub_absolute_pose_blended_odometry_OR_ekf.publish(msg) # publish on the topic /absolute_pose_odometry_OR_ekf in /odom frame, this will be used for visualization and comparison with EKF-based absolute pose


        # Store the odometry-based relative pose
        # This is continuously updated and used as the fallback estimate
        self.last_odom_pose = msg
        self.publish_bumpless_pose()

        # print(f"[ODOM] Published relative pose: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
        # print(f"[ODOM] Angles: yaw={np.degrees(yaw_to_ugv):.1f}°, pitch={np.degrees(pitch_to_ugv):.1f}°")
        # 1️⃣ Predict UGV trajectory in UAV frame
        # THis will update the 'self.trajectory' variable
        self.predict_ugv_trajectory_uav_frame_using_Odometry(
        # self.x is UGV position relative to UAV (in UAV body frame)
        # self.x[3:6, 0] = UGV orientation relative to UAV (roll, pitch, yaw)
        self.ugv_pos_and_orient_in_UAV_frame.flatten(), # self.x is UGV position relative to UAV (in UAV body frame)
        self.uav_position_in_odom_frame, 
        self.uav_yaw_in_odom_frame,
        self.ugv_lin_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link, 
        self.ugv_ang_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link,  
        self.uav_lin_vel_wrt_odom_frame_expressed_in_base_link, 
        self.uav_ang_vel_wrt_odom_frame_expressed_in_base_link,
        self.pred_N, 
        self.dt
        )
            
 


    def run_ekf_mode(self):
        dt = self.dt
    
        # ------------------------------------------------------------------
        # 1. PREDICTION STEP (Purely Linear Inertial Frame Math)
        # ------------------------------------------------------------------
        # Define the 3D Rotation Matrix from UGV body to World (Odom) frame
        R_ugv_to_odom = np.array([
            [np.cos(self.ugv_yaw_in_jackal_odom_frame), -np.sin(self.ugv_yaw_in_jackal_odom_frame), 0],
            [np.sin(self.ugv_yaw_in_jackal_odom_frame),  np.cos(self.ugv_yaw_in_jackal_odom_frame), 0],
            [0, 0, 1]
        ])
        
        # Transform UGV body velocity to absolute World velocity
        v_g_body = self.ugv_lin_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link.flatten()
        v_g_world = R_ugv_to_odom @ v_g_body
        
        # Simple Linear Prediction: State is UGV absolute position in world frame
        # self.ugv_pos_and_orient_in_UAV_frame must be initialized as a 3x1 or 6x1 absolute world vector
        self.ugv_pos_and_orient_in_UAV_frame[0:3, 0] += v_g_world * dt

        # Fully Linear System Matrix: F = Identity (since dx/dt = v)
        F = np.eye(6) 
        
        # Propagate covariance smoothly
        self.P = F @ self.P @ F.T + self.Q

        # ------------------------------------------------------------------
        # 2. UPDATE STEP (Transform Measurement to World Frame)
        # ------------------------------------------------------------------
        update_applied = False
        maha_value = -1.0

        if self.aruco_detected and (self.aruco_pose_measurement is not None):
            # 3D Rotation Matrix from UAV body to World (Odom) frame
            R_uav_to_odom = np.array([
                [np.cos(self.uav_yaw_in_odom_frame), -np.sin(self.uav_yaw_in_odom_frame), 0],
                [np.sin(self.uav_yaw_in_odom_frame),  np.cos(self.uav_yaw_in_odom_frame), 0],
                [0, 0, 1]
            ])
            
            # Relative vector from UAV to UGV in UAV body frame (from ArUco)
            p_rel_uav_frame = np.array([
                self.aruco_pose_measurement.pose.position.x,
                self.aruco_pose_measurement.pose.position.y,
                self.aruco_pose_measurement.pose.position.z
            ])
            
            # Transform the relative measurement into the absolute world frame
            p_rel_world_frame = R_uav_to_odom @ p_rel_uav_frame
            
            # Calculate expected absolute UGV position based on ArUco:
            # UGV_world = UAV_world + Relative_Vector_world
            z_x = self.uav_position_in_odom_frame[0] + p_rel_world_frame[0]
            z_y = self.uav_position_in_odom_frame[1] + p_rel_world_frame[1]
            z_z = self.uav_position_in_odom_frame[2] + p_rel_world_frame[2]
            
            # For debugging purposes, let's publish the transformed ArUco measurement in the world frame to visualize where the EKF thinks the UGV is based on the ArUco measurement alone (without EKF fusion)
            # Pack into a PoseStamped message mapped directly to the 'odom' frame
            debug_msg = PoseStamped()
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = 'odom'
            
            debug_msg.pose.position.x = float(z_x)
            debug_msg.pose.position.y = float(z_y)
            debug_msg.pose.position.z = float(z_z)
            
            # Forward the orientation configuration directly for reference mapping
            # debug_msg.pose.orientation = self.aruco_pose_measurement.pose.orientation
            debug_msg.pose.orientation.x = self.aruco_pose_measurement.pose.orientation.x
            debug_msg.pose.orientation.y = self.aruco_pose_measurement.pose.orientation.y
            debug_msg.pose.orientation.z = self.aruco_pose_measurement.pose.orientation.z
            debug_msg.pose.orientation.w = self.aruco_pose_measurement.pose.orientation.w


            # Broadcast to the offline analysis pipeline
            # pub_debug_kinematics is a publisher that sends the transformed ArUco measurement 
            # in the world frame to a topic named '/debug/transformed_aruco_pose'. 
            # This allows us to visualize where the EKF thinks the UGV is based solely 
            # on the ArUco measurement, without any EKF fusion. 
            # By comparing this debug visualization with the EKF's final estimate, 
            # we can gain insights into how much the EKF is correcting the raw
            #  ArUco measurement and whether there are any systematic biases or errors 
            # in the measurement transformation process.
            
            self.pub_debug_kinematics.publish(debug_msg)
            print("I AM PRINTING DEBUG MSG:", debug_msg)



            q = self.aruco_pose_measurement.pose.orientation
            meas_roll, meas_pitch, meas_yaw = self.quat_to_rpy(q)
            
            # Create our absolute measurement vector
            z = np.array([[z_x], [z_y], [z_z], [meas_roll], [meas_pitch], [meas_yaw]])
            
            # Innovation (Difference in absolute coordinates)
            y = z - self.ugv_pos_and_orient_in_UAV_frame
            for i in range(3, 6):
                y[i, 0] = math.atan2(math.sin(y[i, 0]), math.cos(y[i, 0]))
                
            S = self.P + self.R
            Sinv = np.linalg.pinv(S)
            maha_value = float((y.T @ Sinv @ y)[0, 0])
            
            if maha_value <= self.mahalanobis_threshold:
                K = self.P @ Sinv
                self.ugv_pos_and_orient_in_UAV_frame += K @ y
                self.P = (np.eye(6) - K) @ self.P
                update_applied = True
        # ------------------------------------------------------------------
        # 3. ROS 2 DIAGNOSTICS & PUBLISHING
        # ------------------------------------------------------------------
        self.pub_maha.publish(Float32(data=maha_value))
        self.pub_update_flag.publish(Bool(data=update_applied))
        
        # State is already absolute world frame coordinates! Directly construct message:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        
        msg.pose.position.x = float(self.ugv_pos_and_orient_in_UAV_frame[0, 0])
        msg.pose.position.y = float(self.ugv_pos_and_orient_in_UAV_frame[1, 0])
        msg.pose.position.z = float(self.ugv_pos_and_orient_in_UAV_frame[2, 0])
        
        qx, qy, qz, qw = self.rpy_to_quat(0.0, 0.0, self.ugv_yaw_in_odom_frame)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        
        self.pub_absolute_only_ekf.publish(msg)
        # Store the EKF pose so it can be used later for smooth blending
        # IMPORTANT: do NOT directly publish this to MPC
        self.last_ekf_pose = msg
        self.publish_bumpless_pose()

        # --- Trajectory prediction for EKF---
        # ============= ⭐⭐ CRITICAL FIX: Trajectory prediction ⭐⭐ =============
        # Predict UAV trajectory in WORLD frame (no this is wrong)
        # ⚠️  WARNING:This is incorrect
        # UAV trajectory prediction mjust be in UAV frame not in world frame
        # The predicted UGV trajectory must be in the UAV's body frame for the 
        # NMPC to work correctly, since the NMPC controller needs the relative position error 
        # to compute tracking control actions.
        
        # self.predict_ugv_trajectory_uav_frame_using_EKF(
        #     # self.x is UGV position relative to UAV (in UAV body frame)
        #     # self.x[3:6, 0] = UGV orientation relative to UAV (roll, pitch, yaw)
        #     self.ugv_pos_and_orient_in_UAV_frame.flatten(), # self.x is UGV position relative to UAV (in UAV body frame)
        #     self.uav_position_in_odom_frame, self.uav_yaw_in_odom_frame,
        #     self.ugv_lin_vel_in_jackal_odom_frame, self.ugv_ang_vel_in_jackal_odom_frame,
        #     self.uav_lin_vel_in_odom_frame, self.uav_ang_vel_in_odom_frame,
        #     N=self.pred_N, dt=self.dt
        # )
        self.predict_ugv_trajectory_uav_frame_using_Odometry(
        # self.x is UGV position relative to UAV (in UAV body frame)
        # self.x[3:6, 0] = UGV orientation relative to UAV (roll, pitch, yaw)
        self.ugv_pos_and_orient_in_UAV_frame.flatten(), # self.x is UGV position relative to UAV (in UAV body frame)
        self.uav_position_in_odom_frame, 
        self.uav_yaw_in_odom_frame,
        self.ugv_lin_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link, 
        self.ugv_ang_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link,  
        self.uav_lin_vel_wrt_odom_frame_expressed_in_base_link, 
        self.uav_ang_vel_wrt_odom_frame_expressed_in_base_link,
        self.pred_N, 
        self.dt
        )

   
    def publish_bumpless_pose(self):
        """
        Publishes a smooth, continuous pose estimate to MPC by blending
        odometry and EKF estimates instead of hard-switching between them.
        """

        # --------------------------------------------------------------------------
        # Safety check: if odometry has never been received, we cannot publish
        # Odometry is treated as the baseline estimate
        # --------------------------------------------------------------------------
        if self.last_odom_pose is None:
            return

        # --------------------------------------------------------------------------
        # Detect mode transition (EKF <-> Odometry)
        # If the EKF active flag changed since last cycle, a transition occurred
        # --------------------------------------------------------------------------
        if self.ekf_active != self.prev_mode_ekf:
            # Mark that a mode transition is occurring (useful for debugging/logging)
            self.mode_transition_active = True

            # Update stored mode for next iteration
            self.prev_mode_ekf = self.ekf_active

        # --------------------------------------------------------------------------
        # Define the target blending value based on current mode
        # EKF active   → target_alpha = 1.0
        # EKF inactive → target_alpha = 0.0
        # --------------------------------------------------------------------------
        target_alpha = 1.0 if self.ekf_active else 0.0

        # --------------------------------------------------------------------------
        # Smoothly move blend_alpha toward target_alpha
        # This implements a first-order low-pass filter (exponential smoothing)
        # --------------------------------------------------------------------------
        self.blend_alpha += self.blend_rate * (target_alpha - self.blend_alpha)

        # Clamp blend_alpha to valid range [0, 1]
        self.blend_alpha = np.clip(self.blend_alpha, 0.0, 1.0)

        # --------------------------------------------------------------------------
        # If EKF pose is not yet available, fall back completely to odometry
        # This prevents invalid blending during EKF startup
        # --------------------------------------------------------------------------
        if self.last_ekf_pose is None:
            blended_pose = self.last_odom_pose

        else:
            # ----------------------------------------------------------------------
            # Create a new PoseStamped message for the blended output
            # ----------------------------------------------------------------------
            blended_pose = PoseStamped()

            # Use odometry header for consistency (frame_id + timestamp)
            blended_pose.header = self.last_odom_pose.header

            # ----------------------------------------------------------------------
            # Position blending (linear interpolation)
            # This guarantees continuity even if odometry drifted
            # ----------------------------------------------------------------------
            blended_pose.pose.position.x = (
                (1.0 - self.blend_alpha) * self.last_odom_pose.pose.position.x +
                self.blend_alpha * self.last_ekf_pose.pose.position.x
            )

            blended_pose.pose.position.y = (
                (1.0 - self.blend_alpha) * self.last_odom_pose.pose.position.y +
                self.blend_alpha * self.last_ekf_pose.pose.position.y
            )

            blended_pose.pose.position.z = (
                (1.0 - self.blend_alpha) * self.last_odom_pose.pose.position.z +
                self.blend_alpha * self.last_ekf_pose.pose.position.z
            )

            # ----------------------------------------------------------------------
            # Orientation handling
            # For safety and simplicity, we select orientation based on dominance
            # (avoids quaternion interpolation instability)
            # ----------------------------------------------------------------------
            if self.blend_alpha > 0.5:
                blended_pose.pose.orientation = self.last_ekf_pose.pose.orientation
            else:
                blended_pose.pose.orientation = self.last_odom_pose.pose.orientation

        # --------------------------------------------------------------------------
        # Publish the final smooth pose to MPC
        # MPC/FLC will always subscribes to this topic
        # --------------------------------------------------------------------------
        # self.pub_rel.publish(blended_pose)   # publishing on the topic  /relative_pose_odom_OR_ekf in /base_link frame
        #  self.pub_absolute_pose_blended_odometry_OR_ekf = self.create_publisher(PoseStamped, '/absolute_pose_odometry_OR_ekf', 10) # publishing in  odom frame
        self.pub_absolute_pose_blended_odometry_OR_ekf.publish(blended_pose) # publish on the topic /absolute_pose_odometry_OR_ekf in /odom frame, this will be used in the Fuzzy LOgic Controller
   
      
    def predict_ugv_trajectory_uav_frame_using_Odometry(
            self,
            state_vec,
            uav_pos,
            uav_yaw,
            v_g,
            w_g,
            v_u,     
            w_u,     
                     
            N,           # Prediction horizon length (number of steps)
            dt=0.02):    # Discrete-time step (seconds)
        """
        Predict desired UAV trajectory for NMPC to track UGV.
        All positions are expressed in the UAV body frame (no world-frame conversion).

        Parameters:
            state_vec : np.array
                Current relative state of UGV w.r.t UAV [x, y, z, roll, pitch, yaw].
            uav_pos : np.array
                UAV linear velocity in odom frame [vx, vy].
            uav_yaw : float
                UAV yaw rate (rad/s).
            v_g, w_g : np.array / float
                UGV linear and angular velocities in jacakl/odom frame. ()
            v_u, w_u : np.array / float
                UAV linear and angular velocities in odom frame.
            N : int
                Prediction horizon.
            dt : float
                Time step (s).

        Returns:
            uav_trajectory : list of np.array
                List of UAV positions [x, y, z] in UAV frame."""
        
                # 1️⃣ Predict UGV trajectory in UAV frame
        # # THis will update the 'self.trajectory' variable
        # self.predict_ugv_trajectory_uav_frame_using_Odometry(
        # self.ugv_pos_and_orient_in_UAV_frame.flatten(), # self.x is UGV position relative to UAV (in UAV body frame)
        # self.uav_position_in_odom_frame, self.uav_yaw,
        # self.ugv_lin_vel_in_jackal_odom_frame, self.ugv_ang_vel_in_jackal_odom_frame,  
        # self.uav_lin_vel_in_odom_frame, self.uav_ang_vel_in_odom_frame,
        # self.pred_N, self.dt
        # )
        #------------------------------------------------------------------------------------
        #             Trajectrory of UGV generated by UGV Odometry
        #-----------------------------------------------------------------------------------
        # the specific code that generates the trajectory from odometry data
        # the desired UAV trajectory should be in the UAV frame for the NMPC to function correctly
        # For the NMPC to work correctly, you should feed the UGV trajectory 
        # (expressed in the UAV frame) directly to the NMPC as the reference.
        # You do NOT convert the predicted UGV trajectory into a UAV trajectory.


        # ------------------------------------------------------------
        # Extract the current relative position of the UGV w.r.t UAV
        # from the EKF/odometry state vector.
        #
        # state_vec[0:3] represents:
        #   x_rel : UGV x-position relative to UAV (UAV frame)
        #   y_rel : UGV y-position relative to UAV (UAV frame)
        #   z_rel : UGV z-position relative to UAV (UAV frame)
        # ------------------------------------------------------------
        
        x_rel=self.ugv_pos_and_orient_in_UAV_frame[0,0]
        y_rel=self.ugv_pos_and_orient_in_UAV_frame[1,0]
        z_rel=self.ugv_pos_and_orient_in_UAV_frame[2,0]

        roll=self.ugv_pos_and_orient_in_UAV_frame[3,0]
        pitch=self.ugv_pos_and_orient_in_UAV_frame[4,0]
        yaw=self.ugv_pos_and_orient_in_UAV_frame[5,0]
        qx, qy, qz, qw = self.rpy_to_quat(roll, pitch, yaw) 

        # ------------------------------------------------------------
        # Form a 2D relative position vector for planar motion.
        # This vector represents where the UGV currently appears
        # from the UAV’s body frame.
        # ------------------------------------------------------------
        p_rel = np.array([x_rel, y_rel], dtype=float)

        # ------------------------------------------------------------
        # Initialize a list to store the predicted relative positions
        # of the UGV over the prediction horizon.
        # Each element will be a 3D point [x_rel, y_rel, z_rel].
        # ------------------------------------------------------------
        self.trajectory = []

        # 2️⃣ Define desired offset from UGV (in UAV frame)
        # Better Strategy: If you find you keep losing the marker, try an offset of [-1.0, 0.0, -2.0].
        #  Being 1 meter behind gives the camera a better "look ahead" angle at the UGV.
        desired_offset = np.array([0.0, 0.0, 0.0])  # negative x is "behind" in UAV frame

        # 1. Get the Jackal's velocities (already in /jackal/odom which is ENU)
        v_gx_world = self.ugv_lin_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link[0]
        v_gy_world = self.ugv_lin_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link[1]

        # 2. Get the UAV's current yaw (this must be its yaw in the 'odom' frame)
        uav_yaw = self.uav_yaw_in_odom_frame 
        cos_y = np.cos(uav_yaw)
        sin_y = np.sin(uav_yaw)

        # 3. Rotate World Velocity into UAV Body Frame (FLU)
        # This converts ENU (North/East) to FLU (Front/Left)
        v_gx_flu =  v_gx_world * cos_y + v_gy_world * sin_y
        v_gy_flu = -v_gx_world * sin_y + v_gy_world * cos_y

        # v_g is now ready for your dx/dy calculation
        v_g = np.array([v_gx_flu, v_gy_flu])

        # 4. Angular velocity (Yaw rate)
        # Since both frames share the same Z-up axis, the scalar yaw rate 
        # is the same in both frames.
        w_g = self.ugv_ang_vel_wrt_jackal_odom_frame_expressed_in_jackal_base_link[2]

        v_ux=self.uav_lin_vel_wrt_odom_frame_expressed_in_base_link[0]
        v_uy=self.uav_lin_vel_wrt_odom_frame_expressed_in_base_link[1]
        # This turns the [x, y, z] vector into a single float (yaw rate)
        scalar_wu = float(self.uav_ang_vel_wrt_odom_frame_expressed_in_base_link[2])
        for i in range(N):
           
            
            # Force p_rel elements and w_u to be pure scalars
            curr_px = float(p_rel[0])
            curr_py = float(p_rel[1])
            

            # 1. Predict next UGV relative position (Kinematic Model)
            dx = v_gx_flu - v_ux + (scalar_wu * curr_py)
            dy = v_gy_flu - v_uy - (scalar_wu * curr_px)
                    
            
            # Now these additions will work because dx and dy are single floats
            p_rel[0] += dx * dt
            p_rel[1] += dy * dt

            # 2. Calculate desired UAV position
            ugv_curr_rel = np.array([p_rel[0], p_rel[1], z_rel], dtype=float)
            desired_uav_pos = ugv_curr_rel + desired_offset
            
            self.trajectory.append(desired_uav_pos)
   
        

        # ------------------------------------------------------------
        # Publish the predicted UGV trajectory expressed entirely
        # in the UAV body frame.
        # This trajectory is used directly by the NMPC.
        # ------------------------------------------------------------
        # Publish predicted path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
            # ⚠️  WARNING:This is incorrect, the frame_id should be base_link
        path_msg.header.frame_id = 'base_link'

        for pos in self.trajectory:  
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(pos[0])
            ps.pose.position.y = float(pos[1])
            ps.pose.position.z = float(pos[2])
            
            # For NMPC, orientation might not matter, but set to identity
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.pred_pub.publish(path_msg)
       

      
    def predict_ugv_trajectory_uav_frame_using_EKF(
            self, state_vec, uav_pos, uav_yaw,
            v_g, w_g, v_u, w_u, N, dt):
        """
        Predict desired UAV trajectory for NMPC to track UGV.
        All positions are expressed in the UAV body frame (no world-frame conversion).

        Parameters:
            state_vec : np.array
                Current relative state of UGV w.r.t UAV [x, y, z, roll, pitch, yaw].
            uav_pos : np.array
                UAV linear velocity in UAV frame [vx, vy].
            uav_yaw : float
                UAV yaw rate (rad/s).
            v_g, w_g : np.array / float
                UGV linear and angular velocities in UAV frame.
            v_u, w_u : np.array / float
                UAV linear and angular velocities in UAV frame.
            N : int
                Prediction horizon.
            dt : float
                Time step (s).

        Returns:
            uav_trajectory : list of np.array
                List of UAV positions [x, y, z] in UAV frame.
        """
         # ============= ⭐⭐ NEW: Alternative - Predict desired UAV trajectory EKF mode ⭐⭐ =============
         #the specific code that generates the trajectory from  EKF
        # ⚠️  WARNING: This needs to be analyzed very carefully
         # ------------------------------------------------------------
        # Extract the current relative position of the UGV w.r.t UAV
        # from the EKF/odometry state vector.
        #
        # state_vec[0:3] represents:
        #   x_rel : UGV x-position relative to UAV (UAV frame)
        #   y_rel : UGV y-position relative to UAV (UAV frame)
        #   z_rel : UGV z-position relative to UAV (UAV frame)
        # ------------------------------------------------------------
        x_rel, y_rel, z_rel = state_vec[0:3]

        # ------------------------------------------------------------
        # Form a 2D relative position vector for planar motion.
        # This vector represents where the UGV currently appears
        # from the UAV’s body frame.
        # ------------------------------------------------------------
        p_rel = np.array([x_rel, y_rel], dtype=float)

        # ------------------------------------------------------------
        # Initialize a list to store the predicted relative positions
        # of the UGV over the prediction horizon.
        # Each element will be a 3D point [x_rel, y_rel, z_rel].
        # ------------------------------------------------------------
        self.trajectory = []

        # 2️⃣ Define desired offset from UGV (in UAV frame)
        # Better Strategy: If you find you keep losing the marker, try an offset of [-1.0, 0.0, -2.0].
        #  Being 1 meter behind gives the camera a better "look ahead" angle at the UGV.
        desired_offset = np.array([0.0, 0.0, 0.0])  # negative x is "behind" in UAV frame



        # Ensure velocities are 1D arrays [vx, vy] regardless of input shape
        v_g_flat = np.array(v_g).flatten()
        v_u_flat = np.array(v_u).flatten()

        for i in range(N):
            # Extract scalars using .item() to guarantee we don't have arrays
            v_gx = float(np.array(v_g).flatten()[0])
            v_ux = float(np.array(v_u).flatten()[0])
            v_gy = float(np.array(v_g).flatten()[1])
            v_uy = float(np.array(v_u).flatten()[1])
            
            # Force p_rel elements and w_u to be pure scalars
            curr_px = float(p_rel[0])
            curr_py = float(p_rel[1])
            scalar_wu = float(np.array(w_u).flatten()[0])

            # 1. Predict next UGV relative position (Kinematic Model)
            dx = v_gx - v_ux + (scalar_wu * curr_py)
            dy = v_gy - v_uy - (scalar_wu * curr_px)
            
            # Now these additions will work because dx and dy are single floats
            p_rel[0] += dx * dt
            p_rel[1] += dy * dt

            # 2. Calculate desired UAV position
            ugv_curr_rel = np.array([p_rel[0], p_rel[1], z_rel], dtype=float)
            desired_uav_pos = ugv_curr_rel + desired_offset
            
            self.trajectory.append(desired_uav_pos)




        # 4️⃣ Debug visualization
        # self.print_trajectory_debug("Predicted UGV Trajectory by EKF (UAV frame)", "base_link")
        # ------------------------------------------------------------
        # Publish the predicted UGV trajectory expressed entirely
        # in the UAV body frame.
        # This trajectory is used directly by the NMPC.
        # ------------------------------------------------------------
        # Publish predicted path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
            # ⚠️  WARNING:This is incorrect, the frame_id should be base_link
        path_msg.header.frame_id = 'base_link'

        for pos in self.trajectory:  
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(pos[0])
            ps.pose.position.y = float(pos[1])
            ps.pose.position.z = float(pos[2])
            
            # For NMPC, orientation might not matter, but set to identity
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.pred_pub.publish(path_msg)
       

    # Helper functions
    def quat_to_rpy(self, q):
        w, x, y, z = q.w, q.x, q.y, q.z
        sinr = 2*(w*x + y*z)
        cosr = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr, cosr)

        sinp = 2*(w*y - z*x)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)

        siny = 2*(w*z + x*y)
        cosy = 1 - 2*(y*y + z*z)
        yaw = np.arctan2(siny, cosy)

        return roll, pitch, yaw

    def rpy_to_quat(self, roll, pitch, yaw):
        cy = np.cos(yaw*0.5); sy = np.sin(yaw*0.5)
        cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
        cr = np.cos(roll*0.5); sr = np.sin(roll*0.5)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        return (qx, qy, qz, qw)
    
    # ------------ helpers ------------
    def rpy_to_rot(self,roll, pitch, yaw):
        cr = np.cos(roll); sr = np.sin(roll)
        cp = np.cos(pitch); sp = np.sin(pitch)
        cy = np.cos(yaw); sy = np.sin(yaw)

        R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [-sp,   cp*sr,            cp*cr]])
        return R

    # ============= ⭐⭐ NEW FUNCTION ⭐⭐ =============
    def get_yaw_from_quat(self,q):
        """Extract yaw from quaternion"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        
        return np.arctan2(siny_cosp, cosy_cosp)
    # ==============================================

    def get_quat_from_rpy( self,roll, pitch, yaw):
        """
        Convert Roll, Pitch, Yaw (in radians) to a Quaternion
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        q_w = cr * cp * cy + sr * sp * sy
        q_x = sr * cp * cy - cr * sp * sy
        q_y = cr * sp * cy + sr * cp * sy
        q_z = cr * cp * sy - sr * sp * cy

        return [q_x, q_y, q_z, q_w]


   
  
def main(args=None):
    rclpy.init(args=args)
    node = AbsolutePoseEKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


#------------------------------------------------------------------------------------------
#  100% Verfied information from mutiple sources
#------------------------------------------------------------------------------------------
# In ROS 2 and ArduPilot (when integrated via MAVROS or similar bridges), the coordinate systems follow specific standards that can be confusing because they often mix World and Body conventions.
# 1. The Global Frame (odom / map)

# Standard ROS 2 environments use the ENU (East-North-Up) convention for global frames.

#     X-axis: East

#     Y-axis: North

#     Z-axis: Up

# 2. The Body Frame (base_link)

# The base_link of your UAV follows the FLU (Front-Left-Up) convention. This is the standard for nearly all ROS-based robots:

#     X-axis (Red): Points Forward (out of the "nose" of the drone).

# Y-axis (Green): Points Left.

# Z-axis (Blue): Points Up.