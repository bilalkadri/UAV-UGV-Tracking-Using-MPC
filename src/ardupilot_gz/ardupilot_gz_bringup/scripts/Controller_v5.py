#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import traceback
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_pose
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32
from ardupilot_msgs.srv import ModeSwitch, ArmMotors
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point




# Helper math
# def rpy_to_rot(roll, pitch, yaw):
#     cr = np.cos(roll); sr = np.sin(roll)
#     cp = np.cos(pitch); sp = np.sin(pitch)
#     cy = np.cos(yaw); sy = np.sin(yaw)
#     R = np.array([
#         [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
#         [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
#         [-sp, cp*sr, cp*cr]
#     ])
#     return R

# def ang_vel_transform(roll, pitch):
#     cr = np.cos(roll); sr = np.sin(roll)
#     cp = np.cos(pitch); sp = np.sin(pitch)
#     # clamp cp to avoid divide-by-zero but keep it smooth
#     if abs(cp) < 1e-6:
#         cp = np.sign(cp) * 1e-6 if cp != 0 else 1e-6
#     T = np.array([
#         [1.0, sr*sp/cp, cr*sp/cp],
#         [0.0, cr,       -sr     ],
#         [0.0, sr/cp,    cr/cp   ]
#     ])
#     return T



def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def get_yaw_from_quat(q):
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


class Controller_for_UAV_Node(Node):
    def __init__(self, mode="Fuzzy-PID"):
        super().__init__('controller_for_uav_node')
        
        # Controller Selection: "Fuzzy-PID" or "PID"
        self.mode = mode

        #----------------------------------------------------------------------------------------
        #                         MPC parameters
        #----------------------------------------------------------------------------------------
  
        # Prediction horizon length
        # Larger horizon → MPC plans further ahead and avoids aggressive short-term corrections
        # 20 is okay, but sluggish behavior benefits from a slightly longer horizon
        self.N = 25


        # Prediction time step (seconds)
        # This is the internal prediction resolution
        # Increasing this LOWERS controller bandwidth
        # 0.1 is fast; 0.15–0.2 makes motion smoother
        self.pred_dt = 0.15


        # MPC execution period (seconds)
        # How often the MPC is solved
        # Increasing this directly slows reaction speed
        # 0.1 → aggressive, 0.15–0.2 → sluggish and stable
        self.controller_dt = 0.15


        #----------------------------------------------------------------------------------------
        # State tracking weights
        #----------------------------------------------------------------------------------------

        # Position tracking cost matrix [x, y, z]
        # High values force the UAV to correct position errors aggressively
        # These are intentionally reduced to avoid violent corrections
        self.Q_pos = np.diag([
            10.0,   # x position weight (was 10.0)
            100.0,   # y position weight (was 10.0)
            5.0    # z position weight (was 20.0)
        ])


        # Orientation (angle) tracking cost matrix [roll, pitch, yaw]
        # Lower yaw weight prevents fast spinning to correct small angular noise
        self.Q_ang = np.diag([
            1.0,   # roll
            1.0,   # pitch
            2.0    # yaw (was 10.0)
        ])


        #----------------------------------------------------------------------------------------
        # Control smoothness penalties
        #----------------------------------------------------------------------------------------

        # Penalty on change in control inputs Δu = u(k) - u(k-1)
        # THIS IS THE MOST IMPORTANT PART FOR SMOOTHNESS
        # Large values strongly discourage sudden control changes
        self.R_du = np.diag([
            200.0,  # Δvx penalty (was 100.0)
            200.0,  # Δvy penalty (was 100.0)
            80.0,   # Δvz penalty (was 15.0)
            30.0    # Δyaw_rate penalty (was 1.0)
        ])


        #----------------------------------------------------------------------------------------
        # Field-of-view / visibility constraint weight
        #----------------------------------------------------------------------------------------

        # Weight that keeps the UGV inside camera FOV
        # Lowering this avoids violent lateral motion when target is near edge
        self.W_fov = 120.0   # was 250.0


        #----------------------------------------------------------------------------------------
        # Control input limits (soft constraints via MPC)
        #----------------------------------------------------------------------------------------

        # Maximum horizontal velocity (m/s)
        # Limiting velocity prevents aggressive chase behavior
        self.v_max = 5.0     # was 20.0


        # Maximum vertical velocity (m/s)
        # Vertical motion is especially destabilizing for MPC
        self.vz_max = 0.5    # was 1.0


        # Maximum yaw rate (rad/s)
        # Large yaw rates amplify base_link noise
        self.yawdot_max = 0.25   # was 0.5


        #----------------------------------------------------------------------------------------
        #                         PID parameters
        #----------------------------------------------------------------------------------------
        # Proportional gains for [x, y, z, yaw]
        self.kp = np.array([1.5, 1.5, 1.0, 0.8])
        # Integral gains
        self.ki = np.array([0.01, 0.01, 0.01, 0.01])
        # Derivative gains
        self.kd = np.array([0.1, 0.1, 0.05, 0.05])
        
        self.error_integral = np.zeros(4)
        self.prev_error = np.zeros(4)

        #----------------------------------------------------------------------------------------
        #                         State Variables
        #----------------------------------------------------------------------------------------
        # self.ugv_pos_and_orient_in_UAV_frame = np.zeros(6)
        # self.have_rel = False
        self.ekf_active = False # Default state in Odometry mode
        
        # =====================================================================
        # 🏁 LANDING STATE MACHINE INFRASTRUCTURE
        # =====================================================================
        # Global State Definitions for standard tracking and touchdown routines
        self.STATE_TRACKING = 0
        self.STATE_DESCENT  = 1
        self.STATE_TERMINAL = 2
        self.STATE_LANDED   = 3

        # Initialize tracking variables
        self.current_landing_state = self.STATE_TRACKING
        self.alignment_start_time = None
        self.terminal_phase_start_time = None
        
        # Memory buffers to hold terminal velocities
        self.latched_vx_flu = 0.0
        self.latched_vy_flu = 0.0
        self.latched_vz_flu = 0.0
         # =====================================================================

        # state
        self.ugv_pos_and_orient_in_UAV_frame = np.zeros(6)
        self.have_rel = False

        # odometry
        self.ugv_lin_vel_in_jackal_odom_frame = np.zeros(3)
        self.ugv_ang_vel_in_jackal_odom_frame = np.zeros(3)
        self.v_u = np.zeros(3)
        self.w_u = np.zeros(3)

        self.ugv_position_in_odom_frame =  [0.0, 2.0, 0.0]   # Store UGV position by transforming data from /jacakal/base_link to /odom frame 
        self.ugv_yaw_in_odom_frame=0.0
        self.ugv_absolute_pose_in_odom_frame_BLENDED = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])

        self.ugv_lin_vel_in_jackal_odom_frame = np.zeros(3)      # UGV velocity (jacakl/odom frame),
        self.ugv_ang_vel_in_jackal_odom_frame = np.zeros(3)  # UGV angular velocity (jackal/odom frame), 


        self.uav_position_in_odom_frame = np.zeros(3)
        self.uav_yaw_in_odom_frame = 0.0  # ⭐⭐ NEW: Store UAV yaw ⭐⭐

        self.uav_lin_vel_in_odom_frame = np.zeros(3)      # UAV velocity (body frame) ⭐⭐ CHANGED ⭐⭐
        self.uav_ang_vel_in_odom_frame = np.zeros(3)  # UAV angular velocity (body frame)



        #predicted trajectory generated from EKF node
        # self.nmpc_trajectory_ref=[]
        # Client to change flight modes natively (e.g., switching to LAND mode)
        self.set_mode_client = self.create_client(ModeSwitch, '/ap/mode_switch')

        # Client to switch physical arming states natively (disarming motors)
        self.arming_client = self.create_client(ArmMotors, '/ap/arm_motors')

        
        #----------------------------------------------------------------------------------------
        #                         TF and ROS Setup
        #----------------------------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        # self.create_subscription(PoseStamped, '/relative_pose_odom_OR_ekf', self.rel_pose_odom_OR_ekf_cb, 10)
        self.create_subscription(Odometry, '/jackal/jackal_velocity_controller/odom', self.ugv_pose_cb, 10)
        # self.create_subscription(Path, '/predicted_trajectory', self.predicted_trajectory_cb, 10)
        self.create_subscription(PoseStamped, '/ap/pose/filtered', self.uav_pose_cb, qos_profile)
        self.create_subscription(TwistStamped, '/ap/twist/filtered', self.uav_twist_cb, qos_profile)
        self.create_subscription(String,'/tracking_mode',self.mode_status_cb,10)
        self.create_subscription(PoseStamped,'/absolute_pose_odometry_OR_ekf',self.absolute_pose_odom_OR_ekf_cb,10) # subscribing to the absolute pose measurement from EKF node (Aruco + Odometry)
        # self.pub_absolute_pose_blended_odometry_OR_ekf = self.create_publisher(PoseStamped, '/absolute_pose_odometry_OR_ekf', 10) # publishing in  odom frame

        # Publishers
        # self.Xref_pred_pub = self.create_publisher(Path, '/Xref_MPC_in_Odom_frame_from_predicted_trajectory', 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/controller/cmd_vel', 10)
        #self.error_pub = self.create_publisher(PointStamped, '/mpc/tracking_error', 10)
        self.tracking_error_pub = self.create_publisher( PointStamped,'/controller/tracking_error',10)
         # Create ROS2 Publisher for the landing state pulse
        # Topic type: std_msgs/msg/Int32
        self.landing_state_pub = self.create_publisher(Int32,'/controller/landing_state', 10)
        self.error_horiz_pub = self.create_publisher(Float32, '/controller/e_pos_horiz', 10)
        self.deck_distance_marker_pub = self.create_publisher(Marker, '/visuals/true_distance_beam', 10)
        # Timer Switcher
        self.create_timer(self.controller_dt, self.control_loop)
        
        self.get_logger().info(f"Controller_for_UAV_Node started in {self.mode} mode.")
    
    def send_land_command(self):
        """Commands ArduPilot to switch flight modes natively using ardupilot_msgs."""
        if not self.set_mode_client.service_is_ready():
            self.get_logger().error("❌ Native mode switch service (/ap/mode_switch) not ready!")
            return

        # 🟢 Updated to native ardupilot_msgs service request object
        req = ModeSwitch.Request()
        req.mode = 9  # Mode 9 is the standard firmware integer ID for 'LAND' in ArduPilot
        
        self.get_logger().info("🛬 Sending Native ArduPilot Flight Mode Request: [LAND Mode 9]")
        self.set_mode_client.call_async(req)

    def send_disarm_command(self):
        """Commands ArduPilot to disarm its rotors natively using ardupilot_msgs."""
        if not self.arming_client.service_is_ready():
            self.get_logger().error("❌ Native arming service (/ap/arm_motors) not ready!")
            return

        # 🟢 Updated to native ardupilot_msgs service request object
        req = ArmMotors.Request()
        req.arm = False  # False turns the motors off completely
        
        self.get_logger().warn("⚡ Sending Native ArduPilot Actuator Kill Request: [DISARM]")
        self.arming_client.call_async(req)


    def autonomous_tracking_landing_logic(self):
        """
        Terrain-resilient landing state machine that monitors tracking errors and 
        publishes states as an integer pulse (0 = Tracking, 1 = Descent).
        Protected against rugged terrain fluctuations via direct altitude differential checking.
        """
        # 1. Compute current horizontal tracking error relative to target position
        if self.ekf_active:
            dx, dy, _ = self.ugv_absolute_pose_in_odom_frame_BLENDED[:3] - self.uav_position_in_odom_frame
        else:
            dx, dy, _ = self.ugv_absolute_pose_in_odom_frame_BLENDED[:3] - self.uav_position_in_odom_frame

        # Calculate absolute 2D horizontal distance error
        e_pos_horiz = np.linalg.norm([dx, dy])
        error_msg = Float32()
        error_msg.data = float(e_pos_horiz)  # Ensure type safety as a standard float
        self.error_horiz_pub.publish(error_msg)
        
        # Get current time stamp in seconds
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Ensure class-level variables for the extended states exist
        if not hasattr(self, 'terminal_phase_start_time'):
            self.terminal_phase_start_time = None
        if not hasattr(self, 'latched_vx_flu'):
            self.latched_vx_flu = 0.0
            self.latched_vy_flu = 0.0

        # =================================================================
        # STATE MACHINE TRANSITION AND EXECUTION LOGIC
        # =================================================================
        
        # # Global State Definitions for standard tracking and touchdown routines
        # self.STATE_TRACKING = 0
        # self.STATE_DESCENT  = 1
        # self.STATE_TERMINAL = 2
        # self.STATE_LANDED   = 3


        # 🔁 STATE 0: Hover Tracking & Alignment Verification
        if self.current_landing_state == self.STATE_TRACKING:
            # Check alignment criteria based on your tuned stable error profile
            if e_pos_horiz < 0.45:
                if self.alignment_start_time is None:
                    self.alignment_start_time = current_time
                elif (current_time - self.alignment_start_time) >= 1.5:
                    self.current_landing_state = self.STATE_DESCENT
                    self.get_logger().info("🎯 Target Aligned. Initiating Exponential Descent Phase.")
            else:
                self.alignment_start_time = None  # Reset timer if drone drifts outside bounds
            
            # Execute normal tracking using standard 2.0m vertical altitude
            self.GSPIDFLC_Tracking_and_Landing_Controller(target_altitude_offset=2.0)

        # 📉 STATE 1: Controlled Exponential Descent
        elif self.current_landing_state == self.STATE_DESCENT:
            # Abort Safety: Guard against rapid vehicle acceleration or terrain drops
            if e_pos_horiz > 0.45:  # If horizontal error exceeds safe threshold during descent
                self.current_landing_state = self.STATE_TRACKING
                self.alignment_start_time = None
                self.get_logger().warn("⚠️ Tracking threshold breached! Aborting descent, climbing to safe hover.")
                return

            # 🟢 RUGGED TERRAIN PROTECTIVE LAYER
            # Read absolute UAV height from ArduPilot state and subtract current UGV floor Z elevation
            uav_z = self.uav_position_in_odom_frame[2]
            ugv_z = self.ugv_position_in_odom_frame[2]  # UGV's absolute altitude in the world/odom frame
            true_distance_to_deck = uav_z - ugv_z 
            # self.publish_distance_marker(uav_z, ugv_z)  # Visualize true distance to deck in RViz
           
           
            # If the drone hits the ground cushion (1.0m), switch to open-loop touchdown
            # This is now immune to world Z displacement.......................................s caused by hills or valleys
            if true_distance_to_deck <= 1.5:  # 1.5m threshold to detect ground cushion contact
                # 
                if e_pos_horiz <= 0.15: # Must be within 20cm of center to authorize touchdown drop
                    self.current_landing_state = self.STATE_TERMINAL
                    self.terminal_phase_start_time = current_time
                    self.get_logger().info("🛑 Precision ground cushion reached. Switching to Terminal Touchdown.")
                else:
                    # Hold current altitude and wait for horizontal alignment to recover instead of dropping blindly
                    self.get_logger().warn("⏳ Near deck but misaligned! Holding descent until centered...")
                    self.GSPIDFLC_Tracking_and_Landing_Controller(target_altitude_offset=1.5)
            else:
                # Drive height down smoothly by changing the target offset to 0.0m
                # Your fuzzy adaptive scaling will handle velocity damping naturally
                self.GSPIDFLC_Tracking_and_Landing_Controller(target_altitude_offset=0.0)

        # 🛑 STATE 2: Closed-Loop Forced Descent
        elif self.current_landing_state == self.STATE_TERMINAL: 

                # Abort Safety is ignored during terminal phase for robustness.
                uav_z = self.uav_position_in_odom_frame[2]
                ugv_z = self.ugv_position_in_odom_frame[2]  
                true_distance_to_deck = uav_z - ugv_z 

                # Triggernative native flight controller landing (State 3) when super low
                if true_distance_to_deck <= 1.5 and e_pos_horiz <= 0.2:  
                    self.current_landing_state = self.STATE_LANDED
                    self.get_logger().info("🏁 Deck contacted or imminent. Swapping native land mode.")
                else:
                    # Drive height down but slightly slower than before to maintain thrust for pitching
                    self.GSPIDFLC_Tracking_and_Landing_Controller(target_altitude_offset=0.0)

        # # 🏁 STATE 3: Vehicle Landed
        elif self.current_landing_state == self.STATE_LANDED:
                
                self.get_logger().info("🏁 Countdown complete. Sending ONE-SHOT native service requests...")
                
                # 🛑 1. Freeze local velocity tracking outputs
                self.publish_cmd([0.0, 0.0, 0.0], 0.0) 
                
                # 🛬 2. Trigger native flight controller touchdown mode (Mode 9 = LAND)
                self.send_land_command()               
                
                # ⚡ 3. Terminate rotor propulsion dynamics completely
                self.send_disarm_command()

        # =================================================================
        # 📈 STATE PULSE PUBLISHING
        # =================================================================
        state_msg = Int32()
        state_msg.data = self.current_landing_state
        self.landing_state_pub.publish(state_msg)

    def publish_distance_marker(self, uav_pos, ugv_pos):
        marker = Marker()
        marker.header.frame_id = "odom"  # Match your main world navigation frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "landing_metrics"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Define line thickness
        marker.scale.x = 0.05  # 5cm thick beam
        
        # Define color (e.g., bright orange/yellow so it stands out)
        marker.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9)
        
        # Point A: Ground UGV Center Position
        p1 = Point(x=ugv_pos[0], y=ugv_pos[1], z=ugv_pos[2])
        # Point B: Sky UAV Center Position
        p2 = Point(x=uav_pos[0], y=uav_pos[1], z=uav_pos[2])
        
        marker.points.append(p1)
        marker.points.append(p2)
        
        self.deck_distance_marker_pub.publish(marker)
    def mode_status_cb(self,msg):
        if msg.data == "ODOMETRY_ACTIVE":
            self.ekf_active=False
        elif msg.data == "EKF_ACTIVE":
            self.ekf_active=True

   
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
            self.uav_lin_vel_in_odom_frame = np.array([
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.linear.z
            ])
            self.uav_ang_vel_in_odom_frame = np.array([
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

            self.ugv_lin_vel_in_jackal_odom_frame = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ])
            self.ugv_ang_vel_in_jackal_odom_frame = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ])
            q = msg.pose.pose.orientation
            self.roll_ugv_in_jackal_odom_frame, self.pitch_ugv_in_jackal_odom_frame, self.yaw_ugv_in_jackal_odom_frame = self.quat_to_rpy_msg(q)
            
            # ============= ⭐⭐ GET WORLD POSITION VIA TF ⭐⭐ =============
            # ugv_world_pos is in the /odom frame
            # ugv_world_pos = self.get_ugv_world_position()

            # self.ugv_position_in_odom_frame=self.get_ugv_world_position()
            #  When the above function is called all the data will be automatically stored in class variables:

            self.get_ugv_world_position()

                
        except Exception as e:
            print(f"[ERROR] ugv_pose_cb: {str(e)}")
            
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
            yaw_local = get_yaw_from_quat(t1.transform.rotation)
            
            # Get yaw from the starting offset (t2) using your function
            yaw_offset = get_yaw_from_quat(t2.transform.rotation)

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
            self.uav_yaw_in_odom_frame = get_yaw_from_quat(msg.pose.orientation)
        except Exception:
            self.get_logger().error("❌❌❌❌❌Exception in uav_pose_cb:❌❌❌❌❌\n" + traceback.format_exc())
   

    def control_loop(self):
        """Timer callback that switches between Fuzzy-PID, PID """
        if not self.have_rel:
            self.get_logger().warn("No relative pose received yet", throttle_duration_sec=2.0)
            self.publish_cmd([0.0, 0.0, 0.0], 0.0)
            return

        # Controller mode selection
        if self.mode == "PID":
            self.run_pid_logic()
        elif self.mode == "Fuzzy-PID":
            # 🟢 FIX: Let the state machine take full exclusive control of publishing
            self.autonomous_tracking_landing_logic()
        else:
            self.get_logger().error(f"Invalid mode: {self.mode}")


    def GSPIDFLC_Tracking_and_Landing_Controller(self, target_altitude_offset):
        """
        Fuzzy-PID controller with anti-saturation protection and smooth gain scheduling
        this code is used for landing the UAV on UGV in the DESCENT phase 
        nce:
        """
        # --- Position Error Calculation ---
        if self.ekf_active:
            dx, dy,_ = self.ugv_absolute_pose_in_odom_frame_BLENDED[:3] - self.uav_position_in_odom_frame
        else:
            dx, dy,_ = self.ugv_absolute_pose_in_odom_frame_BLENDED[:3]- self.uav_position_in_odom_frame
        
        #I have identified a big problem
        # dz= self.ugv_absolute_pose_in_odom_frame_BLENDED[2]- self.uav_position_in_odom_frame[2]
        # dz is not the true altitude error, because the EKF is just predicting the xy-pose of the UGV
        # The z value in EKF remain ZERO (and sometimes changes to arbitrary value), so the dz calculated from 
        # the EKF blended pose is not the true altitude error between the UAV and UGV,  
        # not the altitude error relative to the UGV
     
        dz=self.ugv_position_in_odom_frame[2] - self.uav_position_in_odom_frame[2]
        dyaw = self.ugv_yaw_in_odom_frame - self.uav_yaw_in_odom_frame
        dyaw = wrap_angle(dyaw)
        
        # --- Publishing control error on the topic '/controller/tracking_error'---
        msg = PointStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"   # or "map" depending on your system

        msg.point.x = float(dx)
        msg.point.y = float(dy)
        msg.point.z = float(dz)

        self.tracking_error_pub.publish(msg)


        # current_error = np.array([dx, dy, dz - 2.0, dyaw])
        current_error = np.array([dx, dy, dz - target_altitude_offset, dyaw])
        
        e_pos_norm = np.linalg.norm(current_error[:3])
        e_yaw_abs = abs(dyaw)
        
        # =====================================================
        # ANTI-SATURATION VELOCITY LIMITS
        # =====================================================
        # Define strict limits to prevent saturation
        fuzzy_v_max = 2.5      # Keep well below actuator limits
        fuzzy_vz_max = 0.4
        fuzzy_yaw_max = 0.2
        
        # =====================================================
        # ADAPTIVE INTEGRAL WITH BACK-CALCULATION ANTI-WINDUP
        # =====================================================
        if not hasattr(self, 'fuzzy_error_integral'):
            self.fuzzy_error_integral = np.zeros(4)
        
        # Calculate desired integral increment
        integral_increment = current_error * self.controller_dt
        
        # Only integrate position errors (not yaw) when within reasonable range
        if e_pos_norm < 4.0:
            self.fuzzy_error_integral += integral_increment
        else:
            # Aggressive decay when far
            self.fuzzy_error_integral *= 0.9
        
        # Tighter anti-windup limits
        self.fuzzy_error_integral = np.clip(self.fuzzy_error_integral, -0.5, 0.5)
        
        # =====================================================
        # DERIVATIVE WITH ADAPTIVE FILTERING
        # =====================================================
        raw_derivative = (current_error - self.prev_error) / self.controller_dt
        self.prev_error = current_error.copy()
        
        if not hasattr(self, 'filtered_derivative'):
            self.filtered_derivative = np.zeros(4)
        
        # Adaptive filtering: more filtering when oscillating
        de_norm = np.linalg.norm(raw_derivative[:3])
        if de_norm > 3.0:  # High derivative = likely oscillating
            alpha_lpf = 0.1  # Heavy filtering
        elif de_norm > 1.0:
            alpha_lpf = 0.2  # Moderate filtering
        else:
            alpha_lpf = 0.3  # Normal filtering
        
        self.filtered_derivative = alpha_lpf * raw_derivative + (1.0 - alpha_lpf) * self.filtered_derivative
        
        # =====================================================
        # SMOOTH GAIN SCHEDULING WITH SATURATION PREVENTION
        # =====================================================
        
        # Base gains - moderate values
        kp_base = np.array([0.6, 0.6, 0.4, 0.25])
        ki_base = np.array([0.008, 0.008, 0.006, 0.004])
        kd_base = np.array([0.06, 0.06, 0.03, 0.02])
        
        # Calculate maximum allowed Kp to prevent saturation
        # v_max = kp_max * error_max, so kp_max = v_max / error_max
        if e_pos_norm > 0.1:
            kp_max_allowed = fuzzy_v_max / e_pos_norm
        else:
            kp_max_allowed = fuzzy_v_max / 0.1  # Prevent division by zero
        
        # Smooth gain calculation using sigmoid-like function
        # This prevents abrupt gain changes that cause oscillation
        
        
        # Normalized error (0 to 1 range)
        e_norm = np.clip(e_pos_norm / 8.0, 0.0, 1.0)  # 8m as "very far"
        
        # Smooth Kp scaling: higher when far, lower when close
        # Uses smooth transition instead of hard thresholds
        kp_scale = 0.5 + 1.0 * e_norm  # Ranges from 0.5 to 1.5 smoothly
        kp_scale = np.clip(kp_scale, 0.4, min(1.8, kp_max_allowed / np.max(kp_base[:2])))
        
        # Ki scaling: only active when close and not oscillating
        closeness_factor = np.exp(-2.0 * e_pos_norm)  # 1 when close, 0 when far
        oscillation_factor = np.exp(-de_norm / 2.0)   # 1 when smooth, 0 when oscillating
        ki_scale = closeness_factor * oscillation_factor * 1.5
        ki_scale = np.clip(ki_scale, 0.0, 1.5)
        
        # Kd scaling: higher when oscillating or moving fast
        kd_scale = 1.0 + 0.5 * (1.0 - oscillation_factor)  # Increases when oscillating
        kd_scale = np.clip(kd_scale, 0.8, 2.5)
        
        # Apply scaling with saturation check
        kp_adapt = kp_base * kp_scale
        
        # Anti-saturation: reduce Kp if command would saturate
        max_kp_effect = np.max(np.abs(kp_adapt[:2] * current_error[:2]))
        if max_kp_effect > fuzzy_v_max:
            # Scale down Kp to prevent saturation
            kp_reduction = fuzzy_v_max / max_kp_effect * 0.9  # 90% of max to stay within limits
            kp_adapt[:2] *= kp_reduction
        
        ki_adapt = ki_base * ki_scale
        kd_adapt = kd_base * kd_scale
        
        # =====================================================
        # PID OUTPUT WITH BACK-CALCULATION
        # =====================================================
        
        # Calculate preliminary output
        p_term = kp_adapt * current_error
        i_term = ki_adapt * self.fuzzy_error_integral
        d_term = kd_adapt * self.filtered_derivative
        
        output_prelim = p_term + i_term + d_term
        
        # Anti-windup: if output would saturate, reduce integral
        # scale_factor_for_z_controller=10

        # self.STATE_TRACKING = 0
        # self.STATE_DESCENT  = 1
        # self.STATE_TERMINAL = 2
        # self.STATE_LANDED   = 3

        if self.current_landing_state == self.STATE_TRACKING:
            vx_enu_prelim = output_prelim[0]
            vy_enu_prelim = output_prelim[1]
            vz_enu_prelim = 0
        # Group DESCENT and TERMINAL together since both require closed-loop XY tracking
        elif self.current_landing_state in [self.STATE_DESCENT, self.STATE_TERMINAL]:
            vx_enu_prelim = output_prelim[0]
            vy_enu_prelim = output_prelim[1]
            vz_enu_prelim = output_prelim[2]
        
        # Check for saturation
        if abs(vx_enu_prelim) > fuzzy_v_max:
            # Back-calculation: reduce integral to prevent windup
            excess = vx_enu_prelim - np.sign(vx_enu_prelim) * fuzzy_v_max
            self.fuzzy_error_integral[0] -= excess * self.controller_dt * 0.5
            output_prelim[0] = np.sign(vx_enu_prelim) * fuzzy_v_max
        
        if abs(vy_enu_prelim) > fuzzy_v_max:
            excess = vy_enu_prelim - np.sign(vy_enu_prelim) * fuzzy_v_max
            self.fuzzy_error_integral[1] -= excess * self.controller_dt * 0.5
            output_prelim[1] = np.sign(vy_enu_prelim) * fuzzy_v_max

        if abs(vz_enu_prelim) > fuzzy_v_max:
            excess = vz_enu_prelim - np.sign(vz_enu_prelim) * fuzzy_v_max
            self.fuzzy_error_integral[2] -= excess * self.controller_dt * 0.5
            output_prelim[2] = np.sign(vz_enu_prelim) * fuzzy_v_max   
        
        # Extract final commands
        vx_enu, vy_enu, vz_enu, wz_enu = output_prelim

        if self.current_landing_state == self.STATE_TRACKING:
            vz_enu = 0
        elif self.current_landing_state == self.STATE_DESCENT:
            vz_enu = output_prelim[2] 
        elif self.current_landing_state == self.STATE_TERMINAL:
            # ✅ FIX: Lower the forced rate. Restore pitch control authority.
            # -2.0m/s dumps collective thrust, killing pitching authority. 
            # -1.0m/s keeps enough thrust alive for the rear motors to force the nose down.
            vz_enu = -2.0

   
        
        # ENU → FLU rotation
        yaw = self.uav_yaw_in_odom_frame
        vx_flu =  vx_enu * np.cos(yaw) + vy_enu * np.sin(yaw)
        vy_flu = -vx_enu * np.sin(yaw) + vy_enu * np.cos(yaw)
        vz_flu =  vz_enu
        wz_flu =  wz_enu

        # 2. Apply Oscillation Detection & Damping HERE (On raw controller outputs only)
        if not hasattr(self, 'prev_vx_cmd'):
            self.prev_vx_cmd = 0.0
            self.prev_vy_cmd = 0.0
        
        # Detect sign changes (oscillation indicator)
        vx_sign_change = (vx_flu * self.prev_vx_cmd < 0) and (abs(vx_flu) > 0.5) and (abs(self.prev_vx_cmd) > 0.5)
        vy_sign_change = (vy_flu * self.prev_vy_cmd < 0) and (abs(vy_flu) > 0.5) and (abs(self.prev_vy_cmd) > 0.5)

        if vx_sign_change or vy_sign_change:
            damping_factor = 0.7  # Reduce command by 30%
            vx_flu *= damping_factor
            vy_flu *= damping_factor
            if np.random.rand() < 0.3:
                self.get_logger().warn(f"Oscillation detected! Damping applied. Vx_flu:{vx_flu:.2f} Vy_flu:{vy_flu:.2f}")

        # Store the post-damping flu velocities for the next iteration comparison        
        self.prev_vx_cmd = vx_flu
        self.prev_vy_cmd = vy_flu

        # =====================================================================
        # 🟢 MINIMAL VELOCITY FEEDFORWARD INJECTION 
        # =====================================================================
        # Inject UGV Velocity Feedforward to eliminate the long tail on the X-error
        # caused by UGV motion, add the UGV's linear velocity directly to your final outputs. 
        # Read the UGV's target velocity vector directly from your state estimations
        # WE need a feedforward term to prevent the drone from lagging behind the UGV when it is moving at a steady speed.
        # Because a standard PID controller is reactive, it cannot generate a velocity command unless an error already exists ($V = K_p \cdot e$).
        # If your UGV is traveling at, say, 1.5m/s along the $X$-axis, your UAV must maintain a persistent, steady-state $X$-error just to make the PID output match that $1.5\text{ m/s}$ speed. Every time the UGV changes speed, accelerates, or decelerates, the standard PID lags behind, creating those $2\text{ m}$ transient humps you see in the blue line.Meanwhile, 
        # because the UGV isn't moving along the $Y$-axis, the $Y$-error easily converges straight to zero.
        vx_ugv_odom = self.ugv_lin_vel_in_jackal_odom_frame[0]
        vy_ugv_odom = self.ugv_lin_vel_in_jackal_odom_frame[1]

        # Rotate the UGV's Odom frame velocities into the UAV's Forward-Left-Up (FLU) frame
        vx_ugv_ff =  vx_ugv_odom * np.cos(yaw) + vy_ugv_odom * np.sin(yaw)
        vy_ugv_ff = -vx_ugv_odom * np.sin(yaw) + vy_ugv_odom * np.cos(yaw)

        # Introduce a tuning factor (1.1 = inject 115% of target speed to force fast closing)
        k_ff = 1.15

        # Inject the feedforward term directly into your body-frame tracking outputs.
        # This makes your drone actively match target speeds without waiting for tracking errors to build up first.
        # Apply the scaling factor to the body-frame feedforward components
        vx_flu += k_ff * vx_ugv_ff
        vy_flu += k_ff * vy_ugv_ff
        
        # Final clipping (should rarely trigger now due to anti-saturation)
        vx = np.clip(vx_flu, -fuzzy_v_max, fuzzy_v_max)
        vy = np.clip(vy_flu, -fuzzy_v_max, fuzzy_v_max)
        # vz = np.clip(vz_flu, -fuzzy_vz_max, fuzzy_vz_max)

        # ✅ Open up the Z limits during terminal phase, otherwise -2.0m/s gets clipped to -0.4m/s
        if self.current_landing_state == self.STATE_TERMINAL:
            vz = np.clip(vz_flu, -2.5, 2.5)  # Allow a high speed drop
        else:
            vz = np.clip(vz_flu, -fuzzy_vz_max, fuzzy_vz_max)

        wz = np.clip(wz_flu, -fuzzy_yaw_max, fuzzy_yaw_max)
   
      
        
   

        if self.current_landing_state != 2:
            self.latched_vx_flu = vx
            self.latched_vy_flu = vy
        
        self.publish_cmd([vx, vy, vz], wz)


    def run_pid_logic(self):
        
        e_pos=self.ugv_position_in_odom_frame-self.uav_position_in_odom_frame
        
        """Simple PID controller for UAV tracking."""
        # Errors in UAV body frame (FLU)
        # dx, dy, dz = self.ugv_pos_and_orient_in_UAV_frame[0:3]



        dx, dy, dz = self.ugv_absolute_pose_in_odom_frame_BLENDED[:3] - self.uav_position_in_odom_frame
        # dx,dy,dz=self.ugv_position_in_odom_frame-self.uav_position_in_odom_frame
        dyaw = self.ugv_yaw_in_odom_frame- self.uav_yaw_in_odom_frame

        # --- Publishing control error on the topic '/controller/tracking_error'---
        msg = PointStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"   # or "map" depending on your system

        msg.point.x = float(dx)
        msg.point.y = float(dy)
        msg.point.z = float(dz)

        self.tracking_error_pub.publish(msg)


        # Target 2m above UGV
        current_error = np.array([dx, dy, dz - 2.0, dyaw])
        
        # Update Integral and Derivative
        self.error_integral += current_error * self.controller_dt
        # Clip integral to prevent windup
        self.error_integral = np.clip(self.error_integral, -1.0, 1.0)
        
        error_derivative = (current_error - self.prev_error) / self.controller_dt
        self.prev_error = current_error

        # Compute PID Output
        #SInce the error was in ENU therefore the PID output is also in ENU
        # We need to convert it into FLU 
        output = (self.kp * current_error) + (self.ki * self.error_integral) + (self.kd * error_derivative)

        vx_enu, vy_enu, vz_enu, wz_enu = output
        yaw = self.uav_yaw_in_odom_frame

        # Converting from ENU to FLU (the base_link frame of UAV is in FLU)
        # Rotate World (ENU) to Body (FLU)
        # We use the inverse rotation: 
        # vx_body =  vx_world * cos(yaw) + vy_world * sin(yaw)
        # vy_body = -vx_world * sin(yaw) + vy_world * cos(yaw)
        vx_flu =  vx_enu * np.cos(yaw) + vy_enu * np.sin(yaw)
        vy_flu = -vx_enu * np.sin(yaw) + vy_enu * np.cos(yaw)
        vz_flu =  vz_enu # Vertical velocity is frame-independent between ENU and FLU
        wz_flu=wz_enu

        # Clip to max bounds
        vx = np.clip(vx_flu, -self.v_max, self.v_max)
        vy = np.clip(vy_flu, -self.v_max, self.v_max)
        vz = np.clip(vz_flu, -self.vz_max, self.vz_max)
        wz = np.clip(wz_flu, -self.yawdot_max, self.yawdot_max)

        if np.random.rand() < 0.1: # Throttled logging
            self.get_logger().info(f"PID Mode | Error: [{dx:.2f}, {dy:.2f}, {dz:.2f}] Cmd: Vx:{vx:.2f} Cmd: Vy:{vy:.2f} Cmd: Vz:{vz:.2f}")

        self.publish_cmd([vx, vy, vz], wz)

   
   
    def absolute_pose_odom_OR_ekf_cb(self, msg: PoseStamped):
        # This callback is getting UGV position from EKF and is most critical for the control logic
        # It gives the absolute pose of the UGV in the odom frame (world coordinates)
        # This is the belnded ROS topic ('/absolute_pose_odometry_OR_ekf'), 
        # data from odometery and EKF are blended in this ROS topic
        # When the ARUCO marker is not visible then odometry data is p[opulated in this topic
        # but when the ARUCO marker is visible then the EKF estimation is populated in this topic,
        # so this topic always has the best possible estimation of the UGV's absolute pose in the world frame (odom frame)]
        
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        roll, pitch, yaw = self.quat_to_rpy_msg(msg.pose.orientation)
        self.ugv_absolute_pose_in_odom_frame_BLENDED = np.array([x, y, z, roll, pitch, yaw], dtype=float)
        self.have_rel = True

    def publish_cmd(self, v_xyz, yawdot):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id ="base_link"
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = map(float, v_xyz)
        msg.twist.angular.z = float(yawdot)
        self.cmd_pub.publish(msg)

    def quat_to_rpy_msg(self, q):
        # Implementation from your snippet...
        w,x,y,z = q.w, q.x, q.y, q.z
        sinr = 2*(w*x + y*z)
        cosr = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr, cosr)
        sinp = 2*(w*y - z*x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        siny = 2*(w*z + x*y)
        cosy = 1 - 2*(y*y + z*z)
        yaw = np.arctan2(siny, cosy)
        return roll, pitch, yaw
    
   
def main(args=None):
    rclpy.init(args=args)
    
    # Selection logic
    mode_selection = "Fuzzy-PID"  # Set to "PID" or "Fuzzy-PID"
    node = Controller_for_UAV_Node(mode=mode_selection)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()