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



# Helper math
def rpy_to_rot(roll, pitch, yaw):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw); sy = np.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R

def ang_vel_transform(roll, pitch):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    # clamp cp to avoid divide-by-zero but keep it smooth
    if abs(cp) < 1e-6:
        cp = np.sign(cp) * 1e-6 if cp != 0 else 1e-6
    T = np.array([
        [1.0, sr*sp/cp, cr*sp/cp],
        [0.0, cr,       -sr     ],
        [0.0, sr/cp,    cr/cp   ]
    ])
    return T



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
    def __init__(self, mode="MPC"):
        super().__init__('controller_for_uav_node')
        
        # Controller Selection: "MPC" or "PID"
        self.mode = mode

        #----------------------------------------------------------------------------------------
        #                         MPC parameters
        #----------------------------------------------------------------------------------------
        # self.N = 20
        # self.pred_dt = 0.1   
        # self.mpc_dt = 0.1    

        # self.Q_pos = np.diag([10.0, 10.0, 20.0]) 
        # self.Q_ang = np.diag([10.0, 10.0, 10.0])
        # self.R_du = np.diag([100.0, 100.0, 15.0, 1.0])
        # self.W_fov = 250.0
        
        # self.v_max = 20.0
        # self.vz_max = 1.0
        # self.yawdot_max = 0.5

  
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
        self.mpc_dt = 0.15


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
        
        # self.uav_position_in_odom_frame = np.zeros(3)
        # self.uav_yaw_in_odom_frame = 0.0
        # self.uav_lin_vel_in_odom_frame = np.zeros(3)
        # self.uav_ang_vel_in_odom_frame = np.zeros(3)
        
        # self.ugv_position_in_odom_frame = np.array([0.0, 2.0, 0.0])
        # self.ugv_yaw_in_odom_frame = 0.0
        # self.ugv_lin_vel_in_jackal_odom_frame = np.zeros(3)
        
      

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
        self.ugv_absolute_pose_in_odom_frame_EKF_estimation = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])

        self.ugv_lin_vel_in_jackal_odom_frame = np.zeros(3)      # UGV velocity (jacakl/odom frame),
        self.ugv_ang_vel_in_jackal_odom_frame = np.zeros(3)  # UGV angular velocity (jackal/odom frame), 


        self.uav_position_in_odom_frame = np.zeros(3)
        self.uav_yaw_in_odom_frame = 0.0  # ⭐⭐ NEW: Store UAV yaw ⭐⭐

        self.uav_lin_vel_in_odom_frame = np.zeros(3)      # UAV velocity (body frame) ⭐⭐ CHANGED ⭐⭐
        self.uav_ang_vel_in_odom_frame = np.zeros(3)  # UAV angular velocity (body frame)



        #predicted trajectory generated from EKF node
        self.nmpc_trajectory_ref=[]





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
        self.create_subscription(PoseStamped, '/relative_pose_odom_OR_ekf', self.rel_pose_odom_OR_ekf_cb, 10)
        self.create_subscription(Odometry, '/jackal/jackal_velocity_controller/odom', self.ugv_pose_cb, 10)
        self.create_subscription(Path, '/predicted_trajectory', self.predicted_trajectory_cb, 10)
        self.create_subscription(PoseStamped, '/ap/pose/filtered', self.uav_pose_cb, qos_profile)
        self.create_subscription(TwistStamped, '/ap/twist/filtered', self.uav_twist_cb, qos_profile)
        self.create_subscription(String,'/tracking_mode',self.mode_status_cb,10)
        self.create_subscription(PoseStamped,'/absolute_pose_odometry_OR_ekf',self.absolute_pose_odom_OR_ekf_cb,10) # subscribing to the absolute pose measurement from EKF node (Aruco + Odometry)
        # self.pub_absolute_pose_blended_odometry_OR_ekf = self.create_publisher(PoseStamped, '/absolute_pose_odometry_OR_ekf', 10) # publishing in  odom frame

        # Publishers
        self.Xref_pred_pub = self.create_publisher(Path, '/Xref_MPC_in_Odom_frame_from_predicted_trajectory', 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/mpc/cmd_vel', 10)
        self.error_pub = self.create_publisher(PointStamped, '/mpc/tracking_error', 10)

        
        # Timer Switcher
        self.create_timer(self.mpc_dt, self.control_loop)
        
        self.get_logger().info(f"Controller_for_UAV_Node started in {self.mode} mode.")
    
    def mode_status_cb(self,msg):
        if msg.data == "ODOMETRY_ACTIVE":
            self.ekf_active=False
        elif msg.data == "EKF_ACTIVE":
            self.ekf_active=True


    def predicted_trajectory_cb(self, msg):
        """
        Converts the nav_msgs/Path into a NumPy array for NMPC processing.
        """
        # Create a list of [x, y, z] coordinates from the path poses
        path_list = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            z = pose_stamped.pose.position.z
            path_list.append([x, y, z])

        # Convert to NumPy array: shape (N, 3)
        # This matrix can be fed directly into your NMPC solver
        self.nmpc_trajectory_ref = np.array(path_list)
        
        # Optional: Log the received horizon length
        # self.get_logger().info(f"Received trajectory with {len(self.nmpc_trajectory_ref)} points")

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

            # if self.ugv_position_in_odom_frame is not None:
            #     print(f"[TF] UGV world position: {self.ugv_position_in_odom_frame}")
            # else:
            #     print(f"[TF] Could not get UGV world position via TF")
            #     # Fallback to raw odometry (but warn it's local)
            #     self.ugv_position_in_odom_frame = np.array([
            #         msg.pose.pose.position.x,
            #         msg.pose.pose.position.y,
            #         msg.pose.pose.position.z
            #     ])
            #     print(f"[WARNING] Using local odom position: {self.ugv_position_in_odom_frame}")
            
            #  really don't understand why this line is here, why we are running the   self.run_odometry_mode()
            # from here, i need to write some justification here          
            # If not in EKF mode, track with odometry
            # if not self.ekf_active:
            #     self.run_odometry_mode()
                
        except Exception as e:
            print(f"[ERROR] ugv_pose_cb: {str(e)}")
            
    def get_ugv_world_position(self):
        """Get UGV position in world frame , remember world frame is odom, using TF"""
        try:
            
            # This code is getting the transform between two different odometry frames-specifically, 
            # it's finding where jackal/base_link is located in the global odom frame
            # REmember: jackal/odom is the initial positon of the jackal in the simulation world
            # jackal/base_link is fixed on top of the UGV. SO when UGV moves jackal/base_link moves as well

            print("I am in get_ugv_world_position")
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
        """Timer callback that switches between MPC and PID logic."""
        if not self.have_rel:
            self.get_logger().warn("No relative pose received yet", throttle_duration_sec=2.0)
            self.publish_cmd([0.0, 0.0, 0.0], 0.0)
            return

        if self.mode == "MPC":
            self.run_mpc_logic()
        elif self.mode == "PID":
            self.run_pid_logic()
        elif self.mode== "Fuzzy-PID":
            self.run_fuzzy_pid_logic()
        else:
            self.get_logger().error(f"Invalid mode: {self.mode}")

    def run_fuzzy_pid_logic(self):

        # --- Position Error ---
        # dx, dy, dz = self.ugv_position_in_odom_frame - self.uav_position_in_odom_frame
        dx, dy, dz = self.ugv_absolute_pose_in_odom_frame_EKF_estimation[:3] - self.uav_position_in_odom_frame
        
        dyaw = self.ugv_yaw_in_odom_frame - self.uav_yaw_in_odom_frame
        # Wrap dyaw to [-pi, pi] to avoid discontinuous control jumps
        dyaw = ((dyaw + np.pi) % (2 * np.pi)) - np.pi

        current_error = np.array([dx, dy, dz - 2.0, dyaw])

        # --- Error norms for fuzzy scaling ---
        e_pos_norm = np.linalg.norm(current_error[:3])
        e_yaw_abs = abs(dyaw)

       
        # --- Integral with anti-windup constraint ---
        self.error_integral += current_error * self.mpc_dt
        self.error_integral = np.clip(self.error_integral, -1.0, 1.0)

        # --- Derivative calculation with First-Order Low-Pass Filter ---
        raw_derivative = (current_error - self.prev_error) / self.mpc_dt
        self.prev_error = current_error

        # Low-pass filter coefficient (alpha_lpf between 0 and 1)
        # Lower values = smoother but introduces slight delay. 0.2-0.3 is optimal for 10-50Hz loops.
        if not hasattr(self, 'filtered_derivative'):
            self.filtered_derivative = np.zeros(4)
        
        alpha_lpf = 0.25 
        self.filtered_derivative = alpha_lpf * raw_derivative + (1.0 - alpha_lpf) * self.filtered_derivative
        de_norm = np.linalg.norm(self.filtered_derivative[:3])

        # # --- Derivative without Low Pass Filter ---
        # error_derivative = (current_error - self.prev_error) / self.mpc_dt
        # self.prev_error = current_error

        # de_norm = np.linalg.norm(error_derivative[:3])

        # =====================================================
        # FUZZY GAIN SCALING
        # =====================================================

        # Normalize error (tune these thresholds)
        e_scale = 5.0       # meters where error considered "large"
        de_scale = 3.0      # m/s rate considered "fast"

        e_ratio = np.clip(e_pos_norm / e_scale, 0.0, 1.5)
        de_ratio = np.clip(de_norm / de_scale, 0.0, 1.5)

        # --- Fuzzy Rules (smooth nonlinear functions) ---

        # Large error → high Kp
        #kp_scale = 0.02 + 0.3 * e_ratio
        #kp_scale = 0.03 + 0.3 * (1 - e_ratio) # combination 1
        kp_scale = 0.2 + 0.3 * (1-e_ratio) # combination 2
        alpha = 0.5
        #kp_scale = alpha*kp_scale + (1-alpha)*kp_scale# smoothing with original gain, alpha is the smoothing factor between 0 and 1
        # 1. Kp Scale: Small error -> lower gain (smooth); Large error -> higher gain (aggressive tracking)
        # This keeps the drone aggressive when far away, but soft and quiet when hovering on target.
        # kp_scale = 0.4 + 0.6 * e_ratio  # Ranges from 0.4 (close) to 1.0 (far away)




        # Small error → increase Ki
        #ki_scale = 0.07 + 1.0 * (1.0 - e_ratio); combination 1
        ki_scale = 0.07 + 1.0 * (1.0 - e_ratio); #combination 2


        # High derivative → increase Kd
        #kd_scale = 2.5 + 1.0 * de_ratio #combination1
        kd_scale = 2.5 + 1.0 * de_ratio# combination 2



        # Apply scaling
        kp_adapt = self.kp * kp_scale
        ki_adapt = self.ki * ki_scale
        kd_adapt = self.kd * kd_scale

        # =====================================================
        # PID Output
        # =====================================================

        # output = (
        #     kp_adapt * current_error
        #     + ki_adapt * self.error_integral
        #     + kd_adapt * error_derivative
        # )

        output = (
            kp_adapt * current_error
            + ki_adapt * self.error_integral
            + kd_adapt * self.filtered_derivative  # Uses the filtered derivative signal
        )

        vx_enu, vy_enu, vz_enu, wz_enu = output
        yaw = self.uav_yaw_in_odom_frame

        # ENU → FLU rotation (same as your working PID)
        vx_flu =  vx_enu * np.cos(yaw) + vy_enu * np.sin(yaw)
        vy_flu = -vx_enu * np.sin(yaw) + vy_enu * np.cos(yaw)
        vz_flu =  vz_enu
        wz_flu =  wz_enu

        # Clip
        vx = np.clip(vx_flu, -self.v_max, self.v_max)
        vy = np.clip(vy_flu, -self.v_max, self.v_max)
        vz = np.clip(vz_flu, -self.vz_max, self.vz_max)
        wz = np.clip(wz_flu, -self.yawdot_max, self.yawdot_max)

        if np.random.rand() < 0.1:
            self.get_logger().info(
                f"FuzzyPID | e_norm:{e_pos_norm:.2f} "
                f"Kp_scale:{kp_scale:.2f} "
                f"Cmd: Vx:{vx:.2f} Vy:{vy:.2f} Vz:{vz:.2f}"
            )

        self.publish_cmd([vx, vy, vz], wz)

    def run_pid_logic(self):
        

        e_pos=self.ugv_position_in_odom_frame-self.uav_position_in_odom_frame
        
        """Simple PID controller for UAV tracking."""
        # Errors in UAV body frame (FLU)
        # dx, dy, dz = self.ugv_pos_and_orient_in_UAV_frame[0:3]
        dx,dy,dz=self.ugv_position_in_odom_frame-self.uav_position_in_odom_frame
        dyaw = self.ugv_yaw_in_odom_frame- self.uav_yaw_in_odom_frame

        # Target 2m above UGV
        current_error = np.array([dx, dy, dz - 2.0, dyaw])
        
        # Update Integral and Derivative
        self.error_integral += current_error * self.mpc_dt
        # Clip integral to prevent windup
        self.error_integral = np.clip(self.error_integral, -1.0, 1.0)
        
        error_derivative = (current_error - self.prev_error) / self.mpc_dt
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

    def run_mpc_logic(self):
        """Existing MPC logic moved into this function."""
        if not self.have_rel:
            self.get_logger().warn("No relative pose received yet - publishing zero cmd")
            self.publish_cmd([0.0,0.0,0.0], 0.0)
            return

        # DEBUG
        # print(f"\n=== NMPC DEBUG ===")
        # print(f"Current rel state: {self.ugv_pos_and_orient_in_UAV_frame}")
        # print(f"v_g (UGV): {self.ugv_lin_vel_in_jackal_odom_frame}, v_u (UAV): {self.uav_lin_vel_in_odom_frame}")

        # copy states/odometry to local variables to avoid race conditions
        # x0 = self.ugv_pos_and_orient_in_UAV_frame.copy()
        # x0=self.uav_position_in_odom_frame
        v_g = self.ugv_lin_vel_in_jackal_odom_frame  #This must be in the odom frame not in /jacakl/odom frame
        w_g = self.ugv_ang_vel_in_jackal_odom_frame  #This must be in the odom frame not in /jacakl/odom frame
        v_u = self.uav_lin_vel_in_odom_frame
        w_u = self.uav_ang_vel_in_odom_frame

        # 1. Initialize the full matrix with zeros
        # self.N is 50 in your code
        Xref = np.zeros((self.N, 6))

        # 2. Extract the UGV world coordinates in odom frame
        ugv_x = self.ugv_position_in_odom_frame[0]
        ugv_y = self.ugv_position_in_odom_frame[1]
        ugv_z = self.ugv_position_in_odom_frame[2]

        # 3. Extract the UAV world coordinates in odom frame
        uav_x = self.uav_position_in_odom_frame[0]
        uav_y = self.uav_position_in_odom_frame[1]
        uav_z = self.uav_position_in_odom_frame[2]

        # 4. Initializing x0 from odom data
        # This is the "Option B" secret: the error calculation must be in odom frame
        # the error should be in odom frame, so that even if the drone twist and turn
        # the exact distance between the UAV and UGV is not affected
        x0 = np.zeros(6)
        x0[0] = uav_x
        x0[1] = uav_y
        x0[2] = uav_z
        x0[3] = 0
        x0[4] = 0
        x0[5] = self.uav_yaw_in_odom_frame 


        # x0 = np.array([
        #     0.0,  # X: UAV is at its own origin
        #     0.0,  # Y: UAV is at its own origin
        #     0.0,  # Z: UAV is at its own origin (height is relative to current)
        #     v_u[0], # VX: UAV current body-frame forward velocity
        #     v_u[1], # VY: UAV current body-frame lateral velocity
        #     0.0   # Yaw: UAV is facing its own "Forward" axis (0 radians)
        # ])
  
        # 1. Get current UAV heading (World Yaw) in odom frame
        psi = self.uav_yaw_in_odom_frame
        
        # self.nmpc_trajectory_ref is in jackal/odom frame, this data is coming from 
        # Traj_Pred_Using_Sensor_Data.py

        if hasattr(self, 'nmpc_trajectory_ref') and len(self.nmpc_trajectory_ref) > 0:
               # Lookup transform: odom ← jackal/odom
            # transform = self.tf_buffer.lookup_transform(
            #     'odom',           # target frame
            #     'jackal/odom',    # source frame
            #     rclpy.time.Time()
            # )
            try:
                # Attempt to find the transform between global odom and UGV odom
                transform = self.tf_buffer.lookup_transform(
                    'odom',           # target frame
                    'jackal/odom',    # source frame
                    rclpy.time.Time() # Get the latest available transform
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"TF2 Lookup failed: {e}")
                return # Prevent the rest of the code from running without a valid transform

            # Output Path in odom frame
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'odom'

            
            for p in self.nmpc_trajectory_ref:
                # 1. Create PoseStamped in source frame
                ps_in = PoseStamped()
                ps_in.header.frame_id = 'jackal/odom'
                ps_in.pose.position.x = float(p[0])
                ps_in.pose.position.y = float(p[1])
                ps_in.pose.position.z = float(p[2])
                ps_in.pose.orientation.w = 1.0

                # 2. Transform ONLY the .pose part
                # pose_out will be a geometry_msgs.msg.Pose object
                pose_out = do_transform_pose(ps_in.pose, transform)

                # ⭐ ADD OFFSET HERE ⭐
                pose_out.position.z += 2.0   #UAV flying at a constant altitude of 2.0 m
                
                # 3. Create a new PoseStamped to put into the Path
                new_ps = PoseStamped()
                new_ps.header = path_msg.header # Use the target frame 'odom'
                new_ps.pose = pose_out

                path_msg.poses.append(new_ps)

            self.Xref_pred_pub.publish(path_msg)


        # controls
        U = np.zeros((self.N, 4))
        U_prev = np.zeros_like(U)

        # optimization parameters
        iters = 15
        alpha = 0.05
        eps = 1e-3


        # This is the "Option B" secret:
        #  the error should be in odom frame, so that even if the drone twist and turn
        #  the exact distance between the UAV and UGV is not affected
        # precompute base cost
        # x0[0:3] = UAV position in odom frame
        # Xref[k, 0:3] is the UGV position in odom frame obtained by transforming the  /relative_pose_odom_OR_ekf
        # from FLU to ENU frame and then adding the UAV position, so effectively this is coming from Odometry+EKF
    
        # Xref[k, 5] = self.uav_yaw_in_odom_frame
        # U control signals
        # v_g and w_g are in jackal odom frame, since all the calculations are in odom frame
        # these 2 velocities must also be in odom frame not in /jackal/odom frame 
        cost_base = self._simulate_cost(x0, U, Xref)

        for it in range(iters):
            grad = np.zeros_like(U)

            # Efficient forward-difference gradient: perturb one control element at a time
            for i in range(self.N):
                for j in range(4):
                    Up = U.copy()
                    Up[i,j] += eps
                    cost_p = self._simulate_cost(x0, Up, Xref)
                    grad[i,j] = (cost_p - cost_base) / eps

            # gradient step
            U -= alpha * grad

            # projection
            for k in range(self.N):
                U[k,0] = np.clip(U[k,0], -self.v_max, self.v_max)  # vx control signal
                U[k,1] = np.clip(U[k,1], -self.v_max, self.v_max)  # vy control signal
                U[k,2] = np.clip(U[k,2], -self.vz_max, self.vz_max)# vz control signal
                U[k,3] = np.clip(U[k,3], -self.yawdot_max, self.yawdot_max) # yaw control signal

            # update U_prev and base cost for next iteration
            U_prev = U.copy()
            cost_base = self._simulate_cost(x0, U, Xref)

            # simple stopping
            if np.linalg.norm(grad) < 1e-2:
                break

        # extract first command
        u0 = U[0,:]
        vx_cmd, vy_cmd, vz_cmd, yawdot_cmd = float(u0[0]), float(u0[1]), float(u0[2]), float(u0[3])
        # print(f"MPC Command: vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, vz={vz_cmd:.3f}")

        # safety checks
        if not np.isfinite(vx_cmd + vy_cmd + vz_cmd + yawdot_cmd):
            self.get_logger().error("MPC produced non-finite command — sending zero")
            vx_cmd, vy_cmd, vz_cmd, yawdot_cmd = 0.0, 0.0, 0.0, 0.0

        # clip again
        vx_cmd = float(np.clip(vx_cmd, -self.v_max, self.v_max))
        vy_cmd = float(np.clip(vy_cmd, -self.v_max, self.v_max))
        vz_cmd = float(np.clip(vz_cmd, -self.vz_max, self.vz_max))
        yawdot_cmd = float(np.clip(yawdot_cmd, -self.yawdot_max, self.yawdot_max))

        # log every few cycles to avoid slowing down
        if np.random.rand() < 0.25:
            self.get_logger().info(f"MPC Command → vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, vz={vz_cmd:.3f}, yawdot={yawdot_cmd:.3f}")

        # Check if command makes sense:
        dx, dy, dz = self.ugv_pos_and_orient_in_UAV_frame[0:3]
        
        self.publish_cmd([vx_cmd, vy_cmd, vz_cmd], yawdot_cmd)
        
    def _simulate_cost(self, x0, U, Xref):
        # This is the "Option B" secret:
        #  the error should be in odom frame, so that even if the drone twist and turn
        #  the exact distance between the UAV and UGV is not affected
        # Xref[k, 0:3] is the UGV position in odom frame obtained by transforming the  
        # /relative_pose_odom_OR_ekf to odom frame(from FLU to ENU frame and then adding the UAV position),
        # so effectively this is coming from Odometry+EKF
        # x0 is the UAV position in odom frame
        x_sim = x0.copy()  #x0 is the position of the UAV in the odom frame

        total = 0.0

        for k in range(self.N):
            # 1. ERROR CALCULATION (Absolute Error)
            # Xref[k, 0:3] is in the odom frame
            # Xref[k, 0:3] is the UGV position in odom frame obtained by transforming the  /relative_pose_odom_OR_ekf
            # from FLU to ENU frame and then adding the UAV position, so efgfectivekly this is coming from Odometry+EKF

            # x_sim ​: The predicted state vector of the UAV at time step k, 
            # containing its position and velocity in the global frame:
            # --- 1. Calculate Error FIRST (at the start of the step) ---
            e_pos = x_sim[0:3] - Xref[k, 0:3]

            if k == 0:
                error_msg = PointStamped()
                error_msg.header.stamp = self.get_clock().now().to_msg()
                error_msg.header.frame_id = 'odom'
                error_msg.point.x, error_msg.point.y, error_msg.point.z = e_pos.astype(float)
                self.error_pub.publish(error_msg)
           
            # --- 2. Add to Cost ---
            total += e_pos.T @ self.Q_pos @ e_pos

            # --- 3. Predict NEXT state (Integrate) ---
            yaw_sim = x_sim[5] 
            
            # --- 4. Get Control Inputs (UAV Body Frame i.e. FLU)
            vx_uav = U[k, 0]  
            vy_uav = U[k, 1]
            vz_uav = U[k, 2]
            yawrate_uav = U[k, 3]

            # ---5. TRANSFORM Control Signals i.e. Body Velocity in (FLU) to World Velocity in (ENU)
            # This is the "Option B" secret: 
            # We map Body (Forward/Left) to World (North/East)
            vdx_world = vx_uav * np.cos(yaw_sim) - vy_uav * np.sin(yaw_sim)
            vdy_world = vx_uav * np.sin(yaw_sim) + vy_uav * np.cos(yaw_sim)
            vdz_world = vz_uav
            
            # ---6. UPDATE World State (UAV Position in Odom)
            # x_sim[0:3] are now UAV World Coordinates in odom frame
           
            x_sim[0] += vdx_world * self.pred_dt
            x_sim[1] += vdy_world * self.pred_dt
            x_sim[2] += vdz_world * self.pred_dt
            x_sim[5] += yawrate_uav * self.pred_dt

           
        


           
        
        # for k in range(self.N):
        #     # Current relative orientation
        #     roll, pitch, yaw = x_sim[3], x_sim[4], x_sim[5]
        #     R_rel = rpy_to_rot(roll, pitch, yaw)  # UGV relative to UAV
            
        #     vk = U[k, 0:3]  # UAV velocity command in UAV body frame
        #     yawdotk = U[k, 3]
            
        #     # ============= CORRECTED DYNAMICS =============
        #     # Transform UGV velocity to UAV body frame
        #     # v_g is in UGV body frame, need to transform to UAV body frame
        #     # Simplified: Assume same orientation for now
        #     v_g_uav_body = R_rel @ v_g  # Transform UGV velocity to UAV frame
            
        #     # Relative velocity: how relative position changes
        #     # dx/dt = v_ugv_in_uav_frame - v_uav
        #     rel_vel = v_g_uav_body - vk
            
        #     x_sim[0:3] = x_sim[0:3] + rel_vel * self.pred_dt
            
        #     # Angular: UGV should face UAV (yaw → 0)
        #     # Simple: yaw_dot = -yaw (to reduce yaw error)
        #     # rel_omega = np.array([0.0, 0.0, -yaw + yawdotk])
        #     # x_sim[5] = x_sim[5] + rel_omega[2] * self.pred_dt

        #     x_sim[5] = x_sim[5] + (w_g[2] - yawdotk) * self.pred_dt
        #     # ==============================================
            
        #     # Position error
        #     # ❌❌❌❌❌ I think this is the major problem ❌❌❌❌❌
        #     # the error should be in odom frame, so that even if the drone twist and turn
        #     # the exact distance between the UAV and UGV is not affected
        #     # I need to correct this calculation
        #     # x_sim is UGV position and orientation in UAV frame
        #     # X_ref  is also self.ugv_pos_and_orient_in_UAV_frame[0:3]
        #     e_pos = x_sim[0:3] - Xref[k, 0:3]
        #     total += e_pos.T @ self.Q_pos @ e_pos
            
        #     # Yaw error only
        #     yaw_error = wrap_angle(x_sim[5] - Xref[k, 5])
        #     total += self.Q_ang[2, 2] * yaw_error**2
            
        #     # Control effort
        #     total += U[k, :].T @ np.diag([0.1, 0.1, 0.1, 0.01]) @ U[k, :]
            
        #     # FOV cost
        #     horiz_err = np.linalg.norm(x_sim[0:2])
        #     total += self.W_fov * (horiz_err**2) / ((horiz_err**2) + (Xref[k, 2]**2) + 1e-6)
        
        # Smoothness
        for k in range(self.N):
            du = U[k, :] - (np.zeros(4) if k == 0 else U[k-1, :])
            total += du.T @ self.R_du @ du
        
        return float(total)   


    # --- (Include all other helper functions like rel_pose_cb, quat_to_rpy, etc here) ---
    def rel_pose_odom_OR_ekf_cb(self, msg: PoseStamped):
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        roll, pitch, yaw = self.quat_to_rpy_msg(msg.pose.orientation)
        self.ugv_pos_and_orient_in_UAV_frame = np.array([x, y, z, roll, pitch, yaw], dtype=float)
        self.have_rel = True
    
    def absolute_pose_odom_OR_ekf_cb(self, msg: PoseStamped):
        # This callback is getting UGV position from EKF and is most critical for the control logic
        # It gives the absolute pose of the UGV in the odom frame (world coordinates)
        # This is the belnded ROS topic ('/absolute_pose_odometry_OR_ekf'), 
        # data from odometery and EKF are blenede in this ROS topic
        
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        roll, pitch, yaw = self.quat_to_rpy_msg(msg.pose.orientation)
        self.ugv_absolute_pose_in_odom_frame_EKF_estimation = np.array([x, y, z, roll, pitch, yaw], dtype=float)


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
    
    def debug_nmpc_status(self, x0, U, Xref):
        """
        Minimal NMPC Debugger to identify coordinate frame mismatches.
        x0: Current state [x, y, z, r, p, yaw] (Relative)
        U:  Control array [N, 4]
        Xref: Reference trajectory [N, 6]
        """
        print("\n" + "!"*30 + " NMPC COORDINATE DEBUG " + "!"*30)
        
        # 1. STATE VS REFERENCE (Initial Error)
        # This tells you if the NMPC knows where it is relative to the target
        print(f"CURRENT RELATIVE STATE (x0):  X:{x0[0]:.2f}, Y:{x0[1]:.2f}, Z:{x0[2]:.2f}, Yaw:{np.degrees(x0[5]):.1f}°")
        print(f"FIRST TARGET POINT (Xref[0]): X:{Xref[0,0]:.2f}, Y:{Xref[0,1]:.2f}, Z:{Xref[0,2]:.2f}, Yaw:{np.degrees(Xref[0,5]):.1f}°")
        
        pos_error = Xref[0, 0:3] - x0[0:3]
        print(f"IMMEDIATE ERROR VECTOR:      dX:{pos_error[0]:.2f}, dY:{pos_error[1]:.2f}, dZ:{pos_error[2]:.2f}")

        # 2. TRAJECTORY TREND
        # Check if the predicted trajectory is moving away or toward the drone
        if self.N > 1:
            traj_move = Xref[-1, 0:3] - Xref[0, 0:3]
            print(f"TRAJECTORY TREND (N steps):  dX:{traj_move[0]:.2f}, dY:{traj_move[1]:.2f}, dZ:{traj_move[2]:.2f}")

        # 3. CONTROL ALIGNMENT CHECK
        # This is the "Truth Test": if dX is positive, VX should be positive.
        u0 = U[0, :]
        print("-" * 20)
        print(f"MPC COMMAND (u0):            VX:{u0[0]:.3f}, VY:{u0[1]:.3f}, VZ:{u0[2]:.3f}, Wz:{u0[3]:.3f}")
        
        # LOGIC CHECK: Does the command reduce the error?
        # If dX > 0 (Target ahead), VX should be > 0.
        # If dX < 0 (Target behind), VX should be < 0.
        vx_correct = (pos_error[0] > 0 and u0[0] > 0) or (pos_error[0] < 0 and u0[0] < 0)
        vy_correct = (pos_error[1] > 0 and u0[1] > 0) or (pos_error[1] < 0 and u0[1] < 0)
        
        print(f"DIRECTIONAL LOGIC CHECK:     VX: {'✅ CORRECT' if vx_correct else '❌ REVERSED'}")
        print(f"                             VY: {'✅ CORRECT' if vy_correct else '❌ REVERSED'}")
        
        # 4. VELOCITY CONTEXT
        if hasattr(self, 'v_u'):
            print(f"UAV CURR VEL (Body):         VX:{self.v_u[0]:.2f}, VY:{self.v_u[1]:.2f}")

        print("!"*83 + "\n")

# (Include other callbacks as provided in original script)

def main(args=None):
    rclpy.init(args=args)
    
    # Selection logic
    mode_selection = "Fuzzy-PID"  # Set to "MPC" or "PID" or "Fuzzy-PID"
    node = Controller_for_UAV_Node(mode=mode_selection)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()