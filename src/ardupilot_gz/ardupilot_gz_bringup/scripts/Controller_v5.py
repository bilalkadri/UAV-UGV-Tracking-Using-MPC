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
    def __init__(self, mode="Fuzzy-PID"):
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
       
        if not self.have_rel:
            self.get_logger().warn("No relative pose received yet", throttle_duration_sec=2.0)
            self.publish_cmd([0.0, 0.0, 0.0], 0.0)
            return

        if self.mode == "PID":
            self.run_pid_logic()
        elif self.mode== "Fuzzy-PID":
            self.run_fuzzy_pid_logic()
        else:
            self.get_logger().error(f"Invalid mode: {self.mode}")

  

    def run_fuzzy_pid_logic(self):
        """
        Fuzzy-PID controller with anti-saturation protection and smooth gain scheduling
        """
        # --- Position Error Calculation ---
        if self.ekf_active:
            dx, dy, dz = self.ugv_absolute_pose_in_odom_frame_EKF_estimation[:3] - self.uav_position_in_odom_frame
        else:
            dx, dy, dz = self.ugv_position_in_odom_frame - self.uav_position_in_odom_frame
        
        dyaw = self.ugv_yaw_in_odom_frame - self.uav_yaw_in_odom_frame
        dyaw = wrap_angle(dyaw)
        
        current_error = np.array([dx, dy, dz - 2.0, dyaw])
        
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
        integral_increment = current_error * self.mpc_dt
        
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
        raw_derivative = (current_error - self.prev_error) / self.mpc_dt
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
        vx_enu_prelim = output_prelim[0]
        vy_enu_prelim = output_prelim[1]
        
        # Check for saturation
        if abs(vx_enu_prelim) > fuzzy_v_max:
            # Back-calculation: reduce integral to prevent windup
            excess = vx_enu_prelim - np.sign(vx_enu_prelim) * fuzzy_v_max
            self.fuzzy_error_integral[0] -= excess * self.mpc_dt * 0.5
            output_prelim[0] = np.sign(vx_enu_prelim) * fuzzy_v_max
        
        if abs(vy_enu_prelim) > fuzzy_v_max:
            excess = vy_enu_prelim - np.sign(vy_enu_prelim) * fuzzy_v_max
            self.fuzzy_error_integral[1] -= excess * self.mpc_dt * 0.5
            output_prelim[1] = np.sign(vy_enu_prelim) * fuzzy_v_max
        
        # Extract final commands
        vx_enu, vy_enu, vz_enu, wz_enu = output_prelim
        
        # ENU → FLU rotation
        yaw = self.uav_yaw_in_odom_frame
        vx_flu =  vx_enu * np.cos(yaw) + vy_enu * np.sin(yaw)
        vy_flu = -vx_enu * np.sin(yaw) + vy_enu * np.cos(yaw)
        vz_flu =  vz_enu
        wz_flu =  wz_enu
        
        # Final clipping (should rarely trigger now due to anti-saturation)
        vx = np.clip(vx_flu, -fuzzy_v_max, fuzzy_v_max)
        vy = np.clip(vy_flu, -fuzzy_v_max, fuzzy_v_max)
        vz = np.clip(vz_flu, -fuzzy_vz_max, fuzzy_vz_max)
        wz = np.clip(wz_flu, -fuzzy_yaw_max, fuzzy_yaw_max)
        
        # Oscillation detection and damping
        if not hasattr(self, 'prev_vx_cmd'):
            self.prev_vx_cmd = 0.0
            self.prev_vy_cmd = 0.0
        
        # Detect sign changes (oscillation indicator)
        vx_sign_change = (vx * self.prev_vx_cmd < 0) and (abs(vx) > 0.5) and (abs(self.prev_vx_cmd) > 0.5)
        vy_sign_change = (vy * self.prev_vy_cmd < 0) and (abs(vy) > 0.5) and (abs(self.prev_vy_cmd) > 0.5)
        
        if vx_sign_change or vy_sign_change:
            # Apply extra damping when oscillating detected
            damping_factor = 0.7  # Reduce command by 30%
            vx *= damping_factor
            vy *= damping_factor
            if np.random.rand() < 0.3:
                self.get_logger().warn(f"Oscillation detected! Damping applied. Vx:{vx:.2f} Vy:{vy:.2f}")
        
        self.prev_vx_cmd = vx
        self.prev_vy_cmd = vy
        
        # Debug logging
        if np.random.rand() < 0.1:
            self.get_logger().info(
                f"FuzzyPID | Err:{e_pos_norm:.2f}m | "
                f"Kp:[{kp_adapt[0]:.2f},{kp_adapt[1]:.2f}] "
                f"Ki_s:{ki_scale:.2f} Kd_s:{kd_scale:.2f} | "
                f"Cmd:[{vx:.2f},{vy:.2f},{vz:.2f}] | "
                f"Sat:{'YES' if max(abs(vx_enu_prelim),abs(vy_enu_prelim))>fuzzy_v_max*0.95 else 'NO'}"
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

           
  
    
    def absolute_pose_odom_OR_ekf_cb(self, msg: PoseStamped):
        # This callback is getting UGV position from EKF and is most critical for the control logic
        # It gives the absolute pose of the UGV in the odom frame (world coordinates)
        # This is the belnded ROS topic ('/absolute_pose_odometry_OR_ekf'), 
        # data from odometery and EKF are blenede in this ROS topic
        
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        roll, pitch, yaw = self.quat_to_rpy_msg(msg.pose.orientation)
        self.ugv_absolute_pose_in_odom_frame_EKF_estimation = np.array([x, y, z, roll, pitch, yaw], dtype=float)
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