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
import math



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
        self.pred_dt = 0.015


        # MPC execution period (seconds)
        # How often the MPC is solved
        # Increasing this directly slows reaction speed
        # 0.1 → aggressive, 0.15–0.2 → sluggish and stable
        self.mpc_dt = 0.015


   
       
        #----------------------------------------------------------------------------------------
        # State tracking weights (LOWERED TO REDUCE AGGRESSION)
        #----------------------------------------------------------------------------------------
        # Position tracking cost matrix [x, y, z]
        # High values force the UAV to correct position errors aggressively
        # These are intentionally reduced to avoid violent corrections
        self.Q_pos = np.diag([
            0.500,    # x position (was 10)
            0.500,    # y position (was 100 - THIS WAS YOUR MAIN SOURCE OF INSTABILITY)
            10.0     # z position (was 5)
        ])


        # Orientation (angle) tracking cost matrix [roll, pitch, yaw]
        # Lower yaw weight prevents fast spinning to correct small angular noise
        self.Q_ang = np.diag([
            1.0,   # roll
            1.0,   # pitch
            2.0    # yaw (was 10.0)
        ])

 
        
        #----------------------------------------------------------------------------------------
        # Control smoothness penalties (INCREASED FOR DAMPING)
        #----------------------------------------------------------------------------------------
        # R_u penalizes absolute control effort
        self.R_u = np.diag([
        1.0,   # vx
        1.0,   # vy
        0.5,   # vz (less aggressive)
        0.3    # yaw_rate
        ])
        
        # Penalty on change in control inputs Δu = u(k) - u(k-1)
        # THIS IS THE MOST IMPORTANT PART FOR SMOOTHNESS
        # Large values strongly discourage sudden control changes
        self.R_du = np.diag([
            20.0,  # Δvx (was 200) - Heavy damping for longitudinal motion
            20.0,  # Δvy (was 200) - Heavy damping to stop the Y-axis slamming
            7.50,  # Δvz (was 80)  - Stop the vertical "bouncing"
            5.0   # Δyaw_rate (was 30) - Smooth out rotations
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
        self.v_max = 10.0     # was 20.0


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
        self.e_pos = np.zeros(3)

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
        self.create_subscription(PoseStamped, '/relative_pose_odom_OR_ekf', self.rel_pose_cb, 10) # This is the position of the UGV in odom frame
        self.create_subscription(Odometry, '/jackal/jackal_velocity_controller/odom', self.ugv_pose_cb, 10)
        self.create_subscription(Path, '/predicted_trajectory', self.predicted_trajectory_cb, 10)
        self.create_subscription(PoseStamped, '/ap/pose/filtered', self.uav_pose_cb, qos_profile)
        self.create_subscription(TwistStamped, '/ap/twist/filtered', self.uav_twist_cb, qos_profile)
        self.create_subscription(String,'/tracking_mode',self.mode_status_cb,10)


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
          

            # self.get_ugv_world_position()


                
        except Exception as e:
            print(f"[ERROR] ugv_pose_cb: {str(e)}")
            
    # def get_ugv_world_position(self):
    #     """Get UGV position in world frame , remember world frame is odom, using TF"""
    #     try:
            
    #         # This code is getting the transform between two different odometry frames-specifically, 
    #         # it's finding where jackal/base_link is located in the global odom frame
    #         # REmember: jackal/odom is the initial positon of the jackal in the simulation world
    #         # jackal/base_link is fixed on top of the UGV. SO when UGV moves jackal/base_link moves as well

    #         # print("I am in get_ugv_world_position")
    #         # 1. Get UGV position relative to its own start (jackal/base_link -> jackal/odom)
    #         t1 = self.tf_buffer.lookup_transform('jackal/odom', 'jackal/base_link', rclpy.time.Time())
    #         # print("t1",t1)
    #         # 2. Get UGV starting offset relative to world (jackal/odom -> odom)
    #         t2 = self.tf_buffer.lookup_transform('odom', 'jackal/odom', rclpy.time.Time())
    #         # print("t2",t2)
    #         # 3. Combine them: World_Pos = Offset + (Rotation_of_Offset * Local_Pos)
    #         # We use geometry_msgs math or simple addition if rotations are aligned

    #         self.ugv_position_in_odom_frame[0] = t2.transform.translation.x + t1.transform.translation.x
    #         self.ugv_position_in_odom_frame[1]  = t2.transform.translation.y + t1.transform.translation.y
    #         self.ugv_position_in_odom_frame[2]  = t2.transform.translation.z + t1.transform.translation.z
    #         # pos = np.array([ugv_world_x, ugv_world_y, ugv_world_z])
    #         # print("t1.transform.translation.x ", t1.transform.translation.x )
    #         # print("t2.transform.translation.x ", t2.transform.translation.x )
    #         # --- ORIENTATION (Combined Yaw) ---
    #         # Get yaw from the local movement (t1) using your function
    #         yaw_local = get_yaw_from_quat(t1.transform.rotation)
            
    #         # Get yaw from the starting offset (t2) using your function
    #         yaw_offset = get_yaw_from_quat(t2.transform.rotation)

    #         # Total Yaw = Offset Yaw + Local Yaw
    #         combined_yaw = yaw_offset + yaw_local
            
    #         # Normalize the result to stay within [-pi, pi]
    #         self.ugv_yaw_in_odom_frame = np.arctan2(np.sin(combined_yaw), np.cos(combined_yaw))
     
            

                     
    #         # return np.array([ugv_world_x, ugv_world_y, ugv_world_z])
            
    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
    #             tf2_ros.ExtrapolationException) as e:
    #         print(f"❌❌❌❌❌[TF] Error getting transform❌❌❌❌❌: {str(e)}")
            
    #         # Fallback: try other frame combinations
    #         try:
    #             transform = self.tf_buffer.lookup_transform(
    #                 'odom',              # target frame (common odom frame)
    #                 'jackal/base_link',  # source frame (UGV body)
    #                 rclpy.time.Time()
    #             )
    #             ugv_odom_pos_local_variable = transform.transform.translation
    #             return np.array([ugv_odom_pos_local_variable.x, ugv_odom_pos_local_variable.y, ugv_odom_pos_local_variable.z])
    #         except:
    #             return None
            
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
        else:
            self.get_logger().error(f"Invalid mode: {self.mode}")

    def run_pid_logic(self):
        

        self.e_pos=self.ugv_position_in_odom_frame-self.uav_position_in_odom_frame
        
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
        #Since the error was in ENU therefore the PID output is also in ENU
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
        print("⭐⭐Inside MPC⭐⭐")
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
  
        # 1. Get current UAV heading (World Yaw) in odom frame
        psi = self.uav_yaw_in_odom_frame
        
        # self.nmpc_trajectory_ref is in jackal/odom frame, this data is coming from 
        # Traj_Pred_Using_Sensor_Data.py
        # Xref_pred_pub publishes on /Xref_MPC_in_Odom_frame_from_predicted_trajectory , this is
        # the actual reference for the NMPC

        if hasattr(self, 'nmpc_trajectory_ref') and len(self.nmpc_trajectory_ref) > 0:
               # Lookup transform: odom ← jackal/odom
            try:
               # Step 1: jackal/base_link ← jackal/odom
                # t1 = self.tf_buffer.lookup_transform('target_frame', 'source_frame', rclpy.time.Time())
                t1 = self.tf_buffer.lookup_transform('jackal/odom','jackal/base_link',rclpy.time.Time())
                # Step 2: odom ← jackal/base_link
                t2 = self.tf_buffer.lookup_transform('odom', 'jackal/odom', rclpy.time.Time())

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"TF2 Lookup failed inside MPC: {e}")
                return # Prevent the rest of the code from running without a valid transform

            # Output Path in odom frame
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'odom'

            
            for p in self.nmpc_trajectory_ref:
                # 1. Create PoseStamped in source frame
                pose_in = PoseStamped()
                pose_in.header.frame_id = 'jackal/odom'
                pose_in.pose.position.x = float(p[0])
                pose_in.pose.position.y = float(p[1])
                pose_in.pose.position.z = float(p[2])
                pose_in.pose.orientation.w = 1.0

                # 2. Transform ONLY the .pose part
                # pose_out will be a geometry_msgs.msg.Pose object
                # Transform 1: Result is a Pose
                pose_mid = do_transform_pose(pose_in.pose, t2) #transform from 'jackal/odom' -->'odom'
                # print('pose_mid',pose_mid)

                # Transform 2: Pass the result of Transform 1 directly
                # pose_final = do_transform_pose(pose_mid, t2)

                # ⭐ ADD OFFSET HERE ⭐
                # pose_final.position.z += 2.0   #UAV flying at a constant altitude of 2.0 m
                pose_mid.position.z += 2.0   #UAV flying at a constant altitude of 2.0 m

                # 3. Create a new PoseStamped to put into the Path
                new_ps = PoseStamped()
                new_ps.header = path_msg.header # Use the target frame 'odom'
                # new_ps.pose = pose_final
                new_ps.pose = pose_mid

                path_msg.poses.append(new_ps)
            #published in topic /Xref_MPC_in_Odom_frame_from_predicted_trajectory'
            self.Xref_pred_pub.publish(path_msg) 


            num_points = min(len(path_msg.poses), self.N)
            for k in range(num_points):
                ps = path_msg.poses[k]
                Xref[k, 0] = ps.pose.position.x
                Xref[k, 1] = ps.pose.position.y
                Xref[k, 2] = ps.pose.position.z
                Xref[k, 5] = 0.0

            # Fill remaining with last point (VERY IMPORTANT)
            for k in range(num_points, self.N):
                Xref[k, :] = Xref[num_points - 1, :]

        # The control signals must be in the FLU frame
        # Beware the Xref and the x0 (UAV position) are in odom frames
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
        # Xref[k, 0:3] is Xref_pred_pub which publishes on /Xref_MPC_in_Odom_frame_from_predicted_trajectory , 
        # this is  the actual reference for the NMPC
    
        # x0 and Xref are in odom frame
        # U control signals are in FLU frame, the U must be converted to odom (ENU) frame

        cost_base = self._simulate_cost(x0, U, Xref)

        for it in range(iters):
            grad = np.zeros_like(U)
            #Generating the control signals over the control horizon 
            # Efficient forward-difference gradient: perturb one control element at a time
            for i in range(self.N):
                for j in range(4):
                    Up = U.copy()
                    Up[i,j] += eps # <--- GENERATING A TEST SIGNAL (Perturbation)
                    cost_p = self._simulate_cost(x0, Up, Xref) # <--- TESTING IT
                    grad[i,j] = (cost_p - cost_base) / eps # <--- SEEING IF IT HELPED

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
            cost_base = self._simulate_cost(x0, U_prev, Xref)

            # simple stopping
            if np.linalg.norm(grad) < 1e-2:
                break

        # extract first command
        u0 = U[0,:]

        # 1. Extract ENU velocities from your control input/state u0
        vx_uav_ENU = u0[0]
        vy_uav_ENU = u0[1]
        vz_uav_ENU = u0[2]
        yawrate_uav_ENU = u0[3]

        # 2. Get the current orientation (this must be updated from your telemetry/state)
        current_yaw = psi
        
        # 3. Perform the conversion to FLU (Front-Left-Up)
        vx_uav_FLU = vx_uav_ENU * np.cos(current_yaw) + vy_uav_ENU * np.sin(current_yaw)
        vy_uav_FLU = -vx_uav_ENU * np.sin(current_yaw) + vy_uav_ENU * np.cos(current_yaw)

        # 4. Vertical velocity and Yaw Rate remain the same in both frames
        vz_uav_FLU = vz_uav_ENU 
        yawrate_uav_FLU = yawrate_uav_ENU

        # Resulting vector for FLU-based control
        u0_FLU = [vx_uav_FLU, vy_uav_FLU, vz_uav_FLU, yawrate_uav_FLU]
        vx_uav_flu, vy_uav_flu, vz_uav_flu, yawdot_uav_flu = float(u0_FLU[0]), float(u0_FLU[1]), float(u0_FLU[2]), float(u0_FLU[3])
        # print(f"MPC Command: vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, vz={vz_cmd:.3f}")

        # safety checks
        if not np.isfinite(vx_uav_flu + vy_uav_flu + vz_uav_flu + yawdot_uav_flu):
            self.get_logger().error("MPC produced non-finite command — sending zero")
            vx_uav_flu, vy_uav_flu, vz_uav_flu, yawdot_uav_flu = 0.0, 0.0, 0.0, 0.0

        # clip again
        vx_uav_flu = float(np.clip(vx_uav_flu, -self.v_max, self.v_max))
        vy_uav_flu = float(np.clip(vy_uav_flu, -self.v_max, self.v_max))
        vz_uav_flu = float(np.clip(vz_uav_flu, -self.vz_max, self.vz_max))
        yawdot_uav_flu = float(np.clip(yawdot_uav_flu, -self.yawdot_max, self.yawdot_max))

        # log every few cycles to avoid slowing down
        if np.random.rand() < 0.25:
            self.get_logger().info(f"MPC Command → vx={vx_uav_flu:.3f}, vy={vy_uav_flu:.3f}, vz={vz_uav_flu:.3f}, yawdot={yawdot_uav_flu:.3f}")

        # self.e_pos=self.ugv_position_in_odom_frame-self.uav_position_in_odom_frame
        print(f"⭐⭐MPC Command⭐⭐: vx={vx_uav_flu:.3f}, vy={vy_uav_flu:.3f}, vz={vz_uav_flu:.3f}")
        
        # print(f"⭐⭐Tracking Error⭐⭐: ex={self.e_pos[0]:.3f}, ey={self.e_pos[1]:.3f}, ez={self.e_pos[2]:.3f}")
        self.publish_cmd([vx_uav_flu, vy_uav_flu, vz_uav_flu], yawdot_uav_flu)
        
    

    def _simulate_cost(self, x0, U, Xref):
        # x0: [pos_x, pos_y, pos_z, roll, pitch, yaw]
        x_sim = x0.copy() 
        
        # Initialize velocity states from your ODOM frame measurements
        # We track these inside the loop to simulate momentum
        # curr_v_world = self.uav_lin_vel_in_odom_frame.copy() 
        curr_v_world =np.zeros(3)

        total = 0.0

        # Actuator/Airframe Time Constants (Damping factors)
        tau = 0.1  # XY damping (higher = more overdamped/sluggish)
        tau_z = 0.2 # Z damping

        for k in range(self.N):
            # 1. ERROR CALCULATION (Target - Current) in odom frame
            # self.e_pos = Xref[k, 0:3] - x_sim[0:3]
            e_pos = Xref[k, 0:3] - x_sim[0:3] #this is in ENU frame
            
            # print("e_pos=",self.e_pos)
            # total += self.e_pos.T @ self.Q_pos @ self.e_pos
            total += e_pos.T @ self.Q_pos @ e_pos


            # 2. GET CONTROL INPUTS (UAV Body Frame FLU)
            # These are the inputs for which i want to compute the cost
            # vx_uav_FLU = U[k, 0]
            # vy_uav_FLU = U[k, 1]
            # vz_uav_FLU = U[k, 2]
            # yawrate_uav_FLU = U[k, 3]

            # 2. WE Assume thet the CoNTROL INPUTS ARE IN ENU FRAME

            vx_uav_ENU = U[k, 0]
            vy_uav_ENU = U[k, 1]
            vz_uav_ENU = U[k, 2]
            yawrate_uav_ENU = U[k, 3]

            # Control magnitude penalty
            u_vec = np.array([vx_uav_ENU, vy_uav_ENU, vz_uav_ENU, yawrate_uav_ENU])
            
            total += u_vec.T @ self.R_u @ u_vec

            # This adds damping.
            # This is essential for tight behavior.
            if k > 0:
                du = U[k,:] - U[k-1,:]
                total += du.T @ self.R_du @ du

            # 3. TRANSFORM COMMANDED Body Velocity to World Velocity
            # since my model and the error all are in ENU frame , I have to convert the 
            # the control signal which was in FLU to ENU frame 
            # yaw_uav_ENU = x_sim[5]
            # vx_uav_ENU = (vx_uav_FLU * np.cos(yaw_uav_ENU) - vy_uav_FLU * np.sin(yaw_uav_ENU))
            # vy_uav_ENU = (vx_uav_FLU * np.sin(yaw_uav_ENU) + vy_uav_FLU * np.cos(yaw_uav_ENU))
            # vz_uav_ENU = vz_uav_FLU # Assuming odom and body Z are aligned



            # print('Inside the internal model: vx_enu=',vx_uav_ENU,'vy_enu=',vy_uav_ENU)
            # print('Inside the internal model: vx_flu=',vx_uav_FLU,'vy_flu=',vy_uav_FLU)

            # 4. APPLY FIRST-ORDER DYNAMICS IN WORLD FRAME
            # Instead of v = v_target, we simulate the acceleration lag
            # dv = (v_target - v_current) / tau
            curr_v_world[0] += (self.pred_dt / tau) * (vx_uav_ENU - curr_v_world[0])
            curr_v_world[1] += (self.pred_dt / tau) * (vy_uav_ENU - curr_v_world[1])
            curr_v_world[2] += (self.pred_dt / tau_z) * (vz_uav_ENU - curr_v_world[2])

            # 5. UPDATE WORLD POSITION
            x_sim[0] += curr_v_world[0] * self.pred_dt
            x_sim[1] += curr_v_world[1] * self.pred_dt
            x_sim[2] += curr_v_world[2] * self.pred_dt
            x_sim[5] += yawrate_uav_ENU * self.pred_dt
            
            #This forces drone to slow near target.
            # Without velocity penalty, it will always overshoot.
            total += 5.0 * (curr_v_world.T @ curr_v_world)
        
        # Terminal penalty
        # This forces MPC to STOP at final point instead of blasting through.
        # e_terminal = Xref[self.N-1] - x_sim[0:3]
        e_terminal = Xref[self.N-1, 0:3] - x_sim[0:3]  #in ENU frame
        total += 20.0 * (e_terminal.T @ self.Q_pos @ e_terminal)

        return total

    # --- (Include all other helper functions like rel_pose_cb, quat_to_rpy, etc here) ---
    # self.create_subscription(PoseStamped, '/relative_pose_odom_OR_ekf', self.rel_pose_cb, 10)
    # this topic gives the position of the UGV w.r.t to the odom frame 
    def rel_pose_cb(self, msg: PoseStamped):
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        roll, pitch, yaw = self.quat_to_rpy_msg(msg.pose.orientation)
        # self.ugv_pos_and_orient_in_UAV_frame = np.array([x, y, z, roll, pitch, yaw], dtype=float)
        # This is the UGV position in odom frame
        self.ugv_position_in_odom_frame=np.array([x, y, z],dtype=float)
        
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
    mode_selection = "MPC"  # Set to "MPC" or "PID"
    node = Controller_for_UAV_Node(mode=mode_selection)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()