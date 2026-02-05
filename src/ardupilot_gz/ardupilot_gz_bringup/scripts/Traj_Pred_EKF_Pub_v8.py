#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import traceback
from math import radians
import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import TwistStamped

# ------------ helpers ------------
def rpy_to_rot(roll, pitch, yaw):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw); sy = np.sin(yaw)

    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp,   cp*sr,            cp*cr]])
    return R

# ============= ⭐⭐ NEW FUNCTION ⭐⭐ =============
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

# ------------ Node ------------
class RelativePoseEKF(Node):
    def __init__(self):

    # Constructor (__init__) does:
    # Initializes EKF: Creates state vector self.x (6×1) and covariance matrices P, Q, R
    # Sets up subscribers: For UAV/UGV poses, velocities, and ArUco detection
    # Sets up publishers: For relative pose, predicted trajectory, and EKF status
    # Initializes buffers: For storing UAV/UGV velocities and positions
    # Starts timer: Calls ekf_predict_publish() every dt (0.01s) seconds

        super().__init__('relative_pose_ekf_and_trajectory_prediction_node')

        # ============= ⭐⭐ CHANGE: State Interpretation ⭐⭐ =============
        # State is: UGV position RELATIVE TO UAV in UAV body frame
        # x[0:3] = UGV position in UAV body frame (x-forward, y-left, z-up)
        # x[3:6] = UGV orientation RELATIVE to UAV (Euler angles)
        self.x = np.zeros((6,1))
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.2

        pos_sigma = 0.04
        ang_sigma = radians(2.0)
        self.R = np.diag([
            pos_sigma**2, pos_sigma**2, pos_sigma**2,
            ang_sigma**2, ang_sigma**2, ang_sigma**2
        ])

        self.dt = 0.01  

        # Subscribers
        self.create_subscription(Imu, '/imu', self.imu_cb, 10)
        self.create_subscription(Odometry, '/odometry', self.odom_cb, 10)
        self.create_subscription(Odometry, '/jackal/jackal_velocity_controller/odom', self.ugv_pose_cb, 10)
        self.create_subscription(Bool, '/aruco/detected', self.aruco_detected_cb, 10)
        self.create_subscription(PoseStamped, '/aruco/pose', self.aruco_pose_cb, 10)
        
        # UAV Position subscriber
        self.create_subscription(PoseStamped, '/ap/pose/filtered', self.uav_pose_cb, 10)
        
        # ============= ⭐⭐ NEW: UAV Velocity subscriber ⭐⭐ =============
        self.create_subscription(TwistStamped, '/ap/twist/filtered', self.uav_twist_cb, 10)

        # Publishers
        self.pub_rel = self.create_publisher(PoseStamped, '/relative_pose_ekf', 10)
        self.pred_pub = self.create_publisher(Path, '/predicted_trajectory', 10)
        self.pub_update_flag = self.create_publisher(Bool, '/ekf/update_applied', 10)
        self.pub_maha = self.create_publisher(Float32, '/ekf/mahalanobis_distance', 10)
        self.debug_pub = self.create_publisher(PoseStamped, '/debug/ekf_ugv_world', 10)
       
        # Buffers
        self.v_g = np.zeros(3)      # UGV velocity (body frame)
        self.omega_g = np.zeros(3)  # UGV angular velocity (body frame)
        self.v_u = np.zeros(3)      # UAV velocity (body frame) ⭐⭐ CHANGED ⭐⭐
        self.omega_u = np.zeros(3)  # UAV angular velocity (body frame)

        # UAV state
        self.uav_pos = np.zeros(3)
        self.uav_yaw = 0.0  # ⭐⭐ NEW: Store UAV yaw ⭐⭐

        self.roll_g = 0.0
        self.pitch_g = 0.0
        self.yaw_g = 0.0

        self.pred_N = 12
        self.aruco_detected = False
        self.aruco_meas = None
        self.mahalanobis_threshold = 15.0

        self.create_timer(self.dt, self._timer_wrapper)
        self.get_logger().info("RelativePoseEKF node started.")

    def _timer_wrapper(self):
        try:
            self.ekf_predict_publish()
        except Exception:
            self.get_logger().error(
                "Exception in ekf_predict_publish:\n" + traceback.format_exc()
            )

    # Callbacks
    def uav_pose_cb(self, msg):
        # When this callback is executed:
        # Every time a message arrives on /ap/pose/filtered topic
        # Asynchronously - independent of the EKF timer
        # Updates self.uav_pos with UAV's current position (world frame)
        # Updates self.uav_yaw with UAV's heading from quaternion (world frame)
        # Used later by ekf_predict_publish() for transformations and predictions
        # Header says: frame_id: base_link (UAV body frame)
        # But coordinates are large: y: -27.095 suggests world frame
        # Ardupilot often reports pose in local frame but labels it as base_link
        # In practice: These are WORLD/NED coordinates (not body frame)
        # Conclusion: self.uav_pos stores world coordinates despite confusing frame_id label
        try:
            self.uav_pos[0] = msg.pose.position.x
            self.uav_pos[1] = msg.pose.position.y
            self.uav_pos[2] = msg.pose.position.z
            # ⭐⭐ NEW: Extract UAV yaw ⭐⭐
            self.uav_yaw = get_yaw_from_quat(msg.pose.orientation)
        except Exception:
            self.get_logger().error("Exception in uav_pose_cb:\n" + traceback.format_exc())

    # ⭐⭐ NEW: UAV velocity callback ⭐⭐
    def uav_twist_cb(self, msg):
        # When this callback is executed:
        # Every time a message arrives on /ap/twist/filtered topic
        # Asynchronously - independent of other callbacks
        # Updates self.v_u with UAV linear velocity (body frame)
        # Updates self.omega_u with UAV angular velocity (body frame)
        # Used in EKF prediction to calculate relative motion between UAV and UGV
        # Header says: frame_id: base_link
        # Standard ROS convention: Twist is ALWAYS in child_frame/body frame
        # Linear velocity: In UAV body frame (x-forward, y-left, z-up)
        # Angular velocity: In UAV body frame (roll-x, pitch-y, yaw-z)
        # Both self.v_u and self.omega_u are in UAV BODY FRAME
        try:
            self.v_u = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ])
            self.omega_u = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ])
        except Exception:
            self.get_logger().error("Exception in uav_twist_cb:\n" + traceback.format_exc())

    def imu_cb(self, msg):
        # Why IMU callback is commented out?
        # Using filtered velocities instead - /ap/twist/filtered is more accurate
        # IMU gives raw data - noisy and requires integration
        # Ardupilot already provides filtered velocities - better for prediction
        # Simplicity - one velocity source instead of fusing IMU with other sensors
        # Potential conflict - Using both could cause inconsistencies in velocity data
        # ⭐⭐ CHANGE: Use twist/filtered instead of IMU for velocity ⭐⭐
        pass  # We use /ap/twist/filtered now

    def odom_cb(self, msg):
        # What this function does?
        # Receives UGV odometry from /odometry topic
        # Extracts linear velocity → stores in self.v_g (base_link/body frame) 
        # Extracts angular velocity → stores in self.omega_g (base_link/body frame)
        # Provides UGV motion data for EKF prediction step
        # Updates asynchronously whenever new odometry data arrives
        try:
            self.v_g = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ])
            self.omega_g = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ])
        except Exception:
            self.get_logger().error("Exception in odom_cb:\n" + traceback.format_exc())

    #This was the old function
    # def ugv_pose_cb(self, msg):
    #     try:
    #         q = msg.pose.orientation
    #         self.roll_g, self.pitch_g, self.yaw_g = self.quat_to_rpy(q)
    #     except Exception:
    #         self.get_logger().error("Exception in ugv_pose_cb:\n" + traceback.format_exc())

    def ugv_pose_cb(self, msg):
        try:
            # Odometry has: msg.pose.pose.orientation (two .pose!)
            q = msg.pose.pose.orientation
            self.roll_g, self.pitch_g, self.yaw_g = self.quat_to_rpy(q)
            
            # Also get UGV position if needed:
            self.ugv_pos = [msg.pose.pose.position.x, 
                            msg.pose.pose.position.y,
                            msg.pose.pose.position.z]
        except Exception:
            self.get_logger().error("Exception in ugv_pose_cb:\n" + traceback.format_exc())

    def aruco_detected_cb(self, msg):
        # Get the new detection status
        new_detected = bool(msg.data)
        
        # If detection changes from True to False
        if self.aruco_detected and not new_detected:
            print("[EKF] Detection lost, clearing stale measurement")
            self.aruco_meas = None  # CLEAR THE MEASUREMENT
        
        # Update the detection status
        self.aruco_detected = new_detected
        print(f"[EKF] Detection status: {self.aruco_detected}")

    def aruco_pose_cb(self, msg):
                
        # Check if measurement is NaN (marker lost)
        if (math.isnan(msg.pose.position.x) or 
            math.isnan(msg.pose.position.y) or 
            math.isnan(msg.pose.position.z)):
            print("[EKF] Received NaN measurement - marker lost")
            self.aruco_meas = None  # Clear measurement
        else:
            # Valid measurement
            self.aruco_meas = msg
            print(f"[EKF] Received valid ArUco measurement: x={msg.pose.position.x:.3f}")

    # Main EKF procedure
    def ekf_predict_publish(self):
        # ============= ⭐⭐ CRITICAL FIX: Prediction Step ⭐⭐ =============
        # Your EKF state is in UAV body frame, so we need to handle it properly
        
        # --- Prediction step ---
        #    v_g is in UGV body frame (from /odometry), v_u is in UAV body frame (from /ap/twist/filtered)
        # CANNOT subtract directly - different frames! Need transformation
        # rel_vel = v_g - v_u is WRONG - mixing body frames
        # Must transform both to common frame first (UAV body frame)
        # Transform UGV velocity: v_g_uav_body = R_uav.T @ R_ugv @ v_g
        # Correct approach: Transform UGV velocity from UGV body frame to UAV body frame before subtraction.
     
        # rel_vel = self.v_g - self.v_u   
        # rel_omega = self.omega_g - self.omega_u

        # Transform relative velocity from world frame to UAV body frame
        # R_uav = np.array([
        #     [np.cos(self.uav_yaw), -np.sin(self.uav_yaw), 0],
        #     [np.sin(self.uav_yaw), np.cos(self.uav_yaw), 0],
        #     [0, 0, 1]
        # ])
        
        # rel_vel_body = R_uav.T @ rel_vel.reshape(3, 1)

        # # Update state (all in UAV body frame)
        # self.x[0:3, 0] += rel_vel_body.flatten() * self.dt
        # self.x[3:6, 0] += rel_omega * self.dt
        
        # # Wrap angles to [-pi, pi]
        # for i in range(3, 6):
        #     self.x[i, 0] = ((self.x[i, 0] + np.pi) % (2*np.pi)) - np.pi
        #-----------------------------------------------------------------------------
        #------------------------ABove this line is the old code-------------------
        #---------------Below is the new code---------------------------------
        # --- Prediction step ---
        # 

        # ============= ⭐⭐ ISSUE 1: Missing adaptive Q matrix ⭐⭐ =============
        # ADD THIS AT THE BEGINNING:
        # Update Q matrix based on current angular velocity
        omega_mag = np.linalg.norm(self.omega_u)
        rotation_noise_scale = 1.0 + omega_mag * 0.5  # More noise when rotating fast
        innovation = None
        P_diag = None

        # Base noise values (tune these!)
        pos_noise = 0.005
        ang_noise = 0.001
        
        self.Q = np.diag([
            pos_noise * rotation_noise_scale,
            pos_noise * rotation_noise_scale, 
            pos_noise * rotation_noise_scale,
            ang_noise * rotation_noise_scale,
            ang_noise * rotation_noise_scale,
            ang_noise * rotation_noise_scale
        ])



        # NO, this code is NOT transforming yaw. It's creating a ROTATION MATRIX:
        # self.yaw_g is UGV's heading angle (scalar, in radians) Get UGV yaw (from ugv_pose_cb)
        # R_ugv is a 3×3 rotation matrix that rotates vectors by yaw_g around Z-axis
        # Purpose: To transform vectors from UGV body frame to world frame
        # Example: If UGV has yaw = 30°, R_ugv rotates vectors by 30° around Z
        # Use: v_world = R_ugv @ v_body converts UGV body frame velocity to world frame
        R_ugv = np.array([
            [np.cos(self.yaw_g), -np.sin(self.yaw_g), 0],
            [np.sin(self.yaw_g), np.cos(self.yaw_g), 0],
            [0, 0, 1]
        ])
        
      
        # This creates UAV ROTATION MATRIX:
        # self.uav_yaw is UAV's heading angle (scalar, in radians)
        # R_uav is a 3×3 rotation matrix that rotates vectors by uav_yaw around Z-axis
        # Purpose: To transform vectors between UAV body frame and world frame
        # Forward: R_uav @ v_body → transforms from UAV body to world frame
        # Inverse: R_uav.T @ v_world → transforms from world to UAV body frame


        R_uav = np.array([
            [np.cos(self.uav_yaw), -np.sin(self.uav_yaw), 0],
            [np.sin(self.uav_yaw), np.cos(self.uav_yaw), 0],
            [0, 0, 1]
        ])
        
        # self.v_g is the UGV's velocity in the UGV's body frame
        # Transform UGV velocity from UGV body frame to world frame
        v_g_world = R_ugv @ self.v_g.reshape(3, 1)
        
        # Transform UGV velocity from world frame to UAV body frame
        v_g_uav_body = R_uav.T @ v_g_world
        
        # UAV velocity is already in UAV body frame
        v_u_body = self.v_u.reshape(3, 1)
        
        # Calculate relative velocity in UAV body frame
        rel_vel_body = v_g_uav_body - v_u_body
        
        # Angular velocities: both in body frames, but need transformation
        # Transform UGV angular velocity to UAV body frame

        # UGV body frame angular velocity → world frame → UAV body frame
        omega_g_uav_body = R_uav.T @ R_ugv @ self.omega_g.reshape(3, 1)

        # UAV angular velocity is already in UAV body frame
        omega_u_body = self.omega_u.reshape(3, 1)

        # relative angular velocity in UAV body frame
        rel_omega = omega_g_uav_body - omega_u_body
        
        # Update state (all in UAV body frame)
        # self.x[0:3, 0] = UGV position relative to UAV (in UAV body frame)
        # rel_vel_body * dt = displacement due to relative velocity over time step
        # First line: Updates relative position using relative linear velocity
        # self.x[3:6, 0] = UGV orientation relative to UAV (roll, pitch, yaw)
        # Second line: Updates relative orientation using relative angular velocity

        # Your current prediction step is missing the Coriolis/cross-coupling term that arises from rotating reference frames. When you have a position 
        # in a rotating frame (UAV body frame), the derivative has an additional term:
        # self.x[0:3, 0] += rel_vel_body.flatten() * self.dt

        # CRITICAL FIX: Include cross-coupling term
        # For position in rotating frame: dp/dt = v_rel - ω × p
        # ============= ⭐⭐ ISSUE 2: CORRECT CROSS PRODUCT TERM ⭐⭐ =============
        # ω × p where ω is angular velocity of the BODY FRAME (UAV)
        # This should be: dp/dt = v_rel - ω × p
        omega_cross_p = np.cross(omega_u_body.flatten(), self.x[0:3, 0].flatten())


        # Update position with complete kinematics

        self.x[0:3, 0] += (rel_vel_body.flatten() - omega_cross_p) * self.dt

        self.x[3:6, 0] += rel_omega.flatten() * self.dt
        
        #-------------------------------------------------------
        # --------------For Debugging Purpose-------------------
        #-------------------------------------------------------
        ugv_pos_body = self.x[0:3, 0].reshape(3, 1)
        ugv_pos_world = R_uav @ ugv_pos_body  # Transform to world frame

        debug_msg = PoseStamped()
        debug_msg.header.frame_id = 'odom'
        debug_msg.pose.position.x = float(ugv_pos_world[0])
        debug_msg.pose.position.y = float(ugv_pos_world[1])
        debug_msg.pose.position.z = float(ugv_pos_world[2])
        

        # Add orientation from EKF state:
        # self.x[3:6] contains [roll, pitch, yaw] relative to UAV
        roll, pitch, yaw = self.x[3:6, 0].flatten()

        # Convert to quaternion
        #yaw_to_ugv 
        qx, qy, qz, qw = self.rpy_to_quat(roll, pitch, yaw)
        debug_msg.pose.orientation.x = qx
        debug_msg.pose.orientation.y = qy
        debug_msg.pose.orientation.z = qz
        debug_msg.pose.orientation.w = qw
        self.debug_pub.publish(debug_msg)

        # Wrap angles to [-pi, pi]
        for i in range(3, 6):
            self.x[i, 0] = ((self.x[i, 0] + np.pi) % (2*np.pi)) - np.pi

        # ============= ⭐⭐ ISSUE 4: JACOBIAN CALCULATION PROBLEM ⭐⭐ =============
        # omega_u_body is a numpy array with shape (3,1), you need to flatten it
        omega_u_flat = omega_u_body.flatten()
        
        F = np.eye(6)
        
        # Position part: ∂f/∂p = I - [ω×] * dt
        omega_skew = np.array([
            [0, -omega_u_flat[2], omega_u_flat[1]],
            [omega_u_flat[2], 0, -omega_u_flat[0]],
            [-omega_u_flat[1], omega_u_flat[0], 0]
        ])
        
        F[0:3, 0:3] = np.eye(3) - omega_skew * self.dt
        # ============= ⭐⭐ ISSUE 5: ADD ∂f/∂v TERM ⭐⭐ =============
        # Your state has velocity implicitly in the prediction
        # Add the Jacobian for velocity terms if you're using a velocity state
        # If not using velocity state, you need to propagate uncertainty from velocity noise
        # F[0:3, 3:6] = np.eye(3) * self.dt  # Uncomment if using velocity state
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q
    
        
        # Covariance update: self.P = (self.P + self.Q) * 0.98
        # Adds process noise Q to covariance P (uncertainty grows with prediction)
        # Multiplies by 0.98 - Small damping to prevent covariance explosion
        # self.P = (self.P + self.Q) * 0.98

        # --- Update step (ArUco) ---
        update_applied = False   # Flag to track if update was performed
        maha_value = -1.0  # Default Mahalanobis distance value (no measurement)

        # Check if ArUco marker is detected and measurement exists
        # if self.aruco_detected and (self.aruco_meas is not None):
        if (self.aruco_detected and  self.aruco_meas is not None and
             not math.isnan(self.aruco_meas.pose.position.x)):
            try:
                print(f"[EKF UPDATE] Starting update - detection=True, measurement valid")
                # Extract orientation quaternion from ArUco measurement
                q = self.aruco_meas.pose.orientation
                # Convert quaternion to Euler angles (roll, pitch, yaw)
                meas_roll, meas_pitch, meas_yaw = self.quat_to_rpy(q)

                # # Transform from camera frame to UAV body frame if needed
                # # This depends on your camera mounting
                # # Example: if camera is mounted with 180° rotation around Z
                # camera_to_body = np.array([
                #     [-1, 0, 0],
                #     [0, -1, 0],
                #     [0, 0, 1]
                # ])
                
                # # Transform measurement
                # pos_camera = np.array([
                #     self.aruco_meas.pose.position.x,
                #     self.aruco_meas.pose.position.y,
                #     self.aruco_meas.pose.position.z
                # ])
                
                # pos_body = camera_to_body @ pos_camera


                # Create measurement vector z = [x, y, z, roll, pitch, yaw] from ArUco
                z = np.array([
                    [-self.aruco_meas.pose.position.x],
                    [-self.aruco_meas.pose.position.y],
                    [-self.aruco_meas.pose.position.z],
                    [meas_roll],
                    [meas_pitch],
                    [meas_yaw]
                ])
                # Now use pos_body in measurement vector
                # z = np.array([
                #     [pos_body[0]],
                #     [pos_body[1]],
                #     [pos_body[2]],
                #     [meas_roll],
                #     [meas_pitch],
                #     [meas_yaw]
                # ])



                # print(f"ArUco measurement: x={self.aruco_meas.pose.position.x:.2f}, y={self.aruco_meas.pose.position.y:.2f}, z={self.aruco_meas.pose.position.z:.2f}")
                # Calculate innovation (difference between measurement and prediction)
                y = z - self.x
                innovation = y.copy()
                # Wrap angular differences to range [-π, π] to avoid discontinuity
                for idx in range(3,6):
                    ang = (float(y[idx,0]) + np.pi) % (2*np.pi) - np.pi
                    y[idx,0] = ang
                

                # Calculate innovation covariance S = P + R
                S = self.P + self.R

                # Compute inverse of innovation covariance (for Kalman gain)
                try:
                    Sinv = np.linalg.inv(S)   # Try regular inverse
                except:
                    Sinv = np.linalg.pinv(S)   # Use pseudo-inverse if matrix is singular

                # Calculate Mahalanobis distance = y^T * S^(-1) * y
                maha_value = float((y.T @ Sinv @ y)[0,0])

                # Check if measurement is valid (not an outlier)
                if maha_value <= self.mahalanobis_threshold:
                    print(f"[EKF UPDATE] Mahalanobis distance: {maha_value:.3f}, Threshold: {self.mahalanobis_threshold}")
                    # Compute Kalman gain K = P * S^(-1)
                    K = self.P @ Sinv
                    # Update state estimate: x = x + K * y
                    self.x = self.x + K @ y
                     # Update covariance: P = (I - K) * P
                    self.P = (np.eye(6) - K) @ self.P
                    P_diag = np.diag(self.P).copy()

                    # Set flag indicating update was applied
                    update_applied = True

            except Exception:
                self.get_logger().error("Exception during EKF update:\n" + traceback.format_exc())

        # Publish Mahalanobis distance
        maha_msg = Float32()
        maha_msg.data = float(maha_value)
        self.pub_maha.publish(maha_msg)

        # Publish update-applied flag
        flag = Bool()
        flag.data = update_applied
        self.pub_update_flag.publish(flag)

        # --- Publish EKF estimate (relative pose) ---
        # ============= ⭐⭐ FIX: Publish at UAV position pointing to UGV ⭐⭐ =============
        msg = PoseStamped()  #Creates empty PoseStamped message - for publishing relative pose
        msg.header.stamp = self.get_clock().now().to_msg()
        # FIX 1: Set the Frame ID to the UAV's body frame
        msg.header.frame_id = 'base_link' #Sets frame_id to 'base_link' - message coordinates are in UAV frame
        
        # Position: UAV location
        # Sets position to UAV location - uses UAV's current world coordinates
        # msg.pose.position.x = float(self.uav_pos[0])  # self.uav_pos is the  UAV's current position (world frame)
        # msg.pose.position.y = float(self.uav_pos[1])
        # msg.pose.position.z = float(self.uav_pos[2])

        # FIX 2: Set the position to (0,0,0) so the arrow starts exactly at UAV base_link's origin
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0
        

        # Get relative position from EKF
        dx, dy, dz = self.x[0:3, 0].flatten()

        # DEBUG: Print for verification
        
        print(f"ArUco meas - x:{self.aruco_meas.pose.position.x if self.aruco_meas else 'None'}")
        print(f"=== EKF VERIFICATION ===")
        print(f"EKF state - dx:{dx:.2f}, dy:{dy:.2f}, dz:{dz:.2f}")
        # print(f"ArUco measurement: x={self.aruco_meas.pose.position.x:.2f}, y={self.aruco_meas.pose.position.y:.2f}, z={self.aruco_meas.pose.position.z:.2f}")
        print(f"UGV Odometry velocity: vx={ self.ugv_pos[0]:.3f}, vy={self.ugv_pos[1]:.3f}, vz={self.ugv_pos[2]:.3f}")
        print(f"EKF Innovation: {innovation}")
        print(f"EKF Covariance diag: {P_diag}")
        if self.aruco_meas is not None:
            print(
                f"ArUco measurement: "
                f"x={self.aruco_meas.pose.position.x:.2f}, "
                f"y={self.aruco_meas.pose.position.y:.2f}, "
                f"z={self.aruco_meas.pose.position.z:.2f}"
            )
        else:
            print("ArUco measurement: None (no detection)")


        #------------------------------------------------------------
        #          Required for RViz Visualization
        #---------------------------------------------------------
        # Orientation: Point from UAV to predicted UGV position
        # Transform relative position from body to world frame
        # self.x[0:3] is in UAV body frame - UGV's position relative to UAV

        

        rel_pos_body = self.x[0:3, 0].reshape(3, 1)

        #----------------------------------------------------------------
        # ------------------This is all incorrect information------------
        #---------------------------------------------------------------- 
        # RVIZ needs world coordinates - everything must be in 'odom' frame
        # R_uav @ rel_pos_body transforms from UAV body frame to world frame
        # Result: Gets UGV position in world coordinates for visualization
        # Without this: The arrow would point in wrong direction in RVIZ
        # We don't need rel_pos_world, we need the angles implied by rel_pos_body, 
        # but since the message is now in base_link, the vector (rel_pos_body) IS the direction.
        # rel_pos_world = R_uav @ rel_pos_body
        

        # Calculate direction vector
        # The rotation calculation should use the vector *as defined in base_link* # 
        # because the message is published in base_link.
        dx, dy, dz = rel_pos_body.flatten()
        print(f"Relative position: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
       
        # Calculate yaw and pitch relative to the base_link frame 
        # (which is where the message is published)

        # Temporarily hardcode a known position:
        # Test: UGV should be 5m in front, 0m to side, 5m below
        # dx, dy, dz = 5.0, 0.0, -5.0
        yaw_to_ugv = np.arctan2(dy, dx)  # Should be 0° (straight ahead)
        # Arrow should point straight ahead

        # # Add 180° to yaw to point opposite direction
        # yaw_to_ugv = np.arctan2(dy, dx) + np.pi  # Add 180 degrees
        # # Wrap to [-π, π]
        # yaw_to_ugv = ((yaw_to_ugv + np.pi) % (2*np.pi)) - np.pi

        # Pitch: angle relative to the X-Y plane
        # Note: pitch is usually zero for horizontal tracking
        pitch_to_ugv = -np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        
        qx, qy, qz, qw = self.rpy_to_quat(0.0, pitch_to_ugv, yaw_to_ugv)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        
          #-----------------Debug INfo-----------------------
        # Add these prints:
        print(f"=== ARROW DEBUG ===")
        print(f"dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
        print(f"yaw_to_ugv (degrees): {np.degrees(yaw_to_ugv):.1f}°")
        print(f"pitch_to_ugv (degrees): {np.degrees(pitch_to_ugv):.1f}°")
        print(f"Quaternion: x={qx:.3f}, y={qy:.3f}, z={qz:.3f}, w={qw:.3f}")
        print(f"=== END DEBUG ===")



        self.pub_rel.publish(msg)

        # --- Trajectory prediction ---
        # ============= ⭐⭐ CRITICAL FIX: Trajectory prediction ⭐⭐ =============
        # Predict UAV trajectory in WORLD frame
        traj_world = self.predict_desired_uav_trajectory(
            self.x.flatten(), 
            self.uav_pos, self.uav_yaw,
            self.v_g, self.omega_g,
            self.v_u, self.omega_u,
            N=self.pred_N, dt=self.dt
        )

        # Publish predicted path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'

        for pos in traj_world:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(pos[0])
            ps.pose.position.y = float(pos[1])
            ps.pose.position.z = float(pos[2])
            
            # For NMPC, orientation might not matter, but set to identity
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.pred_pub.publish(path_msg)

    # ============= ⭐⭐ NEW: Proper trajectory prediction ⭐⭐ =============
    def predict_ugv_trajectory_world(self, state_vec, uav_pos, uav_yaw, 
                                     v_g, w_g, v_u, w_u, N=50, dt=0.02):
        """
        Predict UGV trajectory in WORLD frame
        Returns: List of [x, y, z] positions in world frame
        """
        # Current relative state
        x_rel, y_rel, z_rel = state_vec[0:3]
        
        # Current UAV rotation matrix (simplified 2D)
        R_uav = np.array([
            [np.cos(uav_yaw), -np.sin(uav_yaw), 0],
            [np.sin(uav_yaw), np.cos(uav_yaw), 0],
            [0, 0, 1]
        ])
        
        # Current UGV position in world frame
        rel_pos_body = np.array([[x_rel], [y_rel], [z_rel]])
        ugv_pos_world = uav_pos.reshape(3, 1) + R_uav @ rel_pos_body
        
        trajectory = []
        current_ugv_pos = ugv_pos_world.flatten()
        current_v_g = v_g.copy()
        
        # Predict future UGV positions
        for i in range(N):
            # Simple constant velocity model for UGV
            current_ugv_pos += current_v_g * dt
            
            trajectory.append(current_ugv_pos.copy())
            
            # Optional: Add velocity prediction (constant acceleration model)
            # current_v_g += ... * dt
        print(f"UAV pos: {self.uav_pos}")
        
        return trajectory

    # ============= ⭐⭐ NEW: Alternative - Predict desired UAV trajectory ⭐⭐ =============
    def predict_desired_uav_trajectory(self, state_vec, uav_pos, uav_yaw,
                                       v_g, w_g, v_u, w_u, N=50, dt=0.02):
        """
        Predict desired UAV trajectory for NMPC (to track UGV)
        Returns: List of [x, y, z] positions where UAV should be
        """
        # First predict UGV trajectory
        ugv_traj = self.predict_ugv_trajectory_world(
            state_vec, uav_pos, uav_yaw, v_g, w_g, v_u, w_u, N, dt
        )
        
        # Desired offset from UGV (adjust based on your tracking requirements)
        # Example: Maintain 2m behind and 2m above UGV
        # desired_offset = np.array([-2.0, 0.0, 2.0])  # In UAV body frame
        # Current: [-2.0, 0.0, 2.0] - too close, try:
        desired_offset = np.array([5.0, 0.0, 10.0])  # 5m behind, 10m above
        
        uav_trajectory = []
        R_uav = np.array([
            [np.cos(uav_yaw), -np.sin(uav_yaw), 0],
            [np.sin(uav_yaw), np.cos(uav_yaw), 0],
            [0, 0, 1]
        ])
        
        for ugv_pos in ugv_traj:
            # Transform offset to world frame and add to UGV position
            offset_world = R_uav @ desired_offset.reshape(3, 1)
            desired_uav_pos = ugv_pos + offset_world.flatten()
            uav_trajectory.append(desired_uav_pos)
        
        return uav_trajectory

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


def main(args=None):
    rclpy.init(args=args)
    node = RelativePoseEKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()