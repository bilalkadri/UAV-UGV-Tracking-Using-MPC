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

# ------------ Node ------------
class UGV_Pose_from_Sensor_Data(Node):
    def __init__(self):
        super().__init__('UGV_Pose_from_Sensor_Data_node')
       
       # This Node will provide the UGV position in Odom frame

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        #----------------------------------------------------------------------------------------
        #                         Class Variables 
        #----------------------------------------------------------------------------------------

        
        # Variables to store UGV information 

        self.ugv_position_in_odom_frame =  [0.0, 2.0, 0.0]   # Store UGV position by transforming data from /jacakal/base_link to /odom frame 
        self.ugv_yaw_in_odom_frame=0.0
        self.ugv_roll_in_jackal_odom_frame =0.0
        self.ugv_pitch_in_jackal_odom_frame =0.0
        self.ugv_yaw_in_jackal_odom_frame = [0.0,0.0,0.0]

        self.ugv_position_in_jackal_odom_frame=[0.0,0.0,0.0]
        self.trajectory = []  # Stores the predicted trajectory
        
        
       
        self.ugv_lin_vel_in_jackal_base_link_frame =[0.0,0.0,0.0]
        self.ugv_ang_vel_in_jackal_base_link_frame =[0.0,0.0,0.0]
      

        #----------------------------------------------------------------------------------------
        #                                       tf transformation
        #----------------------------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
      

        #----------------------------------------------------------------------------------------
        #                                       Subscribers
        #----------------------------------------------------------------------------------------
        #/jackal/odom is a FIXED frame (does not move with the UGV)
        #/jackal/odom is fixed at the initial position of the UGV
        self.create_subscription(Odometry, '/jackal/jackal_velocity_controller/odom', self.ugv_pose_cb, 10) # in jackal/odom frame

        #----------------------------------------------------------------------------------------
        #                                              Publishers
        #----------------------------------------------------------------------------------------
        self.pub_rel = self.create_publisher(PoseStamped, '/relative_pose_odom_OR_ekf', 10) # publishing in base_link frame
        
        self.pub_rel_only_odom = self.create_publisher(PoseStamped, '/relative_pose_odom', 10) # publishing in base_link frame
        
        self.pred_pub = self.create_publisher(Path, '/predicted_trajectory', 10)

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
    
    def rpy_to_quat(self, roll, pitch, yaw):
        cy = np.cos(yaw*0.5); sy = np.sin(yaw*0.5)
        cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
        cr = np.cos(roll*0.5); sr = np.sin(roll*0.5)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        return (qx, qy, qz, qw)

    def ugv_pose_cb(self, msg):
        # """Handle UGV odometry messages"""
        try:
            # self.create_subscription(Odometry, '/jackal/jackal_velocity_controller/odom', self.ugv_pose_cb, 10) 
            # in jackal/odom frame
            # Extract UGV orientation from /jackal/odom  (existing)
            #/jackal/odom is a FIXED frame (does not move with the UGV)
            #/jackal/odom is fixed at the initial position of the UGV
            #/jackal/base_link moves with the UGV(jackal)

            #--------------------------------------------------------------------
            #           Very Important Information
            #-----------------------------------------------------------------------
            #  /jackal/jackal_velocity_controller/odom is a nav_msgs/Odometry message 
            # An Odometry message always contains two different frames by design.
            #"header": {
            # "frame_id": "jackal/odom"
            # },
            # "child_frame_id": "jackal/base_link"
            # Pose i.e. Position and orientation are expressed in jackal/odom frame
            # Twist i.e. Velocities in nav_msgs/Odometry are expressed in the child_frame_id frame
            #  i.e. jackal/base_link frame
                   
            # Therefore the position (pose) of UGV is in jackal/odom frame whereas the 
            # (Twist) velocity of UGV is in jackal/base_link frame 

            self.ugv_position_in_jackal_odom_frame=np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

            # 2. Velocity (Note: Usually expressed in jackal/base_link frame)
            
            self.ugv_lin_vel_in_jackal_base_link_frame = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ])
            self.ugv_ang_vel_in_jackal_base_link_frame = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ])
            q = msg.pose.pose.orientation
            self.ugv_roll_in_jackal_odom_frame, self.ugv_pitch_in_jackal_odom_frame, self.ugv_yaw_in_jackal_odom_frame = self.quat_to_rpy(q)
            
          
            # ============= ⭐⭐ GET WORLD POSITION VIA TF ⭐⭐ =============
            # ugv_world_pos is in the /odom frame
            # ugv_world_pos = self.get_ugv_world_position()

            # self.ugv_position_in_odom_frame=self.get_ugv_world_position()
            #  When the above function is called all the data will be automatically stored in class variables:

            self.get_ugv_position_in_odom_frame()
            self.publishing_the_data_on_ROS_topic()
          
            x0=[self.ugv_position_in_jackal_odom_frame[0],self.ugv_position_in_jackal_odom_frame[1],self.ugv_yaw_in_jackal_odom_frame]
            # ✅ Linear velocity → linear.x
            # Linear velocity in base_link frame
            # v = msg.twist.twist.linear.x      # m/s
            # ✅ Angular velocity → angular.z
            #  Angular velocity about z-axis (yaw rate) in base_link
            # w = msg.twist.twist.angular.z     # rad/s
           
            v=self.ugv_lin_vel_in_jackal_base_link_frame[0]
            w=self.ugv_ang_vel_in_jackal_base_link_frame[2]
            
            self.predict_ugv_trajectory_in_odom_frame_using_Sensor_Data(x0, v, w, dt=0.15, N=10)
                 
        except Exception as e:
            print(f"[ERROR] ugv_pose_cb: {str(e)}")
            traceback.print_exc()
    
    def publishing_the_data_on_ROS_topic(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        
        # Position
        msg.pose.position.x = self.ugv_position_in_odom_frame[0]
        msg.pose.position.y = self.ugv_position_in_odom_frame[1]
        msg.pose.position.z = self.ugv_position_in_odom_frame[2]
        
        # Orientation
        # Convert to quaternion, the orientation of the UGV w.r.t the odom frame 
        qx, qy, qz, qw = self.rpy_to_quat(0, 0, self.ugv_yaw_in_odom_frame)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        
        # Publish to the same topic NMPC uses
        #  self.pub_rel = self.create_publisher(PoseStamped, '/relative_pose_odom_OR_ekf', 10)
        self.pub_rel.publish(msg)   # publishing on the topic  /relative_pose_odom_OR_ekf in /odom frame
        self.pub_rel_only_odom.publish(msg) # publishing on the topic  /relative_pose_odom in /odom frame
 

    def get_ugv_position_in_odom_frame(self):
        """Get UGV position in world frame , remember world frame is odom, using TF"""
        try:
            
            # This code is getting the transform between two different odometry frames-specifically, 
            # it's finding where jackal/base_link is located in the global odom frame
            # REmember: jackal/odom is the initial positon of the jackal in the simulation world
            # jackal/base_link is fixed on top of the UGV. SO when UGV moves jackal/base_link moves as well

            # print("I am in get_ugv_position_in_odom_frame")
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
    

            # --- ORIENTATION (Combined Yaw) ---
            # Get yaw from the local movement (t1) using your function
            yaw_local = self.get_yaw_from_quat(t1.transform.rotation)
            
            # Get yaw from the starting offset (t2) using your function
            yaw_offset = self.get_yaw_from_quat(t2.transform.rotation)

            # Total Yaw = Offset Yaw + Local Yaw
            combined_yaw = yaw_offset + yaw_local
            
            # Normalize the result to stay within [-pi, pi]
            self.ugv_yaw_in_odom_frame = np.arctan2(np.sin(combined_yaw), np.cos(combined_yaw))
         
       
               
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            print(f"❌❌❌❌❌[TF] Error getting transform❌❌❌❌❌: {str(e)}")


    def predict_ugv_trajectory_in_odom_frame_using_Sensor_Data(self,x0, v, w, dt, N):
        """
        State vector (in jackal/odom frame):
            x_k = [ x_k, y_k, θ_k ]ᵀ

        Inputs (measured in jackal/base_link frame):
            v_k = linear velocity
            ω_k = angular velocity

        Discrete-time kinematic model:

            x_{k+1} = x_k + v_k cos(θ_k) Δt
            y_{k+1} = y_k + v_k sin(θ_k) Δt
            θ_{k+1} = θ_k + ω_k Δt
        """

        traj = np.zeros((N + 1, 3))

        # Initial condition:
        # x_0 = [x_0, y_0, θ_0]
        traj[0] = x0

        for k in range(N):
            x_k, y_k, theta_k = traj[k]

            # Coordinate transformation (base_link → odom):
            #   ẋ = v cos(θ)
            #   ẏ = v sin(θ)
            #   θ̇ = ω

            # Forward Euler discretization:
            #   x_{k+1} = x_k + ẋ Δt
            #   y_{k+1} = y_k + ẏ Δt
            #   θ_{k+1} = θ_k + θ̇ Δt

            x_next = x_k + v * np.cos(theta_k) * dt
            y_next = y_k + v * np.sin(theta_k) * dt
            theta_next = theta_k + w * dt

            # print('x_next=',x_next)
            # print('y_next=',y_next)
            # print('theta_next=',theta_next)
            # The traj[k] is in the jackal/odom frame
            traj[k + 1] = [x_next, y_next, theta_next]
            self.trajectory.append([x_next, y_next, theta_next])

        # return traj


        # ------------------------------------------------------------
        # Publish the predicted UGV trajectory expressed entirely
        # in the UAV body frame.
        # This trajectory is used directly by the NMPC.
        # ------------------------------------------------------------
        # Publish predicted path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
            # ⚠️  WARNING:This is incorrect, the frame_id should be base_link
        path_msg.header.frame_id = 'jackal/odom'

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



def main(args=None):
    rclpy.init(args=args)
    node = UGV_Pose_from_Sensor_Data()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()