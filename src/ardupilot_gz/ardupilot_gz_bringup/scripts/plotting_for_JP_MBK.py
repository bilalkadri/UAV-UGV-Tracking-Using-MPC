#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import TransformStamped # Added by Saudah
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy # Added by Saudah

class PlottingForJP(Node):
    def __init__(self):
        super().__init__('plottingforJP')
        
        qos_be = QoSProfile(depth=10) # Added by Saudah
        qos_be.reliability = ReliabilityPolicy.BEST_EFFORT # Added by Saudah
        qos_be.durability = DurabilityPolicy.VOLATILE # Added by Saudah

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers (Common Frame: odom)
        self.ugv_pub = self.create_publisher(PoseStamped, '/plotted/ugv_pose_in_odom_frame', 10)
        self.uav_pub = self.create_publisher(PoseStamped, '/plotted/uav_pose_in_odom_frame', 10)

        # Subscriptions
        self.create_subscription(Odometry, '/jackal/jackal_velocity_controller/odom', self.ugv_callback, 10)
        # self.create_subscription(PoseStamped, '/ap/pose/filtered', self.uav_callback, 10)    
        self.create_subscription(PoseStamped, '/ap/pose/filtered', self.uav_callback, qos_be) # Added by Saudah
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) # Added by Saudah
       




    def transform_and_publish_ugv(self, input_pose, target_frame, pub):
        try:
            # Lookup transform from input frame to target odom frame
            transform = self.tf_buffer.lookup_transform(
                target_frame, 
                input_pose.header.frame_id, 
                rclpy.time.Time()
            )
            # Perform transform
            transformed_pose = do_transform_pose(input_pose.pose, transform)
            
            # Prepare message for plotting
            out_msg = PoseStamped()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.header.frame_id = target_frame
            out_msg.pose = transformed_pose
            pub.publish(out_msg)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Could not transform: {e}')

    def transform_and_publish_uav(self, input_pose, target_frame, pub):
        try:
            # Lookup transform from input frame to target odom frame
            transform = self.tf_buffer.lookup_transform(
                target_frame, 
                input_pose.header.frame_id, 
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            # Perform transform
            transformed_pose = do_transform_pose(input_pose.pose, transform)
            
            # Prepare message for plotting
            out_msg = PoseStamped()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.header.frame_id = target_frame
            out_msg.pose = transformed_pose
            pub.publish(out_msg)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Could not transform: {e}')

    def ugv_callback(self, msg):
        # Convert Odometry to PoseStamped for do_transform_pose
        ugv_p = PoseStamped()
        ugv_p.header = msg.header
        ugv_p.pose = msg.pose.pose
        self.transform_and_publish_ugv(ugv_p, 'odom', self.ugv_pub)

    def uav_callback(self, msg):
        # A big confusion
        # You are absolutely right to call that out. If /ap/pose/filtered is expressing coordinates
        # in base_link (the drone's body frame), then x,y,z would almost always be near 
        # (0,0,0) because the drone is always at the origin of its own body.                                                                                                              always be near (0,0,0) because the drone is always at the origin of its own body.
        # However, in ArduPilot's ROS 2 implementation, there is a common naming confusion we 
        # need to address:
        # The Critical Distinction
        # ArduPilot's Internal Logic: ArduPilot calculates its position relative to the 
        # EKF Origin (where it was armed/turned on).

        # The ROS Message: Even though the frame_id says base_link, ArduPilot often populates 
        # the pose field with the World Position (Local NED or Global) relative to that EKF
        # origin.

        # The Logical Mismatch: If you look at your echo output from earlier, your z value
        #  was 1.97. If that were truly "relative to base_link," it would mean the drone is 2 meters
        # away from itself, which is impossible. Therefore, the data inside that message is 
        #  actually the drone's position in the "World" (the EKF's map), but it is mislabeled 
        # with the wrong frame_id.
        
        
        
        current_sim_time = self.get_clock().now().to_msg()

        # 1. Create a Transform from 'odom' to 'uav/base_link'
        # We treat the 'pose' inside the message as the translation from the origin.
        t = TransformStamped()
        t.header.stamp = current_sim_time
        t.header.frame_id = "odom"           # The Parent (World)
        t.child_frame_id = "uav/base_link"   # The Child (Drone)

        # These values from ArduPilot are actually the World-coordinates
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        
        # Broadcast this so the rest of ROS knows where the UAV is
        self.tf_broadcaster.sendTransform(t)  

        # 2. Publish the PoseStamped for your plotting script
        out = PoseStamped()
        out.header = t.header # Same time and same 'odom' frame
        out.pose.position = msg.pose.position
        out.pose.orientation = msg.pose.orientation
        
        self.uav_pub.publish(out)
        # self.transform_and_publish_uav(uav_p, 'odom', self.uav_pub)


def main(args=None):
    rclpy.init(args=args)
    node = PlottingForJP()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()