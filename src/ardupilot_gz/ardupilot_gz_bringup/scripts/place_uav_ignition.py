#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from ros_gz_interfaces.srv import SetEntityPose

class PlaceDroneIgn(Node):
    def __init__(self):
        super().__init__('place_drone_ign')

        # ✅ Gazebo Ignition service
        self.cli = self.create_client(SetEntityPose, '/world/map/set_pose')

        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('⏳ Waiting for /world/default/set_pose service...')

        # Call once after startup
        self.timer = self.create_timer(3.0, self.place_drone_once)

    def place_drone_once(self):
        self.get_logger().info('🚁 Placing drone on top of Husky...')

        req = SetEntityPose.Request()
        req.entity.name = 'iris'       # <-- Your UAV model name
        req.entity.type = 2            # 2 = model entity type

        # Adjust these according to your Husky’s position
        req.pose.position.x = 0.0
        req.pose.position.y = 0.0
        req.pose.position.z = 0.4      # about 0.3–0.5 m above the UGV
        req.pose.orientation.w = 1.0   # facing forward

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('✅ Drone repositioned successfully!')
        else:
            self.get_logger().error('❌ Failed to move drone.')

        # Stop the timer so it only runs once
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = PlaceDroneIgn()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
