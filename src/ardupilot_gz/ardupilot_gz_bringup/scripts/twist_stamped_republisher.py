#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

class TwistStampedRepublisher(Node):
    def __init__(self):
        super().__init__('twist_stamped_republisher')
        self.sub = self.create_subscription(
            Twist, '/cmd_vel', self.cb, 10)
        self.pub = self.create_publisher(
            TwistStamped, '/jackal/jackal_velocity_controller/cmd_vel', 10)

    def cb(self, msg):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.twist = msg
        self.pub.publish(stamped)

def main():
    rclpy.init()
    node = TwistStampedRepublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
