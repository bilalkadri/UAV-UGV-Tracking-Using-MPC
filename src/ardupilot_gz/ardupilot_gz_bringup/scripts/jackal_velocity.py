#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header

class CircleMover(Node):
    def __init__(self):
        super().__init__('jackal_circle_mover')
        # Publisher now publishes TwistStamped messages
        self.publisher_ = self.create_publisher(
            TwistStamped,
            '/jackal/jackal_velocity_controller/cmd_vel',
            10
        )
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.move_circle)
        self.get_logger().info("Moving Jackal in circular path...")

    def move_circle(self):
        msg = TwistStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        # Set desired circular motion
        msg.twist.linear.x = 0.5    # forward speed (m/s)
        msg.twist.angular.z = 0.5   # turning speed (rad/s)

        # Publish the command
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CircleMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

