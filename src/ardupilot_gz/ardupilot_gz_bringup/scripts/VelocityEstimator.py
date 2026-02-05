#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

# ============================================================
# ================= JACKAL VELOCITY ESTIMATOR ================
# ============================================================

class VelocityEstimator(Node):
    def __init__(self):
        super().__init__('velocity_estimator')

        self.jackal_vel = None
        self.jackal_omega = None

        # subscribe to Jackal odometry
        self.create_subscription(
            Odometry,
            '/odometry',
            self.jackal_odom_cb,
            10
        )

        self.get_logger().info("VelocityEstimator node started.")

    def jackal_odom_cb(self, msg: Odometry):
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular

        self.jackal_vel = np.array([lv.x, lv.y, lv.z])
        self.jackal_omega = np.array([av.x, av.y, av.z])

        print(f"[JACKAL] Linear Velocity (m/s): {self.jackal_vel}")
        print(f"[JACKAL] Angular Velocity (rad/s): {self.jackal_omega}")


# ============================================================
# =========================== MAIN ===========================
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    node = VelocityEstimator()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

