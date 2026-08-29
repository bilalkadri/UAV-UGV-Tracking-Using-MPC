#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarXYZ(Node):

    def __init__(self):
        super().__init__('lidar_xyz')

        self.subscription = self.create_subscription(
            LaserScan,
            '/iris/lidar',
            self.lidar_callback,
            10
        )

        self.get_logger().info('Subscribed to /iris/lidar')

    def lidar_callback(self, msg):

        points = []

        for i, r in enumerate(msg.ranges):

            # Ignore invalid measurements
            if not math.isfinite(r):
                continue

            if r < msg.range_min or r > msg.range_max:
                continue

            # Calculate angle
            theta = msg.angle_min + i * msg.angle_increment

            # Convert polar -> Cartesian
            x = r * math.cos(theta)
            y = r * math.sin(theta)

            # LaserScan is planar
            z = 0.0

            points.append((x, y, z))

        # Print first 20 valid points
        self.get_logger().info(
            f'Valid points: {len(points)}'
        )

        for i, (x, y, z) in enumerate(points[:20]):
            print(
                f'Point {i:03d}: '
                f'X={x:8.3f} m, '
                f'Y={y:8.3f} m, '
                f'Z={z:8.3f} m'
            )


def main(args=None):

    rclpy.init(args=args)

    node = LidarXYZ()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
