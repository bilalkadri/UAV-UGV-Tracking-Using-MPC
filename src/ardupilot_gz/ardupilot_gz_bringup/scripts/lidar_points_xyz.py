#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class LidarXYZ(Node):

    def __init__(self):
        super().__init__('lidar_xyz')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/iris/lidar/points',
            self.lidar_callback,
            10
        )

        self.get_logger().info(
            'Subscribed to /iris/lidar/points'
        )

    def lidar_callback(self, msg):

        points = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )

        points = list(points)

        self.get_logger().info(
            f'Valid 3D points: {len(points)}'
        )

        for i, point in enumerate(points[:20]):

            x = point[0]
            y = point[1]
            z = point[2]

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
