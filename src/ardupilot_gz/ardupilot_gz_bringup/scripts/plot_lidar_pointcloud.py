#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import matplotlib.pyplot as plt


class LidarPointCloud(Node):

    def __init__(self):
        super().__init__('lidar_pointcloud_plot')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/iris/lidar/points',
            self.lidar_callback,
            10
        )

        self.get_logger().info(
            'Waiting for LiDAR PointCloud2 data...'
        )

        self.data_received = False

    def lidar_callback(self, msg):

        if self.data_received:
            return

        self.data_received = True

        # Read X, Y, Z
        points = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )

        points = list(points)

        self.get_logger().info(
            f'Received {len(points)} valid 3D points'
        )

        if len(points) == 0:
            self.get_logger().warn(
                'No valid points received.'
            )
            return

        # Extract coordinates
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        z = [p[2] for p in points]

        # Print some points
        for i in range(min(20, len(points))):
            print(
                f'Point {i:03d}: '
                f'X={x[i]:8.3f} m, '
                f'Y={y[i]:8.3f} m, '
                f'Z={z[i]:8.3f} m'
            )

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            x,
            y,
            z,
            s=2
        )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        ax.set_title('Iris LiDAR 3D Point Cloud')

        plt.tight_layout()

        plt.show()


def main(args=None):

    rclpy.init(args=args)

    node = LidarPointCloud()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
