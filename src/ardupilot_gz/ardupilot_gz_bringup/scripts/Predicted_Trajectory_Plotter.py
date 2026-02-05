#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PredictedTrajectoryPlotter(Node):

    def __init__(self):
        super().__init__('predicted_trajectory_plotter')

        self.subscription = self.create_subscription(
            Path,
            '/predicted_trajectory',
            self.path_callback,
            10
        )

        # Storage for trajectory
        self.x_data = []
        self.y_data = []
        self.z_data = []

        # Matplotlib setup
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title('Predicted Trajectory')

        self.get_logger().info('Subscribed to /predicted_trajectory')

    def path_callback(self, msg: Path):
        # Clear previous data
        self.x_data.clear()
        self.y_data.clear()
        self.z_data.clear()

        for pose_stamped in msg.poses:
            p = pose_stamped.pose.position
            self.x_data.append(p.x)
            self.y_data.append(p.y)
            self.z_data.append(p.z)

        self.update_plot()

    def update_plot(self):
        self.ax.cla()

        self.ax.plot(
            self.x_data,
            self.y_data,
            self.z_data,
            marker='o'
        )

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title('Predicted Trajectory')

        plt.draw()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = PredictedTrajectoryPlotter()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

