#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class EKFPathPublisher(Node):
    def __init__(self):
        super().__init__('ekf_path_publisher')

        # Subscribe to EKF estimated relative pose
        self.sub = self.create_subscription(
            PoseStamped,
            '/relative_pose_ekf',   # EKF publishes PoseStamped here
            self.pose_callback,
            10
        )

        # Publish trajectory (Path) on a DIFFERENT topic
        self.path_pub = self.create_publisher(Path, '/relative_pose_ekf_path', 10)

        # Path message container
        self.path = Path()
        self.path.header.frame_id = "map"  # or your world frame

    def pose_callback(self, msg: PoseStamped):
        # Add new pose to the trajectory
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose

        self.path.poses.append(pose)
        self.path.header.stamp = msg.header.stamp

        # Publish updated path
        self.path_pub.publish(self.path)


def main(args=None):
    rclpy.init(args=args)
    node = EKFPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

