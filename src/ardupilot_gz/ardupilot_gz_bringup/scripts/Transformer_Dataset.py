#!/usr/bin/env python3

import os
import csv
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from tf_transformations import euler_from_quaternion


class AprilTagLidarDataset(Node):

    def __init__(self):
        super().__init__('apriltag_lidar_dataset')

        # ---------------------------------------------------------
        # Topics
        # ---------------------------------------------------------
        self.aruco_topic = '/aruco/pose'
        self.jackal_topic = '/jackal/jackal_velocity_controller/odom'
        self.uav_topic = '/ap/pose/filtered'
        self.lidar_topic = '/iris/lidar/points'

        # ---------------------------------------------------------
        # CSV output
        # ---------------------------------------------------------
        self.output_dir = os.path.expanduser(
            '~/ardu_ws/src/ardupilot_gz/ardupilot_gz_bringup/csvs'
        )

        os.makedirs(self.output_dir, exist_ok=True)

        self.csv_file_path = os.path.join(
            self.output_dir,
            'apriltag_lidar_relative_pose_dataset.csv'
        )

        self.csv_file = open(
            self.csv_file_path,
            'w',
            newline=''
        )

        self.writer = csv.writer(self.csv_file)

        # ---------------------------------------------------------
        # CSV header
        # ---------------------------------------------------------
        self.writer.writerow([
            'timestamp',

            # AprilTag features
            'tag_x',
            'tag_y',
            'tag_z',
            'tag_distance',
            'tag_yaw',

            # LiDAR features
            'lidar_min_range',
            'lidar_mean_range',
            'lidar_std_range',
            'lidar_median_range',
            'lidar_valid_ratio',

            # Labels
            'relative_x',
            'relative_y'
        ])

        self.csv_file.flush()

        # ---------------------------------------------------------
        # Latest measurements
        # ---------------------------------------------------------
        self.latest_tag = None
        self.latest_lidar = None
        self.latest_jackal = None
        self.latest_uav = None

        # Prevent writing the exact same measurements repeatedly
        self.last_written_time = None

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------
        self.tag_sub = self.create_subscription(
            PoseStamped,
            self.aruco_topic,
            self.aruco_callback,
            10
        )

        self.jackal_sub = self.create_subscription(
            Odometry,
            self.jackal_topic,
            self.jackal_callback,
            10
        )

        self.uav_sub = self.create_subscription(
            PoseStamped,
            self.uav_topic,
            self.uav_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            10
        )

        # Check whether all measurements are available
        self.timer = self.create_timer(
            0.02,       # 50 Hz maximum dataset rate
            self.process_data
        )

        self.get_logger().info(
            'AprilTag + LiDAR dataset collector started.'
        )

        self.get_logger().info(
            f'CSV file: {self.csv_file_path}'
        )

    # =============================================================
    # AprilTag callback
    # =============================================================

    def aruco_callback(self, msg):

        p = msg.pose.position
        q = msg.pose.orientation

        # Quaternion -> Euler
        quaternion = [
            q.x,
            q.y,
            q.z,
            q.w
        ]

        roll, pitch, yaw = euler_from_quaternion(quaternion)

        distance = math.sqrt(
            p.x ** 2 +
            p.y ** 2 +
            p.z ** 2
        )

        self.latest_tag = {
            'x': p.x,
            'y': p.y,
            'z': p.z,
            'distance': distance,
            'yaw': yaw
        }

    # =============================================================
    # Jackal callback
    # =============================================================

    def jackal_callback(self, msg):

        p = msg.pose.pose.position

        self.latest_jackal = {
            'x': p.x,
            'y': p.y,
            'z': p.z
        }

    # =============================================================
    # UAV callback
    # =============================================================

    def uav_callback(self, msg):

        p = msg.pose.position

        self.latest_uav = {
            'x': p.x,
            'y': p.y,
            'z': p.z
        }

    # =============================================================
    # LiDAR callback
    # =============================================================

    def lidar_callback(self, msg):

        x_values = []
        y_values = []
        z_values = []

        try:

            # Read x, y, z directly from PointCloud2
            points = point_cloud2.read_points(
                msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            )

            for point in points:

                x, y, z = point

                # Ignore invalid values
                if not (
                    math.isfinite(x) and
                    math.isfinite(y) and
                    math.isfinite(z)
                ):
                    continue

                x_values.append(x)
                y_values.append(y)
                z_values.append(z)

        except Exception as e:

            self.get_logger().error(
                f'LiDAR processing error: {e}'
            )

            return

        if len(x_values) == 0:
            return

        # Convert to numpy arrays
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)
        z_values = np.asarray(z_values)

        # Euclidean distance of each LiDAR point
        ranges = np.sqrt(
            x_values ** 2 +
            y_values ** 2 +
            z_values ** 2
        )

        # Remove zero and unrealistic ranges
        valid = (
            np.isfinite(ranges) &
            (ranges > 0.05) &
            (ranges < 90.0)
        )

        ranges = ranges[valid]

        if len(ranges) == 0:
            return

        # Total expected points from your LiDAR
        total_points = msg.width * msg.height

        valid_ratio = (
            len(ranges) / total_points
            if total_points > 0
            else 0.0
        )

        self.latest_lidar = {
            'min': float(np.min(ranges)),
            'mean': float(np.mean(ranges)),
            'std': float(np.std(ranges)),
            'median': float(np.median(ranges)),
            'valid_ratio': float(valid_ratio)
        }

    # =============================================================
    # Combine measurements
    # =============================================================

    def process_data(self):

        # Need all four measurements
        if self.latest_tag is None:
            return

        if self.latest_lidar is None:
            return

        if self.latest_jackal is None:
            return

        if self.latest_uav is None:
            return

        # ---------------------------------------------------------
        # Relative Jackal position with respect to UAV
        # ---------------------------------------------------------

        relative_x = (
            self.latest_jackal['x'] -
            self.latest_uav['x']
        )

        relative_y = (
            self.latest_jackal['y'] -
            self.latest_uav['y']
        )

        # ---------------------------------------------------------
        # Current ROS time
        # ---------------------------------------------------------

        now = self.get_clock().now()
        timestamp = now.nanoseconds / 1e9

        # Prevent duplicate rows when none of the sensor data
        # has changed sufficiently
        if self.last_written_time is not None:

            if timestamp - self.last_written_time < 0.02:
                return

        self.last_written_time = timestamp

        # ---------------------------------------------------------
        # Write row
        # ---------------------------------------------------------

        row = [

            timestamp,

            # -------------------------
            # AprilTag
            # -------------------------
            self.latest_tag['x'],
            self.latest_tag['y'],
            self.latest_tag['z'],
            self.latest_tag['distance'],
            self.latest_tag['yaw'],

            # -------------------------
            # LiDAR
            # -------------------------
            self.latest_lidar['min'],
            self.latest_lidar['mean'],
            self.latest_lidar['std'],
            self.latest_lidar['median'],
            self.latest_lidar['valid_ratio'],

            # -------------------------
            # Labels
            # -------------------------
            relative_x,
            relative_y
        ]

        self.writer.writerow(row)
        self.csv_file.flush()

        # Print periodically
        if int(timestamp * 10) % 10 == 0:

            self.get_logger().info(
                f'ΔX={relative_x:.3f} m, '
                f'ΔY={relative_y:.3f} m | '
                f'Tag=({self.latest_tag["x"]:.3f}, '
                f'{self.latest_tag["y"]:.3f}, '
                f'{self.latest_tag["z"]:.3f}) | '
                f'LiDAR mean={self.latest_lidar["mean"]:.3f} m'
            )

    # =============================================================
    # Shutdown
    # =============================================================

    def destroy_node(self):

        try:
            self.csv_file.close()

            self.get_logger().info(
                f'Dataset saved to:\n{self.csv_file_path}'
            )

        except Exception:
            pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = AprilTagLidarDataset()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
