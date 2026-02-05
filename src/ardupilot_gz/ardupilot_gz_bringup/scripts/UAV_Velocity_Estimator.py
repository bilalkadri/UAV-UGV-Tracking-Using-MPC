#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped
import numpy as np
import math
import time

class UAVVelocityEstimator(Node):
    def __init__(self):
        super().__init__('uav_velocity_estimator_node')  #This name will appear as ROS-2 Node 

        # Subscribing to IMU
        self.subscription = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10)

        # Publisher for estimated velocities
        self.vel_pub = self.create_publisher(TwistStamped, '/uav/vel_estimated', 10)

        # Initial velocities
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        # For integration
        self.last_time = None

        self.get_logger().info("UAV Velocity Estimator Node Started.")

    def imu_callback(self, msg: Imu):

        # Current time
        now = self.get_clock().now().nanoseconds / 1e9

        # First measurement
        if self.last_time is None:
            self.last_time = now
            return

        dt = now - self.last_time
        self.last_time = now

        if dt <= 0:
            return

        # Extract linear accelerations (body frame)
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        # Remove gravity (optionally)
        az -= 9.81

        # Integrate acceleration -> velocity
        self.vx += ax * dt
        self.vy += ay * dt
        self.vz += az * dt

        # ---- ANGULAR VELOCITY (direct from IMU gyro) ----
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z

        # Prepare message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"

        twist_msg.twist.linear.x = self.vx
        twist_msg.twist.linear.y = self.vy
        twist_msg.twist.linear.z = self.vz

        twist_msg.twist.angular.x = wx
        twist_msg.twist.angular.y = wy
        twist_msg.twist.angular.z = wz

        # Publish
        self.vel_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = UAVVelocityEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

