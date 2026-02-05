#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

def rpy_to_rot(roll, pitch, yaw):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw); sy = np.sin(yaw)

    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp, cp*sr, cp*cr]])
    return R

class RelativePoseEKF(Node):
    def __init__(self):
        super().__init__('relative_pose_ekf')

        # State: x_rel, y_rel, z_rel, roll_rel, pitch_rel, yaw_rel
        self.x = np.zeros((6,1))
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.01

        self.dt = 0.02  # 50 Hz

        # --- SUBSCRIBERS ---
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odometry', self.odom_cb, 10)

        self.ugv_pose_sub = self.create_subscription(
            PoseStamped, '/ugv/pose', self.ugv_pose_cb, 10)

        # Publisher
        self.pub_rel = self.create_publisher(PoseStamped, '/relative_pose_ekf', 10)

        # Buffers
        self.v_g = np.zeros(3)
        self.omega_g = np.zeros(3)

        self.a_u = np.zeros(3)
        self.omega_u = np.zeros(3)
        self.v_u = np.zeros(3)  # UAV velocity from integrated acceleration

        self.roll_g = 0.0; self.pitch_g = 0.0; self.yaw_g = 0.0

        self.create_timer(self.dt, self.ekf_predict_publish)

    # -------------------------
    # CALLBACKS
    # -------------------------
    def imu_cb(self, msg: Imu):
        # UAV acceleration
        self.a_u = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        # UAV angular velocity
        self.omega_u = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Integrate UAV velocity from acceleration
        self.v_u += self.a_u * self.dt

    def odom_cb(self, msg: Odometry):
        # UGV linear velocity
        self.v_g = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        # UGV angular velocity
        self.omega_g = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])

    def ugv_pose_cb(self, msg):
        q = msg.pose.orientation
        self.roll_g, self.pitch_g, self.yaw_g = self.quat_to_rpy(q)

    # -------------------------
    # EKF PREDICTION STEP
    # -------------------------
    def ekf_predict_publish(self):

        R_ag = rpy_to_rot(self.roll_g, self.pitch_g, self.yaw_g)

        # (1) Linear relative motion
        delta_pos = R_ag @ (self.v_g - self.v_u) * self.dt

        # (2) Angular relative motion
        delta_theta = (self.omega_g - self.omega_u) * self.dt

        # Update EKF state
        self.x[0:3,0] += delta_pos
        self.x[3:6,0] += delta_theta

        # Covariance update
        F = np.eye(6)
        self.P = F @ self.P @ F.T + self.Q

        # ----------------------------
        # ⭐ PRINT RELATIVE POSE HERE
        # ----------------------------
        self.get_logger().info(
            f"Relative Position:  x={self.x[0,0]:.3f}, y={self.x[1,0]:.3f}, z={self.x[2,0]:.3f} | "
            f"Relative RPY: roll={self.x[3,0]:.3f}, pitch={self.x[4,0]:.3f}, yaw={self.x[5,0]:.3f}"
        )

        # Publish message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(self.x[0])
        msg.pose.position.y = float(self.x[1])
        msg.pose.position.z = float(self.x[2])
        quat = self.rpy_to_quat(self.x[3,0], self.x[4,0], self.x[5,0])
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.pub_rel.publish(msg)

    # -------------------------
    # Helper functions
    # -------------------------
    def quat_to_rpy(self, q):
        w,x,y,z = q.w, q.x, q.y, q.z
        sinr = 2*(w*x + y*z)
        cosr = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr, cosr)

        sinp = 2*(w*y - z*x)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)

        siny = 2*(w*z + x*y)
        cosy = 1 - 2*(y*y + z*z)
        yaw = np.arctan2(siny, cosy)
        return roll, pitch, yaw

    def rpy_to_quat(self, roll, pitch, yaw):
        cy = np.cos(yaw*0.5); sy = np.sin(yaw*0.5)
        cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
        cr = np.cos(roll*0.5); sr = np.sin(roll*0.5)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        return (qx, qy, qz, qw)


def main(args=None):
    rclpy.init(args=args)
    node = RelativePoseEKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

