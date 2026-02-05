#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu

def rpy_to_rot(roll, pitch, yaw):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw); sy = np.sin(yaw)

    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp, cp*sr, cp*cr]])
    return R

def ang_vel_transform(roll, pitch):
    """
    Transform matrix T so that Euler rates = T * body_angular_rates
    Matches the structure used in the paper (Eq. 4 style).
    """
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    # Avoid division by zero near cp ~ 0
    if np.abs(cp) < 1e-6:
        cp = 1e-6
    T = np.array([
        [1.0, sr*sp/cp, cr*sp/cp],
        [0.0, cr,       -sr     ],
        [0.0, sr/cp,    cr/cp   ]
    ])
    return T

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

        # Publisher: relative pose (existing)
        self.pub_rel = self.create_publisher(PoseStamped, '/relative_pose_ekf', 10)
        # Publisher: predicted trajectory as a Path (NEW)
        self.pred_pub = self.create_publisher(Path, '/predicted_trajectory', 10)

        # Buffers
        self.v_g = np.zeros(3)
        self.omega_g = np.zeros(3)

        self.a_u = np.zeros(3)
        self.omega_u = np.zeros(3)
        self.v_u = np.zeros(3)  # UAV velocity from integrated acceleration

        # UGV orientation (used by original code if needed)
        self.roll_g = 0.0; self.pitch_g = 0.0; self.yaw_g = 0.0

        # Prediction horizon parameters
        self.pred_N = 12

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

        # Integrate UAV velocity from acceleration (simple integrator; keep as you had it)
        self.v_u += self.a_u * self.dt

    def odom_cb(self, msg: Odometry):
        # UGV linear velocity (in UGV body frame)
        self.v_g = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        # UGV angular velocity (in UGV body frame)
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

        # Use relative orientation from EKF for R_AG_rel (rotation from G -> A for relative state)
        roll_rel = float(self.x[3,0])
        pitch_rel = float(self.x[4,0])
        yaw_rel = float(self.x[5,0])

        # Rotation from G to A using relative orientation
        R_ag_rel = rpy_to_rot(roll_rel, pitch_rel, yaw_rel)
        T_ag_rel = ang_vel_transform(roll_rel, pitch_rel)

        # --------(1) Linear relative motion---------
        # rel_vel = R_AG * v_g - v_u   (UGV velocity transformed into UAV frame, minus UAV velocity)
        rel_vel = R_ag_rel @ self.v_g - self.v_u
        delta_pos = rel_vel * self.dt

        # --------(2) Angular relative motion---------
        rel_omega = T_ag_rel @ self.omega_g - self.omega_u
        delta_theta = rel_omega * self.dt

        # Update EKF state
        self.x[0:3,0] += delta_pos
        self.x[3:6,0] += delta_theta

        # Covariance update (simple form; keep as before)
        F = np.eye(6)
        self.P = F @ self.P @ F.T + self.Q

        # ----------------------------
        # ⭐ PRINT RELATIVE POSE HERE
        # ----------------------------
        #self.get_logger().info(
        #  f"Relative Position:  x={self.x[0,0]:.3f}, y={self.x[1,0]:.3f}, z={self.x[2,0]:.3f} | "
        #    f"Relative RPY: roll={self.x[3,0]:.3f}, pitch={self.x[4,0]:.3f}, yaw={self.x[5,0]:.3f}"
        #)

        # ----------------------------
        # Trajectory prediction (12-step horizon) and print predictions
        # ----------------------------
        traj = self.predict_trajectory(self.x.flatten(), self.v_g, self.omega_g,
                                       self.v_u, self.omega_u, N=self.pred_N, dt=self.dt)

        # Print predicted horizon (each step)
        #for i, p in enumerate(traj):
            #px, py, pz, r, pch, yw = p
            #self.get_logger().info(
            #    f"[Prediction] step {i+1:2d}: x={px:.3f}, y={py:.3f}, z={pz:.3f}, "
             #   f"roll={r:.3f}, pitch={pch:.3f}, yaw={yw:.3f}"
            #)

        # Publish current relative pose message (existing)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self.x[0])
        msg.pose.position.y = float(self.x[1])
        msg.pose.position.z = float(self.x[2])
        quat = self.rpy_to_quat(self.x[3,0], self.x[4,0], self.x[5,0])
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.pub_rel.publish(msg)

        # ----------------------------
        # Publish predicted trajectory as nav_msgs/Path (NEW)
        # ----------------------------
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'  # use 'map' as frame for the predicted trajectory
        path_msg.poses = []

        for p in traj:
            px, py, pz, r, pch, yw = p
            ps = PoseStamped()
            ps.header.stamp = path_msg.header.stamp
            ps.header.frame_id = path_msg.header.frame_id
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.position.z = float(pz)
            qx, qy, qz, qw = self.rpy_to_quat(r, pch, yw)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path_msg.poses.append(ps)

        self.pred_pub.publish(path_msg)

    # -------------------------
    # Trajectory prediction helper
    # -------------------------
    def predict_trajectory(self, state_vec, v_g, w_g, v_u, w_u, N=12, dt=0.02):
        """
        state_vec: flattened 6-element array [x_rel, y_rel, z_rel, roll, pitch, yaw]
        v_g: UGV linear velocity in G (3,)
        w_g: UGV angular velocity in G (3,)
        v_u: UAV linear velocity in A (3,)
        w_u: UAV angular velocity in A (3,)
        returns: numpy array shape (N,6) with predicted [x,y,z,roll,pitch,yaw]
        """
        x_rel, y_rel, z_rel, roll_rel, pitch_rel, yaw_rel = state_vec[:6]
        traj = np.zeros((N,6))
        for k in range(N):
            # rotation/transform from the current predicted relative orientation
            R_ag = rpy_to_rot(roll_rel, pitch_rel, yaw_rel)
            T_ag = ang_vel_transform(roll_rel, pitch_rel)

            # predict linear and angular increments
            rel_vel = R_ag @ v_g - v_u
            rel_omega = T_ag @ w_g - w_u

            x_rel += rel_vel[0] * dt
            y_rel += rel_vel[1] * dt
            z_rel += rel_vel[2] * dt

            roll_rel  += rel_omega[0] * dt
            pitch_rel += rel_omega[1] * dt
            yaw_rel   += rel_omega[2] * dt

            # Apply Δ = 0.01 offset to all predicted states (from paper’s equation)
            traj[k, :] = np.array([
                x_rel + 0.01,
                y_rel + 0.01,
                z_rel + 0.01,
                roll_rel + 0.01,
                pitch_rel + 0.01,
                yaw_rel + 0.01
            ])


        return traj

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

