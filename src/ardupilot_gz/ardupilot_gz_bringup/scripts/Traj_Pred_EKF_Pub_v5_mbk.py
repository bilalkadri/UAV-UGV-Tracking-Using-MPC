#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import traceback
from math import radians

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32

# ------------ helpers (same as before) ------------
def rpy_to_rot(roll, pitch, yaw):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw); sy = np.sin(yaw)

    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp,   cp*sr,            cp*cr]])
    return R

def ang_vel_transform(roll, pitch):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    if abs(cp) < 1e-6:
        cp = 1e-6
    return np.array([
        [1.0,  sr*sp/cp,  cr*sp/cp],
        [0.0,  cr,        -sr     ],
        [0.0,  sr/cp,     cr/cp   ]
    ])

# ------------ Node ------------
class RelativePoseEKF(Node):
    def __init__(self):
        super().__init__('relative_pose_ekf_and_trajectory_prediction_node')

        # State: x_rel, y_rel, z_rel, roll_rel, pitch_rel, yaw_rel
        self.x = np.zeros((6,1))
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.1

        pos_sigma = 0.04
        ang_sigma = radians(2.0)
        self.R = np.diag([
            pos_sigma**2, pos_sigma**2, pos_sigma**2,
            ang_sigma**2, ang_sigma**2, ang_sigma**2
        ])

        self.dt = 0.02  # 50 Hz

        # Subscribers
        self.create_subscription(Imu, '/imu', self.imu_cb, 10)
        self.create_subscription(Odometry, '/odometry', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/ugv/pose', self.ugv_pose_cb, 10)

        self.create_subscription(Bool, '/aruco/detected', self.aruco_detected_cb, 10)
        self.create_subscription(PoseStamped, '/aruco/pose', self.aruco_pose_cb, 10)

        # Publishers
        self.pub_rel = self.create_publisher(PoseStamped, '/relative_pose_ekf', 10)
        self.pred_pub = self.create_publisher(Path, '/predicted_trajectory', 10)

        # Existing publisher: update flag
        self.pub_update_flag = self.create_publisher(Bool, '/ekf/update_applied', 10)

        # ⭐ NEW publisher: Mahalanobis distance ⭐
        self.pub_maha = self.create_publisher(Float32, '/ekf/mahalanobis_distance', 10)

        # Buffers
        self.v_g = np.zeros(3)
        self.omega_g = np.zeros(3)
        self.a_u = np.zeros(3)
        self.omega_u = np.zeros(3)
        self.v_u = np.zeros(3)

        self.roll_g = 0.0
        self.pitch_g = 0.0
        self.yaw_g = 0.0

        self.pred_N = 12
        self.aruco_detected = False
        self.aruco_meas = None
        self.mahalanobis_threshold = 15.0

        self.create_timer(self.dt, self._timer_wrapper)
        self.get_logger().info("RelativePoseEKF node started.")

    def _timer_wrapper(self):
        try:
            self.ekf_predict_publish()
        except Exception:
            self.get_logger().error(
                "Exception in ekf_predict_publish:\n" + traceback.format_exc()
            )

    # Callbacks
    def imu_cb(self, msg):
        try:
            self.a_u = np.array([msg.linear_acceleration.x,
                                 msg.linear_acceleration.y,
                                 msg.linear_acceleration.z])
            self.omega_u = np.array([msg.angular_velocity.x,
                                     msg.angular_velocity.y,
                                     msg.angular_velocity.z])
            # self.v_u += self.a_u * self.dt       This line was in the original code, I am commenting it out
            # This line is the smoking gun 👇

            # self.v_u += self.a_u * self.dt

            # Why this explodes Z

            # In ROS:

            # msg.linear_acceleration INCLUDES GRAVITY

            # Typical IMU Z ≈ +9.81 m/s²

            # So you are effectively doing:

            # v_z(t) = ∫ 9.81 dt  → grows linearly
            # z(t)   = ∫ v_z dt   → grows quadratically


            # At 50 Hz:

            # After ~30–60 seconds → millions of meters

            # Which is EXACTLY what you see: z ≈ 2,500,000
            # ✔ This is physically correct math
            # ❌ But wrong state modeling
            # Correct Solution: Ignore IMU linear acceleration for velocity integration
            #(RECOMMENDED FOR UGV): FORCE PLANAR MOTION

            # Your UGV:
            # Moves on ground
            # Does NOT fly
            # Does NOT estimate altitude via IMU
            self.v_u[:] = 0.0

        except Exception:
            self.get_logger().error("Exception in imu_cb:\n" + traceback.format_exc())

    def odom_cb(self, msg):
        try:
            self.v_g = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z])
            self.omega_g = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z])
        except Exception:
            self.get_logger().error("Exception in odom_cb:\n" + traceback.format_exc())

    def ugv_pose_cb(self, msg):
        try:
            q = msg.pose.orientation
            self.roll_g, self.pitch_g, self.yaw_g = self.quat_to_rpy(q)
        except Exception:
            self.get_logger().error("Exception in ugv_pose_cb:\n" + traceback.format_exc())

    def aruco_detected_cb(self, msg):
        self.aruco_detected = bool(msg.data)

    def aruco_pose_cb(self, msg):
        self.aruco_meas = msg

    # Main EKF procedure
    def ekf_predict_publish(self):
        # --- Prediction step ---
        roll_rel = float(self.x[3,0])
        pitch_rel = float(self.x[4,0])
        yaw_rel = float(self.x[5,0])

        R_ag_rel = rpy_to_rot(roll_rel, pitch_rel, yaw_rel)
        T_ag_rel = ang_vel_transform(roll_rel, pitch_rel)

        rel_vel = R_ag_rel @ self.v_g - self.v_u
        rel_omega = T_ag_rel @ self.omega_g - self.omega_u

        self.x[0:3,0] += rel_vel * self.dt
        self.x[3:6,0] += rel_omega * self.dt

        self.P = self.P + self.Q

        # --- Update step ---
        update_applied = False
        maha_value = -1.0   # default, meaning "no maha computed"

        if self.aruco_detected and (self.aruco_meas is not None):
            try:
                q = self.aruco_meas.pose.orientation
                meas_roll, meas_pitch, meas_yaw = self.quat_to_rpy(q)

                z = np.array([
                    [self.aruco_meas.pose.position.x],
                    [self.aruco_meas.pose.position.y],
                    [self.aruco_meas.pose.position.z],
                    [meas_roll],
                    [meas_pitch],
                    [meas_yaw]
                ])

                y = z - self.x
                for idx in range(3,6):
                    ang = (float(y[idx,0]) + np.pi) % (2*np.pi) - np.pi
                    y[idx,0] = ang

                S = self.P + self.R
                try:
                    Sinv = np.linalg.inv(S)
                except:
                    Sinv = np.linalg.pinv(S)

                maha_value = float((y.T @ Sinv @ y)[0,0])

                if maha_value <= self.mahalanobis_threshold:
                    K = self.P @ Sinv
                    self.x = self.x + K @ y
                    self.P = (np.eye(6) - K) @ self.P
                    update_applied = True

            except Exception:
                self.get_logger().error("Exception during EKF update:\n" + traceback.format_exc())

        # ⭐ Publish the Mahalanobis distance ⭐
        maha_msg = Float32()
        maha_msg.data = float(maha_value)
        self.pub_maha.publish(maha_msg)

        # ⭐ Publish update-applied flag ⭐
        flag = Bool()
        flag.data = update_applied
        self.pub_update_flag.publish(flag)

        # --- Publish EKF estimate ---
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.position.x = float(self.x[0,0])
        msg.pose.position.y = float(self.x[1,0])
        # msg.pose.position.z = float(self.x[2,0]) THis was in the original code, z was exploding
        msg.pose.position.z = 0.0

        qx,qy,qz,qw = self.rpy_to_quat(
            float(self.x[3,0]),
            float(self.x[4,0]),
            float(self.x[5,0])
        )
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pub_rel.publish(msg)

        # --- Trajectory prediction ---
        traj = self.predict_trajectory(
            self.x.flatten(), self.v_g, self.omega_g,
            self.v_u, self.omega_u,
            N=self.pred_N, dt=self.dt
        )

        # Publish predicted path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "odom"

        for p in traj:
            px, py, pz, r, pch, yw = p
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.position.z = float(pz)
            qx,qy,qz,qw = self.rpy_to_quat(r, pch, yw)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path_msg.poses.append(ps)

        self.pred_pub.publish(path_msg)

    # Prediction helper
    def predict_trajectory(self, state_vec, v_g, w_g, v_u, w_u, N=12, dt=0.02):
        x_rel, y_rel, z_rel, roll_rel, pitch_rel, yaw_rel = state_vec[:6]
        traj = np.zeros((N,6))
        for i in range(N):
            R_ag = rpy_to_rot(roll_rel, pitch_rel, yaw_rel)
            T_ag = ang_vel_transform(roll_rel, pitch_rel)
            rel_vel = R_ag @ v_g - v_u
            rel_omega = T_ag @ w_g - w_u

            x_rel += rel_vel[0] * dt
            y_rel += rel_vel[1] * dt
            # z_rel += rel_vel[2] * dt THis was in the original code, z was exploding
            z_rel = 0.0

            roll_rel += rel_omega[0] * dt
            pitch_rel += rel_omega[1] * dt
            yaw_rel += rel_omega[2] * dt

            traj[i,:] = [x_rel, y_rel, z_rel, roll_rel, pitch_rel, yaw_rel]

        return traj

    # Helper functions
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

