#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import threading
import time
import math
import cv2

from geometry_msgs.msg import PoseStamped, Pose, PoseArray, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image
from std_msgs.msg import Header

from cv_bridge import CvBridge
from pymavlink import mavutil
import matplotlib.pyplot as plt

# ---------------------------
# Global shared flags / buffers
# ---------------------------
TRACK_FLAG = '0'
TRACK_FLAG_LOCK = threading.Lock()

uav_traj = []
ugv_traj = []

err_x_list = []
err_y_list = []
ERR_LOCK = threading.Lock()

# ---------------------------
# Utility math helpers
# ---------------------------
def rpy_to_rot(roll, pitch, yaw):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw); sy = np.sin(yaw)
    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp, cp*sr, cp*cr]])
    return R

def ang_vel_transform(roll, pitch):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    if np.abs(cp) < 1e-6:
        cp = 1e-6
    T = np.array([
        [1.0, sr*sp/cp, cr*sp/cp],
        [0.0, cr,       -sr     ],
        [0.0, sr/cp,    cr/cp   ]
    ])
    return T

# ---------------------------
# RelativePoseEKF node
# ---------------------------
class RelativePoseEKF(Node):
    def __init__(self):
        super().__init__('relative_pose_ekf')

        self.x = np.zeros((6,1))
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.01

        self.dt = 0.02  # 50 Hz

        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odometry', self.odom_cb, 10)
        self.ugv_pose_sub = self.create_subscription(PoseStamped, '/ugv/pose', self.ugv_pose_cb, 10)

        self.pub_rel = self.create_publisher(PoseStamped, '/relative_pose_ekf', 10)
        self.pred_pub = self.create_publisher(PoseArray, '/predicted_trajectory', 10)

        self.v_g = np.zeros(3)
        self.omega_g = np.zeros(3)

        self.a_u = np.zeros(3)
        self.omega_u = np.zeros(3)
        self.v_u = np.zeros(3)

        self.roll_g = 0.0; self.pitch_g = 0.0; self.yaw_g = 0.0

        self.pred_N = 12

        self.create_timer(self.dt, self.ekf_predict_publish)

    def imu_cb(self, msg: Imu):
        self.a_u = np.array([msg.linear_acceleration.x,
                             msg.linear_acceleration.y,
                             msg.linear_acceleration.z])
        self.omega_u = np.array([msg.angular_velocity.x,
                                 msg.angular_velocity.y,
                                 msg.angular_velocity.z])
        self.v_u += self.a_u * self.dt

    def odom_cb(self, msg: Odometry):
        self.v_g = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        self.omega_g = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])

    def ugv_pose_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        self.roll_g, self.pitch_g, self.yaw_g = self.quat_to_rpy(q)

    def ekf_predict_publish(self):
        roll_rel = float(self.x[3,0])
        pitch_rel = float(self.x[4,0])
        yaw_rel = float(self.x[5,0])

        R_ag_rel = rpy_to_rot(roll_rel, pitch_rel, yaw_rel)
        T_ag_rel = ang_vel_transform(roll_rel, pitch_rel)

        rel_vel = R_ag_rel @ self.v_g - self.v_u
        delta_pos = rel_vel * self.dt

        rel_omega = T_ag_rel @ self.omega_g - self.omega_u
        delta_theta = rel_omega * self.dt

        self.x[0:3,0] += delta_pos
        self.x[3:6,0] += delta_theta

        F = np.eye(6)
        self.P = F @ self.P @ F.T + self.Q

        self.get_logger().info(
            f"Relative Position: x={self.x[0,0]:.3f}, y={self.x[1,0]:.3f}, z={self.x[2,0]:.3f}"
        )

        traj = self.predict_trajectory(self.x.flatten(), self.v_g, self.omega_g,
                                       self.v_u, self.omega_u, N=self.pred_N, dt=self.dt)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self.x[0])
        msg.pose.position.y = float(self.x[1])
        msg.pose.position.z = float(self.x[2])
        qx, qy, qz, qw = self.rpy_to_quat(self.x[3,0], self.x[4,0], self.x[5,0])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pub_rel.publish(msg)

        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'
        pa.poses = []
        for p in traj:
            px, py, pz, r, pch, yw = p
            pose = Pose()
            pose.position.x = float(px)
            pose.position.y = float(py)
            pose.position.z = float(pz)
            qx, qy, qz, qw = self.rpy_to_quat(r, pch, yw)
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            pa.poses.append(pose)
        self.pred_pub.publish(pa)

    def predict_trajectory(self, state_vec, v_g, w_g, v_u, w_u, N=12, dt=0.02):
        x_rel, y_rel, z_rel, roll_rel, pitch_rel, yaw_rel = state_vec[:6]
        traj = np.zeros((N,6))
        for k in range(N):
            R_ag = rpy_to_rot(roll_rel, pitch_rel, yaw_rel)
            T_ag = ang_vel_transform(roll_rel, pitch_rel)
            rel_vel = R_ag @ v_g - v_u
            rel_omega = T_ag @ w_g - w_u
            x_rel += rel_vel[0] * dt
            y_rel += rel_vel[1] * dt
            z_rel += rel_vel[2] * dt
            roll_rel  += rel_omega[0] * dt
            pitch_rel += rel_omega[1] * dt
            yaw_rel   += rel_omega[2] * dt
            traj[k, :] = np.array([x_rel, y_rel, z_rel, roll_rel, pitch_rel, yaw_rel])
        return traj

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

# ---------------------------
# Aruco follower node
# ---------------------------
class ArucoFollower(Node):
    def __init__(self, master):
        super().__init__('aruco_follower')
        self.bridge = CvBridge()
        self.master = master

        self.tag_detected = False
        self.marker_center = None
        self.frame_center = None

        self.image_sub = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.jackal_pub = self.create_publisher(TwistStamped, '/jackal/jackal_velocity_controller/cmd_vel', 10)

        self.get_logger().info("ArucoFollower Node Started")

    def image_callback(self, msg):
        global TRACK_FLAG
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        h, w, _ = frame.shape
        self.frame_center = (w / 2.0, h / 2.0)

        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(frame)
        except Exception:
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame,
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            )

        if ids is not None and len(ids) > 0:
            if not self.tag_detected:
                self.tag_detected = True
                with TRACK_FLAG_LOCK:
                    TRACK_FLAG = '1'
                self.get_logger().info("Marker detected — TRACK_FLAG = 1")

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            c = corners[0][0]
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
            self.marker_center = (cx, cy)

        else:
            if self.tag_detected:
                self.tag_detected = False
                with TRACK_FLAG_LOCK:
                    TRACK_FLAG = '0'
                self.get_logger().info("Marker lost — TRACK_FLAG = 0")
            self.marker_center = None

        try:
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)
        except Exception:
            pass

# ---------------------------
# MAVLINK helper functions
# ---------------------------
TARGET_ALT_M = 2.0
CONN_STR = 'udp:127.0.0.1:14551'
COMMAND_RATE_HZ = 5

def wait_heartbeat(master):
    print("Waiting heartbeat...")
    master.wait_heartbeat()
    print("Heartbeat OK")

def set_mode(master, mode_name):
    modes = master.mode_mapping()
    if mode_name not in modes:
        print(f"Mode {mode_name} not available.")
        return
    mode_id = modes[mode_name]
    for _ in range(10):
        master.mav.set_mode_send(
            master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb and mavutil.mode_string_v10(hb) == mode_name:
            print(f"Mode set to {mode_name}")
            return

def arm_and_wait(master, arm=True):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1 if arm else 0, 0,0,0,0,0,0
    )
    while True:
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb:
            armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            if armed == arm:
                return

def request_streams(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        100000, 0,0,0,0,0
    )

def read_relative_alt(master):
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    return msg.relative_alt / 1000.0 if msg else None

def takeoff(master, alt_m):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0,0,0,0,0,0, alt_m, 0
    )

def send_velocity(master, vx, vy, vz, yaw_rate, start_time):
    try:
        time_boot_ms = int((time.time() - start_time) * 1000)
        master.mav.set_position_target_local_ned_send(
            time_boot_ms,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000011,
            0,0,0,
            vx, vy, vz,
            0,0,0,
            0,
            yaw_rate
        )
    except Exception as e:
        print(f"send_velocity error: {e}")

def land(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0,0,0,0,0,0,0,0
    )

def takeoff_and_wait(master, alt):
    set_mode(master, "GUIDED")
    arm_and_wait(master, True)
    takeoff(master, alt)
    while True:
        alt_now = read_relative_alt(master)
        if alt_now and alt_now >= alt * 0.95:
            break
        time.sleep(0.5)

def wait_for_landing(master):
    while True:
        alt = read_relative_alt(master)
        if alt and alt <= 0.2:
            break
        time.sleep(0.5)
    arm_and_wait(master, False)

# ---------------------------
# Jackal / UGV rectangle motion publisher
# ---------------------------
def publish_rectangle(node, pub, stop_event):
    rate_hz = 20.0
    dt = 1.0 / rate_hz
    Lx = 3.0; Ly = 3.0; corner_radius = 0.6
    v = 0.25; yaw_rate_corner = 0.45

    t1 = (Lx - 2 * corner_radius) / v
    t2 = (Ly - 2 * corner_radius) / v
    t_corner = (3.14159 / 2 * corner_radius) / v
    T = 2 * (t1 + t2) + 4 * t_corner

    t = 0.0
    while not stop_event.is_set() and rclpy.ok():
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()

        if 0 <= t < t1:
            vx = v; wz = 0.0
        elif t1 <= t < t1 + t_corner:
            vx = v; wz = yaw_rate_corner
        elif t1 + t_corner <= t < t1 + t_corner + t2:
            vx = v; wz = 0.0
        elif t1 + t_corner + t2 <= t < t1 + t_corner + t2 + t_corner:
            vx = v; wz = yaw_rate_corner
        elif t1 + 2*t_corner + t2 <= t < t1 + 2*t_corner + t2 + t1:
            vx = v; wz = 0.0
        elif t1 + 2*t_corner + t2 + t1 <= t < t1 + 3*t_corner + t2 + t1:
            vx = v; wz = yaw_rate_corner
        elif t1 + 3*t_corner + t2 + t1 <= t < t1 + 3*t_corner + 2*t2 + t1:
            vx = v; wz = 0.0
        else:
            vx = v; wz = yaw_rate_corner

        msg.twist.linear.x = vx
        msg.twist.angular.z = wz
        pub.publish(msg)

        t += dt
        if t > T:
            t = 0.0
        time.sleep(dt)


