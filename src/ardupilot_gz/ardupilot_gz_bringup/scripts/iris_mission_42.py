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

        # State and covariances
        self.x = np.zeros((6,1))
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.01

        self.dt = 0.02  # 50 Hz

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odometry', self.odom_cb, 10)
        self.ugv_pose_sub = self.create_subscription(PoseStamped, '/ugv/pose', self.ugv_pose_cb, 10)

        # Publishers
        self.pub_rel = self.create_publisher(PoseStamped, '/relative_pose_ekf', 10)
        # <-- CHANGED: publish predicted trajectory as PoseArray (user requested Option 2)
        self.pred_pub = self.create_publisher(PoseArray, '/predicted_trajectory', 10)

        # Buffers for measurements
        self.v_g = np.zeros(3)
        self.omega_g = np.zeros(3)

        self.a_u = np.zeros(3)
        self.omega_u = np.zeros(3)
        self.v_u = np.zeros(3)

        self.roll_g = 0.0; self.pitch_g = 0.0; self.yaw_g = 0.0

        self.pred_N = 12

        self.create_timer(self.dt, self.ekf_predict_publish)

    # Callbacks
    def imu_cb(self, msg: Imu):
        self.a_u = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        self.omega_u = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
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

    # Main predict + publish
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

        # Simple covariance update
        F = np.eye(6)
        self.P = F @ self.P @ F.T + self.Q

        # Log current relative pose
        self.get_logger().info(
            f"Relative Position:  x={self.x[0,0]:.3f}, y={self.x[1,0]:.3f}, z={self.x[2,0]:.3f} | "
            f"Relative RPY: roll={self.x[3,0]:.3f}, pitch={self.x[4,0]:.3f}, yaw={self.x[5,0]:.3f}"
        )

        # Predict trajectory
        traj = self.predict_trajectory(self.x.flatten(), self.v_g, self.omega_g,
                                       self.v_u, self.omega_u, N=self.pred_N, dt=self.dt)

        for i, p in enumerate(traj):
            px, py, pz, r, pch, yw = p
            self.get_logger().debug(
                f"[Prediction] step {i+1:2d}: x={px:.3f}, y={py:.3f}, z={pz:.3f}, "
                f"roll={r:.3f}, pitch={pch:.3f}, yaw={yw:.3f}"
            )

        # Publish current relative pose
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

        # ----------------------------
        # PUBLISH predicted trajectory as PoseArray (user requested)
        # ----------------------------
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

    # Trajectory predictor (unchanged)
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

    # helpers
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
# PID controller class
# ---------------------------
class PID:
    def __init__(self, kp, ki, kd, output_limits=(-1, 1)):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.output_limits = output_limits
        self.integral = 0.0; self.prev_error = 0.0; self.prev_time = time.time()

    def reset(self):
        self.integral = 0.0; self.prev_error = 0.0; self.prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.prev_time if now - self.prev_time > 0 else 1e-6
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        self.prev_error = error; self.prev_time = now
        return output

pid_x = PID(0.002, 0.0, 0.001, (-0.5, 0.5))
pid_y = PID(0.002, 0.0, 0.001, (-0.5, 0.5))

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

        # Camera subscription (topic name kept as '/camera/image')
        self.image_sub = self.create_subscription(Image, '/camera/image', self.image_callback, 10)

        # Optional jackal velocity publisher (if you want to publish TwistStamped to jackal)
        self.jackal_pub = self.create_publisher(TwistStamped, '/jackal/jackal_velocity_controller/cmd_vel', 10)

        # Keep safe log
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

        # detect markers
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(frame)
        except Exception:
            # fallback for older OpenCV
            corners, ids, _ = cv2.aruco.detectMarkers(frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL))

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

        # Show for debugging (non-blocking)
        try:
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)
        except Exception:
            pass

# ---------------------------
# MAVLINK helper functions (unchanged semantics)
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
        print(f"Mode {mode_name} not available on vehicle.")
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
    master.mav.command_long_send(master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1 if arm else 0, 0,0,0,0,0,0)
    while True:
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb:
            armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            if armed == arm:
                return

def request_streams(master):
    master.mav.command_long_send(master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        100000, 0, 0, 0, 0, 0)

def read_relative_alt(master):
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    return msg.relative_alt / 1000.0 if msg else None

def takeoff(master, alt_m):
    master.mav.command_long_send(master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0,0,0,0,0,0, alt_m, 0)

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
    master.mav.command_long_send(master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0,0,0,0,0,0,0,0)

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
# Jackal / UGV motion publisher (optional)
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
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)

# ---------------------------
# follow_ugv PID thread (uses global TRACK_FLAG)
# ---------------------------
def follow_ugv(node: ArucoFollower, master):
    global pid_x, pid_y, uav_traj, err_x_list, err_y_list
    dt = 0.1
    while rclpy.ok():
        with TRACK_FLAG_LOCK:
            local_flag = TRACK_FLAG
        if local_flag != '1':
            time.sleep(0.05)
            continue
        pid_x.reset(); pid_y.reset()
        start_time = time.time()
        node.get_logger().info("follow_ugv: entering PID tracking loop")
        while True:
            with TRACK_FLAG_LOCK:
                if TRACK_FLAG != '1':
                    break
            if node.marker_center and node.frame_center:
                err_x = node.frame_center[0] - node.marker_center[0]
                err_y = node.frame_center[1] - node.marker_center[1]
                with ERR_LOCK:
                    err_x_list.append(err_x); err_y_list.append(err_y)
                    if len(err_x_list) > 1000:
                        err_x_list[:] = err_x_list[-1000:]; err_y_list[:] = err_y_list[-1000:]
                vx = -0.5 * pid_x.update(err_x)
                vy = 0.5 * pid_y.update(err_y)
                vz = 0.0; yaw_rate = 0.0
                send_velocity(master, vx, vy, vz, yaw_rate, start_time)
                if uav_traj:
                    last_x, last_y = uav_traj[-1]
                else:
                    last_x, last_y = 0.0, 0.0
                new_x = last_x + vx * dt
                new_y = last_y + vy * dt
                uav_traj.append((new_x, new_y))
            else:
                with ERR_LOCK:
                    err_x_list.append(None); err_y_list.append(None)
                send_velocity(master, 0.0, 0.0, 0.0, 0.0, start_time)
            time.sleep(dt)
        node.get_logger().info("follow_ugv: marker lost, stopping PID")
        send_velocity(master, 0.0, 0.0, 0.0, 0.0, start_time)
        time.sleep(0.1)

# ---------------------------
# Live plotting thread
# ---------------------------
def plot_errors_thread(stop_event, node_name="plot_thread"):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    line_x, = ax.plot([], [], label='err_x')
    line_y, = ax.plot([], [], label='err_y')
    ax.set_xlabel("Sample index"); ax.set_ylabel("Pixel error")
    ax.set_title("Live err_x / err_y (when TRACK_FLAG == '1')")
    ax.grid(True); ax.legend(); ax.set_xlim(0, 100); ax.set_ylim(-500, 500)
    while not stop_event.is_set() and rclpy.ok():
        with ERR_LOCK:
            xs = list(err_x_list); ys = list(err_y_list)
        valid_indices = [i for i, (a, b) in enumerate(zip(xs, ys)) if a is not None and b is not None]
        if valid_indices:
            plot_x = [xs[i] for i in valid_indices]; plot_y = [ys[i] for i in valid_indices]
            idx = list(range(len(plot_x)))
            line_x.set_data(idx, plot_x); line_y.set_data(idx, plot_y)
            ax.relim(); ax.autoscale_view(True, True, True)
            N = 200; current_len = len(idx)
            ax.set_xlim(max(0, current_len - N), max(N, current_len))
        try:
            fig.canvas.draw(); fig.canvas.flush_events()
        except Exception:
            pass
        time.sleep(0.1)
    plt.close(fig)

# ---------------------------
# MAIN: create mavlink + nodes + executor + threads
# ---------------------------
def main():
    # Setup mavlink connection (used by Aruco follower)
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)
    request_streams(master)

    # Initialize ROS
    rclpy.init()
    ekf_node = RelativePoseEKF()
    aruco_node = ArucoFollower(master)

    # Executor to run both nodes
    executor = MultiThreadedExecutor()
    executor.add_node(ekf_node)
    executor.add_node(aruco_node)

    # Threads for Aruco follower behaviours
    follow_thread = threading.Thread(target=follow_ugv, args=(aruco_node, master), daemon=True)
    follow_thread.start()

    plot_stop_event = threading.Event()
    plot_thread = threading.Thread(target=plot_errors_thread, args=(plot_stop_event,), daemon=True)
    plot_thread.start()

    # Optional: start rectangle publisher for jackal (uncomment if needed)
    # stop_event_rect = threading.Event()
    # pub_thread = threading.Thread(target=publish_rectangle, args=(aruco_node, aruco_node.jackal_pub, stop_event_rect), daemon=True)
    # pub_thread.start()

    # UAV takeoff in background (blocking) — run in a thread if desired
    # takeoff_and_wait(master, TARGET_ALT_M)  # uncomment to takeoff here (blocking)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Keyboard interrupt — shutting down")
    finally:
        # shutdown procedures
        plot_stop_event.set()
        try:
            ekf_node.destroy_node()
            aruco_node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

