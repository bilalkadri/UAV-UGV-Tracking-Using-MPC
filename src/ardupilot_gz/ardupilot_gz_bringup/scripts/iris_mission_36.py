#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import threading
import time
import math
from pymavlink import mavutil
import matplotlib.pyplot as plt

# ============================================================
# ================= GLOBAL TRACKING FLAG =====================
# ============================================================
TRACK_FLAG = '0'
TRACK_FLAG_LOCK = threading.Lock()

# ============================================================
# ========== TRACK FLAG FIRST-DETECTION JACKAL SPEED =========
# ============================================================
JACKAL_SLOWED = False   # <--- NEW FLAG

# ============================================================
# ======== GLOBAL TRAJECTORY LISTS FOR UAV & UGV =============
# ============================================================
uav_traj = []
ugv_traj = []

# ============================================================
# ====================== SIMPLE PID ==========================
# ============================================================
class PID:
    def __init__(self, kp, ki, kd, output_limits=(-1, 1)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.prev_time if now - self.prev_time > 0 else 1e-6

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        self.prev_error = error
        self.prev_time = now
        return output

# ============================================================
# ================= GLOBAL PID CONTROLLERS ===================
# ============================================================
pid_x = PID(0.002, 0.0, 0.001, (-0.5, 0.5))
pid_y = PID(0.002, 0.0, 0.001, (-0.5, 0.5))

# ============================================================
# ================ ARUCO DETECTOR NODE ========================
# ============================================================
class ArucoDetector(Node):
    def __init__(self, master):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.master = master
        self.tag_detected = False
        self.marker_center = None
        self.frame_center = None

        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10
        )

        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        except:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.get_logger().info("Aruco Detector Node Started")

    def image_callback(self, msg):
        global TRACK_FLAG, JACKAL_SLOWED
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        self.frame_center = (w / 2.0, h / 2.0)

        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            if not self.tag_detected:
                self.tag_detected = True
                with TRACK_FLAG_LOCK:
                    TRACK_FLAG = '1'

                # FIRST TIME TRACK_FLAG=1 → SLOW DOWN JACKAL ANGULAR VELOCITY
                if not JACKAL_SLOWED:
                    JACKAL_SLOWED = True
                    print("TRACK_FLAG=1 first time → Jackal omega changed to 0.02")

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

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

# ============================================================
# ================= JACKAL & UAV FOLLOW ======================
# ============================================================
def move_jackal_forward(node, duration=5.0, speed=0.4):
    """UGV moves in a circular trajectory continuously and independently."""
    global ugv_traj, JACKAL_SLOWED

    radius = 25.0
    omega = 0.2     # default value

    jackal_pub = node.create_publisher(
        TwistStamped, '/jackal/jackal_velocity_controller/cmd_vel', 10
    )
    msg = TwistStamped()
    msg.header = Header()

    x, y = radius, 0.0
    theta = 0.0
    dt = 0.1

    while rclpy.ok():
        # If TRACK_FLAG=1 first time → omega becomes 0.02
        if JACKAL_SLOWED:
            omega = 0.02

        linear_speed = radius * omega
        msg.twist.linear.x = linear_speed
        msg.twist.angular.z = omega

        msg.header.stamp = node.get_clock().now().to_msg()
        jackal_pub.publish(msg)

        theta += omega * dt
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        ugv_traj.append((x, y))

        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(dt)

# ============================================================
# FOLLOW UGV WITH PID (Modified Version)
# ============================================================
def follow_ugv(node, master):
    global pid_x, pid_y, uav_traj

    dt = 0.1

    while True:
        with TRACK_FLAG_LOCK:
            local_flag = TRACK_FLAG

        if local_flag != '1':
            time.sleep(0.1)
            continue

        pid_x.reset()
        pid_y.reset()
        start_time = time.time()

        if node.marker_center and node.frame_center:
            err_x = node.frame_center[0] - node.marker_center[0]
            err_y = node.frame_center[1] - node.marker_center[1]

            vx = -2.5 * pid_x.update(err_x)
            vy = 2.5 * pid_y.update(err_y)
            vz = 0.0
            yaw_rate = 0.0

            send_velocity(master, vx, vy, vz, yaw_rate, start_time)

            if uav_traj:
                last_x, last_y = uav_traj[-1]
            else:
                last_x, last_y = 0.0, 0.0

            new_x = last_x + vx * dt
            new_y = last_y + vy * dt
            uav_traj.append((new_x, new_y))
        else:
            send_velocity(master, 0.0, 0.0, 0.0, 0.0, start_time)

        time.sleep(dt)

# ============================================================
# MAVLINK FUNCTIONS (unchanged)
# ============================================================
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
            armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)!=0
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
    return msg.relative_alt/1000.0 if msg else None

def takeoff(master, alt_m):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,0,0,0,0,0,0,alt_m,0
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
            vx,vy,vz,
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
        0,0,0,0,0,0,0,0,0
    )

# ============================================================
# END OF CODE
# ============================================================

