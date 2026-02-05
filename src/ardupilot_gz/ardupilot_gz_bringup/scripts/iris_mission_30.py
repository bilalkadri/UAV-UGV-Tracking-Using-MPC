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
import matplotlib.pyplot as plt  # For plotting trajectories

# ============================================================
# ================= GLOBAL TRACKING FLAG =====================
# ============================================================
TRACK_FLAG = '0'

# ============================================================
# ====================== SIMPLE PID ==========================
# ============================================================
class PID:
    def __init__(self, kp, ki, kd, output_limits=(-1, 1)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()

    def reset(self):
        self.integral = 0
        self.prev_error = 0
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
# ======== GLOBAL TRAJECTORY LISTS FOR UAV & UGV =============
# ============================================================
uav_traj = []
ugv_traj = []

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

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.get_logger().info("Aruco Detector Node Started")

    def image_callback(self, msg):
        global TRACK_FLAG
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        self.frame_center = (w / 2, h / 2)

        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            if not self.tag_detected:
                self.tag_detected = True
                TRACK_FLAG = '1'
                self.get_logger().info("Marker detected — TRACK_FLAG = 1")

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            c = corners[0][0]
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
            self.marker_center = (cx, cy)
        else:
            if self.tag_detected:
                self.tag_detected = False
                TRACK_FLAG = '0'
                self.get_logger().info("Marker lost — TRACK_FLAG = 0")

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

# ============================================================
# ================= JACKAL & UAV FOLLOW ======================
# ============================================================

def move_jackal_forward(node, duration=5.0, speed=0.4):
    """UGV moves in a circular trajectory continuously and independently."""
    global ugv_traj

    radius = 3.0
    omega = 0.2
    linear_speed = radius * omega

    jackal_pub = node.create_publisher(
        TwistStamped, '/jackal/jackal_velocity_controller/cmd_vel', 10
    )
    msg = TwistStamped()
    msg.header = Header()

    msg.twist.linear.x = linear_speed
    msg.twist.angular.z = omega

    x, y = 0.0, 0.0
    theta = 0.0
    dt = 0.1
    start_time = time.time()

    # 🔥 IMPORTANT: Now Jackal keeps moving forever (independent of TRACK_FLAG)
    while True:
        msg.header.stamp = node.get_clock().now().to_msg()
        jackal_pub.publish(msg)

        theta += omega * dt
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        ugv_traj.append((x, y))

        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(dt)

def follow_ugv(node, master):
    global pid_x, pid_y, uav_traj
    print("Drone following UGV using PID...")
    pid_x.reset()
    pid_y.reset()

    start_time = time.time()
    dt = 0.1
    x, y = 0.0, 0.0

    while TRACK_FLAG == '1':
        if node.marker_center and node.frame_center:
            err_x = node.frame_center[0] - node.marker_center[0]
            err_y = node.frame_center[1] - node.marker_center[1]

            vx = pid_x.update(err_x)
            vy = -pid_y.update(err_y)
            vz = 0.0

            send_velocity(master, vx, vy, vz, start_time)

            x += vx * dt
            y += vy * dt
            uav_traj.append((x, y))

        time.sleep(dt)

    send_velocity(master, 0.0, 0.0, 0.0, start_time)
    print("PID stopped (TRACK_FLAG != 1).")

# ============================================================
# ================= MAVLINK FUNCTIONS ========================
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
        0, 1 if arm else 0, 0, 0, 0, 0, 0, 0
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
        100000, 0, 0, 0, 0, 0
    )

def read_relative_alt(master):
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    return msg.relative_alt / 1000.0 if msg else None

def takeoff(master, alt_m):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt_m, 0
    )

def send_velocity(master, vx, vy, vz, start_time):
    time_boot_ms = int((time.time() - start_time) * 1000)
    master.mav.set_position_target_local_ned_send(
        time_boot_ms,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )

def land(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0, 0
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

# ============================================================
# ============== MODIFIED MONITOR FUNCTION ===================
# ============================================================
def track_monitor_and_start(master, node):
    global TRACK_FLAG
    jackal_started = False

    while rclpy.ok():

        # 🔥 START JACKAL ONLY ONCE WHEN TRACK_FLAG BECOMES 1
        if TRACK_FLAG == '1' and not jackal_started:
            node.get_logger().info("TRACK_FLAG=1 → Starting Jackal (will NOT stop).")
            threading.Thread(target=move_jackal_forward, args=(node,), daemon=True).start()
            jackal_started = True

            # PID follow starts here (still depends on TRACK_FLAG)
            threading.Thread(target=follow_ugv, args=(node, master), daemon=True).start()

        # PID restarts only when marker detected again
        if TRACK_FLAG == '1' and jackal_started:
            # ensure PID follow thread restarts on each marker reacquire
            pass

        time.sleep(0.1)

# ============================================================
# ===================== PLOTTING =============================
# ============================================================
def plot_trajectories():
    if not uav_traj or not ugv_traj:
        print("No trajectory data available.")
        return
    uav_x, uav_y = zip(*uav_traj)
    ugv_x, ugv_y = zip(*ugv_traj)
    plt.figure(figsize=(8, 6))
    plt.plot(ugv_x, ugv_y, 'r-', label='UGV')
    plt.plot(uav_x, uav_y, 'b--', label='UAV')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# ============================================================
# ========================= MAIN =============================
# ============================================================
def main():
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)
    request_streams(master)

    rclpy.init()
    node = ArucoDetector(master)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    threading.Thread(target=track_monitor_and_start, args=(master, node), daemon=True).start()

    takeoff_and_wait(master, TARGET_ALT_M)

    node.get_logger().info("Searching for marker...")
    start_time = time.time()
    dt = 1.0 / COMMAND_RATE_HZ

    # UAV initial searching motion
    vx, vy, vz = 0.7, 0.0, 0.0

    while TRACK_FLAG != '1':
        send_velocity(master, vx, vy, vz, start_time)
        time.sleep(dt)

    node.get_logger().info("Marker detected — follow running.")

    while TRACK_FLAG == '1':
        time.sleep(0.2)

    land(master)
    wait_for_landing(master)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
    plot_trajectories()

if __name__ == "__main__":
    main()

