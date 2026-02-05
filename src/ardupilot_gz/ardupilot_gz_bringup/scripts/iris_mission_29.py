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
import matplotlib.pyplot as plt  # ✅ For plotting trajectories

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
pid_x = PID(0.002, 0.0, 0.001, (-0.5, 0.5))  # left/right
pid_y = PID(0.002, 0.0, 0.001, (-0.5, 0.5))  # up/down

# ============================================================
# ======== GLOBAL TRAJECTORY LISTS FOR UAV & UGV =============
# ============================================================
uav_traj = []   # (x, y)
ugv_traj = []   # (x, y)

# ============================================================
# ================ ARUCO DETECTOR NODE (ROS 2) ===============
# ============================================================
class ArucoDetector(Node):
    def __init__(self, master):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.master = master
        self.tag_detected = False
        self.marker_center = None
        self.frame_center = None

        # Camera subscription
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10
        )

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.get_logger().info("Aruco Detector Node Started ✅")

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
                self.get_logger().info("🔶 Marker detected — TRACK_FLAG set to '1'")

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            c = corners[0][0]
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())

            self.marker_center = (cx, cy)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Marker Center ({cx},{cy})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            if self.tag_detected:
                self.tag_detected = False
                TRACK_FLAG = '0'
                self.get_logger().info("⚠️ Marker lost — TRACK_FLAG set to '0'")

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

# ============================================================
# ================ JACKAL & UAV FOLLOW FUNCTIONS =============
# ============================================================

# ******* 🔥 MODIFIED FOR CIRCULAR MOTION ********
def move_jackal_forward(node, duration=5.0, speed=0.4):
    """Moves Jackal (UGV) in a circular trajectory of radius 2 m anticlockwise."""
    global ugv_traj

    radius = 2.0
    omega = 0.2   # rad/s (anticlockwise)
    linear_speed = radius * omega  # = 0.4 m/s

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

    while (time.time() - start_time) < duration and TRACK_FLAG == '1':
        msg.header.stamp = node.get_clock().now().to_msg()
        jackal_pub.publish(msg)

        theta += omega * dt
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        ugv_traj.append((x, y))

        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(dt)

    msg.twist.linear.x = 0.0
    msg.twist.angular.z = 0.0
    jackal_pub.publish(msg)
    node.get_logger().info("✅ Jackal circular motion stopped.")
# ******************************************************

def follow_ugv(node, master):
    global pid_x, pid_y, uav_traj
    print("🚁 Drone started following UGV (PID active, X-Y only)...")
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
            vy = 0.5*vy
            vz = 0.0

            send_velocity(master, vx, vy, vz, start_time)

            x += vx * dt
            y += vy * dt
            uav_traj.append((x, y))

        time.sleep(dt)

    send_velocity(master, 0.0, 0.0, 0.0, start_time)
    print("🛑 Drone stopped following (TRACK_FLAG != '1').")

# ============================================================
# ================= MAVLINK FLIGHT CONTROLLER ================
# ============================================================
TARGET_ALT_M = 2.0
CONN_STR = 'udp:127.0.0.1:14551'
COMMAND_RATE_HZ = 5

def wait_heartbeat(master):
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"✅ Heartbeat from system {master.target_system} component {master.target_component}")

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
            print(f"✅ Mode set to {mode_name}")
            return
    print(f"⚠️ Failed to set mode {mode_name}")

def arm_and_wait(master, arm=True):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1 if arm else 0, 0, 0, 0, 0, 0, 0
    )
    while True:
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb:
            armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            if armed == arm:
                print("✔️ Armed." if arm else "✔️ Disarmed.")
                return

def request_streams(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        1_000_000 // 10, 0, 0, 0, 0, 0
    )

def read_relative_alt(master):
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    return msg.relative_alt / 1000.0 if msg else None

def takeoff(master, alt_m):
    print(f"🚁 Taking off to {alt_m} m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt_m, 0
    )

def send_velocity(master, vx, vy, vz, start_time):
    time_boot_ms = int((time.time() - start_time) * 1000) % 4294967295
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
    print("🛬 Landing...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0, 0
    )

def takeoff_and_wait(master, alt):
    set_mode(master, "GUIDED")
    arm_and_wait(master, True)
    takeoff(master, alt)
    while True:
        alt_now = read_relative_alt(master)
        if alt_now and alt_now >= alt * 0.95:
            print("🚀 Reached target altitude")
            break
        time.sleep(0.5)

def wait_for_landing(master):
    while True:
        alt = read_relative_alt(master)
        if alt and alt <= 0.2:
            break
        time.sleep(0.5)
    arm_and_wait(master, False)
    print("✔️ Landed and disarmed.")

# ============================================================
# ===================== MONITOR & MAIN =======================
# ============================================================
def track_monitor_and_start(master, node):
    global TRACK_FLAG
    already_started = False

    while rclpy.ok():
        if TRACK_FLAG == '1' and not already_started:
            node.get_logger().info("TRACK_FLAG == '1' — starting Jackal and PID follow threads.")
            threading.Thread(target=move_jackal_forward, args=(node, 10.0, 0.4), daemon=True).start()
            threading.Thread(target=follow_ugv, args=(node, master), daemon=True).start()
            already_started = True

        if TRACK_FLAG != '1' and already_started:
            node.get_logger().info("TRACK_FLAG cleared — ready for restart.")
            already_started = False

        time.sleep(0.1)

def plot_trajectories():
    if not uav_traj or not ugv_traj:
        print("⚠️ No trajectory data to plot.")
        return
    uav_x, uav_y = zip(*uav_traj)
    ugv_x, ugv_y = zip(*ugv_traj)

    plt.figure(figsize=(8, 6))
    plt.plot(ugv_x, ugv_y, 'r-', label='UGV Trajectory')
    plt.plot(uav_x, uav_y, 'b--', label='UAV Trajectory')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('UAV and UGV Trajectories')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

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

    node.get_logger().info("Searching for ArUco marker while moving forward...")
    start_time = time.time()
    dt = 1.0 / COMMAND_RATE_HZ
    vx, vy, vz = 0.7, 0.0, 0.0

    while TRACK_FLAG != '1':
        send_velocity(master, vx, vy, vz, start_time)
        time.sleep(dt)

    node.get_logger().info("Marker detected — follow threads running.")

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

