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
import matplotlib.pyplot as plt
from pymavlink import mavutil

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
# ================ ARUCO DETECTOR NODE (ROS 2) ===============
# ============================================================
class ArucoDetector(Node):
    def __init__(self, master):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.master = master
        self.tag_detected = False
        self.marker_center = None
        self.marker_size = None
        self.frame_center = None
        self.following = False

        # PID controllers
        self.pid_x = PID(0.002, 0.0, 0.001, (-0.5, 0.5))  # horizontal
        self.pid_y = PID(0.002, 0.0, 0.001, (-0.5, 0.5))  # vertical
        self.pid_z = PID(0.02, 0.0, 0.005, (-0.8, 0.8))   # forward/back

        # Publisher for Jackal
        self.jackal_pub = self.create_publisher(
            TwistStamped,
            '/jackal/jackal_velocity_controller/cmd_vel',
            10
        )

        # Subscribe to camera topic
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10
        )

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.get_logger().info("Aruco Detector Node Started ✅")

    def move_jackal_forward(self, duration=5.0, speed=0.4):
        msg = TwistStamped()
        msg.header = Header()
        msg.twist.linear.x = speed
        msg.twist.angular.z = 0.0

        start_time = time.time()
        rate = self.create_rate(10)
        while (time.time() - start_time) < duration:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.jackal_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

        msg.twist.linear.x = 0.0
        self.jackal_pub.publish(msg)
        self.get_logger().info("✅ Jackal stopped.")

    def follow_ugv(self):
        """PID-based drone following using ArUco feedback."""
        print("🚁 Drone started following UGV...")
        start_time = time.time()
        dt = 1.0 / 10  # 10 Hz

        while self.tag_detected:
            if self.marker_center and self.frame_center and self.marker_size:
                err_x = self.frame_center[0] - self.marker_center[0]
                err_y = self.frame_center[1] - self.marker_center[1]
                err_z = 100 - self.marker_size  # desired marker size ~100 px

                vx = self.pid_z.update(err_z)     # forward/backward
                vy = self.pid_x.update(err_x)     # left/right
                vz = -self.pid_y.update(err_y)    # up/down

                send_velocity(self.master, vx, vy, vz, start_time)
            time.sleep(dt)
        print("🛑 Drone stopped following (marker lost).")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        self.frame_center = (w / 2, h / 2)
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            if not self.tag_detected:
                self.tag_detected = True
                threading.Thread(target=self.move_jackal_forward, daemon=True).start()
                threading.Thread(target=self.follow_ugv, daemon=True).start()

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            c = corners[0][0]
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())
            size = cv2.contourArea(corners[0])

            self.marker_center = (cx, cy)
            self.marker_size = math.sqrt(size)

            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Marker Center ({cx},{cy})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            if self.tag_detected:
                self.tag_detected = False

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

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
                print("✅ Armed." if arm else "✅ Disarmed.")
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
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0,
        0, 0, alt_m,
        0
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
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, 0,
        0, 0, 0,
        0
    )

def takeoff_and_wait(master, alt):
    set_mode(master, "GUIDED")
    arm_and_wait(master, True)
    takeoff(master, alt)
    while True:
        alt_now = read_relative_alt(master)
        if alt_now and alt_now >= alt * 0.95:
            print("✅ Reached target altitude")
            break
        time.sleep(0.5)

def wait_for_landing(master):
    while True:
        alt = read_relative_alt(master)
        if alt and alt <= 0.2:
            break
        time.sleep(0.5)
    arm_and_wait(master, False)
    print("✅ Landed and disarmed.")

# ============================================================
# ===================== COMBINED MAIN ========================
# ============================================================
def main():
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)
    request_streams(master)

    rclpy.init()
    node = ArucoDetector(master)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    takeoff_and_wait(master, TARGET_ALT_M)

    # 🟢 Drone moves horizontally forward before detection
    print("➡️ Moving forward while searching for ArUco marker...")
    start_time = time.time()
    dt = 1.0 / COMMAND_RATE_HZ
    vx, vy, vz = 0.7, 0.0, 0.0  # horizontal forward speed

    while not node.tag_detected:
        send_velocity(master, vx, vy, vz, start_time)
        time.sleep(dt)

    print("🔶 ArUco marker detected — switching to PID following mode.")
    while node.tag_detected:
        time.sleep(0.5)

    land(master)
    wait_for_landing(master)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

