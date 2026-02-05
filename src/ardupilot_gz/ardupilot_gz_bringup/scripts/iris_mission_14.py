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
# ================ ARUCO DETECTOR NODE (ROS 2) ===============
# ============================================================
class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.tag_detected = False  # Shared variable

        # Publisher now uses TwistStamped messages
        self.jackal_pub = self.create_publisher(
            TwistStamped,
            '/jackal/jackal_velocity_controller/cmd_vel',
            10
        )

        # Subscribe to camera topic
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10
        )

        # Use Original ArUco Dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.get_logger().info("Aruco Detector Node Started ✅ (DICT_ARUCO_ORIGINAL)")

    def move_jackal_forward(self, duration=5.0, speed=0.5):
        """Move Jackal forward for a specified duration using TwistStamped."""
        msg = TwistStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = speed
        msg.twist.angular.z = 0.0

        start_time = time.time()
        rate = self.create_rate(10)  # 10 Hz

        self.get_logger().info(f"🚙 Moving Jackal forward for {duration} seconds...")
        while (time.time() - start_time) < duration:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.jackal_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

        # Stop Jackal after motion
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        msg.header.stamp = self.get_clock().now().to_msg()
        self.jackal_pub.publish(msg)
        self.get_logger().info("✅ Jackal stopped after moving forward.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            if not self.tag_detected:
                print("🔶 Tag Detected")
                self.tag_detected = True

                # Move Jackal forward when tag detected (in background thread)
                threading.Thread(target=self.move_jackal_forward, daemon=True).start()

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID: {marker_id}", (center_x - 20, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)


# ============================================================
# ================= MAVLINK FLIGHT CONTROLLER ================
# ============================================================
TARGET_ALT_M = 2.2
CIRCLE_RADIUS_M = 2.0
CIRCLE_SPEED_DEG_PER_SEC = 30
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
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, alt_m
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
    set_mode(master, "LAND")

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
def mavlink_mission():
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)
    request_streams(master)

    # ========== Phase 1 ==========
    print("\n🚁 === Phase 1: Takeoff → Straight Flight → Land ===")
    takeoff_and_wait(master, TARGET_ALT_M)
    print("➡️ Moving forward for 5 seconds...")
    start_time = time.time()
    duration = 5
    dt = 1.0 / COMMAND_RATE_HZ
    while time.time() - start_time < duration:
        send_velocity(master, 0.7, 0.0, 0.0, start_time)
        time.sleep(dt)

    land(master)
    wait_for_landing(master)

    # ========== Phase 2 ==========
    print("\n🌀 === Phase 2: Takeoff → Circle → Land ===")
    takeoff_and_wait(master, TARGET_ALT_M)
    print("🌀 Flying in a circular path (velocity-based)...")

    start_time = time.time()
    dt = 1.0 / COMMAND_RATE_HZ
    circle_duration = 360 / CIRCLE_SPEED_DEG_PER_SEC
    omega = math.radians(CIRCLE_SPEED_DEG_PER_SEC)

    x_hist, y_hist = [], []
    while True:
        elapsed = time.time() - start_time
        if elapsed > circle_duration:
            break

        angle = omega * elapsed
        vx = -CIRCLE_RADIUS_M * omega * math.sin(angle)
        vy =  CIRCLE_RADIUS_M * omega * math.cos(angle)
        vz = 0
        send_velocity(master, vx, vy, vz, start_time)
        x_hist.append(CIRCLE_RADIUS_M * math.cos(angle))
        y_hist.append(CIRCLE_RADIUS_M * math.sin(angle))
        time.sleep(dt)

    print("✅ Circle complete. Landing...")
    land(master)
    wait_for_landing(master)

    plt.figure()
    plt.plot(y_hist, x_hist, 'b-', linewidth=2)
    plt.title("Drone Circular Trajectory")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    print("✅ Mission complete!")

def main():
    # Start ROS 2 Aruco node in parallel
    rclpy.init()
    node = ArucoDetector()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        # Run MAVLink mission in main thread
        mavlink_mission()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

