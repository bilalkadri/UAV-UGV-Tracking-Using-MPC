#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from apriltag_msgs.msg import AprilTagDetectionArray
from pymavlink import mavutil
import math, time, threading, matplotlib.pyplot as plt

# ==============================
# Configuration
# ==============================
TARGET_ALT_M = 2.0
CIRCLE_RADIUS_M = 2.0
CIRCLE_SPEED_DEG_PER_SEC = 30
CONN_STR = 'udp:127.0.0.1:14551'
COMMAND_RATE_HZ = 5
PAUSE_TIME_SEC = 3.0  # pause duration when tag detected

class DroneMissionNode(Node):
    def __init__(self):
        super().__init__('drone_mission_node')

        # MAVLink setup
        self.master = mavutil.mavlink_connection(CONN_STR)
        self.wait_heartbeat()
        self.request_streams()

        # AprilTag subscription
        self.create_subscription(
            AprilTagDetectionArray,
            '/tag_detections',
            self.tag_callback,
            10
        )

        # Shared flag for tag detection
        self.tag_detected = False
        self.pause_lock = threading.Lock()

        # Run mission in a background thread
        self.mission_thread = threading.Thread(target=self.run_mission)
        self.mission_thread.start()

    # ==============================
    # MAVLink Utility Functions
    # ==============================
    def wait_heartbeat(self):
        self.get_logger().info("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        self.get_logger().info(f"✅ Heartbeat from system {self.master.target_system}")

    def set_mode(self, mode_name):
        modes = self.master.mode_mapping()
        mode_id = modes[mode_name]
        for _ in range(10):
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            hb = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if hb and mavutil.mode_string_v10(hb) == mode_name:
                self.get_logger().info(f"✅ Mode set to {mode_name}")
                return
        self.get_logger().warn(f"⚠️ Failed to set mode {mode_name}")

    def arm_and_wait(self, arm=True):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1 if arm else 0, 0, 0, 0, 0, 0, 0
        )
        while True:
            hb = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if hb:
                armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                if armed == arm:
                    self.get_logger().info("✅ Armed." if arm else "✅ Disarmed.")
                    return

    def request_streams(self):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            1_000_000 // 10, 0, 0, 0, 0, 0
        )

    def read_relative_alt(self):
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        return msg.relative_alt / 1000.0 if msg else None

    def takeoff(self, alt_m):
        self.get_logger().info(f"🚁 Taking off to {alt_m} m...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, alt_m
        )

    def send_velocity(self, vx, vy, vz, start_time):
        time_boot_ms = int((time.time() - start_time) * 1000) % 4294967295
        self.master.mav.set_position_target_local_ned_send(
            time_boot_ms,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )

    def land(self):
        self.get_logger().info("🛬 Landing...")
        self.set_mode("LAND")

    # ==============================
    # Mission Steps
    # ==============================
    def takeoff_and_wait(self, alt):
        self.set_mode("GUIDED")
        self.arm_and_wait(True)
        self.takeoff(alt)
        while True:
            alt_now = self.read_relative_alt()
            if alt_now and alt_now >= alt * 0.95:
                self.get_logger().info("✅ Reached target altitude")
                break
            time.sleep(0.5)

    def wait_for_landing(self):
        while True:
            alt = self.read_relative_alt()
            if alt and alt <= 0.2:
                break
            time.sleep(0.5)
        self.arm_and_wait(False)
        self.get_logger().info("✅ Landed and disarmed.")

    # ==============================
    # Pause mechanism when tag detected
    # ==============================
    def pause_if_tag_detected(self):
        with self.pause_lock:
            if self.tag_detected:
                self.get_logger().info("📸 Tag detected – pausing for inspection...")
                # Stop movement (zero velocity)
                self.send_velocity(0.0, 0.0, 0.0, time.time())
                time.sleep(PAUSE_TIME_SEC)
                self.tag_detected = False
                self.get_logger().info("✅ Resuming mission...")

    # ==============================
    # Mission Execution
    # ==============================
    def run_mission(self):
        # -----------------------------
        # PHASE 1: Straight flight
        # -----------------------------
        self.get_logger().info("🚁 === Phase 1: Takeoff → Straight Flight → Land ===")
        self.takeoff_and_wait(TARGET_ALT_M)

        start_time = time.time()
        duration = 5
        dt = 1.0 / COMMAND_RATE_HZ
        while time.time() - start_time < duration:
            self.pause_if_tag_detected()
            self.send_velocity(1.0, 0.0, 0.0, start_time)
            time.sleep(dt)

        self.land()
        self.wait_for_landing()

        # -----------------------------
        # PHASE 2: Circle flight
        # -----------------------------
        self.get_logger().info("🌀 === Phase 2: Takeoff → Circle → Land ===")
        self.takeoff_and_wait(TARGET_ALT_M)

        dt = 1.0 / COMMAND_RATE_HZ
        circle_duration = 360 / CIRCLE_SPEED_DEG_PER_SEC
        omega = math.radians(CIRCLE_SPEED_DEG_PER_SEC)
        x_hist, y_hist = [], []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > circle_duration:
                break
            self.pause_if_tag_detected()
            angle = omega * elapsed
            vx = -CIRCLE_RADIUS_M * omega * math.sin(angle)
            vy =  CIRCLE_RADIUS_M * omega * math.cos(angle)
            vz = 0
            self.send_velocity(vx, vy, vz, start_time)
            x_hist.append(CIRCLE_RADIUS_M * math.cos(angle))
            y_hist.append(CIRCLE_RADIUS_M * math.sin(angle))
            time.sleep(dt)

        self.land()
        self.wait_for_landing()
        self.get_logger().info("✅ Mission complete!")

        plt.figure()
        plt.plot(y_hist, x_hist, 'b-', linewidth=2)
        plt.title("Drone Circular Trajectory")
        plt.xlabel("East (m)")
        plt.ylabel("North (m)")
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    # ==============================
    # AprilTag Callback
    # ==============================
    def tag_callback(self, msg: AprilTagDetectionArray):
        if msg.detections:
            with self.pause_lock:
                if not self.tag_detected:  # trigger only once per detection window
                    self.tag_detected = True
                    self.get_logger().info("📸 Tag detected")

def main(args=None):
    rclpy.init(args=args)
    node = DroneMissionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

