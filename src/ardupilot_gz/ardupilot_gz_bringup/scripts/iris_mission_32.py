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
TRACK_FLAG_LOCK = threading.Lock()  # for safer reads/writes across threads

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
# ======== GLOBAL TRAJECTORY LISTS FOR UAV & UGV =============
# ============================================================
uav_traj = []  # stores (x, y) positions of UAV in circle frame
ugv_traj = []  # stores (x, y) positions of UGV

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

        # Some OpenCV versions prefer DetectorParameters_create(); this should still work in recent versions:
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        except Exception:
            # fallback for other opencv builds
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.get_logger().info("Aruco Detector Node Started")

    def image_callback(self, msg):
        global TRACK_FLAG
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        self.frame_center = (w / 2.0, h / 2.0)

        corners, ids, _ = self.detector.detectMarkers(frame)

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
            # keep previous marker_center (optional); here we clear it
            self.marker_center = None

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

# ============================================================
# ================= JACKAL & UAV FOLLOW ======================
# ============================================================

def move_jackal_forward(node, duration=5.0, speed=0.4):
    """UGV moves in a circular trajectory continuously and independently."""
    global ugv_traj

    radius = 3.0
    omega = 0.02
    linear_speed = radius * omega

    jackal_pub = node.create_publisher(
        TwistStamped, '/jackal/jackal_velocity_controller/cmd_vel', 10
    )
    msg = TwistStamped()
    msg.header = Header()

    # We'll publish angular z and linear x to create circular motion (differential controller on Jackal)
    msg.twist.linear.x = linear_speed
    msg.twist.angular.z = omega

    x, y = radius, 0.0  # initial position on circle
    theta = 0.0
    dt = 0.1

    # Jackal keeps moving forever (independent of TRACK_FLAG)
    while rclpy.ok():
        msg.header.stamp = node.get_clock().now().to_msg()
        jackal_pub.publish(msg)

        theta += omega * dt
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        ugv_traj.append((x, y))

        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(dt)

def move_uav_circle(master, node):
    """UAV moves in a circular trajectory continuously (PID overlays on top when activated).
       In Option A PID overrides circle when TRACK_FLAG == '1' (so this thread sends circle velocities only
       when TRACK_FLAG == '0')."""
    global uav_traj

    radius = 3.0
    omega = 0.02   # same angular rate as jackal
    dt = 0.1
    theta = 0.0

    start_time = time.time()
    x, y = radius, 0.0

    while True:
        # Compute circle velocities in local frame (derivative of position)
        vx_circle = -radius * omega * math.sin(theta)
        vy_circle = radius * omega * math.cos(theta)
        vz = 0.0

        # If PID is NOT active (TRACK_FLAG == '0'), send circular velocity
        with TRACK_FLAG_LOCK:
            local_flag = TRACK_FLAG

        if local_flag == '0':
            # send circle velocity directly
            send_velocity(master, vx_circle, vy_circle, vz, start_time)

            # integrate to store approximate UAV position on circle
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            uav_traj.append((x, y))

        # If TRACK_FLAG == '1' we do NOT send circle velocity here (PID thread overrides)
        theta += omega * dt
        time.sleep(dt)

def follow_ugv(node, master):
    """PID control loop that activates only while TRACK_FLAG == '1'.
       Option A: PID fully overrides circular motion while active."""
    global pid_x, pid_y, uav_traj

    dt = 0.1

    while True:
        # busy-loop that waits for TRACK_FLAG == '1'
        with TRACK_FLAG_LOCK:
            local_flag = TRACK_FLAG

        if local_flag == '1':
            pid_x.reset()
            pid_y.reset()
            start_time = time.time()

            # track until marker lost
            while True:
                with TRACK_FLAG_LOCK:
                    if TRACK_FLAG != '1':
                        break

                if node.marker_center and node.frame_center:
                    err_x = node.frame_center[0] - node.marker_center[0]
                    err_y = node.frame_center[1] - node.marker_center[1]

                    # PID outputs velocities in image-pixel space -> you may want mapping to world velocities;
                    # here we use the PID outputs directly as vx, vy (small gains). vy is inverted for image coords.
                    vx = pid_x.update(err_x)
                    vy = -pid_y.update(err_y)
                    vz = 0.0

                    send_velocity(master, vx, vy, vz, start_time)

                    # approximate integration to add to uav_traj (not in world units but useful for plotting)
                    # integrate with dt to get relative movement
                    if uav_traj:
                        last_x, last_y = uav_traj[-1]
                    else:
                        last_x, last_y = 0.0, 0.0
                    new_x = last_x + vx * dt
                    new_y = last_y + vy * dt
                    uav_traj.append((new_x, new_y))
                else:
                    # if no marker_center available but TRACK_FLAG says 1, send zero to avoid uncontrolled movement
                    send_velocity(master, 0.0, 0.0, 0.0, start_time)

                time.sleep(dt)

            # marker lost: ensure we stop any PID velocity and let move_uav_circle resume sending circle velocities
            send_velocity(master, 0.0, 0.0, 0.0, start_time)
            print("PID stopped (TRACK_FLAG != '1').")

        # if TRACK_FLAG isn't '1' sleep a bit and re-check
        time.sleep(0.1)

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
    # defensive check
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
    # request global position at ~10Hz (100000 microseconds interval argument kept from your original)
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
    """Send NED-local velocity (vx forward, vy right, vz down) via MAVLink SET_POSITION_TARGET_LOCAL_NED."""
    # Defensive: require master to be connected
    try:
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
    except Exception as e:
        print(f"send_velocity error: {e}")

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
    """Starts jackal circular motion and UAV circular motion once, when marker detected first time.
       PID thread always runs but activates only when TRACK_FLAG == '1'."""
    jackal_started = False
    uav_circle_started = False
    pid_thread_started = False

    while rclpy.ok():
        with TRACK_FLAG_LOCK:
            local_flag = TRACK_FLAG

        # Start jackal and UAV circle threads only once when marker is first detected
        if local_flag == '1' and not jackal_started:
            node.get_logger().info("TRACK_FLAG=1 → Starting Jackal circular motion.")
            threading.Thread(target=move_jackal_forward, args=(node,), daemon=True).start()
            jackal_started = True

        if local_flag == '1' and not uav_circle_started:
            node.get_logger().info("TRACK_FLAG=1 → Starting UAV circular motion (will pause while PID active).")
            threading.Thread(target=move_uav_circle, args=(master, node), daemon=True).start()
            uav_circle_started = True

        # Start the PID thread once (it will monitor TRACK_FLAG internally)
        if not pid_thread_started:
            threading.Thread(target=follow_ugv, args=(node, master), daemon=True).start()
            pid_thread_started = True

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
    plt.plot(ugv_x, ugv_y, 'r-', label='UGV (Jackal)')
    plt.plot(uav_x, uav_y, 'b--', label='UAV')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.title("Trajectories")
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

    # run ROS spin in background
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # monitor thread to start motion threads once marker seen
    threading.Thread(target=track_monitor_and_start, args=(master, node), daemon=True).start()

    # UAV takeoff and search motion
    takeoff_and_wait(master, TARGET_ALT_M)

    node.get_logger().info("Searching for marker...")
    start_time = time.time()
    dt = 1.0 / COMMAND_RATE_HZ

    # UAV initial searching motion (forward)
    vx_search, vy_search, vz = 0.7, 0.0, 0.0

    # keep sending search velocity until marker is detected (TRACK_FLAG becomes '1')
    while True:
        with TRACK_FLAG_LOCK:
            if TRACK_FLAG == '1':
                break
        send_velocity(master, vx_search, vy_search, vz, start_time)
        time.sleep(dt)

    node.get_logger().info("Marker detected — circular motions and PID threads should be running.")

    # Wait while system runs — main thread just monitors TRACK_FLAG and eventually lands when you desire
    try:
        while True:
            # If you want automatic landing when marker lost for long time, implement counter here.
            # For now we keep running until the process is interrupted.
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Keyboard interrupt, landing and shutting down...")

    # STOP: send zero velocities and land
    send_velocity(master, 0.0, 0.0, 0.0, start_time)
    land(master)
    wait_for_landing(master)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
    plot_trajectories()

if __name__ == "__main__":
    main()

