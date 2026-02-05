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
import matplotlib.pyplot as plt  # For plotting trajectories / live errors

from Traj_Pred_EKF_Pub_v2 import RelativePoseEKF
from UAV_Velocity_Estimator import UAVVelocityEstimator
from NMPC_Controller_v4 import SimpleMPCNode

from Plot_Data_v6 import PlotterNode





from rclpy.executors import MultiThreadedExecutor

# ============================================================
# ================= GLOBAL TRACKING FLAG =====================
# ============================================================
TRACK_FLAG = '0'
TRACK_FLAG_LOCK = threading.Lock()  # for safer reads/writes across threads

# ============================================================
# ======== GLOBAL TRAJECTORY LISTS FOR UAV & UGV =============
# ============================================================
uav_traj = []  # stores (x, y) positions of UAV in circle frame
ugv_traj = []  # stores (x, y) positions of UGV

# ============================================================
# ================== LIVE ERROR STORAGE =======================
# ============================================================
err_x_list = []
err_y_list = []
ERR_LOCK = threading.Lock()


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

        # ArUco detector setup (try multiple APIs for compatibility)
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
            # Use first detected marker for center
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

# ============================================================
# ============ MPC VELOCITY SUBSCRIBER NODE ==================
# ============================================================
class MPCVelocitySubscriber(Node):
    def __init__(self, master, start_time):
        """
        Subscribes to /mpc/cmd_vel and sends MPC velocity commands
        directly to the drone via MAVLink.
        """
        super().__init__('mpc_velocity_subscriber')

        self.master = master
        self.start_time = start_time

        # Subscribe to MPC output topic
        self.subscription = self.create_subscription(
            TwistStamped,
            '/mpc/cmd_vel',
            self.cmd_callback,
            20
        )

        self.get_logger().info("MPCVelocitySubscriber Node Started — listening on /mpc/cmd_vel")

    # ----------------------------------------------------------
    #  When MPC publishes velocities, send them directly to UAV
    # ----------------------------------------------------------
    def cmd_callback(self, msg):

        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vz = msg.twist.linear.z
        #vz = 0
        yaw_rate = msg.twist.angular.z

        try:
            send_velocity(self.master, vx, vy, vz, yaw_rate, self.start_time)
        except Exception as e:
            self.get_logger().error(f"Failed sending MPC velocity: {e}")


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
    # request global position at ~10Hz
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

def send_velocity(master, vx, vy, vz, yaw_rate, start_time):
    """Send velocity (vx,vy,vz) + angular velocity (yaw_rate) via SET_POSITION_TARGET_LOCAL_NED."""
    try:
        time_boot_ms = int((time.time() - start_time) * 1000)

        master.mav.set_position_target_local_ned_send(
            time_boot_ms,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000011,   # <-- enable yaw rate + velocities
            0, 0, 0,              # position ignored
            vx, vy, vz,           # linear velocities
            0, 0, 0,              # accel ignored
            0,                    # yaw (ignored)
            yaw_rate              # <-- angular velocity (rad/s)
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

# Move the UGV in a rectangle
def publish_rectangle(node, pub, stop_event):
    """
    Publish smooth (rounded-corner) rectangle motion using a heading controller.
    Rectangle is traced counter-clockwise.
    """

    rate_hz = 20.0
    dt = 1.0 / rate_hz

    # Rectangle parameters (meters)
    Lx = 3.0   # length
    Ly = 3.0   # width
    corner_radius = 0.6  # smooth corner radius (meters)

    # Velocities
    v = 0.25                # forward velocity
    yaw_rate_corner = 0.45  # angular velocity during rounded corners

    # Internal timing
    # straight segment time = (segment_length - 2*corner_radius) / v
    t1 = (Lx - 2 * corner_radius) / v
    t2 = (Ly - 2 * corner_radius) / v
    # corner duration = quarter circle arc = (pi/2 * R) / v
    t_corner = (3.14159 / 2 * corner_radius) / v

    # Complete cycle timing
    T = 2 * (t1 + t2) + 4 * t_corner  # full loop
    t = 0.0

    while not stop_event.is_set() and rclpy.ok():
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()

        # ---- Phase 1: long straight ----
        if 0 <= t < t1:
            vx = v
            wz = 0.0

        # ---- Phase 2: smooth rounded corner ----
        elif t1 <= t < t1 + t_corner:
            vx = v
            wz = yaw_rate_corner

        # ---- Phase 3: short straight ----
        elif t1 + t_corner <= t < t1 + t_corner + t2:
            vx = v
            wz = 0.0

        # ---- Phase 4: rounded corner ----
        elif t1 + t_corner + t2 <= t < t1 + t_corner + t2 + t_corner:
            vx = v
            wz = yaw_rate_corner

        # ---- Phase 5: second long straight ----
        elif t1 + 2*t_corner + t2 <= t < t1 + 2*t_corner + t2 + t1:
            vx = v
            wz = 0.0

        # ---- Phase 6: rounded corner ----
        elif t1 + 2*t_corner + t2 + t1 <= t < t1 + 3*t_corner + t2 + t1:
            vx = v
            wz = yaw_rate_corner

        # ---- Phase 7: second short straight ----
        elif t1 + 3*t_corner + t2 + t1 <= t < t1 + 3*t_corner + 2*t2 + t1:
            vx = v
            wz = 0.0

        # ---- Phase 8: last corner ----
        else:
            vx = v
            wz = yaw_rate_corner

        # ---- Publish ----
        msg.twist.linear.x = vx
        msg.twist.angular.z = wz
        pub.publish(msg)

        # Time update
        t += dt
        if t > T:  # restart rectangle
            t = 0.0

        time.sleep(dt)

    # ---- Publishing zero velocity before stopping ----
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)


# ============================================================
# ========================= MAIN =============================
# ============================================================
def main():
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)
    request_streams(master)

    rclpy.init()
    
    # create nodes
    node = ArucoDetector(master)
    ekf_node = RelativePoseEKF()
    vel_node = UAVVelocityEstimator()
    mpc_node = SimpleMPCNode()
    plotter_node = PlotterNode()

    # NEW MPC subscriber node
    #start_time = time.time()
    #mpc_subscriber = MPCVelocitySubscriber(master,start_time)

    # create an executor and add nodes (use multiple threads to allow callbacks in parallel)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(ekf_node)
    executor.add_node(vel_node)
    executor.add_node(mpc_node)
    executor.add_node(plotter_node)
    #executor.add_node(mpc_subscriber) 
    

    # spin executor in a background thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()


    # optional events for other threads (avoid NameError on cleanup)
    plot_stop_event = threading.Event()
    stop_event_rect = threading.Event()

    # start jackal rectangular motion publisher (optional)
    #jackal_pub = node.create_publisher(
    #    TwistStamped, '/jackal/jackal_velocity_controller/cmd_vel', 10
    #)
    #stop_event_rect = threading.Event()
    #pub_thread = threading.Thread(target=publish_rectangle, args=(node, jackal_pub, stop_event_rect), daemon=True)
    #pub_thread.start()

    # UAV takeoff and hover
    takeoff_and_wait(master, TARGET_ALT_M)

    node.get_logger().info("Takeoff complete. UAV hovering at target altitude.")
    start_time = time.time()
    dt = 1.0 / COMMAND_RATE_HZ

    # --------------------------------------------
    # ADD THIS LINE HERE — STABILIZATION DELAY
    # --------------------------------------------
    time.sleep(1.0)   # ensure UAV is stable after takeoff
    
    # -----------------------------------------------------------
    # START MPC SUBSCRIBER ONLY NOW — AFTER TAKEOFF IS COMPLETE
    # -----------------------------------------------------------
    start_time = time.time()
    mpc_subscriber = MPCVelocitySubscriber(master, start_time)
    executor.add_node(mpc_subscriber)

    node.get_logger().info("MPC subscriber activated — ready to receive /mpc/cmd_vel")

    #try:
        #  UAV initial searching motion (forward / yaw scan) until marker detected.
        #while rclpy.ok():
            #with TRACK_FLAG_LOCK:
                #local_flag = TRACK_FLAG

            #if local_flag == '0':
                #vx_search, vy_search, vz, yaw_rate = 0.0, 0.0, 0.0, 0.1

                # keep sending search velocity until marker is detected (TRACK_FLAG becomes '1')
                #while True:
                    #with TRACK_FLAG_LOCK:
                        #if TRACK_FLAG == '1':
                            #break
                    #send_velocity(master, vx_search, vy_search, vz, yaw_rate, start_time)
                    #time.sleep(dt)
                # when marker found the follow_ugv thread will take over
            #else:
                # small sleep to avoid busy-looping when marker already seen
                #time.sleep(0.1)

    #except KeyboardInterrupt:
        #print("Keyboard interrupt — cleaning up and landing...")

    # graceful shutdown: stop threads and land
    plot_stop_event.set()
    stop_event_rect.set()

    try:
        send_velocity(master, 0.0, 0.0, 0.0, 0.0, start_time)
        land(master)
        wait_for_landing(master)
    except Exception as e:
        print("Error during landing sequence:", e)

            # Shutdown rclpy executor cleanly
    
    try:
        executor.shutdown(wait=False)
    except Exception:
        pass

        # remove nodes from executor, then destroy
    try:
        executor.remove_node(node)
        executor.remove_node(ekf_node)
        executor.remove_node(vel_node)
        executor.remove_node(mpc_node)
        executor.remove_node(plotter_node)
        executor.remove_node(mpc_subscriber)
    except Exception:
        pass

    try:
        node.destroy_node()
    except Exception:
        pass

    try:
        ekf_node.destroy_node()
    except Exception:
        pass

    try:
        vel_node.destroy_node()          
    except Exception:
        pass

    try:
        mpc_node.destroy_node()          
    except Exception:
        pass
        
    try:
        plotter_node.destroy_node()
    except Exception:
        pass

    try:
        mpc_subscriber.destroy_node()
    except Exception:
        pass


    # finally shutdown rclpy and other resources
    try:
        rclpy.shutdown()
    except Exception:
        pass

    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()

