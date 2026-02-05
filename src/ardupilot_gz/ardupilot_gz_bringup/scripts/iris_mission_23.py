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

        # PID terms
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Limit output
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        # Save state
        self.prev_error = error
        self.prev_time = now
        return output

# ============================================================
# ================= GLOBAL PID CONTROLLERS ===================
# ============================================================
# Tuned gains for smoother control
pid_x = PID(0.003, 0.00001, 0.002, (-0.8, 0.8))  # horizontal
pid_y = PID(0.003, 0.00001, 0.002, (-0.8, 0.8))  # vertical

# ============================================================
# ======== GLOBAL TRAJECTORY LISTS FOR UAV & UGV =============
# ============================================================
uav_traj = []
ugv_traj = []

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
        self.filtered_center = None  # smoothed marker center

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

            # Apply simple low-pass filter to reduce noise
            if self.filtered_center is None:
                self.filtered_center = (cx, cy)
            else:
                alpha = 0.3
                fx = alpha * cx + (1 - alpha) * self.filtered_center[0]
                fy = alpha * cy + (1 - alpha) * self.filtered_center[1]
                self.filtered_center_

