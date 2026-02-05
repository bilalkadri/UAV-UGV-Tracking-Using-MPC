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


# ============================================================
# ================= GLOBAL TRACKING FLAG =====================
# ============================================================
TRACK_FLAG = '0'
TRACK_FLAG_LOCK = threading.Lock()  # for safer reads/writes across threads



# ============================================================
# ================ ARUCO DETECTOR NODE ========================
# ============================================================
class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        self.bridge = CvBridge()
        self.master = None
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

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()





