#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import threading
import numpy as np

# ===== Optional Global Flags =====
TRACK_FLAG = '0'
TRACK_FLAG_LOCK = threading.Lock()

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        self.bridge = CvBridge()

        # states
        self.tag_detected = False
        self.marker_center = None
        self.frame_center = None

        # camera parameters (replace with your calibration values)
        self.marker_size = 0.20  # meters
        self.camera_matrix = np.array([
            [600, 0, 320],
            [0, 600, 240],
            [0, 0, 1]
        ], dtype=float)
        self.dist_coeffs = np.zeros((5, 1))

        # subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

        # publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 10)
        self.center_pub = self.create_publisher(PointStamped, '/aruco/marker_center', 10)
        self.detected_pub = self.create_publisher(Bool, '/aruco/detected', 10)

        # ArUco setup
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        except Exception:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.get_logger().info("Aruco Detector Node Started")

    def image_callback(self, msg):
        global TRACK_FLAG

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w, _ = frame.shape
        self.frame_center = (w / 2.0, h / 2.0)

        corners, ids, _ = self.detector.detectMarkers(frame)

        detected_msg = Bool()
        detected_msg.data = False

        if ids is not None and len(ids) > 0:
            if not self.tag_detected:
                self.tag_detected = True
                with TRACK_FLAG_LOCK:
                    TRACK_FLAG = '1'
                self.get_logger().info("Marker detected — TRACK_FLAG = 1")

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # first marker center
            c = corners[0][0]
            cx = float(c[:, 0].mean())
            cy = float(c[:, 1].mean())
            self.marker_center = (cx, cy)

            # publish marker center
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "camera"
            point_msg.point.x = cx
            point_msg.point.y = cy
            point_msg.point.z = 0.0
            self.center_pub.publish(point_msg)

            # === FIXED LINE HERE ===
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            # convert to PoseStamped
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera"

            pose_msg.pose.position.x = float(tvec[0])
            pose_msg.pose.position.y = float(tvec[1])
            pose_msg.pose.position.z = float(tvec[2])

            # rotation
            rot_matrix, _ = cv2.Rodrigues(rvec)
            quat = self.rotation_matrix_to_quaternion(rot_matrix)
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            self.pose_pub.publish(pose_msg)

            detected_msg.data = True

        else:
            if self.tag_detected:
                self.tag_detected = False
                with TRACK_FLAG_LOCK:
                    TRACK_FLAG = '0'
                self.get_logger().info("Marker lost — TRACK_FLAG = 0")
            self.marker_center = None

        self.detected_pub.publish(detected_msg)

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
        qx = (R[2,1] - R[1,2]) / (4*qw)
        qy = (R[0,2] - R[2,0]) / (4*qw)
        qz = (R[1,0] - R[0,1]) / (4*qw)
        return [qx, qy, qz, qw]


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
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

