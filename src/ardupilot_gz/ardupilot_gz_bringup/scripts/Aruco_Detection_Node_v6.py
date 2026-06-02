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
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

# ===== Optional Global Flags =====
TRACK_FLAG = '0'
TRACK_FLAG_LOCK = threading.Lock()

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        self.bridge = CvBridge()

        # states
        self.tag_detected = False
        self.marker_center = None
        self.frame_center = None
        self.last_valid_yaw = 0.0
        self.is_yaw_initialized = False  # Caught first-frame offsets safely

        # for transformation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # camera parameters (replace with your calibration values)
        self.marker_size = 0.1  # meters
        self.camera_matrix = np.array([
            [205.4696273803711, 0.0, 320.0],
            [0.0, 205.4696559906006, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        self.dist_coeffs = np.zeros(5, dtype=float)

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
            point_msg.point.x = cx
            point_msg.point.y = cy
            point_msg.point.z = 0.0
            self.center_pub.publish(point_msg)

            # Estimate pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            # 1. Create Pose in the Camera Optical Frame
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_optical_frame"
                
            pose_msg.pose.position.x = float(tvec[0])
            pose_msg.pose.position.y = float(tvec[1])
            pose_msg.pose.position.z = float(tvec[2])

            # Get raw rotation matrix and raw quaternion from OpenCV
            rot_matrix, _ = cv2.Rodrigues(rvec)
            quat_raw = self.rotation_matrix_to_quaternion(rot_matrix)

            # 2. Extract the raw measured Euler angles
            siny_cosp = 2 * (quat_raw[3] * quat_raw[2] + quat_raw[0] * quat_raw[1])
            cosy_cosp = 1 - 2 * (quat_raw[1] * quat_raw[1] + quat_raw[2] * quat_raw[2])
            measured_yaw = np.arctan2(siny_cosp, cosy_cosp)

            # Extract raw roll and pitch so we preserve them
            sinr_cosp = 2 * (quat_raw[3] * quat_raw[0] + quat_raw[1] * quat_raw[2])
            cosr_cosp = 1 - 2 * (quat_raw[0] * quat_raw[0] + quat_raw[1] * quat_raw[1])
            measured_roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            sinp = 2 * (quat_raw[3] * quat_raw[1] - quat_raw[2] * quat_raw[0])
            measured_pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

            # 3. Calculate the true, minimal angular step change between frames
            if not self.is_yaw_initialized:
                # First frame catch: lock the current raw yaw as our starting anchor
                self.last_valid_yaw = measured_yaw
                self.is_yaw_initialized = True
                wrapped_yaw_delta = 0.0
            else:
                # Normal operation: compute relative delta variations safely
                raw_yaw_delta = measured_yaw - self.last_valid_yaw
                wrapped_yaw_delta = np.arctan2(np.sin(raw_yaw_delta), np.cos(raw_yaw_delta))

                # Check if the PnP solver suddenly flipped 180 degrees
                if abs(wrapped_yaw_delta - raw_yaw_delta) > (np.pi / 2.0):
                    self.get_logger().warn("[ANTI-DISCONTINUITY] Blocked a 180-degree ArUco orientation flip.")
            
            # 4. Step your historical baseline forward ONLY by the true, clean delta
            self.last_valid_yaw += wrapped_yaw_delta
            self.last_valid_yaw = np.arctan2(np.sin(self.last_valid_yaw), np.cos(self.last_valid_yaw))

            # 5. Reconstruct the clean quaternion using your latched smooth yaw
            cy = np.cos(self.last_valid_yaw * 0.5)
            sy = np.sin(self.last_valid_yaw * 0.5)
            cp = np.cos(measured_pitch * 0.5)
            sp = np.sin(measured_pitch * 0.5)
            cr = np.cos(measured_roll * 0.5)
            sr = np.sin(measured_roll * 0.5)

            quat = [
                sr * cp * cy - cr * sp * sy,  # x
                cr * sp * cy + sr * cp * sy,  # y
                cr * cp * sy - sr * sp * cy,  # z
                cr * cp * cy + sr * sp * sy   # w
            ]

            # 6. Safely send the fully continuous quaternion out to the EKF pipeline
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            # 2. Transform from camera_optical_frame to base_link
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 
                    'camera_optical_frame', 
                    rclpy.time.Time()
                )

                # Transform the pose
                pose_base_link = do_transform_pose(pose_msg.pose, transform)

                # 3. Create final message to publish
                final_msg = PoseStamped()
                final_msg.header.stamp = self.get_clock().now().to_msg()
                final_msg.header.frame_id = "base_link"
                final_msg.pose = pose_base_link

                self.pose_pub.publish(final_msg)
                
            except Exception as e:
                self.get_logger().error(f"TF Transform failed: {e}")

            detected_msg.data = True

        else:
            # Marker is genuinely lost
            if self.tag_detected:
                self.tag_detected = False
                self.is_yaw_initialized = False  # Allows clean baseline lock on re-detection
                with TRACK_FLAG_LOCK:
                    TRACK_FLAG = '0'
                self.get_logger().info("Marker lost — TRACK_FLAG = 0")
            self.marker_center = None
    
            # Publish a "null" pose with NaNs
            null_pose = PoseStamped()
            null_pose.header.stamp = self.get_clock().now().to_msg()
            null_pose.header.frame_id = "base_link"
            null_pose.pose.position.x = float('nan')
            null_pose.pose.position.y = float('nan')
            null_pose.pose.position.z = float('nan')
            self.pose_pub.publish(null_pose)

        self.detected_pub.publish(detected_msg)

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        """Robust Sheppard's method to prevent division-by-zero near 180-deg configurations."""
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
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