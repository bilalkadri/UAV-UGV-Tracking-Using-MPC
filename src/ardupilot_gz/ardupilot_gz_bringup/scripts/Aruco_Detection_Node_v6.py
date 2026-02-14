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

        #for transformation
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
        print(ids)

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
            # In ROS, frame_id refers to physical space (meters). 
            # Pixel values should be published in a custom RegionOfInterest or Point2D message,
            #  or just ignored if you only need the 3D pose. 
            # Visualizing this in Foxglove will create a point 320 meters away from your drone.
            # therefor point_msg.header.frame_id should not be set 
            # point_msg.header.frame_id = "pitch_link" #this is the last link in the tf tree, the camera is connected to this link 
            point_msg.point.x = cx
            point_msg.point.y = cy
            point_msg.point.z = 0.0
            self.center_pub.publish(point_msg)

            # Estimate pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]



            # Based on the model.sdf file provided, there is a slight misunderstanding 
            # in your setup: the camera is not 
            # directly attached to the roll_link. Instead, it is attached to the pitch_link.
            # The gimbal follows a standard serial chain where each link is a child of the previous one:
            # gimbal_link: The base plate attached to the UAV.
            # yaw_link: Attached to the base plate via the yaw_joint.
            # roll_link: Attached to the yaw_link via the roll_joint.
            # pitch_link: Attached to the roll_link via the pitch_joint.
            # Frame ID: Th
            
            #Static mapping from "camera_optical_frame"  to "pitch_link" has been defined in the
            #launch file 
            # ros2 launch ardupilot_gz_bringup complete_control_system_v5.launch.py 
            # Define the static transform from pitch_link to camera_optical_frame
            # # Args: x y z yaw pitch roll parent_frame child_frame
            # camera_optical_tf = Node(
            #     package='tf2_ros',
            #     executable='static_transform_publisher',
            #     name='camera_base_to_optical',
            #     arguments=['0', '0', '0', '-1.5708', '0', '-1.5708', 'pitch_link', 'camera_optical_frame'],
            #     parameters=[{'use_sim_time': use_sim_time}]
            # )

            # 1. Create Pose in the Camera Optical Frame
            # convert to PoseStamped
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_optical_frame"

                
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

            # 2. Transform from roll_link to base_link
            try:
                # Get the latest transform in the tree
                # transform = self.tf_buffer.lookup_transform(
                #     'base_link', 
                #     'camera_optical_frame', 
                #     rclpy.time.Time())
                
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 
                    'camera_optical_frame', 
                    rclpy.time.Time(),  
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



            # self.pose_pub.publish(pose_msg)

            detected_msg.data = True

            # ===== Added print statements =====
            # print("===== ArUco Marker Detected =====")
            # print(f"Marker Center (pixels): x={cx:.2f}, y={cy:.2f}")
            # print(f"Marker Position (m): x={tvec[0]:.3f}, y={tvec[1]:.3f}, z={tvec[2]:.3f}")
            # print(f"Marker Orientation (quaternion): x={quat[0]:.3f}, y={quat[1]:.3f}, z={quat[2]:.3f}, w={quat[3]:.3f}")
            # print(f"Detection Status: {detected_msg.data}")
            # print("=================================")

        else:
            if self.tag_detected:
                self.tag_detected = False
                with TRACK_FLAG_LOCK:
                    TRACK_FLAG = '0'
                self.get_logger().info("Marker lost — TRACK_FLAG = 0")
            self.marker_center = None

            
    
            # Publish a "null" pose with zeros or NaN
            null_pose = PoseStamped()
            null_pose.header.stamp = self.get_clock().now().to_msg()
            null_pose.header.frame_id = "base_link"
            null_pose.pose.position.x = float('nan')
            null_pose.pose.position.y = float('nan')
            null_pose.pose.position.z = float('nan')
            self.pose_pub.publish(null_pose)

            # print lost status
            # print("===== ArUco Marker Lost =====")
            # print("Detection Status: False")
            # print("==============================")

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

