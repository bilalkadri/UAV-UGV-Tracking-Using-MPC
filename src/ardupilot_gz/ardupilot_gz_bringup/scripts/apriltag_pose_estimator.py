#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import apriltag
import numpy as np
import transforms3d as tf_transformations

from tf2_ros import TransformBroadcaster


class AprilTagPoseEstimator(Node):
    def __init__(self):
        super().__init__('apriltag_pose_estimator')

        # Parameters
        self.declare_parameter('camera_topic', '/uav/camera/image_raw')
        self.declare_parameter('tag_size', 0.15)  # meters
        self.declare_parameter('camera_matrix', [640, 0, 320, 0, 640, 240, 0, 0, 1])  # fx, fy, cx, cy
        self.declare_parameter('camera_frame', 'uav_camera_frame')
        self.declare_parameter('ugv_frame', 'ugv_frame')

        # Read params
        self.camera_topic = self.get_parameter('camera_topic').value
        self.tag_size = self.get_parameter('tag_size').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.ugv_frame = self.get_parameter('ugv_frame').value

        cam_matrix = np.array(self.get_parameter('camera_matrix').value).reshape(3, 3)
        self.camera_matrix = cam_matrix
        self.dist_coeffs = np.zeros((4, 1))  # assume no distortion

        # Initialize detector
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/ugv/relative_pose', 10)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)

        self.get_logger().info("AprilTag Pose Estimator Node started.")

    def image_callback(self, msg):
        # Convert image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        detections = self.detector.detect(cv_image)

        if len(detections) == 0:
            self.get_logger().warn_throttle(5.0, "No AprilTags detected.")
            return

        # Use first detected tag
        detection = detections[0]

        # Estimate pose
        obj_points = np.array([
            [-self.tag_size / 2, -self.tag_size / 2, 0],
            [ self.tag_size / 2, -self.tag_size / 2, 0],
            [ self.tag_size / 2,  self.tag_size / 2, 0],
            [-self.tag_size / 2,  self.tag_size / 2, 0],
        ])

        img_points = np.array([
            detection.corners[0],
            detection.corners[1],
            detection.corners[2],
            detection.corners[3],
        ])

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points,
                                           self.camera_matrix, self.dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            return

        # Convert rotation vector to quaternion
        rot_mat, _ = cv2.Rodrigues(rvec)
        quat = tf_transformations.quaternion_from_matrix(
            np.vstack((np.hstack((rot_mat, [[0], [0], [0]])), [0, 0, 0, 1]))
        )

        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = self.camera_frame
        pose_msg.pose.position.x = tvec[0][0]
        pose_msg.pose.position.y = tvec[1][0]
        pose_msg.pose.position.z = tvec[2][0]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

        # Publish TF transform (camera → UGV)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.ugv_frame
        t.transform.translation.x = tvec[0][0]
        t.transform.translation.y = tvec[1][0]
        t.transform.translation.z = tvec[2][0]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


