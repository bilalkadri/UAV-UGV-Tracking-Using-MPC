#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        self.get_logger().info("Aruco Detector Node Started ✅")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None:
            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                self.get_logger().info(f"Detected marker IDs: {ids.flatten().tolist()}")
        
        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

