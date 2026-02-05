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
        
        # Subscribe to camera topic
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)
        
        # ✅ Use the Original ArUco Dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        self.get_logger().info("Aruco Detector Node Started ✅ (DICT_ARUCO_ORIGINAL)")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Debug: Confirm image shape
        print(f"Frame shape: {frame.shape}")
        
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            # Draw bounding boxes and IDs
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i, marker_id in enumerate(ids.flatten()):
                # Compute the center of the marker
                c = corners[i][0]
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())
                
                # Draw a small circle at the center
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                
                # Label the marker ID above the center
                cv2.putText(frame, f"ID: {marker_id}", (center_x - 20, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Log detections in terminal
            self.get_logger().info(f"Detected marker IDs: {ids.flatten().tolist()}")
        else:
            # Optional: You can comment this line if it floods logs
            self.get_logger().info("No markers detected in frame.")
        
        # Show the image with overlays
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

