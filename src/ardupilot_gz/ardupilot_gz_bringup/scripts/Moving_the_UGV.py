#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
import math

class JackalMover(Node):
    def __init__(self, mode="LINEAR"):
        super().__init__("jackal_mover")
        
        # Mode Selection: "LINEAR" or "RECTANGLE" or "ROTATE"
        self.mode = mode

        # Publisher & Subscriber
        self.pub = self.create_publisher(TwistStamped, "/jackal/jackal_velocity_controller/cmd_vel", 10)
        self.sub = self.create_subscription(Bool, "/aruco/detected", self.aruco_detected_cb, 10)

        # Common State Variables
        self.started_once = False
        self.state = "IDLE"
        self.elapsed_time = 0.0
        self.dt = 0.01    
        
        # --- Linear Mode Params ---
        self.dist = 10.0
        self.v_fwd, self.v_bwd = 0.3, -0.3
        self.pause_dur = 1.0
        self.t_fwd = self.dist / abs(self.v_fwd)
        self.t_bwd = self.dist / abs(self.v_bwd)

        # --- Rectangle Mode Params ---
        self.Lx, self.Ly = 10.0, 10.0
        self.radius = 0.6
        self.v_rec = 0.25
        self.w_rec = 0.45
        self.t_str_x = (self.Lx - 2 * self.radius) / self.v_rec
        self.t_str_y = (self.Ly - 2 * self.radius) / self.v_rec
        self.t_turn = (math.pi/2 * self.radius) / self.v_rec
        self.total_rect_time = 2 * (self.t_str_x + self.t_str_y) + 4 * self.t_turn

        # Timer
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info(f"Jackal Mover initialized in {self.mode} mode. Waiting for ArUco...")

    def aruco_detected_cb(self, msg: Bool):
        if msg.data and not self.started_once:
            self.started_once = True
            self.state = "START"
            self.get_logger().info(f"ArUco Detected! Starting {self.mode} motion.")

    def control_loop(self):
        if self.state == "IDLE":
            return

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        if self.mode == "LINEAR":
            self.execute_linear_logic(msg)
        elif self.mode == "RECTANGLE":
            self.execute_rectangle_logic(msg)
        elif self.mode == "ROTATE":
            self.execute_rotate_logic(msg)


        self.pub.publish(msg)
        self.elapsed_time += self.dt

    def execute_rotate_logic(self,msg):
        """Logic for Rotation"""
        msg.twist.linear.z=1

    def execute_linear_logic(self, msg):
        """Logic for Forward 5m / Backward 5m"""
        if self.state == "START": self.state = "FORWARD"
        
        if self.state == "FORWARD":
            if self.elapsed_time < self.t_fwd:
                msg.twist.linear.x = self.v_fwd
            else:
                self.state = "PAUSE_FWD"; self.elapsed_time = 0.0
        
        elif self.state == "PAUSE_FWD":
            if self.elapsed_time >= self.pause_dur:
                self.state = "BACKWARD"; self.elapsed_time = 0.0
        
        elif self.state == "BACKWARD":
            if self.elapsed_time < self.t_bwd:
                msg.twist.linear.x = self.v_bwd
            else:
                self.state = "PAUSE_BWD"; self.elapsed_time = 0.0
        
        elif self.state == "PAUSE_BWD":
            if self.elapsed_time >= self.pause_dur:
                self.state = "FORWARD"; self.elapsed_time = 0.0

    def execute_rectangle_logic(self, msg):
        """Logic for Continuous Rectangle Pattern"""
        t = self.elapsed_time % self.total_rect_time
        
        # Segment timing check
        if t < self.t_str_x: # Side 1
            msg.twist.linear.x, msg.twist.angular.z = self.v_rec, 0.0
        elif t < (self.t_str_x + self.t_turn): # Corner 1
            msg.twist.linear.x, msg.twist.angular.z = self.v_rec, self.w_rec
        elif t < (self.t_str_x + self.t_turn + self.t_str_y): # Side 2
            msg.twist.linear.x, msg.twist.angular.z = self.v_rec, 0.0
        elif t < (self.t_str_x + 2*self.t_turn + self.t_str_y): # Corner 2
            msg.twist.linear.x, msg.twist.angular.z = self.v_rec, self.w_rec
        elif t < (2*self.t_str_x + 2*self.t_turn + self.t_str_y): # Side 3
            msg.twist.linear.x, msg.twist.angular.z = self.v_rec, 0.0
        elif t < (2*self.t_str_x + 3*self.t_turn + self.t_str_y): # Corner 3
            msg.twist.linear.x, msg.twist.angular.z = self.v_rec, self.w_rec
        elif t < (2*self.t_str_x + 3*self.t_turn + 2*self.t_str_y): # Side 4
            msg.twist.linear.x, msg.twist.angular.z = self.v_rec, 0.0
        else: # Corner 4
            msg.twist.linear.x, msg.twist.angular.z = self.v_rec, self.w_rec

def main(args=None):
    rclpy.init(args=args)
    
    # CHANGE THIS LINE TO SWITCH MOTIONS:
    # Use "LINEAR" for Forward/Backward or 
    # "RECTANGLE" for the rectangle or
    #  "ROTATE"  for rotating at a place
    node = JackalMover(mode="LINEAR")

     # ==========================
     # 
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()