#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped

import threading
import time


class PlotterNode(Node):

    def __init__(self):
        super().__init__('plotter_node')

        # Storage
        self.odom_hist = []
        self.rel_pose_hist = []
        self.vel_est_hist = []
        self.mpc_cmd_hist = []

        # Subscribers
        self.create_subscription(Odometry, '/odometry', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/relative_pose_ekf', self.ekf_callback, 10)
        self.create_subscription(TwistStamped, '/uav/vel_estimated', self.vel_callback, 10)
        self.create_subscription(TwistStamped, '/mpc/cmd_vel', self.mpc_cmd_callback, 10)

        # Plotting thread
        self.run_plot = True
        self.plot_thread = threading.Thread(target=self.live_plot_loop, daemon=True)
        self.plot_thread.start()

        self.get_logger().info("Plotter Node Started ✔")

    # ================= CALLBACKS ==================

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.odom_hist.append((x, y))

    def ekf_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.rel_pose_hist.append((x, y))

    def vel_callback(self, msg):
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        self.vel_est_hist.append((vx, vy))

    def mpc_cmd_callback(self, msg):
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        self.mpc_cmd_hist.append((vx, vy))

    # ================= PLOTTING LOOP ==================

    def live_plot_loop(self):
        plt.ion()
        fig, ax = plt.subplots()

        while self.run_plot:
            ax.clear()

            # 1 — ODOMETRY (UAV actual path)
            if len(self.odom_hist) > 0:
                xs = [p[0] for p in self.odom_hist]
                ys = [p[1] for p in self.odom_hist]
                ax.plot(xs, ys, 'b-', label="Odometry (UAV path)")

            # 2 — EKF relative pose
            if len(self.rel_pose_hist) > 0:
                xs = [p[0] for p in self.rel_pose_hist]
                ys = [p[1] for p in self.rel_pose_hist]
                ax.plot(xs, ys, 'g--', label="EKF Relative Pose")

            # 3 — MPC command velocities (arrows)
            if len(self.mpc_cmd_hist) > 0 and len(self.odom_hist) > 0:
                x, y = self.odom_hist[-1]
                vx, vy = self.mpc_cmd_hist[-1]
                ax.arrow(x, y, vx * 0.5, vy * 0.5,
                         head_width=0.1, color='r', label="MPC Command")

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("UAV Tracking and EKF Visualization")
            ax.grid(True)
            ax.legend()

            plt.pause(0.05)
            time.sleep(0.05)

    # ================= CLEANUP ==================

    def destroy_node(self):
        self.run_plot = False
        time.sleep(0.2)
        plt.close('all')
        super().destroy_node()


def main():
    rclpy.init()
    node = PlotterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

