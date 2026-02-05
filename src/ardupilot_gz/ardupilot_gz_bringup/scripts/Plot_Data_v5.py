#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped

import threading
import time
import numpy as np


class PlotterNode(Node):

    def __init__(self):
        super().__init__('plotter_node')

        # Storage
        self.odom_hist = []          # [(x, y)]
        self.odom_vel_hist = []      # [(vx, vy, vz)]
        self.odom_ang_hist = []      # [(wx, wy, wz)]
        self.rel_pose_hist = []      # [(x, y)]
        self.vel_est_hist = []       # [(vx, vy)]
        self.mpc_cmd_hist = []       # [(vx, vy)]

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

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        self.odom_vel_hist.append((vx, vy, vz))

        wx = msg.twist.twist.angular.x
        wy = msg.twist.twist.angular.y
        wz = msg.twist.twist.angular.z
        self.odom_ang_hist.append((wx, wy, wz))

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

    # ============= SAVE DATA FUNCTION (INSERTED HERE) =============

    def save_data(self):
        self.get_logger().info("Saving plot data to disk...")

        np.savetxt("odom_path.csv", np.array(self.odom_hist), delimiter=",",
                   header="x,y", comments='')

        np.savetxt("odom_linear_vel.csv", np.array(self.odom_vel_hist), delimiter=",",
                   header="vx,vy,vz", comments='')

        np.savetxt("odom_angular_vel.csv", np.array(self.odom_ang_hist), delimiter=",",
                   header="wx,wy,wz", comments='')

        np.savetxt("ekf_relative_pose.csv", np.array(self.rel_pose_hist), delimiter=",",
                   header="x,y", comments='')

        np.savetxt("vel_estimated.csv", np.array(self.vel_est_hist), delimiter=",",
                   header="vx,vy", comments='')

        np.savetxt("mpc_cmd_vel.csv", np.array(self.mpc_cmd_hist), delimiter=",",
                   header="vx_cmd,vy_cmd", comments='')

        self.get_logger().info("✔ All plot data saved!")

    # ================= PLOTTING LOOP ==================

    def live_plot_loop(self):
        plt.ion()

        # Create windows
        fig1, ax1 = plt.subplots(num="Window 1 - Odometry Path")
        fig2, ax2 = plt.subplots(num="Window 2 - EKF Relative Pose")
        fig3, ax3 = plt.subplots(num="Window 3 - Velocities (Estimated + MPC)")
        fig4, ax4 = plt.subplots(num="Window 4 - Odometry Linear Velocities")
        fig5, ax5 = plt.subplots(num="Window 5 - Odometry Angular Velocities")

        while self.run_plot:

            # Window 1
            ax1.clear()
            if len(self.odom_hist) > 0:
                xs = [p[0] for p in self.odom_hist]
                ys = [p[1] for p in self.odom_hist]
                ax1.plot(xs, ys, 'b-', label="Odometry Path")
            ax1.set_title("UAV Odometry Path")
            ax1.set_xlabel("X (m)")
            ax1.set_ylabel("Y (m)")
            ax1.grid(True)
            ax1.legend()
            fig1.canvas.draw()
            fig1.canvas.flush_events()

            # Window 2
            ax2.clear()
            if len(self.rel_pose_hist) > 0:
                xs = [p[0] for p in self.rel_pose_hist]
                ys = [p[1] for p in self.rel_pose_hist]
                ax2.plot(xs, ys, 'g--', label="Relative Pose (EKF)")
            ax2.set_title("EKF Relative Pose")
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.grid(True)
            ax2.legend()
            fig2.canvas.draw()
            fig2.canvas.flush_events()

            # Window 3
            ax3.clear()
            if len(self.vel_est_hist) > 0:
                vx = [v[0] for v in self.vel_est_hist]
                vy = [v[1] for v in self.vel_est_hist]
                ax3.plot(vx, label="Estimated vx")
                ax3.plot(vy, label="Estimated vy")

            if len(self.mpc_cmd_hist) > 0:
                vx_cmd = [v[0] for v in self.mpc_cmd_hist]
                vy_cmd = [v[1] for v in self.mpc_cmd_hist]
                ax3.plot(vx_cmd, '--', label="MPC vx_cmd")
                ax3.plot(vy_cmd, '--', label="MPC vy_cmd")

            ax3.set_title("Velocities (Estimated & MPC Command)")
            ax3.set_xlabel("Time step")
            ax3.set_ylabel("Velocity (m/s)")
            ax3.grid(True)
            ax3.legend()
            fig3.canvas.draw()
            fig3.canvas.flush_events()

            # Window 4
            ax4.clear()
            if len(self.odom_vel_hist) > 0:
                vx = [v[0] for v in self.odom_vel_hist]
                vy = [v[1] for v in self.odom_vel_hist]
                vz = [v[2] for v in self.odom_vel_hist]
                ax4.plot(vx, label="Odometry vx")
                ax4.plot(vy, label="Odometry vy")
                ax4.plot(vz, label="Odometry vz")

            ax4.set_title("Odometry Linear Velocities")
            ax4.set_xlabel("Time step")
            ax4.set_ylabel("Velocity (m/s)")
            ax4.grid(True)
            ax4.legend()
            fig4.canvas.draw()
            fig4.canvas.flush_events()

            # Window 5
            ax5.clear()
            if len(self.odom_ang_hist) > 0:
                wx = [v[0] for v in self.odom_ang_hist]
                wy = [v[1] for v in self.odom_ang_hist]
                wz = [v[2] for v in self.odom_ang_hist]
                ax5.plot(wx, label="Angular wx")
                ax5.plot(wy, label="Angular wy")
                ax5.plot(wz, label="Angular wz")

            ax5.set_title("Odometry Angular Velocities")
            ax5.set_xlabel("Time step")
            ax5.set_ylabel("Angular Velocity (rad/s)")
            ax5.grid(True)
            ax5.legend()
            fig5.canvas.draw()
            fig5.canvas.flush_events()

            time.sleep(0.05)

    # ================= CLEANUP ==================

    def destroy_node(self):
        # Stop plotting
        self.run_plot = False
        time.sleep(0.2)

        # SAVE DATA BEFORE EXIT
        self.save_data()

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

