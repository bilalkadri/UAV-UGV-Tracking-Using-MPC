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

        # Thread-safety lock
        self.lock = threading.Lock()

        # Storage
        self.odom_hist = []                  # [(x, y)]
        self.odom_vel_hist = []              # [(vx, vy, vz)]
        self.odom_ang_hist = []              # [(wx, wy, wz)]
        self.rel_pose_hist = []              # [(x, y)]
        self.rel_pose_orient_hist = []       # [(qx, qy, qz, qw)]
        self.vel_est_hist = []               # [(vx, vy)]
        self.mpc_cmd_hist = []               # [(vx, vy)]

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
        with self.lock:
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
        with self.lock:
            # store position
            x = msg.pose.position.x
            y = msg.pose.position.y
            self.rel_pose_hist.append((x, y))

            # ==== NEW: store orientation ====
            qx = msg.pose.orientation.x
            qy = msg.pose.orientation.y
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w
            self.rel_pose_orient_hist.append((qx, qy, qz, qw))

    def vel_callback(self, msg):
        with self.lock:
            vx = msg.twist.linear.x
            vy = msg.twist.linear.y
            self.vel_est_hist.append((vx, vy))

    def mpc_cmd_callback(self, msg):
        with self.lock:
            vx = msg.twist.linear.x
            vy = msg.twist.linear.y
            self.mpc_cmd_hist.append((vx, vy))

    # ============= SAVE DATA FUNCTION =============

    def save_data(self):
        self.get_logger().info("Saving plot data to disk...")

        with self.lock:
            odom = list(self.odom_hist)
            odom_vel = list(self.odom_vel_hist)
            odom_ang = list(self.odom_ang_hist)
            rel_pose = list(self.rel_pose_hist)
            rel_orient = list(self.rel_pose_orient_hist)
            vel_est = list(self.vel_est_hist)
            mpc_cmd = list(self.mpc_cmd_hist)

        try:
            if len(odom) > 0:
                np.savetxt("odom_path.csv", np.array(odom), delimiter=",", header="x,y", comments='')
            else:
                open("odom_path.csv", "w").write("x,y\n")

            if len(odom_vel) > 0:
                np.savetxt("odom_linear_vel.csv", np.array(odom_vel), delimiter=",", header="vx,vy,vz", comments='')
            else:
                open("odom_linear_vel.csv", "w").write("vx,vy,vz\n")

            if len(odom_ang) > 0:
                np.savetxt("odom_angular_vel.csv", np.array(odom_ang), delimiter=",", header="wx,wy,wz", comments='')
            else:
                open("odom_angular_vel.csv", "w").write("wx,wy,wz\n")

            if len(rel_pose) > 0:
                np.savetxt("ekf_relative_pose.csv", np.array(rel_pose), delimiter=",", header="x,y", comments='')
            else:
                open("ekf_relative_pose.csv", "w").write("x,y\n")

            # ==== NEW: Save orientation ====
            if len(rel_orient) > 0:
                np.savetxt("ekf_relative_pose_orientation.csv",
                           np.array(rel_orient),
                           delimiter=",",
                           header="qx,qy,qz,qw",
                           comments='')
            else:
                open("ekf_relative_pose_orientation.csv", "w").write("qx,qy,qz,qw\n")

            if len(vel_est) > 0:
                np.savetxt("vel_estimated.csv", np.array(vel_est), delimiter=",", header="vx,vy", comments='')
            else:
                open("vel_estimated.csv", "w").write("vx,vy\n")

            if len(mpc_cmd) > 0:
                np.savetxt("mpc_cmd_vel.csv", np.array(mpc_cmd), delimiter=",", header="vx_cmd,vy_cmd", comments='')
            else:
                open("mpc_cmd_vel.csv", "w").write("vx_cmd,vy_cmd\n")

            self.get_logger().info("✔ All plot data saved!")

        except Exception as e:
            self.get_logger().error(f"Failed to save data: {e}")

    # ================= PLOTTING LOOP ==================

    def live_plot_loop(self):
        plt.ion()

        fig1, ax1 = plt.subplots(num="Window 1 - Odometry Path")
        fig2, ax2 = plt.subplots(num="Window 2 - EKF Relative Pose")
        fig3, ax3 = plt.subplots(num="Window 3 - Velocities (Estimated + MPC)")
        fig4, ax4 = plt.subplots(num="Window 4 - Odometry Linear Velocities")
        fig5, ax5 = plt.subplots(num="Window 5 - Odometry Angular Velocities")
        fig6, ax6 = plt.subplots(num="Window 6 - EKF Orientation (Quaternion)")

        while self.run_plot:

            with self.lock:
                local_odom = list(self.odom_hist)
                local_rel = list(self.rel_pose_hist)
                local_rel_orient = list(self.rel_pose_orient_hist)
                local_vel_est = list(self.vel_est_hist)
                local_mpc = list(self.mpc_cmd_hist)
                local_odom_vel = list(self.odom_vel_hist)
                local_odom_ang = list(self.odom_ang_hist)

            # Window 1
            ax1.clear()
            if len(local_odom) > 0:
                xs = [p[0] for p in local_odom]
                ys = [p[1] for p in local_odom]
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
            if len(local_rel) > 0:
                xs = [p[0] for p in local_rel]
                ys = [p[1] for p in local_rel]
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
            if len(local_vel_est) > 0:
                ax3.plot([v[0] for v in local_vel_est], label="Estimated vx")
                ax3.plot([v[1] for v in local_vel_est], label="Estimated vy")

            if len(local_mpc) > 0:
                ax3.plot([v[0] for v in local_mpc], '--', label="MPC vx_cmd")
                ax3.plot([v[1] for v in local_mpc], '--', label="MPC vy_cmd")

            ax3.set_title("Velocities (Estimated & MPC Command)")
            ax3.set_xlabel("Time step")
            ax3.set_ylabel("Velocity (m/s)")
            ax3.grid(True)
            ax3.legend()
            fig3.canvas.draw()
            fig3.canvas.flush_events()

            # Window 4
            ax4.clear()
            if len(local_odom_vel) > 0:
                ax4.plot([v[0] for v in local_odom_vel], label="Odometry vx")
                ax4.plot([v[1] for v in local_odom_vel], label="Odometry vy")
                ax4.plot([v[2] for v in local_odom_vel], label="Odometry vz")

            ax4.set_title("Odometry Linear Velocities")
            ax4.set_xlabel("Time step")
            ax4.set_ylabel("Velocity (m/s)")
            ax4.grid(True)
            ax4.legend()
            fig4.canvas.draw()
            fig4.canvas.flush_events()

            # Window 5
            ax5.clear()
            if len(local_odom_ang) > 0:
                ax5.plot([v[0] for v in local_odom_ang], label="Angular wx")
                ax5.plot([v[1] for v in local_odom_ang], label="Angular wy")
                ax5.plot([v[2] for v in local_odom_ang], label="Angular wz")

            ax5.set_title("Odometry Angular Velocities")
            ax5.set_xlabel("Time step")
            ax5.set_ylabel("Angular Velocity (rad/s)")
            ax5.grid(True)
            ax5.legend()
            fig5.canvas.draw()
            fig5.canvas.flush_events()

            # ===== NEW WINDOW 6: Quaternion components =====
            ax6.clear()
            if len(local_rel_orient) > 0:
                qx = [q[0] for q in local_rel_orient]
                qy = [q[1] for q in local_rel_orient]
                qz = [q[2] for q in local_rel_orient]
                qw = [q[3] for q in local_rel_orient]

                ax6.plot(qx, label="qx")
                ax6.plot(qy, label="qy")
                ax6.plot(qz, label="qz")
                ax6.plot(qw, label="qw")

            ax6.set_title("EKF Orientation (Quaternion)")
            ax6.set_xlabel("Time step")
            ax6.set_ylabel("Quaternion Value")
            ax6.grid(True)
            ax6.legend()
            fig6.canvas.draw()
            fig6.canvas.flush_events()

            time.sleep(0.05)

    # ================= CLEANUP ==================

    def destroy_node(self):
        self.run_plot = False
        if self.plot_thread.is_alive():
            self.plot_thread.join(timeout=1.0)
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
