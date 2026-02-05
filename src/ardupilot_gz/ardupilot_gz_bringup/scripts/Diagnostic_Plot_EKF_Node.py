#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import csv
import threading


class EKFMonitor(Node):
    def __init__(self):
        super().__init__('ekf_monitor_node')

        # Parameters
        self.buffer_size = 200
        self.mahalanobis_threshold = 11.0

        # Data buffers
        self.md_data = deque(maxlen=self.buffer_size)
        self.update_data = deque(maxlen=self.buffer_size)
        self.time_data = deque(maxlen=self.buffer_size)
        self.counter = 0

        # Subscribers
        self.create_subscription(Float32, '/ekf/mahalanobis_distance', self.md_cb, 10)
        self.create_subscription(Bool, '/ekf/update_applied', self.update_cb, 10)

        # Setup matplotlib
        plt.style.use('seaborn-darkgrid')
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()

        self.line_md, = self.ax1.plot([], [], 'b-', label='Mahalanobis Distance')
        self.line_update, = self.ax2.plot([], [], 'r-', label='Update Applied')
        self.line_thresh, = self.ax1.plot([], [], 'g--', label='Threshold')

        self.ax1.set_xlabel('Time Step')
        self.ax1.set_ylabel('Mahalanobis Distance', color='b')
        self.ax2.set_ylabel('Update Applied', color='r')

        self.ax1.set_ylim(0, 50)
        self.ax2.set_ylim(-0.1, 1.1)

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

        # Prepare CSV log files
        self.init_csv_files()

    def init_csv_files(self):
        """Create CSV headers."""
        with open("md_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_step", "mahalanobis_distance"])

        with open("update_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_step", "update_flag"])

        with open("combined_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_step", "mahalanobis_distance", "update_flag"])

    def md_cb(self, msg):
        self.md_data.append(msg.data)
        self.time_data.append(self.counter)

        # Log MD to separate file
        with open("md_data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.counter, msg.data])

        self.counter += 1

    def update_cb(self, msg):
        flag = int(msg.data)
        self.update_data.append(flag)

        # Log Update flag to separate file
        with open("update_data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.counter, flag])

        # Combined log only when MD and update both exist
        if len(self.time_data) > 0:
            with open("combined_data.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.counter, 
                                 self.md_data[-1] if len(self.md_data) > 0 else None,
                                 flag])

    def update_plot(self, frame):
        if len(self.time_data) == 0:
            return

        # Pad update_data if shorter
        while len(self.update_data) < len(self.time_data):
            self.update_data.append(0)

        self.line_md.set_data(self.time_data, self.md_data)
        self.line_update.set_data(self.time_data, self.update_data)

        # Threshold line
        thresh_line = [self.mahalanobis_threshold] * len(self.time_data)
        self.line_thresh.set_data(self.time_data, thresh_line)

        self.ax1.set_xlim(0, self.time_data[-1] + 1)
        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper right')

    def run(self):
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = EKFMonitor()

    # Spin in separate thread so callbacks run
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run()  # Show the plot
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

