
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import open3d as o3d
import numpy as np
import threading
import time


# ============================================================
# CHANGE THIS IF YOUR POINTCLOUD2 TOPIC HAS A DIFFERENT NAME
# ============================================================
LIDAR_TOPIC = '/iris/lidar/points'


class LidarOpen3D(Node):

    def __init__(self):

        super().__init__('lidar_open3d_viewer')

        # Latest point cloud
        self.latest_points = None

        # Lock for ROS/Open3D thread communication
        self.lock = threading.Lock()

        # ROS 2 subscription
        self.subscription = self.create_subscription(
            PointCloud2,
            LIDAR_TOPIC,
            self.lidar_callback,
            10
        )

        self.get_logger().info(
            f'Subscribed to {LIDAR_TOPIC}'
        )

    def lidar_callback(self, msg):

        try:

            # Read X, Y, Z from PointCloud2
            points = point_cloud2.read_points(
                msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            )

            # Convert directly to NumPy
            points = np.array(
                list(points),
                dtype=np.float64
            )

            if points.size == 0:
                return

            # Store latest cloud
            with self.lock:
                self.latest_points = points

        except Exception as e:

            self.get_logger().error(
                f'Point cloud conversion error: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = LidarOpen3D()

    # --------------------------------------------------------
    # ROS spinning in a separate thread
    # --------------------------------------------------------

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )

    ros_thread.start()

    # --------------------------------------------------------
    # Open3D setup
    # --------------------------------------------------------

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name='Iris LiDAR - Real-Time 3D Point Cloud',
        width=1280,
        height=720
    )

    # Empty point cloud
    pcd = o3d.geometry.PointCloud()

    vis.add_geometry(
        pcd,
        reset_bounding_box=True
    )

    # Coordinate frame
    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0,
            origin=[0, 0, 0]
        )
    )

    vis.add_geometry(
        coordinate_frame
    )

    # --------------------------------------------------------
    # Rendering options
    # --------------------------------------------------------

    render_option = vis.get_render_option()

    render_option.point_size = 2.0

    # --------------------------------------------------------
    # Main visualization loop
    # --------------------------------------------------------

    first_cloud = True

    try:

        while rclpy.ok():

            # Check if new point cloud exists
            with node.lock:

                if node.latest_points is not None:

                    points = node.latest_points.copy()

                else:

                    points = None

            # ------------------------------------------------
            # Update point cloud
            # ------------------------------------------------

            if points is not None:

                # Update XYZ coordinates
                pcd.points = o3d.utility.Vector3dVector(
                    points
                )

                # ------------------------------------------------
                # Color points according to Z height
                # ------------------------------------------------

                z = points[:, 2]

                z_min = np.min(z)
                z_max = np.max(z)

                if z_max > z_min:

                    normalized_z = (
                        (z - z_min) /
                        (z_max - z_min)
                    )

                else:

                    normalized_z = np.zeros_like(z)

                # Simple RGB coloring based on height
                colors = np.zeros(
                    (len(points), 3),
                    dtype=np.float64
                )

                colors[:, 0] = normalized_z
                colors[:, 1] = 1.0 - normalized_z
                colors[:, 2] = 0.5

                pcd.colors = (
                    o3d.utility.Vector3dVector(colors)
                )

                # Update Open3D
                vis.update_geometry(pcd)

                # Print statistics occasionally
                if first_cloud:

                    print()
                    print('======================================')
                    print('      LiDAR Point Cloud Received')
                    print('======================================')
                    print(
                        f'Number of points : {len(points)}'
                    )
                    print(
                        f'X range          : '
                        f'{np.min(points[:, 0]):.3f} '
                        f'to '
                        f'{np.max(points[:, 0]):.3f} m'
                    )
                    print(
                        f'Y range          : '
                        f'{np.min(points[:, 1]):.3f} '
                        f'to '
                        f'{np.max(points[:, 1]):.3f} m'
                    )
                    print(
                        f'Z range          : '
                        f'{np.min(points[:, 2]):.3f} '
                        f'to '
                        f'{np.max(points[:, 2]):.3f} m'
                    )
                    print('======================================')
                    print()

                    first_cloud = False

            # ------------------------------------------------
            # Process Open3D events
            # ------------------------------------------------

            if not vis.poll_events():
                break

            vis.update_renderer()

            # Small delay to avoid excessive CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:

        print('\nStopping LiDAR viewer...')

    finally:

        vis.destroy_window()

        node.destroy_node()

        rclpy.shutdown()

        ros_thread.join(timeout=1.0)


if __name__ == '__main__':

    main()


