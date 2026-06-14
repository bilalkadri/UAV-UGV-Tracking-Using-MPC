#!/usr/bin/env python3 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import threading
import time
import math
from pymavlink import mavutil
import matplotlib.pyplot as plt  # For plotting trajectories / live errors

#from Traj_Pred_EKF_Pub_v2 import RelativePoseEKF
#from UAV_Velocity_Estimator import UAVVelocityEstimator
#from NMPC_Controller_v4 import SimpleMPCNode

#from Plot_Data_v6 import PlotterNode
from rclpy.executors import MultiThreadedExecutor

# ============================================================
# ================= GLOBAL TRACKING FLAG =====================
# ============================================================
# TRACK_FLAG = '0'
# TRACK_FLAG_LOCK = threading.Lock()  # for safer reads/writes across threads

# ============================================================
# ======== GLOBAL TRAJECTORY LISTS FOR UAV & UGV =============
# ============================================================
uav_traj = []  # stores (x, y) positions of UAV in circle frame
ugv_traj = []  # stores (x, y) positions of UGV

# ============================================================
# ================== LIVE ERROR STORAGE =======================
# ============================================================
err_x_list = []
err_y_list = []
ERR_LOCK = threading.Lock()


# ============================================================
# ============ CONTROLLER VELOCITY SUBSCRIBER NODE ==================
# ============================================================
class ControllerVelocitySubscriber(Node):
    def __init__(self, master, start_time):
        """
        Subscribes to /controller/cmd_vel and sends controller velocity commands
        directly to the drone via MAVLink.
        """
        super().__init__('controller_velocity_subscriber')

        self.master = master
        self.start_time = start_time

        # Subscribe to MPC output topic
        self.subscription = self.create_subscription(
            TwistStamped,
            '/controller/cmd_vel',
            self.cmd_callback,
            20
        )

        self.get_logger().info("ControllerVelocitySubscriber Node Started — listening on /controller/cmd_vel")

    # ----------------------------------------------------------
    #  When Controller publishes velocities, send them directly to UAV
    # ----------------------------------------------------------
    def cmd_callback(self, msg):
        
        scale = 0.5  # Adjust this scale factor as needed to tune the responsiveness of the drone to the controller's output
        #scale = 0.09

        vx_flu = msg.twist.linear.x*scale
        vy_flu = msg.twist.linear.y*scale
        vz_flu = msg.twist.linear.z*scale
        # vz_flu = 0
        yaw_rate_flu = msg.twist.angular.z

        # The /controller/cmd_vel produced by the MPC Node is in the FLU frame,
        # we have to convert FLU to MAV_FRAME_BODY_NED (This is actually Forward-Right-Down) 
        
    
        # There are many frames that can be used in mavlink, following are two frames
        # mavutil.mavlink.MAV_FRAME_LOCAL_NED
        # mavutil.mavlink.MAV_FRAME_BODY_NED (This actually uses Forward-Right-Down).

        # "LOCAL": Origin of local frame is fixed relative to earth. Unless otherwise specified this origin is the origin of the vehicle position-estimator ("EKF").
        # "BODY": Origin of local frame travels with the vehicle. NOTE, "BODY" does NOT indicate alignment of frame axis with vehicle attitude.

        # Strategy 1: The "Body Frame" Approach (Recommended)
        # Since you are already converting your MPC output (ENU) into the UAV's base_link (FLU) inside your publish_cmd function, 
        # you should tell Mavlink to interpret the velocities as Body-Fixed.
        # Change the constant to: MAV_FRAME_BODY_NED
       

        # Adjustment: Since I am receiving  /controller/cmd_vel as FLU (Forward-Left-Up), I just need to 
        # flip Y and Z:
        # vx_mav = vx_flu (Forward)
        # vy_mav = -vy_flu (Left $\rightarrow$ Right)
        # vz_mav = -vz_flu (Up $\rightarrow$ Down)

        vx_BODY_NED_MAVLINK=vx_flu
        vy_BODY_NED_MAVLINK=-vy_flu
        vz_BODY_NED_MAVLINK=-vz_flu
        yaw_BODY_NED_MAVLINK=yaw_rate_flu
        

        try:
            send_velocity(self.master, vx_BODY_NED_MAVLINK, vy_BODY_NED_MAVLINK, vz_BODY_NED_MAVLINK, yaw_BODY_NED_MAVLINK, self.start_time)
        except Exception as e:
            self.get_logger().error(f"Failed sending controller velocity: {e}")


# ============================================================
# ================= MAVLINK FUNCTIONS ========================
# ============================================================
TARGET_ALT_M = 2.0
CONN_STR = 'udp:127.0.0.1:14551'
COMMAND_RATE_HZ = 5

def wait_heartbeat(master):
    print("Waiting heartbeat...")
    master.wait_heartbeat()
    print("Heartbeat OK")

def set_mode(master, mode_name):
    modes = master.mode_mapping()
    # defensive check
    if mode_name not in modes:
        print(f"Mode {mode_name} not available on vehicle.")
        return
    mode_id = modes[mode_name]
    for _ in range(10):
        master.mav.set_mode_send(
            master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb and mavutil.mode_string_v10(hb) == mode_name:
            print(f"Mode set to {mode_name}")
            return

def arm_and_wait(master, arm=True):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1 if arm else 0, 0, 0, 0, 0, 0, 0
    )
    while True:
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb:
            armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            if armed == arm:
                return

def request_streams(master):
    # request global position at ~10Hz
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        100000, 0, 0, 0, 0, 0
    )

def read_relative_alt(master):
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    return msg.relative_alt / 1000.0 if msg else None

def takeoff(master, alt_m):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt_m, 0
    )

def send_velocity(master, vx, vy, vz, yaw_rate, start_time):
    """Send velocity (vx,vy,vz) + angular velocity (yaw_rate) via SET_POSITION_TARGET_LOCAL_NED."""
    try:
        time_boot_ms = int((time.time() - start_time) * 1000)
        
        # print(f"[SEND] Sending to drone: vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}, yaw_rate={yaw_rate:.3f}")
        

        master.mav.set_position_target_local_ned_send(
            time_boot_ms,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000011,   # <-- enable yaw rate + velocities
            0, 0, 0,              # position ignored
            vx, vy, vz,           # linear velocities
            0, 0, 0,              # accel ignored
            0,                    # yaw (ignored)
            yaw_rate              # <-- angular velocity (rad/s)
        )
    except Exception as e:
        print(f"send_velocity error: {e}")

def land(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    )

def takeoff_and_wait(master, alt):
    set_mode(master, "GUIDED")
    arm_and_wait(master, True)
    takeoff(master, alt)
    while True:
        alt_now = read_relative_alt(master)
        if alt_now and alt_now >= alt * 0.95:
            break
        time.sleep(0.5)

def wait_for_landing(master):
    while True:
        alt = read_relative_alt(master)
        if alt and alt <= 0.2:
            break
        time.sleep(0.5)
    arm_and_wait(master, False)




# ============================================================
# ========================= MAIN =============================
# ============================================================
def main():
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)
    request_streams(master)

    rclpy.init()
    
    # create an executor and add nodes (use multiple threads to allow callbacks in parallel)
    executor = MultiThreadedExecutor()

    # spin executor in a background thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()


    # optional events for other threads (avoid NameError on cleanup)
    plot_stop_event = threading.Event()
    stop_event_rect = threading.Event()

    # UAV takeoff and hover
    takeoff_and_wait(master, TARGET_ALT_M)

    #node.get_logger().info("Takeoff complete. UAV hovering at target altitude.")
    start_time = time.time()
    dt = 1.0 / COMMAND_RATE_HZ

    # --------------------------------------------
    # ADD THIS LINE HERE — STABILIZATION DELAY
    # --------------------------------------------
    time.sleep(1.0)   # ensure UAV is stable after takeoff
    
    # -----------------------------------------------------------
    # START CONTROLLER SUBSCRIBER ONLY NOW — AFTER TAKEOFF IS COMPLETE
    # -----------------------------------------------------------
    start_time = time.time()
    controller_subscriber = ControllerVelocitySubscriber(master, start_time)
    executor.add_node(controller_subscriber)

    #node.get_logger().info("Controller subscriber activated — ready to receive /controller/cmd_vel")

    # ---------------------------
    # Keep main alive (run until Ctrl+C)
    # ---------------------------
    try:
        while rclpy.ok():
            #   Optionally monitor TRACK_FLAG or other conditions here
            time.sleep(0.1)
    except KeyboardInterrupt:
         print("KeyboardInterrupt received — starting shutdown/landing")

    plot_stop_event.set()
    stop_event_rect.set()

    try:
        send_velocity(master, 0.0, 0.0, 0.0, 0.0, start_time)
        land(master)
        wait_for_landing(master)
    except Exception as e:
        print("Error during landing sequence:", e)

            
    
    try:
        executor.shutdown(wait=False)
    except Exception:
        pass

        
    try:
        executor.remove_node(controller_subscriber)
    except Exception:
        pass



    try:
        controller_subscriber.destroy_node()
    except Exception:
        pass


    # finally shutdown rclpy and other resources
    try:
        rclpy.shutdown()
    except Exception:
        pass

    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()

