#!/usr/bin/env python3
import time
from pymavlink import mavutil
import matplotlib.pyplot as plt

TARGET_ALT_M = 2.0
HOVER_SEC = 5
MOVE_SEC = 10
CONN_STR = 'udp:127.0.0.1:14551'

def wait_heartbeat(master):
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Heartbeat from system {master.target_system} component {master.target_component}")

def set_mode(master, mode_name):
    modes = master.mode_mapping()
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
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1 if arm else 0, 0, 0, 0, 0, 0, 0
    )
    while True:
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb:
            armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            if armed == arm:
                print("Armed." if arm else "Disarmed.")
                return

def request_streams(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        1_000_000 // 10, 0, 0, 0, 0, 0
    )

def read_global_position(master):
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    if msg:
        x = msg.lat / 1e7
        y = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0
        return x, y, alt
    return None, None, None

def takeoff(master, alt_m):
    print(f"Takeoff to {alt_m} m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, alt_m
    )

def send_velocity(master, vx, vy, vz):
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )

def land(master):
    print("Landing...")
    set_mode(master, "LAND")

def main():
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)
    request_streams(master)

    set_mode(master, "GUIDED")
    arm_and_wait(master, True)
    takeoff(master, TARGET_ALT_M)

    # Position lists for plotting
    x_data, y_data = [], []

    print("Waiting to reach target altitude...")
    while True:
        x, y, alt = read_global_position(master)
        if alt and alt >= TARGET_ALT_M * 0.95:
            print("Reached target altitude.")
            break
        if x and y:
            x_data.append(x)
            y_data.append(y)
        time.sleep(0.5)

    print(f"Hovering for {HOVER_SEC} seconds...")
    hover_start = time.time()
    while time.time() - hover_start < HOVER_SEC:
        x, y, alt = read_global_position(master)
        if x and y:
            x_data.append(x)
            y_data.append(y)
        time.sleep(0.5)

    print(f"Moving horizontally for {MOVE_SEC} seconds...")
    move_start = time.time()
    while time.time() - move_start < MOVE_SEC:
        send_velocity(master, vx=1.0, vy=0.0, vz=0.0)
        x, y, alt = read_global_position(master)
        if x and y:
            x_data.append(x)
            y_data.append(y)
        time.sleep(0.5)

    send_velocity(master, vx=0.0, vy=0.0, vz=0.0)
    time.sleep(1)

    land(master)
    while True:
        x, y, alt = read_global_position(master)
        if alt and alt <= 0.2:
            break
        if x and y:
            x_data.append(x)
            y_data.append(y)
        time.sleep(0.5)

    arm_and_wait(master, False)
    print("Mission complete.")

    # Plot the UAV X-Y trajectory
    plt.figure(figsize=(7, 6))
    plt.plot(x_data, y_data, 'b-', linewidth=2, label='UAV Path')
    plt.scatter(x_data[0], y_data[0], color='green', label='Start')
    plt.scatter(x_data[-1], y_data[-1], color='red', label='End')
    plt.title("UAV X-Y Position Trajectory")
    plt.xlabel("Latitude (°)")
    plt.ylabel("Longitude (°)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

