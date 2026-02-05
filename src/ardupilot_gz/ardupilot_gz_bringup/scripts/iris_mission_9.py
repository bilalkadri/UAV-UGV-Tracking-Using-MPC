#!/usr/bin/env python3
import time
import math
import matplotlib.pyplot as plt
from pymavlink import mavutil

# ==============================
# Configuration
# ==============================
TARGET_ALT_M = 2.0
CIRCLE_RADIUS_M = 2.0
CIRCLE_SPEED_DEG_PER_SEC = 30  # degrees per second
CONN_STR = 'udp:127.0.0.1:14551'
COMMAND_RATE_HZ = 5  # Hz

# ==============================
# MAVLink Utility Functions
# ==============================
def wait_heartbeat(master):
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"✅ Heartbeat from system {master.target_system} component {master.target_component}")

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
            print(f"✅ Mode set to {mode_name}")
            return
    print(f"⚠️ Failed to set mode {mode_name}")

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
                print("✅ Armed." if arm else "✅ Disarmed.")
                return

def request_streams(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        1_000_000 // 10, 0, 0, 0, 0, 0
    )

def read_relative_alt(master):
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    return msg.relative_alt / 1000.0 if msg else None

def takeoff(master, alt_m):
    print(f"🚁 Taking off to {alt_m} m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, alt_m
    )

def send_velocity(master, vx, vy, vz, start_time):
    """Send velocity command in local NED frame"""
    time_boot_ms = int((time.time() - start_time) * 1000) % 4294967295
    master.mav.set_position_target_local_ned_send(
        time_boot_ms,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,  # only velocity enabled
        0, 0, 0,  # position ignored
        vx, vy, vz,
        0, 0, 0,  # accel
        0, 0       # yaw, yaw_rate
    )

def land(master):
    print("🛬 Landing...")
    set_mode(master, "LAND")

# ==============================
# Mission Helper Functions
# ==============================
def takeoff_and_wait(master, alt):
    set_mode(master, "GUIDED")
    arm_and_wait(master, True)
    takeoff(master, alt)
    while True:
        alt_now = read_relative_alt(master)
        if alt_now and alt_now >= alt * 0.95:
            print("✅ Reached target altitude")
            break
        time.sleep(0.5)

def wait_for_landing(master):
    while True:
        alt = read_relative_alt(master)
        if alt and alt <= 0.2:
            break
        time.sleep(0.5)
    arm_and_wait(master, False)
    print("✅ Landed and disarmed.")

# ==============================
# Main Function
# ==============================
def main():
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)
    request_streams(master)

    # ==========================
    # PHASE 1: Straight flight
    # ==========================
    print("\n🚁 === Phase 1: Takeoff → Straight Flight → Land ===")
    takeoff_and_wait(master, TARGET_ALT_M)

    print("➡️ Moving backward for 5 seconds...")
    start_time = time.time()
    duration = 5
    dt = 1.0 / COMMAND_RATE_HZ
    while time.time() - start_time < duration:
        send_velocity(master, 1.0, 0.0, 0.0, start_time)  # move forward (x direction)
        time.sleep(dt)

    land(master)
    wait_for_landing(master)

    # ==========================
    # PHASE 2: Circular path
    # ==========================
    print("\n🌀 === Phase 2: Takeoff → Circle → Land ===")
    takeoff_and_wait(master, TARGET_ALT_M)

    print("🌀 Flying in a circular path (velocity-based commands)...")
    start_time = time.time()
    dt = 1.0 / COMMAND_RATE_HZ
    circle_duration = 360 / CIRCLE_SPEED_DEG_PER_SEC
    omega = math.radians(CIRCLE_SPEED_DEG_PER_SEC)

    # Store trajectory for plotting
    x_hist, y_hist = [], []

    while True:
        elapsed = time.time() - start_time
        if elapsed > circle_duration:
            break

        angle = omega * elapsed
        vx = -CIRCLE_RADIUS_M * omega * math.sin(angle)
        vy =  CIRCLE_RADIUS_M * omega * math.cos(angle)
        vz = 0

        send_velocity(master, vx, vy, vz, start_time)
        x_hist.append(CIRCLE_RADIUS_M * math.cos(angle))
        y_hist.append(CIRCLE_RADIUS_M * math.sin(angle))
        time.sleep(dt)

    print("✅ Circle complete. Initiating landing...")
    land(master)
    wait_for_landing(master)

    # ==========================
    # Plot Trajectory
    # ==========================
    plt.figure()
    plt.plot(y_hist, x_hist, 'b-', linewidth=2)
    plt.title("Drone Circular Trajectory")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    print("✅ Mission complete!")

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    main()

