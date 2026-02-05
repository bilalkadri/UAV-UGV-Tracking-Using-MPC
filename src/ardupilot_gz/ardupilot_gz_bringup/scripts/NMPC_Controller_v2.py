#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry

# Helper math
def rpy_to_rot(roll, pitch, yaw):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw); sy = np.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R

def ang_vel_transform(roll, pitch):
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    # clamp cp to avoid divide-by-zero but keep it smooth
    if abs(cp) < 1e-6:
        cp = np.sign(cp) * 1e-6 if cp != 0 else 1e-6
    T = np.array([
        [1.0, sr*sp/cp, cr*sp/cp],
        [0.0, cr,       -sr     ],
        [0.0, sr/cp,    cr/cp   ]
    ])
    return T

def quat_to_rpy_msg(q):
    w,x,y,z = q.w, q.x, q.y, q.z
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class SimpleMPCNode(Node):
    def __init__(self):
        super().__init__('simple_mpc_controller')

        # Topics
        self.rel_pose_topic = '/relative_pose_ekf'
        self.ugv_odom_topic = '/odometry'
        self.uav_odom_topic = '/uav/vel_estimated'
        self.cmd_pub_topic = '/mpc/cmd_vel'

        # MPC parameters
        self.N = 12
        self.pred_dt = 0.2   # prediction timestep
        self.mpc_dt = 0.1    # control loop rate

        # weights
        self.Q_pos = np.diag([100.0, 100.0, 50.0])
        self.Q_ang = np.diag([10.0, 10.0, 10.0])
        self.R_du = np.diag([1.0, 1.0, 1.0, 0.1])
        self.W_fov = 200.0

        # control bounds
        self.v_max = 2.0
        self.vz_max = 1.0
        self.yawdot_max = 1.0

        # state
        self.rel_state = np.zeros(6)
        self.have_rel = False

        # odometry
        self.v_g = np.zeros(3)
        self.w_g = np.zeros(3)
        self.v_u = np.zeros(3)
        self.w_u = np.zeros(3)

        # subs / pubs
        self.create_subscription(PoseStamped, self.rel_pose_topic, self.cb_rel_pose, 10)
        self.create_subscription(Odometry, self.ugv_odom_topic, self.cb_ugv_odom, 10)
        self.create_subscription(Odometry, self.uav_odom_topic, self.cb_uav_odom, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_pub_topic, 10)

        # timer
        self.create_timer(self.mpc_dt, self.mpc_loop)

        self.get_logger().info("SimpleMPCNode started. Topics: rel_pose=%s, ugv_odom=%s, uav_odom=%s"%
                               (self.rel_pose_topic, self.ugv_odom_topic, self.uav_odom_topic))

    def cb_rel_pose(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        roll, pitch, yaw = quat_to_rpy_msg(msg.pose.orientation)
        self.rel_state = np.array([x, y, z, roll, pitch, yaw], dtype=float)
        self.have_rel = True

    def cb_ugv_odom(self, msg: Odometry):
         yaw = quat_to_rpy_msg(msg.pose.pose.orientation)[2]

         self.v_g = np.array([
             msg.twist.twist.linear.x,
             msg.twist.twist.linear.y,
             msg.twist.twist.linear.z
         ])

         self.w_g = np.array([0.0, 0.0, msg.twist.twist.angular.z])


    def cb_uav_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        roll, pitch, yaw = quat_to_rpy_msg(q)

        # body → world rotation
        R = rpy_to_rot(roll, pitch, yaw)

        # UAV linear velocity in world frame
        self.v_u = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

        # angular velocity (IMU frame)
        wx = msg.twist.twist.angular.x
        wy = msg.twist.twist.angular.y
        wz = msg.twist.twist.angular.z

        # convert to rpy rates
        T = ang_vel_transform(roll, pitch)
        self.w_u = T @ np.array([wx, wy, wz])


    def mpc_loop(self):
        if not self.have_rel:
            self.get_logger().warn("No relative pose received yet - publishing zero cmd")
            self.publish_cmd([0.0,0.0,0.0], 0.0)
            return

        # copy states/odometry to local variables to avoid race conditions
        x0 = self.rel_state.copy()
        v_g = self.v_g.copy()
        w_g = self.w_g.copy()
        v_u = self.v_u.copy()
        w_u = self.w_u.copy()

        # reference
        z_ref = 0.6
        Xref = np.zeros((self.N, 6))
        for k in range(self.N):
            Xref[k,0:3] = np.array([0.0, 0.0, z_ref])
            Xref[k,3:6] = np.array([0.0, 0.0, 0.0])

        # controls
        U = np.zeros((self.N, 4))
        U_prev = np.zeros_like(U)

        # optimization parameters
        iters = 8
        alpha = 0.2
        eps = 1e-3

        # precompute base cost
        cost_base = self._simulate_cost(x0, U, Xref, v_g, w_g, v_u, w_u)

        for it in range(iters):
            grad = np.zeros_like(U)

            # Efficient forward-difference gradient: perturb one control element at a time
            for i in range(self.N):
                for j in range(4):
                    Up = U.copy()
                    Up[i,j] += eps
                    cost_p = self._simulate_cost(x0, Up, Xref, v_g, w_g, v_u, w_u)
                    grad[i,j] = (cost_p - cost_base) / eps

            # gradient step
            U -= alpha * grad

            # projection
            for k in range(self.N):
                U[k,0] = np.clip(U[k,0], -self.v_max, self.v_max)
                U[k,1] = np.clip(U[k,1], -self.v_max, self.v_max)
                U[k,2] = np.clip(U[k,2], -self.vz_max, self.vz_max)
                U[k,3] = np.clip(U[k,3], -self.yawdot_max, self.yawdot_max)

            # update U_prev and base cost for next iteration
            U_prev = U.copy()
            cost_base = self._simulate_cost(x0, U, Xref, v_g, w_g, v_u, w_u)

            # simple stopping
            if np.linalg.norm(grad) < 1e-2:
                break

        # extract first command
        u0 = U[0,:]
        vx_cmd, vy_cmd, vz_cmd, yawdot_cmd = float(u0[0]), float(u0[1]), float(u0[2]), float(u0[3])

        # safety checks
        if not np.isfinite(vx_cmd + vy_cmd + vz_cmd + yawdot_cmd):
            self.get_logger().error("MPC produced non-finite command — sending zero")
            vx_cmd, vy_cmd, vz_cmd, yawdot_cmd = 0.0, 0.0, 0.0, 0.0

        # clip again
        vx_cmd = float(np.clip(vx_cmd, -self.v_max, self.v_max))
        vy_cmd = float(np.clip(vy_cmd, -self.v_max, self.v_max))
        vz_cmd = float(np.clip(vz_cmd, -self.vz_max, self.vz_max))
        yawdot_cmd = float(np.clip(yawdot_cmd, -self.yawdot_max, self.yawdot_max))

        # log every few cycles to avoid slowing down
        if np.random.rand() < 0.25:
            self.get_logger().info(f"MPC Command → vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, vz={vz_cmd:.3f}, yawdot={yawdot_cmd:.3f}")

        self.publish_cmd([vx_cmd, vy_cmd, vz_cmd], yawdot_cmd)

    def _simulate_cost(self, x0, U, Xref, v_g, w_g, v_u, w_u):
        x_sim = x0.copy()
        total = 0.0
        for k in range(self.N):
            roll, pitch, yaw = x_sim[3], x_sim[4], x_sim[5]
            R_ag = rpy_to_rot(roll, pitch, yaw)
            T_ag = ang_vel_transform(roll, pitch)

            vk = U[k, 0:3]
            yawdotk = U[k, 3]

            # Use correct relative dynamics: R_ag * (v_g - v_u) - v_cmd
            rel_vel = R_ag @ (v_g - v_u) - vk
            x_sim[0:3] = x_sim[0:3] + rel_vel * self.pred_dt

            rel_omega = T_ag @ (w_g - w_u) - np.array([0.0, 0.0, yawdotk])
            x_sim[3:6] = x_sim[3:6] + rel_omega * self.pred_dt

            # angle wrap for error
            e_pos = x_sim[0:3] - Xref[k,0:3]
            total += e_pos.T @ self.Q_pos @ e_pos

            e_ang = wrap_angle(x_sim[3:6] - Xref[k,3:6])
            total += e_ang.T @ self.Q_ang @ e_ang

            total += U[k,:].T @ np.diag([0.1,0.1,0.1,0.01]) @ U[k,:]

            horiz_err = np.linalg.norm(x_sim[0:2])
            total += self.W_fov * (horiz_err**2) / ((horiz_err**2) + (Xref[k,2]**2) + 1e-6)

        # smoothness
        for k in range(self.N):
            du = U[k,:] - (np.zeros(4) if k==0 else U[k-1,:])
            total += du.T @ self.R_du @ du

        return float(total)

    def publish_cmd(self, v_xyz, yawdot):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(v_xyz[0])
        msg.twist.linear.y = float(v_xyz[1])
        msg.twist.linear.z = float(v_xyz[2])
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(yawdot)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

