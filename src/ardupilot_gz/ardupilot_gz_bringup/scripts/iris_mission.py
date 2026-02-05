

#!/usr/bin/env python3

import rospy
import time
from geometry_msgs.msg import TwistStamped
from mavros_msgs.srv import CommandBool, CommandTOL
from mavros_msgs.msg import State

def wait_for_connection():
    rospy.loginfo("Waiting for FCU connection...")
    rospy.wait_for_message("/mavros/state", State)
    rospy.loginfo("Connected to FCU.")

def arm():
    rospy.wait_for_service('/mavros/cmd/arming')
    try:
        arm_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        arm_srv(True)
        rospy.loginfo("Drone armed")
    except rospy.ServiceException as e:
        rospy.logerr("Arming failed: %s" % e)

def takeoff(altitude=2.0):
    rospy.wait_for_service('/mavros/cmd/takeoff')
    try:
        takeoff_srv = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
        takeoff_srv(min_pitch=0.0, yaw=0.0, latitude=0.0,
                    longitude=0.0, altitude=altitude)
        rospy.loginfo("Takeoff to altitude %.1f m" % altitude)
        time.sleep(5)  # wait for reaching altitude
    except rospy.ServiceException as e:
        rospy.logerr("Takeoff failed: %s" % e)

def move_forward(duration=10.0, speed=1.0):
    pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',
                          TwistStamped, queue_size=10)
    rospy.loginfo("Moving forward for %.1f seconds..." % duration)
    vel_msg = TwistStamped()
    vel_msg.twist.linear.x = speed  # forward in x-direction
    vel_msg.twist.linear.y = 0.0
    vel_msg.twist.linear.z = 0.0
    vel_msg.twist.angular.z = 0.0

    rate = rospy.Rate(10)  # 10 Hz
    start_time = time.time()
    while time.time() - start_time < duration and not rospy.is_shutdown():
        vel_msg.header.stamp = rospy.Time.now()
        pub.publish(vel_msg)
        rate.sleep()

def land():
    rospy.wait_for_service('/mavros/cmd/land')
    try:
        land_srv = rospy.ServiceProxy('/mavros/cmd/land', CommandTOL)
        land_srv(min_pitch=0.0, yaw=0.0, latitude=0.0,
                 longitude=0.0, altitude=0.0)
        rospy.loginfo("Landing...")
    except rospy.ServiceException as e:
        rospy.logerr("Landing failed: %s" % e)

if __name__ == '__main__':
    rospy.init_node('iris_auto_mission')

    wait_for_connection()
    arm()
    takeoff(altitude=2.0)
    move_forward(duration=10.0, speed=1.0)
    land()
