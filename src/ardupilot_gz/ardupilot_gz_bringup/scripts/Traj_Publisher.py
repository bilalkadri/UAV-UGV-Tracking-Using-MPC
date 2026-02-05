from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

path_msg = Path()
path_msg.header.frame_id = "map"

# Fill poses
for point in predicted_points:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = point[0]
    pose.pose.position.y = point[1]
    pose.pose.position.z = point[2]
    path_msg.poses.append(pose)

publisher.publish(path_msg)
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

path_msg = Path()
path_msg.header.frame_id = "map"

# Fill poses
for point in predicted_points:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = point[0]
    pose.pose.position.y = point[1]
    pose.pose.position.z = point[2]
    path_msg.poses.append(pose)

publisher.publish(path_msg)

