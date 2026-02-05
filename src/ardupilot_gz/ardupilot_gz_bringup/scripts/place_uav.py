#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import ModelStates, EntityState

class PlaceUAV(Node):
    def __init__(self):
        super().__init__('place_uav_on_ugv')

        # Change these names if your models have different entity names
        self.ugv_name = 'husky'
        self.uav_name = 'iris'
        self.height_offset = 0.6   # meters above the UGV top surface

        self.ugv_pose = None
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.sub = self.create_subscription(ModelStates, '/model_states', self.callback, 10)

        self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        self.client.wait_for_service()
        self.get_logger().info('Service available. Waiting for model states...')

    def callback(self, msg: ModelStates):
        # Get UGV pose
        if self.ugv_name in msg.name and self.uav_name in msg.name:
            ugv_idx = msg.name.index(self.ugv_name)
            uav_idx = msg.name.index(self.uav_name)

            ugv_pose = msg.pose[ugv_idx]
            uav_pose = msg.pose[uav_idx]

            # Compute new UAV pose
            new_pose = uav_pose
            new_pose.position.x = ugv_pose.position.x
            new_pose.position.y = ugv_pose.position.y
            new_pose.position.z = ugv_pose.position.z + self.height_offset

            # Build request
            req = SetEntityState.Request()
            req.state = EntityState()
            req.state.name = self.uav_name
            req.state.pose = new_pose

            # Send request
            future = self.client.call_async(req)
            future.add_done_callback(self.done_callback)

            self.get_logger().info(f'UAV positioned above {self.ugv_name}.')
            # Stop after one update
            rclpy.shutdown()

    def done_callback(self, future):
        if future.result() is not None:
            self.get_logger().info('UAV successfully placed on top of UGV.')
        else:
            self.get_logger().error('Failed to set UAV position.')

def main(args=None):
    rclpy.init(args=args)
    node = PlaceUAV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
