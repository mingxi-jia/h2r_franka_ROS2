import cloud_proxy as cloud_utils
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import os
import numpy as np

def get_screenshot(cloud_proxy, fname, args=None,):
    try:
        cloud_proxy.get_env_screenshot(fname)
        print(f'Screenshot {fname} recorded')
    except KeyboardInterrupt:
        pass


#ros2 subscriber to record pose 
class PoseListener(Node):
    def __init__(self):
        super().__init__("ee_pose_listener")
        self.listener = self.create_subscription(PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.get_ee_pose, 10)
        self.ee_pos = None
        self.ee_quat = None
        rclpy.spin_once(self, timeout_sec=0.01)

    def get_ee_pose(self, msg):
        self.ee_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])


if __name__ == "__main__":

    demo_type = 'mug_on_rack'
    demo_name = 'crossword_mug_skinny_rack'
    demo_root_dir = '/home/mingxi/code/h2r_franka_ROS2_skye/demo_data'
    demo_folder = os.path.join(demo_root_dir, demo_type, demo_name)
    os.makedirs(demo_folder, exist_ok=True)
    demo_initial_screenshot = 'initial_scene.npy'
    demo_final_screenshot = 'final_scene.npy'
    demo_transform = 'ee_transform'

    rclpy.init(args=None)
    pose_subscriber = PoseListener()
    rclpy.spin_once(pose_subscriber)
    workspace_size = np.array([[0.3, 0.7],
                    [-0.2, 0.2],
                    [-0.02, 1.0]]) 
    cloud_proxy = cloud_utils.CloudProxy(workspace_size=workspace_size, use_inhand=False)

    input("Record initial grasp pose - press enter when finished")
    get_screenshot(cloud_proxy, os.path.join(demo_folder, demo_initial_screenshot))
    rclpy.spin_once(pose_subscriber)
    initial_transform = (pose_subscriber.ee_pos, pose_subscriber.ee_quat) # (self, PoseStamped, '/franka_robot_state_broadcaster/current_pose')
    print(initial_transform)

    input("Record place pose - press enter when finished")

    rclpy.spin_once(pose_subscriber)
    get_screenshot(cloud_proxy, os.path.join(demo_folder, demo_final_screenshot))
    final_transform = (pose_subscriber.ee_pos, pose_subscriber.ee_quat)
    print(final_transform)

    np.savez(os.path.join(demo_folder, demo_transform),
              init_pos=initial_transform[0], 
              init_quat=initial_transform[1],
              final_pos=final_transform[0], 
              final_quat=final_transform[1])

    pose_subscriber.destroy_node()
    cloud_proxy.destroy_node()
    rclpy.shutdown()




    # save

