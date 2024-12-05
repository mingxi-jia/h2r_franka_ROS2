import cloud_proxy as cloud_utils
import rclpy
from rclpy.node import Node
import os
import numpy as np


def get_screenshot(cloud_proxy, fname, args=None,):
    try:
        cloud_proxy.get_env_screenshot(fname)
        print(f'Screenshot {fname} recorded')
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    scene_type = 'mug_on_rack'
    scene_name = 'crossword_mug_skinny_rack'
    scene_root_dir = '/home/mingxi/code/h2r_franka_ROS2_skye/grasp_data'
    scene_folder = os.path.join(scene_root_dir, scene_type, scene_name)
    os.makedirs(scene_folder, exist_ok=True)
    scene_initial_screenshot = 'initial_scene.npy'

    rclpy.init(args=None)
    workspace_size = np.array([[0.3, 0.7],
                    [-0.2, 0.2],
                    [-0.02, 1.0]]) 
    cloud_proxy = cloud_utils.CloudProxy(workspace_size=workspace_size, use_inhand=False)

    input("Record initial pcl scene - press enter when finished")
    get_screenshot(cloud_proxy, os.path.join(scene_folder, scene_initial_screenshot))
    cloud_proxy.destroy_node()
    rclpy.shutdown()
