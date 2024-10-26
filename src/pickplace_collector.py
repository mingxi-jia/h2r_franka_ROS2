import os
import sys
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf
import pickle

from src.envs.gem_ws import GEMWS
import panda_utils.demo_util_pp as demo_util


def capture_observation(env:GEMWS):
    """Capture multi-view RGB-D observations from the environment."""
    return env.get_multi_obs()

def plot_image(image:np.array):
    """Display an RGB image."""
    plt.imshow(image / 255)
    plt.show(block=False)
    plt.pause(1)


def get_action_data(env:GEMWS):
    """Retrieve positional data for the pick and place task and format as metadata."""
    pose_data = {}
    for action in ['pick', 'place']:
        input(f"{action.capitalize()}ing: move the arm to the {action} position and press ENTER")
        pose_data[f'robot_{action}_pos'] = env.panda_arm.get_ee_pose()
        pose_data[f'pixel_{action}'] = env.get_ee_pixel_xy()
    return pose_data

def get_env_metadata(env:GEMWS):
    metadata = dict()
    metadata['intrinsics'] = env.cloud_proxy.get_all_cam_intrinsic()
    metadata['extrinsics']  = env.cloud_proxy.get_all_cam_extrinsic()
    return metadata

def save_demo(demo:dict, file_path:str):
    with open(file_path, 'wb') as file:
        pickle.dump(demo, file)

def main():
    # Initialization
    rospy.init_node('image_proxy')
    env = GEMWS()
    rospy.sleep(2)
    env.arm_reset()
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'demo_{current_time}.pkl'
    demonstrations = dict()

    metadata = get_env_metadata(env)
    demonstrations['metadata'] = metadata
    
    demo_index, collection_ongoing = 0, True

    while collection_ongoing:
        # Capture demonstration episode
        instruction = input("Enter instruction and press Enter (format: {PICKOBJ} and {PLACEOBJ}):")
        multiview_rgbd_image = capture_observation(env)
        
        plot_image(multiview_rgbd_image['kevin'])
        action = get_action_data(env)
        
        # Choose action for the captured demo
        action = input("Options: [r] record, [a] abandon, [e] end: ").strip().lower()
        if action in ['r', 'r ']:
            print("saved and continue")
            demo = {'obs': multiview_rgbd_image, 'action': action, 'instruction': instruction}
            demonstrations[f'demo_{demo_index}'] = demo
            demo_index += 1
            save_demo(demo, filename)
        elif action in ['a', 'a ']:
            continue  # Start new episode
        elif action in ['e', 'e ']:
            collection_ongoing = False  # End collection
        else:
            print("abandon and continue")
            continue  # Start new episode

    print('end')


if __name__ == '__main__':
    main()
