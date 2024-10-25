import os
import sys
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf

from src.envs.gem_ws import GEMWS
import panda_utils.demo_util_pp as demo_util
from lepp.parser import parse_instruction

# Paths Setup
sys.path.extend(['./', '..', '/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP'])

# Constants
Z_MIN_ROBOT = -0.1621
WS_CENTER = [-0.445, -0.079, -0.1625]
ACTION_SEQUENCE = 'pxyzr'
OBS_SOURCE = 'raw'
MAX_Z, MIN_Z = 0.12, 0.02

# Workspace bounds
WORKSPACE = np.array([
    [WS_CENTER[0] - 0.15, WS_CENTER[0] + 0.15],
    [WS_CENTER[1] - 0.15, WS_CENTER[1] + 0.15],
    [WS_CENTER[2], 0.50]
])


def capture_observation(env):
    """Capture multi-view RGB-D observations from the environment."""
    return env.get_multi_obs()


def plot_image(image):
    """Display an RGB image."""
    plt.imshow(np.rot90(image[180:476, 500:730, :3] / 255, 2))
    plt.show(block=False)
    plt.pause(1)


def get_action_data(tf_listener, env, instruction):
    """Retrieve positional data for the pick and place task and format as metadata."""
    pose_data = {}
    for action in ['pick', 'place']:
        input(f"{action.capitalize()}ing: move the arm to the {action} position and press ENTER")
        xyz = env.ur5.tool_position.copy()
        pose_data[f'{action}_pos'] = np.concatenate((xyz, env.ur5.joint_values.copy()))
        pose_data[f'{action}_quat'] = env.ur5.tool_quat.copy()
        pose_data[f'{action}_xyz_base'], pose_data[f'{action}_quat_base'] = tf_listener.lookup_transform('/base_link', '/tool0_controller', rospy.Time(0))
        pose_data[f'labelled_x_{action}'], pose_data[f'labelled_y_{action}'] = demo_util.xy2pixel([xyz[0], xyz[1]])
    pose_data['instruction'] = instruction
    return pose_data

def get_env_metadata(env:GEMWS):
    metadata = dict()
    metadata['intrinsics'] = env.cloud_proxy.get_all_cam_intrinsic()
    metadata['extrinsics']  = env.cloud_proxy.get_all_cam_extrinsic()
    return metadata

def save_demos(demos, filename):
    """Save demonstration data to disk."""
    today_date = datetime.now().strftime("%Y%m%d")
    data_path = os.path.join('demo_rss', today_date)
    os.makedirs(data_path, exist_ok=True)
    file_path = os.path.join(data_path, filename)
    np.save(file_path, demos)
    print(f'Demos (total of {len(demos)}) saved at {os.getcwd()}/{file_path}')


def main():
    # Initialization
    rospy.init_node('image_proxy')
    env = GEMWS(ws_center=WS_CENTER, action_sequence=ACTION_SEQUENCE, obs_source=OBS_SOURCE)
    tf_listener = tf.TransformListener()
    rospy.sleep(2)
    env.arm_reset()
    
    pixel_x_reso, pixel_y_reso = env.cloud_proxy.img_width, env.cloud_proxy.img_height
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'demo_{current_time}.npy'
    demonstrations = dict()

    metadata = get_env_metadata(env)
    demonstrations['metadata'] = metadata
    
    demo_index, collection_ongoing = 0, True

    while collection_ongoing:
        # Capture demonstration episode
        instruction = input("Enter pick&place instruction and press Enter:")
        multiview_rgbd_image = capture_observation(env)
        
        plot_image(multiview_rgbd_image[1])
        action = env.convert_robot_xy_to_pixel_xy()
        
        # Choose action for the captured demo
        action = input("Options: [r] record, [a] abandon, [re] record new, [ae] abandon new, [rs] record save, [as] abandon save: ").strip().lower()
        if action in ['r', 're', 'rs']:
            episode = {'obs': multiview_rgbd_image, 'action': action}
            demonstrations[f'demo_{demo_index}'] = episode
        if action in ['re', 'ae']:
            continue  # Start new episode
        if action in ['rs', 'as']:
            collection_ongoing = False  # End collection

    save_demos(demonstrations, filename)


if __name__ == '__main__':
    main()
