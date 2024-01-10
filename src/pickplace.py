import os
import sys
import collections
import time
from datetime import datetime

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

sys.path.append('./')
sys.path.append('..')

import rospy
from src.envs.env import Env
import utils.demo_util_pp as demo_util

def moveAndPlace(env):
    place_pose_1 = [-0.37496883, -1.85229093,  1.99809742, -1.71823579, -1.56919986,
       -0.37594825]
    place_pose_2 = [ 0.79398787, -1.03516657,  1.43296528, -1.96834976, -1.57040912,
        0.79418302]
    env.ur5.moveToJ(place_pose_1)
    env.ur5.waitUntilNotMoving()
    env.ur5.moveToJ(place_pose_2)
    env.ur5.gripper.openGripper()

if __name__ == '__main__':
    rospy.init_node('image_proxy')

    global model, alg, action_sequence, in_hand_mode, workspace, max_z, min_z
    # ws_center = [-0.5539, 0.0298, -0.1625]
    ws_center = [-0.445, -0.079, -0.1625]
    workspace = np.asarray([[ws_center[0]-0.15, ws_center[0]+0.15],
                            [ws_center[1]-0.15, ws_center[1]+0.15],
                            [ws_center[2], 0.50]])
    max_z = 0.12
    min_z = 0.02

    Z_MIN_ROBOT = -0.1621
    pre_pick_height = Z_MIN_ROBOT + 0.3
    pick_height = Z_MIN_ROBOT + 0.0
    pre_place_height = Z_MIN_ROBOT + 0.3
    place_height = Z_MIN_ROBOT + 0.2


    action_sequence = 'pxyzr'
    obs_source = 'raw'
    env = Env(ws_center=ws_center, action_sequence=action_sequence, obs_source=obs_source)
    rospy.sleep(2)
    # env.ur5.moveToHome()
    # env.ur5.gripper.openGripper()
    env.ur5.gripper.closeGripper()

    # Testing Observations and Image, to be deleted once done all the jobs
    rgbd = env.getObs(None)
    print(f"RGB shape is :{rgbd[:,:,:3].shape}")
    plt.imshow(rgbd[:, :, :3].numpy().astype(int))

    plt.show(block=False)
    # Testing ends
    current_time = time.strftime("%Y,%m,%d,%H,%M,%S").replace(',', '-')
    file_name = f'demo_{current_time}'
    demos = []
    # demos = np.load('add the path of file that needs modification', allow_pickle=True)
    num_demos = 0
    collection_ongoing = True
    while collection_ongoing == True:
        episode_ongoing = True
        episode = []
        while episode_ongoing:
            env.ur5.moveToHome()
            input("Press Enter to take photo...")
            rgbd = env.getObs(None)
            plt.imshow(rgbd[:,:,:3].numpy().astype(int))
            plt.show(block=False)
            plt.pause(1)
            description = input("PICKING Please manually move to the grasping pose.\n"
                  "Then type the object description and Enter to execute and save...")
            xyz = env.ur5.tool_position.copy()
            p0_theta = env.ur5.joint_values.copy()[-1]
            quat0 = env.ur5.tool_quat.copy()
            pose0 = np.concatenate((xyz, env.ur5.joint_values.copy()))
            print(f'grasping pose is at {pose0}')
            labelled_x_pick, labelled_y_pick = demo_util.xy2pixel([xyz[0], xyz[1]])
            labelled_rgb = rgbd[:, :, :3].numpy().astype(int)
            labelled_rgb[labelled_x_pick-2: labelled_x_pick+2, labelled_y_pick-2: labelled_y_pick+2] = 255
            plt.imshow(labelled_rgb)
            plt.show(block=False)
            plt.pause(1)
            env.ur5.gripper.closeGripper()


            confirmation = input("PLACING Please manually move to the grasping pose and press enter.\n")
            xyz = env.ur5.tool_position.copy()
            p1_theta = env.ur5.joint_values.copy()[-1]
            quat1 = env.ur5.tool_quat.copy()
            pose1 = np.concatenate((xyz, env.ur5.joint_values.copy()))
            print(f'grasping pose is at {pose1}')
            labelled_x_place, labelled_y_place = demo_util.xy2pixel([xyz[0], xyz[1]])
            labelled_rgb = rgbd[:, :, :3].numpy().astype(int)
            labelled_rgb[labelled_x_place - 2: labelled_x_place + 2, labelled_y_place - 2: labelled_y_place + 2] = 255
            plt.imshow(labelled_rgb)
            plt.show(block=False)
            plt.pause(1)
            env.ur5.gripper.openGripper()
            demo = {'depth': (1.08 - rgbd[..., 3] / 1000.).numpy(),
                    'rgb': (rgbd[..., :3]).numpy(),
                    'p0': [labelled_x_pick, labelled_y_pick],
                    'p0_theta': p0_theta,
                    'pose0': pose0,
                    'quat0': quat0,
                    'p1': [labelled_x_place, labelled_y_place],
                    'p1_theta': p1_theta,
                    'pose1': pose1,
                    'quat1': quat1,
                    'instruction': description}
            x = input('Enter r to record this demo and continue episode;\n'
                      'Enter a to abandon this demo and continue episode;\n'
                      'Enter re to record this demo and start a new episode;\n'
                      'Enter ae to abandon this demo and start a new episode;\n'
                      'Enter rs to record this demo, save and end;\n'
                      'Enter as to abandon this demo, save and end.')
            if x == 'r':
                episode.append(demo)
                num_demos += 1
                env.ur5.moveToHome()
            elif x == 'a':
                env.ur5.moveToHome()
            elif x == 're':
                episode.append(demo)
                num_demos += 1
                env.ur5.moveToHome()
                episode_ongoing = False
                if len(episode) > 0:
                    demos.append(episode)
            elif x == 'ae':
                env.ur5.moveToHome()
                episode_ongoing = False
                if len(episode) > 0:
                    demos.append(episode)
            elif x == 'rs':
                episode.append(demo)
                demos.append(episode)
                num_demos += 1
                env.ur5.moveToHome()
                episode_ongoing = False
                collection_ongoing = False
            elif x == 'as':
                if len(episode) > 0:
                    demos.append(episode)
                episode_ongoing = False
                collection_ongoing = False
            else:
                env.ur5.moveToHome()

    if not os.path.exists('demo_rss'):
        os.makedirs('demo_rss')
    today_date = datetime.now().strftime("%Y%m%d")

    data_path = os.path.join('demo_rss', today_date)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    file_path = os.path.join(data_path, file_name)
    np.save(file_path, demos)
    print(f'demos (total of {num_demos}) are saved at {os.getcwd()}/{file_name}')




