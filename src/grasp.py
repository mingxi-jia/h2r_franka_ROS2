import os
import sys
import collections
import time

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

sys.path.append('./')
sys.path.append('..')

import rospy
from src.envs.env import Env
import panda_utils.demo_util as demo_util

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
    ws_center = [-0.5539, 0.0298, -0.1625]
    workspace = np.asarray([[ws_center[0]-0.15, ws_center[0]+0.15],
                            [ws_center[1]-0.15, ws_center[1]+0.15],
                            [ws_center[2], 0.50]])
    max_z = 0.12
    min_z = 0.02



    action_sequence = 'pxyzr'
    obs_source = 'raw'
    env = Env(ws_center=ws_center, action_sequence=action_sequence, obs_source=obs_source)
    rospy.sleep(2)
    env.ur5.moveToHome()
    env.ur5.gripper.closeGripper()
    # env.ur5.gripper.closeGripper()

    # Testing Observations and Image, to be deleted once done all the jobs
    rgbd = env.getObs(None)
    print(f"RGB shape is :{rgbd[:,:,:3].shape}")
    plt.imshow(rgbd[:, :, :3].numpy().astype(int))

    plt.show(block=False)
    # Testing ends
    current_time = time.strftime("%Y,%m,%d,%H,%M,%S").replace(',', '-')
    file_name = f'demo_{current_time}'
    demos = []
    num_demos = 0
    collection_ongoing = True
    while collection_ongoing == True:
        env.ur5.moveToHome()
        input("Press Enter to take photo...")
        rgbd = env.getObs(None)
        plt.imshow(rgbd[:,:,:3].numpy().astype(int))
        plt.show(block=False)
        plt.pause(1)
        description = input("Please manually move to the grasping pose.\n"
              "Then type the object description and Enter to execute and save...")
        xyz = env.ur5.tool_position.copy()
        rz = env.ur5.joint_values.copy()
        quat = env.ur5.tool_quat.copy()
        pose = np.concatenate((xyz, rz))
        print(f'grasping pose is at {pose}')
        labelled_x, labelled_y = demo_util.xy2pixel([xyz[0], xyz[1]])
        labelled_rgb = rgbd[:, :, :3].numpy().astype(int)
        labelled_rgb[labelled_x-2: labelled_x+2, labelled_y-2: labelled_y+2] = 255
        plt.imshow(labelled_rgb)
        plt.show(block=False)
        plt.pause(1)
        env.ur5.gripper.closeGripper()
        x = input('Enter r to record this demo and continue;\n'
                  'Enter a to abandon this demo and continue;\n'
                  'Enter rs to record this demo, save and end;\n'
                  'Enter as to abandon this demo, save and end.')
        if x == 'r':
            demo = {'rgbd': rgbd,
                    'pose': pose,
                    'quat': quat,
                    'description': description}
            demos.append(demo)
            num_demos += 1
            moveAndPlace(env)
        elif x == 'a':
            moveAndPlace(env)
        elif x == 'rs':
            demo = {'rgbd': rgbd,
                    'pose': pose,
                    'quat': quat,
                    'description': description}
            demos.append(demo)
            num_demos += 1
            moveAndPlace(env)
            collection_ongoing = False
        elif x == 'as':
            collection_ongoing = False
        else:
            moveAndPlace(env)

    np.save(file_name, demos)
    print(f'demos (total of {num_demos}) are saved at {os.getcwd()}/{file_name}')




