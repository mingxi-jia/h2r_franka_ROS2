import os
import time
from termcolor import colored
import numpy as np
from networks.pick_agent import Actor
import argparse
import torch
import rospy
import matplotlib.pyplot as plt
from instructions import generate_pick_instruction_shape_part

import pickle

from src.envs.env import Env
import src.simulator.utils as utils
import utils.demo_util_pp as demo_util
import sys


import torch
from cliport import agents
from omegaconf import OmegaConf

import os
import pickle
import json
import wandb
import re


import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment
from cliport.utils import utils
from lepp.clip_preprocess import CLIP_processor
# from lepp.parser import parse_instruction
from lepp.dataset_tool import dataTool
from cliport.dataset_real import RealDataset
# import src.utils.demo_util


# import time
# from termcolor import colored
# from networks.pick_agent_lan_img import Actor as ActorLO
# import clip
def load_hydra_config(config_path):
    return OmegaConf.load(config_path)

def process_obs(rgb, depth):
    img = np.concatenate((rgb,
                          depth[Ellipsis, None],
                          depth[Ellipsis, None],
                          depth[Ellipsis, None]), axis=2)
    # img = utils.preprocess(img, dist='real')
    return img

def rotatePixelCoordinate(image_shape: tuple, pixel_xy: np.array, rotate_angle: float):
    '''
    We define x, y to be row and column respectively
    rotate_angle is in rad
    '''
    image_shape = np.array(image_shape)
    image_center = image_shape[:2]//2
    rotation_mat = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)],
                             [np.sin(rotate_angle), np.cos(rotate_angle)]])
    pixel_xy = (pixel_xy-image_center).reshape(2, 1)
    length = np.sqrt(pixel_xy[0]**2+pixel_xy[1]**2)
    result = rotation_mat.dot(pixel_xy).reshape(2,)+np.array([image_center[1],image_center[0]])
    result_x = np.clip(result[0], 0, image_shape[1])
    result_y = np.clip(result[1], 0, image_shape[0])
    return np.array([result_x, result_y]).astype(int)


def rotateImage90(image: np.array):
    return np.rot90(image)

def visualize(rgb, depth, p0, p1, p0_theta, p1_theta, clip_features):
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(rgb/255)
    ax[1].imshow(depth)
    ax[2].imshow(clip_features)
    # ax[2].imshow(clip_feature_pick[..., 0])
    # ax[3].imshow(clip_feature_place[..., 0])
    p0_theta = (p0_theta + 2*np.pi) % (2*np.pi)
    p1_theta = p0_theta + p1_theta
    print('row, column, rotz:', p0[0], p0[1])
    ax[0].plot(p0[1], p0[0], marker='o', color="green")
    ax[0].plot(p1[1], p1[0], marker='x', color="red")
    arrow_length = 30
    ax[0].arrow(p0[1], p0[0],
                arrow_length*np.cos(p0_theta),
                -arrow_length*np.sin(p0_theta),
                width=0.005,
                color='green')
    ax[0].arrow(p1[1], p1[0],
                arrow_length*np.cos(p1_theta),
                -arrow_length*np.sin(p1_theta),
                width=0.005,
                color='red')
    fig.canvas.draw()
    plt.show(block=False)
    plt.pause(1)


def parse_instruction(instruction):
    pick, place = instruction.split(' and ')
    pick = " ".join(pick.split(' ')[1:])
    place = " ".join(place.split(' ')[2:])
    return pick, place

if __name__ == "__main__":
    rospy.init_node('eval_pp')

    exp_path = '/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP/exps'
    agent_name = 'LEPP'
    tcfg = load_hydra_config('/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP/cliport/cfg/train.yaml')
    tcfg['lepp']['model_name'] = 'unetl-score-vit-postLinearMul'
    tcfg['lepp']['pick_kernel_name'] = 'unetl'
    tcfg['lepp']['place_kernel_name'] = 'eunet'
    tcfg['train']['task'] = 'pick-part-in-brown-box-processed'
    tcfg['dataset']['type'] = 'realtable'
    tcfg['dataset']['use_image_goal'] = True
    tcfg['lepp']['logit_out_channel'] = 3
    tcfg['train']['n_demos'] = 100


    training_npy = '/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP/data/' + tcfg['train']['task'] + '.npy'
    data_path = '/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP/data/'
    train_ds = RealDataset(training_npy, tcfg, n_demos=100)
    agent = agents.names[agent_name](None, tcfg, None, None)
    path = os.path.join(exp_path, "pick-part-in-brown-box-processed"+f"-n-train{tcfg['train']['n_demos']}", "checkpoints")
    agent.load(os.path.join(path, 'steps=10000.pt'))

    # clip_processor = CLIP_processor()
    querytool = dataTool(train_ds, tcfg['train']['task'], 'val')

    global model, alg, action_sequence, in_hand_mode, workspace, max_z, min_z
    # ws_center = [-0.5539, 0.0298, -0.1625]
    ws_center = [-0.445,-0.079,-0.1625]
    workspace = np.asarray([[ws_center[0] - 0.15, ws_center[0] + 0.15],
                            [ws_center[1] - 0.15, ws_center[1] + 0.15],
                            [ws_center[2], 0.50]])
    max_z = 0.12
    min_z = 0.02



    action_sequence = 'pxyzr'
    obs_source = 'raw'
    env = Env(ws_center=ws_center, action_sequence=action_sequence, obs_source=obs_source)
    pixel_small_reso, pixel_big_reso = env.cloud_proxy.img_width, env.cloud_proxy.img_height
    rospy.sleep(2)
    env.ur5.moveToHome()
    rospy.sleep(2)

    while True:
        instruction = input("Type instruction and then Press Enter take photo...")
        depth, rgb, clip_feature_pick, clip_feature_place = env.cloud_proxy.getObs(instruction)
        clip_feature_pick = clip_feature_pick[..., None]
        clip_feature_place = clip_feature_place[..., None]

        pick_inst, place_inst = parse_instruction(instruction)
        pick_text_feature = env.cloud_proxy.clip_processor.get_clip_text_feature(pick_inst)
        place_text_feature = env.cloud_proxy.clip_processor.get_clip_text_feature(place_inst)
        pick_crop, pick_similarity = querytool.query_crop(pick_text_feature)
        place_crop, place_similarity = querytool.query_crop(place_text_feature)
        # pick_crop = None
        # place_crop = None

        if pick_crop is not None:
            _, _, clip_feature_pick_crop, _ = env.cloud_proxy.clip_processor.get_clip_feature_from_text_and_image(rgb, pick_inst, pick_crop, kernel_size=40, stride=20)
            clip_feature_pick = (clip_feature_pick + clip_feature_pick_crop)/2

        if place_crop is not None:
            _, _, clip_feature_place_crop, _ = env.cloud_proxy.clip_processor.get_clip_feature_from_text_and_image(rgb, place_inst, place_crop,
                                                                                             kernel_size=40, stride=20)
            clip_feature_place = (clip_feature_place + clip_feature_place_crop) / 2

        img = process_obs(rgb, depth)
        # img = rotateImage90(img)
        # clip_feature_pick = rotateImage90(clip_feature_pick)
        # clip_feature_place = rotateImage90(clip_feature_place)
        obs = {'img': img,
               'clip_pick': clip_feature_pick,
               'clip_place': clip_feature_place}
        info = {'lang_goal': instruction}


        Z_MIN_ROBOT=-0.1621
        pre_pick_height = Z_MIN_ROBOT+0.3
        pick_height = Z_MIN_ROBOT-0.005
        pre_place_height = Z_MIN_ROBOT+0.3
        place_height = Z_MIN_ROBOT+0.2

        act = agent.actReal(obs, info)

        p0_x, p0_y, p0_theta = act['pick']
        p1_x, p1_y, p1_theta = act['place']
        visualize(img[..., :3], img[..., 3], [p0_x, p0_y], [p1_x, p1_y], p0_theta, p1_theta, clip_feature_pick[...,0])

        # p0_x, p0_y = rotatePixelCoordinate(img.shape, np.array([p0_x, p0_y]), -np.pi/2)
        # p1_x, p1_y = rotatePixelCoordinate(img.shape, np.array([p1_x, p1_y]), -np.pi/2)
        # p0_theta = p0_theta - np.pi/2

        x0, y0 = demo_util.pixel2xy([p0_x, p0_y], pixel_small_reso, pixel_big_reso)
        x1, y1 = demo_util.pixel2xy([p1_x, p1_y], pixel_small_reso, pixel_big_reso)
        # p0_theta = p0_theta
        p0_theta = (p0_theta + 2 * np.pi) % (2 * np.pi)
        p1_theta = (p0_theta + p1_theta  + 2 * np.pi) % (2 * np.pi)
        # p1_theta = p1_theta + 1.57

        env.ur5.moveToP(x0, y0, pre_pick_height, 0,0,p0_theta)
        env.ur5.moveToP(x0, y0, pick_height, 0, 0, p0_theta)
        env.ur5.gripper.closeGripper()
        time.sleep(1)

        env.ur5.moveToP(x0, y0, pre_place_height, 0, 0, p0_theta)
        env.ur5.moveToP(x1, y1, pre_place_height, 0, 0, p1_theta)
        env.ur5.moveToP(x1, y1, place_height, 0, 0, p1_theta)
        env.ur5.gripper.openGripper()
        time.sleep(1)

        env.ur5.moveToHome()

        confirmation = input('q to quit; c to continue.')
        if confirmation=='q':
            break
