import os
import sys
import time
import copy
import math
import collections

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

sys.path.append('./')
sys.path.append('..')
from agents.models_x import *
from agents.fc.dqn_x_rot_in_hand import DQNXRotInHand
from agents.fc.policy_x_rot_in_hand import PolicyRotInHand
from agents.fc.dqn_x_rot_in_hand_anneal import DQNXRotInHandAnneal
from agents.fc.dqn_x_rot_in_hand_margin import DQNXRotInHandMargin

from utils.parameters_x import *

import rospy
from img_proxy import ImgProxy
from env import Env

Transition = collections.namedtuple('Transition', 'state obs action reward next_state next_obs done')
ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done expert')
Policy = collections.namedtuple('Policy', 'state obs action')

def creatAgent():
    diag_length = float(heightmap_size) * np.sqrt(2)
    diag_length = int(np.ceil(diag_length / 32) * 32)
    fcn = FCNInHandDynamicFilterPyramid(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)

    if alg == 'dagger':
        agent = PolicyRotInHand(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives,
                                sl, buffer_type=='per', num_rotations, half_rotation, patch_size, divide_factor)
    elif alg == 'dqn':
        agent = DQNXRotInHand(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives, sl,
                              buffer_type=='per', num_rotations, half_rotation, patch_size)
    elif alg == 'dqn_sl_anneal':
        agent = DQNXRotInHandAnneal(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                    num_primitives, sl, buffer_type=='per', num_rotations, half_rotation, patch_size)
    elif alg == 'dqn_margin':
        agent = DQNXRotInHandMargin(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                    num_primitives, sl, buffer_type=='per', num_rotations, half_rotation, patch_size,
                                    margin, margin_l, margin_weight, margin_beta, divide_factor)
    else:
        raise NotImplementedError
    return agent

if __name__ == '__main__':
    rospy.init_node('image_proxy')
    ws_center = [-0.533, 0.04, 0.09]
    workspace = np.asarray([[ws_center[0]-0.15, ws_center[0]+0.15],
                            [ws_center[1]-0.15, ws_center[1]+0.15],
                            [0, 0.50]])

    model = 'dfpyramid'
    agent = creatAgent()
    pre = '/home/dian/Downloads/snapshot_block_stacking'
    agent.loadModel(pre)
    agent.eval()
    agent.initHis(1)

    env = Env()
    obs = env.getObs()
    state = torch.zeros(1)
    q_map, action_idx, action = agent.getEGreedyActions(state, obs, 0, 0)
    env.ur5.moveToHome()
    env.ur5.pick(action[0, 0].item(), action[0, 1].item(), 0.1, action[0, 2].item())
    print(1)

