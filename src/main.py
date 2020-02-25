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
from src.agents.models_x import *
from src.agents.fc.dqn_x_rot_in_hand import DQNXRotInHand
from src.agents.fc.policy_x_rot_in_hand import PolicyRotInHand
from src.agents.fc.dqn_x_rot_in_hand_anneal import DQNXRotInHandAnneal
from src.agents.fc.dqn_x_rot_in_hand_margin import DQNXRotInHandMargin

from src.utils.parameters_x import *

from src.utils import torch_utils

import rospy
from src.img_proxy import ImgProxy
from src.env import Env

Transition = collections.namedtuple('Transition', 'state obs action reward next_state next_obs done')
ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done expert')
Policy = collections.namedtuple('Policy', 'state obs action')

def creatAgent():
    # fcn = FCNSmall(1, num_primitives).to(device)
    diag_length = float(heightmap_size) * np.sqrt(2)
    diag_length = int(np.ceil(diag_length / 32) * 32)
    if model == 'fcn':
        fcn = FCNInHand(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'fc':
        fcn = FCNInHandDomainFC(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'dfpyramid':
        fcn = FCNInHandDynamicFilterPyramid(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'df':
        fcn = FCNInHandDynamicFilter(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'pyramid':
        fcn = FCNInHandPyramidSep(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'dfu':
        fcn = FCNInHandDynamicFilterU(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'fit':
        fcn = From2Fit(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'res':
        fcn = ResNet(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'resu':
        fcn = ResU(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'resucat':
        fcn = ResUCat(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    elif model == 'ucat':
        fcn = UCat(1, num_primitives, domain_shape=(1, diag_length, diag_length)).to(device)
    else:
        raise NotImplementedError

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

def plotQMaps(q_value_maps, save=None, j=0):
    idx = torch_utils.argmax3d(q_value_maps)[0]
    fig = plt.figure(figsize=(6, 9))
    grid = AxesGrid(fig, 111, nrows_ncols=(4, 2))
    for i in range(q_value_maps.size(1)):
        ax = grid[i]
        ax.set_axis_off()
        im = ax.imshow(q_value_maps[0, i], vmin=q_value_maps.min(), vmax=q_value_maps.max())
    for i in range(q_value_maps.size(1), len(grid)):
        grid[i].remove()
    # cbar = ax.cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)
    grid[idx[0]].scatter(x=idx[2], y=idx[1], c='r', s=20)
    if not save:
        fig.show()
    else:
        fig.savefig(os.path.join(save, '{}_qmap.png'.format(j)))
        plt.close(fig)

if __name__ == '__main__':
    rospy.init_node('image_proxy')
    ws_center = [-0.5257, -0.0098, 0.1]
    workspace = np.asarray([[ws_center[0]-0.15, ws_center[0]+0.15],
                            [ws_center[1]-0.15, ws_center[1]+0.15],
                            [0, 0.50]])

    # model = 'dfpyramid'
    model = 'resucat'
    agent = creatAgent()
    # pre = '/home/dian/Downloads/dqfd/snapshot_house_building_3'
    # pre = '/home/dian/Downloads/dqfd/snapshot_house_building_2'
    # pre = '/home/dian/Downloads/dqfd/snapshot_house_building_1'
    # pre = '/home/dian/Downloads/dqfd/snapshot_block_stacking'

    pre = '/home/dian/Downloads/resucatperlin0.02/snapshot_house_building_3'
    agent.loadModel(pre)
    agent.eval()
    agent.initHis(1)

    env = Env()
    rospy.sleep(2)
    env.ur5.moveToHome()
    j=0
    while True:
        j += 1
        obs = env.getObs()
        plt.imshow(obs[0, 0])
        plt.axis('off')
        # plt.savefig(os.path.join('/home/dian/Documents/obs', '{}_obs.png'.format(j)))
        plt.show()
        state = torch.tensor([env.ur5.holding_state], dtype=torch.float32)
        q_map, action_idx, action = agent.getEGreedyActions(state, obs, 0, 0)
        # plotQMaps(q_map, '/home/dian/Documents/qmap', j)
        plotQMaps(q_map)
        pixels = action_idx[:, :2]
        patch = agent.getImgPatch(obs, pixels)
        env.step((state.item(), action[0, 0].item(), action[0, 1].item(), action[0, 2].item()))
        
        agent.updateHis(patch, action[:, 2], torch.tensor([env.ur5.holding_state], dtype=torch.float32), None, torch.zeros(1))
