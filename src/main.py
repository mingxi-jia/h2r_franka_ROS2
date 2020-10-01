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
from agents.fc.policy_x_rot_in_hand import PolicyXRotInHand
from agents.fc.dqn_x_rot_in_hand_anneal import DQNXRotInHandAnneal
from agents.fc.dqn_x_rot_in_hand_margin import DQNXRotInHandMargin
from agents.hierarchy.policy_rot_in_hand import PolicyRotInHand
from agents.hierarchy.in_hand_agent import DQNRotMaxInHand, DQNRotMulInHand, DQNRotMaxSharedInHand
from agents.hierarchy.margin_agent import DQNRotMaxInHandMargin, DQNRotMulInHandMargin, DQNRotMaxSharedInHandMargin, DQNPRotMaxSharedInHandMargin
from agents.hierarchy.rot_z_agent import DQNRotZMaxSharedInHand, DQNRotZMaxSharedInHandMargin, PolicyRotZInHand, PolicyRotZSharedInHand
from agents.hierarchy.rot_2_agent import DQNRot2MaxSharedInHand, DQNRot2MaxSharedInHandMargin, PolicyRot2InHand, DQNRot2MulInHand, PolicyRot2SharedInHand
from agents.hierarchy.zrr_agent import DQNZRRMaxSharedInHand, DQNZRRMaxSharedInHandMargin, PolicyZRRInHand, PolicyZRRSharedInHand
from agents.hierarchy.deictic_agent import DQNRotDeicticMaxSharedInHandMargin, DQNRotDeicticMaxSharedInHand, DQNRotZDeicticMaxSharedInHandMargin, DQNRotZDeicticMaxSharedInHand, DQNRot2DeicticMaxSharedInHand
from agents.hierarchy.policy_shared_in_hand import PolicySharedInHand
from agents.hierarchy.policy_3l_rz_rxz import Policy3LMaxSharedRzRxZ
from agents.hierarchy.dqn_3l_max_shared import DQN3LMaxShared
from agents.hierarchy.dqn_3l_rz_rxz import DQN3LMaxSharedRzRxZ
from agents.hierarchy.margin_3l_max_shared import Margin3LMaxShared
from agents.hierarchy.margin_3l_rz_rxz import Margin3LMaxSharedRzRxZ
from agents.hierarchy.dqn_3l_6d import DQN3L6DMaxShared
from agents.hierarchy.dqn_5l import DQN5LMaxShared

from src.utils.parameters_x import *

from src.utils import torch_utils
from src.utils.torch_utils import rand_perlin_2d

import rospy
from src.img_proxy import ImgProxy
from src.env import Env

Transition = collections.namedtuple('Transition', 'state obs action reward next_state next_obs done')
ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done expert')
Policy = collections.namedtuple('Policy', 'state obs action')
if env == 'block_placing':
    num_rx = 4
    min_rx = -np.pi/8
    max_rx = np.pi/8
else:
    num_rx = 7
    min_rx = -np.pi / 6
    max_rx = np.pi / 6
def creatAgent():
    # fcn = FCNSmall(1, num_primitives).to(device)
    diag_length = float(heightmap_size) * np.sqrt(2)
    diag_length = int(np.ceil(diag_length / 32) * 32)
    if in_hand_mode == 'proj':
        patch_channel = 3
    else:
        patch_channel = 1
    patch_shape = (patch_channel, patch_size, patch_size)
    if alg.find('deictic') >= 0:
        q2_input_shape = (patch_channel * 2, patch_size, patch_size)
    else:
        q2_input_shape = (patch_channel+1, patch_size, patch_size)

    if action_sequence == 'xyzrp':
        q2_output_size = num_primitives * num_rotations * num_zs
    elif action_sequence == 'xyrrp':
        q2_output_size = num_primitives * num_rotations * num_rx
    elif action_sequence == 'xyzrrp':
        q2_output_size = num_primitives * num_rotations * num_rx * num_zs
    else:
        q2_output_size = num_primitives * num_rotations

    if model == 'fcn':
        fcn = FCNInHand(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'fc':
        fcn = FCNInHandDomainFC(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'dfpyramid':
        fcn = FCNInHandDynamicFilterPyramid(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'df':
        fcn = FCNInHandDynamicFilter(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'pyramid':
        fcn = FCNInHandPyramidSep(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'dfu':
        fcn = FCNInHandDynamicFilterU(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'fit':
        fcn = From2Fit(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'res':
        fcn = ResNet(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'resu':
        fcn = ResU(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'resucat':
        if alg.find('shared') >= 0:
            fcn = ResUCatShared(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
        else:
            fcn = ResUCat(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'resucat96':
        if alg.find('shared') >= 0:
            fcn = ResUCatShared96(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
        else:
            raise NotImplementedError
    elif model == 'ucat':
        fcn = UCat(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'res34':
        fcn = ResNet34(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'res101':
        fcn = ResNet101(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'res34_96':
        fcn = Res34Shared96(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    elif model == 'res101_96':
        fcn = Res101Shared96(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)
    else:
        raise NotImplementedError

    if alg in ['dqn_hier_mul', 'margin_hier_mul']:
        cnn = CNNResSig(image_shape=q2_input_shape, n_outputs=q2_output_size).to(device)
    elif alg in ['dagger_hier']:
        cnn = CNNRes(image_shape=q2_input_shape, n_outputs=q2_output_size).to(device)
    elif alg in ['dqn_hier_max', 'margin_hier_max']:
        cnn = CNNResI((1, diag_length, diag_length), q2_input_shape, q2_output_size).to(device)
    elif alg.find('shared') >= 0:
        if model.find('96') >= 0:
            cnn = CNNResShared96(q2_input_shape, q2_output_size).to(device)
        else:
            cnn = CNNResShared(q2_input_shape, q2_output_size).to(device)

    if alg == 'dagger':
        agent = PolicyXRotInHand(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives,
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
    elif alg == 'dqn_hier_mul':
        if action_sequence.count('r') == 2:
            agent = DQNRot2MulInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr,
                                           gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                           half_rotation,
                                           patch_size, num_rx, min_rx, max_rx)
        else:
            agent = DQNRotMulInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                num_primitives, sl, buffer_type =='per', num_rotations, half_rotation, patch_size)
    elif alg == 'dagger_hier':
        if action_sequence == 'xyzrp':
            agent = PolicyRotZInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                    num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size, num_zs, min_z, max_z)
        elif action_sequence == 'xyrrp':
            agent = PolicyRot2InHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                     num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size,
                                     num_rx, min_rx, max_rx)
        elif action_sequence == 'xyzrrp':
            agent = PolicyZRRInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                     num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size,
                                     num_rx, min_rx, max_rx, num_zs, min_z, max_z)
        else:
            agent = PolicyRotInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                    num_primitives, sl, buffer_type=='per', num_rotations, half_rotation, patch_size)

    elif alg == 'dagger_hier_shared':
        if action_sequence == 'xyzrp':
            agent = PolicyRotZSharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                    num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size,
                                           num_zs, min_z, max_z)
        elif action_sequence == 'xyrrp':
            agent = PolicyRot2SharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                           num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size,
                                           num_rx, min_rx, max_rx)
        elif action_sequence == 'xyzrrp':
            agent = PolicyZRRSharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                          num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation,
                                          patch_size, num_rx, min_rx, max_rx, num_zs, min_z, max_z)
        else:
            agent = PolicySharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                        num_primitives, sl, buffer_type=='per', num_rotations, half_rotation, patch_size)

    elif alg == 'margin_hier_mul':
        agent = DQNRotMulInHandMargin(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                      num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size,
                                      margin, margin_l, margin_weight, margin_beta)

    elif alg == 'dqn_hier_max':
        agent = DQNRotMaxInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                num_primitives, sl, buffer_type =='per', num_rotations, half_rotation, patch_size)
    elif alg == 'margin_hier_max':
        agent = DQNRotMaxInHandMargin(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                      num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size,
                                      margin, margin_l, margin_weight, margin_beta)
    elif alg == 'dqn_hier_max_shared':
        if action_sequence == 'xyzrp':
            agent = DQNRotZMaxSharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr,
                                           gamma,
                                           num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation,
                                           patch_size, num_zs, min_z, max_z)
        elif action_sequence == 'xyrrp':
            agent = DQNRot2MaxSharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr,
                                           gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                           half_rotation,
                                           patch_size, num_rx, min_rx, max_rx)
        elif action_sequence == 'xyzrrp':
            agent = DQNZRRMaxSharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr,
                                           gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                           half_rotation, patch_size, num_rx, min_rx, max_rx, num_zs, min_z, max_z)
        else:
            agent = DQNRotMaxSharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                          num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation,
                                          patch_size)
    elif alg == 'dqn_hier_max_shared_deictic':
        if action_sequence.find('z') >= 0:
            agent = DQNRotZDeicticMaxSharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution,
                                                  device, lr, gamma, num_primitives, sl, buffer_type == 'per',
                                                  num_rotations, half_rotation, patch_size, num_zs, min_z, max_z)
        elif action_sequence.count('r') == 2:
            agent = DQNRot2DeicticMaxSharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr,
                                                  gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                                  half_rotation, patch_size, num_rx, min_rx, max_rx)
        else:
            agent = DQNRotDeicticMaxSharedInHand(fcn, cnn, action_space, workspace, heightmap_resolution,
                                                 device, lr, gamma, num_primitives, sl, buffer_type == 'per',
                                                 num_rotations, half_rotation, patch_size)

    elif alg == 'margin_hier_max_shared':
        if action_sequence == 'xyzrp':
            agent = DQNRotZMaxSharedInHandMargin(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr,
                                                 gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                                 half_rotation, patch_size, margin, margin_l, margin_weight,
                                                 margin_beta,
                                                 num_zs, min_z, max_z)
        elif action_sequence == 'xyrrp':
            agent = DQNRot2MaxSharedInHandMargin(fcn, cnn, action_space, workspace, heightmap_resolution, device,
                                                 lr, gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                                 half_rotation, patch_size, margin, margin_l, margin_weight,
                                                 margin_beta,
                                                 num_rx, min_rx, max_rx)
        elif action_sequence == 'xyzrrp':
            agent = DQNZRRMaxSharedInHandMargin(fcn, cnn, action_space, workspace, heightmap_resolution, device,
                                                 lr, gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                                 half_rotation, patch_size, margin, margin_l, margin_weight,
                                                 margin_beta,
                                                 num_rx, min_rx, max_rx, num_zs, min_z, max_z)
        else:
            agent = DQNRotMaxSharedInHandMargin(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr,
                                                gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                                half_rotation, patch_size, margin, margin_l, margin_weight, margin_beta)
    elif alg == 'margin_hier_max_shared_deictic':
        if action_sequence.find('z') >= 0:
            cnn = CNNResShared96((6, 24, 24), num_primitives).to(device)
            agent = DQNRotZDeicticMaxSharedInHandMargin(fcn, cnn, action_space, workspace, heightmap_resolution, device,
                                                        lr,
                                                        gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                                        half_rotation, patch_size, margin, margin_l, margin_weight,
                                                        margin_beta, num_zs, min_z, max_z)
        else:
            cnn = CNNResShared96((2, 24, 24), num_primitives).to(device)
            agent = DQNRotDeicticMaxSharedInHandMargin(fcn, cnn, action_space, workspace, heightmap_resolution, device,
                                                       lr,
                                                       gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                                       half_rotation, patch_size, margin, margin_l, margin_weight,
                                                       margin_beta)

    elif alg == 'margin_p_hier_max_shared':
        if model == 'resucat96':
            fcn_p = ResUCat96Sig(1, 1, domain_shape=(1, diag_length, diag_length)).to(device)
            cnn_p = CNNResSig(image_shape=(1, 24, 24), n_outputs=1 * num_rotations).to(device)
        else:
            raise NotImplementedError
        agent = DQNPRotMaxSharedInHandMargin(fcn, cnn, fcn_p, cnn_p, action_space, workspace, heightmap_resolution,
                                             device, lr,
                                             gamma, num_primitives, sl, buffer_type == 'per', num_rotations,
                                             half_rotation, patch_size, margin, margin_l, margin_weight, margin_beta)

    elif alg == 'dqn_hier_max_shared_3l':
        if action_sequence == 'xyzrrp':
            q2_output_size = num_primitives * num_rotations
            q3_output_size = num_primitives * num_zs * num_rx
            q2 = CNNResShared(q2_input_shape, q2_output_size).to(device)
            q3 = CNNResShared(q2_input_shape, q3_output_size).to(device)
            agent = DQN3LMaxSharedRzRxZ(fcn, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                        num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size,
                                        num_rx, min_rx, max_rx, num_zs, min_z, max_z)
        elif action_sequence == 'xyzrp':
            q2_output_size = num_primitives * num_rotations
            q3_output_size = num_primitives * num_zs
            q2 = CNNResShared(q2_input_shape, q2_output_size).to(device)
            q3 = CNNResShared(q2_input_shape, q3_output_size).to(device)

            agent = DQN3LMaxShared(fcn, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                   num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation,
                                   patch_size, num_zs, min_z, max_z)
        elif action_sequence == 'xyzrrrp':
            q2_output_size = num_primitives * num_rotations * num_zs
            q3_output_size = num_primitives * num_rx * num_rx
            q2_input_shape = (patch_channel + 1, patch_size, patch_size)
            q3_input_shape = (patch_channel + 3, patch_size, patch_size)
            q2 = CNNResShared(q2_input_shape, q2_output_size).to(device)
            q3 = CNNResShared(q3_input_shape, q3_output_size).to(device)
            if half_rotation:
                rz_range = (0, (num_rotations - 1) * np.pi / num_rotations)
            else:
                rz_range = (0, (num_rotations - 1) * 2 * np.pi / num_rotations)
            agent = DQN3L6DMaxShared(fcn, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                     num_primitives,
                                     patch_size, sl, buffer_type == 'per', num_rotations, rz_range, num_rx,
                                     (min_rx, max_rx),
                                     num_rx, (min_rx, max_rx), num_zs, (min_z, max_z))
    elif alg == 'dqn_hier_max_shared_5l' or alg == 'dqn_5l':
        if action_sequence != 'xyzrrrp':
            raise NotImplementedError
        q3_output_size = num_primitives * num_zs
        q2_output_size = num_primitives * num_rotations
        q4_output_size = num_primitives * num_rx
        q5_output_size = num_primitives * num_rx
        q2_input_shape = (patch_channel + 1, patch_size, patch_size)
        q3_input_shape = (patch_channel + 1, patch_size, patch_size)
        q4_input_shape = (patch_channel + 3, patch_size, patch_size)
        q5_input_shape = (patch_channel + 3, patch_size, patch_size)
        q2 = CNNResShared(q2_input_shape, q2_output_size).to(device)
        q3 = CNNResShared(q3_input_shape, q3_output_size).to(device)
        q4 = CNNResShared(q4_input_shape, q4_output_size).to(device)
        q5 = CNNResShared(q5_input_shape, q5_output_size).to(device)

        if half_rotation:
            rz_range = (0, (num_rotations - 1) * np.pi / num_rotations)
        else:
            rz_range = (0, (num_rotations - 1) * 2 * np.pi / num_rotations)

        agent = DQN5LMaxShared(fcn, q2, q3, q4, q5, action_space, workspace, heightmap_resolution, device, lr, gamma,
                               num_primitives,
                               patch_size, sl, buffer_type == 'per', num_rotations, rz_range, num_rx, (min_rx, max_rx),
                               num_rx, (min_rx, max_rx), num_zs, (min_z, max_z))

    elif alg == 'margin_hier_max_shared_3l':
        if action_sequence == 'xyzrrp':
            q2_output_size = num_primitives * num_rotations
            q3_output_size = num_primitives * num_zs * num_rx
            q2 = CNNResShared(q2_input_shape, q2_output_size).to(device)
            q3 = CNNResShared(q2_input_shape, q3_output_size).to(device)
            agent = Margin3LMaxSharedRzRxZ(fcn, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                           num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size,
                                           margin, margin_l, margin_weight, margin_beta,
                                           num_rx, min_rx, max_rx, num_zs, min_z, max_z)

        elif action_sequence == 'xyzrp':
            q2_output_size = num_primitives * num_rotations
            q3_output_size = num_primitives * num_zs
            q2 = CNNResShared(q2_input_shape, q2_output_size).to(device)
            q3 = CNNResShared(q2_input_shape, q3_output_size).to(device)

            agent = Margin3LMaxShared(fcn, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                      num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation,
                                      patch_size, margin, margin_l, margin_weight, margin_beta, num_zs, min_z, max_z)
        else:
            raise NotImplementedError
    elif alg == 'dagger_hier_shared_3l':
        if action_sequence == 'xyzrrp':
            q2_output_size = num_primitives * num_rotations
            q3_output_size = num_primitives * num_zs * num_rx
            q2 = CNNResShared(q2_input_shape, q2_output_size).to(device)
            q3 = CNNResShared(q2_input_shape, q3_output_size).to(device)
            agent = Policy3LMaxSharedRzRxZ(fcn, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                           num_primitives, sl, buffer_type == 'per', num_rotations, half_rotation, patch_size,
                                           num_rx, min_rx, max_rx, num_zs, min_z, max_z)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    if model.find('96') > 0:
        agent.padding = 96
    return agent

def addPerlinNoiseToObs(obs, c):
    for i in range(1):
        obs[i, 0] += (c * rand_perlin_2d((90, 90), (
            int(np.random.choice([1, 2, 3, 5, 6], 1)[0]),
            int(np.random.choice([1, 2, 3, 5, 6], 1)[0]))) + c)

def addPerlinNoiseToInHand(in_hand, c):
    for i in range(1):
        in_hand[i, 0] += (c * rand_perlin_2d((24, 24), (
            int(np.random.choice([1, 2], 1)[0]),
            int(np.random.choice([1, 2], 1)[0]))) + c)


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

    global model, alg, action_sequence, in_hand_mode, workspace, max_z, min_z
    ws_center = [-0.524, -0.0098, -0.092]
    workspace = np.asarray([[ws_center[0]-0.15, ws_center[0]+0.15],
                            [ws_center[1]-0.15, ws_center[1]+0.15],
                            [ws_center[2], 0.50]])
    max_z = 0.12
    min_z = 0.02
    # model = 'dfpyramid'
    model = 'resucat'
    alg = 'dqn_hier_max_shared_5l'
    action_sequence = 'xyzrrrp'
    in_hand_mode = 'proj'
    agent = creatAgent()
    agent.q3_input = 'proj'
    # pre = '/home/dian/Downloads/dqfd/snapshot_house_building_3'
    # pre = '/home/dian/Downloads/dqfd/snapshot_house_building_2'
    # pre = '/home/dian/Downloads/dqfd/snapshot_house_building_1'
    # pre = '/home/dian/Downloads/dqfd/snapshot_block_stacking'

    # pre = '/home/dian/Downloads/4h1_7_sdqfd_perlin/ran_tilt_4h1_7_sdqfd_3l_qt_0628_4_12420761_2/models/snapshot_tilt_house_building_1'
    # pre = '/home/dian/Downloads/4h1_9_sdqfd_perlin/models/snapshot_tilt_house_building_1'
    # pre = '/home/dian/Downloads/h3_9_sdqfd_perlin/models/snapshot_tilt_house_building_3'
    # pre = '/home/dian/Downloads/4h1_12_sdqfd_perlin/models/snapshot_tilt_house_building_1'
    # pre = '/home/dian/Downloads/h3_13_sdqfd_perlin/models/snapshot_tilt_house_building_3'
    # pre = '/home/dian/Downloads/h4_13_sdqfd_perlin/models/snapshot_tilt_house_building_4'
    # pre = '/home/dian/Downloads/4h1_15_sdqfd_perlin/models/snapshot_tilt_house_building_1'
    # pre = '/home/dian/Downloads/h3_15_sdqfd_perlin/models/snapshot_tilt_house_building_3'
    # pre = '/home/dian/Downloads/imh6_2_sdqfd_perlin/models/snapshot_tilt_improvise_house_building_6'
    # pre = '/home/dian/Downloads/imh6_3/models/snapshot_tilt_improvise_house_building_6'
    # pre = '/home/dian/Downloads/imh2_2_sdqfd_perlin/models/snapshot_tilt_improvise_house_building_2'
    # pre = '/home/dian/Downloads/imh2_4_tmp/models/snapshot_tilt_improvise_house_building_2'
    # pre = '/home/dian/Downloads/imh6_4_tmp/models/snapshot_tilt_improvise_house_building_6'

    # pre = '/home/dian/Downloads/h3_6d/snapshot_tilt_house_building_3'
    # pre = '/home/dian/Downloads/h4_6d/3/snapshot_tilt_house_building_4'

    # pre = '/home/dian/Downloads/4h1_6d/1/models/snapshot_tilt_house_building_1'
    # pre = '/home/dian/Downloads/h3_6d/1/models/snapshot_tilt_house_building_3'
    # pre = '/home/dian/Downloads/h4_6d_3/4/models/snapshot_tilt_house_building_4'
    pre = '/home/dian/Downloads/h4_6d_5l/ran_tilt_6d_h4_sdqfd_5l_4_0928_8_14614980_2/models/snapshot_tilt_house_building_4'

    agent.loadModel(pre)
    agent.eval()
    # agent.initHis(1)

    env = Env(ws_center=ws_center, action_sequence=action_sequence)
    rospy.sleep(2)
    env.ur5.moveToHome()
    j=0
    action = None
    while True:
        j += 1
        obs, in_hand = env.getObs(action)
        # addPerlinNoiseToObs(obs, 0.005)
        # addPerlinNoiseToInHand(in_hand, 0.005)
        state = torch.tensor([env.ur5.holding_state], dtype=torch.float32)
        q_map, action_idx, action = agent.getEGreedyActions(state, in_hand, obs, 0, 0)
        plt.imshow(obs[0, 0])
        plt.colorbar()
        plt.axis('off')
        plt.scatter(action_idx[0, 1], action_idx[0, 0], c='r')
        plt.savefig(os.path.join('/home/dian/Documents/obs', '{}_obs.png'.format(j)))
        plt.show()

        # plotQMaps(q_map, '/home/dian/Documents/qmap', j)
        # plotQMaps(q_map)
        plt.imshow(q_map[0])
        plt.scatter(action_idx[0, 1], action_idx[0, 0], c='r')
        plt.axis('off')
        plt.savefig(os.path.join('/home/dian/Documents/qmap', '{}_qmap.png'.format(j)))
        plt.show()
        # pixels = action_idx[:, :2]
        # patch = agent.getImgPatch(obs, pixels)
        # action = torch.cat(action[0])
        action = [*list(map(lambda x: x.item(), action[0])), state.item()]
        if action[2] <= min_z and state.item() == 0:
            action[2] -= 0.01
        z_offset_threshold = -0.04 if state.item() == 0 else 0
        safe_region_extent = 5
        local_region = obs[0, 0,
                       int(max(action_idx[0, 0] - safe_region_extent, 0)):int(min(action_idx[0, 0] + safe_region_extent, heightmap_size)),
                       int(max(action_idx[0, 1] - safe_region_extent, 0)):int(min(action_idx[0, 1] + safe_region_extent, heightmap_size))]
        safe_z_pos = local_region.max() + z_offset_threshold
        if action[2] < safe_z_pos:
            print('z {} too small, clipping to {}'.format(action[2], safe_z_pos))
            action[2] = safe_z_pos.item()

        action[2] += ws_center[2]
        env.step(action)
        
        # agent.updateHis(patch, action[:, 2], torch.tensor([env.ur5.holding_state], dtype=torch.float32), None, torch.zeros(1))
