from agents.hierarchy.in_hand_agent import DQNRotMaxSharedInHand
from utils import torch_utils

import numpy as np
import torch
import torch.nn.functional as F

class PolicySharedInHand(DQNRotMaxSharedInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        DQNRotMaxSharedInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                           num_primitives, sl, per, num_rotations, half_rotation, patch_size)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        batch_size = states.size(0)
        heightmap_size = obs[0].size(2)

        pixel = action_idx[:, 0:2]
        phis = action_idx[:, 2:]

        q1_output, obs_encoding = self.forwardFCN(states, obs)
        q1_output = q1_output.reshape(batch_size, -1)
        q1_target = action_idx[:, 0] * heightmap_size + action_idx[:, 1]
        q1_loss = F.cross_entropy(q1_output, q1_target)

        q2_output = self.forwardPhiNet(states, obs, pixel, obs_encoding)
        q2_output = q2_output.reshape(batch_size, -1)
        q2_target = action_idx[:, 2]
        q2_loss = F.cross_entropy(q2_output, q2_target)

        loss = q1_loss + q2_loss

        self.fcn_optimizer.zero_grad()
        self.phi_optimizer.zero_grad()
        loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.phi_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_optimizer.step()

        self.loss_calc_dict = {}

        return (q1_loss.item(), q2_loss.item()), torch.tensor(0.)