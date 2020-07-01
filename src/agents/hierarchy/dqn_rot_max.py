from copy import deepcopy
from utils import torch_utils

import numpy as np

import torch
import torch.nn.functional as F
from agents.hierarchy.dqn_rot_base import DQNRotBase

class DQNRotMax(DQNRotBase):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                num_primitives = 1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):

        super().__init__(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives, sl, per, num_rotations, half_rotation, patch_size)

    def forwardPhiNet(self, states, obs, pixels, target_net=False, to_cpu=False):
        patch = self.getImgPatch(obs.to(self.device), pixels.to(self.device))
        phi_net = self.phi_net if not target_net else self.target_phi
        phi_output = phi_net(obs, patch).reshape(states.size(0), self.num_primitives, -1)[torch.arange(0, states.size(0)),
                                                                                          states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

    def expandObs(self, obs, n):
        expanded_obs = obs.expand(n, -1, -1, -1, -1)
        expanded_obs = expanded_obs.reshape(expanded_obs.size(0)*expanded_obs.size(1), expanded_obs.size(2), expanded_obs.size(3), expanded_obs.size(4))
        return expanded_obs

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]
        phis = action_idx[:, 2:]

        if self.sl:
            q_target = self.gamma ** step_lefts
        else:
            with torch.no_grad():
                q1_map_prime = self.forwardFCN(next_states, next_obs, target_net=True)
                x_star = torch_utils.argmax2d(q1_map_prime)
                q2_prime = self.forwardPhiNet(next_states, next_obs, x_star, target_net=True)
                q2 = q2_prime.max(1)[0]
                q_prime = q2
                q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q2_output = self.forwardPhiNet(states, obs, pixel)
        q2_pred = q2_output[torch.arange(batch_size), phis[:, 0]]
        q1_output = self.forwardFCN(states, obs)
        q1_pred = q1_output[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]

        self.loss_calc_dict['q1_output'] = q1_output
        self.loss_calc_dict['q2_output'] = q2_output

        with torch.no_grad():
            q2_target_net_output = self.forwardPhiNet(states, obs, pixel, target_net=True).max(1)[0]

        q1_target = torch.stack((q2_target_net_output, q_target), dim=1).max(1)[0]

        q1_td_loss = F.smooth_l1_loss(q1_pred, q1_target)
        q2_td_loss = F.smooth_l1_loss(q2_pred, q_target)
        td_loss = q1_td_loss + q2_td_loss
        with torch.no_grad():
            td_error = torch.abs(q2_pred - q_target)

        return td_loss, td_error
