from copy import deepcopy
from utils import torch_utils

import numpy as np

import torch
import torch.nn.functional as F
from agents.hierarchy.dqn_rot_max import DQNRotMax

class DQNRotMaxShared(DQNRotMax):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                num_primitives = 1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        super().__init__(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        self.expert_sl = False

    def forwardFCN(self, states, obs, target_net=False, to_cpu=False):
        fcn = self.fcn if not target_net else self.target_fcn
        obs = obs.to(self.device)
        padding_width = int((self.padding - obs.size(2)) / 2)
        # pad obs
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        q_value_maps, obs_encoding = fcn(obs)[torch.arange(0, states.size(0)), states.long()]
        q_value_maps = q_value_maps[:, :, padding_width: -padding_width, padding_width: -padding_width]

        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps, obs_encoding

    def forwardPhiNet(self, states, obs, pixels, obs_encoding, target_net=False, to_cpu=False):
        # obs_encoding = obs_encoding.detach()
        patch = self.getImgPatch(obs.to(self.device), pixels.to(self.device))
        phi_net = self.phi_net if not target_net else self.target_phi
        phi_output = phi_net(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[torch.arange(0, states.size(0)),
                                                                                          states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

    def getEGreedyActions(self, states, obs, eps, coef=0.01):
        with torch.no_grad():
            q_value_maps, obs_encoding = self.forwardFCN(states, obs, to_cpu=True)
        # q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        pixels = torch_utils.argmax2d(q_value_maps).long()
        with torch.no_grad():
            phi_output = self.forwardPhiNet(states, obs, pixels, obs_encoding, to_cpu=True)
        phi = torch.argmax(phi_output, 1)

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps

        if type(obs) is tuple:
            hm, ih = obs
        else:
            hm = obs
        for i, m in enumerate(rand_mask):
            if m:
                pixel_candidates = torch.nonzero(hm[i, 0]>0.01)
                rand_pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))]
                pixels[i] = rand_pixel

        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.phi_size)
        phi[rand_mask] = rand_phi.long()

        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        phi_idx, phi_a = self.decodePhi(phi)
        actions = torch.cat((x, y, phi_a), dim=1)
        action_idx = torch.cat((pixels, phi_idx), dim=1)

        return q_value_maps, action_idx, actions

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]
        phis = action_idx[:, 2:]

        if self.sl:
            q_target = self.gamma ** step_lefts
        elif self.expert_sl:
            q_target_sl = self.gamma ** step_lefts
            with torch.no_grad():
                q1_map_prime, obs_prime_encoding = self.forwardFCN(next_states, next_obs, target_net=True)
                x_star = torch_utils.argmax2d(q1_map_prime)
                q2_prime = self.forwardPhiNet(next_states, next_obs, x_star, obs_prime_encoding, target_net=True)
                q2 = q2_prime.max(1)[0]
                q_prime = q2
                q_target_td = rewards + self.gamma * q_prime * non_final_masks

            q_target = q_target_td
            q_target[is_experts] = q_target_sl[is_experts]

        else:
            with torch.no_grad():
                q1_map_prime, obs_prime_encoding = self.forwardFCN(next_states, next_obs, target_net=True)
                x_star = torch_utils.argmax2d(q1_map_prime)
                q2_prime = self.forwardPhiNet(next_states, next_obs, x_star, obs_prime_encoding, target_net=True)
                q2 = q2_prime.max(1)[0]
                q_prime = q2
                q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q1_output, obs_encoding = self.forwardFCN(states, obs)
        q1_pred = q1_output[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]
        q2_output = self.forwardPhiNet(states, obs, pixel, obs_encoding)
        q2_pred = q2_output[torch.arange(batch_size), phis[:, 0]]


        self.loss_calc_dict['q1_output'] = q1_output
        self.loss_calc_dict['q2_output'] = q2_output

        # with torch.no_grad():
        #     q2_target_net_output = self.forwardPhiNet(states, obs, pixel, obs_encoding, target_net=True).max(1)[0]
        #
        # q1_target = torch.stack((q2_target_net_output, q_target), dim=1).max(1)[0]

        q1_td_loss = F.smooth_l1_loss(q1_pred, q_target)
        q2_td_loss = F.smooth_l1_loss(q2_pred, q_target)
        td_loss = q1_td_loss + q2_td_loss

        with torch.no_grad():
            td_error = torch.abs(q2_pred - q_target)

        return td_loss, td_error
