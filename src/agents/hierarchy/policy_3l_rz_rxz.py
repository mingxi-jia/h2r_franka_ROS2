from agents.hierarchy.dqn_3l_rz_rxz import DQN3LMaxSharedRzRxZ
from utils import torch_utils

import numpy as np
import torch
import torch.nn.functional as F

class Policy3LMaxSharedRzRxZ(DQN3LMaxSharedRzRxZ):
    def __init__(self, q1, q2, q3, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_rx=8, min_rx=-np.pi / 4, max_rx=np.pi / 4, num_zs=16, min_z=0.02, max_z=0.17):
        super().__init__(q1, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                     num_primitives, sl, per, num_rotations, half_rotation, patch_size,
                                     num_rx, min_rx, max_rx, num_zs, min_z, max_z)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        batch_size = states.size(0)
        heightmap_size = obs[0].size(2)

        pixel = action_idx[:, 0:2]
        phis = action_idx[:, 2:]
        a2 = (phis / self.a3_size).long().reshape(phis.size(0), 1)
        a3 = (phis % self.a3_size).long().reshape(phis.size(0), 1)

        q1_output, obs_encoding = self.forwardQ1(states, obs)
        q1_output = q1_output.reshape(batch_size, -1)
        q1_target = action_idx[:, 0] * heightmap_size + action_idx[:, 1]
        q1_loss = F.cross_entropy(q1_output, q1_target)

        q2_output = self.forwardQ2(states, obs, pixel, obs_encoding)
        q2_output = q2_output.reshape(batch_size, -1)
        q2_target = a2[:, 0]
        q2_loss = F.cross_entropy(q2_output, q2_target)

        q3_output = self.forwardQ3(states, obs, pixel, a2, obs_encoding)
        q3_output = q3_output.reshape(batch_size, -1)
        q3_target = a3[:, 0]
        q3_loss = F.cross_entropy(q3_output, q3_target)

        loss = q1_loss + q2_loss + q3_loss

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.q3_optimizer.zero_grad()

        loss.backward()

        for param in self.q1.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q1_optimizer.step()

        for param in self.q2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q2_optimizer.step()

        for param in self.q3.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q3_optimizer.step()

        self.loss_calc_dict = {}

        return (q1_loss.item(), q2_loss.item(), q3_loss.item()), torch.tensor(0.)