from copy import deepcopy
from utils import torch_utils

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from agents.hierarchy.dqn_3l_max_shared import DQN3LMaxShared

class DQN3LMaxSharedRzRxZ(DQN3LMaxShared):
    def __init__(self, q1, q2, q3, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_rx=8, min_rx=-np.pi / 4, max_rx=np.pi / 4, num_zs=16, min_z=0.02, max_z=0.17):
        DQN3LMaxShared.__init__(self, q1, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma,
                 num_primitives, sl, per, num_rotations, half_rotation, patch_size, num_zs, min_z, max_z)
        self.num_rx = num_rx
        self.rx_range = [min_rx, max_rx]
        self.rxs = torch.from_numpy(np.linspace(self.rx_range[0], self.rx_range[1], self.num_rx)).float()

        self.a3_size = num_zs * num_rx

    def decodeA3(self, a3):
        rx_id = a3 / self.num_zs
        z_id = a3 % self.num_zs
        rx = self.rxs[rx_id].reshape(a3.size(0), 1)
        z = self.zs[z_id].reshape(a3.size(0), 1)
        return a3.unsqueeze(1), torch.cat((rx, z), dim=1)

    def decodeActions(self, pixels, a2_id, a3_id):
        a2_id, a2 = self.decodeA2(a2_id)
        a3_id, a3 = self.decodeA3(a3_id)
        rz_id = a2_id
        rx_z_id = a3_id
        rz = a2
        rx = a3[:, 0:1]
        z = a3[:, 1:2]
        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(pixels.size(0), 1)
        y = ((90 - pixels[:, 0].float()) * self.heightmap_resolution + self.workspace[1][0]).reshape(pixels.size(0), 1)
        actions = torch.cat((x, y, z, rz, rx), dim=1)
        action_idx = torch.cat((pixels, rz_id * self.a3_size + rx_z_id), dim=1)
        return action_idx, actions

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        z = plan[:, 2:3]
        rz = plan[:, 3:4]
        rx = plan[:, 4:5]
        states = plan[:, 5:6]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size - 1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size - 1)

        if not self.half_rotation:
            rz_id = (rz.expand(-1, self.num_rotations + 1) - torch.cat(
                (self.rotations, torch.tensor([np.pi * 2])))).abs().argmin(1)
            rz_id[rz_id==self.num_rotations] = 0
        else:
            rz_id = (rz.expand(-1, self.num_rotations) - self.rotations).abs().argmin(1)

        rx_id = (rx.expand(-1, self.num_rx) - self.rxs).abs().argmin(1)
        z_id = (z.expand(-1, self.num_zs) - self.zs).abs().argmin(1)

        action_idx, actions = self.decodeActions(torch.cat((pixel_x, pixel_y), dim=1), rz_id, rx_id * self.num_zs + z_id)
        return action_idx, actions

