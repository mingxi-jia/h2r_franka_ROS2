import torch
import numpy as np
from agents.hierarchy.margin_agent import DQNRotMaxSharedInHand, DQNRotMaxSharedInHandMargin, DQNRotMulInHand
from agents.hierarchy.policy_rot_in_hand import PolicyRotInHand
from agents.hierarchy.policy_shared_in_hand import PolicySharedInHand

class ZRRAgent:
    def __init__(self, num_zs=16, min_z=0.02, max_z=0.17, num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4):
        self.num_zs = num_zs
        self.z_range = [min_z, max_z]
        self.zs = torch.from_numpy(np.linspace(self.z_range[0], self.z_range[1], self.num_zs)).float()

        self.num_rx = num_rx
        self.rx_range = [min_rx, max_rx]
        self.rxs = torch.from_numpy(np.linspace(self.rx_range[0], self.rx_range[1], self.num_rx)).float()

    def decodePhi(self, phi):
        rz_id = (phi / (self.num_rx * self.num_zs)).long().reshape(phi.size(0), 1)
        rz = self.rotations[rz_id].reshape(phi.size(0), 1)

        rx_id = ((phi % (self.num_rx * self.num_zs)) / self.num_zs).long().reshape(phi.size(0), 1)
        rx = self.rxs[rx_id].reshape(phi.size(0), 1)

        z_id = ((phi % (self.num_rx * self.num_zs)) % self.num_zs).long().reshape(phi.size(0), 1)
        z = self.zs[z_id].reshape(phi.size(0), 1)

        return phi.unsqueeze(1), torch.cat((z, rz, rx), dim=1)

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        z = plan[:, 2:3]
        rz = plan[:, 3:4]
        rx = plan[:, 4:5]
        states = plan[:, 5:6]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size-1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size-1)

        if not self.half_rotation:
            rz_id = (rz.expand(-1, self.num_rotations + 1) - torch.cat(
                (self.rotations, torch.tensor([np.pi * 2])))).abs().argmin(1).unsqueeze(1)
            rz_id[rz_id==self.num_rotations] = 0
        else:
            rz_id = (rz.expand(-1, self.num_rotations) - self.rotations).abs().argmin(1).unsqueeze(1)

        rx_id = (rx.expand(-1, self.num_rx) - self.rxs).abs().argmin(1).unsqueeze(1)
        z_id = (z.expand(-1, self.num_zs) - self.zs).abs().argmin(1).unsqueeze(1)

        x = (pixel_x.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_y.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        z = self.zs[z_id]
        rx = self.rxs[rx_id]
        rz = self.rotations[rz_id]
        actions = torch.cat((x, y, z, rz, rx), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, rz_id * self.num_rx * self.num_zs + rx_id * self.num_zs + rz_id), dim=1)
        return action_idx, actions

class DQNZRRMaxSharedInHand(ZRRAgent, DQNRotMaxSharedInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4, num_zs=16, min_z=0.02, max_z=0.17):
        DQNRotMaxSharedInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        ZRRAgent.__init__(self, num_zs, min_z, max_z, num_rx, min_rx, max_rx)

class DQNZRRMulInHand(ZRRAgent, DQNRotMulInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4, num_zs=16, min_z=0.02, max_z=0.17):
        DQNRotMulInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        ZRRAgent.__init__(self, num_zs, min_z, max_z, num_rx, min_rx, max_rx)

class DQNZRRMaxSharedInHandMargin(ZRRAgent, DQNRotMaxSharedInHandMargin):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0,
                 num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4, num_zs=16, min_z=0.02, max_z=0.17):
        DQNRotMaxSharedInHandMargin.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size, margin, margin_l,
                                             margin_weight, softmax_beta)
        ZRRAgent.__init__(self, num_zs, min_z, max_z, num_rx, min_rx, max_rx)

class PolicyZRRInHand(ZRRAgent, PolicyRotInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4, num_zs=16, min_z=0.02, max_z=0.17):
        PolicyRotInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        ZRRAgent.__init__(self, num_zs, min_z, max_z, num_rx, min_rx, max_rx)

class PolicyZRRSharedInHand(ZRRAgent, PolicySharedInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4, num_zs=16, min_z=0.02, max_z=0.17):
        PolicySharedInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                    num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        ZRRAgent.__init__(self, num_zs, min_z, max_z, num_rx, min_rx, max_rx)
