import torch
import numpy as np
from agents.hierarchy.margin_agent import DQNRotMaxSharedInHand, DQNRotMaxSharedInHandMargin, DQNRotMulInHand
from agents.hierarchy.policy_rot_in_hand import PolicyRotInHand
from agents.hierarchy.policy_shared_in_hand import PolicySharedInHand

class Rot2Agent:
    def __init__(self, num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4):
        self.num_rx = num_rx
        self.rx_range = [min_rx, max_rx]
        self.rxs = torch.from_numpy(np.linspace(self.rx_range[0], self.rx_range[1], self.num_rx)).float()
        # self.zs = torch.tensor([(self.z_range[1]-self.z_range[0]) / self.num_zs * i for i in range(self.num_zs)]) + self.z_range[0]

    def decodePhi(self, phi):
        rz_id = (phi / self.num_rx).long().reshape(phi.size(0), 1)
        rz = self.rotations[rz_id].reshape(phi.size(0), 1)
        rx_id = (phi % self.num_rx).long().reshape(phi.size(0), 1)
        rx = self.rxs[rx_id].reshape(phi.size(0), 1)

        return phi.unsqueeze(1), torch.cat((rz, rx), dim=1)

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        rz = plan[:, 2:3]
        rx = plan[:, 3:4]
        states = plan[:, 4:5]
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

        x = (pixel_x.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_y.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        rx = self.rxs[rx_id]
        rz = self.rotations[rz_id]
        actions = torch.cat((x, y, rz, rx), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, rz_id * self.num_rx + rx_id), dim=1)
        return action_idx, actions

class DQNRot2MaxSharedInHand(Rot2Agent, DQNRotMaxSharedInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4):
        DQNRotMaxSharedInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        Rot2Agent.__init__(self, num_rx, min_rx, max_rx)

class DQNRot2MulInHand(Rot2Agent, DQNRotMulInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4):
        DQNRotMulInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        Rot2Agent.__init__(self, num_rx, min_rx, max_rx)

class DQNRot2MaxSharedInHandMargin(Rot2Agent, DQNRotMaxSharedInHandMargin):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0, num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4):
        DQNRotMaxSharedInHandMargin.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size, margin, margin_l,
                                             margin_weight, softmax_beta)
        Rot2Agent.__init__(self, num_rx, min_rx, max_rx)

class PolicyRot2InHand(Rot2Agent, PolicyRotInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24, num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4):
        PolicyRotInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        Rot2Agent.__init__(self, num_rx, min_rx, max_rx)

class PolicyRot2SharedInHand(Rot2Agent, PolicySharedInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24, num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4):
        PolicySharedInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                    num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        Rot2Agent.__init__(self, num_rx, min_rx, max_rx)