import torch
import torch.nn.functional as F
import numpy as np
import time

from scipy.ndimage import median_filter

from agents.hierarchy.margin_agent import DQNRotMaxSharedInHand, DQNRotMaxSharedInHandMargin
from agents.hierarchy.rot_z_agent import DQNRotZMaxSharedInHand, DQNRotZMaxSharedInHandMargin
from agents.hierarchy.rot_2_agent import DQNRot2MaxSharedInHand, DQNRot2MaxSharedInHandMargin

class DeicticRotMaxSharedAgent:
    def __init__(self):
        pass

    def getRotatedPatch(self, patch, batch_size):
        affine_mats = []
        for rot in self.rotations:
            affine_mat = np.asarray([[np.cos(rot), np.sin(rot), 0],
                                     [-np.sin(rot), np.cos(rot), 0]])
            affine_mat.shape = (2, 3, 1)
            affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float().to(self.device)
            affine_mats.append(affine_mat)
        affine_mats = torch.cat(affine_mats)
        affine_mats = affine_mats.repeat(batch_size, 1, 1)

        flow_grid = F.affine_grid(affine_mats, patch.size())
        rotated_patch = F.grid_sample(patch, flow_grid, mode='bilinear')
        return rotated_patch

    def forwardPhiNet(self, states, obs, pixels, obs_encoding, target_net=False, to_cpu=False):
        obs, in_hand = obs
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        phi_net = self.phi_net if not target_net else self.target_phi
        patch = self.getImgPatch(obs, pixels.to(self.device))
        patch = patch.unsqueeze(1).repeat(1, self.num_rotations, 1, 1, 1)
        in_hand = in_hand.unsqueeze(1).repeat(1, self.num_rotations, 1, 1, 1)
        obs_encoding = obs_encoding.unsqueeze(1).repeat(1, self.num_rotations, 1, 1, 1)
        patch = patch.reshape(patch.size(0) * patch.size(1), patch.size(2), patch.size(3), patch.size(4))
        in_hand = in_hand.reshape(in_hand.size(0) * in_hand.size(1), in_hand.size(2), in_hand.size(3), in_hand.size(4))
        obs_encoding = obs_encoding.reshape(obs_encoding.size(0) * obs_encoding.size(1), obs_encoding.size(2), obs_encoding.size(3), obs_encoding.size(4))

        rotated_patch = self.getRotatedPatch(patch, states.size(0))

        phi_input = torch.cat((rotated_patch, in_hand), dim=1)
        predictions = phi_net(obs_encoding, phi_input)
        predictions = predictions.reshape((states.shape[0], -1, predictions.size(1)))
        predictions = predictions.permute(0, 2, 1)[torch.arange(0, states.size(0)), states.long()]

        if to_cpu:
            predictions = predictions.cpu()
        return predictions

class DQNRotDeicticMaxSharedInHandMargin(DeicticRotMaxSharedAgent, DQNRotMaxSharedInHandMargin):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0):
        DQNRotMaxSharedInHandMargin.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size, margin, margin_l,
                                             margin_weight, softmax_beta)
        DeicticRotMaxSharedAgent.__init__(self)

class DQNRotDeicticMaxSharedInHand(DeicticRotMaxSharedAgent, DQNRotMaxSharedInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        DQNRotMaxSharedInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        DeicticRotMaxSharedAgent.__init__(self)

class DeicticRotZMaxSharedAgent(DeicticRotMaxSharedAgent):
    def __init__(self):
        super().__init__()

    def forwardPhiNet(self, states, obs, pixels, obs_encoding, target_net=False, to_cpu=False):
        obs, in_hand = obs
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        phi_net = self.phi_net if not target_net else self.target_phi
        patch = self.getImgPatch(obs, pixels.to(self.device))
        patch = patch.unsqueeze(1).repeat(1, self.num_rotations, 1, 1, 1)
        patch = patch.reshape(patch.size(0) * patch.size(1), patch.size(2), patch.size(3), patch.size(4))
        patch = self.getRotatedPatch(patch, states.size(0))
        patch = patch.reshape(states.size(0), self.num_rotations, patch.size(1), patch.size(2), patch.size(3))
        patch = patch.unsqueeze(2).repeat(1, 1, self.num_zs, self.patch_size, 1, 1)
        patch = patch.reshape(patch.size(0)*patch.size(1)*patch.size(2), patch.size(3), patch.size(4), patch.size(5))

        zs = self.zs.unsqueeze(1).repeat(1, self.patch_size).to(self.device)
        z_offset = torch.tensor([(self.patch_size/2-i)*self.heightmap_resolution for i in range(self.patch_size)]).to(self.device)
        z_offset = z_offset.unsqueeze(0).repeat(self.num_zs, 1)
        zs = zs + z_offset
        zs = zs.reshape(1, 1, zs.size(0), zs.size(1), 1, 1)
        zs = zs.repeat(states.size(0), self.num_rotations, 1, 1, self.patch_size, self.patch_size)
        zs = zs.reshape(states.size(0)*self.num_rotations*self.num_zs, self.patch_size, self.patch_size, self.patch_size)
        zs[zs < self.heightmap_resolution] = 100

        occupancy = patch > zs
        proj = torch.stack((occupancy.sum(1), occupancy.sum(2), occupancy.sum(3)), dim=1).float()

        in_hand = in_hand.reshape(in_hand.size(0), 1, 1, in_hand.size(1), in_hand.size(2), in_hand.size(3))
        in_hand = in_hand.repeat(1, self.num_rotations, self.num_zs, 1, 1, 1)
        in_hand = in_hand.reshape(states.size(0)*self.num_rotations*self.num_zs, in_hand.size(-3), in_hand.size(-2), in_hand.size(-1))

        obs_encoding = obs_encoding.unsqueeze(1).unsqueeze(1).repeat(1, self.num_rotations, self.num_zs, 1, 1, 1)
        obs_encoding = obs_encoding.reshape(states.size(0)*self.num_rotations*self.num_zs, obs_encoding.size(-3), obs_encoding.size(-2), obs_encoding.size(-1))

        phi_input = torch.cat((proj, in_hand), dim=1)
        predictions = phi_net(obs_encoding, phi_input)

        predictions = predictions.reshape((states.shape[0], -1, predictions.size(1)))
        predictions = predictions.permute(0, 2, 1)[torch.arange(0, states.size(0)), states.long()]

        if to_cpu:
            predictions = predictions.cpu()
        return predictions


class DQNRotZDeicticMaxSharedInHandMargin(DeicticRotZMaxSharedAgent, DQNRotZMaxSharedInHandMargin):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0, num_zs=16, min_z=0.02, max_z=0.17):
        DQNRotZMaxSharedInHandMargin.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                              num_primitives, sl, per, num_rotations, half_rotation, patch_size, margin, margin_l,
                                              margin_weight, softmax_beta, num_zs, min_z, max_z)
        DeicticRotZMaxSharedAgent.__init__(self)

class DQNRotZDeicticMaxSharedInHand(DeicticRotZMaxSharedAgent, DQNRotZMaxSharedInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_zs=16, min_z=0.02, max_z=0.17):
        DQNRotZMaxSharedInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                              num_primitives, sl, per, num_rotations, half_rotation, patch_size, num_zs, min_z, max_z)
        DeicticRotZMaxSharedAgent.__init__(self)

class DeicticRot2MaxSharedAgent(DeicticRotMaxSharedAgent):
    def __init__(self):
        super().__init__()
        self.map = None
        self.initTMap()

    def initTMap(self):
        maps = []
        for i, rx in enumerate(self.rxs):
            occupancy = np.ones((24, 24, 24))
            point = np.argwhere(occupancy)
            point = point - self.patch_size / 2
            R = np.array([[np.cos(-rx), 0, np.sin(-rx)],
                          [0, 1, 0],
                          [-np.sin(-rx), 0, np.cos(-rx)]])
            rotated_point = R.dot(point.T)
            rotated_point = rotated_point + self.patch_size / 2
            rotated_point = np.round(rotated_point).astype(int)
            rotated_point = rotated_point.T.reshape(1, self.patch_size, self.patch_size, self.patch_size, 3)
            maps.append(rotated_point)
        self.map = np.concatenate(maps)




    def forwardPhiNet(self, states, obs, pixels, obs_encoding, target_net=False, to_cpu=False):
        obs, in_hand = obs
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        phi_net = self.phi_net if not target_net else self.target_phi
        ori_patch = self.getImgPatch(obs, pixels.to(self.device))
        patch = ori_patch.unsqueeze(1).repeat(1, self.num_rotations, 1, 1, 1)
        patch = patch.reshape(patch.size(0) * patch.size(1), patch.size(2), patch.size(3), patch.size(4))
        patch = self.getRotatedPatch(patch, states.size(0))
        patch = patch.reshape(states.size(0), self.num_rotations, patch.size(1), patch.size(2), patch.size(3))
        patch = patch.repeat(1, 1, self.patch_size, 1, 1)
        # patch = patch.reshape(patch.size(0)*patch.size(1), patch.size(2), patch.size(3), patch.size(4))


        zs = ori_patch[:, 0, int(self.patch_size/2)-4:int(self.patch_size/2)+4,
                             int(self.patch_size/2)-4:int(self.patch_size/2)+4].reshape(states.size(0), -1).max(1)[0]
        zs[states.bool()] += 0.01
        zs[~states.bool()] -= 0.01
        zs[zs < 0.02] = 0.02
        zs = zs.unsqueeze(1).repeat(1, self.patch_size)

        z_offset = torch.tensor([(self.patch_size/2-i)*self.heightmap_resolution for i in range(self.patch_size)]).to(self.device)
        z_offset = z_offset.unsqueeze(0).repeat(states.size(0), 1)
        zs = zs + z_offset # Bx24

        zs = zs.reshape(zs.size(0), 1, zs.size(1), 1, 1)
        zs = zs.repeat(1, self.num_rotations, 1, self.patch_size, self.patch_size)
        # zs = zs.reshape(states.size(0)*self.num_rotations, self.patch_size, self.patch_size, self.patch_size)
        zs[zs < -self.heightmap_resolution] = 100

        ori_occupancy = (patch > zs).cpu().numpy()
        ori_occupancy = ori_occupancy.reshape(ori_occupancy.shape[0] * ori_occupancy.shape[1], ori_occupancy.shape[2],
                                              ori_occupancy.shape[3], ori_occupancy.shape[4])
        point_w_d = np.argwhere(ori_occupancy)
        dimension = point_w_d[:, 0]
        point = point_w_d[:, 1:4]

        mapped_point = self.map[:, point[:, 0], point[:, 1], point[:, 2]]

        occupancies = []
        for i in range(self.num_rx):
            # R = np.array([[np.cos(-rx), 0, np.sin(-rx)],
            #               [0, 1, 0],
            #               [-np.sin(-rx), 0, np.cos(-rx)]])
            # rotated_point = R.dot(point.T)
            # rotated_point = rotated_point + self.patch_size / 2
            # rotated_point = np.round(rotated_point).astype(int)
            rotated_point = mapped_point[i].T
            d = dimension[(np.logical_and(0 < rotated_point.T, rotated_point.T < self.patch_size)).all(1)].T.astype(int)
            rotated_point = rotated_point.T[(np.logical_and(0 < rotated_point.T, rotated_point.T < self.patch_size)).all(1)].T

            occupancy = np.zeros((ori_occupancy.shape[0], self.patch_size, self.patch_size, self.patch_size))
            occupancy[d, rotated_point[0], rotated_point[1], rotated_point[2]] = 1
            for j in range(occupancy.shape[0]):
                occupancy[j] = median_filter(occupancy[j], size=2)
            # occupancy = median_filter(occupancy, size=2)
            occupancy = np.ceil(occupancy)
            occupancies.append(occupancy.reshape(occupancy.shape[0], 1, occupancy.shape[1], occupancy.shape[2], occupancy.shape[3]))

        occupancy = np.concatenate(occupancies, 1)
        occupancy = occupancy.reshape(occupancy.shape[0]*occupancy.shape[1], self.patch_size, self.patch_size, self.patch_size)
        occupancy = torch.from_numpy(occupancy).to(self.device)
        proj = torch.stack((occupancy.sum(1), occupancy.sum(2), occupancy.sum(3)), dim=1).float()

        in_hand = in_hand.reshape(in_hand.size(0), 1, 1, in_hand.size(1), in_hand.size(2), in_hand.size(3))
        in_hand = in_hand.repeat(1, self.num_rotations, self.num_rx, 1, 1, 1)
        in_hand = in_hand.reshape(states.size(0)*self.num_rotations*self.num_rx, in_hand.size(-3), in_hand.size(-2), in_hand.size(-1))

        obs_encoding = obs_encoding.unsqueeze(1).unsqueeze(1).repeat(1, self.num_rotations, self.num_rx, 1, 1, 1)
        obs_encoding = obs_encoding.reshape(states.size(0)*self.num_rotations*self.num_rx, obs_encoding.size(-3), obs_encoding.size(-2), obs_encoding.size(-1))

        phi_input = torch.cat((proj, in_hand), dim=1)
        predictions = phi_net(obs_encoding, phi_input)

        predictions = predictions.reshape((states.shape[0], -1, predictions.size(1)))
        predictions = predictions.permute(0, 2, 1)[torch.arange(0, states.size(0)), states.long()]

        if to_cpu:
            predictions = predictions.cpu()
        return predictions

class DQNRot2DeicticMaxSharedInHand(DeicticRot2MaxSharedAgent, DQNRot2MaxSharedInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_rx=8, min_rx=-np.pi/4, max_rx=np.pi/4):
        DQNRot2MaxSharedInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                        num_primitives, sl, per, num_rotations, half_rotation, patch_size, num_rx, min_rx, max_rx)
        DeicticRot2MaxSharedAgent.__init__(self)