import numpy as np

from abc import abstractmethod

import torch
import torch.nn.functional as F
import utils.torch_utils as torch_utils

from agents.base_agent import BaseAgent

class HierarchyAgent(BaseAgent):
    def __init__(self, action_space, workspace, heightmap_resolution, device, num_rotations=8, half_rotation=False,
                 num_heights=10, height_range=(1, 0.1), lr=1e-4, patch_size=64):
        super(HierarchyAgent, self).__init__(action_space, workspace, heightmap_resolution, device, num_rotations,
                                             half_rotation, num_heights, height_range, lr)
        self.patch_size = patch_size

    @abstractmethod
    def getQValueMap(self, states, obs):
        raise NotImplementedError('Sub-agents should implement this method')

    def getEGreedyXYs(self, states, obs, eps):
        # with torch.no_grad():
        #     q_value_maps = self.fcn(obs.to(self.device)).cpu()
        q_value_maps = self.getQValueMap(states, obs)
        q_value_maps += torch.randn_like(q_value_maps) * eps

        pixels = torch_utils.argmax2d(q_value_maps[torch.arange(0, states.size(0)), states.long()]).cpu().long()
        xys = self.getXYFromPixels(states, pixels)
        return q_value_maps, pixels, xys

    def getXYFromPixels(self, states, pixels):
        x = pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]
        y = pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]

        return torch.stack((x, y), dim=-1)

    def getImgPatch(self, obs, center_pixel):
        batch_size = obs.size(0)
        img_size = obs.size(2)
        transition = (center_pixel - obs.size(2) / 2).float().flip(1)
        transition_scaled = transition / obs.size(2) * 2

        affine_mat = torch.eye(2).unsqueeze(0).expand(batch_size, -1, -1).float()
        if obs.is_cuda:
            affine_mat = affine_mat.to(self.device)
        affine_mat = torch.cat((affine_mat, transition_scaled.unsqueeze(2).float()), dim=2)
        flow_grid = F.affine_grid(affine_mat, obs.size())
        transformed = F.grid_sample(obs, flow_grid, mode='nearest', padding_mode='zeros')
        patch = transformed[:, :,
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2),
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2)]
        return patch

    def coordToPixel(self, coord):
        yx = ((coord - torch.tensor([self.workspace[0][0], self.workspace[1][0]])) / self.heightmap_resolution).long()
        y = yx[:, 0]
        x = yx[:, 1]
        return torch.stack((x, y), 1)

    def _loadBatchToDevice(self, batch):
        images = []
        xys = []
        labels = []
        positives = []
        for d in batch:
            images.append(d.image)
            xys.append(d.xy)
            labels.append(d.labels)
            positives.append(d.positive)
        image_tensor = torch.tensor(images).unsqueeze(1).to(self.device)
        xy_tensor = torch.tensor(xys).to(self.device)
        label_tensor = torch.tensor(labels).to(self.device)
        positive_tensor = torch.tensor(positives).to(self.device)

        return image_tensor, xy_tensor, label_tensor, positive_tensor
