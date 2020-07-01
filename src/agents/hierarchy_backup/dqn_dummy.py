from agents.hierarchy_backup.hierarchy_separate import HierarchySeparate
from agents.models import FCN, CNN
from utils import torch_utils

import torch
import torch.nn.functional as F

class DQNDummy(HierarchySeparate):
    def __init__(self, action_space, workspace, heightmap_resolution, device, num_rotations=8, half_rotation=False,
                 num_heights=10, height_range=(1, 0.1), lr=1e-4, patch_size=64, gamma=0.99):
        super(DQNDummy, self).__init__(action_space, workspace, heightmap_resolution, device, num_rotations,
                                       half_rotation,
                                       num_heights, height_range, lr, patch_size)

        self.target_fcn = FCN().to(device)
        self.target_rot = CNN((1, patch_size, patch_size), num_rotations).to(device)
        self.target_z = CNN((1, patch_size, patch_size), num_heights).to(device)
        self.updateTarget()

        self.gamma = gamma

    def getEGreedyActions(self, states, obs, eps):
        with torch.no_grad():
            q_value_maps = self.fcn(obs.to(self.device)).cpu()
        q_value_maps += torch.randn_like(q_value_maps) * eps
        pixels = torch_utils.argmax2d(q_value_maps[torch.arange(0, states.size(0)), states.long()]).cpu().long()
        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        patch = self.getImgPatch(obs, pixels)
        with torch.no_grad():
            rot_output = self.rot_net(patch.to(self.device)).cpu()
        rot_id = torch.argmax(rot_output, 1).reshape(states.size(0), 1)
        rotation = self.rotations[rot_id].reshape(states.size(0), 1)

        with torch.no_grad():
            height_output = self.z_net(patch.to(self.device)).cpu()
        z_id = torch.argmax(height_output, 1).reshape(states.size(0), 1)
        z = self.heights[z_id].reshape(states.size(0), 1)
        actions = torch.cat((x, y, z, rotation), dim=1)
        action_idx = torch.cat((pixels, z_id, rot_id), dim=1)

        return q_value_maps, action_idx, actions

    def updateTarget(self):
        self.target_fcn.load_state_dict(self.fcn.state_dict())
        self.target_rot.load_state_dict(self.rot_net.state_dict())
        self.target_z.load_state_dict(self.z_net.state_dict())

    def update(self, batch):
        obs_t, actions, rewards, obs_tp1, non_final_masks = self._loadBatchToDevice(batch)
        batch_size = obs_t.size(0)

        num_non_final_obs = torch.sum(non_final_masks)
        next_state_values = torch.zeros(batch_size, device=self.device)
        if num_non_final_obs > 0:
            non_final_obs_tp1 = torch.stack([s for d, s in zip(non_final_masks, obs_tp1) if d])
            next_state_values[non_final_masks] = self.target_fcn(non_final_obs_tp1).view(num_non_final_obs, -1).max(1)[0]

        fcn_output = self.fcn(obs_t)
        x_value = fcn_output[torch.arange(0, batch_size), 0, actions[:, 0], actions[:, 1]]
        x_target = rewards + self.gamma * next_state_values
        fcn_loss = F.mse_loss(x_value, x_target)
        self.fcn_optimizer.zero_grad()
        fcn_loss.backward()
        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        pixel = torch.stack((actions[:, 0], actions[:, 1]), dim=1)
        patch_tensor = self.getImgPatch(obs_t.cpu(), pixel).to(self.device)
        rot_output = self.rot_net(patch_tensor)
        rot_value = rot_output[torch.arange(0, batch_size), actions[:, 3]]
        rot_target = rewards + self.gamma * next_state_values
        rot_loss = F.mse_loss(rot_value, rot_target)
        self.rot_optimizer.zero_grad()
        rot_loss.backward()
        for param in self.rot_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.rot_optimizer.step()

        z_output = self.z_net(patch_tensor)
        z_value = z_output[torch.arange(0, batch_size), actions[:, 2]]
        z_target = rewards + self.gamma * next_state_values
        z_loss = F.mse_loss(z_value, z_target)
        self.z_optimizer.zero_grad()
        z_loss.backward()
        for param in self.z_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.z_optimizer.step()

        return (fcn_loss.item(), rot_loss.item(), z_loss.item()), None

    def train(self):
        self.fcn.train()
        self.rot_net.train()
        self.z_net.train()

    def _loadBatchToDevice(self, batch):
        obs_t, actions, rewards, obs_tp1, masks = batch
        obs_t = obs_t.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).squeeze(1)
        obs_tp1 = obs_tp1.to(self.device)
        non_final_masks = (masks ^ 1).bool().to(self.device)

        return obs_t, actions, rewards, obs_tp1, non_final_masks