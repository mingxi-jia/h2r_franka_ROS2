import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from utils import torch_utils
from agents.fc.dqn_x_rot_in_hand import DQNXRotInHand

class PolicyRotInHand(DQNXRotInHand):
    def __init__(self, fcn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.99,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=True, patch_size=24, update_divide_factor=2):
        super().__init__(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives, sl, per,
                         num_rotations, half_rotation, patch_size)
        self.update_divide_factor = update_divide_factor


    def update(self, batch):
        states, obs, action_idx = self._loadBatchToDevice(batch)
        batch_size = states.size(0)
        heightmap_size = obs[0].size(2)
        divide_factor = self.update_divide_factor
        small_batch_size = int(batch_size/divide_factor)
        total_loss = 0
        fake_td_error = torch.tensor(0.)
        self.fcn_optimizer.zero_grad()
        for i in range(divide_factor):
            s_states = states[small_batch_size*i:small_batch_size*(i+1)]
            s_obs = (obs[0][small_batch_size*i:small_batch_size*(i+1)], obs[1][small_batch_size*i:small_batch_size*(i+1)])
            s_action_idx = action_idx[small_batch_size*i:small_batch_size*(i+1)]
            output = self.forwardFCN(s_states, s_obs)
            output = output.reshape(small_batch_size, -1)
            target = s_action_idx[:, 2] * heightmap_size * heightmap_size + s_action_idx[:, 0] * heightmap_size + s_action_idx[:, 1]
            loss = F.cross_entropy(output, target) / divide_factor
            loss.backward()
            total_loss += loss.item() / divide_factor

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        # fake_td_error = torch.abs(output - target).detach()
        return total_loss, fake_td_error

    def _loadBatchToDevice(self, batch):
        states = []
        images = []
        in_hands = []
        actions = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs[0])
            in_hands.append(d.obs[1])
            actions.append(d.action)
        states_tensor = torch.stack(states).long().to(self.device)
        image_tensor = torch.stack(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        in_hand_tensor = torch.stack(in_hands).to(self.device)
        if len(in_hand_tensor.shape) == 3:
            in_hand_tensor = in_hand_tensor.unsqueeze(1)
        action_tensor = torch.stack(actions).to(self.device)

        return states_tensor, (image_tensor, in_hand_tensor), action_tensor
