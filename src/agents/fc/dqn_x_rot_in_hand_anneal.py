import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from copy import deepcopy
from utils import torch_utils
from agents.fc.dqn_x_rot_in_hand import DQNXRotInHand

class DQNXRotInHandAnneal(DQNXRotInHand):
    def __init__(self, fcn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.99,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=True, patch_size=24):
        super().__init__(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives, sl, per,
                         num_rotations, half_rotation, patch_size)

    def update(self, batch, sl_weight):
        states, obs, action_idx, step_left, next_states, next_obs, non_final_masks = self._loadBatchToDevice(batch)
        batch_size = states.size(0)
        
        rewards = (step_left == 0).float()
        q_sl = self.gamma ** step_left

        with torch.no_grad():
            divide_factor = 2
            small_batch_size = int(batch_size / divide_factor)
            qs_rl = []
            for i in range(divide_factor):
                s_next_states = next_states[small_batch_size * i:small_batch_size * (i + 1)]
                s_next_obs = (next_obs[0][small_batch_size * i:small_batch_size * (i + 1)],
                              next_obs[1][small_batch_size * i:small_batch_size * (i + 1)])
                s_rewards = rewards[small_batch_size * i:small_batch_size * (i + 1)]
                s_non_final_masks = non_final_masks[small_batch_size * i:small_batch_size * (i + 1)]
                q_map_prime = self.forwardFCN(s_next_states, s_next_obs, target_net=True)
                q_prime = q_map_prime.reshape((small_batch_size, -1)).max(1)[0]
                q = s_rewards + self.gamma * q_prime * s_non_final_masks
                qs_rl.append(q.detach())
            q_rl = torch.cat(qs_rl)

        q = sl_weight * q_sl + (1-sl_weight) * q_rl

        q_output = self.forwardFCN(states, obs, specific_rotations=action_idx[:, 2:3].cpu())[torch.arange(0, batch_size), 0, action_idx[:, 0], action_idx[:, 1]]
        loss = F.smooth_l1_loss(q_output, q)
        self.fcn_optimizer.zero_grad()
        loss.backward()
        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        td_error = torch.abs(q_output - q).detach()
        return loss.item(), td_error