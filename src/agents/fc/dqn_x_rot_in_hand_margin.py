import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from agents.fc.dqn_x_rot_in_hand import DQNXRotInHand
from utils import torch_utils

class DQNXRotInHandMargin(DQNXRotInHand):
    def __init__(self, fcn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.99,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=True, patch_size=24, margin='ce',
                 margin_l=0.1, margin_weight=1.0, softmax_beta=1.0, update_divide_factor=4):
        super().__init__(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives, sl, per,
                         num_rotations, half_rotation, patch_size)
        if margin not in ['ce', 'bce', 'bcel', 'l']:
            raise NotImplementedError
        self.margin = margin
        self.margin_l = margin_l
        self.margin_weight = margin_weight
        self.softmax_beta = softmax_beta
        self.update_divide_factor = update_divide_factor

    def _loadBatchToDevice(self, batch):
        states = []
        images = []
        in_hands = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        next_in_hands = []
        dones = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs[0])
            in_hands.append(d.obs[1])
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs[0])
            next_in_hands.append(d.next_obs[1])
            dones.append(d.done)
            is_experts.append(d.expert)
        states_tensor = torch.stack(states).long().to(self.device)
        image_tensor = torch.stack(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        in_hand_tensor = torch.stack(in_hands).to(self.device)
        if len(in_hand_tensor.shape) == 3:
            in_hand_tensor = in_hand_tensor.unsqueeze(1)
        xy_tensor = torch.stack(xys).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        next_in_hands_tensor = torch.stack(next_in_hands).to(self.device)
        if len(next_in_hands_tensor.shape) == 3:
            next_in_hands_tensor = next_in_hands_tensor.unsqueeze(1)
        dones_tensor = torch.stack(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        is_experts_tensor = torch.stack(is_experts).bool().to(self.device)

        return states_tensor, (image_tensor, in_hand_tensor), xy_tensor, rewards_tensor, next_states_tensor, \
               (next_obs_tensor, next_in_hands_tensor), non_final_masks, is_experts_tensor

    # def update(self, batch):
    #     if self.per:
    #         # (states, obs, action_idx, rewards, next_states, next_obs, non_final_masks), weights, idxes = self._loadPrioritizedBatchToDevice(batch)
    #         raise NotImplementedError
    #     else:
    #         states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, is_experts = self._loadBatchToDevice(
    #             batch)
    #     batch_size = states.size(0)
    #     divide_factor = 2
    #     small_batch_size = int(batch_size/divide_factor)
    #     total_loss = 0
    #     td_error = 0
    #     self.fcn_optimizer.zero_grad()
    #     for i in range(divide_factor):
    #         s_states = states[small_batch_size*i:small_batch_size*(i+1)]
    #         s_obs = (obs[0][small_batch_size*i:small_batch_size*(i+1)], obs[1][small_batch_size*i:small_batch_size*(i+1)])
    #         s_action_idx = action_idx[small_batch_size*i:small_batch_size*(i+1)]
    #         s_rewards = rewards[small_batch_size*i:small_batch_size*(i+1)]
    #         s_next_states = next_states[small_batch_size*i:small_batch_size*(i+1)]
    #         s_next_obs = (next_obs[0][small_batch_size*i:small_batch_size*(i+1)], next_obs[1][small_batch_size*i:small_batch_size*(i+1)])
    #         s_non_final_masks = non_final_masks[small_batch_size*i:small_batch_size*(i+1)]
    #         s_is_experts = is_experts[small_batch_size*i:small_batch_size*(i+1)]
    #
    #         if self.sl:
    #             q = self.gamma ** s_rewards
    #         else:
    #             with torch.no_grad():
    #                 q_map_prime = self.forwardFCN(s_next_states, s_next_obs, target_net=True)
    #                 q_prime = q_map_prime.reshape((small_batch_size, -1)).max(1)[0]
    #                 q = s_rewards + self.gamma * q_prime * s_non_final_masks
    #                 q = q.detach()
    #         q_map = self.forwardFCN(s_states, s_obs)
    #         q_output = q_map[torch.arange(0, small_batch_size), s_action_idx[:, 2], s_action_idx[:, 0], s_action_idx[:, 1]]
    #         if self.per:
    #             # td_loss = self.criterion(q_output, q, weights, torch.ones(batch_size).to(self.device))
    #             raise NotImplementedError
    #         else:
    #             td_loss = F.smooth_l1_loss(q_output, q)
    #
    #         if s_is_experts.sum() == 0:
    #             margin_loss = 0
    #         else:
    #             softmax = F.softmax(q_map.reshape(small_batch_size, -1), dim=1).reshape(q_map.shape)
    #             margin_map = torch.zeros_like(q_map).to(self.device)
    #             margin_map[torch.arange(0, small_batch_size), s_action_idx[:, 2], s_action_idx[:, 0], s_action_idx[:, 1]] = 1
    #             margin_loss = F.binary_cross_entropy(softmax[s_is_experts], margin_map[s_is_experts])
    #
    #         loss = (td_loss + margin_loss) / divide_factor
    #         loss.backward()
    #
    #         total_loss += loss.item() / divide_factor
    #         td_error += torch.abs(q_output - q).detach() / divide_factor
    #
    #     for param in self.fcn.parameters():
    #         param.grad.data.clamp_(-1, 1)
    #     self.fcn_optimizer.step()
    #     return total_loss, td_error

    def update(self, batch):
        total_batch_size = len(batch)
        divide_factor = self.update_divide_factor
        batch_size = int(total_batch_size / divide_factor)
        total_loss = 0
        total_td_errors = []
        self.fcn_optimizer.zero_grad()
        for i in range(divide_factor):
            small_batch = batch[batch_size*i:batch_size*(i+1)]
            if self.per:
                # (states, obs, action_idx, rewards, next_states, next_obs, non_final_masks), weights, idxes = self._loadPrioritizedBatchToDevice(batch)
                raise NotImplementedError
            else:
                states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, is_experts = self._loadBatchToDevice(
                    small_batch)
            heightmap_size = obs[0].size(2)
            if self.sl:
                q = self.gamma ** rewards

            else:
                with torch.no_grad():
                    q_map_prime = self.forwardFCN(next_states, next_obs, target_net=True)
                    q_prime = q_map_prime.reshape((batch_size, -1)).max(1)[0]
                    q = rewards + self.gamma * q_prime * non_final_masks
                    q = q.detach()

            q_map = self.forwardFCN(states, obs)
            q_output = q_map[torch.arange(0, batch_size), action_idx[:, 2], action_idx[:, 0], action_idx[:, 1]]
            if self.per:
                # td_loss = self.criterion(q_output, q, weights, torch.ones(batch_size).to(self.device))
                raise NotImplementedError
            else:
                td_loss = F.smooth_l1_loss(q_output, q)

            # cross entropy
            if self.margin == 'ce':
                expert_q_map = q_map[is_experts]
                if expert_q_map.size(0) == 0:
                    margin_loss = 0
                else:
                    target = action_idx[is_experts, 2] * heightmap_size * heightmap_size + action_idx[
                        is_experts, 0] * heightmap_size + action_idx[is_experts, 1]
                    margin_loss = F.cross_entropy(self.softmax_beta*expert_q_map.reshape(expert_q_map.size(0), -1), target)

            # binary cross entropy
            elif self.margin == 'bce':
                expert_q_map = q_map[is_experts]
                if expert_q_map.size(0) == 0:
                    margin_loss = 0
                else:
                    margin_map = torch.zeros_like(q_map)
                    margin_map[torch.arange(0, batch_size), action_idx[:, 2], action_idx[:, 0], action_idx[:, 1]] = 1
                    margin_map = margin_map[is_experts]
                    softmax = F.softmax(self.softmax_beta*expert_q_map.reshape(is_experts.sum(), -1), dim=1).reshape(expert_q_map.shape)
                    margin_loss = F.binary_cross_entropy(softmax, margin_map)

            # binary cross entropy with logits
            elif self.margin == 'bcel':
                margin_map = torch.zeros_like(q_map)
                margin_map[torch.arange(0, batch_size), action_idx[:, 2], action_idx[:, 0], action_idx[:, 1]] = 1
                margin_loss = F.binary_cross_entropy_with_logits(self.softmax_beta*q_map[is_experts], margin_map[is_experts])
                if torch.isnan(margin_loss):
                    margin_loss = 0

            # l margin
            else:
                # margin_map = torch.ones_like(q_map) * self.margin_l
                # margin_map[torch.arange(0, batch_size), action_idx[:, 2], action_idx[:, 0], action_idx[:, 1]] = 0
                # margin_q_map = q_map + margin_map
                # margin_q_max = margin_q_map.reshape(batch_size, -1).max(1)[0]
                # margin_loss = (margin_q_max - q_output)[is_experts]
                # margin_loss = margin_loss.mean()
                # if torch.isnan(margin_loss):
                #     margin_loss = 0

                margin_loss = 0
                num_expert = is_experts.sum().item()
                for j in range(batch_size):
                    if not is_experts[j]:
                        continue
                    qm = q_map[j]
                    qe = q_output[j]
                    over_q = qm[qm > qe - self.margin_l]
                    if over_q.shape[0] == 0:
                        continue
                    over_q_target = torch.ones_like(over_q) * qe - self.margin_l

                    # margin_losses.append(((over_q - over_q_target)**2).mean())
                    margin_loss += (over_q - over_q_target).mean()/num_expert

            loss = td_loss + self.margin_weight * margin_loss

            loss.backward()
            total_loss += (loss.item()/divide_factor)
            total_td_errors.append(torch.abs(q_output - q).detach().cpu())

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        return total_loss, torch.cat(total_td_errors)