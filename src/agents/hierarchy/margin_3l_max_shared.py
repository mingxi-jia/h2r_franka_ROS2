from copy import deepcopy
from utils import torch_utils
from agents.hierarchy.dqn_3l_max_shared import DQN3LMaxShared
import numpy as np

import torch
import torch.nn.functional as F

class Margin3LMaxShared(DQN3LMaxShared):
    def __init__(self, q1, q2, q3, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0, num_zs=16, min_z=0.02, max_z=0.17):
        super().__init__(q1, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives,
                         sl, per, num_rotations, half_rotation, patch_size, num_zs, min_z, max_z)
        self.margin = margin
        self.margin_l = margin_l
        self.margin_weight = margin_weight
        self.softmax_beta = softmax_beta

    def calcMarginLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        q1_output = self.loss_calc_dict['q1_output']
        q2_output = self.loss_calc_dict['q2_output']
        q3_output = self.loss_calc_dict['q3_output']

        phis = action_idx[:, 2:]
        a2 = (phis / self.a3_size).long().reshape(phis.size(0), 1)
        a3 = (phis % self.a3_size).long().reshape(phis.size(0), 1)

        q1_pred = q1_output[torch.arange(0, batch_size), action_idx[:, 0], action_idx[:, 1]]
        q2_pred = q2_output[torch.arange(batch_size), a2[:, 0]]
        q3_pred = q3_output[torch.arange(batch_size), a3[:, 0]]

        if self.margin == 'ce':
            expert_q1_output = q1_output[is_experts]
            expert_q2_output = q2_output[is_experts]
            expert_q3_output = q3_output[is_experts]

            if expert_q1_output.size(0) == 0:
                q1_margin_loss = 0
                q2_margin_loss = 0
                q3_margin_loss = 0
            else:
                target_1 = action_idx[is_experts, 0] * self.heightmap_size + action_idx[is_experts, 1]
                target_2 = a2[is_experts, 0]
                target_3 = a3[is_experts, 0]

                q1_margin_loss = F.cross_entropy(self.softmax_beta * expert_q1_output.reshape(expert_q1_output.size(0), -1),
                                                 target_1)
                q2_margin_loss = F.cross_entropy(expert_q2_output.reshape(expert_q2_output.size(0), -1), target_2)
                q3_margin_loss = F.cross_entropy(expert_q3_output.reshape(expert_q3_output.size(0), -1), target_3)

        elif self.margin == 'oril':
            margin_1 = torch.ones_like(q1_output) * self.margin_l
            margin_2 = torch.ones_like(q2_output) * self.margin_l
            margin_3 = torch.ones_like(q3_output) * self.margin_l
            margin_1[torch.arange(0, batch_size), action_idx[:, 0], action_idx[:, 1]] = 0
            margin_2[torch.arange(0, batch_size), a2[:, 0]] = 0
            margin_3[torch.arange(0, batch_size), a3[:, 0]] = 0
            margin_output_1 = q1_output + margin_1
            margin_output_2 = q2_output + margin_2
            margin_output_3 = q3_output + margin_3
            margin_output_1_max = margin_output_1.reshape(batch_size, -1).max(1)[0]
            margin_output_2_max = margin_output_2.reshape(batch_size, -1).max(1)[0]
            margin_output_3_max = margin_output_3.reshape(batch_size, -1).max(1)[0]
            q1_margin_loss = (margin_output_1_max - q1_pred)[is_experts]
            q2_margin_loss = (margin_output_2_max - q2_pred)[is_experts]
            q3_margin_loss = (margin_output_3_max - q3_pred)[is_experts]
            q1_margin_loss = q1_margin_loss.mean()
            q2_margin_loss = q2_margin_loss.mean()
            q3_margin_loss = q3_margin_loss.mean()
            if torch.isnan(q1_margin_loss) or torch.isnan(q2_margin_loss) or torch.isnan(q3_margin_loss):
                q1_margin_loss = 0
                q2_margin_loss = 0
                q3_margin_loss = 0

        elif self.margin == 'l':
            q1_margin_losses = []
            q2_margin_losses = []
            q3_margin_losses = []
            for j in range(batch_size):
                if not is_experts[j]:
                    q1_margin_losses.append(torch.tensor(0).float().to(self.device))
                    q2_margin_losses.append(torch.tensor(0).float().to(self.device))
                    q3_margin_losses.append(torch.tensor(0).float().to(self.device))
                    continue
                # calculate q3 margin loss
                q3e = q3_pred[j]
                q_all_3 = q3_output[j]
                over_q3 = q_all_3[q_all_3 > q3e - self.margin_l]
                if over_q3.shape[0] == 0:
                    q3_margin_losses.append(torch.tensor(0).float().to(self.device))
                else:
                    over_q3_target = torch.ones_like(over_q3) * q3e - self.margin_l
                    q3_margin_losses.append((over_q3 - over_q3_target).mean())

                # calculate q2 margin loss
                q2e = q2_pred[j]
                q_all_2 = q2_output[j]
                over_q2 = q_all_2[q_all_2 > q2e - self.margin_l]
                if over_q2.shape[0] == 0:
                    q2_margin_losses.append(torch.tensor(0).float().to(self.device))
                else:
                    over_q2_target = torch.ones_like(over_q2) * q2e - self.margin_l
                    q2_margin_losses.append((over_q2 - over_q2_target).mean())

                # calculate q1 margin loss
                q1e = q1_pred[j]
                qm = q1_output[j]
                over_q1 = qm[qm > q1e - self.margin_l]
                if over_q1.shape[0] == 0:
                    q1_margin_losses.append(torch.tensor(0).float().to(self.device))
                else:
                    over_q1_target = torch.ones_like(over_q1) * q1e - self.margin_l
                    q1_margin_losses.append((over_q1 - over_q1_target).mean())

            q1_margin_loss = torch.stack(q1_margin_losses).mean()
            q2_margin_loss = torch.stack(q2_margin_losses).mean()
            q3_margin_loss = torch.stack(q3_margin_losses).mean()

        return q1_margin_loss, q2_margin_loss, q3_margin_loss

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()
        q1_margin_loss, q2_margin_loss, q3_margin_loss = self.calcMarginLoss()

        loss = td_loss + self.margin_weight * q1_margin_loss + self.margin_weight * q2_margin_loss + self.margin_weight * q3_margin_loss

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

        return loss.item(), td_error