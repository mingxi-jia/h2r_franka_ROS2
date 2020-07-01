from agents.hierarchy.dqn_rot_mul_in_hand import DQNRotMulInHand
from utils import torch_utils


import torch
import torch.nn.functional as F

class DQNRotMulInHandMargin(DQNRotMulInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives = 1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0):
        super().__init__(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                         num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        self.margin = margin
        self.margin_l = margin_l
        self.margin_weight = margin_weight
        self.softmax_beta = softmax_beta

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()

        td_loss, td_error = self.calcTDLoss()

        q_target = self.loss_calc_dict['q_target']
        q1_output = self.loss_calc_dict['q1_output']
        q2_output = self.loss_calc_dict['q2_output']
        q1_pred = q1_output[torch.arange(0, batch_size), action_idx[:, 0], action_idx[:, 1]]

        q1_margin_losses = []
        q2_margin_losses = []
        for j in range(batch_size):
            if not is_experts[j]:
                q1_margin_losses.append(torch.tensor(0).float().to(self.device))
                continue
            qe = q_target[j]
            # calculate q2 margin loss
            q_all_phi = q1_pred[j].detach() * q2_output[j]
            over_q2 = q_all_phi[q_all_phi > qe - self.margin_l]
            if over_q2.shape[0] == 0:
                q2_margin_losses.append(torch.tensor(0).float().to(self.device))
            else:
                over_q2_target = torch.ones_like(over_q2) * qe - self.margin_l
                q2_margin_losses.append((over_q2 - over_q2_target).mean())

            # calculate q1 margin loss
            qm = q1_output[j]
            over_q1 = qm[qm > qe - self.margin_l]
            if over_q1.shape[0] == 0:
                q1_margin_losses.append(torch.tensor(0).float().to(self.device))
            else:
                over_q1_target = torch.ones_like(over_q1) * qe - self.margin_l
                q1_margin_losses.append((over_q1 - over_q1_target).mean())

        q1_margin_loss = torch.stack(q1_margin_losses).mean()
        q2_margin_loss = torch.stack(q2_margin_losses).mean()

        loss = td_loss + self.margin_weight * q1_margin_loss + self.margin_weight * q2_margin_loss

        # q1_loss = q1_td_loss + self.margin_weight * q1_margin_loss
        # q2_loss = q2_td_loss + self.margin_weight * q2_margin_loss

        self.fcn_optimizer.zero_grad()
        self.phi_optimizer.zero_grad()
        loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.phi_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_optimizer.step()

        self.loss_calc_dict = {}

        return loss.item(), td_error

