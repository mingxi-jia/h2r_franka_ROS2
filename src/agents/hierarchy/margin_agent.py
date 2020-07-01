import torch
import torch.nn.functional as F
from agents.hierarchy.in_hand_agent import DQNRotMulInHand, DQNRotMaxInHand, DQNRotMaxSharedInHand, DQNPRotMaxSharedInHand

class MarginAgent:
    def __init__(self, margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0):
        self.margin = margin
        self.margin_l = margin_l
        self.margin_weight = margin_weight
        self.softmax_beta = softmax_beta

    def calcMarginLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        q1_output = self.loss_calc_dict['q1_output']
        q2_output = self.loss_calc_dict['q2_output']

        q1_pred = q1_output[torch.arange(0, batch_size), action_idx[:, 0], action_idx[:, 1]]
        q2_pred = q2_output[torch.arange(batch_size), action_idx[:, 2]]

        if self.margin == 'ce':
            expert_q1_output = q1_output[is_experts]
            expert_q2_output = q2_output[is_experts]

            if expert_q1_output.size(0) == 0:
                q1_margin_loss = 0
                q2_margin_loss = 0
            else:
                target_1 = action_idx[is_experts, 0] * self.heightmap_size + action_idx[is_experts, 1]
                target_2 = action_idx[is_experts, 2]

                q1_margin_loss = F.cross_entropy(self.softmax_beta * expert_q1_output.reshape(expert_q1_output.size(0), -1),
                                                 target_1)
                q2_margin_loss = F.cross_entropy(expert_q2_output.reshape(expert_q2_output.size(0), -1), target_2)

        elif self.margin == 'oril':
            margin_1 = torch.ones_like(q1_output) * self.margin_l
            margin_2 = torch.ones_like(q2_output) * self.margin_l
            margin_1[torch.arange(0, batch_size), action_idx[:, 0], action_idx[:, 1]] = 0
            margin_2[torch.arange(0, batch_size), action_idx[:, 2]] = 0
            margin_output_1 = q1_output + margin_1
            margin_output_2 = q2_output + margin_2
            margin_output_1_max = margin_output_1.reshape(batch_size, -1).max(1)[0]
            margin_output_2_max = margin_output_2.reshape(batch_size, -1).max(1)[0]
            q1_margin_loss = (margin_output_1_max - q1_pred)[is_experts]
            q2_margin_loss = (margin_output_2_max - q2_pred)[is_experts]
            q1_margin_loss = q1_margin_loss.mean()
            q2_margin_loss = q2_margin_loss.mean()
            if torch.isnan(q1_margin_loss) or torch.isnan(q2_margin_loss):
                q1_margin_loss = 0
                q2_margin_loss = 0


        else:
            q1_margin_losses = []
            q2_margin_losses = []
            for j in range(batch_size):
                if not is_experts[j]:
                    q1_margin_losses.append(torch.tensor(0).float().to(self.device))
                    q2_margin_losses.append(torch.tensor(0).float().to(self.device))
                    continue
                # calculate q2 margin loss
                q2e = q2_pred[j]
                q_all_phi = q2_output[j]
                over_q2 = q_all_phi[q_all_phi > q2e - self.margin_l]
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

        return q1_margin_loss, q2_margin_loss


    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()
        q1_margin_loss, q2_margin_loss = self.calcMarginLoss()

        loss = td_loss + self.margin_weight * q1_margin_loss + self.margin_weight * q2_margin_loss

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

class DQNRotMulInHandMargin(MarginAgent, DQNRotMulInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0):
        DQNRotMulInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        MarginAgent.__init__(self, margin, margin_l, margin_weight, softmax_beta)


class DQNRotMaxInHandMargin(MarginAgent, DQNRotMaxInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0):
        DQNRotMaxInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                 num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        MarginAgent.__init__(self, margin, margin_l, margin_weight, softmax_beta)


class DQNRotMaxSharedInHandMargin(MarginAgent, DQNRotMaxSharedInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0):
        DQNRotMaxSharedInHand.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                       num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        MarginAgent.__init__(self, margin, margin_l, margin_weight, softmax_beta)

class DQNPRotMaxSharedInHandMargin(MarginAgent, DQNPRotMaxSharedInHand):
    def __init__(self, fcn, cnn, fcn_p, cnn_p, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0):
        DQNPRotMaxSharedInHand.__init__(self, fcn, cnn, fcn_p, cnn_p, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                        num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        MarginAgent.__init__(self, margin, margin_l, margin_weight, softmax_beta)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()
        q1_margin_loss, q2_margin_loss = self.calcMarginLoss()

        loss = td_loss + self.margin_weight * q1_margin_loss + self.margin_weight * q2_margin_loss

        self.fcn_optimizer.zero_grad()
        self.phi_optimizer.zero_grad()
        loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.phi_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_optimizer.step()

        p_loss = self.calcPLoss()
        self.fcn_p_optimizer.zero_grad()
        self.phi_p_optimizer.zero_grad()
        p_loss.backward()

        for param in self.fcn_p.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_p_optimizer.step()

        for param in self.phi_net_p.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_p_optimizer.step()

        self.loss_calc_dict = {}

        return (loss.item(), p_loss.item()), td_error