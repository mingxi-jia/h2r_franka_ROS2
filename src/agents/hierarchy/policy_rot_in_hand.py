from agents.hierarchy.dqn_rot_mul_in_hand import DQNRotMulInHand
from utils import torch_utils

import numpy as np
import torch
import torch.nn.functional as F

class PolicyRotInHand(DQNRotMulInHand):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives = 1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        super().__init__(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                         num_primitives, sl, per, num_rotations, half_rotation, patch_size)

    def update(self, batch):
        states, obs, action_idx = self._loadBatchToDevice(batch)
        batch_size = states.size(0)
        heightmap_size = obs[0].size(2)

        pixel = action_idx[:, 0:2]
        phis = action_idx[:, 2:]

        q1_output = self.forwardFCN(states, obs).reshape(batch_size, -1)
        q1_target = action_idx[:, 0] * heightmap_size + action_idx[:, 1]
        q1_loss = F.cross_entropy(q1_output, q1_target)

        q2_output = self.forwardPhiNet(states, obs, pixel).reshape(batch_size, -1)
        q2_target = action_idx[:, 2]
        q2_loss = F.cross_entropy(q2_output, q2_target)

        self.fcn_optimizer.zero_grad()
        q1_loss.backward()
        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        self.phi_optimizer.zero_grad()
        q2_loss.backward()
        for param in self.phi_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_optimizer.step()

        return (q1_loss.item(), q2_loss.item()), torch.tensor(0.)

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

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.01):
        with torch.no_grad():
            q_value_maps = self.forwardFCN(states, (obs, in_hand), to_cpu=True)
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef

        m = torch.multinomial(F.softmax(q_value_maps.reshape(states.size(0), -1), 1), 1)
        d = obs.size(2)
        # pixels = torch.cat(((m / d).view(-1, 1), (m % d).view(-1, 1)), dim=1)
        pixels = torch_utils.argmax2d(q_value_maps).long()
        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        with torch.no_grad():
            phi_output = self.forwardPhiNet(states, (obs, in_hand), pixels, to_cpu=True)

        phi = torch.argmax(phi_output, 1)
        # phi = torch.multinomial(F.softmax(phi_output, 1), 1).squeeze(1)
        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps
        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.phi_size)
        phi[rand_mask] = rand_phi.long()
        phi_idx, phi_a = self.decodePhi(phi)
        actions = torch.cat((x, y, phi_a), dim=1)
        action_idx = torch.cat((pixels, phi_idx), dim=1)

        return q_value_maps, action_idx, actions