from agents.hierarchy_backup.phi.dqn_phi import DQNPhi
from utils import torch_utils

import numpy as np

import torch
import torch.nn.functional as F

class DQNPhiPrimitive(DQNPhi):
    def __init__(self, *args, **kwargs):
        super(DQNPhiPrimitive, self).__init__(*args, **kwargs)

    def encodeInput(self, states, img):
        states_reshape = states.reshape(states.shape[0], 1, 1, 1)
        states_expanded = states_reshape.expand(img.size())
        return torch.cat((img, states_expanded.float()), dim=1)

    def getEGreedyPhi(self, states, obs, eps, pixels):
        patch = self.getImgPatch(obs, pixels.to(self.device))
        states = states.to(self.device)
        cnn_input = self.encodeInput(states, patch)

        with torch.no_grad():
            phi_output = self.phi_net(cnn_input).cpu().reshape(states.size(0),
                                                           self.num_primitives, -1)[torch.arange(0, states.size(0)),
                                                                                    states.long()]
        phi = torch.argmax(phi_output, 1)

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.num_rotations * self.num_heights)
        phi[rand_mask] = rand_phi.long()
        rot_id = (phi / self.num_rotations).long().reshape(states.size(0), 1)
        rotation = self.rotations[rot_id].reshape(states.size(0), 1)
        z_id = (phi % self.num_heights).long().reshape(states.size(0), 1)
        z = self.heights[z_id].reshape(states.size(0), 1)
        return rot_id, z_id, rotation, z

    def getEGreedyActions(self, states, obs, eps):
        obs = obs.to(self.device)
        states = states.to(self.device)
        fcn_input = self.encodeInput(states, obs)
        with torch.no_grad():
            q_value_maps = self.fcn(fcn_input).cpu()
        argmax = torch_utils.argmax3d(q_value_maps)
        primitives = argmax[:, 0]

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_primitives = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.num_primitives)
        primitives[rand_mask] = rand_primitives.long()

        q_value_maps += torch.randn_like(q_value_maps) * eps * 0.01
        pixels = torch_utils.argmax2d(q_value_maps[torch.arange(0, states.size(0)), states.long()]).cpu().long()
        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(obs.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(obs.size(0), 1)

        rot_id, z_id, rotation, z = self.getEGreedyPhi(primitives, obs, eps, pixels)

        actions = torch.cat((primitives.unsqueeze(1).float(), x, y, z, rotation), dim=1)
        action_idx = torch.cat((primitives.unsqueeze(1), pixels, z_id, rot_id), dim=1)
        return q_value_maps, action_idx, actions

    def getEGreedyActionsGreedyX(self, states, obs, eps):
        obs = obs.to(self.device)
        states = states.to(self.device)
        fcn_input = self.encodeInput(states, obs)
        with torch.no_grad():
            q_value_maps = self.fcn(fcn_input).cpu()
        argmax = torch_utils.argmax3d(q_value_maps)
        primitives = argmax[:, 0]

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_primitives = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.num_primitives)
        primitives[rand_mask] = rand_primitives.long()

        pixels = torch_utils.argmax2d(q_value_maps[torch.arange(0, states.size(0)), states.long()]).cpu().long()
        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        rot_id, z_id, rotation, z = self.getEGreedyPhi(states, obs, eps, pixels)

        actions = torch.cat((primitives.unsqueeze(1).float(), x, y, z, rotation), dim=1)
        action_idx = torch.cat((primitives.unsqueeze(1), pixels, z_id, rot_id), dim=1)
        return q_value_maps, action_idx, actions

    def samplePhis(self, states, obs, center_pixels, num_phis):
        patch = self.getImgPatch(obs.to(self.device), center_pixels.to(self.device))
        states = states.to(self.device)
        with torch.no_grad():
            phi_output = self.phi_net(self.encodeInput(states, patch)).cpu().reshape(states.size(0),
                                                                                     self.num_primitives, -1)[torch.arange(0, states.size(0)),
                                                                                                              states.long()]
        phis = []
        for output in phi_output.cpu().numpy():
            p = np.exp(output) / np.exp(output).sum()
            phi = torch.tensor(np.random.choice(np.arange(len(output)), num_phis, replace=False, p=p))
            rot_id = (phi / self.num_rotations).long().reshape(num_phis, 1)
            z_id = (phi % self.num_heights).long().reshape(num_phis, 1)
            phis.append(torch.cat((rot_id, z_id), dim=1).unsqueeze(1))
        return torch.cat(phis, dim=1)

    def updateIterate(self, batch):
        states, image, pixel, primitives, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(batch)
        batch_size = image.size(0)
        phi_size = phis.size(1)
        channel_size = image.size(1)
        heightmap_size = image.size(2)

        q = []
        with torch.no_grad():
            for i in range(batch_size):
                nxt_o = next_obs[i]
                nxt_s = next_states[i]
                q1_map_prime = self.target_fcn(self.encodeInput(nxt_s, nxt_o))
                argmax = torch_utils.argmax3d(q1_map_prime)
                primitive_star = argmax[:, 0]
                x_star = argmax[:, 1:]
                q1 = q1_map_prime[torch.arange(0, phi_size), primitive_star, x_star[:, 0], x_star[:, 1]]
                patch = self.getImgPatch(nxt_o, x_star)
                q2_prime = self.target_phi(self.encodeInput(nxt_s, patch)).reshape(phi_size, -1)
                q2 = q2_prime.max(1)[0]
                q_prime = q1 * q2
                q.append(rewards[i] + self.gamma * q_prime * non_final_masks[i])

            q = torch.stack(q)
            max_phi_q = q.max(1)[0]
            max_phi_q_expanded = max_phi_q.reshape(batch_size, 1).expand(batch_size, phi_size)

        q1_output = self.fcn(self.encodeInput(states, image))[torch.arange(0, batch_size), primitives, pixel[:, 0], pixel[:, 1]]
        current_patch = self.getImgPatch(image, pixel).to(self.device)
        q2_output = self.phi_net(self.encodeInput(states, current_patch)).reshape(batch_size, self.num_primitives, -1)[
            torch.arange(0, batch_size), primitives]
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)
        non_zero_mask = max_phi_q_expanded != 0
        fcn_loss = F.smooth_l1_loss(q1_output, max_phi_q)
        phi_loss = F.smooth_l1_loss(q2_output[non_zero_mask], q[non_zero_mask] / max_phi_q_expanded[non_zero_mask])

        self.fcn_optimizer.zero_grad()
        fcn_loss.backward()
        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        self.phi_optimizer.zero_grad()
        phi_loss.backward()
        for param in self.phi_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_optimizer.step()

        return (fcn_loss.item(), phi_loss.item()), None

    def _loadBatchToDevice(self, batch):
        states = []
        images = []
        xys = []
        primitives = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        phis = []
        for d in batch:
            states.append(d.state)
            images.append(d.image)
            xys.append(d.xy)
            primitives.append(d.primitive)
            rewards.append(d.rewards)
            next_states.append(d.next_states)
            next_obs.append(torch.stack(d.next_obs))
            dones.append(d.dones)
            phis.append(d.phis)
        states_tensor = torch.tensor(states).long().to(self.device)
        image_tensor = torch.tensor(images).unsqueeze(1).to(self.device)
        xy_tensor = torch.tensor(xys).to(self.device)
        primitives_tensor = torch.tensor(primitives).long().to(self.device)
        rewards_tensor = torch.tensor(rewards).to(self.device)
        next_states_tensor = torch.tensor(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        dones_tensor = torch.tensor(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        phis_tensor = torch.tensor(phis).long().to(self.device)
        return states_tensor, image_tensor, xy_tensor, primitives_tensor, rewards_tensor, next_states_tensor, next_obs_tensor, non_final_masks, phis_tensor
