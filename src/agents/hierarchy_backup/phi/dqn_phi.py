from copy import deepcopy
from agents.hierarchy_backup.phi.hierarchy_phi import HierarchyPhi
from utils import torch_utils

import numpy as np

import torch
import torch.nn.functional as F

class DQNPhi(HierarchyPhi):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, num_rotations=8,
                 half_rotation=False, num_heights=10, height_range=(1, 0.1), lr=1e-4, patch_size=64, gamma=0.99,
                 iter_update=True, num_primitives=1, num_input_channel=1, sl=False, one_a=False, per=False):
        super(DQNPhi, self).__init__(fcn, cnn, action_space, workspace, heightmap_resolution, device, num_rotations,
                                     half_rotation, num_heights, height_range, lr, patch_size, num_primitives,
                                     num_input_channel)

        self.target_fcn = deepcopy(fcn)
        self.target_phi = deepcopy(cnn)
        self.updateTarget()
        self.gamma = gamma
        self.iter_update = iter_update

        self.criterion = torch_utils.WeightedHuberLoss()

        self.sl = sl
        self.one_a = one_a
        self.per = per

    def setCandidatePos(self, pos_candidate):
        self.fcn.setCandidatePos(pos_candidate)
        self.target_fcn.setCandidatePos(pos_candidate)

    def forwardPhiNet(self, states, obs, pixels, target_net=False, to_cpu=False):
        patch = self.getImgPatch(obs.to(self.device), pixels.to(self.device))
        phi_net = self.phi_net if not target_net else self.target_phi
        phi_output = phi_net(patch).reshape(states.size(0), self.num_primitives, -1)[torch.arange(0, states.size(0)),
                                                                                     states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

    def forwardFCN(self, states, obs, target_net=False, to_cpu=False):
        fcn = self.fcn if not target_net else self.target_fcn
        obs = obs.to(self.device)
        q_value_maps = fcn(obs)[torch.arange(0, states.size(0)), states.long()]

        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps

    def getEGreedyPhi(self, states, obs, eps, pixels):
        # patch = self.getImgPatch(obs, pixels.to(self.device))
        # with torch.no_grad():
        #     phi_output = self.phi_net(patch).cpu().reshape(states.size(0),
        #                                                    self.num_primitives, -1)[torch.arange(0, states.size(0)),
        #                                                                             states.long()]
        with torch.no_grad():
            phi_output = self.forwardPhiNet(states, obs, pixels, to_cpu=True)
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

    def getEGreedyActions(self, states, obs, eps, coef=0.01):
        # obs = obs.to(self.device)
        # with torch.no_grad():
        #     q_value_maps = self.fcn(obs).cpu()
        # q_value_maps += torch.randn_like(q_value_maps) * eps * 0.01
        # pixels = torch_utils.argmax2d(q_value_maps[torch.arange(0, states.size(0)), states.long()]).cpu().long()
        q_value_maps = self.forwardFCN(states, obs, to_cpu=True)
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        pixels = torch_utils.argmax2d(q_value_maps).long()
        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        rot_id, z_id, rotation, z = self.getEGreedyPhi(states, obs, eps, pixels)

        actions = torch.cat((x, y, z, rotation), dim=1)
        action_idx = torch.cat((pixels, z_id, rot_id), dim=1)
        return q_value_maps, action_idx, actions

    def getEGreedyActionsGreedyX(self, states, obs, eps):
        q_value_maps = self.forwardFCN(states, obs, to_cpu=True)
        pixels = torch_utils.argmax2d(q_value_maps).long()
        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        rot_id, z_id, rotation, z = self.getEGreedyPhi(states, obs, eps, pixels)

        actions = torch.cat((x, y, z, rotation), dim=1)
        action_idx = torch.cat((pixels, z_id, rot_id), dim=1)
        return q_value_maps, action_idx, actions

    def getSoftmaxActions(self, states, obs, eps):
        obs = obs.to(self.device)
        with torch.no_grad():
            q_value_maps = self.fcn(obs).cpu()

        pixels = []
        q_maps = q_value_maps[torch.arange(0, states.size(0)), states.long()]
        for q_map in q_maps:
            topk_value, topk_idx = torch.topk(q_map.flatten(), int(max(obs.size(2)*obs.size(3)*eps, 1)))
            topk_prob = torch.nn.Softmax(0)(topk_value).cpu().numpy()
            flatten_pixel = np.random.choice(topk_idx.cpu().numpy(), p=topk_prob)
            pixels.append([int(flatten_pixel/obs.size(2)), int(flatten_pixel%obs.size(2))])
        pixels = torch.tensor(pixels)

        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        rot_id, z_id, rotation, z = self.getEGreedyPhi(states, obs, eps, pixels)

        actions = torch.cat((x, y, z, rotation), dim=1)
        action_idx = torch.cat((pixels, z_id, rot_id), dim=1)
        return q_value_maps, action_idx, actions

    def getEGreedySoftmaxActions(self, states, obs, eps):
        obs = obs.to(self.device)
        with torch.no_grad():
            q_value_maps = self.fcn(obs).cpu()

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps

        height, width = q_value_maps.shape[2], q_value_maps.shape[3]
        maps_hands = q_value_maps[torch.arange(0, states.size(0)), states.long()]
        flat_maps = maps_hands.reshape((maps_hands.shape[0], height * width))
        m = torch.distributions.Categorical(logits=flat_maps)
        indices = m.sample()
        softmax_pixels = torch.tensor(np.unravel_index(indices, (height, width))).permute(1, 0).long()
        greedy_pixels = torch_utils.argmax2d(q_value_maps[torch.arange(0, states.size(0)), states.long()]).cpu().long()

        pixels = greedy_pixels
        pixels[rand_mask] = softmax_pixels[rand_mask]

        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        rot_id, z_id, rotation, z = self.getEGreedyPhi(states, obs, eps, pixels)

        actions = torch.cat((x, y, z, rotation), dim=1)
        action_idx = torch.cat((pixels, z_id, rot_id), dim=1)
        return q_value_maps, action_idx, actions

    def getActionFromPlan(self, plan):
        states = plan[:, 0:1]
        x = plan[:, 1:2]
        y = plan[:, 2:3]
        z = plan[:, 3:4]
        rot = plan[:, 4:5]
        pixel_y = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_x = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        z_id = (z.expand(-1, self.num_heights) - self.heights).abs().argmin(1).unsqueeze(1)
        rot_id = (rot.expand(-1, self.num_rotations) - self.rotations).abs().argmin(1).unsqueeze(1)

        x = (pixel_y.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_x.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        z = self.heights[z_id].reshape(states.size(0), 1)
        rotation = self.rotations[rot_id].reshape(states.size(0), 1)
        actions = torch.cat((x, y, z, rotation), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, z_id, rot_id), dim=1)
        return action_idx, actions

    def getActionAndPatchFromPlan(self, plan, obs):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        z = plan[:, 2:3]
        rot = plan[:, 3:4]
        states = plan[:, 4:5]
        pixel_y = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_x = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        z_id = (z.expand(-1, self.num_heights) - self.heights).abs().argmin(1).unsqueeze(1)
        rot_id = (rot.expand(-1, self.num_rotations) - self.rotations).abs().argmin(1).unsqueeze(1)

        x = (pixel_y.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_x.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        z = self.heights[z_id].reshape(states.size(0), 1)
        rotation = self.rotations[rot_id].reshape(states.size(0), 1)
        actions = torch.cat((x, y, z, rotation), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, z_id, rot_id), dim=1)

        patch = self.getImgPatch(obs[:, 0:1].to(self.device), torch.cat((pixel_x, pixel_y), dim=1).to(self.device))
        return action_idx, actions, patch

    def updateTarget(self):
        self.target_fcn.load_state_dict(self.fcn.state_dict())
        self.target_phi.load_state_dict(self.phi_net.state_dict())

    def samplePhis(self, states, obs, center_pixels, num_phis):
        # patch = self.getImgPatch(obs.to(self.device), center_pixels.to(self.device))
        # with torch.no_grad():
        #     phi_output = self.phi_net(patch).cpu().reshape(states.size(0),
        #                                                    self.num_primitives, -1)[torch.arange(0, states.size(0)),
        #                                                                             states.long()]
        with torch.no_grad():
            phi_output = self.forwardPhiNet(states, obs, center_pixels, to_cpu=True)
        phis = []
        for output in phi_output.cpu().numpy():
            p = np.exp(output) / np.exp(output).sum()
            phi = torch.tensor(np.random.choice(np.arange(len(output)), num_phis, replace=False, p=p))
            rot_id = (phi / self.num_rotations).long().reshape(num_phis, 1)
            z_id = (phi % self.num_heights).long().reshape(num_phis, 1)
            phis.append(torch.cat((rot_id, z_id), dim=1).unsqueeze(1))
        return torch.cat(phis, dim=1)

    def reshapeNextObs(self, next_obs, batch_size):
        next_obs = next_obs.reshape(batch_size, next_obs.shape[-3], next_obs.shape[-2], next_obs.shape[-1])
        return next_obs

    def update(self, batch):
        if self.one_a:
            return self.update1a(batch)
        else:
            return self.updateNormal(batch)

    def updateNormal(self, batch):
        if self.per:
            (states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis), weights, idxes = self._loadPrioritizedBatchToDevice(batch)
        else:
            states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(batch)
        batch_size = states.size(0)
        phi_size = phis.size(1)

        # SL target
        if self.sl:
            q = self.gamma ** rewards

        # RL target
        else:
            if self.iter_update:
                q = []
                with torch.no_grad():
                    for i in range(batch_size):
                        q1_map_prime = self.forwardFCN(next_states[i], next_obs[i], target_net=True)
                        x_star = torch_utils.argmax2d(q1_map_prime)
                        q1 = q1_map_prime[torch.arange(0, phi_size), x_star[:, 0], x_star[:, 1]]
                        q2_prime = self.forwardPhiNet(next_states[i], next_obs[i], x_star, target_net=True)
                        q2 = q2_prime.max(1)[0]
                        q_prime = q1 * q2
                        q.append(rewards[i] + self.gamma * q_prime * non_final_masks[i])
                    q = torch.stack(q)

            else:
                next_obs = self.reshapeNextObs(next_obs, batch_size*phi_size)
                next_states = next_states.reshape(batch_size * phi_size)
                with torch.no_grad():
                    q1_map_prime = self.forwardFCN(next_states, next_obs, target_net=True)
                    x_star = torch_utils.argmax2d(q1_map_prime)
                    q1 = q1_map_prime[torch.arange(0, batch_size * phi_size), x_star[:, 0], x_star[:, 1]].reshape(batch_size, phi_size)
                    q2_prime = self.forwardPhiNet(next_states, next_obs, x_star, target_net=True)
                    q2 = q2_prime.max(1)[0]
                    q2 = q2.reshape(batch_size, phi_size)
                    q_prime = q1 * q2
                    q = rewards + self.gamma * q_prime * non_final_masks

        max_phi_q = q.max(1)[0]
        max_phi_q_expanded = max_phi_q.reshape(batch_size, 1).expand(batch_size, phi_size)
        non_zero_mask = max_phi_q_expanded != 0

        q2_output = self.forwardPhiNet(states, image, pixel)
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)
        q1_output = self.forwardFCN(states, image)[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]

        if self.per:
            fcn_loss = self.criterion(q1_output, max_phi_q, weights, torch.ones(batch_size).to(self.device))
            max_phi_q += 1e-4
            phi_loss = self.criterion(q2_output, q / max_phi_q, weights, non_zero_mask.float())
        else:
            fcn_loss = F.smooth_l1_loss(q1_output, max_phi_q.clone())
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

        with torch.no_grad():
            td_error = torch.abs((q1_output.unsqueeze(1) * q2_output - q).mean(1))

        return (fcn_loss.item(), phi_loss.item()), td_error

    def update1a(self, batch):
        if self.per:
            (states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis), weights, idxes = self._loadPrioritizedBatchToDevice(batch)
        else:
            states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(batch)
        batch_size = states.size(0)
        phi_size = phis.size(1)
        next_obs = self.reshapeNextObs(next_obs, batch_size * phi_size)
        next_states = next_states.reshape(batch_size * phi_size)
        # SL target
        if self.sl:
            q = self.gamma ** rewards
        # RL target
        else:
            with torch.no_grad():
                q1_map_prime = self.forwardFCN(next_states, next_obs, target_net=True)
                x_star = torch_utils.argmax2d(q1_map_prime)
                q1 = q1_map_prime[torch.arange(0, batch_size * phi_size), x_star[:, 0], x_star[:, 1]].reshape(batch_size,
                                                                                                              phi_size)
                q2_prime = self.forwardPhiNet(next_states, next_obs, x_star, target_net=True)
                q2 = q2_prime.max(1)[0]
                q2 = q2.reshape(batch_size, phi_size)
                q_prime = q1 * q2
                q = rewards + self.gamma * q_prime * non_final_masks

        q = q.squeeze(1)
        with torch.no_grad():
            q2_target_output = self.forwardPhiNet(states, image, pixel, target_net=True)
            q1_target_output = self.forwardFCN(states, image, target_net=True)[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]

        # max_phi_q = q2_target_output.max(1)[0] * q1_target_output
        max_phi_q = torch.stack(((q2_target_output.max(1)[0] * q1_target_output), q), dim=1).max(1)[0]
        non_zero_mask = max_phi_q != 0

        q2_output = self.forwardPhiNet(states, image, pixel)
        q2_output = q2_output[torch.arange(batch_size), phis[:, 0]]
        q1_output = self.forwardFCN(states, image)[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]

        if self.per:
            fcn_loss = self.criterion(q1_output, max_phi_q, weights, torch.ones(batch_size).to(self.device))
            max_phi_q += 1e-4
            phi_loss = self.criterion(q2_output, q / max_phi_q, weights, non_zero_mask.float())
        else:
            fcn_loss = F.smooth_l1_loss(q1_output, max_phi_q.clone())
            phi_loss = F.smooth_l1_loss(q2_output[non_zero_mask], q[non_zero_mask] / max_phi_q[non_zero_mask])

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

        with torch.no_grad():
            td_error = torch.abs(q1_output * q2_output - q)

        return (fcn_loss.item(), phi_loss.item()), td_error

    # def updateSL(self, batch):
    #     states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(
    #         batch)
    #     batch_size = states.size(0)
    #     phi_size = phis.size(1)
    #
    #     non_final_masks[rewards==0] = 1
    #     q = self.gamma ** rewards * non_final_masks
    #
    #     max_phi_q = q.max(1)[0]
    #     max_phi_q_expanded = max_phi_q.reshape(batch_size, 1).expand(batch_size, phi_size)
    #     non_zero_mask = max_phi_q_expanded != 0
    #
    #     q2_output = self.forwardPhiNet(states, image, pixel)
    #     q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
    #                           dim=1)
    #     q1_output = self.forwardFCN(states, image)[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]
    #
    #     fcn_loss = F.smooth_l1_loss(q1_output, max_phi_q)
    #     phi_loss = F.smooth_l1_loss(q2_output[non_zero_mask], q[non_zero_mask] / max_phi_q_expanded[non_zero_mask])
    #
    #     self.fcn_optimizer.zero_grad()
    #     fcn_loss.backward()
    #     for param in self.fcn.parameters():
    #         param.grad.data.clamp_(-1, 1)
    #     self.fcn_optimizer.step()
    #
    #     self.phi_optimizer.zero_grad()
    #     phi_loss.backward()
    #     for param in self.phi_net.parameters():
    #         param.grad.data.clamp_(-1, 1)
    #     self.phi_optimizer.step()
    #
    #     return (fcn_loss.item(), phi_loss.item()), None

    def updatePrioritized(self, batch):
        (states, image, pixel, rewards, next_states, next_obs, non_final_masks), weights, idxes= self._loadPrioritizedBatchToDevice(batch)
        batch_size = image.size(0)
        phi_size = self.num_rotations * self.num_heights
        channel_size = image.size(1)
        heightmap_size = image.size(2)

        q = []
        with torch.no_grad():
            for i in range(batch_size):
                nxt_o = next_obs[i]
                q1_map_prime = self.target_fcn(nxt_o)[torch.arange(0, phi_size), next_states[i]].reshape(phi_size,
                                                                                                         heightmap_size,
                                                                                                         heightmap_size)
                x_star = torch_utils.argmax2d(q1_map_prime)
                q1 = q1_map_prime[torch.arange(0, phi_size), x_star[:, 0], x_star[:, 1]]
                patch = self.getImgPatch(nxt_o, x_star)
                q2_prime = self.target_phi(patch).reshape(phi_size, self.num_primitives, -1)[
                    torch.arange(0, phi_size), next_states[i]]
                q2 = q2_prime.max(1)[0]
                q_prime = q1 * q2
                q.append(rewards[i] + self.gamma * q_prime * non_final_masks[i])

            q = torch.stack(q)
            max_phi_q = q.max(1)[0]
            max_phi_q_expanded = max_phi_q.reshape(batch_size, 1).expand(batch_size, phi_size)

        q1_output = self.fcn(image)[torch.arange(0, batch_size), states, pixel[:, 0], pixel[:, 1]]
        current_patch = self.getImgPatch(image, pixel).to(self.device)
        q2_output = self.phi_net(current_patch).reshape(batch_size, self.num_primitives, -1)[
            torch.arange(0, batch_size), states]

        non_zero_mask = max_phi_q_expanded != 0
        non_zero_mask = non_zero_mask.float()
        fcn_loss = self.criterion(q1_output, max_phi_q, weights, torch.ones(batch_size).to(self.device))
        max_phi_q_expanded += 1e-4
        phi_loss = self.criterion(q2_output, q/max_phi_q_expanded, weights, non_zero_mask)

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

        with torch.no_grad():
            q1_td_error = torch.abs(q1_output - max_phi_q)
            q2_td_error = torch.abs(q2_output - q/max_phi_q_expanded)
            q2_td_error = q2_td_error * non_zero_mask
            q2_td_error = q2_td_error.view(batch_size, -1).sum(dim=1)

        return (fcn_loss.item(), phi_loss.item()), (q1_td_error+q2_td_error)/2

    def train(self):
        self.fcn.train()
        self.phi_net.train()
        self.target_fcn.eval()
        self.target_phi.eval()

    def eval(self):
        self.fcn.eval()
        self.phi_net.eval()
        self.target_fcn.eval()
        self.target_phi.eval()

    def _loadBatchToDevice(self, batch):
        states = []
        images = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        phis = []
        for d in batch:
            states.append(d.state)
            images.append(d.image)
            xys.append(d.xy)
            rewards.append(d.rewards)
            next_states.append(d.next_states)
            next_obs.append(torch.stack(d.next_obs))
            dones.append(d.dones)
            phis.append(d.phis)
        states_tensor = torch.tensor(states).long().to(self.device)
        image_tensor = torch.tensor(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        xy_tensor = torch.tensor(xys).to(self.device)
        rewards_tensor = torch.tensor(rewards).to(self.device)
        next_states_tensor = torch.tensor(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        dones_tensor = torch.tensor(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        phis_tensor = torch.tensor(phis).long().to(self.device)

        return states_tensor, image_tensor, xy_tensor, rewards_tensor, next_states_tensor, next_obs_tensor, non_final_masks, phis_tensor

    def _loadPrioritizedBatchToDevice(self, batch):
        batch, weights, idxes = batch
        weights = torch.from_numpy(weights).float().to(self.device)
        return self._loadBatchToDevice(batch), weights, idxes