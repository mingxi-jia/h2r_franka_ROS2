from agents.hierarchy_backup.phi.dqn_phi import DQNPhi
from utils import torch_utils

import torch
import torch.nn.functional as F

class DQNPhi2(DQNPhi):
    def __init__(self, *args, **kwargs):
        self.num_top_x = kwargs.pop('num_top_x', 0)
        # self.sl = kwargs.pop('sl', False)
        self.update_max = kwargs.pop('max', False)
        super(DQNPhi2, self).__init__(*args, **kwargs)
        self.num_pos_candidate = self.action_space[1] ** 2

    def update(self, batch):
        if self.sl:
            return self.updateSL(batch, min(self.num_pos_candidate, self.num_top_x))
        elif self.update_max:
            return self.updateMax(batch)
        else:
            return self.updateMultipleX(batch, min(self.num_pos_candidate, self.num_top_x))

    def setCandidatePos(self, pos_candidate):
        super().setCandidatePos(pos_candidate)
        self.num_pos_candidate = len(pos_candidate[0]) * len(pos_candidate[1])

    def updateSamplePhi(self, batch, topkx=False, k=1):
        states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(batch)
        batch_size = image.size(0)
        phi_size = phis.size(1)
        channel_size = image.size(1)
        heightmap_size = image.size(2)

        q = []
        with torch.no_grad():
            for i in range(batch_size):
                if not topkx:
                    q1_map_prime = self.target_fcn(next_obs[i])[torch.arange(0, phi_size), next_states[i]].reshape(phi_size,
                                                                                                             heightmap_size,
                                                                                                             heightmap_size)
                    x_star = torch_utils.argmax2d(q1_map_prime)
                    patch = self.getImgPatch(next_obs[i], x_star)
                    q2_prime = self.target_phi(patch).reshape(phi_size, self.num_primitives, -1)[
                        torch.arange(0, phi_size), next_states[i]]
                    q2 = q2_prime.max(1)[0]
                    q.append(rewards[i] + self.gamma * q2 * non_final_masks[i])
                else:
                    q.append(self.getQ2WithTopKX(next_states[i], next_obs[i], rewards[i], non_final_masks[i], k))

            q = torch.stack(q)
            # max_phi_q = q.max(1)[0]

        q1_output = self.fcn(image)[torch.arange(0, batch_size), states, pixel[:, 0], pixel[:, 1]]
        current_patch = self.getImgPatch(image, pixel).to(self.device)
        q2_output = self.phi_net(current_patch).reshape(batch_size, self.num_primitives, -1)[
            torch.arange(0, batch_size), states]
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)

        with torch.no_grad():
            q2_target_output = self.target_phi(current_patch).reshape(batch_size, self.num_primitives, -1)[
                torch.arange(0, batch_size), states]
            q2_target_output_max = q2_target_output.max(1)[0]
            q1_target = q2_target_output_max

        fcn_loss = F.smooth_l1_loss(q1_output, q1_target)
        phi_loss = F.smooth_l1_loss(q2_output, q)

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

    def getQ2WithTopKX(self, next_states, next_obs, rewards, non_final_masks, k=5):
        batch_size = next_obs.size(0)
        heightmap_size = next_obs.size(2)
        q1_map_prime = self.target_fcn(next_obs)[torch.arange(0, batch_size), next_states].reshape(batch_size,
                                                                                                   heightmap_size,
                                                                                                   heightmap_size)
        q1_map_prime_flatten = q1_map_prime.reshape(batch_size, -1)
        top_k_x_star = q1_map_prime_flatten.topk(k)[1]
        top_x = torch.stack((top_k_x_star / heightmap_size, top_k_x_star % heightmap_size), dim=2).permute(1, 0, 2)

        q2_estimates = []
        for x in top_x:
            patch = self.getImgPatch(next_obs, x)
            q2_prime = self.target_phi(patch).reshape(batch_size, self.num_primitives, -1)[
                torch.arange(0, batch_size), next_states]
            q2 = rewards + self.gamma * q2_prime.max(1)[0] * non_final_masks
            q2_estimates.append(q2)
        q2_estimates = torch.stack(q2_estimates)
        q2 = q2_estimates.max(0)[0]
        return q2

    def reshapeNextObs(self, next_obs, batch_size):
        next_obs = next_obs.reshape(batch_size, next_obs.shape[-3], next_obs.shape[-2], next_obs.shape[-1])
        return next_obs

    def updateSL(self, batch, num_x=10):
        states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(
            batch)
        batch_size = states.size(0)
        phi_size = phis.size(1)
        heightmap_size = self.action_space[1]

        non_final_masks[rewards==0] = 1
        q = self.gamma ** rewards * non_final_masks

        max_phi_q = q.max(1)[0]

        q2_output = self.forwardPhiNet(states, image, pixel)
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)

        q1_output = self.forwardFCN(states, image)[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]

        fcn_loss = F.smooth_l1_loss(q1_output, max_phi_q)
        phi_loss = F.smooth_l1_loss(q2_output, q)

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

    def updateMultipleX(self, batch, num_x=10):
        states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(
            batch)
        batch_size = states.size(0)
        phi_size = phis.size(1)
        heightmap_size = self.action_space[1]

        if self.iter_update:
            q = []
            with torch.no_grad():
                for i in range(batch_size):
                    q1_map_prime = self.forwardFCN(next_states[i], next_obs[i], target_net=True)
                    x_star = torch_utils.argmax2d(q1_map_prime)
                    q2_prime = self.forwardPhiNet(next_states[i], next_obs[i], x_star, target_net=True)
                    q2 = q2_prime.max(1)[0]
                    q.append(rewards[i] + self.gamma * q2 * non_final_masks[i])

                q = torch.stack(q)

        else:
            next_obs = self.reshapeNextObs(next_obs, batch_size*phi_size)
            next_states = next_states.reshape(batch_size * phi_size)
            with torch.no_grad():
                q1_map_prime = self.forwardFCN(next_states, next_obs, target_net=True)
                x_star = torch_utils.argmax2d(q1_map_prime)
                q2_prime = self.forwardPhiNet(next_states, next_obs, x_star, target_net=True)
                q2 = q2_prime.max(1)[0]
                q2 = q2.reshape(batch_size, phi_size)
                q = rewards + self.gamma * q2 * non_final_masks

        q2_output = self.forwardPhiNet(states, image, pixel)
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)

        q1_map = self.forwardFCN(states, image)
        q1_map_flatten = q1_map.reshape(batch_size, -1)
        topk_x = q1_map_flatten.topk(num_x)[1]
        top_x = torch.stack((topk_x / heightmap_size, topk_x % heightmap_size), dim=2).permute(1, 0, 2)
        xs = torch.cat((top_x, pixel.unsqueeze(0)))
        q1_outputs = []
        q1_targets = []
        for x in xs:
            with torch.no_grad():
                q2_target_output = self.forwardPhiNet(states, image, x, target_net=True)
            q1_targets.append(q2_target_output.max(1)[0])
            q1_outputs.append(q1_map[torch.arange(0, batch_size), x[:, 0], x[:, 1]])

        q1_targets = torch.stack(q1_targets, dim=1)
        q1_outputs = torch.stack(q1_outputs, dim=1)

        fcn_loss = F.smooth_l1_loss(q1_outputs, q1_targets)
        phi_loss = F.smooth_l1_loss(q2_output, q)

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

    def updateMax(self, batch):
        states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(
            batch)
        batch_size = states.size(0)
        phi_size = phis.size(1)
        heightmap_size = self.action_space[1]

        if self.iter_update:
            q = []
            with torch.no_grad():
                for i in range(batch_size):
                    q1_map_prime = self.forwardFCN(next_states[i], next_obs[i], target_net=True)
                    x_star = torch_utils.argmax2d(q1_map_prime)
                    q2_prime = self.forwardPhiNet(next_states[i], next_obs[i], x_star, target_net=True)
                    q2 = q2_prime.max(1)[0]
                    q.append(rewards[i] + self.gamma * q2 * non_final_masks[i])

                q = torch.stack(q)

        else:
            next_obs = self.reshapeNextObs(next_obs, batch_size*phi_size)
            next_states = next_states.reshape(batch_size * phi_size)
            with torch.no_grad():
                q1_map_prime = self.forwardFCN(next_states, next_obs, target_net=True)
                x_star = torch_utils.argmax2d(q1_map_prime)
                q2_prime = self.forwardPhiNet(next_states, next_obs, x_star, target_net=True)
                q2 = q2_prime.max(1)[0]
                q2 = q2.reshape(batch_size, phi_size)
                q = rewards + self.gamma * q2 * non_final_masks

        max_phi_q = q.max(1)[0]

        q2_output = self.forwardPhiNet(states, image, pixel)
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)

        q1_output = self.forwardFCN(states, image)[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]

        fcn_loss = F.smooth_l1_loss(q1_output, max_phi_q)
        phi_loss = F.smooth_l1_loss(q2_output, q)

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

    def updatePrioritized(self, batch):
        (states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis), weights, idxes= self._loadPrioritizedBatchToDevice(batch)

        batch_size = image.size(0)
        phi_size = phis.size(1)
        channel_size = image.size(1)
        heightmap_size = image.size(2)

        q = []
        with torch.no_grad():
            for i in range(batch_size):
                q1_map_prime = self.target_fcn(next_obs[i])[torch.arange(0, phi_size), next_states[i]].reshape(phi_size,
                                                                                                               heightmap_size,
                                                                                                               heightmap_size)
                x_star = torch_utils.argmax2d(q1_map_prime)
                patch = self.getImgPatch(next_obs[i], x_star)
                q2_prime = self.target_phi(patch).reshape(phi_size, self.num_primitives, -1)[
                    torch.arange(0, phi_size), next_states[i]]
                q2 = q2_prime.max(1)[0]
                q.append(rewards[i] + self.gamma * q2 * non_final_masks[i])

            q = torch.stack(q)

        q1_output = self.fcn(image)[torch.arange(0, batch_size), states, pixel[:, 0], pixel[:, 1]]
        current_patch = self.getImgPatch(image, pixel).to(self.device)
        q2_output = self.phi_net(current_patch).reshape(batch_size, self.num_primitives, -1)[
            torch.arange(0, batch_size), states]
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)

        with torch.no_grad():
            q2_target_output = self.target_phi(current_patch).reshape(batch_size, self.num_primitives, -1)[
                torch.arange(0, batch_size), states]
            q2_target_output_max = q2_target_output.max(1)[0]
            q1_target = q2_target_output_max

        fcn_loss = F.smooth_l1_loss(q1_output, q1_target)
        phi_loss = F.smooth_l1_loss(q2_output, q)


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
            td_error = torch.abs(q2_output - q)
            td_error = td_error.view(batch_size, -1).sum(dim=1)

        return (fcn_loss.item(), phi_loss.item()), td_error

    def updatePrioritizedMultipleX(self, batch, num_x=10):
        (states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis), weights, idxes= self._loadPrioritizedBatchToDevice(batch)

        batch_size = image.size(0)
        phi_size = phis.size(1)
        channel_size = image.size(1)
        heightmap_size = image.size(2)

        q = []
        with torch.no_grad():
            for i in range(batch_size):
                q1_map_prime = self.target_fcn(next_obs[i])[torch.arange(0, phi_size), next_states[i]].reshape(phi_size,
                                                                                                               heightmap_size,
                                                                                                               heightmap_size)
                x_star = torch_utils.argmax2d(q1_map_prime)
                patch = self.getImgPatch(next_obs[i], x_star)
                q2_prime = self.target_phi(patch).reshape(phi_size, self.num_primitives, -1)[
                    torch.arange(0, phi_size), next_states[i]]
                q2 = q2_prime.max(1)[0]
                q.append(rewards[i] + self.gamma * q2 * non_final_masks[i])

            q = torch.stack(q)

        current_patch = self.getImgPatch(image, pixel).to(self.device)
        q2_output = self.phi_net(current_patch).reshape(batch_size, self.num_primitives, -1)[
            torch.arange(0, batch_size), states]
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)

        q1_map = self.fcn(image)[torch.arange(0, batch_size), states].reshape(batch_size, heightmap_size,
                                                                              heightmap_size)
        q1_map_flatten = q1_map.reshape(batch_size, -1)
        topk_x = q1_map_flatten.topk(num_x)[1]
        top_x = torch.stack((topk_x / heightmap_size, topk_x % heightmap_size), dim=2).permute(1, 0, 2)
        xs = torch.cat((top_x, pixel.unsqueeze(0)))
        q1_outputs = []
        q1_targets = []
        for x in xs:
            patch = self.getImgPatch(image, x)
            q2_target_output = self.target_phi(patch).reshape(batch_size, self.num_primitives, -1)[
                torch.arange(0, batch_size), states]
            q1_targets.append(q2_target_output.max(1)[0])
            q1_outputs.append(q1_map[torch.arange(0, batch_size), x[:, 0], x[:, 1]])

        q1_targets = torch.stack(q1_targets, dim=1)
        q1_outputs = torch.stack(q1_outputs, dim=1)

        fcn_loss = self.criterion(q1_outputs, q1_targets, weights, torch.ones_like(q1_outputs).to(self.device))
        phi_loss = self.criterion(q2_output, q, weights, torch.ones_like(q2_output).to(self.device))

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
            td_error = torch.abs(q2_output - q)
            td_error = td_error.view(batch_size, -1).sum(dim=1)

        return (fcn_loss.item(), phi_loss.item()), td_error

    # def _loadBatchToDevice(self, batch):
    #     states = []
    #     images = []
    #     xys = []
    #     rewards = []
    #     next_states = []
    #     next_obs = []
    #     dones = []
    #     phis = []
    #     for d in batch:
    #         states.append(d.state)
    #         images.append(d.image)
    #         xys.append(d.xy)
    #         rewards.append(d.rewards[0])
    #         next_states.append(d.next_states[0])
    #         next_obs.append(d.next_obs[0])
    #         dones.append(d.dones[0])
    #         phis.append(d.phis[0])
    #     states_tensor = torch.tensor(states).long().to(self.device)
    #     image_tensor = torch.tensor(images).unsqueeze(1).to(self.device)
    #     xy_tensor = torch.tensor(xys).to(self.device)
    #     rewards_tensor = torch.tensor(rewards).to(self.device)
    #     next_states_tensor = torch.tensor(next_states).long().to(self.device)
    #     next_obs_tensor = torch.stack(next_obs).to(self.device)
    #     dones_tensor = torch.tensor(dones).int()
    #     non_final_masks = (dones_tensor ^ 1).float().to(self.device)
    #     phis_tensor = torch.tensor(phis).long().to(self.device)
    #
    #     return states_tensor, image_tensor, xy_tensor, rewards_tensor, next_states_tensor, next_obs_tensor, non_final_masks, phis_tensor
