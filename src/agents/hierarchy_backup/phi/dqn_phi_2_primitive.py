from copy import deepcopy
from agents.hierarchy_backup.phi.dqn_phi_2 import DQNPhi2
from utils import torch_utils

import numpy as np

import torch
import torch.nn.functional as F

class DQNPhi2Primitive(DQNPhi2):
    def __init__(self, pri_net, *args, **kwargs):
        super(DQNPhi2Primitive, self).__init__(*args, **kwargs)
        # TODO: separate num_primitive_states and num_primitive_actions
        self.pri_net = pri_net
        self.target_pri = deepcopy(pri_net)
        # if small_net:
        #     self.pri_net = CNNSmall((num_input_channel+num_primitives, action_space[1], action_space[1]), num_primitives).to(device)
        #     self.target_pri = CNNSmall((num_input_channel+num_primitives, action_space[1], action_space[1]), num_primitives).to(device)
        # else:
        #     self.pri_net = CNN((num_input_channel+num_primitives, action_space[1], action_space[1]), num_primitives).to(device)
        #     self.target_pri = CNN((num_input_channel+num_primitives, action_space[1], action_space[1]), num_primitives).to(device)
        self.pri_optimizer = torch.optim.Adam(self.pri_net.parameters(), lr=self.lr)
        self.updateTargetPri()

    def updateTargetPri(self):
        self.target_pri.load_state_dict(self.pri_net.state_dict())

    def encodeInput(self, states, img):
        states_one_hot = torch.eye(self.num_primitives).unsqueeze(2).\
            reshape(1, self.num_primitives, self.num_primitives, 1).\
            expand(states.size(0), -1, -1, -1)[torch.arange(states.size(0)), states.long()]
        states_reshape = states_one_hot.unsqueeze(-1)
        states_expanded = states_reshape.expand(-1, -1, img.size(2), img.size(3))
        if img.is_cuda:
            states_expanded = states_expanded.to(self.device)
        return torch.cat((img, states_expanded.float()), dim=1)

    def getEGreedyPrimitiveActions(self, states, obs, eps):
        pri_input = self.encodeInput(states, obs)
        with torch.no_grad():
            pri_output = self.pri_net(pri_input.to(self.device)).cpu()
        pri = torch.argmax(pri_output, 1)
        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_pri = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.num_primitives)
        pri[rand_mask] = rand_pri.long()
        return pri.float()

    def updateSamplePhi(self, batch):
        states, image, pixel, primitives, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(batch)
        batch_size = image.size(0)
        heightmap_size = image.size(2)
        phi_size = phis.size(1)

        q = []
        with torch.no_grad():
            for i in range(batch_size):
                nxt_o = next_obs[i]
                nxt_s = next_states[i]
                pri_input = self.encodeInput(nxt_s, nxt_o)
                pri_output = self.target_pri(pri_input)
                pri_star = torch.argmax(pri_output, dim=1)
                q1_map_prime = self.target_fcn(nxt_o)[torch.arange(0, phi_size), pri_star].reshape(phi_size,
                                                                                                        heightmap_size,
                                                                                                        heightmap_size)
                x_star = torch_utils.argmax2d(q1_map_prime)
                patch = self.getImgPatch(nxt_o, x_star)
                q2_prime = self.target_phi(patch).reshape(phi_size, self.num_primitives, -1)[
                    torch.arange(0, phi_size), next_states[i]]
                q2 = q2_prime.max(1)[0]
                q.append(rewards[i] + self.gamma * q2 * non_final_masks[i])

            q = torch.stack(q)
            max_phi_q = q.max(1)[0]

        q0_output = self.pri_net(self.encodeInput(states, image))[torch.arange(0, batch_size), primitives]
        q1_output = self.fcn(image)[torch.arange(0, batch_size), primitives, pixel[:, 0], pixel[:, 1]]
        current_patch = self.getImgPatch(image, pixel).to(self.device)
        q2_output = self.phi_net(current_patch).reshape(batch_size, self.num_primitives, -1)[
            torch.arange(0, batch_size), primitives]
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)

        with torch.no_grad():
            q1_target_output = self.target_fcn(image)[torch.arange(0, batch_size), primitives].reshape(batch_size, -1)
            q0_target = q1_target_output.max(1)[0]

            q2_target_output = self.target_phi(current_patch).reshape(batch_size, self.num_primitives, -1)[
                torch.arange(0, batch_size), primitives]
            q1_target = q2_target_output.max(1)[0]

        pri_loss = F.smooth_l1_loss(q0_output, q0_target)
        fcn_loss = F.smooth_l1_loss(q1_output, q1_target)
        phi_loss = F.smooth_l1_loss(q2_output, q)

        self.pri_optimizer.zero_grad()
        pri_loss.backward()
        for param in self.pri_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.pri_optimizer.step()

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

        return (pri_loss.item(), fcn_loss.item(), phi_loss.item()), None

    def updateMultipleX(self, batch, num_x=10):
        states, image, pixel, primitives, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(
            batch)
        batch_size = image.size(0)
        heightmap_size = image.size(2)
        phi_size = phis.size(1)

        q = []
        with torch.no_grad():
            for i in range(batch_size):
                nxt_o = next_obs[i]
                nxt_s = next_states[i]
                pri_input = self.encodeInput(nxt_s, nxt_o)
                pri_output = self.target_pri(pri_input)
                pri_star = torch.argmax(pri_output, dim=1)
                q1_map_prime = self.target_fcn(nxt_o)[torch.arange(0, phi_size), pri_star].reshape(phi_size,
                                                                                                   heightmap_size,
                                                                                                   heightmap_size)
                x_star = torch_utils.argmax2d(q1_map_prime)
                patch = self.getImgPatch(nxt_o, x_star)
                q2_prime = self.target_phi(patch).reshape(phi_size, self.num_primitives, -1)[
                    torch.arange(0, phi_size), next_states[i]]
                q2 = q2_prime.max(1)[0]
                q.append(rewards[i] + self.gamma * q2 * non_final_masks[i])

            q = torch.stack(q)

        q0_output = self.pri_net(self.encodeInput(states, image))
        current_patch = self.getImgPatch(image, pixel).to(self.device)
        q2_output = self.phi_net(current_patch).reshape(batch_size, self.num_primitives, -1)[
            torch.arange(0, batch_size), primitives]
        q2_output = torch.cat([q2_output[torch.arange(batch_size), phis[:, i]].reshape(-1, 1) for i in range(phi_size)],
                              dim=1)

        q1_map = self.fcn(image)[torch.arange(0, batch_size), primitives].reshape(batch_size, heightmap_size,
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
                torch.arange(0, batch_size), primitives]
            q1_targets.append(q2_target_output.max(1)[0])
            q1_outputs.append(q1_map[torch.arange(0, batch_size), x[:, 0], x[:, 1]])

        q1_targets = torch.stack(q1_targets, dim=1)
        q1_outputs = torch.stack(q1_outputs, dim=1)

        q1_target_output = self.target_fcn(image).reshape(batch_size, self.num_primitives, -1)
        q0_target = q1_target_output.max(2)[0]

        pri_loss = F.smooth_l1_loss(q0_output, q0_target)
        fcn_loss = F.smooth_l1_loss(q1_outputs, q1_targets)
        phi_loss = F.smooth_l1_loss(q2_output, q)

        self.pri_optimizer.zero_grad()
        pri_loss.backward()
        for param in self.pri_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.pri_optimizer.step()

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

        return (pri_loss.item(), fcn_loss.item(), phi_loss.item()), None

    def update(self, batch):
        states, image, pixel, primitives, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(batch)
        batch_size = image.size(0)
        heightmap_size = image.size(2)
        with torch.no_grad():
            pri_input = self.encodeInput(next_states, next_obs)
            pri_output = self.target_pri(pri_input)
            pri_star = torch.argmax(pri_output, dim=1)

            q1_map_prime = self.target_fcn(next_obs)[torch.arange(0, batch_size), pri_star].reshape(batch_size,
                                                                                                     heightmap_size,
                                                                                                     heightmap_size)
            x_star = torch_utils.argmax2d(q1_map_prime)
            patch = self.getImgPatch(next_obs, x_star)
            q2_prime = self.target_phi(patch).reshape(batch_size, self.num_primitives, -1)[
                torch.arange(0, batch_size), pri_star]
            q2_target = rewards + self.gamma * q2_prime.max(1)[0] * non_final_masks

        q0_output = self.pri_net(self.encodeInput(states, image))[torch.arange(0, batch_size), primitives]
        q1_output = self.fcn(image)[torch.arange(0, batch_size), primitives, pixel[:, 0], pixel[:, 1]]
        current_patch = self.getImgPatch(image, pixel).to(self.device)
        q2_output = self.phi_net(current_patch).reshape(batch_size, self.num_primitives, -1)[
            torch.arange(0, batch_size), primitives, phis]

        with torch.no_grad():
            q1_target_output = self.target_fcn(image)[torch.arange(0, batch_size), primitives].reshape(batch_size, -1)
            q0_target = q1_target_output.max(1)[0]

            q2_target_output = self.target_phi(current_patch).reshape(batch_size, self.num_primitives, -1)[
                torch.arange(0, batch_size), primitives]
            q1_target = q2_target_output.max(1)[0]

        pri_loss = F.smooth_l1_loss(q0_output, q0_target)
        fcn_loss = F.smooth_l1_loss(q1_output, q1_target)
        phi_loss = F.smooth_l1_loss(q2_output, q2_target)

        self.pri_optimizer.zero_grad()
        pri_loss.backward()
        for param in self.pri_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.pri_optimizer.step()

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

        return (pri_loss.item(), fcn_loss.item(), phi_loss.item()), None

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
