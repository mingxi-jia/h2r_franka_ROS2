from copy import deepcopy
from utils import torch_utils

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

class DQN3LMaxShared:
    def __init__(self, q1, q2, q3, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 num_zs=16, min_z=0.02, max_z=0.17):
        self.lr = lr
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.target_q1 = deepcopy(q1)
        self.target_q2 = deepcopy(q2)
        self.target_q3 = deepcopy(q3)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q3_optimizer = torch.optim.Adam(self.q3.parameters(), lr=self.lr, weight_decay=1e-5)

        self.updateTarget()
        self.gamma = gamma
        self.criterion = torch_utils.WeightedHuberLoss()
        self.num_primitives = num_primitives
        self.action_space = action_space
        self.workspace = workspace
        self.heightmap_resolution = heightmap_resolution
        self.device = device
        self.sl = sl
        self.per = per

        self.heightmap_size = 90
        self.padding = 128
        self.patch_size = patch_size

        self.num_rotations = num_rotations
        self.half_rotation = half_rotation
        self.a2_size = num_rotations
        if self.half_rotation:
            self.rotations = torch.tensor([np.pi / self.num_rotations * i for i in range(self.num_rotations)])
        else:
            self.rotations = torch.tensor([(2 * np.pi) / self.num_rotations * i for i in range(self.num_rotations)])

        self.num_zs = num_zs
        self.z_range = [min_z, max_z]
        self.zs = torch.from_numpy(np.linspace(self.z_range[0], self.z_range[1], self.num_zs)).float()
        self.a3_size = num_zs

        self.loss_calc_dict = {}

    def updateTarget(self):
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q3.load_state_dict(self.q3.state_dict())

    def forwardQ1(self, states, obs, target_net=False, to_cpu=False):
        obs, in_hand = obs
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        padding_width = int((self.padding - obs.size(2)) / 2)
        q1 = self.q1 if not target_net else self.target_q1
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        q_value_maps, obs_encoding = q1(obs, in_hand)
        q_value_maps = q_value_maps[torch.arange(0, states.size(0)), states.long(), padding_width: -padding_width,
                       padding_width: -padding_width]
        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps, obs_encoding

    def forwardQ2(self, states, obs, pixels, obs_encoding, target_net=False, to_cpu=False):
        obs, in_hand = obs
        patch = self.getQ2Input(obs.to(self.device), pixels.to(self.device))
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q2 = self.q2 if not target_net else self.target_q2
        phi_output = q2(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

    def forwardQ3(self, states, obs, pixels, a2, obs_encoding, target_net=False, to_cpu=False):
        obs, in_hand = obs
        patch = self.getQ3Input(obs.to(self.device), pixels.to(self.device), a2.to(self.device))
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q3 = self.q3 if not target_net else self.target_q3
        phi_output = q3(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

    def getQ2Input(self, obs, center_pixel):
        batch_size = obs.size(0)
        img_size = obs.size(2)
        transition = (center_pixel - obs.size(2) / 2).float().flip(1)
        transition_scaled = transition / obs.size(2) * 2

        affine_mat = torch.eye(2).unsqueeze(0).expand(batch_size, -1, -1).float()
        if obs.is_cuda:
            affine_mat = affine_mat.to(self.device)
        affine_mat = torch.cat((affine_mat, transition_scaled.unsqueeze(2).float()), dim=2)
        flow_grid = F.affine_grid(affine_mat, obs.size())
        transformed = F.grid_sample(obs, flow_grid, mode='nearest', padding_mode='zeros')
        patch = transformed[:, :,
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2),
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2)]
        return patch

    def getQ3Input(self, obs, center_pixel, rz):
        rz = rz.cpu()
        rz = self.rotations[rz]
        batch_size = obs.size(0)
        img_size = obs.size(2)
        transition = (center_pixel - obs.size(2) / 2).float().flip(1)
        transition_scaled = transition / obs.size(2) * 2

        affine_mats = []
        for rot in rz:
            affine_mat = np.asarray([[np.cos(rot), np.sin(rot)],
                                     [-np.sin(rot), np.cos(rot)]])
            affine_mat.shape = (2, 2, 1)
            affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float().to(self.device)
            affine_mats.append(affine_mat)
        affine_mat = torch.cat(affine_mats)
        if obs.is_cuda:
            affine_mat = affine_mat.to(self.device)
        affine_mat = torch.cat((affine_mat, transition_scaled.unsqueeze(2).float()), dim=2)
        flow_grid = F.affine_grid(affine_mat, obs.size())
        transformed = F.grid_sample(obs, flow_grid, mode='bilinear', padding_mode='zeros')
        patch = transformed[:, :,
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2),
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2)]
        return patch

    def encodeInHand(self, input_img, in_hand_img):
        if input_img.size(2) == in_hand_img.size(2):
            return torch.cat((input_img, in_hand_img), dim=1)
        else:
            resized_in_hand = F.interpolate(in_hand_img, size=(input_img.size(2), input_img.size(3)),
                                            mode='nearest')
        return torch.cat((input_img, resized_in_hand), dim=1)

    def decodeA2(self, a2):
        rot_id = a2
        rotation = self.rotations[rot_id].reshape(a2.size(0), 1)
        return rot_id.unsqueeze(1), rotation

    def decodeA3(self, a3):
        z_id = a3
        z = self.zs[z_id].reshape(a3.size(0), 1)
        return z_id.unsqueeze(1), z

    def decodeActions(self, pixels, a2_id, a3_id):
        a2_id, a2 = self.decodeA2(a2_id)
        a3_id, a3 = self.decodeA3(a3_id)
        rot_id = a2_id
        z_id = a3_id
        rot = a2
        z = a3
        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(pixels.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(pixels.size(0), 1)
        actions = torch.cat((x, y, z, rot), dim=1)
        action_idx = torch.cat((pixels, rot_id * self.num_zs + z_id), dim=1)
        return action_idx, actions

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.01):
        with torch.no_grad():
            q_value_maps, obs_encoding = self.forwardQ1(states, (obs, in_hand), to_cpu=True)
        pixels = torch_utils.argmax2d(q_value_maps).long()
        with torch.no_grad():
            q2_output = self.forwardQ2(states, (obs, in_hand), pixels, obs_encoding, to_cpu=True)
        a2_id = torch.argmax(q2_output, 1)
        with torch.no_grad():
            q3_output = self.forwardQ3(states, (obs, in_hand), pixels, a2_id, obs_encoding, to_cpu=True)
        a3_id = torch.argmax(q3_output, 1)

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps

        if type(obs) is tuple:
            hm, ih = obs
        else:
            hm = obs
        for i, m in enumerate(rand_mask):
            if m:
                pixel_candidates = torch.nonzero(hm[i, 0]>0.01)
                rand_pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))]
                pixels[i] = rand_pixel

        rand_a2 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a2_size)
        a2_id[rand_mask] = rand_a2.long()

        rand_a3 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a3_size)
        a3_id[rand_mask] = rand_a3.long()

        action_idx, actions = self.decodeActions(pixels, a2_id, a3_id)

        return q_value_maps, action_idx, actions

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        z = plan[:, 2:3]
        rot = plan[:, 3:4]
        states = plan[:, 4:5]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size-1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size-1)

        rot_id = (rot.expand(-1, self.num_rotations) - self.rotations).abs().argmin(1).unsqueeze(1)
        z_id = (z.expand(-1, self.num_zs) - self.zs).abs().argmin(1).unsqueeze(1)

        action_idx, actions = self.decodeActions(torch.cat((pixel_x, pixel_y), dim=1), rot_id, z_id)
        return action_idx, actions

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]
        phis = action_idx[:, 2:]
        a2 = (phis / self.a3_size).long().reshape(phis.size(0), 1)
        a3 = (phis % self.a3_size).long().reshape(phis.size(0), 1)

        with torch.no_grad():
            q1_map_prime, obs_prime_encoding = self.forwardQ1(next_states, next_obs, target_net=True)
            x_star = torch_utils.argmax2d(q1_map_prime)
            q2_prime = self.forwardQ2(next_states, next_obs, x_star, obs_prime_encoding, target_net=True)
            a2_star = torch.argmax(q2_prime, 1)
            q3_prime = self.forwardQ3(next_states, next_obs, x_star, a2_star, obs_prime_encoding, target_net=True)
            q3 = q3_prime.max(1)[0]
            q_prime = q3
            q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q1_output, obs_encoding = self.forwardQ1(states, obs)
        q1_pred = q1_output[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]
        q2_output = self.forwardQ2(states, obs, pixel, obs_encoding)
        q2_pred = q2_output[torch.arange(batch_size), a2[:, 0]]
        q3_output = self.forwardQ3(states, obs, pixel, a2, obs_encoding)
        q3_pred = q3_output[torch.arange(batch_size), a3[:, 0]]

        self.loss_calc_dict['q1_output'] = q1_output
        self.loss_calc_dict['q2_output'] = q2_output
        self.loss_calc_dict['q3_output'] = q3_output

        # with torch.no_grad():
        #     q2_target_net_output = self.forwardQ2(states, obs, pixel, obs_encoding, target_net=True).max(1)[0]
        #     q3_target_net_output = self.forwardQ3(states, obs, pixel, a2, obs_encoding, target_net=True).max(1)[0]
        #
        # q1_target = torch.stack((q2_target_net_output, q3_target_net_output, q_target), dim=1).max(1)[0]
        # q2_target = torch.stack((q3_target_net_output, q_target), dim=1).max(1)[0]

        q1_td_loss = F.smooth_l1_loss(q1_pred, q_target)
        q2_td_loss = F.smooth_l1_loss(q2_pred, q_target)
        q3_td_loss = F.smooth_l1_loss(q3_pred, q_target)
        td_loss = q1_td_loss + q2_td_loss + q3_td_loss

        # num_x = 8
        # heightmap_size = 90
        # q1_map_flatten = q1_output.reshape(batch_size, -1)
        # topk_x = q1_map_flatten.topk(num_x)[1]
        # top_x = torch.stack((topk_x / heightmap_size, topk_x % heightmap_size), dim=2).permute(1, 0, 2)
        # q1_topk_outputs = []
        # q1_topk_targets = []
        # for x in top_x:
        #     with torch.no_grad():
        #         q2_target_output = self.forwardQ2(states, obs, x, obs_encoding, target_net=True).max(1)[0]
        #     q1_topk_targets.append(q2_target_output)
        #     q1_topk_outputs.append(q1_output[torch.arange(0, batch_size), x[:, 0], x[:, 1]])
        # q1_topk_targets = torch.stack(q1_topk_targets, dim=1).mean(1)
        # q1_topk_outputs = torch.stack(q1_topk_outputs, dim=1).mean(1)
        # combined_q1_pred = torch.stack((q1_pred, q1_topk_outputs), dim=1)
        # combined_q1_target = torch.stack((q1_target, q1_topk_targets), dim=1)
        #
        # num_a2 = 4
        # q2_all_flatten = q2_output.reshape(batch_size, -1)
        # topk_a2 = q2_all_flatten.topk(num_a2)[1].permute(1, 0)
        # q2_topk_outputs = []
        # q2_topk_targets = []
        # for a2 in topk_a2:
        #     with torch.no_grad():
        #         q3_target_output = self.forwardQ3(states, obs, pixel, a2, obs_encoding, target_net=True).max(1)[0]
        #     q2_topk_targets.append(q3_target_output)
        #     q2_topk_outputs.append(q2_output[torch.arange(0, batch_size), a2])
        # q2_topk_targets = torch.stack(q2_topk_targets, dim=1).mean(1)
        # q2_topk_outputs = torch.stack(q2_topk_outputs, dim=1).mean(1)
        # combined_q2_pred = torch.stack((q2_pred, q2_topk_outputs), dim=1)
        # combined_q2_target = torch.stack((q2_target, q2_topk_targets), dim=1)
        #
        # q1_td_loss = F.smooth_l1_loss(combined_q1_pred, combined_q1_target)
        # q2_td_loss = F.smooth_l1_loss(combined_q2_pred, combined_q2_target)
        # q3_td_loss = F.smooth_l1_loss(q3_pred, q_target)
        # td_loss = q1_td_loss + q2_td_loss + q3_td_loss

        with torch.no_grad():
            td_error = torch.abs(q3_pred - q_target)

        return td_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.q3_optimizer.zero_grad()
        td_loss.backward()

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

        return td_loss.item(), td_error


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
        step_lefts = []
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
            step_lefts.append(d.step_left)
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
        step_lefts_tensor = torch.stack(step_lefts).to(self.device)
        is_experts_tensor = torch.stack(is_experts).bool().to(self.device)

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = (image_tensor, in_hand_tensor)
        self.loss_calc_dict['action_idx'] = xy_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = (next_obs_tensor, next_in_hands_tensor)
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor

        return states_tensor, (image_tensor, in_hand_tensor), xy_tensor, rewards_tensor, next_states_tensor, \
               (next_obs_tensor, next_in_hands_tensor), non_final_masks, step_lefts_tensor, is_experts_tensor

    def _loadLossCalcDict(self):
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']
        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def loadModel(self, path_pre):
        q1_path = path_pre + '_q1.pt'
        print('loading {}'.format(q1_path))
        self.q1.load_state_dict(torch.load(q1_path, map_location='cuda:0'))
        q2_path = path_pre + '_q2.pt'
        print('loading {}'.format(q2_path))
        self.q2.load_state_dict(torch.load(q2_path, map_location='cuda:0'))
        q3_path = path_pre + '_q3.pt'
        print('loading {}'.format(q3_path))
        self.q3.load_state_dict(torch.load(q3_path, map_location='cuda:0'))
        self.updateTarget()

    def saveModel(self, path):
        torch.save(self.q1.state_dict(), '{}_q1.pt'.format(path))
        torch.save(self.q2.state_dict(), '{}_q2.pt'.format(path))
        torch.save(self.q3.state_dict(), '{}_q3.pt'.format(path))

    def getSaveState(self):
        return {
            'q1': self.q1.state_dict(),
            'q1_target': self.q1.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2': self.q2.state_dict(),
            'q2_target': self.q2.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'q3': self.q3.state_dict(),
            'q3_target': self.q3.state_dict(),
            'q3_optimizer': self.q3_optimizer.state_dict()
        }

    def loadFromState(self, save_state):
        self.q1.load_state_dict(save_state['q1'])
        self.target_q1.load_state_dict(save_state['q1_target'])
        self.q1_optimizer.load_state_dict(save_state['q1_optimizer'])

        self.q2.load_state_dict(save_state['q2'])
        self.target_q2.load_state_dict(save_state['q2_target'])
        self.q2_optimizer.load_state_dict(save_state['q2_optimizer'])

        self.q3.load_state_dict(save_state['q3'])
        self.target_q3.load_state_dict(save_state['q3_target'])
        self.q3_optimizer.load_state_dict(save_state['q3_optimizer'])


    def train(self):
        self.q1.train()
        self.q2.train()
        self.q3.train()

    def eval(self):
        self.q1.eval()
        self.q2.eval()
        self.q3.eval()

    def getModelStr(self):
        return 'q1:' + str(self.q1) + '\n' + 'q2:' + str(self.q2) + '\n' + 'q3:' + str(self.q3)

    def getCurrentObs(self, in_hand, obs):
        obss = []
        for i, o in enumerate(obs):
            obss.append((o.squeeze(), in_hand[i].squeeze()))
        return obss