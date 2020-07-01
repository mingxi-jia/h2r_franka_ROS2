from copy import deepcopy
from utils import torch_utils

import numpy as np

import torch
import torch.nn.functional as F

class DQNRotBase:
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                num_primitives = 1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):

        self.lr = lr
        self.fcn = fcn
        self.phi_net = cnn
        self.target_fcn = deepcopy(fcn)
        self.target_phi = deepcopy(cnn)
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.phi_optimizer = torch.optim.Adam(self.phi_net.parameters(), lr=self.lr, weight_decay=1e-5)

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
        self.phi_size = num_rotations
        if self.half_rotation:
            self.rotations = torch.tensor([np.pi / self.num_rotations * i for i in range(self.num_rotations)])
        else:
            self.rotations = torch.tensor([(2 * np.pi) / self.num_rotations * i for i in range(self.num_rotations)])

        self.loss_calc_dict = {}

    def updateTarget(self):
        self.target_fcn.load_state_dict(self.fcn.state_dict())
        self.target_phi.load_state_dict(self.phi_net.state_dict())

    def getImgPatch(self, obs, center_pixel):
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

    def forwardFCN(self, states, obs, target_net=False, to_cpu=False):
        fcn = self.fcn if not target_net else self.target_fcn
        obs = obs.to(self.device)
        padding_width = int((self.padding - obs.size(2)) / 2)
        # pad obs
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        q_value_maps = fcn(obs)[torch.arange(0, states.size(0)), states.long()]
        q_value_maps = q_value_maps[:, :, padding_width: -padding_width, padding_width: -padding_width]

        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps

    def forwardPhiNet(self, states, obs, pixels, target_net=False, to_cpu=False):
        patch = self.getImgPatch(obs.to(self.device), pixels.to(self.device))
        phi_net = self.phi_net if not target_net else self.target_phi
        phi_output = phi_net(patch).reshape(states.size(0), self.num_primitives, -1)[torch.arange(0, states.size(0)),
                                                                                     states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

    def decodePhi(self, phi):
        rot_id = phi
        rotation = self.rotations[rot_id].reshape(phi.size(0), 1)
        return rot_id.unsqueeze(1), rotation

    def getEGreedyActions(self, states, obs, eps, coef=0.01):
        with torch.no_grad():
            q_value_maps = self.forwardFCN(states, obs, to_cpu=True)
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        pixels = torch_utils.argmax2d(q_value_maps).long()
        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        with torch.no_grad():
            phi_output = self.forwardPhiNet(states, obs, pixels, to_cpu=True)

        phi = torch.argmax(phi_output, 1)
        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps
        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.phi_size)
        phi[rand_mask] = rand_phi.long()
        phi_idx, phi_a = self.decodePhi(phi)
        actions = torch.cat((x, y, phi_a), dim=1)
        action_idx = torch.cat((pixels, phi_idx), dim=1)

        return q_value_maps, action_idx, actions

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        rot = plan[:, 2:3]
        states = plan[:, 3:4]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size-1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size-1)
        rot_id = (rot.expand(-1, self.num_rotations) - self.rotations).abs().argmin(1).unsqueeze(1)

        x = (pixel_x.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_y.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        rot = self.rotations[rot_id]
        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, rot_id), dim=1)
        return action_idx, actions

    def calcTDLoss(self):
        raise NotImplementedError

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.fcn_optimizer.zero_grad()
        self.phi_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.phi_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_optimizer.step()

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
        fcn_path = path_pre + '_fcn.pt'
        print('loading {}'.format(fcn_path))
        self.fcn.load_state_dict(torch.load(fcn_path))
        cnn_path = path_pre + '_cnn.pt'
        print('loading {}'.format(cnn_path))
        self.phi_net.load_state_dict(torch.load(cnn_path))
        self.updateTarget()

    def saveModel(self, path):
        torch.save(self.fcn.state_dict(), '{}_fcn.pt'.format(path))
        torch.save(self.phi_net.state_dict(), '{}_cnn.pt'.format(path))

    def getSaveState(self):
        return {
            'fcn': self.fcn.state_dict(),
            'fcn_target': self.target_fcn.state_dict(),
            'fcn_optimizer': self.fcn_optimizer.state_dict(),
            'cnn': self.phi_net.state_dict(),
            'cnn_target': self.target_phi.state_dict(),
            'cnn_optimizer': self.phi_optimizer.state_dict()
        }

    def loadFromState(self, save_state):
        self.fcn.load_state_dict(save_state['fcn'])
        self.target_fcn.load_state_dict(save_state['fcn_target'])
        self.fcn_optimizer.load_state_dict(save_state['fcn_optimizer'])

        self.phi_net.load_state_dict(save_state['cnn'])
        self.target_phi.load_state_dict(save_state['cnn_target'])
        self.phi_optimizer.load_state_dict(save_state['cnn_optimizer'])

    def train(self):
        self.fcn.train()
        self.phi_net.train()

    def eval(self):
        self.fcn.eval()
        self.phi_net.eval()

    def getModelStr(self):
        return 'fcn:' + str(self.fcn) + '\n' + 'cnn:' + str(self.phi_net)
