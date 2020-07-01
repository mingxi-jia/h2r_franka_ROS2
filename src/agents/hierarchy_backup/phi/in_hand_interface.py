import torch
import torch.nn.functional as F

import numpy as np
from scipy import ndimage

from agents.hierarchy_backup.phi.his_base import HisBase
from utils import torch_utils

class InHandInterface(HisBase):
    def __init__(self):
        HisBase.__init__(self)
        self.empty_in_hand = torch.zeros((1, 1, self.patch_size, self.patch_size))

    def getCurrentObs(self, obs):
        obss = []
        for i, o in enumerate(obs):
            obss.append((o.squeeze().numpy(), self.his[i].squeeze().numpy()))
        return obss

    def getNextObs(self, patch, rotation, height, states_, obs_, dones):
        in_hand_img_ = self.getInHandImage(patch, rotation, height).cpu()
        in_hand_img_[1 - states_.bool()] = self.empty_in_hand.clone()
        in_hand_img_[dones.bool()] = self.empty_in_hand.clone()
        obss_ = []
        for i, o in enumerate(obs_):
            obss_.append((o, in_hand_img_[i]))
        return obss_

    def updateHis(self, patch, rotation, height, states_, obs_, dones):
        in_hand_img_ = self.getInHandImage(patch, rotation, height).cpu()
        in_hand_img_[1 - states_.bool()] = self.empty_in_hand.clone()
        in_hand_img_[dones.bool()] = self.empty_in_hand.clone()
        self.his = in_hand_img_

    def encodeInHand(self, input_img, in_hand_img):
        if input_img.size(2) == in_hand_img.size(2):
            return torch.cat((input_img, in_hand_img), dim=1)
        else:
            resized_in_hand = F.interpolate(in_hand_img, size=(input_img.size(2), input_img.size(3)),
                                            mode='nearest')
        return torch.cat((input_img, resized_in_hand), dim=1)

    def getInHandImage(self, patch, rot, z):
        with torch.no_grad():
            patch = patch.cpu().numpy()
            in_hand_imgs = []
            for i, img in enumerate(patch):
                depth_heightmap = np.squeeze(img)
                depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=2, order=0)
                diag_length = float(depth_heightmap_2x.shape[0]) * np.sqrt(2)
                diag_length = np.ceil(diag_length / 32) * 32
                padding_width = int((diag_length - depth_heightmap_2x.shape[0]) / 2)
                depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
                depth_heightmap_2x = torch.tensor(depth_heightmap_2x).unsqueeze(0).unsqueeze(0).to(self.device)

                rotate_theta = rot[i].item()
                # clockwise
                affine_mat = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                         [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat.shape = (2, 3, 1)
                affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float().to(self.device)
                flow_grid = F.affine_grid(affine_mat, depth_heightmap_2x.size())

                # Rotate images clockwise
                depth_heightmap_2x_rotated = F.grid_sample(depth_heightmap_2x, flow_grid, mode='nearest')

                # depth_heightmap_2x_rotated = ndimage.rotate(depth_heightmap_2x, -np.rad2deg(rot[i].item()), reshape=False)
                depth_heightmap_rotated = depth_heightmap_2x_rotated[:, :, padding_width: -padding_width:2,
                                                                     padding_width: -padding_width:2]
                in_hand_img = depth_heightmap_rotated - z[i].item()
                # in_hand_img = depth_heightmap_rotated
                in_hand_imgs.append(in_hand_img)

            return torch.cat(in_hand_imgs, dim=0)

    def forwardPhiNet(self, states, obs, pixels, target_net=False, to_cpu=False):
        obs, in_hand = obs
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        patch = self.getImgPatch(obs[:, 0:1].to(self.device), pixels.to(self.device))
        patch = self.encodeInHand(patch, in_hand)

        phi_net = self.phi_net if not target_net else self.target_phi
        phi_output = phi_net(patch).reshape(states.size(0), self.num_primitives, -1)[torch.arange(0, states.size(0)),
                                                                                     states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

    def forwardFCN(self, states, obs, target_net=False, to_cpu=False):
        obs, in_hand = obs
        fcn = self.fcn if not target_net else self.target_fcn
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        q_value_maps = fcn(obs, in_hand)[torch.arange(0, states.size(0)), states.long()]

        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps

    def getEGreedyActions(self, states, obs, eps, coef=0.1):
        # obs, in_hand = obs
        obs = obs.to(self.device)
        in_hand = self.his.to(self.device)
        with torch.no_grad():
            q_value_maps = self.fcn(obs, in_hand).cpu()
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        pixels = torch_utils.argmax2d(q_value_maps[torch.arange(0, states.size(0)), states.long()]).cpu().long()
        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        patch, rot_id, z_id, rotation, z = self.getEGreedyPhi(states, obs, eps, pixels)

        actions = torch.cat((x, y, z, rotation), dim=1)
        action_idx = torch.cat((pixels, z_id, rot_id), dim=1)
        return q_value_maps, patch, action_idx, actions

    def getEGreedyPhi(self, states, obs, eps, pixels):
        # obs, in_hand = obs
        # in_hand = self.his.to(self.device)
        patch = self.getImgPatch(obs[:, 0:1], pixels.to(self.device))
        # encoded_patch = self.encodeInHand(patch, in_hand)
        #
        # with torch.no_grad():
        #     phi_output = self.phi_net(encoded_patch).cpu().reshape(states.size(0),
        #                                                    self.num_primitives, -1)[torch.arange(0, states.size(0)),
        #                                                                             states.long()]
        with torch.no_grad():
            phi_output = self.forwardPhiNet(states, (obs, self.his), pixels, to_cpu=True)

        phi = torch.argmax(phi_output, 1)

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.num_rotations * self.num_heights)
        phi[rand_mask] = rand_phi.long()
        rot_id = (phi / self.num_rotations).long().reshape(states.size(0), 1)
        rotation = self.rotations[rot_id].reshape(states.size(0), 1)
        z_id = (phi % self.num_heights).long().reshape(states.size(0), 1)
        z = self.heights[z_id].reshape(states.size(0), 1)
        return patch, rot_id, z_id, rotation, z

    def samplePhis(self, states, obs, center_pixels, num_phis):
        # obs, in_hand = obs
        # in_hand = self.his.to(self.device)
        # patch = self.getImgPatch(obs.to(self.device), center_pixels.to(self.device))
        # encoded_patch = self.encodeInHand(patch, in_hand.to(self.device))
        # with torch.no_grad():
        #     phi_output = self.phi_net(encoded_patch).cpu().reshape(states.size(0),
        #                                                    self.num_primitives, -1)[torch.arange(0, states.size(0)),
        #                                                                             states.long()]
        with torch.no_grad():
            phi_output = self.forwardPhiNet(states, (obs, self.his), center_pixels, to_cpu=True)

        phis = []
        for output in phi_output.cpu().numpy():
            p = np.exp(output) / np.exp(output).sum()
            phi = torch.tensor(np.random.choice(np.arange(len(output)), num_phis, replace=False, p=p))
            rot_id = (phi / self.num_rotations).long().reshape(num_phis, 1)
            z_id = (phi % self.num_heights).long().reshape(num_phis, 1)
            phis.append(torch.cat((rot_id, z_id), dim=1).unsqueeze(1))
        return torch.cat(phis, dim=1)

    def reshapeNextObs(self, next_obs, batch_size):
        next_obs = zip(*next_obs)
        next_obs, next_in_hand = next_obs
        next_obs = torch.stack(next_obs)
        next_in_hand = torch.stack(next_in_hand)
        next_obs = next_obs.reshape(batch_size, next_obs.shape[-3], next_obs.shape[-2], next_obs.shape[-1])
        next_in_hand = next_in_hand.reshape(batch_size, next_in_hand.shape[-3], next_in_hand.shape[-2], next_in_hand.shape[-1])
        return next_obs, next_in_hand

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
        phis = []
        for d in batch:
            states.append(d.state)
            images.append(d.image[0])
            in_hands.append(d.image[1])
            xys.append(d.xy)
            rewards.append(d.rewards)
            next_states.append(d.next_states)
            tmp = list(zip(*d.next_obs))
            next_obs.append(torch.stack(tmp[0]))
            next_in_hands.append(torch.stack(tmp[1]))
            dones.append(d.dones)
            phis.append(d.phis)
        states_tensor = torch.tensor(states).long().to(self.device)
        image_tensor = torch.tensor(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        in_hand_tensor = torch.tensor(in_hands).to(self.device)
        if len(in_hand_tensor.shape) == 3:
            in_hand_tensor = in_hand_tensor.unsqueeze(1)
        xy_tensor = torch.tensor(xys).to(self.device)
        rewards_tensor = torch.tensor(rewards).to(self.device)
        next_states_tensor = torch.tensor(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        next_in_hands_tensor = torch.stack(next_in_hands).to(self.device)
        dones_tensor = torch.tensor(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        phis_tensor = torch.tensor(phis).long().to(self.device)

        return states_tensor, (image_tensor, in_hand_tensor), xy_tensor, rewards_tensor, next_states_tensor, \
               tuple(zip(*(next_obs_tensor, next_in_hands_tensor))), non_final_masks, phis_tensor