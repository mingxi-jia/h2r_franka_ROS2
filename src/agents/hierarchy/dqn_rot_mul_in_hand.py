from agents.hierarchy.dqn_rot_mul import DQNRotMul

import torch
import torch.nn.functional as F

class DQNRotMulInHand(DQNRotMul):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives = 1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        super().__init__(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                         num_primitives, sl, per, num_rotations, half_rotation, patch_size)

    def getCurrentObs(self, in_hand, obs):
        obss = []
        for i, o in enumerate(obs):
            obss.append((o.squeeze(), in_hand[i].squeeze()))
        return obss

    def encodeInHand(self, input_img, in_hand_img):
        if input_img.size(2) == in_hand_img.size(2):
            return torch.cat((input_img, in_hand_img), dim=1)
        else:
            resized_in_hand = F.interpolate(in_hand_img, size=(input_img.size(2), input_img.size(3)),
                                            mode='nearest')
        return torch.cat((input_img, resized_in_hand), dim=1)

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
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)

        padding_width = int((self.padding - obs.size(2)) / 2)

        fcn = self.fcn if not target_net else self.target_fcn
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        q_value_maps = fcn(obs, in_hand)[torch.arange(0, states.size(0)), states.long(), padding_width: -padding_width, padding_width: -padding_width]
        # q_value_maps = q_value_maps[:, :, padding_width: -padding_width, padding_width: -padding_width]

        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.01):
        return super(DQNRotMulInHand, self).getEGreedyActions(states, (obs, in_hand), eps, coef)

