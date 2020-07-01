import torch
import torch.nn.functional as F

from agents.hierarchy.dqn_rot_mul import DQNRotMul
from agents.hierarchy.dqn_rot_max import DQNRotMax
from agents.hierarchy.dqn_rot_max_shared import DQNRotMaxShared
from agents.hierarchy.dqn_p_rot_max_shared import DQNPRotMaxShared

class InHandAgent:
    def __init__(self):
        pass

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
        q_value_maps = fcn(obs, in_hand)[torch.arange(0, states.size(0)), states.long(), padding_width: -padding_width,
                       padding_width: -padding_width]
        # q_value_maps = q_value_maps[:, :, padding_width: -padding_width, padding_width: -padding_width]

        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.01):
        return super().getEGreedyActions(states, (obs, in_hand), eps, coef)

class InHandSharedAgent(InHandAgent):
    def __init__(self):
        super().__init__()

    def forwardFCN(self, states, obs, target_net=False, to_cpu=False):
        obs, in_hand = obs
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        padding_width = int((self.padding - obs.size(2)) / 2)
        fcn = self.fcn if not target_net else self.target_fcn
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        q_value_maps, obs_encoding = fcn(obs, in_hand)
        q_value_maps = q_value_maps[torch.arange(0, states.size(0)), states.long(), padding_width: -padding_width,
                       padding_width: -padding_width]
        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps, obs_encoding

    def forwardPhiNet(self, states, obs, pixels, obs_encoding, target_net=False, to_cpu=False):
        obs, in_hand = obs
        patch = self.getImgPatch(obs.to(self.device), pixels.to(self.device))
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        phi_net = self.phi_net if not target_net else self.target_phi
        phi_output = phi_net(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

class InHandSharedPAgent(InHandSharedAgent):
    def __init__(self):
        super().__init__()

    def forwardFCNP(self, states, obs, to_cpu=False):
        fcn = self.fcn_p
        obs, in_hand = obs

        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        padding_width = int((self.padding - obs.size(2)) / 2)
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        q_value_maps = fcn(obs, in_hand)[torch.arange(0, states.size(0)), 0]
        q_value_maps = q_value_maps[:, padding_width: -padding_width, padding_width: -padding_width]

        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps

    def forwardPhiNetP(self, states, obs, pixels, to_cpu=False):
        phi_net = self.phi_net_p
        obs, in_hand = obs

        patch = self.getImgPatch(obs.to(self.device), pixels.to(self.device))
        phi_output = phi_net(patch).reshape(states.size(0), 1, -1)[
            torch.arange(0, states.size(0)),
            0]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

class DQNRotMulInHand(InHandAgent, DQNRotMul):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        DQNRotMul.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                           num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        InHandAgent.__init__(self)


class DQNRotMaxInHand(InHandAgent, DQNRotMax):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        DQNRotMax.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                           num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        InHandAgent.__init__(self)

    def forwardPhiNet(self, states, obs, pixels, target_net=False, to_cpu=False):
        obs, in_hand = obs
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)

        padding_width = int((self.padding - obs.size(2)) / 2)
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)

        patch = self.getImgPatch(obs[:, 0:1].to(self.device), pixels.to(self.device))
        patch = self.encodeInHand(patch, in_hand)

        phi_net = self.phi_net if not target_net else self.target_phi
        phi_output = phi_net(obs, patch).reshape(states.size(0), self.num_primitives, -1)[torch.arange(0, states.size(0)),
                                                                                     states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

    def expandObs(self, obs, n):
        obs, in_hand = obs
        expanded_obs = obs.expand(n, -1, -1, -1, -1)
        expanded_obs = expanded_obs.reshape(expanded_obs.size(0)*expanded_obs.size(1), expanded_obs.size(2), expanded_obs.size(3), expanded_obs.size(4))
        expanded_in_hand = in_hand.expand(n, -1, -1, -1, -1)
        expanded_in_hand = expanded_in_hand.reshape(expanded_in_hand.size(0)*expanded_in_hand.size(1), expanded_in_hand.size(2), expanded_in_hand.size(3), expanded_in_hand.size(4))
        return expanded_obs, expanded_in_hand

class DQNRotMaxSharedInHand(InHandSharedAgent, DQNRotMaxShared):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        DQNRotMaxShared.__init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma,
                           num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        InHandSharedAgent.__init__(self)

class DQNPRotMaxSharedInHand(InHandSharedPAgent, DQNPRotMaxShared):
    def __init__(self, fcn, cnn, fcn_p, cnn_p, action_space, workspace, heightmap_resolution, device, lr=1e-4,
                 gamma=0.9, num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        DQNPRotMaxShared.__init__(self, fcn, cnn, fcn_p, cnn_p, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                  num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        InHandSharedPAgent.__init__(self)
