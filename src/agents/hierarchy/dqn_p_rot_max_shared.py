from utils import torch_utils

import numpy as np

import torch
import torch.nn.functional as F
from agents.hierarchy.dqn_rot_max_shared import DQNRotMaxShared

class DQNPRotMaxShared(DQNRotMaxShared):
    def __init__(self, fcn, cnn, fcn_p, cnn_p, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                num_primitives = 1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24):
        super().__init__(fcn, cnn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives, sl, per, num_rotations, half_rotation, patch_size)
        self.fcn_p = fcn_p
        self.phi_net_p = cnn_p

        self.fcn_p_optimizer = torch.optim.Adam(self.fcn_p.parameters(), lr=self.lr)
        self.phi_p_optimizer = torch.optim.Adam(self.phi_net_p.parameters(), lr=self.lr)

    def forwardFCNP(self, states, obs, to_cpu=False):
        fcn = self.fcn_p
        obs = obs.to(self.device)
        padding_width = int((self.padding - obs.size(2)) / 2)
        # pad obs
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        q_value_maps = fcn(obs)[torch.arange(0, states.size(0)), states.long()]
        q_value_maps = q_value_maps[:, :, padding_width: -padding_width, padding_width: -padding_width]

        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps

    def forwardPhiNetP(self, states, obs, pixels, to_cpu=False):
        phi_net = self.phi_net_p
        patch = self.getImgPatch(obs.to(self.device), pixels.to(self.device))
        phi_output = phi_net(patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output

    def getEGreedyActions(self, states, obs, eps, coef=0.01):
        with torch.no_grad():
            q_value_maps, obs_encoding = self.forwardFCN(states, obs, to_cpu=True)
            affordance = self.forwardFCNP(states, obs, to_cpu=True)
            q_value_maps[states==0] = q_value_maps[states==0] * affordance[states==0]
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        pixels = torch_utils.argmax2d(q_value_maps).long()
        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        with torch.no_grad():
            phi_output = self.forwardPhiNet(states, obs, pixels, obs_encoding, to_cpu=True)
            affordance = self.forwardPhiNetP(states, obs, pixels, to_cpu=True)
            phi_output[states==0] = phi_output[states==0] * affordance[states==0]

        phi = torch.argmax(phi_output, 1)
        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps
        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.phi_size)
        phi[rand_mask] = rand_phi.long()
        phi_idx, phi_a = self.decodePhi(phi)
        actions = torch.cat((x, y, phi_a), dim=1)
        action_idx = torch.cat((pixels, phi_idx), dim=1)

        return q_value_maps, action_idx, actions

    def calcPLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]
        phis = action_idx[:, 2:]

        fcn_output = self.forwardFCNP(states, obs)[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]
        fcn_pred = fcn_output[states==0]
        fcn_target = torch.zeros_like(states).float()
        fcn_target[(states==0) & (next_states==1)] = 1
        fcn_target = fcn_target[states==0]

        phi_output = self.forwardPhiNetP(states, obs, pixel)[torch.arange(batch_size), phis[:, 0]]
        phi_pred = phi_output[states==0]
        phi_target = torch.zeros_like(states).float()
        phi_target[(states==0) & (next_states==1)] = 1
        phi_target = phi_target[states==0]

        fcn_p_loss = F.smooth_l1_loss(fcn_pred, fcn_target)
        phi_p_loss = F.smooth_l1_loss(phi_pred, phi_target)

        return fcn_p_loss+phi_p_loss


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

        p_loss = self.calcPLoss()
        self.fcn_p_optimizer.zero_grad()
        self.phi_p_optimizer.zero_grad()
        p_loss.backward()

        for param in self.fcn_p.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_p_optimizer.step()

        for param in self.phi_net_p.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_p_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item()+p_loss.item(), td_error

    def loadModel(self, path_pre):
        fcn_path = path_pre + '_fcn.pt'
        print('loading {}'.format(fcn_path))
        self.fcn.load_state_dict(torch.load(fcn_path))
        cnn_path = path_pre + '_cnn.pt'
        print('loading {}'.format(cnn_path))
        self.phi_net.load_state_dict(torch.load(cnn_path))
        self.updateTarget()

        fcn_p_path = path_pre + '_fcn_p.pt'
        print('loading {}'.format(fcn_p_path))
        self.fcn_p.load_state_dict(torch.load(fcn_p_path))
        cnn_p_path = path_pre + '_cnn_p.pt'
        print('loading {}'.format(cnn_p_path))
        self.phi_net_p.load_state_dict(torch.load(cnn_p_path))


    def saveModel(self, path):
        torch.save(self.fcn.state_dict(), '{}_fcn.pt'.format(path))
        torch.save(self.phi_net.state_dict(), '{}_cnn.pt'.format(path))
        torch.save(self.fcn_p.state_dict(), '{}_fcn_p.pt'.format(path))
        torch.save(self.phi_net_p.state_dict(), '{}_cnn_p.pt'.format(path))

    def getSaveState(self):
        return {
            'fcn': self.fcn.state_dict(),
            'fcn_target': self.target_fcn.state_dict(),
            'fcn_optimizer': self.fcn_optimizer.state_dict(),
            'cnn': self.phi_net.state_dict(),
            'cnn_target': self.target_phi.state_dict(),
            'cnn_optimizer': self.phi_optimizer.state_dict(),
            'fcn_p': self.fcn_p.state_dict(),
            'fcn_p_optimizer': self.fcn_p_optimizer.state_dict(),
            'cnn_p': self.phi_net_p.state_dict(),
            'cnn_p_optimizer': self.phi_p_optimizer.state_dict()
        }

    def loadFromState(self, save_state):
        self.fcn.load_state_dict(save_state['fcn'])
        self.target_fcn.load_state_dict(save_state['fcn_target'])
        self.fcn_optimizer.load_state_dict(save_state['fcn_optimizer'])

        self.phi_net.load_state_dict(save_state['cnn'])
        self.target_phi.load_state_dict(save_state['cnn_target'])
        self.phi_optimizer.load_state_dict(save_state['cnn_optimizer'])

        self.fcn_p.load_state_dict(save_state['fcn_p'])
        self.fcn_p_optimizer.load_state_dict(save_state['fcn_p_optimizer'])

        self.phi_net_p.load_state_dict(save_state['cnn_p'])
        self.phi_p_optimizer.load_state_dict(save_state['cnn_p_optimizer'])

    def train(self):
        self.fcn.train()
        self.phi_net.train()

        self.fcn_p.train()
        self.phi_net_p.train()

    def eval(self):
        self.fcn.eval()
        self.phi_net.eval()

        self.fcn_p.eval()
        self.phi_net_p.eval()

    def getModelStr(self):
        return 'fcn:' + str(self.fcn) + '\n' + 'cnn:' + str(self.phi_net) + '\n' + 'fcn_p' + str(self.fcn_p) + '\n' + 'cnn_p:' + str(self.phi_net_p)
