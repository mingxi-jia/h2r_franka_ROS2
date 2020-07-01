from utils import torch_utils

import torch
import torch.nn.functional as F

class V2VFunctionInterface:
    def __init__(self, v_net):
        self.v_net = v_net
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=self.lr)

    def forwardVNet(self, states, obs):
        if type(obs) is tuple:
            obs = tuple(map(lambda o: o.to(self.device), obs))
        else:
            obs = obs.to(self.device)
        v_output = self.v_net(obs).reshape(states.size(0), -1)[torch.arange(0, states.size(0)), states.long()]
        return v_output

    def updateMultipleX(self, batch, num_x=10):
        states, image, pixel, rewards, next_states, next_obs, non_final_masks, phis = self._loadBatchToDevice(
            batch)
        batch_size = states.size(0)
        phi_size = phis.size(1)
        heightmap_size = self.action_space[1]

        next_obs = self.reshapeNextObs(next_obs, batch_size * phi_size)
        next_states = next_states.reshape(batch_size * phi_size)
        with torch.no_grad():
            q_prime = self.forwardVNet(next_states, next_obs)
        q_prime = q_prime.reshape(batch_size, phi_size)
        q = rewards + self.gamma * q_prime * non_final_masks

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
                q2_target_output = self.forwardPhiNet(states, image, x)
            q1_targets.append(q2_target_output.max(1)[0])
            q1_outputs.append(q1_map[torch.arange(0, batch_size), x[:, 0], x[:, 1]])

        q1_targets = torch.stack(q1_targets, dim=1)
        q1_outputs = torch.stack(q1_outputs, dim=1)

        v_output = self.forwardVNet(states, image)
        v_target = q1_map.reshape(batch_size, -1).max(1)[0].detach()

        fcn_loss = F.smooth_l1_loss(q1_outputs, q1_targets)
        phi_loss = F.smooth_l1_loss(q2_output, q)
        v_loss = F.smooth_l1_loss(v_output, v_target)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        for param in self.v_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.v_optimizer.step()

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

        return (fcn_loss.item(), phi_loss.item(), v_loss.item()), None

    def saveModel(self, path):
        torch.save(self.fcn.state_dict(), '{}_fcn.pt'.format(path))
        torch.save(self.phi_net.state_dict(), '{}_phi.pt'.format(path))
        torch.save(self.v_net.state_dict(), '{}_v.pt'.format(path))

    def loadModel(self, path_pre):
        fcn_path = path_pre + '_fcn.pt'
        phi_path = path_pre + '_phi.pt'
        v_path = path_pre + '_v.pt'
        print('loading {}'.format(fcn_path))
        self.fcn.load_state_dict(torch.load(fcn_path))
        print('loading {}'.format(phi_path))
        self.phi_net.load_state_dict(torch.load(phi_path))
        print('loading {}'.format(v_path))
        self.v_net.load_state_dict(torch.load(v_path))
