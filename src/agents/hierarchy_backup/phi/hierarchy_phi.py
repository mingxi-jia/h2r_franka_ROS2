import torch
import torch.nn.functional as F
import utils.torch_utils as torch_utils

from agents.models import FCN, CNN, FCNSmall, CNNSmall
from agents.hierarchy_backup.hierarchy_agent import HierarchyAgent

class HierarchyPhi(HierarchyAgent):
    def __init__(self, fcn, cnn, action_space, workspace, heightmap_resolution, device, num_rotations=8, half_rotation=False,
                 num_heights=10, height_range=(1, 0.1), lr=1e-4, patch_size=64, num_primitives=1, num_input_channel=1):
        super(HierarchyPhi, self).__init__(action_space, workspace, heightmap_resolution, device, num_rotations,
                                           half_rotation, num_heights, height_range, lr, patch_size)

        self.fcn = fcn
        self.phi_net = cnn
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr)
        self.phi_optimizer = torch.optim.Adam(self.phi_net.parameters(), lr=self.lr)

        self.num_primitives = num_primitives
        self.num_input_channel = num_input_channel

    def getQValueMap(self, states, obs):
        with torch.no_grad():
            q_value_maps = self.fcn(obs.to(self.device)).cpu()
        return q_value_maps

    def getGreedyAction(self, states, obs):
        with torch.no_grad():
            q_value_maps = self.fcn(obs.to(self.device)).cpu()
        pixels = torch_utils.argmax2d(q_value_maps[torch.arange(0, states.size(0)), states.long()]).cpu().long()
        xys = self.getXYFromPixels(states, pixels)
        patch = self.getImgPatch(obs, pixels)

        with torch.no_grad():
            phi_output = self.phi_net(patch.to(self.device))
        phi = torch.argmax(phi_output, 1)
        rot_id = (phi/self.num_rotations).long()
        rotation = self.rotations[rot_id].reshape(states.size(0), 1)
        z_id = (phi%self.num_heights).long()
        z = self.heights[z_id].reshape(states.size(0), 1)
        actions = torch.cat((xys, z, rotation), dim=1)
        return actions

    def getEGreedyActions(self, states, obs, eps):
        obs = obs.to(self.device)
        with torch.no_grad():
            q_value_maps = self.fcn(obs).cpu()
        q_value_maps += torch.randn_like(q_value_maps) * eps * 0.1

        # import matplotlib.pyplot as plt
        # plt.imshow(q_value_maps[0, 0].cpu());plt.show()

        pixels = torch_utils.argmax2d(q_value_maps[torch.arange(0, states.size(0)), states.long()]).cpu().long()
        x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        patch = self.getImgPatch(obs, pixels.to(self.device))

        # plt.imshow(patch[0, 0].cpu());plt.show()

        with torch.no_grad():
            phi_output = self.phi_net(patch.to(self.device)).cpu()
        phi = torch.argmax(phi_output, 1)
        rot_id = (phi/self.num_rotations).long().reshape(states.size(0), 1)
        rotation = self.rotations[rot_id].reshape(states.size(0), 1)
        z_id = (phi%self.num_heights).long().reshape(states.size(0), 1)
        z = self.heights[z_id].reshape(states.size(0), 1)

        actions = torch.cat((x, y, z, rotation), dim=1)
        action_idx = torch.cat((pixels, z_id, rot_id), dim=1)
        return q_value_maps, action_idx, actions

    def update(self, batch):
        image_tensor, xy_tensor, label_tensor, positive_tensor = self._loadBatchToDevice(batch)
        pixel = xy_tensor

        # optimize fcn
        fcn_output = self.fcn(image_tensor)
        fcn_prediction = fcn_output[torch.arange(0, len(batch)), 0, pixel[:, 0], pixel[:, 1]]
        fcn_target = positive_tensor.float()
        fcn_loss = F.mse_loss(fcn_prediction, fcn_target)
        self.fcn_optimizer.zero_grad()
        fcn_loss.backward()
        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        # optimize rot
        patch_tensor = self.getImgPatch(image_tensor, pixel)
        mask = positive_tensor
        phi_output = self.phi_net(patch_tensor.to(self.device))
        phi_label = label_tensor.reshape(label_tensor.size(0), -1)
        phi_target = phi_label.float()
        phi_loss = F.binary_cross_entropy_with_logits(phi_output[mask], phi_target[mask])
        self.phi_optimizer.zero_grad()
        phi_loss.backward()
        for param in self.phi_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_optimizer.step()

        return fcn_loss.item(), phi_loss.item()

    def saveModel(self, path):
        torch.save(self.fcn.state_dict(), '{}_fcn.pt'.format(path))
        torch.save(self.phi_net.state_dict(), '{}_phi.pt'.format(path))

    def loadModel(self, path_pre):
        fcn_path = path_pre + '_fcn.pt'
        phi_path = path_pre + '_phi.pt'
        print('loading {}'.format(fcn_path))
        self.fcn.load_state_dict(torch.load(fcn_path))
        print('loading {}'.format(phi_path))
        self.phi_net.load_state_dict(torch.load(phi_path))

