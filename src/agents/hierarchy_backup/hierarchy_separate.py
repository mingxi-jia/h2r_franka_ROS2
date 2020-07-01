import torch
import torch.nn.functional as F
import utils.torch_utils as torch_utils

from agents.models import FCN, CNN
from agents.hierarchy_backup.hierarchy_agent import HierarchyAgent

class HierarchySeparate(HierarchyAgent):
    def __init__(self, action_space, workspace, heightmap_resolution, device, num_rotations=8, half_rotation=False,
                 num_heights=10, height_range=(1, 0.1), lr=1e-4, patch_size=64):
        super(HierarchySeparate, self).__init__(action_space, workspace, heightmap_resolution, device, num_rotations, half_rotation,
                                                num_heights, height_range, lr, patch_size)

        self.fcn = FCN().to(device)
        self.rot_net = CNN((1, patch_size, patch_size), num_rotations).to(device)
        self.z_net = CNN((1, patch_size, patch_size), num_heights).to(device)

        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr)
        self.rot_optimizer = torch.optim.Adam(self.rot_net.parameters(), lr=self.lr)
        self.z_optimizer = torch.optim.Adam(self.z_net.parameters(), lr=self.lr)

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
            rot_output = self.rot_net(patch.to(self.device))
        rot = torch.argmax(rot_output, 1)
        rotation = self.rotations[rot].reshape(states.size(0), 1)

        with torch.no_grad():
            height_output = self.z_net(patch.to(self.device))
        z_id = torch.argmax(height_output, 1)
        z = self.heights[z_id].reshape(states.size(0), 1)
        actions = torch.cat((xys, z, rotation), dim=1)
        return actions

    def update(self, batch):
        image_tensor, xy_tensor, label_tensor, positive_tensor = self._loadBatchToDevice(batch)
        pixel = self.coordToPixel(xy_tensor.cpu()).to(self.device)

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
        patch_tensor = self.getImgPatch(image_tensor.cpu(), pixel)
        mask = positive_tensor
        rot_output = self.rot_net(patch_tensor.to(self.device))
        rot_label = label_tensor.sum(dim=2)
        rot_label[rot_label>1] = 1
        rot_target = rot_label.float()
        rot_loss = F.binary_cross_entropy_with_logits(rot_output[mask], rot_target[mask])
        self.rot_optimizer.zero_grad()
        rot_loss.backward()
        for param in self.rot_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.rot_optimizer.step()

        z_output = self.z_net(patch_tensor.to(self.device))
        z_label = label_tensor.sum(dim=1)
        z_label[z_label>1] = 1
        z_target = z_label.float()
        z_loss = F.binary_cross_entropy_with_logits(z_output[mask], z_target[mask])
        self.z_optimizer.zero_grad()
        z_loss.backward()
        for param in self.z_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.z_optimizer.step()

        return fcn_loss.item(), rot_loss.item(), z_loss.item()

    def saveModel(self, path):
        torch.save(self.fcn.state_dict(), '{}_fcn.pt'.format(path))
        torch.save(self.rot_net.state_dict(), '{}_rot.pt'.format(path))
        torch.save(self.z_net.state_dict(), '{}_z.pt'.format(path))

    def loadModel(self, fcn_path, rot_path, z_path=None):
        self.fcn.load_state_dict(torch.load(fcn_path))
        self.rot_net.load_state_dict(torch.load(rot_path))
        if z_path:
            self.z_net.load_state_dict(torch.load(z_path))
