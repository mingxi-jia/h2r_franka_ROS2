import torch
import torch.nn.functional as F
import utils.torch_utils as torch_utils

from agents.models import CNN
from agents.hierarchy_backup.hierarchy_separate import HierarchySeparate


class HierarchyDual(HierarchySeparate):
    def __init__(self, action_space, workspace, heightmap_resolution, device, num_rotations=8, half_rotation=False,
                 num_heights=10, height_range=(1, 0.1), lr=1e-4, patch_size=64):
        super(HierarchyDual, self).__init__(action_space, workspace, heightmap_resolution, device, num_rotations,
                                            half_rotation,
                                            num_heights, height_range, lr, patch_size)

        self.phi_net = CNN((1, patch_size, patch_size), num_heights*num_rotations).to(device)

        self.phi_optimizer = torch.optim.Adam(self.phi_net.parameters(), lr=self.lr)

    def getGreedyActionFromPhi(self, states, obs):
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
        patch_tensor = self.getImgPatch(image_tensor, pixel)
        mask = positive_tensor
        rot_output = self.rot_net(patch_tensor.to(self.device))
        rot_label = label_tensor.sum(dim=2)
        rot_label[rot_label > 1] = 1
        rot_target = rot_label.float()
        rot_loss = F.binary_cross_entropy_with_logits(rot_output[mask], rot_target[mask])
        self.rot_optimizer.zero_grad()
        rot_loss.backward()
        for param in self.rot_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.rot_optimizer.step()

        z_output = self.z_net(patch_tensor.to(self.device))
        z_label = label_tensor.sum(dim=1)
        z_label[z_label > 1] = 1
        z_target = z_label.float()
        z_loss = F.binary_cross_entropy_with_logits(z_output[mask], z_target[mask])
        self.z_optimizer.zero_grad()
        z_loss.backward()
        for param in self.z_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.z_optimizer.step()

        phi_output = self.phi_net(patch_tensor.to(self.device))
        phi_label = label_tensor.reshape(label_tensor.size(0), -1)
        phi_target = phi_label.float()
        phi_loss = F.binary_cross_entropy_with_logits(phi_output[mask], phi_target[mask])
        self.phi_optimizer.zero_grad()
        phi_loss.backward()
        for param in self.phi_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.phi_optimizer.step()

        return fcn_loss.item(), rot_loss.item(), z_loss.item(), phi_loss.item()

    def saveModel(self, path):
        torch.save(self.fcn.state_dict(), '{}_fcn.pt'.format(path))
        torch.save(self.rot_net.state_dict(), '{}_rot.pt'.format(path))
        torch.save(self.z_net.state_dict(), '{}_z.pt'.format(path))
        torch.save(self.phi_net.state_dict(), '{}_phi.pt'.format(path))

    def loadModel(self, fcn_path, rot_path, z_path, phi_path):
        self.fcn.load_state_dict(torch.load(fcn_path))
        self.rot_net.load_state_dict(torch.load(rot_path))
        self.z_net.load_state_dict(torch.load(z_path))
        self.phi_net.load_state_dict(torch.load(phi_path))

