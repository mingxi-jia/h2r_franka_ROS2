from agents.hierarchy_backup.phi.in_hand_interface import InHandInterface

import torch

class InHandHisInterface(InHandInterface):
    def __init__(self):
        InHandInterface.__init__(self)

    def initHis(self, num_processes):
        self.his = torch.zeros((num_processes, self.num_his_channel, self.patch_size, self.patch_size))

    def getNextObs(self, patch, rotation, height, states_, obs_, dones):
        in_hand_img_ = self.getInHandImage(patch, rotation, height).cpu()
        in_hand_img_[1 - states_.bool()] = self.empty_in_hand.clone()
        in_hand_img_[dones.bool()] = self.empty_in_hand.clone()
        his = torch.cat((in_hand_img_, patch.cpu(),  self.his[:, :-2]), dim=1)
        obss_ = []
        for i, o in enumerate(obs_):
            obss_.append((o, his[i]))
        return obss_

    def updateHis(self, patch, rotation, height, states_, obs_, dones):
        in_hand_img_ = self.getInHandImage(patch, rotation, height).cpu()
        in_hand_img_[1 - states_.bool()] = self.empty_in_hand.clone()
        in_hand_img_[dones.bool()] = self.empty_in_hand.clone()
        self.his = torch.cat((in_hand_img_, patch.cpu(), self.his[:, :-2]), dim=1)
        self.his[dones.bool()] = torch.zeros((1, self.num_his_channel, self.patch_size, self.patch_size))

