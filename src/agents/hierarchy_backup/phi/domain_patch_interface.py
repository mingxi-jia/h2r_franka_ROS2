import torch
import torch.nn.functional as F

class DomainPatchInterface:
    def __init__(self):
        pass

    def getImgPatch(self, obs, center_pixel):
        patch = super().getImgPatch(obs, center_pixel)
        whole_obs = F.interpolate(obs, (self.patch_size, self.patch_size), mode='bilinear', align_corners=False)
        return torch.cat((patch, whole_obs), 1)

    def getInHandImage(self, patch, rot, z):
        patch = patch[:, 0:1]
        return super().getInHandImage(patch, rot, z)