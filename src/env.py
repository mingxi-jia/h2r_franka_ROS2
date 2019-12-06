from ur5 import UR5
from img_proxy import ImgProxy
import skimage
import torch
import ros_numpy
import numpy as np

class Env:
    def __init__(self):
        self.ur5 = UR5()
        self.img_proxy = ImgProxy()

    def getObs(self):
        obs = self.img_proxy.getImage()
        obs = obs[240 - 100:240 + 100, 320 - 100:320 + 100]
        obs[np.isnan(obs)] = 0
        obs = -obs
        obs -= obs.min()
        obs = skimage.transform.resize(obs, (90, 90))
        obs = obs.reshape(1, 1, 90, 90)
        return torch.tensor(obs).to(torch.float32)