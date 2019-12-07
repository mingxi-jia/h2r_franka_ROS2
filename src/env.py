from src.ur5 import UR5
from src.img_proxy import ImgProxy
import skimage
import scipy
import torch
import ros_numpy
import numpy as np

class Env:
    def __init__(self):
        self.ur5 = UR5()
        self.img_proxy = ImgProxy()
        self.heightmap = np.zeros((90, 90))

    def getObs(self):
        obs = self.img_proxy.getImage()
        obs = obs[240 - 100:240 + 100, 320 - 100:320 + 100]
        # obs[np.isnan(obs)] = obs[np.logical_not(np.isnan(obs))].max()
        mask = np.isnan(obs)
        obs[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), obs[~mask])
        obs = -obs
        obs -= obs.min()
        obs = skimage.transform.resize(obs, (90, 90))
        obs = scipy.ndimage.rotate(obs, 90)
        self.heightmap = obs
        # obs = obs.T
        obs = obs.reshape(1, 1, 90, 90)
        return torch.tensor(obs).to(torch.float32)

    def getGripperClosed(self):
        return self.ur5.gripper.isClosed()

    def getSafeHeight(self, x, y):
        return self.heightmap[x-5:x+5, y-5:y+5].max()
