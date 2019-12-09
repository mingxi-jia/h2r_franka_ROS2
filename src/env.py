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
        self.ws_center = [-0.5257, -0.0098, 0.09]

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
        self.heightmap = obs.copy()
        # obs = obs.T
        b = np.linspace(0.01, 0, 90).reshape(1, 90).repeat(90, axis=0)
        obs -= b
        obs *= 0.8
        obs[obs < 0.01] = 0
        obs = obs.reshape(1, 1, 90, 90)
        return torch.tensor(obs).to(torch.float32)

    def getGripperClosed(self):
        return self.ur5.gripper.isClosed()

    def getSafeHeight(self, x, y):
        return self.heightmap[max(x-5, 0):min(x+5, 90), max(y-5, 0):min(y+5, 90)].max()
