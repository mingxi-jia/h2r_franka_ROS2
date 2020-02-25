from src.ur5 import UR5
from src.img_proxy import ImgProxy
import skimage
import scipy
import torch
import ros_numpy
import numpy as np
import matplotlib.pyplot as plt

class Env:
    def __init__(self, ws_center=(-0.5257, -0.0098, 0.095), ws_x=0.3, ws_y=0.3, cam_resolution=0.0015, obs_size=(90, 90),
                 action_sequence='pxyr'):
        self.ws_center = ws_center
        self.ws_x = ws_x
        self.ws_y = ws_y
        self.workspace = np.asarray([[ws_center[0] - ws_x/2, ws_center[0] + ws_x/2],
                                     [ws_center[1] - ws_y/2, ws_center[1] + ws_y/2],
                                     [ws_center[2], ws_center[2]+0.5]])
        self.cam_resolution = cam_resolution
        self.obs_size = obs_size
        self.action_sequence = action_sequence
        self.heightmap_resolution = ws_x / obs_size[0]

        self.ur5 = UR5()
        self.img_proxy = ImgProxy()
        self.heightmap = np.zeros((self.obs_size[0], self.obs_size[1]))

        # Motion primatives
        self.PICK_PRIMATIVE = 0
        self.PLACE_PRIMATIVE = 1

        self.ee_offset = 0.095

    def _getXYFromPixels(self, x_pixel, y_pixel):
        x = y_pixel * self.heightmap_resolution + self.workspace[0][0]
        y = (self.obs_size[0]-x_pixel) * self.heightmap_resolution + self.workspace[1][0]
        return x, y

    def _getPixelsFromXY(self, x, y):
        '''
        Get the x/y pixels on the heightmap for the given coordinates
        Args:
          - x: X coordinate
          - y: Y coordinate
        Returns: (x, y) in pixels corresponding to coordinates
        '''
        y_pixel = (x - self.workspace[0][0]) / self.heightmap_resolution
        x_pixel = self.obs_size[0] - ((y - self.workspace[1][0]) / self.heightmap_resolution)

        return int(x_pixel), int(y_pixel)

    def _getPrimativeHeight(self, motion_primative, x, y):
        '''
        Get the z position for the given action using the current heightmap.
        Args:
          - motion_primative: Pick/place motion primative
          - x: X coordinate for action
          - y: Y coordinate for action
          - offset: How much to offset the action along approach vector
        Returns: Valid Z coordinate for the action
        '''
        x_pixel, y_pixel = self._getPixelsFromXY(x, y)
        local_region = self.heightmap[int(max(x_pixel - self.obs_size[0] / 20, 0)):
                                      int(min(x_pixel + self.obs_size[0] / 20, self.obs_size[0])),
                                      int(max(y_pixel - self.obs_size[1] / 20, 0)):
                                      int(min(y_pixel + self.obs_size[1] / 20, self.obs_size[1]))]
        safe_z_pos = np.max(local_region) + self.workspace[2][0]
        safe_z_pos = safe_z_pos - 0.04 if motion_primative == self.PICK_PRIMATIVE else safe_z_pos - 0.006

        return safe_z_pos

    def _getSpecificAction(self, action):
        """
        decode input action base on self.action_sequence
        Args:
          action: action tensor
        Returns: motion_primative, x, y, z, rot
        """
        primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a),
                                                          ['p', 'x', 'y', 'z', 'r'])
        motion_primative = action[primative_idx] if primative_idx != -1 else self.ur5.holding_state
        x = action[x_idx]
        y = action[y_idx]
        z = action[z_idx] if z_idx != -1 else self._getPrimativeHeight(motion_primative, x, y)
        rot = action[rot_idx] if rot_idx != -1 else 0
        return motion_primative, x, y, z, rot

    def _preProcessObs(self, obs):
        obs = scipy.ndimage.median_filter(obs, 1)
        b = np.linspace(1, 0, 90).reshape(1, 90).repeat(90, axis=0)
        a = np.linspace(0.5, 1, 90).reshape(1, 90).repeat(90, axis=0).T
        b = b * a * 0.01
        obs -= b
        obs *= 0.8
        obs[obs < 0.007] = 0
        return obs

    def getObs(self):
        # get img from camera
        obs = self.img_proxy.getImage()
        # cut img base on workspace size
        pixel_range = (self.ws_x / self.cam_resolution, self.ws_y / self.cam_resolution)
        obs = obs[240 - int(pixel_range[1]/2): 240 + int(pixel_range[1]/2),
                  320 - int(pixel_range[0]/2): 320 + int(pixel_range[0]/2)]
        # process nans
        mask = np.isnan(obs)
        obs[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), obs[~mask])
        # reverse img s.t. table is 0
        obs = -obs
        # obs -= obs.min()
        obs -= obs.reshape(-1)[obs.reshape(-1).argpartition(100)[:100]].mean()
        # resize img to obs size
        obs = skimage.transform.resize(obs, self.obs_size)
        # rotate img
        obs = scipy.ndimage.rotate(obs, 90)
        # save obs copy
        self.heightmap = obs.copy()
        obs = self._preProcessObs(obs)
        obs = obs.reshape(1, 1, obs.shape[0], obs.shape[1])
        return torch.tensor(obs).to(torch.float32)

    def step(self, action):
        p, x, y, z, r = self._getSpecificAction(action)
        if p == self.PICK_PRIMATIVE:
            self.ur5.pick(x, y, z, r)
        elif p == self.PLACE_PRIMATIVE:
            self.ur5.place(x, y, z, r)
        else:
            raise NotImplementedError

    def getGripperClosed(self):
        return self.ur5.gripper.isClosed()

    def getSafeHeight(self, x, y):
        return self.heightmap[max(x-5, 0):min(x+5, 90), max(y-5, 0):min(y+5, 90)].max()
