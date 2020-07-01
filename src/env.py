from src.ur5 import UR5
from src.img_proxy import ImgProxy
import skimage
import scipy
import torch
import ros_numpy
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sk_transform
from scipy.ndimage import median_filter


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

        self.in_hand_size = 24
        self.heightmap_size = 90
        self.in_hand_mode = 'proj'

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

    def _decodeAction(self, action):
        """
        decode input action base on self.action_sequence
        Args:
          action: action tensor

        Returns: motion_primative, x, y, z, rot

        """
        primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a),
                                                          ['p', 'x', 'y', 'z', 'r'])
        motion_primative = action[primative_idx] if primative_idx != -1 else 0
        x = action[x_idx]
        y = action[y_idx]
        z = action[z_idx] if z_idx != -1 else self._getPrimativeHeight(motion_primative, x, y)
        rz, ry, rx = 0, np.pi, 0
        if self.action_sequence.count('r') <= 1:
            rz = action[rot_idx] if rot_idx != -1 else 0
            ry = 0
            rx = 0
        elif self.action_sequence.count('r') == 2:
            rz = action[rot_idx]
            ry = action[rot_idx + 1]
            rx = 0
        elif self.action_sequence.count('r') == 3:
            rz = action[rot_idx]
            ry = action[rot_idx + 1]
            rx = action[rot_idx + 2]

        # [-pi, 0] is easier for the arm(kuka) to execute
        while rz < -np.pi:
            rz += np.pi
            rx = -rx
            ry = -ry
        while rz > 0:
            rz -= np.pi
            rx = -rx
            ry = -ry
        rot = (rx, ry, rz)

        return motion_primative, x, y, z, rot
    def _preProcessObs(self, obs):
        obs = scipy.ndimage.median_filter(obs, 1)
        b = np.linspace(1, 0, 90).reshape(1, 90).repeat(90, axis=0)
        a = np.linspace(0.5, 1, 90).reshape(1, 90).repeat(90, axis=0).T
        b = b * a * 0.01
        obs -= b
        # obs *= 0.8
        obs[obs < 0.007] = 0
        return obs

    def _getPixelsFromPos(self, x, y):
        '''
        Get the x/y pixels on the heightmap for the given coordinates
        Args:
          - x: X coordinate
          - y: Y coordinate
        Returns: (x, y) in pixels corresponding to coordinates
        '''
        y_pixel = (x - self.workspace[0][0]) / self.heightmap_resolution
        x_pixel = 90 - (y - self.workspace[1][0]) / self.heightmap_resolution

        # x = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(pixels.size(0), 1)
        # y = ((90 - pixels[:, 0].float()) * self.heightmap_resolution + self.workspace[1][0]).reshape(pixels.size(0), 1)

        return int(x_pixel), int(y_pixel)

    def getInHandOccupancyGridProj(self, crop, z, rot):
        rx, ry, rz = rot
        # crop = zoom(crop, 2)
        crop = np.round(crop, 5)
        size = self.in_hand_size

        zs = np.array([z + (size / 2 - i) * (self.heightmap_resolution) for i in range(size)])
        zs = zs.reshape((-1, 1, 1))
        zs = zs.repeat(size, 1).repeat(size, 2)
        zs[zs < -(self.heightmap_resolution)] = 100
        c = crop.reshape(1, size, size).repeat(size, 0)
        ori_occupancy = c > zs

        # transform into points
        point = np.argwhere(ori_occupancy)
        # center
        point = point - size / 2
        R = np.array([[np.cos(-rx), 0, np.sin(-rx)],
                      [0, 1, 0],
                      [-np.sin(-rx), 0, np.cos(-rx)]])
        point = R.dot(point.T)
        point = point + size / 2
        point = np.round(point).astype(int)
        point = point.T[(np.logical_and(0 < point.T, point.T < size)).all(1)].T

        occupancy = np.zeros((size, size, size))
        occupancy[point[0], point[1], point[2]] = 1
        occupancy = median_filter(occupancy, size=2)
        occupancy = np.ceil(occupancy)

        projection = np.stack((occupancy.sum(0), occupancy.sum(1), occupancy.sum(2)))
        return projection

    def getInHandImage(self, heightmap, x, y, z, rot, next_heightmap):
        (rx, ry, rz) = rot
        # Pad heightmaps for grasps near the edges of the workspace
        heightmap = np.pad(heightmap, int(self.in_hand_size / 2), 'constant', constant_values=0.0)
        next_heightmap = np.pad(next_heightmap, int(self.in_hand_size / 2), 'constant', constant_values=0.0)

        x, y = self._getPixelsFromPos(x, y)
        x = np.clip(x, self.in_hand_size / 2, self.heightmap_size - 1 - self.in_hand_size / 2)
        y = np.clip(y, self.in_hand_size / 2, self.heightmap_size - 1 - self.in_hand_size / 2)
        x = x + int(self.in_hand_size / 2)
        y = y + int(self.in_hand_size / 2)

        # Get the corners of the crop
        x_min = int(x - self.in_hand_size / 2)
        x_max = int(x + self.in_hand_size / 2)
        y_min = int(y - self.in_hand_size / 2)
        y_max = int(y + self.in_hand_size / 2)

        # Crop both heightmaps
        crop = heightmap[x_min:x_max, y_min:y_max]
        if self.in_hand_mode.find('sub') > -1:
            next_crop = next_heightmap[x_min:x_max, y_min:y_max]

            # Adjust the in-hand image to remove background objects
            next_max = np.max(next_crop)
            crop[crop >= next_max] -= next_max

        # end_effector rotate counter clockwise along z, so in hand img rotate clockwise
        crop = sk_transform.rotate(crop, np.rad2deg(-rz))

        if self.in_hand_mode.find('proj') > -1:
            return self.getInHandOccupancyGridProj(crop, z, rot)
        else:
            return crop.reshape((self.in_hand_size, self.in_hand_size, 1))

    def getHeightmap(self):
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

    def getEmptyInHand(self):
        if self.in_hand_mode.find('proj') > -1:
            return np.zeros((3, self.in_hand_size, self.in_hand_size))
        else:
            return np.zeros((1, self.in_hand_size, self.in_hand_size))

    def getObs(self, action=None):
        old_heightmap = self.heightmap
        self.heightmap = self.getHeightmap()

        if action is None or self.ur5.holding_state == False:
            in_hand_img = self.getEmptyInHand()
        else:
            motion_primative, x, y, z, rot = self._decodeAction(action)
            z -= self.ws_center[2]
            in_hand_img = self.getInHandImage(old_heightmap[0, 0], x, y, z, rot, self.heightmap[0, 0])
        in_hand_img = in_hand_img.reshape(1, in_hand_img.shape[0], in_hand_img.shape[1], in_hand_img.shape[2])
        return self.heightmap, torch.tensor(in_hand_img).to(torch.float32)

    def step(self, action):
        p, x, y, z, r = self._decodeAction(action)
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
