from src.ur5 import UR5
from src.img_proxy import ImgProxy
from src.cloud_proxy import CloudProxy
import skimage
import scipy
import torch
import ros_numpy
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sk_transform
from scipy.ndimage import median_filter
from src.utils import transformation
from sklearn.impute import SimpleImputer

class Env:
    def __init__(self, ws_center=(-0.5539, 0.0298, -0.145), ws_x=0.3, ws_y=0.3, cam_resolution=0.00155, obs_size=(90, 90),
                 action_sequence='pxyr', in_hand_mode='proj', pick_offset=0.1, place_offset=0.1, in_hand_size=24,
                 obs_source='reconstruct', safe_z_region=1 / 20, place_open_pos=0):
        assert obs_source == 'reconstruct' or 'raw'
        self.ws_center = ws_center
        self.ws_x = ws_x
        self.ws_y = ws_y
        self.workspace = np.asarray([[ws_center[0] - ws_x/2, ws_center[0] + ws_x/2],
                                     [ws_center[1] - ws_y/2, ws_center[1] + ws_y/2],
                                     [ws_center[2], ws_center[2]+0.4]])
        self.cam_resolution = cam_resolution
        self.obs_size = obs_size
        self.action_sequence = action_sequence
        self.heightmap_resolution = ws_x / obs_size[0]

        self.ur5 = UR5(pick_offset, place_offset, place_open_pos)
        self.img_proxy = ImgProxy()
        self.cloud_proxy = CloudProxy()
        self.old_heightmap = np.zeros((self.obs_size[0], self.obs_size[1]))
        self.heightmap = np.zeros((self.obs_size[0], self.obs_size[1]))

        # Motion primatives
        self.PICK_PRIMATIVE = 0
        self.PLACE_PRIMATIVE = 1

        self.in_hand_size = in_hand_size
        self.heightmap_size = obs_size[0]
        self.in_hand_mode = in_hand_mode

        self.ee_offset = 0.095

        self.pick_offset = 0.02
        self.place_offset_1 = 0.005
        self.place_offset_2 = 0.001

        self.obs_source = obs_source
        self.safe_z_region = safe_z_region

    def _getXYFromPixels(self, x_pixel, y_pixel):
        x = x_pixel * self.heightmap_resolution + self.workspace[0][0]
        y = y_pixel * self.heightmap_resolution + self.workspace[1][0]
        return x, y

    def _getPixelsFromXY(self, x, y):
        '''
        Get the x/y pixels on the heightmap for the given coordinates
        Args:
          - x: X coordinate
          - y: Y coordinate
        Returns: (x, y) in pixels corresponding to coordinates
        '''
        x_pixel = (x - self.workspace[0][0]) / self.heightmap_resolution
        y_pixel = (y - self.workspace[1][0]) / self.heightmap_resolution

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
        local_region = self.heightmap[int(max(x_pixel - self.obs_size[0] * self.safe_z_region, 0)):
                                      int(min(x_pixel + self.obs_size[0] * self.safe_z_region, self.obs_size[0])),
                                      int(max(y_pixel - self.obs_size[1] * self.safe_z_region, 0)):
                                      int(min(y_pixel + self.obs_size[1] * self.safe_z_region, self.obs_size[1]))]
        safe_z_pos = np.median(local_region.flatten()[(-local_region).flatten().argsort()[:25]]) + self.workspace[2][0]
        safe_z_pos = safe_z_pos - self.pick_offset if motion_primative == self.PICK_PRIMATIVE else safe_z_pos + self.place_offset_1 + self.place_offset_2
        safe_z_pos = max(safe_z_pos, self.workspace[2][0])

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
        # while rz < -np.pi:
        #     rz += np.pi
        #     rx = -rx
        #     ry = -ry
        # while rz > 0:
        #     rz -= np.pi
        #     rx = -rx
        #     ry = -ry
        rot = (rx, ry, rz)

        return motion_primative, x, y, z, rot
    def _preProcessObs(self, obs):
        # obs = scipy.ndimage.median_filter(obs, 2)
        b = np.linspace(-0.5, 0.5, self.heightmap_size).reshape(1, self.heightmap_size).repeat(self.heightmap_size, axis=0)
        # a = np.linspace(0.5, 1, self.heightmap_size).reshape(1, self.heightmap_size).repeat(self.heightmap_size, axis=0).T
        b = b * 0.01
        obs += b
        # obs *= 0.9
        # obs[obs < 0.007] = 0
        return obs

    def getInHandOccupancyGridProj(self, crop, z, rot):
        rx, ry, rz = rot
        # crop = zoom(crop, 2)
        crop = np.round(crop, 5)
        size = self.in_hand_size

        zs = np.array([z + (-size / 2 + i) * (self.heightmap_resolution) for i in range(size)])
        zs = zs.reshape((1, 1, -1))
        zs = zs.repeat(size, 0).repeat(size, 1)
        # zs[zs<-(self.heightmap_resolution)] = 100
        c = crop.reshape(size, size, 1).repeat(size, 2)
        ori_occupancy = c > zs

        # transform into points
        point = np.argwhere(ori_occupancy)
        # center
        ori_point = point - size / 2
        R = transformation.euler_matrix(rx, ry, rz)[:3, :3].T
        point = R.dot(ori_point.T)
        point = point + size / 2
        point = np.round(point).astype(int)
        point = point.T[(np.logical_and(0 < point.T, point.T < size)).all(1)].T

        occupancy = np.zeros((size, size, size))
        occupancy[point[0], point[1], point[2]] = 1
        occupancy = median_filter(occupancy, size=2)
        occupancy = np.ceil(occupancy)

        projection = np.stack((occupancy.sum(0), occupancy.sum(1), occupancy.sum(2)))
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # axs[0].imshow(projection[0])
        # axs[1].imshow(projection[1])
        # axs[2].imshow(projection[2])
        # fig.show()
        return projection

    def getInHandImage(self, heightmap, x, y, z, rot, next_heightmap):
        (rx, ry, rz) = rot
        # Pad heightmaps for grasps near the edges of the workspace
        heightmap = np.pad(heightmap, int(self.in_hand_size / 2), 'constant', constant_values=0.0)
        next_heightmap = np.pad(next_heightmap, int(self.in_hand_size / 2), 'constant', constant_values=0.0)

        x, y = self._getPixelsFromXY(x, y)
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

        if self.in_hand_mode.find('proj') > -1:
            return self.getInHandOccupancyGridProj(crop, z, rot)
        else:
            # end_effector rotate counter clockwise along z, so in hand img rotate clockwise
            crop = sk_transform.rotate(crop, np.rad2deg(-rz))
            return crop.reshape(1, self.in_hand_size, self.in_hand_size)

    # def getHeightmapRaw(self):
    #     # get img from camera
    #     obs = self.img_proxy.getImage()
    #     # cut img base on workspace size
    #     pixel_range = (self.ws_x / self.cam_resolution, self.ws_y / self.cam_resolution)
    #     obs = obs[242 - int(pixel_range[1]/2): 242 + int(pixel_range[1]/2),
    #               320 - int(pixel_range[0]/2): 320 + int(pixel_range[0]/2)]
    #     # process nans
    #     imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    #     obs = imputer.fit_transform(obs)
    #     # reverse img s.t. table is 0
    #     obs = -obs
    #     # obs -= obs.min()
    #     obs -= -0.90987
    #     # resize img to obs size
    #     obs = skimage.transform.resize(obs, self.obs_size)
    #     # rotate img
    #     obs = scipy.ndimage.rotate(obs, 90)
    #     # save obs copy
    #     # self.heightmap = obs.copy()
    #     obs = self._preProcessObs(obs)
    #     # obs = obs.reshape(1, 1, obs.shape[0], obs.shape[1])
    #     return obs

    def getHeightmapReconstruct(self):
        # get img from camera
        obss = []
        i = 0
        while i < 10:
            obs = self.cloud_proxy.getProjectImg(self.ws_x, self.obs_size[0])
            if obs.shape != (256, 256):
                continue
            else:
                obss.append(obs)
                i += 1
        obs = np.median(obss, axis=0)
        # reverse img s.t. table is 0
        obs = -obs
        # obs -= obs.min()
        obs -= -1.0268  # for daul bin setup
        # obs -= -1.045  # for HaoJie baseline
        # rotate img
        # obs = scipy.ndimage.rotate(obs, 180)
        # save obs copy
        # self.heightmap = obs.copy()
        obs = self._preProcessObs(obs)
        # obs = obs.reshape(1, 1, obs.shape[0], obs.shape[1])
        return obs

    def getEmptyInHand(self):
        if self.in_hand_mode.find('proj') > -1:
            return np.zeros((3, self.in_hand_size, self.in_hand_size))
        else:
            return np.zeros((1, self.in_hand_size, self.in_hand_size))

    def getObs(self, action=None):
        old_heightmap = self.heightmap
        if self.obs_source == 'reconstruct':
            self.heightmap = self.getHeightmapReconstruct()
        else:
            self.heightmap = self.getHeightmapRaw()

        if action is None or self.ur5.holding_state == False:
            in_hand_img = self.getEmptyInHand()
        else:
            motion_primative, x, y, z, rot = self._decodeAction(action)
            z -= self.ws_center[2]
            in_hand_img = self.getInHandImage(self.old_heightmap, x, y, z, rot, self.heightmap)
        in_hand_img = in_hand_img.reshape(1, in_hand_img.shape[0], in_hand_img.shape[1], in_hand_img.shape[2])
        heightmap = self.heightmap.reshape(1, 1, self.heightmap.shape[0], self.heightmap.shape[1])

        return torch.tensor(heightmap).to(torch.float32), torch.tensor(in_hand_img).to(torch.float32)

    def step(self, action):
        p, x, y, z, r = self._decodeAction(action)
        if p == self.PICK_PRIMATIVE:
            self.ur5.pick(x, y, z, r)
            self.place_offset_1 = z - self.workspace[2, 0]
        elif p == self.PLACE_PRIMATIVE:
            self.ur5.place(x, y, z, r)
        else:
            raise NotImplementedError
        self.old_heightmap = self.heightmap

    def getGripperClosed(self):
        return self.ur5.gripper.isClosed()

    def getSafeHeight(self, x, y):
        return self.heightmap[max(x-5, 0):min(x+5, 90), max(y-5, 0):min(y+5, 90)].max()

    def plotObs(self, cam_resolution):
        self.cam_resolution = cam_resolution
        obs = self.getHeightmapReconstruct()
        plt.imshow(obs[0, 0])
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    import rospy
    # plt.style.use('grayscale')
    rospy.init_node('image_proxy')
    # env = Env(ws_x=0.4, ws_y=0.4, obs_size=(128, 128))
    env = Env(ws_x=0.8, ws_y=0.8, obs_size=(256, 256), obs_source='reconstruct')
    while True:
        obs, in_hand = env.getObs(None)
        plt.imshow(obs[0, 0])
        plt.plot((128, 128), (0, 255), color='r', linewidth=1)
        plt.plot((0, 255), (145, 145), color='r', linewidth=1)
        plt.scatter(16, 63, color='y', linewidths=1, marker='+')
        plt.scatter(16, 143, color='y', linewidths=1, marker='+')
        plt.scatter(96, 63, color='y', linewidths=1, marker='+')
        plt.scatter(96, 143, color='y', linewidths=1, marker='+')
        plt.scatter(160, 63, color='y', linewidths=1, marker='+')
        plt.scatter(160, 143, color='y', linewidths=1, marker='+')
        plt.scatter(240, 63, color='y', linewidths=1, marker='+')
        plt.scatter(240, 143, color='y', linewidths=1, marker='+')
        plt.colorbar()
        fig, axs = plt.subplots(nrows=1, ncols=2)
        obs0 = axs[0].imshow(obs[0, 0, 63:143, 16:96])
        fig.colorbar(obs0, ax=axs[0])
        obs1 = axs[1].imshow(obs[0, 0, 63:143, 160:240])
        fig.colorbar(obs1, ax=axs[1])
        plt.show()
