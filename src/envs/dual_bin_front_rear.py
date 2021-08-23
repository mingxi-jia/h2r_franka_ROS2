import numpy as np
from src.envs.env import *

class Bin():
    def __init__(self, center_rc, center_ws, size_pixle, name=None):
        self.center_rc = center_rc
        self.center_ws = center_ws
        self.size_pixle = size_pixle
        self.name = name
        self.empty_thres = 0.006

    def GetVertexRC(self):
        '''
        get the vertexs of the bin in [row, colonm]
        '''
        return [[int(self.center_rc[0] - self.size_pixle / 2), int(self.center_rc[1] - self.size_pixle / 2)],
                [int(self.center_rc[0] - self.size_pixle / 2), int(self.center_rc[1] + self.size_pixle / 2)],
                [int(self.center_rc[0] + self.size_pixle / 2), int(self.center_rc[1] - self.size_pixle / 2)],
                [int(self.center_rc[0] + self.size_pixle / 2), int(self.center_rc[1] + self.size_pixle / 2)]]

    def GetRangeRC(self):
        '''
        get the vertexs of the bin in [row, colonm]
        '''
        return [int(self.center_rc[0] - self.size_pixle / 2), int(self.center_rc[0] + self.size_pixle / 2),
                int(self.center_rc[1] - self.size_pixle / 2), int(self.center_rc[1] + self.size_pixle / 2)]

    def GetObs(self, obs):
        pixle_range = self.GetRangeRC()
        return obs[0, 0, pixle_range[0]:pixle_range[1], pixle_range[2]:pixle_range[3]]

    def IsEmpty(self, obs):
        return np.all(self.GetObs(obs) < self.empty_thres)


class DualBinFrontRear(Env):
    def __init__(self, ws_center=(-0.5539, 0.0298, -0.145), ws_x=0.8, ws_y=0.8, cam_resolution=0.00155,
                 cam_size=(256, 256), action_sequence='xyrp', in_hand_mode='proj', pick_offset=0.05, place_offset=0.05,
                 in_hand_size=24, obs_source='reconstruct', safe_z_region=1 / 20, place_open_pos=0):
        super().__init__(ws_center, ws_x, ws_y, cam_resolution, cam_size, action_sequence, in_hand_mode, pick_offset,
                       place_offset, in_hand_size, obs_source, safe_z_region, place_open_pos)

        self.gripper_reach = 0.05
        # self.gripper_reach = 0.0
        assert (ws_x / cam_size[0]) == (ws_y / cam_size[1])
        self.pixel_size = ws_x / cam_size[0]
        self.cam_size = cam_size
        self.release_z = 0.15
        left_bin_center_rc = [103, 56]  # rc: row, colonm
        right_bin_center_rc = [103, 200]
        left_bin_center_ws = self.pixel2meter(left_bin_center_rc)
        right_bin_center_ws = self.pixel2meter(right_bin_center_rc)
        self.bin_size = 0.25
        self.bin_size_pixle = 80
        self.left_bin = Bin(left_bin_center_rc, left_bin_center_ws, self.bin_size_pixle, name='left_bin')
        self.right_bin = Bin(right_bin_center_rc, right_bin_center_ws, self.bin_size_pixle, name='right_bin')
        self.move_action = (ws_center[0], ws_center[1], 0.20 + self.workspace[2][0], (0, 0, 0))  # xyzr
        self.bins = [self.left_bin, self.right_bin]
        self.picking_bin_id = None

    def checkWS(self):
        obs, in_hand = env.getObs(None)
        plt.imshow(obs[0, 0])
        plt.plot((128, 128), (0, 255), color='r', linewidth=1)
        plt.plot((0, 255), (145, 145), color='r', linewidth=1)
        plt.scatter(self.left_bin.center_rc[1], self.left_bin.center_rc[0], color='r', linewidths=1, marker='+')
        plt.scatter(self.right_bin.center_rc[1], self.right_bin.center_rc[0], color='r', linewidths=1, marker='+')
        left_bin_vertexs_rc = self.left_bin.GetVertexRC()
        right_bin_vertexs_rc = self.right_bin.GetVertexRC()
        for vertex_rc in left_bin_vertexs_rc:
            plt.scatter(vertex_rc[1], vertex_rc[0], color='y', linewidths=1, marker='+')
        for vertex_rc in right_bin_vertexs_rc:
            plt.scatter(vertex_rc[1], vertex_rc[0], color='y', linewidths=1, marker='+')
        # plt.scatter(16, 63, color='y', linewidths=1, marker='+')
        # plt.scatter(16, 143, color='y', linewidths=1, marker='+')
        # plt.scatter(96, 63, color='y', linewidths=1, marker='+')
        # plt.scatter(96, 143, color='y', linewidths=1, marker='+')
        # plt.scatter(160, 63, color='y', linewidths=1, marker='+')
        # plt.scatter(160, 143, color='y', linewidths=1, marker='+')
        # plt.scatter(240, 63, color='y', linewidths=1, marker='+')
        # plt.scatter(240, 143, color='y', linewidths=1, marker='+')
        plt.colorbar()
        fig, axs = plt.subplots(nrows=1, ncols=2)
        obs0 = axs[0].imshow(self.left_bin.GetObs(obs))
        fig.colorbar(obs0, ax=axs[0])
        obs1 = axs[1].imshow(self.right_bin.GetObs(obs))
        fig.colorbar(obs1, ax=axs[1])
        plt.show()

    def pixel2meter(self, pixel, rc=None):
        if rc == 'r':  # row
            return (pixel - self.cam_size[1] / 2) * self.pixel_size + self.ws_center[1]
        elif rc == 'c':  # colonm
            return (pixel - self.cam_size[0] / 2) * self.pixel_size + self.ws_center[0]
        elif len(pixel) == 2: # rc
            return [self.pixel2meter(pixel[1], 'c'), self.pixel2meter(pixel[0], 'r')]
        raise NotImplementedError

    def place_action(self):
        while 1:
            xy = np.random.normal(0, self.bin_size / 6, (2))
            if np.all(-(self.bin_size / 2) < xy < (self.bin_size / 2)):
                break
        return (xy[0], xy[1], 0.15 + self.workspace[2][0], (0, 0, 0))  # xyzr

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
        local_region = self.heightmap[int(max(x_pixel - self.cam_size[0] * self.safe_z_region, 0)):
                                      int(min(x_pixel + self.cam_size[0] * self.safe_z_region, self.cam_size[0])),
                                      int(max(y_pixel - self.cam_size[1] * self.safe_z_region, 0)):
                                      int(min(y_pixel + self.cam_size[1] * self.safe_z_region, self.cam_size[1]))]
        if motion_primative == self.PICK_PRIMATIVE:
            safe_z_pos = np.median(local_region.flatten()[(-local_region).flatten().argsort()[:25]]) + \
                         self.workspace[2][0]
            safe_z_pos = safe_z_pos - self.gripper_reach
        else:
            safe_z_pos = self.release_z + self.workspace[2][0]
        safe_z_pos = max(safe_z_pos, self.workspace[2][0])

        return safe_z_pos

    def _decodeAction(self, action, bin_id):
        p, x, y, z, r = super()._decodeAction(action)
        assert -(self.bin_size / 2) < x < (self.bin_size / 2) and\
               -(self.bin_size / 2) < y < (self.bin_size / 2)
        bin_center_ws = self.bins[bin_id].center_ws
        x += bin_center_ws[0]
        y += bin_center_ws[1]
        return p, x, y, z, r

    def step(self, action):
        '''
        In this env, the agent only control pick action.
        A place action will be added by the env.
        '''
        if self.picking_bin_id is None:
            for bin, id in enumerate(self.bins):
                if not bin.IsEmpty():
                    self.picking_bin_id = id
                    break
        # pick
        p, x, y, z, r = self._decodeAction(action, self.picking_bin_id)
        self.ur5.only_pick(x, y, z, r)
        # move
        x, y, z, r = self.move_action
        rx, ry, rz = r
        self.ur5.moveToP(x, y, z, rx, ry, rz)

        # place
        x, y, z, r = self._decodeAction(self.place_action(), (self.picking_bin_id + 1) % 2)
        reward = int(not self.ur5.only_place(x, y, z, r, return_isClosed=True))
        self.old_heightmap = self.heightmap

        # Observation
        cam_obs = self.getObs(None)
        if self.bins[self.picking_bin_id].IsEmpty(cam_obs): # if one episode ends
            self.picking_bin_id = (self.picking_bin_id + 1) % 2
            done = True
        else:
            done = False
        obs = self.bins[self.picking_bin_id].GetObs(cam_obs)

        # move
        x, y, z, r = self.move_action
        rx, ry, rz = r
        self.ur5.moveToP(x, y, z, rx, ry, rz)

        return obs, reward, done

if __name__ == '__main__':
    import rospy
    rospy.init_node('image_proxy')
    env = DualBinFrontRear(ws_x=0.8, ws_y=0.8, cam_size=(256, 256), obs_source='reconstruct')
    while True:
        env.checkWS()
