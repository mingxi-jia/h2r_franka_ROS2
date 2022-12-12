import logging
import time
import numpy as np
from src.envs.env import *
from scipy.ndimage.interpolation import rotate
import rospy
from src.bin_constrain import ZProtect
import matplotlib.pyplot as plt
from torchvision import transforms as T
import threading

class Bin():
    def __init__(self, center_rc, center_ws, size_pixel, action_range_pixel, sensor_type, name=None, empty_thres=10):
        self.center_rc = center_rc
        self.center_ws = center_ws
        self.size_pixel = size_pixel
        self.action_range_pixel = action_range_pixel
        self.inner_padding = int((size_pixel - action_range_pixel) / 2)
        # self.inner_inner_padding = 10
        assert self.inner_padding != 0
        self.name = name
        self.empty_rgb_thres = empty_thres
        self.empty_pixel_thres = 10
        self.empty_img = None
        self.last_img = None
        self.sensor_type = sensor_type
        # self.empty_thres = 0.
        self.blurrer = T.GaussianBlur(kernel_size=7, sigma=3)
        self.obs = None

    def reset(self, obs):
        self.empty_img = self.GetActionWiseObs(obs, pre_process=False)
        self.empty_img = self.blurrer(self.empty_img)

    def updateLastImg(self, obs):
        last_img = self.GetActionWiseObs(obs, pre_process=False)
        self.last_img = self.blurrer(last_img)

    def GetVertexRC(self):
        '''
        get the vertexs of the bin in [row, colonm]
        '''
        return [[int(self.center_rc[0] - (self.size_pixel - 1) / 2), int(self.center_rc[1] - (self.size_pixel - 1) / 2)],
                [int(self.center_rc[0] - (self.size_pixel - 1) / 2), int(self.center_rc[1] + (self.size_pixel - 1) / 2)],
                [int(self.center_rc[0] + (self.size_pixel - 1) / 2), int(self.center_rc[1] - (self.size_pixel - 1) / 2)],
                [int(self.center_rc[0] + (self.size_pixel - 1) / 2), int(self.center_rc[1] + (self.size_pixel - 1) / 2)]]

    def GetRangeRC(self):
        '''
        get the vertexs of the bin in [row, colonm]
        '''
        return [int(self.center_rc[0] - (self.size_pixel - 1) / 2), int(self.center_rc[0] + (self.size_pixel - 1) / 2),
                int(self.center_rc[1] - (self.size_pixel - 1) / 2), int(self.center_rc[1] + (self.size_pixel - 1) / 2)]

    def GetObs(self, obs, pre_process=True):
        pixel_range = self.GetRangeRC()
        bin_obs = obs[:, :, pixel_range[0]:pixel_range[1]+1, pixel_range[2]:pixel_range[3]+1]
        if pre_process:
            if self.sensor_type == 'nrgbd':
                bin_obs[:, :3] /= 10
            elif self.sensor_type == '000d':
                bin_obs[:, :3] = 0
            elif self.sensor_type == 'nrgb0':
                bin_obs[:, :3] /= 10
                bin_obs[:, -1] = 0
            elif self.sensor_type == 'd':
                bin_obs = bin_obs[:, -1]
            else:
                assert self.sensor_type == 'rgbd'
        return bin_obs

    def GetActionWiseObs(self, obs, pre_process=True):
        bin_obs = self.GetObs(obs, pre_process=pre_process)

        return bin_obs[0, :, self.inner_padding:-self.inner_padding, self.inner_padding:-self.inner_padding]

    def IsImageDifferent(self, new_img, baseline_img):
        # wheather there are less than empty_thres pixles that changed less than empty_thres
        bin_img = self.GetActionWiseObs(new_img, pre_process=False)
        bin_img = self.blurrer(bin_img)
        diff_img = bin_img - baseline_img
        diff_img = (diff_img[:-1] * 255).abs().sum(axis=0)
        # plt.figure()
        # plt.imshow(diff_img)
        # plt.colorbar()
        # plt.title('Difference image for ' + self.name)
        # plt.show()
        diff_img = diff_img > self.empty_rgb_thres
        return diff_img.sum() < self.empty_pixel_thres

    def IsEmpty(self, obs):
        is_empty = self.IsImageDifferent(obs, self.empty_img)
        # if is_empty:
        #     self.reset(obs)
        return is_empty

    def IsObjectPlaced(self, obs):
        if self.last_img is not None:
            return self.IsImageDifferent(obs, self.last_img)
        else:
            self.updateLastImg(obs)
            return None

class DualBinFrontRear(Env):
    def __init__(self, ws_center=(-0.4448, -0.17, -0.145), ws_x=0.8, ws_y=0.8, cam_resolution=0.00155, sensor_type='rgbd',
                 cam_size=(256, 256), action_sequence='xyrp', in_hand_mode='proj', pick_offset=0.05, place_offset=0.05,
                 in_hand_size=24, obs_source='reconstruct', safe_z_region=1 / 20, place_open_pos=0, bin_size=0, bin_size_pixel=112,
                 z_heuristic=None):
        super().__init__(ws_center=ws_center, ws_x=ws_x, ws_y=ws_y, cam_resolution=cam_resolution, obs_size=cam_size,
                         action_sequence=action_sequence, in_hand_mode=in_hand_mode, pick_offset=pick_offset,
                         place_offset=place_offset, in_hand_size=in_hand_size, obs_source=obs_source,
                         safe_z_region=safe_z_region, place_open_pos=place_open_pos)

        self.gripper_depth = 0.055
        # self.gripper_depth = 0.0
        assert (ws_x / cam_size[0]) == (ws_y / cam_size[1])
        assert bin_size != 0
        self.pixel_size = ws_x / cam_size[0]
        self.cam_size = cam_size
        self.release_z = 0.2
        left_bin_center_rc = [102.5, 55.5]  # rc: row, colonm
        right_bin_center_rc = [102.5, 199.5]
        left_bin_center_ws = self.pixel2ws(left_bin_center_rc)
        right_bin_center_ws = self.pixel2ws(right_bin_center_rc)
        self.action_range = 0.25  # !!! important, action safety guarantee
        self.action_range_pixel = int(0.25 / self.pixel_size)  # !!! important, action safety guarantee
        # self.bin_size
        self.bin_size_pixel = bin_size_pixel
        self.in_hand_size = 32
        empty_thres = 50
        self.left_bin = Bin(left_bin_center_rc, left_bin_center_ws, self.bin_size_pixel, self.action_range_pixel,
                            sensor_type=sensor_type, name='left_bin', empty_thres=empty_thres)
        self.right_bin = Bin(right_bin_center_rc, right_bin_center_ws, self.bin_size_pixel, self.action_range_pixel,
                             sensor_type=sensor_type, name='right_bin', empty_thres=empty_thres)
        self.move_action = ((left_bin_center_ws[0] + right_bin_center_ws[0]) / 2,
                            (left_bin_center_ws[1] + right_bin_center_ws[1]) / 2,
                            0.28 + self.workspace[2][0], (0, 0, 0))  # xyzr
        self.bins = [self.left_bin, self.right_bin]
        self.picking_bin_id = None
        self.r_action = (0, 0, 1.571)
        self.z_heuristic = z_heuristic
        self.num_step = 0
        self.max_step = 45
        self.srd = None

        # bin z protection
        self.z_bin_constrain = ZProtect(bin_size + 0.035, None, 55, 0.1)

        self.State = None
        self.Reward = None
        self.Action = None
        self.IsRobotReady = None
        self.SENTINEL = None

    def getObs(self, action=None):
        obs, in_hand = super(DualBinFrontRear, self).getObs(action=action)
        # obs[obs > 0.2] = obs.mean()
        # return obs.clip(max=0.2), in_hand
        return obs, in_hand

    def checkWS(self):
        obs, in_hand = self.getObs(None)
        d_img = obs[0, -1]
        rgb_img = obs[0, :-1].permute(1, 2, 0).clone()
        rgb_img += 0.5
        rgb_img *= 255
        rgb_img = rgb_img.int()
        for img in [d_img, rgb_img]:
            plt.figure()
            plt.imshow(img)
            plt.colorbar()
            plt.plot((127.5, 127.5), (0, 255), color='r', linewidth=1)
            # plt.plot((0, 255), (145, 145), color='r', linewidth=1)
            # plt.plot((0, 255), (150, 150), color='r', linewidth=1, alpha=0.8)
            plt.scatter(127.5, 127.5, color='w', linewidths=2, marker='+')
            plt.scatter(self.left_bin.center_rc[1], self.left_bin.center_rc[0], color='r', linewidths=1, marker='+')
            plt.scatter(self.right_bin.center_rc[1], self.right_bin.center_rc[0], color='r', linewidths=1, marker='+')
            left_bin_vertexs_rc = self.left_bin.GetVertexRC()
            right_bin_vertexs_rc = self.right_bin.GetVertexRC()
            for vertex_rc in left_bin_vertexs_rc:
                plt.scatter(vertex_rc[1], vertex_rc[0], color='y', linewidths=1, marker='+')
            for vertex_rc in right_bin_vertexs_rc:
                plt.scatter(vertex_rc[1], vertex_rc[0], color='y', linewidths=1, marker='+')
            plt.scatter(16, 63, color='r', linewidths=1, marker='+')
            plt.scatter(16, 142, color='r', linewidths=1, marker='+')
            plt.scatter(95, 63, color='r', linewidths=1, marker='+')
            plt.scatter(95, 142, color='r', linewidths=1, marker='+')
            plt.scatter(160, 63, color='r', linewidths=1, marker='+')
            plt.scatter(160, 142, color='r', linewidths=1, marker='+')
            plt.scatter(239, 63, color='r', linewidths=1, marker='+')
            plt.scatter(239, 142, color='r', linewidths=1, marker='+')
            plt.scatter(127.5, 127.5, color='g', linewidths=1, marker='+')  # center of the workspace

        # plt.colorbar()
        fig, axs = plt.subplots(nrows=1, ncols=2)
        obs0 = axs[0].imshow(self.left_bin.GetObs(obs, pre_process=True)[0, -1].clamp(-0.02, 0.015))
        fig.colorbar(obs0, ax=axs[0])
        obs1 = axs[1].imshow(self.right_bin.GetObs(obs, pre_process=True)[0, -1].clamp(-0.02, 0.015))
        fig.colorbar(obs1, ax=axs[1])

        fig, axs = plt.subplots(nrows=1, ncols=2)
        obs0 = axs[0].imshow(self.left_bin.GetObs(obs, pre_process=True)[0, :-1].permute(1,2,0))
        obs1 = axs[1].imshow(self.right_bin.GetObs(obs, pre_process=True)[0, :-1].permute(1,2,0))

        fig, axs = plt.subplots(nrows=1, ncols=2)
        img0 = self.left_bin.GetObs(obs, pre_process=False)[0, :-1].permute(1,2,0)
        img0 += 0.5
        img0 *= 255
        obs0 = axs[0].imshow(img0.int())
        img1 = self.right_bin.GetObs(obs, pre_process=False)[0, :-1].permute(1,2,0)
        img1 += 0.5
        img1 *= 255
        obs1 = axs[1].imshow(img1.int())

        plt.show()

    def pixel2ws(self, pixel, rc=None):
        if rc == 'r':  # row
            return (pixel - (self.cam_size[1] - 1) / 2) * self.pixel_size + self.ws_center[0]
        elif rc == 'c':  # colonm
            return (pixel - (self.cam_size[0] - 1) / 2) * self.pixel_size + self.ws_center[1]
        elif len(pixel) == 2:  # rc
            return [self.pixel2ws(pixel[0], 'r'), self.pixel2ws(pixel[1], 'c')]
        raise NotImplementedError

    def place_action(self):
        if not self.ur5.holding_state:
            if self.action_sequence == 'xyrp':
                return (0, 0, 0, 0)
            elif self.action_sequence == 'xyzrp':
                return (0, 0, 0, 0, 0)
        while 1:
            xy = np.random.normal(0, self.action_range / 6, (2))
            # place obj in 2/3 size of workspace
            if ((-(self.action_range / 3) < xy) & (xy < (self.action_range / 3))).all():
                break
        rz = np.random.uniform(0, np.pi)
        if self.action_sequence == 'xyrp':
            return (xy[0], xy[1], rz, 0)  # xyrp
        elif self.action_sequence == 'xyzrp':
            return (xy[0], xy[1], 0, rz, 0)  # xyzrp

    def _getPixelsFromXY(self, x, y):
        '''
        Get the x/y pixels on the rgbd_img for the given coordinates
        Args:
          - x: X coordinate
          - y: Y coordinate
        Returns: (x, y) in pixels corresponding to coordinates
        '''
        y_pixel = (x - self.workspace[0][0]) / self.rgbd_img_resolution
        x_pixel = (y - self.workspace[1][0]) / self.rgbd_img_resolution

        return int(x_pixel), int(y_pixel)

    def isActionEmpty(self, action, hm_thres):
        '''

        :param action: input action to the env
        :return: whether the rigion in action (a_x, a_y) is empty
        '''
        primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a),
                                                          ['p', 'x', 'y', 'z', 'r'])
        x = action[x_idx]
        y = action[y_idx]
        assert -(self.action_range / 2) <= x <= (self.action_range / 2) and \
               -(self.action_range / 2) <= y <= (self.action_range / 2)
        bin_center_ws = self.bins[self.picking_bin_id].center_ws
        x += bin_center_ws[0]
        y += bin_center_ws[1]
        col_pixel, row_pixel = self._getPixelsFromXY(x, y)
        local_region = self.rgbd_img[0, -1, int(max(row_pixel - self.in_hand_size * self.safe_z_region / 2, 0)):
                                     int(min(row_pixel + self.in_hand_size * self.safe_z_region / 2, self.cam_size[1])),
                                     int(max(col_pixel - self.in_hand_size * self.safe_z_region / 2, 0)):
                                     int(min(col_pixel + self.in_hand_size * self.safe_z_region / 2, self.cam_size[0]))]
        hm_at_action = np.median(local_region.flatten()[(-local_region).flatten().argsort()[:25]])
        return hm_at_action < hm_thres

    def _getPrimitiveHeight(self, motion_primative, x, y, rz=None, z=None, bin_z=0):
        '''
        Get the z position for the given action using the current rgbd_img.
        Args:
          - motion_primative: Pick/place motion primative
          - x: X coordinate for action
          - y: Y coordinate for action
          - offset: How much to offset the action along approach vector
        Returns: Valid Z coordinate for the action
        '''
        col_pixel, row_pixel = self._getPixelsFromXY(x, y)
        # local_region = K.warp_affine(local_region.unsqueeze(0).unsqueeze(0),
        #                              transform, (self.in_hand_size, self.in_hand_size),
        #                              mode='nearest', padding_mode='border').squeeze(0).squeeze(0)

        if motion_primative == self.PICK_PRIMATIVE:
            local_region = self.rgbd_img[0, -1, int(max(row_pixel - self.in_hand_size, 0)):
                                         int(min(row_pixel + self.in_hand_size, self.cam_size[1])),
                                         int(max(col_pixel - self.in_hand_size, 0)):
                                         int(min(col_pixel + self.in_hand_size, self.cam_size[0]))] # local_region is x4 large as ih_img

            local_region = rotate(local_region, angle=-rz * 180 / np.pi, reshape=False)
            patch = local_region[(self.in_hand_size - 16):(self.in_hand_size + 16),
                                 (self.in_hand_size - 4):(self.in_hand_size + 4)]
            if z is None:
                egde = patch.copy()
                egde[5:-5] = 0
                # safe_z_pos = max(np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) - self.gripper_depth,
                #                  np.mean(egde.flatten()[(-egde).flatten().argsort()[2:12]]) - self.gripper_depth / 1.5)
                # Only safe with z collision detection
                center = local_region[(self.in_hand_size - 4):(self.in_hand_size + 4),
                                      (self.in_hand_size - 4):(self.in_hand_size + 4)]
                safe_z_pos = np.mean(center.flatten()[(-center).flatten().argsort()[2:12]]) - self.gripper_depth
            else:
                safe_z_pos = np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) + z

            safe_z_pos = safe_z_pos + self.workspace[2, 0] + bin_z
        else:
            safe_z_pos = self.release_z + self.workspace[2, 0]
        safe_z_pos = max(safe_z_pos, self.workspace[2, 0])
        safe_z_pos = min(safe_z_pos, self.workspace[2, 1])
        assert self.workspace[2][0] <= safe_z_pos <= self.workspace[2][1]

        return safe_z_pos

    def _decodeAction(self, action, bin_id):
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
        rot = (rx, ry, rz)
        # bin_z = self.z_bin_constrain.z_protection_func((x, y, rz))
        bin_z = 0
        assert -(self.action_range / 2) <= x <= (self.action_range / 2) and \
               -(self.action_range / 2) <= y <= (self.action_range / 2)
        bin_center_ws = self.bins[bin_id].center_ws
        x += bin_center_ws[0]
        y += bin_center_ws[1]
        if self.z_heuristic == 'residual' and z_idx != -1:
            z = self._getPrimitiveHeight(motion_primative, x, y, rz, z=action[z_idx])
        elif z_idx != -1:
            z = action[z_idx]
        else:
            z = self._getPrimitiveHeight(motion_primative, x, y, rz, bin_z=bin_z)
        # z = action[z_idx] if z_idx != -1 else self._getPrimitiveHeight(motion_primative, x, y, rz)

        return motion_primative, x, y, z, rot

    def reset(self):
        cam_obs, _ = self.getObs(None)
        # print('getting obs')
        if self.picking_bin_id is None:
            empty_obs, _ = self.getObs(None)  # Wait for camera adjust img
            is_bins_empty = bool(input('Are bins loaded?'))
            if is_bins_empty:
                for id, bin in enumerate(self.bins):
                    bin.reset(empty_obs)
            else:
                raise NotImplementedError('Please load bins.')
            cam_obs, _ = self.getObs(None)
            for id, bin in enumerate(self.bins):
                if not bin.IsEmpty(cam_obs):
                    self.picking_bin_id = id
                    break
            if self.picking_bin_id is None:
                raise NotImplementedError  # when both bins are empty

        return torch.tensor([0], dtype=torch.float32).view(1),\
               torch.zeros((1, 1, self.in_hand_size, self.in_hand_size)).to(torch.float32),\
               torch.tensor(self.bins[self.picking_bin_id].GetObs(cam_obs)
                            .reshape(1, -1, self.bin_size_pixel, self.bin_size_pixel)).to(torch.float32)

    def p_reset(self):
        all_state = self.reset()
        # print('get obs')
        self.State.set_var('reset', all_state)
        self.IsRobotReady.set_var('reset', True)

    def p_sensor_processing(self):
        while True:
            # print('processing img')
            # Observation
            # print('got request')
            cam_obs, _ = self.getObs(None)
            # print('got cam_obs')
            if self.bins[self.picking_bin_id].IsEmpty(cam_obs) or self.num_step > self.max_step:  # if one episode ends
                self.num_step = 0
                # self.IsRobotReady.get_var('sensor_processing')
                self.picking_bin_id = (self.picking_bin_id + 1) % 2
                done = True
                # print('episode ends')
                return False
            else:
                done = False
                obs = self.bins[self.picking_bin_id].GetObs(cam_obs).reshape(1, 1, self.bin_size_pixel,
                                                                             self.bin_size_pixel)
                # print('got obs')
                all_state = (torch.tensor([0], dtype=torch.float32).view(1), \
                             torch.zeros((1, 1, self.in_hand_size, self.in_hand_size)).to(torch.float32), \
                             obs.to(torch.float32))
                self.State.set_var('sensor_processing', all_state)
                return True
        print('sensor_processing killed')

    def p_picking(self, action):
        # print('pick at: ', action)
        assert self.picking_bin_id is not None
        # pick
        p, x, y, z, r = self._decodeAction(action, self.picking_bin_id)
        self.ur5.only_pick_fast(x, y, z, r, check_gripper_close_when_pick=True)
        self.r_action = r
        # print('finished picking')

    def p_move_reward(self, return_reward=True):
        self.ur5.gripper.closeGripper()
        # move
        x, y, z, r = self.move_action
        # Add some random noise to protect robot
        y += np.random.uniform(-0.02, 0.02)
        z += np.random.uniform(-0.02, 0.02)
        # place_action = self._decodeAction(self.place_action(), (self.picking_bin_id + 1) % 2)
        # rx, ry, rz = place_action[-1]
        rx, ry, rz = self.r_action
        self.ur5.moveToPT(x, y, z, rx, ry, rz, t=1, t_wait_reducing=0.5)
        # self.ur5.moveToBinCenter()
        reward = self.ur5.checkGripperState()
        # rospy.sleep(0.2)

        reward = torch.tensor(reward, dtype=torch.float32).view(1)
        if self.Reward is not None and return_reward:
            self.Reward.set_var('move_reward', reward)
        # print('moved to the center, reward: ', reward)
        return reward.item()

    def p_place_move_center(self, is_request=True):
        while True:
            # place
            # print('placing')
            p, x, y, z, r = self._decodeAction(self.place_action(), (self.picking_bin_id + 1) % 2)
            z = self.release_z + self.workspace[2][0]
            # self.ur5.only_place_fast(x, y, z, r, no_action_when_empty=False, move2_prepose=False)

            rx, ry, rz = r
            # T = transformation.euler_matrix(rx, ry, rz)
            # pre_pos = np.array([x, y, z])
            # pre_pos += self.pick_offset * T[:3, 2]
            # pre_pos[2] += self.place_offset
            # if move2_prepose:
            #     self.moveToP(*pre_pos, rx, ry, rz)
            # self.ur5.moveToPT(x, y, z, rx, ry, rz, t=1.2, t_wait_reducing=0.5)
            self.ur5.moveToPT(x, y, z, rx, ry, rz, t=.9, t_wait_reducing=0.6)
            # self.gripper.openGripper(position=self.place_open_pos)
            self.ur5.gripper.openGripper()
            if is_request:
                sensor_success = self.p_sensor_processing()
            else:
                sensor_success = True
                rospy.sleep(0.7)

            self.ur5.holding_state = 0
            # if move2_prepose:
            #     self.moveToP(*pre_pos, rx, ry, rz)
            # self.old_heightmap = self.heightmap

            # move
            x, y, z, r = self.move_action
            rx, ry, rz = r
            # print('moving back bins center')
            self.ur5.moveToPT(x, y, z, rx, ry, rz, t=0.7)

            if sensor_success:
                self.IsRobotReady.set_var('place', True)
                break
        # print('robot is ready for picking')

    def sendAction(self, action):
        self.srd = self.step(action)

    def getSRD(self):
        assert self.srd is not None
        return self.srd

    def ObsThread(self):
        self.obs = self.getObs(None)

    def step(self, action):
        '''
        In this env, the agent only control pick action.
        A place action will be added by the env automatically.
        '''
        assert self.picking_bin_id is not None
        # pick
        p, x, y, z, r = self._decodeAction(action, self.picking_bin_id)
        self.ur5.only_pick_fast(x, y, z, r, check_gripper_close_when_pick=True)
        self.num_step += 1
        r_action = r
        # move
        x, y, z, r = self.move_action
        # place_action =
        # rx, ry, rz = place_action[-1]
        rx, ry, rz = r_action
        self.ur5.moveToPT(x, y, z, rx, ry, rz, t=0.8)
        reward = self.ur5.checkGripperState()
        # place
        p, x, y, z, r = self._decodeAction(self.place_action(), (self.picking_bin_id + 1) % 2)
        z = self.release_z + self.workspace[2][0]
        self.ur5.only_place_fast(x, y, z, r, no_action_when_empty=False, move2_prepose=False)
        self.old_rgbd_img = self.rgbd_img
        # Observation
        rospy.sleep(0.1)
        obs_thread = threading.Thread(target=self.ObsThread)
        obs_thread.start()
        rospy.sleep(0.3)

        # move
        x, y, z, r = self.move_action
        rx, ry, rz = r
        self.ur5.moveToPT(x, y, z, rx, ry, rz, t=0.8)

        obs_thread.join()
        cam_obs, _ = self.obs
        if self.bins[self.picking_bin_id].IsEmpty(cam_obs) or self.num_step > self.max_step:  # if the bin is emptyed
            self.num_step = 0
            self.picking_bin_id = (self.picking_bin_id + 1) % 2
            done = True
            # place at the center of the bin
            p, x, y, z, r = self._decodeAction(self.place_action(), (self.picking_bin_id + 1) % 2)
            z = self.release_z + self.workspace[2][0]
            # self.ur5.only_place_fast(x, y, z, r, no_action_when_empty=False, move2_prepose=False)
            # rospy.sleep(0.2)
            rx, ry, rz = r
            self.ur5.moveToPT(x, y, z, rx, ry, rz, t=0.8)
            self.old_rgbd_img = self.rgbd_img
            # rospy.sleep(0.2)

            # Observation
            obs_thread = threading.Thread(target=self.ObsThread)
            obs_thread.start()
            rospy.sleep(0.3)

            # move
            x, y, z, r = self.move_action
            rx, ry, rz = r
            self.ur5.moveToPT(x, y, z, rx, ry, rz, t=0.8)

            obs_thread.join()
            cam_obs, _ = self.obs
        else:
            done = False
        obs = self.bins[self.picking_bin_id].GetObs(cam_obs).reshape(1, -1, self.bin_size_pixel, self.bin_size_pixel)


        return torch.tensor([0], dtype=torch.float32).view(1),\
               torch.zeros((1, 1, self.in_hand_size, self.in_hand_size)).to(torch.float32),\
               torch.tensor(obs.clone().detach(), dtype=torch.float32).to(torch.float32),\
               torch.tensor(reward, dtype=torch.float32).view(1),\
               torch.tensor(done, dtype=torch.float32).view(1)

    def getStepLeft(self):
        return torch.tensor(100).view(1)

    def close(self):
        self.ur5.moveToHome()


if __name__ == '__main__':
    import rospy
    rospy.init_node('image_proxy')
    env = DualBinFrontRear(ws_x=0.8, ws_y=0.8, ws_center=(-0.4448, -0.17, -0.147), cam_size=(256, 256),
                           obs_source='reconstruct', bin_size=0.4, bin_size_pixel=112)
    while True:
        env.checkWS()

        # start_time = time.time()
        # env.getObs(None)
        # print('get obs: ', time.time() - start_time)
        # env.reset()
