# initialized with: tray height, width, inclination, gripper size
# input: (x, y (unit: int), theta (0 ~ pi))
# output: min_safe_z
import numpy as np
# import torch
import math


class ZProtect():
    def __init__(self, bin_in, bin_out, bin_angle, gripper_size):
        '''
        :param bin_in: bottom size of the bin, action space size (unit: meter)
        :param bin_out: full size of bin, observation space size (unit: meter)
        :param bin_angle: bin angle (unit: degree)
        :param gripper_size: length of the gripper (unit: meter)
        '''
        self.bin_in = bin_in
        self.bin_out = bin_out
        self.bin_angle = bin_angle
        self.bin_radian = math.radians(self.bin_angle)
        self.gripper_size = gripper_size

    def z_protection_func(self, gripper_loc):
        '''
        :param gripper_loc: a tuple of center of gripper location (x, y, theta(rz, in radians))
        :return: min_safe_z: a safe height for the z (from the bottom 0)
        '''
        x, y, theta = gripper_loc
        # location calculation for gripper
        #print(np.cos(theta))
        gripper_l = (x + self.gripper_size / 2 * np.sin(theta), y - self.gripper_size / 2 * np.cos(theta))
        #print('gripper l: ', gripper_l)
        gripper_r = (x - self.gripper_size / 2 * np.sin(theta), y + self.gripper_size / 2 * np.cos(theta))
        #print('gripper r: ', gripper_r)


        # check whether gripper is out of the action space
        gripper_l_x_in = False
        gripper_l_y_in = False
        gripper_r_x_in = False
        gripper_r_y_in = False

        if abs(gripper_l[0]) <= self.bin_in / 2:
            gripper_l_x_in = True
        if abs(gripper_l[1]) <= self.bin_in / 2:
            gripper_l_y_in = True

        if abs(gripper_r[0]) <= self.bin_in / 2:
            gripper_r_x_in = True
        if abs(gripper_r[1]) <= self.bin_in / 2:
            gripper_r_y_in = True

        #print('gripper_l_x_in: ', gripper_l_x_in)
        #print('gripper_l_y_in: ', gripper_l_y_in)
        #print('gripper_r_x_in: ', gripper_r_x_in)
        #print('gripper_r_y_in: ', gripper_r_y_in)

        if gripper_l_x_in:
            if gripper_l_y_in:
                z_height_l = 0
            else:
                z_height_l = np.tan(self.bin_radian) * (np.abs(gripper_l[1]) - self.bin_in / 2)
        else:
            z_height_l = np.tan(self.bin_radian) * (np.abs(gripper_l[0]) - self.bin_in / 2)
        #print('z_height_l: ', z_height_l)

        if gripper_r_x_in:
            if gripper_r_y_in:
                z_height_r = 0
            else:
                z_height_r = np.tan(self.bin_radian) * (np.abs(gripper_r[1]) - self.bin_in / 2)

        else:
            z_height_r = np.tan(self.bin_radian) * (np.abs(gripper_r[0]) - self.bin_in / 2)
        #print('z_height_r: ', z_height_r)

        if z_height_l >= z_height_r:
            min_safe_z = z_height_l
        else:
            min_safe_z = z_height_r # min_safe_z = max()

        #print('min_safe_z: ', min_safe_z)
        return min_safe_z

# if __name__ == '__main__':
#     protect = Protect(20, 30, 45, 4)
#
#     # 0: 1.0000
#     #protect.z_protection_func((-9, 9, 0))
#
#     # 30: 0.73205
#     #protect.z_protection_func((9, 9, math.pi / 6))
#
#     # 45: 0.41421
#     #protect.z_protection_func((-9, -9, math.pi / 4))
#
#     # 90: 1.0000
#     #protect.z_protection_func((9, -9, math.pi / 2))
#
#     # 150: 1.0000
#     #protect.z_protection_func((-9, 9, math.pi / 6 * 5))
#
#     # 180: 1.0000
#     #protect.z_protection_func((9, -9, math.pi))
#
#     # -45: 1.0000
#     protect.z_protection_func((9, 9, - math.pi / 4))

