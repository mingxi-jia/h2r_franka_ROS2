import numpy as np

from cloud_proxy import CloudProxy
from panda import PandaArmControl

class GEMWS:
    def __init__(self):
        
        self.cloud_proxy:CloudProxy = CloudProxy()
        self.panda_arm:PandaArmControl = PandaArmControl()

        self.intrinsics = self.cloud_proxy.get_all_cam_intrinsic()
        self.extrinsics = self.cloud_proxy.get_all_cam_extrinsic()
        self.X_MIN_ROBOT = self.cloud_proxy.workspace[0,0]
        self.X_MAX_ROBOT = self.cloud_proxy.workspace[0,1]
        self.Y_MIN_ROBOT = self.cloud_proxy.workspace[1,0]
        self.Y_MAX_ROBOT = self.cloud_proxy.workspace[1,1]

        self.img_width = 320
        self.img_height = 320
        self.z_table = -0.02
        self.proj_pose = None
        self.rgb_value = [0,0,0]

        self.extrinsic2 = self.extrinsics['kevin']
        self.table_to_topdown_height = self.extrinsic2[2, -1]

    def get_multi_obs(self):
        rgbds = dict()
        self.cloud_proxy.clear_cache()
        for cam_name in self.cloud_proxy.fixed_cam_names:
            depth = self.cloud_proxy.get_depth_image(cam_name)
            rgb = self.cloud_proxy.get_rgb_image(cam_name)
            rgbds[cam_name] = np.concatenate([rgb, depth[..., None]], axis=-1)
        return rgbds

    def arm_reset(self):
        self.panda_arm.go_home()
        self.panda_arm.open_gripper()

    def get_ee_pixel_xy(self):
        # upper left corner of the action space is the pixel origin, x as row, y as column
        xy_in_robot_frame = self.panda_arm.get_ee_pose()
        x, y = xy_in_robot_frame
        # pixel_x = (pixel_x_reso/ _x_reso) * (x - UPPER_XY)
        # pixel_y = (pixel_y_reso/ _y_reso) * (y - LEFT_XY)
        pixel_x = (x - self.X_MIN_ROBOT) / (self.X_MAX_ROBOT - self.X_MIN_ROBOT) * self.img_height
        pixel_y = (y - self.Y_MIN_ROBOT) / (self.Y_MAX_ROBOT - self.Y_MIN_ROBOT) * self.img_width
        pixel = np.array([pixel_x, pixel_y]).astype(int)
        # rotated_pixel = rotatePixelCoordinate([pixel_small_reso, pixel_big_reso], pixel, np.pi/2)
        return pixel