import numpy as np
import rospy

from legacy.cloud_proxy_ros1 import CloudProxy
from h2r_franka_ros2.legacy.panda import PandaArmControl

class GEMWS:
    def __init__(self):
        

        self.workspace = np.array([[0.35, 0.75],
                                    [-0.2, 0.2],
                                    [-0.02, 1.0]])

        self.cloud_proxy:CloudProxy = CloudProxy(self.workspace)
        self.robot:PandaArmControl = PandaArmControl()

        print("waiting for cameras")
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
        self.z = 0.1

        self.extrinsic2 = self.extrinsics['kevin']
        self.table_to_topdown_height = self.extrinsic2[2, -1]

    def get_multi_obs(self):
        rgbds = dict()
        self.cloud_proxy.clear_cache()
        for cam_name in self.cloud_proxy.cam_names:
            print(f"getting camera {cam_name}")
            depth = self.cloud_proxy.get_depth_image(cam_name)
            rgb = self.cloud_proxy.get_rgb_image(cam_name)
            rgbds[cam_name] = np.concatenate([rgb, depth[..., None]], axis=-1)
        print("get_multi_obs done")
        return rgbds

    def get_topdown_obs(self):
        self.cloud_proxy.clear_cache()
        cam_name= 'kevin'
        print(f"getting camera {cam_name}")
        depth = self.cloud_proxy.get_depth_image(cam_name)
        rgb = self.cloud_proxy.get_rgb_image(cam_name)
        rgbd = np.concatenate([rgb, depth[..., None]], axis=-1)
        # return rgbd[]
        pass


    def proprocess_multi_obs((self, obs:dict):
        pass

    def get_clip_cloud(self)::
        pass

    def get_topdown_relevancy(self):
        pass

    def arm_reset(self):
        self.robot.go_home()
        self.robot.open_gripper()

    def get_ee_pixel_xy(self):
        # upper left corner of the action space is the pixel origin, x as row, y as column
        x, y, z = self.robot.get_ee_position()
        # pixel_x = (pixel_x_reso/ _x_reso) * (x - UPPER_XY)
        # pixel_y = (pixel_y_reso/ _y_reso) * (y - LEFT_XY)
        pixel_x = (x - self.X_MIN_ROBOT) / (self.X_MAX_ROBOT - self.X_MIN_ROBOT) * self.img_height
        pixel_y = (y - self.Y_MIN_ROBOT) / (self.Y_MAX_ROBOT - self.Y_MIN_ROBOT) * self.img_width
        pixel = np.array([pixel_x, pixel_y]).astype(int)
        # rotated_pixel = rotatePixelCoordinate([pixel_small_reso, pixel_big_reso], pixel, np.pi/2)
        return pixel

    def move_ee_to_xyr(self, xyzr):
        x, y, z, topdown_rotation = xyzr
        self.robot.move_ee_to_pose(x, y, z, np.pi, 0., topdown_rotation)

    def move_ee_to_xyrpy(self, xyrpy):
        x, y, z, topdown_rotation = xyzr
        self.robot.move_ee_to_pose(x, y, z, np.pi, 0., topdown_rotation)

    def plot_multiview(self, multiview_rgbd:dict):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3)
        for i, (cam_name, rgb_depth) in enumerate(multiview_rgbd.items()):
            ax = axs[i//3, i%3]
            ax.imshow(rgb_depth[..., :3]/255)
            ax.set_title(cam_name)
        plt.show(block=False)

    def calirate_ws_corners(self):
        self.arm_reset()
        #xyzr
        print("going to upper_left_pose")
        upper_left_pose = [self.cloud_proxy.workspace[0,0], self.cloud_proxy.workspace[1,0], self.z+0.1, 0.]
        self.move_ee_to_xyr(upper_left_pose)
        print("going to upper_right_pose")
        upper_right_pose = [self.cloud_proxy.workspace[0,0], self.cloud_proxy.workspace[1,1], self.z+0.1, 0.]
        self.move_ee_to_xyr(upper_right_pose)
        print("going to down_right_pose")
        down_right_pose = [self.cloud_proxy.workspace[0,1], self.cloud_proxy.workspace[1,1], self.z+0.1, 0.]
        self.move_ee_to_xyr(down_right_pose)
        print("going to upper_right_pose")
        down_left_pose = [self.cloud_proxy.workspace[0,1], self.cloud_proxy.workspace[1,0], self.z+0.1, 0.]
        self.move_ee_to_xyr(down_left_pose)

if __name__ == "__main__":
    rospy.init_node('panda_ws')
    env = GEMWS()
    multiview_rgbd = env.get_multi_obs()
    env.plot_multiview(multiview_rgbd)
    env.calirate_ws_corners()
    print(1)