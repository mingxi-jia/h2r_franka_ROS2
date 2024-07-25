import time

import rospy
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
import ros_numpy
import tf2_ros
import tf

import torch
from sklearn.impute import SimpleImputer
from skimage.restoration import inpaint
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
from cv_bridge import CvBridge

import skimage.transform
from skimage.transform import rotate
import open3d
import sys
import os
currect_path = "/"+os.path.join(*os.path.abspath(__file__).split("/")[:-2])
sys.path.append(os.path.join(currect_path, "src"))
sys.path.append(currect_path)
from utils import demo_util_pp, transformation
import sys
sys.path.append("/home/master_oogway/panda_ws/src/GEM2.0")
from lepp.clip_preprocess import CLIP_processor
from cloud_proxy_three import CloudProxy
from panda import PandaArmControl
# import scipy

KEVIN_WS_BOUNDS = [86, 472, 318, 602] # [top, bottom, left, right]
KEVIN_HEIGHT_OFFSET = -1.05271 # plusing offset always means realworld to object space

class PandaWorkspace:
    def __init__(self, clip_processor=None, kernel_sizes=[40, 90, 140], stride=10, real_robot=True):
        if real_robot:
            self.cloud_proxy = CloudProxy()
            self.intrinsics = self.cloud_proxy.get_all_cam_intrinsic()
            self.extrinsics = self.cloud_proxy.get_all_cam_extrinsic()
        else:
            self.cloud_proxy = None
            self.intrinsics = np.load(os.path.join(currect_path,"parameters/intrinsics.npy"), allow_pickle=True).item()
            self.extrinsics = np.load(os.path.join(currect_path,"parameters/extrinsics.npy"), allow_pickle=True).item()
        # RT base_link
        self.workspace = np.array([[0.28, 0.71],
                                    [-0.32, 0.26],
                                    [-0.05, 1.0]])
        self.center = self.workspace.mean(-1)
        self.x_size = self.workspace[0].max() - self.workspace[0].min()
        self.x_half = self.x_size/2
        self.y_size = self.workspace[1].max() - self.workspace[1].min()
        self.y_half = self.y_size/2
        self.z_min = self.workspace[2].min()


        if clip_processor is None:
            self.clip_processor = CLIP_processor()
        else:
            self.clip_processor = clip_processor
        self.kernel_sizes = kernel_sizes
        self.stride = stride

        self.img_width = 284
        self.img_height = 386
        self.z_table = -0.02
        self.proj_pose = None
        self.rgb_value = [0,0,0]

        self.extrinsic2 = self.extrinsics['kevin']
        self.table_to_topdown_height = self.extrinsic2[2, -1]

        self.panda_arm = PandaArmControl()

    def get_intrinsics(self, camera_id):
        if self.cloud_proxy is None:
            intrinsics = self.intrinsics[camera_id]
        else:
            self.cloud_proxy.get_cam_intrinsic(camera_id)
        return intrinsics

    def get_pointcloud_from_depth(self, rgb, depth, intrinsics, instruction, visualize=False):
        kernel_sizes = self.kernel_sizes
        stride = self.stride
        clip_feature = np.zeros(rgb.shape)
        for size in kernel_sizes:
            feature = self.clip_processor.get_clip_feature(rgb, instruction, size, stride, normalization=False)[0]
            threshold_value = np.sort(feature.flatten())[-8000 - 50 * (200 - size)]
            feature[feature < threshold_value] = 0

            clip_feature += feature
            
        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
        
        
        points = np.float32([px, py, depth, rgb[..., 0], rgb[..., 1], rgb[..., 2], clip_feature[...,0]]).transpose(1, 2, 0)
        cloud = points.reshape(-1,7)
        # z_constrain= (cloud[:,2]>0.1) & (cloud[:,2]<1.1)
        # cloud = cloud[z_constrain]
        if visualize:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(cloud[:,:3])
            pcd.colors = open3d.utility.Vector3dVector(cloud[:, 3:6]/255)
            open3d.visualization.draw_geometries([pcd])
        # the output cloud is in rgb frame
        return cloud
    
    def transform_clip_cloud_to_base(self, cloud, cam_id):
        if self.cloud_proxy is None:
            T = self.extrinsics[cam_id]
        else:
            return self.cloud_proxy.transform_clip_cloud_to_base(cloud, cam_id)
        cloud_RT_base = self.transform(cloud[:, :3], T)
        return np.concatenate([cloud_RT_base, cloud[:, 3:]], axis=1)

    def cloud_preprocess(self, cloud):
        cloud, rgb_clip = self.getFilteredPointCloud(cloud[:, :3], cloud[:, 3:])
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cloud)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
        cloud = np.asarray(cl.points)
        rgb_clip = rgb_clip[ind]
        cloud = np.concatenate([cloud, rgb_clip], axis=1)
        return cloud
    
    def clear_cache(self):
        self.depth1 = None
        self.depth2 = None
        self.depth3 = None
        self.rgbimg1 = None
        self.rgbimg2 = None
        self.rgbimg3 = None
        self.cloud1 = None
        self.cloud2 = None
        self.cloud3 = None
        self.rgb1 = None
        self.rgb2 = None
        self.rgb3 = None

    def pad_bottom_cloud(self, cloud, rgb_value=None):
        # generate 'fake' point cloud for area outside the bins
        if rgb_value is None:
            rgb_value = self.rgb_value
        r, g, b = rgb_value
        padding_more = 0.0
        x = np.arange((self.center[0]-self.x_half*2)*1000, (self.center[0]+self.x_half*2)*1000, 2)
        y = np.arange((self.center[1]-self.y_half*2)*1000, (self.center[1]+self.y_half*2)*1000, 2)
        xx, yy = np.meshgrid(x, y)
        xx = xx/1000
        yy = yy/1000
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        pts = np.concatenate([xx, yy, np.ones_like(yy)*(self.z_table), np.ones_like(yy)*r, np.ones_like(yy)*g, np.ones_like(yy)*b, np.zeros_like(yy)], 1)
        cloud = np.concatenate([cloud, pts], axis=0)
        return cloud
    
    def get_clip_cloud(self, rgb, depth, cam_name, instruction):
        intrinsic = self.intrinsics[cam_name]
        cloud = self.get_pointcloud_from_depth(rgb, depth, intrinsic, instruction)
        cloud = self.transform_clip_cloud_to_base(cloud, cam_name)
        cloud = self.cloud_preprocess(cloud)
        return cloud

    
    def get_xyzrgb_points(self, cloud_array, remove_nans=True, dtype=np.float):
        '''Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
        '''
        # remove crap points
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
            cloud_array = cloud_array[mask]
        
        # pull out x, y, and z values
        xyz = np.zeros(cloud_array.shape + (3,), dtype=dtype)
        xyz[...,0] = cloud_array['x']
        xyz[...,1] = cloud_array['y']
        xyz[...,2] = cloud_array['z']

        rgb_arr = ros_numpy.point_cloud2.split_rgb_field(cloud_array)
        rbg = np.zeros(cloud_array.shape + (3,), dtype=dtype)
        rbg[...,0] = rgb_arr['r']
        rbg[...,1] = rgb_arr['g']
        rbg[...,2] = rgb_arr['b']

        return xyz, rbg
    
    def lookup_transform(self, fromFrame, toFrame, lookupTime=rospy.Time(0)):
        """
        Lookup a transform in the TF tree.
        :param fromFrame: the frame from which the transform is calculated
        :type fromFrame: string
        :param toFrame: the frame to which the transform is calculated
        :type toFrame: string
        :return: transformation matrix from fromFrame to toFrame
        :rtype: 4x4 np.array
        """

        transformMsg = self.tfBuffer.lookup_transform(toFrame, fromFrame, lookupTime, rospy.Duration(1.0))
        translation = transformMsg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transformMsg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        T = tf.transformations.quaternion_matrix(quat)
        T[0:3, 3] = pos
        return T
    
    def transform(self, cloud, T, isPosition=True):
        '''Apply the homogeneous transform T to the point cloud. Use isPosition=False if transforming unit vectors.'''

        n = cloud.shape[0]
        cloud = cloud.T
        augment = np.ones((1, n)) if isPosition else np.zeros((1, n))
        cloud = np.concatenate((cloud, augment), axis=0)
        cloud = np.dot(T, cloud)
        cloud = cloud[0:3, :].T
        return cloud
    


    def interpolate(self, depth):
        """
        Fill nans in depth image
        """
        # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
        mask = np.logical_not(np.isnan(depth))
        # array of (number of points, 2) containing the x,y coordinates of the valid values only
        xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

        # the valid values in the first, second, third color channel,  as 1D arrays (in the same order as their coordinates in xym)
        data0 = np.ravel(depth[:, :][mask])

        # three separate interpolators for the separate color channels
        interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)

        # interpolate the whole image, one color channel at a time
        result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

        return result0
    
    def interpolate_zero(self, depth):
        """
        Fill nans in depth image
        """
        depth[np.isnan(depth)] = 0.

        return depth
    
    def getFilteredPointCloud(self, cloud, rgb, manual_offset=None):

        # filter ws x
        x_cond = (cloud[:, 0] < self.center[0] + self.x_half) * (cloud[:, 0] > self.center[0] - self.x_half)
        cloud = cloud[x_cond]
        rgb = rgb[x_cond]
        # filter ws y
        y_cond = (cloud[:, 1] < self.center[1] + self.y_half) * (cloud[:, 1] > self.center[1] - self.y_half)
        cloud = cloud[y_cond]
        rgb = rgb[y_cond]
        # filter ws z
        z_cond = (cloud[:, 2] < self.center[2].max()) * (cloud[:, 2] > self.z_min)
        cloud = cloud[z_cond]
        rgb = rgb[z_cond]

        if manual_offset is not None:
            # actually shouldn't do this. We should do cam intrinsic calib
            cloud[:, 0] += manual_offset[0]
            cloud[:, 1] += manual_offset[1]
            cloud[:, 2] += manual_offset[2]

        return cloud, rgb
    
    def getFusedPointCloud(self):
        """
        get new point cloud, set self.cloud
        """
        start_time = time.time()
        while (self.cloud1 is None or self.cloud2 is None or self.cloud3 is None or self.rgb1 is None or self.rgb2 is None or self.rgb3 is None):
            rospy.sleep(0.1)

        cloud1, rgb1 = self.getFilteredPointCloud(self.cloud1, self.rgb1)
        cloud2, rgb2 = self.getFilteredPointCloud(self.cloud2, self.rgb2)
        cloud3, rgb3 = self.getFilteredPointCloud(self.cloud3, self.rgb3)
        
        cloud = np.concatenate((cloud1, cloud2, cloud3))
        rgb = np.concatenate((rgb1, rgb2, rgb3))
        

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cloud)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
        cloud = np.asarray(cl.points)
        rgb = rgb[ind]
        
        # generate 'fake' point cloud for area outside the bins
        x = np.arange((self.center[0]-self.x_half*2)*1000, (self.center[0]+self.x_half*2)*1000, 2)
        y = np.arange((self.center[1]-self.y_half*2)*1000, (self.center[1]+self.y_half*2)*1000, 2)
        xx, yy = np.meshgrid(x, y)
        xx = xx/1000
        yy = yy/1000
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        pts = np.concatenate([xx, yy, np.ones_like(yy)*(self.z_min)], 1)
        cloud = np.concatenate([cloud, pts])
        self.cloud = cloud

        if self.allow_color_cloud:
            predefined_color = [255,255,255]
            padding_rgb = np.ones([pts.shape[0], 3]) 
            rgb = np.concatenate([rgb, padding_rgb])
            cloud_color = np.concatenate([cloud, rgb], axis=-1)
            self.cloud_color = cloud_color
        else:
            cloud_color = None
            self.cloud_color = cloud_color

        return cloud1, cloud2, cloud3, cloud, cloud_color

    def getProjectImg(self, target_size, img_size, gripper_pos=(-0.5, 0, 0.1)):
        """
        return orthographic projection depth img from self.cloud
        target_size: img coverage size in meters
        img_size: img pixel size
        gripper_pos: the pos of the camera
        return depth image
        """
        if self.cloud is None:
            self.getNewPointCloud(gripper_pos)
        cloud = np.copy(self.cloud)
        cloud = cloud[(cloud[:, 2] < max(gripper_pos[2], self.z_min + 0.05))]
        view_matrix = transformation.euler_matrix(0, np.pi, 0).dot(np.eye(4))
        # view_matrix = np.eye(4)
        view_matrix[:3, 3] = [gripper_pos[0], -gripper_pos[1], gripper_pos[2]]
        view_matrix = transformation.euler_matrix(0, 0, -np.pi/2).dot(view_matrix)
        augment = np.ones((1, cloud.shape[0]))
        pts = np.concatenate((cloud.T, augment), axis=0)
        projection_matrix = np.array([
            [1 / (target_size / 2), 0, 0, 0],
            [0, 1 / (target_size / 2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        tran_world_pix = np.matmul(projection_matrix, view_matrix)
        pts = np.matmul(tran_world_pix, pts)
        # pts[1] = -pts[1]
        pts[0] = (pts[0] + 1) * img_size / 2
        pts[1] = (pts[1] + 1) * img_size / 2

        pts[0] = np.round_(pts[0])
        pts[1] = np.round_(pts[1])
        mask = (pts[0] >= 0) * (pts[0] < img_size) * (pts[1] > 0) * (pts[1] < img_size)
        pts = pts[:, mask]
        # dense pixel index
        mix_xy = (pts[1].astype(int) * img_size + pts[0].astype(int))
        # lexsort point cloud first on dense pixel index, then on z value
        ind = np.lexsort(np.stack((pts[2], mix_xy)))
        # bin count the points that belongs to each pixel
        bincount = np.bincount(mix_xy)
        # cumulative sum of the bin count. the result indicates the cumulative sum of number of points for all previous pixels
        cumsum = np.cumsum(bincount)
        # rolling the cumsum gives the ind of the first point that belongs to each pixel.
        # because of the lexsort, the first point has the smallest z value
        cumsum = np.roll(cumsum, 1)
        cumsum[0] = bincount[0]
        cumsum[cumsum == np.roll(cumsum, -1)] = 0
        # pad for unobserved pixels
        cumsum = np.concatenate((cumsum, -1 * np.ones(img_size * img_size - cumsum.shape[0]))).astype(int)

        depth = pts[2][ind][cumsum]
        depth[cumsum == 0] = np.nan
        depth = depth.reshape(img_size, img_size)
        # fill nans
        depth = self.interpolate(depth)
        return depth
    
    def getTopDownProjectImg(self, cloud_rgbc, projection_height):
        """
        return orthographic projection depth img from self.cloud
        target_size: img coverage size in meters
        img_size: img pixel size
        gripper_pos: the pos of the camera
        return depth image
        """
        proj_pos = [self.center[0], self.center[1], projection_height]
        target_size = np.max([self.x_half, self.y_half])*2
        img_size = np.max([self.img_height, self.img_width])

        
        cloud = np.copy(cloud_rgbc[:, :3])
        rgbc = np.copy(cloud_rgbc[:, 3:])

        view_matrix = transformation.euler_matrix(0, np.pi, 0).dot(np.eye(4))
        view_matrix[:3, 3] = [proj_pos[0], -proj_pos[1], proj_pos[2]]
        view_matrix = transformation.euler_matrix(0, 0, 0).dot(view_matrix)
        augment = np.ones((1, cloud.shape[0]))
        pts = np.concatenate((cloud.T, augment), axis=0)
        scale = 1.
        projection_matrix = np.array([
            [scale / (target_size / 2), 0, 0, 0],
            [0, scale / (target_size / 2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        tran_world_pix = np.matmul(projection_matrix, view_matrix)
        pts = np.matmul(tran_world_pix, pts)
        pts[0] = (pts[0] + 1) * img_size / 2
        pts[1] = (pts[1] + 1) * img_size / 2

        pts[0] = np.round_(pts[0])
        pts[1] = np.round_(pts[1])
        mask = (pts[0] >= 0) * (pts[0] < img_size) * (pts[1] > 0) * (pts[1] < img_size)
        pts = pts[:, mask]
        # dense pixel index
        mix_xy = (pts[1].astype(int) * img_size + pts[0].astype(int))
        # lexsort point cloud first on dense pixel index, then on z value
        ind = np.lexsort(np.stack((pts[2], mix_xy)))
        # bin count the points that belongs to each pixel
        bincount = np.bincount(mix_xy)
        # cumulative sum of the bin count. the result indicates the cumulative sum of number of points for all previous pixels
        cumsum = np.cumsum(bincount)
        # rolling the cumsum gives the ind of the first point that belongs to each pixel.
        # because of the lexsort, the first point has the smallest z value
        cumsum = np.roll(cumsum, 1)
        cumsum[0] = bincount[0]
        cumsum[cumsum == np.roll(cumsum, -1)] = 0
        # pad for unobserved pixels
        cumsum = np.concatenate((cumsum, -1 * np.ones(img_size * img_size - cumsum.shape[0]))).astype(int)

        depth = pts[2][ind][cumsum]
        depth[cumsum == 0] = np.nan
        depth = depth.reshape(img_size, img_size)
        # fill nans
        depth = self.interpolate(depth)

        rgbc = rgbc.T
        rgbc = rgbc[:, mask].T
        rgbc = rgbc[ind][cumsum]
        rgb = rgbc[:, :3]
        clip = rgbc[:, 3:4]
        rgb[cumsum == 0] = np.array(self.rgb_value)
        clip[cumsum == 0] = np.nan
        rgbc = np.concatenate([rgb, clip], axis=-1)
        rgbc = rgbc.reshape(img_size, img_size, 4)
        clip = self.interpolate(rgbc[..., 3])
        rgbc = np.concatenate([rgbc[..., :3], clip[..., None]], axis=2)


        img_left, img_right =  np.array([-self.img_width//2, self.img_width//2]) + img_size//2
        img_top, img_down =  np.array([-self.img_height//2, self.img_height//2]) + img_size//2
        depth = depth[img_top:img_down, img_left:img_right]
        rgbc = rgbc[img_top:img_down, img_left:img_right]
        return depth, rgbc
    
    def _preProcessObs(self, obs, kernel_size=5):
        obs = scipy.ndimage.median_filter(obs, kernel_size)
        return obs

    def _preProcessRGBC(self, obs, kernel_size=3):
        # obs = obs[..., :3]
        if len(obs.shape) == 2:
            obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
        obss = []
        for channel in range(obs.shape[-1]):
            a = scipy.ndimage.median_filter(obs[..., channel], kernel_size)
            obss.append(a)
        obs = np.stack(obss).transpose(1,2,0)
        return obs
    
    def get_height_map_reconstruct(self, cloud_rgbc, proj_height=None):
        if proj_height is None:
            proj_height = self.table_to_topdown_height

        depth, rgbc = self.getTopDownProjectImg(cloud_rgbc, proj_height)
        
        depth = self._preProcessObs(depth)
        rgb = cv2.bilateralFilter(rgbc[...,:3].astype(np.uint8), 15, 40, 40)

        clip_feature = rgbc[..., 3:4]
        clip_feature = (clip_feature - clip_feature.min()) / (clip_feature.max() - clip_feature.min())
        rgbc = np.concatenate([rgb, clip_feature], axis=2)
        return depth, rgbc
    
    def get_topdown_clip_from_multi(self, separate_clouds, proj_height=None):
        if proj_height is None:
            proj_height = self.table_to_topdown_height

        clip_feature = np.zeros((self.img_height, self.img_width))
        for cloud in separate_clouds:
            _, rgbc_tmp = self.getTopDownProjectImg(cloud, proj_height)
            clip_feature += rgbc_tmp[..., 3]
        clip_feature = (clip_feature - clip_feature.min()) / (clip_feature.max() - clip_feature.min() + 1e-6)
        clip_feature = (clip_feature - 0.5) * 80
        clip_feature = torch.special.expit(torch.tensor(clip_feature)).numpy()

        return clip_feature
        

    def get_multi_obs(self):
        self.clear_cache()
        depth = self.get_depth_image(1)
        rgb = self.get_rgb_image(1)
        rgbd1 = np.concatenate([rgb, depth[..., None]], axis=-1)
        intrinsic1 = self.getCamIntrinsic(1)
        extrinsic1 = self.getCamExtrinsic(1)

        depth = self.get_depth_image(2)
        rgb = self.get_rgb_image(2)
        rgbd2 = np.concatenate([rgb, depth[..., None]], axis=-1)
        intrinsic2 = self.getCamIntrinsic(2)
        extrinsic2 = self.getCamExtrinsic(2)

        depth = self.get_depth_image(3)
        rgb = self.get_rgb_image(3)
        rgbd3 = np.concatenate([rgb, depth[..., None]], axis=-1)
        intrinsic3 = self.getCamIntrinsic(3)
        extrinsic3 = self.getCamExtrinsic(3)
        return rgbd1, rgbd2, rgbd3, intrinsic1, intrinsic2, intrinsic3, extrinsic1, extrinsic2, extrinsic3
    
    def idx_to_pos(self, shape, x, y):
        x = 0.6 * x / shape[0] - 0.31
        y = 0.77 - 0.45 * y / shape[1]
        return x, y
    
    def get_close_look(self, instruction):
        # get images and depths from the cameras
        now = time.time()
        top, bottom, left, right = KEVIN_WS_BOUNDS
        bob_image = self.cloud_proxy.get_rgb_image('bob')
        bob_depth = self.cloud_proxy.get_depth_image('bob')
        kevin_image = self.cloud_proxy.get_rgb_image('kevin')
        kevin_depth = self.cloud_proxy.get_depth_image('kevin')
        stuart_image = self.cloud_proxy.get_rgb_image('stuart')
        stuart_depth = self.cloud_proxy.get_depth_image('stuart')

        # get clip feature point clouds from the images and depths
        bob_cloud = self.get_clip_cloud(bob_image, bob_depth, 'bob', instruction)
        kevin_cloud = self.get_clip_cloud(kevin_image, kevin_depth, 'kevin', instruction)
        stuart_cloud = self.get_clip_cloud(stuart_image, stuart_depth, 'stuart', instruction)

        # fuse clip feature point clouds
        fused_cloud = np.concatenate([bob_cloud, kevin_cloud, stuart_cloud], axis=0)
        fused_cloud = self.pad_bottom_cloud(fused_cloud, rgb_value=[27, 40, 40])
        feature_map = self.get_topdown_clip_from_multi([bob_cloud, kevin_cloud, stuart_cloud], self.extrinsics['kevin'][2, -1])
        # topdown_rgb = np.array(kevin_image)[86:472, 318:602]/255.
        # plt.imshow(topdown_rgb * 0.5 + feature_map[...,None])

        # get approximate position of the object
        x0, y0 = np.unravel_index(np.argmax(feature_map), feature_map.shape)
        x, y = self.idx_to_pos(feature_map.shape, x0, y0)

        print('feature generation time taken:', time.time() - now)

        kevin_target_depths = kevin_depth[top:bottom, left:right][x0-30:x0+30, y0-30:y0+30]
        kevin_target_depth = np.sort(kevin_target_depths.flatten())[:100].mean()

        target_height = - kevin_target_depth - KEVIN_HEIGHT_OFFSET + 0.2


        # move the robot to the approximate position of the object
        self.panda_arm.move_camera_to_pose(x, y, 0.5, 3.14, 0, -0.8)

        # in case of extreme noise
        if target_height < 0.107 or target_height > 1.0:
            target_height = 0.5
        
        self.panda_arm.move_camera_to_pose(x, y, target_height, 3.14, 0, -0.8)

        return self.cloud_proxy.get_rgb_image('tim')
    
    def go_home(self):
        self.panda_arm.move_gripper_to_pose(0.1, 0.2, 0.6, 3.14, 0, -0.8)

        


def main():
    rospy.init_node('panda_ws')
    panda_ws = PandaWorkspace()
    panda_ws.go_home()

    instruction = 'rubber duck'
    close_look_image = panda_ws.get_close_look(instruction)
    image = Image.fromarray(close_look_image)
    image.save(f'/home/master_oogway/panda_ws/src/GEM2.0/data/close_up/{instruction}.png')
    plt.imshow(close_look_image)
    panda_ws.go_home()


if __name__ == '__main__':
    main()