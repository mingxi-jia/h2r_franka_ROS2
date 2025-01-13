import time

import rospy
from sensor_msgs.msg import Image
import ros_numpy
import tf
import math
import torch.nn.functional as F

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
from PIL import Image
import torchvision.transforms as T
from scipy.spatial.transform import Rotation as R
import json
import open3d as o3d
from f3rm.features.clip import clip

import open3d
import sys
import os
currect_path = "/"+os.path.join(*os.path.abspath(__file__).split("/")[:-2])
from panda_utils import transformation
sys.path.append("/home/master_oogway/panda_ws/code/GEM2.0")
from lepp.clip_dino_seg import clip_dino_seg
from lepp.clip_preprocess import CLIP_processor
from f3rm_robot.load import load_nerfstudio_outputs
from legacy.cloud_proxy_ros1 import CloudProxy
from h2r_franka_ros2.legacy.panda import PandaArmControl

KEVIN_WS_BOUNDS = [86, 472, 318, 602] # [top, bottom, left, right]
KEVIN_HEIGHT_OFFSET = -1.05271 # plusing offset always means realworld to object space
DIST_ABOVE_TARGET = 0.12
RADIUS_AROUND_TARGET = 0.16
REAL_WORLD_WS_BOUNDS = [0.22, 0.7, -0.31, 0.28]
# PHIS = [np.pi/3, np.pi/5, np.pi/8]
PHIS = [np.pi/6]
GRIPPER_PIXELS = 100

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
        self.workspace = np.array([REAL_WORLD_WS_BOUNDS[:2],
                                    REAL_WORLD_WS_BOUNDS[2:],
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
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device='cuda')

        self.img_width = 284
        self.img_height = 386
        self.z_table = -0.02
        self.proj_pose = None
        self.rgb_value = [0,0,0]

        self.extrinsic2 = self.extrinsics['kevin']
        self.table_to_topdown_height = self.extrinsic2[2, -1]

        self.panda_arm = PandaArmControl()

    def get_intrinsic(self, camera_id):
        if self.cloud_proxy is None:
            intrinsic = self.intrinsics[camera_id]
        else:
            intrinsic = self.cloud_proxy.get_cam_intrinsic(camera_id)
        return intrinsic
    
    def get_extrinsic(self, camera_id):
        if self.cloud_proxy is None:
            extrinsic = self.extrinsics[camera_id]
        else:
            extrinsic = self.cloud_proxy.get_cam_extrinsic(camera_id)
        return extrinsic

    def get_pointcloud_from_depth(self, rgb, depth, intrinsics, instruction, visualize=False):
        kernel_sizes = self.kernel_sizes
        stride = self.stride
        clip_feature = np.zeros(rgb.shape)
        for size in kernel_sizes:
            feature = self.clip_processor.get_clip_feature(rgb, instruction, size, stride, normalization=False)[0]
            threshold_value = np.sort(feature.flatten())[-8000 - 50 * (200 - size)]
            feature[feature < threshold_value] = 0

            clip_feature += feature * (140.0 / size)
            
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
    
    def td_image_idx_to_pos(self, shape, x, y):
        y0, y1, x0, x1 = REAL_WORLD_WS_BOUNDS
        x = x0 + (x1 - x0) * x / shape[0]
        y = y1 + (y0 - y1) * y / shape[1]
        return x, y
    
    def uv_to_pos(self, u, v, depth, camera_id):
        """
        Calculate the real-world coordinates of a pixel given the intrinsic matrix,
        extrinsic matrix, and depth.
        
        Parameters:
        K (np.array): Intrinsic matrix (3x3).
        extrinsic_matrix (np.array): Extrinsic matrix (4x4).
        pixel_coords (tuple): Pixel coordinates (u, v).
        depth (float): Depth value at the pixel.
        
        Returns:
        np.array: World coordinates (X, Y, Z).
        """

        # Extrinsic matrix
        extrinsic_matrix = self.get_extrinsic(camera_id)

        # Intrinsic matrix
        K = self.get_intrinsic(camera_id)
        K_inv = np.linalg.inv(K)
        
        # Camera coordinates (homogeneous)
        pixel_vector = np.array([v, u, 1])
        camera_coords = depth * K_inv.dot(pixel_vector)
        
        # Convert to homogeneous coordinates
        camera_coords_homogeneous = np.append(camera_coords, 1)
        
        # Transform to world coordinates using the extrinsic matrix
        world_coords_homogeneous = extrinsic_matrix @ camera_coords_homogeneous
        
        # Convert from homogeneous to Cartesian coordinates
        world_coords = world_coords_homogeneous[:3] / world_coords_homogeneous[3]
        
        return world_coords
    
    def get_target_position(self, instruction):

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

        # visualizatino of the feature generated
        # topdown_rgb = np.array(kevin_image)[86:472, 318:602]/255.
        # plt.imshow(topdown_rgb * 0.5 + feature_map[...,None])

        # get approximate position of the object
        x0, y0 = np.unravel_index(np.argmax(feature_map), feature_map.shape)
        x, y = self.td_image_idx_to_pos(feature_map.shape, x0, y0)

        print('feature generation time taken:', time.time() - now)

        kevin_target_depths = kevin_depth[top:bottom, left:right][x0-30:x0+30, y0-30:y0+30]
        kevin_target_depth = np.sort(kevin_target_depths.flatten())[:100].mean()
        target_height = - kevin_target_depth - KEVIN_HEIGHT_OFFSET + DIST_ABOVE_TARGET

        return x, y, target_height
    
    def get_close_look(self, x, y, z):

        # move the robot to the approximate position of the object
        self.panda_arm.move_camera_to_pose(y, x, 0.6, np.pi, 0, np.pi/2)

        # in case of extreme noise
        if z < 0.107 or z > 0.8:
            z = 0.5
        
        step = (z - 0.6) / 3
        for i in range(4):
            self.panda_arm.move_camera_to_pose(y, x, 0.6 + step * i, np.pi, 0, np.pi/2)

        rospy.sleep(1)

        return self.cloud_proxy.get_rgb_image('tim')
    
    def circle_target_and_capture_images(self, x, y, z, r, n, path):
        idx = 0
        self.panda_arm.add_safe_guard()
        images = []
        depths = []
        poses = []
        for i in range(n):
            theta = (2 * math.pi / n) * i

            for phi in PHIS:
                x_offset = r * math.cos(theta) * math.cos(phi)
                cam_x = x + x_offset
                y_offset = r * math.sin(theta) * math.cos(phi)
                cam_y = y + y_offset
                z_offset = r * math.sin(phi)
                cam_z = z + z_offset

                # Calculate orientation to face the target
                direction = np.array([x - cam_x, y - cam_y, z - cam_z])
                direction = direction / np.linalg.norm(direction)  # normalize the direction vector
                up = np.array([0, 0, -1])
                right = np.cross(up, direction)
                right = right / np.linalg.norm(right)
                up = np.cross(direction, right)
                rotation_matrix = np.column_stack((right, up, direction))
                rotation = R.from_matrix(rotation_matrix).as_euler('xyz')
                try:
                    # if phi == np.pi/4:
                    #     self.panda_arm.move_camera_to_pose(cam_x, cam_y, 0.4, np.pi, 0, rotation[2])
                    self.panda_arm.move_camera_to_pose(cam_x, cam_y, cam_z, rotation[0], rotation[1], rotation[2])
                except TimeoutError:
                    continue
                # Capture image with 'tim' camera
                image = Image.fromarray(self.cloud_proxy.get_rgb_image('tim')[GRIPPER_PIXELS:, :, :])
                if not os.path.exists(path):
                    os.makedirs(path)
                img_path = f'{path}/frame_{idx}.png'
                image.save(img_path)
                images.append(image)
                depths.append(self.cloud_proxy.get_depth_image('tim')[GRIPPER_PIXELS:, :])
                poses.append(self.get_extrinsic('tim'))
                idx += 1
        tim_extrinsic = self.cloud_proxy.get_cam_extrinsic('tim').tolist()
        d = {'extrinsic': tim_extrinsic}
        json.dump(d, open(f'{path}/tim_extrinsic.json', 'w'), indent=4)
        self.panda_arm.remove_safe_guard()
        return images, depths, poses

    def sample_clip_features(self, feature_field, grid_resolution, bounding_box, encoded_instruction, device='cuda'):
        """
        Sample the NeRF model over a 3D grid and create a voxel grid based on similarity to the encoded instruction.
        
        Args:
            nerf_model: The trained NeRF model.
            grid_resolution: The resolution of the 3D voxel grid (number of voxels along each dimension).
            bounding_box: A tuple ((xmin, ymin, zmin), (xmax, ymax, zmax)) defining the 3D space to sample.
            encoded_instruction: The encoded feature vector of the language instruction.
            viewer_utils: The ViewerUtils instance.
            device: The device to perform computations on.
            
        Returns:
            voxel_grid: A 3D numpy array representing the voxel grid based on similarity to the encoded instruction.
        """
        # Create a 3D grid of points
        x = np.linspace(bounding_box[0][0], bounding_box[1][0], grid_resolution)
        y = np.linspace(bounding_box[0][1], bounding_box[1][1], grid_resolution)
        z = np.linspace(bounding_box[0][2], bounding_box[1][2], grid_resolution)
        grid = np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)
        
        # Convert the grid to a torch tensor
        grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
        
        # Sample the NeRF model at the grid points
        clip_dict = feature_field(grid_tensor)
        clip_features = clip_dict['feature']
        density = clip_dict['density']
        
        # Normalize the CLIP features
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity with the encoded instruction
        similarity = (clip_features @ encoded_instruction.T).squeeze(-1)
        
        # Reshape the similarity scores to the grid resolution
        similarity = similarity.view(grid_resolution, grid_resolution, grid_resolution)
        
        # Apply a threshold to the similarity scores to create the voxel grid
        threshold1 = 0.0  # Adjust this threshold as needed
        threshold2 = 8
        density = torch.reshape(density, similarity.shape)
        voxel_grid = ((density > threshold2) * (similarity > threshold1)).cpu().detach().numpy()
        
        return voxel_grid

    def get_3d_mask(self, config_path, nerf_to_local, instruction, bounding_box):
        load_state = load_nerfstudio_outputs(config_path)

        # Get features and density for each demo from feature field
        feature_field = load_state.feature_field_adapter()
        text = clip.tokenize(instruction).cuda()
        encoded_instruction = self.clip_model.encode_text(text=text)
        encoded_instruction /= encoded_instruction.norm(dim=-1, keepdim=True)
        bounding_box_nerf = (bounding_box[0] @ nerf_to_local.T[:3][:3], bounding_box[1] @ nerf_to_local.T[:3][:3])
        features_in_nerf = self.sample_clip_features(feature_field, 128, bounding_box_nerf, encoded_instruction)
        ## TODO: bug with transformation, and question about resolution
        # features_in_local = features_in_nerf @ nerf_to_local

        return features_in_nerf
    
    def reconstruct_point_clouds(self, depths, masks, camera_poses):
        """
        Reconstructs two 3D point clouds from depth maps, masks, and camera poses.

        Parameters:
            depths (list of np.array): List of depth maps (H, W) for each frame.
            masks (list of np.array): List of masks (H, W) for each frame.
            camera_poses (list of np.array): List of camera poses (4x4) for each frame.
            intrinsics (np.array): Camera intrinsic matrix (3x3).

        Returns:
            full_point_cloud (np.array): 3D point cloud of all points seen by the camera (N, 3).
            masked_point_cloud (np.array): 3D point cloud of points shown by the mask (M, 3).
        """
        full_point_cloud = []
        masked_point_cloud = []
        intrinsics = self.get_intrinsic('tim')

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        for depth, mask, pose in zip(depths, masks, camera_poses):
            H, W = depth.shape
            i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

            # Convert depth map to 3D points in camera space
            z = depth
            x = (i - cx) * z / fx
            y = (j - cy) * z / fy

            points_camera_space = np.stack((x, y, z), axis=-1).reshape(-1, 3)

            # Transform points to world space
            points_camera_space_homogeneous = np.concatenate([points_camera_space, np.ones((points_camera_space.shape[0], 1))], axis=-1)
            points_world_space = (pose @ points_camera_space_homogeneous.T).T[:, :3]

            # Add all points to the full point cloud
            full_point_cloud.append(points_world_space)

            # Add masked points to the masked point cloud
            mask_flatten = mask.flatten()
            masked_points_world_space = points_world_space[mask_flatten > 0]
            masked_point_cloud.append(masked_points_world_space)

        # Combine all points into a single point cloud
        full_point_cloud = np.concatenate(full_point_cloud, axis=0)
        masked_point_cloud = np.concatenate(masked_point_cloud, axis=0)

        return full_point_cloud, masked_point_cloud
    
    def visualize_point_clouds(self, full_point_cloud, masked_point_cloud):
        """
        Visualizes two 3D point clouds using Open3D.

        Parameters:
            full_point_cloud (np.array): 3D point cloud of all points (N, 3).
            masked_point_cloud (np.array): 3D point cloud of masked points (M, 3).
        """

        # Convert full point cloud to Open3D format
        full_pc = o3d.geometry.PointCloud()
        full_pc.points = o3d.utility.Vector3dVector(full_point_cloud)
        full_pc.paint_uniform_color([0, 0.651, 0.929])  # Light blue color

        # Convert masked point cloud to Open3D format
        masked_pc = o3d.geometry.PointCloud()
        masked_pc.points = o3d.utility.Vector3dVector(masked_point_cloud)
        masked_pc.paint_uniform_color([1, 0.706, 0])  # Orange color

        # Visualize the point clouds
        o3d.visualization.draw_geometries([full_pc, masked_pc])
    

def main():
    rospy.init_node('panda_ws')
    panda_ws = PandaWorkspace()
    # panda_ws.panda_arm.go_home()
    # panda_ws.panda_arm.move_gripper_width(0.1)

    instruction = 'teal_cube'

    x, y, z = panda_ws.get_target_position(instruction)

    #####################################################################################################
    ####### to get topdown semenatic map

    # close_look_image_array = panda_ws.get_close_look(x, y, z)

    
    # close_look_image = Image.fromarray(close_look_image_array)
    # close_look_image.save(f'/home/master_oogway/panda_ws/code/GEM2.0/data/close_up/{instruction}.png')
    # mask = clip_dino_seg(close_look_image, instruction)

    # padding_size = (KEVIN_WS_BOUNDS[1] - KEVIN_WS_BOUNDS[0] - KEVIN_WS_BOUNDS[3] + KEVIN_WS_BOUNDS[2]) // 2
    
    # mask_image = Image.fromarray(np.array(mask.detach().cpu()))

    # transform = T.Compose([
    #     T.Resize((KEVIN_WS_BOUNDS[3] - KEVIN_WS_BOUNDS[2], KEVIN_WS_BOUNDS[3] - KEVIN_WS_BOUNDS[2])),
    #     T.Pad(padding=(padding_size, 0, padding_size, 0)),
    # ])
    # mask_image = transform(mask_image)
    # mask_image.save(f'/home/master_oogway/panda_ws/code/GEM2.0/data/close_up/{instruction}_mask.png')

    # import matplotlib.pyplot as plt
    # plt.imshow(close_look_image)
    # plt.show(block=False)

    # u, v = input('Enter the pixel coordinates of the object: input format "u,v"').split(',')
    # grasping_point = panda_ws.uv_to_pos(int(u), int(v), DIST_ABOVE_TARGET, 'tim')
    # panda_ws.panda_arm.move_gripper_to_pose(grasping_point[0], grasping_point[1], 0.2, 3.14, 0, -0.8)
    # panda_ws.panda_arm.move_gripper_to_pose(grasping_point[0], grasping_point[1], 0.04, 3.14, 0, -0.8)
    # panda_ws.panda_arm.move_gripper_width(0) 
    #####################################################################################################
    #####################################################################################################
    ## to get the nerf images or gsplat images

    # n = 10
    # nerf_path = f'/home/master_oogway/panda_ws/code/GEM2.0/data/nerf/{instruction}'
    # splat_path = f'/home/master_oogway/panda_ws/code/InstantSplat/data/custom/{instruction}/{n}_views/images'
    # panda_ws.circle_target_and_capture_images(y, x, z/3., RADIUS_AROUND_TARGET, n, nerf_path)
    # cam_in_world = json.load(open(f'{nerf_path}/tim_extrinsic.json', 'r'))['extrinsic']
    # cam_in_world = np.array(cam_in_world)
    # cam_in_nerf = json.load(open(f'/home/master_oogway/panda_ws/code/GEM2.0/data/nerf/{instruction}/transforms.json', 'r'))['frames'][0]['transform_matrix']
    # cam_in_nerf = np.array(cam_in_nerf)
    # nerf_to_world = np.dot(cam_in_world, np.linalg.inv(cam_in_nerf))
    # world_to_local = np.array([
    #                     [1, 0, 0, -x],
    #                     [0, 1, 0, -y],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 0, 1]
    #                 ])
    # nerf_to_local = np.dot(world_to_local, nerf_to_world)

    # # ATTENTION: it is called nerf to world, but it is actually nerf to local

    # json.dump({'nerf_to_world': nerf_to_local.tolist()}, open(f'/home/master_oogway/panda_ws/code/GEM2.0/data/nerf/{instruction}/nerf_to_world.json', 'w'), indent=4)
    #####################################################################################################
    #####################################################################################################
    # ## to get the clip features

    # nerf_to_local = np.array(json.load(open(f'/home/master_oogway/panda_ws/code/GEM2.0/data/nerf/{instruction}/nerf_to_world.json', 'r'))['nerf_to_world'])
    # features = panda_ws.get_3d_mask('/home/master_oogway/panda_ws/code/f3rm/outputs/teal_cube/f3rm/2024-08-17_220342/config.yml',
    #                                  nerf_to_local, instruction, ((-0.5, -0.5, 0), (0.5, 0.5, 0.5)))

    # Then convolve with the picking kernel to determine the picking position in the nerf space,
    # and use nerf_to_world to get the exact picking position in the world space.
    # In the post processing of data, convert the picking position from world to nerf space.
    #####################################################################################################
    ## to get 3D point clouds from flood fill

    n = 3
    image_path = f'/home/master_oogway/panda_ws/code/GEM2.0/data/point_cloud_from_depth/{instruction}'
    images, depths, poses = panda_ws.circle_target_and_capture_images(y, x, z, RADIUS_AROUND_TARGET, n, image_path)
    panda_ws.panda_arm.go_home()
    masks = [clip_dino_seg(image, instruction) for image in images]
    size = min(depths[0].shape)
    reshaped_masks = [F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(size, size), 
                                    mode='bilinear', align_corners=False).squeeze(0).squeeze(0) for mask in masks]
    resized_mask = torch.zeros(depths[0].shape)
    left = (depths[0].shape[0] - size) // 2
    right = left + size
    top = (depths[0].shape[1] - size) // 2
    down = top + size
    resized_masks = []
    for i in range(len(reshaped_masks)):
        resized_mask[left:right, top:down] = reshaped_masks[i]
        resized_masks.append(resized_mask)
    full_point_cloud, masked_point_cloud = panda_ws.reconstruct_point_clouds(depths, resized_masks, poses)
    panda_ws.visualize_point_clouds(full_point_cloud, masked_point_cloud)

    #####################################################################################################
    panda_ws.panda_arm.go_home()
    panda_ws.panda_arm.move_gripper_width(0.1)

if __name__ == '__main__':
    main()