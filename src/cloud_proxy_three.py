import time

import rospy
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
import ros_numpy
import tf2_ros
import tf

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
from panda_utils import demo_util_pp, transformation
import sys
sys.path.append("/home/master_oogway/panda_ws/src/GEM2.0")

# import scipy

class CloudProxy:
    def __init__(self):
        self.fixed_cam_names = ['dave', 'mel', 'kevin', 'bob', 'stuart']
        # self.fixed_cam_names = ['stuart']
        self.inhand_cam_name = 'tim'
        self.cam_names = self.fixed_cam_names + [self.inhand_cam_name]
        self.base_frame = 'panda_link0'

        self.topic1_depth = "/bob/aligned_depth_to_color/image_raw"
        self.topic2_depth = "/kevin/aligned_depth_to_color/image_raw"
        self.topic3_depth = "/stuart/aligned_depth_to_color/image_raw"
        self.topic4_depth = "/tim/aligned_depth_to_color/image_raw"
        self.topic5_depth = "/dave/aligned_depth_to_color/image_raw"
        self.topic6_depth = "/mel/aligned_depth_to_color/image_raw"
        self.sub1_depth = rospy.Subscriber(self.topic1_depth, Image, self.call_back_depth1, queue_size=1)
        self.depth1 = None
        self.sub2_depth = rospy.Subscriber(self.topic2_depth, Image, self.call_back_depth2, queue_size=1)
        self.depth2 = None
        self.sub3_depth = rospy.Subscriber(self.topic3_depth, Image, self.call_back_depth3, queue_size=1)
        self.depth3 = None
        self.sub4_depth = rospy.Subscriber(self.topic4_depth, Image, self.call_back_depth4, queue_size=1)
        self.depth4 = None
        self.sub5_depth = rospy.Subscriber(self.topic5_depth, Image, self.call_back_depth5, queue_size=1)
        self.depth5 = None
        self.sub6_depth = rospy.Subscriber(self.topic6_depth, Image, self.call_back_depth6, queue_size=1)
        self.depth6 = None

        self.topic1_info = "/bob/color/camera_info"
        self.topic2_info = "/kevin/color/camera_info"
        self.topic3_info = "/stuart/color/camera_info"
        self.topic4_info = "/tim/color/camera_info"
        self.topic5_info = "/dave/color/camera_info"
        self.topic6_info = "/mel/color/camera_info"
        self.sub1_info = rospy.Subscriber(self.topic1_info, CameraInfo, self.call_back_info1, queue_size=1)
        self.info1 = None
        self.sub2_info = rospy.Subscriber(self.topic2_info, CameraInfo, self.call_back_info2, queue_size=1)
        self.info2 = None
        self.sub3_info = rospy.Subscriber(self.topic3_info, CameraInfo, self.call_back_info3, queue_size=1)
        self.info3 = None
        self.sub4_info = rospy.Subscriber(self.topic4_info, CameraInfo, self.call_back_info4, queue_size=1)
        self.info4 = None
        self.sub5_info = rospy.Subscriber(self.topic5_info, CameraInfo, self.call_back_info5, queue_size=1)
        self.info5 = None
        self.sub6_info = rospy.Subscriber(self.topic6_info, CameraInfo, self.call_back_info6, queue_size=1)
        self.info6 = None

        self.topic1_rgb = "/bob/color/image_raw"
        self.topic2_rgb = "/kevin/color/image_raw"
        self.topic3_rgb = "/stuart/color/image_raw"
        self.topic4_rgb = "/tim/color/image_raw"
        self.topic5_rgb = "/dave/color/image_raw"
        self.topic6_rgb = "/mel/color/image_raw"
        self.sub1_rgb = rospy.Subscriber(self.topic1_rgb, Image, self.call_back_rgb1, queue_size=1)
        self.rgbimg1 = None
        self.sub2_rgb = rospy.Subscriber(self.topic2_rgb, Image, self.call_back_rgb2, queue_size=1)
        self.rgbimg2 = None
        self.sub3_rgb = rospy.Subscriber(self.topic3_rgb, Image, self.call_back_rgb3, queue_size=1)
        self.rgbimg3 = None
        self.sub4_rgb = rospy.Subscriber(self.topic4_rgb, Image, self.call_back_rgb4, queue_size=1)
        self.rgbimg4 = None
        self.sub5_rgb = rospy.Subscriber(self.topic5_rgb, Image, self.call_back_rgb5, queue_size=1)
        self.rgbimg5 = None
        self.sub6_rgb = rospy.Subscriber(self.topic6_rgb, Image, self.call_back_rgb6, queue_size=1)
        self.rgbimg6 = None

        self.rgb1_frame = "bob_color_optical_frame"
        self.rgb2_frame = "kevin_color_optical_frame"
        self.rgb3_frame = "stuart_color_optical_frame"
        self.rgb4_frame = "tim_color_optical_frame"
        self.rgb5_frame = "dave_color_optical_frame"
        self.rgb6_frame = "mel_color_optical_frame"

        # RT base_link
        # self.workspace = np.array([[0.28, 0.71],
        #                             [-0.32, 0.26],
        #                             [-0.1, 1.0]])
        self.workspace = np.array([[0.20, 0.75],
                                    [-0.30, 0.25],
                                    [-0.02, 1.0]])
        self.center = self.workspace.mean(-1)
        self.x_size = self.workspace[0].max() - self.workspace[0].min()
        self.x_half = self.x_size/2
        self.y_size = self.workspace[1].max() - self.workspace[1].min()
        self.y_half = self.y_size/2
        self.z_min = self.workspace[2].min()

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

        rospy.sleep(1) # key thing, dont delete

    def call_back_cloud1(self, msg):
        self.msg1 = msg
        cloudTime = self.msg1.header.stamp
        cloudFrame = self.msg1.header.frame_id
        cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(self.msg1)
        cloud, rgb = self.get_xyzrgb_points(cloud_arr, remove_nans=True)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        T = self.lookup_transform(cloudFrame, self.base_frame, rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud1 = cloud
        self.rgb1 = rgb
    
    def call_back_cloud2(self, msg):
        self.msg2 = msg
        cloudTime = self.msg2.header.stamp
        cloudFrame = self.msg2.header.frame_id
        cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(self.msg2)
        cloud, rgb = self.get_xyzrgb_points(cloud_arr, remove_nans=True)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        T = self.lookup_transform(cloudFrame, self.base_frame, rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud2 = cloud
        self.rgb2 = rgb

    def call_back_cloud3(self, msg):
        self.msg3 = msg
        cloudTime = self.msg3.header.stamp
        cloudFrame = self.msg3.header.frame_id
        cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(self.msg3)
        cloud, rgb = self.get_xyzrgb_points(cloud_arr, remove_nans=True)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        T = self.lookup_transform(cloudFrame, self.base_frame, rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud3 = cloud
        self.rgb3 = rgb
    
    def call_back_cloud4(self, msg):
        self.msg4 = msg
        cloudTime = self.msg4.header.stamp
        cloudFrame = self.msg4.header.frame_id
        cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(self.msg4)
        cloud, rgb = self.get_xyzrgb_points(cloud_arr, remove_nans=True)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        T = self.lookup_transform(cloudFrame, self.base_frame, rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud4 = cloud
        self.rgb4 = rgb
    
    def call_back_info1(self, msg):
        self.info1 = np.array(msg.K).reshape(3,3)
    
    def call_back_info2(self, msg):
        self.info2 = np.array(msg.K).reshape(3,3)

    def call_back_info3(self, msg):
        self.info3 = np.array(msg.K).reshape(3,3)

    def call_back_info4(self, msg):
        self.info4 = np.array(msg.K).reshape(3,3)
    
    def call_back_info5(self, msg):
        self.info5 = np.array(msg.K).reshape(3,3)

    def call_back_info6(self, msg):
        self.info6 = np.array(msg.K).reshape(3,3)

    def call_back_depth1(self, msg):
        self.depth1 = ros_numpy.numpify(msg)

    def call_back_depth2(self, msg):
        self.depth2 = ros_numpy.numpify(msg)

    def call_back_depth3(self, msg):
        self.depth3 = ros_numpy.numpify(msg)
    
    def call_back_depth4(self, msg):
        self.depth4 = ros_numpy.numpify(msg)

    def call_back_depth5(self, msg):
        self.depth5 = ros_numpy.numpify(msg)
    
    def call_back_depth6(self, msg):
        self.depth6 = ros_numpy.numpify(msg)

    def call_back_rgb1(self, msg):
        self.rgbimg1 = ros_numpy.numpify(msg)

    def call_back_rgb2(self, msg):
        self.rgbimg2 = ros_numpy.numpify(msg)

    def call_back_rgb3(self, msg):
        self.rgbimg3 = ros_numpy.numpify(msg)
    
    def call_back_rgb4(self, msg):
        self.rgbimg4 = ros_numpy.numpify(msg)
    
    def call_back_rgb5(self, msg):
        self.rgbimg5 = ros_numpy.numpify(msg)

    def call_back_rgb6(self, msg):
        self.rgbimg6 = ros_numpy.numpify(msg)

    def get_depth_image(self, cam_id, iteration=10):
        images = []
        for _ in range(iteration):
            if cam_id == 1 or cam_id == 'bob':
                while self.depth1 is None:
                    rospy.sleep(0.01)
                images.append(self.depth1/1000)
            elif cam_id == 2 or cam_id == 'kevin':
                while self.depth2 is None:
                    rospy.sleep(0.01)
                images.append(self.depth2/1000)
            elif cam_id == 3 or cam_id == 'stuart':
                while self.depth3 is None:
                    rospy.sleep(0.01)
                images.append(self.depth3/1000)
            elif cam_id == 4 or cam_id == 'tim':
                while self.depth4 is None:
                    rospy.sleep(0.01)
                images.append(self.depth4/1000)
            elif cam_id == 5 or cam_id == 'dave':
                while self.depth5 is None:
                    rospy.sleep(0.01)
                images.append(self.depth5/1000)
            elif cam_id == 6 or cam_id == 'mel':
                while self.depth6 is None:
                    rospy.sleep(0.01)
                images.append(self.depth6/1000)
            else:
                NotImplementedError
        image = np.median(images, axis=0)
        return image

    def get_rgb_image(self, cam_id):
        if cam_id == 1 or cam_id == 'bob':
            while self.rgbimg1 is None:
                rospy.sleep(0.01)
            return self.rgbimg1
        elif cam_id == 2 or cam_id == 'kevin':
            while self.rgbimg2 is None:
                rospy.sleep(0.01)
            return self.rgbimg2
        elif cam_id == 3 or cam_id == 'stuart':
            while self.rgbimg3 is None:
                rospy.sleep(0.01)
            return self.rgbimg3
        elif cam_id == 4 or cam_id == 'tim':
            while self.rgbimg4 is None:
                rospy.sleep(0.01)
            return self.rgbimg4
        elif cam_id == 5 or cam_id == 'dave':
            while self.rgbimg5 is None:
                rospy.sleep(0.01)
            return self.rgbimg5
        elif cam_id == 6 or cam_id == 'mel':
            while self.rgbimg6 is None:
                rospy.sleep(0.01)
            return self.rgbimg6
        else:
            NotImplementedError

    
    def transform_cloud_to_base(self, cloud, cam_id):
        if cam_id == 1 or cam_id == 'bob':
            rgb_frame = self.rgb1_frame
        elif cam_id == 2 or cam_id == 'kevin':
            rgb_frame = self.rgb2_frame
            self.proj_pose = self.lookup_transform(rgb_frame, self.base_frame, rospy.Time(0))
        elif cam_id == 3 or cam_id == 'stuart':
            rgb_frame = self.rgb3_frame
        elif cam_id == 4 or cam_id == 'tim':
            rgb_frame = self.rgb4_frame
        elif cam_id == 5 or cam_id == 'dave':
            rgb_frame = self.rgb5_frame
        elif cam_id == 6 or cam_id == 'mel':
            rgb_frame = self.rgb6_frame
        # if cam_id == 4 or cam_id == 'tim':
        #     rgb_frame = self.rgb4_frame
        T = self.lookup_transform(rgb_frame, self.base_frame, rospy.Time(0))
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
        self.depth4 = None
        self.depth5 = None
        self.depth6 = None
        self.rgbimg1 = None
        self.rgbimg2 = None
        self.rgbimg3 = None
        self.rgbimg4 = None
        self.rgbimg5 = None
        self.rgbimg6 = None
        self.cloud1 = None
        self.cloud2 = None
        self.cloud3 = None
        self.cloud4 = None
        self.cloud5 = None
        self.cloud6 = None

    def pad_bottom_cloud(self, cloud, rgb_value=[255,255,255]):
        # generate 'fake' point cloud for area outside the bins
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
    
    def get_fused_clip_cloud(self, instruction):
        self.clear_cache()
        cloud1 = self.get_pointcloud_from_depth(1, instruction)
        cloud1 = self.transform_clip_cloud_to_base(cloud1, 1)
        cloud2 = self.get_pointcloud_from_depth(2, instruction)
        cloud2 = self.transform_clip_cloud_to_base(cloud2, 2)
        cloud3 = self.get_pointcloud_from_depth(3, instruction)
        cloud3 = self.transform_clip_cloud_to_base(cloud3, 3)
        cloud = np.concatenate([cloud1, cloud2, cloud3], axis=0)
        # cloud = np.concatenate([cloud1, cloud3], axis=0)
        
        cloud = self.cloud_preprocess(cloud)
        cloud1 = self.cloud_preprocess(cloud1)
        cloud2 = self.cloud_preprocess(cloud2)
        cloud3 = self.cloud_preprocess(cloud3)
        
        cloud = self.pad_bottom_cloud(cloud, rgb_value=self.rgb_value)
        cloud2 = self.pad_bottom_cloud(cloud2, rgb_value=self.rgb_value)
        

        return cloud, cloud1, cloud2, cloud3
    
    def get_pointcloud_in_cam_frame(self, cam_name):
        rgb = self.get_rgb_image(cam_name)
        depth = self.get_depth_image(cam_name)
        intrinsics = self.get_cam_intrinsic(cam_name)

        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
        
        points = np.float32([px, py, depth, rgb[..., 0], rgb[..., 1], rgb[..., 2]]).transpose(1, 2, 0)
        cloud = points.reshape(-1,6)
        # z_constrain= (cloud[:,2]>0.1) & (cloud[:,2]<1.1)
        # cloud = cloud[z_constrain]
        visualize = False
        if visualize:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(cloud[:,:3])
            pcd.colors = open3d.utility.Vector3dVector(cloud[:, 3:6]/255)
            open3d.visualization.draw_geometries([pcd])
        return cloud

    def get_fused_cloud(self, instruction):
        self.clear_cache()
        cloud1 = self.get_pointcloud_from_depth(1)
        cloud1 = self.transform_clip_cloud_to_base(cloud1, 1)
        cloud2 = self.get_pointcloud_from_depth(2)
        cloud2 = self.transform_clip_cloud_to_base(cloud2, 2)
        cloud3 = self.get_pointcloud_from_depth(3)
        cloud3 = self.transform_clip_cloud_to_base(cloud3, 3)
        cloud = np.concatenate([cloud1, cloud2, cloud3], axis=0)
        # cloud = np.concatenate([cloud1, cloud3], axis=0)
        
        cloud = self.cloud_preprocess(cloud)
        cloud1 = self.cloud_preprocess(cloud1)
        cloud2 = self.cloud_preprocess(cloud2)
        cloud3 = self.cloud_preprocess(cloud3)
        
        cloud = self.pad_bottom_cloud(cloud, rgb_value=self.rgb_value)
        cloud2 = self.pad_bottom_cloud(cloud2, rgb_value=self.rgb_value)
        

        return cloud, cloud1, cloud2, cloud3
    
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

        return xyz, rbgs
    
    def transform(self, cloud, T, isPosition=True):
        '''Apply the homogeneous transform T to the point cloud. Use isPosition=False if transforming unit vectors.'''

        n = cloud.shape[0]
        cloud = cloud.T
        augment = np.ones((1, n)) if isPosition else np.zeros((1, n))
        cloud = np.concatenate((cloud, augment), axis=0)
        cloud = np.dot(T, cloud)
        cloud = cloud[0:3, :].T
        return cloud

    def lookup_transform(self, fromFrame, toFrame, lookupTime=rospy.Time(0)):
        """
        Lookup a transform in the TF tree.for instruction in instructions:
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
    
    def get_cam_intrinsic(self, cam_id):
        if cam_id == 1 or cam_id == 'bob':
            while self.info1 is None:
                rospy.sleep(0.01)
            intrinsics = self.info1
        if cam_id == 2 or cam_id == 'kevin':
            while self.info2 is None:
                rospy.sleep(0.01)
            intrinsics = self.info2
        if cam_id == 3 or cam_id == 'stuart':
            while self.info3 is None:
                rospy.sleep(0.01)
            intrinsics = self.info3
        if cam_id == 4 or cam_id == 'tim':
            while self.info4 is None:
                rospy.sleep(0.01)
            intrinsics = self.info4
        if cam_id == 5 or cam_id == 'dave':
            while self.info5 is None:
                rospy.sleep(0.01)
            intrinsics = self.info5
        if cam_id == 6 or cam_id == 'mel':
            while self.info6 is None:
                rospy.sleep(0.01)
            intrinsics = self.info6
        return intrinsics

    def get_cam_extrinsic(self, cam_id):
        if cam_id == 1 or cam_id == 'bob':
            rgb_frame = self.rgb1_frame
            extrinsic = self.lookup_transform(rgb_frame, self.base_frame, rospy.Time(0))
        if cam_id == 2 or cam_id == 'kevin':
            rgb_frame = self.rgb2_frame
            extrinsic = self.lookup_transform(rgb_frame, self.base_frame, rospy.Time(0))
        if cam_id == 3 or cam_id == 'stuart':
            rgb_frame = self.rgb3_frame
            extrinsic = self.lookup_transform(rgb_frame, self.base_frame, rospy.Time(0))
        if cam_id == 4 or cam_id == 'tim':
            rgb_frame = self.rgb4_frame
            extrinsic = self.lookup_transform(rgb_frame, self.base_frame, rospy.Time(0))
        if cam_id == 5 or cam_id == 'dave':
            rgb_frame = self.rgb5_frame
            extrinsic = self.lookup_transform(rgb_frame, self.base_frame, rospy.Time(0))
        if cam_id == 5 or cam_id == 'mel':
            rgb_frame = self.rgb6_frame
            extrinsic = self.lookup_transform(rgb_frame, self.base_frame, rospy.Time(0))
        return extrinsic
    
    def get_all_cam_extrinsic(self):
        extrinsics = {}
        for cam_id in self.cam_names:
            extrinsics[cam_id] = self.get_cam_extrinsic(cam_id)
        return extrinsics
    
    def get_all_cam_intrinsic(self):
        intrinsics = {}
        for cam_id in self.cam_names:
            intrinsics[cam_id] = self.get_cam_intrinsic(cam_id)
        return intrinsics
    
    def get_whole_pc_in_robot_frame(self, allow_in_hand=False):
        clouds = []
        for cam in self.fixed_cam_names:
            cloud = self.get_pointcloud_in_cam_frame(cam)
            cloud = self.transform_cloud_to_base(cloud, cam)
            clouds.append(cloud)

        if allow_in_hand:
            cloud_inhand = self.get_pointcloud_in_cam_frame(self.inhand_cam_name)
            cloud_inhand = self.transform_cloud_to_base(cloud_inhand, 4)
            clouds.append(cloud_inhand)

        cloud = np.concatenate(clouds, axis=0)
        return cloud
    
    def get_workspace_pc(self, allow_in_hand=False):
        self.clear_cache()
        cloud = self.get_whole_pc_in_robot_frame(allow_in_hand)
        # filter ws x
        x_cond = (cloud[:, 0] < self.center[0] + self.x_half) * (cloud[:, 0] > self.center[0] - self.x_half)
        cloud = cloud[x_cond]
        # filter ws y
        y_cond = (cloud[:, 1] < self.center[1] + self.y_half) * (cloud[:, 1] > self.center[1] - self.y_half)
        cloud = cloud[y_cond]
        # filter ws z
        z_cond = (cloud[:, 2] < self.center[2].max()) * (cloud[:, 2] > self.z_min)
        cloud = cloud[z_cond]

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cloud[:,:3])
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=3.5)
        cloud = cloud[ind]

        return cloud


def main():
    rospy.init_node('test')
    cloud_proxy = CloudProxy()
    cloud_proxy.get_cam_extrinsic('bob')
    instructions = ['cup', 'hammer', 'cube']
    for instruction in instructions:
        cloud_proxy.cloud = None
        cloud, cloud1, cloud2, cloud3 = cloud_proxy.get_fused_clip_cloud(instruction)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cloud[:, :3])
        clip_cloud = np.copy(cloud[:, 6:7])
        clip_cloud = (clip_cloud - clip_cloud.min()) / (clip_cloud.max() - clip_cloud.min())
        pcd.colors = open3d.utility.Vector3dVector(cloud1[:, 3:6] / 255 * 0.5 + clip_cloud)
        open3d.visualization.draw_geometries([pcd])
    #     depth, rgbc, rgbc_average = cloud_proxy.getHeightmapReconstruct(cloud, [cloud1, cloud2, cloud3])
    #     depth, rgbc, rgbc_average = cloud_proxy.getHeightmapReconstruct(cloud2)
    #     rgbc = cloud_proxy._preProcessRGBC(rgbc)
    cloud_proxy.get_pointcloud_from_depth(1, 'cube')

def test_point_cloud():
    rospy.init_node('test')
    cloud_proxy = CloudProxy()
    cloud = cloud_proxy.get_workspace_pc()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cloud[:,:3])
    pcd.colors = open3d.utility.Vector3dVector(cloud[:, 3:6]/255)
    open3d.visualization.draw_geometries([pcd])

    # For saving pcds and images

    # for cam_name in cloud_proxy.fixed_cam_names:
    #     image = cloud_proxy.get_rgb_image(cam_name)
    #     np.save('/home/master_oogway/panda_ws/code/helping_hands_panda/data/skye_teapot_data/rgb_{}_3.npy'.format(cam_name), image)

    # np.save('/home/master_oogway/panda_ws/code/helping_hands_panda/data/skye_teapot_data/pcd_points_3.npy', np.array(pcd.points))
    # np.save('/home/master_oogway/panda_ws/code/helping_hands_panda/data/skye_teapot_data/pcd_colors_3.npy', np.array(pcd.colors))

    print(1)


if __name__ == '__main__':
    # main()
    test_point_cloud()