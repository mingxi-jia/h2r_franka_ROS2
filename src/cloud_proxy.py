#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
import tf_transformations
from geometry_msgs.msg import TransformStamped
import time
import open3d 

CAM_INDEX_MAP = {
            'bob': 0,
            'kevin': 1,
            'stuart': 2,
            'dave': 3,
            'mel': 4,
            'tim': 5, # the in hand cam needs 
        }
CAM_INDEX_MAP = {k: v for k, v in sorted(CAM_INDEX_MAP.items(), key=lambda item: item[1])}
# Initialize depth topics and subscriptions
TOPICS_DEPTH = {
    'bob': "/bob/aligned_depth_to_color/image_raw",
    'kevin': "/kevin/aligned_depth_to_color/image_raw",
    'stuart': "/stuart/aligned_depth_to_color/image_raw",
    'tim': "/tim/aligned_depth_to_color/image_raw",
    'dave': "/dave/aligned_depth_to_color/image_raw",
    'mel': "/mel/aligned_depth_to_color/image_raw"
}
TOPICS_RGB = {
    'bob': "/bob/color/image_raw",
    'kevin': "/kevin/color/image_raw",
    'stuart': "/stuart/color/image_raw",
    'tim': "/tim/color/image_raw",
    'dave': "/dave/color/image_raw",
    'mel': "/mel/color/image_raw"
}
RGB_FRAMES = {
    'bob': "bob_color_optical_frame",
    'kevin': "kevin_color_optical_frame",
    'stuart': "stuart_color_optical_frame",
    'tim': "tim_color_optical_frame",
    'dave': "dave_color_optical_frame",
    'mel': "mel_color_optical_frame"
}
TOPICS_CAM_INFO = {
    'bob': "/bob/color/camera_info",
    'kevin': "/kevin/color/camera_info",
    'stuart': "/stuart/color/camera_info",
    'tim': "/tim/color/camera_info",
    'dave': "/dave/color/camera_info",
    'mel': "/mel/color/camera_info"
}

class CloudProxy(Node):
    def __init__(self, workspace_size: np.array, use_inhand=False):
        super().__init__('cloud_proxy')

        # Camera names and frames
        self.fixed_cam_names = list(CAM_INDEX_MAP.keys())[:5]
        if use_inhand:
            self.inhand_cam_name = 'tim'
            self.fixed_cam_names.append('tim')
        self.cam_names = self.fixed_cam_names
        self.base_frame = 'fr3_link0'
        self.use_inhand = use_inhand

        self.depth_images = [None] * len(self.cam_names)
        self.camera_intrinsics = [None] * len(self.cam_names)
        self.rgb_images = [None] * len(self.cam_names)

        # Create subscriptions for depth, camera info, and RGB
        for i, cam in enumerate(self.cam_names):
            rgb_topic = TOPICS_RGB[cam]
            self.create_subscription(Image, rgb_topic, lambda msg, idx=i: self.rgb_callback(msg, idx), 10)
            depth_topic = TOPICS_DEPTH[cam]
            self.create_subscription(Image, depth_topic, lambda msg, idx=i: self.depth_callback(msg, idx), 10)
            info_topic = TOPICS_CAM_INFO[cam]
            self.create_subscription(CameraInfo, info_topic, lambda msg, idx=i: self.info_callback(msg, idx), 10)
        

        self.workspace = workspace_size
        self.center = self.workspace.mean(-1)
        self.x_size = self.workspace[0].max() - self.workspace[0].min()
        self.x_half = self.x_size / 2
        self.y_size = self.workspace[1].max() - self.workspace[1].min()
        self.y_half = self.y_size / 2
        self.z_min = self.workspace[2].min()

        # Transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread = True)

        self.bridge = CvBridge()

        # Wait for initialization
        rclpy.spin_once(self, timeout_sec=1)
        time.sleep(1)
        
        self.camera_extrinsics = dict()
        self.look_and_set_extrinsics()


    # ----------------------------ROS related functions---------------------------
    def depth_callback(self, msg, index):
        # h, w = msg.height, msg.width
        self.depth_images[index] = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) / 1000

    def info_callback(self, msg, index):
        self.camera_intrinsics[index] = np.array(msg.k).reshape(3,3)

    def rgb_callback(self, msg, index):
        self.rgb_images[index] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
    
    # ----------------------------Basic Camera functions---------------------------
    def get_rgb_image(self, cam_id):
        # Map camera IDs to their respective indices in the depth_images list
        if cam_id not in CAM_INDEX_MAP:
            raise NotImplementedError(f"Camera ID {cam_id} not supported")
        
        index = CAM_INDEX_MAP[cam_id]
        while self.rgb_images[index] is None:
            rclpy.spin_once(self, timeout_sec=0.01)
        rgb = self.rgb_images[index]
        return rgb
    
    def get_depth_image(self, cam_id, iteration=1):
        # Map camera IDs to their respective indices in the depth_images list
        if cam_id not in CAM_INDEX_MAP:
            raise NotImplementedError(f"Camera ID {cam_id} not supported")
        
        index = CAM_INDEX_MAP[cam_id]
        images = []

        # Collect multiple depth images for averaging
        for _ in range(iteration):
            while self.depth_images[index] is None:
                rclpy.spin_once(self, timeout_sec=0.01)
            
            images.append(self.depth_images[index])
            self.clear_depth(index)

        # Return the median of the collected images
        return np.median(images, axis=0)
    
    def get_cam_intrinsic(self, cam_id):
        if cam_id not in CAM_INDEX_MAP:
            raise NotImplementedError(f"Camera ID {cam_id} not supported")
        
        index = CAM_INDEX_MAP[cam_id]
        while self.camera_intrinsics[index] is None:
            rclpy.spin_once(self, timeout_sec=0.01)
        intrinsics = self.camera_intrinsics[index]
        return intrinsics
    
    def get_cam_extrinsic(self, cam_id):
        if cam_id not in CAM_INDEX_MAP:
            raise NotImplementedError(f"Camera ID {cam_id} not supported")
        
        rgb_frame = RGB_FRAMES[cam_id]
        T = self.lookup_transform(rgb_frame, self.base_frame)
        return T
    
    def look_and_set_extrinsics(self):
        print("init: getting camera extrinsics")
        for cam in self.cam_names:
            extrinsic = self.get_cam_extrinsic(cam)
            self.camera_extrinsics[cam] = extrinsic
    
    # ----------------------------Util functions---------------------------
    def clear_depth(self, cam_index):
        self.depth_images[cam_index] = None
    
    def clear_cache(self):
        self.depth_images = [None] * len(self.cam_names)
        self.rgb_images = [None] * len(self.cam_names)
        
    def transform(self, cloud, T, isPosition=True):
        '''Apply the homogeneous transform T to the point cloud. Use isPosition=False if transforming unit vectors.'''

        n = cloud.shape[0]
        cloud = cloud.T
        augment = np.ones((1, n)) if isPosition else np.zeros((1, n))
        cloud = np.concatenate((cloud, augment), axis=0)
        cloud = np.dot(T, cloud)
        cloud = cloud[0:3, :].T
        return cloud
    
    def lookup_transform(self, fromFrame, toFrame, lookupTime=Time(seconds=0)):
        """
        Lookup a transform in the TF tree.for instruction in instructions:
        :param fromFrame: the frame from which the transform is calculated
        :type fromFrame: string
        :param toFrame: the frame to which the transform is calculated
        :type toFrame: string
        :return: transformation matrix from fromFrame to toFrame
        :rtype: 4x4 np.array
        """
        keep_trying = True
        while keep_trying:
            try:
                transformMsg = self.tf_buffer.lookup_transform(toFrame, fromFrame, lookupTime, Duration(seconds=5))
                keep_trying = False
            except:
                rclpy.spin_once(self, timeout_sec=1)
                
        
        translation = transformMsg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transformMsg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        T = tf_transformations.quaternion_matrix(quat)
        T[0:3, 3] = pos
        return T
    
    def get_all_cam_extrinsic(self):
        # for cam_id in self.cam_names:
        #     extrinsics[cam_id] = self.get_cam_extrinsic(cam_id)
        extrinsics = self.camera_extrinsics
        return extrinsics
    
    def get_all_cam_intrinsic(self):
        intrinsics = {}
        for cam_id in self.cam_names:
            intrinsics[cam_id] = self.get_cam_intrinsic(cam_id)
        return intrinsics
    # ----------------------------Point cloud functions---------------------------
    def transform_cloud_to_base(self, cloud, cam_id):
        rgb_frame = RGB_FRAMES[cam_id]
        T = self.lookup_transform(rgb_frame, self.base_frame)
        cloud_RT_base = self.transform(cloud[:, :3], T)
        return np.concatenate([cloud_RT_base, cloud[:, 3:]], axis=1)
    
    def get_pointcloud_in_cam_frame(self, cam_name):
        # (H, W, 3)
        rgb = self.get_rgb_image(cam_name)
        # (H, W)
        depth = self.get_depth_image(cam_name)
        # (3, 3)
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
    
    def get_env_screenshot(self, file_name):
        screenshot = dict()
        # get workspace meta
        screenshot['workspace_size'] = self.workspace
        screenshot['cam_names'] = self.cam_names
        screenshot['enable_inhand'] = self.use_inhand
        # get camera meta
        extrinsic_dict = self.get_all_cam_extrinsic()
        intrinsic_dict = self.get_all_cam_intrinsic()
        screenshot['extrinsic'] = extrinsic_dict
        screenshot['intrinsic'] = intrinsic_dict
        # get multiview images
        rgb_dict = dict()
        depth_dict = dict()
        for cam in self.cam_names:
            rgb_dict[cam] = self.get_rgb_image(cam)
            depth_dict[cam] = self.get_depth_image(cam)
        screenshot['rgbs'] = rgb_dict
        screenshot['depths'] = depth_dict
        # get workspace pointcloud
        pc = self.get_workspace_pc()
        screenshot['point_cloud'] = pc
        # save
        np.save(file_name, screenshot)
        
    
def main(args=None):
    rclpy.init(args=args)
    workspace_size = np.array([[0.3, 0.7],
                        [-0.2, 0.2],
                        [-0.02, 1.0]]) 
    cloud_proxy = CloudProxy(workspace_size=workspace_size, use_inhand=False)

    try:
        cloud_proxy.get_env_screenshot('screenshot.npy')
        print(1)
    except KeyboardInterrupt:
        pass
    finally:
        # this is needed because of ROS2 mechanism.
        # without destroy_node(), it somehow wont work if you restart the program
        cloud_proxy.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()
