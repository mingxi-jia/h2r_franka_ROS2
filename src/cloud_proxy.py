#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
import time
import open3d 
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import copy
import cv2
from scipy.spatial.transform import Rotation
from configs.camera_configs import (IMAGE_HEIGHT, IMAGE_WIDTH, CAM_INDEX_MAP, CAM_INDEX_MAP, TOPICS_DEPTH, TOPICS_RGB,
                                    RGB_FRAMES, TOPICS_CAM_INFO)

class CloudProxy(Node):
    def __init__(self, workspace_size: np.array, use_inhand=False, base_frame='fr3_link0'):
        super().__init__('cloud_proxy')

        # Camera names and frames
        self.fixed_cam_names = list(CAM_INDEX_MAP.keys())[:5]
        self.inhand_cam_name = 'tim'
        self.cam_names = copy.copy(self.fixed_cam_names)
        if use_inhand:
            print("Inhand camera activated")
            self.cam_names.append(self.inhand_cam_name)
        else:
            print("Inhand camera not activated")
        self.base_frame = base_frame
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
        # (self, PoseStamped, '/franka_robot_state_broadcaster/current_pose')

        self.workspace = workspace_size
        self.center = self.workspace.mean(-1) if base_frame=='fr3_link0' else np.array([0., 0., self.workspace.mean(-1)[-1]])
        self.x_size = self.workspace[0].max() - self.workspace[0].min()
        self.x_half = self.x_size / 2
        self.y_size = self.workspace[1].max() - self.workspace[1].min()
        self.y_half = self.y_size / 2
        self.z_min = self.workspace[2].min()
        
        # Transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread = True)
        # Initialize TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        self.bridge = CvBridge()

        # Wait for initialization
        rclpy.spin_once(self, timeout_sec=1)
        time.sleep(1)
        
        self.camera_extrinsics = dict()
        # self.look_and_set_extrinsics()
        self.look_and_set_extrinsics(base_frame)
        # Publish the transform
        # self.create_timer(0.1, lambda: self.publish_workspace_transform(self.center, [0.,0.,0.,1.]))


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
    
    def get_cam_extrinsic(self, cam_id, base_frame):
        if cam_id not in CAM_INDEX_MAP:
            raise NotImplementedError(f"Camera ID {cam_id} not supported")
        
        rgb_frame = RGB_FRAMES[cam_id]
        # print(cam_id)
        # print(self.use_inhand)
        # print(base_frame)
        # print(rgb_frame)
        # exit(0)
        T = self.lookup_transform(rgb_frame, base_frame)
        print(f"got {cam_id} pose.")
        return T
    
    def look_and_set_extrinsics(self, base_frame=None):
        if base_frame is None:
            base_frame = self.base_frame
        print(f"init: getting camera extrinsics in frame {base_frame}")
        for cam in self.cam_names:
            extrinsic = self.get_cam_extrinsic(cam, base_frame)
            self.camera_extrinsics[cam] = extrinsic
    
    # ----------------------------Util functions---------------------------
    def spin_for_30_times(self):
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.1)

    def clear_depth(self, cam_index):
        self.depth_images[cam_index] = None
    
    def clear_cache(self):
        self.spin_for_30_times()
        self.depth_images = [None] * len(self.cam_names)
        self.rgb_images = [None] * len(self.cam_names)
        if self.use_inhand:
            self.camera_extrinsics = dict()
            self.look_and_set_extrinsics(self.base_frame)
            time.sleep(0.5)
        self.spin_for_30_times()
        
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
                transformMsg = self.tf_buffer.lookup_transform(toFrame, fromFrame, lookupTime, Duration(seconds=1))
                keep_trying = False
            except Exception as e:
                print(e)
                rclpy.spin_once(self, timeout_sec=1)
                
        
        translation = transformMsg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transformMsg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        T = np.eye(4)
        R = Rotation.from_quat(quat).as_matrix()
        T[:3, :3] = R
        T[0:3, 3] = pos
        return T
    
    def get_all_cam_extrinsic(self):
        for cam_id in self.cam_names:
            self.camera_extrinsics[cam_id] = self.get_cam_extrinsic(cam_id, self.base_frame)
        # self.update_inhand_extrinsic()
        # extrinsics = self.camera_extrinsics
        return self.camera_extrinsics
    
    def get_all_cam_intrinsic(self):
        intrinsics = {}
        for cam_id in self.cam_names:
            intrinsics[cam_id] = self.get_cam_intrinsic(cam_id)
        return intrinsics
    
    def publish_workspace_transform(self, translation, rotation):
        """
        Publish a static transform from fr3_link to workspace_link.
        
        :param translation: A tuple (x, y, z) representing the translation
        :param rotation: A tuple (x, y, z, w) representing the quaternion
        """
        print("publishing workspace origin")
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'fr3_link'
        t.child_frame_id = 'workspace_link'

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]

        self.tf_broadcaster.sendTransform(t)
    
    def mask_out_photo(self, photo: np.ndarray, point_cloud: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """
        Masks out pixels in the photo that do not project onto any point in the point cloud.
        Parameters:
            photo (np.ndarray): RGB image array of shape (height, width, 3).
            point_cloud (np.ndarray): Array of 3D points of shape (N, 3), where N is the number of points.
            intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
            extrinsics (np.ndarray): 4x4 camera extrinsic matrix.
        Returns:
            np.ndarray: Masked photo with unprojected areas set to black.
        """
        extrinsics = np.linalg.inv(extrinsics)
        # Get image dimensions
        height, width, _ = photo.shape
        # Initialize a black mask
        mask = np.zeros((height, width), dtype=np.uint8)
        # Convert each 3D point to homogeneous coordinates (N, 4)
        point_cloud_h = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        # Transform points to camera coordinates using extrinsics (4x4)
        camera_coords = (extrinsics @ point_cloud_h.T).T  # Result is (N, 4)
        # Keep only points in front of the camera (where Z > 0)
        valid_points = camera_coords[:, 2] > 0
        camera_coords = camera_coords[valid_points]
        # Project to image plane using intrinsics
        image_coords = (intrinsics @ camera_coords[:, :3].T).T  # Result is (N, 3)
        # Normalize by the depth (Z-coordinate)
        u = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        v = (image_coords[:, 1] / image_coords[:, 2]).astype(int)
        # Filter points that fall within the image boundaries
        valid_pixels = (0 <= u) & (u < width) & (0 <= v) & (v < height)
        u, v = u[valid_pixels], v[valid_pixels]
        # Set valid pixels in the mask to white (255)
        mask[v, u] = 255
        # Create a 3-channel mask for the RGB photo
        mask_3ch = cv2.merge([mask, mask, mask])
        # Apply the mask to the photo, setting unprojected pixels to black
        masked_photo = np.where(mask_3ch == 255, photo, 0)
        return masked_photo, mask

    def project_pointcloud_to_image(self, pointcloud, intrinsics, extrinsics, image_width, image_height):
        """
        Project an xyzrgb point cloud onto a 2D image plane using camera intrinsics and extrinsics.
        Args:
            pointcloud (np.ndarray): Nx6 array with columns representing x, y, z, r, g, b.
            intrinsics (np.ndarray): 3x3 camera intrinsics matrix.
            extrinsics (np.ndarray): 4x4 camera extrinsics matrix.
            image_width (int): Width of the output image.
            image_height (int): Height of the output image.
        Returns:
            np.ndarray: 2D RGB image array with projected point cloud colors.
        """
        extrinsics = np.linalg.inv(extrinsics)
        # Separate xyz and rgb data
        xyz = pointcloud[:, :3]  # Nx3
        rgb = pointcloud[:, 3:6]  # Nx3
        # Transform xyz coordinates using the extrinsics to the camera coordinate system
        # Add a column of ones for homogeneous coordinates
        xyz_h = np.hstack((xyz, np.ones((xyz.shape[0], 1))))  # Nx4
        xyz_camera = (extrinsics @ xyz_h.T).T  # Nx4
        # Discard points behind the camera
        valid_points = xyz_camera[:, 2] > 0
        xyz_camera = xyz_camera[valid_points]
        rgb = rgb[valid_points]
        # Project the 3D points onto the 2D image plane using the intrinsics
        xy_image = (intrinsics @ xyz_camera[:, :3].T).T  # Nx3
        # Normalize by the z coordinate to get pixel coordinates
        xy_image /= xy_image[:, 2:3]
        # Round to nearest integer pixel coordinates
        x_pixels = np.round(xy_image[:, 0]).astype(int)
        y_pixels = np.round(xy_image[:, 1]).astype(int)
        # Initialize an empty image
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        # Filter points that fall within the image bounds
        valid_indices = (x_pixels >= 0) & (x_pixels < image_width) & (y_pixels >= 0) & (y_pixels < image_height)
        x_pixels = x_pixels[valid_indices]
        y_pixels = y_pixels[valid_indices]
        rgb = rgb[valid_indices]
        # Map rgb values to image pixels
        image[y_pixels, x_pixels] = rgb.astype(np.uint8)
        return image
        
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
    
    def get_whole_pc_in_base_frame(self, get_inhand=False):
        clouds = []
        for cam in self.fixed_cam_names:
            cloud = self.get_pointcloud_in_cam_frame(cam)
            cloud = self.transform_cloud_to_base(cloud, cam)
            clouds.append(cloud)

        cloud_inhand = None
        if get_inhand and self.use_inhand:
            self.update_inhand_extrinsic()
            cloud_inhand = self.get_pointcloud_in_cam_frame(self.inhand_cam_name)
            cloud_inhand = self.transform_cloud_to_base(cloud_inhand, self.inhand_cam_name)
            clouds.append(cloud_inhand)

        cloud = np.concatenate(clouds, axis=0)
        return cloud, cloud_inhand
    
    def clip_pc(self, cloud):
        # filter ws x
        x_cond = (cloud[:, 0] < self.center[0] + self.x_half) * (cloud[:, 0] > self.center[0] - self.x_half)
        cloud = cloud[x_cond]
        # filter ws y
        y_cond = (cloud[:, 1] < self.center[1] + self.y_half) * (cloud[:, 1] > self.center[1] - self.y_half)
        cloud = cloud[y_cond]
        # filter ws z
        z_cond = (cloud[:, 2] < self.center[2].max()) * (cloud[:, 2] > self.z_min)
        cloud = cloud[z_cond]
        return cloud

    def get_workspace_pc(self, clip_pc_size=True, get_inhand=False):
        self.clear_cache()
        cloud, cloud_inhand = self.get_whole_pc_in_base_frame(get_inhand)
        if clip_pc_size:
            cloud = self.clip_pc(cloud)
            if get_inhand:
                cloud_inhand = self.clip_pc(cloud_inhand)


        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cloud[:,:3])
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=3.5)
        cloud = cloud[ind]

        return cloud, cloud_inhand
    
    def update_inhand_extrinsic(self):
        cam = self.inhand_cam_name
        self.camera_extrinsics[cam] = None # clear cache
        extrinsic = self.get_cam_extrinsic(self.inhand_cam_name, self.base_frame)
        self.camera_extrinsics[cam] = extrinsic
    
    
    
    def get_env_screenshot(self, file_name=None, clip_pc_size=True, save_point_cloud=True):
        for _ in range(2):
            self.clear_cache()
            
        screenshot = dict()
        # get workspace meta
        screenshot['workspace_size'] = self.workspace
        screenshot['cam_names'] = self.cam_names
        screenshot['enable_inhand'] = self.use_inhand
        screenshot['base_frame'] = self.base_frame
        # get camera meta
        extrinsic_dict = self.get_all_cam_extrinsic()
        intrinsic_dict = self.get_all_cam_intrinsic()
        screenshot['extrinsic'] = extrinsic_dict
        screenshot['intrinsic'] = intrinsic_dict
        # get multiview images
        rgb_dict = dict()
        depth_dict = dict()
        
        for cam in self.cam_names:
            print(f'getting images from {cam}')
            rgb_dict[cam] = self.get_rgb_image(cam)
            depth_dict[cam] = self.get_depth_image(cam)
        screenshot['rgbs'] = rgb_dict
        screenshot['depths'] = depth_dict

        # get workspace pointcloud
        pc_dict = dict()
        if save_point_cloud:
            pc_dict['fix'], pc_dict['inhand'] = self.get_workspace_pc(clip_pc_size=clip_pc_size, get_inhand=False)
            mask_rgb_dict = dict()
            mask_dict = dict()
            for cam in self.cam_names:
                # mask_rgb[cam] = self.project_pointcloud_to_image(pc_inhand, intrinsic_dict[cam], extrinsic_dict[cam], IMAGE_WIDTH, IMAGE_HEIGHT)
                cloud_type = 'inhand' if cam == self.inhand_cam_name else 'fix'
                mask_rgb_dict[cam], mask_dict[cam] = self.mask_out_photo(rgb_dict[cam], pc_dict[cloud_type][:,:3], intrinsic_dict[cam], extrinsic_dict[cam])
                
            
            screenshot['point_cloud'] = pc_dict['fix']
            screenshot['point_cloud_inhand'] = pc_dict['inhand']
            screenshot['mask_rgb'] = mask_rgb_dict
            screenshot['mask'] = mask_dict


        if file_name is not None:
            # save
            np.save(file_name, screenshot)
        
    
def manual_tim_collection(args=None):
    rclpy.init(args=args)
    workspace_size = np.array([[0.3, 0.7],
                        [-0.2, 0.2],
                        [-0.02, 1.0]]) 
    cloud_proxy = CloudProxy(workspace_size=workspace_size, use_inhand=True, base_frame='workspace_link')
    image_index = 0
    try:
        while True:
            # cloud_proxy.get_env_screenshot()
            cloud_proxy.get_env_screenshot(f'tim_1111_{image_index}.npy')
            image_index += 1
            a = input("press enter to continue and e to stop")
            if a == 'e':
                break
    except KeyboardInterrupt:
        pass
    finally:
        # this is needed because of ROS2 mechanism.
        # without destroy_node(), it somehow wont work if you restart the program
        cloud_proxy.destroy_node()
        rclpy.shutdown()

def get_screenshot(args=None,):
    rclpy.init(args=args)
    workspace_size = np.array([[0.3, 0.7],
                        [-0.2, 0.2],
                        [-0.02, 1.0]]) 
    cloud_proxy = CloudProxy(workspace_size=workspace_size, use_inhand=False)
    try:
        cloud_proxy.get_env_screenshot('skye_bowl_mug_grasped_final.npy')#'1109.npy'
        print(1)
    except KeyboardInterrupt:
        pass
    finally:
        # this is needed because of ROS2 mechanism.
        # without destroy_node(), it somehow wont work if you restart the program
        cloud_proxy.destroy_node()
        rclpy.shutdown()
             
if __name__ == '__main__':
    get_screenshot()
    #manual_tim_collection()
