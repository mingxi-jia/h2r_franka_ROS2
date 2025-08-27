#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time


import tf2_ros
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped, TransformStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from franka_msgs.action import Grasp, Homing, Move
from franka_msgs.msg import GraspEpsilon
from rclpy.action import ActionClient

import copy
import cv2
import time
import open3d 
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial.transform import Rotation as R

from gem_camera_configs import TOPICS_DEPTH, TOPICS_RGB, TOPICS_CAM_INFO, RGB_FRAMES, BASE_FRAME, INHAND_CAMERA_NAME
from closeloop_camera_configs import TOPICS_DEPTH as CLOSE_TOPICS_DEPTH, TOPICS_RGB as CLOSE_TOPICS_RGB, \
    TOPICS_CAM_INFO as CLOSE_TOPICS_CAM_INFO, RGB_FRAMES as CLOSE_RGB_FRAMES, BASE_FRAME as CLOSE_BASE_FRAME

CONFIG_MAP = {
            'gem': {
                'TOPICS_DEPTH': TOPICS_DEPTH,
                'TOPICS_RGB': TOPICS_RGB,
                'TOPICS_CAM_INFO': TOPICS_CAM_INFO,
                'RGB_FRAMES': RGB_FRAMES,
                'BASE_FRAME': BASE_FRAME,
                'collect_freq': 1.5
            },
            'closeloop': {
                'TOPICS_DEPTH': CLOSE_TOPICS_DEPTH,
                'TOPICS_RGB': CLOSE_TOPICS_RGB,
                'TOPICS_CAM_INFO': CLOSE_TOPICS_CAM_INFO,
                'RGB_FRAMES': CLOSE_RGB_FRAMES,
                'BASE_FRAME': CLOSE_BASE_FRAME,
                'collect_freq': 5
            }
        }

class CloudSynchronizer(Node):
    def __init__(self, configuration='gem'):
        super().__init__('cloud_synchronizer')

        # cofigure 
        if configuration in CONFIG_MAP.keys():
            config = CONFIG_MAP[configuration]
        else:
            NotImplementedError
        
        self.TOPICS_DEPTH = config['TOPICS_DEPTH']
        self.TOPICS_RGB = config['TOPICS_RGB']
        self.TOPICS_CAM_INFO = config['TOPICS_CAM_INFO']
        self.RGB_FRAMES = config['RGB_FRAMES']
        self.BASE_FRAME = config['BASE_FRAME']
        self.INHAND_CAMERA_NAME = INHAND_CAMERA_NAME
        self.collect_freq = config['collect_freq']

        # setup cloud
        self.time_diff = 1. / self.collect_freq

        self.ee_pose = None
        self.camera_intrinsics = dict()
        self.camera_extrinsics = dict()
        self.rgb_dict = dict()
        self.depth_dict = dict()

        # Define camera names
        self.camera_names = list(self.TOPICS_CAM_INFO.keys())  # Add all camera names here
        self.camera_names.sort()

        # Synchronizer
        self.rgb_subs = []
        self.depth_subs = []
        for camera_name in self.camera_names:
            rgb_sub = Subscriber(self, Image, self.TOPICS_RGB[camera_name])
            depth_sub = Subscriber(self, Image, self.TOPICS_DEPTH[camera_name])
            self.rgb_subs.append(rgb_sub)
            self.depth_subs.append(depth_sub)

            info_topic = self.TOPICS_CAM_INFO[camera_name]
            self.camera_intrinsics[camera_name] = None
            self.camera_extrinsics[camera_name] = None
            self.rgb_dict[camera_name] = None
            self.depth_dict[camera_name] = None
            self.create_subscription(CameraInfo, info_topic, lambda msg, cam=camera_name: self.info_callback(msg, cam), 10)

        self.pose_sub = Subscriber(self, PoseStamped, '/franka_robot_state_broadcaster/current_pose')
        self.get_logger().info("All data subscriber ready.")

        all_subscribers = self.rgb_subs + self.depth_subs + [self.pose_sub]
        self.ts = ApproximateTimeSynchronizer(
            all_subscribers,
            queue_size=20,          
            slop=self.time_diff
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info("Data Synchronizer ready.")
        
        # Transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread = True)

        self.bridge = CvBridge()

        self.get_logger().info("Initializing camera poses.")
        self.camera_extrinsics: dict = self.get_all_camera_extrinsics()
        self.get_logger().info("Initializing camera intrinsics.")
        self.camera_intrinsics: dict = self.get_all_camera_intrinsics()
        self.get_logger().info("Camera ex/intrinsics done.")

        self.grasp_client = ActionClient(self, Grasp, '/franka_gripper/grasp')
        self.grasp_client.wait_for_server()
        self.get_logger().info(f'grasp agent ready!')

        self.homing_client = ActionClient(self, Move, '/franka_gripper/move')
        self.homing_client.wait_for_server()
        self.get_logger().info(f'homing agent ready!')

        self.speed = 0.2
        self.grasping = False

    def sync_callback(self, *args):
        num_cameras = len(self.camera_names)
        rgb_images = args[:num_cameras]
        depth_images = args[num_cameras:num_cameras * 2]
        pose_msg: PoseStamped = args[-1]
        pose = pose_msg.pose

        # Process camera data
        rgb_data, depth_data = dict(), dict()
        for i, camera_name in enumerate(self.camera_names):
            try:
                rgb_data[camera_name] = np.array(rgb_images[i].data).reshape((rgb_images[i].height, rgb_images[i].width, -1))
                depth_data[camera_name] = self.bridge.imgmsg_to_cv2(depth_images[i], "16UC1").astype(np.float32) / 1000
            except Exception as e:
                self.get_logger().warn(f"Failed to process data for camera {camera_name}: {str(e)}")

        self.rgb_dict = rgb_data
        self.depth_dict = depth_data
        self.ee_pose = {'xyz_RT_base': [pose.position.x, pose.position.y, pose.position.z],
                        'qxqyqzqw_RT_base': [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]}
        
        # self.get_logger().info("Synchronized data step updated.")

    
    # ----------------------------Basic Camera functions---------------------------
    # ----------------------------Extrinsics
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
                # transformMsg = self.tf_buffer.lookup_transform_async(toFrame, fromFrame, lookupTime, Duration(seconds=1))

                keep_trying = False
            except:
                rclpy.spin_once(self, timeout_sec=1.)
                
        translation = transformMsg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transformMsg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        
        transform = np.eye(4)
        transform[:3, 3] = np.array(pos)
        transform[:3, :3] = R.from_quat(quat).as_matrix()
        return transform
    
    def get_camera_extrinsics(self, camera_name):
        extrinsic = self.lookup_transform(self.RGB_FRAMES[camera_name], 
                                          self.BASE_FRAME
                                          )
        return extrinsic

    def get_all_camera_extrinsics(self):
        for cam in self.camera_names:
            self.get_logger().info(f"Initializing {cam} pose.")
            self.camera_extrinsics[cam] = self.get_camera_extrinsics(cam)
        return self.camera_extrinsics
    
    def update_inhand_extrinsic(self):
        cam = self.INHAND_CAMERA_NAME
        extrinsic = self.get_camera_extrinsics(self.INHAND_CAMERA_NAME, self.BASE_FRAME)
        self.camera_extrinsics[cam] = extrinsic
    
    # ----------------------------Intrinsics

    def info_callback(self, msg, camera_name):
        self.camera_intrinsics[camera_name] = np.array(msg.k).reshape(3,3)
        
    def get_cam_intrinsic(self, camera_name):
        if camera_name not in self.camera_names:
            raise NotImplementedError(f"Camera ID {camera_name} not supported")
        
        while self.camera_intrinsics[camera_name] is None:
            rclpy.spin_once(self, timeout_sec=0.01)
        intrinsics = self.camera_intrinsics[camera_name]
        return intrinsics
    
    def get_all_camera_intrinsics(self):
        for cam in self.camera_names:
            self.get_logger().info(f"Initializing {cam} intrinsics.")
            self.camera_intrinsics[cam] = self.get_cam_intrinsic(cam)
        return self.camera_intrinsics
    
    #--------------------------EE POSE
    def get_ee_pose(self):
        return self.ee_pose
    
    #-------------------------OBSERVATION
    def get_raw_rgbs(self):
        return self.rgb_dict

    def get_raw_depths(self):
        return self.depth_dict
    
    # def get_cam_extrinsic(self, cam_id, base_frame):
    #     if cam_id not in CAM_INDEX_MAP:
    #         raise NotImplementedError(f"Camera ID {cam_id} not supported")
        
    #     rgb_frame = RGB_FRAMES[cam_id]
    #     T = self.lookup_transform(rgb_frame, base_frame)
    #     print(f"got {cam_id} pose.")
    #     return T
    
    # def look_and_set_extrinsics(self, base_frame=None):
    #     if base_frame is None:
    #         base_frame = self.base_frame
    #     print(f"init: getting camera extrinsics in frame {base_frame}")
    #     camera_extrinsics = dict()
    #     for cam in self.cam_names:
    #         extrinsic = self.get_cam_extrinsic(cam, base_frame)
    #         camera_extrinsics[cam] = extrinsic
    #     return camera_extrinsics
    
    # ----------------------------Util functions---------------------------

    def clear_cache(self):
        for camera_name in self.camera_names:
            self.rgb_dict[camera_name] = None
            self.depth_dict[camera_name] = None
    
    # ----------------------Grasp functions---------------------------
    def send_grasp_goal(self, width=0.00, force=50.0):
        epsilon = GraspEpsilon()
        epsilon.inner = 0.05
        epsilon.outer = 0.05
        goal_msg = Grasp.Goal(width=width, speed=self.speed, force=force, epsilon=epsilon)
        future = self.grasp_client.send_goal_async(goal_msg)
        future.add_done_callback(self.grasp_response_callback)

    def grasp_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            # self.get_logger().info('Grasp goal rejected.')
            return
        # self.get_logger().info('Grasp goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.grasp_result_callback)

    def grasp_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Grasp result: {result}')

    def send_homing_goal(self):
        goal_msg = Move.Goal(width=0.1, speed=self.speed)
        future = self.homing_client.send_goal_async(goal_msg)
        future.add_done_callback(self.homing_response_callback)

    def homing_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            # self.get_logger().info('Homing goal rejected.')
            return
        # self.get_logger().info('Homing goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.homing_result_callback)

    def homing_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Homing result: {result}')
    
def main():
    rclpy.init()
    collector = CloudSynchronizer('gem')
    try:
        rclpy.spin(collector)
    except (KeyboardInterrupt):
        pass
    finally:
        collector.destroy_node()
        rclpy.shutdown()   
             
if __name__ == '__main__':
    main()
