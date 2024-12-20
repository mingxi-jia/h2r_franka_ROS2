import sys
import copy
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from pynput import keyboard

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from std_srvs.srv import Trigger
from moveit_msgs.action import MoveGroup
from sensor_msgs.msg import Image, JointState, Joy
from geometry_msgs.msg import PoseStamped, TwistStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, RobotState, JointConstraint

from panda_utils.configs import JOINT_HOME
from panda_utils.panda_arm import ArmControl
from sensing_utils.cloud_sychronizer import CloudSynchronizer


def is_none_in_dict(data_dict: dict):
    if data_dict is None:
        return True
    for k, v in data_dict.items():
        if v is None:
            return True
    return False

class Actor():
    def __init__(self, camera_name='dave', experiment_folder=None):
        self.executor = MultiThreadedExecutor()

        print("initializing cloud proxy")
        self.cloud_synchronizer = CloudSynchronizer('closeloop')
        self.executor.add_node(self.cloud_synchronizer)

        self.contron_freq = 2 # according to openVLA doc 6 is maximum for 4090
        self.robot = ArmControl(JOINT_HOME)
        self.executor.add_node(self.robot)

        self.executor_thread = threading.Thread(target=self.run_executor, daemon=True)
        self.executor_thread.start()

        self.acting = False
        
        if experiment_folder is not None:
            from agents.openvla import OpenVLAAgent
            self.agent = OpenVLAAgent(experiment_folder, is_test=True)
            self.instruction = input("what is the instruction for this episode: ")
        else:
            print("No model loaded. Testing mode.")
        
        # Thread for keyboard input
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        self.camera_name = camera_name

        self.start_new_episode()
    
    def run_executor(self):
        self.executor.spin()

    def keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'g':  # next episode
                    self.acting = False
                    print("g key pressed. Goal is triggered. Reset..")
                    self.start_new_episode()
                elif key.char == 'b':  # save and continue
                    self.acting = False
                    print("b key pressed. Bad test. Reset..")
                    self.start_new_episode()
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def move_robot_by_relative_action(self, current_pose, action_rel):
        x, y, z = current_pose[:3] + action_rel[:3]
        
        current_pose_R = R.from_quat(current_pose[3:]).as_matrix()
        goal_rotation = R.from_euler('XYZ', action_rel[3:6]).as_matrix() @ current_pose_R
        goal_pose_quat = R.from_matrix(goal_rotation).as_quat()
        
        self.robot.goto(x, y, z, goal_pose_quat)

        gripper_action = int(action_rel[6])
        if gripper_action != self.gripper_state:
            self.gripper_state = gripper_action
            if gripper_action == 0:
                self.robot.close_gripper()
            else:
                self.robot.open_gripper()
        

    def start_new_episode(self):
        """Starts a new episode after confirming the robot reset."""
        self.gripper_state = 1
        self.robot.open_gripper()
        self.robot.robot_reset_by_JOINT()
        # 
        self.acting = True
        # Simulating robot reset and starting a new episode

    def get_observation(self):
        self.cloud_synchronizer.clear_cache()
        raw_multiview_rgbs, raw_multiview_depths, ee_pose = None, None, None
        time.sleep(1.0)
        while is_none_in_dict(raw_multiview_rgbs) or is_none_in_dict(raw_multiview_depths) or ee_pose is None:
            ee_pose = self.cloud_synchronizer.get_ee_pose()
            raw_multiview_rgbs = self.cloud_synchronizer.get_raw_rgbs()
            raw_multiview_depths = self.cloud_synchronizer.get_raw_depths()
            time.sleep(1.0)
            print("waiting for obs")
        multiview_rgbs, multiview_depths = copy.copy(raw_multiview_rgbs), copy.copy(raw_multiview_depths)
        ee_pose = copy.copy(ee_pose)
        ee_pose = ee_pose['xyz_RT_base'] + ee_pose['qxqyqzqw_RT_base']
        sensor_dict = {'rgbs': multiview_rgbs, 
                       'depth': multiview_depths,
                       'ee_pose': ee_pose,}
        return sensor_dict
    
    def action_loop(self):
        while True:
            if self.acting:
                sensor_dict = self.get_observation()
                print("Actor receives sensor input.")
                rgb = sensor_dict['rgbs'][self.camera_name]
                depth = sensor_dict['depths'][self.camera_name]
                current_pose = sensor_dict['ee_pose']
                action_rel = self.agent.act(rgb, self.instruction)
                self.move_robot_by_relative_action(current_pose, action_rel)

    def test_action_loop(self):
        self.robot.reset()
        example_data = np.load("/home/mingxi/code/datasets/subsampled_demo0.npy", allow_pickle=True)
        for step in  example_data:
            action_rel = step['action']
            print(action_rel)
            sensor_dict = self.get_observation()
            current_pose = sensor_dict['ee_pose']
            self.move_robot_by_relative_action(current_pose, action_rel)


def main(args=None):
    rclpy.init(args=args)
    experiment_folder = "/home/mingxi/code/gem/openVLA/logs/openvla-7b+franka_pick_place_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    actor = Actor('dave')
    # actor.action_loop()
    actor.test_action_loop()

if __name__ == '__main__':
    main()
