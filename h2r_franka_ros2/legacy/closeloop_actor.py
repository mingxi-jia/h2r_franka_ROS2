import sys
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from pynput import keyboard

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_srvs.srv import Trigger
from moveit_msgs.action import MoveGroup
from sensor_msgs.msg import Image, JointState, Joy
from geometry_msgs.msg import PoseStamped, TwistStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, RobotState, JointConstraint

from panda_utils.configs import JOINT_HOME
from h2r_franka_ros2.legacy.execution import Executor

from agents.openvla import OpenVLAAgent

class Actor():
    def __init__(self, camera_name):

        self.contron_freq = 2 # according to openVLA doc 6 is maximum for 4090

        self.robot = Executor(camera_name)
        self.executor_thread = threading.Thread(target=self.spin_executor, daemon=True)
        self.executor_thread.start()

        self.acting = False
        experiment_folder = "/home/mingxi/code/gem/openVLA/logs/openvla-7b+franka_pick_place_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug"
        self.agent = OpenVLAAgent(experiment_folder, is_test=True)
        self.instruction = input("what is the instruction for this episode: ")
        
        # Thread for keyboard input
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        self.start_new_episode(proceed=False)

    def spin_executor(self):
        rclpy.spin(self.robot)

    def keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'g':  # next episode
                    self.acting = False
                    print("g key pressed. Goal is triggered. Reset..")
                    self.start_new_episode(proceed=False)
                elif key.char == 'b':  # save and continue
                    self.acting = False
                    print("b key pressed. Bad test. Reset..")
                    self.start_new_episode(proceed=True)
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def act(self, rgb, depth, current_pose, franka_joint):
        current_pose_R = R.from_quat(current_pose[3:]).as_matrix()
        
        # Example action: send a movement command based on pose and twist action
        action_rel = self.agent.act(rgb, self.instruction)
        goal_xyz = current_pose[:3] + action_rel[:3]
        print(action_rel)
        goal_rotation = R.from_euler('XYZ', action_rel[3:6]).as_matrix() @ current_pose_R
        goal_pose_quat = R.from_matrix(goal_rotation).as_quat()

        self.robot.move_robot(goal_xyz, goal_pose_quat, action_rel[6])
        time.sleep(1/self.contron_freq)

    def start_new_episode(self, proceed=True):
        """Starts a new episode after confirming the robot reset."""
        
        if proceed:
            pass
        self.robot.open_gripper()
        self.robot.robot_reset_by_JOINT()
        # 
        self.acting = True
        # Simulating robot reset and starting a new episode

    def action_loop(self):
        while True:
            if self.acting:
                sensor_dict = self.robot.get_sensor_input()
                if sensor_dict is None:
                    print("Actor receives no sensor input. Waiting.")
                    continue
                print("Actor receives sensor input.")
                rgb = sensor_dict['rgb']
                depth = sensor_dict['depth']
                current_pose = sensor_dict['current_pose']
                franka_joint = sensor_dict['franka_joint']
                self.act(rgb, depth, current_pose, franka_joint)
                self.robot.clear_cache()

def main(args=None):
    rclpy.init(args=args)
    camera_name = 'dave'
    actor = Actor(camera_name)
    actor.action_loop()

if __name__ == '__main__':
    main()
