import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
import argparse
import sys
import time
import os
import threading
import copy

import yaml
import numpy as np
from PIL import Image
from pynput import keyboard
from pynput.keyboard import Key

from sensing_utils.cloud_sychronizer import CloudSynchronizer
from panda_utils.panda_arm import ArmControl, DummyRobot
from panda_utils.configs import PICKPLACE_JOINT_HOME

# from agents.gem import GEM
from LEPP.agent import GEMAgent
from agents.openvla import OpenVLAAgent
            

def is_none_in_dict(data_dict: dict):
    if data_dict is None:
        return True
    for k, v in data_dict.items():
        if v is None:
            return True
    return False


class PickPlaceActor():
    def __init__(self, experiment_folder, agent_type='gem', enable_robot=True) -> None:
        self.executor = MultiThreadedExecutor()

        print("initializing cloud proxy")
        self.cloud_synchronizer = CloudSynchronizer('gem')
        self.executor.add_node(self.cloud_synchronizer)

        print("initializing robot")
        if enable_robot:
            self.robot = ArmControl(joint_home=PICKPLACE_JOINT_HOME)
            self.executor.add_node(self.robot)
        else:
            # for debugging
            self.robot = DummyRobot()

        self.executor_thread = threading.Thread(target=self.run_executor, daemon=True)
        self.executor_thread.start()

        # self.cloud_synchronizer.send_homing_goal()
        self.robot.open_gripper()

        self.intrinsics = self.cloud_synchronizer.get_all_camera_intrinsics()
        self.extrinsics = self.cloud_synchronizer.get_all_camera_extrinsics()

        # initialize agent
        if agent_type == 'gem':
            self.agent = GEMAgent(experiment_folder)
        elif agent_type == 'openvla':
            self.agent = OpenVLAAgent(experiment_folder, is_test=True)

        # self.raw_multiview_rgbs = None
        # self.raw_multiview_depths = None

    def run_executor(self):
        self.executor.spin()
        
    def get_observation(self):
        self.cloud_synchronizer.clear_cache()
        raw_multiview_rgbs, raw_multiview_depths = None, None
        time.sleep(1.0)
        while is_none_in_dict(raw_multiview_rgbs) or is_none_in_dict(raw_multiview_depths):
            ee_pose = self.cloud_synchronizer.get_ee_pose()
            raw_multiview_rgbs = self.cloud_synchronizer.get_raw_rgbs()
            raw_multiview_depths = self.cloud_synchronizer.get_raw_depths()
            time.sleep(1.0)
            print("waiting for obs")
        multiview_rgbs, multiview_depths = copy.copy(raw_multiview_rgbs), copy.copy(raw_multiview_depths)
        return multiview_rgbs, multiview_depths
                    
 
    def start_acting_loop(self):
        """
        Main loop to manage the pick place process.
        """
        while True:
            self.robot.reset()
            sys.stdin.flush() # clean keyboard buffer
            pick_obj = input("Type in pick object:")
            place_obj = input("Type in place object:")
            instruction = {'instruction': f'pick {pick_obj} and place into {place_obj}', 'pick_obj': pick_obj, 'place_obj': place_obj}
            
            rgbs, depths = self.get_observation()
            observation_dict = {'rgbs': rgbs, "depths":depths, "instruction":instruction}
            xyr_actions = self.agent.pickplace(observation_dict)
            if xyr_actions is not None:
                self.robot.pick(*xyr_actions['pick'])
                self.robot.place(*xyr_actions['place'])
            else:
                print("robot action is invalid")

            # self.robot.pick(0.45, 0.2, np.pi/2)
            # self.robot.place(0.45, -0.2, 0.)
            
    def print_dataset_info(self):
        pass

    def shutdown(self):
        self.executor.shutdown()
        self.cloud_synchronizer.destroy_node()
        self.robot.destroy_node()

def main():
    rclpy.init()
    experiment_folder = "/home/mingxi/code/gem/LEPP/exps/LEPP-unetl-score-vit-postLinearMul-3-unetl-eunet-ParTrue-TopdownFalse-CropTrue-Ratio0.2-Vlmnormal-augTrue"
    # experiment_folder = "/home/mingxi/code/gem/openVLA/logs/openvla-7b+openloop_pick_place_dataset+b2+lr-0.0005+lora-r32+dropout-0.0"
    agent_type = 'gem'
    actor = PickPlaceActor(experiment_folder, agent_type)
    try:
        actor.start_acting_loop()
    except (KeyboardInterrupt, ExternalShutdownException):
        actor.print_dataset_info()
    finally:
        actor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()