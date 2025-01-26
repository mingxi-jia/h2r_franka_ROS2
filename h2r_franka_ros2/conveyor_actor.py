import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
import argparse
import sys
import time
import os
import threading
import copy
from scipy.spatial.transform import Rotation as R

import yaml
import numpy as np
from PIL import Image
from pynput import keyboard
from pynput.keyboard import Key

from sensing_utils.cloud_sychronizer import CloudSynchronizer
from panda_utils.panda_arm import ArmControl, DummyRobot
from panda_utils.configs import PICKPLACE_JOINT_HOME

# from agents.gem import GEM
from LEPP.gem_agent import GEMAgent
from LEPP.cliport_agent import CLIPortAgent
from agents.openvla import OpenVLAAgent
from agents.heuristic_agent import HeuristicAgent
            

def is_none_in_dict(data_dict: dict):
    if data_dict is None:
        return True
    for k, v in data_dict.items():
        if v is None:
            return True
    return False

def convert_ee_pose_to_xyr(ee_pose):
    x, y, z = ee_pose['xyz_RT_base']
    qxqyqzqw = ee_pose['qxqyqzqw_RT_base']
    _, _, rz = R.from_quat(qxqyqzqw).as_euler('XYZ')
    return np.array([x, y, rz])

class PickPlaceActor():
    def __init__(self, experiment_folder, agent_type='gem', enable_robot=True) -> None:
        assert os.path.exists(experiment_folder), f"{experiment_folder} does not exist."

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
        self.ee_pose = None

        # self.intrinsics = self.cloud_synchronizer.get_all_camera_intrinsics()
        # self.extrinsics = self.cloud_synchronizer.get_all_camera_extrinsics()

        # initialize agent
        if agent_type == 'gem':
            self.expert = GEMAgent(experiment_folder, self.cloud_synchronizer.camera_intrinsics, self.cloud_synchronizer.camera_extrinsics)
        elif agent_type == 'human':
            self.expert = None

        print(f"using {agent_type} as collector.")
        self.agent_type = agent_type
        self.agent = HeuristicAgent(self.cloud_synchronizer.camera_intrinsics, self.cloud_synchronizer.camera_extrinsics)


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
            action_type = input(f"1. Collect demo by {self.agent_type} | 2. Perform recorded action. \nType in the index here:")
            pick_obj = input("Type in pick object:")
            place_obj = input("Type in place object:")
            rgbs, depths = self.get_observation()
            instruction = {'instruction': f'pick {pick_obj} and place into {place_obj}', 'pick_obj': pick_obj, 'place_obj': 'None'}
            observation_dict = {'rgbs': rgbs, "depths": depths, "instruction": instruction}
            
            action_z = 0.277 if 'mug' in place_obj else 0.23

            if action_type == "1":
                print(f"AGENT: collecting demo for picking {pick_obj}")
                
                if self.agent_type == 'gem':
                    xyr_actions, _ = self.expert.pickplace(observation_dict)
                    self.robot.pick(*xyr_actions['pick'], z=action_z)
                    self.robot.place_box(place_obj, z=action_z)
                elif self.agent_type == 'human':
                    print("Starting the data collection loop. Please swtich to programming on robot pc.")
                    input("MOVE THE ROBOT AND PRESS ',' to close gripper")
                    ee_pose = self.cloud_synchronizer.get_ee_pose()
                    print(f"the current ee pose is {self.ee_pose}")
                    self.cloud_synchronizer.send_grasp_goal()

                    input("MOVE THE ROBOT AND PRESS ',' to open gripper")
                    self.cloud_synchronizer.send_homing_goal()

                    xyr_actions = dict()
                    xyr_actions['pick'] = convert_ee_pose_to_xyr(ee_pose)
                    xyr_actions['place'] = None
                    input("Please swtich to execution on robot pc. type y to confirm.")

                is_store = input("Do you want to save this demo (y or n):")
                if is_store == 'y':
                    self.agent.record_demo(observation_dict, pick_obj, xyr_actions['pick'])
                    print("demos is recorded")
                else:
                    print("demos skipped")

            elif action_type == "2":
                print(f"AGENT: Perform recorded action for {pick_obj}")
                xyr_actions = self.agent.pickplace(observation_dict)
                self.robot.pick(*xyr_actions['pick'], z=action_z)
                self.robot.place_box(place_obj, z=action_z)

            observation_dict = None

            

            
    def print_dataset_info(self):
        pass

    def shutdown(self):
        self.executor.shutdown()
        self.cloud_synchronizer.destroy_node()
        self.robot.destroy_node()

def get_agent_type(experiment_folder: str):
    if experiment_folder.split('/')[-1].startswith('LEPP'):
        agent_type = 'gem'
    elif experiment_folder.split('/')[-1].startswith('openvla'): 
        agent_type = 'openvla'
    elif experiment_folder.split('/')[-1].startswith('cliport'): 
        agent_type = 'cliport'
    else:
        NotImplementedError
    print(f"AGENT: Currently using {agent_type}")
    return agent_type

def main():
    rclpy.init()
    experiment_folder = "/home/mingxi/code/Shivam/LEPP/exps/pick-part-in-box-real-n-train109/LEPP-unetl-score-vit-postLinearMul-3-unetl-eunet-ParTrue-TopdownFalse-CropTrue-Ratio0.2-Vlmnormal-augTrue"
    # experiment_folder = "/home/mingxi/code/gem/LEPP/exps/pick-part-in-box-real-n-train16/cliport-unetl-3-none-none-ParTrue-TopdownFalse-CropTrue-Ratio0.2-Vlmnormal-augTrue"
    agent_type = get_agent_type(experiment_folder)
    
    actor = PickPlaceActor(experiment_folder, agent_type=agent_type)
    #actor = PickPlaceActor(experiment_folder, agent_type="human")
    try:
        actor.start_acting_loop()
    except (KeyboardInterrupt, ExternalShutdownException):
        actor.print_dataset_info()
    finally:
        actor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()