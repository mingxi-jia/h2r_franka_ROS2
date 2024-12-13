import rclpy
from rclpy.executors import ExternalShutdownException
import argparse
import sys
import time
import os
import threading

import yaml
import numpy as np
from PIL import Image
from pynput import keyboard
from pynput.keyboard import Key

from sensing_utils.cloud_sychronizer import CloudSynchronizer

from agents.gem import GEM

class PickPlaceActor():
    def __init__(self) -> None:
        print("initializing cloud proxy")
        self.cloud_synchronizer = CloudSynchronizer()
        self.cloud_thread = threading.Thread(target=self.spin_cloud_proxy, daemon=True)
        self.cloud_thread.start()

        self.cloud_synchronizer.send_homing_goal()

        task_name = input("Input task name:")

        # initialize dataset
        self.index = 0
        root_path = "/home/mingxi/code/datasets"
        self.dataset_path = os.path.join(root_path, task_name, time.strftime("%Y_%m_%d_%H_%M"))
        self.episode_path = os.path.join(self.dataset_path, f"demo_{self.index}")
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.episode_path, exist_ok=True)

        self.intrinsics = self.cloud_synchronizer.get_all_camera_intrinsics()
        self.extrinsics = self.cloud_synchronizer.get_all_camera_extrinsics()
        
    
    def spin_cloud_proxy(self):
        rclpy.spin(self.cloud_synchronizer)

    def grasp_listener(self):
        def on_press(key):
            try:
                if key.char == '0':  
                    if self.cloud_synchronizer.grasping:
                        self.cloud_synchronizer.send_homing_goal()
                        self.cloud_synchronizer.grasping = False
                    else:
                        self.cloud_synchronizer.send_grasp_goal()
                        self.cloud_synchronizer.grasping = True
                    time.sleep(0.5)

            except AttributeError:
                pass
            except Exception as e:
                print(f"Unhandled error in keyboard listener: {e}")

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        
    def get_observation(self):
        self.cloud_synchronizer.clear_cache()
        time.sleep(1.0)
        raw_multiview_rgbs = self.cloud_synchronizer.get_raw_rgbs()
        raw_multiview_depths = self.cloud_synchronizer.get_raw_depths()
        return raw_multiview_rgbs, raw_multiview_depths
                    
                    
        
    def start_acting_loop(self):
        """
        Main loop to manage the pick place process.
        """
        while True:
            self.robot.reset()
            pick_obj = input("Type in pick object:")
            place_obj = input("Type in place object:")
            instruction = {'pick_obj': pick_obj, 'place_obj': place_obj}
            rgbs, depths = self.get_observation()
            xyr_actions = self.agent.predict(rgbs, depths, instruction, self.intrinsics, self.extrinsics)
            self.robot.pick(xyr_actions['pick'])
            time.sleep(3.0)
            self.robot.place(xyr_actions['place'])
            time.sleep(3.0)
            
    def print_dataset_info(self):
        pass

    def shutdown(self):
        self.cloud_synchronizer.destroy_node()

def main():
    rclpy.init()
    collector = PickPlaceActor()
    try:
        collector.start_acting_loop()
    except (KeyboardInterrupt, ExternalShutdownException):
        collector.print_dataset_info()
    finally:
        collector.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()