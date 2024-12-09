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

from cloud_sychronizer import CloudSynchronizer


def save_npy_data(root_path, modality: str, data: np.array):
    np.save(os.path.join(root_path, f"{modality}.npy"), data)

def save_dict_data(root_path, modality: str, data: dict):
    # convert numpy array to list
    for key, value in data.items():
        if isinstance(value, (np.ndarray)):
            data[key] = value.tolist()

    yaml_file = os.path.join(root_path, f"{modality}.yaml")
    with open(yaml_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

class PickPlaceCollector():
    def __init__(self) -> None:
        print("initializing cloud proxy")
        self.cloud_synchronizer = CloudSynchronizer()
        self.cloud_thread = threading.Thread(target=self.spin_cloud_proxy, daemon=True)
        self.cloud_thread.start()

        task_name = input("Input task name:")

        # initialize dataset
        self.index = 0
        root_path = "/home/mingxi/code/datasets"
        self.dataset_path = os.path.join(root_path, task_name, time.strftime("%Y_%m_%d_%H_%M"))
        self.episode_path = os.path.join(self.dataset_path, f"demo_{self.index}")
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.episode_path, exist_ok=True)

        intrinsics = self.cloud_synchronizer.get_all_camera_intrinsics()
        save_dict_data(self.dataset_path, 'intrinsics', intrinsics)
        extrinsics = self.cloud_synchronizer.get_all_camera_extrinsics()
        save_dict_data(self.dataset_path, 'extrinsics', extrinsics)

        # Thread for keyboard input
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
    
    def spin_cloud_proxy(self):
        rclpy.spin(self.cloud_synchronizer)
        
    def keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'p':  # next episode
                    self.cloud_synchronizer.clear_cache()
                    time.sleep(1.0)
                    raw_multiview_rgbs = self.cloud_synchronizer.get_raw_rgbs()
                    raw_multiview_depths = self.cloud_synchronizer.get_raw_depths()
                    
                    success = self.save_observation(raw_multiview_rgbs, raw_multiview_depths)
                    if success:
                        pick_obj = input("input pick object name:")
                        place_obj = input("input place object name:")
                        instructions = {'instruction': f"pick {pick_obj} and place on {place_obj}",
                                        'pick_obj': pick_obj,
                                        'place_obj': place_obj}
                        save_dict_data(self.episode_path, "instruction", instructions)
                        print("obs input saved.")
                    else:
                        print("no obs input.")

                elif key.char == '1':  # save pick loc
                    print("save pick loc.")
                    ee_pose = self.cloud_synchronizer.get_ee_pose()
                    save_dict_data(self.episode_path, "pick_pose", ee_pose)
                    # self.close_gripper()
                elif key.char == '2':  # save place loc
                    print("save place loc.")
                    ee_pose = self.cloud_synchronizer.get_ee_pose()
                    save_dict_data(self.episode_path, "place_pose", ee_pose)
                    # self.open_gripper()
                elif key.char == '=':  # continue
                    valid = self.check_valid_previous_demo()
                    if valid:
                        self.index += 1
                        self.episode_path = os.path.join(self.dataset_path, f"demo_{self.index}")
                        os.makedirs(self.episode_path, exist_ok=True)
                        print("save demo.")
                    else:
                        print("The previous demo is not completes. Either lack of obs or actions. Failed to continue.")
                
            except AttributeError:
                pass
            except Exception as e:
                print(f"Unhandled error in keyboard listener: {e}")

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def check_valid_previous_demo(self):
        is_valid_images = os.path.exists(os.path.join(self.episode_path, "rgb_kevin.png"))
        is_valid_pick = os.path.exists(os.path.join(self.episode_path, "pick_pose.yaml"))
        is_valid_place = os.path.exists(os.path.join(self.episode_path, "place_pose.yaml"))
        return is_valid_images and is_valid_pick and is_valid_place

    def save_observation(self, rgbs: dict, depths: dict):
        try:
            camera_list = rgbs.keys()

            for cam in camera_list:
                rgb = Image.fromarray(rgbs[cam])
                rgb.save(os.path.join(self.episode_path, f"rgb_{cam}.png"))
                
                # store real depth
                depth = depths[cam]
                np.save(os.path.join(self.episode_path, f"depth_{cam}_raw.npy"), depth)

                # store depth visulization
                depth_min, depth_max = 0., 1.3
                depth = np.clip(depth, depth_min, depth_max)
                depth = np.asarray(depth / depth_max * 255., dtype=np.uint8)
                depth = Image.fromarray(depth)
                depth.save(os.path.join(self.episode_path, f"depth_{cam}_0to2m.png"))
            return True
        except:
            return False
        
    def start_collection_loop(self):
        """
        Main loop to manage the collection process.
        """
        print("Starting the data collection loop. Use the following keys:")
        print("'p' - Save sensor inputs (RGB, Depth) and task instructions")
        print("'1' - Save pick location")
        print("'2' - Save place location")
        print("'=' - Move to the next demo/episode")
        print("'Ctrl+C' - Exit and save dataset information")
        while True:
            time.sleep(0.5)  # Keeps the loop alive and lets the threads work
            
    def print_dataset_info(self):
        pass

    def shutdown(self):
        self.cloud_synchronizer.destroy_node()

def main():
    rclpy.init()
    collector = PickPlaceCollector()
    try:
        collector.start_collection_loop()
    except (KeyboardInterrupt, ExternalShutdownException):
        collector.print_dataset_info()
    finally:
        collector.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()