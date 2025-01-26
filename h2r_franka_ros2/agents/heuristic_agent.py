import os
import numpy as np
from PIL import Image
import yaml

from lepp.clip_utils import get_cloud_from_depth
from lepp.vision_utils import cloud_preprocess, visualize_pointcloud
from lepp.configs import REAL_WORKSPACE_SIZE, REAL_TOPDOWN_RGBD_WORKSPACE, FEATURE_Z_MIN

def load_test_images(folder_path):
    raw_multiview_rgbs = dict()
    raw_multiview_depths = dict()
    cam = 'kevin'

    rgb_path = os.path.join(folder_path, f"rgb_{cam}.png")
    rgb = np.array(Image.open(rgb_path))
    raw_multiview_rgbs[cam] = rgb
    
    depth_path = os.path.join(folder_path, f"depth_{cam}_raw.npy")
    depth = np.load(depth_path)
    raw_multiview_depths[cam] = depth

    return raw_multiview_rgbs, raw_multiview_depths

def get_point_cloud(rgbs: dict, depths: dict, intrinsics: dict, extrinsics: dict):

    intrinsic = np.array(intrinsics).reshape(3,3)
    extrinsic = np.array(extrinsics).reshape(4,4)
    rgb, depth = rgbs, depths
    cloud = get_cloud_from_depth(rgb, depth, intrinsic, extrinsic)
    cloud = cloud_preprocess(cloud, REAL_WORKSPACE_SIZE)

    return cloud

def load_yaml(yaml_path: str):
    with open(yaml_path) as file:
        data = yaml.safe_load(file)
    return data

def save_skill_yaml(skill_dict):
    with open('skills.yaml', 'w') as yaml_file:
        yaml.dump(skill_dict, yaml_file, default_flow_style=False)


class HeuristicAgent():
    def __init__(self, intrinsics=None, extrinsics=None):
        self.skills_in_object_frame = dict()

        if intrinsics is None:
            print("AGENT: loading existing intrinsics")
            self.intrinsics = load_yaml(os.path.join(experiment_folder, 'intrinsics.yaml'))  
        else:
            print("AGENT: loading extrinsics ros topics")
            self.intrinsics = intrinsics
        if extrinsics is None:
            print("AGENT: loading existing extrinsics")
            self.extrinsics = load_yaml(os.path.join(experiment_folder, 'extrinsics.yaml'))
        else:
            print("AGENT: loading extrinsics from tf tree")
            self.extrinsics = extrinsics

    def get_topdown_centroid(self, observation_dict):
        cam = 'kevin'
        multiview_rgbs, multiview_depths = observation_dict['rgbs'], observation_dict['depths']
        topdown_cloud = get_point_cloud(multiview_rgbs[cam], multiview_depths[cam], self.intrinsics[cam], self.extrinsics[cam])
        centroid = np.mean(topdown_cloud[:,:2], axis=0)
        return centroid

    def record_demo(self, observation_dict, object_name, xyr: np.array):
        centroid_xy = self.get_topdown_centroid(observation_dict)
        self.skills_in_object_frame[object_name] = (xyr - np.array([centroid_xy[0], centroid_xy[1], 0.])).tolist()
        save_skill_yaml(self.skills_in_object_frame)

    def pickplace(self, observation_dict):
        instruction = observation_dict['instruction']
        
        pick_goal = instruction['pick_obj']
        centroid_xy = self.get_topdown_centroid(observation_dict)

        if pick_goal in self.skills_in_object_frame.keys():
            local_action = np.array(self.skills_in_object_frame[pick_goal])
            pick_action =  local_action + np.array([centroid_xy[0], centroid_xy[1], 0.])
        else:
            print("no skill found. Going to pick object's centroid.")
            pick_action = np.array([centroid_xy[0], centroid_xy[1], np.pi/2])

        return {
                'pick': pick_action,
                'place': None,
                }

if __name__ == "__main__":
    experiment_folder = "/home/mingxi/code/Shivam/LEPP/exps/pick-part-in-box-real-n-train32/LEPP-unetl-score-vit-postLinearMul-3-unetl-eunet-ParTrue-TopdownFalse-CropTrue-Ratio0.2-Vlmnormal-augTrue"
    agent = HeuristicAgent()

    multiview_rgbs, multiview_depths = load_test_images('/home/mingxi/code/raw_datasets/convoyer_object14/2025_01_24_00_01/demo_0')
    instruction = {'instruction': 'pick red mug and place into brown box', 'pick_obj': 'red mug', 'place_obj': 'brown box'}

    observation_dict = dict()
    observation_dict['rgbs'], observation_dict['depths'], observation_dict['instruction'] =  multiview_rgbs, multiview_depths, instruction
    action = agent.pickplace(observation_dict)
    print(action)