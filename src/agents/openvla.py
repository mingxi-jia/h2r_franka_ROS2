import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from openVLA.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openVLA.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openVLA.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import os
import json
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from lepp.vision_utils import getTopDownProjectImg, get_crop, convert_robot_action_to_pixel_action, transform, convert_pixel_action_to_robot_action
from lepp.real_dataset_processor import (fuse_text_and_visual_maps, de_preprocess_action, load_yaml, fuse_text_and_visual_maps, preprocess_topdown_rgbd,
                                         image_rot270)
from lepp.clip_utils import get_clip_obs

def zoom_in(rgb, percentage=0.6):
    height, width, channel = rgb.shape
    zoomed_height, zoomed_width = int(height*percentage), int(width*percentage)
    image_center = (height//2, width//2)
    return rgb[image_center[0]-zoomed_height//2:image_center[0]+zoomed_height//2, image_center[1]-zoomed_width//2:image_center[1]+zoomed_width//2]

def preprocess_rgb(rgb: np.array, is_aug: bool):
    assert rgb.dtype == np.uint8
    # crop to a square 480 480
    height, width, channel = rgb.shape
    rgb = rgb[:, width//2-height//2:width//2+height//2]
    # resize to 224x224 as used in openVLA
    rgb = zoom_in(rgb, percentage=0.45)
    rgb = cv2.resize(rgb, (224, 224), 
               interpolation = cv2.INTER_LINEAR)
    if is_aug:
        rgb = apply_center_crop(rgb)
    return rgb 

def apply_center_crop(rgb: np.array):
    original_size = rgb.shape[:2]
    crop_scale = 0.9
    sqrt_crop_scale = math.sqrt(crop_scale)
    t_h_half=int(sqrt_crop_scale * rgb.shape[0])//2
    t_w_half=int(sqrt_crop_scale * rgb.shape[1])//2
    center_h, center_w = rgb.shape[0]//2, rgb.shape[1]//2
    rgb = rgb[center_h-t_h_half:center_h+t_h_half, center_w-t_w_half:center_w+t_w_half, :]
    rgb = Image.fromarray(rgb)
    rgb = rgb.resize(original_size, Image.Resampling.BILINEAR)
    return np.array(rgb)

def construct_delta_action_from_abs_poses(pose_t: np.array, pose_t_plus_1: np.array, grasp_signal_t_plus_1: np.array):
    """
    input:
        pose_t: raw end effector pose (x y z qx qy qz qw gripper_width)
        grasp_signal_t_plus_1: grasping signal, 1 meaning grasping
    output:
        delta_action (x y z roll pitch yaw gripper_width): delta action between pose_t and pose_t_plus_1 
    """
    delta_action = np.zeros(7).astype(np.float32)

    # calculate delta xyz
    delta_xyz = pose_t_plus_1[:3] - pose_t[:3]
    delta_action[:3] = delta_xyz

    # calculate delta rpy
    r1, r2 = R.from_quat(pose_t[3:7]), R.from_quat(pose_t_plus_1[3:7])
    r1_mat, r2_mat = r1.as_matrix(), r2.as_matrix()
    rrel_mat = r2_mat @ r1_mat.T
    rrel_euler = R.from_matrix(rrel_mat).as_euler('XYZ')
    delta_action[3:6] = rrel_euler
    # flip grasping signal to align with openvla definitions: +1=open and 0=close
    delta_action[6] = np.asarray(1 - grasp_signal_t_plus_1, dtype=np.float32)
    return delta_action

def preprocess_openvla_obs(multiview_rgbs: dict, multiview_depths: dict, intrinsics: dict, extrinsics: dict):
    multiview_tfeatures_picking = dict()
    for k, v in multiview_depths.items():
        multiview_tfeatures_picking[k] = np.ones_like(v)
    topdown_rgb, _, _ = get_clip_obs(multiview_rgbs, multiview_depths, multiview_tfeatures_picking, intrinsics, extrinsics)
    # organize topdown image and actions to robot view point
    topdown_rgb = image_rot270(topdown_rgb)

    return topdown_rgb

class OpenVLAAgent:
    def __init__(self, experiment_folder: str, intrinsics: np.array, extrinsics: np.array) -> None:
        print("AGENT: initializing OpenVLA agent")

        self.height, self.width, self.channel = 224, 224, 3
        # Register OpenVLA model to HF AutoClasses (not needed if you pushed model to HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # experiment_folder = "/home/mingxi/mingxi_ws/gem/openvla/logs/openvla-7b+franka_pick_place_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug" 
        # Load Processor & VLA
        use_pretrain = False
        if use_pretrain:
            experiment_folder = "openvla/openvla-7b"
        self.processor = AutoProcessor.from_pretrained(experiment_folder, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            experiment_folder,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_safetensors=True,
        ).to("cuda:0")

        dataset_statistics_path = os.path.join(experiment_folder, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                norm_stats = json.load(f)
            self.vla.norm_stats = norm_stats
        
        self.is_aug = True if 'image_aug' in experiment_folder else False
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def act(self, image: np.array, instruction: str):
        image = np.asarray(preprocess_rgb(image, is_test=self.is_test), dtype=np.uint8)
        assert image.shape == (self.height, self.width, self.channel), f"image shape should be {(self.height, self.width, self.channel)} instead of {image.shape}"
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(image)).to("cuda:0", dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key=None, do_sample=False)
        return action
    
    def pickplace(self, observation_dict: dict):
        print("AGENT: openvla is processing.")

        multiview_rgbs, multiview_depths, instruction = observation_dict['rgbs'], observation_dict['depths'], observation_dict['instruction']

        pick_instruction = f"pick {instruction['pick_obj']}"
        place_instruction = f"place {instruction['pick_obj']} on {instruction['place_obj']}"

        topdown_rgb = preprocess_openvla_obs(multiview_rgbs, multiview_depths, self.intrinsics, self.extrinsics)
        if self.is_aug:
            image = apply_center_crop(topdown_rgb)

        prompt = f"In: What action should the robot take to {pick_instruction}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(image)).to("cuda:0", dtype=torch.bfloat16)
        pick_action = self.vla.predict_action(**inputs, unnorm_key=None, do_sample=False)
        pick_xyr = np.array([pick_action[0]*224, pick_action[1]*224, pick_action[5]*np.pi*2])

        prompt = f"In: What action should the robot take to {place_instruction}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(image)).to("cuda:0", dtype=torch.bfloat16)
        place_action = self.vla.predict_action(**inputs, unnorm_key=None, do_sample=False)
        place_xyr = np.array([place_action[0]*224, place_action[1]*224, place_action[5]*np.pi*2])

        p0_pixel_action, p1_pixel_action = de_preprocess_action(pick_xyr, place_xyr)

        pick_robot_action = convert_pixel_action_to_robot_action(p0_pixel_action)
        place_robot_action = convert_pixel_action_to_robot_action(p1_pixel_action)

        primitive = 'pickplace' #TODO

        return {'pick': pick_robot_action, 'place': place_robot_action}, primitive


if __name__ == "__main__":
    experiment_folder = "/home/mingxi/code/gem/openVLA/logs/openvla-7b+franka_pick_place_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug-1225demo20"
    agent = OpenVLAAgent(experiment_folder)
    image = np.random.random([480,640,3]) * 255
    image = image.astype(np.uint8)
    action = agent.act(image, "pick mouse and place into box")
    print(action)