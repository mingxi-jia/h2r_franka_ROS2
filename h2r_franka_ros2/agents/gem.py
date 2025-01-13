import numpy as np
import torch

# Remember to export agent PYTHONPATH


class GEM:
    def __init__(self, experiment_folder, intrinsics, extrinsics):
        self.model = load_model_from_folder(experiment_folder)

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def preprocess_obs(self, rgbs, depths, instruction):
        return


    def act(self, rgbs, depths, instruction:str):
        observation = self.preprocess_obs(rgbs, depths, instruction)
        pick_action, place_action = self.model(observation)
        return {'pick_action': pick_action, 'place_action':place_action}
