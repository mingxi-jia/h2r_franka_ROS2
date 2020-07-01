import math

import torch
import numpy as np
import numpy.random as npr

import torch
import torch.nn.functional as F
import utils.torch_utils as torch_utils

class BaseAgent(object):
    def __init__(self, action_space, workspace, heightmap_resolution, device, num_rotations=8, half_rotation=False,
                 num_heights=10, height_range=(1, 0.1), lr=1e-4):
        '''
        Base agent class that generate random actions and e-greedy actions
        using the greedy actions method required by any sub classes.
        '''
        self.action_space = action_space
        self.workspace = workspace
        self.heightmap_resolution = heightmap_resolution
        self.device = device
        self.num_rotations = num_rotations
        self.half_rotation = half_rotation
        if self.half_rotation:
            self.rotations = torch.tensor([np.pi / self.num_rotations * i for i in range(self.num_rotations)])
        else:
            self.rotations = torch.tensor([(2*np.pi)/self.num_rotations * i for i in range(self.num_rotations)])
        self.num_heights = num_heights
        self.height_range = height_range
        self.heights = torch.tensor(np.linspace(height_range[0], height_range[1], self.num_heights),
                                    dtype=torch.float)

        self.lr = lr

    def getEGreedyActions(self, states, obs, eps):
        pass

    def seed(self, seed):
        '''
        Set the random torch seed

        Args:
          - seed: Integer to use as torch seed
        '''
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
