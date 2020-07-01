from copy import deepcopy
from utils import torch_utils
from agents.hierarchy.dqn_3l_rz_rxz import DQN3LMaxSharedRzRxZ
from agents.hierarchy.margin_3l_max_shared import Margin3LMaxShared
import numpy as np

import torch
import torch.nn.functional as F

class Margin3LMaxSharedRzRxZ(DQN3LMaxSharedRzRxZ, Margin3LMaxShared):
    def __init__(self, q1, q2, q3, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=False, patch_size=24,
                 margin='ce', margin_l=0.1, margin_weight=1.0, softmax_beta=1.0,
                 num_rx=8, min_rx=-np.pi / 4, max_rx=np.pi / 4, num_zs=16, min_z=0.02, max_z=0.17):
        DQN3LMaxSharedRzRxZ.__init__(self, q1, q2, q3, action_space, workspace, heightmap_resolution, device, lr, gamma,
                                     num_primitives, sl, per, num_rotations, half_rotation, patch_size,
                                     num_rx, min_rx, max_rx, num_zs, min_z, max_z)
        self.margin = margin
        self.margin_l = margin_l
        self.margin_weight = margin_weight
        self.softmax_beta = softmax_beta

    def update(self, batch):
        return Margin3LMaxShared.update(self, batch)