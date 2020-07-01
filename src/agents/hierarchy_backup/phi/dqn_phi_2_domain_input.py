from agents.hierarchy_backup.phi.dqn_phi_2 import DQNPhi2

import torch


class DQN2DomainInput(DQNPhi2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forwardPhiNet(self, states, obs, pixels, target_net=False, to_cpu=False):
        obs = obs.to(self.device)
        pixels = pixels.to(self.device)
        patch = self.getImgPatch(obs, pixels)
        phi_net = self.phi_net if not target_net else self.target_phi
        phi_output = phi_net((obs, patch)).reshape(states.size(0), self.num_primitives, -1)[torch.arange(0, states.size(0)),
                                                                                     states.long()]
        if to_cpu:
            phi_output = phi_output.cpu()
        return phi_output