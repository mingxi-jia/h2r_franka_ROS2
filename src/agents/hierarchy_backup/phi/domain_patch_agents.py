from agents.hierarchy_backup.phi.dqn_phi_2 import DQNPhi2
from agents.hierarchy_backup.phi.dqn_phi_agents_in_hand import DQN2InHand, DQNInHand
from agents.hierarchy_backup.phi.domain_patch_interface import DomainPatchInterface

class DQNPhiInHandDomainPatch(DomainPatchInterface, DQNInHand):
    def __init__(self, *args, **kwargs):
        DQNInHand.__init__(self, *args, **kwargs)
        DomainPatchInterface.__init__(self)


class DQNPhi2InHandDomainPatch(DomainPatchInterface, DQN2InHand):
    def __init__(self, *args, **kwargs):
        DQN2InHand.__init__(self, *args, **kwargs)
        DomainPatchInterface.__init__(self)


class DQNPhi2DomainPatch(DomainPatchInterface, DQNPhi2):
    def __init__(self, *args, **kwargs):
        DQNPhi2.__init__(self, *args, **kwargs)
        DomainPatchInterface.__init__(self)