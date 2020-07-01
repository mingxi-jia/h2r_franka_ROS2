from agents.hierarchy_backup.phi.dqn_phi import DQNPhi
from agents.hierarchy_backup.phi.dqn_phi_2 import DQNPhi2
from agents.hierarchy_backup.phi.in_hand_interface import InHandInterface

class DQN2InHand(InHandInterface, DQNPhi2):
    def __init__(self, *args, **kwargs):
        DQNPhi2.__init__(self, *args, **kwargs)
        InHandInterface.__init__(self)

class DQNInHand(InHandInterface, DQNPhi):
    def __init__(self, *args, **kwargs):
        DQNPhi.__init__(self, *args, **kwargs)
        InHandInterface.__init__(self)