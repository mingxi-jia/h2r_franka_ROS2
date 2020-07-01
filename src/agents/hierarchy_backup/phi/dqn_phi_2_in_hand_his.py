from agents.hierarchy_backup.phi.dqn_phi_agents_in_hand import DQN2InHand
from agents.hierarchy_backup.phi.in_hand_his_interface import InHandHisInterface


class DQN2InHandHis(InHandHisInterface, DQN2InHand):
    def __init__(self, *args, **kwargs):
        self.num_his_channel = kwargs.pop('num_his_channel', 7)
        DQN2InHand.__init__(self, *args, **kwargs)
        InHandHisInterface.__init__(self)



