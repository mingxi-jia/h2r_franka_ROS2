from agents.hierarchy_backup.phi.dqn_phi_2 import DQNPhi2
from agents.hierarchy_backup.phi.dqn_phi_2_domain_input import DQN2DomainInput
from agents.hierarchy_backup.phi.dqn_phi_agents_in_hand_domain_input import DQN2InHandDomain
from agents.hierarchy_backup.phi.v2_v_function_interface import V2VFunctionInterface

class DQN2V(V2VFunctionInterface, DQNPhi2):
    def __init__(self, *args, **kwargs):
        v_net = kwargs.pop('v_net')
        DQNPhi2.__init__(self, *args, **kwargs)
        V2VFunctionInterface.__init__(self, v_net)

class DQN2DomainInputV(V2VFunctionInterface, DQN2DomainInput):
    def __init__(self, *args, **kwargs):
        v_net = kwargs.pop('v_net')
        DQN2DomainInput.__init__(self, *args, **kwargs)
        V2VFunctionInterface.__init__(self, v_net)

class DQN2InHandDomainV(V2VFunctionInterface, DQN2InHandDomain):
    def __init__(self, *args, **kwargs):
        v_net = kwargs.pop('v_net')
        DQN2InHandDomain.__init__(self, *args, **kwargs)
        V2VFunctionInterface.__init__(self, v_net)