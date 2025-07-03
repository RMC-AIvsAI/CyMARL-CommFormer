from CybORG.Shared import Observation
from CybORG.Simulator.Actions import Action
from CybORG.Simulator.Actions.AbstractActions import Monitor
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockZoneTraffic
from CybORG.Simulator.Session import VelociraptorServer


class Block(Action):
    def __init__(self, subnet: str, session: int, agent: str):
        super().__init__()
        self.subnet = subnet
        self.agent = agent
        self.session = session
        self.action_cost = 0

    def execute(self, state) -> Observation:
        self.action_cost = -2.0
        
        # discount action cost if a root or SYSTEM red session is present
        other_agent_actions = [action for agent, action in state.actions[-1].items() if agent != self.agent and agent != 'Red' and agent != 'Green']
        for action in other_agent_actions:
            if action.name == 'Analyse':
                red_sessions = state.hosts[action.hostname].sessions['Red']
                if any(red_sessions) and (state.sessions['Red'][red_sessions[0]].username == 'root' or state.sessions['Red'][red_sessions[0]].username == 'SYSTEM'):
                    self.action_cost = 1
                    break  # Exit the loop once the condition is met
        
        obs = Observation(True)
        sub_action = BlockZoneTraffic(session=self.session, agent=self.agent, subnet=self.subnet)
        sub_action.execute(state)
        return obs
    
    @property
    def cost(self):
        return self.action_cost

    def __str__(self):
        return f"{self.__class__.__name__} {self.subnet}"

