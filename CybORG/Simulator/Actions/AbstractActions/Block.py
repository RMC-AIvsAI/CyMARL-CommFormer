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
        self.cost_multiplier = 1
        self.action_cost = 0

    def execute(self, state) -> Observation:
        # perform monitor at start of action
        #monitor = Monitor(session=self.session, agent=self.agent)
        #obs = monitor.execute(state)
        if len(state.actions) > 1:
            last_agent_action= [action for agent, action in state.actions[-2].items() if agent == self.agent]
            other_agent_actions = [action for agent, action in state.actions[-1].items() if agent != self.agent and agent != 'Red']
            if last_agent_action[0].name == 'Block' and last_agent_action[0].subnet == self.subnet:
                self.cost_multiplier *= 2
            else:
                self.cost_multiplier = 1
            
        obs = Observation(True)
        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]
        sub_action = BlockZoneTraffic(session=self.session, agent=self.agent, subnet=self.subnet)
        sub_action.execute(state)
        return obs
    
    @property
    def cost(self):
        return -1.0 * self.cost_multiplier

    def __str__(self):
        return f"{self.__class__.__name__} {self.subnet}"

