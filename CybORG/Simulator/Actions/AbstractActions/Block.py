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

    def execute(self, state) -> Observation:
        # perform monitor at start of action
        #monitor = Monitor(session=self.session, agent=self.agent)
        #obs = monitor.execute(state)
        obs = Observation(True)
        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]
        sub_action = BlockZoneTraffic(session=self.session, agent=self.agent, subnet=self.subnet)
        sub_action.execute(state)
        return obs
    
    @property
    def cost(self):
        return -1.0

    def __str__(self):
        return f"{self.__class__.__name__} {self.subnet}"

