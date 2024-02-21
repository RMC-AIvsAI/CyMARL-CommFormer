from CybORG.Shared import Observation
from CybORG.Simulator.Actions import Action
from CybORG.Simulator.Actions.AbstractActions import Monitor
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import AllowZoneTraffic
from CybORG.Simulator.Session import VelociraptorServer


class UnBlock(Action):
    def __init__(self, subnet: str, session: int, agent: str):
        super().__init__()
        self.subnet = subnet
        self.agent = agent
        self.session = session
        self.action_cost = -1.0

    def execute(self, state) -> Observation:
        # perform monitor at start of action
        #monitor = Monitor(session=self.session, agent=self.agent)
        #obs = monitor.execute(state)
        self.action_cost = -1.0
        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]
        sub_action = AllowZoneTraffic(session=self.session, agent=self.agent, subnet=self.subnet)
        obs = sub_action.execute(state)
        if obs.action_succeeded:
            self.action_cost = 0
        return obs

    def __str__(self):
        return f"{self.__class__.__name__} {self.subnet}"

    @property
    def cost(self):
        return self.action_cost