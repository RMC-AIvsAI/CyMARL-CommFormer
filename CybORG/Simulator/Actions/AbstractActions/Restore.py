

from CybORG.Shared import Observation
from .Monitor import Monitor
from CybORG.Simulator.Actions import Action
from CybORG.Simulator.Actions.ConcreteActions.RestoreFromBackup import RestoreFromBackup
from CybORG.Simulator.Session import VelociraptorServer
from CybORG.Simulator.Actions.AbstractActions import Monitor

class Restore(Action):
    def __init__(self, session: int, agent: str, hostname: str):
        super().__init__()
        self.agent = agent
        self.session = session
        self.hostname = hostname
        self.action_cost = 0
        self.mapping = {
            'Low': 1.0,
            'Medium': 1.0,
            'High': 10.0
        }

    def execute(self, state) -> Observation:
        self.action_cost = -1 * self.mapping[state.scenario.hosts[self.hostname].confidentiality_value]

        obs = Observation(False)
        if self.session not in state.sessions[self.agent]:
            return obs
        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]
        # find relevant session on the chosen host
        sessions = [s for s in state.sessions[self.agent].values() if s.hostname == self.hostname]
        if len(sessions) > 0:
            if self.hostname in parent_session.sus_pids and any(parent_session.sus_pids[self.hostname]):
                session = state.np_random.choice(sessions)
                obs.set_success(True)
                # restore host
                action = RestoreFromBackup(session=self.session, agent=self.agent, target_session=session.ident)
                action.execute(state)

        return obs

    @property
    def cost(self):
        return self.action_cost

    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"
