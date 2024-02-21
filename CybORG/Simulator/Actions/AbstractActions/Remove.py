

from CybORG.Shared import Observation
from .Monitor import Monitor
from CybORG.Simulator.Actions import Action
from CybORG.Simulator.Actions.ConcreteActions.StopProcess import StopProcess
from CybORG.Simulator.Session import VelociraptorServer
from CybORG.Simulator.State import State


class Remove(Action):
    def __init__(self, session: int, agent: str, hostname: str):
        super().__init__()
        self.agent = agent
        self.session = session
        self.hostname = hostname
        self.action_success = False
        self.any_sus_pids = False
        self.blocked = False

    def execute(self, state: State) -> Observation:
        # perform monitor at start of action
        #monitor = Monitor(session=self.session, agent=self.agent)
        #obs = monitor.execute(state)
        self.action_success = False
        self.any_sus_pids = False
        self.blocked = False
        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]
        # find relevant session on the chosen host
        sessions = [s for s in state.sessions[self.agent].values() if s.hostname == self.hostname]
        obs = Observation(False)
        if len(sessions) > 0:
            session = state.np_random.choice(sessions)
            # remove suspicious processes
            if self.hostname in parent_session.sus_pids:
                obs_success = []
                for sus_pid in parent_session.sus_pids[self.hostname]:
                    action = StopProcess(session=self.session, agent=self.agent, target_session=session.ident, pid=sus_pid)
                    sub_obs = action.execute(state)
                    obs_success.append(sub_obs.action_succeeded)
                if obs_success:
                    self.any_sus_pids = True
                    if any(obs_success):
                        obs.set_success(True)
                        self.action_success = True     
            if self.action_success:
                subnet = state.hostname_subnet_map[self.hostname]
                if state.blocks:
                    self.blocked = any(subnet in sublist for sublist in state.blocks.values())
            return obs
        else:
            return obs

    @property
    def cost(self):
        if not self.action_success:
            if not self.any_sus_pids:
                return -1.0
            else:
                return 0.0
        else:
            if self.blocked:
                return 1.0
            else:
                return 0.1
    
    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"
