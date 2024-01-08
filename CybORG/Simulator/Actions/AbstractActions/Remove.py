

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

    def execute(self, state: State) -> Observation:
        # perform monitor at start of action
        monitor = Monitor(session=self.session, agent=self.agent)
        obs = monitor.execute(state)

        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]
        # find relevant session on the chosen host
        sessions = [s for s in state.sessions[self.agent].values() if s.hostname == self.hostname]
        if len(sessions) > 0:
            session = state.np_random.choice(sessions)
            #obs = Observation(True)
            # remove suspicious processes
            if self.hostname in parent_session.sus_pids:
                obs_success = []
                for sus_pid in parent_session.sus_pids[self.hostname]:
                    action = StopProcess(session=self.session, agent=self.agent, target_session=session.ident, pid=sus_pid)
                    sub_obs = action.execute(state)
                    obs_success.append(sub_obs.action_succeeded)
                if any(obs_success):
                    self.action_success = True
                else:
                    self.action_success = False
                    obs.set_success(False)
            else:
                self.action_success = False
                obs.set_success(False)
            # remove suspicious files
            return obs
        else:
            self.action_success = False
            obs.set_success(False)
            return obs

    @property
    def cost(self):
        if not self.action_success:
            return -0.1
        else:
            return 0
    
    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"
