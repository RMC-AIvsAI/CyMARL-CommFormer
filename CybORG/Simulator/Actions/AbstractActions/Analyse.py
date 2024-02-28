

from CybORG.Shared import Observation
from .Monitor import Monitor
from CybORG.Simulator.Actions import Action
from CybORG.Simulator.Actions.AbstractActions import Monitor
from CybORG.Simulator.Session import VelociraptorServer
from CybORG.Simulator.Actions.ConcreteActions.DensityScout import DensityScout
from CybORG.Simulator.Actions.ConcreteActions.SigCheck import SigCheck


class Analyse(Action):
    def __init__(self, session: int, agent: str, hostname: str):
        super().__init__()
        self.agent = agent
        self.session = session
        self.hostname = hostname
        self.action_cost = 0.0

    def execute(self, state) -> Observation:
        obs = Observation()
        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]

        self.action_cost = -10.0
        artefacts = [DensityScout, SigCheck]
        # find relevant session on the chosen host
        sessions = [s for s in state.sessions[self.agent].values() if s.hostname == self.hostname]
        # If there are sessions then continue with analyse
        if len(sessions) > 0:
            session = state.np_random.choice(sessions)
            # run the artifacts on the chosen host
            for artifact in artefacts:
                sub_action = artifact(agent=self.agent, session=self.session, target_session=session.ident)
                sub_obs = sub_action.execute(state)
                obs.combine_obs(sub_obs)
            
            if self.hostname in obs.data:
                if 'Files' in obs.data[self.hostname]:
                    for file in obs.data[self.hostname]['Files']:
                        if 'Density' in file:
                            if file['Density'] >= 0.9:
                                red_pid = [s for s in state.sessions['Red'].values() if s.hostname == self.hostname][0].pid
                                if self.hostname not in parent_session.sus_pids:
                                    self.action_cost = 0.2
                                else:
                                    if red_pid not in parent_session.sus_pids[self.hostname]:
                                        self.action_cost = 0.2
                                # When exploit is not detected, and once the host is escalated add the process in suspected pid list
                                parent_session.add_sus_pids(hostname=self.hostname, pid=int(red_pid))              
            return obs
        else:
            obs.set_success(False)
            return obs

    @property
    def cost(self):
        return self.action_cost

    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"
    
