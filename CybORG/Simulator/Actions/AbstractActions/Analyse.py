

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
        self.action_success = False

    def execute(self, state) -> Observation:
        # perform monitor at start of action
        #monitor = Monitor(session=self.session, agent=self.agent)
        #obs = monitor.execute(state)
        obs = Observation()
        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]
        #if any(state.hosts[self.hostname].sessions['Red']):
        #    self.action_success = True
        #else:
        #    self.action_success = False
        self.action_success = False
        artefacts = [DensityScout, SigCheck]
        # find relevant session on the chosen host
        sessions = [s for s in state.sessions[self.agent].values() if s.hostname == self.hostname]
        # find other agents action
        other_agent_actions = [action for agent, action in state.actions[-1].items() if agent != self.agent and agent != 'Red']
        action_valid = False
        if any(other_agent_actions):
            if other_agent_actions[0].name == 'Block' and other_agent_actions[0].subnet == state.hostname_subnet_map[self.hostname]:
                action_valid = True
        
        if len(sessions) > 0 and action_valid:
            session = state.np_random.choice(sessions)
            # run the artifacts on the chosen host
            #obs = Observation(True)
            for artifact in artefacts:
                sub_action = artifact(agent=self.agent, session=self.session, target_session=session.ident)
                sub_obs = sub_action.execute(state)
                obs.combine_obs(sub_obs)
            
            if self.hostname in obs.data:
                if 'Files' in obs.data[self.hostname]:
                    for file in obs.data[self.hostname]['Files']:
                        if 'Density' in file:
                            if file['Density'] >= 0.8:
                                self.action_success = True

                                red_pid = [s for s in state.sessions['Red'].values() if s.hostname == self.hostname][0].pid
                                parent_session.add_sus_pids(hostname=self.hostname, pid=int(red_pid))
                                             
            return obs
        else:
            obs.set_success(False)
            return obs

    @property
    def cost(self):
        if not self.action_success:
            return -0.5
        else:
            return 0.0

    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"
    
