from random import choice

from CybORG.Shared import Observation
from CybORG.Simulator.Actions import Action
from CybORG.Simulator.Actions.AbstractActions import Monitor
from CybORG.Simulator.Actions.ConcreteActions.RemoveFile import RemoveFile
from CybORG.Simulator.Session import VelociraptorServer


class DataRepair(Action):
    def __init__(self, session: int, agent: str, hostname: str):
        super().__init__()
        self.agent = agent
        self.session = session
        self.hostname = hostname

    def execute(self, state) -> Observation:
        # perform monitor at start of action
        #monitor = Monitor(session=self.session, agent=self.agent)
        #obs = monitor.execute(state)
    
        parent_session: VelociraptorServer = state.sessions[self.agent][self.session]
        # find relevant session on the chosen host
        sessions = [s for s in state.sessions[self.agent].values() if s.hostname == self.hostname]
        if len(sessions) > 0:
            session = choice(sessions)
            obs = Observation(True)
            # remove suspicious files
            if self.hostname in parent_session.sus_files:
                for path, sus_file in parent_session.sus_files[self.hostname]:
                    sub_action = RemoveFile(agent=self.agent, session=self.session, target_session=session.ident, path=path, file_name=sus_file)
                    obs = sub_action.execute(state)
                if obs.success == True:
                    del parent_session.sus_files[self.hostname]
            return obs
        else:
            return Observation(False)

    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"
    
