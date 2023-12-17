from random import choice

from CybORG.Shared import Observation
from CybORG.Simulator.Actions import Action
from CybORG.Simulator.Actions.ConcreteActions.AddFile import AddFile
from CybORG.Simulator.State import State


class Tamper(Action):
    def __init__(self, session: int, agent: str, hostname: str):
        super().__init__()
        self.agent = agent
        self.session = session
        self.hostname = hostname

    def execute(self, state: State) -> Observation:
        # find session on the chosen host
        sessions = [s for s in state.sessions[self.agent].values() if s.hostname == self.hostname]
        if len(sessions) == 0:
            # no valid session could be found on chosen host
            return Observation(success=False)
        # find if any session are already SYSTEM or root
        session = None
        for s in sessions:
            # else find if session is Admin or sudo
            if s.username == 'root' or s.username == 'SYSTEM':
                session = s.ident
                obs = Observation(success=True)
                obs.add_session_info(hostid=self.hostname, **s.get_state())
                break
        # else use random session
        if session is None:
            session = choice(sessions).ident

        # put file on host using chosen session
        sub_action = AddFile(session=self.session, agent=self.agent, target_session=session)
        obs = sub_action.execute(state)

        return obs

    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"
