from CybORG.Shared import Observation
from CybORG.Simulator.Actions.ConcreteActions.LocalAction import LocalAction
from CybORG.Simulator.Host import Host
from CybORG.Simulator.Process import Process
from CybORG.Simulator.State import State


class RemoveFile(LocalAction):
    def __init__(self, session: int, agent: str, target_session: int, path: str, file_name: str):
        super(RemoveFile, self).__init__(session, agent)
        self.file_name = file_name
        self.path = path
        self.target_session = target_session

    def execute(self, state: State) -> Observation:
        obs = Observation()
        obs.set_success(False)
        if self.session not in state.sessions[self.agent] or self.target_session not in state.sessions[self.agent]:
            return obs
        target_host: Host = state.hosts[state.sessions[self.agent][self.target_session].host]
        session = state.sessions[self.agent][self.session]
        target_session = state.sessions[self.agent][self.target_session]

        if not session.active or not target_session.active:
            return obs

        file = target_host.get_file(self.file_name, self.path)
        if file is not None:
            obs.set_success(True)
            target_host.files.remove(file)
            # note that files in scenario will not repopulate
        return obs
