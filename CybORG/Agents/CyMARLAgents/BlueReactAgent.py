from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Simulator.Actions import Monitor, Remove, Restore, Misinform
import random


class BlueReactRemoveAgent(BaseAgent):
    def __init__(self, agent_name=None):
        self.host_list = []
        self.last_action = None
        self.agent_name = agent_name

    def train(self, results: Results):
        pass

    def get_action(self, observation, action_space):
        # add suspicious hosts to the hostlist if monitor found something
        # added line to reflect changes in blue actions
        # it looks weird but allows for raw cyborg interface
        if isinstance(observation, Results.Results):
            observation = observation.observation
        if self.last_action is not None and self.last_action == 'Monitor':
            for host_name, host_info in [(value['System info']['Hostname'], value) for key, value in observation.items() if key != 'success']:
                if host_name not in self.host_list and host_name != 'User0' and 'Processes' in host_info and len([i for i in host_info['Processes'] if 'PID' in i]) > 0:
                    self.host_list.append(host_name)
        # assume a single session in the action space
        session = list(action_space['session'].keys())[0]
        if len(self.host_list) == 0:
            self.last_action = 'Monitor'
            return Monitor(agent=self.agent_name, session=session)
        else:
            self.last_action = 'Remove'
            return Remove(hostname=self.host_list.pop(0), agent=self.agent_name, session=session)

    def end_episode(self):
        self.host_list = []
        self.last_action = None

    def set_initial_values(self, action_space, observation):
        pass


class BlueReactRestoreAgent(BaseAgent):
    def __init__(self, agent_name=None):
        self.host_list = []
        self.last_action = None
        self.agent_name = agent_name

    def train(self, results: Results):
        pass

    def get_action(self, observation, action_space):
        if isinstance(observation, Results.Results):
            observation = observation.observation
        if self.last_action is not None and self.last_action == 'Monitor':
            for host_name, host_info in [(value['System info']['Hostname'], value) for key, value in observation.items() if key != 'success']:
                if host_name not in self.host_list and host_name != 'User0' and 'Processes' in host_info and len([i for i in host_info['Processes'] if 'PID' in i]) > 0:
                    self.host_list.append(host_name)
        # assume a single session in the action space
        session = list(action_space['session'].keys())[0]
        if len(self.host_list) == 0:
            self.last_action = 'Monitor'
            return Monitor(agent=self.agent_name, session=session)
        else:
            self.last_action = 'Restore'
            return Restore(hostname=self.host_list.pop(0), agent=self.agent_name, session=session)

    def end_episode(self):
        self.host_list = []
        self.last_action = None

    def set_initial_values(self, action_space, observation):
        pass

class BlueReactMisinformAgent(BaseAgent):
    def __init__(self, agent_name=None):
        self.host_list = []
        self.last_action = None
        self.agent_name = agent_name
        self.turn_count = 0

    def train(self, results: Results):
        pass

    def get_action(self, observation, action_space):
        if isinstance(observation, Results.Results):
            observation = observation.observation
        if self.last_action is not None and self.last_action == 'Monitor':
            for host_name, host_info in [(value['System info']['Hostname'], value) for key, value in observation.items() if key != 'success']:
                if host_name not in self.host_list and host_name != 'User0' and 'Processes' in host_info and len([i for i in host_info['Processes'] if 'PID' in i]) > 0:
                    self.host_list.append(host_name)
        # assume a single session in the action space
        session = list(action_space['session'].keys())[0]

        if self.turn_count < 5 and len(observation) > 1:
            self.turn_count += 1
            self.last_action = 'Misinform'
            random_hostname = random.choice([value['System info']['Hostname'] for key, value in observation.items() if key != 'success'])
            return Misinform(hostname=random_hostname, agent=self.agent_name, session=session)

        if len(self.host_list) == 0:
            self.last_action = 'Monitor'
            return Monitor(agent=self.agent_name, session=session)
        else:
            self.last_action = 'Restore'
            return Restore(hostname=self.host_list.pop(0), agent=self.agent_name, session=session)

    def end_episode(self):
        self.host_list = []
        self.last_action = None
        self.turn_count = 0

    def set_initial_values(self, action_space, observation):
        pass
