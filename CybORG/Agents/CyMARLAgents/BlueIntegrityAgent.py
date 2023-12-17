from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Simulator.Actions import Monitor, Remove, Restore, DataRepair, Misinform
import random


class BlueIntegrityRestoreAgent(BaseAgent):
    def __init__(self, agent_name=None):
        self.host_list = []
        self.rest_list = []
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

        # This agent will first try to perform DataRepair, then restore the host (regardless of outcome)
        if len(self.host_list) > 0 and random.randrange(2) > 0.5 :
            self.last_action = 'DataRepair'
            hostname = self.host_list.pop(0)
            obs = DataRepair(hostname=hostname, agent=self.agent_name, session=session)
            self.rest_list.append(hostname)
            return obs
        elif len(self.rest_list) > 0:
            self.last_action = 'Restore'
            return Restore(hostname=self.rest_list.pop(0), agent=self.agent_name, session=session)
        else:
            self.last_action = 'Monitor'
            return Monitor(agent=self.agent_name, session=session)

    def end_episode(self):
        self.host_list = []
        self.last_action = None

    def set_initial_values(self, action_space, observation):
        pass

class BlueIntegrityMisinformAgent(BaseAgent):
    def __init__(self, agent_name=None):
        self.host_list = []
        self.rest_list = []
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

        # This agent will first try to perform DataRepair, then restore the host (regardless of outcome)
        if len(self.host_list) > 0 and random.randrange(2) > 0.5 :
            self.last_action = 'DataRepair'
            hostname = self.host_list.pop(0)
            obs = DataRepair(hostname=hostname, agent=self.agent_name, session=session)
            self.rest_list.append(hostname)
            return obs
        elif len(self.rest_list) > 0:
            self.last_action = 'Restore'
            return Restore(hostname=self.rest_list.pop(0), agent=self.agent_name, session=session)
        else:
            self.last_action = 'Monitor'
            return Monitor(agent=self.agent_name, session=session)

    def end_episode(self):
        self.host_list = []
        self.last_action = None
        self.turn_count = 0

    def set_initial_values(self, action_space, observation):
        pass