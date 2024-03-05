from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Simulator.Actions import Sleep, GreenPingSweep, GreenPortScan, GreenConnection

class GreenAgent(BaseAgent):
    def __init__(self, np_random=None):
        super().__init__(np_random)
        self.action_space = [
                Sleep,
                # GreenPingSweep,
                GreenPortScan,
                # GreenConnection, 
                ]
        self.hostnames = None
        self.subnets = None
        self.step_count = 0

    def get_action(self,observation,action_space):
        if self.step_count == 0:
            self.subnets = list(action_space['allowed_subnets'].keys())
            self.hostnames = [host for host, val in action_space['hostname'].items() if val]
            self.step_count += 1
        action = self.np_random.choice(self.action_space)
        if action == Sleep:
            return Sleep()
        elif action == GreenPingSweep:
            subnet = self.np_random.choice(self.subnets)
            return action(subnet=subnet,session=0,agent='Green')
        else:
            hostname = self.np_random.choice(self.hostnames)
            return action(hostname=hostname,session=0,agent='Green')

    def train(self,results):
        pass

    def end_episode(self):
        self.hostnames = None
        self.subnets = None
        self.step_count = 0

    def set_initial_values(self,action_space,observation):
        pass
