# Copyright DST Group. Licensed under the MIT license.
from collections import namedtuple

from CybORG.Shared import Scenario
from CybORG.Shared.Enums import OperatingSystemType
from CybORG.Shared.RewardCalculator import RewardCalculator
import pprint

from CybORG.Simulator.Actions.AbstractActions import Restore

WIN_REWARD = 0
LOSE_REWARD = 0
SESSION_REWARD = 0
ROOT_SESSION_REWARD = 0
LOSE_SESSION_REWARD = 0
LOSE_ROOT_SESSION_REWARD = 0

REWARD_MAX_DECIMAL_PLACES = 1
HostReward = namedtuple('HostReward','confidentiality availability')

# CyMARL - confidentiality
class PwnRewardCalculator(RewardCalculator):
    # this reward calculator provides a reward to Red due to changes in the number of privileged sessions
    # For DIAL implementation added rewards for exploited hosts. Exploited hosts now will receive the score of confidentiality value
    # and privilege escalated hosts will receive double the confidentiality value
    def __init__(self, agent_name: str, scenario: Scenario):
        super().__init__(agent_name)
        self.old_total = 0
        self.scenario = scenario
        self.mapping = {'None': 0.0,
                        'Low': 0.1,
                        'Medium': 1.0,
                        'High': 10.0}

        self.compromised_hosts = {}

    def reset(self):
        self.old_total = 0

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool):
        root_sessions = 0
        system_sessions = 0
        user_sessions = 0
        self.compromised_hosts = {}
        for host, info in current_state.items():
            if host == 'success':
                continue

            if 'Sessions' in info:
                for session in info['Sessions']:
                    if session['Agent'] == self.agent_name:
                        
                        if info['System info']['OSType'] == OperatingSystemType.LINUX:
                            # count the number of root sessions
                            if session['Username'] == 'root':
                                confidentiality_value = self.mapping[self.scenario.get_host(host).get_confidentiality_value('Low')]
                                #root_sessions += (confidentiality_value * 2)
                                root_sessions += confidentiality_value
                                #self.compromised_hosts[host] = (confidentiality_value * 2)
                                self.compromised_hosts[host] = confidentiality_value
                                break
                            #else:
                                # count the number of user sessions
                            #    confidentiality_value = self.mapping[self.scenario.get_host(host).get_confidentiality_value('Low')]
                            #    user_sessions += confidentiality_value
                            #    self.compromised_hosts[host] = confidentiality_value
                            #    break
                        
                        if info['System info']['OSType'] == OperatingSystemType.WINDOWS:
                            # count the number of SYSTEM sessions
                            if session['Username'] == 'SYSTEM':
                                confidentiality_value = self.mapping[self.scenario.get_host(host).get_confidentiality_value('Low')]
                                #system_sessions += (confidentiality_value * 2)
                                system_sessions += confidentiality_value
                                #self.compromised_hosts[host] = (confidentiality_value * 2)
                                self.compromised_hosts[host] = confidentiality_value
                                break
                            #else:
                            #    # count the number of user sessions
                            #    confidentiality_value = self.mapping[self.scenario.get_host(host).get_confidentiality_value('Low')]
                            #    user_sessions += confidentiality_value
                            #    self.compromised_hosts[host] = confidentiality_value
                            #    break


        # find the difference from the old privileged sessions
        total = root_sessions + system_sessions + user_sessions
        reward = total #- self.old_total
        self.old_total = total
        return round(reward, REWARD_MAX_DECIMAL_PLACES)

# CyMARL - Availability
class DoSRewardCalculator(RewardCalculator):
    # calculates the reward for disrupting the network
    def __init__(self, agent_name: str, scenario: Scenario):
        super(DoSRewardCalculator, self).__init__(agent_name)
        self.scenario = scenario
        self.procs = {}
        self.mapping = {'None': 0.0,
                        'Low': 0.1,
                        'Medium': 0.2,
                        'High': 10.0}

        self.impacted_hosts = {}
        self.disrupted = []
        self.first_pass = True

    def reset(self):
        self.procs = {}
        self.impacted_hosts = {}
        self.disrupted = []

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool):

        for hostname, data in current_state.items():
            if hostname == 'success':
                continue
            if "Processes" in data:
                # iterate over files in the current obs and give reward for each 'secret.txt'
                for proc in data['Processes']:
                    if proc['Process Name'] == 'DoS.exe' and hostname not in self.disrupted:
                        self.disrupted.append(hostname)


        # get availability values                    
        availability_scores = [self.mapping[self.scenario.get_host(i).get_availability_value('None')] for i in self.disrupted]
        # sum disrupted hosts
        self.impacted_hosts = dict(zip(self.disrupted,availability_scores))

        reward = sum(availability_scores)
        return round(reward, REWARD_MAX_DECIMAL_PLACES)

# CyMARL - Integrity
class TamperRewardCalculator(RewardCalculator):
    # calculates the reward for modifying data on the hosts
    def __init__(self, agent_name: str, scenario: Scenario):
        super(TamperRewardCalculator, self).__init__(agent_name)
        self.scenario = scenario
        self.files = []
        self.mapping = {'None': 0.0,
                        'Low': 0.1,
                        'Medium': 1.0,
                        'High': 10.0}

    def reset(self):
        self.files = []
        self.impacted_hosts = {}

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool):
        for hostname, data in current_state.items():
            if hostname == 'success':
                continue
            if "Files" in data:
                # iterate over files in the current obs and give reward for each 'secret.txt'
                for file in data['Files']:
                    if file['File Name'] == 'secret.txt' and hostname not in self.files:
                        self.files.append(hostname)
                                
        # get availability values (to be changed to integrity values)             
        integrity_scores = [self.mapping[self.scenario.get_host(i).get_integrity_value('None')] for i in self.files]
        self.impacted_hosts = dict(zip(self.files, integrity_scores))
        reward = sum(integrity_scores)
        return round(reward, REWARD_MAX_DECIMAL_PLACES)

# CybORG - OT Availability
class DisruptRewardCalculator(RewardCalculator):
    # calculates the reward for disrupting the network
    def __init__(self, agent_name: str, scenario: Scenario):
        super(DisruptRewardCalculator, self).__init__(agent_name)
        if self.agent_name is None:
            self._log_warning(f'Setting agent_name to Red in {__class__.__name__} because input was None. Probably Scenario1 running.')
            self.agent_name = 'Red'

        self.scenario = scenario
        self.ots = None
        self.mapping = {'None': 0.0,
                        'Low': 0.1,
                        'Medium': 1.0,
                        'High': 10.0}

        self.impacted_hosts = {}

    def reset(self):
        self.ots = None

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool):
        self.impacted_hosts = {}
        # currently a single 1 reward is recieved if the Impact action is successful
        # TODO: Add in test to see what required services are up

        disrupted = []
        if self.ots is None:
            self.ots = []
            for host_info in current_state.values():
                if type(host_info) is not dict:
                    continue
                if 'Processes' in host_info:
                    for proc in host_info['Processes']:
                        if proc['Process Name'] == 'OTService':
                            self.ots.append(host_info['System info']['Hostname'])


        for host_info in current_state.values():
            working = False
            if type(host_info) is not dict:
                continue

            if self.ots:
                if host_info['System info']['Hostname'] in self.ots:
                    if 'Processes' in host_info:
                        for proc in host_info['Processes']:
                            if proc['Process Name'] == 'OTService':
                                working = True
                        if not working:
                            disrupted.append(host_info['System info']['Hostname'])

        availability_scores = [self.mapping[self.scenario.get_host(i).get_availability_value(default='Low')] for i in disrupted]
        self.impacted_hosts = dict(zip(disrupted,availability_scores))

        # CyMARL - Restore action scales with availability value
        for agent_name in action.keys():
            agent_action = action[agent_name]
            if agent_action.name == 'Restore':
                availability_scores.append(self.mapping[self.scenario.get_host(agent_action.hostname).get_availability_value(default='Low')])
                
        reward = sum(availability_scores)
        return round(reward, REWARD_MAX_DECIMAL_PLACES)


class HybridImpactPwnRewardCalculator(RewardCalculator):
    # Hybrid of availability and confidentiality reward calculator
    def __init__(self, agent_name: str, scenario: Scenario):
        super(HybridImpactPwnRewardCalculator, self).__init__(agent_name)
        self.pwn_calculator = PwnRewardCalculator(agent_name, scenario)
        self.disrupt_calculator = DisruptRewardCalculator(agent_name, scenario)
        self.host_scores = {}

    def reset(self):
        self.pwn_calculator.reset()
        self.disrupt_calculator.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = self.pwn_calculator.calculate_reward(current_state, action, agent_observations, done) \
                 + self.disrupt_calculator.calculate_reward(current_state, action, agent_observations, done)

        self._compute_host_scores(current_state.keys())
        return round(reward, REWARD_MAX_DECIMAL_PLACES)

    def _compute_host_scores(self, hostnames):
        self.host_scores = {}
        compromised_hosts = self.pwn_calculator.compromised_hosts
        impacted_hosts = self.disrupt_calculator.impacted_hosts
        for host in hostnames:
            if host == 'success':
                continue
            compromised = compromised_hosts[host] if host in compromised_hosts else 0
            impacted = impacted_hosts[host] if host in impacted_hosts else 0
            reward_state = HostReward(compromised,impacted)  
                                    # confidentiality, availability
            self.host_scores[host] = reward_state

        
