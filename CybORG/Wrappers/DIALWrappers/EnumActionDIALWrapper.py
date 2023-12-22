import copy
import inspect, pprint
from typing import Union

from CybORG.Agents.SimpleAgents import BaseAgent
from CybORG.Wrappers import BaseWrapper
from CybORG.Shared import Results


class EnumActionDIALWrapper(BaseWrapper):
    def __init__(self, env: Union[type, BaseWrapper] = None):
        super().__init__(env)
        self.possible_actions = {}
        self.action_signature = {}
        self.get_action_space('Red')

    def step(self, agent=None, action: int = None) -> Results:
        if action is not None:
            action = self.possible_actions[action]
        return super().step(agent, action)
    
    def multi_step(self, actions: dict):
        for agent_name, action in actions.items():
            if action is not None:
                actions[agent_name] = self.possible_actions[agent_name][action]
        return self.env.multi_step(actions)

    def action_space_change(self, action_space: dict = {}) -> int:
        # CyMARL revision
        if not isinstance(action_space, dict):
            return None
        possible_actions = []
        temp = {}
        params = ['action']
        agent = list(action_space['agent'].keys())[0]
        for i, action in enumerate(action_space['action']):
            if action not in self.action_signature:
                self.action_signature[action] = inspect.signature(action).parameters
            param_list = [{}]
            for p in self.action_signature[action]:
                if p == 'priority':
                    continue
                temp[p] = []
                if p not in params:
                    params.append(p)

                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            if "Blue" in agent and p == "hostname":
                                if key == "Defender":
                                    continue
                                if key == "User0":
                                    continue
                                if not val:
                                    continue
                            p_dict[p] = key
                            new_param_list.append({key: value for key, value in p_dict.items()})
                    param_list = new_param_list
            for p_dict in param_list:
                possible_actions.append(action(**p_dict))

        if agent != 'Red':
            possible_actions_sorted = [possible_actions[0]] + sorted(possible_actions[1:], key=lambda x: x.hostname[-1])
        else:
            possible_actions_sorted = possible_actions

        self.possible_actions[list(action_space['agent'].keys())[0]] = possible_actions_sorted
        return len(possible_actions_sorted)

