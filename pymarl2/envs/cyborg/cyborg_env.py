import inspect
import os
import copy

import numpy as np
from pymarl2.envs.multiagentenv import MultiAgentEnv
from gym.spaces import flatdim

from CybORG import CybORG
from CybORG.Shared.Scenarios.FileReaderScenarioGenerator import \
    FileReaderScenarioGenerator
from CybORG.Wrappers import MultiAgentGymWrapper, BlueTableWrapper, EnumActionWrapper, FixedFlatWrapper, BlueTableDIALWrapper, EnumActionDIALWrapper, MultiAgentDIALWrapper


class CyborgMultiAgentEnv(MultiAgentEnv):
    # Multi-Agent Environment adapted from SMAC
    def __init__(self, map_name, time_limit=100, action_masking=False, wrapper_type='raw', no_obs=False, **kwargs):
        self.episode_limit = time_limit
        self.action_masking = action_masking
        self._env = self._create_env(map_name, time_limit, wrapper_type)
        
        self.n_agents = len(self._env.agents)
        self._agent_ids = list(self._env.agents)
        self._obs = None
        self.info = None

        self.longest_action_space = max(self._env.action_spaces.values(), key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_spaces.values(), key=lambda x: x.shape
        )
        self.max_hosts = max(list(len(self.get_agent_hosts(agent)) for agent in self._agent_ids))
        
    def step(self, actions):
        """ Returns reward, terminated, info """
        assert(len(actions) == self.n_agents)
        actions = [int(a) for a in actions]
        action_dict = dict(zip(self._agent_ids, actions))
        self._obs, reward, done, self.info = self._env.step(action_dict)
        self._obs = list(self._obs.values())

        return float(sum(reward.values()) / self.n_agents), all(done.values()), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs = []
        for agent in range(self.n_agents):
            agent_obs = self._obs[agent]
            flattened_obs = sum(agent_obs, [])  # Concatenate sublists
            padded_obs = flattened_obs + [0] * (self.get_obs_size() - len(flattened_obs))
            #padded_obs.extend([self.step_count])
            obs.append(padded_obs)
        return obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self.get_obs(), axis=0).flatten().astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """

        agent_name = self._agent_ids[agent_id]
        actions = self.get_possible_actions(agent_name)
        valid_actions = copy.deepcopy(actions)
        obs = self._obs[agent_id]
        for i in range(len(valid_actions)):
            if i == 0 or i == len(valid_actions) - 1: # Sleep action and block action is always available
                valid_actions[i] = 1
            else:
                host_obs = obs[(i - 1) % self.max_hosts]
                if sum(host_obs) > 1: # If he host has more than 1 bit then all actions are avaliable
                    valid_actions[i] = 1
                elif sum(host_obs) == 1 and host_obs[0] == 0: # If he host has only 1 bit in its obs and the scan bit is 0 then all actions are avaliable
                    valid_actions[i] = 1
                else:
                    valid_actions[i] = 0

        return valid_actions

    def get_possible_actions(self, agent):
        return self._env.get_possible_actions(agent)
    
    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = list(self._obs.values())
        self._elapsed_steps = 0

        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        #trace = {}
        #for agent, data in self.info.items():
        #    trace[agent] = {"action": data['action'],
        #                    #"attempted_action": data['action'].action,
        #                    "reward": data['reward'],
        #                    #"obs": list(data['observation'])
        #                    }
        #return trace
        return {}

    def get_agent_hosts(self, agent):
        return self._env.get_agent_hosts(agent)
    
    def _wrap_env(self, env, wrapper_type):
        if wrapper_type == 'vector':
            return MultiAgentDIALWrapper(BlueTableDIALWrapper(EnumActionDIALWrapper(env), output_mode='vector'))
        else:
            return MultiAgentGymWrapper(FixedFlatWrapper(EnumActionWrapper(env)))

    def _create_env(self, map_name, time_limit, wrapper_type):
        # Get the directory containing cyborg
        cyborg_dir = os.path.dirname(os.path.dirname(inspect.getfile(CybORG)))
        path = cyborg_dir + f'/CybORG/Shared/Scenarios/scenario_files/{map_name}.yaml'
        norm_path = os.path.normpath(path)

        # Make scenario from specified file
        sg = FileReaderScenarioGenerator(norm_path)
        cyborg = CybORG(scenario_generator=sg, time_limit=time_limit)
        env = self._wrap_env(cyborg, wrapper_type)
        return env