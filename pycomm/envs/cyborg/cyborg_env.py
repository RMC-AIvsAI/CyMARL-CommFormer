import inspect
import os
import sys
import copy
import numpy

import torch
from gym.spaces import flatdim
from pycomm.envs.multiagentenv import MultiAgentEnv

from CybORG import CybORG
from CybORG.Shared.Scenarios.FileReaderScenarioGenerator import \
    FileReaderScenarioGenerator
from CybORG.Wrappers import MultiAgentDIALWrapper, BlueTableDIALWrapper, EnumActionDIALWrapper

class CyborgEnv(MultiAgentEnv):

    def __init__(self, map_name, time_limit=100, action_limiting=False, comm_limiting=False, wrapper_type='table', **kwargs):
        self.action_limiting = action_limiting
        self.comm_limiting = comm_limiting
        self.episode_limit = time_limit
        self._env = self._create_env(map_name, time_limit, wrapper_type)
        
        self.n_agents = len(self._env.agents)
        self._agent_ids = list(self._env.agents)
        self._obs = None
        
        # CommFormer specific attributes
        self.observation_space = list(self._env.observation_spaces.values())
        self.share_observation_space = list(self._env.share_observation_spaces.values())
        self.action_space = list(self._env.action_spaces.values())

        # DIAL and other algorithm specific attributes
        self.longest_action_space = max(self._env.action_spaces.values(), key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_spaces.values(), key=lambda x: x.shape
        )
        self.step_count = 0
        self.max_hosts = max(list(len(self.get_agent_hosts(agent)) for agent in self._agent_ids))
        self.all_obs = {}

    def reset(self):
        # Returns initial observations and states
        self.step_count = 0
        self._obs = self._env.reset()
        self._obs = list(self._obs.values())
        self.all_obs = {}
        self.all_obs[self.step_count] = copy.deepcopy(self._obs)

        return self.get_state()
    
    def step(self, actions):
        # Returns reward, terminated
        actions = actions - 1
        actions = actions.tolist()
        action_dict = dict(zip(self._agent_ids, actions))
        # works with underlying multiagentdialwrapper to get the results
        self._obs, reward, done, info = self._env.step(action_dict)
        self._obs = list(self._obs.values())
        self.step_count += 1
        self.all_obs[self.step_count] = copy.deepcopy(self._obs)
        # CommFormer specific change. FUTURE WORK: incorporate if statement to differentiate between DIAL run and CommFormer run.
        return torch.tensor(list(reward.values())), list(done.values()), str(info['Red']['action'])
        # old version, compatible with DIAL. Reference only.
        #return torch.tensor(list(reward.values())), int(all(done.values())), str(info['Red']['action'])

    def get_obs(self):
        # Returns all agent observations in a list
        return self._obs

    def get_obs_agent(self, agent_id):
        # Returns observation for agent_id
        return self._obs[agent_id]

    def get_obs_size(self):
        # Returns the shape of the observation 
        return flatdim(self.longest_observation_space)

    def get_state(self):
        
        if self.step_count < self.episode_limit:
            state = []
            for agent in range(self.n_agents):
                agent_obs = self._obs[agent]
                flattened_obs = sum(agent_obs, [])  # Concatenate sublists
                padded_obs = flattened_obs + [0] * (self.get_obs_size() - len(flattened_obs))
                #padded_obs.extend([self.step_count])
                state.append(padded_obs)
            state_tensor = torch.tensor(state, dtype=torch.long)
            return state_tensor
        return torch.zeros(self.n_agents, self.get_obs_size())

    def get_state_size(self):
        # Returns the shape of the state
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_possible_actions(self, agent):
        return self._env.get_possible_actions(agent)
    
    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, comm_vector, step, agent_id):
        # Returns the available actions and comm_limited for agent_id 
        comm_lim = self.get_comm_limited(step, agent_id)
        agent_name = self._agent_ids[agent_id]
        actions = self.get_possible_actions(agent_name)
        
        if not self.action_limiting:
            valid_actions = [1 for _ in range(len(actions))]
            return torch.tensor(valid_actions, dtype=torch.long), comm_lim
        
        analyse_available = True # Even with limiting actions analyse is available
        if comm_vector is not None: # If comm_vector is provided that means analyse is available only if another agent sends a signal
            analyse_available = False
            for i in comm_vector.numpy():
                if i >= 0.5:
                    analyse_available = True

        valid_actions = copy.deepcopy(actions)
        obs = self.all_obs[step][agent_id]
        for i, action in enumerate(actions):
            if action.name == 'Sleep' or action.name == 'Block': # Sleep action and block action is always available
                valid_actions[i] = 1
            else:
                host_obs = obs[(i - 1) % self.max_hosts]
                if sum(host_obs) > 1: # If he host has more than 1 bit then all actions are avaliable
                    valid_actions[i] = 1
                elif sum(host_obs) == 1 and host_obs[0] == 0: # If he host has only 1 bit in its obs and the scan bit is 0 then all actions are avaliable
                    valid_actions[i] = 1
                elif action.name == 'Analyse' and analyse_available:
                    valid_actions[i] = 1
                else:
                    valid_actions[i] = 0

        return torch.tensor(valid_actions, dtype=torch.long), comm_lim

    def get_total_actions(self):
        # Returns the total number of actions an agent could ever take 
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed):
        self._env.set_seed(seed)

    def get_stats(self, steps):
        #TODO
        return 0

    def get_comm_limited(self, step, agent_id):
        """
        Limiting communication, to only when an agent sees a 1 in any of its observation space
        """
        comm_lim = False
        if self.comm_limiting:
            for i, obs in enumerate(self.all_obs[step]):
                if i == agent_id:
                    activity = any(1 in host for host in obs if len(host) == 4)
                    if activity:
                        comm_lim = True
                    else:
                        comm_lim = False
        return comm_lim
    
    def get_agent_hosts(self, agent):
        return self._env.get_agent_hosts(agent)

    def get_env_info(self):
        env_info = {"obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "hosts_per_agent": self.max_hosts,
                    "possible_actions": {agent: self.get_possible_actions(agent) for agent in self._agent_ids}
        }
        return env_info
    
    def _wrap_env(self, env, wrapper_type):
        try:
            if wrapper_type == 'vector':
                return MultiAgentDIALWrapper(BlueTableDIALWrapper(EnumActionDIALWrapper(env), output_mode='vector'))
                #return MultiAgentDIALWrapper(EnumActionDIALWrapper(BlueTableDIALWrapper(env, output_mode='vector')))
            else:
                raise ValueError(f"Unsupported wrapper type: {wrapper_type}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit()

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
    