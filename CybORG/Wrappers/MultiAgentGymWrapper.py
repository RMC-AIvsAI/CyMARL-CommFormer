from typing import Optional

from CybORG import CybORG
from CybORG.Wrappers import BaseWrapper

from gym.spaces import Discrete, MultiDiscrete, Box, Dict
import warnings
import random
from gym import spaces
import numpy as np

from CybORG.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper, RedTableWrapper, EnumActionWrapper
from CybORG.Shared.CommsRewardCalculator import CommsAvailabilityRewardCalculator

class MultiAgentGymWrapper(BaseWrapper):
    def __init__(self, env: BaseWrapper, max_steps: int = 100):
        super().__init__(env)

        self.observation_size = {agent: 
            len(self.env.get_observation(agent)) for agent in self.agents}

        self._observation_spaces = {agent: Box(
                -1.0, 64.0, shape=(self.observation_size[agent],), 
                dtype=np.float32) for agent in self.agents}

        self._action_spaces = {agent: Discrete(
                self.env.get_action_space(agent)) for agent in self.agents}

        self.max_steps = max_steps
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0. for agent in self.agents}
        self.infos = {}

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> dict:
        self.env.reset()
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0. for agent in self.agents}
        self.infos = {}
        return {agent: self.env.get_observation(agent) for agent in self.agents}

    def step(self, actions: dict):

        # Multi-step is a product of CyMARL for coop agent control
        results = self.env.multi_step(actions)
        
        obs = {agent: results[agent].observation for agent in self.agents}
        rew = {agent: results[agent].reward for agent in self.agents}
        done = {agent: results[agent].done for agent in self.agents}
        # Adds red agent info for traces
        info = {agent: vars(results[agent]) for agent in self.agents + ['Red']} 
        return obs, rew, done, info

    def observation_change(self, obs):
        return np.array(obs, dtype=int)

    def render(self, mode="human"):
        # Insert code from phillip
        return self.env.render(mode)

    def close(self):
        # Insert code from phillip
        return self.env.close()

    def observation_space(self, agent: str):
        '''
        Returns the observation space for a single agent

        Parameters:
            agent -> str
        '''
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        '''
        Returns the action space for a single agent

        Parameters:
            agent -> str
        '''
        return self._action_spaces[agent]

    @property
    def observation_spaces(self):
        '''
        Returns the observation space for every possible agent
        '''
        try:
            return {agent: self.observation_space(agent) for agent in self.agents}
        except AttributeError:
            raise AttributeError(
                "The base environment does not have an `observation_spaces` dict attribute. Use the environments `observation_space` method instead"
            )

    @property
    def action_spaces(self):
        '''
        Returns the action space for every possible agent
        '''
        try:
            return {agent: self.action_space(agent) for agent in self.agents}
        except AttributeError:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            )

    def get_rewards(self):
        '''
        Returns the rewards for every possible agent
        '''
        try:
            return {agent: self.get_reward(agent) for agent in self.agents}
        except AttributeError:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            )

    def get_dones(self):
        '''
        Returns the dones for every possible agent
        '''
        try:
            return {agent: self.get_done(agent) for agent in self.agents}
        except AttributeError:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            )

    def get_reward(self, agent):
        '''
        Returns the reward for a single agent

        Parameters:
            agent -> str
        '''
        return self.rewards[agent]

    def get_done(self, agent):
        '''
        Returns the dones for a single agent

        Parameters:
            agent -> str
        '''
        return self.dones[agent]

    def get_action_space(self, agent):
        '''
        Obtains the action_space of the specified agent

        Parameters:
            agent -> str
        '''
        return self.env.get_action_space(agent)

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_last_actions(self, agent):
        return self.get_attr('get_last_action')(agent)

    #@property
    #def agents(self) -> list: #for interface with PPZ env
    #    return [agent for agent in self.env.agents if not self.dones[agent]]

    @property
    def agents(self):
        return self.get_attr('agents')

    #@property
    #def agents(self):
    #    return self.get_attr('agents')
