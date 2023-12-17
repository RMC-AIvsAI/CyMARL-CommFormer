"""
Switch game

This class manages the state of the Switch game among multiple agents.

RIAL Actions:

1 = Nothing
2 = Tell
3 = On
4 = Off
"""

import numpy as np
import torch

from utils.dotdic import DotDic 

class SwitchGame:

	def __init__(self, map_name, time_limit=6, **kwargs):
		self.game_actions = DotDic({
			'NOTHING': 1,
			'TELL': 2
		})

		self.game_states = DotDic({
			'OUTSIDE': 0,
			'INSIDE': 1,
		})

		self.game_comm_limited = True

		if map_name == "small":
			self.n_agents = 3
		elif map_name == "large":
			self.n_agents = 4
		else:
			self.n_agents = 2

		self.action_space = len(self.game_actions.keys())
		self.obs_space = 1
		self.time_limit = 4 * self.n_agents - 6

		self.reward_all_live = 1
		self.reward_all_die = -1

	def reset(self):
		# Step count
		self.step_count = 0

		# Rewards
		self.reward = torch.zeros(self.n_agents)

		# Who has been in the room?
		self.has_been = torch.zeros(self.time_limit, self.n_agents)

		# Terminal state
		self.terminal = 0

		# Active agent
		self.active_agent = torch.zeros(self.time_limit, dtype=torch.long) # 1-indexed agents
		for step in range(self.time_limit):
			agent_id = 1 + np.random.randint(self.n_agents)
			self.active_agent[step] = agent_id
			self.has_been[step][agent_id - 1] = 1

		return self.get_state()

	def get_action_range(self, a_total, step, agent_id):
		"""
		Return 1-indexed indices into Q vector for valid actions and communications (so 0 represents no-op)
		"""
		agent_id = agent_id + 1
		action_dtype = torch.long
		action_range = torch.zeros((2), dtype=action_dtype)
		comm_range = torch.zeros((2), dtype=action_dtype)
		if self.active_agent[step] == agent_id:
			action_range = torch.tensor([1, self.action_space], dtype=action_dtype)
			comm_range = torch.tensor(
				[self.action_space + 1, a_total], dtype=action_dtype)
		else:
			action_range = torch.tensor([1, 1], dtype=action_dtype)

		return action_range, comm_range

	def get_comm_limited(self, step, agent_id):
		comm_lim = 0
		if step > 0 and agent_id == self.active_agent[step]:
			comm_lim = self.active_agent[step - 1]
		return comm_lim

	def get_reward(self, a_t):
		# Return reward for action a_t by active agent
		active_agent_idx = self.active_agent[self.step_count].item() - 1
		if a_t[active_agent_idx].item() == self.game_actions.TELL and not self.terminal:
			has_been = self.has_been[:self.step_count + 1].sum(0).gt(0).sum(0).item()
			if has_been == self.n_agents:
				self.reward.fill_(self.reward_all_live)
			else:
				self.reward.fill_(self.reward_all_die)
			self.terminal = 1
		elif self.step_count == self.time_limit - 1 and not self.terminal:
			self.terminal = 1

		return self.reward.clone(), self.terminal

	def step(self, a_t):
		reward, terminal = self.get_reward(a_t)
		self.step_count += 1

		return reward, terminal

	def get_state(self):
		state = torch.zeros(self.n_agents, self.obs_space, dtype=torch.long)

		# Get the state of the game
		if (self.step_count < self.time_limit):
			for a in range(1, self.n_agents + 1):
				for o in range(self.obs_space):
					if self.active_agent[self.step_count] == a:
						state[a - 1][o] = self.game_states.INSIDE

		return state

	def god_strategy_reward(self, steps):
		reward = 0
		has_been = self.has_been[:self.time_limit].sum(0).gt(0).sum().item()
		if has_been == self.n_agents:
			reward = self.reward_all_live

		return reward

	def naive_strategy_reward(self):
		pass

	def get_stats(self, steps):
		#stats = DotDic({})
		#stats.god_reward = self.god_strategy_reward(steps)
		return self.god_strategy_reward(steps)

	def describe_game(self, b=0):
		print('has been:', self.has_been[b])
		print('num has been:', self.has_been[b].sum(0).gt(0).sum().item())
		print('active agents: ', self.active_agent[b])
		print('reward:', self.reward[b])

	def get_env_info(self):
		env_info = {"obs_shape": self.obs_space,
                    "n_actions": self.action_space,
                    "n_agents": self.n_agents,
                    "episode_limit": self.time_limit}
		return env_info