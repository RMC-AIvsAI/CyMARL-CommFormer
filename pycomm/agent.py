"""
Create agents for communication games
"""
import copy

import numpy as np
import torch
from torch.optim import Adam, RMSprop
from torch.nn.utils import clip_grad_norm_

from utils.dotdic import DotDic
from components.dru import DRU

class CNetAgent:
	def __init__(self, opt, device, model, target, index):
		self.opt = opt
		self.game = None
		self.model = model
		self.model_target = target
		self.device = device
		for p in self.model_target.parameters():
			p.requires_grad = False

		self.episodes_seen = 0
		self.dru = DRU(opt.game_comm_sigma, device, opt.model_comm_narrow, opt.game_comm_hard)
		self.id = index
		if opt.optimizer == "adam":
			self.optimizer = Adam(params=model.get_params(), lr=opt.learningrate, weight_decay=0)
		else:
			self.optimizer = RMSprop(params=model.get_params(), lr=opt.learningrate, momentum=opt.momentum)

	def reset(self):
		self.model.reset_parameters()
		self.model_target.reset_parameters()
		self.episodes_seen = 0

	def _eps_flip(self, eps):
		# Sample Bernoulli with P(True) = eps
		return np.random.rand(self.opt.bs_run) < eps

	def _random_choice(self, items):
		return torch.from_numpy(np.random.choice(items, 1)).item()

	def select_action_and_comm(self, action_range, comm_range, q, eps=0, target=False, train_mode=False):
		# eps-Greedy action selector
		if not train_mode:
			eps = 0
		opt = self.opt
		#action_range, comm_range = self.game.get_action_range(opt.game_action_space_total, step, self.id)
		action = torch.zeros(opt.bs_run, dtype=torch.long)
		action_value = torch.zeros(opt.bs_run)
		comm_vector = torch.zeros(opt.bs_run)

		should_select_random_a = self._eps_flip(eps)

		# Get action + comm
		for b in range(opt.bs_run):
			a_range = [i for i, x in enumerate(action_range[b]) if x == 1]
			#a_range = range(action_range[b, 0].item() - 1, action_range[b, 1].item())
			if should_select_random_a[b]:
				action[b] = self._random_choice(a_range)
				action_value[b] = q[b, action[b]]
			else:
				action_value[b], index = q[b, a_range].max(0)
				action[b] = torch.tensor(a_range)[index]
			action[b] = action[b] + 1

			q_c_range = range(opt.game_action_space, opt.game_action_space_total)
			if opt.comm_enabled and comm_range[b, 1].item() > 0:
				comm_vector[b] = self.dru.forward(q[b, q_c_range], train_mode=train_mode) # apply DRU

		return (action, action_value), (comm_vector)

	def episode_loss(self, episode):
		opt = self.opt
		total_loss = torch.zeros(opt.bs)
		for b in range(opt.bs):
			b_steps = episode.steps[b].item()
			for step in range(b_steps):
				record = episode.step_records[step]
				for i in range(opt.game_nagents):
					td_action = 0
					r_t = record.r_t[b][i]
					q_a_t = record.q_a_t[b][i]

					if record.a_t[b][i].item() > 0:
						if record.terminal[b].item() > 0:
							td_action = r_t - q_a_t
						else:
							next_record = episode.step_records[step + 1]
							q_next_max = next_record.q_a_max_t[b][i]
							td_action = r_t + opt.gamma * q_next_max - q_a_t

					loss_t = (td_action ** 2)
					total_loss[b] = total_loss[b] + loss_t
		loss = total_loss.sum()
		loss = loss/(self.opt.bs * self.opt.game_nagents)
		return loss

	def learn_from_episode(self, episode):
		self.optimizer.zero_grad()
		loss = self.episode_loss(episode)
		loss.backward(retain_graph=not self.opt.model_know_share)
		clip_grad_norm_(parameters=self.model.get_params(), max_norm=10)
		self.optimizer.step()

		self.episodes_seen = self.episodes_seen + 1
		if self.episodes_seen % self.opt.step_target == 0:
			self.model_target.load_state_dict(self.model.state_dict())

	def cuda(self):
		self.model.cuda()
		self.model_target.cuda()