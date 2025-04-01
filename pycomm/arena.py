import copy

import numpy as np
import torch
from torch.autograd import Variable

from multiprocessing import Pipe, Process
from envs import REGISTRY as env_REGISTRY # type: ignore
from functools import partial

from utils.dotdic import DotDic # type: ignore

class Arena:
	def __init__(self, opt, env_args, device):
		self.opt = opt
		self.env_args = env_args
		self.device = device

		# create multiprocessing pipes
		self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.opt.bs_run)])

		# create the CyborgEnv function
		env_fn = env_REGISTRY[self.opt.game.lower()]

		# create an empty list to store worker processes
		self.ps = []

		# initialize the worker processes
		for i, worker_conn in enumerate(self.worker_conns):
			ps = Process(target=env_worker, 
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.env_args))))
			self.ps.append(ps)
	
		# start the worker processes in daemon mode (runs in background and exits alongside main program)
		for p in self.ps:
			p.daemon = True
			p.start()
	
		# get environment info
		self.parent_conns[0].send(("get_env_info", None))
		self.env_info = self.parent_conns[0].recv()
		self.possible_actions = self.env_info["possible_actions"]
		# adds the number of agents, action space, observation space, number of steps and hosts per agent to the options variable
		self.update_opt()
		
	def update_opt(self):
		self.opt.game_nagents = self.env_info["n_agents"]
		self.opt.game_action_space = self.env_info["n_actions"]
		self.opt.game_obs_space = self.env_info["obs_shape"]
		self.opt.nsteps = self.env_info["episode_limit"]
		self.opt.hosts_per_agent = self.env_info["hosts_per_agent"]

		if self.opt.comm_enabled:
			self.opt.game_action_space_total = self.opt.game_action_space + 1 # only allowing 1 bit of comms in this version
		else:
			self.opt.game_action_space_total = self.opt.game_action_space

	def reset(self):
		state = torch.zeros(self.opt.bs_run, self.opt.game_nagents, self.opt.game_obs_space).to(self.device)
		# Reset the envs
		for parent_conn in self.parent_conns:
			parent_conn.send(("reset", None))

		# Get the state back
		for bs, parent_conn in enumerate(self.parent_conns):
			data = parent_conn.recv()
			data_dtype = data["state"].dtype
			state[bs] = data["state"]

		return state.to(dtype=data_dtype)
	
	def run_episode(self, agents, buffer, eps, train_mode=False):
		opt = self.opt
		step = 0
		# reset state at time t (s_t)
		s_t = self.reset()

		# create episode variable holding all the data for the episode
		episode = buffer.create_episode(opt.bs_run)

		# appends step records for the initial state, 0 across the board
		episode.step_records.append(buffer.create_step_record(opt.bs_run))
		episode.step_records[-1].s_t = s_t
		episode_steps = train_mode and opt.nsteps + 1 or opt.nsteps

		# Loop through the episode
		while step < episode_steps and episode.ended.sum() < opt.bs_run:
			
			# appends step records for the next state
			episode.step_records.append(buffer.create_step_record(opt.bs_run))

			# Loop through each agent
			for i in range(1, opt.game_nagents + 1):
				# Get received messages per agent per batch
				agent = agents[i]
				agent_idx = i - 1
				comm = None
				if opt.comm_enabled:
					comm = episode.step_records[step].comm.clone()
					mask = torch.ones(opt.game_nagents, dtype=torch.bool)
					mask[agent_idx] = False
					comm = comm[:, mask]

				# Get prev action per batch
				prev_action = None
				if opt.model_action_aware:
					prev_action = torch.ones(opt.bs_run, dtype=torch.long).to(self.device)
					for b in range(opt.bs_run):
						if step > 0 and episode.step_records[step - 1].a_t[b, agent_idx] > 0:
							prev_action[b] = episode.step_records[step - 1].a_t[b, agent_idx]

				# Batch agent index for input into model
				batch_agent_index = torch.zeros(opt.bs_run, dtype=torch.long).fill_(agent_idx).to(self.device)

				agent_inputs = {
					's_t': episode.step_records[step].s_t[:, agent_idx],
					'messages': comm,
					'hidden': episode.step_records[step].hidden[agent_idx, :], # Hidden state
					'prev_action': prev_action,
					'agent_index': batch_agent_index
				}
				episode.step_records[step].agent_inputs.append(agent_inputs)

				# Compute model output (Q function + message bits)
				hidden_t, q_t = agent.model(**agent_inputs)
				episode.step_records[step + 1].hidden[agent_idx] = hidden_t.squeeze()

				# Choose next action and comm using eps-greedy selector
				action_range, comm_range = self.get_action_range(comm, opt.game_action_space, opt.game_action_space_total, step, agent_idx)
				(action, action_value), (comm_vector) = \
					agent.select_action_and_comm(action_range, comm_range, q_t, eps=eps, train_mode=train_mode)

				# Store action + comm
				episode.step_records[step].a_t[:, agent_idx] = action
				episode.step_records[step].q_a_t[:, agent_idx] = action_value
				if opt.comm_enabled:
					episode.step_records[step + 1].comm[:, agent_idx] = comm_vector

			# Update game state
			a_t = episode.step_records[step].a_t
			episode.step_records[step].r_t, episode.step_records[step].terminal, state, info = \
				self.get_step(a_t)
			episode.step_records[step].red_actions.extend(info)

			# Accumulate steps
			if step < opt.nsteps:
				for b in range(opt.bs_run):
					if not episode.ended[b]:
						episode.steps[b] = episode.steps[b] + 1
						episode.r[b] = episode.r[b] + episode.step_records[step].r_t[b]

					if episode.step_records[step].terminal[b]:
						episode.ended[b] = 1

			# Target-network forward pass
			if opt.model_target and train_mode:
				for i in range(1, opt.game_nagents + 1):
					agent_target = agents[i]
					agent_idx = i - 1

					agent_inputs = episode.step_records[step].agent_inputs[agent_idx]
					# import pdb; pdb.set_trace()
					comm_target = agent_inputs.get('messages', None)

					if opt.comm_enabled and opt.model_dial:
						comm_target = episode.step_records[step].comm_target.clone()
						mask = torch.ones(opt.game_nagents, dtype=torch.bool)
						mask[agent_idx] = False
						comm_target = comm_target[:, mask]

					# comm_target.retain_grad()
					agent_target_inputs = copy.copy(agent_inputs)
					agent_target_inputs['messages'] = Variable(comm_target)
					agent_target_inputs['hidden'] = \
						episode.step_records[step].hidden_target[agent_idx, :]
					hidden_target_t, q_target_t = agent_target.model_target(**agent_target_inputs)
					episode.step_records[step + 1].hidden_target[agent_idx] = \
						hidden_target_t.squeeze()

					# Choose next arg max action and comm
					action_range_t, comm_range_t = self.get_action_range(comm_target, opt.game_action_space, opt.game_action_space_total, step, agent_idx)
					(action, action_value), (comm_vector) = \
						agent_target.select_action_and_comm(action_range_t, comm_range_t, q_target_t, eps=0, target=True, train_mode=True)

					# save target actions, comm, and q_a_t, q_a_max_t
					episode.step_records[step].q_a_max_t[:, agent_idx] = action_value
					if opt.model_dial:
						if opt.comm_enabled:
							episode.step_records[step + 1].comm_target[:, agent_idx] = comm_vector

			# Update step
			step = step + 1
			if episode.ended.sum().item() < opt.bs_run:
				episode.step_records[step].s_t = state

		# Collect stats
		episode.game_stats = self.get_stats(episode.steps)

		return episode

	def get_step(self, actions):
		reward = torch.zeros(self.opt.bs_run, self.opt.game_nagents, dtype=torch.float).to(self.device)
		terminal = torch.zeros(self.opt.bs_run, dtype=torch.long).to(self.device)
		state = torch.zeros(self.opt.bs_run, self.opt.game_nagents, self.opt.game_obs_space).to(self.device)
		info = []
		for bs, parent_conn in enumerate(self.parent_conns):
			parent_conn.send(("step", actions[bs]))

		# Get the action range
		for bs, parent_conn in enumerate(self.parent_conns):
			data = parent_conn.recv()
			reward[bs] = data["reward"]
			state[bs] = data["state"]
			data_dtype = data["state"].dtype
			terminal[bs] = data["terminated"]
			info.append(data["info"])

		return reward, terminal, state.to(dtype=data_dtype), info
	
	def get_action_range(self, comm, action_space, a_total, step, agent_idx):
		#TODO incase if implementing valid action check which will result in action space being different for all agents

		action_range = torch.zeros((self.opt.bs_run, action_space), dtype=torch.long).to(self.device)
		comm_range = torch.zeros((self.opt.bs_run, 2), dtype=torch.long).to(self.device)
		for bs, parent_conn in enumerate(self.parent_conns):
			if self.opt.limit_analyse and comm is not None:
				c_data = comm[bs].view(-1).detach()
			else:
				c_data = None
			action_range_data = [c_data, step, agent_idx]
			parent_conn.send(("get_action_range", action_range_data))

		# Get the action range
		for bs, parent_conn in enumerate(self.parent_conns):
			data = parent_conn.recv()

			action_range[bs] = data[0]
			if data[1]:
				comm_range[bs] = torch.tensor([action_space + 1, a_total], dtype=torch.long)
			else:
				comm_range[bs] = torch.tensor([0, 0], dtype=torch.long)
			
		return action_range, comm_range
		
	def get_stats(self, steps):
		stats = DotDic({})
		reward = torch.zeros(self.opt.bs_run).to(self.device)
		for bs, parent_conn in enumerate(self.parent_conns):
			parent_conn.send(("get_stats", steps[bs]))

		# Get the action range
		for bs, parent_conn in enumerate(self.parent_conns):
			data = parent_conn.recv()
			reward[bs] = data["god_reward"]

		stats.god_reward = reward
		return stats
	
def env_worker(remote, env_fn):
    	# Make environment
		env = env_fn.x()
		while True:
			cmd, data = remote.recv()
			if cmd == "step":
				actions = data
				# Take a step in the environment
				reward, terminated, info = env.step(actions)
				# Return the state
				state = env.get_state()
				remote.send({
					# Data for the next timestep needed to pick an action
					"state": state,
					# Rest of the data for the current timestep
					"reward": reward,
					"terminated": terminated,
					"info": info
				})
			elif cmd == "reset":
				state = env.reset()
				remote.send({
					"state": state
				})
			elif cmd == "get_action_range":
				comm = data[0]
				step = data[1]
				agent_id = data[2]
				remote.send(env.get_avail_agent_actions(comm, step, agent_id))
			elif cmd == "close":
				env.close()
				remote.close()
				break
			elif cmd == "get_env_info":
				remote.send(env.get_env_info())
			elif cmd == "get_stats":
				steps = data
				god_reward = env.get_stats(steps)
				remote.send({
					"god_reward": god_reward
				})
			else:
				raise NotImplementedError

class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)