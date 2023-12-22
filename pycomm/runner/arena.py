import copy

import numpy as np
import torch
from torch.autograd import Variable

from multiprocessing import Pipe, Process
from envs import REGISTRY as env_REGISTRY
from functools import partial

from components.episode import Episode
from utils.dotdic import DotDic

class Arena:
	def __init__(self, opt, env_args):
		self.opt = opt
		self.env_args = env_args
		self.eps = opt.eps_finish
		self.device = torch.device("cuda" if opt.device == 'cuda' and torch.cuda.is_available() else "cpu")
		self.episode = Episode(opt, self.device)

		self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.opt.bs)])
		env_fn = env_REGISTRY[self.opt.game.lower()]
		self.ps = []

		for i, worker_conn in enumerate(self.worker_conns):
			ps = Process(target=env_worker, 
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.env_args))))
			self.ps.append(ps)
	
		for p in self.ps:
			p.daemon = True
			p.start()
	
		self.parent_conns[0].send(("get_env_info", None))
		self.env_info = self.parent_conns[0].recv()
	
		self.update_opt()
		
	def update_opt(self):
		self.opt.game_nagents = self.env_info["n_agents"]
		self.opt.game_action_space = self.env_info["n_actions"]
		self.opt.game_obs_space = self.env_info["obs_shape"]
		self.opt.nsteps = self.env_info["episode_limit"]

		if self.opt.comm_enabled:
			self.opt.game_action_space_total = self.opt.game_action_space + self.opt.game_comm_bits
		else:
			self.opt.game_action_space_total = self.opt.game_action_space

	def reset(self):
		state = torch.zeros(self.opt.bs, self.opt.game_nagents, self.opt.game_obs_space).to(self.device)
		# Reset the envs
		for parent_conn in self.parent_conns:
			parent_conn.send(("reset", None))

		# Get the state back
		for bs, parent_conn in enumerate(self.parent_conns):
			data = parent_conn.recv()
			data_dtype = data["state"].dtype
			state[bs] = data["state"]

		return state.to(dtype=data_dtype)
	
	def run_episode(self, agents, train_mode=False):
		self.episode.reset()
		episode = self.episode

		opt = self.opt
		self.eps = self.eps * opt.eps_decay

		step = 0
		s_t = self.reset()
		episode.step_records.append(episode.create_step_record())
		episode.step_records[-1].s_t = s_t
		episode_steps = train_mode and opt.nsteps + 1 or opt.nsteps
		while step < episode_steps and episode.ended.sum() < opt.bs:
			episode.step_records.append(episode.create_step_record())

			for i in range(1, opt.game_nagents + 1):
				# Get received messages per agent per batch
				agent = agents[i]
				agent_idx = i - 1
				comm = None
				if opt.comm_enabled:
					comm = episode.step_records[step].comm.clone()
					comm_limited = self.get_comm_limited(step, agent.id)
					if comm_limited is not None:
						comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
						for b in range(opt.bs):
							if comm_limited[b].item() > 0:
								comm_lim[b] = comm[b][comm_limited[b] - 1]
						comm = comm_lim
					else:
						mask = torch.ones(opt.game_nagents, dtype=torch.bool)
						mask[agent_idx] = False
						comm = comm[:, mask]
						#comm[:, agent_idx].zero_()

				# Get prev action per batch
				prev_action = None
				if opt.model_action_aware:
					prev_action = torch.ones(opt.bs, dtype=torch.long).to(self.device)
					if not opt.model_dial:
						prev_message = torch.ones(opt.bs, dtype=torch.long)
					for b in range(opt.bs):
						if step > 0 and episode.step_records[step - 1].a_t[b, agent_idx] > 0:
							prev_action[b] = episode.step_records[step - 1].a_t[b, agent_idx]
						if not opt.model_dial:
							if step > 0 and episode.step_records[step - 1].a_comm_t[b, agent_idx] > 0:
								prev_message[b] = episode.step_records[step - 1].a_comm_t[b, agent_idx]
					if not opt.model_dial:
						prev_action = (prev_action, prev_message)

				# Batch agent index for input into model
				batch_agent_index = torch.zeros(opt.bs, dtype=torch.long).fill_(agent_idx).to(self.device)

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
				action_range, comm_range = self.get_action_range(opt.game_action_space_total, step, agent_idx)
				(action, action_value), (comm_vector, comm_action, comm_value) = \
					agent.select_action_and_comm(action_range, comm_range, q_t, eps=self.eps, train_mode=train_mode)

				# Store action + comm
				episode.step_records[step].a_t[:, agent_idx] = action
				episode.step_records[step].q_a_t[:, agent_idx] = action_value
				episode.step_records[step + 1].comm[:, agent_idx] = comm_vector
				if not opt.model_dial:
					episode.step_records[step].a_comm_t[:, agent_idx] = comm_action
					episode.step_records[step].q_comm_t[:, agent_idx] = comm_value

			# Update game state
			a_t = episode.step_records[step].a_t
			episode.step_records[step].r_t, episode.step_records[step].terminal, state = \
				self.get_step(a_t)

			# Accumulate steps
			if step < opt.nsteps:
				for b in range(opt.bs):
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
						comm_limited = self.get_comm_limited(step, agent.id)
						if comm_limited is not None:
							comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
							for b in range(opt.bs):
								if comm_limited[b].item() > 0:
									comm_lim[b] = comm_target[b][comm_limited[b] - 1]
							comm_target = comm_lim
						else:
							mask = torch.ones(opt.game_nagents, dtype=torch.bool)
							mask[agent_idx] = False
							comm_target = comm_target[:, mask]
							#comm_target[:, agent_idx].zero_()

					# comm_target.retain_grad()
					agent_target_inputs = copy.copy(agent_inputs)
					agent_target_inputs['messages'] = Variable(comm_target)
					agent_target_inputs['hidden'] = \
						episode.step_records[step].hidden_target[agent_idx, :]
					hidden_target_t, q_target_t = agent_target.model_target(**agent_target_inputs)
					episode.step_records[step + 1].hidden_target[agent_idx] = \
						hidden_target_t.squeeze()

					# Choose next arg max action and comm
					action_range_t, comm_range_t = self.get_action_range(opt.game_action_space_total, step, agent_idx)
					(action, action_value), (comm_vector, comm_action, comm_value) = \
						agent_target.select_action_and_comm(action_range_t, comm_range_t, q_target_t, eps=0, target=True, train_mode=True)

					# save target actions, comm, and q_a_t, q_a_max_t
					episode.step_records[step].q_a_max_t[:, agent_idx] = action_value
					if opt.model_dial:
						episode.step_records[step + 1].comm_target[:, agent_idx] = comm_vector
					else:
						episode.step_records[step].q_comm_max_t[:, agent_idx] = comm_value

			# Update step
			step = step + 1
			if episode.ended.sum().item() < opt.bs:
				episode.step_records[step].s_t = state

		# Collect stats
		episode.game_stats = self.get_stats(episode.steps)

		return episode

	def get_step(self, actions):
		reward = torch.zeros(self.opt.bs, self.opt.game_nagents, dtype=torch.float).to(self.device)
		terminal = torch.zeros(self.opt.bs, dtype=torch.long).to(self.device)
		state = torch.zeros(self.opt.bs, self.opt.game_nagents, self.opt.game_obs_space).to(self.device)
		for bs, parent_conn in enumerate(self.parent_conns):
			parent_conn.send(("step", actions[bs]))

		# Get the action range
		for bs, parent_conn in enumerate(self.parent_conns):
			data = parent_conn.recv()
			reward[bs] = data["reward"]
			state[bs] = data["state"]
			data_dtype = data["state"].dtype
			terminal[bs] = data["terminated"]

		return reward, terminal, state.to(dtype=data_dtype)
	
	def get_action_range(self, a_total, step, agent_idx):
		#TODO incase if implementing valid action check which will result in action space being different for all agents

		action_range = torch.zeros((self.opt.bs, 2), dtype=torch.long).to(self.device)
		comm_range = torch.zeros((self.opt.bs, 2), dtype=torch.long).to(self.device)
		action_range_data = [a_total, step, agent_idx]
		for parent_conn in self.parent_conns:
			parent_conn.send(("get_action_range", action_range_data))

		# Get the action range
		for bs, parent_conn in enumerate(self.parent_conns):
			data = parent_conn.recv()

			action_range[bs] = data[0]
			comm_range[bs] = data[1]
			
		return action_range, comm_range
	
	def get_comm_limited(self, step, agent_idx):
		#TODO for limiting communication to only agents who observe compromised hosts

		if self.opt.game_comm_limited:
			comm_limited = torch.zeros(self.opt.bs, dtype=torch.long).to(self.device)
			comm_data = [step, agent_idx]
			for parent_conn in self.parent_conns:
				parent_conn.send(("get_comm", comm_data))

			for bs, parent_conn in enumerate(self.parent_conns):
				c_l = parent_conn.recv()
				comm_limited[bs] = c_l
			return comm_limited
		return None
		
	def get_stats(self, steps):
		stats = DotDic({})
		reward = torch.zeros(self.opt.bs).to(self.device)
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
				reward, terminated = env.step(actions)
				# Return the state
				state = env.get_state()
				remote.send({
					# Data for the next timestep needed to pick an action
					"state": state,
					# Rest of the data for the current timestep
					"reward": reward,
					"terminated": terminated,
					#"info": env_info
				})
			elif cmd == "reset":
				state = env.reset()
				remote.send({
					"state": state
				})
			elif cmd == "get_comm":
				step = data[0]
				agent_id = data[1]
				remote.send(env.get_comm_limited(step, agent_id))
			elif cmd == "get_action_range":
				a_total = data[0]
				step = data[1]
				agent_id = data[2]
				remote.send(env.get_action_range(a_total, step, agent_id))
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