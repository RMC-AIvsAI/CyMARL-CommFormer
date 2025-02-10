from functools import partial
import json, csv, copy
import torch
import time

from modules import REGISTRY as cnet_Registry
from pycomm.agent import CNetAgent
from pycomm.arena import Arena
from components.episode import Episode, PlayGame

def init_action_and_comm_bits(opt):
	opt.comm_enabled = opt.game_comm_bits > 0
	if opt.model_comm_narrow is None:
		opt.model_comm_narrow = opt.model_dial
	if not opt.model_comm_narrow and opt.game_comm_bits > 0:
		opt.game_comm_bits = 2 ** opt.game_comm_bits
	return opt

def init_opt(opt):
	if not opt.model_rnn_layers:
		opt.model_rnn_layers = 2
	if opt.model_avg_q is None:
		opt.model_avg_q = True
	if opt.eps_decay is None:
		opt.eps_decay = 1.0
	opt = init_action_and_comm_bits(opt)
	return opt

def create_cnet(opt):
	game_name = opt.game.lower()
	return cnet_Registry[game_name](opt)

def create_agents(opt, device):
	agents = [None] # 1-index agents
	
	# instantiates the neural networks that will train the agents
	cnet = create_cnet(opt)
	cnet_target = copy.deepcopy(cnet)

	# creates the agents and assigns the cnet and cnet_target to the agents
	for i in range(1, opt.game_nagents + 1):
		agents.append(CNetAgent(opt, device, model=cnet, target=cnet_target, index=i))

		# if the model_know_share option is set to False, each agent will have its own cnet and cnet_target models
		if not opt.model_know_share:
			cnet = create_cnet(opt)
			cnet_target = copy.deepcopy(cnet)
	return agents

def save_episode_and_reward_to_csv(file, writer, train_e, train_r, test_e, test_r):
	writer.writerow({'Train_Episode': train_e, 'Train_Reward': train_r, 'Test_Episode': test_e, 'Test_Reward': test_r})
	file.flush()

# Linearly anneal epsilon rate
def linear_schedule(opt, delta, timesteps):
	return max(opt.eps_finish, opt.eps_start - delta * timesteps)
	
def run_trial(opt, env_args, result_path=None, verbose=False):
	# Initialize action and comm bit settings
	opt = init_opt(opt)

	# checks to see if cuda is available, note that the config file must have the device parameter set to 'cuda' for this to work
	device = torch.device("cuda" if opt.device == 'cuda' and torch.cuda.is_available() else "cpu")

	# create the environment, includes starting multiprocessing workers and pipes and updates options to include the environment information
	arena = Arena(opt, env_args, device)

	# create the agents and their associated rnn models
	agents = create_agents(opt, device)

	# if the device is set to cuda, move the agents to the gpu
	if device == torch.device('cuda'):
		for agent in agents:
			if agent is not None:
				agent.cuda()

	# write the trial options and results to a csv file
	test_callback = None
	if result_path:
		result_out = open(result_path + '/trial.csv', 'w')

		csv_meta = '#' + json.dumps(opt) + '\n'

		result_out.write(csv_meta)

		writer = csv.DictWriter(result_out, fieldnames=['Train_Episode', 'Train_Reward', 'Test_Episode', 'Test_Reward'])
		writer.writeheader()
		test_callback = partial(save_episode_and_reward_to_csv, result_out, writer)

	# reset the agents and the buffer
	for agent in agents[1:]:
		agent.reset()
	start_time = time.time()
	buffer = Episode(opt, device)
	rewards = []
	
	# calculate rate of epsilon decay per episode
	delta = (opt.eps_start - opt.eps_finish) / opt.eps_anneal_time
	total_timesteps = 0

	# train the agents
	for e in range(opt.nepisodes):
		# calculate epsilon for the episode
		eps = linear_schedule(opt, delta, total_timesteps)
		buffer.reset()
        
		# run episode
		for batch in range(opt.bs//opt.bs_run):
			episode = arena.run_episode(agents, buffer, eps=eps, train_mode=True)
			buffer.add_episode(episode)
		episode_batch = buffer.combine_episodes()
		total_timesteps += episode_batch.steps.sum().item()
		norm_r = average_reward(opt, episode_batch, opt.bs, normalized=opt.normalized_reward)
		
		# output epoch average steps and reward
		if verbose:
			print('train epoch:', e, 'avg steps:', episode_batch.steps.float().mean().item(), 'avg reward:', norm_r)
		
		# save epoch, average reward, and average steps to csv file
		if test_callback:
			test_callback(e, norm_r, 0, 0)

		# learn from episode
		if opt.model_know_share:
			agents[1].learn_from_episode(episode_batch)
		else:
			for agent in agents[1:]:
				agent.learn_from_episode(episode_batch)

		# read from buffer and output turn, red action, agent actions, reward, message sent,
		# and state for completed episodes. Also outputs the total reward for all steps in the episode.
		# saves the output to a file every 10 episodes
		if e % opt.step_test == 0:
			game = PlayGame(opt, result_path + '/policies/' + str(e) + '.txt')
			game.open_file()
			episode = arena.run_episode(agents, buffer, eps=0, train_mode=False)
			norm_r = average_reward(opt, episode, opt.bs_run, normalized=opt.normalized_reward)
			rewards.append(norm_r)
			if test_callback:
				test_callback(0, 0, e, norm_r)
			game.play_game(episode)
			game.close_file()
			print('TEST EPOCH:', e, 'avg steps:', episode.steps.float().mean().item(), 'avg reward:', norm_r)
	
			if e == opt.nepisodes - 1:
				end_time = time.time()
				game = PlayGame(opt, result_path + '/final.txt')
				game.open_file()
				for _ in range(100):
					episode = arena.run_episode(agents, buffer, eps=0, train_mode=False)
					game.play_game(episode)
				game.close_file()
	total_time = end_time - start_time

	result_out.write("Start Time: " + str(start_time) + "\n")
	result_out.write("End Time: " + str(end_time) + "\n")
	result_out.write("Total_Time: " + str(total_time) + "\n")
	if result_path:
		result_out.close()

def average_reward(opt, episode, batch_size, normalized=True):
    reward = episode.r.sum()/(batch_size * opt.game_nagents)
    if normalized:
        god_reward = episode.game_stats.god_reward.sum()/batch_size
        if reward == god_reward:
            reward = 1
        elif god_reward == 0:
            reward = 0
        else:
            reward = reward/god_reward
    return float(reward)
