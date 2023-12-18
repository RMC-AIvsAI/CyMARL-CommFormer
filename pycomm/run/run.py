from functools import partial
import json, csv, copy

from modules import REGISTRY as cnet_Registry
from learner.agent import CNetAgent
from runner.arena import Arena

def init_action_and_comm_bits(opt):
	opt.comm_enabled = opt.game_comm_bits > 0 and opt.game_nagents > 1
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

def create_agents(opt):
	agents = [None] # 1-index agents
	cnet = create_cnet(opt)
	cnet_target = copy.deepcopy(cnet)
	for i in range(1, opt.game_nagents + 1):
		agents.append(CNetAgent(opt, model=cnet, target=cnet_target, index=i))
		if not opt.model_know_share:
			cnet = create_cnet(opt)
			cnet_target = copy.deepcopy(cnet)
	return agents

def save_episode_and_reward_to_csv(file, writer, e, r):
	writer.writerow({'episode': e, 'reward': r})
	file.flush()

def save_episode_to_json(json_file, episode_index, episode):
    episode_info = {
        "episode_index": episode_index,
        "steps": episode.steps.tolist(),
        "rewards": episode.r.tolist(),
        # Add more fields as needed
    }
    json.dump(episode_info, json_file)
    json_file.write('\n')

def run_trial(opt, env_args, result_path=None, result_path_json=None, verbose=False):
	# Initialize action and comm bit settings
	opt = init_opt(opt)
	arena = Arena(opt, env_args)
	agents = create_agents(arena.opt)
	
	test_callback = None
	if result_path:
		result_out = open(result_path, 'w')
		#result_out_json = open(result_path_json, 'w')

		csv_meta = '#' + json.dumps(opt) + '\n'
		#json_meta = '#' + json.dumps(opt) + '\n'

		result_out.write(csv_meta)
		#result_out_json.write(json_meta)

		writer = csv.DictWriter(result_out, fieldnames=['episode', 'reward'])
		writer.writeheader()
		test_callback = partial(save_episode_and_reward_to_csv, result_out, writer)


	for agent in agents[1:]:
		agent.reset()

	rewards = []
	for e in range(opt.nepisodes):
        # run episode
		episode = arena.run_episode(agents, train_mode=True)
		norm_r = average_reward(opt, episode, normalized=opt.normalized_reward)
		if verbose:
			print('train epoch:', e, 'avg steps:', episode.steps.float().mean().item(), 'avg reward:', norm_r)
		if opt.model_know_share:
			agents[1].learn_from_episode(episode)
		else:
			for agent in agents[1:]:
				agent.learn_from_episode(episode)
		"""
		if result_out_json:
			save_episode_to_json(result_out_json, e, episode)
		"""
		if e % opt.step_test == 0:
			episode = arena.run_episode(agents, train_mode=False)
			norm_r = average_reward(opt, episode, normalized=opt.normalized_reward)
			rewards.append(norm_r)
			if test_callback:
				test_callback(e, norm_r)
			print('TEST EPOCH:', e, 'avg steps:', episode.steps.float().mean().item(), 'avg reward:', norm_r)

	if result_path:
		result_out.close()
	"""
	if result_path_json:
		result_out_json.close()
	"""
def average_reward(opt, episode, normalized=True):
    reward = episode.r.sum()/(opt.bs * opt.game_nagents)
    if normalized:
        god_reward = episode.game_stats.god_reward.sum()/opt.bs
        if reward == god_reward:
            reward = 1
        elif god_reward == 0:
            reward = 0
        else:
            reward = reward/god_reward
    return float(reward)
