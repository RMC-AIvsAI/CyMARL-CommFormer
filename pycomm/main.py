import argparse, datetime, copy, os, json
from pathlib import Path

from utils.dotdic import DotDic
from run.run import run_trial
	
def construct_env_args(opt, map_name):
	env_args = {}
	env_args["key"] = opt.game.lower()
	if map_name:
		env_args["map_name"] = map_name
	else:
		env_args["map_name"] = opt.game.lower()
	if not opt.nsteps:
		env_args["time_limit"] = 30
	else:
		env_args["time_limit"] = opt.nsteps
	if not opt.wrapper_type:
		env_args["wrapper_type"] = "table"
	else:
		env_args["wrapper_type"] = opt.wrapper_type
	env_args["no_obs"] = False
	env_args["action_masking"] = False
	env_args["episode_trace"] = False

	return env_args

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config_path', type=str, help='path to existing options file')
	parser.add_argument('-r', '--results_path', type=str, help='path to results directory')
	parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
	parser.add_argument('-s', '--start_index', type=int, default=0, help='starting index for trial output')
	parser.add_argument('-v', '--verbose', action='store_true', help='prints training epoch rewards if set')
	parser.add_argument('-m', '--map_name', type=str, help='name of the map')
	args = parser.parse_args()

	opt = DotDic(json.loads(open(args.config_path, 'r').read()))

	# Create environment arguments
	env_args = construct_env_args(opt, args.map_name)

	result_path = None
	if args.results_path:
		result_path = args.map_name and os.path.join(args.results_path, Path(args.map_name).stem) or \
			os.path.join(args.results_path, Path(args.config_path).stem)

	for i in range(args.ntrials):
		trial_result_path = None
		trial_result_path_json = None
		if result_path:
			trial_result_path = result_path + '_' + str(i + args.start_index) + '_' + str(datetime.datetime.now().strftime("%Y_%m_%d_T%H_%M_%S_%f")) + '.csv'
			trial_result_path_json = result_path + '_' + str(i + args.start_index) + '_' + str(datetime.datetime.now().strftime("%Y_%m_%d_T%H_%M_%S_%f")) + '.json'
		trial_opt = copy.deepcopy(opt)
		run_trial(trial_opt, env_args, result_path=trial_result_path, result_path_json=trial_result_path_json, verbose=args.verbose)

