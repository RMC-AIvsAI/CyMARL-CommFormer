import argparse, datetime, copy, os, json
from pathlib import Path

from utils.dotdic import DotDic # type: ignore
from pycomm.run import run_trial, load_model_and_evaluate
	
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
		env_args["wrapper_type"] = "vector"
	else:
		env_args["wrapper_type"] = opt.wrapper_type
	if not opt.action_limiting:
		env_args["action_limiting"] = False
	else:
		env_args["action_limiting"] = opt.action_limiting
	if not opt.game_comm_limited:
		env_args["comm_limiting"] = False
	else:
		env_args["comm_limiting"] = opt.game_comm_limited
	env_args["no_obs"] = False
	
	env_args["episode_trace"] = False

	return env_args

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config_path', type=str, required=True, help='path to existing options file')
	parser.add_argument('-m', '--map_name', type=str, required=True, help='name of the map')
	parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
	parser.add_argument('-s', '--start_index', type=int, default=0, help='starting index for trial output')
	parser.add_argument('-r', '--results', action='store_true', help='stores training results if set')
	parser.add_argument('-v', '--verbose', action='store_true', help='prints training epoch rewards if set')
	parser.add_argument('-l', '--model_path', type=str, help='path to load the model for evaluation')
	
	# parse the command line arguments
	args = parser.parse_args()

	# load the options file
	opt = DotDic(json.loads(open(args.config_path, 'r').read()))

	# create environment arguments
	env_args = construct_env_args(opt, args.map_name)

	if args.model_path:
		model_dir = os.path.dirname(args.model_path)
		evaluation_dir = os.path.join(model_dir, Path('evaluation').stem)
		os.makedirs(evaluation_dir, exist_ok=True)
		load_model_and_evaluate(opt, env_args, args.model_path, evaluation_dir)
	else:
		result_path = None
		if args.results:
			result_path = "results/dial"
			result_path = os.path.join(result_path, Path(args.map_name).stem)
			start_time = str(datetime.datetime.now().strftime("%Y_%m_%d_T%H_%M_%S_%f"))
			result_path = os.path.join(result_path, Path(start_time).stem)
			# Create the results folder if it doesn't exist
			os.makedirs(result_path, exist_ok=True)
			policies = os.path.join(result_path, Path('policies').stem)
			os.makedirs(policies, exist_ok=True)
		for i in range(args.ntrials):
			trial_result_path = None
			
			if result_path:
				trial_result_path = os.path.join(result_path, Path(str(i + args.start_index)).stem)
			trial_opt = copy.deepcopy(opt)
			run_trial(trial_opt, env_args, result_path=result_path, verbose=args.verbose)

