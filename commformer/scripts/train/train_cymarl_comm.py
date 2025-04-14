#!/usr/bin/env python
import sys
import os
import wandb # type: ignore
import socket
import setproctitle # type: ignore
import numpy as np
from pathlib import Path
import torch
# Dynamically determine the path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent.parent))
from commformer.config import get_config
from pycomm.envs.cyborg.cyborg_env import CyborgEnv as CyborgEnv
from commformer.runner.shared.cyborg_runner import CybORGRunner as Runner
from commformer.envs.env_wrappers import CybORG_SubprocVecEnv as SubprocVecEnv
from commformer.envs.env_wrappers import CybORG_DummyVecEnv as DummyVecEnv

"""Train script for CyMARL-Commformer."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "CybORG":
                # Extract relevant arguments from all_args
                map_name = all_args.scenario_name # cyborg expects map name variable not scenario name
                # If CybORG time limit is not specified, use the episode length
                if all_args.time_limit is None:
                    time_limit = all_args.episode_length
                else:
                    time_limit = all_args.time_limit
                action_limiting = all_args.action_limiting
                wrapper_type = all_args.wrapper_type
                
                # Pass the extracted arguments to CyborgEnv
                env = CyborgEnv(map_name=map_name, time_limit=time_limit, action_limiting=action_limiting, wrapper_type=wrapper_type, use_CommFormer=True)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        # single environment wrapper - only for debugging
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "CybORG":
                # Extract relevant arguments from all_args
                map_name = all_args.scenario_name # cyborg expects map name variable not scenario name
                # If CybORG time limit is not specified, use the episode length
                if all_args.time_limit is None:
                    time_limit = all_args.episode_length
                else:
                    time_limit = all_args.time_limit
                action_limiting = all_args.action_limiting
                wrapper_type = all_args.wrapper_type
                
                # Pass the extracted arguments to CyborgEnv
                env = CyborgEnv(map_name=map_name, time_limit=time_limit, action_limiting=action_limiting, wrapper_type=wrapper_type, use_CommFormer=True)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        # single environment wrapper - only for debugging
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    # environment specific arguments
    parser.add_argument('--scenario_name', type=str, default='confidentiality_small', 
                        help="Which scenario to run on")
    parser.add_argument('--tensor_obs', action="store_true", default=False,
                        help="Do you want a tensor observation")
    parser.add_argument('--eval_episode_length', type=int, default=20)
    parser.add_argument('--time_limit', type=int, 
                        help='Underlying CybORG environment episode limit. In CyMARL, this is the number of steps per episode.')
    parser.add_argument('--action_limiting', type=bool, default=False, 
                        help='CybORG action limiting/masking flag')
    parser.add_argument('--wrapper_type', type=str, default='vector', 
                        help='CybORG wrapper type')

    # uses argparse library functions, imported in get_config, to parse the arguments
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # load general hyperparameters for commformer
    parser = get_config()
    # add environment specific parameters to the parser and parse the command line
    all_args = parse_args(args, parser)

    if "dec" in all_args.algorithm_name:
        all_args.dec_actor = True
        all_args.share_actor = False


    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        import time
        timestr = time.strftime("%y%m%d-%H%M%S")
        curr_run = all_args.prefix_name + "-" + timestr
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.alg_seed)
    torch.cuda.manual_seed_all(all_args.alg_seed)
    np.random.seed(all_args.alg_seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
