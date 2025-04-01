# Learning to Communicate In Cooperative MARL for Autonomous Cyber Operations

This codebase is adapted from the switch riddle game from [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://github.com/minqi/learning-to-communicate-pytorch).

## More info
More generally, `main.py` takes multiple arguments:

| Arg | Short | Description | Required? |
| ------ | ------ | ------- | ------- | 
| --config_path | -c | path to JSON configuration file | ✅ |
| --map_name | -m | CybORG scenario to choose | ✅ |
| --ntrials | -n | number of trials to run | - |
| --start_index | -s | start-index used as suffix in result filenames | - |
| --verbose | -v | prints results per training epoch to stdout if set | - |
| --results | -r | if set will save the results in the results folder | - |

##### Configuration
JSON configuration files passed to `main.py` should consist of the following key-value pairs:

| Key | Description | Type |
| ------ | ------ | ------- |
| game | name of the game, e.g. "cyborg" | string |
| game_comm_limited | true if only some agents can communicate at each step | bool |
| comm_enabled | true if communication is enabled | bool |
| game_comm_sigma | standard deviation of Gaussian noise applied by DRU | float |
| game_comm_hard | true if use hard discretization, soft approximation otherwise | bool |
| nsteps | maximum number of game steps | int |
| gamma | reward discount factor for Q-learning | float |
| model_dial | true if agents should use DIAL | bool |
| model_comm_narrow | true if DRU should use sigmoid for regularization, softmax otherwise | bool |
| model_target | true if learning should use a target Q-network | bool |
| model_bn | true if learning should use batch normalization | bool |
| model_know_share | true if agents should share parameters | bool |
| model_action_aware | true if each agent should know their last action | bool |
| model_rnn_size | dimension of rnn hidden state | int |
| bs | batch size of episodes | int |
| bs_run | number of parallel environments | int |
| learningrate | learning rate for optimizer (RMSProp) | float |
| momentum | momentum for optimizer (RMSProp) | float |
| eps_start | exploration rate for epsilon-greedy exploration at the start of the game | float |
| eps_finish | exploration rate for epsilon-greedy exploration to anneal to | float |
| eps_anneal_time | timesteps to anneal the exploration rate to eps_finish | float |
| nepisodes | number of epochs, each consisting of <bs> parallel episodes | int |
| step_test | perform a test episode every this many steps | int |
| step_target | update target network every this many steps | int |