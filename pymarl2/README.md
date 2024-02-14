# CyMARL: A MARL extension of CybORG

CyMARL is a [PyMARL](https://github.com/oxwhirl/pymarl) environment that extends [CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2) for cooperative multi-agent reinforcement learning. This repository is adapted from [PyMARL2](https://github.com/hijkzzz/pymarl2/) and [CybORG v3.1](https://github.com/cage-challenge/CybORG/tree/dd586a39b129fb21b7ef4c15d388ad809f24882f).


## Experimentation
CyMARL is run via command line interface as a PyMARL environment. To run a single experiment, the key "cyborg" is used as the `env-config` value and a map name must be provided.
```
python pymarl2\main.py --config=qmix --env-config=cyborg with env_args.map_name=confidentiality_small
```
Note that experiments will default to 1M timesteps, this can be changed by modifying the `t_max` value.
```
python pymarl2\main.py --config=qmix --env-config=cyborg with env_args.map_name=confidentiality_small t_max=250000
```
For running sequential experiements, batch scripts are used which pull parameters from files in `\runner\config\`. The output of each experiment is stored in `results\sacred\<map_name>\`. Best practice is to add a `name` value to the python command to avoid mixing up experiments using the same map.
