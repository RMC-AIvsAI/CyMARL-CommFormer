# CyMARL: A MARL extension of CybORG

CyMARL is a [PyMARL](https://github.com/oxwhirl/pymarl) environment that extends [CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2) for cooperative multi-agent reinforcement learning. This repository is adapted from [PyMARL2](https://github.com/hijkzzz/pymarl2/) and [CybORG v3.1](https://github.com/cage-challenge/CybORG/tree/dd586a39b129fb21b7ef4c15d388ad809f24882f).

## Installation instructions
CyMARL has been tested using Windows 10. To install, begin with a new Anaconda environment following these steps:
```
conda create -n cymarl python=3.9 -y
conda activate cymarl
# get pytorch
pip install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# install CyMARL (includes PyMARL and CybORG)
cd "path/to/cymarl/"
pip install -e ./
# install SMAC dependencies
pip install git+https://github.com/oxwhirl/smac.git
```
To verify the installation of PyMARL, run an experiement:
```
python pymarl2\main.py --config=qmix_predator_prey --env-config=stag_hunt with env_args.map_name=stag_hunt
```
To verify the installation and configuration of CybORG, run the debugging script:
```
python runner\basic_marl.py
```

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
