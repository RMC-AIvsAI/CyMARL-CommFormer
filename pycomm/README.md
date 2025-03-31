# This Readme file version has been replaced by the one found in the CyMARL-CommFormer root folder.
# This version is kept for reference purposes only.
## Overview
This codebase adapts 2 libraries: CyMARL and DIAL to train Multiple Agents in Autonomous Cyber Operations (CybORG). Collectively will be called CyMARL. More information can be found in the respective library folders: [CyMARL](pymarl2/) and [DIAL](pycomm/).



## Installation instructions
Codebase has been tested using Windows 10 and Windows 11. To install, begin with a new Anaconda environment following these steps:
```
conda create -n cymarl python=3.9 -y
conda activate cymarl
# get pytorch
pip install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# install CyMARL and DIAL (includes PyMARL, DIAL and CybORG)
cd "path/to/cymarl/"
pip install -e ./
# install SMAC dependencies
pip install git+https://github.com/oxwhirl/smac.git
```
To verify the installation of PyMARL, run an experiement:
```
python pymarl2\main.py --config=qmix_predator_prey --env-config=stag_hunt with env_args.map_name=stag_hunt
```
To verify the installation of DIAL, run an experiement:
```
python pycomm/main.py -c pycomm/config/cyborg_dial.json -m confidentiality_small -r
```
To verify the installation and configuration of CybORG, run the debugging script:
```
python runner\basic_marl.py
```

## Experimentation
To run experiments on QMix algorithm: 
```
python pymarl2\main.py --config=qmix --env-config=cyborg with env_args.map_name=confidentiality_small
```
Note that experiments will default to 1M timesteps, this can be changed by modifying the `t_max` value.
```
python pymarl2\main.py --config=qmix --env-config=cyborg with env_args.map_name=confidentiality_small t_max=250000
```
For running sequential experiements, batch scripts are used which pull parameters from files in `\runner\config\`. The output of each experiment is stored in `results\sacred\<map_name>\`. Best practice is to add a `name` value to the python command to avoid mixing up experiments using the same map.

To run experiments on DIAL algorithm: 
```
python pycomm/main.py -c pycomm/config/cyborg_dial.json -m confidentiality_small -r
```