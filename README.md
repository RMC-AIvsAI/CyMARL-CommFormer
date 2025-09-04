## Overview
This codebase adapts 2 libraries: CyMARL and CommFormer to train Multiple Agents in Autonomous Cyber Operations (CybORG). Collectively this work will be called CyMARL-CommFormer. 

More information can be found in the respective library folders: [CyMARL](pycomm/) and [CommFormer](commformer/).

This work builds upon the work completed by Faizan Contractor and Dr. Ranwa Al Mallah in Learning to Communicate in Multi-Agent Reinforcement Learning for Autonomous Cyber Defence.

## Installation instructions
Codebase has been tested using Windows 11. To install, begin with a new Anaconda environment following these steps:

### Using the GUI
1. Install Anaconda Navigator
2. Click on the Environments tab in Anaconda Navigator
3. Click on the Create button, choose a name for the environment and choose python version 3.8.x. (3.9.x is the expected python version for CyMARL but 3.8.x, the default required version for CommFormer, has been tested and confirmed working)
4. Click on the Home tab once the environment is created and select the application of your choice to continue installation (anaconda_prompt terminal, Windows terminal, or VS Code terminal - enter the commands seen in the Package Installation via its terminal after ensuring the correct environment is loaded)

### Using a terminal of your choice
```
conda create -n cymarl-commformer python=3.8 -y #(3.9 also works)
conda activate cymarl-commformer
```

### Package Installation
```
# clone the repository
git clone https://github.com/alexwtp/CyMARL-CommFormer.git

# install CyMARL-CommFormer (includes PyMARL, DIAL, CybORG and CommFormer)
# change directory to the folder containing the cloned CyMARL-CommFormer
pip install -r requirements.txt
```
To verify the installation of CyMARL-CommFormer, run an experiement using the command line below:
```
python .\commformer\train\train_cymarl_comm.py --env_name "CybORG" --algorithm_name "commformer_dec" --scenario_name "confidentiality_small"` --num_agents 2 --eval_episode_length 10 --n_rollout_threads 4 --seed 4 --episode_length 30 --num_env_steps 1000 --log_interval 10
```

## Experimentation
To run experiments on CyMARL-CommFormer use either pre-built scripts found in commformer\scripts or via command line: 

2 homogeneous CommFormer agents, 2 million steps. Used to establish homogeneous baseline.
```
python .\train\train_cymarl_comm.py --env_name "CybORG" --algorithm_name "commformer_dec" --scenario_name "confidentiality_small"`
--num_agents 2 --eval_episode_length 10 --n_rollout_threads 4 --seed 4 --episode_length 30 --num_env_steps 2000000 --log_interval 10
```

## Baseline Results and Analysis
Baseline results for CyMARL-CommFormer can be found in the [Baseline Results](commformer/baseline_results/) folder.

A Jupyter Notebook containing the analysis conducted using these results can be found in [Analysis](commformer_experiment_analysis.ipynb).
