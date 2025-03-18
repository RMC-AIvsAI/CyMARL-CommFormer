## Overview
This codebase adapts 2 libraries: CyMARL and CommFormer to train Multiple Agents in Autonomous Cyber Operations (CybORG). Collectively this work will be called CyMARL-CommFormer. 

More information can be found in the respective library folders: [CyMARL](pycomm/) and [CommFormer](commformer/).

This work builds upon the work completed in DIAL.

## Installation instructions
Codebase has been tested using Windows 10 and Windows 11. To install, begin with a new Anaconda environment following these steps:

### Using the GUI
1. Install Anaconda Navigator
2. Click on the Environments tab in Anaconda Navigator
3. Click on the Create button, choose a name for the environment and choose python version 3.9.x (3.8.x for CommFormer-TBC)
4. Click on the Home tab once the environment is created and select the application of your choice to continue installation (anaconda_prompt terminal, Windows terminal, or VS Code terminal - enter the commands seen in the Package Installation via its terminal after ensuring the correct environment is loaded)

### Using a terminal of your choice
```
conda create -n cymarl python=3.9 -y #(3.8 for CommFormer-TBC)
conda activate cymarl
```

### Package Installation
```
# get pytorch
pip install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# clone the repository
git clone https://github.com/alexwtp/CyMARL-CommFormer.git

# install CyMARL and DIAL (includes PyMARL, DIAL and CybORG)
# change directory to the folder containing CyMARL
cd .\CyMARL-CommFormer\
pip install -e ./
# install SMAC dependencies
pip install git+https://github.com/oxwhirl/smac.git
```
To verify the installation of PyMARL, run an experiement:
```
python pymarl2\main.py --config=qmix_predator_prey --env-config=stag_hunt with env_args.map_name=stag_hunt t_max=10500  
```
To verify the installation of DIAL, run an experiement:
```
python pycomm/main.py -c pycomm/config/cyborg_dial.json -m confidentiality_small -r
```
To verify the installation and configuration of CybORG, run the debugging script:
Note: This script no longer functions due to changes made during the implementation of DIAL. No current plans to fix this.
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
# note that this command will run into an error (AttributeError: 'BlueTableDIALWrapper' object has no attribute 'close') when completing the training episode
python pymarl2\main.py --config=qmix --env-config=cyborg with env_args.map_name=confidentiality_small t_max=25000
```
For running sequential experiements, batch scripts are used which pull parameters from files in `\runner\config\`. The output of each experiment is stored in `results\sacred\<map_name>\`. Best practice is to add a `name` value to the python command to avoid mixing up experiments using the same map.

To run experiments on DIAL algorithm: 
```
python pycomm/main.py -c pycomm/config/cyborg_dial.json -m confidentiality_small -r
```
