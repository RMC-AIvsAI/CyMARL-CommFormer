#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n cymarl python=3.9 -y
# conda activate cymarl
# cd "path/to/cymarl/"

pip install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

#pip install protobuf==3.19.5 sacred==0.8.2 numpy scipy gym==0.26.0 matplotlib seaborn
#pip install pyyaml==5.3.1 pygame==2.3.0 pytest probscale imageio snakeviz tensorboard-logger
# install in cymarl directory
pip install -e ./
pip install git+https://github.com/oxwhirl/smac.git
