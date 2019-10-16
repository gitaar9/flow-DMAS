#!/bin/bash

#Dear assessor, this file is meant to explain the steps to follow in order to install everything that is required for our project. This file was tested on a clean install of Ubuntu 18, however it should also work on Ubuntu 16 (with changing out the last bash script at the end of this file, see their comments). Run the following commands sequentially, or run this script by using "source ProjectInstall.sh".

sudo apt-get install curl -y							#If not installed already
sudo apt-get install git -y							#If not installed already
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh 						#Bunch of enters required, then type 'yes', followed by another enter and finally type another 'yes'

#Kill current terminal and open a new one!

source ~/.bashrc
git clone https://github.com/gitaar9/flow-DMAS.git				#git clone https://github.com/flow-project/flow.git
cd flow-DMAS
conda env create -f environment.yml
source activate flow 								#conda activate flow
pip install -e .
#scripts/setup_sumo_ubuntu1604.sh 						#For ubuntu version 16
scripts/setup_sumo_ubuntu1804.sh  						#For ubuntu version 18

#Kill current terminal and open a new one!

#The following will start training the model, for more information see the README.md found on github https://github.com/gitaar9/flow-DMAS/tree/thijs_improving_the_new_reward_function
source activate flow
python3 examples/rllib/multiagent_exps/multiagent_bottleneck.py

#The following command can be used to remove the conda environment, if so desired:
#conda env remove -n flow

#Sources:
#https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart
#https://flow.readthedocs.io/en/latest/flow_setup.html
