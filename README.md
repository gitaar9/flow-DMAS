# Details

DMAS group 1:
<ul>
  <li>Daniel Bick (s3145697)</li>
  <li>Thijs Eker (s2576597)</li>
  <li>Ewout Bergsma (s3441423)</li>
  <li>Thomas Bakker (s2468360)</li>  
</ul>

[Original github repository](https://github.com/flow-project/flow)

[Reference paper](https://arxiv.org/abs/1710.05465)

# Introduction

By using and adjusting the above mentioned orignial github repository we constructed a multi agent traffic simulation, using reinforcement learning. To use our hello world follow these steps:
<ul>
  <li>git clone https://github.com/gitaar9/flow-DMAS.git</li>
  <li>git checkout thijs_improving_the_new_reward_function</li>
  <li>follow the instructions in the "RUNNING_INSTRUCTIONS.sh" file (may want to open it in a text editor)</li>
</ul>

The following two files are created by us:
<ul>
  <li>/flow-DMAS/examples/rllib/multiagent_exps/multiagent_bottleneck.py</li>
  <li>/flow-DMAS/flow/envs/multiagent/bottleneck.py</li>
</ul>
Important to note, the "multiagent_bottleneck.py" is the file that should be ran, if one would want to train the model. The "bottleneck.py" holds the actual functionality, as the other file is mainly used for parameter setting.

# Visualization of the trained model

