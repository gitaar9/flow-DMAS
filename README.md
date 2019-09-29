# Details

DMAS group A1:
<ul>
  <li>Daniel Bick (s3145697)</li>
  <li>Thijs Eker (s2576597)</li>
  <li>Ewout Bergsma (s3441423)</li>
  <li>Thomas Bakker (s2468360)</li>  
</ul>

[Original github repository](https://github.com/flow-project/flow)

[Reference paper](https://arxiv.org/abs/1710.05465)

# Introduction

By using and adjusting the above mentioned orignial github repository we constructed a multi agent traffic simulation, using reinforcement learning. To install the project follow these steps:
<ul>
  <li>git clone https://github.com/gitaar9/flow-DMAS.git</li>
  <li>git checkout thijs_improving_the_new_reward_function</li>
  <li>follow the instructions in the "RUNNING_INSTRUCTIONS.sh" file (may want to open it in a text editor)</li>
</ul>

The following two files are created by us:
<ul>
  <li>examples/rllib/multiagent_exps/multiagent_bottleneck.py</li>
  <li>flow/envs/multiagent/bottleneck.py</li>
</ul>
Important to note, the "multiagent_bottleneck.py" is the file that should be ran, if one would want to train the model. The "bottleneck.py" holds the actual functionality, as the other file is mainly used for parameter setting. Additionally, our report can also be found on this page: "Report_alphaversion.pdf".

# Visualization of the trained model

To visualize the simulation, follow these steps:
<ul>
  <li>Navigate to the base directory of git repo</li>
  <li>Activate conde env: conda activate flow</li>
  <li>Run sumo with the model stored at iteration 300: python3 flow/visualize/visualizer_rllib.py trained_model_09-29 300</li>
  <li>In sumo click the green play button 3 times (we think this has to do with the fact we use 4 cores)</li>
  <li>When the play button is pressed the 4th time, the simulation will run (for an average of 200 steps)</li>
  <li>With the step button next to the play button, you can step through the simulation to get a better view of what the cars are doing.</li>
</ul>

The red cars are the cars controlled by the trained model, white cars are controlled by a simple algorithm designed to mimick human behaviour (provided by the flow library). Forcing the simulation to go beyond the initial ~200 steps will make yellow cars enter the simulation, currently the reason is unknown.
