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

Dear assessors, welcome to the Github page of our project. The file you are currently reading is meant to explain all that is required to train, visualize and obtain results of our project. All other <code>README.md</code> files are from the original repository and not our work.

By using and adjusting the above mentioned orignial github repository we constructed a multi agent traffic simulation, using reinforcement learning. The following files with code were created by us for creating the RL agents:
<ul>
  <li><code>examples/rllib/multiagent_exps/multiagent_bottleneck.py</code></li>
  <li><code>flow/envs/multiagent/bottleneck.py</code></li>  
</ul>


The <code>multiagent_bottleneck.py</code> file is the main file to run, if one would want to train the model. The <code>bottleneck.py</code> file holds the environment classes that are used in the aforementioned main file, thus holding code for the state, action and reward. 

The following files hold code that were created by us in order to obtain results:

<ul>
  <li><code>flow/visualize/visualizer_rllib.py</code> 	<- Main file to obtain results</li>
  <li><code>flow/benchmarks/baselines/our_bottleneck.py</code> 	<- Baseline variant of flow/benchmarks/our_bottleneck.py this file can be run</li>
  <li><code>flow/benchmarks/our_bottleneck.py</code> 		<-  The main benchmark file</li>
  <li><code>flow/benchmarks/our_bottleneck_10_perc_av.py</code> 	<- same as our_bottleneck.py but with 10% AVs</li>
  <li><code>flow/benchmarks/rllib/our_ppo_runner.py</code> 	<- The file to run the benchmark with PPO agents<li>
</ul>

Example of command in order to run the code for obtaining results, running from root of this project:
<code>python3 flow/visualize/visualizer_rllib.py LOCATION NCHECKPOINT</code> EDDIIITT!!!!

Additionally, our report can also be found on this page: <code>Report.pdf</code>.

# Train model
In order to train a model Google Colab can be used. Following these instructions:
<ul>
    <li>Download the <code>Google_Colab.ipynb</code> found on this page</li>
    <li>Go to the Google Colab website: https://colab.research.google.com</li>
    <li>Log in to a Google account (if not logged in already)</li>
    <li>Upload the file that was uploaded in step 1 (file > upload notebook > navigate to downloaded file and select)</li>
    <li>Follow the instructions in the file that has just opened in the browser</li>
</ul>

# Install project locally

Before installing the project that there are two videos uploaded to Youtube, for visualization purposes. Additionally one can train the model using Google Colab. If one would still want to install the project locally, follow these steps:
<ul>
  <li>Git clone https://github.com/gitaar9/groupA1-coop-av-flow.git</li>
  <li>Navigate to the base directory of git repo</li>
  <li>Follow the instructions in the <code>Install_locally.sh</code> file (may want to open it in a text editor)
</ul>

# Visualization of the trained model

For visualization of the simulation there are two options. Firstly, we have uploaded two visualization videos to Youtube:
<ul>
  <li>Reward function r<sub>velocity</sub> with 50% AV: https://youtu.be/O65Y8ObD3qI</li>
  <li>Reward function r<sub>time</sub> with 10% AV: https://youtu.be/np_Wf7nnfUo</li>
</ul>

The second option is to render visualizations locally. This requires to install the project locally first. Afterwards follow these instructions:

<ul>
  <li>Navigate to the root of this project</li>
  <li>Activate conde env: <code>conda activate flow</code></li>
  <li>
    Run sumo with the model stored at iteration 300:<br>
    <code>python3 flow/visualize/visualizer_rllib.py trained_model_09-29 300</code>
  </li>
  <li>In sumo click the green play button 3 times (we think this has to do with the fact we use 4 cores)</li>
  <li>When the play button is pressed the 4th time, the simulation will run (for an average of 200 steps)</li>
  <li>With the step button next to the play button, you can step through the simulation to get a better view of what the cars are doing.</li>
</ul>

The red cars are the cars controlled by the trained model, white cars are controlled by a simple algorithm designed to mimick human behaviour (provided by the flow library).
