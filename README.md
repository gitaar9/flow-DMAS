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

Dear assessors, welcome to the Github page of our project. The file you are currently reading is meant to explain all that is required to train, visualize and obtain results of our project. All other README.md files are from the original repository and not our work.

By using and adjusting the above mentioned orignial github repository we constructed a multi agent traffic simulation, using reinforcement learning. The following two files were created by us:
<ul>
  <li>examples/rllib/multiagent_exps/multiagent_bottleneck.py</li>
  <li>flow/envs/multiagent/bottleneck.py</li>
  <li>BENCHMARK FILE THIJS</li>
</ul>
The "multiagent_bottleneck.py" file is the main file to run, if one would want to train the model. The "bottleneck.py" file holds the environment classes that are used in the aforementioned main file, thus holding code for the state, action and reward. Furthermore, BENCHMARK FILE(S)! Additionally, our report can also be found on this page: "Report.pdf".

# Train model
In order to train a model Google Colab can be used. Following these instructions:
<ul>
    <li>Download the "Google_notebook.ipynb" found on this page</li>
    <li>Go to the [Google Colab website](https://colab.research.google.com)</li>
    <li>Log in to a Google account (if not logged in already)</li>
    <li>Upload the file that was uploaded in step 1 (file > upload notebook > navigate to downloaded file and select)</li>
    <li>Follow the instructions in the file that has just opened in the browser</li>
</ul>

# Visualization of the trained model

To visualize the simulation, follow these steps:
<ul>
  <li>Navigate to the base directory of git repo</li>
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
