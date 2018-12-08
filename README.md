[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

For this project, we will use reinforcement learning to play [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)!

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the root folder of this repository, and unzip (or decompress) the file.

3. Install all the required dependencies:

    The main requirements of this project are Python==3.6, numpy, matplotlib, jupyter, pytorch and unity-agents. To ease its installation, I recommend the following procedure:

    - [Install miniconda](https://conda.io/docs/user-guide/install/index.html).

      > Feel free to skip this step, if you already have anaconda or miniconda installed in your machine.

      > For Linux 64-bit users, I would recommend trying the step outlined [here](#installation-for-conda-and-linux-x86-64-users)

    - Creating the environment.

      `conda create -n drlnd-collab-compet python=3.6`

    - Activate the environment

      `conda activate drlnd-collab-compet`

    - Installing dependencies.

      `pip install -r requirements.txt`
      

### Installation for conda and Linux x86-64 users

You can use the environment [YAML file](environment_linux_x86-64.yml) provided with repo as follows:

`conda env create -f environment_linux_x86-64.yml`

### Instructions

Launch a jupyter notebook and follow the tutorial in [Tennis.ipynb](Tennis.ipynb) to get started with training your own agent!

> In case you close the shell running the jupyter server, don't forget to activate the environment. `conda activate drlnd-collab-compet`

## Do you like the project?

Please gimme a ⭐️ in the GitHub banner 😉. I am also open for discussions especially accompany with ☕ or 🍺.