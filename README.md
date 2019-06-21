# Wolpertinger Training with DDPG (Pytorch, Multi-GPU/single-GPU/CPU)
## Overview
Pytorch version of Wolpertinger Training with DDPG. <br>
The code is compatible with training in multi-GPU, single-GPU or CPU. <br>

## Dependencies
* python 3.6.8
* torch 1.1.0
* [OpenAI gym](https://github.com/openai/gym)
* [pyflann](http://www.galaxysofts.com/new/pyflann-for-python-3x/)
  * This is the library (FLANN ([(Muja & Lowe, 2014](https://ieeexplore.ieee.org/abstract/document/6809191))) with approximate nearest-neighbor methods allowed for logarithmic-time lookup complexity relative to the number of actions. However, the python binding of FLANN (pyflann) is written for python 2 and is no longer maintained. Please refer to [pyflann](http://www.galaxysofts.com/new/pyflann-for-python-3x/) for the pyflann package compatible with python3. Just download and place it in your (virtual) environment.

## Project Reference
* [Original paper of Wolpertinger Training with DDPG, Google DeepMind](https://arxiv.org/abs/1512.07679)
* https://github.com/ghliu/pytorch-ddpg
* https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces

## TODO
* Module of testing the trained policy
* Upload the training result
