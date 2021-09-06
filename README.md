# Wolpertinger Training with DDPG (Pytorch, Multi-GPU/single-GPU/CPU)
## Overview
Pytorch version of Wolpertinger Training with DDPG (paper: [Deep Reinforcement Learning in Large Discrete Action Spaces](https://arxiv.org/pdf/1512.07679.pdf)). <br>
The code is compatible with training in multi-GPU, single-GPU or CPU. <br>
It is also compatible with both **continuous** and **discrete** control of OpenAI gym. <br>
In continuous case, I discretize the action space to use wolpertinger-DDPG training algorithm.

## Dependencies
* python 3.6.8
* torch 1.1.0
* [OpenAI gym](https://github.com/openai/gym)
  * If you get an RunTimeError:NotImplementedError in ActionWrapper.step while training with gym, replace your `gym/core.py` file with [`core.py`](https://github.com/openai/gym/blob/master/gym/core.py) in openai/gym.
* [pyflann](http://www.galaxysofts.com/new/pyflann-for-python-3x/)
  * This is the library (FLANN, [Muja & Lowe, 2014](https://ieeexplore.ieee.org/abstract/document/6809191)) with approximate nearest-neighbor methods allowed for logarithmic-time lookup complexity relative to the number of actions. However, the python binding of FLANN (pyflann) is written for python 2 and is no longer maintained.
  * To use this package, please put the whole directory [`pyflann`](pyflann) into your (virtual) python environment.
  * Please refer to [pyflann](http://www.galaxysofts.com/new/pyflann-for-python-3x/) for a more detailed instruction if needed.
  

## Usage
* In Pendulum-v0 (continuous control), discretize the continuous action space to a discrete action spaces with 200000 actions.
    ```bash
    python main.py --env 'Pendulum-v0' --max-actions 200000
    ```
* In CartPole-v1 (discrete control), ```--max-actions``` is not needed.
    ```bash
    python main.py --env 'CartPole-v1'
    ```
* To use CPU only:
    ```bash
    python main.py --gpu-ids -1
    ```
* To use single-GPU only:
    ```bash
    python main.py --gpu-ids 0 --gpu-nums 1
    ```
* To use multi-GPU (e.g., use GPU-0 and GPU-1):
    ```bash
    python main.py --gpu-ids 0 1 --gpu-nums 2
    ```
## Result
* Please refer to [`output`](output) for the trained models and training log.
  * Pendulum-v0: a gym environment with **continuous** action space.
  * CartPole-v1: a gym environment with **discrete** action space.

## Project Reference
* [Original paper of Wolpertinger Training with DDPG, Google DeepMind](https://arxiv.org/abs/1512.07679)
* I used and modified part of the code in https://github.com/ghliu/pytorch-ddpg under Apache License 2.0.
* I used and modified part of the code in https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces under MIT License.
