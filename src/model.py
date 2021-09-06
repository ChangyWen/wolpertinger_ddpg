#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=128, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # out = self.tanh(out)
        out = self.softsign(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=128, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # concatenate along columns
        c_in = torch.cat([out,a],len(a.shape)-1)
        out = self.fc2(c_in)
        out = self.relu(out)
        out = self.fc3(out)
        return out