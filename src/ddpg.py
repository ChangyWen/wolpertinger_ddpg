#!/usr/bin/env python
# -*- coding: utf-8 -*-

# [reference] Use and modified code in https://github.com/ghliu/pytorch-ddpg

import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, args, nb_states, nb_actions):
        USE_CUDA = torch.cuda.is_available()
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states =  nb_states
        self.nb_actions= nb_actions
        self.gpu_ids = [i for i in range(args.gpu_nums)] if USE_CUDA and args.gpu_nums > 0 else [-1]
        self.gpu_used = True if self.gpu_ids[0] >= 0 else False

        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg).double()
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg).double()
        self.actor_optim = Adam(self.actor.parameters(), lr=args.p_lr, weight_decay=args.weight_decay)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg).double()
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg).double()
        self.critic_optim = Adam(self.critic.parameters(), lr=args.c_lr, weight_decay=args.weight_decay)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions,
                                                       theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau_update = args.tau_update
        self.gamma = args.gamma

        # Linear decay rate of exploration policy
        self.depsilon = 1.0 / args.epsilon
        # initial exploration rate
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        self.continious_action_space = False

    def update_policy(self):
        pass

    def cuda_convert(self):
        if len(self.gpu_ids) == 1:
            if self.gpu_ids[0] >= 0:
                with torch.cuda.device(self.gpu_ids[0]):
                    print('model cuda converted')
                    self.cuda()
        if len(self.gpu_ids) > 1:
            self.data_parallel()
            self.cuda()
            self.to_device()
            print('model cuda converted and paralleled')

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def data_parallel(self):
        self.actor = nn.DataParallel(self.actor, device_ids=self.gpu_ids)
        self.actor_target = nn.DataParallel(self.actor_target, device_ids=self.gpu_ids)
        self.critic = nn.DataParallel(self.critic, device_ids=self.gpu_ids)
        self.critic_target = nn.DataParallel(self.critic_target, device_ids=self.gpu_ids)

    def to_device(self):
        self.actor.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.actor_target.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.critic.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.critic_target.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        # self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        # proto action
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t]), gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])),
            gpu_used=self.gpu_used
        ).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        # self.a_t = action
        return action

    def reset(self, s_t):
        self.s_t = s_t
        self.random_process.reset_states()

    def load_weights(self, dir):
        if dir is None: return

        if self.gpu_used:
            # load all tensors to GPU (gpu_id)
            ml = lambda storage, loc: storage.cuda(self.gpu_ids)
        else:
            # load all tensors to CPU
            ml = lambda storage, loc: storage

        self.actor.load_state_dict(
            torch.load('output/{}/actor.pkl'.format(dir), map_location=ml)
        )

        self.critic.load_state_dict(
            torch.load('output/{}/critic.pkl'.format(dir), map_location=ml)
        )
        print('model weights loaded')


    def save_model(self,output):
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] > 0:
            with torch.cuda.device(self.gpu_ids[0]):
                torch.save(
                    self.actor.state_dict(),
                    '{}/actor.pt'.format(output)
                )
                torch.save(
                    self.critic.state_dict(),
                    '{}/critic.pt'.format(output)
                )
        elif len(self.gpu_ids) > 1:
            torch.save(self.actor.module.state_dict(),
                       '{}/actor.pt'.format(output)
            )
            torch.save(self.actor.module.state_dict(),
                       '{}/critic.pt'.format(output)
                       )
        else:
            torch.save(
                self.actor.state_dict(),
                '{}/actor.pt'.format(output)
            )
            torch.save(
                self.critic.state_dict(),
                '{}/critic.pt'.format(output)
            )

    def seed(self,seed):
        torch.manual_seed(seed)
        if len(self.gpu_ids) > 0:
            torch.cuda.manual_seed_all(seed)
