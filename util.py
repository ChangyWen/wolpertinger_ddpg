#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch.autograd import Variable
import logging


def to_numpy(var, gpu_used=False):
    return var.cpu().data.numpy().astype(np.float64) if gpu_used else var.data.numpy().astype(np.float64)

def to_tensor(ndarray, volatile=False, requires_grad=False, gpu_used=False, gpu_0 = 0):
    if gpu_used:
        return Variable(torch.from_numpy(ndarray).cuda(device=gpu_0).type(torch.cuda.DoubleTensor),
                        volatile=volatile,
                        requires_grad=requires_grad)
    else:
        return Variable(torch.from_numpy(ndarray).type(torch.DoubleTensor),
                        volatile=volatile,
                        requires_grad=requires_grad)

def soft_update(target, source, tau_update):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau_update) + param.data * tau_update
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)