import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env
# Referencing the stablebaselines docs: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html

import sys
sys.path.append("../")

import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

import pickle as pkl

# Reward Model Code
from preference_model import construct_PrefNet, paired_cross_entropy_loss, PrefNet

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

from preprocess import mean, std, preprocess_input_function

# load the data
# must use unaugmented (original) dataset
from settings import train_push_dir, test_dir
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.agents.trainer_template import build_trainer



def PPnet_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()
    action_dist = dist_class(logits)
    log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
    policy.entropy = action_dist.entropy().mean()
    
    return
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    ray.init()
    
    config = {
        "env": PPnetEnv
        "env_config": {
            
        }
        "num_gpus":
    }
    
    num_epochs = 100
    
    '''
    PPnet_policy = build_torch_policy(name='PPnetPolicy', 
                                     get_default_config=config, 
                                     loss_fn=PPnet_loss, 
                                     postprocess_fn=None, 
                                     extra_action_out_fn=model_value_predictions, 
                                     extra_grad_process_fn=apply_grad_clipping, 
                                     optimizer_fn=torch_optimizer, 
                                     mixins=[ValueNetworkMixin])
    '''
    
    PPnet_policy = PPnetPolicy(observation_space, action_space, config)
    PPnet_env = PPnetEnv(gym.Env)
    
    # Exact update from the algorithm to the policy network?
    algo = PPnet_A3C(PPnet_policy, env)
    for i in range(num_epochs):
        algo.train_one_step()
        
    ray.shutdown()