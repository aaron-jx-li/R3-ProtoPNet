import gym, ray
from gym import spaces

#from stable_baselines3.common.env_checker import check_env
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


"""
Todos:
    
"""

"""
Parameters:
    model: The ProtoPNet model to use to generate bounding boxes
    reward_fn: The learned reward function to use to evaluate bounding boxes
    k: the number of images per prototype from comparison
"""

class PPnetEnv(gym.Env):

    def __init__(self, reward_fn, img_dir = train_push_dir, k = 3, r_fn_arch = "resnet18",
                HEIGHT = 224, WIDTH = 224, N_CHANNELS = 3):
        super(PPnetEnv, self).__init__()
        
        # Define action and observation space
        # Action Space: The bounding box info (low_left, low_right, high_left, high_right) * k
        self.action_space = spaces.Box(low = 0, high = 255, shape = (k, 4), dtype = np.float32)
        
        # Example for using image as input:
        # TODO: Change the below so that it's a vector already processed
        ## That way images are already concatenated
        self.observation_space = spaces.Box(low=0, high=255, shape=
                                            (N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        
        # Image Transformations--unclear if needed
        self.normalize = transforms.Normalize(mean=mean,
                                         std=std)

        self.trans = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # Load the Data
        train_dir = train_push_dir

        batch_size = k

        # train set: do not normalize
        self.train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ]))
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=1, pin_memory=False)
        self.train_iter = enumerate(self.train_loader)
        # Load the reward function
        
        # Construct the reward model
        ## Note: User must provide matching architecture to load properly
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prefnet = construct_PrefNet(r_fn_arch)
        prefnet.to(self.device)
        #prefnet.load_state_dict(torch.load(reward_fn))
        
        self.reward_fn = prefnet
        
        # Image Directory
        self.img_dir = img_dir
        self.k = k
        
        # Q: Need to have an initial state?
        self.state = None

    # Not so necessary if the major work is done in Algorithm?
    def step(self, action):
        # Execute one time step within the environment
        
        ### Evaluate Action with Reward
        
        # if invalid action, return extremely low reward
        # or just check to make sure they can't do that
        # Hard to check if invalid if entire image is provided
        
        

        
        # TODO: remove these dummy variables?
        
        reward = self.reward_fn(action)
        obs = next(self.train_iter)
        # done should always be true: the episode is only ever one step
        #done = True
        if not next(self.train_iter):
            done = True
        
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        # Aka, sample the appropraite amount of images
        for image, label in self.train_loader:
            self.state = image
            break
            
        # TODO: currently returns as a tensor--is this necessary to change?
        return self.state

    

#if __name__ == "__main__":
    
    # Some code to verify that the Environment is Valid for the
#    env = PPnetEnv("./datasets/first_pref_model", img_dir = "./datasets/")
#     # It will check your custom environment and output additional warnings if needed
#     check_env(env)