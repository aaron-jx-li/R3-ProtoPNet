import sys
import torch  
import numpy as np  
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions.categorical import Categorical
import math
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from preprocess import mean, std, preprocess_input_function
from settings import train_batch_size, test_batch_size, train_push_batch_size, train_dir, test_dir, train_push_dir
from settings import base_architecture, img_size, prototype_shape, num_classes, prototype_activation_function, add_on_layers_type
from receptive_field import compute_rf_prototype
from tqdm import tqdm
from settings import joint_optimizer_lrs, joint_lr_step_size
import train_and_test as tnt
from torch.utils.data import Subset
import time
import heapq
from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
import cv2
import argparse

class R3_PPnet(nn.Module):
    def __init__(self, PPnet, preference_model, k=1, p=5, learning_rate=1e-4, dummy_reward=False, train_batch_size=80):
        super(R3_PPnet, self).__init__()
        
        
        self.PPnet = PPnet.cuda()
        self.k = k
        self.pf_model = preference_model.cuda()
        
        self.PPnet_multi = self.PPnet
        for p in self.PPnet_multi.module.features.parameters():
            p.requires_grad = False
        for p in self.PPnet_multi.module.add_on_layers.parameters():
            p.requires_grad = False
        self.PPnet_multi.module.prototype_vectors.requires_grad = True
        for p in self.PPnet_multi.module.last_layer.parameters():
            p.requires_grad = False
        
        self.train_batch_size = train_batch_size
        self.num_epoch = 0
        policy_optimizer_specs = [#{'params': self.PPnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, 
                                  #{'params': self.PPnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                                  #{'params': self.PPnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                  #{'params': self.PPnet.module.add_on_layers.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
                                  {'params': self.PPnet.module.prototype_vectors, 'lr': 1e-3, 'weight_decay': 1e-4}
                                  ]
        self.policy_optimizer = torch.optim.Adam(policy_optimizer_specs)
        self.num_iteration = 0
        
        
    def get_heatmaps(self, batch_x, labels, pid):
        self.PPnet_multi.eval()
        n_prototypes = self.PPnet_multi.module.num_prototypes
        prototype_shape = self.PPnet_multi.module.prototype_shape
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
        protoL_rf_info = self.PPnet_multi.module.proto_layer_rf_info
        
        batch_x = batch_x.cuda()
        protoL_input_torch, proto_dist_torch = self.PPnet_multi.module.push_forward(batch_x)
        
        proto_dist_ = proto_dist_torch.view(proto_dist_torch.shape[0], proto_dist_torch.shape[1], -1)
        # proto_dist_: (batch_size, 1000, 49)
        all_distances = torch.amin(proto_dist_, axis=-1)
        # all_distances: (batch_size, 1000)
        target_p = [pid]
        
        proto_dist = torch.clone(proto_dist_torch)
       
        proto_dist = torch.transpose(proto_dist, 0, 1)
        dists = all_distances[:, target_p]
        dists = torch.transpose(dists, 0, 1)
        
        # proto_dist: (1000, batch_size)
        proto_dist_numpy = proto_dist.detach().cpu().numpy()
        
        
        original_imgs = []
        rescaled_patterns = []
        
        for j in target_p:
            
            original_imgs_j = []
            rescaled_patterns_j = []
           
            for i in range(len(batch_x)):
                closest_patch_indices_in_distance_map_j = list(np.unravel_index(np.argmin(proto_dist_numpy[j][i],axis=None), proto_dist_numpy[j][i].shape))
                closest_patch_indices_in_distance_map_j = [0] + closest_patch_indices_in_distance_map_j
                
                closest_patch_indices_in_img = compute_rf_prototype(batch_x.size(2), closest_patch_indices_in_distance_map_j, protoL_rf_info)
                closest_patch = \
                    batch_x[i, :, closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2], closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]
                closest_patch = closest_patch.cpu().numpy()
                closest_patch = np.transpose(closest_patch, (1, 2, 0))

                original_img = batch_x[i].cpu().numpy()
                original_img = np.transpose(original_img, (1, 2, 0))
                
                
                if self.PPnet_multi.module.prototype_activation_function == 'log':
                    act_pattern = np.log((proto_dist_numpy[j][i] + 1)/(proto_dist_numpy[j][i] + self.PPnet_multi.module.epsilon))
                elif self.PPnet_multi.module.prototype_activation_function == 'linear':
                    act_pattern = max_dist - proto_dist_numpy[j][i]
                else:
                    act_pattern = prototype_activation_function_in_numpy(proto_dist_numpy[j][i])

                patch_indices = closest_patch_indices_in_img[1:5]
                          
                upsampled_act_pattern = cv2.resize(act_pattern, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
                rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
                rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
                
                original_img = original_img - np.amin(original_img)
                original_img = original_img / np.amax(original_img)
                original_imgs_j.append(original_img)
                rescaled_patterns_j.append(rescaled_act_pattern)
                
            
            original_imgs.append(original_imgs_j)
            rescaled_patterns.append(rescaled_patterns_j)
    
        return original_imgs, rescaled_patterns, dists
    
    '''
    imgs: (1000, 1, 224, 224, 3), patterns: (1000, 1, 224, 224)
    '''
    def get_rewards(self, imgs, patterns):
        
        num_p = len(imgs)
        assert num_p == len(patterns)
        batch_size = len(imgs[0])
        assert batch_size == len(patterns[0])
        
        with torch.no_grad():
            rewards = torch.empty((num_p, batch_size))
            for i in range(num_p):
                for j in range(batch_size):
                    img = np.array([imgs[i][j]])
                    img = np.transpose(img, (0, 3, 1, 2))
                    img = torch.from_numpy(img).cuda()
                
                    pattern = patterns[i][j]
                    pattern = np.array([pattern, pattern, pattern])
                    pattern = torch.from_numpy(np.array([pattern])).cuda()
                    reward = self.pf_model(img, pattern) 
                    rewards[i][j] = reward
                
        return rewards
        
    def joint_update(self, rewards, dists):

        assert rewards.shape[0] == len(dists)
        assert rewards.shape[1] == len(dists[0])
        
        self.policy_optimizer.zero_grad()
        loss = 0
        for i in range(rewards.shape[0]):
            for j in range(rewards.shape[1]):
                if dists[i][j] > 0:
                    loss -= (1 / (100*dists[i][j]+1)) * rewards[i][j]

        if loss != 0:
            loss.backward()
            self.policy_optimizer.step()
        return 
    
    
    def run(self, batch_x, labels, pid, update=True):

        original_imgs, rescaled_patterns, dists = self.get_heatmaps(batch_x, labels, pid)
        rewards = self.get_rewards(original_imgs, rescaled_patterns)
        if update:
            loss = self.joint_update(rewards, dists)
        
        return torch.mean(rewards)
   
def reselect_prototypes(r3_ppnet, update_threshold, select_threshold, bad_prototype_idx, train_dataset):
    # get the heatmaps by searching for the closest images in the entire dataset
    # heatmaps: (2000, 1, 224, 224, 3)
    # rewards: (2000,)
    # patch_ids: (2000, 3, (idx tuple))
    prototype_shape = r3_ppnet.PPnet_multi.module.prototype_shape
    
    initial_bad_rewards = []
    final_bad_rewards = []    
    patch_ids = []
    failed_p = []

    success_count = 0
    # loop over each prototype
    for k in tqdm(range(len(bad_prototype_idx))):
        j = bad_prototype_idx[k]
        r3_ppnet.PPnet_multi.eval()
        n_prototypes = r3_ppnet.PPnet_multi.module.num_prototypes
        prototype_shape = r3_ppnet.PPnet_multi.module.prototype_shape
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
        protoL_rf_info = r3_ppnet.PPnet_multi.module.proto_layer_rf_info
        class_identity = torch.argmax(r3_ppnet.PPnet_multi.module.prototype_class_identity[j])
        class_idx = [x for x, y in enumerate(train_dataset.targets) if y == class_identity]
        class_start_idx = class_idx[0]
        class_dataset = Subset(train_dataset, class_idx)
        dataloader = torch.utils.data.DataLoader(class_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)
        batch_x, label = next(iter(dataloader))
        batch_x = batch_x.cuda()
        with torch.no_grad():
            initial_reward = r3_ppnet.run(batch_x, label, j, update=False)
        
        
        initial_bad_rewards.append(initial_reward.detach().cpu().numpy())
        # conv_outs: (80, 128, 7, 7)
        conv_outs = r3_ppnet.PPnet_multi.module.conv_features(batch_x)
        batch_idx = np.arange(batch_x.shape[0])
        class_idx_batch = batch_idx[label == class_identity]
        class_outs = conv_outs[label == class_identity]
        if class_outs.shape[0] == 0:
            continue
        height = class_outs.shape[2]
        width = class_outs.shape[3]
        success = False
        for _ in range(200):
            img_idx = np.random.choice(class_outs.shape[0])
            h = np.random.choice(height)
            w = np.random.choice(width)
            
            patch_candidate = class_outs[img_idx, :, h:h+1, w:w+1]
            r3_ppnet.PPnet_multi.module.prototype_vectors.data[j] = patch_candidate

            # distances: (7, 7)
            with torch.no_grad():
                patch_reward = r3_ppnet.run(batch_x, label, j, update=False)
            
            idx_tuple = (class_start_idx + img_idx, h, w)
            if patch_reward > select_threshold and idx_tuple not in patch_ids:
                patch_ids.append(idx_tuple)
                success = True
                success_count += 1
                break           
  
        if success:
            print("Prototype " + str(j) + " success!")
            final_bad_rewards.append(patch_reward.detach().cpu().numpy())
        else:
            print("Prototype " + str(j) + " failed!")
            failed_p.append(j)
    print("bad prototype number: ", len(bad_prototype_idx))
    if len(initial_bad_rewards) > 0:
        print("initial mean bad reward: ", np.mean(np.array(initial_bad_rewards)))
    if len(final_bad_rewards) > 0:
        print("final mean bad reward: ", np.mean(np.array(final_bad_rewards)))
    print("number of success: ", success_count)
    return final_bad_rewards, bad_prototype_idx, patch_ids, failed_p

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}


class PrefNet(nn.Module):

    def __init__(self, img_features, pattern_features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=False,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck', 
                k = 1):

        super(PrefNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.k = k
        
        self.img_features = img_features
        self.pattern_features = pattern_features

        self.img_conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3),
            nn.Sigmoid()
            )
        
        self.pattern_conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3),
            nn.Sigmoid()
            )
        
        self.final_fc = nn.Sequential(
            # change here
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        
            )
        
        if init_weights:
            self._initialize_weights()
            
            
    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        return x
    
    def forward(self, x, p):
        
        x = self.img_features(x)
        x = self.img_conv(x)
        
        p = self.pattern_features(p)
        p = self.pattern_conv(p)
        
        out = torch.flatten(p, 1) 
        out = self.final_fc(out)
        
        return out

    
    def _initialize_weights(self):
        for m in self.img_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for m in self.pattern_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




            
def construct_PrefNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(1000, 128, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck',
                    k = 1):
    img_features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    pattern_features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = img_features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return PrefNet(img_features=img_features,
                   pattern_features=pattern_features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type,
                 k = k)


def paired_cross_entropy_loss(out1, out2, targets):
    
    total_loss = 0
    for i in range(len(targets)):
        
        if targets[i] == -1:
            p1 = torch.exp(out1[i])/(torch.exp(out1[i]) + torch.exp(out2[i]))
            loss = - torch.log(p1)
        elif targets[i] == 1:
            p2 = torch.exp(out2[i])/(torch.exp(out1[i]) + torch.exp(out2[i]))
            loss = - torch.log(p2)

        else:
            p1 = torch.exp(out1[i])/(torch.exp(out1[i]) + torch.exp(out2[i]))
            p2 = torch.exp(out2[i])/(torch.exp(out1[i]) + torch.exp(out2[i]))
            loss = - (0.5*torch.log(p1) + 0.5*torch.log(p2))
            
        total_loss += loss
    return total_loss

class ImagePatch:

    def __init__(self, patch, label, distance,
                 original_img=None, act_pattern=None, patch_indices=None, img_id=None):
        self.patch = patch
        self.label = label
        self.negative_distance = -distance

        self.original_img = original_img
        self.act_pattern = act_pattern
        self.patch_indices = patch_indices
        self.img_id = img_id

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


class ImagePatchInfo:

    def __init__(self, label, distance):
        self.label = label
        self.negative_distance = -distance

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance

def find_k_nearest_patches_to_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                                         prototype_network_parallel, # pytorch network with prototype_vectors
                                         k=1,
                                         batch_size = 80,
                                         preprocess_input_function=None, # normalize if needed
                                         full_save=False, # save all the images
                                         root_dir_for_saving_images=None,
                                         log=print,
                                         prototype_activation_function_in_numpy=None, heatmap_ratio = 1.0):
    prototype_network_parallel.eval()
    '''
    full_save=False will only return the class identity of the closest
    patches, but it will not save anything.
    '''
    log('find nearest patches')
    start = time.time()
    n_prototypes = prototype_network_parallel.module.num_prototypes
    
    prototype_shape = prototype_network_parallel.module.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info

    heaps = []
    # allocate an array of n_prototypes number of heaps
    for _ in range(n_prototypes):
        # a heap in python is just a maintained list
        heaps.append([])

    for idx, (search_batch_input, search_y) in tqdm(enumerate(dataloader)):
        if preprocess_input_function is not None:
            search_batch = preprocess_input_function(search_batch_input)

        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.cuda()
            protoL_input_torch, proto_dist_torch = \
                prototype_network_parallel.module.push_forward(search_batch)
        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
        
        # proto_dist_: (80, 1000, 7, 7)

        for img_idx, distance_map in enumerate(proto_dist_):
            for j in range(n_prototypes):
                # find the closest patches in this batch to prototype j

                closest_patch_distance_to_prototype_j = np.amin(distance_map[j])
                closest_patch_indices_in_distance_map_j = \
                    list(np.unravel_index(np.argmin(distance_map[j],axis=None),
                                          distance_map[j].shape))
                patch_idx = closest_patch_indices_in_distance_map_j
                
                closest_patch_indices_in_distance_map_j = [0] + closest_patch_indices_in_distance_map_j
                closest_patch_indices_in_img = \
                    compute_rf_prototype(search_batch.size(2),
                                         closest_patch_indices_in_distance_map_j,
                                         protoL_rf_info)
                closest_patch = \
                    search_batch_input[img_idx, :,
                                       closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2],
                                       closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]
                closest_patch = closest_patch.numpy()
                closest_patch = np.transpose(closest_patch, (1, 2, 0))

                original_img = search_batch_input[img_idx].numpy()
                original_img = np.transpose(original_img, (1, 2, 0))

                if prototype_network_parallel.module.prototype_activation_function == 'log':
                    act_pattern = np.log((distance_map[j] + 1)/(distance_map[j] + prototype_network_parallel.module.epsilon))
                elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                    act_pattern = max_dist - distance_map[j]
                else:
                    act_pattern = prototype_activation_function_in_numpy(distance_map[j])

                # 4 numbers: height_start, height_end, width_start, width_end
                patch_indices = closest_patch_indices_in_img[1:5]

                # construct the closest patch object
                closest_patch = ImagePatch(patch=closest_patch,
                                           label=search_y[img_idx],
                                           distance=closest_patch_distance_to_prototype_j,
                                           original_img=original_img,
                                           act_pattern=act_pattern,
                                           patch_indices=patch_idx,
                                           img_id = idx * batch_size + img_idx)

                # add to the j-th heap 
                if len(heaps[j]) < k:
                    heapq.heappush(heaps[j], closest_patch)
                else:
                    heapq.heappushpop(heaps[j], closest_patch)
                    

    # after looping through the dataset every heap will
    # have the k closest prototypes
    original_imgs = []
    patterns = []
    patch_ids = []
    for j in range(n_prototypes):
        # finally sort the heap; the heap only contains the k closest
        # but they are not ranked yet
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]

        original_imgs_j = []
        patterns_j = []
        patch_ids_j = []
        for i, patch in enumerate(heaps[j]):
            
            img_size = patch.original_img.shape[0]
            upsampled_act_pattern = cv2.resize(patch.act_pattern,
                                               dsize=(img_size, img_size),
                                               interpolation=cv2.INTER_CUBIC)
            rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
            rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
            
            original_imgs_j.append(patch.original_img)
            patterns_j.append(rescaled_act_pattern)
            patch_ids_j.append((patch.img_id, patch.patch_indices[0], patch.patch_indices[1]))
        original_imgs.append(original_imgs_j)
        patterns.append(patterns_j)
        patch_ids.append(patch_ids_j)
    end = time.time()
    log('\tfind nearest patches time: \t{0}'.format(end - start))

    return original_imgs, patterns, patch_ids

'''
Update each prototype
'''
def joint_update(r3_ppnet, p_indices, reselect_threshold, update_threshold, select_threshold, patch_ids, train_dataset, num_p=10):
    
    # patch_ids: np.array (200, 5, 3)
    bad_prototype_idx = []
    num_reweighed = 0
    for p_idx in tqdm(p_indices):
        
        
        start = time.time()
        old_vec = torch.clone(r3_ppnet.PPnet_multi.module.prototype_vectors.data)
        
        class_idx = p_idx // num_p
        if p_idx % num_p == 0:
            class_idx = p_idx // num_p
            class_imgs = [x for x, y in enumerate(train_dataset.targets) if y == class_idx]
            class_dataset = Subset(train_dataset, class_imgs)
            dataloader = torch.utils.data.DataLoader(class_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=False)
            batch, label = next(iter(dataloader))
        best_vec = torch.clone(r3_ppnet.PPnet_multi.module.prototype_vectors.data[p_idx])
        best_iter = 0
        best_reward = r3_ppnet.run(batch, label, p_idx)
        initial_reward = best_reward

        class_p_idx = p_idx % num_p
        if initial_reward > update_threshold:
            for i in range(len(old_vec)):
                r3_ppnet.PPnet_multi.module.prototype_vectors.data[i] = old_vec[i]
            continue
        if initial_reward < reselect_threshold:
            bad_prototype_idx.append(p_idx)
            for i in range(len(old_vec)):
                r3_ppnet.PPnet_multi.module.prototype_vectors.data[i] = old_vec[i]
            continue
        
        # Reward reweighing
        print("Prototype: " + str(p_idx) + " initial reward: " + str(initial_reward))
        convergence_counter = 0
        for i in range(200):
            avg_reward = r3_ppnet.run(batch, label, p_idx)
            if avg_reward > best_reward:
                convergence_counter = 0
                best_vec = torch.clone(r3_ppnet.PPnet_multi.module.prototype_vectors.data[p_idx])
                best_iter = i
                best_reward = avg_reward
            convergence_counter += 1
            if convergence_counter > 20:
                break

        r3_ppnet.PPnet_multi.module.prototype_vectors.data[p_idx] = best_vec
        num_reweighed += 1
        end = time.time()
        print("Prototype: " + str(p_idx) + " best reward: " + str(best_reward) + " iter: " + str(best_iter) + " time: " + str(end - start))
        for i in range(len(old_vec)):
            if i != p_idx:
                r3_ppnet.PPnet_multi.module.prototype_vectors.data[i] = old_vec[i]
        
    #print("Number of high-reward duplicates: ", num_duplicate)   
    print("Number of bad prototypes: ", len(bad_prototype_idx))
    print("Number of reweighed prototypes: ", num_reweighed)
    final_bad_rewards, bad_prototype_idx, patch_ids, failed_p = reselect_prototypes(r3_ppnet, reselect_threshold, select_threshold, bad_prototype_idx, train_dataset)
    return bad_prototype_idx, patch_ids, failed_p

def R2_debug(args):
    ppnet = torch.load(args.model_path)
    ppnet = torch.nn.DataParallel(ppnet)
    pf_model = torch.load(args.rm_path)

    num_classes = args.num_classes
    img_size = args.img_size
    num_prototypes = args.num_prototypes

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
            args.train_push_dir,
            transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_push_batch_size, shuffle=False,
        num_workers=1, pin_memory=False)

    test_dataset = datasets.ImageFolder(
            args.test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=1, pin_memory=False)

    aug_train_dataset = datasets.ImageFolder(
            args.train_dir,
            transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    aug_train_loader = torch.utils.data.DataLoader(
        aug_train_dataset, batch_size=args.train_batch_size, shuffle=False,
        num_workers=1, pin_memory=False)
    
    r3_ppnet = R3_PPnet(ppnet, pf_model)
    # patch_ids: (2000, 1, 1, 3)
    original_imgs, patterns, patch_ids = find_k_nearest_patches_to_prototypes(train_loader, r3_ppnet.PPnet_multi, k=1, batch_size=80, preprocess_input_function=None, full_save=False, root_dir_for_saving_images=None, log=print, prototype_activation_function_in_numpy=None, heatmap_ratio=1.0)
    num_p = int(args.num_prototypes // args.num_classes)
    patch_ids = np.array(patch_ids).reshape(args.num_classes, num_p, 3)
    # np.save('./patch_ids.npy', patch_ids)

    bad_prototype_idx, patch_ids, failed_p = joint_update(r3_ppnet, np.arange(args.num_prototypes), 0.50, 0.90, 0.90, patch_ids, train_dataset, num_p=num_p)

    torch.save(r3_ppnet.PPnet_multi.module, f"{args.save_path}r2_debugged.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug a pretrained ProtoPNet with a reward model")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_prototypes", type=int, default=2000)
    parser.add_argument("--num_classes", type=int, default=200)

    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--train_push_batch_size", type=int, default=128)
    parser.add_argument("--train_dir", type=str, default=train_dir)
    parser.add_argument("--test_dir", type=str, default=test_dir)
    parser.add_argument("--train_push_dir", type=str, default=train_push_dir)

    parser.add_argument("--rm_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()
    R2_debug(args)