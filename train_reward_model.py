import numpy as np
import numpy.random as npr
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
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
from settings import train_dir, test_dir, train_push_dir, train_batch_size, test_batch_size, train_push_batch_size
from settings import base_architecture, img_size, prototype_shape, num_classes, prototype_activation_function, add_on_layers_type
from receptive_field import compute_rf_prototype
from preference_model import construct_PrefNet, paired_cross_entropy_loss, PrefNet
from tqdm import tqdm
from settings import joint_optimizer_lrs, joint_lr_step_size
import train_and_test as tnt
from torch.utils.data import Subset
from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
from receptive_field import compute_proto_layer_rf_info_v2
import argparse

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
        
        # this has to be named features to allow the precise loading
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
        
        #self.fc1 = nn.Linear(6400, 512)
        #self.fc2 = nn.Linear(512, 32)
        #self.fc3 = nn.Linear(32, 1)
        #self.fc1 = nn.Linear(64, 16)
        

        if init_weights:
            self._initialize_weights()
            
            
    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        # Insert k and then img size
        x = self.features(x)
        #print("base features: ", x.shape)
        #x = self.add_on_layers(x)
        return x
    
    def forward(self, x, p):
        
        x = self.img_features(x)
        x = self.img_conv(x)
        
        p = self.pattern_features(p)
        p = self.pattern_conv(p)
        
        #out = torch.cat((x, p), dim=1)
        out = torch.flatten(p, 1) 
        out = self.final_fc(out)
        
        return out

    
    def _initialize_weights(self):
        for m in self.img_conv.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for m in self.pattern_conv.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
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

def test_reward_model(prefnet, test_set, images, patterns, batch_size):
    acc = []
    error_count = 0
    all_idx = np.arange(len(test_set))
    for batch_i in tqdm(range(len(test_set)//batch_size + 1)):
        prefnet.eval()
        
        idx = all_idx[batch_i*batch_size:(batch_i+1)*batch_size]
        
        left_imgs = torch.zeros((batch_size, 3, 224, 224))
        right_imgs = torch.zeros((batch_size, 3, 224, 224))
        left_patterns = torch.zeros((batch_size, 3, 224, 224))
        right_patterns = torch.zeros((batch_size, 3, 224, 224))
        targets = []
        for i in range(len(idx)):
            index = idx[i]
            left_imgs[i] = images[test_set[index][0]][0]
            right_imgs[i] = images[test_set[index][1]][0]
            targets.append(test_set[index][2])
            left_patterns[i] = patterns[test_set[index][0]][0]
            right_patterns[i] = patterns[test_set[index][1]][0]
        
        targets = torch.tensor(targets).cuda().float()
        
        out1 = prefnet(left_imgs.cuda().float(), left_patterns.cuda().float())
        out2 = prefnet(right_imgs.cuda().float(), right_patterns.cuda().float())

        for i in range(len(targets)):

            if out1[i] > out2[i]:
                y_pred = -1

            else:
                y_pred = 1

            if y_pred == targets[i]:
                acc.append(1)
            else:
                error_count += 1
                acc.append(0)
            
    return np.mean(acc), error_count

def train_reward_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Reload the data while skipping the first row
    data_df = pd.read_csv(args.data_file_path, skiprows=1)

    # Rename the columns for clarity and drop any irrelevant columns
    data_df.columns = ['index', 'img_id', 'p_id', 'rating_mean', 'rating_std']
    data_df = data_df.drop(columns=['index'])

    # Convert 'rating_mean' column to float
    data_df['rating_mean'] = data_df['rating_mean'].astype(float)
    print(data_df.head())

    split = 0.5
    df_len = len(data_df)
    train_set = []
    test_set = []
    split_idx = int(df_len*split)
    for i in range(split_idx):
        for j in range(i+1, split_idx):
            # print(data_df.iloc[i]['rating'])
            if data_df.iloc[i]['rating_mean'] > data_df.iloc[j]['rating_mean'] + 0.5:
                train_set.append([i, j, -1])
            elif data_df.iloc[i]['rating_mean'] + 0.5 < data_df.iloc[j]['rating_mean']:
                train_set.append([i, j, 1])
                
    for i in range(split_idx, df_len):
        for j in range(i+1, df_len):
            if data_df.iloc[i]['rating_mean'] > data_df.iloc[j]['rating_mean'] + 1.5:
                test_set.append([i, j, -1])
            elif data_df.iloc[i]['rating_mean'] + 1.5 < data_df.iloc[j]['rating_mean']:
                test_set.append([i, j, 1])
    
    images = []
    patterns = []
    for i in range(len(data_df)):
        img = args.images_path + str(int(data_df.iloc[i]['img_id'])) + '_' + str(int(data_df.iloc[i]['p_id'])) + '.png'
        img = plt.imread(img)[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        images.append(torch.from_numpy(np.array([img])))
        pattern = args.patterns_path + str(int(data_df.iloc[i]['img_id'])) + '_' + str(int(data_df.iloc[i]['p_id'])) + '.npy'
        pattern = np.load(pattern)
        pattern = np.array([pattern, pattern, pattern])
        patterns.append(torch.from_numpy(np.array([pattern])))

    prefnet = construct_PrefNet("resnet50")
    prefnet.to(device)
    prefnet.train()

    for p in prefnet.final_fc.parameters():
        p.requires_grad = True
    for p in prefnet.img_features.parameters():
        p.requires_grad = True
    for p in prefnet.pattern_features.parameters():
        p.requires_grad = True
    for p in prefnet.img_conv.parameters():
        p.requires_grad = True
    for p in prefnet.pattern_conv.parameters():
        p.requires_grad = True

    pref_optimizer = optim.Adam([{'params': prefnet.img_features.parameters(), 'lr': 1e-4, 'weight_decay':1e-4}, {'params': prefnet.pattern_features.parameters(), 'lr': 1e-4, 'weight_decay':1e-4},
                                #{'params': prefnet.add_on_layers.parameters(), 'lr': 1e-4}, 
                                {'params': prefnet.img_conv.parameters(), 'lr': 1e-3, 'weight_decay':1e-3}, {'params': prefnet.pattern_conv.parameters(), 'lr': 1e-3, 'weight_decay':1e-3}, 
                                {'params': prefnet.final_fc.parameters(), 'lr': 1e-3, 'weight_decay':1e-3}
                                #{'params': prefnet.fc1.parameters(), 'lr': 1e-4}, {'params': prefnet.fc2.parameters(), 'lr': 1e-4}, {'params': prefnet.fc3.parameters(), 'lr': 1e-4}
                                ])

    pref_lr_step_size = 1
    gamma = 0.1
    pref_lr_scheduler = torch.optim.lr_scheduler.StepLR(pref_optimizer, step_size=pref_lr_step_size, gamma=0.1)

    lr = 1e-5
    weight_decay = 1e-5
    pref_optimizer = optim.Adam([{'params': prefnet.img_features.parameters(), 'lr': lr, 'weight_decay':weight_decay}, {'params': prefnet.pattern_features.parameters(), 'lr': lr, 'weight_decay':weight_decay},
                                #{'params': prefnet.add_on_layers.parameters(), 'lr': 1e-4}, 
                                {'params': prefnet.img_conv.parameters(), 'lr': lr, 'weight_decay':weight_decay}, {'params': prefnet.pattern_conv.parameters(), 'lr': lr, 'weight_decay':weight_decay}, 
                                {'params': prefnet.final_fc.parameters(), 'lr': lr, 'weight_decay':weight_decay}
                                #{'params': prefnet.fc1.parameters(), 'lr': 1e-4}, {'params': prefnet.fc2.parameters(), 'lr': 1e-4}, {'params': prefnet.fc3.parameters(), 'lr': 1e-4}
                                ])
    
    epochs = 5
    batch_size = 32

    for epoch in range(0, epochs):
        shuffled_idx = np.random.permutation(len(train_set))
        for batch_i in range(len(train_set)//batch_size + 1):
            prefnet.train()
            if batch_i % 100 == 0:
                last_100_losses = []
                last_100_error_count = 0
            idx = shuffled_idx[batch_i*batch_size:(batch_i+1)*batch_size]
            
            left_imgs = torch.zeros((batch_size, 3, 224, 224))
            right_imgs = torch.zeros((batch_size, 3, 224, 224))
            left_patterns = torch.zeros((batch_size, 3, 224, 224))
            right_patterns = torch.zeros((batch_size, 3, 224, 224))
            targets = []
            for i in range(len(idx)):
                index = idx[i]
                left_imgs[i] = images[train_set[index][0]][0]
                right_imgs[i] = images[train_set[index][1]][0]
                targets.append(train_set[index][2])
                left_patterns[i] = patterns[train_set[index][0]][0]
                right_patterns[i] = patterns[train_set[index][1]][0]
            
            targets = torch.tensor(targets).cuda().float()
            
            out1 = prefnet(left_imgs.cuda().float(), left_patterns.cuda().float())
            out2 = prefnet(right_imgs.cuda().float(), right_patterns.cuda().float())

            
            pref_optimizer.zero_grad()   
            
            for i in range(len(targets)):
                if out1[i] > out2[i] and targets[i] == 1:
                    last_100_error_count += 1

                elif out1[i] < out2[i] and targets[i] == -1:
                    last_100_error_count += 1
                    
            loss = paired_cross_entropy_loss(out1, out2, targets)
            
            loss.backward()
            pref_optimizer.step()   
            
            last_100_losses.append(loss.data.cpu().numpy()[0])
            
            #if batch_i % 100 == 0:
            #    print(epoch, batch_i, np.sum(last_100_losses))
            #if batch_i % 100 == 99:
            #    print(epoch, batch_i, last_100_error_count)
            if batch_i % 500 == 499:
                test_acc, test_error_count = test_reward_model(prefnet, test_set, images, patterns, batch_size)
                print(epoch, batch_i, test_acc, test_error_count)
                if test_acc > 0.75:
                    torch.save(prefnet, f"{args.save_path}/{test_acc:.4f}.pth")
        pref_lr_scheduler.step()

    test_acc, test_error_count = test_reward_model(prefnet, test_set, images, patterns, batch_size)
    print(test_acc, test_error_count)

    torch.save(prefnet, f"{args.save_path}/{test_acc:.4f}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model with collected human feedback")
    parser.add_argument("--data_file_path", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--patterns_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()
    train_reward_model(args)