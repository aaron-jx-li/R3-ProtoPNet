import os
import shutil
from collections import Counter
import numpy as np
import torch
import torch.utils.data
import pandas as pd
import argparse
import re
from helpers import makedir
import model
import save
from log import create_logger
import train_and_test as tnt
from helpers import makedir
import find_nearest
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import push
from preprocess import mean, std, preprocess_input_function
from settings import train_dir, test_dir, train_push_dir, num_classes, prototype_shape, add_on_layers_type
from settings import img_size, num_classes, prototype_activation_function, add_on_layers_type
from find_nearest import find_k_nearest_patches_to_prototypes
from settings import coefs
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, test_dir, train_push_dir, img_size, batch_sizes):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize])
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=16, pin_memory=True)
    
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    )
    train_push_loader = DataLoader(train_push_dataset, batch_size=batch_sizes[2], shuffle=False, num_workers=16)
    
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize])
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=16)
    
    return train_loader, train_push_loader, test_loader

def train_protopnet(args):
    # Initialize model
    if args.load_path is None:
        ppnet = model.construct_PPNet(
            base_architecture=args.architecture,
            pretrained=True,
            img_size=args.img_size,
            prototype_shape=args.prototype_shape,
            num_classes=args.num_classes,
            prototype_activation_function=args.prototype_activation_function,
            add_on_layers_type=args.add_on_layers
        )
    else:
        ppnet = torch.load(args.load_path)

    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet).cuda()
    
    # Define optimizers and learning rates
    optimizers = {
        'warm': optim.Adam([
            {'params': ppnet.add_on_layers.parameters(), 'lr': args.warm_lr},
            {'params': ppnet.prototype_vectors, 'lr': args.warm_lr},
        ]),
        'joint': optim.Adam([
            {'params': ppnet_multi.module.features.parameters(), 'lr': args.joint_features_lr, 'weight_decay': 1e-3},
            {'params': ppnet_multi.module.add_on_layers.parameters(), 'lr': args.joint_add_on_lr, 'weight_decay': 1e-3},
            {'params': ppnet_multi.module.prototype_vectors, 'lr': args.joint_prototypes_lr},
        ]),
        'last_layer': optim.Adam([
            {'params': ppnet.last_layer.parameters(), 'lr': args.last_layer_lr}
        ])
    }

    lr_scheduler = optim.lr_scheduler.StepLR(optimizers['joint'], step_size=args.lr_step_size, gamma=args.gamma)
    
    # Create data loaders
    train_loader, train_push_loader, test_loader = get_data_loaders(
        args.train_dir, args.test_dir, args.train_push_dir, args.img_size,
        [args.train_batch_size, args.test_batch_size, args.train_push_batch_size]
    )
    
    log, logclose = create_logger(log_filename=os.path.join(args.save_path, 'train.log'))
    class_specific = True

    # Training logic placeholder
    print("Training model with configurations:", args)
    save_path = args.save_path
    for epoch in range(args.num_epochs):
        print('epoch: \t{0}'.format(epoch))
        
        if epoch in range(args.num_warm_epochs) and args.load_path is None:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=optimizers['warm'],
                    class_specific=class_specific, coefs=coefs, log=log)
            
        else:
            tnt.joint(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=optimizers['joint'],
                    class_specific=class_specific, coefs=coefs, log=log)
            lr_scheduler.step()

        if epoch in args.push_epochs:
            _ = push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=None,
                prototype_self_act_filename_prefix=None,
                proto_bound_boxes_filename_prefix=None,
                save_prototype_class_identity=True,
                log=log,
                bounding_box_tracker=None)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=save_path, model_name=f"ep{epoch}_pushed", accu=accu,
                                        target_accu=args.target_acc, log=log)

            

            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                # Fine tune the last layers
                for i in range(5):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=optimizers['last_layer'],
                                class_specific=class_specific, coefs=coefs, log=log)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
                    print("Test accuracy: ", accu)
                    save.save_model_w_condition(model=ppnet, model_dir=save_path, model_name=f"ep{epoch}_pushed_finetuned", accu=accu,
                                    target_accu=args.target_acc, log=log)
                
                # Save the last with the final layer fine-tuned
            
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        print(f"Test Accuracy = {accu}")
        save.save_model_w_condition(model=ppnet_multi.module, model_dir=save_path, model_name=f"ep{epoch}", accu=accu,
                                    target_accu=args.target_acc, log=log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPNet with various configurations")
    parser.add_argument("--architecture", type=str, choices=["vgg19", "resnet34", "densenet161"], default="vgg19")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_prototypes", type=int, default=2000)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--prototype_shape", type=int, default=(2000, 256, 1, 1))
    parser.add_argument("--prototype_activation_function", type=str, default='log')
    parser.add_argument("--add_on_layers", type=str, default='regular')
    
    parser.add_argument("--warm_lr", type=float, default=1e-3)
    parser.add_argument("--joint_features_lr", type=float, default=1e-4)
    parser.add_argument("--joint_add_on_lr", type=float, default=1e-3)
    parser.add_argument("--joint_prototypes_lr", type=float, default=1e-3)
    parser.add_argument("--last_layer_lr", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_warm_epochs", type=int, default=5)
    parser.add_argument("--push_epochs", type=list, default=[9, 19, 29, 39, 49])

    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--train_push_batch_size", type=int, default=128)
    
    parser.add_argument("--train_dir", type=str, default=train_dir)
    parser.add_argument("--test_dir", type=str, default=test_dir)
    parser.add_argument("--train_push_dir", type=str, default=train_push_dir)

    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--target_acc", type=float, default=0.70)
    
    args = parser.parse_args()
    train_protopnet(args)
