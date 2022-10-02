import os
import shutil

import torch
import torch.utils.data
import pandas as pd
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

from bounding_box_metrics import bounding_box_overlap
from find_nearest import find_k_nearest_patches_to_prototypes

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
    from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    model_dir = '/scratch/users/jiaxun1218/saved_models/' + base_architecture + '/' + experiment_run + '/'
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # load the data
    from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size

    normalize = transforms.Normalize(mean=mean,
                                 std=std)

    # all datasets
    # train set
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=2, pin_memory=False)
    # push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
    ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=2, pin_memory=False)
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=2, pin_memory=False)

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    # construct the model
    print(base_architecture, img_size, prototype_shape, num_classes, prototype_activation_function, add_on_layers_type)
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
    #if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # define optimizer
    from settings import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
     {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    from settings import warm_optimizer_lrs
    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from settings import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # weighting of different training losses
    from settings import coefs

    # number of training epochs, number of warm epochs, push start epoch, push epochs
    from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs, push_saved_epochs

    ## Tracking Code
    bounding_box_tracker = []

    ## 

    # train the model
    log('start training')
    import copy

    overlaps = {
        "epoch": [],
        "overlap": []
    }
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)

    #     accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
    #                     class_specific=class_specific, log=log)
    #     save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
    #                                 target_accu=0.70, log=log)

        if epoch >= push_start and epoch in push_epochs:
            # Only save the model in in push_saved_epochs
            if epoch in push_saved_epochs:
                bounding_box_tracker = push.push_prototypes(
                    train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    class_specific=class_specific,
                    preprocess_input_function=preprocess_input_function, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                    save_prototype_class_identity=True,
                    log=log,
                    bounding_box_tracker=bounding_box_tracker)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                            target_accu=0.70, log=log)

                # updated bounding_box_tracker is used  
                overlap = bounding_box_overlap(bounding_box_tracker, prototype_shape[0], num_classes)
                overlaps["epoch"].append(epoch)
                overlaps["overlap"].append(overlap)   

                pd.DataFrame(overlaps).to_csv(model_dir + "overlap_" + str(epoch) + ".csv")

            else:
                bounding_box_tracker = push.push_prototypes(
                    train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    class_specific=class_specific,
                    preprocess_input_function=preprocess_input_function, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                    save_prototype_class_identity=True,
                    log=log,
                    bounding_box_tracker=bounding_box_tracker)
    #             accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
    #                             class_specific=class_specific, log=log)
    #             save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
    #                                         target_accu=0.70, log=log)

                overlap = bounding_box_overlap(bounding_box_tracker, prototype_shape[0], num_classes)
                overlaps["epoch"].append(epoch)
                overlaps["overlap"].append(overlap)

            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                # Fine tune the last layers
                for i in range(8):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                  class_specific=class_specific, coefs=coefs, log=log)

                # Save the last with the final layer fine-tuned
                if epoch in push_saved_epochs:
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    print("Test accuracy: ", accu)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=0.70, log=log)
                    
                    #find_k_nearest_patches_to_prototypes(test_loader, prototype_network_parallel=ppnet_multi, k=3, preprocess_input_function=None, full_save=True, root_dir_for_saving_images='/scratch/users/jiaxun1218/saved_models/' + base_architecture + "/" + experiment_run + "/epoch-" + str(epoch) + "/")

    logclose()

