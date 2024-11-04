## Overview

This repo contains the official code for the paper ["Improving Prototypical Visual Explanations with Reward Reweighing, Reselection, and Retraining"](https://arxiv.org/abs/2307.03887)
(published in ICML 2024). Our proposed framework is called Reward Reweighing, Reselecting, and Retraining (R3) concept-level debugging framework, which
performs three additional corrective updates to a pretrained Prototypical Part Network (ProtoPNet) in an offline and efficient manner based on human feedback.

## Instructions for preparing the CUB-200-2011 data:

1. Download the dataset `CUB_200_2011.tgz` from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. Unpack `CUB_200_2011.tgz`
3. Crop the images using information from `bounding_boxes.txt` (included in the dataset)
4. Split the cropped images into training and test sets, using `train_test_split.txt` (included in the dataset)
5. Put the cropped training images in the directory `"./datasets/cub200_cropped/train_cropped/"`
6. Put the cropped test images in the directory `"./datasets/cub200_cropped/test_cropped/"`
7. Augment the training set using img_aug.py (included in this code package)
   -- this will create an augmented training set in the following directory:
      `"./datasets/cub200_cropped/train_cropped_augmented/"`

## Instructions for reproducing our experiments:
1. In `settings.py`, provide the appropriate strings for `data_path`, `train_dir`, `test_dir`,
`train_push_dir`:   
(1) `data_path` is the root directory where the dataset resides
    -- if you followed the instructions for preparing the data, `data_path` should be `"./datasets/cub200_cropped/"`  
(2) `train_dir` is the directory containing the augmented training set
    -- if you followed the instructions for preparing the data, `train_dir` should be `data_path` + `"train_cropped_augmented/"`  
(3) `test_dir` is the directory containing the test set
    -- if you followed the instructions for preparing the data, `test_dir` should be `data_path` + `"test_cropped/"`  
(4) `train_push_dir` is the directory containing the original (unaugmented) training set
    -- if you followed the instructions for preparing the data, `train_push_dir` should be `data_path + "train_cropped/"`
   
2. Train a baseline ProtoPNet, as detailed in the paper ["This Looks Like That: Deep Learning for Interpretable Image Recognition"](https://arxiv.org/abs/1806.10574).  
`python train_protopnet.py --save_path [local path for saving the ProtoPNet]`
3. Train reward model:  
`python train_reward_model.py --data_file_path [local path for the rating file] --images_path [local path for saved images] --patterns_path [local path for saved upsampled activation patterns] --save_path [local path for saving the reward model]`  
4. Perform R2 debugging:  
`python R2_debug.py --rm_path [local path for saved reward model] --model_path [local path for saved ProtoPNet] --save_path [local path for saving the ProtoPNet]`  
5. Retraining:  
`python train_protopnet.py --load_path [local path for the saved model after R2 updates] --save_path [local path for saving the new ProtoPNet]`  
