base_architecture = 'resnet34'
img_size = 224
prototype_shape = (2000, 256, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '009'

# data_path = '/scratch/users/jiaxun1218/data/' # czh
data_path = '/scratch/users/jiaxun1218/car_data/' # czh
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'
train_batch_size = 80 #80
test_batch_size = 100   #100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 50 #405
num_warm_epochs = 5

push_start = 0
push_saved_epochs = [10, 70, 100, 200, 300, 405]
push_epochs = [10, 50, 70, 100, 200, 300, 405]
