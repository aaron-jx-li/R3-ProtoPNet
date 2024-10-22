import Augmentor
import os
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

# datasets_root_dir = './datasets/cub200_cropped/'
# datasets_root_dir  = '/Users/macbook/Desktop/Research/Berkeley Yu Group/Datasets/CUB_200_2011_cropped/'
#datasets_root_dir = '/scratch/users/jiaxun1218/car_data/' # czh
#dir = datasets_root_dir + 'train_cropped/'
#target_dir = datasets_root_dir + 'train_cropped_augmented/'

datasets_root_dir = '/accounts/projects/binyu/jiaxun1218/ProtoPNet/stanford_cars/car_data/car_data/'
src_dir = datasets_root_dir + 'train/'
target_dir = datasets_root_dir + 'train_augmented/'

makedir(target_dir)
folders = [os.path.join(src_dir, folder) for folder in os.listdir(src_dir)]
target_folders = [os.path.join(target_dir, folder) for folder in os.listdir(src_dir)]

for i in range(len(folders)):

    fd = folders[i]
    tfd = target_folders[i]
    
    makedir(tfd)
    
    # Some Jupyter Processing
    if ".ipynb_checkpoints" in fd:
        continue
    
    # rotation

    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    print("Folders: ", fd, tfd)
    print(os.getcwd())
    for i in range(5):
        p.process()
    del p
    
    
    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for i in range(5):
        p.process()
    del p
    # shear
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    for i in range(5):
        p.process()
    del p
    # random_distortion
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    p.flip_left_right(probability=0.5)
    for i in range(5):
        p.process()
    del p
