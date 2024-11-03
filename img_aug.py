import imgaug.augmenters as iaa
import imageio
import os
import numpy as np

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

root_dir = '/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/CUB/'
src_dir = root_dir + 'train_cropped/'
target_dir = root_dir + 'train_cropped_augmented/'

makedir(target_dir)
folders = [os.path.join(src_dir, folder) for folder in os.listdir(src_dir)]
target_folders = [os.path.join(target_dir, folder) for folder in os.listdir(src_dir)]

# Define augmentation pipelines
rotate = iaa.Sequential([
    iaa.Rotate((-15, 15)),
    iaa.Fliplr(0.5)
])
rotate.name = 'rotate'

skew = iaa.Sequential([
    iaa.ShearX((10, -10)),
    iaa.Fliplr(0.5)
])
skew.name = 'skew'

shear = iaa.Sequential([
    iaa.ShearY((-10, 10)),
    iaa.Fliplr(0.5)
])
shear.name = 'shear'

random_distort = iaa.Sequential([
    iaa.ElasticTransformation(alpha=5.0, sigma=0.25),
    iaa.Fliplr(0.5)
])
random_distort.name = 'distort'

# Function to load and augment images
def augment_images(folder, target_folder, augmenter, num_augmentations=5):
    images = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    for img_path in images:
        image = imageio.imread(img_path)
        for i in range(num_augmentations):
            aug_image = augmenter(image=image)
            new_img_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{augmenter.name}_aug_{i}.jpg"
            imageio.imwrite(os.path.join(target_folder, new_img_name), aug_image)

# Apply augmentations
for fd, tfd in zip(folders, target_folders):
    if os.path.exists(tfd) and len(os.listdir(tfd)) > 0:
        print(f"Skipping {fd} as augmented images already exist in {tfd}")
        continue
    makedir(tfd)

    print("Processing folders:", fd, tfd)
    augment_images(fd, tfd, rotate)
    augment_images(fd, tfd, skew)
    augment_images(fd, tfd, shear)
    augment_images(fd, tfd, random_distort)