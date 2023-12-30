from PIL import Image
import natsort
import os
import numpy as np
import glob
from dataloader import *
import random
from pathlib import Path
import natsort
import shutil

from collections import Counter
full_path = r'C:\Users\p.dyroey\Documents\GitHub\moonrise-fm\moonrisefm\data\real_moon\*'
full_glob = glob.glob(full_path)
file_names = []
for filename in natsort.natsorted(full_glob):
    file_names.append(filename.split('/')[-1])
print(f"found {len(file_names)} images")

## load images
images = []
for image_name in file_names:
    file_name = image_name
    image = Image.open(file_name)
    images.append(image)
print(f"nr images loaded: {len(images)}")

def make_image_dir(path_dir):
    path = Path(path_dir)
    # remove folder if it exists
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
# patches
def random_patches(image, n=1000, patch_h=48, patch_w=48):
    '''
    Extract randomly cropped images and masks. Adapted from:
    https://github.com/orobix/retina-unet/blob/master/lib/extract_patches.py

    Inputs:
        image : array
            grayscale or RGB image
        mask : array
            RGB image
        n : int
            number of patches to extract from image
        patch_h : int
            patch height
        patch_w : int
            patch width

    Outputs:
        patches : list[array]
            extracted patches
        patch_masks : list[array]
            mask of extracted patches
    '''
    print(image.shape)
    img_h, img_w = image.shape[:2]

    patches = []

    for _ in range(n):

        x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
        y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
        patch = image[
                y_center-int(patch_h/2):y_center+int(patch_h/2),
                x_center-int(patch_w/2):x_center+int(patch_w/2)
                ]
        patches.append(patch)

    return patches

def augment_rectangular(data):
    '''agument annotation masks with all combinations of flipping up&down and left&right

    Inputs:
        data : tuple[list]
            list of patch images and masks

    Outputs:
        data_aug : tuple[list]
            list of augmented patch images and masks

    '''

    data_aug = []
    for patch in data:
        patch_ud = np.flipud(patch)
        patch_lr = np.fliplr(patch)
        patch_lr_ud = np.flipud(patch_lr)

        data_aug.extend([(patch), (patch_lr), (patch_ud), (patch_lr_ud)])

    return data_aug

def save_patches(export_dir, data):
    '''Export patches and masks for model training

    Inputs:
        export_dir : str
            path of directory in which images will be saved
        data : tuple[list]
            list of patch images and masks

    Outputs:
        None
    '''

    save_dir_images = os.path.join(export_dir, 'images')

    make_image_dir(save_dir_images)
    data_sorted = [image for sublist in data for image in sublist]
    # breakpoint()
    for i, patch in enumerate(data_sorted):
        file_name = f'p{i}'
        save_image_path = os.path.join(save_dir_images, file_name + '.png')
        # breakpoint()
        patch = Image.fromarray(np.uint8(patch))

        patch.save(save_image_path)

# number of patches per image (should depend on patch and image size)
n = 40
data = []
for img in images:
    img = np.array(img)
    patches = random_patches(img, n=n, patch_h=128, patch_w=128)
    data.append(patches)
data_aug=data
# data_aug = augment_rectangular(data)
random.shuffle(data_aug)

export_dir = r'C:\Users\p.dyroey\Documents\GitHub\moonrise-fm\moonrisefm\data\real_moon_patches_256'
save_patches(export_dir, data_aug)