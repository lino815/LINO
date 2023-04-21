import matplotlib.pyplot as plt
from PIL import Image
import json
import natsort
import os
import numpy as np
import glob
from moonrise_fm.dataloader import *
from moonrise_fm.lib import *
from moonrise_fm.model import *
PATH_PARAMETERS = r'C:\Users\Y.li\Desktop\Moonrise_crater'
# load image names
file_names = []
for filename in natsort.natsorted(glob(os.path.join(PATH_PARAMETERS, 'images', '*'))):
    file_names.append(filename.split('/')[-1])
print(f"found {len(file_names)} images")


# load masks
masks = []
for image_name in file_names:
    # image_name = re.sub("_img", "_mask", image_name)
    # image_name = re.sub("images", "masks", image_name)
    # file_name = os.path.join(data_dir, params['mask_folder'],  image_name)
    file_name = os.path.abspath(image_name).replace("_img", "_mask").replace(
        'images', 'masks')
    print(image_name)
    print(file_name)
    mask = Image.open(file_name)
    masks.append(np.array(mask))
print(f"no masks: {len(masks)}")

# list labels
shape_to_label = {"_background_": 0,
                  "ground": 1,
                  "shadow": 2,
                  "stone": 3,
                  "crater-edge": 4,
                  "crater-slope": 5
                  }
label_to_shape = {v: k for k, v in shape_to_label.items()}

labels = set()
for mask in masks:
    # if len(mask.shape) != 3:
    #     mask = rgb2mask(mask)
    labels = labels.union(set([label_to_shape[label] for label in np.unique(mask)]))

# get size of images
h = []
w = []

for image_name in file_names:
    file_name = image_name
    image = np.array(Image.open(file_name))
    h.append(image.shape[0])
    w.append(image.shape[1])

d = {'h-range': [min(h), max(h)],
     'w-range': [min(w), max(w)]}

## load images
images = []
for image_name in file_names:
    file_name = image_name
    image = Image.open(file_name)
    images.append(image)
print(f"no images: {len(images)}")


# patches
# number of patches per image (should depend on patch and image size)
n = 20
label_to_shape = {v: k for k, v in shape_to_label.items()}
d_pixels = {k: 0 for k in shape_to_label.keys()}
d_images = {k: 0 for k in shape_to_label.keys()}
feature_labels = set([v for k, v in shape_to_label.items() if k != '_background_'])

data = []

for img, mask in zip(images, masks):
    img = np.array(img)
    mask = np.array(mask)

    print(img.shape)
    print(mask.shape)
    # print(mask.shape)
    # print(len(mask.shape))
    # error but cant find the solution. the image is already 2d so no rgb2mask needed. but tests dont work
    # mask = rgb2mask(mask)
    patches, patch_masks = random_patches(img, mask, n=n, patch_h=1024, patch_w=1024)
    for patch, patch_mask in zip(patches, patch_masks):
        # consider only patch containing feature_labels
        if len(set(np.unique(patch_mask)).intersection(feature_labels)) > 0:
            data.append((patch, patch_mask))

            count = Counter(patch_mask.flatten().tolist())
            for label, num in count.most_common():
                d_pixels[label_to_shape[label]] += num
                d_images[label_to_shape[label]] += 1

print('pixel per class = ', d_pixels)
print('images per class = ', d_images)

data_aug = augment_rectangular(data)
random.shuffle(data_aug)

# F = [val / (d_images[key] * patch_mask.size) for key, val in d_pixels.items()]
# w = np.median(F) / F
# print('cross entropy loss class weights = ', w)

export_dir = r'C:\Users\Y.li\Desktop\moonrise2_patches_crater'
save_patches(export_dir, data_aug)