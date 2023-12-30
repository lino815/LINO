from PIL import Image
import natsort
import os
import numpy as np
import glob
from dataloader import *
from img_lib import *
import random
import natsort
from pathlib import Path
from collections import Counter
# load image names
raw_path = Path(r"C:\Users\Y.li\Desktop\data")
path = raw_path / "2023_424"
patch_size = 256
number_of_patches = 40
class_names = 'class_names.txt'


full_path = path / 'images'
full_glob = full_path.glob('*')
file_names = []
for filename in natsort.natsorted(full_glob):
    file_names.append(filename.name)
print(f"found {len(file_names)} images")

# load masks
mask_dir = path / 'masks'
masks = []
for image_name in file_names:
    # image_name = re.sub("_img", "_mask", image_name)
    # image_name = re.sub("images", "masks", image_name)
    # file_name = os.path.join(data_dir, params['mask_folder'],  image_name)
    temp_path = path / "masks" / image_name
    file_name = str(temp_path).replace("_img", "_mask")
    if not os.path.isfile(file_name):
        file_name = os.path.abspath(file_name).replace(".jpg", ".png")
    print(image_name)
    print(file_name)
    mask = Image.open(file_name)
    masks.append(np.array(mask))
print(f"no masks: {len(masks)}")

# read list of class names and make dict
sorted_dict = {}
file_path = path / class_names
if file_path.is_file():
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines.sort()  # Sort the lines
        for i, line in enumerate(lines, start=0):
            sorted_dict[i] = line.strip()
    label_to_shape = sorted_dict
    shape_to_label = {v: k for k, v in label_to_shape.items()}
else:
    shape_to_label = {"value 0": 0,
                      "value 1": 1,
                      "value 2": 2,
                      "value 3": 3,
                      "value 4": 4,
                      "value 5": 5
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
    file_name = path / "images" / image_name
    image = np.array(Image.open(file_name))
    h.append(image.shape[0])
    w.append(image.shape[1])

d = {'h-range': [min(h), max(h)],
     'w-range': [min(w), max(w)]}

## load images
images = []
for image_name in file_names:
    file_name = path / "images" / image_name
    image = Image.open(file_name)
    images.append(image)
print(f"no images: {len(images)}")


# patches
# number of patches per image (should depend on patch and image size)

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
    patches, patch_masks = random_patches(img, mask, n=number_of_patches, patch_h=patch_size, patch_w=patch_size)
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

export_dir = path.parent / (str(path.name) + '_patches_' + str(patch_size))
print(export_dir)
save_patches(export_dir, data_aug)