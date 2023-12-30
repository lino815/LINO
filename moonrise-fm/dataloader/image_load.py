import os
from glob import glob
import natsort
import json
import numpy as np
from PIL import Image
import json

from moonrise_fm.dataloader.utils import *
from moonrise_fm.lib.utils import *

def get_image_names(mode='train'):
    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']
    dir = params['dir']
    if mode=='train':
        file_dir = params['train_dir']
    if mode=='test':
        file_dir = params['test_dir']
    data_dir = os.path.join(dir, file_dir)
    file_names = []
    #for filename in natsort.natsorted(glob(f'{data_dir}/**/*.png', recursive=True)):
    #        # file_names.append(filename.split("\\")[-1])
    #        file_names.append(filename)
    for filename in natsort.natsorted(glob(os.path.join(data_dir, params['image_folder'], '*'))):
        file_names.append(filename.split('/')[-1])
    print(f"found {len(file_names)} images")
    return file_names

def list_labels(file_names, mode='train'):

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    masks = load_masks(file_names,mode=mode)
    shape_to_label = params['label_to_value']
    label_to_shape = {v:k for k,v in shape_to_label.items()}
    
    labels = set()
    for mask in masks:
        # if len(mask.shape) != 3:
        #     mask = rgb2mask(mask)
        labels = labels.union(set([label_to_shape[label] for label in np.unique(mask)]))
        
    return labels

def get_sizes(image_names, mode='train'):

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    if mode=='train':
        params['train_dir']
    if mode=='test':
        params['test_dir']

    h = []
    w = []
    
    for image_name in image_names:

        # file_name = os.path.join(data_dir, params['image_folder'],  image_name)
        file_name = image_name
        image = np.array(Image.open(file_name))
        
        h.append(image.shape[0])
        w.append(image.shape[1])
        
    d = {'h-range': [min(h), max(h)],
         'w-range': [min(w), max(w)]}
    
    return d 

def load_images(image_names, mode='train'):

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    if mode=='train':
        params['train_dir']
    if mode=='test':
        params['test_dir']

    resize_w = params['resize_width']
    bool(params['equalize'])
      
    images = []
    for image_name in image_names:

        # file_name = os.path.join(data_dir, params['image_folder'],  image_name)
        file_name = image_name
        image = Image.open(file_name)
        
        if resize_w is not None: 
            orig_w, orig_h = image.size[:2]
            resize_h = int(resize_w/orig_w*orig_h)
            image = np.array(image.resize((resize_w,resize_h), Image.BILINEAR))
            
        images.append(image)
    print(f"no images: {len(images)}")
    return images

def load_masks(image_names, mode='train'):
    
    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']
    dir = params['dir']
    if mode == 'train':
        file_dir = params['train_dir']
    if mode == 'test':
        file_dir = params['test_dir']
    data_dir = os.path.join(dir, file_dir)

    resize_w = params['resize_width']
    masks = []
    for image_name in image_names:  
        # image_name = re.sub("_img", "_mask", image_name)
        # image_name = re.sub("images", "masks", image_name)
        # file_name = os.path.join(data_dir, params['mask_folder'],  image_name)
        file_name = os.path.abspath(image_name).replace("notebooks", "data").replace("_img", "_mask").replace(params['image_folder'], params['mask_folder'])
        print(data_dir)
        print(params['mask_folder'])
        print(image_name)
        print(file_name)
        mask = Image.open(file_name)
        if resize_w is not None:
            orig_w, orig_h = mask.size[:2]
            resize_h = int(resize_w/orig_w*orig_h)
            mask = mask.resize((resize_w,resize_h), Image.NEAREST)

        masks.append(np.array(mask))
    print(f"no masks: {len(masks)}")
    return masks
