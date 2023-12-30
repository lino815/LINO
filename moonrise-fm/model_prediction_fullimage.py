import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from img_lib.utils import mask2rgb
from unet3plus.unet3plus import UNet_3Plus # assuming that you have a file called Unet_3Plus.py that contains your model implementation
import numpy as np
# Define the path to your saved checkpoint
checkpoint_path = r"C:\Users\p.dyroey\Documents\GitHub\moonrise-fm\lightning_logs\\tile4_patches_256\unet3+\version_22\checkpoints\epoch=49-step=3950.ckpt"

# Load the saved checkpoint
model = UNet_3Plus.load_from_checkpoint(checkpoint_path, n_channels=3, n_classes=6)

# Set the model to evaluation mode
model.eval()

# Define the transform to be applied to the input image
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# todo look after orobix and france1 model inference for better overlap calculations
def split_image(image, block_size=256, overlap=0.2):
    # Bestimmen der Anzahl von Blöcken in jeder Dimension
    overlap_px = int(block_size * overlap)
    padding = block_size
    non_overlap_size = block_size // 2
    image_pad = np.pad(image, [(padding, padding), (padding, padding), (0, 0)], mode='reflect')

    n_blocks_x = int(image.shape[0] / non_overlap_size) + 1
    n_blocks_y = int(image.shape[1] / non_overlap_size) + 1

    # Erstellen von leeren Arrays für die Patches und deren Indizes
    patches = []
    patch_indices = []

    # Erstellen von Blöcken durch Durchlaufen des Bildes und Auswählen der entsprechenden Pixel
    for i in range(n_blocks_x):
        for j in range(n_blocks_y):
            x_start = padding + i * non_overlap_size - padding//2 # padding zu richtigem bild + patch nummer mit richtigen pixel - overlap für
            y_start = padding + j * non_overlap_size - padding//2
            x_end = x_start + block_size
            y_end = y_start + block_size
            patch = image_pad[x_start:x_end, y_start:y_end, :]
            patches.append(patch)
            patch_indices.append((i, j))

    # Umformen des Eingabebildes in einen 4D-Array mit der Form (n_blocks_x, n_blocks_y, block_size, block_size, n_channels)
    # patches = np.reshape(image, (n_blocks_x, block_size, n_blocks_y, block_size, -1))
    # patches = np.transpose(patches, (0, 2, 1, 3, 4))
    return patches, patch_indices, image.shape, padding

# Funktion zum Zusammenführen von kleinen Bildern in ein großes Bild
def merge_image(patches: list, patch_indices: list[tuple], padding: int, original_shape, block_size=256):
    # Erstellen eines leeren Arrays für das zusammengeführte Bild
    merged_image = np.zeros((original_shape[0]+block_size, original_shape[1]+block_size))
    unpadding_val = block_size // 2
    non_overlap_size = block_size // 2
    # Zusammenführen der Patches in das zusammengeführte Bild durch Einfügen an der entsprechenden Position
    for i, j in patch_indices:
        x_start = i * non_overlap_size
        y_start = j * non_overlap_size
        x_end = x_start + non_overlap_size
        y_end = y_start + non_overlap_size
        image = patches.pop(0)
        unpadded = image[unpadding_val:non_overlap_size+unpadding_val, unpadding_val:non_overlap_size+unpadding_val]
        merged_image[x_start:x_end, y_start:y_end] = unpadded
    merged_image = merged_image[0:original_shape[0], 0:original_shape[1]]
    return merged_image


# Define the path to the folder containing the test images
image_folder_path = r"C:\Users\p.dyroey\Documents\GitHub\moonrise-fm\moonrisefm\data\moonrise_test\images"
path = Path(image_folder_path)
image_list = []
mask_list = []
# Loop through all the images in the folder and apply the model to each one
for filename in tqdm(os.listdir(image_folder_path), desc="images", position=0):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the test image
        image_path = path / filename
        image = Image.open(image_path).convert("RGB")
        image_whc = np.array(image)
        # image_cwh = np.moveaxis(image_whc, -1, 0)

        patches, patch_indices, pad_shape, padding = split_image(image_whc, 256)
        patches_seg = []

        for patch in tqdm(patches, desc="patches", position=1, leave=False):
            image_tensor = transform(patch)
            # Add batch dimension to the image tensor
            image_tensor = image_tensor.unsqueeze(0)
            # Run the image through the model
            with torch.no_grad():
                output = model(image_tensor)
            # Extrahieren des Vorhersagewerts und der Klasse
            # make argmax to get single image
            probs = torch.softmax(output, dim=1)
            masks_pred = torch.argmax(probs, dim=1)
            # remove batch dimension
            single_pred_image = masks_pred.numpy()[0, :, :]
            patches_seg.append(single_pred_image)

        image_seg = merge_image(patches=patches_seg, patch_indices=patch_indices, padding=64, original_shape=image_whc.shape, block_size=256)

        image_list.append(image)
        mask_list.append(image_seg)

        # Ausgabe des Vorhersagewerts und der Klasse
        save_mask_path = path.parents[0] / "masks" / f'{image_path.stem}_pred.png' # Der neue Pfad
        rgb_image = mask2rgb(image_seg)

        patch = Image.fromarray(rgb_image)
        patch.save(save_mask_path)

import napari
viewer = napari.view_image(image_list, name='input', rgb=True)
viewer = napari.view_image(mask_list, name='output', rgb=True)
