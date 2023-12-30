import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from img_lib.utils import mask2rgb
from unet3plus.unet3plus import UNet_3Plus # assuming that you have a file called Unet_3Plus.py that contains your model implementation

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

# Define the path to the folder containing the test images
image_folder_path = r"C:\Users\p.dyroey\Documents\GitHub\moonrise-fm\moonrisefm\data\real_moon_patches_256\images"
path = Path(image_folder_path)
# Loop through all the images in the folder and apply the model to each one
for filename in tqdm(os.listdir(image_folder_path)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the test image
        image_path = path / filename
        image = Image.open(image_path).convert("RGB")

        # Apply the transform to the test image
        image_tensor = transform(image)

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

        # Ausgabe des Vorhersagewerts und der Klasse
        save_mask_path = path.parents[0] / "masks" / f'{image_path.stem}_pred.png'  # Der neue Pfad

        rgb_image = mask2rgb(single_pred_image)
        patch = Image.fromarray(rgb_image)
        patch.save(save_mask_path)