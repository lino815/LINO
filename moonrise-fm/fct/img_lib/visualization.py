import matplotlib.pyplot as plt
import numpy as np
import cv2

from img_lib.utils import mask2rgb


def make_tile(path_image, path_mask, tile_size):

    patch_id = path_image.split('/')[-1][:-4]
    patch = cv2.imread(path_image)
    patch_mask = cv2.cvtColor(cv2.imread(path_mask),cv2.COLOR_BGR2RGB) 
    patch_tot = np.hstack((patch, patch_mask))
    patch_tot = cv2.resize(patch_tot, dsize=tile_size)
    patch_tot = cv2.putText(patch_tot, patch_id, (5,patch_tot.shape[0]-5), cv2.FONT_HERSHEY_PLAIN, 1.25, (255, 255, 0), thickness=2)

    return patch_tot

def compose_image(path_images, path_masks, tile_size, n=4):
    
    n_images = len(path_images)
    m = int(np.ceil(n_images / n))
    complete_image = np.zeros((m*tile_size[1], n*tile_size[0], 3), dtype=np.uint8)

    counter = 0
    for i in range(m):
        ys = i*tile_size[1]
        ye = ys+tile_size[1]
        for j in range(n):
            xs = j*tile_size[0]
            xe = xs+tile_size[0]
            if counter==n_images:
                break
            path_image = path_images[counter]
            path_mask = path_masks[counter]
            patch_tile = make_tile(path_image, path_mask, tile_size)
            counter += 1
            complete_image[ys:ye, xs:xe, :] = patch_tile
        if counter==n_images:
            break 
            
    return complete_image

def plot_patches(complete_image, tile_size, index=0, n_rows=8):
    
    y_start = (index)*tile_size[1]*n_rows
    y_end = y_start + tile_size[1]*n_rows
    
    print(y_start, y_end)
    
    fig,ax = plt.subplots(1,1,figsize=(20,20))
    ax.imshow(complete_image[y_start:y_end,:,:])
    plt.show()

def plot_net_predictions(imgs, true_masks, masks_pred, batch_size):
    # print(batch_size)
    # fig, ax = plt.subplots(3, 2)
    # # fig, ax = plt.subplots(3, 2) # changed for batch_size
    # for i in range(1):
    
    fig, ax = plt.subplots(3, batch_size, figsize=(20, 15))
    
    for i in range(batch_size):
        img = np.transpose(imgs[i].squeeze().cpu().detach().numpy(), (1, 2, 0))
        mask_pred = masks_pred[i].cpu().detach().numpy()
        mask_true = true_masks[i].cpu().detach().numpy()
        # print(1, i, img.shape)
        # print(img.shape, mask_pred.shape, mask_true.shape)
        if len(img.shape) != 2:
            num_channels = img.shape[-1]

            if num_channels == 1:
                # Plot grayscale image
                ax[0, i].imshow(img.squeeze(), cmap='gray')
            elif num_channels == 3:
                # Plot RGB image
                ax[0, i].imshow(img)
            else:
                # Plot the first channel of the image stack
                sdifsdijf = img[..., 0]
                ax[0, i].imshow(img[..., 0], cmap='gray')
        else:
            ax[0, i].imshow(img)
        ax[1,i].imshow(mask2rgb(mask_pred))
        ax[1,i].set_title('Predicted')
        ax[2,i].imshow(mask2rgb(mask_true))
        ax[2,i].set_title('Ground truth')
        ax[0, i].axis('off')
        ax[1, i].axis('off')
        ax[2, i].axis('off')
    fig.tight_layout()
    return fig