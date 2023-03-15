import os
from glob import glob
import natsort
import numpy as np
import pytorch_lightning
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
from pathlib import Path
from lib.utils import rgb2mask

class CoreDataset(Dataset):
    
    def __init__(self, path, transform=None):
        print(os.getcwd())
        print(path)
        glob_path = glob(os.path.join(path, 'images', '*.png'))
        self.path_images = natsort.natsorted(glob_path)
        self.path_masks = natsort.natsorted(glob(os.path.join(path, 'masks', '*.png')))
        self.transform = transform
        print(f" {len(self.path_images)} images, {len(self.path_masks)} masks")
        
    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.path_images[idx])
        mask = Image.open(self.path_masks[idx])
        
        sample = {'image':image, 'mask':mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
# add image normalization transform at some point
   
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image, mask = sample['image'], sample['mask']  
        # standard scaling would be probably better then dividing by 255 (subtract mean and divide by std of the dataset)
        image = np.array(image)/255
        # convert colors to "flat" labels
        mask = rgb2mask(np.array(mask))
        # mask = np.array(mask)
        sample = {'image': torch.from_numpy(image).permute(2,0,1).float(),
                  'mask': torch.from_numpy(mask).long(), 
                 }
        
        return sample

class UNetDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, batch_size=16, dir='.', val_ratio=0.2, test_ratio=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.path = dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    def setup(self, stage=None):
        dataset = CoreDataset(self.path, transform=transforms.Compose([ToTensor()]))
        val_len = int(self.val_ratio * len(dataset))
        test_len = int(self.test_ratio * len(dataset))
        train_len = len(dataset) - val_len - test_len
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_len, val_len, test_len])
    def train_dataloader(self):
        return DataLoader(self.train_dataset, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, drop_last=True)
def make_datasets(path, val_ratio):
    dataset = CoreDataset(path, transform = transforms.Compose([ToTensor()]))
    # train_len
    val_len = int(val_ratio*len(dataset))
    test_len = int(val_ratio * len(dataset))
    lengths = [len(dataset)-val_len-test_len, val_len,test_len]
    train_dataset, val_dataset, test_dataset= random_split(dataset, lengths)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8,0.1,0.1])

    return train_dataset, val_dataset, test_dataset


def make_dataloaders(path, val_ratio, params):
    train_dataset, val_dataset, test_dataset= make_datasets(path, val_ratio)
    train_loader = DataLoader(train_dataset, drop_last=True, **params)
    val_loader = DataLoader(val_dataset, drop_last=True, **params)
    test_loader = DataLoader(test_dataset, drop_last=True, **params)

    return train_loader, val_loader, test_loader