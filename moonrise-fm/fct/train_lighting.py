import pytorch_lightning as pl
from optparse import OptionParser
from unet.utils import make_checkpoint_dir
from unet.unet_lighting import LitUnet
from unet3plus.unet3plus import UNet_3Plus
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from fct.fct_utils.data_utils import get_acdc, convert_masks
import random


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():

    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-f', '--folder', dest='folder',
                      default='input', help='folder name')
    parser.add_option('-m', '--model', dest='model',
                      default='input', help='model name')
    parser.add_option('-o', '--optimzer', dest='optimizer',
                      default='SGD', help='optimizer name')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    setup_seed(0)
    # initialisiert
    now = datetime.datetime.now()
    args = get_args()

    # Datensatz laden
    # dir_data = f'./moonrise_fm/data/{args.folder}'
    dir_data = './moonrise-fm/data/moonrise3_crater_patches_512'
    dir_checkpoint = f'./moonrise-fm/checkpoints/{args.folder}_t{now.strftime("%Y%m_%H%M")}_e{args.epochs}_b{args.batchsize}_lr{args.lr}/'
    dir_summary = f'./moonrise-fm/runs/{args.folder}_t{now.strftime("%Y%m_%H%M")}_e{args.epochs}_b{args.batchsize}_lr{args.lr}'
    params = {'batch_size': args.batchsize, 'shuffle': True, 'num_workers': 4}
    in_channels, out_channels = 3, 6
    optimizer_name = 'SGD'
    make_checkpoint_dir(dir_checkpoint)
    args.model = 'unet3+'
    # val_ratio = 0.2
    # test_ratio = 0.1
    # data_module = UNetDataModule(dir=dir_data, val_ratio=val_ratio, test_ratio=test_ratio)
    val_ratio = 0.1
    # train_loader, val_loader, test_loader = make_dataloaders(dir_data, val_ratio, params)
    # get data
    # training
    acdc_data, _, _ = get_acdc('./moonrise-fm/data/adac_ready2/training')
    acdc_data[1] = convert_masks(acdc_data[1])
    acdc_data[0] = np.transpose(acdc_data[0], (0, 3, 1, 2))  # for the channels
    acdc_data[1] = np.transpose(acdc_data[1], (0, 3, 1, 2))  # for the channels
    acdc_data[0] = torch.Tensor(acdc_data[0])  # convert to tensors
    acdc_data[1] = torch.Tensor(acdc_data[1])  # convert to tensors
    acdc_data = TensorDataset(acdc_data[0], acdc_data[1])
    train_loader = DataLoader(acdc_data, batch_size=args.batch_size)
    # validation
    acdc_data, _, _ = get_acdc('./moonrise-fm/data/adac_ready2/ACDC/testing')
    acdc_data[1] = convert_masks(acdc_data[1])
    acdc_data[0] = np.transpose(acdc_data[0], (0, 3, 1, 2))  # for the channels
    acdc_data[1] = np.transpose(acdc_data[1], (0, 3, 1, 2))  # for the channels
    acdc_data[0] = torch.Tensor(acdc_data[0])  # convert to tensors
    acdc_data[1] = torch.Tensor(acdc_data[1])  # convert to tensors
    acdc_data = TensorDataset(acdc_data[0], acdc_data[1])
    val_loader = DataLoader(acdc_data, batch_size=args.batch_size)

        # train_loader, val_loader, test_loader = make_dataloaders(dir_data, val_ratio, params)
        # hyperparameters = dict(
        #     batchsize=args.batchsize,
        #     lr=args.lr,
        #     optimizer_name=optimizer_name
        # )


    # Lastmodelltraining
    if args.model == 'unet':
        model = LitUnet(n_channels=in_channels, n_classes=out_channels, lr=args.lr)
    if args.model == 'unet3+':
        model = UNet_3Plus(n_channels=in_channels, n_classes=out_channels, lr=args.lr)
    if args.model == 'fct':
        model = LitUnet(n_channels=in_channels, n_classes=out_channels, lr=args.lr)
    # print(pl.utilities.model_summary.ModelSummary(model))
    # print(model)
    checkpoint_callback = ModelCheckpoint(filename=dir_checkpoint+'CP.pth', monitor="val_loss")

    trainer = pl.Trainer(limit_train_batches=args.batchsize,
                         max_epochs=args.epochs,
                         logger=True,
                         optimizer=args.optimizer_name,
                         accelerator='gpu',
                         devices=1,
                         precision=16)
    hyperparameters = dict(batchsize=args.batchsize,
                           lr=args.lr,
                           optimizer_name=args.optimizer_name,
                           model=args.model,
                           max_epochs=args.epochs,
                           )
    print("hyperparameters", hyperparameters)
    trainer.logger.log_hyperparams(hyperparameters)
    # print(pl.utilities.model_summary.ModelSummary(model))
    # print(model)
    checkpoint_callback = ModelCheckpoint(filename=dir_checkpoint+'CP.pth', monitor="val_loss")
    # trainer.fit(model=model, datamodule=data_module)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)