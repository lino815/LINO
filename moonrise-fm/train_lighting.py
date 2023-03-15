import pytorch_lightning as pl
from optparse import OptionParser
from unet.load_data import UNetDataModule
from unet.utils import make_checkpoint_dir
from unet.load_data import make_dataloaders
from unet.unet_lighting import LitUnet
from unet3plus.unet3plus import UNet_3Plus
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassRecall, MulticlassF1Score, MulticlassPrecision
import datetime


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

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    # initialisiert
    now = datetime.datetime.now()
    args = get_args()

    # Datensatz laden
    # dir_data = f'./moonrise_fm/data/{args.folder}'
    dir_data = f'./moonrise-fm/data/moonrise3_crater_patches_512'
    dir_checkpoint = f'./checkpoints/{args.folder}_t{now.strftime("%Y%m_%H%M")}_e{args.epochs}_b{args.batchsize}_lr{args.lr}/'
    dir_summary = f'./runs/{args.folder}_t{now.strftime("%Y%m_%H%M")}_e{args.epochs}_b{args.batchsize}_lr{args.lr}'
    params = {'batch_size': args.batchsize, 'shuffle': True, 'num_workers': 4}

    make_checkpoint_dir(dir_checkpoint)

    # val_ratio = 0.2
    # test_ratio = 0.1
    # data_module = UNetDataModule(dir=dir_data, val_ratio=val_ratio, test_ratio=test_ratio)
    val_ratio = 0.1
    # train_loader, val_loader, test_loader = make_dataloaders(dir_data, val_ratio, params)

    logger = pl.loggers.TensorBoardLogger("tb_logs", name=f'{args.folder}')

    # Lastmodelltraining
    # model = LitUnet(n_channels=3, n_classes=3)
    model = UNet_3Plus(in_channels=3, n_classes=6, lr=1e-3)
    # print(pl.utilities.model_summary.ModelSummary(model))
    # print(model)
    checkpoint_callback = ModelCheckpoint(filename=dir_checkpoint+f'CP.pth', monitor="val_loss")
    trainer = pl.Trainer(limit_train_batches=args.batchsize, max_epochs=args.epochs, logger=logger, accelerator='gpu', devices=1)
    # trainer.fit(model=model, datamodule=data_module)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)