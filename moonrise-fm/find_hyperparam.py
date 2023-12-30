import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import optuna
from unet.load_data import make_dataloaders
from img_lib.pytorch_pruner_fix import PyTorchLightningPruningCallback
import os
from pathlib import Path
import datetime
import torch
import random
import numpy as np
from unet.unet_lighting import LitUnet
from unet3plus.unet3plus import UNet_3Plus
from emunet.emunet import EMUnet
from optparse import OptionParser
import optparse
from transunet.transunet import TransUnetLightning


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=160,
                      type='int', help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=0,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0,
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load', default=False,
                      help='load file model')
    parser.add_option('-f', '--folder', dest='folder', default='QA_dark_patches_256',
                      help='folder name')
    parser.add_option('-m', '--model', dest='model', default='seabed',
                      type='str', help='model name')
    parser.add_option('-p', '--optimzer', dest='optimizer', default=False,
                      help='optimizer name')
    parser.add_option('-i', '--input', dest='inp', default=3,
                      type='int', help='input channels')
    parser.add_option('-o', '--output', dest='outp', default=4,
                      type='int', help='output channels')
    parser.add_option('-w', '--weight_decay', dest='weight_decay', default=0,
                      type='float', help='weight_decay')
    parser.add_option('--loss', dest='loss', default='',
                      type='str', help='loss function name')
    parser.add_option('-s', '--scheduler', dest='scheduler', default='',
                      type='str', help='scheduler name')
    (options, args) = parser.parse_args()
    return options


def objective(trial: optuna.trial.Trial, args: optparse.Values) -> float:
    model_name = args.model
    epoch = args.epochs
    print(args)
    if args.batchsize == 0:
        if model_name == 'unet3+':
            batchsize = 32
        elif model_name == 'unet':
            batchsize = 32
        elif model_name == 'seabed':
            batchsize = 32
        elif model_name == 'transunet':
            batchsize = 32
    else:
        batchsize = args.batchsize

    if args.lr == 0:
        lr = trial.suggest_float("learning rate", 1e-6, 2*1e-3, log=True)
    else:
        lr = args.lr

    if args.optimizer == 0:
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    else:
        optimizer_name = args.optimizer


    if args.weight_decay == 0:
        weight_decay = trial.suggest_float("weight_decay", 0.0005, 0.005, log=True)
    else:
        weight_decay = args.weight_decay

    if args.loss != '':
        loss_name = args.loss
    else:
        loss_name = trial.suggest_categorical("loss", ["CELoss", "FocalLoss", "DiceLoss"])

    if args.scheduler != '':
        scheduler_name = args.scheduler
    else:
        scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "CosineAnnealingLR", "ExponentialLR"])

    in_channels, out_channels = args.inp, args.outp
    # TODO which working directory should be used?
    # base_dir = Path('./data/')
    # dir_data = 'tile4_patches_256'
    # dir_data = 'moonrise_crater_new_patches_256'
    # dir_data = 'prime_data'
    dir_data = args.folder
    dir = Path('./data/') / dir_data
    params = {'batch_size': batchsize, 'shuffle': True, 'num_workers': 8}

    if model_name == 'unet':
        model = LitUnet(n_channels=in_channels, n_classes=out_channels, lr=lr, scheduler_name=scheduler_name,
                        loss_name=loss_name, weight_decay=weight_decay)
    if model_name == 'unet3+':
        model = UNet_3Plus(n_channels=in_channels, n_classes=out_channels, lr=lr, scheduler_name=scheduler_name,
                           loss_name=loss_name, weight_decay=weight_decay)
    if model_name == 'seabed':
        model = EMUnet(n_channels=in_channels, n_classes=out_channels, lr=lr, scheduler_name=scheduler_name,
                       loss_name=loss_name, weight_decay=weight_decay)

    if model_name == 'transunet':
        model = TransUnetLightning(n_channels=in_channels, n_classes=out_channels, lr=lr, scheduler_name=scheduler_name,
                                   loss_name=loss_name, weight_decay=weight_decay)

    val_ratio = 0.1
    train_loader, val_loader, test_loader = make_dataloaders(dir, val_ratio, params)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"lightning_logs/{dir_data}/")
    trainer = pl.Trainer(enable_checkpointing=True,
                         default_root_dir=f"lightning_logs/{dir_data}/",
                         max_epochs=epoch,
                         logger=tb_logger,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=1,
                         callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_dice')],
                         precision=16
                         )
    hyperparameters = dict(batchsize=batchsize,
                           lr=lr,
                           optimizer_name=optimizer_name,
                           model=model_name,
                           max_epochs=epoch,
                           data=dir_data,
                           params=params,
                           scheduler_name=scheduler_name,
                           loss_name=loss_name,
                           weight_decay=weight_decay
                           )
    print("hyperparameters", hyperparameters)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics['val_dice'].detach()


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(0)
    args = get_args()
    now = datetime.datetime.now()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    study = optuna.create_study(direction="maximize",
                                study_name=f't{now.strftime("%Y%m%d_%H%M")}-{args.folder}-{args.model}',
                                pruner=optuna.pruners.MedianPruner(
                                    n_startup_trials=2, n_warmup_steps=5, interval_steps=3
                                ),
                                storage="sqlite:///db.sqlite3"
                                )
    study.optimize(lambda trial: objective(trial, args), n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()
