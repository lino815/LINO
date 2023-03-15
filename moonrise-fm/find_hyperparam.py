import pytorch_lightning as pl
import optuna
from unet.load_data import make_dataloaders
from optuna.integration import PyTorchLightningPruningCallback
from seabed_Unet.emunet import EMUnet
#from unet.unet_lighting import LitUnet
import torch
import os

from unet3plus import UNet_3Plus
def objective(trial: optuna.trial.Trial) -> float:
    # para
    epoch = 50
    # batchsize = trial.suggest_int("batchsize", 1, 4)
    batchsize = 4
    lr = trial.suggest_float("learning rate", 1e-3, 1)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    dir_data = f'./moonrise-fm/data/moonrise3_crater_patches_512'
    # params = {'batch_size': batchsize, 'shuffle': True, 'num_workers': 4}
    params = {'batch_size': batchsize, 'shuffle': True, 'num_workers': 8}

    model = EMUnet(n_channels=3, n_classes=4, lr=lr, optimizer=optimizer_name)
    val_ratio = 0.1
    train_loader, val_loader, test_loader = make_dataloaders(dir_data, val_ratio, params)

    # trainer = pl.Trainer(
    #     logger=True,
    #     enable_checkpointing=False,
    #     max_epochs=epoch,
    #     gpus=0 if torch.cuda.is_available() else None,
    #     callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    # )
    trainer = pl.Trainer(enable_checkpointing=False,
                         max_epochs=epoch,
                         logger=True,
                         accelerator='gpu',
                         devices=1,
                         callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_jaccard')],
                         precision=16
                         )
    hyperparameters = dict(batchsize=batchsize,
                           lr=lr,
                           optimizer_name=optimizer_name
                           )
    print("hyperparameters", hyperparameters)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["'val_jaccard'"].detachm()



from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)



if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50)

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
