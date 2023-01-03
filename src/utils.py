import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor


def get_lr_monitor_callback(logging_interval='epoch'):
    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    return lr_monitor


def get_early_stop_callback(patience=50,
                            monitored_variable='val_loss',
                            mode='min'):
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=monitored_variable,
        mode=mode,
        patience=patience,
        verbose=True,
    )
    return early_stop_callback


def get_model_ckpt_callback():
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
    )
    return checkpoint_callback


def get_logger(logs_dir):
    logger = pl.loggers.TensorBoardLogger(logs_dir.parent, logs_dir.name)
    return logger


def get_checkpoint_path(experiment_dir, version=0):
    cps_dir = experiment_dir / f'version_{version}' / 'checkpoints'
    checkpoint_paths = list(cps_dir.glob('epoch=*.ckpt'))
    if not checkpoint_paths:
        raise FileNotFoundError('No checkpoints found')
    if len(checkpoint_paths) > 1:
        raise ValueError('More than one checkpoint found')
    return checkpoint_paths[0]


def find_lr(trainer, model, figure_path=None, print_results=False):
    lr_finder = trainer.lr_find(model, num_training=50)
    # Results can be found in
    if print_results:
        print(lr_finder.results)
    # Plot with
    if figure_path is not None:
        fig = lr_finder.plot(suggest=True)
        fig.savefig(figure_path, dpi=400)
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    return new_lr


def find_best_lr(model, trainer, figure_path):
    figure_path.mkdir()
    new_lr = find_lr(trainer, model, figure_path)
    print('Best learning rate:', new_lr)
    return new_lr