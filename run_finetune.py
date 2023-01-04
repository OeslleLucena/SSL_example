import argparse
import pytorch_lightning as pl
from src.model_finetune import LitViT
from src.utils import get_logger, get_lr_monitor_callback, get_checkpoint_path


def train_model(args):
    trainer = pl.Trainer.from_argparse_args(args)
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    pl.seed_everything(args.seed) # To be reproducible
    model = LitViT(args)
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # add program level args
    parser.add_argument('--data_path', type=str,
                        default='/add/a/path')
    parser.add_argument('--pretrained_path', type=str,
                        default='/add/a/path')
    parser.add_argument('--json_path', type=str,
                        default='/add/a/path')
    parser.add_argument('--experiment_logs_dir', type=str,
                        default='/add/a/path')
    parser.add_argument('--seed', type=int, default=45)

    temp_args, _ = parser.parse_known_args()

    # add model specific args
    parser = LitViT.add_model_specific_args(parser)

    # modify pytorch lightning args
    args = parser.parse_args()
    args.logger = get_logger(temp_args.experiment_logs_dir)
    lr_monitor = get_lr_monitor_callback(logging_interval='epoch')
    args.callbacks = [lr_monitor]
    args.default_root_dir = str(temp_args.experiment_logs_dir)
    args.checkpoint_callback= True
    args.accelerator = 'gpu'
    args.devices = 1
    args.amp_backend = 'native'
    args.precision = 16
    args.norm = True
    args.gradient_clip_val = 0.01
    args.check_val_every_n_epoch = 1
    args.row_log_interval = 1
    args.num_sanity_val_steps = 0

    train_model(args)
