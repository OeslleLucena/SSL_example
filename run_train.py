import argparse
from pathlib import Path
import pytorch_lightning as pl
from src.model import LitUNETR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('--runs_dir', type=str,
                        default='/home/ol18/Codes/SSL_example')
    parser.add_argument('--exp_name', type=str,
                        default='logdir')
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--version', type=int, default=0)


    # add model specific args
    parser = LitUNETR.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(exist_ok=True)
    experiment_dir = runs_dir / args.exp_name
    # lr_monitor = get_lr_monitor_callback(logging_interval='epoch')
    # args.logger = get_logger(experiment_dir)
    # args.callbacks = [lr_monitor]
    args.default_root_dir = str(experiment_dir)
    # args.checkpoint_callback= True
    args.accelerator = 'gpu'
    args.devices = 1

    # args.row_log_interval = 1
    # args.check_val_every_n_epoch = 1
    # args.max_epochs = 400
    # args.gradient_clip_val = 0.01
    # args.num_sanity_val_steps = 0
    # args.amp_level = 'O2'
    # args.precision = 16
    # args.norm = True
    # args.net_mode = 'train'
    # args.uq_flag = False
    # args.drop_rate = 0 #0.05
    # args.batch_size = 1
    # args.patch_size = 64
    # args.samples_per_volume = 1
    # args.learning_rate = 1e-3
    # args.init_idx = 3
    # args.num_pool = 4
    # args.num_output_channels = 73
    # args.lr_step_size = 50
    # args.accumulate_grad_batches = 3 #new
    # args.weights_flag = True
    # args.weights_version = 3
    # args.loss_type = 'BCE'

    # is resume from ckpt needed?
    cps_dir = experiment_dir / f'version_{args.version}' / 'checkpoints'
    checkpoint_paths = list(cps_dir.glob('epoch=*.ckpt'))

    trainer = pl.Trainer.from_argparse_args(args)
    model = LitUNETR(args)
    trainer.fit(model)

    # if checkpoint_paths:
    #     print('[INFO] Resume training')
    #     if len(checkpoint_paths) > 1:
    #         checkpoint_paths.sort()
    #         print(checkpoint_paths)
    #         cpkt = checkpoint_paths[-1] # pick the latest
    #     else:
    #         cpkt = checkpoint_paths[0]
    #     args.resume_from_checkpoint=cpkt
    #     # init the trainer like this
    #     trainer = pl.Trainer.from_argparse_args(args)
    #     model = LitUnet(args)
    #     trainer.fit(model)
    #
    # else:
    #     # init the trainer like this
    #     trainer = pl.Trainer.from_argparse_args(args)
    #     dict_args = vars(args)
    #     del dict_args['callbacks']  # remove from hyperparameters yaml file
    #     del dict_args['logger']  # remove from hyperparameters yaml file
    #     model = LitUnet(args)
    #     # Run lr finder
    #     # lr_finder = trainer.tuner.lr_find(model, num_training=30)
    #     # suggested_lr = lr_finder.suggestion()
    #     # args.learning_rate = suggested_lr
    #     # print(suggested_lr)
    #     # model = LitUnet(args)
    #     trainer.fit(model)
