import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
import pytorch_lightning as pl
from monai.utils import set_determinism, first
from monai.networks.nets import UNETR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


TRAINING = 'training'
VALIDATION = 'validation'
INFERENCE = 'inference'


class LitViT(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.net = UNETR(
                    in_channels=1,
                    out_channels=14,
                    img_size=(96, 96, 96),
                    feature_size=16,
                    hidden_size=768,
                    mlp_dim=3072,
                    num_heads=12,
                    pos_embed="conv",
                    norm_name="instance",
                    res_block=True,
                    conv_block=True,
                    dropout_rate=0.0,
                )

        # Load ViT backbone weights into UNETR
        if self.hparams.pretrained:
            print('Loading Weights from the Path {}'.format(self.hparams.pretrained_path))
            vit_dict = torch.load(self.hparams.pretrained_path)
            vit_weights = vit_dict['state_dict']

            # Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
            # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
            # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
            # while pretraining with ViTAutoEnc and are not a part of ViT backbone.
            model_dict = self.net.vit.state_dict()
            vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
            model_dict.update(vit_weights)
            self.net.vit.load_state_dict(model_dict)
            del model_dict, vit_weights, vit_dict
            print('Pretrained Weights Succesfully Loaded !')

        elif not self.hparams.pretrained:
            print('No weights were loaded, all weights being used are randomly initialized!')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--mlp_dim', type=int, default=3072)
        parser.add_argument('--hidden_size', type=int, default=768)
        parser.add_argument('--img_size', type=tuple, default=(96, 96, 96))
        parser.add_argument('--patch_size', type=tuple, default=(16, 16, 16))
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--samples_per_volume', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--num_input_channels', type=int, default=1)
        parser.add_argument('--pretrained', type=bool, default=True)
        return parser

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        set_determinism(seed=self.hparams.seed)
        json_path =  Path(self.hparams.json_path)
        data_path =  Path(self.hparams.data_path)


        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        datalist = load_decathlon_datalist(base_dir=data_path,
                                           data_list_file_path=json_path,
                                           is_segmentation=True,
                                           data_list_key="training")

        val_files = load_decathlon_datalist(base_dir=data_path,
                                            data_list_file_path=json_path,
                                            is_segmentation=True,
                                            data_list_key="validation")


        # Define DataLoader using MONAI, CacheDataset needs to be used
        self.train_ds = CacheDataset(
                            data=datalist,
                            transform=train_transforms,
                            cache_num=24,
                            cache_rate=1.0,
                            num_workers=4,
                        )
        self.val_ds = CacheDataset(
                        data=val_files,
                        transform=val_transforms,
                        cache_num=6,
                        cache_rate=1.0,
                        num_workers=4
                    )

        self.dice_metric = DiceMetric(include_background=True,
                                      reduction="mean",
                                      get_not_nans=False)
        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_label = AsDiscrete(to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=14)


    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          num_workers=self.hparams.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          num_workers=self.hparams.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs = batch['image']
        labels = batch['label']
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        labels = batch['label']
        sw_batch_size = 4
        outputs = sliding_window_inference(inputs, self.hparams.img_size, sw_batch_size,
                                           self.forward)
        labels_convert = [self.post_label(label) for label in decollate_batch(labels)]
        output_convert = [self.post_pred(pred) for pred in decollate_batch(outputs)]

        loss = self.criterion(output_convert, labels_convert)
        dice = self.dice_metric(y_pred=output_convert, y=labels_convert).mean()

        self.log('val', loss,
                 on_step=True,
                 prog_bar=True,
                 logger=True)
        self.log('val_dice', dice, on_epoch=True, prog_bar=True, logger=True)

        return dice