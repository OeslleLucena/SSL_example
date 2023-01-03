import json
import time
import torch
# import torch.optim as optim
# import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import pytorch_lightning as pl

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled
)

TRAINING = 'training'
VALIDATION = 'validation'
INFERENCE = 'inference'


class LitUNETR(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.net = ViTAutoEnc(in_channels=self.hparams.num_input_channels,
                              img_size=self.hparams.img_size,
                              patch_size=self.hparams.patch_size,
                              pos_embed='conv',
                              hidden_size=self.hparams.hidden_size,
                              mlp_dim=self.hparams.mlp_dim)

        self.get_paths()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--mlp_dim', type=int, default=3072)
        parser.add_argument('--hidden_size', type=int, default=768)
        parser.add_argument('--img_size', type=int, default=(96, 96, 96))
        parser.add_argument('--patch_size', type=int, default=(16, 16, 16))
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--samples_per_volume', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--num_input_channels', type=int, default=1)


        parser.add_argument('--data_path', type=str,
                            default='/media/ol18/Elements/Datasets/CT-Covid-19-August2020')
        parser.add_argument('--weights_path', type=str,
                            default='/media/ol18/Seagate Expansion Drive/'
                                    'Datasets/TractSeg_Data/label_weights')
        parser.add_argument('--json_path', type=str,
                            default='/home/ol18/Codes/SSL_example/self_supervised_pretraining/json_files/'
                                    'tcia_covid19/dataset_split.json')

        return parser

    def forward(self, x):
        return self.net(x)

    def get_paths(self):
        json_path =  Path(self.hparams.json_path)
        data_path =  Path(self.hparams.data_path)

        with open(json_path, 'r') as json_f:
            json_data = json.load(json_f)

        train_data = json_data[TRAINING]
        val_data = json_data[VALIDATION]

        for idx, each_d in enumerate(train_data):
            train_data[idx]['image'] = data_path / train_data[idx]['image']

        for idx, each_d in enumerate(val_data):
            val_data[idx]['image'] = data_path / val_data[idx]['image']

        self.train_data = train_data
        self.val_data = train_data

        self.recon_loss = L1Loss()
        self.contrastive_loss = ContrastiveLoss(batch_size=self.hparams.batch_size * 2,
                                                temperature=0.05)

    def prepare_data(self):
        set_determinism(seed=self.hparams.seed)

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    2.0, 2.0, 2.0), mode=("bilinear")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
                RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
                CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
                OneOf(transforms=[
                    RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True,
                                       max_spatial_size=32),
                    RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False,
                                       max_spatial_size=64),
                ]
                ),
                RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
                # Please note that that if image, image_2 are called via the same transform call because of the determinism
                # they will get augmented the exact same way which is not the required case here, hence two calls are made
                OneOf(transforms=[
                    RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True,
                                       max_spatial_size=32),
                    RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False,
                                       max_spatial_size=64),
                ]
                ),
                RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8)
            ]
        )

        # Define DataLoader using MONAI, CacheDataset needs to be used
        self.train_ds = Dataset(data=self.train_data, transform=train_transforms)
        self.val_ds = Dataset(data=self.val_data, transform=train_transforms)


    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs_1 = batch['image']
        inputs_2 = batch['image_2']
        gt_input = batch['gt_image']

        outputs_v1, hidden_v1 = self(inputs_1)
        outputs_v2, hidden_v2 = self(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        r_loss = self.recon_loss(outputs_v1, gt_input)
        cl_loss = self.contrastive_loss(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss + cl_loss * r_loss

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)


    #
    # def validation_step(self, batch, batch_idx):
    #     images, labels = batch
    #     logits_predicted = self(images)
    #
    #     loss = F.cross_entropy(logits_predicted, labels)
    #     acc = torch.sum(torch.eq(torch.argmax(logits_predicted, -1), labels).to(torch.float32)) / len(labels)
    #
    #     log = {'val_loss': loss,
    #            'val_acc': acc}
    #
    #     return log
    #
    # def validation_epoch_end(self, outputs):
    #     mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     mean_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
    #
    #     self.log('val/loss', mean_loss, prog_bar=True)
    #     self.log('val/acc', mean_acc, prog_bar=True)