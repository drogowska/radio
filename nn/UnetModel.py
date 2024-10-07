import glob
import os
import torch
from monai.networks.nets import UNet, FlexibleUNet
from monai.losses import DiceLoss,DiceCELoss
from monai.metrics import DiceMetric, compute_dice, compute_hausdorff_distance, compute_average_surface_distance
from monai.inferers import sliding_window_inference
from utils import save_nifti_file, save_subject_np, save_subject_nii
from torch.nn import CrossEntropyLoss

from pytorch_lightning import LightningModule

import numpy as np
import nibabel as nib


class UnetModel(LightningModule):
    @classmethod
    def load(cls, v_num: int):
        files = glob.glob(f'E:/radio/logs/FlexibleUNet_{cls.__name__}/version_{v_num}/checkpoints/*.ckpt')
        if len(files) != 1:
            raise ValueError
        return cls.load_from_checkpoint(files[0])
    
    def __init__(self, stage):
        super().__init__()
        self.model = FlexibleUNet(
            in_channels=1,
            out_channels=1,
            # backbone='efficientnet-b2',
            backbone='resnet34',
            # decoder_channels=(256,128,32,16,8),
            # channels=(16, 32, 64, 128, 256),
            spatial_dims=3,
            # dropout=0.1,
            pretrained=True
        )
        # self.loss = DiceLoss(to_onehot_y=False, sigmoid=True)
        self.learning_rate = 1e-3
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.stage = stage
        weight = 10.0
        weight_tensor = torch.tensor(weight, dtype=torch.float) if weight is not None else None

        self.loss = DiceCELoss(to_onehot_y=False, sigmoid=True,
                weight=weight_tensor
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]
        outputs = self(images)
        loss = self.loss(outputs, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return  dict(  
            loss=loss,
        )

    def validation_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]
        outputs = sliding_window_inference(images, (256, 256, 256), 4, self.forward)
        loss = self.loss(outputs, masks)
        outputs = torch.sigmoid(outputs)
        outputs = outputs > 0.5
        self.log(f"val_loss",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dict(
            loss=loss,
        )
    
    def predict_step(self, batch, batch_idx):
        images, mask, name = batch["image"], batch["mask"], batch["name"]
        roi_size = (256, 256, 256)  
        outputs = sliding_window_inference(images, roi_size, 4, self.model)
        outputs = torch.sigmoid(outputs)
        outputs = outputs > 0.5
     
        return [outputs, images, mask, name]


class UnetLADCube(UnetModel):
    def __init__(self):
        super().__init__(stage='unet_LAD_cube')

class UnetLADAll(UnetModel):
    def __init__(self):
        super().__init__(stage='unet_LAD_all')

class UnetBreastCancerCube(UnetModel):
    def __init__(self):
        super().__init__(stage='unet_cancer_breast_cube')


class UnetCancerBedCube(UnetModel):
    def __init__(self):
        super().__init__(stage='unet_cancer_bed_cube')

class UnetCancerBedAll(UnetModel):
    def __init__(self):
        super().__init__(stage='unet_cancer_bed_all')
